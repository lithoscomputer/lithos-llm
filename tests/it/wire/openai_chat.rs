//! Wire parity for the OpenAI Chat Completions dialect.
//!
//! This is the dialect with the most provider skins behind it. OpenAI defined
//! `POST /v1/chat/completions`; OpenRouter, DeepSeek, Venice, Modal, Kimi, and
//! MiniMax each re-implement it and disagree about the details. The fixtures
//! here therefore pin three things no other dialect file has to:
//!
//! 1. Several spellings of the same usage counter, all of which are
//!    **inclusive** and must decode into the same disjoint buckets.
//! 2. A cost the provider reports in-band. No other protocol does this, and a
//!    catalog estimate must never overwrite it.
//! 3. What survives only in [`Response::raw`]. Skins put fields there that this
//!    crate deliberately does not model, so the raw document carries more
//!    weight here than anywhere else.
//!
//! The catalog provider is named `compat` rather than `openai_compatible` on
//! purpose. Raw provider options are keyed by the catalog provider id, while
//! opaque replay parts are keyed by the codec's own `openai_compatible`
//! namespace. Two different names prove the two namespaces are independent.

use std::net::TcpListener;
use std::thread;
use std::time::Duration;

use futures_util::StreamExt as _;
use httpmock::{Method, MockServer};
use lithos_llm::catalog::{Catalog, adapter_ids, codec_ids};
use lithos_llm::types::{
    ContentPart, CostSource, Error, ErrorKind, ImageContent, MediaSource, Message, ReasoningEffort,
    ResponseFormat, RetryClassification, Role, ToolCall, ToolChoice, ToolResult,
};
use lithos_llm::{Request, Response};
use serde_json::{Value, json};

use crate::support::{self, WireCapture, WireProvider};

/// The catalog provider id, which is also the raw-options namespace.
const PROVIDER: &str = "compat";

/// The catalog model id, which is the second half of a request selector.
const MODEL: &str = "compat-chat";

/// The provider's own model name, which is what reaches the wire.
const API_MODEL: &str = "vendor/compat-chat-v2";

/// The only endpoint this dialect has.
const PATH: &str = "/v1/chat/completions";

// ===========================================================================
// Fixtures
// ===========================================================================

fn provider() -> WireProvider<'static> {
    WireProvider::new(
        PROVIDER,
        adapter_ids::OPENAI_COMPATIBLE,
        codec_ids::OPENAI_CHAT,
        MODEL,
    )
    .with_api_model(API_MODEL)
    .with_auth("{ type = \"bearer\" }")
}

fn selector() -> String {
    provider().selector()
}

/// The catalog every fixture uses unless it is about pricing.
///
/// [`WireProvider`] claims every capability, which is what lets an encoding
/// fixture reach the codec with content this dialect cannot carry. A real
/// catalog entry describes the dialect truthfully instead; see
/// [`truthful_catalog`].
fn plain_catalog(base_url: &str) -> Catalog {
    provider().catalog(base_url)
}

/// The capability set an honest Chat Completions catalog entry declares.
///
/// The protocol carries neither documents nor audio, so a catalog that says so
/// makes the client refuse those requests up front.
const TEXT_AND_IMAGE_CAPABILITIES: &str = "{ text = true, images = true, audio = false, \
     documents = false, tools = true, structured_output = true, reasoning = true, caching = true, \
     cache_breakpoints = true, sampling = true }";

/// The same set for a model that declares no prompt caching.
///
/// The prompt-cache breakpoints are gated on `caching` and
/// `cache_breakpoints` together, so a fixture that pins their absence needs a
/// catalog entry that denies them.
const NO_CACHING_CAPABILITIES: &str = "{ text = true, images = true, audio = true, \
     documents = true, tools = true, structured_output = true, reasoning = true, caching = false, \
     sampling = true }";

fn truthful_catalog(base_url: &str) -> Catalog {
    provider()
        .with_capabilities(TEXT_AND_IMAGE_CAPABILITIES)
        .catalog(base_url)
}

/// The same catalog with rates attached, for the cost-precedence fixtures.
///
/// The model table is the last thing [`WireProvider::toml`] renders, so an
/// appended key lands inside it.
fn priced_catalog(base_url: &str) -> Catalog {
    let source = format!(
        "{}pricing = {{ input_usd_micros_per_million = 1000000, \
         output_usd_micros_per_million = 2000000 }}\n",
        provider().toml(base_url)
    );
    support::catalog_from_toml("wire-priced", &source)
}

/// A plain text completion, in the envelope every skin agrees on.
fn text_response() -> Value {
    json!({
        "id": "chatcmpl-wire",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": API_MODEL,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "Hello." },
            "finish_reason": "stop",
        }],
        "usage": { "prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16 },
    })
}

/// A completion that answers with one tool call.
fn tool_call_response() -> Value {
    json!({
        "id": "chatcmpl-wire",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": API_MODEL,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_paris",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"city\": \"Paris\"}",
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": { "prompt_tokens": 42, "completion_tokens": 9, "total_tokens": 51 },
    })
}

// ===========================================================================
// Drivers
// ===========================================================================

/// Builds a catalog for a running mock server.
type CatalogFor = fn(&str) -> Catalog;

/// Completes one request against a canned JSON body.
///
/// Returns the captured wire request and the decoded response, which is the
/// pair every non-streaming fixture snapshots.
async fn exchange(request: Request, body: &Value) -> (WireCapture, Response) {
    exchange_with(plain_catalog, request, body).await
}

async fn exchange_with(
    catalog: CatalogFor,
    request: Request,
    body: &Value,
) -> (WireCapture, Response) {
    let server = MockServer::start_async().await;
    let client = support::client_for(
        catalog(&server.base_url()),
        PROVIDER,
        support::bearer_credentials(),
    );
    let (mock, slot) = support::mount_capture(&server, PATH, body);

    let response = client
        .complete(request)
        .await
        .expect("the completion should succeed");

    mock.assert_async().await;
    (support::captured(&slot), response)
}

/// Streams one request against a canned SSE transcript.
///
/// Returns the captured wire request and every serialized stream item, in
/// order, including a terminal `Err` when the stream failed.
async fn stream(request: Request, frames: &[&str]) -> (WireCapture, Vec<Value>) {
    let server = MockServer::start_async().await;
    let client = support::client_for(
        plain_catalog(&server.base_url()),
        PROVIDER,
        support::bearer_credentials(),
    );
    let transcript = support::sse_data_transcript(frames);
    let (mock, slot) = support::mount_capture_sse(&server, PATH, &transcript);

    let stream = client
        .stream(request)
        .await
        .expect("the stream should be accepted");
    let events = support::collect_stream_events(stream).await;

    mock.assert_async().await;
    (support::captured(&slot), events)
}

/// Sends one request the codec must refuse and returns the error.
///
/// The refusal happens while encoding, so the mock must record no calls at
/// all. A codec that dropped or substituted the offending part instead would
/// dispatch a request the caller never wrote.
async fn refusal(request: Request) -> Error {
    refusal_with(plain_catalog, request).await
}

/// [`refusal`] against a catalog of the caller's choosing.
async fn refusal_with(catalog: CatalogFor, request: Request) -> Error {
    let server = MockServer::start_async().await;
    let client = support::client_for(
        catalog(&server.base_url()),
        PROVIDER,
        support::bearer_credentials(),
    );
    let (mock, _slot) = support::mount_capture(&server, PATH, &text_response());

    let error = client
        .complete(request)
        .await
        .expect_err("the codec should refuse this request");

    mock.assert_calls_async(0).await;
    error
}

/// Sends the base request against an HTTP failure and returns the error.
async fn failure(status: u16, retry_after: Option<&str>, body: &Value) -> Error {
    let server = MockServer::start_async().await;
    let client = support::client_for(
        plain_catalog(&server.base_url()),
        PROVIDER,
        support::bearer_credentials(),
    );
    let mock = server.mock(|when, then| {
        when.method(Method::POST).path(PATH);
        let then = then
            .status(status)
            .header("content-type", "application/json")
            .json_body(body.clone());
        // `Then` is a builder whose methods take and return it by value, so the
        // optional header is added by rebuilding rather than by mutation.
        match retry_after {
            Some(retry_after) => then.header("retry-after", retry_after),
            None => then,
        };
    });

    let error = client
        .complete(support::base_request(&selector()))
        .await
        .expect_err("the call should fail");

    mock.assert_async().await;
    error
}

/// One error as the serializable projection applications receive.
fn error_json(error: &Error) -> Value {
    serde_json::to_value(error.data()).expect("error data should serialize")
}

/// The completed response of a stream, as a JSON value.
fn completed(events: &[Value]) -> &Value {
    events
        .iter()
        .find(|event| event.get("type").and_then(Value::as_str) == Some("completed"))
        .and_then(|event| event.get("response"))
        .expect("the stream should carry one completed response")
}

// ===========================================================================
// The shared corpus
// ===========================================================================

#[tokio::test]
async fn encodes_the_base_request() {
    let (wire, response) = exchange(support::base_request(&selector()), &text_response()).await;

    crate::json_snapshot!(wire);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn encodes_a_multi_turn_conversation() {
    let (wire, response) =
        exchange(support::multi_turn_request(&selector()), &text_response()).await;

    crate::json_snapshot!(wire);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn encodes_tools_and_decodes_a_tool_call() {
    let (wire, response) = exchange(
        support::tools_request(&selector(), None),
        &tool_call_response(),
    )
    .await;

    crate::json_snapshot!(wire);
    crate::json_snapshot!(response);
}

// The four tool-choice fixtures pin the request only. They send the same tool
// set as `encodes_tools_and_decodes_a_tool_call` against the same canned body,
// so a second copy of that decoded response in four more snapshots would pin
// nothing the first one does not already pin.

#[tokio::test]
async fn encodes_tool_choice_auto() {
    let (wire, _) = exchange(
        support::tools_request(&selector(), Some(ToolChoice::Auto)),
        &tool_call_response(),
    )
    .await;

    crate::json_snapshot!(wire);
}

#[tokio::test]
async fn encodes_tool_choice_none() {
    // Unlike Anthropic, this dialect keeps sending the tool definitions and
    // spells the refusal as `tool_choice: "none"`.
    let (wire, _) = exchange(
        support::tools_request(&selector(), Some(ToolChoice::None)),
        &text_response(),
    )
    .await;

    crate::json_snapshot!(wire);
}

#[tokio::test]
async fn encodes_tool_choice_required() {
    let (wire, _) = exchange(
        support::tools_request(&selector(), Some(ToolChoice::Required)),
        &tool_call_response(),
    )
    .await;

    crate::json_snapshot!(wire);
}

#[tokio::test]
async fn encodes_tool_choice_named() {
    let (wire, _) = exchange(
        support::tools_request(
            &selector(),
            Some(ToolChoice::Tool {
                name: "get_weather".to_owned(),
            }),
        ),
        &tool_call_response(),
    )
    .await;

    crate::json_snapshot!(wire);
}

#[tokio::test]
async fn encodes_a_tool_round_trip() {
    // A tool result is its own wire message here, so the two results of one
    // canonical tool turn expand into two `role: "tool"` messages. The
    // protocol has no error marker on a tool result at all: the failed
    // Madrid result is indistinguishable from the successful Paris one once
    // it reaches the wire, which is why `is_error` disappears from the
    // request snapshot. The loss is reported rather than silent — the decoded
    // response carries an `unsupported_control` warning naming it.
    let (wire, response) = exchange(
        support::tool_round_trip_request(&selector()),
        &text_response(),
    )
    .await;

    crate::json_snapshot!(wire);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn encodes_replayed_reasoning_as_reasoning_content() {
    // Kimi and DeepSeek want the assistant's own reasoning back when a turn
    // continues, so the signed block replays into `reasoning_content`. There
    // is no field for a signature and none for redacted reasoning, so the
    // signature is dropped and the redacted block is skipped rather than sent
    // as ordinary reasoning text.
    let (wire, response) = exchange(
        support::reasoning_round_trip_request(&selector()),
        &text_response(),
    )
    .await;

    crate::json_snapshot!(wire);
    crate::json_snapshot!(response);
}

/// The capabilities of a Chat Completions row that takes no forced tool
/// choice, as OpenRouter's Claude Fable 5.1 row does.
const NO_FORCED_CHOICE_CAPABILITIES: &str = "{ text = true, images = true, tools = true, \
                                             forced_tool_choice = false, structured_output = \
                                             true, reasoning = true, caching = true, \
                                             cache_breakpoints = true }";

#[tokio::test]
async fn refuses_a_forced_tool_choice_the_model_rejects_before_dispatch() {
    // The gate lives in the client, so an aggregator row that denies forced
    // tool choice refuses it on this dialect exactly as the native Anthropic
    // codec does: typed, unretried, and before any HTTP request.
    let catalog = |base_url: &str| {
        provider()
            .with_capabilities(NO_FORCED_CHOICE_CAPABILITIES)
            .catalog(base_url)
    };
    let error = refusal_with(
        catalog,
        support::tools_request(
            &selector(),
            Some(ToolChoice::Tool {
                name: "get_weather".to_owned(),
            }),
        ),
    )
    .await;

    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    assert_eq!(error.retry_classification(), RetryClassification::Never);
    crate::json_snapshot!(error_json(&error));

    // `auto` on the same row still reaches the wire unchanged.
    let (wire, _) = exchange_with(
        catalog,
        support::tools_request(&selector(), Some(ToolChoice::Auto)),
        &tool_call_response(),
    )
    .await;
    assert_eq!(wire.body["tool_choice"], json!("auto"));
}

#[tokio::test]
async fn rejects_a_custom_tool_before_dispatch() {
    // A codec that quietly downgraded the custom tool to a function tool would
    // send a request the model cannot answer.
    let error = refusal(support::custom_tool_request(&selector())).await;

    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    crate::json_snapshot!(error_json(&error));
}

#[tokio::test]
async fn encodes_both_image_sources() {
    // The corpus attachment requests each pair an image with a PDF, and this
    // dialect now refuses any request carrying a document, so neither can
    // reach the encoder. This hand-built request holds only the two image
    // sources, which is what the `image_url` shape has to be pinned against.
    //
    // Both sources produce the same part. A URL is sent as-is and keeps its
    // `detail`; inline bytes become a `data:` URL, because there is no
    // separate base64 shape in this protocol.
    let request = Request::builder()
        .model(selector())
        .message(Message::new(Role::User, [
            ContentPart::Text {
                text: "Describe both images.".to_owned(),
            },
            ContentPart::Image(ImageContent {
                source: MediaSource::url("https://example.com/cat.png"),
                detail: Some("high".to_owned()),
            }),
            ContentPart::Image(ImageContent {
                source: MediaSource::base64("aW1hZ2UtYnl0ZXM=", "image/png"),
                detail: None,
            }),
        ]))
        .max_output_tokens(128)
        .build()
        .expect("the image request should build");

    let (wire, response) = exchange(request, &text_response()).await;

    let parts = &wire.body["messages"][0]["content"];
    assert_eq!(parts[1]["image_url"]["url"], "https://example.com/cat.png");
    assert_eq!(parts[1]["image_url"]["detail"], "high");
    assert_eq!(
        parts[2]["image_url"]["url"], "data:image/png;base64,aW1hZ2UtYnl0ZXM=",
        "inline bytes become a data: URL"
    );
    crate::json_snapshot!(wire);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn refuses_a_url_document_before_dispatch() {
    // This protocol carries no document part. The codec refuses the request
    // rather than dropping the PDF, and rather than substituting placeholder
    // text for it — a dropped attachment is silent data loss, and a
    // substituted one is worse, because a model would answer the placeholder
    // as if the caller had written it.
    let error = refusal(support::url_attachments_request(&selector())).await;

    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    crate::json_snapshot!(error_json(&error));
}

#[tokio::test]
async fn refuses_an_inline_document_before_dispatch() {
    // The refusal is about the part, not the source, so inline bytes are
    // refused the same way a URL is.
    let error = refusal(support::inline_attachments_request(&selector())).await;

    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    crate::json_snapshot!(error_json(&error));
}

/// INTENTIONAL DIFFERENCE: the reference encoded audio for this dialect as an
/// `input_audio` part whatever the skin behind it accepted, and a skin that
/// did not silently dropped it or answered a placeholder. The protocol as the
/// compatible skins implement it carries no audio, so the codec refuses the
/// request instead. A catalog entry that claims `audio` can still let a
/// specific skin take it.
#[tokio::test]
async fn refuses_audio_before_dispatch() {
    let error = refusal(support::audio_request(&selector())).await;

    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    crate::json_snapshot!(error_json(&error));
}

#[tokio::test]
async fn a_truthful_catalog_refuses_a_document_before_the_codec_does() {
    // Two layers refuse a document, and this pins the outer one. A catalog
    // that describes this dialect truthfully sets `documents = false`, so the
    // client refuses at request validation and the codec never runs. The
    // messages differ, which is how a reader tells the layers apart: this one
    // names the model, the codec's names the provider. Audio is gated
    // identically.
    let server = MockServer::start_async().await;
    let client = support::client_for(
        truthful_catalog(&server.base_url()),
        PROVIDER,
        support::bearer_credentials(),
    );
    let (mock, _slot) = support::mount_capture(&server, PATH, &text_response());

    let error = client
        .complete(support::url_attachments_request(&selector()))
        .await
        .expect_err("a document should be refused");

    mock.assert_calls_async(0).await;
    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    crate::json_snapshot!(error_json(&error));
}

#[tokio::test]
async fn encodes_a_json_object_response_format() {
    let (wire, response) = exchange(
        support::response_format_request(&selector(), ResponseFormat::JsonObject),
        &text_response(),
    )
    .await;

    crate::json_snapshot!(wire);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn encodes_a_json_schema_response_format() {
    // The schema is wrapped in a named `json_schema` object and marked strict.
    let (wire, response) = exchange(
        support::response_format_request(&selector(), support::json_schema_format()),
        &text_response(),
    )
    .await;

    crate::json_snapshot!(wire);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn encodes_sampling_controls_and_stop_sequences() {
    let (wire, response) = exchange(support::sampling_request(&selector()), &text_response()).await;

    let stop = wire
        .body
        .get("stop")
        .expect("stop sequences should be sent");
    assert_eq!(stop, &json!(["END", "STOP"]), "stop keeps corpus order");
    // `Request` stores sampling controls as `f32`, and JSON carries `f64`.
    // Widening with `.into()` would put the binary32 value on the wire, so a
    // caller's 0.7 would reach the provider as 0.699999988079071. The shared
    // `common::sampling` helper renders the shortest decimal that identifies
    // the `f32` instead, which is what the snapshot pins.
    assert_eq!(wire.body["temperature"], json!(0.7));
    assert_eq!(wire.body["top_p"], json!(0.9));
    crate::json_snapshot!(wire);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn reports_request_metadata_rather_than_sending_it() {
    // Only OpenAI itself documents `metadata` on this endpoint, and a strict
    // skin rejects the whole request over one field it does not know. The tags
    // are dropped and the loss is reported, which is how this crate treats
    // every control a protocol cannot carry.
    let (wire, response) = exchange(support::metadata_request(&selector()), &text_response()).await;

    assert!(
        wire.body.get("metadata").is_none(),
        "metadata must not reach a compatible skin"
    );
    let codes: Vec<&str> = response
        .warnings
        .iter()
        .map(|warning| warning.code.as_str())
        .collect();
    assert_eq!(codes, ["unsupported_control"]);
    crate::json_snapshot!(wire);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn encodes_reasoning_effort_untranslated() {
    // The canonical level names are this dialect's own vocabulary, so the
    // effort passes straight through. Dropping it silently would send a
    // reasoning request the provider answers at its default depth.
    let request = Request::builder()
        .model(selector())
        .user("Hello")
        .reasoning_effort(ReasoningEffort::Xhigh)
        .max_output_tokens(128)
        .build()
        .expect("the effort request should build");

    let (wire, _) = exchange(request, &text_response()).await;

    assert_eq!(wire.body["reasoning_effort"], json!("xhigh"));
    crate::json_snapshot!(wire);
}

#[tokio::test]
async fn sends_a_developer_message_as_a_system_message() {
    // OpenAI itself accepts `developer`; the skins behind this dialect accept
    // system, user, assistant, and tool, and reject anything else. A developer
    // instruction therefore travels as the closest role every skin knows.
    let request = Request::builder()
        .model(selector())
        .message(Message::text(Role::Developer, "Keep it short."))
        .user("Hello")
        .max_output_tokens(128)
        .build()
        .expect("the developer request should build");

    let (wire, _) = exchange(request, &text_response()).await;

    assert_eq!(wire.body["messages"][0]["role"], json!("system"));
    crate::json_snapshot!(wire);
}

#[tokio::test]
async fn a_json_only_tool_result_sends_the_bare_value() {
    // This protocol takes a string for a tool result. A result made only of
    // JSON parts sends the value itself; sending the `ContentPart` envelope
    // would hand the model the crate's own wrapper to reason about.
    let request = Request::builder()
        .model(selector())
        .user("What is the weather in Paris?")
        .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
            ToolCall::function("call_paris", "get_weather", json!({ "city": "Paris" })),
        )]))
        .message(Message::new(Role::Tool, [ContentPart::ToolResult(
            ToolResult {
                tool_call_id: "call_paris".to_owned(),
                name:         Some("get_weather".to_owned()),
                content:      vec![ContentPart::Json {
                    value: json!({ "city": "Paris", "temp_c": 18 }),
                }],
                is_error:     false,
            },
        )]))
        .max_output_tokens(128)
        .build()
        .expect("the json result request should build");

    let (wire, _) = exchange(request, &text_response()).await;

    assert_eq!(
        wire.body["messages"][2]["content"],
        json!(r#"{"city":"Paris","temp_c":18}"#)
    );
    crate::json_snapshot!(wire);
}

// ===========================================================================
// Prompt-cache breakpoints
// ===========================================================================

#[tokio::test]
async fn a_cacheable_model_gets_anthropic_style_breakpoints() {
    // Aggregators forward `cache_control` to an upstream Anthropic model. Two
    // breakpoints land: the last system message, and the second-to-last user
    // turn, so each iteration of an agent loop reads the prefix the previous
    // one wrote. Marking a message converts its plain-string content into the
    // one-part array form, which is the only shape that can carry the
    // annotation; every unmarked message keeps the plain string.
    let (wire, _) = exchange(support::multi_turn_request(&selector()), &text_response()).await;

    let messages = &wire.body["messages"];
    assert_eq!(
        messages[0]["content"][0]["cache_control"],
        json!({ "type": "ephemeral" })
    );
    assert_eq!(
        messages[1]["content"][0]["cache_control"],
        json!({ "type": "ephemeral" }),
        "the second-to-last user turn"
    );
    assert!(
        messages[3]["content"].is_string(),
        "the newest user turn is not a reusable prefix"
    );
    // The body itself is pinned by `encodes_a_multi_turn_conversation`, which
    // sends the same request; a second copy here would pin nothing new.
}

#[tokio::test]
async fn a_model_without_caching_gets_no_breakpoints() {
    // The catalog decides. A model that cannot cache would reject the
    // annotation, so the request keeps the plain-string content form every
    // skin accepts.
    let server = MockServer::start_async().await;
    let client = support::client_for(
        provider()
            .with_capabilities(NO_CACHING_CAPABILITIES)
            .catalog(&server.base_url()),
        PROVIDER,
        support::bearer_credentials(),
    );
    let (mock, slot) = support::mount_capture(&server, PATH, &text_response());

    client
        .complete(support::multi_turn_request(&selector()))
        .await
        .expect("the completion should succeed");

    mock.assert_async().await;
    let wire = support::captured(&slot);
    assert!(
        !wire.body.to_string().contains("cache_control"),
        "a model without caching gets no breakpoints"
    );
    crate::json_snapshot!(wire);
}

/// INTENTIONAL DIFFERENCE: the reference keyed raw provider options by the
/// adapter name, so every skin behind this dialect shared one
/// `openai_compatible` namespace and an option meant for one gateway leaked to
/// all of them. Options here are keyed by the catalog provider id, per the
/// approved request-controls plan, so `compat` and a failover candidate each
/// see only their own.
#[tokio::test]
async fn sends_only_the_selected_provider_option_namespace() {
    let (wire, response) = exchange(
        support::provider_options_request(&selector(), PROVIDER, "openai"),
        &text_response(),
    )
    .await;

    let body = wire.body.as_object().expect("a JSON object body");
    assert_eq!(
        body.get("service_tier"),
        Some(&json!("flex")),
        "the selected namespace reaches the wire"
    );
    assert!(
        !body.contains_key("unreachable_option"),
        "the failover namespace must leave no trace"
    );
    assert!(
        !body.contains_key("auto_cache"),
        "a control key is consumed, never sent"
    );
    crate::json_snapshot!(wire);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn a_raw_provider_option_overrides_a_generated_field() {
    // The shared corpus cannot show an override for this dialect: its raw
    // option is spelled `max_output_tokens`, which is the OpenAI Responses
    // field name, while Chat Completions caps output with `max_tokens`. The
    // corpus option is therefore sent verbatim as an extra key, which is the
    // escape hatch working as designed but not an override. This fixture
    // builds the override case by hand.
    let request = Request::builder()
        .model(selector())
        .user("Hello")
        .temperature(0.2)
        .max_output_tokens(128)
        .provider_option(PROVIDER, "temperature", json!(0.9))
        .provider_option(PROVIDER, "max_tokens", json!(512))
        .build()
        .expect("the override request should build");

    let (wire, _) = exchange(request, &text_response()).await;

    assert_eq!(wire.body["temperature"], json!(0.9));
    assert_eq!(wire.body["max_tokens"], json!(512));
    crate::json_snapshot!(wire);
}

#[tokio::test]
async fn replays_provider_native_content_losslessly() {
    // An opaque part of this dialect names the assistant message field it
    // replays into, so `openai_compatible.reasoning` is written back as
    // `reasoning`. The corpus fills that field with an object because it is
    // shared across dialects; a real skin puts a string there and uses
    // `reasoning_details` for structured replay. The contract this pins is
    // that the payload survives byte for byte, whatever shape it has.
    let (wire, response) = exchange(
        support::replay_request(&selector(), "openai_compatible"),
        &text_response(),
    )
    .await;

    let assistant = &wire.body["messages"][1];
    assert_eq!(
        assistant["tool_calls"][0]["function"]["arguments"],
        json!("{\"city\": \"Paris\"}"),
        "the provider's own argument text replays byte for byte"
    );
    assert!(
        assistant.get("other_provider").is_none(),
        "another provider's opaque part must be ignored, not replayed"
    );
    crate::json_snapshot!(wire);
    crate::json_snapshot!(response);
}

// ===========================================================================
// The structured reasoning channel
// ===========================================================================

#[tokio::test]
async fn reasoning_details_survive_a_complete_response() {
    // OpenRouter puts signed or encrypted reasoning in `reasoning_details`,
    // and the upstream model rejects a continued turn that does not send it
    // back. The array is kept verbatim as one opaque part: only the model that
    // wrote the encrypted members can read them, so nothing is normalized out.
    let details = json!([
        { "type": "reasoning.encrypted", "id": "rs-1", "data": "AQ==", "index": 0 },
        { "type": "reasoning.text", "text": "Two cities to look up.", "index": 1 },
    ]);
    let body = json!({
        "id": "chatcmpl-details",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello.",
                "reasoning_details": details,
            },
            "finish_reason": "stop",
        }],
        "usage": { "prompt_tokens": 11, "completion_tokens": 5 },
    });

    let (_, response) = exchange(support::base_request(&selector()), &body).await;

    let part = response.content.first().expect("an opaque part");
    match part {
        ContentPart::Opaque { kind, data } => {
            assert_eq!(kind, "openai_compatible.reasoning_details");
            assert_eq!(data, &details, "the payload survives byte for byte");
        }
        other => panic!("expected an opaque part, got {other:?}"),
    }
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn a_replayed_reasoning_details_part_returns_to_its_own_field() {
    // The decode and encode halves are one contract: an opaque part named
    // after the message field it came from replays into that field. Without
    // this round trip the aggregator sees an unsigned turn.
    let details = json!([{ "type": "reasoning.encrypted", "id": "rs-1", "data": "AQ==" }]);
    let request = Request::builder()
        .model(selector())
        .user("What is the weather in Paris?")
        .message(Message::new(Role::Assistant, [
            ContentPart::opaque("openai_compatible.reasoning_details", details.clone()),
            ContentPart::Text {
                text: "Looking it up.".to_owned(),
            },
        ]))
        .user("Thanks.")
        .max_output_tokens(128)
        .build()
        .expect("the replay request should build");

    let (wire, _) = exchange(request, &text_response()).await;

    assert_eq!(wire.body["messages"][1]["reasoning_details"], details);
    crate::json_snapshot!(wire);
}

#[tokio::test]
async fn streamed_reasoning_details_coalesce_before_the_part_is_built() {
    // A stream splits each logical detail across chunks and interleaves two of
    // them. Fragments are matched by `type` and `index`, so the two details
    // rebuild as two entries rather than four, and the text members
    // concatenate in wire order. Half a signature replayed upstream is
    // rejected, so the coalescing is what makes the stream path usable at all.
    let (_, events) = stream(
        support::base_request(&selector()),
        &[
            r#"{"id":"chatcmpl-stream","choices":[{"index":0,"delta":{"role":"assistant","reasoning_details":[{"type":"reasoning.text","index":0,"text":"Two cities "}]}}]}"#,
            r#"{"id":"chatcmpl-stream","choices":[{"index":0,"delta":{"reasoning_details":[{"type":"reasoning.text","index":1,"text":"and one "}]}}]}"#,
            r#"{"id":"chatcmpl-stream","choices":[{"index":0,"delta":{"reasoning_details":[{"type":"reasoning.text","index":0,"text":"to look up."}]}}]}"#,
            r#"{"id":"chatcmpl-stream","choices":[{"index":0,"delta":{"reasoning_details":[{"type":"reasoning.text","index":1,"text":"time zone.","signature":"sig-1"}]}}]}"#,
            r#"{"id":"chatcmpl-stream","choices":[{"index":0,"delta":{"content":"Hello."},"finish_reason":"stop"}]}"#,
            "[DONE]",
        ],
    )
    .await;

    support::assert_stream_contract(&events);

    let response = completed(&events);
    let content = response["content"]
        .as_array()
        .expect("the completed content");
    assert_eq!(content[0]["type"], "opaque");
    assert_eq!(content[0]["kind"], "openai_compatible.reasoning_details");
    assert_eq!(
        content[0]["data"],
        json!([
            { "type": "reasoning.text", "index": 0, "text": "Two cities to look up." },
            {
                "type": "reasoning.text",
                "index": 1,
                "text": "and one time zone.",
                "signature": "sig-1",
            },
        ])
    );
    crate::json_snapshot!(events);
}

// ===========================================================================
// Usage
// ===========================================================================

#[tokio::test]
async fn inclusive_usage_decodes_into_disjoint_buckets() {
    // Every counter this protocol reports is inclusive: `prompt_tokens`
    // already contains both cache details, and `completion_tokens` already
    // contains the reasoning detail. The decoder subtracts them so the five
    // canonical buckets are disjoint and `total` is their plain sum.
    let body = json!({
        "id": "chatcmpl-usage",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "ok" },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 200,
            "completion_tokens": 10,
            "total_tokens": 210,
            "prompt_tokens_details": { "cached_tokens": 50, "cache_write_tokens": 100 },
            "completion_tokens_details": { "reasoning_tokens": 4 },
        },
    });

    let (_, response) = exchange(support::base_request(&selector()), &body).await;

    assert_eq!(response.usage.input, 50, "200 - 50 cached - 100 written");
    assert_eq!(response.usage.cache_read, 50);
    assert_eq!(response.usage.cache_write, 100);
    assert_eq!(response.usage.output, 6, "10 - 4 reasoning");
    assert_eq!(response.usage.reasoning, 4);
    assert_eq!(response.usage.total(), 210, "the buckets are disjoint");
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn flat_usage_spellings_decode_the_same_way() {
    // DeepSeek reports its automatic cache hits as a flat
    // `prompt_cache_hit_tokens`, and Modal reports reasoning as a flat
    // `reasoning_tokens`. Both are inclusive in exactly the same way as the
    // nested details.
    let body = json!({
        "id": "chatcmpl-usage",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "ok" },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 53,
            "completion_tokens": 66,
            "total_tokens": 119,
            "prompt_cache_hit_tokens": 41,
            "prompt_cache_miss_tokens": 12,
            "reasoning_tokens": 54,
        },
    });

    let (_, response) = exchange(support::base_request(&selector()), &body).await;

    assert_eq!(response.usage.input, 12, "53 - 41 cache hits");
    assert_eq!(response.usage.cache_read, 41);
    assert_eq!(response.usage.output, 12, "66 - 54 reasoning");
    assert_eq!(response.usage.reasoning, 54);
    assert_eq!(response.usage.total(), 119);
    crate::json_snapshot!(response);
}

// ===========================================================================
// In-band cost
// ===========================================================================

#[tokio::test]
async fn openrouter_reports_cost_in_band() {
    let body = json!({
        "id": "chatcmpl-cost",
        "provider": "Fireworks",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "ok" },
            "finish_reason": "stop",
        }],
        "usage": { "prompt_tokens": 200, "completion_tokens": 10, "cost": 0.0042 },
    });

    let (_, response) = exchange(support::base_request(&selector()), &body).await;

    let cost = response.cost.expect("a provider-reported cost");
    assert_eq!(cost.usd_micros, 4200);
    assert_eq!(cost.source, CostSource::Provider);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn venice_reports_a_top_level_cost() {
    let body = json!({
        "id": "chatcmpl-cost",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "ok" },
            "finish_reason": "stop",
        }],
        "usage": { "prompt_tokens": 4, "completion_tokens": 2 },
        // `diem` is Venice's own currency and is deliberately not read.
        "cost": { "usd": 0.25, "diem": 9.5 },
    });

    let (_, response) = exchange(support::base_request(&selector()), &body).await;

    let cost = response.cost.expect("a provider-reported cost");
    assert_eq!(cost.usd_micros, 250_000);
    assert_eq!(cost.source, CostSource::Provider);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn usage_cost_wins_over_a_top_level_cost() {
    let body = json!({
        "id": "chatcmpl-cost",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "ok" },
            "finish_reason": "stop",
        }],
        "usage": { "prompt_tokens": 4, "completion_tokens": 2, "cost": 0.001 },
        "cost": { "usd": 0.25 },
    });

    let (_, response) = exchange(support::base_request(&selector()), &body).await;

    let cost = response.cost.expect("a provider-reported cost");
    assert_eq!(cost.usd_micros, 1000, "usage.cost has precedence");
    assert_eq!(cost.source, CostSource::Provider);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn a_catalog_estimate_never_overwrites_a_provider_cost() {
    let body = json!({
        "id": "chatcmpl-cost",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "ok" },
            "finish_reason": "stop",
        }],
        "usage": { "prompt_tokens": 200, "completion_tokens": 10, "cost": 0.0042 },
    });

    let (_, response) =
        exchange_with(priced_catalog, support::base_request(&selector()), &body).await;

    // The catalog would price this call at 200 + 20 = 220 micros. The
    // provider's own figure is authoritative and stands.
    let cost = response.cost.expect("a provider-reported cost");
    assert_eq!(cost.usd_micros, 4200);
    assert_eq!(cost.source, CostSource::Provider);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn a_catalog_estimate_fills_in_a_missing_provider_cost() {
    let body = json!({
        "id": "chatcmpl-cost",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "ok" },
            "finish_reason": "stop",
        }],
        "usage": { "prompt_tokens": 200, "completion_tokens": 10 },
    });

    let (_, response) =
        exchange_with(priced_catalog, support::base_request(&selector()), &body).await;

    let cost = response.cost.expect("a catalog estimate");
    assert_eq!(cost.usd_micros, 220, "200 input at 1 plus 10 output at 2");
    assert_eq!(cost.source, CostSource::Catalog);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn unmodeled_provider_fields_survive_only_in_raw() {
    // This is the escape-hatch fixture, and it matters more here than in any
    // other dialect. Skins decorate the envelope with fields this crate
    // deliberately does not model, and `raw` is the only place they survive.
    let body = json!({
        "id": "chatcmpl-raw",
        "provider": "Fireworks",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "ok" },
            "finish_reason": "stop",
            "native_finish_reason": "eos",
        }],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 2,
            // An upstream figure, not what this call is billed. It is not a
            // cost this crate reports.
            "cost_details": { "upstream_inference_cost": 0.5 },
            "prompt_tokens_details": { "audio_tokens": 3 },
        },
    });

    let (_, response) = exchange(support::base_request(&selector()), &body).await;

    assert!(
        response.cost.is_none(),
        "an upstream inference cost is not a provider-reported cost"
    );
    let raw = response.raw.as_ref().expect("the untouched success body");
    assert_eq!(raw["usage"]["cost_details"]["upstream_inference_cost"], 0.5);
    assert_eq!(raw["usage"]["prompt_tokens_details"]["audio_tokens"], 3);
    assert_eq!(raw["provider"], "Fireworks");
    assert_eq!(raw["choices"][0]["native_finish_reason"], "eos");
    crate::json_snapshot!(response);
}

// ===========================================================================
// Streaming
// ===========================================================================

#[tokio::test]
async fn streaming_always_asks_for_usage() {
    // Without `stream_options.include_usage` no compatible skin sends a usage
    // chunk at all, and a streamed call would report no tokens. It is
    // therefore unconditional rather than opt-in.
    let (wire, events) = stream(
        support::base_request(&selector()),
        &[
            r#"{"id":"chatcmpl-stream","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello."},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-stream","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            "[DONE]",
        ],
    )
    .await;

    assert_eq!(wire.body["stream"], json!(true));
    assert_eq!(
        wire.body["stream_options"],
        json!({ "include_usage": true })
    );
    support::assert_stream_contract(&events);
    crate::json_snapshot!(wire);
    crate::json_snapshot!(events);
}

#[tokio::test]
async fn streams_reasoning_text_and_interleaved_tool_calls() {
    // INTENTIONAL DIFFERENCE from the reference implementation. Fabro emits no
    // reasoning stream events for this dialect at all: it accumulates
    // `delta.reasoning` and `delta.reasoning_content` silently and only
    // reveals them on the final response, so a caller streaming a reasoning
    // model sees nothing until the end. We open a real reasoning block and
    // emit `reasoning_delta` events, the same as every other dialect. Both
    // spellings feed the one block, because a skin may switch between them.
    let (wire, events) = stream(
        support::tools_request(&selector(), None),
        &[
            r#"{"id":"chatcmpl-stream","object":"chat.completion.chunk","created":1700000000,"choices":[{"index":0,"delta":{"role":"assistant","reasoning":"Two cities "},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-stream","choices":[{"index":0,"delta":{"reasoning_content":"to look up."}}]}"#,
            r#"{"id":"chatcmpl-stream","choices":[{"index":0,"delta":{"content":"Looking "}}]}"#,
            r#"{"id":"chatcmpl-stream","choices":[{"index":0,"delta":{"content":"both up."}}]}"#,
            r#"{"id":"chatcmpl-stream","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_paris","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}"#,
            r#"{"id":"chatcmpl-stream","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_madrid","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}"#,
            r#"{"id":"chatcmpl-stream","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]}}]}"#,
            r#"{"id":"chatcmpl-stream","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\"city\":"}}]}}]}"#,
            r#"{"id":"chatcmpl-stream","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Paris\"}"}}]}}]}"#,
            r#"{"id":"chatcmpl-stream","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\"Madrid\"}"}}]}}]}"#,
            r#"{"id":"chatcmpl-stream","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
            "[DONE]",
        ],
    )
    .await;

    support::assert_stream_contract(&events);

    // The protocol carries no block ids, so the decoder synthesizes them from
    // the accumulation slot. Two interleaved calls must stay two blocks.
    let blocks: Vec<&str> = events
        .iter()
        .filter(|event| event.get("type").and_then(Value::as_str) == Some("content_block_start"))
        .filter_map(|event| event.get("id").and_then(Value::as_str))
        .collect();
    assert_eq!(blocks, ["reasoning-0", "block-0", "tool-0", "tool-1"]);

    let response = completed(&events);
    let content = response["content"]
        .as_array()
        .expect("the completed content");
    assert_eq!(content[0]["type"], "reasoning");
    assert_eq!(content[0]["text"], "Two cities to look up.");
    // The block id and the provider's tool-call id are different things, and
    // each call keeps its own.
    assert_eq!(content[2]["id"], "call_paris");
    assert_eq!(content[2]["arguments"], json!({ "city": "Paris" }));
    assert_eq!(content[3]["id"], "call_madrid");
    assert_eq!(content[3]["arguments"], json!({ "city": "Madrid" }));

    crate::json_snapshot!(wire);
    crate::json_snapshot!(events);
}

#[tokio::test]
async fn a_final_empty_choices_chunk_carries_usage_and_cost() {
    // Usage arrives in its own trailing chunk whose `choices` array is empty.
    // A decoder that reads usage only from a chunk with a choice would report
    // no tokens for every streamed call.
    let (_, events) = stream(
        support::base_request(&selector()),
        &[
            r#"{"id":"chatcmpl-stream","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}"#,
            r#"{"id":"chatcmpl-stream","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"stop"}]}"#,
            r#"{"id":"chatcmpl-stream","choices":[],"usage":{"prompt_tokens":11,"completion_tokens":5,"total_tokens":16,"prompt_tokens_details":{"cached_tokens":3},"cost":0.0001}}"#,
            "[DONE]",
        ],
    )
    .await;

    support::assert_stream_contract(&events);

    let usage: Vec<&Value> = events
        .iter()
        .filter(|event| event.get("type").and_then(Value::as_str) == Some("usage"))
        .collect();
    assert_eq!(usage.len(), 1, "one cumulative snapshot");
    assert_eq!(usage[0]["usage"]["input"], 8, "11 - 3 cached");
    assert_eq!(usage[0]["usage"]["cache_read"], 3);

    let response = completed(&events);
    assert_eq!(response["usage"]["output"], 5);
    assert_eq!(
        response["cost"],
        json!({ "usd_micros": 100, "source": "provider" })
    );
    crate::json_snapshot!(events);
}

#[tokio::test]
async fn a_streamed_response_carries_no_raw_document() {
    // This protocol supplies no terminal response object, so there is nothing
    // to put in `raw`. It deliberately holds the provider's own document or
    // nothing, never a replayed log of stream events.
    let (_, events) = stream(
        support::base_request(&selector()),
        &[
            r#"{"id":"chatcmpl-stream","choices":[{"index":0,"delta":{"content":"Hello."},"finish_reason":"stop"}]}"#,
            "[DONE]",
        ],
    )
    .await;

    support::assert_stream_contract(&events);
    let response = completed(&events);
    assert!(
        response.get("raw").is_none(),
        "a streamed response must not synthesize a raw document"
    );
    crate::json_snapshot!(events);
}

#[tokio::test]
async fn a_stream_error_chunk_ends_the_stream() {
    let (_, events) = stream(
        support::base_request(&selector()),
        &[
            r#"{"id":"chatcmpl-stream","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}"#,
            r#"{"error":{"message":"rate limit reached","type":"rate_limit_error","code":"rate_limit_exceeded"}}"#,
        ],
    )
    .await;

    // `assert_stream_contract` enforces the half that matters: a failed stream
    // emits no `completed` event, so a caller can never mistake a truncated
    // answer for a whole one.
    support::assert_stream_contract(&events);
    let last = events.last().expect("the stream should produce items");
    assert_eq!(last["type"], "error");
    assert_eq!(last["error"]["provider_code"], "rate_limit_exceeded");
    assert!(
        !events
            .iter()
            .any(|event| event.get("type").and_then(Value::as_str) == Some("completed")),
        "a failed stream must not complete"
    );
    crate::json_snapshot!(events);
}

/// A stream the transport cuts off after content completes once, as
/// incomplete.
///
/// INTENTIONAL DIFFERENCE: the reference synthesized a `Stop` finish when
/// content had started and the stream ended without `[DONE]`, so a cut answer
/// read as a whole one. The completed response says it is incomplete instead.
#[tokio::test]
async fn a_stream_cut_off_after_content_completes_as_incomplete() {
    let (_, events) = stream(
        support::base_request(&selector()),
        &[
            r#"{"id":"chatcmpl-cut","choices":[{"index":0,"delta":{"role":"assistant","content":"Half an"}}]}"#,
            r#"{"id":"chatcmpl-cut","choices":[{"index":0,"delta":{"content":" answer"}}]}"#,
        ],
    )
    .await;

    support::assert_stream_contract(&events);
    let response = completed(&events);
    assert_eq!(
        response["finish_reason"],
        json!("incomplete"),
        "the provider never said why it stopped"
    );
    assert_eq!(response["content"][0]["text"], json!("Half an answer"));
    crate::json_snapshot!(events);
}

/// A stream the transport cuts off before any content still completes, as an
/// incomplete response with nothing in it.
///
/// INTENTIONAL DIFFERENCE: the reference emitted no terminal event at all
/// here, so a caller could not tell a cut stream from a consumer bug. Every
/// successful stream ends with exactly one `completed`.
#[tokio::test]
async fn a_stream_cut_off_before_content_completes_as_incomplete() {
    let (_, events) = stream(support::base_request(&selector()), &[
        r#"{"id":"chatcmpl-cut","choices":[{"index":0,"delta":{"role":"assistant"}}]}"#,
    ])
    .await;

    support::assert_stream_contract(&events);
    let response = completed(&events);
    assert_eq!(response["finish_reason"], json!("incomplete"));
    assert_eq!(response["content"], json!([]));
    crate::json_snapshot!(events);
}

#[tokio::test]
async fn venice_reports_a_top_level_cost_in_a_stream() {
    // Venice puts its cost beside `usage` in the trailing chunk, not inside
    // it, and prices in both dollars and its own currency. Only the dollars
    // are read, and they are provider-reported.
    let (_, events) = stream(
        support::base_request(&selector()),
        &[
            r#"{"id":"chatcmpl-cost","choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"#,
            r#"{"id":"chatcmpl-cost","choices":[],"usage":{"prompt_tokens":4,"completion_tokens":2},"cost":{"usd":0.25,"diem":9.5}}"#,
            "[DONE]",
        ],
    )
    .await;

    support::assert_stream_contract(&events);
    let response = completed(&events);
    assert_eq!(
        response["cost"],
        json!({ "usd_micros": 250_000, "source": "provider" })
    );
    assert_eq!(response["usage"]["input"], 4);
    crate::json_snapshot!(events);
}

// ===========================================================================
// Classified errors
// ===========================================================================

#[tokio::test]
async fn a_200_with_no_choices_fails_to_decode() {
    // Some skins answer a filtered or aborted call with a 200 whose `choices`
    // array is empty. Decoding that as an empty success would hand the caller
    // a finished response the model never wrote, so the body fails to decode
    // and the untouched document rides along for diagnosis.
    let server = MockServer::start_async().await;
    let client = support::client_for(
        plain_catalog(&server.base_url()),
        PROVIDER,
        support::bearer_credentials(),
    );
    let (mock, _slot) = support::mount_capture(
        &server,
        PATH,
        &json!({ "id": "chatcmpl-empty", "object": "chat.completion", "choices": [] }),
    );

    let error = client
        .complete(support::base_request(&selector()))
        .await
        .expect_err("an empty choices array is not a successful response");

    mock.assert_async().await;
    assert_eq!(error.kind(), ErrorKind::ResponseDecode);
    crate::json_snapshot!(error_json(&error));
}

#[tokio::test]
async fn a_request_timeout_expiring_mid_stream_is_never_retried() {
    // A raw socket stands in for a provider that starts streaming and then
    // stalls past the caller's request timeout; the mock server cannot stall
    // mid-body. The provider is already executing the call when the budget
    // expires, so the failure must keep the complete path's never-retry rule
    // instead of classifying as a retryable network fault.
    let listener = TcpListener::bind("127.0.0.1:0").expect("a local socket should bind");
    let base_url = format!(
        "http://{}",
        listener.local_addr().expect("the socket has an address")
    );
    thread::spawn(move || {
        use std::io::{Read as _, Write as _};
        if let Ok((mut socket, _)) = listener.accept() {
            let mut request = [0_u8; 8192];
            let _ = socket.read(&mut request);
            let _ = socket.write_all(
                b"HTTP/1.1 200 OK\r\n\
                  content-type: text/event-stream\r\n\
                  content-length: 65536\r\n\r\n\
                  data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
            );
            let _ = socket.flush();
            thread::sleep(Duration::from_secs(5));
        }
    });

    let client = support::client_for(
        plain_catalog(&base_url),
        PROVIDER,
        support::bearer_credentials(),
    );
    let request = Request::builder()
        .model(selector())
        .user("Hello")
        .timeout(Duration::from_millis(300))
        .build()
        .expect("the request should build");

    let mut stream = client
        .stream(request)
        .await
        .expect("the headers arrive before the timeout, so the stream opens");
    let mut terminal = None;
    while let Some(item) = stream.next().await {
        if let Err(error) = item {
            terminal = Some(error);
            break;
        }
    }

    let error = terminal.expect("the stalled stream should fail");
    assert_eq!(error.kind(), ErrorKind::Timeout);
    assert_eq!(error.retry_classification(), RetryClassification::Never);
}

#[tokio::test]
async fn classifies_a_rejected_credential() {
    let error = failure(
        401,
        None,
        &json!({ "error": {
            "message": "Incorrect API key provided",
            "type": "invalid_request_error",
            "code": "invalid_api_key",
        } }),
    )
    .await;

    assert_eq!(error.kind(), ErrorKind::Authentication);
    assert_eq!(error.retry_classification().delay(), None);
    crate::json_snapshot!(error_json(&error));
}

#[tokio::test]
async fn classifies_a_blocked_account() {
    let error = failure(
        403,
        None,
        &json!({ "error": {
            "message": "Your account has been deactivated",
            "type": "invalid_request_error",
            "code": "account_deactivated",
        } }),
    )
    .await;

    assert_eq!(error.kind(), ErrorKind::AccessDenied);
    crate::json_snapshot!(error_json(&error));
}

#[tokio::test]
async fn classifies_a_missing_model() {
    let error = failure(
        404,
        None,
        &json!({ "error": {
            "message": "The model `vendor/compat-chat-v2` does not exist",
            "type": "invalid_request_error",
            "code": "model_not_found",
        } }),
    )
    .await;

    assert_eq!(error.kind(), ErrorKind::NotFound);
    crate::json_snapshot!(error_json(&error));
}

#[tokio::test]
async fn classifies_throttling_with_an_http_date_retry_after() {
    // Providers may answer `Retry-After` with an HTTP-date rather than a
    // number of seconds. A far-future date exercises the date parser without
    // depending on the clock landing inside a narrow window.
    let error = failure(
        429,
        Some("Fri, 01 Jan 2100 00:00:00 GMT"),
        &json!({ "error": {
            "message": "Rate limit reached for requests",
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded",
        } }),
    )
    .await;

    assert_eq!(error.kind(), ErrorKind::RateLimit);
    let delay = error
        .retry_after()
        .expect("an HTTP-date Retry-After should parse into a delay");
    assert!(
        delay > Duration::from_secs(2_000_000_000),
        "the parsed delay should run to the year 2100, got {delay:?}"
    );

    // The delay counts down from the moment of the call, so it changes on
    // every run. Normalize it rather than pinning a value that can never
    // match twice.
    let mut data = error_json(&error);
    data["retry"] = json!({ "type": "after", "after_millis": "[COUNTDOWN]" });
    data["provider_retry_after_millis"] = json!("[COUNTDOWN]");
    crate::json_snapshot!(data);
}

#[tokio::test]
async fn classifies_a_spent_quota_as_not_retryable() {
    // A 429 normally means throttling, which backoff clears. This one reports
    // spent credit, which backoff never clears, so the classification must be
    // `quota_exceeded` and the retry advice must be `never`.
    let error = failure(
        429,
        Some("60"),
        &json!({ "error": {
            "message": "You exceeded your current quota, please check your plan and billing details",
            "type": "insufficient_quota",
            "code": "insufficient_quota",
        } }),
    )
    .await;

    assert_eq!(error.kind(), ErrorKind::QuotaExceeded);
    assert_eq!(
        error.retry_after(),
        None,
        "a Retry-After must not make a spent quota look retryable"
    );
    crate::json_snapshot!(error_json(&error));
}

#[tokio::test]
async fn classifies_a_provider_side_failure() {
    let error = failure(
        500,
        None,
        &json!({ "error": {
            "message": "The server had an error while processing your request",
            "type": "server_error",
            "code": Value::Null,
        } }),
    )
    .await;

    assert_eq!(error.kind(), ErrorKind::Server);
    assert_eq!(
        error.retry_classification().delay(),
        None,
        "a 5xx without Retry-After is safe to repeat on the caller's schedule"
    );
    crate::json_snapshot!(error_json(&error));
}

// ===========================================================================
// Token counting
// ===========================================================================

#[tokio::test]
async fn count_input_tokens_is_unavailable_and_sends_nothing() {
    let server = MockServer::start_async().await;
    let client = support::client_for(
        plain_catalog(&server.base_url()),
        PROVIDER,
        support::bearer_credentials(),
    );
    let (mock, _slot) = support::mount_capture(&server, PATH, &text_response());

    let count = client
        .count_input_tokens(support::base_request(&selector()))
        .await
        .expect("an absent count endpoint is not an error");

    // This dialect has no count endpoint. Reporting `None` is the honest
    // answer; falling back to a local estimate would hand the caller a number
    // no compatible skin agrees with.
    assert!(count.is_none());
    mock.assert_calls_async(0).await;
}
