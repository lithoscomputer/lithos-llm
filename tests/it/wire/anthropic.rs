//! Anthropic Messages wire parity.
//!
//! Every test here drives the public client API — `complete`, `stream`, and
//! `count_input_tokens` — against a local mock that captures the request the
//! codec produced. Both halves of the exchange are pinned: the captured wire
//! request, which is what we encoded, and the decoded canonical result, which
//! is what we decoded.
//!
//! # Intentional differences from the reference implementation
//!
//! Three are pinned here and marked at the fixture that pins them:
//!
//! - Content-block ids are stable and surfaced on every stream event.
//! - `redacted_thinking` survives the STREAMING path. The reference kept it in
//!   the blocking path and dropped it while streaming.
//! - A truncated stream still emits exactly one `completed`, and that
//!   completion says `incomplete` rather than `stop`. The reference emitted no
//!   terminal event at all, which left a caller with nothing to act on.

use httpmock::{Method, Mock, MockServer};
use lithos_llm::types::{
    ContentPart, ErrorKind, ImageContent, MediaSource, Message, ReasoningEffort, ResponseFormat,
    RetryClassification, Role, Speed, ToolCall, ToolChoice, ToolDefinition, ToolResult,
};
use lithos_llm::{Client, Request};
use serde_json::{Value, json};

use crate::support::{self, WireProvider};

/// The catalog provider id, which is also the provider-options namespace and
/// the opaque-part namespace this codec replays.
const PROVIDER: &str = "anthropic";

/// The catalog model id. It differs from the API model on purpose, so a
/// snapshot proves the wire carries the API id rather than the catalog id.
const MODEL: &str = "claude";

/// The provider's own model id, which is what `model` must carry on the wire.
const API_MODEL: &str = "claude-sonnet-4-6";

/// The generation endpoint, for both blocking and streaming calls.
const MESSAGES_PATH: &str = "/v1/messages";

/// The provider-authoritative input token count endpoint.
const COUNT_PATH: &str = "/v1/messages/count_tokens";

/// The namespace a second provider's options live in, which must never reach
/// the Anthropic wire.
const OTHER_PROVIDER: &str = "openai";

// ===========================================================================
// Catalog, client, and canned provider payloads
// ===========================================================================

/// The Anthropic provider every test in this file routes through.
fn anthropic() -> WireProvider<'static> {
    WireProvider::new(PROVIDER, "anthropic", "anthropic-messages", MODEL)
        .with_api_model(API_MODEL)
        .with_auth("{ type = \"header\", name = \"x-api-key\" }")
}

/// Builds a client pointed at `server` and returns it with the request
/// selector its catalog answers to.
fn client_for(server: &MockServer) -> (Client, String) {
    let provider = anthropic();
    let client = support::client_for(
        provider.catalog(&server.base_url()),
        PROVIDER,
        support::header_credentials("x-api-key"),
    );
    (client, provider.selector())
}

/// The ordinary text answer most encoding tests are answered with.
fn text_response() -> Value {
    json!({
        "id": "msg_01WireText",
        "type": "message",
        "role": "assistant",
        "model": API_MODEL,
        "content": [{ "type": "text", "text": "Hello back." }],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": { "input_tokens": 11, "output_tokens": 5 }
    })
}

/// A response that calls one tool, which is what the tool fixtures provoke.
fn tool_use_response() -> Value {
    json!({
        "id": "msg_01WireTool",
        "type": "message",
        "role": "assistant",
        "model": API_MODEL,
        "content": [
            { "type": "text", "text": "Looking it up." },
            {
                "type": "tool_use",
                "id": "toolu_01Weather",
                "name": "get_weather",
                "input": { "city": "Paris" }
            }
        ],
        "stop_reason": "tool_use",
        "stop_sequence": null,
        "usage": { "input_tokens": 96, "output_tokens": 41 }
    })
}

/// A response that answers with the structured document the caller asked for.
fn json_response() -> Value {
    json!({
        "id": "msg_01WireJson",
        "type": "message",
        "role": "assistant",
        "model": API_MODEL,
        "content": [{ "type": "text", "text": "{\"city\":\"Paris\",\"population\":2102650}" }],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": { "input_tokens": 24, "output_tokens": 18 }
    })
}

// ===========================================================================
// Local fixtures
// ===========================================================================
//
// These are dialect-specific and deliberately NOT in the shared corpus: each
// exercises an Anthropic-only contract that no other dialect has. The shared
// corpus is never edited to suit this file.

/// A request carrying all three prompt-cache anchors at once.
///
/// The system prompt, the last tool definition, and the conversation prefix
/// each take a breakpoint. No shared corpus entry carries all three, because
/// no other dialect places breakpoints this way.
fn cacheable_request(model: &str, auto_cache: Option<bool>) -> Request {
    let mut builder = Request::builder()
        .model(model)
        .system("Keep it short.")
        .user("First")
        .message(Message::text(Role::Assistant, "Answer"))
        .user("Second")
        .message(Message::text(Role::Assistant, "Answer"))
        .user("Third")
        .tool(ToolDefinition::function(
            "get_weather",
            "Reads the current weather for a city",
            json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"],
            }),
        ))
        .max_output_tokens(128);
    if let Some(auto_cache) = auto_cache {
        builder = builder.provider_option(PROVIDER, "auto_cache", json!(auto_cache));
    }
    builder.build().expect("the cacheable request should build")
}

/// A request whose raw options replace two fields the codec generated.
///
/// The shared `provider_options_request` cannot express this for Anthropic:
/// none of its keys — `service_tier`, `max_output_tokens`, `auto_cache` — is a
/// field the Anthropic codec generates for that request, so it can prove
/// namespace isolation but not override.
fn override_request(model: &str) -> Request {
    Request::builder()
        .model(model)
        .user("Hello")
        .temperature(0.2)
        .max_output_tokens(64)
        .provider_option(PROVIDER, "max_tokens", json!(2048))
        .provider_option(PROVIDER, "temperature", json!(0.9))
        .provider_option(PROVIDER, "thinking", json!({ "type": "enabled" }))
        .provider_option(PROVIDER, "auto_cache", json!(false))
        .build()
        .expect("the override request should build")
}

/// A request that fills every field the count endpoint accepts.
///
/// `thinking` only ever reaches the wire through a raw provider option, so the
/// shared corpus cannot produce a count body that exercises all six accepted
/// fields at once.
fn count_request(model: &str) -> Request {
    Request::builder()
        .model(model)
        .system("Keep it short.")
        .user("What is the weather in Paris?")
        .tool(ToolDefinition::function(
            "get_weather",
            "Reads the current weather for a city",
            json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"],
            }),
        ))
        .tool_choice(ToolChoice::Auto)
        .temperature(0.7)
        .stop_sequence("END")
        .metadata_entry("tenant", "acme")
        .max_output_tokens(256)
        .provider_option(PROVIDER, "thinking", json!({ "type": "enabled" }))
        .provider_option(PROVIDER, "auto_cache", json!(false))
        .build()
        .expect("the count request should build")
}

// ===========================================================================
// Mock helpers
// ===========================================================================

/// Mounts a `POST` mock that answers with a status, headers, and a JSON body.
///
/// `support::mount_capture` always answers 200 with no extra headers, which is
/// what an encoding test needs. Error classification and rate-limit reporting
/// need control over both, and neither needs the request captured.
fn mount_answer<'server>(
    server: &'server MockServer,
    path: &str,
    status: u16,
    headers: &[(&str, &str)],
    body: &Value,
) -> Mock<'server> {
    let path = path.to_owned();
    let headers: Vec<(String, String)> = headers
        .iter()
        .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
        .collect();
    let body = body.to_string();
    server.mock(move |when, then| {
        when.method(Method::POST).path(path);
        let mut then = then
            .status(status)
            .header("content-type", "application/json")
            .body(body);
        for (name, value) in headers {
            then = then.header(name, value);
        }
    })
}

/// The number of `cache_control` breakpoints anywhere in a wire body.
fn breakpoints(value: &Value) -> usize {
    match value {
        Value::Object(object) => {
            let own = usize::from(object.contains_key("cache_control"));
            own + object.values().map(breakpoints).sum::<usize>()
        }
        Value::Array(items) => items.iter().map(breakpoints).sum(),
        _ => 0,
    }
}

/// Whether a captured request carries one header value.
fn has_header(capture: &support::WireCapture, name: &str, value: &str) -> bool {
    capture
        .headers
        .iter()
        .any(|(key, held)| key == name && held == value)
}

// ===========================================================================
// The shared corpus
// ===========================================================================

#[tokio::test]
async fn encodes_the_base_request() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let (_mock, slot) = support::mount_capture(&server, MESSAGES_PATH, &text_response());

    let response = client
        .complete(support::base_request(&model))
        .await
        .expect("the base request should complete");

    crate::json_snapshot!(support::captured(&slot));
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn encodes_a_multi_turn_conversation() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let (_mock, slot) = support::mount_capture(&server, MESSAGES_PATH, &text_response());

    let response = client
        .complete(support::multi_turn_request(&model))
        .await
        .expect("the multi-turn request should complete");

    // The system turn is hoisted out of `messages` into `system`, which is the
    // only place Anthropic accepts it.
    crate::json_snapshot!(support::captured(&slot));
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn encodes_tools_and_decodes_a_tool_call() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let (_mock, slot) = support::mount_capture(&server, MESSAGES_PATH, &tool_use_response());

    let response = client
        .complete(support::tools_request(&model, None))
        .await
        .expect("the tools request should complete");

    crate::json_snapshot!(support::captured(&slot));
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn encodes_tool_choice_auto() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let (_mock, slot) = support::mount_capture(&server, MESSAGES_PATH, &tool_use_response());

    let response = client
        .complete(support::tools_request(&model, Some(ToolChoice::Auto)))
        .await
        .expect("the auto tool choice should complete");

    crate::json_snapshot!(support::captured(&slot));
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn encodes_tool_choice_none() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let (_mock, slot) = support::mount_capture(&server, MESSAGES_PATH, &text_response());

    let response = client
        .complete(support::tools_request(&model, Some(ToolChoice::None)))
        .await
        .expect("the none tool choice should complete");

    // We keep `tools` and send `tool_choice: {"type":"none"}`. The reference
    // dropped the tool definitions entirely, which changed the prompt the
    // model saw and therefore the prompt cache prefix.
    crate::json_snapshot!(support::captured(&slot));
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn encodes_tool_choice_required() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let (_mock, slot) = support::mount_capture(&server, MESSAGES_PATH, &tool_use_response());

    let response = client
        .complete(support::tools_request(&model, Some(ToolChoice::Required)))
        .await
        .expect("the required tool choice should complete");

    // Anthropic spells "call some tool" as `any`, not `required`.
    crate::json_snapshot!(support::captured(&slot));
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn encodes_a_named_tool_choice() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let (_mock, slot) = support::mount_capture(&server, MESSAGES_PATH, &tool_use_response());

    let response = client
        .complete(support::tools_request(
            &model,
            Some(ToolChoice::Tool {
                name: "get_weather".to_owned(),
            }),
        ))
        .await
        .expect("the named tool choice should complete");

    crate::json_snapshot!(support::captured(&slot));
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn encodes_a_tool_round_trip() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let (_mock, slot) = support::mount_capture(&server, MESSAGES_PATH, &text_response());

    let response = client
        .complete(support::tool_round_trip_request(&model))
        .await
        .expect("the tool round trip should complete");

    let captured = support::captured(&slot);
    // Anthropic has no tool role and rejects consecutive same-role turns, so
    // the two parallel results — each its own canonical message — merge into
    // ONE `user` turn carrying both `tool_result` blocks, in call order.
    assert_eq!(
        captured.body["messages"].as_array().map(Vec::len),
        Some(3),
        "the two tool messages must merge into one user turn"
    );
    let results = &captured.body["messages"][2]["content"];
    assert_eq!(results.as_array().map(Vec::len), Some(2));
    assert_eq!(results[0]["type"], "tool_result");
    assert_eq!(results[0]["tool_use_id"], json!("call_paris"));
    // A failed result is marked by `is_error`, not by different content.
    assert_eq!(results[0]["is_error"], json!(false));
    assert_eq!(results[1]["tool_use_id"], json!("call_madrid"));
    assert_eq!(results[1]["is_error"], json!(true));

    crate::json_snapshot!(captured);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn encodes_replayed_reasoning() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let (_mock, slot) = support::mount_capture(&server, MESSAGES_PATH, &text_response());

    let response = client
        .complete(support::reasoning_round_trip_request(&model))
        .await
        .expect("the reasoning round trip should complete");

    let captured = support::captured(&slot);
    let blocks = &captured.body["messages"][1]["content"];
    // The signature is what lets Anthropic accept replayed thinking. Losing it
    // makes the provider reject the next turn.
    assert_eq!(blocks[0]["type"], "thinking");
    assert_eq!(blocks[0]["signature"], json!("sig-abc"));
    // A redacted block is a different block type carrying an opaque `data`
    // blob, never ordinary thinking text.
    assert_eq!(
        blocks[1],
        json!({
            "type": "redacted_thinking",
            "data": "cmVkYWN0ZWQtcGF5bG9hZA=="
        })
    );

    crate::json_snapshot!(captured);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn rejects_custom_tools_before_dispatch() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let (mock, _slot) = support::mount_capture(&server, MESSAGES_PATH, &text_response());

    let error = client
        .complete(support::custom_tool_request(&model))
        .await
        .expect_err("Anthropic has no custom tool encoding");

    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    assert_eq!(error.retry_classification(), RetryClassification::Never);
    // Rejecting before dispatch is the contract: a downgraded custom tool
    // would change what the model is asked to produce, and a rejected request
    // must not cost the caller a provider call.
    assert_eq!(
        mock.calls_async().await,
        0,
        "an unsupported capability must fail before any HTTP request"
    );

    crate::json_snapshot!(error.data());
}

#[tokio::test]
async fn encodes_url_attachments() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let (_mock, slot) = support::mount_capture(&server, MESSAGES_PATH, &text_response());

    let response = client
        .complete(support::url_attachments_request(&model))
        .await
        .expect("the URL attachments request should complete");

    let captured = support::captured(&slot);
    let blocks = &captured.body["messages"][0]["content"];
    assert_eq!(
        blocks[1]["source"],
        json!({
            "type": "url",
            "url": "https://example.com/cat.png"
        })
    );
    assert_eq!(
        blocks[2]["source"],
        json!({
            "type": "url",
            "url": "https://example.com/report.pdf"
        })
    );
    // `detail` has no Anthropic equivalent and is dropped rather than guessed
    // at; the document name becomes `title`.
    assert!(blocks[1].get("detail").is_none());
    assert_eq!(blocks[2]["title"], json!("report.pdf"));

    crate::json_snapshot!(captured);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn encodes_inline_attachments() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let (_mock, slot) = support::mount_capture(&server, MESSAGES_PATH, &text_response());

    let response = client
        .complete(support::inline_attachments_request(&model))
        .await
        .expect("the inline attachments request should complete");

    let captured = support::captured(&slot);
    let blocks = &captured.body["messages"][0]["content"];
    assert_eq!(
        blocks[1]["source"],
        json!({
            "type": "base64",
            "media_type": "image/png",
            "data": "aW1hZ2UtYnl0ZXM="
        })
    );
    assert_eq!(
        blocks[2]["source"],
        json!({
            "type": "base64",
            "media_type": "application/pdf",
            "data": "cGRmLWJ5dGVz"
        })
    );

    crate::json_snapshot!(captured);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn rejects_audio_before_dispatch() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let (mock, _slot) = support::mount_capture(&server, MESSAGES_PATH, &text_response());

    let error = client
        .complete(support::audio_request(&model))
        .await
        .expect_err("the Messages API has no audio input block");

    // Audio is refused, not dropped and not substituted. A silent drop would
    // let the model answer a prompt the caller never sent, and inventing
    // placeholder text would feed the model content it would read as real.
    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    assert!(error.message().contains("audio content"));
    assert_eq!(
        mock.calls_async().await,
        0,
        "an unencodable part must fail before any HTTP request"
    );

    // Counting refuses exactly what completion refuses. A request the provider
    // would not accept must not come back with a token count, and it must not
    // reach the wire to find that out.
    let (count_mock, _count_slot) =
        support::mount_capture(&server, COUNT_PATH, &json!({ "input_tokens": 7 }));
    let counted = client
        .count_input_tokens(support::audio_request(&model))
        .await
        .expect_err("the count endpoint refuses what complete refuses");
    assert_eq!(counted.kind(), ErrorKind::InvalidRequest);
    assert_eq!(counted.provider_code(), Some("unsupported_capability"));
    assert_eq!(
        count_mock.calls_async().await,
        0,
        "a refused count must not reach the wire either"
    );

    crate::json_snapshot!(error.data());
}

/// A tool result carrying text and an image the tool produced.
///
/// The shared corpus has only text-only tool results, so nothing else reaches
/// the `plain_text` flattening this exposes.
fn image_tool_result_request(model: &str) -> Request {
    Request::builder()
        .model(model)
        .user("Chart last week's sales.")
        .tool(ToolDefinition::function(
            "render_chart",
            "Renders a chart",
            json!({ "type": "object" }),
        ))
        .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
            ToolCall::function("call_chart", "render_chart", json!({ "range": "7d" })),
        )]))
        .message(Message::new(Role::Tool, [ContentPart::ToolResult(
            ToolResult {
                tool_call_id: "call_chart".to_owned(),
                name:         Some("render_chart".to_owned()),
                content:      vec![
                    ContentPart::Text {
                        text: "Here is the chart.".to_owned(),
                    },
                    ContentPart::Image(ImageContent {
                        source: MediaSource::base64("Y2hhcnQtYnl0ZXM=", "image/png"),
                        detail: None,
                    }),
                ],
                is_error:     false,
            },
        )]))
        .provider_option(PROVIDER, "auto_cache", json!(false))
        .max_output_tokens(128)
        .build()
        .expect("the image tool result request should build")
}

#[tokio::test]
async fn keeps_an_image_a_tool_returned() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let (_mock, slot) = support::mount_capture(&server, MESSAGES_PATH, &text_response());

    let response = client
        .complete(image_tool_result_request(&model))
        .await
        .expect("the image tool result request should complete");

    let captured = support::captured(&slot);
    let result = &captured.body["messages"][2]["content"][0];
    // `tool_result.content` takes an array of blocks here, not only a string,
    // and it accepts image blocks. An image a tool produced reaches the model
    // instead of being flattened away with the rest of the non-text content.
    assert_eq!(result["content"][0]["type"], json!("text"));
    assert_eq!(result["content"][0]["text"], json!("Here is the chart."));
    assert_eq!(result["content"][1]["type"], json!("image"));
    assert_eq!(
        result["content"][1]["source"]["data"],
        json!("Y2hhcnQtYnl0ZXM=")
    );

    // Nothing was lost, so nothing is reported. The warning fires only for
    // content this protocol genuinely cannot carry.
    assert!(
        response.warnings.is_empty(),
        "content the codec carries must not warn: {:?}",
        response.warnings
    );

    // This gap is now closed: the block array is what the protocol documents,
    // and the warning it used to raise is gone. Snapshotting the empty warning
    // list keeps that visible, so a regression that reintroduces flattening
    // shows up as a warning appearing rather than only as a body diff.
    crate::json_snapshot!(captured);
    crate::json_snapshot!(response.warnings);
}

#[tokio::test]
async fn encodes_a_json_object_response_format() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let (_mock, slot) = support::mount_capture(&server, MESSAGES_PATH, &json_response());

    let response = client
        .complete(support::response_format_request(
            &model,
            ResponseFormat::JsonObject,
        ))
        .await
        .expect("the JSON object format should complete");

    // Anthropic carries structured output in `output_config.format`, not in a
    // synthetic tool. A permissive object schema is what "any JSON" means.
    crate::json_snapshot!(support::captured(&slot));
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn encodes_a_json_schema_response_format() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let (_mock, slot) = support::mount_capture(&server, MESSAGES_PATH, &json_response());

    let response = client
        .complete(support::response_format_request(
            &model,
            support::json_schema_format(),
        ))
        .await
        .expect("the JSON schema format should complete");

    let captured = support::captured(&slot);
    // The schema travels raw. Anthropic has no name wrapper, so the corpus
    // format name is dropped rather than invented into the payload.
    assert_eq!(
        captured.body["output_config"]["format"]["type"],
        "json_schema"
    );
    assert!(
        captured.body["output_config"]["format"]
            .get("name")
            .is_none()
    );

    crate::json_snapshot!(captured);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn encodes_sampling_controls_and_stop_sequences() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let (_mock, slot) = support::mount_capture(&server, MESSAGES_PATH, &text_response());

    let response = client
        .complete(support::sampling_request(&model))
        .await
        .expect("the sampling request should complete");

    let captured = support::captured(&slot);
    // Order matters: Anthropic stops at the first match, so a reordered list
    // is a different request.
    assert_eq!(captured.body["stop_sequences"], json!(["END", "STOP"]));

    crate::json_snapshot!(captured);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn encodes_request_metadata() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let (_mock, slot) = support::mount_capture(&server, MESSAGES_PATH, &text_response());

    let response = client
        .complete(support::metadata_request(&model))
        .await
        .expect("the metadata request should complete");

    let captured = support::captured(&slot);
    // Anthropic is one of the two dialects that carry request metadata, so it
    // is sent rather than warned about.
    assert_eq!(captured.body["metadata"], json!({ "tenant": "acme" }));
    assert!(response.warnings.is_empty());

    crate::json_snapshot!(captured);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn sends_only_the_selected_provider_namespace() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let (_mock, slot) = support::mount_capture(&server, MESSAGES_PATH, &text_response());

    let response = client
        .complete(support::provider_options_request(
            &model,
            PROVIDER,
            OTHER_PROVIDER,
        ))
        .await
        .expect("the provider options request should complete");

    let captured = support::captured(&slot);
    let body = captured
        .body
        .as_object()
        .expect("a Messages body is an object");
    // The failover candidate's namespace leaves no trace at all.
    assert!(!body.contains_key("unreachable_option"));
    assert!(
        !captured
            .body
            .to_string()
            .contains("this must never be sent")
    );
    // `auto_cache` is a control key: the codec consumes it and never sends it.
    assert!(!body.contains_key("auto_cache"));
    assert_eq!(breakpoints(&captured.body), 0);
    // Everything else in the selected namespace reaches the wire verbatim,
    // including keys Anthropic itself does not define.
    assert_eq!(body["service_tier"], json!("flex"));
    assert_eq!(body["max_output_tokens"], json!(256));

    crate::json_snapshot!(captured);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn replays_provider_native_content() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let (_mock, slot) = support::mount_capture(&server, MESSAGES_PATH, &text_response());

    let response = client
        .complete(support::replay_request(&model, PROVIDER))
        .await
        .expect("the replay request should complete");

    let captured = support::captured(&slot);
    let blocks = &captured.body["messages"][1]["content"];
    // Our own opaque part replays verbatim, and the one belonging to another
    // provider is skipped so the request still encodes after failover.
    assert_eq!(
        blocks[0],
        json!({
            "id": "rs_replay",
            "encrypted_content": "opaque-payload"
        })
    );
    assert_eq!(blocks[1]["type"], "tool_use");
    assert_eq!(blocks.as_array().map(Vec::len), Some(2));

    crate::json_snapshot!(captured);
    crate::json_snapshot!(response);
}

// ===========================================================================
// Provider options
// ===========================================================================

#[tokio::test]
async fn raw_options_override_generated_fields() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let (_mock, slot) = support::mount_capture(&server, MESSAGES_PATH, &text_response());

    let response = client
        .complete(override_request(&model))
        .await
        .expect("the override request should complete");

    let captured = support::captured(&slot);
    // Raw options are authoritative: they are merged last and replace what the
    // codec generated, rather than being overwritten by it.
    assert_eq!(captured.body["max_tokens"], json!(2048));
    assert_eq!(captured.body["temperature"], json!(0.9));
    // A field the codec never generates is simply added.
    assert_eq!(captured.body["thinking"], json!({ "type": "enabled" }));

    crate::json_snapshot!(captured);
    crate::json_snapshot!(response);
}

// ===========================================================================
// Prompt cache
// ===========================================================================

#[tokio::test]
async fn auto_cache_marks_the_system_tools_and_conversation_prefix() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let (_mock, slot) = support::mount_capture(&server, MESSAGES_PATH, &text_response());

    client
        .complete(cacheable_request(&model, None))
        .await
        .expect("the cacheable request should complete");

    let captured = support::captured(&slot);
    // The system prompt becomes a block array purely so it can carry a
    // breakpoint; a bare string cannot.
    assert_eq!(
        captured.body["system"][0]["cache_control"],
        json!({ "type": "ephemeral" })
    );
    // The breakpoint goes on the LAST tool definition, so the whole tool block
    // is inside the cached prefix.
    assert_eq!(
        captured.body["tools"][0]["cache_control"],
        json!({ "type": "ephemeral" })
    );
    // Three user turns interleaved with two assistant turns: the prefix
    // breakpoint lands on the second-to-last user turn, at message index 2, so
    // the next agent-loop iteration reads what this one wrote.
    assert_eq!(
        captured.body["messages"][2]["content"][0]["cache_control"],
        json!({ "type": "ephemeral" })
    );
    assert_eq!(breakpoints(&captured.body), 3);

    crate::json_snapshot!(captured);
}

#[tokio::test]
async fn auto_cache_disabled_sends_no_breakpoints() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let (_mock, slot) = support::mount_capture(&server, MESSAGES_PATH, &text_response());

    client
        .complete(cacheable_request(&model, Some(false)))
        .await
        .expect("the opted-out request should complete");

    let captured = support::captured(&slot);
    assert_eq!(breakpoints(&captured.body), 0);
    // Without a breakpoint to carry, the system prompt stays a plain string.
    assert!(captured.body["system"].is_string());
    // The control key itself never reaches the wire.
    assert!(!captured.body.to_string().contains("auto_cache"));

    crate::json_snapshot!(captured);
}

// ===========================================================================
// Output controls, betas, and limits
// ===========================================================================

/// Capabilities for a model that takes named effort levels, as
/// claude-sonnet-4-6 does.
const LEVELS_CAPABILITIES: &str = "{ text = true, tools = true, structured_output = true, \
                                    reasoning = true, reasoning_effort_levels = true }";

/// Capabilities for a reasoning model that takes no effort levels, as
/// claude-sonnet-4-5 does. Effort reaches such a model as a thinking budget.
const BUDGET_CAPABILITIES: &str =
    "{ text = true, tools = true, structured_output = true, reasoning = true }";

/// The catalog output limit the model-limit fixture declares.
const CATALOG_LIMITS: &str = "{ context_tokens = 200000, max_output_tokens = 64000 }";

/// Builds a client whose model declares `capabilities` and, when one is given,
/// catalog `limits`.
///
/// `WireProvider` renders the model table last, so an appended key lands on
/// the model.
fn client_with(server: &MockServer, capabilities: &str, limits: Option<&str>) -> (Client, String) {
    let provider = anthropic().with_capabilities(capabilities);
    let mut toml = provider.toml(&server.base_url());
    if let Some(limits) = limits {
        toml.push_str("limits = ");
        toml.push_str(limits);
        toml.push('\n');
    }
    let client = support::client_for(
        support::catalog_from_toml("wire", &toml),
        PROVIDER,
        support::header_credentials("x-api-key"),
    );
    (client, provider.selector())
}

/// A request that asks for reasoning effort under an explicit output limit.
///
/// A thinking budget is a share of that limit, so the limit has to be pinned
/// for the budget to be.
fn effort_request(model: &str, effort: ReasoningEffort) -> Request {
    Request::builder()
        .model(model)
        .user("Plan a trip to Paris.")
        .reasoning_effort(effort)
        .max_output_tokens(8000)
        .build()
        .expect("the effort request should build")
}

/// A request carrying a tool choice, reasoning effort, and a JSON answer.
///
/// All three output controls at once, so one fixture shows which of them a
/// forced tool choice suppresses.
fn tool_choice_effort_request(model: &str, choice: ToolChoice) -> Request {
    Request::builder()
        .model(model)
        .user("What is the weather in Paris?")
        .tool(ToolDefinition::function(
            "get_weather",
            "Reads the current weather for a city",
            json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"],
            }),
        ))
        .tool_choice(choice)
        .reasoning_effort(ReasoningEffort::High)
        .response_format(ResponseFormat::JsonObject)
        .max_output_tokens(8000)
        .build()
        .expect("the tool choice effort request should build")
}

/// A tool result carrying a structured document the tool returned.
///
/// The shared corpus has only text-only tool results, so nothing else reaches
/// the JSON translation this pins.
fn json_tool_result_request(model: &str) -> Request {
    Request::builder()
        .model(model)
        .user("What is the weather in Paris?")
        .tool(ToolDefinition::function(
            "get_weather",
            "Reads the current weather for a city",
            json!({ "type": "object" }),
        ))
        .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
            ToolCall::function("call_weather", "get_weather", json!({ "city": "Paris" })),
        )]))
        .message(Message::new(Role::Tool, [ContentPart::ToolResult(
            ToolResult {
                tool_call_id: "call_weather".to_owned(),
                name:         Some("get_weather".to_owned()),
                content:      vec![
                    ContentPart::Json {
                        value: json!({ "city": "Paris", "high_c": 21 }),
                    },
                    ContentPart::Text {
                        text: "Fetched at 09:00.".to_owned(),
                    },
                ],
                is_error:     false,
            },
        )]))
        .provider_option(PROVIDER, "auto_cache", json!(false))
        .max_output_tokens(128)
        .build()
        .expect("the JSON tool result request should build")
}

#[tokio::test]
async fn a_levels_model_takes_an_effort_level_and_adaptive_thinking() {
    let server = MockServer::start_async().await;
    let (client, model) = client_with(&server, LEVELS_CAPABILITIES, None);
    let (_mock, slot) = support::mount_capture(&server, MESSAGES_PATH, &text_response());

    client
        .complete(effort_request(&model, ReasoningEffort::High))
        .await
        .expect("the effort request should complete");

    let captured = support::captured(&slot);
    assert_eq!(captured.body["output_config"]["effort"], json!("high"));
    // Effort guides the thinking allocation rather than replacing it: without
    // a `thinking` object the model does not reason at all, so a levels model
    // gets both.
    assert_eq!(captured.body["thinking"], json!({ "type": "adaptive" }));

    crate::json_snapshot!(captured);
}

#[tokio::test]
async fn a_model_without_effort_levels_takes_a_thinking_budget() {
    let server = MockServer::start_async().await;
    let (client, model) = client_with(&server, BUDGET_CAPABILITIES, None);
    let (_mock, slot) = support::mount_capture(&server, MESSAGES_PATH, &text_response());

    client
        .complete(effort_request(&model, ReasoningEffort::High))
        .await
        .expect("the effort request should complete");

    let captured = support::captured(&slot);
    // Three quarters of the output limit, which is what `high` means to a
    // model that takes a budget instead of a level.
    assert_eq!(
        captured.body["thinking"],
        json!({ "type": "enabled", "budget_tokens": 6000 })
    );
    assert_eq!(captured.body["max_tokens"], json!(8000));
    // Sending `effort` too would ask the model to honor a control it does not
    // take, which is the request the endpoint rejects.
    assert_eq!(captured.body.get("output_config"), None);

    crate::json_snapshot!(captured);
}

#[tokio::test]
async fn a_thinking_budget_lifts_an_output_limit_it_would_not_fit_under() {
    let server = MockServer::start_async().await;
    let (client, model) = client_with(&server, BUDGET_CAPABILITIES, None);
    let (_mock, slot) = support::mount_capture(&server, MESSAGES_PATH, &text_response());

    client
        .complete(effort_request(&model, ReasoningEffort::Max))
        .await
        .expect("the effort request should complete");

    // The highest effort budgets the whole limit, and the budget has to sit
    // strictly below `max_tokens`, so the limit grows to make room for it.
    let captured = support::captured(&slot);
    assert_eq!(captured.body["thinking"]["budget_tokens"], json!(8000));
    assert_eq!(captured.body["max_tokens"], json!(9024));
}

#[tokio::test]
async fn thinking_follows_the_model_when_no_effort_is_asked_for() {
    let server = MockServer::start_async().await;
    let (_mock, slot) = support::mount_capture(&server, MESSAGES_PATH, &text_response());

    let (levels_client, model) = client_with(&server, LEVELS_CAPABILITIES, None);
    levels_client
        .complete(support::base_request(&model))
        .await
        .expect("the base request should complete");

    // A levels model reasons only when the request says so, so the adaptive
    // object goes out even though the caller asked for no effort.
    let captured = support::captured(&slot);
    assert_eq!(captured.body["thinking"], json!({ "type": "adaptive" }));
    assert_eq!(captured.body.get("output_config"), None);

    let (budget_client, model) = client_with(&server, BUDGET_CAPABILITIES, None);
    budget_client
        .complete(support::base_request(&model))
        .await
        .expect("the base request should complete");

    // A model without levels is either natively adaptive, and rejects the
    // toggle, or takes an explicit budget the caller did not ask for.
    let captured = support::captured(&slot);
    assert_eq!(captured.body.get("thinking"), None);
}

#[tokio::test]
async fn a_forced_tool_choice_drops_thinking_and_output_config() {
    let mut encoded = Vec::new();

    for choice in [ToolChoice::Auto, ToolChoice::Required, ToolChoice::Tool {
        name: "get_weather".to_owned(),
    }] {
        let server = MockServer::start_async().await;
        let (client, model) = client_with(&server, LEVELS_CAPABILITIES, None);
        let (_mock, slot) = support::mount_capture(&server, MESSAGES_PATH, &tool_use_response());
        let forced = !matches!(choice, ToolChoice::Auto);

        let response = client
            .complete(tool_choice_effort_request(&model, choice))
            .await
            .expect("the tool choice effort request should complete");

        let captured = support::captured(&slot);
        let codes: Vec<&str> = response
            .warnings
            .iter()
            .map(|warning| warning.code.as_str())
            .collect();
        if forced {
            // Anthropic rejects extended thinking together with a forced tool
            // choice, so both output controls go — the effort level, and the
            // structured-output format that shares the object with it.
            assert_eq!(captured.body.get("thinking"), None, "{captured:?}");
            assert_eq!(captured.body.get("output_config"), None, "{captured:?}");
            // Dropping a control the caller asked for is reported, never
            // silent.
            assert_eq!(codes, ["unsupported_control", "unsupported_control"]);
        } else {
            assert_eq!(captured.body["thinking"], json!({ "type": "adaptive" }));
            assert_eq!(captured.body["output_config"]["effort"], json!("high"));
            assert!(codes.is_empty(), "{:?}", response.warnings);
        }

        encoded.push(json!({
            "tool_choice": captured.body.get("tool_choice"),
            "thinking": captured.body.get("thinking"),
            "output_config": captured.body.get("output_config"),
            "warnings": response.warnings,
        }));
    }

    crate::json_snapshot!(encoded);
}

#[tokio::test]
async fn max_tokens_falls_back_to_the_model_limit() {
    let server = MockServer::start_async().await;
    let (_mock, slot) = support::mount_capture(&server, MESSAGES_PATH, &text_response());

    let (limited_client, model) = client_with(&server, LEVELS_CAPABILITIES, Some(CATALOG_LIMITS));
    limited_client
        .complete(uncapped_request(&model))
        .await
        .expect("the uncapped request should complete");

    // Anthropic requires `max_tokens`, and the model's own output limit is
    // the only honest answer for a caller who named none. A small fixed
    // default truncates a long generation with nothing to show for it.
    let captured = support::captured(&slot);
    assert_eq!(captured.body["max_tokens"], json!(64_000));

    let (unlimited_client, model) = client_with(&server, LEVELS_CAPABILITIES, None);
    unlimited_client
        .complete(uncapped_request(&model))
        .await
        .expect("the uncapped request should complete");

    // A model the catalog records no limit for falls back to the generous
    // default rather than to a small one.
    let captured = support::captured(&slot);
    assert_eq!(captured.body["max_tokens"], json!(65_536));
}

/// A request that names no output limit, so the codec has to pick one.
fn uncapped_request(model: &str) -> Request {
    Request::builder()
        .model(model)
        .user("Write the whole report.")
        .build()
        .expect("the uncapped request should build")
}

#[tokio::test]
async fn beta_headers_leave_the_body_and_become_one_header() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let (_mock, slot) = support::mount_capture(&server, MESSAGES_PATH, &text_response());

    client
        .complete(
            Request::builder()
                .model(&model)
                .user("Hello")
                .max_output_tokens(128)
                .provider_option(
                    PROVIDER,
                    "beta_headers",
                    json!(["context-1m-2025-08-07", "files-api-2025-04-14"]),
                )
                .build()
                .expect("the beta headers request should build"),
        )
        .await
        .expect("the beta headers request should complete");

    let captured = support::captured(&slot);
    // Anthropic takes several betas as one comma-separated header value.
    assert!(
        has_header(
            &captured,
            "anthropic-beta",
            "context-1m-2025-08-07,files-api-2025-04-14"
        ),
        "{:?}",
        captured.headers
    );
    // This is a header control, not a body field. Left among the options it
    // would be merged into the JSON body, which the endpoint rejects.
    assert_eq!(captured.body.get("beta_headers"), None);

    crate::json_snapshot!(captured);
}

#[tokio::test]
async fn the_fast_speed_tier_carries_its_own_beta() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let (_mock, slot) = support::mount_capture(&server, MESSAGES_PATH, &text_response());

    client
        .complete(fast_request(&model, None))
        .await
        .expect("the fast request should complete");

    // The body field alone does nothing: without the beta the endpoint
    // ignores the tier, and the caller waits at standard latency for a
    // request they meant to be fast.
    let captured = support::captured(&slot);
    assert_eq!(captured.body["speed"], json!("fast"));
    assert!(
        has_header(&captured, "anthropic-beta", "fast-mode-2026-02-01"),
        "{:?}",
        captured.headers
    );

    crate::json_snapshot!(captured);

    client
        .complete(fast_request(
            &model,
            Some(json!(["fast-mode-2026-02-01", "files-api-2025-04-14"])),
        ))
        .await
        .expect("the fast request should complete");

    // A caller who already listed the fast beta does not get it twice.
    let captured = support::captured(&slot);
    assert!(
        has_header(
            &captured,
            "anthropic-beta",
            "fast-mode-2026-02-01,files-api-2025-04-14"
        ),
        "{:?}",
        captured.headers
    );
}

/// A fast-tier request, optionally naming betas of its own.
fn fast_request(model: &str, betas: Option<Value>) -> Request {
    let mut builder = Request::builder()
        .model(model)
        .user("Hello")
        .speed(Speed::Fast)
        .max_output_tokens(128);
    if let Some(betas) = betas {
        builder = builder.provider_option(PROVIDER, "beta_headers", betas);
    }
    builder.build().expect("the fast request should build")
}

#[tokio::test]
async fn keeps_json_a_tool_returned() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let (_mock, slot) = support::mount_capture(&server, MESSAGES_PATH, &text_response());

    let response = client
        .complete(json_tool_result_request(&model))
        .await
        .expect("the JSON tool result request should complete");

    let captured = support::captured(&slot);
    let result = &captured.body["messages"][2]["content"][0];
    // Anthropic has no structured-output block, so the document travels as
    // the text a tool would have printed — the same translation message
    // content uses. Filtering it out sent the model an EMPTY result.
    assert_eq!(result["content"][0]["type"], json!("text"));
    assert_eq!(
        result["content"][0]["text"],
        json!("{\"city\":\"Paris\",\"high_c\":21}")
    );
    assert_eq!(result["content"][1]["text"], json!("Fetched at 09:00."));

    // Nothing was lost, so nothing is reported.
    assert!(
        response.warnings.is_empty(),
        "content the codec carries must not warn: {:?}",
        response.warnings
    );

    crate::json_snapshot!(captured);
}

// ===========================================================================
// Usage, raw payloads, headers, and rate limits
// ===========================================================================

#[tokio::test]
async fn decodes_disjoint_usage_without_subtraction() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let usage_response = json!({
        "id": "msg_01WireUsage",
        "type": "message",
        "role": "assistant",
        "model": API_MODEL,
        "content": [
            { "type": "thinking", "thinking": "Long deliberation.", "signature": "sig-usage" },
            { "type": "text", "text": "Done." }
        ],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 50,
            "output_tokens": 1200,
            "cache_read_input_tokens": 9000,
            "cache_creation_input_tokens": 1000
        }
    });
    let (_mock, _slot) = support::mount_capture(&server, MESSAGES_PATH, &usage_response);

    let response = client
        .complete(support::base_request(&model))
        .await
        .expect("the usage request should complete");

    // Anthropic's counters are ALREADY disjoint: `input_tokens` excludes both
    // cache counters. Subtracting here would under-report input on every
    // cached call, so the decoder assigns rather than subtracts.
    assert_eq!(response.usage.input, 50);
    assert_eq!(response.usage.output, 1200);
    assert_eq!(response.usage.cache_read, 9000);
    // Cache writes come from `cache_creation_input_tokens`, which is nonzero
    // only when the request carried breakpoints.
    assert_eq!(response.usage.cache_write, 1000);
    // Thinking tokens are billed inside `output_tokens` with no separate
    // counter. Splitting them out would be fabrication, so reasoning stays 0
    // even though this response contains a thinking block.
    assert_eq!(response.usage.reasoning, 0);
    assert_eq!(response.usage.total(), 11_250);
    assert_eq!(response.usage.billable_output(), 1200);

    crate::json_snapshot!(response);
}

#[tokio::test]
async fn keeps_the_raw_success_document() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let document = tool_use_response();
    let (_mock, _slot) = support::mount_capture(&server, MESSAGES_PATH, &document);

    let response = client
        .complete(support::base_request(&model))
        .await
        .expect("the request should complete");

    // The whole success body is retained exactly as the provider sent it, so a
    // caller can read a field this crate does not model.
    assert_eq!(response.raw.as_ref(), Some(&document));
}

#[tokio::test]
async fn sends_the_anthropic_version_header() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let (_mock, slot) = support::mount_capture(&server, MESSAGES_PATH, &text_response());

    client
        .complete(support::base_request(&model))
        .await
        .expect("the request should complete");

    let captured = support::captured(&slot);
    assert!(
        has_header(&captured, "anthropic-version", "2023-06-01"),
        "every Anthropic request declares the protocol version: {:?}",
        captured.headers
    );
    // The credential goes in `x-api-key`, not `authorization`. The harness
    // redacts the value, so the snapshot proves placement without printing a
    // key-shaped string.
    assert!(has_header(&captured, "x-api-key", "[redacted]"));
    assert!(
        !captured
            .headers
            .iter()
            .any(|(name, _)| name == "authorization")
    );
}

#[tokio::test]
async fn reports_rate_limit_headers() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let _mock = mount_answer(
        &server,
        MESSAGES_PATH,
        200,
        &[
            ("anthropic-ratelimit-requests-limit", "1000"),
            ("anthropic-ratelimit-requests-remaining", "999"),
            ("anthropic-ratelimit-requests-reset", "2026-08-28T18:30:00Z"),
            ("anthropic-ratelimit-tokens-limit", "80000"),
            ("anthropic-ratelimit-tokens-remaining", "79904"),
            ("anthropic-ratelimit-tokens-reset", "2026-08-28T18:31:00Z"),
            // A header from the other family must not bleed in.
            ("anthropic-ratelimit-input-tokens-limit", "40000"),
        ],
        &text_response(),
    );

    let response = client
        .complete(support::base_request(&model))
        .await
        .expect("the request should complete");

    let limits = response
        .rate_limits
        .as_ref()
        .expect("the anthropic-ratelimit-* family should be parsed");
    assert_eq!(limits.request_limit, Some(1000));
    assert_eq!(limits.request_remaining, Some(999));
    assert_eq!(limits.token_limit, Some(80000));
    assert_eq!(limits.token_remaining, Some(79904));
    // Request and token resets stay separate and keep Anthropic's own RFC 3339
    // formatting. This crate does not parse them into an instant.
    assert_ne!(limits.request_reset, limits.token_reset);
    assert_eq!(
        limits.request_reset.as_deref(),
        Some("2026-08-28T18:30:00Z")
    );
    assert_eq!(limits.token_reset.as_deref(), Some("2026-08-28T18:31:00Z"));

    // The shared `[TIMESTAMP]` filter rewrites both reset values, so the
    // snapshot pins the counters and the presence of both fields. That the two
    // resets stay distinct is held by the assertions above, not by the
    // snapshot.
    crate::json_snapshot!(limits);
}

// ===========================================================================
// Streaming
// ===========================================================================

/// A realistic transcript: split usage, thinking with a signature delta,
/// visible text, and an interleaved tool call.
fn stream_transcript() -> String {
    support::sse_transcript(&[
        (
            "message_start",
            r#"{"type":"message_start","message":{"id":"msg_01WireStream","type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":47,"cache_read_input_tokens":1024,"cache_creation_input_tokens":256,"output_tokens":1}}}"#,
        ),
        ("ping", r#"{"type":"ping"}"#),
        (
            "content_block_start",
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#,
        ),
        (
            "content_block_delta",
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"The user wants "}}"#,
        ),
        (
            "content_block_delta",
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"today's weather."}}"#,
        ),
        (
            "content_block_delta",
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"EqQBCgIYAhIM1gbc"}}"#,
        ),
        (
            "content_block_stop",
            r#"{"type":"content_block_stop","index":0}"#,
        ),
        (
            "content_block_start",
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
        ),
        (
            "content_block_delta",
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Let me check "}}"#,
        ),
        (
            "content_block_delta",
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"the forecast."}}"#,
        ),
        (
            "content_block_stop",
            r#"{"type":"content_block_stop","index":1}"#,
        ),
        (
            "content_block_start",
            r#"{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_01Weather","name":"get_weather","input":{}}}"#,
        ),
        (
            "content_block_delta",
            r#"{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}"#,
        ),
        (
            "content_block_delta",
            r#"{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":" \"Paris\"}"}}"#,
        ),
        (
            "content_block_stop",
            r#"{"type":"content_block_stop","index":2}"#,
        ),
        (
            "message_delta",
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":89}}"#,
        ),
        ("message_stop", r#"{"type":"message_stop"}"#),
    ])
}

/// The cumulative usage snapshot carried by the last `usage` event.
fn last_usage(events: &[Value]) -> &Value {
    events
        .iter()
        .rev()
        .find(|event| event["type"] == "usage")
        .and_then(|event| event.get("usage"))
        .expect("the stream should carry at least one usage snapshot")
}

#[tokio::test]
async fn streams_text_reasoning_and_a_tool_call() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let (_mock, slot) = support::mount_capture_sse(&server, MESSAGES_PATH, &stream_transcript());

    let stream = client
        .stream(support::tools_request(&model, None))
        .await
        .expect("the stream should start");
    let events = support::collect_stream_events(stream).await;

    support::assert_stream_contract(&events);

    let captured = support::captured(&slot);
    assert_eq!(captured.body["stream"], json!(true));

    // Usage is split across TWO events. `message_start` carries the input and
    // both cache counters; `message_delta` carries only the output count. A
    // decoder that reads `message_delta` alone loses every input count
    // silently, so the final snapshot must still hold all of them.
    let usage = last_usage(&events);
    assert_eq!(usage["input"], json!(47));
    assert_eq!(usage["cache_read"], json!(1024));
    assert_eq!(usage["cache_write"], json!(256));
    assert_eq!(usage["output"], json!(89));
    assert_eq!(usage["reasoning"], json!(0));

    // Content-block ids are stable and surfaced on every event for the block.
    // The reference exposed no block identity at all, which made two open
    // blocks indistinguishable. The provider's tool-call id stays on the call.
    let start = events
        .iter()
        .find(|event| {
            event["type"] == "content_block_start" && event["kind"]["type"] == "tool_call"
        })
        .expect("the tool call should open a block");
    assert_eq!(start["id"], json!("block-2"));
    assert_eq!(start["kind"]["id"], json!("toolu_01Weather"));

    crate::json_snapshot!(captured);
    crate::json_snapshot!(events);
}

#[tokio::test]
async fn a_streamed_response_has_no_raw_document() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let transcript = support::sse_transcript(&[
        (
            "message_start",
            r#"{"type":"message_start","message":{"id":"msg_01WireRaw","type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[],"usage":{"input_tokens":8,"output_tokens":0}}}"#,
        ),
        (
            "content_block_start",
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"Hi"}}"#,
        ),
        (
            "content_block_stop",
            r#"{"type":"content_block_stop","index":0}"#,
        ),
        (
            "message_delta",
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}"#,
        ),
        ("message_stop", r#"{"type":"message_stop"}"#),
    ]);
    let (_mock, _slot) = support::mount_capture_sse(&server, MESSAGES_PATH, &transcript);

    let stream = client
        .stream(support::base_request(&model))
        .await
        .expect("the stream should start");
    let events = support::collect_stream_events(stream).await;

    support::assert_stream_contract(&events);

    let completed = events
        .last()
        .expect("a successful stream ends with a completed event");
    assert_eq!(completed["type"], json!("completed"));
    // Anthropic sends no terminal response document. `raw` stays empty rather
    // than being synthesized from accumulated stream state, because an
    // accumulated log is not what the provider said.
    assert!(
        completed["response"].get("raw").is_none(),
        "a streamed response must not synthesize a raw document: {completed}"
    );
    // The seed text on `content_block_start` is content, not a placeholder.
    assert_eq!(
        completed["response"]["content"],
        json!([{ "type": "text", "text": "Hi" }])
    );

    crate::json_snapshot!(events);
}

#[tokio::test]
async fn streaming_keeps_redacted_thinking() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let transcript = support::sse_transcript(&[
        (
            "content_block_start",
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"EroBCkYIBBgCKkDx"}}"#,
        ),
        (
            "content_block_stop",
            r#"{"type":"content_block_stop","index":0}"#,
        ),
        // A real Anthropic stream always reports why it stopped before
        // `message_stop`. Leaving it out would complete this stream as
        // `incomplete`, which belongs to the truncation fixture, not here.
        (
            "message_delta",
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":14}}"#,
        ),
        ("message_stop", r#"{"type":"message_stop"}"#),
    ]);
    let (_mock, _slot) = support::mount_capture_sse(&server, MESSAGES_PATH, &transcript);

    let stream = client
        .stream(support::base_request(&model))
        .await
        .expect("the stream should start");
    let events = support::collect_stream_events(stream).await;

    support::assert_stream_contract(&events);

    // INTENTIONAL DIFFERENCE: the reference dropped `redacted_thinking` in the
    // streaming path while keeping it in the blocking path, so a streamed
    // conversation could not be replayed. The blob arrives whole on the start
    // event and no delta follows it.
    let ended = events
        .iter()
        .find(|event| event["type"] == "content_block_end")
        .expect("the redacted block should end");
    assert_eq!(
        ended["part"],
        json!({
            "type": "reasoning",
            "text": "EroBCkYIBBgCKkDx",
            "redacted": true
        })
    );

    crate::json_snapshot!(events);
}

#[tokio::test]
async fn a_stream_error_event_ends_the_stream() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let transcript = support::sse_transcript(&[
        (
            "message_start",
            r#"{"type":"message_start","message":{"id":"msg_01WireFail","type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[],"usage":{"input_tokens":8,"output_tokens":0}}}"#,
        ),
        (
            "content_block_start",
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        ),
        (
            "content_block_delta",
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Partial"}}"#,
        ),
        (
            "error",
            r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
        ),
    ]);
    let (_mock, _slot) = support::mount_capture_sse(&server, MESSAGES_PATH, &transcript);

    let stream = client
        .stream(support::base_request(&model))
        .await
        .expect("the stream should start");
    let events = support::collect_stream_events(stream).await;

    support::assert_stream_contract(&events);

    let last = events.last().expect("the stream produced events");
    assert_eq!(last["type"], json!("error"));
    assert_eq!(last["error"]["provider_code"], json!("overloaded_error"));
    // A mid-stream error carries no HTTP status, so it is classified from its
    // code alone.
    assert!(last["error"].get("status").is_none());
    // A failed stream emits no `completed` event: the partial text is not a
    // response the caller may treat as finished.
    assert!(
        !events.iter().any(|event| event["type"] == "completed"),
        "a failed stream must not complete: {events:?}"
    );

    crate::json_snapshot!(events);
}

#[tokio::test]
async fn a_truncated_stream_completes_once_and_says_it_is_incomplete() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let transcript = support::sse_transcript(&[
        (
            "message_start",
            r#"{"type":"message_start","message":{"id":"msg_01WireCut","type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[],"usage":{"input_tokens":8,"output_tokens":0}}}"#,
        ),
        (
            "content_block_start",
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        ),
        (
            "content_block_delta",
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Half a sen"}}"#,
        ),
    ]);
    let (_mock, _slot) = support::mount_capture_sse(&server, MESSAGES_PATH, &transcript);

    let stream = client
        .stream(support::base_request(&model))
        .await
        .expect("the stream should start");
    let events = support::collect_stream_events(stream).await;

    support::assert_stream_contract(&events);

    // INTENTIONAL DIFFERENCE: the transport ended without `message_stop`, and
    // the reference emitted no terminal event at all, leaving the caller to
    // guess. We close the open block and emit exactly one `completed` carrying
    // the partial text, so every successful stream has a terminal event.
    let completions = events
        .iter()
        .filter(|event| event["type"] == "completed")
        .count();
    assert_eq!(completions, 1);
    let response = events
        .last()
        .map(|event| &event["response"])
        .expect("the last event is the completion");
    assert_eq!(
        response["content"],
        json!([{ "type": "text", "text": "Half a sen" }])
    );
    // Completing is not the same as finishing. The provider never said why it
    // stopped, so the finish reason says the answer is cut off rather than
    // defaulting to `stop` and reading as a model that finished its sentence.
    assert_eq!(response["finish_reason"], json!("incomplete"));

    crate::json_snapshot!(events);
}

#[tokio::test]
async fn a_streamed_refusal_fails_the_stream() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let transcript = support::sse_transcript(&[
        (
            "message_start",
            r#"{"type":"message_start","message":{"id":"msg_01WireRefusal","type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[],"usage":{"input_tokens":19,"output_tokens":0}}}"#,
        ),
        (
            "message_delta",
            r#"{"type":"message_delta","delta":{"stop_reason":"refusal","stop_details":{"type":"refusal","explanation":"This asks for working malware."}},"usage":{"output_tokens":3}}"#,
        ),
        ("message_stop", r#"{"type":"message_stop"}"#),
    ]);
    let (_mock, _slot) = support::mount_capture_sse(&server, MESSAGES_PATH, &transcript);

    let stream = client
        .stream(support::base_request(&model))
        .await
        .expect("the stream should start");
    let events = support::collect_stream_events(stream).await;

    support::assert_stream_contract(&events);

    let last = events.last().expect("the stream produced events");
    assert_eq!(last["type"], json!("error"));
    // A refusal is a failure, not a short answer. The stream fails on the
    // event that carries the reason, before `message_stop` can complete it as
    // a success a caller would read as the model having nothing to say.
    assert_eq!(last["error"]["kind"], json!("content_filter"));
    assert_eq!(last["error"]["provider_code"], json!("refusal"));
    assert_eq!(last["error"]["retry"], json!({ "type": "never" }));
    // The provider's own account of the refusal reaches the caller.
    assert!(
        last["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("This asks for working malware.")),
        "{last:?}"
    );
    assert!(
        !events.iter().any(|event| event["type"] == "completed"),
        "a refused stream must not complete: {events:?}"
    );

    crate::json_snapshot!(events);
}

// ===========================================================================
// Error classification
// ===========================================================================

/// One classified-error case: an HTTP status, response headers, and a body.
struct ErrorCase {
    name:    &'static str,
    status:  u16,
    headers: &'static [(&'static str, &'static str)],
    body:    Value,
}

fn error_cases() -> Vec<ErrorCase> {
    vec![
        ErrorCase {
            name:    "authentication",
            status:  401,
            headers: &[],
            body:    json!({
                "type": "error",
                "error": { "type": "authentication_error", "message": "invalid x-api-key" }
            }),
        },
        ErrorCase {
            name:    "access denied",
            status:  403,
            headers: &[],
            body:    json!({
                "type": "error",
                "error": {
                    "type": "permission_error",
                    "message": "Your API key does not have permission to use the specified resource."
                }
            }),
        },
        ErrorCase {
            name:    "not found",
            status:  404,
            headers: &[],
            body:    json!({
                "type": "error",
                "error": { "type": "not_found_error", "message": "model: claude-does-not-exist" }
            }),
        },
        ErrorCase {
            name:    "rate limit",
            status:  429,
            headers: &[("retry-after", "30")],
            body:    json!({
                "type": "error",
                "error": {
                    "type": "rate_limit_error",
                    "message": "Number of requests has exceeded your rate limit."
                }
            }),
        },
        ErrorCase {
            name:    "overloaded",
            status:  529,
            headers: &[],
            body:    json!({
                "type": "error",
                "error": { "type": "overloaded_error", "message": "Overloaded" }
            }),
        },
        ErrorCase {
            name:    "spent credit",
            status:  429,
            headers: &[],
            body:    json!({
                "type": "error",
                "error": {
                    "type": "rate_limit_error",
                    "message": "Your credit balance is too low to access the Anthropic API."
                }
            }),
        },
    ]
}

#[tokio::test]
async fn classifies_error_responses() {
    let mut classified = Vec::new();

    for case in error_cases() {
        let server = MockServer::start_async().await;
        let (client, model) = client_for(&server);
        let _mock = mount_answer(
            &server,
            MESSAGES_PATH,
            case.status,
            case.headers,
            &case.body,
        );

        let error = match client.complete(support::base_request(&model)).await {
            Ok(response) => panic!("the {} case should fail, got {response:?}", case.name),
            Err(error) => error,
        };

        classified.push(json!({ "case": case.name, "error": error.data() }));
    }

    let kinds: Vec<&str> = classified
        .iter()
        .map(|entry| {
            entry["error"]["kind"]
                .as_str()
                .expect("every classified error has a kind")
        })
        .collect();
    assert_eq!(kinds, [
        "authentication",
        "access_denied",
        "not_found",
        "rate_limit",
        "server",
        "quota_exceeded"
    ]);

    // A 429 with `Retry-After` is retryable on the provider's own schedule.
    assert_eq!(
        classified[3]["error"]["retry"],
        json!({ "type": "after", "after_millis": 30_000 })
    );
    // A 529 is retryable too, but with no advised delay.
    assert_eq!(classified[4]["error"]["retry"], json!({ "type": "safe" }));
    // Spent credit arrives as a `rate_limit_error` and is NOT retryable:
    // backoff never restores a balance. Only the message separates the two.
    assert_eq!(classified[5]["error"]["retry"], json!({ "type": "never" }));
    assert_eq!(
        classified[5]["error"]["provider_code"],
        json!("rate_limit_error")
    );

    crate::json_snapshot!(classified);
}

#[tokio::test]
async fn a_refusal_fails_instead_of_decoding_as_an_empty_answer() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let _mock = mount_answer(
        &server,
        MESSAGES_PATH,
        200,
        &[],
        &json!({
            "id": "msg_01WireRefusal",
            "type": "message",
            "role": "assistant",
            "model": API_MODEL,
            "content": [],
            "stop_reason": "refusal",
            "stop_details": {
                "type": "refusal",
                "explanation": "This asks for working malware."
            },
            "usage": { "input_tokens": 19, "output_tokens": 3 }
        }),
    );

    let error = client
        .complete(support::base_request(&model))
        .await
        .expect_err("a refusal is a failure, not a successful empty answer");

    // The HTTP status is 200, so only the stop reason separates a refusal from
    // an answer. Decoding it as success hid the refusal from the caller and
    // from the middleware that classifies it.
    assert_eq!(error.kind(), ErrorKind::ContentFilter);
    assert_eq!(error.provider_code(), Some("refusal"));
    assert_eq!(error.retry_classification(), RetryClassification::Never);
    assert!(
        error.message().contains("This asks for working malware."),
        "the provider's explanation reaches the caller: {}",
        error.message()
    );

    crate::json_snapshot!(error.data());
}

#[tokio::test]
async fn rejects_a_success_body_that_is_not_a_messages_response() {
    let mut rejected = Vec::new();

    for (case, body) in [
        ("empty object", json!({})),
        (
            "no content",
            json!({
                "id": "msg_01WireBare",
                "model": API_MODEL,
                "usage": { "input_tokens": 4, "output_tokens": 0 }
            }),
        ),
        (
            "no usage",
            json!({
                "id": "msg_01WireBare",
                "model": API_MODEL,
                "content": [{ "type": "text", "text": "Hello back." }]
            }),
        ),
        (
            "a gateway document",
            json!({ "status": "ok", "upstream": "anthropic" }),
        ),
    ] {
        let server = MockServer::start_async().await;
        let (client, model) = client_for(&server);
        let _mock = mount_answer(&server, MESSAGES_PATH, 200, &[], &body);

        let error = match client.complete(support::base_request(&model)).await {
            Ok(response) => panic!("the {case} case should fail, got {response:?}"),
            Err(error) => error,
        };

        // A 200 that carries none of the fields every Messages response has is
        // not a Messages response. Decoding it leniently reported a model that
        // said nothing, which reads as a real, empty answer.
        assert_eq!(error.kind(), ErrorKind::ResponseDecode, "{case}");
        rejected.push(json!({ "case": case, "error": error.data() }));
    }

    crate::json_snapshot!(rejected);
}

// ===========================================================================
// Token counting
// ===========================================================================

#[tokio::test]
async fn counts_input_tokens_without_max_tokens() {
    let server = MockServer::start_async().await;
    let (client, model) = client_for(&server);
    let (_mock, slot) =
        support::mount_capture(&server, COUNT_PATH, &json!({ "input_tokens": 2095 }));

    let count = client
        .count_input_tokens(count_request(&model))
        .await
        .expect("the count request should succeed")
        .expect("Anthropic has a native count endpoint");

    assert_eq!(count.tokens(), 2095);
    // The count is attributed to the canonical catalog model, not to the
    // selector the request used.
    assert_eq!(count.model().model().as_str(), MODEL);

    let captured = support::captured(&slot);
    assert_eq!(captured.path, COUNT_PATH);
    let body = captured
        .body
        .as_object()
        .expect("a count body is an object");
    let mut keys: Vec<&str> = body.keys().map(String::as_str).collect();
    keys.sort_unstable();
    // The count body is a NARROWING of the generation body. `max_tokens` is
    // required on /v1/messages and rejected here, and every other
    // generation-only field — temperature, stop_sequences, metadata, stream —
    // is dropped too.
    assert_eq!(keys, [
        "messages",
        "model",
        "system",
        "thinking",
        "tool_choice",
        "tools"
    ]);
    // Raw provider options are merged before the narrowing, so an option
    // cannot smuggle a generation-only field onto this endpoint.
    assert!(!body.contains_key("max_tokens"));
    assert!(!body.contains_key("stream"));
    assert!(has_header(&captured, "anthropic-version", "2023-06-01"));

    crate::json_snapshot!(captured);
}
