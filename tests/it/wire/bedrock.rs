//! Amazon Bedrock Converse wire snapshots.
//!
//! This dialect had no wire coverage in the reference implementation at all,
//! so nothing here is a port. Every fixture is written against the Converse
//! API shape and the approved `bedrock-transport` plan, and each intentional
//! difference from the reference is documented at the fixture that shows it.
//!
//! The intentional differences pinned below, in the order they appear:
//!
//! - The model id is percent-encoded into the path. The reference interpolated
//!   it raw, which split an ARN-style inference-profile id into extra path
//!   segments and made the call unroutable.
//! - A custom tool is refused before dispatch. The reference sent a custom tool
//!   through the plain function path, so the grammar was silently lost.
//! - URL media is refused before dispatch. The reference dropped the part
//!   silently, so the model answered about an attachment it never received.
//! - Request metadata produces an `unsupported_control` warning. The reference
//!   dropped the map with no signal.
//! - The `auto_cache` control key never reaches the wire. The reference had no
//!   known-keys list for Bedrock and forwarded a stray `auto_cache` onto the
//!   Converse body.
//! - `count_input_tokens` calls Bedrock Runtime `CountTokens`. The reference
//!   implemented no Bedrock count path at all.
//! - A `stopReason` of `refusal` fails the call on both the blocking and the
//!   streaming path. The reference mapped it to a ContentFilter finish reason
//!   and returned a successful, empty response.
//! - A tool name or `toolUseId` in the message history that Converse rejects
//!   refuses the request. The reference rewrote both onto the Converse
//!   character set with a hash suffix.

#[cfg(feature = "bedrock-aws")]
use std::process::Command;
#[cfg(feature = "bedrock-aws")]
use std::{env, fs, process};

use httpmock::{Method, MockServer};
use lithos_llm::credentials::{Credentials, SecretValue};
use lithos_llm::types::{
    ContentPart, ErrorData, FinishReason, Message, ReasoningContent, Request, Response, Role,
    ToolCall, ToolChoice, ToolDefinition, ToolResult,
};
use lithos_llm::{Client, Error};
use serde_json::{Value, json};

use crate::support::{self, WireCapture, WireProvider};

/// The catalog provider id, which is also the raw provider-options namespace
/// and the opaque replay namespace this codec claims.
const PROVIDER: &str = "bedrock";
/// The catalog model id, which is what a request selector names.
const MODEL: &str = "sonnet";
/// The provider model id, which is what reaches the URL path.
const API_MODEL: &str = "anthropic.claude-sonnet-4-6";

const BEDROCK_BEARER_AUTH: &str = "{ type = \"bedrock_bearer\" }";
#[cfg(feature = "bedrock-aws")]
const AWS_AUTH: &str = "{ type = \"aws\", region = \"us-east-1\" }";

// ===========================================================================
// Harness
// ===========================================================================

/// The provider under test: Converse behind a long-lived Bedrock API key.
fn wire_provider() -> WireProvider<'static> {
    WireProvider::new(PROVIDER, "bedrock", "bedrock-converse", MODEL)
        .with_api_model(API_MODEL)
        .with_auth(BEDROCK_BEARER_AUTH)
}

/// The request selector every corpus constructor is given.
fn selector() -> String {
    wire_provider().selector()
}

/// A long-lived Bedrock API key, sent as a bearer token.
fn bedrock_credentials() -> Credentials {
    Credentials::BedrockBearer(SecretValue::new(support::TEST_API_KEY))
}

/// The path of one Bedrock runtime operation, as it should reach the wire.
fn operation_path(api_model: &str, operation: &str) -> String {
    format!("/model/{api_model}/{operation}")
}

fn client(server: &MockServer, provider: &WireProvider<'_>, credentials: Credentials) -> Client {
    support::client_for(provider.catalog(&server.base_url()), PROVIDER, credentials)
}

/// A Converse success body wrapping one content block list.
fn converse_response(content: &Value) -> Value {
    json!({
        "output": { "message": { "role": "assistant", "content": content } },
        "stopReason": "end_turn",
        "usage": { "inputTokens": 12, "outputTokens": 7, "totalTokens": 19 },
    })
}

/// The plain-text Converse success body most fixtures answer with.
fn text_response() -> Value {
    converse_response(&json!([{ "text": "Hello." }]))
}

/// Drives one complete call against a mock Converse endpoint.
async fn complete(request: Request, response_body: &Value) -> (WireCapture, Response) {
    complete_for(&wire_provider(), request, response_body).await
}

/// Drives one complete call for a specific provider description.
async fn complete_for(
    provider: &WireProvider<'_>,
    request: Request,
    response_body: &Value,
) -> (WireCapture, Response) {
    let server = MockServer::start_async().await;
    let (_mock, slot) = support::mount_capture(
        &server,
        &operation_path(provider.api_model, "converse"),
        response_body,
    );

    let response = client(&server, provider, bedrock_credentials())
        .complete(request)
        .await
        .expect("the Converse call should succeed");

    (support::captured(&slot), response)
}

/// Drives one complete call the provider answers with a failing document.
///
/// The HTTP call succeeds, so this covers a body the codec itself rejects
/// rather than a classified transport failure.
async fn complete_error(request: Request, response_body: &Value) -> Error {
    let server = MockServer::start_async().await;
    let (_mock, _slot) = support::mount_capture(
        &server,
        &operation_path(API_MODEL, "converse"),
        response_body,
    );

    client(&server, &wire_provider(), bedrock_credentials())
        .complete(request)
        .await
        .expect_err("the Converse call should fail")
}

/// Drives one call the codec must refuse before it dispatches.
///
/// Returns the error and asserts that the mock endpoint was never reached, so
/// a refusal that still spent a provider call cannot pass.
async fn refuse(request: Request) -> Error {
    let server = MockServer::start_async().await;
    let provider = wire_provider();
    let body = text_response();
    let (mock, slot) =
        support::mount_capture(&server, &operation_path(API_MODEL, "converse"), &body);

    let error = client(&server, &provider, bedrock_credentials())
        .complete(request)
        .await
        .expect_err("the call should be refused");

    assert_eq!(
        mock.calls_async().await,
        0,
        "a refused request must never reach the provider"
    );
    assert!(
        slot.lock()
            .expect("the capture slot should not be poisoned")
            .is_none(),
        "a refused request must capture nothing"
    );
    error
}

// ===========================================================================
// The canonical corpus
// ===========================================================================

#[tokio::test]
async fn encodes_the_base_request() {
    let body = text_response();

    let (capture, response) = complete(support::base_request(&selector()), &body).await;

    crate::json_snapshot!(capture);
    crate::json_snapshot!(response);
    // The whole success document is retained verbatim, so a caller can read a
    // Converse field this crate does not normalize.
    assert_eq!(response.raw.as_ref(), Some(&body));
}

#[tokio::test]
async fn encodes_a_multi_turn_conversation() {
    // Converse carries the system instruction in its own `system` block list,
    // never as a message, and the two user turns bracket the assistant turn.
    let (capture, response) =
        complete(support::multi_turn_request(&selector()), &text_response()).await;

    crate::json_snapshot!(capture);
    crate::json_snapshot!(response);
}

/// The captured body with `toolConfig` removed, plus the removed value.
fn split_tool_config(capture: &WireCapture) -> (Value, Value) {
    let mut body = capture.body.clone();
    let tool_config = body
        .as_object_mut()
        .and_then(|body| body.remove("toolConfig"))
        .unwrap_or(Value::Null);
    (body, tool_config)
}

#[tokio::test]
async fn encodes_every_tool_choice_mode() {
    let body = text_response();

    // The unset choice is the baseline: its whole request is pinned, and every
    // other mode is then held to changing nothing but `toolConfig`.
    let (capture, _) = complete(support::tools_request(&selector(), None), &body).await;
    let (baseline, unset) = split_tool_config(&capture);
    crate::json_snapshot!(capture);

    let modes = [
        ("auto", ToolChoice::Auto),
        ("none", ToolChoice::None),
        ("required", ToolChoice::Required),
        ("tool", ToolChoice::Tool {
            name: "get_weather".to_owned(),
        }),
    ];
    let mut configs = vec![json!({ "tool_choice": "unset", "toolConfig": unset })];
    for (name, choice) in modes {
        let (capture, _) = complete(support::tools_request(&selector(), Some(choice)), &body).await;
        let (rest, tool_config) = split_tool_config(&capture);
        assert_eq!(
            rest, baseline,
            "the {name} tool choice changed something other than `toolConfig`"
        );
        configs.push(json!({ "tool_choice": name, "toolConfig": tool_config }));
    }

    // `auto` is the Converse wire default, so it is omitted rather than sent.
    // `none` has no Converse spelling: withholding the whole `toolConfig` is
    // the only faithful encoding, so that entry is null.
    crate::json_snapshot!(configs);
}

#[tokio::test]
async fn decodes_a_tool_call_response() {
    let body = json!({
        "output": { "message": { "role": "assistant", "content": [
            { "text": "Looking that up." },
            { "toolUse": {
                "toolUseId": "tooluse_paris",
                "name": "get_weather",
                "input": { "city": "Paris" },
            } },
        ] } },
        "stopReason": "tool_use",
        "usage": { "inputTokens": 40, "outputTokens": 22, "totalTokens": 62 },
    });

    let (_capture, response) = complete(
        support::tools_request(&selector(), Some(ToolChoice::Auto)),
        &body,
    )
    .await;

    crate::json_snapshot!(response);
}

#[tokio::test]
async fn encodes_a_tool_round_trip() {
    // Converse has no tool role: both results ride in a `user` message, and
    // the failed one is marked with `status: "error"` rather than a separate
    // block.
    //
    // The two results are separate canonical messages but answer one assistant
    // turn, and Converse alternates roles, so they merge into a single turn.
    // Sending them as two consecutive `user` messages would be rejected.
    let (capture, response) = complete(
        support::tool_round_trip_request(&selector()),
        &text_response(),
    )
    .await;

    let messages = capture.body["messages"]
        .as_array()
        .expect("the body carries a message list");
    let roles: Vec<&str> = messages
        .iter()
        .filter_map(|message| message["role"].as_str())
        .collect();
    assert_eq!(roles, ["user", "assistant", "user"]);
    let results: Vec<&str> = messages[2]["content"]
        .as_array()
        .expect("the merged turn carries a content list")
        .iter()
        .filter_map(|block| block["toolResult"]["toolUseId"].as_str())
        .collect();
    assert_eq!(results, ["call_paris", "call_madrid"]);

    crate::json_snapshot!(capture);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn encodes_replayed_reasoning() {
    let body = converse_response(&json!([
        { "reasoningContent": { "reasoningText": {
            "text": "97 has no divisor below 10.",
            "signature": "sig-def",
        } } },
        { "reasoningContent": { "redactedContent": "cmVkYWN0ZWQtcmVwbHk=" } },
        { "text": "Yes, 97 is prime." },
    ]));

    let (capture, response) =
        complete(support::reasoning_round_trip_request(&selector()), &body).await;

    // The signature Bedrock issued is echoed back inside `reasoningText`,
    // because Bedrock validates it. A redacted block has no signature of its
    // own and rides as the sealed `redactedContent` payload instead of text.
    crate::json_snapshot!(capture);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn refuses_a_custom_tool_before_dispatch() {
    // INTENTIONAL DIFFERENCE. The reference mapped every tool through the
    // plain function path, so a custom tool's grammar was silently replaced by
    // an input schema. Converse has function tools only, so the request is
    // refused instead of quietly downgraded.
    let error = refuse(support::custom_tool_request(&selector())).await;

    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    crate::json_snapshot!(error.data());
}

#[tokio::test]
async fn refuses_url_media_before_dispatch() {
    // INTENTIONAL DIFFERENCE. Converse has no URL media source and this crate
    // never fetches remote media on the caller's behalf. The reference dropped
    // the part silently, which asked the model to describe an attachment it
    // had never been given.
    let error = refuse(support::url_attachments_request(&selector())).await;

    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    crate::json_snapshot!(error.data());
}

#[tokio::test]
async fn refuses_audio_before_dispatch() {
    // Converse carries no audio block. Dropping the part and answering anyway
    // would let the model reply to a prompt the caller never sent, so the
    // request is refused with nothing dispatched.
    let error = refuse(support::audio_request(&selector())).await;

    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    crate::json_snapshot!(error.data());
}

#[tokio::test]
async fn encodes_inline_attachments() {
    // Both media blocks carry the base64 payload under `source.bytes`, and the
    // declared media type is mapped onto the Converse `format` enum:
    // `image/png` becomes `png` and `application/pdf` becomes `pdf`. A
    // document also carries a `name`, which Converse requires.
    let (capture, response) = complete(
        support::inline_attachments_request(&selector()),
        &text_response(),
    )
    .await;

    crate::json_snapshot!(capture);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn encodes_sampling_controls() {
    let (capture, response) =
        complete(support::sampling_request(&selector()), &text_response()).await;

    // Every sampling control lives under `inferenceConfig`, and the stop
    // sequences keep the order the caller gave them.
    assert_eq!(
        capture.body["inferenceConfig"]["stopSequences"],
        json!(["END", "STOP"])
    );
    crate::json_snapshot!(capture);
    // This corpus entry also carries one metadata entry, so the same warning
    // the metadata fixture pins appears here too.
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn warns_that_request_metadata_cannot_be_sent() {
    // INTENTIONAL DIFFERENCE. Converse has no request-metadata field. The
    // reference dropped `request.metadata` with no signal; folding it into
    // some other field would change the prompt, so it is reported instead.
    let (capture, response) =
        complete(support::metadata_request(&selector()), &text_response()).await;

    assert!(
        !capture.body.to_string().contains("tenant"),
        "no metadata may reach the wire: {}",
        capture.body
    );
    let codes: Vec<&str> = response
        .warnings
        .iter()
        .map(|warning| warning.code.as_str())
        .collect();
    assert_eq!(codes, ["unsupported_control"]);
    crate::json_snapshot!(capture);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn merges_only_the_selected_provider_options() {
    let request = support::provider_options_request(&selector(), PROVIDER, "openai");

    let (capture, _) = complete(request, &text_response()).await;

    let body = capture.body.to_string();
    assert!(
        !body.contains("unreachable_option"),
        "another provider's namespace must leave no trace: {body}"
    );
    // INTENTIONAL DIFFERENCE. `auto_cache` is a control key this codec
    // consumes. The reference kept no known-keys list for Bedrock, so a stray
    // `auto_cache` was forwarded onto the Converse body.
    assert!(
        !body.contains("auto_cache"),
        "a control key must never reach the wire: {body}"
    );
    crate::json_snapshot!(capture);
}

#[tokio::test]
async fn replays_bedrock_opaque_content() {
    // The opaque part in this codec's own namespace goes back on the wire
    // verbatim and another provider's is dropped.
    //
    // Two things this fixture deliberately does not claim. The corpus opaque
    // payload is shaped like an OpenAI reasoning item, so what lands in the
    // message content below is not a valid Converse block; the fixture pins
    // the passthrough, and a real Bedrock replay part holds a real Converse
    // block. And `toolUse.input` is a JSON document rather than a string, so
    // the tool call's original argument bytes cannot survive here the way they
    // do on a protocol that sends arguments as text.
    let (capture, response) = complete(
        support::replay_request(&selector(), PROVIDER),
        &text_response(),
    )
    .await;

    let body = capture.body.to_string();
    assert!(
        !body.contains("rs_ignored"),
        "another provider's opaque part must be dropped: {body}"
    );
    crate::json_snapshot!(capture);
    crate::json_snapshot!(response);
}

// ===========================================================================
// Dialect specifics
// ===========================================================================

#[tokio::test]
async fn percent_encodes_the_model_id_in_the_path() {
    // INTENTIONAL DIFFERENCE, and the reason this fixture exists. The
    // reference interpolated the model id into the path with no encoding at
    // all, so the `/` inside an ARN-style inference-profile id split into
    // extra path segments and the call was unroutable. `%` is encoded first so
    // that the escape introduced for `/` is never re-encoded, and the signer
    // signs the encoded URL, so client and server still apply exactly one
    // encoding pass to the same bytes.
    let body = text_response();
    let mut captures = Vec::new();
    for (api_model, path) in [
        (
            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/lithos-shared",
            "/model/arn:aws:bedrock:us-east-1:123456789012:inference-profile%2Flithos-shared/\
             converse",
        ),
        ("weird%model/v1", "/model/weird%25model%2Fv1/converse"),
    ] {
        let provider = wire_provider().with_api_model(api_model);
        let server = MockServer::start_async().await;
        let (_mock, slot) = support::mount_capture(&server, path, &body);

        client(&server, &provider, bedrock_credentials())
            .complete(support::base_request(&selector()))
            .await
            .expect("the Converse call should succeed");

        let capture = support::captured(&slot);
        assert_eq!(capture.path, path);
        captures.push(json!({ "api_model": api_model, "path": capture.path }));
    }

    crate::json_snapshot!(captures);
}

#[tokio::test]
async fn reads_usage_without_subtracting_anything() {
    // Converse counters are already disjoint, like Anthropic's: `inputTokens`
    // excludes both cache counters, so nothing is subtracted. Reasoning tokens
    // are folded into `outputTokens` with no separate counter, so `reasoning`
    // stays zero rather than guessing at a split.
    let mut body = text_response();
    body["usage"] = json!({
        "inputTokens": 30,
        "outputTokens": 628,
        "totalTokens": 658,
        "cacheReadInputTokens": 1024,
        "cacheWriteInputTokens": 512,
    });

    let (_capture, response) = complete(support::base_request(&selector()), &body).await;

    assert_eq!(response.usage.input, 30);
    assert_eq!(response.usage.output, 628);
    assert_eq!(response.usage.reasoning, 0);
    assert_eq!(response.usage.cache_read, 1024);
    assert_eq!(response.usage.cache_write, 512);
    // The provider's own `totalTokens` is ignored; the disjoint buckets sum to
    // more than it, because it excludes the cache counters too.
    assert_eq!(response.usage.total(), 2194);
    crate::json_snapshot!(response.usage);
}

#[tokio::test]
async fn a_refusal_fails_the_call() {
    // INTENTIONAL DIFFERENCE. The reference mapped `stopReason: "refusal"` to
    // a ContentFilter finish reason and returned a successful, empty response.
    // A refusal is a failure, not a short answer, so it now fails the call the
    // same way it does on the direct Anthropic route: one uniform
    // cross-provider contract, visible to retry middleware, rather than a
    // silent empty answer on this route alone.
    //
    // `refusal` is not a member of the documented Converse `stopReason` enum.
    // Bedrock passes it through from the Anthropic models that emit it, which
    // is the same observed behavior the reference implementation mapped.
    //
    // Converse carries no field holding the model's account of the refusal, so
    // the error has no explanation to pass on — unlike the Anthropic route,
    // where `stop_details.explanation` supplies one.
    let body = json!({
        "output": { "message": { "role": "assistant", "content": [] } },
        "stopReason": "refusal",
        "usage": { "inputTokens": 12, "outputTokens": 0, "totalTokens": 12 },
    });

    let error = complete_error(support::base_request(&selector()), &body).await;

    assert_eq!(error.provider_code(), Some("refusal"));
    assert_eq!(error.data().raw_data.as_ref(), Some(&body));
    crate::json_snapshot!(error.data());
}

#[tokio::test]
async fn a_streamed_refusal_fails_before_the_stream_completes() {
    // The streaming half of the same contract. `messageStop` carries the stop
    // reason, so the stream fails there, before the terminal `metadata` event
    // can complete it as a success.
    let frames = vec![
        support::bedrock_event_frame("messageStart", &json!({ "role": "assistant" })),
        support::bedrock_event_frame("messageStop", &json!({ "stopReason": "refusal" })),
        support::bedrock_event_frame(
            "metadata",
            &json!({ "usage": { "inputTokens": 12, "outputTokens": 0 } }),
        ),
    ];

    let (_capture, events) = stream(support::base_request(&selector()), &frames).await;

    support::assert_stream_contract(&events);
    let failures: Vec<&Value> = events
        .iter()
        .filter(|event| event["type"] == "error")
        .collect();
    let [failure] = failures.as_slice() else {
        panic!("expected exactly one error item, got {events:#?}");
    };
    assert_eq!(failure["error"]["kind"], "content_filter");
    assert_eq!(failure["error"]["provider_code"], "refusal");
    assert!(
        !events.iter().any(|event| event["type"] == "ended"),
        "a refused stream must not complete"
    );
    crate::json_snapshot!(events);
}

#[tokio::test]
async fn a_context_window_stop_decodes_as_length() {
    // Converse's own name for a generation that ran out of context. It is the
    // same outcome as `max_tokens`, so it normalizes to the same finish
    // reason instead of an unrecognized one.
    let mut body = text_response();
    body["stopReason"] = json!("model_context_window_exceeded");

    let (_capture, response) = complete(support::base_request(&selector()), &body).await;

    assert_eq!(response.finish_reason, FinishReason::Length);
    crate::json_snapshot!(response);
}

/// A conversation replaying one historical tool call and its result.
fn replayed_tool_request(name: &str, id: &str) -> Request {
    Request::builder()
        .model(selector())
        .user("What is the weather in Paris?")
        .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
            ToolCall::function(id, name, json!({ "city": "Paris" })),
        )]))
        .message(Message::new(Role::Tool, [ContentPart::ToolResult(
            ToolResult {
                tool_call_id: id.to_owned(),
                name:         Some(name.to_owned()),
                content:      vec![ContentPart::Text {
                    text: "18C and clear".to_owned(),
                }],
                is_error:     false,
            },
        )]))
        .build()
        .expect("the replayed tool request should build")
}

#[tokio::test]
async fn refuses_historical_tool_identifiers_converse_rejects() {
    // INTENTIONAL DIFFERENCE. The reference rewrote a foreign tool name or id
    // onto the Converse character set with a hash suffix. Rewriting is lossy —
    // `mcp.server.tool` and `mcp_server_tool` collapse onto one value, and a
    // rewritten id no longer matches the call the caller holds — so the
    // request is refused with the offending value named instead.
    //
    // A conversation that began on another provider is where this bites: a
    // dotted Gemini tool name or an over-long id would otherwise die at AWS
    // with an opaque ValidationException.
    let dotted = refuse(replayed_tool_request("mcp.server.tool", "call_1")).await;
    assert_eq!(dotted.provider_code(), Some("unsupported_capability"));

    let long_id = refuse(replayed_tool_request("get_weather", &"a".repeat(65))).await;
    assert_eq!(long_id.provider_code(), Some("unsupported_capability"));

    crate::json_snapshot!(json!({
        "dotted_name": dotted.data(),
        "over_long_id": long_id.data(),
    }));
}

#[tokio::test]
async fn a_replayed_tool_call_id_keeps_its_dots_and_colons() {
    // `toolUseId` allows `.` and `:` on top of the tool-name set, so an id in
    // that shape replays unchanged rather than being refused or rewritten.
    let (capture, _) = complete(
        replayed_tool_request("get_weather", "functions.get_weather:4"),
        &text_response(),
    )
    .await;

    assert_eq!(
        capture.body["messages"][1]["content"][0]["toolUse"]["toolUseId"],
        "functions.get_weather:4"
    );
    crate::json_snapshot!(capture);
}

#[tokio::test]
async fn tool_result_content_keeps_json_and_drops_reasoning() {
    // INTENTIONAL DIFFERENCE from this crate's own earlier behavior, not from
    // the reference. `toolResult.content` is a block list whose members
    // include `json`, so structured results reach the model as themselves and
    // no flattening warning is due. Reasoning has no member of that union: it
    // is dropped rather than encoded as a `reasoningContent` block Converse
    // would reject, and the drop is reported.
    let request = Request::builder()
        .model(selector())
        .user("Chart the quarters.")
        .message(Message::new(Role::Tool, [ContentPart::ToolResult(
            ToolResult {
                tool_call_id: "call_chart".to_owned(),
                name:         Some("chart".to_owned()),
                content:      vec![
                    ContentPart::Json {
                        value: json!({ "quarters": [1, 2] }),
                    },
                    ContentPart::Reasoning(ReasoningContent {
                        text:             "the third quarter is the outlier".to_owned(),
                        signature:        None,
                        signature_origin: None,
                        redacted:         false,
                    }),
                ],
                is_error:     false,
            },
        )]))
        .build()
        .expect("the tool result request should build");

    let (capture, response) = complete(request, &text_response()).await;

    let body = capture.body.to_string();
    assert!(body.contains("quarters"), "{body}");
    assert!(!body.contains("reasoningContent"), "{body}");
    let codes: Vec<&str> = response
        .warnings
        .iter()
        .map(|warning| warning.code.as_str())
        .collect();
    assert_eq!(codes, ["unsupported_control"]);
    crate::json_snapshot!(capture);
    crate::json_snapshot!(response);
}

/// Two user turns and one tool, so every cache-point position has a home.
fn caching_request(auto_cache: Option<bool>) -> Request {
    let mut builder = Request::builder()
        .model(selector())
        .system("Keep it short.")
        .user("What is the capital of France?")
        .message(Message::text(Role::Assistant, "Paris."))
        .user("And of Spain?")
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
    builder.build().expect("the caching request should build")
}

#[tokio::test]
async fn auto_cache_places_cache_points_by_default() {
    let (capture, _) = complete(caching_request(None), &text_response()).await;

    // One after the system prompt, one after the last tool, and one on the
    // second-to-last user turn, so each agent-loop iteration reuses the prefix
    // the previous one wrote instead of paying to write a prefix the next turn
    // invalidates.
    let marker = json!({ "cachePoint": { "type": "default" } });
    assert_eq!(capture.body["system"][1], marker);
    assert_eq!(capture.body["toolConfig"]["tools"][1], marker);
    assert_eq!(capture.body["messages"][0]["content"][1], marker);
    crate::json_snapshot!(capture);
}

#[tokio::test]
async fn auto_cache_false_sends_no_cache_points() {
    let (capture, _) = complete(caching_request(Some(false)), &text_response()).await;

    let body = capture.body.to_string();
    assert!(!body.contains("cachePoint"), "{body}");
    assert!(!body.contains("auto_cache"), "{body}");
    crate::json_snapshot!(capture);
}

/// The capabilities of a hosted family that cannot cache.
const NO_CACHING_CAPABILITIES: &str =
    "{ text = true, tools = true, tool_choice = { required = true, named = true } }";

#[tokio::test]
async fn a_model_that_cannot_cache_sends_no_cache_points() {
    // Bedrock allows passthrough, so a caller can name any hosted family, and
    // most of them have no prompt cache. A `cachePoint` block is not ignored
    // by such a family: Converse rejects the whole request with a
    // ValidationException, so every call to a Llama, Mistral, or DeepSeek
    // model would fail. The caller's `auto_cache` control is a veto on top of
    // this, not a substitute for it.
    let provider = wire_provider().with_capabilities(NO_CACHING_CAPABILITIES);

    let (capture, _) = complete_for(&provider, caching_request(None), &text_response()).await;

    let body = capture.body.to_string();
    assert!(!body.contains("cachePoint"), "{body}");
    crate::json_snapshot!(capture);
}

#[tokio::test]
async fn a_raw_option_overrides_one_generated_inference_field() {
    // Raw options win, and two objects at the same key merge recursively, so
    // overriding `maxTokens` keeps the `temperature` the codec encoded.
    let request = Request::builder()
        .model(selector())
        .user("Hello")
        .max_output_tokens(128)
        .temperature(0.5)
        .provider_option(PROVIDER, "inferenceConfig", json!({ "maxTokens": 4096 }))
        .provider_option(PROVIDER, "guardrailConfig", json!({ "trace": "enabled" }))
        .build()
        .expect("the override request should build");

    let (capture, _) = complete(request, &text_response()).await;

    assert_eq!(capture.body["inferenceConfig"]["maxTokens"], 4096);
    assert_eq!(capture.body["inferenceConfig"]["temperature"], 0.5);
    crate::json_snapshot!(capture);
}

// ===========================================================================
// Streaming
// ===========================================================================

/// The `ConverseStream` transcript every streaming fixture is read against.
///
/// Blocks 0 and 2 interleave on purpose: their deltas alternate, so a decoder
/// that keyed content by arrival order rather than by `contentBlockIndex`
/// would merge them.
fn stream_frames() -> Vec<Vec<u8>> {
    vec![
        support::bedrock_event_frame("messageStart", &json!({ "role": "assistant" })),
        support::bedrock_event_frame(
            "contentBlockDelta",
            &json!({ "contentBlockIndex": 0, "delta": { "text": "Checking" } }),
        ),
        support::bedrock_event_frame(
            "contentBlockStart",
            &json!({
                "contentBlockIndex": 2,
                "start": { "toolUse": { "toolUseId": "tooluse_paris", "name": "get_weather" } },
            }),
        ),
        support::bedrock_event_frame(
            "contentBlockDelta",
            &json!({ "contentBlockIndex": 0, "delta": { "text": " Paris." } }),
        ),
        support::bedrock_event_frame(
            "contentBlockDelta",
            &json!({
                "contentBlockIndex": 2,
                "delta": { "toolUse": { "input": "{\"city\":" } },
            }),
        ),
        support::bedrock_event_frame(
            "contentBlockDelta",
            &json!({
                "contentBlockIndex": 1,
                "delta": { "reasoningContent": { "text": "Paris is in France." } },
            }),
        ),
        support::bedrock_event_frame(
            "contentBlockDelta",
            &json!({
                "contentBlockIndex": 1,
                "delta": { "reasoningContent": { "signature": "sig-stream" } },
            }),
        ),
        support::bedrock_event_frame(
            "contentBlockDelta",
            &json!({
                "contentBlockIndex": 2,
                "delta": { "toolUse": { "input": " \"Paris\"}" } },
            }),
        ),
        support::bedrock_event_frame("contentBlockStop", &json!({ "contentBlockIndex": 0 })),
        support::bedrock_event_frame("contentBlockStop", &json!({ "contentBlockIndex": 1 })),
        support::bedrock_event_frame("contentBlockStop", &json!({ "contentBlockIndex": 2 })),
        support::bedrock_event_frame("messageStop", &json!({ "stopReason": "tool_use" })),
        support::bedrock_event_frame(
            "metadata",
            &json!({
                "usage": {
                    "inputTokens": 30,
                    "outputTokens": 628,
                    "totalTokens": 658,
                    "cacheReadInputTokens": 1024,
                    "cacheWriteInputTokens": 512,
                },
                "metrics": { "latencyMs": 412 },
            }),
        ),
    ]
}

/// Drives one stream over real binary event-stream frames.
async fn stream(request: Request, frames: &[Vec<u8>]) -> (WireCapture, Vec<Value>) {
    let server = MockServer::start_async().await;
    let provider = wire_provider();
    let (_mock, slot) = support::mount_capture_event_stream(
        &server,
        &operation_path(API_MODEL, "converse-stream"),
        frames,
    );

    let events = support::collect_stream_events(
        client(&server, &provider, bedrock_credentials())
            .stream(request)
            .await
            .expect("the ConverseStream call should be accepted"),
    )
    .await;

    (support::captured(&slot), events)
}

#[tokio::test]
async fn streams_text_reasoning_and_a_tool_call() {
    let (capture, events) = stream(support::base_request(&selector()), &stream_frames()).await;

    support::assert_stream_contract(&events);
    // The stream path differs from the complete path only in the operation
    // segment of the URL.
    assert_eq!(
        capture.path,
        operation_path(API_MODEL, "converse-stream").as_str()
    );
    // UNDOCUMENTED DIFFERENCE, pinned so it is a decision rather than an
    // accident: the reference asked for the binary framing with
    // `accept: application/vnd.amazon.eventstream`, and the captured headers
    // below carry the client default instead. Bedrock frames a
    // `converse-stream` response the same way either way, so this is a
    // difference in what is requested, not in what arrives.
    // Converse identifies a block only by its ordinal, so the ordinal becomes
    // the stable block id, and interleaved blocks keep separate identities.
    let ids: Vec<&str> = events
        .iter()
        .filter_map(|event| event.get("id").and_then(Value::as_str))
        .collect();
    assert!(ids.contains(&"block-0"), "{ids:?}");
    assert!(ids.contains(&"block-1"), "{ids:?}");
    assert!(ids.contains(&"block-2"), "{ids:?}");

    let completed = events
        .last()
        .expect("the stream should produce events")
        .clone();
    assert_eq!(completed["type"], "ended");
    // Bedrock sends no final response document, only the terminal `metadata`
    // usage event, so a streamed response keeps `raw` unset rather than
    // inventing an accumulated log of events.
    assert!(completed["response"].get("raw").is_none());

    crate::json_snapshot!(capture);
    crate::json_snapshot!(events);
}

#[tokio::test]
async fn an_empty_text_delta_opens_no_text_block() {
    // The reference suppressed an empty text delta. Passing it on opens a text
    // block that carries no text, and the completed response ends with an
    // empty part that re-encodes to nothing on the next turn.
    let frames = vec![
        support::bedrock_event_frame("messageStart", &json!({ "role": "assistant" })),
        support::bedrock_event_frame(
            "contentBlockDelta",
            &json!({ "contentBlockIndex": 0, "delta": { "text": "" } }),
        ),
        support::bedrock_event_frame("contentBlockStop", &json!({ "contentBlockIndex": 0 })),
        support::bedrock_event_frame("messageStop", &json!({ "stopReason": "end_turn" })),
        support::bedrock_event_frame(
            "metadata",
            &json!({ "usage": { "inputTokens": 12, "outputTokens": 0 } }),
        ),
    ];

    let (_capture, events) = stream(support::base_request(&selector()), &frames).await;

    support::assert_stream_contract(&events);
    assert!(
        !events
            .iter()
            .any(|event| event["type"] == "text_delta" || event["type"] == "text_start"),
        "an empty delta must open no text block: {events:#?}"
    );
    let completed = events.last().expect("the stream should produce events");
    assert_eq!(completed["type"], "ended");
    assert_eq!(completed["response"]["content"], json!([]));
    crate::json_snapshot!(events);
}

#[tokio::test]
async fn an_exception_frame_fails_the_stream() {
    // Bedrock reports a modeled failure in band, after a successful HTTP
    // status, so a stream can fail long after its response headers arrived.
    let frames = vec![
        support::bedrock_event_frame("messageStart", &json!({ "role": "assistant" })),
        support::bedrock_event_frame(
            "contentBlockDelta",
            &json!({ "contentBlockIndex": 0, "delta": { "text": "Chec" } }),
        ),
        support::bedrock_exception_frame(
            "throttlingException",
            &json!({
                "__type": "com.amazon.bedrock#ThrottlingException",
                "message": "Too many requests, please wait before trying again.",
            }),
        ),
    ];

    let (_capture, events) = stream(support::base_request(&selector()), &frames).await;

    support::assert_stream_contract(&events);
    let failures: Vec<&Value> = events
        .iter()
        .filter(|event| event["type"] == "error")
        .collect();
    let [failure] = failures.as_slice() else {
        panic!("expected exactly one error item, got {events:#?}");
    };
    assert_eq!(failure["error"]["kind"], "rate_limit");
    assert_eq!(failure["error"]["provider_code"], "ThrottlingException");
    assert!(
        !events.iter().any(|event| event["type"] == "ended"),
        "a failed stream must not complete"
    );
    crate::json_snapshot!(events);
}

#[tokio::test]
async fn an_error_frame_fails_the_stream() {
    // The other failure class. An `error` frame carries its diagnostics in the
    // frame headers rather than in the payload and has no `:event-type` at
    // all, so a decoder that reads only `:event-type` sees nothing to do and
    // lets the stream complete as if the model had finished. Both the code and
    // the message below come from headers, over an empty payload.
    let frames = vec![
        support::bedrock_event_frame("messageStart", &json!({ "role": "assistant" })),
        support::bedrock_event_frame(
            "contentBlockDelta",
            &json!({ "contentBlockIndex": 0, "delta": { "text": "Chec" } }),
        ),
        support::encode_event_stream_frame(
            &[
                (":message-type", "error"),
                (":error-code", "ModelStreamErrorException"),
                (":error-message", "The model stopped responding mid-stream."),
            ],
            b"{}",
        ),
    ];

    let (_capture, events) = stream(support::base_request(&selector()), &frames).await;

    support::assert_stream_contract(&events);
    let failures: Vec<&Value> = events
        .iter()
        .filter(|event| event["type"] == "error")
        .collect();
    let [failure] = failures.as_slice() else {
        panic!("expected exactly one error item, got {events:#?}");
    };
    assert_eq!(failure["error"]["kind"], "server");
    assert_eq!(
        failure["error"]["provider_code"],
        "ModelStreamErrorException"
    );
    assert_eq!(
        failure["error"]["message"],
        "provider bedrock The model stopped responding mid-stream."
    );
    assert!(
        !events.iter().any(|event| event["type"] == "ended"),
        "a failed stream must not complete"
    );
    crate::json_snapshot!(events);
}

// ===========================================================================
// Classified errors
// ===========================================================================

/// Drives one call against a failing endpoint and returns the classified error.
async fn failure(status: u16, body: &Value) -> ErrorData {
    let server = MockServer::start_async().await;
    let provider = wire_provider();
    let path = operation_path(API_MODEL, "converse");
    let response_body = body.clone();
    let _mock = server.mock(move |when, then| {
        when.method(Method::POST).path(path.clone());
        then.status(status)
            .header("content-type", "application/json")
            .json_body(response_body.clone());
    });

    client(&server, &provider, bedrock_credentials())
        .complete(support::base_request(&selector()))
        .await
        .expect_err("a failing status should produce an error")
        .data()
}

#[tokio::test]
async fn classifies_provider_failures() {
    // The bearer path and the SigV4 path spell the message key differently —
    // `Message` for one, `message` for the other — and both are read.
    let cases = [
        (
            401,
            json!({
                "__type": "UnrecognizedClientException",
                "message": "The security token included in the request is invalid.",
            }),
        ),
        (
            403,
            json!({
                "__type": "com.amazon.bedrock#AccessDeniedException",
                "Message": "You do not have access to the model with the specified model ID.",
            }),
        ),
        (
            429,
            json!({
                "__type": "com.amazon.bedrock#ThrottlingException",
                "message": "Too many requests, please wait before trying again.",
            }),
        ),
        (
            429,
            json!({
                "__type": "com.amazon.bedrock#ServiceQuotaExceededException",
                "message": "Your request rate exceeds the service quota for your account.",
            }),
        ),
        (
            400,
            json!({
                "__type": "com.amazon.bedrock#ValidationException",
                "message": "The value at toolConfig.tools failed to satisfy the constraint.",
            }),
        ),
        (
            404,
            json!({
                "__type": "com.amazon.bedrock#ResourceNotFoundException",
                "message": "Could not resolve the foundation model from the provided model id.",
            }),
        ),
        (
            500,
            json!({
                "__type": "com.amazon.bedrock#InternalServerException",
                "message": "An internal server error occurred.",
            }),
        ),
    ];

    let mut classified = Vec::new();
    for (status, body) in cases {
        let data = failure(status, &body).await;
        classified.push(json!({ "status": status, "error": data }));
    }

    // Throttling is a rate limit and is safe to retry. A quota that is spent
    // is not: the same code arrives on the same status, and only the code
    // separates the two.
    let kinds: Vec<&Value> = classified
        .iter()
        .map(|entry| &entry["error"]["kind"])
        .collect();
    assert_eq!(kinds, [
        "authentication",
        "access_denied",
        "rate_limit",
        "quota_exceeded",
        "invalid_request",
        "not_found",
        "server",
    ]);
    assert_eq!(classified[2]["error"]["retry"]["type"], "safe");
    assert_eq!(classified[3]["error"]["retry"]["type"], "never");
    crate::json_snapshot!(classified);
}

// ===========================================================================
// Token counting
// ===========================================================================

#[tokio::test]
async fn counts_input_tokens_through_the_native_endpoint() {
    // INTENTIONAL DIFFERENCE. The reference implemented no Bedrock count path
    // at all, so a caller always fell back to a local estimate. The approved
    // plan requires the real Runtime `CountTokens` operation.
    let server = MockServer::start_async().await;
    let provider = wire_provider();
    let body = json!({ "inputTokens": 41 });
    let (_mock, slot) =
        support::mount_capture(&server, &operation_path(API_MODEL, "count-tokens"), &body);

    let count = client(&server, &provider, bedrock_credentials())
        .count_input_tokens(support::multi_turn_request(&selector()))
        .await
        .expect("the count-tokens call should succeed")
        .expect("Bedrock counts input tokens natively");

    let capture = support::captured(&slot);
    // The body carries the Converse prompt and nothing else: `inferenceConfig`
    // and `toolConfig` shape generation rather than the prompt, and
    // `CountTokens` rejects them.
    assert!(capture.body["input"]["converse"].is_object());
    assert!(
        capture.body["input"]["converse"]
            .get("inferenceConfig")
            .is_none()
    );
    assert_eq!(count.tokens(), 41);
    assert_eq!(count.model().to_string(), selector());
    crate::json_snapshot!(capture);
}

// ===========================================================================
// Authentication
// ===========================================================================

/// The provider used for both authentication arms.
///
/// The AWS scheme accepts either credential, so one catalog description drives
/// both halves and nothing but the credential differs between them. Without
/// `bedrock-aws` this provider does not build at all: the adapter reports a
/// construction issue rather than silently downgrading to bearer, which is why
/// everything below is gated on the feature.
#[cfg(feature = "bedrock-aws")]
fn aws_provider() -> WireProvider<'static> {
    WireProvider::new(PROVIDER, "bedrock", "bedrock-converse", MODEL)
        .with_api_model(API_MODEL)
        .with_auth(AWS_AUTH)
}

/// The environment variable naming the file the signed child process writes.
#[cfg(feature = "bedrock-aws")]
const SIGV4_CAPTURE_FILE: &str = "LITHOS_LLM_SIGV4_CAPTURE";

/// The name of the child test, which is also its filter on the command line.
#[cfg(feature = "bedrock-aws")]
const SIGV4_CHILD_TEST: &str = "wire::bedrock::sigv4_child_writes_the_authentication_captures";

/// Drives the same request twice against one endpoint, once per credential.
///
/// Both calls share a mock server, so the two captures differ in nothing the
/// harness normalizes away: the same host, the same port, and the same path.
#[cfg(feature = "bedrock-aws")]
async fn authentication_captures() -> Value {
    let server = MockServer::start_async().await;
    let provider = aws_provider();
    let body = text_response();
    let (_mock, slot) =
        support::mount_capture(&server, &operation_path(API_MODEL, "converse"), &body);

    client(&server, &provider, bedrock_credentials())
        .complete(support::base_request(&selector()))
        .await
        .expect("the bearer call should succeed");
    let bearer = support::captured(&slot);

    client(&server, &provider, Credentials::AwsDefaultChain {
        region: Some("us-east-1".to_owned()),
    })
    .complete(support::base_request(&selector()))
    .await
    .expect("the signed call should succeed");
    let signed = support::captured(&slot);

    json!({ "bearer": bearer, "signed": signed })
}

/// Produces both authentication captures in a child process.
///
/// SigV4 credentials reach this crate only through the AWS default provider
/// chain, which reads the process environment. This crate denies `unsafe`, so
/// a test cannot write an environment variable in process; re-running this one
/// test in a child that carries static credentials is the hermetic way to sign
/// without depending on whatever AWS configuration a developer or a CI runner
/// happens to have. Nothing reaches AWS: the endpoint is a local mock.
///
/// This test is ignored because it is a fixture for the test below rather than
/// a test of its own.
#[cfg(feature = "bedrock-aws")]
#[ignore = "a fixture for a_signed_request_differs_only_in_authentication, which supplies the \
            credentials it needs"]
#[tokio::test]
async fn sigv4_child_writes_the_authentication_captures() {
    let path = env::var(SIGV4_CAPTURE_FILE)
        .expect("the parent test names the file to write the captures into");

    let captures = authentication_captures().await;

    fs::write(
        path,
        serde_json::to_vec(&captures).expect("the captures should serialize"),
    )
    .expect("the captures should be written");
}

#[cfg(feature = "bedrock-aws")]
#[tokio::test]
async fn a_signed_request_differs_only_in_authentication() {
    let path = env::temp_dir().join(format!("lithos-llm-sigv4-captures-{}.json", process::id()));
    let output = Command::new(env::current_exe().expect("the test binary should have a path"))
        .args(["--exact", "--ignored", SIGV4_CHILD_TEST])
        .env(SIGV4_CAPTURE_FILE, &path)
        // The AWS documentation example key pair, resolved by the environment
        // provider at the head of the default chain. `AWS_SESSION_TOKEN` is
        // cleared so an ambient temporary credential cannot add a fourth signed
        // header and change what this test pins.
        .env("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE")
        .env(
            "AWS_SECRET_ACCESS_KEY",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
        )
        .env("AWS_REGION", "us-east-1")
        .env("AWS_EC2_METADATA_DISABLED", "true")
        .env_remove("AWS_SESSION_TOKEN")
        .env_remove("AWS_PROFILE")
        .output()
        .expect("the test binary should re-run");
    assert!(
        output.status.success(),
        "the signed child process failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let captures: Value = serde_json::from_slice(
        &fs::read(&path).expect("the child process should have written its captures"),
    )
    .expect("the captures should parse");
    drop(fs::remove_file(&path));

    let bearer = &captures["bearer"];
    let signed = &captures["signed"];
    // One encoded request feeds both authentication arms, so the method, the
    // URL, and the body cannot differ between them.
    assert_eq!(signed["method"], bearer["method"]);
    assert_eq!(signed["path"], bearer["path"]);
    assert_eq!(signed["body"], bearer["body"]);

    // The signed request carries exactly one header the bearer request does
    // not: the SigV4 timestamp. Both carry an `authorization` header, whose
    // value the harness redacts, so the two header lists are otherwise equal.
    // No `x-amz-content-sha256` is sent; the payload hash lives inside the
    // signature rather than in a header of its own.
    let header_names = |capture: &Value| -> Vec<String> {
        capture["headers"]
            .as_array()
            .expect("a capture carries a header list")
            .iter()
            .map(|header| header[0].as_str().unwrap_or_default().to_owned())
            .collect()
    };
    let mut expected = header_names(bearer);
    expected.push("x-amz-date".to_owned());
    expected.sort();
    assert_eq!(header_names(signed), expected);

    crate::json_snapshot!(captures);
}
