//! Gemini `generateContent` wire parity.
//!
//! Gemini is the odd dialect in three ways, and each one has fixtures here.
//!
//! 1. The model and the operation live in the **path**, not the body:
//!    `/v1beta/models/{api_model}:generateContent`,
//!    `:streamGenerateContent?alt=sse`, and `:countTokens`. The catalog
//!    `api_model` differs from the catalog model id on purpose, so every
//!    snapshot shows which of the two the codec puts on the wire.
//! 2. The usage counters are inclusive in one direction and exclusive in the
//!    other, so [`usage_buckets_are_disjoint`] asserts the arithmetic rather
//!    than trusting the snapshot.
//! 3. The protocol supplies no tool-call ids and no stream block ids, so this
//!    codec synthesizes both.
//!
//! # Intentional differences from the reference implementation
//!
//! - Synthesized tool-call ids are **deterministic**
//!   (`{name}-{ordinal}-{responseId}`, or `{name}-{ordinal}` when the payload
//!   carries no response id). The reference minted a `Uuid::new_v4()` per call,
//!   which is why its snapshots needed a UUID scrubber and ours do not.
//! - The default `safetySettings` entry is spelled in camelCase. The reference
//!   sent the same value under `safety_settings`; proto-JSON reads both
//!   spellings as one field, and camelCase matches every other key this encoder
//!   writes.
//! - Stream content-block ids are stable per run (`text-0`, `reasoning-0`,
//!   `tool-0`). The reference reused one UUID for all text in a stream.
//! - Thought signatures are carried onto reasoning parts as
//!   [`ReasoningContent::signature`]. The reference dropped them there and kept
//!   them only on tool calls.
//! - `{"error": …}` chunks fail the stream. The reference parsed them as an
//!   empty chunk and completed normally.
//!
//! [`ReasoningContent::signature`]: lithos_llm::types::ReasoningContent::signature

use httpmock::{Method, MockServer};
use lithos_llm::types::{
    ContentPart, Error, ErrorKind, FinishReason, ImageContent, MediaSource, Message,
    ReasoningEffort, Response, ResponseFormat, RetryClassification, Role, Speed, ToolCall,
    ToolChoice, ToolDefinition, ToolResult,
};
use lithos_llm::{Client, Request};
use serde_json::{Value, json};

use crate::support::{
    self, WireCapture, assert_stream_contract, captured, client_for, collect_stream_events,
    header_credentials, mount_capture, mount_capture_sse, sse_data_transcript,
};

/// The canonical catalog provider id, which is also the codec's replay and
/// provider-option namespace.
const PROVIDER: &str = "gemini";

/// The catalog model id, which is the half of the selector a caller types.
const MODEL: &str = "gemini-2.5-pro";

/// The provider's own model id. It differs from [`MODEL`] so that a snapshot
/// showing `MODEL` in a URL would be a visible bug.
const API_MODEL: &str = "gemini-2.5-pro-002";

/// The header Gemini authenticates with. The harness redacts its value.
const API_KEY_HEADER: &str = "x-goog-api-key";

const GENERATE_PATH: &str = "/v1beta/models/gemini-2.5-pro-002:generateContent";
const STREAM_PATH: &str = "/v1beta/models/gemini-2.5-pro-002:streamGenerateContent";
const COUNT_PATH: &str = "/v1beta/models/gemini-2.5-pro-002:countTokens";

// ===========================================================================
// Harness
// ===========================================================================

fn provider() -> support::WireProvider<'static> {
    support::WireProvider::new(PROVIDER, "gemini", "gemini-generate", MODEL)
        .with_api_model(API_MODEL)
        .with_auth("{ type = \"header\", name = \"x-goog-api-key\" }")
}

/// The request selector every corpus constructor is called with.
fn selector() -> String {
    provider().selector()
}

fn client(server: &MockServer) -> Client {
    client_for(
        provider().catalog(&server.base_url()),
        PROVIDER,
        header_credentials(API_KEY_HEADER),
    )
}

/// Runs one blocking `generateContent` call and returns both halves.
async fn complete(request: Request, body: &Value) -> (WireCapture, Response) {
    let server = MockServer::start_async().await;
    let client = client(&server);
    let (mock, slot) = mount_capture(&server, GENERATE_PATH, body);

    let response = client
        .complete(request)
        .await
        .expect("the generateContent call should succeed");

    mock.assert_async().await;
    (captured(&slot), response)
}

/// Runs one `streamGenerateContent` call and returns both halves.
async fn stream(request: Request, transcript: &str) -> (WireCapture, Vec<Value>) {
    let server = MockServer::start_async().await;
    let client = client(&server);
    let (mock, slot) = mount_capture_sse(&server, STREAM_PATH, transcript);

    let stream = client
        .stream(request)
        .await
        .expect("the streamGenerateContent call should start");
    let events = collect_stream_events(stream).await;

    mock.assert_async().await;
    (captured(&slot), events)
}

/// Answers one `generateContent` call with an HTTP failure.
async fn failure(status: u16, body: &Value) -> Error {
    let server = MockServer::start_async().await;
    let client = client(&server);
    let body = body.clone();
    let mock = server.mock(|when, then| {
        when.method(Method::POST).path(GENERATE_PATH);
        then.status(status)
            .header("content-type", "application/json")
            .json_body(body);
    });

    let error = client
        .complete(support::base_request(&selector()))
        .await
        .expect_err("a failing status should not decode into a response");

    mock.assert_async().await;
    error
}

// ===========================================================================
// Provider payloads
// ===========================================================================

/// A plain text answer with a usage block.
fn text_response() -> Value {
    json!({
        "responseId": "resp-gemini-1",
        "modelVersion": "gemini-2.5-pro-002",
        "candidates": [{
            "index": 0,
            "content": { "role": "model", "parts": [{ "text": "Hello there." }] },
            "finishReason": "STOP",
        }],
        "usageMetadata": {
            "promptTokenCount": 11,
            "candidatesTokenCount": 5,
            "totalTokenCount": 16,
        },
    })
}

/// One function call, with the thought signature Gemini 3 requires on replay.
fn function_call_response() -> Value {
    json!({
        "responseId": "resp-gemini-2",
        "modelVersion": "gemini-2.5-pro-002",
        "candidates": [{
            "index": 0,
            "content": { "role": "model", "parts": [
                { "text": "Checking the weather." },
                { "functionCall": { "name": "get_weather", "args": { "city": "Paris" } },
                  "thoughtSignature": "sig-call" },
            ] },
            "finishReason": "STOP",
        }],
        "usageMetadata": {
            "promptTokenCount": 42,
            "candidatesTokenCount": 12,
            "thoughtsTokenCount": 7,
            "totalTokenCount": 61,
        },
    })
}

/// A JSON answer, for the structured-output fixtures.
fn json_response() -> Value {
    json!({
        "responseId": "resp-gemini-3",
        "modelVersion": "gemini-2.5-pro-002",
        "candidates": [{
            "index": 0,
            "content": { "role": "model", "parts": [
                { "text": "{\"city\":\"Paris\",\"population\":2102650}" },
            ] },
            "finishReason": "STOP",
        }],
        "usageMetadata": { "promptTokenCount": 9, "candidatesTokenCount": 14 },
    })
}

/// One Gemini gRPC error document.
fn grpc_error(code: u16, status: &str, message: &str) -> Value {
    json!({ "error": { "code": code, "message": message, "status": status } })
}

// ===========================================================================
// The canonical request corpus
// ===========================================================================

#[tokio::test]
async fn base_request() {
    let (request, response) = complete(support::base_request(&selector()), &text_response()).await;

    crate::json_snapshot!(request);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn multi_turn_request() {
    // System text is lifted out of `contents` into `systemInstruction`, and the
    // assistant turn becomes the `model` role.
    let (request, response) =
        complete(support::multi_turn_request(&selector()), &text_response()).await;

    crate::json_snapshot!(request);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn tools_request() {
    let (request, response) = complete(
        support::tools_request(&selector(), None),
        &function_call_response(),
    )
    .await;

    crate::json_snapshot!(request);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn tool_choice_auto() {
    let (request, response) = complete(
        support::tools_request(&selector(), Some(ToolChoice::Auto)),
        &function_call_response(),
    )
    .await;

    crate::json_snapshot!(request);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn tool_choice_none() {
    let (request, response) = complete(
        support::tools_request(&selector(), Some(ToolChoice::None)),
        &text_response(),
    )
    .await;

    crate::json_snapshot!(request);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn tool_choice_required() {
    // Gemini spells "call some tool" as mode `ANY`, with no allow list.
    let (request, response) = complete(
        support::tools_request(&selector(), Some(ToolChoice::Required)),
        &function_call_response(),
    )
    .await;

    crate::json_snapshot!(request);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn tool_choice_named() {
    // A named tool is the same `ANY` mode narrowed by `allowedFunctionNames`.
    let (request, response) = complete(
        support::tools_request(
            &selector(),
            Some(ToolChoice::Tool {
                name: "get_weather".to_owned(),
            }),
        ),
        &function_call_response(),
    )
    .await;

    crate::json_snapshot!(request);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn tool_round_trip_request() {
    // Two deliberate departures from the reference implementation are pinned
    // here. Both were reviewed and kept.
    //
    // 1. Each tool result arrives as its own canonical message, and consecutive
    //    same-role messages merge into one turn — so two parallel calls answer in a
    //    single `user` turn holding both `functionResponse` parts, which is
    //    Gemini's canonical shape. The reference sent each result as its own turn.
    //    Anthropic and Bedrock require the same merge, since both protocols
    //    alternate roles strictly.
    // 2. A failed tool result has no separate wire shape: it rides inside the
    //    free-form `functionResponse.response` object, under the `error` key
    //    Google's guidance prefers. The reference sent an `is_error` flag there,
    //    which the model reads as ordinary output rather than as a failure.
    let (request, response) = complete(
        support::tool_round_trip_request(&selector()),
        &text_response(),
    )
    .await;

    let contents = request.body["contents"]
        .as_array()
        .expect("contents should be an array");
    assert_eq!(contents.len(), 3);
    let roles: Vec<&str> = contents
        .iter()
        .filter_map(|turn| turn["role"].as_str())
        .collect();
    assert_eq!(roles, ["user", "model", "user"]);
    // Google's guidance reports a failure under an `error` key. The model
    // reads this payload, and a key it recognizes as an error reads as one,
    // where a boolean flag beside `output` reads as ordinary output.
    assert_eq!(
        contents[2]["parts"][1]["functionResponse"]["response"],
        json!({ "error": "the city is unknown" })
    );

    crate::json_snapshot!(request);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn reasoning_round_trip_request() {
    let (request, response) = complete(
        support::reasoning_round_trip_request(&selector()),
        &text_response(),
    )
    .await;

    // Both reasoning parts replay as `thought: true` parts, and the signed one
    // keeps its `thoughtSignature`. Gemini has no redacted-reasoning shape, so
    // the redacted part replays as ordinary thought text with no signature.
    let parts = &request.body["contents"][1]["parts"];
    assert_eq!(parts[0]["thought"], json!(true));
    assert_eq!(parts[0]["thoughtSignature"], json!("sig-abc"));
    assert_eq!(parts[1]["thought"], json!(true));
    assert_eq!(parts[1].get("thoughtSignature"), None);

    crate::json_snapshot!(request);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn custom_tools_are_rejected_before_dispatch() {
    let server = MockServer::start_async().await;
    let client = client(&server);
    let (mock, _slot) = mount_capture(&server, GENERATE_PATH, &text_response());

    let error = client
        .complete(support::custom_tool_request(&selector()))
        .await
        .expect_err("Gemini cannot encode a custom tool");

    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    assert_eq!(
        error.retry_classification(),
        RetryClassification::Never,
        "a request this protocol cannot express is not worth resending"
    );
    // Refusing before dispatch is the contract: a downgraded custom tool must
    // never reach the provider, so nothing may be sent at all.
    assert_eq!(
        mock.calls_async().await,
        0,
        "the codec must refuse a custom tool without sending a request"
    );
}

#[tokio::test]
async fn url_attachments_request() {
    let (request, response) = complete(
        support::url_attachments_request(&selector()),
        &text_response(),
    )
    .await;

    // A URL source becomes `fileData.fileUri`. Gemini expects a Files API URI
    // there and does not fetch arbitrary public URLs, so a public URL like the
    // ones here is REJECTED by the provider. That is deliberate: the caller
    // sees a failure rather than losing the attachment quietly, which is the
    // line this crate draws. The alternative — dropping the part — would be
    // silent data loss.
    //
    // It declares no `mimeType`, because the canonical `MediaSource::Url`
    // carries none and inventing one would be worse, and it drops the image
    // `detail` hint, which this protocol has no field for.
    let parts = &request.body["contents"][0]["parts"];
    assert_eq!(
        parts[1]["fileData"],
        json!({ "fileUri": "https://example.com/cat.png" })
    );
    assert_eq!(
        parts[2]["fileData"],
        json!({ "fileUri": "https://example.com/report.pdf" })
    );

    crate::json_snapshot!(request);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn inline_attachments_request() {
    let (request, response) = complete(
        support::inline_attachments_request(&selector()),
        &text_response(),
    )
    .await;

    // Inline bytes become `inlineData`, which does declare a `mimeType`. The
    // document name has no field here and is dropped.
    let parts = &request.body["contents"][0]["parts"];
    assert_eq!(
        parts[1]["inlineData"],
        json!({ "mimeType": "image/png", "data": "aW1hZ2UtYnl0ZXM=" })
    );
    assert_eq!(
        parts[2]["inlineData"],
        json!({ "mimeType": "application/pdf", "data": "cGRmLWJ5dGVz" })
    );

    crate::json_snapshot!(request);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn audio_request() {
    let (request, response) = complete(support::audio_request(&selector()), &text_response()).await;

    // Gemini is one of the few dialects that accepts audio, and it takes it
    // through the same `inlineData` part every other medium uses.
    assert_eq!(
        request.body["contents"][0]["parts"][1]["inlineData"],
        json!({ "mimeType": "audio/wav", "data": "YXVkaW8tYnl0ZXM=" })
    );

    crate::json_snapshot!(request);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn response_format_json_object() {
    let (request, response) = complete(
        support::response_format_request(&selector(), ResponseFormat::JsonObject),
        &json_response(),
    )
    .await;

    // A JSON object request sets the MIME type only; there is no schema to send.
    assert_eq!(
        request.body["generationConfig"]["responseMimeType"],
        json!("application/json")
    );
    assert_eq!(
        request.body["generationConfig"].get("responseJsonSchema"),
        None
    );

    crate::json_snapshot!(request);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn response_format_json_schema() {
    let (request, response) = complete(
        support::response_format_request(&selector(), support::json_schema_format()),
        &json_response(),
    )
    .await;

    // The schema goes to `responseJsonSchema`, which takes plain JSON Schema.
    // The format name has no field in this protocol and is dropped.
    assert_eq!(
        request.body["generationConfig"]["responseJsonSchema"]["required"],
        json!(["city", "population"])
    );

    crate::json_snapshot!(request);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn response_format_text_leaves_the_output_mode_alone() {
    let (request, _) = complete(
        support::response_format_request(&selector(), ResponseFormat::Text),
        &text_response(),
    )
    .await;

    // `Text` is this protocol's own default, so it sets nothing. Declaring
    // `application/json` for it — which this codec used to do for every format
    // variant — makes the model answer in JSON to a caller who asked for prose.
    assert_eq!(
        request.body["generationConfig"].get("responseMimeType"),
        None
    );
    assert!(
        !request.body.to_string().contains("application/json"),
        "a text request must not name a JSON output mode anywhere"
    );
}

#[tokio::test]
async fn sampling_request() {
    let (request, response) =
        complete(support::sampling_request(&selector()), &text_response()).await;

    // Every sampling control lands inside `generationConfig`, and the stop
    // sequences keep the order the caller gave them.
    let generation = &request.body["generationConfig"];
    // The canonical controls are `f32` and this body is `f64` JSON. Widening
    // one directly would send 0.7 as `0.699999988079071`, so the shared
    // `sampling` helper round-trips through the shortest decimal instead and
    // the provider sees the number the caller wrote.
    assert_eq!(generation["temperature"], json!(0.7));
    assert_eq!(generation["topP"], json!(0.9));
    assert_eq!(generation["stopSequences"], json!(["END", "STOP"]));
    assert_eq!(generation["maxOutputTokens"], json!(128));

    crate::json_snapshot!(request);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn metadata_request_warns_and_sends_nothing() {
    let (request, response) =
        complete(support::metadata_request(&selector()), &text_response()).await;

    // This protocol has no request-metadata field, so the entry is dropped.
    let body = request
        .body
        .as_object()
        .expect("the request body should be an object");
    assert!(!body.contains_key("metadata"));
    assert!(
        !request.body.to_string().contains("acme"),
        "no part of the dropped metadata may reach the wire"
    );

    // The loss is reported instead, which is the whole encode-time warning
    // channel proved end to end: the codec attaches the warning, the adapter
    // moves it off the encoded request, and it arrives on the decoded response.
    let codes: Vec<&str> = response
        .warnings
        .iter()
        .map(|warning| warning.code.as_str())
        .collect();
    assert_eq!(codes, ["unsupported_control"]);

    crate::json_snapshot!(request);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn speed_and_reasoning_effort_are_reported_not_sent() {
    // Neither control has a field in this protocol. Gemini does have a thinking
    // budget, but it takes a token count, and turning an effort level into a
    // defensible one needs per-model reasoning limits the catalog does not
    // carry. Warning is what the crate does with a control it cannot express;
    // guessing a budget would change how much the caller is billed.
    let request = Request::builder()
        .model(selector())
        .user("Hello")
        .speed(Speed::Fast)
        .reasoning_effort(ReasoningEffort::High)
        .max_output_tokens(128)
        .build()
        .expect("the dropped controls request should build");

    let (request, response) = complete(request, &text_response()).await;

    let body = request.body.to_string();
    assert!(!body.contains("thinkingConfig"), "{body}");
    assert!(!body.contains("speed"), "{body}");

    let warnings: Vec<(&str, &str)> = response
        .warnings
        .iter()
        .map(|warning| (warning.code.as_str(), warning.message.as_str()))
        .collect();
    assert_eq!(warnings, [
        (
            "unsupported_control",
            "this provider protocol does not support the speed control",
        ),
        (
            "unsupported_control",
            "this provider protocol does not support the reasoning effort control",
        ),
    ]);
}

#[tokio::test]
async fn default_safety_settings_are_injected() {
    let (request, _) = complete(support::base_request(&selector()), &text_response()).await;

    // Google's own defaults block a good deal of ordinary text. The reference
    // implementation relaxed the dangerous-content filter for a caller who
    // asked for nothing, and this keeps that behavior, so a prompt that worked
    // before the migration still works after it.
    //
    // One deliberate change of spelling: the reference sent `safety_settings`.
    // Proto-JSON accepts both spellings for the same field, and every other key
    // this encoder writes is camelCase, so the default is `safetySettings`.
    assert_eq!(
        request.body["safetySettings"],
        json!([{
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_ONLY_HIGH",
        }])
    );
}

#[tokio::test]
async fn caller_safety_settings_replace_the_default() {
    let request = Request::builder()
        .model(selector())
        .user("Hello")
        .provider_option(
            PROVIDER,
            "safetySettings",
            json!([{ "category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE" }]),
        )
        .max_output_tokens(128)
        .build()
        .expect("the safety settings request should build");

    let (request, _) = complete(request, &text_response()).await;

    // The default is a fallback, not a floor: what the caller wrote is what is
    // sent, and the dangerous-content entry is not added beside it.
    assert_eq!(
        request.body["safetySettings"],
        json!([{ "category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE" }])
    );
}

#[tokio::test]
async fn a_tool_result_recovers_its_function_name_from_the_call() {
    // `functionResponse` names the function, not the call. A canonical
    // `ToolResult` carries the name only when the application kept it, so the
    // encoder recovers it from the assistant turn that made the call. Without
    // that, this request would name the function `call_paris`, which matches no
    // declared function.
    let request = Request::builder()
        .model(selector())
        .user("What is the weather in Paris?")
        .tool(ToolDefinition::function(
            "get_weather",
            "Reads the current weather for a city",
            json!({ "type": "object", "properties": { "city": { "type": "string" } } }),
        ))
        .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
            ToolCall::function("call_paris", "get_weather", json!({ "city": "Paris" })),
        )]))
        .message(Message::new(Role::Tool, [ContentPart::ToolResult(
            ToolResult {
                tool_call_id: "call_paris".to_owned(),
                name:         None,
                content:      vec![ContentPart::Text {
                    text: "18C and clear".to_owned(),
                }],
                is_error:     false,
            },
        )]))
        .max_output_tokens(128)
        .build()
        .expect("the nameless tool result request should build");

    let (request, _) = complete(request, &text_response()).await;

    let result = &request.body["contents"][2]["parts"][0]["functionResponse"];
    assert_eq!(result["name"], json!("get_weather"));
    assert_eq!(result["id"], json!("call_paris"));
}

#[tokio::test]
async fn provider_options_request() {
    let (request, response) = complete(
        support::provider_options_request(&selector(), PROVIDER, "openai"),
        &text_response(),
    )
    .await;

    let body = request
        .body
        .as_object()
        .expect("the request body should be an object");
    // Only the selected namespace is merged. The failover candidate's options
    // leave no trace at all.
    assert_eq!(body["service_tier"], json!("flex"));
    assert!(!body.contains_key("unreachable_option"));
    assert!(
        !request.body.to_string().contains("this must never be sent"),
        "another provider's namespace must not reach the wire"
    );
    // `auto_cache` is a codec control, consumed rather than forwarded.
    assert!(!body.contains_key("auto_cache"));

    crate::json_snapshot!(request);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn replay_request() {
    let (request, response) = complete(
        support::replay_request(&selector(), PROVIDER),
        &function_call_response(),
    )
    .await;

    let parts = &request.body["contents"][1]["parts"];
    // Intentional difference from every string-argument dialect: Gemini takes
    // `functionCall.args` as a JSON object, not as argument text, so there are
    // no original bytes to replay. `ToolCall::raw_arguments` is deliberately
    // ignored here; the object is re-serialized, which is the only thing this
    // protocol can carry.
    assert_eq!(parts[1]["functionCall"]["args"], json!({ "city": "Paris" }));
    // This codec's own opaque part replays verbatim; another provider's is
    // dropped, so the same conversation stays sendable after failover.
    //
    // What that first part contains is worth saying out loud: it is the
    // corpus's synthetic payload, and it is not a valid Gemini `Part`, so the
    // real API would reject it. Nothing that this codec produces can reach
    // this path — `decode_response` emits text, reasoning, and tool calls, and
    // never an opaque part — so only hand-built content gets here. The
    // contract under test is the lossless-replay rule (own namespace kept,
    // every other namespace dropped), not the realism of the payload.
    assert_eq!(
        parts[0],
        json!({ "id": "rs_replay", "encrypted_content": "opaque-payload" })
    );
    assert_eq!(parts.as_array().map(Vec::len), Some(2));
    assert!(
        !request.body.to_string().contains("rs_ignored"),
        "another provider's opaque part must not reach the wire"
    );

    // One case this fixture deliberately does not reach: a message whose parts
    // are ALL foreign-namespace opaque parts. Every part resolves to `None`,
    // `generate_body` then skips the empty message, and the whole turn
    // disappears from `contents` with no warning. That is intentional — the
    // turn genuinely has nothing this protocol can carry, and dropping it is
    // what keeps a conversation sendable after failover — but it is silent,
    // which is worth knowing before anyone revisits replay.
    assert!(
        request.body["contents"]
            .as_array()
            .is_some_and(|contents| contents.len() == 3),
        "this turn survives because it still holds a tool call"
    );

    crate::json_snapshot!(request);
    crate::json_snapshot!(response);
}

/// A tool result carrying an image beside its text.
///
/// The shared corpus has text-only tool results, so this request is written
/// here. It is the one shape that reaches the `plain_text` flattening in
/// `encode_tool_result`.
fn media_tool_result_request() -> Request {
    Request::builder()
        .model(selector())
        .user("What does the chart show?")
        .tool(ToolDefinition::function(
            "read_chart",
            "Reads a chart image",
            json!({ "type": "object", "properties": {} }),
        ))
        .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
            ToolCall::function("call_chart", "read_chart", json!({})),
        )]))
        .message(Message::new(Role::Tool, [ContentPart::ToolResult(
            ToolResult {
                tool_call_id: "call_chart".to_owned(),
                name:         Some("read_chart".to_owned()),
                content:      vec![
                    ContentPart::Text {
                        text: "Revenue by quarter.".to_owned(),
                    },
                    ContentPart::Image(ImageContent::new(MediaSource::base64(
                        "Y2hhcnQtYnl0ZXM=",
                        "image/png",
                    ))),
                ],
                is_error:     false,
            },
        )]))
        .max_output_tokens(128)
        .build()
        .expect("the media tool result request should build")
}

#[tokio::test]
async fn a_tool_result_with_media_flattens_to_text_and_warns() {
    let (request, response) = complete(media_tool_result_request(), &text_response()).await;

    // `functionResponse.response.output` is a string, so `encode_tool_result`
    // keeps the text parts and the image is gone. Encoding it into the
    // free-form `response` object would be better, but that wire shape is not
    // verified against a live API, so the loss is reported rather than guessed
    // at.
    let result = &request.body["contents"][2]["parts"][0]["functionResponse"];
    assert_eq!(
        result["response"],
        json!({ "output": "Revenue by quarter." })
    );
    assert!(
        !request.body.to_string().contains("Y2hhcnQtYnl0ZXM="),
        "the flattened image must not reach the wire in some other position"
    );

    // The loss is a warning, not a refusal: the text still reaches the model,
    // so this is partial delivery rather than a caller's own attachment
    // vanishing. That is the same line drawn for dropped request metadata.
    let warnings: Vec<(&str, &str)> = response
        .warnings
        .iter()
        .map(|warning| (warning.code.as_str(), warning.message.as_str()))
        .collect();
    assert_eq!(warnings, [(
        "unsupported_control",
        "this provider protocol does not support non-text tool result content",
    )]);

    crate::json_snapshot!(request);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn a_text_only_tool_result_does_not_warn() {
    // The guard is narrow on purpose: an ordinary tool round trip loses
    // nothing, so it must stay silent. Without this, the warning above could
    // start firing for every tool call and no fixture would notice.
    let (_, response) = complete(
        support::tool_round_trip_request(&selector()),
        &text_response(),
    )
    .await;

    assert!(response.warnings.is_empty());
}

// ===========================================================================
// Usage
// ===========================================================================

#[tokio::test]
async fn usage_buckets_are_disjoint() {
    let body = json!({
        "responseId": "resp-gemini-usage",
        "candidates": [{
            "content": { "role": "model", "parts": [{ "text": "Done." }] },
            "finishReason": "STOP",
        }],
        "usageMetadata": {
            "promptTokenCount": 200,
            "cachedContentTokenCount": 180,
            "toolUsePromptTokenCount": 400,
            "candidatesTokenCount": 200,
            "thoughtsTokenCount": 300,
            "totalTokenCount": 1100,
        },
    });

    let (_, response) = complete(support::base_request(&selector()), &body).await;

    // Three separate rules, asserted separately because a snapshot would let
    // any of them silently change into any other.
    //
    // 1. `cachedContentTokenCount` is INSIDE `promptTokenCount`, so it is
    //    subtracted back out of `input`.
    assert_eq!(response.usage.cache_read, 180);
    // 2. `toolUsePromptTokenCount` sits OUTSIDE `promptTokenCount`, so it is added
    //    to `input`: (200 - 180) + 400.
    assert_eq!(response.usage.input, 420);
    // 3. `candidatesTokenCount` EXCLUDES `thoughtsTokenCount`, so `output` passes
    //    through untouched and reasoning is taken as reported.
    assert_eq!(response.usage.output, 200);
    assert_eq!(response.usage.reasoning, 300);
    // Creating a Gemini cache is a separate `cachedContents` call, so a
    // generation response never reports a cache write.
    assert_eq!(response.usage.cache_write, 0);
    assert_eq!(response.usage.total(), 1100);

    crate::json_snapshot!(response.usage);
}

// ===========================================================================
// Decoding details
// ===========================================================================

/// The ids of every tool call in a response, in order.
fn call_ids(response: &Response) -> Vec<&str> {
    response
        .content
        .iter()
        .filter_map(|part| match part {
            ContentPart::ToolCall(call) => Some(call.id.as_str()),
            _ => None,
        })
        .collect()
}

/// Two calls to the same tool, with the response id the payload should scope
/// its synthesized call ids by.
fn two_call_response(response_id: &str) -> Value {
    json!({
        "responseId": response_id,
        "candidates": [{
            "content": { "role": "model", "parts": [
                { "functionCall": { "name": "search", "args": { "query": "rust" } } },
                { "text": "and then" },
                { "functionCall": { "name": "search", "args": { "query": "gemini" } } },
            ] },
            "finishReason": "STOP",
        }],
    })
}

#[tokio::test]
async fn synthesized_tool_call_ids_are_deterministic() {
    // Gemini sends no call ids. This codec synthesizes them rather than minting
    // a UUID, so decoding one payload twice gives one answer and a snapshot
    // needs no id scrubbing.
    let body = two_call_response("resp-gemini-ids");

    let (_, first) = complete(support::base_request(&selector()), &body).await;
    let (_, second) = complete(support::base_request(&selector()), &body).await;

    // The name and the ordinal say which call this is; the response id says
    // which response it came from.
    assert_eq!(call_ids(&first), [
        "search-0-resp-gemini-ids",
        "search-1-resp-gemini-ids"
    ]);
    assert_eq!(
        first.content, second.content,
        "decoding one payload twice must produce identical ids"
    );

    crate::json_snapshot!(first.content);
}

#[tokio::test]
async fn synthesized_tool_call_ids_do_not_collide_across_turns() {
    // The plain `{name}-{ordinal}` form repeats: a conversation that calls
    // `search` on two turns carries `search-0` twice, so an application keyed
    // by call id collides and the replayed history sends duplicate ids on the
    // wire. The response id separates the turns.
    let (_, first) = complete(
        support::base_request(&selector()),
        &two_call_response("resp-turn-1"),
    )
    .await;
    let (_, second) = complete(
        support::base_request(&selector()),
        &two_call_response("resp-turn-2"),
    )
    .await;

    assert_eq!(call_ids(&first), [
        "search-0-resp-turn-1",
        "search-1-resp-turn-1"
    ]);
    assert_eq!(call_ids(&second), [
        "search-0-resp-turn-2",
        "search-1-resp-turn-2"
    ]);
}

#[tokio::test]
async fn a_payload_without_a_response_id_keeps_the_bare_synthesized_ids() {
    // Not every route sends `responseId`. The fallback is the id these calls
    // have always had, so nothing is worse for its absence.
    let mut body = two_call_response("unused");
    body.as_object_mut()
        .expect("the body should be an object")
        .remove("responseId");

    let (_, response) = complete(support::base_request(&selector()), &body).await;

    assert_eq!(call_ids(&response), ["search-0", "search-1"]);
}

#[tokio::test]
async fn a_blocked_prompt_is_an_error_not_an_empty_answer() {
    // Gemini refuses a prompt with HTTP 200 and a body that has no candidates
    // at all, only `promptFeedback`. Decoding that as a successful empty
    // response makes a blocked prompt indistinguishable from a model that had
    // nothing to say, so it fails instead.
    let error = failure(
        200,
        &json!({ "promptFeedback": { "blockReason": "SAFETY" } }),
    )
    .await;

    assert_eq!(error.kind(), ErrorKind::ContentFilter);
    assert_eq!(error.provider_code(), Some("SAFETY"));
    // A blocked prompt is blocked every time; resending it is pointless.
    assert_eq!(error.retry_classification(), RetryClassification::Never);

    crate::json_snapshot!(error.data());
}

#[tokio::test]
async fn a_body_with_no_candidates_fails_to_decode() {
    // Without a block reason there is nothing to classify: a 200 with no
    // candidates is a malformed document, not a refusal.
    let error = failure(200, &json!({ "responseId": "resp-gemini-empty" })).await;

    assert_eq!(error.kind(), ErrorKind::ResponseDecode);
    assert_eq!(error.provider_code(), None);
}

#[tokio::test]
async fn a_function_call_turn_finishes_as_a_tool_call() {
    // Gemini reports `STOP` even when the whole turn is a function call, so an
    // agent loop that dispatches tools on `ToolCall` would never run them.
    let (_, response) = complete(
        support::tools_request(&selector(), None),
        &function_call_response(),
    )
    .await;

    assert_eq!(response.finish_reason, FinishReason::ToolCall);
    assert_eq!(
        response.raw.as_ref().and_then(|raw| raw
            .pointer("/candidates/0/finishReason")
            .and_then(Value::as_str)),
        Some("STOP"),
        "the provider's own word is kept in `raw`; only the canonical reason is corrected"
    );
}

#[tokio::test]
async fn a_truncated_function_call_keeps_the_provider_reason() {
    // Only `STOP` is corrected. `MAX_TOKENS` is the more specific fact: the
    // call was cut off, and reporting it as a complete tool call would send a
    // half-written call back to the model.
    let mut body = function_call_response();
    body["candidates"][0]["finishReason"] = json!("MAX_TOKENS");

    let (_, response) = complete(support::tools_request(&selector(), None), &body).await;

    assert_eq!(response.finish_reason, FinishReason::Length);
}

#[tokio::test]
async fn the_raw_success_document_is_preserved() {
    let body = function_call_response();

    let (_, response) = complete(support::base_request(&selector()), &body).await;

    assert_eq!(
        response.raw.as_ref(),
        Some(&body),
        "a blocking response keeps the provider document exactly as sent"
    );
}

#[tokio::test]
async fn raw_options_merge_into_the_generation_config() {
    // The shared corpus carries only top-level provider options, so this one
    // request is written here to pin the recursive half of option merging:
    // overriding one key inside `generationConfig` must keep its siblings.
    let request = Request::builder()
        .model(selector())
        .user("Hello")
        .temperature(0.2)
        .stop_sequences(["END"])
        .max_output_tokens(128)
        .provider_option(PROVIDER, "generationConfig", json!({ "temperature": 0.9 }))
        .provider_option(
            PROVIDER,
            "safetySettings",
            json!([{ "category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE" }]),
        )
        .build()
        .expect("the generation config override request should build");

    let (request, _) = complete(request, &text_response()).await;

    let generation = &request.body["generationConfig"];
    assert_eq!(generation["temperature"], json!(0.9), "the raw option wins");
    assert_eq!(
        generation["maxOutputTokens"],
        json!(128),
        "a sibling the codec encoded survives the override"
    );
    assert_eq!(generation["stopSequences"], json!(["END"]));

    crate::json_snapshot!(request);
}

// ===========================================================================
// Streaming
// ===========================================================================

/// A realistic `streamGenerateContent` transcript.
///
/// It carries a thought run, a text run, and one complete function call, it
/// repeats the running usage totals the way Gemini does, and every chunk
/// repeats the response id the way Gemini does.
fn stream_transcript() -> String {
    sse_data_transcript(&[
        r#"{"responseId":"resp-gemini-stream","candidates":[{"content":{"role":"model","parts":[{"text":"Let me check","thought":true}]}}],"usageMetadata":{"promptTokenCount":18,"cachedContentTokenCount":6,"candidatesTokenCount":5}}"#,
        r#"{"responseId":"resp-gemini-stream","candidates":[{"content":{"role":"model","parts":[{"text":" the forecast.","thought":true,"thoughtSignature":"sig-stream-think"}]}}]}"#,
        r#"{"responseId":"resp-gemini-stream","candidates":[{"content":{"role":"model","parts":[{"text":"Checking"}]}}]}"#,
        r#"{"responseId":"resp-gemini-stream","candidates":[{"content":{"role":"model","parts":[{"text":" now."}]}}]}"#,
        r#"{"responseId":"resp-gemini-stream","candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"city":"Paris"}},"thoughtSignature":"sig-stream-call"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":18,"cachedContentTokenCount":6,"candidatesTokenCount":14,"thoughtsTokenCount":9}}"#,
    ])
}

/// The `type` discriminator of every collected stream item, in order.
fn kinds(events: &[Value]) -> Vec<&str> {
    events
        .iter()
        .map(|event| {
            event["type"]
                .as_str()
                .expect("every stream item should carry a type")
        })
        .collect()
}

/// The block ids of every event of one type, in order.
fn block_ids<'a>(events: &'a [Value], kind: &str) -> Vec<&'a str> {
    events
        .iter()
        .filter(|event| event["type"] == kind)
        .map(|event| {
            event["id"]
                .as_str()
                .expect("a block event should carry an id")
        })
        .collect()
}

#[tokio::test]
async fn stream_transcript_assigns_one_block_per_run() {
    let (request, events) = stream(
        support::tools_request(&selector(), None),
        &stream_transcript(),
    )
    .await;

    // The operation and the SSE opt-in both live in the URL, so the captured
    // path is where streaming is visible at all. The snapshot shows the
    // request headers carry `accept: */*` rather than `text/event-stream`:
    // Gemini keys the framing off `?alt=sse` alone, so the header is not sent,
    // and adding it would change every codec's request for the sake of a
    // hypothetical proxy that reads it.
    assert_eq!(
        request.path,
        "/v1beta/models/gemini-2.5-pro-002:streamGenerateContent?alt=sse"
    );

    assert_stream_contract(&events);

    // The protocol has no opening event, so the first chunk is what starts the
    // stream. `Started` still leads, carrying the response id that chunk named,
    // which is what every other codec does.
    let first = events.first().expect("the stream should produce events");
    assert_eq!(first["type"], json!("started"));
    assert_eq!(first["id"], json!("resp-gemini-stream"));

    // A run ends when the part kind changes; that is the only end signal the
    // protocol gives. Ids are stable per run rather than one UUID for the
    // whole stream.
    assert_eq!(block_ids(&events, "content_block_start"), [
        "reasoning-0",
        "text-0",
        "tool-0"
    ]);
    assert_eq!(block_ids(&events, "content_block_end"), [
        "reasoning-0",
        "text-0",
        "tool-0"
    ]);

    // A `functionCall` arrives whole in one chunk, so its block opens and
    // closes back to back with no delta in between.
    let types = kinds(&events);
    let start = types
        .iter()
        .rposition(|kind| *kind == "content_block_start")
        .expect("the tool call should open a block");
    assert_eq!(types[start + 1], "content_block_end");
    assert!(
        !types.contains(&"tool_call_delta"),
        "Gemini delivers complete calls, so a tool block carries no delta"
    );

    crate::json_snapshot!(request);
    crate::json_snapshot!(events);
}

#[tokio::test]
async fn stream_usage_is_last_wins_and_the_completed_response_carries_no_raw() {
    let (_, events) = stream(
        support::tools_request(&selector(), None),
        &stream_transcript(),
    )
    .await;

    let completed = events.last().expect("the stream should produce events");
    assert_eq!(completed["type"], "completed");
    let response = &completed["response"];

    // Two chunks reported usage. The totals are snapshots of the same call, so
    // the last one replaces the first rather than adding to it: accumulating
    // would give input 24 and output 19.
    assert_eq!(response["usage"]["input"], json!(12), "18 - 6, once");
    assert_eq!(response["usage"]["output"], json!(14));
    assert_eq!(response["usage"]["reasoning"], json!(9));
    assert_eq!(response["usage"]["cache_read"], json!(6));

    // Gemini sends no final response document, so there is nothing to keep.
    // `raw` is skipped when it is `None`, so its absence is the assertion.
    assert_eq!(response.get("raw"), None);

    // The stream carried a response id, so the completed response keeps it.
    assert_eq!(response["id"], json!("resp-gemini-stream"));
    // Gemini said `STOP` on a turn that called a tool, and the stream corrects
    // that the same way the blocking path does.
    assert_eq!(response["finish_reason"], json!("tool_call"));

    // The tool call keeps the deterministic synthesized id — scoped by the
    // response id, exactly as the blocking path scopes it — and the thought
    // signature that Gemini 3 needs back on the next turn.
    let call = &response["content"][2];
    assert_eq!(call["type"], json!("tool_call"));
    assert_eq!(call["id"], json!("get_weather-0-resp-gemini-stream"));
    assert_eq!(
        call["provider_metadata"]["gemini"]["thoughtSignature"],
        json!("sig-stream-call")
    );
}

#[tokio::test]
async fn stream_error_chunk_ends_the_stream_without_completing() {
    let transcript = sse_data_transcript(&[
        r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"Partial"}]}}]}"#,
        r#"{"error":{"code":429,"message":"Resource has been exhausted (e.g. check quota).","status":"RESOURCE_EXHAUSTED"}}"#,
    ]);

    let (_, events) = stream(support::base_request(&selector()), &transcript).await;

    assert_stream_contract(&events);

    let types = kinds(&events);
    assert_eq!(
        types.last(),
        Some(&"error"),
        "an error chunk must terminate the stream"
    );
    assert!(
        !types.contains(&"completed"),
        "a failed stream must never complete"
    );

    let error = &events.last().expect("the stream should produce events")["error"];
    // A mid-stream gRPC status is classified exactly as an HTTP one would be.
    assert_eq!(error["kind"], json!("rate_limit"));
    assert_eq!(error["provider_code"], json!("RESOURCE_EXHAUSTED"));

    crate::json_snapshot!(events);
}

// ===========================================================================
// Classified errors
// ===========================================================================

#[tokio::test]
async fn classifies_the_grpc_error_statuses() {
    let cases = [
        (
            401,
            "UNAUTHENTICATED",
            "API key not valid. Please pass a valid API key.",
            ErrorKind::Authentication,
        ),
        (
            403,
            "PERMISSION_DENIED",
            "Generative Language API has not been used in this project before.",
            ErrorKind::AccessDenied,
        ),
        (
            404,
            "NOT_FOUND",
            "models/gemini-2.5-pro-002 is not found for API version v1beta.",
            ErrorKind::NotFound,
        ),
        (
            429,
            "RESOURCE_EXHAUSTED",
            "Resource has been exhausted (e.g. check quota).",
            ErrorKind::RateLimit,
        ),
        (
            400,
            "INVALID_ARGUMENT",
            "Invalid JSON payload received. Unknown name \"foo\".",
            ErrorKind::InvalidRequest,
        ),
        (
            503,
            "UNAVAILABLE",
            "The model is overloaded. Please try again later.",
            ErrorKind::Server,
        ),
    ];

    let mut data = Vec::new();
    for (status, grpc_status, message, expected) in cases {
        let error = failure(status, &grpc_error(status, grpc_status, message)).await;

        assert_eq!(error.kind(), expected, "{grpc_status}");
        assert_eq!(error.provider_code(), Some(grpc_status), "{grpc_status}");
        data.push(serde_json::to_value(error.data()).expect("error data should serialize"));
    }

    // Throttling clears with backoff, and a provider-side failure is worth
    // another attempt; nothing else here is.
    assert_eq!(data[3]["retry"], json!({ "type": "safe" }));
    assert_eq!(data[5]["retry"], json!({ "type": "safe" }));
    assert_eq!(data[0]["retry"], json!({ "type": "never" }));

    crate::json_snapshot!(data);
}

#[tokio::test]
async fn a_spent_quota_is_not_a_rate_limit() {
    // Gemini reports both throttling and an exhausted billing quota as
    // RESOURCE_EXHAUSTED with HTTP 429. Only the message separates them, and
    // backoff never clears the second one.
    let error = failure(
        429,
        &grpc_error(
            429,
            "RESOURCE_EXHAUSTED",
            "You exceeded your current quota. Please check your plan and billing details.",
        ),
    )
    .await;

    assert_eq!(error.kind(), ErrorKind::QuotaExceeded);
    assert_eq!(error.provider_code(), Some("RESOURCE_EXHAUSTED"));
    assert_eq!(error.retry_classification(), RetryClassification::Never);

    crate::json_snapshot!(error.data());
}

// ===========================================================================
// Token counting
// ===========================================================================

#[tokio::test]
async fn count_input_tokens_wraps_the_whole_generate_body() {
    let server = MockServer::start_async().await;
    let client = client(&server);
    let (mock, slot) = mount_capture(&server, COUNT_PATH, &json!({ "totalTokens": 4242 }));

    let count = client
        .count_input_tokens(support::tools_request(&selector(), None))
        .await
        .expect("the countTokens call should succeed")
        .expect("Gemini has a native count endpoint");

    mock.assert_async().await;
    let request = captured(&slot);

    // The whole generateContent body is wrapped under one key and nothing is
    // stripped: the sampling controls, the system instruction, and the tools
    // all count toward the reported total, so the count matches what the
    // generation request would really send.
    let body = request
        .body
        .as_object()
        .expect("the request body should be an object");
    assert_eq!(body.len(), 1);
    assert!(!body.contains_key("contents"));
    let wrapped = &request.body["generateContentRequest"];
    assert!(wrapped.get("contents").is_some());
    assert!(wrapped.get("tools").is_some());
    assert!(wrapped.get("systemInstruction").is_some());
    assert_eq!(wrapped["generationConfig"]["maxOutputTokens"], json!(128));
    // The API reference marks the nested `model` required; the model in the
    // URL path does not populate it.
    assert_eq!(wrapped["model"], "models/gemini-2.5-pro-002");

    // The response field is camelCase, unlike every snake_case count response
    // in the other dialects.
    assert_eq!(count.tokens(), 4242);
    assert_eq!(count.model().model().as_str(), MODEL);

    crate::json_snapshot!(request);
}
