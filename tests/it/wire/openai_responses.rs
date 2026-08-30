//! The OpenAI Responses dialect, `POST /v1/responses`.
//!
//! This dialect is the widest of the five. It is the only one that encodes a
//! custom tool, the only one that replays provider-native reasoning items
//! verbatim, the only one whose stream terminates with a complete response
//! document, and the only one with a native input token count endpoint.
//!
//! These tests drive the public client API against a local mock server and pin
//! both halves of each exchange. The codec's own unit tests in
//! `src/codecs/openai.rs` already cover translation in isolation; what is
//! pinned here is the HTTP-level contract — method, path, headers, the exact
//! body bytes, and what a caller observes coming back.

use httpmock::{Method, MockServer};
use lithos_llm::types::{
    ContentPart, ErrorKind, FinishReason, ImageContent, MediaSource, Message, ReasoningEffort,
    ResponseFormat, RetryClassification, Role, ToolCall, ToolChoice, ToolDefinition, ToolResult,
};
use lithos_llm::{Client, Request};
use serde_json::{Value, json};

use crate::support;

/// The catalog model id every test routes to.
const MODEL: &str = "gpt-5.6-luna";

/// The provider's own model id, which is what must appear on the wire.
///
/// It differs from [`MODEL`] on purpose: a snapshot showing the catalog id
/// would mean the codec sent a selector rather than an API model.
const API_MODEL: &str = "gpt-5.6-luna-2026-04-01";

/// The catalog provider id, which is also the provider-options namespace.
const PROVIDER: &str = "openai";

/// The generation route.
const RESPONSES_PATH: &str = "/v1/responses";

/// The native input token count route.
const COUNT_PATH: &str = "/v1/responses/input_tokens";

/// The provider under test: bearer auth, every capability, a distinct API
/// model.
fn provider() -> support::WireProvider<'static> {
    support::WireProvider::new(PROVIDER, "openai", "openai-responses", MODEL)
        .with_api_model(API_MODEL)
        .with_auth("{ type = \"bearer\" }")
}

/// The request selector for the provider under test.
fn selector() -> String {
    provider().selector()
}

/// A mock server and a client pointed at it.
async fn wire() -> (MockServer, Client) {
    let server = MockServer::start_async().await;
    let client = support::client_for(
        provider().catalog(&server.base_url()),
        PROVIDER,
        support::bearer_credentials(),
    );
    (server, client)
}

// ===========================================================================
// Canned provider documents
// ===========================================================================

/// A completed response carrying one visible message.
fn text_document() -> Value {
    json!({
        "id": "resp_text",
        "object": "response",
        "created_at": 1_772_000_000_u64,
        "status": "completed",
        "model": API_MODEL,
        "output": [{
            "type": "message",
            "id": "msg_text",
            "status": "completed",
            "role": "assistant",
            "content": [{ "type": "output_text", "text": "Hello there.", "annotations": [] }],
        }],
        "usage": {
            "input_tokens": 12,
            "input_tokens_details": { "cached_tokens": 0 },
            "output_tokens": 4,
            "output_tokens_details": { "reasoning_tokens": 0 },
            "total_tokens": 16,
        },
    })
}

/// A completed response carrying an opaque reasoning item and a function call.
///
/// The reasoning item has no visible summary and does carry
/// `encrypted_content`, which is the shape that must survive as an opaque
/// replay part rather than as reasoning text.
fn function_call_document() -> Value {
    json!({
        "id": "resp_tool",
        "object": "response",
        "status": "completed",
        "model": API_MODEL,
        "output": [
            {
                "type": "reasoning",
                "id": "rs_tool",
                "summary": [],
                "encrypted_content": "gAAAAAopaque-reasoning-state",
            },
            {
                "type": "function_call",
                "id": "fc_tool",
                "call_id": "call_weather",
                "name": "get_weather",
                "arguments": "{\"city\":\"Paris\"}",
                "status": "completed",
            },
        ],
        "usage": {
            "input_tokens": 44,
            "input_tokens_details": { "cached_tokens": 0 },
            "output_tokens": 21,
            "output_tokens_details": { "reasoning_tokens": 16 },
            "total_tokens": 65,
        },
    })
}

/// A completed response carrying one custom tool call.
fn custom_tool_call_document() -> Value {
    json!({
        "id": "resp_custom",
        "object": "response",
        "status": "completed",
        "model": API_MODEL,
        "output": [{
            "type": "custom_tool_call",
            "id": "ctc_patch",
            "call_id": "call_patch",
            "name": "apply_patch",
            "input": "*** Begin Patch\n*** End Patch",
            "status": "completed",
        }],
        "usage": {
            "input_tokens": 31,
            "input_tokens_details": { "cached_tokens": 0 },
            "output_tokens": 12,
            "output_tokens_details": { "reasoning_tokens": 0 },
            "total_tokens": 43,
        },
    })
}

// ===========================================================================
// The shared corpus
// ===========================================================================

/// A completed response whose output interleaves reasoning with the items it
/// anchors.
///
/// This is the shape a replay has to reproduce exactly: each `reasoning` item
/// names the item that must follow it, so the assistant `message` between the
/// two reasoning items cannot be reordered or reconstructed without an `id`.
fn interleaved_document() -> Value {
    json!({
        "id": "resp_interleaved",
        "object": "response",
        "status": "completed",
        "model": API_MODEL,
        "output": [
            {
                "type": "reasoning",
                "id": "rs_first",
                "summary": [],
                "encrypted_content": "gAAAAAfirst-reasoning-state",
            },
            {
                "type": "message",
                "id": "msg_interleaved",
                "status": "completed",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "Checking now.", "annotations": [] }],
            },
            {
                "type": "reasoning",
                "id": "rs_second",
                "summary": [],
                "encrypted_content": "gAAAAAsecond-reasoning-state",
            },
            {
                "type": "function_call",
                "id": "fc_interleaved",
                "call_id": "call_weather",
                "name": "get_weather",
                "arguments": "{\"city\":\"Paris\"}",
                "status": "completed",
            },
        ],
        "usage": {
            "input_tokens": 20,
            "input_tokens_details": { "cached_tokens": 0 },
            "output_tokens": 9,
            "output_tokens_details": { "reasoning_tokens": 4 },
            "total_tokens": 29,
        },
    })
}

#[tokio::test]
async fn encodes_the_base_request() {
    let (server, client) = wire().await;
    let (_mock, slot) = support::mount_capture(&server, RESPONSES_PATH, &text_document());

    let response = client
        .complete(support::base_request(&selector()))
        .await
        .expect("the base request should complete");

    crate::json_snapshot!(support::captured(&slot));
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn encodes_a_multi_turn_conversation() {
    let (server, client) = wire().await;
    let (_mock, slot) = support::mount_capture(&server, RESPONSES_PATH, &text_document());

    let response = client
        .complete(support::multi_turn_request(&selector()))
        .await
        .expect("the multi-turn request should complete");

    crate::json_snapshot!(support::captured(&slot));
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn encodes_tools_and_decodes_a_tool_call() {
    let (server, client) = wire().await;
    let (_mock, slot) = support::mount_capture(&server, RESPONSES_PATH, &function_call_document());

    let response = client
        .complete(support::tools_request(&selector(), None))
        .await
        .expect("the tools request should complete");

    crate::json_snapshot!(support::captured(&slot));
    crate::json_snapshot!(response);
}

/// The four selection modes share one tool set, so only the request differs.
///
/// Decoding a tool call is pinned once, by
/// [`encodes_tools_and_decodes_a_tool_call`]; repeating it here would pin the
/// same canned document four more times without pinning anything new.
#[tokio::test]
async fn encodes_tool_choice_auto() {
    let (server, client) = wire().await;
    let (_mock, slot) = support::mount_capture(&server, RESPONSES_PATH, &text_document());

    client
        .complete(support::tools_request(&selector(), Some(ToolChoice::Auto)))
        .await
        .expect("the tool choice request should complete");

    crate::json_snapshot!(support::captured(&slot));
}

#[tokio::test]
async fn encodes_tool_choice_none() {
    let (server, client) = wire().await;
    let (_mock, slot) = support::mount_capture(&server, RESPONSES_PATH, &text_document());

    client
        .complete(support::tools_request(&selector(), Some(ToolChoice::None)))
        .await
        .expect("the tool choice request should complete");

    // Unlike the Anthropic dialect, this protocol keeps the tool list when the
    // choice is `none`, because `tool_choice: "none"` is the field that
    // suppresses calling. Dropping the tools would change what the model was
    // told it has.
    crate::json_snapshot!(support::captured(&slot));
}

#[tokio::test]
async fn encodes_tool_choice_required() {
    let (server, client) = wire().await;
    let (_mock, slot) = support::mount_capture(&server, RESPONSES_PATH, &text_document());

    client
        .complete(support::tools_request(
            &selector(),
            Some(ToolChoice::Required),
        ))
        .await
        .expect("the tool choice request should complete");

    crate::json_snapshot!(support::captured(&slot));
}

#[tokio::test]
async fn encodes_a_named_tool_choice() {
    let (server, client) = wire().await;
    let (_mock, slot) = support::mount_capture(&server, RESPONSES_PATH, &text_document());

    client
        .complete(support::tools_request(
            &selector(),
            Some(ToolChoice::Tool {
                name: "get_weather".to_owned(),
            }),
        ))
        .await
        .expect("the tool choice request should complete");

    crate::json_snapshot!(support::captured(&slot));
}

/// Two tool calls and their two results, one of which failed.
///
/// `function_call_output` has no error field, so a failed result is marked by
/// `"status": "incomplete"` on the output item and by nothing else.
#[tokio::test]
async fn encodes_a_tool_round_trip() {
    let (server, client) = wire().await;
    let (_mock, slot) = support::mount_capture(&server, RESPONSES_PATH, &text_document());

    let response = client
        .complete(support::tool_round_trip_request(&selector()))
        .await
        .expect("the tool round trip request should complete");

    crate::json_snapshot!(support::captured(&slot));
    crate::json_snapshot!(response);
}

/// Reasoning replayed as history is dropped, and the caller is told.
///
/// Intentional difference from the reference implementation, and from every
/// other dialect here: this protocol cannot accept reasoning text as an input
/// item. Only the provider's own `reasoning` item, replayed verbatim through an
/// `openai.*` opaque part, anchors the reasoning chain. Both
/// [`ContentPart::Reasoning`] parts in this corpus entry — the signed one and
/// the redacted one — therefore leave no trace on the wire. Replay for this
/// dialect is pinned by [`replays_its_own_namespace_and_skips_others`].
///
/// The drop is correct but not silent: the decoded response carries one
/// `unsupported_control` warning, so an application can see that the history it
/// sent was not the history the provider received.
///
/// [`ContentPart::Reasoning`]: lithos_llm::types::ContentPart::Reasoning
#[tokio::test]
async fn drops_replayed_reasoning_text_and_warns() {
    let (server, client) = wire().await;
    let (_mock, slot) = support::mount_capture(&server, RESPONSES_PATH, &text_document());

    let response = client
        .complete(support::reasoning_round_trip_request(&selector()))
        .await
        .expect("the reasoning round trip request should complete");

    assert_eq!(
        response.warnings.len(),
        1,
        "dropping replayed reasoning text must warn exactly once: {:?}",
        response.warnings,
    );
    crate::json_snapshot!(support::captured(&slot));
    crate::json_snapshot!(response);
}

/// The only dialect that carries custom tools.
///
/// Three shapes are pinned at once: the `{"type":"custom",...}` definition, the
/// `custom_tool_call` input item whose payload is `input` rather than
/// `arguments`, and the `custom_tool_call_output` result item, which has no
/// error channel at all. The decode half pins that a `custom_tool_call` output
/// item comes back as a custom [`ToolCall`] whose arguments stay a string.
///
/// [`ToolCall`]: lithos_llm::types::ToolCall
#[tokio::test]
async fn encodes_and_decodes_a_custom_tool_round_trip() {
    let (server, client) = wire().await;
    let (_mock, slot) =
        support::mount_capture(&server, RESPONSES_PATH, &custom_tool_call_document());

    let response = client
        .complete(support::custom_tool_request(&selector()))
        .await
        .expect("the custom tool request should complete");

    crate::json_snapshot!(support::captured(&slot));
    crate::json_snapshot!(response);
}

/// A tool result carrying an image loses the image, and says so.
///
/// A `function_call_output` takes a string, so an image a tool returned cannot
/// travel with it. The text alongside it still reaches the model, which is why
/// this warns rather than refusing the way audio does.
///
/// The shared corpus has no tool result with media in it — every corpus tool
/// result is text — so this request is built here, as the replay and JSON tool
/// result tests below do for the same reason.
#[tokio::test]
async fn warns_when_a_tool_result_carries_media() {
    let (server, client) = wire().await;
    let (_mock, slot) = support::mount_capture(&server, RESPONSES_PATH, &text_document());
    let request = Request::builder()
        .model(selector())
        .user("Screenshot the page.")
        .tool(ToolDefinition::function(
            "screenshot",
            "Captures the current page",
            json!({ "type": "object", "properties": {} }),
        ))
        .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
            ToolCall::function("call_shot", "screenshot", json!({})),
        )]))
        .message(Message::new(Role::Tool, [ContentPart::ToolResult(
            ToolResult {
                tool_call_id: "call_shot".to_owned(),
                name:         Some("screenshot".to_owned()),
                content:      vec![
                    ContentPart::Text {
                        text: "captured".to_owned(),
                    },
                    ContentPart::Image(ImageContent {
                        source: MediaSource::base64("c2hvdC1ieXRlcw==", "image/png"),
                        detail: None,
                    }),
                ],
                is_error:     false,
            },
        )]))
        .max_output_tokens(128)
        .build()
        .expect("the media tool result request should build");

    let response = client
        .complete(request)
        .await
        .expect("a lost image warns rather than failing the call");

    let captured = support::captured(&slot);
    assert!(
        !captured.body.to_string().contains("c2hvdC1ieXRlcw=="),
        "the image bytes cannot travel in a function_call_output",
    );
    assert_eq!(
        response.warnings.len(),
        1,
        "losing tool result media must warn exactly once: {:?}",
        response.warnings,
    );
    crate::json_snapshot!(captured);
    crate::json_snapshot!(response.warnings);
}

/// An image and a document the provider fetches itself.
///
/// Both ride as URLs the provider resolves: an image as `input_image.image_url`
/// and a document as `input_file.file_url`. The document keeps its `filename`,
/// which is what the model sees when it cites the file.
#[tokio::test]
async fn encodes_url_attachments() {
    let (server, client) = wire().await;
    let (_mock, slot) = support::mount_capture(&server, RESPONSES_PATH, &text_document());

    client
        .complete(support::url_attachments_request(&selector()))
        .await
        .expect("the URL attachments request should complete");

    crate::json_snapshot!(support::captured(&slot));
}

/// Inline bytes, which this protocol takes as a `data:` URL.
///
/// The two attachment kinds put that URL in different fields: an image in
/// `image_url` and a document in `file_data`. Sending a document's bytes in the
/// image field, or the other way round, is a mistake only a fixture holding
/// both can catch.
#[tokio::test]
async fn encodes_inline_attachments() {
    let (server, client) = wire().await;
    let (_mock, slot) = support::mount_capture(&server, RESPONSES_PATH, &text_document());

    client
        .complete(support::inline_attachments_request(&selector()))
        .await
        .expect("the inline attachments request should complete");

    crate::json_snapshot!(support::captured(&slot));
}

/// Audio is refused before dispatch, not faked.
///
/// This protocol takes no audio input. The codec could substitute a text
/// placeholder, and an earlier version did, but that puts words the caller
/// never wrote into their prompt and the model answers them. Failing early is
/// the safer contract, and the test proves it is early: the mock must record
/// zero calls, so nothing was sent and nothing was billed.
/// INTENTIONAL DIFFERENCE: the reference sent audio to this endpoint as an
/// `input_audio` part the Responses API does not define, and the provider
/// answered a prompt the caller never wrote. Audio is refused before dispatch
/// instead.
#[tokio::test]
async fn rejects_audio_before_dispatch() {
    let (server, client) = wire().await;
    let (mock, _slot) = support::mount_capture(&server, RESPONSES_PATH, &text_document());

    let error = client
        .complete(support::audio_request(&selector()))
        .await
        .expect_err("this protocol cannot carry audio");

    mock.assert_calls_async(0).await;
    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    crate::json_snapshot!(error.data());
}

#[tokio::test]
async fn encodes_a_json_object_response_format() {
    let (server, client) = wire().await;
    let (_mock, slot) = support::mount_capture(&server, RESPONSES_PATH, &text_document());

    client
        .complete(support::response_format_request(
            &selector(),
            ResponseFormat::JsonObject,
        ))
        .await
        .expect("the response format request should complete");

    crate::json_snapshot!(support::captured(&slot));
}

/// The schema goes under `text.format`, named and strict.
#[tokio::test]
async fn encodes_a_json_schema_response_format() {
    let (server, client) = wire().await;
    let (_mock, slot) = support::mount_capture(&server, RESPONSES_PATH, &text_document());

    client
        .complete(support::response_format_request(
            &selector(),
            support::json_schema_format(),
        ))
        .await
        .expect("the response format request should complete");

    crate::json_snapshot!(support::captured(&slot));
}

/// Sampling controls and metadata reach the wire; stop sequences do not.
///
/// `stop` is the one worth stating out loud: the reference implementation
/// sends it, but the live `/v1/responses` endpoint answers a `stop` member
/// with a 400 "Unknown parameter: 'stop'" on every model, reasoning or not
/// (probed against gpt-5.6-luna, gpt-5.4, and gpt-4o on 2026-08-30). The
/// codec drops the sequences and warns, so the one warning here is the
/// pinned behavior, not an accident.
///
/// The sampling values are pinned as the decimals the caller wrote. `Request`
/// holds them as `f32`, and widening one to `f64` would put
/// `0.699999988079071` on the wire for a caller's `0.7`.
#[tokio::test]
async fn encodes_sampling_controls_and_drops_stop_sequences() {
    let (server, client) = wire().await;
    let (_mock, slot) = support::mount_capture(&server, RESPONSES_PATH, &text_document());

    let response = client
        .complete(support::sampling_request(&selector()))
        .await
        .expect("the sampling request should complete");

    let captured = support::captured(&slot);
    assert_eq!(
        captured.body.get("stop"),
        None,
        "the live endpoint rejects a stop member, so none may be sent",
    );
    assert_eq!(
        captured.body.get("temperature").map(Value::to_string),
        Some("0.7".to_owned()),
        "the caller's own decimal must reach the wire",
    );
    assert_eq!(
        captured.body.get("top_p").map(Value::to_string),
        Some("0.9".to_owned()),
    );
    assert_eq!(
        response.warnings.len(),
        1,
        "the dropped stop sequences must be the one warning: {:?}",
        response.warnings,
    );
    crate::json_snapshot!(captured);
    crate::json_snapshot!(response);
}

/// A reasoning effort reaches the wire as `reasoning.effort`.
#[tokio::test]
async fn encodes_reasoning_effort() {
    let (server, client) = wire().await;
    let (_mock, slot) = support::mount_capture(&server, RESPONSES_PATH, &text_document());
    let request = Request::builder()
        .model(selector())
        .user("Think hard about this.")
        .reasoning_effort(ReasoningEffort::High)
        .build()
        .expect("the effort request should build");

    let response = client
        .complete(request)
        .await
        .expect("the effort request should complete");

    let captured = support::captured(&slot);
    assert_eq!(
        captured.body.get("reasoning"),
        Some(&json!({ "effort": "high" })),
        "the normalized effort level is the provider's own spelling here",
    );
    crate::json_snapshot!(captured);
    crate::json_snapshot!(response);
}

/// A `status: incomplete` document on the blocking path decodes as `Length`.
///
/// A consumer's tool loop branches on the finish reason, so a turn the output
/// limit cut short must never read as one the model finished on its own.
#[tokio::test]
async fn an_incomplete_response_decodes_as_length() {
    let (server, client) = wire().await;
    let mut document = text_document();
    document["status"] = json!("incomplete");
    document["incomplete_details"] = json!({ "reason": "max_output_tokens" });
    let (_mock, _slot) = support::mount_capture(&server, RESPONSES_PATH, &document);

    let response = client
        .complete(support::base_request(&selector()))
        .await
        .expect("an incomplete answer still decodes");

    assert_eq!(response.finish_reason, FinishReason::Length);
    assert_eq!(response.text(), "Hello there.");
    crate::json_snapshot!(response);
}

/// The Codex deployment streams every call and takes no sampling controls.
///
/// A blocking `complete` against a Codex-mode provider is served by a
/// streaming request whose events are assembled into one response, and the
/// sampling fields the deployment rejects are left off the wire. This is
/// adapter behavior layered over the codec, so only a wire test can pin it.
#[tokio::test]
async fn codex_mode_streams_a_complete_call_and_drops_sampling_controls() {
    let server = MockServer::start_async().await;
    let source = format!(
        "{}\n[providers.\"{PROVIDER}\".adapter_options]\nmode = \"codex\"\n",
        provider().toml(&server.base_url())
    );
    let client = support::client_for(
        support::catalog_from_toml("wire-codex", &source),
        PROVIDER,
        support::bearer_credentials(),
    );
    let completed =
        json!({ "type": "response.completed", "response": text_document() }).to_string();
    let transcript = support::sse_transcript(&[
        (
            "response.created",
            r#"{"type":"response.created","response":{"id":"resp_text","status":"in_progress"}}"#,
        ),
        (
            "response.output_item.added",
            r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_text","role":"assistant","content":[]}}"#,
        ),
        (
            "response.output_text.delta",
            r#"{"type":"response.output_text.delta","item_id":"msg_text","delta":"Hello there."}"#,
        ),
        (
            "response.output_item.done",
            r#"{"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_text","role":"assistant","content":[{"type":"output_text","text":"Hello there."}]}}"#,
        ),
        ("response.completed", &completed),
    ]);
    // Codex mode posts to the unversioned `/responses` path — the live
    // deployment's `/v1/responses` is an HTML 403 (2026-08-30).
    let (_mock, slot) = support::mount_capture_sse(&server, "/responses", &transcript);

    let response = client
        .complete(support::sampling_request(&selector()))
        .await
        .expect("a codex-mode complete call should be served by a stream");

    let captured = support::captured(&slot);
    assert_eq!(
        captured.body.get("stream"),
        Some(&json!(true)),
        "codex mode streams a blocking call"
    );
    assert_eq!(
        captured.body.get("temperature"),
        None,
        "the deployment rejects sampling controls"
    );
    assert_eq!(captured.body.get("top_p"), None);
    assert_eq!(response.text(), "Hello there.");
    crate::json_snapshot!(captured);
    crate::json_snapshot!(response);
}

#[tokio::test]
async fn encodes_request_metadata() {
    let (server, client) = wire().await;
    let (_mock, slot) = support::mount_capture(&server, RESPONSES_PATH, &text_document());

    let response = client
        .complete(support::metadata_request(&selector()))
        .await
        .expect("the metadata request should complete");

    assert!(
        response.warnings.is_empty(),
        "this protocol carries request metadata natively: {:?}",
        response.warnings,
    );
    crate::json_snapshot!(support::captured(&slot));
}

/// Raw provider options: only the selected namespace, and it wins.
///
/// Three separate contracts are pinned here. `max_output_tokens` is set by both
/// the typed field and the raw option, and the raw value wins. `service_tier`
/// has no typed field and is added. `auto_cache` is a control key the codec
/// consumes, so it must not appear. The failover candidate's namespace must
/// leave no trace at all.
#[tokio::test]
async fn merges_only_the_selected_provider_options() {
    let (server, client) = wire().await;
    let (_mock, slot) = support::mount_capture(&server, RESPONSES_PATH, &text_document());

    client
        .complete(support::provider_options_request(
            &selector(),
            PROVIDER,
            "anthropic",
        ))
        .await
        .expect("the provider options request should complete");

    let captured = support::captured(&slot);
    let body = captured
        .body
        .as_object()
        .expect("the request body should be a JSON object");
    assert_eq!(
        body.get("max_output_tokens"),
        Some(&json!(256)),
        "a raw option must override the field the codec generated",
    );
    assert!(
        !body.contains_key("auto_cache"),
        "a control key must never reach the wire",
    );
    assert!(
        !captured.body.to_string().contains("unreachable_option"),
        "another provider's namespace must leave no trace",
    );
    crate::json_snapshot!(captured);
}

/// Lossless replay for this codec's own namespace.
///
/// The `openai.reasoning` opaque part is re-emitted verbatim as an input item
/// and comes first, so the `function_call` it anchors follows it. The
/// `other_provider.reasoning` part is skipped without comment, which is what
/// keeps a conversation sendable after failover. `raw_arguments` replays byte
/// for byte — note the space after the colon, which a re-serialization would
/// remove and which would break the provider's prompt cache. The `item_id` from
/// this codec's provider metadata comes back as the item's `id`.
#[tokio::test]
async fn replays_its_own_namespace_and_skips_others() {
    let (server, client) = wire().await;
    let (_mock, slot) = support::mount_capture(&server, RESPONSES_PATH, &text_document());

    client
        .complete(support::replay_request(&selector(), "openai"))
        .await
        .expect("the replay request should complete");

    let captured = support::captured(&slot);
    assert!(
        !captured.body.to_string().contains("rs_ignored"),
        "an opaque part for another provider must not reach the wire",
    );
    crate::json_snapshot!(captured);
}

/// An assistant turn survives a complete round trip through this protocol.
///
/// This is the reference implementation's `reasoning_message_function_call`
/// round trip, run end to end: the provider's own output items come back as
/// content, the caller replays that content unchanged, and every item reaches
/// the wire with the identity it was issued with.
///
/// Three things are pinned, all of which the provider enforces:
///
/// 1. The assistant `message` item is sent verbatim, with its `id` and
///    `status`, rather than rebuilt from the text part. A reasoning item names
///    the item that must follow it, and a reconstructed message has no id to be
///    named by.
/// 2. Items keep their original order, so each reasoning item still sits
///    directly before the item it anchors. Hoisting the replay items to the
///    front would pair `rs_second` with the wrong item.
/// 3. The assistant's text travels once. It rides inside the preserved item, so
///    the extracted text part must not also become an item of its own.
#[tokio::test]
async fn replays_an_assistant_turn_with_its_own_items() {
    let (server, client) = wire().await;
    let (_mock, slot) = support::mount_capture(&server, RESPONSES_PATH, &interleaved_document());

    let first = client
        .complete(support::base_request(&selector()))
        .await
        .expect("the first request should complete");

    let replay = Request::builder()
        .model(selector())
        .user("What is the weather in Paris?")
        .message(Message::new(Role::Assistant, first.content.clone()))
        .max_output_tokens(128)
        .build()
        .expect("the replay request should build");
    client
        .complete(replay)
        .await
        .expect("the replay request should complete");

    let captured = support::captured(&slot);
    let input = captured.body["input"]
        .as_array()
        .expect("the replayed body should carry input items");
    let identities: Vec<Value> = input
        .iter()
        .map(|item| json!([item.get("type"), item.get("id")]))
        .collect();
    assert_eq!(identities, vec![
        json!([null, null]),
        json!(["reasoning", "rs_first"]),
        json!(["message", "msg_interleaved"]),
        json!(["reasoning", "rs_second"]),
        json!(["function_call", "fc_interleaved"]),
    ]);
    assert_eq!(
        captured.body.to_string().matches("Checking now.").count(),
        1,
        "the preserved message item already carries the assistant text",
    );
    crate::json_snapshot!(captured);
    crate::json_snapshot!(first);
}

/// A tool result that is only JSON sends the JSON, not the envelope around it.
///
/// A `function_call_output` takes a string, so the value has to be serialized.
/// Serializing the `ContentPart` list instead would hand the model
/// `[{"type":"json","value":…}]`, which is this crate's own shape rather than
/// anything the tool returned.
#[tokio::test]
async fn sends_a_json_tool_result_as_the_bare_value() {
    let (server, client) = wire().await;
    let (_mock, slot) = support::mount_capture(&server, RESPONSES_PATH, &text_document());
    let request = Request::builder()
        .model(selector())
        .user("What is the weather in Paris?")
        .tool(ToolDefinition::function(
            "get_weather",
            "Looks up the weather",
            json!({ "type": "object", "properties": {} }),
        ))
        .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
            ToolCall::function("call_weather", "get_weather", json!({ "city": "Paris" })),
        )]))
        .message(Message::new(Role::Tool, [ContentPart::ToolResult(
            ToolResult {
                tool_call_id: "call_weather".to_owned(),
                name:         Some("get_weather".to_owned()),
                content:      vec![ContentPart::Json {
                    value: json!({ "temperature_c": 18, "sky": "clear" }),
                }],
                is_error:     false,
            },
        )]))
        .max_output_tokens(128)
        .build()
        .expect("the JSON tool result request should build");

    client
        .complete(request)
        .await
        .expect("the JSON tool result request should complete");

    let captured = support::captured(&slot);
    let output = captured.body["input"][2]["output"]
        .as_str()
        .expect("the tool result should be a function_call_output");
    assert_eq!(
        serde_json::from_str::<Value>(output).expect("the output should be the tool's own JSON"),
        json!({ "temperature_c": 18, "sky": "clear" }),
    );
    crate::json_snapshot!(captured);
}

// ===========================================================================
// Dialect-specific decoding
// ===========================================================================

/// Inclusive provider counters become disjoint canonical buckets.
///
/// `input_tokens` already includes `input_tokens_details.cached_tokens` and
/// `input_tokens_details.cache_write_tokens`, and `output_tokens` already
/// includes `output_tokens_details.reasoning_tokens`. Adding the buckets a
/// caller sees must therefore reproduce the provider's own total, never
/// double-count. The write counter is real billing data: the GPT-5.6 family
/// prices cache writes at 1.25x input and reported them through
/// `cache_write_tokens` in a live pair on 2026-08-30.
#[tokio::test]
async fn decodes_inclusive_usage_into_disjoint_buckets() {
    let (server, client) = wire().await;
    let document = json!({
        "id": "resp_usage",
        "object": "response",
        "status": "completed",
        "model": API_MODEL,
        "output": [],
        "usage": {
            "input_tokens": 100,
            "input_tokens_details": { "cached_tokens": 80, "cache_write_tokens": 15 },
            "output_tokens": 50,
            "output_tokens_details": { "reasoning_tokens": 20 },
            "total_tokens": 150,
        },
    });
    let (_mock, _slot) = support::mount_capture(&server, RESPONSES_PATH, &document);

    let response = client
        .complete(support::base_request(&selector()))
        .await
        .expect("the usage request should complete");

    assert_eq!(
        response.usage.input, 5,
        "input must exclude cached and cache-written tokens"
    );
    assert_eq!(
        response.usage.output, 30,
        "output must exclude reasoning tokens"
    );
    assert_eq!(response.usage.reasoning, 20);
    assert_eq!(response.usage.cache_read, 80);
    assert_eq!(
        response.usage.cache_write, 15,
        "the billed cache write must land in its own bucket"
    );
    assert_eq!(
        response.usage.total(),
        150,
        "the disjoint buckets must add up to the provider's own total",
    );
    crate::json_snapshot!(response);
}

/// A `function_call` with no name never becomes a tool call.
///
/// The model emits these for its own bookkeeping. There is no tool to route one
/// to and no result to send back, so it is not content, and it must not make
/// the turn look like a tool call to a caller branching on the finish reason.
#[tokio::test]
async fn drops_model_internal_tool_calls() {
    let (server, client) = wire().await;
    let document = json!({
        "id": "resp_internal",
        "object": "response",
        "status": "completed",
        "model": API_MODEL,
        "output": [
            {
                "type": "function_call",
                "id": "fc_internal",
                "call_id": "call_internal",
                "name": "",
                "arguments": "{}",
                "status": "completed",
            },
            {
                "type": "message",
                "id": "msg_internal",
                "status": "completed",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "All done.", "annotations": [] }],
            },
        ],
    });
    let (_mock, _slot) = support::mount_capture(&server, RESPONSES_PATH, &document);

    let response = client
        .complete(support::base_request(&selector()))
        .await
        .expect("the request should complete");

    assert!(
        !response
            .content
            .iter()
            .any(|part| matches!(part, ContentPart::ToolCall(_))),
        "a call with no name is not a call the caller can answer",
    );
    assert_eq!(
        response.finish_reason,
        FinishReason::Stop,
        "a model-internal item must not report a tool call",
    );
    crate::json_snapshot!(response);
}

/// The provider's whole success body survives on [`Response::raw`].
///
/// The document carries fields this crate has no typed home for —
/// `service_tier`, `web_search_call` — and a caller must still be able to read
/// them.
///
/// [`Response::raw`]: lithos_llm::types::Response::raw
#[tokio::test]
async fn preserves_the_raw_success_document() {
    let (server, client) = wire().await;
    let document = json!({
        "id": "resp_raw",
        "object": "response",
        "status": "completed",
        "model": API_MODEL,
        "service_tier": "default",
        "output": [
            { "type": "web_search_call", "id": "ws_1", "status": "completed" },
            {
                "type": "message",
                "id": "msg_raw",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "Sunny." }],
            },
        ],
        "usage": {
            "input_tokens": 5,
            "input_tokens_details": { "cached_tokens": 0 },
            "output_tokens": 2,
            "output_tokens_details": { "reasoning_tokens": 0 },
            "total_tokens": 7,
        },
    });
    let (_mock, _slot) = support::mount_capture(&server, RESPONSES_PATH, &document);

    let response = client
        .complete(support::base_request(&selector()))
        .await
        .expect("the raw document request should complete");

    assert_eq!(
        response.raw.as_ref(),
        Some(&document),
        "the raw body must be the provider's own document, unchanged",
    );
    assert_eq!(response.text(), "Sunny.");
    crate::json_snapshot!(response);
}

/// Rate-limit headers, with request resets and token resets kept apart.
///
/// The two reset values use different formats on purpose. Reading either one
/// into the other's field is a real bug that a single-format fixture cannot
/// catch. Both keep the provider's own text; this crate parses neither.
#[tokio::test]
async fn decodes_rate_limit_headers() {
    let (server, client) = wire().await;
    let body = text_document().to_string();
    let _mock = server.mock(move |when, then| {
        when.method(Method::POST).path(RESPONSES_PATH);
        then.status(200)
            .header("content-type", "application/json")
            .header("x-ratelimit-limit-requests", "10000")
            .header("x-ratelimit-remaining-requests", "9999")
            .header("x-ratelimit-reset-requests", "6m0s")
            .header("x-ratelimit-limit-tokens", "2000000")
            .header("x-ratelimit-remaining-tokens", "1998000")
            .header("x-ratelimit-reset-tokens", "1.5s")
            .body(body);
    });

    let response = client
        .complete(support::base_request(&selector()))
        .await
        .expect("the rate limited request should complete");

    let limits = response
        .rate_limits
        .as_ref()
        .expect("the response should carry rate limits");
    assert_eq!(limits.request_reset.as_deref(), Some("6m0s"));
    assert_eq!(limits.token_reset.as_deref(), Some("1.5s"));
    crate::json_snapshot!(limits);
}

// ===========================================================================
// Streaming
// ===========================================================================

/// The reasoning item the streaming transcript closes with.
fn streamed_reasoning_item() -> Value {
    json!({
        "type": "reasoning",
        "id": "rs_stream",
        "summary": [{ "type": "summary_text", "text": "Weighing the options." }],
        "encrypted_content": "gAAAAAstreamed-reasoning-state",
    })
}

/// The message item the streaming transcript closes with.
fn streamed_message_item() -> Value {
    json!({
        "type": "message",
        "id": "msg_stream",
        "status": "completed",
        "role": "assistant",
        "content": [{ "type": "output_text", "text": "Looking that up.", "annotations": [] }],
    })
}

/// The function call item the streaming transcript closes with.
fn streamed_function_call_item() -> Value {
    json!({
        "type": "function_call",
        "id": "fc_stream",
        "call_id": "call_weather",
        "name": "get_weather",
        "arguments": "{\"city\":\"Paris\"}",
        "status": "completed",
    })
}

/// A realistic transcript: reasoning, then text, then a tool call.
///
/// Every frame carries both an `event:` line and a `data:` line, as the real
/// API does, and the frames the codec ignores — `response.in_progress`,
/// `response.reasoning_summary_part.added`, `response.output_text.done` — are
/// present so the test proves they are ignored rather than assuming it.
fn stream_transcript() -> String {
    let frames: Vec<(&str, String)> = vec![
        (
            "response.created",
            json!({
                "type": "response.created",
                "sequence_number": 0,
                "response": { "id": "resp_stream", "object": "response", "status": "in_progress" },
            })
            .to_string(),
        ),
        (
            "response.in_progress",
            json!({
                "type": "response.in_progress",
                "sequence_number": 1,
                "response": { "id": "resp_stream", "status": "in_progress" },
            })
            .to_string(),
        ),
        (
            "response.output_item.added",
            json!({
                "type": "response.output_item.added",
                "sequence_number": 2,
                "output_index": 0,
                "item": { "type": "reasoning", "id": "rs_stream", "summary": [] },
            })
            .to_string(),
        ),
        (
            "response.reasoning_summary_part.added",
            json!({
                "type": "response.reasoning_summary_part.added",
                "sequence_number": 3,
                "item_id": "rs_stream",
                "part": { "type": "summary_text", "text": "" },
            })
            .to_string(),
        ),
        (
            "response.reasoning_summary_text.delta",
            json!({
                "type": "response.reasoning_summary_text.delta",
                "sequence_number": 4,
                "item_id": "rs_stream",
                "output_index": 0,
                "delta": "Weighing ",
            })
            .to_string(),
        ),
        (
            "response.reasoning_summary_text.delta",
            json!({
                "type": "response.reasoning_summary_text.delta",
                "sequence_number": 5,
                "item_id": "rs_stream",
                "output_index": 0,
                "delta": "the options.",
            })
            .to_string(),
        ),
        (
            "response.output_item.done",
            json!({
                "type": "response.output_item.done",
                "sequence_number": 6,
                "output_index": 0,
                "item": streamed_reasoning_item(),
            })
            .to_string(),
        ),
        (
            "response.output_item.added",
            json!({
                "type": "response.output_item.added",
                "sequence_number": 7,
                "output_index": 1,
                "item": {
                    "type": "message",
                    "id": "msg_stream",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                },
            })
            .to_string(),
        ),
        (
            "response.output_text.delta",
            json!({
                "type": "response.output_text.delta",
                "sequence_number": 8,
                "item_id": "msg_stream",
                "output_index": 1,
                "content_index": 0,
                "delta": "Looking ",
            })
            .to_string(),
        ),
        (
            "response.output_text.delta",
            json!({
                "type": "response.output_text.delta",
                "sequence_number": 9,
                "item_id": "msg_stream",
                "output_index": 1,
                "content_index": 0,
                "delta": "that up.",
            })
            .to_string(),
        ),
        (
            "response.output_text.done",
            json!({
                "type": "response.output_text.done",
                "sequence_number": 10,
                "item_id": "msg_stream",
                "output_index": 1,
                "content_index": 0,
                "text": "Looking that up.",
            })
            .to_string(),
        ),
        (
            "response.output_item.done",
            json!({
                "type": "response.output_item.done",
                "sequence_number": 11,
                "output_index": 1,
                "item": streamed_message_item(),
            })
            .to_string(),
        ),
        (
            "response.output_item.added",
            json!({
                "type": "response.output_item.added",
                "sequence_number": 12,
                "output_index": 2,
                "item": {
                    "type": "function_call",
                    "id": "fc_stream",
                    "call_id": "call_weather",
                    "name": "get_weather",
                    "arguments": "",
                    "status": "in_progress",
                },
            })
            .to_string(),
        ),
        (
            "response.function_call_arguments.delta",
            json!({
                "type": "response.function_call_arguments.delta",
                "sequence_number": 13,
                "item_id": "fc_stream",
                "output_index": 2,
                "delta": "{\"city\":",
            })
            .to_string(),
        ),
        (
            "response.function_call_arguments.delta",
            json!({
                "type": "response.function_call_arguments.delta",
                "sequence_number": 14,
                "item_id": "fc_stream",
                "output_index": 2,
                "delta": "\"Paris\"}",
            })
            .to_string(),
        ),
        (
            "response.output_item.done",
            json!({
                "type": "response.output_item.done",
                "sequence_number": 15,
                "output_index": 2,
                "item": streamed_function_call_item(),
            })
            .to_string(),
        ),
        (
            "response.completed",
            json!({
                "type": "response.completed",
                "sequence_number": 16,
                "response": {
                    "id": "resp_stream",
                    "object": "response",
                    "status": "completed",
                    "model": API_MODEL,
                    "output": [
                        streamed_reasoning_item(),
                        streamed_message_item(),
                        streamed_function_call_item(),
                    ],
                    "usage": {
                        "input_tokens": 41,
                        "input_tokens_details": { "cached_tokens": 8 },
                        "output_tokens": 27,
                        "output_tokens_details": { "reasoning_tokens": 19 },
                        "total_tokens": 68,
                    },
                },
            })
            .to_string(),
        ),
    ];

    let frames: Vec<(&str, &str)> = frames
        .iter()
        .map(|(event, data)| (*event, data.as_str()))
        .collect();
    support::sse_transcript(&frames)
}

/// A stream carrying reasoning, text, and a tool call.
///
/// Two intentional differences from the reference implementation are visible in
/// the event snapshot:
///
/// 1. Content-block ids are stable and surfaced. The reference always sent
///    `None` for text and reasoning blocks, so a consumer could not tell two
///    open blocks apart. Ours are the provider's own item ids — `rs_stream`,
///    `msg_stream`, `fc_stream` — which is what makes the block contract in
///    `support::assert_stream_contract` checkable at all.
/// 2. A `tool_call_delta` carries only the raw fragment the provider sent. The
///    reference carried the accumulated argument text on every delta, which
///    made a consumer that concatenated deltas produce quadratic garbage.
///
/// The terminal `completed` response is also this dialect's alone: it comes
/// from the provider's `response.completed` document, so it carries the
/// provider's id, usage, and complete raw body rather than anything this crate
/// assembled.
#[tokio::test]
async fn streams_reasoning_text_and_a_tool_call() {
    let (server, client) = wire().await;
    let (_mock, slot) = support::mount_capture_sse(&server, RESPONSES_PATH, &stream_transcript());

    let stream = client
        .stream(support::base_request(&selector()))
        .await
        .expect("the stream request should be accepted");
    let events = support::collect_stream_events(stream).await;

    support::assert_stream_contract(&events);

    let captured = support::captured(&slot);
    assert_eq!(
        captured.body.get("stream"),
        Some(&json!(true)),
        "a streaming call must ask the provider to stream",
    );

    let completed = events
        .last()
        .expect("the stream should produce events")
        .clone();
    assert_eq!(
        completed.get("type").and_then(Value::as_str),
        Some("completed"),
    );
    let raw = completed
        .pointer("/response/raw")
        .expect("this dialect's completed response must carry the provider's own document");
    assert_eq!(
        raw.get("id").and_then(Value::as_str),
        Some("resp_stream"),
        "the raw document must be the one `response.completed` carried",
    );
    assert_eq!(
        completed.pointer("/response/usage/cache_read"),
        Some(&json!(8)),
        "the terminal usage must come from the provider's own document",
    );

    crate::json_snapshot!(captured);
    crate::json_snapshot!(events);
}

/// A mid-stream `error` event ends the stream with an `Err` item.
///
/// The contract that matters is negative: no `completed` event may follow, so a
/// consumer can never mistake a failed generation for a finished one. The text
/// block that was already open simply stays open.
#[tokio::test]
async fn a_mid_stream_error_terminates_without_completing() {
    let (server, client) = wire().await;
    let transcript = support::sse_transcript(&[
        (
            "response.created",
            r#"{"type":"response.created","response":{"id":"resp_failed","status":"in_progress"}}"#,
        ),
        (
            "response.output_item.added",
            r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_failed","role":"assistant","content":[]}}"#,
        ),
        (
            "response.output_text.delta",
            r#"{"type":"response.output_text.delta","item_id":"msg_failed","delta":"Part"}"#,
        ),
        (
            "error",
            r#"{"type":"error","code":"server_error","message":"the model stream failed","param":null,"sequence_number":4}"#,
        ),
    ]);
    let (_mock, _slot) = support::mount_capture_sse(&server, RESPONSES_PATH, &transcript);

    let stream = client
        .stream(support::base_request(&selector()))
        .await
        .expect("the stream request should be accepted");
    let events = support::collect_stream_events(stream).await;

    support::assert_stream_contract(&events);
    assert!(
        !events
            .iter()
            .any(|event| event.get("type").and_then(Value::as_str) == Some("completed")),
        "a failed stream must not complete",
    );
    assert_eq!(
        events
            .last()
            .and_then(|event| event.get("type"))
            .and_then(Value::as_str),
        Some("error"),
        "the error must be the last item of the stream",
    );

    crate::json_snapshot!(events);
}

/// A stream that stops without `response.completed` still completes, as
/// `incomplete`.
///
/// This dialect normally takes its finish reason from the terminal
/// `response.completed` document. A connection that drops mid-generation never
/// sends one, and the two things a consumer must still get are the partial
/// content it did receive and an honest finish reason. `incomplete` is that
/// reason: the assembler no longer defaults to `stop`, which would have told a
/// caller the model finished its answer when the transport merely stopped.
///
/// Intentional difference from the reference implementation, which emitted no
/// terminal event at all for a truncated Responses stream. A consumer there had
/// to infer truncation from the stream simply ending, which is
/// indistinguishable from a consumer bug.
#[tokio::test]
async fn a_truncated_stream_completes_as_incomplete() {
    let (server, client) = wire().await;
    let transcript = support::sse_transcript(&[
        (
            "response.created",
            r#"{"type":"response.created","response":{"id":"resp_cut","status":"in_progress"}}"#,
        ),
        (
            "response.output_item.added",
            r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_cut","role":"assistant","content":[]}}"#,
        ),
        (
            "response.output_text.delta",
            r#"{"type":"response.output_text.delta","item_id":"msg_cut","delta":"Half a sen"}"#,
        ),
        (
            "response.output_item.done",
            r#"{"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_cut","role":"assistant","content":[{"type":"output_text","text":"Half a sen"}]}}"#,
        ),
    ]);
    let (_mock, _slot) = support::mount_capture_sse(&server, RESPONSES_PATH, &transcript);

    let stream = client
        .stream(support::base_request(&selector()))
        .await
        .expect("the stream request should be accepted");
    let events = support::collect_stream_events(stream).await;

    support::assert_stream_contract(&events);
    let completed = events
        .last()
        .expect("a truncated stream should still complete");
    assert_eq!(
        completed.pointer("/response/finish_reason"),
        Some(&json!("incomplete")),
        "a truncated stream must not report that the model stopped on its own",
    );
    assert_eq!(
        completed.pointer("/response/raw"),
        None,
        "there is no provider document to carry when the stream was cut short",
    );

    crate::json_snapshot!(events);
}

/// A `data:` frame that is not JSON is skipped, not fatal.
///
/// Proxies inject their own keepalive payloads into the stream. Killing a
/// generation that is already half delivered because of a frame carrying no
/// model output would trade a whole response for a comment.
#[tokio::test]
async fn ignores_a_stream_frame_that_is_not_json() {
    let (server, client) = wire().await;
    let transcript = support::sse_transcript(&[
        (
            "response.created",
            r#"{"type":"response.created","response":{"id":"resp_keepalive","status":"in_progress"}}"#,
        ),
        ("ping", "keepalive"),
        (
            "response.output_item.added",
            r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_keepalive","role":"assistant","content":[]}}"#,
        ),
        (
            "response.output_text.delta",
            r#"{"type":"response.output_text.delta","item_id":"msg_keepalive","delta":"Still here."}"#,
        ),
        (
            "response.output_item.done",
            r#"{"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_keepalive","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Still here."}]}}"#,
        ),
        (
            "response.completed",
            r#"{"type":"response.completed","response":{"id":"resp_keepalive","status":"completed","output":[]}}"#,
        ),
    ]);
    let (_mock, _slot) = support::mount_capture_sse(&server, RESPONSES_PATH, &transcript);

    let stream = client
        .stream(support::base_request(&selector()))
        .await
        .expect("the stream request should be accepted");
    let events = support::collect_stream_events(stream).await;

    support::assert_stream_contract(&events);
    let completed = events.last().expect("the stream should produce events");
    assert_eq!(
        completed.get("type").and_then(Value::as_str),
        Some("completed"),
        "an unparseable frame must not end the stream",
    );
    assert_eq!(
        completed.pointer("/response/content/0/text"),
        Some(&json!("Still here.")),
    );
    crate::json_snapshot!(events);
}

/// A stream with no `response.created` still starts.
///
/// The response id rides that event, and a proxy is free not to forward it.
/// Latching `Started` on the first event the provider does send keeps a
/// consumer from seeing deltas for a stream it was never told about; the id is
/// simply absent.
#[tokio::test]
async fn starts_a_stream_that_never_announced_itself() {
    let (server, client) = wire().await;
    let transcript = support::sse_transcript(&[
        (
            "response.output_item.added",
            r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_late","role":"assistant","content":[]}}"#,
        ),
        (
            "response.output_text.delta",
            r#"{"type":"response.output_text.delta","item_id":"msg_late","delta":"No preamble."}"#,
        ),
        (
            "response.output_item.done",
            r#"{"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_late","status":"completed","role":"assistant","content":[{"type":"output_text","text":"No preamble."}]}}"#,
        ),
        (
            "response.completed",
            r#"{"type":"response.completed","response":{"id":"resp_late","status":"completed","output":[]}}"#,
        ),
    ]);
    let (_mock, _slot) = support::mount_capture_sse(&server, RESPONSES_PATH, &transcript);

    let stream = client
        .stream(support::base_request(&selector()))
        .await
        .expect("the stream request should be accepted");
    let events = support::collect_stream_events(stream).await;

    support::assert_stream_contract(&events);
    assert_eq!(
        events.first().and_then(|event| event.get("type")),
        Some(&json!("started")),
        "a stream must start before it delivers anything",
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| event.get("type") == Some(&json!("started")))
            .count(),
        1,
        "a later event must not start the stream a second time",
    );
    crate::json_snapshot!(events);
}

// ===========================================================================
// Classified failures
// ===========================================================================

/// One HTTP failure the provider can return.
struct FailureCase {
    /// The snapshot label, which is what a reader scans the table by.
    name:        &'static str,
    status:      u16,
    retry_after: Option<&'static str>,
    body:        Value,
}

fn failure_cases() -> Vec<FailureCase> {
    vec![
        FailureCase {
            name:        "unauthorized",
            status:      401,
            retry_after: None,
            body:        json!({ "error": {
                "message": "Incorrect API key provided.",
                "type": "invalid_request_error",
                "code": "invalid_api_key",
            }}),
        },
        FailureCase {
            name:        "access_denied",
            status:      403,
            retry_after: None,
            body:        json!({ "error": {
                "message": "Country, region, or territory not supported.",
                "type": "request_forbidden",
                "code": "unsupported_country_region_territory",
            }}),
        },
        FailureCase {
            name:        "model_not_found",
            status:      404,
            retry_after: None,
            body:        json!({ "error": {
                "message": "The model `gpt-nonexistent` does not exist.",
                "type": "invalid_request_error",
                "code": "model_not_found",
            }}),
        },
        FailureCase {
            name:        "rate_limited_with_retry_after",
            status:      429,
            retry_after: Some("30"),
            body:        json!({ "error": {
                "message": "Rate limit reached for requests.",
                "type": "requests",
                "code": "rate_limit_exceeded",
            }}),
        },
        FailureCase {
            name:        "insufficient_quota",
            status:      429,
            retry_after: None,
            body:        json!({ "error": {
                "message": "You exceeded your current quota, please check your plan.",
                "type": "insufficient_quota",
                "code": "insufficient_quota",
            }}),
        },
        FailureCase {
            name:        "server_error",
            status:      500,
            retry_after: None,
            body:        json!({ "error": {
                "message": "The server had an error processing your request.",
                "type": "server_error",
                "code": null,
            }}),
        },
    ]
}

/// The classification of every HTTP failure family, as one table.
///
/// The pair worth reading closely is `rate_limited_with_retry_after` against
/// `insufficient_quota`. Both arrive as HTTP 429 and both say "limit" in
/// English, but only the first clears with backoff. The second is spent credit,
/// so it must classify as `quota_exceeded` and must never be retried.
#[tokio::test]
async fn classifies_http_failures() {
    let mut classified = Vec::new();
    for case in failure_cases() {
        let (server, client) = wire().await;
        let body = case.body.to_string();
        let status = case.status;
        let retry_after = case.retry_after;
        let _mock = server.mock(move |when, then| {
            when.method(Method::POST).path(RESPONSES_PATH);
            let then = then
                .status(status)
                .header("content-type", "application/json")
                .body(body);
            if let Some(value) = retry_after {
                then.header("retry-after", value);
            }
        });

        let error = client
            .complete(support::base_request(&selector()))
            .await
            .expect_err("the provider returned a failure status");

        if case.name == "rate_limited_with_retry_after" {
            assert_eq!(error.kind(), ErrorKind::RateLimit);
            assert_eq!(
                error.data().retry,
                RetryClassification::After { millis: 30_000 },
                "a `Retry-After` on throttling must carry into the classification",
            );
        }
        if case.name == "insufficient_quota" {
            assert_eq!(
                error.kind(),
                ErrorKind::QuotaExceeded,
                "spent quota is not throttling, even at HTTP 429",
            );
            assert_eq!(
                error.data().retry,
                RetryClassification::Never,
                "backoff never clears spent quota",
            );
        }

        classified.push(json!({ "case": case.name, "error": error.data() }));
    }

    crate::json_snapshot!(classified);
}

// ===========================================================================
// Native input token counting
// ===========================================================================

/// The count endpoint takes the generation body projected onto an allowlist.
///
/// The corpus request carries `temperature`, `top_p`, `metadata`, and
/// `max_output_tokens`, and the generation body would also carry `stream`,
/// `store`, and `include`. None of them are accepted by this endpoint, and the
/// projection runs after raw provider options merge, so a stray merged key is
/// dropped too. What survives is the input itself and the fields that change
/// how it is counted.
#[tokio::test]
async fn pins_the_count_tokens_body() {
    let (server, client) = wire().await;
    let (_mock, slot) = support::mount_capture(
        &server,
        COUNT_PATH,
        &json!({ "object": "response.input_tokens", "input_tokens": 42 }),
    );

    let count = client
        .count_input_tokens(support::sampling_request(&selector()))
        .await
        .expect("the token count request should succeed")
        .expect("this dialect has a native token count endpoint");

    let captured = support::captured(&slot);
    let body = captured
        .body
        .as_object()
        .expect("the count request body should be a JSON object");
    for dropped in [
        "temperature",
        "max_output_tokens",
        "top_p",
        "stop",
        "metadata",
        "stream",
        "store",
        "include",
    ] {
        assert!(
            !body.contains_key(dropped),
            "the count endpoint does not accept `{dropped}`",
        );
    }
    assert_eq!(count.tokens(), 42);

    crate::json_snapshot!(captured);
    crate::json_snapshot!(json!({
        "tokens": count.tokens(),
        "model":  count.model(),
    }));
}

/// A count body with the wrong `object` discriminator is a decode failure.
///
/// The count is a number, so nothing in its value can flag a body that came
/// from the wrong route or the wrong API version. The discriminator is the only
/// check available, and dropping it would let a plain response document be read
/// as a token count.
#[tokio::test]
async fn rejects_a_count_response_with_the_wrong_object() {
    let (server, client) = wire().await;
    let (_mock, _slot) = support::mount_capture(
        &server,
        COUNT_PATH,
        &json!({ "object": "response", "input_tokens": 42 }),
    );

    let error = client
        .count_input_tokens(support::base_request(&selector()))
        .await
        .expect_err("a wrong discriminator must not be read as a count");

    assert_eq!(error.kind(), ErrorKind::ResponseDecode);
    crate::json_snapshot!(error.data());
}
