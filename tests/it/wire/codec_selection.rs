//! Per-call codec selection on the built-in `http` adapter.
//!
//! A provider lists the codecs its host speaks and each row reaches a subset
//! of them; the client picks one per call by the operation's family, in
//! provider order. These tests pin where each call lands: the path the mock
//! server saw is the codec that served it. No snapshot is taken, because the
//! bodies are the ones the per-dialect files already pin; what this file
//! pins is the routing.

use httpmock::{Method, MockServer};
use lithos_llm::catalog::codec_ids;
use lithos_llm::types::ErrorKind;
use lithos_llm::{Evaluation, Request};
use serde_json::{Value, json};

use crate::support;

const CHAT_PATH: &str = "/v1/chat/completions";
const MESSAGES_PATH: &str = "/v1/messages";
const COUNT_PATH: &str = "/v1/messages/count_tokens";
const RESPONSES_PATH: &str = "/v1/responses";
const EVALUATION_PATH: &str = "/v4/ai/evaluation-model";

/// A provider that speaks Chat Completions first and Anthropic Messages
/// second, with one row on both and one narrowed to Messages.
fn two_generation_codecs(base_url: &str) -> String {
    format!(
        r#"
        schema_version = 1

        [providers.multi]
        display_name = "Multi"
        codecs = ["{chat}", "{messages}"]
        base_url = "{base_url}"
        default_model = "both"
        auth = {{ type = "bearer" }}

        [providers.multi.models.both]
        display_name = "Both"
        api_model = "vendor/both"
        capabilities = {{ text = true }}

        [providers.multi.models.narrowed]
        display_name = "Narrowed"
        api_model = "vendor/narrowed"
        codecs = ["{messages}"]
        capabilities = {{ text = true }}
        "#,
        chat = codec_ids::OPENAI_CHAT,
        messages = codec_ids::ANTHROPIC_MESSAGES,
    )
}

/// A provider shaped like the built-in `vercel` one: Chat for the language
/// models and the gateway's evaluation protocol for `jev`. `sonnet` is a
/// structured-output row that judges; `jev` is the one native evaluator.
fn mixed_provider(base_url: &str) -> String {
    format!(
        r#"
        schema_version = 1

        [providers.vercel]
        display_name = "Vercel"
        codecs = ["{chat}", "{evaluation}"]
        base_url = "{base_url}"
        allow_passthrough = true
        default_model = "sonnet"
        auth = {{ type = "bearer" }}

        [providers.vercel.models.sonnet]
        display_name = "Sonnet"
        api_model = "anthropic/claude-sonnet-5"
        capabilities = {{ text = true, response_format = {{ json_schema = true }} }}

        [providers.vercel.models.jev]
        display_name = "Jev"
        api_model = "typesafe-ai/jev"
        capabilities = {{ evaluation = {{ choice = true, score = true, boolean = true }} }}
        "#,
        chat = codec_ids::OPENAI_CHAT,
        evaluation = codec_ids::VERCEL_EVALUATION,
    )
}

/// A Responses-only provider whose row claims structured output and so
/// judges; its provider lists no evaluation codec at all.
fn responses_provider(base_url: &str) -> String {
    format!(
        r#"
        schema_version = 1

        [providers.oai]
        display_name = "OpenAI"
        codecs = ["{responses}"]
        base_url = "{base_url}"
        default_model = "gpt"
        auth = {{ type = "bearer" }}

        [providers.oai.models.gpt]
        display_name = "GPT"
        api_model = "gpt-5"
        capabilities = {{ text = true, response_format = {{ json_schema = true }} }}
        "#,
        responses = codec_ids::OPENAI_RESPONSES,
    )
}

/// A passthrough-friendly provider on Anthropic Messages alone, which has a
/// count endpoint, so a passthrough row can complete, stream, and count.
fn messages_provider(base_url: &str) -> String {
    format!(
        r#"
        schema_version = 1

        [providers.anthropic]
        display_name = "Anthropic"
        codecs = ["{messages}"]
        base_url = "{base_url}"
        allow_passthrough = true
        auth = {{ type = "bearer" }}
        "#,
        messages = codec_ids::ANTHROPIC_MESSAGES,
    )
}

fn client(toml: &str, provider: &str) -> lithos_llm::Client {
    support::client_for(
        support::catalog_from_toml("wire-codecs", toml),
        provider,
        support::bearer_credentials(),
    )
}

fn chat_reply(text: &str) -> Value {
    json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "vendor/model",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": text },
            "finish_reason": "stop",
        }],
        "usage": { "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12 },
    })
}

fn messages_reply(text: &str) -> Value {
    json!({
        "id": "msg_01Wire",
        "type": "message",
        "role": "assistant",
        "model": "vendor/model",
        "content": [{ "type": "text", "text": text }],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": { "input_tokens": 11, "output_tokens": 5 },
    })
}

fn responses_reply(text: &str) -> Value {
    json!({
        "id": "resp_1",
        "object": "response",
        "created_at": 1_772_000_000_u64,
        "status": "completed",
        "model": "gpt-5",
        "output": [{
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [{ "type": "output_text", "text": text, "annotations": [] }],
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

/// One boolean question, which a judge answers under the key `q0`.
fn one_boolean(model: &str) -> Evaluation {
    Evaluation::builder()
        .model(model)
        .state("I was charged twice.")
        .boolean("requests_refund", "Refund requested?")
        .build()
        .expect("the evaluation should build")
}

/// Mounts a mock that answers any POST, to prove a path was never reached.
fn mount_never<'server>(server: &'server MockServer, path: &str) -> httpmock::Mock<'server> {
    let path = path.to_owned();
    server.mock(move |when, then| {
        when.method(Method::POST).path(path);
        then.status(200).json_body(json!({}));
    })
}

#[tokio::test]
async fn a_row_on_two_generation_codecs_completes_through_the_first() {
    let server = MockServer::start_async().await;
    let client = client(&two_generation_codecs(&server.base_url()), "multi");
    let (chat, slot) = support::mount_capture(&server, CHAT_PATH, &chat_reply("Hello back."));
    let messages = mount_never(&server, MESSAGES_PATH);

    let response = client
        .complete(support::base_request("multi/both"))
        .await
        .expect("the completion should succeed");

    chat.assert_async().await;
    messages.assert_calls_async(0).await;
    assert_eq!(support::captured(&slot).path, CHAT_PATH);
    assert_eq!(response.text(), "Hello back.");
}

#[tokio::test]
async fn a_row_narrowed_to_the_second_codec_completes_through_it() {
    let server = MockServer::start_async().await;
    let client = client(&two_generation_codecs(&server.base_url()), "multi");
    let chat = mount_never(&server, CHAT_PATH);
    let (messages, slot) =
        support::mount_capture(&server, MESSAGES_PATH, &messages_reply("Hello back."));

    let response = client
        .complete(support::base_request("multi/narrowed"))
        .await
        .expect("the completion should succeed");

    messages.assert_async().await;
    chat.assert_calls_async(0).await;
    assert_eq!(support::captured(&slot).path, MESSAGES_PATH);
    assert_eq!(response.text(), "Hello back.");
}

#[tokio::test]
async fn a_structured_output_row_on_a_mixed_provider_judges_through_chat() {
    let server = MockServer::start_async().await;
    let client = client(&mixed_provider(&server.base_url()), "vercel");
    let (chat, slot) = support::mount_capture(&server, CHAT_PATH, &chat_reply(r#"{"q0":0.9}"#));
    let evaluation_endpoint = mount_never(&server, EVALUATION_PATH);

    let verdict = client
        .evaluate(one_boolean("vercel/sonnet"))
        .await
        .expect("the judge should answer");

    chat.assert_async().await;
    evaluation_endpoint.assert_calls_async(0).await;
    let captured = support::captured(&slot);
    assert_eq!(captured.path, CHAT_PATH);
    assert!(
        captured.body.get("response_format").is_some(),
        "the judge asks for a schema-bound object: {}",
        captured.body
    );
    assert!(
        verdict
            .boolean("requests_refund")
            .expect("the verdict answers the question")
            .is_likely()
    );
}

#[tokio::test]
async fn a_passthrough_evaluation_on_a_mixed_provider_judges_through_chat() {
    let server = MockServer::start_async().await;
    let client = client(&mixed_provider(&server.base_url()), "vercel");
    let (chat, _slot) = support::mount_capture(&server, CHAT_PATH, &chat_reply(r#"{"q0":0.2}"#));
    let evaluation_endpoint = mount_never(&server, EVALUATION_PATH);

    let verdict = client
        .evaluate(one_boolean("vercel/not-in-catalog"))
        .await
        .expect("a passthrough row judges");

    chat.assert_async().await;
    evaluation_endpoint.assert_calls_async(0).await;
    assert!(
        !verdict
            .boolean("requests_refund")
            .expect("the verdict answers the question")
            .is_likely()
    );
}

#[tokio::test]
async fn a_structured_output_row_with_no_evaluation_codec_judges_through_responses() {
    let server = MockServer::start_async().await;
    let client = client(&responses_provider(&server.base_url()), "oai");
    let (responses, slot) =
        support::mount_capture(&server, RESPONSES_PATH, &responses_reply(r#"{"q0":0.75}"#));

    let verdict = client
        .evaluate(one_boolean("oai/gpt"))
        .await
        .expect("the judge should answer");

    responses.assert_async().await;
    assert_eq!(support::captured(&slot).path, RESPONSES_PATH);
    assert!(
        (verdict
            .boolean("requests_refund")
            .expect("the verdict answers the question")
            .probability
            - 0.75)
            .abs()
            < f64::EPSILON
    );
}

#[tokio::test]
async fn a_passthrough_row_on_a_generation_provider_completes_streams_and_counts() {
    let server = MockServer::start_async().await;
    let client = client(&messages_provider(&server.base_url()), "anthropic");
    let transcript = support::sse_transcript(&[
        (
            "message_start",
            r#"{"type":"message_start","message":{"id":"msg_01Stream","type":"message","role":"assistant","model":"vendor/model","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":4,"output_tokens":1}}}"#,
        ),
        (
            "content_block_start",
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        ),
        (
            "content_block_delta",
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Streamed."}}"#,
        ),
        (
            "content_block_stop",
            r#"{"type":"content_block_stop","index":0}"#,
        ),
        (
            "message_delta",
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":3}}"#,
        ),
        ("message_stop", r#"{"type":"message_stop"}"#),
    ]);
    let (count, _) = support::mount_capture(&server, COUNT_PATH, &json!({ "input_tokens": 7 }));
    // Both completion shapes share one path; the body's `stream` flag tells
    // them apart.
    let (blocking, _) = support::mount_capture(&server, MESSAGES_PATH, &messages_reply("Done."));

    let response = client
        .complete(support::base_request("anthropic/not-in-catalog"))
        .await
        .expect("a passthrough completion goes out on the provider's codec");
    assert_eq!(response.text(), "Done.");
    assert_eq!(response.model.model().as_str(), "not-in-catalog");
    blocking.assert_async().await;

    let tokens = client
        .count_input_tokens(support::base_request("anthropic/not-in-catalog"))
        .await
        .expect("the count should succeed")
        .expect("Messages has a count endpoint");
    assert_eq!(tokens.tokens(), 7);
    count.assert_async().await;

    blocking.delete_async().await;
    let (streaming, _) = support::mount_capture_sse(&server, MESSAGES_PATH, &transcript);
    let stream = client
        .stream(support::base_request("anthropic/not-in-catalog"))
        .await
        .expect("a passthrough stream opens on the provider's codec");
    let events = support::collect_stream_events(stream).await;
    support::assert_stream_contract(&events);
    streaming.assert_async().await;
    assert!(
        events
            .iter()
            .any(|event| event["type"] == "text_delta" && event["text"] == "Streamed."),
        "{events:?}"
    );
}

#[tokio::test]
async fn a_passthrough_generation_call_on_an_evaluation_only_provider_is_refused() {
    let server = MockServer::start_async().await;
    let toml = format!(
        r#"
        schema_version = 1

        [providers.typesafe]
        display_name = "TypeSafe"
        codecs = ["{evaluation}"]
        base_url = "{base_url}"
        allow_passthrough = true
        auth = {{ type = "bearer" }}
        "#,
        evaluation = codec_ids::VERCEL_EVALUATION,
        base_url = server.base_url(),
    );
    let client = client(&toml, "typesafe");
    let any = server
        .mock_async(|when, then| {
            when.method(Method::POST);
            then.status(200).json_body(json!({}));
        })
        .await;

    let request = Request::builder()
        .model("typesafe/jev-latest")
        .user("Hello")
        .build()
        .expect("the request should build");
    let error = client
        .complete(request)
        .await
        .expect_err("no generation codec reaches a passthrough row here");

    any.assert_calls_async(0).await;
    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    assert!(
        error
            .message()
            .contains("no generation codec reaches model typesafe/jev-latest"),
        "{}",
        error.message()
    );
}
