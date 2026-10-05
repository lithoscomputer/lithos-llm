use std::error::Error as StdError;

use serde_json::{Map, Value, json, to_string};

use super::{Codec, OPAQUE_PREFIX, OpenAiChatCodec, REASONING_DETAILS, REASONING_DETAILS_KIND};
use crate::codecs::test_support::{resolved, resolved_in};
use crate::resolver::ResolvedRoute;
use crate::transport::SseEvent;
use crate::types::{
    CacheHint, ContentBlockKind, ContentPart, CostSource, Error, ErrorKind, FinishReason, Message,
    ReasoningContent, ReasoningEffort, Request, Response, RetryClassification, Role, Speed,
    StreamEvent, ToolCall, ToolDefinition, ToolResult,
};

const MODEL: &str = "openai/gpt-5.6-luna";

/// The catalog provider namespace `MODEL` resolves to.
const NAMESPACE: &str = "openai";

fn route() -> Result<ResolvedRoute, Box<dyn StdError>> {
    let request = Request::builder().model(MODEL).user("Hello").build()?;
    Ok(resolved(request)?.route().clone())
}

/// Decodes one complete body against the `MODEL` route.
fn decode(body: Value) -> Result<Response, Box<dyn StdError>> {
    Ok(OpenAiChatCodec::default().decode_response(&route()?, body)?)
}

fn object(value: Value) -> Result<Map<String, Value>, Box<dyn StdError>> {
    match value {
        Value::Object(map) => Ok(map),
        other => Err(format!("expected a JSON object, got {other}").into()),
    }
}

/// Feeds chunks through one stream decoder and finishes the stream.
fn stream(chunks: Vec<Value>) -> Result<Vec<StreamEvent>, Box<dyn StdError>> {
    let mut decoder = OpenAiChatCodec::default().stream_decoder(&route()?);
    let mut events = Vec::new();
    for chunk in chunks {
        events.extend(decoder.decode(SseEvent {
            event: None,
            data:  chunk.to_string(),
        })?);
    }
    events.extend(decoder.finish()?);
    Ok(events)
}

fn completed(events: &[StreamEvent]) -> Vec<&Response> {
    events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::Ended { response } => Some(response.as_ref()),
            _ => None,
        })
        .collect()
}

fn tool_parts(response: &Response) -> Vec<&ToolCall> {
    response
        .content
        .iter()
        .filter_map(|part| match part {
            ContentPart::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect()
}

#[test]
fn the_details_kind_is_the_prefix_and_the_field() {
    // The constant replaces a runtime `format!`; this keeps it honest against
    // the two parts `encode_chat_message` splits it back into.
    assert_eq!(
        REASONING_DETAILS_KIND,
        format!("{OPAQUE_PREFIX}{REASONING_DETAILS}")
    );
}

#[test]
fn encodes_chat_tool_calls_and_results() -> Result<(), Box<dyn StdError>> {
    let call = resolved(
        Request::builder()
            .model(MODEL)
            .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
                ToolCall::function("call-1", "weather", json!({ "city": "Boston" })),
            )]))
            .message(Message::new(Role::Tool, [ContentPart::ToolResult(
                ToolResult {
                    tool_call_id: "call-1".to_owned(),
                    name:         Some("weather".to_owned()),
                    content:      vec![ContentPart::Text {
                        text: "snow".to_owned(),
                    }],
                    is_error:     false,
                },
            )]))
            .build()?,
    )?;

    let encoded = OpenAiChatCodec::default().encode(&call, false)?;

    assert_eq!(encoded.body["messages"][0]["role"], "assistant");
    assert_eq!(encoded.body["messages"][0]["tool_calls"][0]["id"], "call-1");
    assert_eq!(
        encoded.body["messages"][0]["tool_calls"][0]["function"]["arguments"],
        json!(r#"{"city":"Boston"}"#)
    );
    assert_eq!(encoded.body["messages"][1]["role"], "tool");
    assert_eq!(encoded.body["messages"][1]["tool_call_id"], "call-1");
    assert_eq!(encoded.body["messages"][1]["content"], "snow");
    Ok(())
}

#[test]
fn inclusive_usage_becomes_disjoint_buckets() -> Result<(), Box<dyn StdError>> {
    let response = decode(json!({
        "choices": [{ "message": { "content": "ok" }, "finish_reason": "stop" }],
        "usage": {
            "prompt_tokens": 200,
            "completion_tokens": 10,
            "prompt_tokens_details": { "cached_tokens": 50, "cache_write_tokens": 100 },
            "completion_tokens_details": { "reasoning_tokens": 4 },
        },
    }))?;

    assert_eq!(response.usage.input, 50);
    assert_eq!(response.usage.cache_read, 50);
    assert_eq!(response.usage.cache_write, 100);
    assert_eq!(response.usage.output, 6);
    assert_eq!(response.usage.reasoning, 4);
    assert_eq!(response.usage.total(), 210);
    Ok(())
}

#[test]
fn flat_usage_spellings_decode_the_same_way() -> Result<(), Box<dyn StdError>> {
    let response = decode(json!({
        "choices": [{ "message": { "content": "ok" } }],
        "usage": {
            "prompt_tokens": 53,
            "completion_tokens": 66,
            "prompt_cache_hit_tokens": 41,
            "reasoning_tokens": 54,
        },
    }))?;

    assert_eq!(response.usage.input, 12);
    assert_eq!(response.usage.cache_read, 41);
    assert_eq!(response.usage.output, 12);
    assert_eq!(response.usage.reasoning, 54);
    Ok(())
}

#[test]
fn anthropic_style_cache_writes_decode_from_either_position() -> Result<(), Box<dyn StdError>> {
    // Venice fronting a Claude model reports cache writes as
    // `cache_creation_input_tokens`, both nested and flat, and never as
    // `cache_write_tokens`. Live calls on 2026-08-29 sent exactly this
    // shape.
    let response = decode(json!({
        "choices": [{ "message": { "content": "ok" } }],
        "usage": {
            "prompt_tokens": 15002,
            "completion_tokens": 12,
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "cache_creation_input_tokens": 13204,
            },
            "cache_creation_input_tokens": 13204,
        },
    }))?;

    assert_eq!(response.usage.cache_write, 13204);
    assert_eq!(response.usage.input, 1798);
    assert_eq!(response.usage.total(), 15014);
    Ok(())
}

#[test]
fn a_nested_detail_wins_over_its_flat_spelling() -> Result<(), Box<dyn StdError>> {
    let response = decode(json!({
        "choices": [{ "message": { "content": "ok" } }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_tokens_details": { "cached_tokens": 30 },
            "prompt_cache_hit_tokens": 90,
            "completion_tokens_details": { "reasoning_tokens": 5 },
            "reasoning_tokens": 19,
        },
    }))?;

    assert_eq!(response.usage.cache_read, 30);
    assert_eq!(response.usage.reasoning, 5);
    Ok(())
}

#[test]
fn the_speed_control_is_reported_and_not_billed() -> Result<(), Box<dyn StdError>> {
    let call = resolved(
        Request::builder()
            .model(MODEL)
            .user("Hello")
            .speed(Speed::Fast)
            .build()?,
    )?;

    let encoded = OpenAiChatCodec::default().encode(&call, false)?;

    let messages: Vec<&str> = encoded
        .warnings
        .iter()
        .map(|warning| warning.message.as_str())
        .collect();
    assert_eq!(messages, [
        "this provider protocol does not support the speed control",
    ]);
    // Nothing speed-shaped may be guessed at on the wire, and cost must
    // not price a fast tier the provider never served.
    let body = encoded.body.to_string();
    assert!(!body.contains("speed"), "{body}");
    assert!(!body.contains("service_tier"), "{body}");
    assert_eq!(encoded.applied_speed, None);
    Ok(())
}

#[test]
fn openrouter_usage_cost_is_provider_reported() -> Result<(), Box<dyn StdError>> {
    let response = decode(json!({
        "choices": [{ "message": { "content": "ok" } }],
        "usage": { "prompt_tokens": 200, "completion_tokens": 10, "cost": 0.0042 },
    }))?;

    let cost = response.cost.ok_or("expected a provider cost")?;
    assert_eq!(cost.usd_micros, 4200);
    assert_eq!(cost.source, CostSource::Provider);
    Ok(())
}

#[test]
fn venice_top_level_cost_is_provider_reported() -> Result<(), Box<dyn StdError>> {
    let response = decode(json!({
        "choices": [{ "message": { "content": "ok" } }],
        "usage": { "prompt_tokens": 4, "completion_tokens": 2 },
        "cost": { "usd": 0.25, "diem": 9.5 },
    }))?;

    let cost = response.cost.ok_or("expected a provider cost")?;
    assert_eq!(cost.usd_micros, 250_000);
    assert_eq!(cost.source, CostSource::Provider);
    Ok(())
}

#[test]
fn usage_cost_wins_over_the_top_level_cost() -> Result<(), Box<dyn StdError>> {
    let response = decode(json!({
        "choices": [{ "message": { "content": "ok" } }],
        "usage": { "prompt_tokens": 4, "completion_tokens": 2, "cost": 0.001 },
        "cost": { "usd": 0.25 },
    }))?;

    let cost = response.cost.ok_or("expected a provider cost")?;
    assert_eq!(cost.usd_micros, 1000);
    Ok(())
}

#[test]
fn upstream_inference_cost_is_not_parsed_but_survives_in_raw() -> Result<(), Box<dyn StdError>> {
    let response = decode(json!({
        "provider": "Fireworks",
        "choices": [{ "message": { "content": "ok" }, "native_finish_reason": "eos" }],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 2,
            "cost_details": { "upstream_inference_cost": 0.5 },
            "prompt_tokens_details": { "audio_tokens": 3 },
        },
    }))?;

    assert!(response.cost.is_none());
    let raw = response.raw.ok_or("expected the raw body")?;
    assert_eq!(raw["usage"]["cost_details"]["upstream_inference_cost"], 0.5);
    assert_eq!(raw["usage"]["prompt_tokens_details"]["audio_tokens"], 3);
    assert_eq!(raw["provider"], "Fireworks");
    assert_eq!(raw["choices"][0]["native_finish_reason"], "eos");
    Ok(())
}

#[test]
fn decodes_reasoning_and_tool_calls() -> Result<(), Box<dyn StdError>> {
    let response = decode(json!({
        "id": "chatcmpl-1",
        "choices": [{
            "message": {
                "content": "done",
                "reasoning_content": "thought about it",
                "tool_calls": [{
                    "id": "call-1",
                    "function": { "name": "weather", "arguments": "{\"city\":\"Boston\"}" },
                }],
            },
            "finish_reason": "tool_calls",
        }],
    }))?;

    assert_eq!(response.id.as_deref(), Some("chatcmpl-1"));
    let reasoning = response.content.first().ok_or("expected reasoning")?;
    assert!(matches!(reasoning, ContentPart::Reasoning(_)));
    let call = tool_parts(&response)
        .first()
        .copied()
        .ok_or("expected a tool call")?
        .clone();
    assert_eq!(call.input.wire_value(), json!({ "city": "Boston" }));
    assert_eq!(Some(call.input.raw()), Some(r#"{"city":"Boston"}"#));
    Ok(())
}

#[test]
fn a_complete_tool_call_wins_over_a_stop_finish_reason() -> Result<(), Box<dyn StdError>> {
    // The complete-path twin of
    // `a_streamed_tool_call_wins_over_a_stop_finish_reason`: qwen on
    // Venice answers a forced tool call with `finish_reason: "stop"`,
    // and both paths must decode that exchange the same way.
    let response = decode(json!({
        "choices": [{
            "message": {
                "content": "",
                "tool_calls": [{
                    "id": "call-1",
                    "function": { "name": "weather", "arguments": "{\"city\":\"Boston\"}" },
                }],
            },
            "finish_reason": "stop",
        }],
    }))?;

    assert_eq!(response.finish_reason, FinishReason::ToolCall);
    Ok(())
}

#[test]
fn raw_provider_options_win_and_controls_never_reach_the_wire() -> Result<(), Box<dyn StdError>> {
    let options = object(json!({
        "temperature": 0.9,
        "seed": 7,
        "auto_cache": false,
    }))?;
    let call = resolved(
        Request::builder()
            .model(MODEL)
            .user("Hello")
            .temperature(0.2)
            .provider_options(NAMESPACE, options)
            .provider_option("anthropic", "top_k", json!(40))
            .build()?,
    )?;

    let encoded = OpenAiChatCodec::default().encode(&call, false)?;

    assert_eq!(encoded.body["temperature"], json!(0.9));
    assert_eq!(encoded.body["seed"], json!(7));
    assert_eq!(encoded.body.get("auto_cache"), None);
    assert_eq!(encoded.body.get("top_k"), None);
    Ok(())
}

#[test]
fn stop_sequences_encode_in_order_and_metadata_is_reported() -> Result<(), Box<dyn StdError>> {
    let call = resolved(
        Request::builder()
            .model(MODEL)
            .user("Hello")
            .stop_sequences(["END", "STOP"])
            .metadata_entry("trace_id", "t789")
            .build()?,
    )?;

    let encoded = OpenAiChatCodec::default().encode(&call, false)?;

    assert_eq!(encoded.body["stop"], json!(["END", "STOP"]));
    // Only OpenAI itself takes `metadata` here, so the tags are dropped
    // with a warning rather than risking a strict skin's rejection.
    assert_eq!(encoded.body.get("metadata"), None);
    let warnings: Vec<&str> = encoded
        .warnings
        .iter()
        .map(|warning| warning.code.as_str())
        .collect();
    assert_eq!(warnings, ["unsupported_control"]);
    Ok(())
}

#[test]
fn reasoning_effort_reaches_the_wire_untranslated() -> Result<(), Box<dyn StdError>> {
    let call = resolved(
        Request::builder()
            .model(MODEL)
            .user("Hello")
            .reasoning_effort(ReasoningEffort::Xhigh)
            .build()?,
    )?;

    let encoded = OpenAiChatCodec::default().encode(&call, false)?;

    assert_eq!(encoded.body["reasoning_effort"], json!("xhigh"));
    Ok(())
}

#[test]
fn a_developer_message_is_sent_as_a_system_message() -> Result<(), Box<dyn StdError>> {
    let call = resolved(
        Request::builder()
            .model(MODEL)
            .message(Message::text(Role::Developer, "Keep it short."))
            .user("Hello")
            .build()?,
    )?;

    let encoded = OpenAiChatCodec::default().encode(&call, false)?;

    assert_eq!(encoded.body["messages"][0]["role"], "system");
    Ok(())
}

#[test]
fn a_json_only_tool_result_sends_the_bare_value() -> Result<(), Box<dyn StdError>> {
    let call = resolved(
        Request::builder()
            .model(MODEL)
            .message(Message::new(Role::Tool, [ContentPart::ToolResult(
                ToolResult {
                    tool_call_id: "call-1".to_owned(),
                    name:         Some("weather".to_owned()),
                    content:      vec![ContentPart::Json {
                        value: json!({ "city": "Boston", "temp_c": 4 }),
                    }],
                    is_error:     false,
                },
            )]))
            .build()?,
    )?;

    let encoded = OpenAiChatCodec::default().encode(&call, false)?;

    assert_eq!(
        encoded.body["messages"][0]["content"],
        json!(r#"{"city":"Boston","temp_c":4}"#),
        "the value itself, not the ContentPart envelope"
    );
    Ok(())
}

#[test]
fn a_json_string_tool_result_sends_the_raw_text() -> Result<(), Box<dyn StdError>> {
    // A JSON part whose value is a bare string means that text. The
    // reference encoder sent it unquoted; serializing the value would
    // hand the model the quoted JSON literal `"72F and sunny"` instead.
    let call = resolved(
        Request::builder()
            .model(MODEL)
            .message(Message::new(Role::Tool, [ContentPart::ToolResult(
                ToolResult {
                    tool_call_id: "call-1".to_owned(),
                    name:         Some("weather".to_owned()),
                    content:      vec![ContentPart::Json {
                        value: json!("72F and sunny"),
                    }],
                    is_error:     false,
                },
            )]))
            .build()?,
    )?;

    let encoded = OpenAiChatCodec::default().encode(&call, false)?;

    assert_eq!(
        encoded.body["messages"][0]["content"],
        json!("72F and sunny")
    );
    Ok(())
}

#[test]
fn an_empty_text_tool_result_sends_an_empty_string() -> Result<(), Box<dyn StdError>> {
    // A command with no stdout answers with nothing. Serializing the
    // ContentPart envelope instead would hand the model spurious JSON as
    // the tool's answer.
    let call = resolved(
        Request::builder()
            .model(MODEL)
            .message(Message::new(Role::Tool, [ContentPart::ToolResult(
                ToolResult {
                    tool_call_id: "call-1".to_owned(),
                    name:         Some("shell".to_owned()),
                    content:      vec![ContentPart::Text {
                        text: String::new(),
                    }],
                    is_error:     false,
                },
            )]))
            .build()?,
    )?;

    let encoded = OpenAiChatCodec::default().encode(&call, false)?;

    assert_eq!(encoded.body["messages"][0]["content"], json!(""));
    Ok(())
}

#[test]
fn several_text_parts_join_into_the_plain_string_form() -> Result<(), Box<dyn StdError>> {
    // A strict text-only skin accepts only string content; the part-array
    // form is reserved for messages carrying media. Two text parts join
    // unseparated, as the reference client sent them.
    let call = resolved(
        Request::builder()
            .model(MODEL)
            .message(Message::new(Role::User, [
                ContentPart::Text {
                    text: "First paragraph. ".to_owned(),
                },
                ContentPart::Text {
                    text: "Second paragraph.".to_owned(),
                },
            ]))
            .build()?,
    )?;

    let encoded = OpenAiChatCodec::default().encode(&call, false)?;

    assert_eq!(
        encoded.body["messages"][0]["content"],
        json!("First paragraph. Second paragraph.")
    );
    Ok(())
}

#[test]
fn noncanonical_reasoning_details_are_ignored() -> Result<(), Box<dyn StdError>> {
    let details = json!([{ "type": "reasoning.encrypted", "id": "rs-1", "data": "AQ==" }]);
    let call = resolved(
        Request::builder()
            .model(MODEL)
            .user("hi")
            .message(Message::new(Role::Assistant, [
                ContentPart::opaque("openai_compat_reasoning_details", details.clone()),
                ContentPart::Text {
                    text: "Looking it up.".to_owned(),
                },
            ]))
            .user("thanks")
            .build()?,
    )?;

    let encoded = OpenAiChatCodec::default().encode(&call, false)?;

    assert!(
        encoded.body["messages"][1]
            .get("reasoning_details")
            .is_none()
    );
    Ok(())
}

#[test]
fn several_reasoning_parts_replay_concatenated() -> Result<(), Box<dyn StdError>> {
    // A multi-block reasoning history — an Anthropic conversation failing
    // over to a Chat route — replays as one `reasoning_content` string
    // joined unseparated, byte for byte what the reference client sent;
    // the same rule the text join follows.
    let call = resolved(
        Request::builder()
            .model(MODEL)
            .user("hi")
            .message(Message::new(Role::Assistant, [
                ContentPart::Reasoning(ReasoningContent {
                    text:             "First block. ".to_owned(),
                    signature:        None,
                    signature_origin: None,
                    redacted:         false,
                }),
                ContentPart::Reasoning(ReasoningContent {
                    text:             "Second block.".to_owned(),
                    signature:        None,
                    signature_origin: None,
                    redacted:         false,
                }),
                ContentPart::Text {
                    text: "Answer.".to_owned(),
                },
            ]))
            .user("thanks")
            .build()?,
    )?;

    let encoded = OpenAiChatCodec::default().encode(&call, false)?;

    assert_eq!(
        encoded.body["messages"][1]["reasoning_content"],
        json!("First block. Second block.")
    );
    Ok(())
}

#[test]
fn a_body_without_choices_fails_to_decode() -> Result<(), Box<dyn StdError>> {
    let error = OpenAiChatCodec::default()
        .decode_response(&route()?, json!({ "id": "chatcmpl-1", "choices": [] }))
        .err()
        .ok_or("expected an empty choices array to fail")?;

    assert_eq!(error.kind(), ErrorKind::ResponseDecode);
    assert_eq!(error.retry_classification(), RetryClassification::Safe);
    Ok(())
}

#[test]
fn an_invalid_stream_chunk_fails_retryably() -> Result<(), Box<dyn StdError>> {
    // A chunk that is not JSON is indistinguishable from mid-stream
    // corruption; the old client retried it and this keeps that contract.
    let mut decoder = OpenAiChatCodec::default().stream_decoder(&route()?);

    let error = decoder
        .decode(SseEvent {
            event: None,
            data:  "{\"id\": \"chatcmpl-1\", \"choi".to_owned(),
        })
        .err()
        .ok_or("expected the truncated chunk to fail the stream")?;

    assert_eq!(error.kind(), ErrorKind::StreamDecode);
    assert_eq!(error.retry_classification(), RetryClassification::Safe);
    Ok(())
}

#[test]
fn an_explicit_null_error_member_is_not_an_error() -> Result<(), Box<dyn StdError>> {
    // A skin that spells out `"error": null` on success chunks must not
    // fail every stream it serves.
    let mut decoder = OpenAiChatCodec::default().stream_decoder(&route()?);

    let events = decoder.decode(SseEvent {
        event: None,
        data:  json!({
            "id": "chatcmpl-1",
            "error": null,
            "choices": [{ "delta": { "content": "hi" } }],
        })
        .to_string(),
    })?;

    assert!(
        events
            .iter()
            .any(|event| matches!(event, StreamEvent::TextDelta { text, .. } if text == "hi")),
        "{events:?}"
    );
    Ok(())
}

#[test]
fn an_array_form_content_delta_decodes_like_the_blocking_path() -> Result<(), Box<dyn StdError>> {
    // A skin that streams content as an array of parts must not complete
    // as an empty success; the blocking path already reads both shapes.
    let mut decoder = OpenAiChatCodec::default().stream_decoder(&route()?);

    let events = decoder.decode(SseEvent {
        event: None,
        data:  json!({
            "id": "chatcmpl-1",
            "choices": [{ "delta": { "content": [
                { "type": "text", "text": "Hel" },
                { "type": "text", "text": "lo" },
            ] } }],
        })
        .to_string(),
    })?;

    assert!(
        events
            .iter()
            .any(|event| matches!(event, StreamEvent::TextDelta { text, .. } if text == "Hello")),
        "{events:?}"
    );
    Ok(())
}

#[test]
fn a_tool_call_fragment_without_an_index_fails_retryably() -> Result<(), Box<dyn StdError>> {
    // `index` is the accumulation slot. Defaulting a missing one to slot
    // 0 would silently merge parallel calls into one call with garbled
    // arguments; the reference decoder required the field and failed the
    // chunk retryably.
    let mut decoder = OpenAiChatCodec::default().stream_decoder(&route()?);

    let error = decoder
        .decode(SseEvent {
            event: None,
            data:  json!({
                "id": "chatcmpl-1",
                "choices": [{ "delta": { "tool_calls": [{
                    "id": "call-1",
                    "function": { "name": "weather", "arguments": "{}" },
                }] } }],
            })
            .to_string(),
        })
        .err()
        .ok_or("expected the index-less fragment to fail the stream")?;

    assert_eq!(error.kind(), ErrorKind::StreamDecode);
    assert_eq!(error.retry_classification(), RetryClassification::Safe);
    Ok(())
}

#[test]
fn arguments_before_identity_assemble_once_identity_arrives() -> Result<(), Box<dyn StdError>> {
    // Some skins deterministically stream `{index, function: {arguments}}`
    // before the fragment that carries the call's identity. The slot
    // opens on the arguments alone, and the late id and name repair the
    // block; failing the stream instead would retry forever against such
    // a skin.
    let events = stream(vec![
        json!({ "id": "chatcmpl-1", "choices": [{ "delta": { "tool_calls": [
            { "index": 0, "function": { "arguments": "{\"q\":" } },
        ] } }] }),
        json!({ "id": "chatcmpl-1", "choices": [{ "delta": { "tool_calls": [
            { "index": 0, "id": "call-1",
              "function": { "name": "search", "arguments": "\"rust\"}" } },
        ] } }] }),
        json!({ "id": "chatcmpl-1", "choices": [{ "delta": {}, "finish_reason": "tool_calls" }] }),
    ])?;

    let starts = events
        .iter()
        .filter(|event| matches!(event, StreamEvent::ContentBlockStart { .. }))
        .count();
    assert_eq!(starts, 1, "the late identity must not reopen the block");

    let responses = completed(&events);
    assert_eq!(responses.len(), 1);
    let calls = tool_parts(responses[0]);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "call-1");
    assert_eq!(calls[0].name, "search");
    assert_eq!(calls[0].input.wire_value(), json!({ "q": "rust" }));
    Ok(())
}

#[test]
fn arguments_only_to_stream_end_keep_the_synthesized_identity() -> Result<(), Box<dyn StdError>> {
    // When the identity never arrives at all, the call still assembles —
    // the reference decoder emitted an identity-less call — with the
    // synthesized block id standing in for the call id and an empty name.
    let events = stream(vec![
        json!({ "id": "chatcmpl-1", "choices": [{ "delta": { "tool_calls": [
            { "index": 0, "function": { "arguments": "{\"q\":" } },
        ] } }] }),
        json!({ "id": "chatcmpl-1", "choices": [{ "delta": { "tool_calls": [
            { "index": 0, "function": { "arguments": "\"rust\"}" } },
        ] } }] }),
        json!({ "id": "chatcmpl-1", "choices": [{ "delta": {}, "finish_reason": "tool_calls" }] }),
    ])?;

    let responses = completed(&events);
    assert_eq!(responses.len(), 1);
    assert_eq!(responses[0].finish_reason, FinishReason::ToolCall);
    let calls = tool_parts(responses[0]);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "tool-0");
    assert_eq!(calls[0].name, "");
    assert_eq!(calls[0].input.wire_value(), json!({ "q": "rust" }));
    Ok(())
}

#[test]
fn tool_call_identity_arriving_on_a_later_fragment_is_kept() -> Result<(), Box<dyn StdError>> {
    // A skin may open the slot with the name and send the provider call
    // id on a later fragment. Discarding the late id would answer the
    // call with the synthesized block id, which the provider rejects on
    // the next turn.
    let events = stream(vec![
        json!({ "id": "chatcmpl-1", "choices": [{ "delta": { "tool_calls": [
            { "index": 0, "function": { "name": "search", "arguments": "{\"q\":" } },
        ] } }] }),
        json!({ "id": "chatcmpl-1", "choices": [{ "delta": { "tool_calls": [
            { "index": 0, "id": "call-real", "function": { "arguments": "\"rust\"}" } },
        ] } }] }),
        json!({ "id": "chatcmpl-1", "choices": [{ "delta": {}, "finish_reason": "tool_calls" }] }),
    ])?;

    let responses = completed(&events);
    assert_eq!(responses.len(), 1);
    let calls = tool_parts(responses[0]);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "call-real");
    assert_eq!(calls[0].name, "search");
    assert_eq!(calls[0].input.wire_value(), json!({ "q": "rust" }));
    Ok(())
}

#[test]
fn a_streamed_tool_call_wins_over_a_stop_finish_reason() -> Result<(), Box<dyn StdError>> {
    // Some skins stream tool calls yet report `finish_reason: "stop"`.
    // The streamed blocks are the ground truth — an agent loop keyed on
    // `ToolCall` must see the calls it is meant to execute.
    let events = stream(vec![
        json!({ "id": "chatcmpl-1", "choices": [{ "delta": { "tool_calls": [
            { "index": 0, "id": "call-1",
              "function": { "name": "search", "arguments": "{\"q\":\"rust\"}" } },
        ] } }] }),
        json!({ "id": "chatcmpl-1", "choices": [{ "delta": {}, "finish_reason": "stop" }] }),
    ])?;

    let responses = completed(&events);
    assert_eq!(responses.len(), 1);
    assert_eq!(responses[0].finish_reason, FinishReason::ToolCall);
    Ok(())
}

#[test]
fn a_streamed_tool_call_without_a_finish_reason_fails() -> Result<(), Box<dyn StdError>> {
    // Even syntactically complete arguments cannot prove the model has
    // finished the call. In particular, an empty prefix must not become {}.
    for arguments in ["", "{\"q\":", "{}"] {
        let mut decoder = OpenAiChatCodec::default().stream_decoder(&route()?);
        let events = decoder.decode(SseEvent {
            event: None,
            data:  json!({ "id": "chatcmpl-1", "choices": [{ "delta": {
                "tool_calls": [{ "index": 0, "id": "call-1",
                    "function": { "name": "search", "arguments": arguments } }],
            } }] })
            .to_string(),
        })?;
        assert!(completed(&events).is_empty());
        let error = decoder.finish().expect_err("unfinished tool call");
        assert_eq!(error.kind(), ErrorKind::StreamDecode);
        assert_eq!(error.retry_classification(), RetryClassification::Safe);
    }
    Ok(())
}

#[test]
fn a_tool_call_cut_at_the_output_limit_is_dropped() -> Result<(), Box<dyn StdError>> {
    // Verified live: `finish_reason: length` beside a call whose
    // arguments are a JSON prefix. The call leaves the content and a
    // warning names it.
    let response = decode(json!({
        "id": "chatcmpl-1",
        "choices": [{
            "message": {
                "tool_calls": [{
                    "id": "call-1",
                    "function": {
                        "name": "write_note",
                        "arguments": "{\"title\":\"Rome\",\"body\":\"Rome began",
                    },
                }],
            },
            "finish_reason": "length",
        }],
    }))?;

    assert!(tool_parts(&response).is_empty());
    assert_eq!(response.finish_reason, FinishReason::Length);
    assert_eq!(response.warnings.len(), 1);
    assert_eq!(response.warnings[0].code, "truncated_tool_call");
    assert!(response.warnings[0].message.contains("write_note"));
    Ok(())
}

#[test]
fn a_non_stop_finish_reason_is_kept_despite_streamed_tool_calls() -> Result<(), Box<dyn StdError>> {
    // Truncation trumps inference: a `length` stop on a partial call is
    // still a truncated answer, and the partial call is not a call.
    let events = stream(vec![
        json!({ "id": "chatcmpl-1", "choices": [{ "delta": { "tool_calls": [
            { "index": 0, "id": "call-1",
              "function": { "name": "search", "arguments": "{\"q\":" } },
        ] } }] }),
        json!({ "id": "chatcmpl-1", "choices": [{ "delta": {}, "finish_reason": "length" }] }),
    ])?;

    let responses = completed(&events);
    assert_eq!(responses.len(), 1);
    assert_eq!(responses[0].finish_reason, FinishReason::Length);
    assert!(tool_parts(responses[0]).is_empty());
    assert_eq!(responses[0].warnings.len(), 1);
    assert_eq!(responses[0].warnings[0].code, "truncated_tool_call");
    Ok(())
}

#[test]
fn a_refusal_fails_instead_of_decoding() -> Result<(), Box<dyn StdError>> {
    // The structured-output refusal channel: content null, refusal text.
    // Decoding it as an empty success would hide the refusal from the
    // caller and from failover — the contract H8 set for Anthropic and
    // Bedrock extends here.
    let body = json!({
        "id": "chatcmpl-1",
        "choices": [{
            "message": { "role": "assistant", "content": null, "refusal": "I can't do that." },
            "finish_reason": "stop",
        }],
    });

    let error = OpenAiChatCodec::default()
        .decode_response(&route()?, body.clone())
        .expect_err("a refusal must fail the call");

    assert_eq!(error.kind(), ErrorKind::ContentFilter);
    assert_eq!(error.provider_code(), Some("refusal"));
    assert!(error.to_string().contains("I can't do that."), "{error}");
    assert_eq!(error.raw_data(), Some(&body));
    Ok(())
}

#[test]
fn a_streamed_refusal_fails_the_stream_with_the_whole_explanation() -> Result<(), Box<dyn StdError>>
{
    let mut decoder = OpenAiChatCodec::default().stream_decoder(&route()?);
    for fragment in ["I can't ", "do that."] {
        decoder.decode(SseEvent {
            event: None,
            data:  json!({
                "id": "chatcmpl-1",
                "choices": [{ "delta": { "refusal": fragment } }],
            })
            .to_string(),
        })?;
    }

    let error = decoder
        .finish()
        .expect_err("a streamed refusal must fail the stream");

    assert_eq!(error.kind(), ErrorKind::ContentFilter);
    assert_eq!(error.provider_code(), Some("refusal"));
    assert!(error.to_string().contains("I can't do that."), "{error}");
    Ok(())
}

#[test]
fn a_choice_without_a_message_fails_to_decode() -> Result<(), Box<dyn StdError>> {
    let error = OpenAiChatCodec::default()
        .decode_response(
            &route()?,
            json!({ "id": "chatcmpl-1", "choices": [{ "finish_reason": "stop" }] }),
        )
        .err()
        .ok_or("expected a choice without a message to fail")?;

    assert_eq!(error.kind(), ErrorKind::ResponseDecode);
    assert_eq!(error.retry_classification(), RetryClassification::Safe);
    Ok(())
}

#[test]
fn a_tool_call_without_its_identity_fails_to_decode() -> Result<(), Box<dyn StdError>> {
    // A call missing its id cannot be answered; one missing its name
    // cannot be dispatched. Both must fail retryably instead of decoding
    // into a part that poisons the replayed conversation.
    for tool_call in [
        json!({ "type": "function", "function": { "name": "f", "arguments": "{}" } }),
        json!({ "id": "call-1", "type": "function", "function": { "arguments": "{}" } }),
    ] {
        let error = OpenAiChatCodec::default()
            .decode_response(
                &route()?,
                json!({
                    "id": "chatcmpl-1",
                    "choices": [{ "message": { "tool_calls": [tool_call] } }],
                }),
            )
            .err()
            .ok_or("expected a tool call without id or name to fail")?;
        assert_eq!(error.kind(), ErrorKind::ResponseDecode);
        assert_eq!(error.retry_classification(), RetryClassification::Safe);
    }
    Ok(())
}

/// A one-provider catalog whose model opts into `cache_control`
/// breakpoints, the way an aggregator fronting an Anthropic model does.
const BREAKPOINT_CATALOG: &str = r#"
    schema_version = 1

    [providers.gateway]
    display_name = "Gateway"
    base_url = "http://127.0.0.1"
    default_model = "fronted"
    auth = { type = "bearer" }

    [providers.gateway.models.fronted]
    display_name = "Fronted"
    api_model = "fronted-v1"
    capabilities = { text = true, caching = true }
protocol_options = { cache_breakpoints = true }
"#;

/// A one-provider catalog whose model takes a cache routing hint, the
/// way Venice and OpenAI take `prompt_cache_key`.
const ROUTING_CATALOG: &str = r#"
    schema_version = 1

    [providers.gateway]
    display_name = "Gateway"
    base_url = "http://127.0.0.1"
    default_model = "routed"
    auth = { type = "bearer" }

    [providers.gateway.models.routed]
    display_name = "Routed"
    api_model = "routed-v1"
    capabilities = { text = true, tools = true, caching = true, cache_routing = true, tool_choice = { required = true, named = true } }
"#;

fn routed(request: Request) -> Result<Value, Box<dyn StdError>> {
    let call = resolved_in(ROUTING_CATALOG, request)?;
    Ok(OpenAiChatCodec::default().encode(&call, false)?.body)
}

#[test]
fn the_default_cache_hint_sends_a_stable_fingerprint() -> Result<(), Box<dyn StdError>> {
    let request = || {
        Request::builder()
            .model("gateway/routed")
            .system("Keep it short.")
            .user("Hello")
            .build()
    };
    let first = routed(request()?)?;
    let second = routed(request()?)?;

    let key = first["prompt_cache_key"]
        .as_str()
        .ok_or("no prompt_cache_key was sent")?;
    assert!(key.starts_with("lithos-"), "unexpected key shape: {key}");
    assert_eq!(first["prompt_cache_key"], second["prompt_cache_key"]);

    // A different system prefix routes elsewhere; a different user turn
    // does not, so an agent loop keeps one replica.
    let other_system = routed(
        Request::builder()
            .model("gateway/routed")
            .system("Answer in French.")
            .user("Hello")
            .build()?,
    )?;
    assert_ne!(first["prompt_cache_key"], other_system["prompt_cache_key"]);
    let other_turn = routed(
        Request::builder()
            .model("gateway/routed")
            .system("Keep it short.")
            .user("A different question entirely")
            .build()?,
    )?;
    assert_eq!(first["prompt_cache_key"], other_turn["prompt_cache_key"]);
    Ok(())
}

#[test]
fn an_explicit_cache_key_is_sent_verbatim() -> Result<(), Box<dyn StdError>> {
    let body = routed(
        Request::builder()
            .model("gateway/routed")
            .user("Hello")
            .cache_key("tenant-42")
            .build()?,
    )?;
    assert_eq!(body["prompt_cache_key"], json!("tenant-42"));
    Ok(())
}

#[test]
fn a_disabled_cache_hint_sends_nothing() -> Result<(), Box<dyn StdError>> {
    let body = routed(
        Request::builder()
            .model("gateway/routed")
            .user("Hello")
            .cache_hint(CacheHint::Disabled)
            .build()?,
    )?;
    assert_eq!(body.get("prompt_cache_key"), None);
    Ok(())
}

#[test]
fn auto_cache_off_suppresses_the_fingerprint() -> Result<(), Box<dyn StdError>> {
    let body = routed(
        Request::builder()
            .model("gateway/routed")
            .user("Hello")
            .provider_option("gateway", "auto_cache", json!(false))
            .build()?,
    )?;
    assert_eq!(body.get("prompt_cache_key"), None);
    Ok(())
}

#[test]
fn a_raw_prompt_cache_key_wins_over_the_fingerprint() -> Result<(), Box<dyn StdError>> {
    let body = routed(
        Request::builder()
            .model("gateway/routed")
            .user("Hello")
            .provider_option("gateway", "prompt_cache_key", json!("raw-key"))
            .build()?,
    )?;
    assert_eq!(body["prompt_cache_key"], json!("raw-key"));
    Ok(())
}

#[test]
fn no_routing_capability_means_no_fingerprint() -> Result<(), Box<dyn StdError>> {
    let call = resolved_in(BREAKPOINT_CATALOG, multi_turn("gateway/fronted")?)?;
    let encoded = OpenAiChatCodec::default().encode(&call, false)?;
    assert_eq!(encoded.body.get("prompt_cache_key"), None);
    Ok(())
}

/// A one-provider catalog with provider-level default request options,
/// the way a Venice row turns off the injected system prompt.
const DEFAULT_OPTIONS_CATALOG: &str = r#"
    schema_version = 1

    [providers.gateway]
    display_name = "Gateway"
    base_url = "http://127.0.0.1"
    default_model = "fronted"
    auth = { type = "bearer" }

    [providers.gateway.default_options]
    venice_parameters = { include_venice_system_prompt = false, strip_thinking_response = false }

    [providers.gateway.models.fronted]
    display_name = "Fronted"
    api_model = "fronted-v1"
    capabilities = { text = true }
"#;

#[test]
fn catalog_default_options_reach_the_wire() -> Result<(), Box<dyn StdError>> {
    let call = resolved_in(
        DEFAULT_OPTIONS_CATALOG,
        Request::builder()
            .model("gateway/fronted")
            .user("Hello")
            .build()?,
    )?;

    let encoded = OpenAiChatCodec::default().encode(&call, false)?;

    assert_eq!(
        encoded.body["venice_parameters"]["include_venice_system_prompt"],
        json!(false)
    );
    Ok(())
}

#[test]
fn request_options_win_over_catalog_defaults_key_by_key() -> Result<(), Box<dyn StdError>> {
    let call = resolved_in(
        DEFAULT_OPTIONS_CATALOG,
        Request::builder()
            .model("gateway/fronted")
            .user("Hello")
            .provider_option(
                "gateway",
                "venice_parameters",
                json!({ "include_venice_system_prompt": true }),
            )
            .build()?,
    )?;

    let encoded = OpenAiChatCodec::default().encode(&call, false)?;

    // The request overrides the one key it names; the default's sibling
    // key survives, because option namespaces merge recursively.
    assert_eq!(
        encoded.body["venice_parameters"]["include_venice_system_prompt"],
        json!(true)
    );
    assert_eq!(
        encoded.body["venice_parameters"]["strip_thinking_response"],
        json!(false)
    );
    Ok(())
}

#[test]
fn a_control_key_in_catalog_defaults_is_consumed() -> Result<(), Box<dyn StdError>> {
    let catalog = BREAKPOINT_CATALOG.replace(
        "[providers.gateway.models.fronted]",
        "[providers.gateway.default_options]\n\
         auto_cache = false\n\n\
         [providers.gateway.models.fronted]",
    );
    let call = resolved_in(&catalog, multi_turn("gateway/fronted")?)?;

    let encoded = OpenAiChatCodec::default().encode(&call, false)?;

    let rendered = to_string(&encoded.body)?;
    assert_eq!(rendered.matches("cache_control").count(), 0);
    assert_eq!(encoded.body.get("auto_cache"), None);
    Ok(())
}

fn multi_turn(model: &str) -> Result<Request, Box<dyn StdError>> {
    Ok(Request::builder()
        .model(model)
        .system("Keep it short.")
        .user("What is the capital of France?")
        .message(Message::text(Role::Assistant, "Paris."))
        .user("And of Spain?")
        .build()?)
}

#[test]
fn auto_cache_marks_the_system_message_and_the_prefix() -> Result<(), Box<dyn StdError>> {
    let call = resolved_in(BREAKPOINT_CATALOG, multi_turn("gateway/fronted")?)?;

    let encoded = OpenAiChatCodec::default().encode(&call, false)?;

    let messages = &encoded.body["messages"];
    assert_eq!(
        messages[0]["content"][0]["cache_control"]["type"],
        "ephemeral"
    );
    assert_eq!(
        messages[1]["content"][0]["cache_control"]["type"],
        "ephemeral"
    );
    assert_eq!(
        messages[1]["content"][0]["text"],
        "What is the capital of France?"
    );
    assert!(
        messages[3]["content"].is_string(),
        "the last user turn keeps the plain string form"
    );
    Ok(())
}

#[test]
fn auto_cache_false_places_no_breakpoints() -> Result<(), Box<dyn StdError>> {
    let call = resolved_in(
        BREAKPOINT_CATALOG,
        Request::builder()
            .model("gateway/fronted")
            .system("Keep it short.")
            .user("What is the capital of France?")
            .message(Message::text(Role::Assistant, "Paris."))
            .user("And of Spain?")
            .provider_option("gateway", "auto_cache", json!(false))
            .build()?,
    )?;

    let encoded = OpenAiChatCodec::default().encode(&call, false)?;

    assert_eq!(
        to_string(&encoded.body)?.matches("cache_control").count(),
        0
    );
    Ok(())
}

#[test]
fn caching_without_breakpoint_support_places_no_breakpoints() -> Result<(), Box<dyn StdError>> {
    // The builtin model declares `caching` for its pricing but not
    // `cache_breakpoints`. A skin whose caching is automatic can reject
    // the part-array rewrite, so its content must stay untouched.
    let call = resolved(multi_turn(MODEL)?)?;

    let encoded = OpenAiChatCodec::default().encode(&call, false)?;

    assert_eq!(
        to_string(&encoded.body)?.matches("cache_control").count(),
        0
    );
    assert!(
        encoded.body["messages"][0]["content"].is_string(),
        "unmarked messages keep the plain string form"
    );
    Ok(())
}

#[test]
fn reasoning_details_are_preserved_verbatim() -> Result<(), Box<dyn StdError>> {
    let details = json!([
        { "type": "reasoning.encrypted", "id": "rs-1", "data": "opaque" },
        { "type": "reasoning.text", "text": "step one", "index": 0 },
    ]);
    let response = decode(json!({
        "choices": [{
            "message": { "content": "done", "reasoning_details": details },
            "finish_reason": "stop",
        }],
    }))?;

    let first = response.content.first().ok_or("expected an opaque part")?;
    match first {
        ContentPart::Opaque { kind, data } => {
            assert_eq!(kind, "openai_compatible.reasoning_details");
            assert_eq!(data, &details);
        }
        other => return Err(format!("expected an opaque part, got {other:?}").into()),
    }
    Ok(())
}

#[test]
fn same_type_unindexed_complete_details_stay_separate_entries() -> Result<(), Box<dyn StdError>> {
    // Multi-block reasoning through an aggregator: two entries of one
    // type, no indexes, each sealed by its own signature. Coalescing them
    // would discard the second signature and fail verification upstream
    // on replay — only a stream coalesces, to undo its own fragmenting.
    let details = json!([
        { "type": "reasoning.text", "text": "step one", "signature": "sig-1" },
        { "type": "reasoning.text", "text": "step two", "signature": "sig-2" },
    ]);
    let response = decode(json!({
        "choices": [{
            "message": { "content": "done", "reasoning_details": details },
            "finish_reason": "stop",
        }],
    }))?;

    let first = response.content.first().ok_or("expected an opaque part")?;
    match first {
        ContentPart::Opaque { kind, data } => {
            assert_eq!(kind, "openai_compatible.reasoning_details");
            assert_eq!(data, &details);
        }
        other => return Err(format!("expected an opaque part, got {other:?}").into()),
    }
    Ok(())
}

#[test]
fn streamed_detail_fragments_coalesce_by_type_and_index() -> Result<(), Box<dyn StdError>> {
    let events = stream(vec![
        json!({ "id": "chatcmpl-1", "choices": [{ "delta": { "reasoning_details": [
            { "type": "reasoning.text", "index": 0, "text": "first " },
        ] } }] }),
        json!({ "choices": [{ "delta": { "reasoning_details": [
            { "type": "reasoning.text", "index": 1, "text": "second " },
        ] } }] }),
        json!({ "choices": [{ "delta": { "reasoning_details": [
            { "type": "reasoning.text", "index": 0, "text": "half" },
        ] } }] }),
        json!({ "choices": [{ "delta": { "reasoning_details": [
            { "type": "reasoning.text", "index": 1, "text": "half", "signature": "sig" },
        ] } }] }),
        json!({ "choices": [{ "delta": { "content": "done" }, "finish_reason": "stop" }] }),
    ])?;

    let response = completed(&events)
        .first()
        .copied()
        .ok_or("expected a completed response")?;
    let first = response.content.first().ok_or("expected an opaque part")?;
    match first {
        ContentPart::Opaque { kind, data } => {
            assert_eq!(kind, "openai_compatible.reasoning_details");
            assert_eq!(
                data,
                &json!([
                    { "type": "reasoning.text", "index": 0, "text": "first half" },
                    {
                        "type": "reasoning.text",
                        "index": 1,
                        "text": "second half",
                        "signature": "sig",
                    },
                ])
            );
        }
        other => return Err(format!("expected an opaque part, got {other:?}").into()),
    }
    Ok(())
}

#[test]
fn a_custom_tool_is_rejected_before_dispatch() -> Result<(), Box<dyn StdError>> {
    let call = resolved(
        Request::builder()
            .model(MODEL)
            .user("Hello")
            .tool(ToolDefinition::custom(
                "apply_patch",
                "Applies a patch",
                json!({ "type": "grammar" }),
            ))
            .build()?,
    )?;

    let error = OpenAiChatCodec::default()
        .encode(&call, false)
        .err()
        .ok_or("expected a custom tool to be rejected")?;

    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    Ok(())
}

#[test]
fn count_tokens_is_unavailable() -> Result<(), Box<dyn StdError>> {
    let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;

    assert!(
        OpenAiChatCodec::default()
            .encode_count_tokens(&call)
            .is_none()
    );
    Ok(())
}

#[test]
fn streaming_always_requests_usage() -> Result<(), Box<dyn StdError>> {
    let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;

    let streamed = OpenAiChatCodec::default().encode(&call, true)?;
    let complete = OpenAiChatCodec::default().encode(&call, false)?;

    assert_eq!(streamed.body["stream"], json!(true));
    assert_eq!(
        streamed.body["stream_options"],
        json!({ "include_usage": true })
    );
    // A blocking request omits both members entirely; a strict skin may
    // reject an explicit `stream: false`.
    assert_eq!(complete.body.get("stream"), None);
    assert_eq!(complete.body.get("stream_options"), None);
    Ok(())
}

#[test]
fn interleaved_tool_calls_keep_separate_blocks() -> Result<(), Box<dyn StdError>> {
    let events = stream(vec![
        json!({ "id": "chatcmpl-1", "choices": [{ "delta": {
            "tool_calls": [{ "index": 0, "id": "call-a", "function": { "name": "alpha", "arguments": "" } }],
        } }] }),
        json!({ "choices": [{ "delta": {
            "tool_calls": [{ "index": 1, "id": "call-b", "function": { "name": "beta", "arguments": "{\"b\":" } }],
        } }] }),
        json!({ "choices": [{ "delta": {
            "tool_calls": [{ "index": 0, "function": { "arguments": "{\"a\":1}" } }],
        } }] }),
        json!({ "choices": [{ "delta": {
            "tool_calls": [{ "index": 1, "function": { "arguments": "2}" } }],
        } }] }),
        json!({ "choices": [{ "delta": {}, "finish_reason": "tool_calls" }] }),
    ])?;

    let starts: Vec<(String, ContentBlockKind)> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ContentBlockStart { id, kind } => {
                Some((id.as_str().to_owned(), kind.clone()))
            }
            _ => None,
        })
        .collect();
    assert_eq!(starts.len(), 2);
    assert_eq!(starts[0].0, "tool-0");
    assert_eq!(starts[1].0, "tool-1");

    let response = completed(&events)
        .first()
        .copied()
        .ok_or("expected a completed response")?;
    let calls = tool_parts(response);
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].id, "call-a");
    assert_eq!(calls[0].name, "alpha");
    assert_eq!(calls[0].input.wire_value(), json!({ "a": 1 }));
    assert_eq!(calls[1].id, "call-b");
    assert_eq!(calls[1].input.wire_value(), json!({ "b": 2 }));
    Ok(())
}

#[test]
fn reasoning_deltas_produce_a_reasoning_block() -> Result<(), Box<dyn StdError>> {
    let events = stream(vec![
        json!({ "id": "chatcmpl-1", "choices": [{ "delta": { "reasoning": "step " } }] }),
        json!({ "choices": [{ "delta": { "reasoning_content": "one" } }] }),
        json!({ "choices": [{ "delta": { "content": "answer" }, "finish_reason": "stop" }] }),
    ])?;

    let reasoning: String = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ReasoningDelta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(reasoning, "step one");

    let blocks: Vec<&str> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ContentBlockStart { id, .. } => Some(id.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(blocks, ["reasoning-0", "block-0"]);

    let response = completed(&events)
        .first()
        .copied()
        .ok_or("expected a completed response")?;
    let first = response.content.first().ok_or("expected reasoning")?;
    match first {
        ContentPart::Reasoning(reasoning) => assert_eq!(reasoning.text, "step one"),
        other => return Err(format!("expected reasoning, got {other:?}").into()),
    }
    assert_eq!(response.text(), "answer");
    Ok(())
}

#[test]
fn the_final_empty_choices_chunk_carries_usage() -> Result<(), Box<dyn StdError>> {
    let events = stream(vec![
        json!({ "id": "chatcmpl-1", "choices": [{ "delta": { "content": "Hel" } }] }),
        json!({ "choices": [{ "delta": { "content": "lo" }, "finish_reason": "stop" }] }),
        json!({ "choices": [], "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 5,
            "prompt_tokens_details": { "cached_tokens": 3 },
            "cost": 0.0001,
        } }),
    ])?;

    let usage: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::Usage { usage } => Some(*usage),
            _ => None,
        })
        .collect();
    assert_eq!(usage.len(), 1);
    assert_eq!(usage[0].input, 8);
    assert_eq!(usage[0].cache_read, 3);

    let responses = completed(&events);
    assert_eq!(responses.len(), 1);
    let response = responses[0];
    assert_eq!(response.id.as_deref(), Some("chatcmpl-1"));
    assert_eq!(response.text(), "Hello");
    assert_eq!(response.usage.output, 5);
    let cost = response.cost.ok_or("expected a provider cost")?;
    assert_eq!(cost.usd_micros, 100);
    assert_eq!(cost.source, CostSource::Provider);
    // This protocol supplies no terminal response document.
    assert!(response.raw.is_none());
    Ok(())
}

#[test]
fn a_stream_error_chunk_ends_the_stream() -> Result<(), Box<dyn StdError>> {
    let mut decoder = OpenAiChatCodec::default().stream_decoder(&route()?);

    let error: Error = decoder
        .decode(SseEvent {
            event: None,
            data:  json!({ "error": {
                "message": "rate limit reached",
                "code": "rate_limit_exceeded",
            } })
            .to_string(),
        })
        .err()
        .ok_or("expected an error chunk to fail the stream")?;

    assert_eq!(error.provider_code(), Some("rate_limit_exceeded"));
    Ok(())
}

#[test]
fn the_done_terminator_completes_the_stream() -> Result<(), Box<dyn StdError>> {
    let mut decoder = OpenAiChatCodec::default().stream_decoder(&route()?);
    decoder.decode(SseEvent {
        event: None,
        data:  json!({
            "id": "chatcmpl-1",
            "choices": [{ "delta": { "content": "ok" }, "finish_reason": "stop" }],
        })
        .to_string(),
    })?;

    let events = decoder.decode(SseEvent {
        event: None,
        data:  "[DONE]".to_owned(),
    })?;

    let responses = completed(&events);
    assert_eq!(responses.len(), 1, "the terminator completes the response");
    assert_eq!(responses[0].finish_reason, FinishReason::Stop);
    assert!(decoder.finish()?.is_empty(), "completing is idempotent");
    Ok(())
}
