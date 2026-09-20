use std::error::Error as StdError;

use serde_json::{Value, json};

use super::{AnthropicMessagesCodec, Codec};
use crate::codecs::test_support::{resolved, resolved_in};
use crate::transport::SseEvent;
use crate::types::{
    ContentPart, ErrorKind, FinishReason, ImageContent, MediaSource, Message, ReasoningContent,
    ReasoningEffort, Request, ResponseFormat, RetryClassification, Role, Speed, StreamEvent,
    ToolChoice, ToolDefinition, ToolResult,
};

const MODEL: &str = "anthropic/claude-sonnet-4-6";

/// One SSE frame as the transport would hand it to the decoder.
fn sse(event: &str, data: &Value) -> SseEvent {
    SseEvent {
        event: Some(event.to_owned()),
        data:  data.to_string(),
    }
}

/// Counts every `cache_control` breakpoint anywhere in a body.
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

/// A request with a system prompt, tools, and three user turns.
fn cacheable_request() -> Result<Request, Box<dyn StdError>> {
    Ok(Request::builder()
        .model(MODEL)
        .system("You are terse.")
        .user("First")
        .message(Message::text(Role::Assistant, "Answer"))
        .user("Second")
        .message(Message::text(Role::Assistant, "Answer"))
        .user("Third")
        .tool(ToolDefinition::function(
            "lookup",
            "Look something up",
            json!({ "type": "object" }),
        ))
        .build()?)
}

/// Drives a decoder over a whole transcript, then finishes it.
fn stream(events: Vec<SseEvent>) -> Result<Vec<StreamEvent>, Box<dyn StdError>> {
    let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;
    let mut decoder = AnthropicMessagesCodec.stream_decoder(call.route());

    let mut decoded = Vec::new();
    for event in events {
        decoded.extend(decoder.decode(event)?);
    }
    decoded.extend(decoder.finish()?);
    Ok(decoded)
}

#[test]
fn a_signature_repeated_on_start_and_delta_is_not_doubled() -> Result<(), Box<dyn StdError>> {
    // The documented protocol sends one signature_delta and none on the
    // start snapshot — but a provider that puts the whole blob in both
    // places must not produce a concatenated signature the replay
    // verifier rejects. Each arrival replaces, as the reference decoder
    // did.
    let events = stream(vec![
        sse(
            "message_start",
            &json!({ "type": "message_start", "message": { "id": "msg_1" } }),
        ),
        sse(
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": { "type": "thinking", "thinking": "", "signature": "sig-1" },
            }),
        ),
        sse(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "thinking_delta", "thinking": "step one" },
            }),
        ),
        sse(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "signature_delta", "signature": "sig-1" },
            }),
        ),
        sse(
            "content_block_stop",
            &json!({ "type": "content_block_stop", "index": 0 }),
        ),
    ])?;

    let reasoning = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::ContentBlockEnd {
                part: ContentPart::Reasoning(part),
                ..
            } => Some(part),
            _ => None,
        })
        .ok_or("expected a reasoning part")?;
    assert_eq!(reasoning.signature.as_deref(), Some("sig-1"));
    Ok(())
}

#[test]
fn a_signature_on_the_stop_event_replaces_the_captured_one() -> Result<(), Box<dyn StdError>> {
    // The documented protocol never sends a signature on
    // content_block_stop, but the reference decoder preferred one that
    // arrived there; a dialect sending it only on the stop event must
    // not close the block unsigned.
    let events = stream(vec![
        sse(
            "message_start",
            &json!({ "type": "message_start", "message": { "id": "msg_1" } }),
        ),
        sse(
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": { "type": "thinking", "thinking": "" },
            }),
        ),
        sse(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "thinking_delta", "thinking": "step one" },
            }),
        ),
        sse(
            "content_block_stop",
            &json!({
                "type": "content_block_stop",
                "index": 0,
                "content_block": { "type": "thinking", "signature": "sig-stop" },
            }),
        ),
    ])?;

    let reasoning = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::ContentBlockEnd {
                part: ContentPart::Reasoning(part),
                ..
            } => Some(part),
            _ => None,
        })
        .ok_or("expected a reasoning part")?;
    assert_eq!(reasoning.signature.as_deref(), Some("sig-stop"));
    Ok(())
}

#[test]
fn a_tool_use_cut_at_the_output_limit_is_dropped() -> Result<(), Box<dyn StdError>> {
    // Verified live: `stop_reason: max_tokens` beside a `tool_use` block
    // whose input is `{}`. The block is not a call and leaves the content.
    let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;

    let response = AnthropicMessagesCodec.decode_response(
        call.route(),
        json!({
            "id": "msg-1",
            "model": "claude-sonnet-4-6",
            "content": [
                { "type": "tool_use", "id": "toolu_1", "name": "write_note", "input": {} }
            ],
            "stop_reason": "max_tokens",
            "usage": { "input_tokens": 10, "output_tokens": 30 }
        }),
    )?;

    assert_eq!(response.content, Vec::new());
    assert_eq!(response.finish_reason, FinishReason::Length);
    assert_eq!(response.warnings.len(), 1);
    assert_eq!(response.warnings[0].code, "truncated_tool_call");
    Ok(())
}

#[test]
fn a_streamed_tool_use_cut_at_the_output_limit_is_dropped_from_the_completed_response()
-> Result<(), Box<dyn StdError>> {
    // Verified live: the stream opens the block, sends one empty
    // `input_json_delta`, and reports `max_tokens` on `message_delta`.
    let events = stream(vec![
        sse(
            "message_start",
            &json!({ "type": "message_start", "message": { "id": "msg_1" } }),
        ),
        sse(
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "write_note",
                    "input": {},
                },
            }),
        ),
        sse(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "input_json_delta", "partial_json": "" },
            }),
        ),
        sse(
            "content_block_stop",
            &json!({ "type": "content_block_stop", "index": 0 }),
        ),
        sse(
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": { "stop_reason": "max_tokens" },
                "usage": { "output_tokens": 30 },
            }),
        ),
        sse("message_stop", &json!({ "type": "message_stop" })),
    ])?;

    let Some(StreamEvent::Ended { response }) = events.last() else {
        return Err("expected the stream to end with a completed event".into());
    };
    assert_eq!(response.content, Vec::new());
    assert_eq!(response.finish_reason, FinishReason::Length);
    assert_eq!(response.warnings.len(), 1);
    assert_eq!(response.warnings[0].code, "truncated_tool_call");
    Ok(())
}

#[test]
fn a_streamed_server_tool_use_block_keeps_its_streamed_input() -> Result<(), Box<dyn StdError>> {
    // A server-side block opens with an empty `input` and streams the
    // real value through `input_json_delta`. The assembled opaque part
    // must carry the streamed input — the shape the blocking decoder
    // keeps whole — or a replay misrepresents what the model did.
    let events = stream(vec![
        sse(
            "message_start",
            &json!({ "type": "message_start", "message": { "id": "msg_1" } }),
        ),
        sse(
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "server_tool_use",
                    "id": "srvtoolu_1",
                    "name": "web_search",
                    "input": {},
                },
            }),
        ),
        sse(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "input_json_delta", "partial_json": "{\"query\":" },
            }),
        ),
        sse(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "input_json_delta", "partial_json": "\"rust\"}" },
            }),
        ),
        sse(
            "content_block_stop",
            &json!({ "type": "content_block_stop", "index": 0 }),
        ),
        sse("message_stop", &json!({ "type": "message_stop" })),
    ])?;

    let parts: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ContentBlockEnd { part, .. } => Some(part.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(parts, vec![ContentPart::Opaque {
        kind: "anthropic.server_tool_use".to_owned(),
        data: json!({
            "type": "server_tool_use",
            "id": "srvtoolu_1",
            "name": "web_search",
            "input": { "query": "rust" },
        }),
    }]);
    Ok(())
}

#[test]
fn encodes_output_controls_and_decodes_tool_calls() -> Result<(), Box<dyn StdError>> {
    let call = resolved(
        Request::builder()
            .model(MODEL)
            .user("Plan a trip")
            .reasoning_effort(ReasoningEffort::High)
            .speed(Speed::Fast)
            .stop_sequence("END")
            .stop_sequence("STOP")
            .metadata_entry("user_id", "u-1")
            .response_format(ResponseFormat::JsonSchema {
                name:   "trip".to_owned(),
                schema: json!({ "type": "object" }),
            })
            .build()?,
    )?;
    let codec = AnthropicMessagesCodec;

    let encoded = codec.encode(&call, false)?;

    assert_eq!(encoded.body["output_config"]["effort"], "high");
    assert_eq!(
        encoded.body["output_config"]["format"]["type"],
        "json_schema"
    );
    assert_eq!(encoded.body["speed"], "fast");
    // The speed reached the wire, so cost estimation may price it.
    assert_eq!(encoded.applied_speed, Some(Speed::Fast));
    assert_eq!(encoded.body["stop_sequences"], json!(["END", "STOP"]));
    assert_eq!(encoded.body["metadata"], json!({ "user_id": "u-1" }));
    // The model takes effort levels, so it also gets the adaptive
    // thinking object, and no request limit means the model's own.
    assert_eq!(encoded.body["thinking"], json!({ "type": "adaptive" }));
    assert_eq!(encoded.body["max_tokens"], 128_000);
    assert_eq!(encoded.body["stream"], false);
    assert!(encoded.url.ends_with("/v1/messages"));
    assert!(
        encoded
            .headers
            .contains(&("anthropic-version".to_owned(), "2023-06-01".to_owned()))
    );
    // The fast tier is a beta, so the body field travels with its header.
    assert!(encoded.headers.contains(&(
        "anthropic-beta".to_owned(),
        "fast-mode-2026-02-01".to_owned()
    )));

    let response = codec.decode_response(
        call.route(),
        json!({
            "id": "msg-1",
            "model": "claude-sonnet-4-6",
            "content": [
                { "type": "thinking", "thinking": "checked", "signature": "sig" },
                { "type": "tool_use", "id": "tool-1", "name": "lookup", "input": { "q": "x" } }
            ],
            "stop_reason": "tool_use",
            "usage": { "input_tokens": 10, "output_tokens": 4 }
        }),
    )?;

    assert!(matches!(
        response.content.as_slice(),
        [ContentPart::Reasoning(_), ContentPart::ToolCall(tool_call)]
            if tool_call.id == "tool-1" && tool_call.name == "lookup"
    ));
    assert!(response.raw.is_some());
    assert_eq!(response.cost, None);
    Ok(())
}

#[test]
fn a_foreign_signed_reasoning_part_is_skipped_with_a_warning() -> Result<(), Box<dyn StdError>> {
    // A Gemini-minted signature cannot verify here; replaying it fails
    // the request, so the part is dropped and the drop is reported. A
    // signed part with no recorded origin — persisted before origins
    // existed — still replays.
    let foreign = ReasoningContent {
        text:             "thought".to_owned(),
        signature:        Some("gemini-sig".to_owned()),
        signature_origin: Some("gemini".to_owned()),
        redacted:         false,
    };
    let legacy = ReasoningContent {
        text:             "older thought".to_owned(),
        signature:        Some("sig".to_owned()),
        signature_origin: Some("anthropic".to_owned()),
        redacted:         false,
    };
    let call = resolved(
        Request::builder()
            .model(MODEL)
            .user("Hello")
            .message(Message::new(Role::Assistant, [
                ContentPart::Reasoning(foreign),
                ContentPart::Reasoning(legacy),
                ContentPart::Text {
                    text: "answer".to_owned(),
                },
            ]))
            .user("Continue")
            .build()?,
    )?;

    let encoded = AnthropicMessagesCodec.encode(&call, false)?;

    let body = encoded.body.to_string();
    assert!(!body.contains("gemini-sig"), "{body}");
    assert!(body.contains("older thought"), "{body}");
    assert!(
        encoded
            .warnings
            .iter()
            .any(|warning| warning.message.contains("signed by another provider")),
        "{:?}",
        encoded.warnings
    );
    Ok(())
}

#[test]
fn json_object_instructs_the_system_text_instead_of_a_schema() -> Result<(), Box<dyn StdError>> {
    let codec = AnthropicMessagesCodec;

    // No schema in Anthropic's structured-output subset says "any JSON
    // object", so the format rides on the system text; a request with a
    // system prompt gets the instruction appended after it.
    let call = resolved(
        Request::builder()
            .model(MODEL)
            .system("You are terse.")
            .user("List the planets")
            .response_format(ResponseFormat::JsonObject)
            .provider_option("anthropic", "auto_cache", json!(false))
            .build()?,
    )?;
    let encoded = codec.encode(&call, false)?;
    assert_eq!(encoded.body.get("output_config"), None);
    assert_eq!(
        encoded.body["system"],
        "You are terse.\n\nYou must respond with valid JSON only, no other text."
    );

    // Without one, the instruction is the whole system text, and the
    // count body keeps it: counting must see what generation sends.
    let bare = resolved(
        Request::builder()
            .model(MODEL)
            .user("List the planets")
            .response_format(ResponseFormat::JsonObject)
            .provider_option("anthropic", "auto_cache", json!(false))
            .build()?,
    )?;
    let encoded = codec.encode(&bare, false)?;
    assert_eq!(encoded.body.get("output_config"), None);
    assert_eq!(
        encoded.body["system"],
        "You must respond with valid JSON only, no other text."
    );
    let counted = codec
        .encode_count_tokens(&bare)
        .ok_or("Anthropic should count tokens")??;
    assert_eq!(
        counted.body["system"],
        "You must respond with valid JSON only, no other text."
    );
    Ok(())
}

#[test]
fn a_whitespace_only_system_prompt_is_omitted() -> Result<(), Box<dyn StdError>> {
    // Templating commonly leaves a system prompt of pure whitespace. The
    // old encoder dropped it, and with auto-cache on it would otherwise
    // become a blank cached text block the provider rejects.
    let call = resolved(
        Request::builder()
            .model(MODEL)
            .system("   \n\t")
            .user("Hello")
            .build()?,
    )?;

    let encoded = AnthropicMessagesCodec.encode(&call, false)?;

    assert_eq!(encoded.body.get("system"), None);
    Ok(())
}

#[test]
fn usage_buckets_stay_disjoint_without_subtraction() -> Result<(), Box<dyn StdError>> {
    let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;

    let response = AnthropicMessagesCodec.decode_response(
        call.route(),
        json!({
            "id": "msg-1",
            "model": "claude-sonnet-4-6",
            "content": [],
            "usage": {
                "input_tokens": 50,
                "output_tokens": 1200,
                "cache_read_input_tokens": 9000,
                "cache_creation_input_tokens": 1000
            }
        }),
    )?;

    assert_eq!(response.usage.input, 50);
    assert_eq!(response.usage.output, 1200);
    assert_eq!(response.usage.cache_read, 9000);
    assert_eq!(response.usage.cache_write, 1000);
    assert_eq!(response.usage.reasoning, 0);
    assert_eq!(response.usage.total(), 11_250);
    Ok(())
}

#[test]
fn raw_options_win_and_controls_are_consumed() -> Result<(), Box<dyn StdError>> {
    let call = resolved(
        Request::builder()
            .model(MODEL)
            .user("Hello")
            .max_output_tokens(64)
            .provider_option("anthropic", "max_tokens", json!(2048))
            .provider_option("anthropic", "thinking", json!({ "type": "enabled" }))
            .provider_option("anthropic", "auto_cache", json!(false))
            .provider_option("openai", "max_tokens", json!(9))
            .build()?,
    )?;

    let encoded = AnthropicMessagesCodec.encode(&call, false)?;

    assert_eq!(encoded.body["max_tokens"], 2048);
    assert_eq!(encoded.body["thinking"], json!({ "type": "enabled" }));
    assert_eq!(encoded.body.get("auto_cache"), None);
    assert_eq!(breakpoints(&encoded.body), 0);
    Ok(())
}

#[test]
fn a_raw_thinking_option_with_a_forced_tool_choice_warns() -> Result<(), Box<dyn StdError>> {
    let call = resolved(
        Request::builder()
            .model(MODEL)
            .user("Look it up")
            .tool(ToolDefinition::function(
                "lookup",
                "Look something up",
                json!({ "type": "object" }),
            ))
            .tool_choice(ToolChoice::Required)
            .provider_option("anthropic", "thinking", json!({ "type": "enabled" }))
            .build()?,
    )?;

    let encoded = AnthropicMessagesCodec.encode(&call, false)?;

    // The raw option is authoritative, so it stays in the body even
    // though Anthropic rejects it beside a forced choice; the warning is
    // the caller's only signal before the provider's 400.
    assert_eq!(encoded.body["thinking"], json!({ "type": "enabled" }));
    assert_eq!(encoded.body["tool_choice"], json!({ "type": "any" }));
    assert!(encoded.warnings.iter().any(|warning| {
        warning.code == "unsupported_control"
            && warning
                .message
                .contains("a thinking provider option with a forced tool choice")
    }));
    Ok(())
}

#[test]
fn auto_cache_places_breakpoints_by_default() -> Result<(), Box<dyn StdError>> {
    let call = resolved(cacheable_request()?)?;

    let encoded = AnthropicMessagesCodec.encode(&call, false)?;

    assert_eq!(
        encoded.body["system"][0]["cache_control"]["type"],
        "ephemeral"
    );
    assert_eq!(
        encoded.body["tools"][0]["cache_control"]["type"],
        "ephemeral"
    );
    // Three user turns interleaved with two assistant turns: the prefix
    // breakpoint lands on the second one, at message index 2.
    assert_eq!(
        encoded.body["messages"][2]["content"][0]["cache_control"]["type"],
        "ephemeral"
    );
    assert_eq!(breakpoints(&encoded.body), 3);
    Ok(())
}

#[test]
fn auto_cache_false_places_no_breakpoints() -> Result<(), Box<dyn StdError>> {
    let call = resolved(
        Request::builder()
            .model(MODEL)
            .system("You are terse.")
            .user("First")
            .message(Message::text(Role::Assistant, "Answer"))
            .user("Second")
            .message(Message::text(Role::Assistant, "Answer"))
            .user("Third")
            .tool(ToolDefinition::function(
                "lookup",
                "Look something up",
                json!({ "type": "object" }),
            ))
            .provider_option("anthropic", "auto_cache", json!(false))
            .build()?,
    )?;

    let encoded = AnthropicMessagesCodec.encode(&call, false)?;

    assert_eq!(breakpoints(&encoded.body), 0);
    assert!(encoded.body["system"].is_string());
    Ok(())
}

#[test]
fn custom_tools_are_rejected_before_dispatch() -> Result<(), Box<dyn StdError>> {
    let call = resolved(
        Request::builder()
            .model(MODEL)
            .user("Patch it")
            .tool(ToolDefinition::custom(
                "apply_patch",
                "Apply a patch",
                json!({ "type": "grammar" }),
            ))
            .build()?,
    )?;

    let Err(error) = AnthropicMessagesCodec.encode(&call, false) else {
        return Err("a custom tool has no Anthropic encoding".into());
    };

    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(error.provider_code(), Some("unsupported_capability"));

    let counted = AnthropicMessagesCodec
        .encode_count_tokens(&call)
        .ok_or("Anthropic has a count-tokens endpoint")?;
    let Err(counted) = counted else {
        return Err("the count endpoint must reject a custom tool too".into());
    };
    assert_eq!(counted.provider_code(), Some("unsupported_capability"));
    Ok(())
}

#[test]
fn media_encodes_base64_and_url_sources() -> Result<(), Box<dyn StdError>> {
    let call = resolved(
        Request::builder()
            .model(MODEL)
            .message(Message::new(Role::User, [
                ContentPart::Image(ImageContent::new(MediaSource::base64("QUJD", "image/png"))),
                ContentPart::Image(ImageContent::new(MediaSource::url(
                    "https://example.com/cat.png",
                ))),
            ]))
            .provider_option("anthropic", "auto_cache", json!(false))
            .build()?,
    )?;

    let encoded = AnthropicMessagesCodec.encode(&call, false)?;
    let blocks = &encoded.body["messages"][0]["content"];

    assert_eq!(
        blocks[0]["source"],
        json!({ "type": "base64", "media_type": "image/png", "data": "QUJD" })
    );
    assert_eq!(
        blocks[1]["source"],
        json!({ "type": "url", "url": "https://example.com/cat.png" })
    );
    Ok(())
}

#[test]
fn redacted_thinking_round_trips_through_the_blocking_path() -> Result<(), Box<dyn StdError>> {
    let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;
    let codec = AnthropicMessagesCodec;

    let response = codec.decode_response(
        call.route(),
        json!({
            "id": "msg-1",
            "model": "claude-sonnet-4-6",
            "content": [{ "type": "redacted_thinking", "data": "ENCRYPTED" }],
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 4, "output_tokens": 2 }
        }),
    )?;

    let [ContentPart::Reasoning(reasoning)] = response.content.as_slice() else {
        return Err("expected one reasoning part".into());
    };
    assert!(reasoning.redacted);
    assert_eq!(reasoning.text, "ENCRYPTED");

    let replay = resolved(
        Request::builder()
            .model(MODEL)
            .user("Hello")
            .message(Message::new(Role::Assistant, [ContentPart::Reasoning(
                ReasoningContent {
                    text:             "ENCRYPTED".to_owned(),
                    signature:        None,
                    signature_origin: None,
                    redacted:         true,
                },
            )]))
            .provider_option("anthropic", "auto_cache", json!(false))
            .build()?,
    )?;
    let encoded = codec.encode(&replay, false)?;

    assert_eq!(
        encoded.body["messages"][1]["content"][0],
        json!({ "type": "redacted_thinking", "data": "ENCRYPTED" })
    );
    Ok(())
}

#[test]
fn streaming_folds_split_usage_into_one_snapshot() -> Result<(), Box<dyn StdError>> {
    let events = stream(vec![
        sse(
            "message_start",
            &json!({
                "type": "message_start",
                "message": {
                    "id": "msg-1",
                    "usage": {
                        "input_tokens": 11,
                        "cache_read_input_tokens": 2,
                        "cache_creation_input_tokens": 1,
                        "output_tokens": 0
                    }
                }
            }),
        ),
        sse(
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": { "stop_reason": "end_turn" },
                "usage": { "output_tokens": 5 }
            }),
        ),
        sse("message_stop", &json!({ "type": "message_stop" })),
    ])?;

    let snapshots: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::Usage { usage } => Some(*usage),
            _ => None,
        })
        .collect();

    let [first, last] = snapshots.as_slice() else {
        return Err(format!("expected two usage snapshots, got {snapshots:?}").into());
    };
    assert_eq!(first.input, 11);
    assert_eq!(first.output, 0);
    assert_eq!(last.input, 11);
    assert_eq!(last.cache_read, 2);
    assert_eq!(last.cache_write, 1);
    assert_eq!(last.output, 5);
    assert_eq!(last.reasoning, 0);
    Ok(())
}

#[test]
fn streaming_assembles_blocks_and_completes_once() -> Result<(), Box<dyn StdError>> {
    let events = stream(vec![
        sse(
            "message_start",
            &json!({ "type": "message_start", "message": { "id": "msg-1" } }),
        ),
        sse("ping", &json!({ "type": "ping" })),
        sse(
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": { "type": "thinking", "thinking": "" }
            }),
        ),
        sse(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "thinking_delta", "thinking": "step" }
            }),
        ),
        sse(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "signature_delta", "signature": "sig" }
            }),
        ),
        sse(
            "content_block_stop",
            &json!({ "type": "content_block_stop", "index": 0 }),
        ),
        sse(
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": 1,
                "content_block": { "type": "tool_use", "id": "toolu_1", "name": "lookup" }
            }),
        ),
        sse(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": 1,
                "delta": { "type": "input_json_delta", "partial_json": "{\"q\":" }
            }),
        ),
        sse(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": 1,
                "delta": { "type": "input_json_delta", "partial_json": "\"rust\"}" }
            }),
        ),
        sse(
            "content_block_stop",
            &json!({ "type": "content_block_stop", "index": 1 }),
        ),
        sse(
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": { "stop_reason": "tool_use" },
                "usage": { "output_tokens": 7 }
            }),
        ),
        sse("message_stop", &json!({ "type": "message_stop" })),
    ])?;

    let starts = events
        .iter()
        .filter(|event| matches!(event, StreamEvent::ContentBlockStart { .. }))
        .count();
    let ends = events
        .iter()
        .filter(|event| matches!(event, StreamEvent::ContentBlockEnd { .. }))
        .count();
    let completions: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::Ended { response } => Some(response),
            _ => None,
        })
        .collect();

    assert_eq!(starts, 2);
    assert_eq!(ends, 2);

    let [response] = completions.as_slice() else {
        return Err("expected exactly one completed event".into());
    };
    assert_eq!(response.id.as_deref(), Some("msg-1"));
    assert_eq!(response.raw, None);
    assert_eq!(response.usage.output, 7);

    let [
        ContentPart::Reasoning(reasoning),
        ContentPart::ToolCall(tool_call),
    ] = response.content.as_slice()
    else {
        return Err(format!("unexpected content {:?}", response.content).into());
    };
    assert_eq!(reasoning.text, "step");
    assert_eq!(reasoning.signature.as_deref(), Some("sig"));
    assert_eq!(tool_call.id, "toolu_1");
    assert_eq!(tool_call.name, "lookup");
    assert_eq!(tool_call.input.wire_value(), json!({ "q": "rust" }));
    assert_eq!(Some(tool_call.input.raw()), Some("{\"q\":\"rust\"}"));
    Ok(())
}

#[test]
fn streaming_keeps_redacted_thinking() -> Result<(), Box<dyn StdError>> {
    let events = stream(vec![
        sse(
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": { "type": "redacted_thinking", "data": "ENCRYPTED" }
            }),
        ),
        sse(
            "content_block_stop",
            &json!({ "type": "content_block_stop", "index": 0 }),
        ),
        sse("message_stop", &json!({ "type": "message_stop" })),
    ])?;

    // The sealed blob must not leak through live reasoning deltas.
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, StreamEvent::ReasoningDelta { .. })),
        "a redacted block leaked a reasoning delta: {events:?}"
    );
    let ended = events.iter().find_map(|event| match event {
        StreamEvent::ContentBlockEnd { part, .. } => Some(part),
        _ => None,
    });

    let Some(ContentPart::Reasoning(reasoning)) = ended else {
        return Err(format!("expected a reasoning block end, got {events:?}").into());
    };
    assert!(reasoning.redacted);
    assert_eq!(reasoning.text, "ENCRYPTED");
    Ok(())
}

#[test]
fn a_stream_error_event_ends_the_stream() -> Result<(), Box<dyn StdError>> {
    let request = Request::builder().model(MODEL).user("Hello").build()?;
    let call = resolved(request)?;
    let mut decoder = AnthropicMessagesCodec.stream_decoder(call.route());

    let error = decoder
        .decode(sse(
            "error",
            &json!({
                "type": "error",
                "error": { "type": "overloaded_error", "message": "Overloaded" }
            }),
        ))
        .expect_err("an error event fails the stream");

    assert_eq!(error.provider_code(), Some("overloaded_error"));
    assert!(error.message().contains("Overloaded"));
    Ok(())
}

#[test]
fn a_success_body_that_is_not_a_messages_response_fails_retryably() -> Result<(), Box<dyn StdError>>
{
    let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;

    // A gateway error page reserialized as a 200 is indistinguishable
    // from a garbled body, so the failure must stay retryable — the
    // classification the reference client gave it.
    let error = AnthropicMessagesCodec
        .decode_response(call.route(), json!({ "error": "gateway" }))
        .expect_err("a structureless 200 must fail to decode");

    assert_eq!(error.kind(), ErrorKind::ResponseDecode);
    assert_eq!(error.retry_classification(), RetryClassification::Safe);
    Ok(())
}

#[test]
fn a_stream_event_that_is_not_json_fails_retryably() -> Result<(), Box<dyn StdError>> {
    let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;
    let mut decoder = AnthropicMessagesCodec.stream_decoder(call.route());

    let error = decoder
        .decode(SseEvent {
            event: None,
            data:  "not json".to_owned(),
        })
        .expect_err("a garbled event must fail the stream");

    assert_eq!(error.kind(), ErrorKind::StreamDecode);
    assert_eq!(error.retry_classification(), RetryClassification::Safe);
    Ok(())
}

#[test]
fn an_input_fragment_for_an_unopened_block_fails_the_stream() -> Result<(), Box<dyn StdError>> {
    // Anthropic announces every tool call in a content_block_start
    // carrying its id and name. When that start is lost, assembling the
    // fragments would fabricate a nameless call whose replay poisons the
    // conversation, so the stream fails retryably instead — the contract
    // R2-24 set for the Chat codec and R3-10 for Bedrock.
    let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;
    let mut decoder = AnthropicMessagesCodec.stream_decoder(call.route());

    let error = decoder
        .decode(sse(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "input_json_delta", "partial_json": "{\"q\":\"rust\"}" },
            }),
        ))
        .err()
        .ok_or("expected the orphan input fragment to fail the stream")?;

    assert_eq!(error.kind(), ErrorKind::StreamDecode);
    assert_eq!(error.retry_classification(), RetryClassification::Safe);
    Ok(())
}

#[test]
fn a_refusal_response_fails_instead_of_decoding() -> Result<(), Box<dyn StdError>> {
    let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;
    let body = json!({
        "id": "msg-1",
        "content": [],
        "stop_reason": "refusal",
        "stop_details": { "explanation": "it asks for malware" }
    });

    let error = AnthropicMessagesCodec
        .decode_response(call.route(), body.clone())
        .expect_err("a refusal must fail the call");

    assert_eq!(error.kind(), ErrorKind::ContentFilter);
    assert_eq!(error.provider_code(), Some("refusal"));
    assert!(error.message().contains("it asks for malware"), "{error}");
    assert_eq!(error.raw_data(), Some(&body));
    Ok(())
}

#[test]
fn a_streamed_refusal_ends_the_stream_as_an_error() -> Result<(), Box<dyn StdError>> {
    let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;
    let mut decoder = AnthropicMessagesCodec.stream_decoder(call.route());

    decoder.decode(sse(
        "message_start",
        &json!({
            "type": "message_start",
            "message": { "id": "msg-1", "usage": { "input_tokens": 3 } }
        }),
    ))?;
    let error = decoder
        .decode(sse(
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": "refusal",
                    "stop_details": { "explanation": "it asks for malware" }
                }
            }),
        ))
        .expect_err("a refusal must fail the stream");

    assert_eq!(error.kind(), ErrorKind::ContentFilter);
    assert_eq!(error.provider_code(), Some("refusal"));
    assert!(error.message().contains("it asks for malware"), "{error}");
    Ok(())
}

/// A catalog whose model reasons but takes no effort levels, like
/// claude-sonnet-4-5.
const BUDGET_MODEL_CATALOG: &str = r#"
    schema_version = 1

    [providers.anthropic]
    display_name = "Anthropic"
    codecs = ["anthropic-messages"]
    base_url = "http://127.0.0.1"
    default_model = "claude-sonnet-4-5"
    auth = { type = "none" }

    [providers.anthropic.models."claude-sonnet-4-5"]
    display_name = "Budget Claude"
    api_model = "claude-sonnet-4-5"
    capabilities = { text = true, tools = true, reasoning = true, tool_choice = { required = true, named = true } }
"#;

/// The budget-model catalog with passthrough allowed.
const PASSTHROUGH_CATALOG: &str = r#"
    schema_version = 1

    [providers.anthropic]
    display_name = "Anthropic"
    codecs = ["anthropic-messages"]
    base_url = "http://127.0.0.1"
    allow_passthrough = true
    default_model = "claude-sonnet-4-5"
    auth = { type = "none" }

    [providers.anthropic.models."claude-sonnet-4-5"]
    display_name = "Budget Claude"
    api_model = "claude-sonnet-4-5"
    capabilities = { text = true, tools = true, reasoning = true, tool_choice = { required = true, named = true } }
"#;

#[test]
fn a_passthrough_model_takes_the_effort_dialect() -> Result<(), Box<dyn StdError>> {
    // Passthrough serves models newer than the catalog, and those reject
    // a manual thinking toggle — so the uncataloged guess is the modern
    // dialect, as the reference client guessed. The adaptive thinking
    // object stays gated on the declared capability, so nothing else
    // about the request changes.
    let call = resolved_in(
        PASSTHROUGH_CATALOG,
        Request::builder()
            .model("anthropic/claude-next")
            .user("Hello")
            .reasoning_effort(ReasoningEffort::High)
            .build()?,
    )?;

    let encoded = AnthropicMessagesCodec.encode(&call, false)?;

    assert_eq!(encoded.body["output_config"]["effort"], json!("high"));
    assert_eq!(encoded.body.get("thinking"), None);
    Ok(())
}

#[test]
fn a_model_without_effort_levels_takes_a_thinking_budget() -> Result<(), Box<dyn StdError>> {
    let call = resolved_in(
        BUDGET_MODEL_CATALOG,
        Request::builder()
            .model("anthropic/claude-sonnet-4-5")
            .user("Hello")
            .reasoning_effort(ReasoningEffort::High)
            .max_output_tokens(8000)
            .build()?,
    )?;
    let codec = AnthropicMessagesCodec;

    let encoded = codec.encode(&call, false)?;

    assert_eq!(
        encoded.body["thinking"],
        json!({ "type": "enabled", "budget_tokens": 6000 })
    );
    assert_eq!(encoded.body["max_tokens"], 8000);
    // Sending `effort` too would ask for a control the model does not
    // take.
    assert_eq!(encoded.body.get("output_config"), None);

    // The count body keeps the thinking budget: counting must see the
    // same request generation sends.
    let counted = codec
        .encode_count_tokens(&call)
        .ok_or("Anthropic should count tokens")??;
    assert_eq!(
        counted.body["thinking"],
        json!({ "type": "enabled", "budget_tokens": 6000 })
    );
    assert_eq!(counted.body.get("max_tokens"), None);
    Ok(())
}

#[test]
fn a_raw_thinking_option_replaces_the_derived_budget() -> Result<(), Box<dyn StdError>> {
    // Merging the raw object into the derived budget would leave a stray
    // `budget_tokens` beside `"type": "disabled"`, which the API rejects,
    // and would keep an output limit lifted for a budget nobody sends.
    let call = resolved_in(
        BUDGET_MODEL_CATALOG,
        Request::builder()
            .model("anthropic/claude-sonnet-4-5")
            .user("Hello")
            .reasoning_effort(ReasoningEffort::Max)
            .max_output_tokens(2048)
            .provider_option("anthropic", "thinking", json!({ "type": "disabled" }))
            .build()?,
    )?;
    let codec = AnthropicMessagesCodec;

    let encoded = codec.encode(&call, false)?;
    assert_eq!(encoded.body["thinking"], json!({ "type": "disabled" }));
    assert_eq!(encoded.body["max_tokens"], 2048);

    // The count body must carry the same clean object generation sends.
    let counted = codec
        .encode_count_tokens(&call)
        .ok_or("Anthropic should count tokens")??;
    assert_eq!(counted.body["thinking"], json!({ "type": "disabled" }));
    Ok(())
}

#[test]
fn a_thinking_budget_keeps_its_floor_and_fits_under_the_limit() -> Result<(), Box<dyn StdError>> {
    let codec = AnthropicMessagesCodec;

    // A quarter of 1200 is under the provider floor; the floor wins and
    // still fits under the limit.
    let floored = codec.encode(
        &resolved_in(
            BUDGET_MODEL_CATALOG,
            Request::builder()
                .model("anthropic/claude-sonnet-4-5")
                .user("Hello")
                .reasoning_effort(ReasoningEffort::Low)
                .max_output_tokens(1200)
                .build()?,
        )?,
        false,
    )?;
    assert_eq!(floored.body["thinking"]["budget_tokens"], 1024);
    assert_eq!(floored.body["max_tokens"], 1200);

    // Max effort budgets the whole limit, so the limit grows to keep the
    // budget strictly below it.
    let lifted = codec.encode(
        &resolved_in(
            BUDGET_MODEL_CATALOG,
            Request::builder()
                .model("anthropic/claude-sonnet-4-5")
                .user("Hello")
                .reasoning_effort(ReasoningEffort::Max)
                .max_output_tokens(2048)
                .build()?,
        )?,
        false,
    )?;
    assert_eq!(lifted.body["thinking"]["budget_tokens"], 2048);
    assert_eq!(lifted.body["max_tokens"], 3072);
    Ok(())
}

#[test]
fn count_tokens_narrows_the_body_and_drops_max_tokens() -> Result<(), Box<dyn StdError>> {
    let call = resolved(
        Request::builder()
            .model(MODEL)
            .system("You are terse.")
            .user("Hello")
            .max_output_tokens(256)
            .temperature(0.5)
            .top_p(0.9)
            .stop_sequence("END")
            .metadata_entry("user_id", "u-1")
            .speed(Speed::Fast)
            .reasoning_effort(ReasoningEffort::High)
            .tool(ToolDefinition::function(
                "lookup",
                "Look something up",
                json!({ "type": "object" }),
            ))
            .tool_choice(ToolChoice::Auto)
            .provider_option("anthropic", "thinking", json!({ "type": "enabled" }))
            .build()?,
    )?;
    let codec = AnthropicMessagesCodec;

    let encoded = codec
        .encode_count_tokens(&call)
        .ok_or("Anthropic has a count-tokens endpoint")??;

    let body = encoded
        .body
        .as_object()
        .ok_or("the count body must be an object")?;
    let mut keys: Vec<&str> = body.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(keys, [
        "messages",
        "model",
        "system",
        "thinking",
        "tool_choice",
        "tools"
    ]);
    assert!(encoded.url.ends_with("/v1/messages/count_tokens"));
    assert!(
        encoded
            .headers
            .contains(&("anthropic-version".to_owned(), "2023-06-01".to_owned()))
    );

    let tokens = codec.decode_count_tokens(call.route(), json!({ "input_tokens": 123 }))?;
    assert_eq!(tokens, 123);
    Ok(())
}

#[test]
fn a_tool_result_image_becomes_a_block_and_does_not_warn() -> Result<(), Box<dyn StdError>> {
    // `tool_result.content` takes an array of blocks here, so an image a
    // tool produced survives instead of being flattened away.
    let request = Request::builder()
        .model(MODEL)
        .user("Chart it.")
        .message(Message::new(Role::Tool, [ContentPart::ToolResult(
            ToolResult {
                tool_call_id: "call-1".to_owned(),
                name:         Some("chart".to_owned()),
                content:      vec![
                    ContentPart::Text {
                        text: "Revenue by quarter.".to_owned(),
                    },
                    ContentPart::Image(ImageContent::new(MediaSource::base64("aW1n", "image/png"))),
                ],
                is_error:     false,
            },
        )]))
        .build()?;

    let encoded = AnthropicMessagesCodec.encode(&resolved(request)?, false)?;

    let blocks = &encoded.body["messages"][0]["content"][1]["content"];
    assert_eq!(blocks[0]["type"], "text");
    assert_eq!(blocks[1]["type"], "image");
    assert_eq!(blocks[1]["source"]["data"], "aW1n");
    assert!(
        encoded.warnings.is_empty(),
        "content this codec carries must not warn"
    );
    Ok(())
}
