use std::error::Error as StdError;

use serde_json::{Value, json};

use super::{COUNT_TOKENS_FIELDS, Codec, MESSAGE_KIND, OpenAiResponsesCodec, REASONING_KIND};
use crate::adapter::ResolvedCall;
use crate::codecs::test_support::{resolved, resolved_in};
use crate::transport::SseEvent;
use crate::types::{
    ContentBlockId, ContentPart, ErrorKind, FinishReason, ImageContent, MediaSource, Message,
    ReasoningContent, Request, Response, RetryClassification, Role, Speed, StreamEvent,
    ToolArguments, ToolCall, ToolCallKind, ToolDefinition, ToolInput, ToolResult,
};

const MODEL: &str = "openai/gpt-5.6-luna";

/// The public deployment's codec.
fn codec() -> OpenAiResponsesCodec {
    OpenAiResponsesCodec::new(false)
}

fn call(request: Request) -> Result<ResolvedCall, Box<dyn StdError>> {
    resolved(request)
}

fn sse(data: &Value) -> SseEvent {
    SseEvent {
        event: None,
        data:  data.to_string(),
    }
}

/// Every content part carried by a block-end event, in order.
fn ended_parts(events: &[StreamEvent]) -> Vec<ContentPart> {
    events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ContentBlockEnd { part, .. } => Some(part.clone()),
            _ => None,
        })
        .collect()
}

/// Every reasoning delta of a stream, in order.
fn reasoning_deltas(events: &[StreamEvent]) -> Vec<&str> {
    events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ReasoningDelta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

/// The single completed response of a stream.
fn completed(events: &[StreamEvent]) -> Result<Response, Box<dyn StdError>> {
    let mut responses = events.iter().filter_map(|event| match event {
        StreamEvent::Ended { response } => Some(response.clone()),
        _ => None,
    });
    let response = responses
        .next()
        .ok_or("the stream emitted no completed event")?;
    if responses.next().is_some() {
        return Err("the stream emitted more than one completed event".into());
    }
    Ok(*response)
}

/// Checks one start, then deltas, then one end for every block.
fn assert_block_boundaries(events: &[StreamEvent]) -> Result<(), Box<dyn StdError>> {
    let mut open: Vec<&ContentBlockId> = Vec::new();
    let mut closed: Vec<&ContentBlockId> = Vec::new();

    for event in events {
        match event {
            StreamEvent::ContentBlockStart { id, .. } => {
                if open.contains(&id) || closed.contains(&id) {
                    return Err(format!("{id:?} started twice").into());
                }
                open.push(id);
            }
            StreamEvent::TextDelta { id, .. }
            | StreamEvent::ReasoningDelta { id, .. }
            | StreamEvent::ToolCallDelta { id, .. } => {
                if !open.contains(&id) {
                    return Err(format!("{id:?} sent a delta before its start").into());
                }
            }
            StreamEvent::ContentBlockEnd { id, .. } => {
                if !open.contains(&id) {
                    return Err(format!("{id:?} ended without a start").into());
                }
                open.retain(|open_id| *open_id != id);
                closed.push(id);
            }
            StreamEvent::Started { .. }
            | StreamEvent::Usage { .. }
            | StreamEvent::RateLimits { .. }
            | StreamEvent::Ended { .. } => {}
        }
    }

    if !open.is_empty() {
        return Err(format!("blocks left open: {open:?}").into());
    }
    Ok(())
}

#[test]
fn tool_calls_and_results_keep_their_protocol_identity() -> Result<(), Box<dyn StdError>> {
    let mut call_part = ToolCall::function("call_abc", "search", json!({ "query": "rust" }));
    call_part.input =
        ToolInput::Function(ToolArguments::from_raw("{\"query\":\"rust\"}".to_owned()));
    call_part
        .provider_metadata
        .insert("openai".to_owned(), json!({ "item_id": "fc_123" }));
    let request = Request::builder()
        .model(MODEL)
        .user("Search for rust")
        .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
            call_part,
        )]))
        .message(Message::new(Role::Tool, [ContentPart::ToolResult(
            ToolResult {
                tool_call_id: "call_abc".to_owned(),
                name:         Some("search".to_owned()),
                content:      vec![ContentPart::Text {
                    text: "2 matches".to_owned(),
                }],
                is_error:     false,
            },
        )]))
        .build()?;
    let codec = codec();

    let encoded = codec.encode(&call(request)?, false)?;

    assert_eq!(
        encoded.body["input"][1],
        json!({
            "type": "function_call",
            "id": "fc_123",
            "call_id": "call_abc",
            "name": "search",
            "arguments": "{\"query\":\"rust\"}",
        })
    );
    assert_eq!(
        encoded.body["input"][2],
        json!({
            "type": "function_call_output",
            "call_id": "call_abc",
            "output": "2 matches",
        })
    );

    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();
    let response = codec.decode_response(
        &route,
        json!({
            "id": "resp_1",
            "status": "completed",
            "output": [{
                "type": "function_call",
                "id": "fc_123",
                "call_id": "call_abc",
                "name": "search",
                "arguments": "{\"query\":\"rust\"}",
            }],
        }),
    )?;

    let [ContentPart::ToolCall(decoded)] = response.content.as_slice() else {
        return Err("expected one tool call".into());
    };
    assert_eq!(decoded.id, "call_abc");
    assert_eq!(decoded.input.wire_value(), json!({ "query": "rust" }));
    assert_eq!(Some(decoded.input.raw()), Some("{\"query\":\"rust\"}"));
    assert_eq!(
        decoded.provider_metadata.get("openai"),
        Some(&json!({ "item_id": "fc_123" }))
    );
    Ok(())
}

#[test]
fn raw_options_win_and_foreign_namespaces_are_ignored() -> Result<(), Box<dyn StdError>> {
    let request = Request::builder()
        .model(MODEL)
        .user("Hello")
        .temperature(0.2)
        .provider_option("openai", "temperature", json!(0.9))
        .provider_option("openai", "auto_cache", json!(false))
        .provider_option("anthropic", "top_k", json!(40))
        .build()?;

    let encoded = codec().encode(&call(request)?, false)?;

    assert_eq!(encoded.body["temperature"], json!(0.9));
    let body = encoded.body.as_object().ok_or("expected a JSON object")?;
    assert!(!body.contains_key("auto_cache"));
    assert!(!body.contains_key("top_k"));
    Ok(())
}

#[test]
fn stop_sequences_are_dropped_with_a_warning() -> Result<(), Box<dyn StdError>> {
    let request = Request::builder()
        .model(MODEL)
        .user("Hello")
        .stop_sequences(["END", "STOP"])
        .metadata_entry("trace_id", "t789")
        .build()?;

    let encoded = codec().encode(&call(request)?, false)?;

    // The live API rejects a `stop` member outright (2026-08-30), so the
    // sequences never reach the wire and the caller is warned instead.
    let body = encoded.body.as_object().ok_or("expected a JSON object")?;
    assert!(!body.contains_key("stop"));
    assert_eq!(encoded.body["metadata"], json!({ "trace_id": "t789" }));
    assert_eq!(
        encoded.warnings.len(),
        1,
        "dropping the stop sequences must warn: {:?}",
        encoded.warnings,
    );
    Ok(())
}

#[test]
fn each_speed_selects_its_service_tier() -> Result<(), Box<dyn StdError>> {
    for (speed, tier) in [
        (Speed::Fast, "priority"),
        (Speed::Balanced, "auto"),
        (Speed::Economical, "flex"),
    ] {
        let request = Request::builder()
            .model(MODEL)
            .user("Hello")
            .speed(speed)
            .build()?;

        let encoded = codec().encode(&call(request)?, false)?;

        assert_eq!(encoded.body["service_tier"], json!(tier));
    }
    Ok(())
}

#[test]
fn codex_mode_hoists_instructions_and_drops_sampling_controls() -> Result<(), Box<dyn StdError>> {
    let request = Request::builder()
        .model(MODEL)
        .system("Be brief")
        .user("Hello")
        .temperature(0.2)
        .top_p(0.5)
        .max_output_tokens(256)
        .build()?;

    let encoded = OpenAiResponsesCodec::new(true).encode(&call(request)?, true)?;

    assert_eq!(encoded.body["instructions"], json!("Be brief"));
    assert_eq!(
        encoded.body["input"],
        json!([{
            "role": "user",
            "content": [{ "type": "input_text", "text": "Hello" }],
        }])
    );
    assert_eq!(encoded.body["stream"], json!(true));
    let body = encoded.body.as_object().ok_or("expected a JSON object")?;
    assert!(!body.contains_key("temperature"));
    assert!(!body.contains_key("top_p"));
    assert!(!body.contains_key("max_output_tokens"));
    assert_eq!(encoded.warnings.len(), 3);
    // The deployment hangs its endpoint off the base path with no
    // version segment; `/v1/responses` there is an HTML 403.
    assert!(
        encoded.url.ends_with("/responses") && !encoded.url.contains("/v1/"),
        "codex mode must post to the unversioned path: {}",
        encoded.url
    );
    Ok(())
}

#[test]
fn codex_mode_reports_no_native_token_count() -> Result<(), Box<dyn StdError>> {
    let request = Request::builder().model(MODEL).user("Hello").build()?;
    // The Codex deployment serves no `/responses/input_tokens` (an HTML
    // 403 on 2026-08-30), so codex mode must say "no native count"
    // rather than build a request that cannot succeed.
    assert!(
        OpenAiResponsesCodec::new(true)
            .encode_count_tokens(&call(request)?)
            .is_none()
    );
    Ok(())
}

#[test]
fn inclusive_usage_becomes_disjoint_buckets() -> Result<(), Box<dyn StdError>> {
    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();

    let response = codec().decode_response(
        &route,
        json!({
            "id": "resp_1",
            "status": "completed",
            "output": [],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "input_tokens_details": { "cached_tokens": 80, "cache_write_tokens": 15 },
                "output_tokens_details": { "reasoning_tokens": 20 },
            },
        }),
    )?;

    assert_eq!(response.usage.input, 5);
    assert_eq!(response.usage.output, 30);
    assert_eq!(response.usage.reasoning, 20);
    assert_eq!(response.usage.cache_read, 80);
    assert_eq!(response.usage.cache_write, 15);
    assert_eq!(response.usage.total(), 150);
    assert_eq!(response.cost, None);
    Ok(())
}

#[test]
fn encrypted_reasoning_is_requested_without_the_catalog_flag() -> Result<(), Box<dyn StdError>> {
    // An overlay entry that forgets `reasoning = true` on a reasoning
    // model must not silently lose the encrypted content its next turn
    // needs; a model without reasoning ignores the include.
    let call = resolved_in(
        r#"
        schema_version = 1

        [providers.openai]
        display_name = "OpenAI"
        codecs = ["openai-responses"]
        base_url = "https://api.openai.com"
        default_model = "overlay"
        auth = { type = "bearer" }

        [providers.openai.models.overlay]
        display_name = "Overlay"
        api_model = "overlay-v1"
        capabilities = { text = true }
        "#,
        Request::builder()
            .model("openai/overlay")
            .user("hi")
            .build()?,
    )?;

    let encoded = codec().encode(&call, false)?;

    assert_eq!(
        encoded.body["include"],
        json!(["reasoning.encrypted_content"])
    );
    Ok(())
}

#[test]
fn a_routing_model_sends_the_cache_fingerprint() -> Result<(), Box<dyn StdError>> {
    let call = resolved_in(
        r#"
        schema_version = 1

        [providers.openai]
        display_name = "OpenAI"
        codecs = ["openai-responses"]
        base_url = "https://api.openai.com"
        default_model = "routed"
        auth = { type = "bearer" }

        [providers.openai.models.routed]
        display_name = "Routed"
        api_model = "routed-v1"
        capabilities = { text = true, caching = true, cache_routing = true }
        "#,
        Request::builder()
            .model("openai/routed")
            .system("Keep it short.")
            .user("hi")
            .build()?,
    )?;

    let encoded = codec().encode(&call, false)?;

    let key = encoded.body["prompt_cache_key"]
        .as_str()
        .ok_or("no prompt_cache_key was sent")?;
    assert!(key.starts_with("lithos-"), "unexpected key shape: {key}");
    Ok(())
}

#[test]
fn a_summary_only_reasoning_item_is_kept_for_replay() -> Result<(), Box<dyn StdError>> {
    // Function calls must replay behind their reasoning item, and only
    // the original item with its id satisfies the pairing — even when it
    // carries a visible summary and no encrypted payload. With no
    // `content`, the summary blocks stand in as the reasoning text,
    // joined by the blank line a consumer puts between them when it
    // reads the summary off the opaque item.
    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();
    let item = json!({
        "type": "reasoning",
        "id": "rs_1",
        "summary": [
            { "type": "summary_text", "text": "Checked the files." },
            { "type": "summary_text", "text": "Nothing to change." },
        ],
    });

    let response = codec().decode_response(
        &route,
        json!({ "id": "resp_1", "status": "completed", "output": [item.clone()] }),
    )?;

    assert_eq!(response.content, vec![
        ContentPart::Reasoning(ReasoningContent {
            text:             "Checked the files.\n\nNothing to change.".to_owned(),
            signature:        None,
            signature_origin: None,
            redacted:         false,
        }),
        ContentPart::opaque(REASONING_KIND, item),
    ]);
    Ok(())
}

#[test]
fn a_body_without_an_output_array_fails_to_decode() -> Result<(), Box<dyn StdError>> {
    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();

    // An empty object and an error-shaped 200 both lack the `output`
    // array every Responses document carries; neither may decode as a
    // successful empty response.
    for body in [json!({}), json!({ "error": { "message": "boom" } })] {
        let error = codec()
            .decode_response(&route, body)
            .err()
            .ok_or("expected a body without output to fail")?;
        assert_eq!(error.kind(), ErrorKind::ResponseDecode);
        assert_eq!(error.retry_classification(), RetryClassification::Safe);
    }
    Ok(())
}

#[test]
fn an_unknown_response_status_keeps_its_spelling() -> Result<(), Box<dyn StdError>> {
    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();

    let response = codec().decode_response(
        &route,
        json!({ "id": "resp_1", "status": "cancelled", "output": [] }),
    )?;

    // The old decoder preserved unknown statuses; collapsing `cancelled`
    // into `Stop` would report an answer the model never finished.
    assert_eq!(
        response.finish_reason,
        FinishReason::Other("cancelled".to_owned())
    );
    Ok(())
}

#[test]
fn the_raw_success_document_is_preserved() -> Result<(), Box<dyn StdError>> {
    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();
    let document = json!({
        "id": "resp_1",
        "status": "completed",
        "service_tier": "default",
        "output": [{
            "type": "message",
            "id": "msg_1",
            "content": [{ "type": "output_text", "text": "hello" }],
        }],
    });

    let response = codec().decode_response(&route, document.clone())?;

    assert_eq!(response.raw, Some(document));
    assert_eq!(response.text(), "hello");
    Ok(())
}

#[test]
fn custom_tools_round_trip() -> Result<(), Box<dyn StdError>> {
    let request = Request::builder()
        .model(MODEL)
        .tool(ToolDefinition::custom(
            "apply_patch",
            "Edits files",
            json!({ "type": "grammar", "syntax": "lark" }),
        ))
        .user("Patch the file")
        .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
            ToolCall::custom("call_001", "apply_patch", "*** Begin Patch"),
        )]))
        .message(Message::new(Role::Tool, [ContentPart::ToolResult(
            ToolResult {
                tool_call_id: "call_001".to_owned(),
                name:         Some("apply_patch".to_owned()),
                content:      vec![ContentPart::Text {
                    text: "Success".to_owned(),
                }],
                is_error:     false,
            },
        )]))
        .build()?;
    let codec = codec();
    let call = call(request)?;

    let encoded = codec.encode(&call, false)?;

    assert_eq!(
        encoded.body["tools"][0],
        json!({
            "type": "custom",
            "name": "apply_patch",
            "description": "Edits files",
            "format": { "type": "grammar", "syntax": "lark" },
        })
    );
    assert_eq!(
        encoded.body["input"][1],
        json!({
            "type": "custom_tool_call",
            "call_id": "call_001",
            "name": "apply_patch",
            "input": "*** Begin Patch",
        })
    );
    assert_eq!(
        encoded.body["input"][2],
        json!({
            "type": "custom_tool_call_output",
            "call_id": "call_001",
            "output": "Success",
        })
    );

    let response = codec.decode_response(
        call.route(),
        json!({
            "id": "resp_1",
            "status": "completed",
            "output": [{
                "type": "custom_tool_call",
                "id": "ctc_def456",
                "call_id": "call_001",
                "name": "apply_patch",
                "input": "*** Begin Patch",
            }],
        }),
    )?;

    let [ContentPart::ToolCall(decoded)] = response.content.as_slice() else {
        return Err("expected one tool call".into());
    };
    assert_eq!(decoded.input.kind(), ToolCallKind::Custom);
    assert_eq!(
        decoded.input.wire_value(),
        Value::String("*** Begin Patch".to_owned())
    );
    assert_eq!(Some(decoded.input.raw()), Some("*** Begin Patch"));
    Ok(())
}

#[test]
fn opaque_reasoning_replays_and_other_namespaces_are_skipped() -> Result<(), Box<dyn StdError>> {
    let item = json!({
        "type": "reasoning",
        "id": "rs_1",
        "summary": [],
        "encrypted_content": "gAAAA",
    });
    let request = Request::builder()
        .model(MODEL)
        .user("Hello")
        .message(Message::new(Role::Assistant, [
            ContentPart::opaque("openai.reasoning", item.clone()),
            ContentPart::opaque("anthropic.thinking", json!({ "signature": "sig" })),
            ContentPart::Reasoning(ReasoningContent {
                text:             "step one".to_owned(),
                signature:        None,
                signature_origin: None,
                redacted:         false,
            }),
            ContentPart::Text {
                text: "hello".to_owned(),
            },
        ]))
        .build()?;

    let encoded = codec().encode(&call(request)?, false)?;

    assert_eq!(encoded.body["input"][1], item);
    assert_eq!(
        encoded.body["input"][2],
        json!({
            "role": "assistant",
            "content": [{ "type": "output_text", "text": "hello" }],
        })
    );
    assert_eq!(
        encoded.body["input"].as_array().map(Vec::len),
        Some(3),
        "an opaque part for another provider must not reach the wire",
    );
    Ok(())
}

#[test]
fn noncanonical_replay_kinds_and_metadata_are_ignored() -> Result<(), Box<dyn StdError>> {
    let mut tool = ToolCall::function("call_1", "search", json!({}));
    tool.provider_metadata
        .insert("id".to_owned(), json!("fc_old"));
    let request = Request::builder()
        .model(MODEL)
        .user("hi")
        .message(Message::new(Role::Assistant, [
            ContentPart::opaque(
                "openai_reasoning",
                json!({"type":"reasoning","id":"rs_old"}),
            ),
            ContentPart::opaque("openai_message", json!({"type":"message","id":"msg_old"})),
            ContentPart::ToolCall(tool),
        ]))
        .build()?;
    let encoded = codec().encode(&call(request)?, false)?;
    assert_eq!(encoded.body["input"].as_array().map(Vec::len), Some(2));
    assert!(encoded.body["input"][1].get("id").is_none());
    Ok(())
}

#[test]
fn a_reasoning_item_decodes_to_visible_text_and_a_replay_part() -> Result<(), Box<dyn StdError>> {
    // When the item carries `content`, the reasoning part is that trace
    // alone; the summary stays on the opaque item.
    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();

    let response = codec().decode_response(
        &route,
        json!({
            "id": "resp_1",
            "status": "completed",
            "output": [{
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{ "type": "summary_text", "text": "Checked the files." }],
                "content": [{ "type": "reasoning_text", "text": "checked" }],
                "encrypted_content": "gAAAA",
            }],
        }),
    )?;

    let [ContentPart::Reasoning(reasoning), opaque] = response.content.as_slice() else {
        return Err("expected reasoning text and its replay part".into());
    };
    assert_eq!(reasoning.text, "checked");
    assert_eq!(opaque.opaque_namespace(), Some("openai"));
    Ok(())
}

#[test]
fn reasoning_content_entries_join_with_a_blank_line() -> Result<(), Box<dyn StdError>> {
    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();

    let response = codec().decode_response(
        &route,
        json!({
            "id": "resp_1",
            "status": "completed",
            "output": [{
                "type": "reasoning",
                "id": "rs_1",
                "summary": [],
                "content": [
                    { "type": "reasoning_text", "text": "First." },
                    { "type": "reasoning_text", "text": "Second." },
                ],
            }],
        }),
    )?;

    let [ContentPart::Reasoning(reasoning), _opaque] = response.content.as_slice() else {
        return Err("expected reasoning text and its replay part".into());
    };
    assert_eq!(reasoning.text, "First.\n\nSecond.");
    Ok(())
}

#[test]
fn an_entry_of_an_unknown_type_is_not_reasoning_text() -> Result<(), Box<dyn StdError>> {
    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();
    let decode = |item: Value| -> Result<Vec<ContentPart>, Box<dyn StdError>> {
        Ok(codec()
            .decode_response(
                &route,
                json!({ "id": "resp_1", "status": "completed", "output": [item] }),
            )?
            .content)
    };

    // A text-bearing `content` entry of another type is skipped.
    let parts = decode(json!({
        "type": "reasoning",
        "id": "rs_1",
        "summary": [],
        "content": [
            { "type": "reasoning_text", "text": "checked" },
            { "type": "reasoning_note", "text": "hidden" },
        ],
    }))?;
    let [ContentPart::Reasoning(reasoning), _opaque] = parts.as_slice() else {
        return Err("expected reasoning text and its replay part".into());
    };
    assert_eq!(reasoning.text, "checked");

    // `content` made only of such entries carries no trace, so the
    // summary stands in, and a summary block of another type is skipped
    // the same way.
    let parts = decode(json!({
        "type": "reasoning",
        "id": "rs_1",
        "summary": [
            { "type": "summary_text", "text": "Checked the files." },
            { "type": "summary_note", "text": "hidden" },
        ],
        "content": [{ "type": "reasoning_note", "text": "hidden" }],
    }))?;
    let [ContentPart::Reasoning(reasoning), _opaque] = parts.as_slice() else {
        return Err("expected reasoning text and its replay part".into());
    };
    assert_eq!(reasoning.text, "Checked the files.");
    Ok(())
}

#[test]
fn a_lost_item_announcement_recovers_the_call_identity() -> Result<(), Box<dyn StdError>> {
    // The argument fragments arrive before any `output_item.added`, so
    // they latch the assembler's fallback block — call id equal to the
    // item id, no name. The terminal item event carries the real
    // identity, and the assembled part must not keep the fallback's.
    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();
    let mut decoder = codec().stream_decoder(&route);
    let transcript = vec![
        json!({ "type": "response.created", "response": { "id": "resp_1" } }),
        json!({
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_123",
            "delta": "{\"query\":",
        }),
        json!({
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_123",
            "delta": "\"rust\"}",
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc_123",
                "call_id": "call_abc",
                "name": "search",
                "arguments": "{\"query\":\"rust\"}",
            },
        }),
    ];

    let mut events = Vec::new();
    for event in transcript {
        events.extend(decoder.decode(sse(&event))?);
    }
    events.extend(decoder.finish()?);

    let parts = ended_parts(&events);
    let [ContentPart::ToolCall(recovered)] = parts.as_slice() else {
        return Err(format!("expected one tool call part, got {parts:?}").into());
    };
    assert_eq!(recovered.id, "call_abc");
    assert_eq!(recovered.name, "search");
    assert_eq!(recovered.input.wire_value(), json!({ "query": "rust" }));
    Ok(())
}

#[test]
fn a_lost_added_for_an_internal_call_leaves_no_phantom_part() -> Result<(), Box<dyn StdError>> {
    // Argument deltas for an unannounced item latch the fallback block,
    // and the terminal item then reveals a model-internal call with no
    // name. Blocking decode drops the item, and the stream must match:
    // the latched block closes for consumers that saw it open, but no
    // nameless ToolCall joins the content and the finish reason stays
    // the document's.
    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();
    let mut decoder = codec().stream_decoder(&route);
    let transcript = vec![
        json!({ "type": "response.created", "response": { "id": "resp_1" } }),
        json!({
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_9",
            "delta": "{\"q\":1}",
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": { "type": "function_call", "id": "fc_9", "arguments": "{\"q\":1}" },
        }),
        json!({
            "type": "response.completed",
            "response": { "id": "resp_1", "status": "completed", "output": [] },
        }),
    ];

    let mut events = Vec::new();
    for event in transcript {
        events.extend(decoder.decode(sse(&event))?);
    }
    events.extend(decoder.finish()?);

    assert_block_boundaries(&events)?;
    let response = completed(&events)?;
    assert_eq!(response.content, Vec::new());
    assert_eq!(response.finish_reason, FinishReason::Stop);
    Ok(())
}

#[test]
fn a_lost_argument_delta_is_healed_by_the_terminal_item() -> Result<(), Box<dyn StdError>> {
    // One argument fragment never arrives. The terminal item carries the
    // complete arguments, so the assembled call must not keep the
    // truncation — and the missing tail goes out as an ordinary delta so
    // a consumer concatenating fragments stays correct too.
    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();
    let mut decoder = codec().stream_decoder(&route);
    let transcript = vec![
        json!({ "type": "response.created", "response": { "id": "resp_1" } }),
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc_123",
                "call_id": "call_abc",
                "name": "search",
            },
        }),
        json!({
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_123",
            "delta": "{\"query\":",
        }),
        // The fragment carrying "\"rust\"}" is lost in transit.
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc_123",
                "call_id": "call_abc",
                "name": "search",
                "arguments": "{\"query\":\"rust\"}",
            },
        }),
    ];

    let mut events = Vec::new();
    for event in transcript {
        events.extend(decoder.decode(sse(&event))?);
    }
    events.extend(decoder.finish()?);

    let tails: Vec<&str> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ToolCallDelta { arguments, .. } => Some(arguments.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(tails, ["{\"query\":", "\"rust\"}"]);
    let parts = ended_parts(&events);
    let [ContentPart::ToolCall(healed)] = parts.as_slice() else {
        return Err(format!("expected one tool call part, got {parts:?}").into());
    };
    assert_eq!(healed.input.wire_value(), json!({ "query": "rust" }));
    assert_eq!(Some(healed.input.raw()), Some("{\"query\":\"rust\"}"));
    Ok(())
}

#[test]
fn a_garbled_argument_buffer_is_replaced_by_the_terminal_item() -> Result<(), Box<dyn StdError>> {
    // The streamed fragments disagree with the terminal item — not a
    // prefix, so something was mangled in transit. The terminal item is
    // the ground truth and replaces the buffer outright.
    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();
    let mut decoder = codec().stream_decoder(&route);
    let transcript = vec![
        json!({ "type": "response.created", "response": { "id": "resp_1" } }),
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc_123",
                "call_id": "call_abc",
                "name": "search",
            },
        }),
        json!({
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_123",
            "delta": "\"rust\"}",
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc_123",
                "call_id": "call_abc",
                "name": "search",
                "arguments": "{\"query\":\"rust\"}",
            },
        }),
    ];

    let mut events = Vec::new();
    for event in transcript {
        events.extend(decoder.decode(sse(&event))?);
    }
    events.extend(decoder.finish()?);

    let parts = ended_parts(&events);
    let [ContentPart::ToolCall(replaced)] = parts.as_slice() else {
        return Err(format!("expected one tool call part, got {parts:?}").into());
    };
    assert_eq!(replaced.input.wire_value(), json!({ "query": "rust" }));
    assert_eq!(Some(replaced.input.raw()), Some("{\"query\":\"rust\"}"));
    Ok(())
}

#[test]
fn a_text_only_tool_message_answering_a_custom_call_routes_as_custom()
-> Result<(), Box<dyn StdError>> {
    // The custom call rides earlier in the request; the answering tool
    // message carries only text and a call id — no name, and the tool is
    // not redeclared. A function_call_output against a custom_tool_call
    // is rejected by the provider, so the call-id route must apply here
    // exactly as it does on the ToolResult path.
    let request = Request::builder()
        .model(MODEL)
        .user("Patch the file")
        .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
            ToolCall::custom("call_001", "apply_patch", "*** Begin Patch"),
        )]))
        .message(
            Message::new(Role::Tool, [ContentPart::Text {
                text: "Success".to_owned(),
            }])
            .with_tool_call_id("call_001"),
        )
        .build()?;

    let encoded = codec().encode(&call(request)?, false)?;

    assert_eq!(
        encoded.body["input"][2],
        json!({
            "type": "custom_tool_call_output",
            "call_id": "call_001",
            "output": "Success",
        })
    );
    Ok(())
}

#[test]
fn a_streamed_message_without_text_emits_no_empty_text_part() -> Result<(), Box<dyn StdError>> {
    // An empty assistant message streams no output_text. Blocking decode
    // of the same body pushes no text part, and the streamed response
    // must match: only the opaque replay item, no Text("") and no stray
    // start/end pair.
    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();
    let mut decoder = codec().stream_decoder(&route);
    let item = json!({
        "type": "message",
        "id": "msg_1",
        "status": "completed",
        "role": "assistant",
        "content": [],
    });
    let transcript = vec![
        json!({ "type": "response.created", "response": { "id": "resp_1" } }),
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": { "type": "message", "id": "msg_1", "role": "assistant", "content": [] },
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": item,
        }),
    ];

    let mut events = Vec::new();
    for event in transcript {
        events.extend(decoder.decode(sse(&event))?);
    }
    events.extend(decoder.finish()?);

    assert_block_boundaries(&events)?;
    let response = completed(&events)?;
    let [ContentPart::Opaque { kind, data }] = response.content.as_slice() else {
        return Err(format!("expected only the opaque item, got {:?}", response.content).into());
    };
    assert_eq!(kind, MESSAGE_KIND);
    assert_eq!(data, &item);
    Ok(())
}

#[test]
fn a_refusal_part_fails_instead_of_decoding() -> Result<(), Box<dyn StdError>> {
    // The structured-output refusal channel: a `refusal` content part in
    // the message item. Decoding it as an empty success would hide the
    // refusal from the caller and from failover — the contract H8 set
    // for Anthropic and Bedrock, and R2-29 extended to Chat, extends
    // here.
    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();
    let body = json!({
        "id": "resp_1",
        "status": "completed",
        "output": [{
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "content": [{ "type": "refusal", "refusal": "I can't help with that." }],
        }],
    });

    let error = codec()
        .decode_response(&route, body.clone())
        .expect_err("a refusal must fail the call");

    assert_eq!(error.kind(), ErrorKind::ContentFilter);
    assert_eq!(error.provider_code(), Some("refusal"));
    assert!(
        error.to_string().contains("I can't help with that."),
        "{error}"
    );
    assert_eq!(error.raw_data(), Some(&body));
    Ok(())
}

#[test]
fn a_streamed_refusal_fails_at_the_terminal_event() -> Result<(), Box<dyn StdError>> {
    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();
    let mut decoder = codec().stream_decoder(&route);
    let item = json!({
        "type": "message",
        "id": "msg_1",
        "status": "completed",
        "role": "assistant",
        "content": [{ "type": "refusal", "refusal": "I can't help with that." }],
    });
    decoder.decode(sse(
        &json!({ "type": "response.created", "response": { "id": "resp_1" } }),
    ))?;
    decoder.decode(sse(&json!({
        "type": "response.output_item.added",
        "output_index": 0,
        "item": { "type": "message", "id": "msg_1", "role": "assistant", "content": [] },
    })))?;
    decoder.decode(sse(&json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "item": item,
    })))?;

    let error = decoder
        .decode(sse(&json!({
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "status": "completed",
                "output": [item],
            },
        })))
        .expect_err("a refusal must fail the stream");

    assert_eq!(error.kind(), ErrorKind::ContentFilter);
    assert_eq!(error.provider_code(), Some("refusal"));
    assert!(
        error.to_string().contains("I can't help with that."),
        "{error}"
    );
    Ok(())
}

#[test]
fn a_terminal_document_without_output_completes_from_the_streamed_blocks()
-> Result<(), Box<dyn StdError>> {
    // A middlebox that strips the terminal document below the decodable
    // shape must not fail a stream whose answer already streamed. The id,
    // usage, and status it does carry are salvaged.
    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();
    let mut decoder = codec().stream_decoder(&route);
    let transcript = vec![
        json!({ "type": "response.created", "response": { "id": "resp_1" } }),
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": { "type": "message", "id": "msg_1", "role": "assistant", "content": [] },
        }),
        json!({ "type": "response.output_text.delta", "item_id": "msg_1", "delta": "Hello" }),
        json!({
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "status": "completed",
                "usage": { "input_tokens": 7, "output_tokens": 2 },
            },
        }),
    ];

    let mut events = Vec::new();
    for event in transcript {
        events.extend(decoder.decode(sse(&event))?);
    }
    events.extend(decoder.finish()?);

    let response = completed(&events)?;
    assert_eq!(response.id.as_deref(), Some("resp_1"));
    assert_eq!(response.finish_reason, FinishReason::Stop);
    assert_eq!(response.usage.input, 7);
    assert_eq!(response.usage.output, 2);
    let [ContentPart::Text { text }] = response.content.as_slice() else {
        return Err(format!("expected the streamed text, got {:?}", response.content).into());
    };
    assert_eq!(text, "Hello");
    Ok(())
}

#[test]
fn a_flattened_terminal_event_completes_from_its_own_fields() -> Result<(), Box<dyn StdError>> {
    // A gateway may flatten the terminal document into the event itself.
    // The old client read either shape; the fields are salvaged from the
    // event object instead of completing empty-handed.
    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();
    let mut decoder = codec().stream_decoder(&route);
    let transcript = vec![
        json!({ "type": "response.created", "response": { "id": "resp_1" } }),
        json!({ "type": "response.output_text.delta", "item_id": "msg_1", "delta": "Hi" }),
        json!({
            "type": "response.completed",
            "id": "resp_1",
            "status": "completed",
            "usage": { "input_tokens": 3, "output_tokens": 1 },
        }),
    ];

    let mut events = Vec::new();
    for event in transcript {
        events.extend(decoder.decode(sse(&event))?);
    }
    events.extend(decoder.finish()?);

    let response = completed(&events)?;
    assert_eq!(response.id.as_deref(), Some("resp_1"));
    assert_eq!(response.finish_reason, FinishReason::Stop);
    assert_eq!(response.usage.input, 3);
    Ok(())
}

#[test]
fn a_stream_transcript_produces_one_block_per_item() -> Result<(), Box<dyn StdError>> {
    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();
    let mut decoder = codec().stream_decoder(&route);
    let transcript = vec![
        json!({ "type": "response.created", "response": { "id": "resp_stream" } }),
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": { "type": "message", "id": "msg_1", "role": "assistant", "content": [] },
        }),
        json!({ "type": "response.output_text.delta", "item_id": "msg_1", "delta": "Hel" }),
        json!({ "type": "response.output_text.delta", "item_id": "msg_1", "delta": "lo" }),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "message",
                "id": "msg_1",
                "content": [{ "type": "output_text", "text": "Hello" }],
            },
        }),
        json!({
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {
                "type": "function_call",
                "id": "fc_123",
                "call_id": "call_abc",
                "name": "search",
            },
        }),
        json!({
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_123",
            "delta": "{\"qu",
        }),
        json!({
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_123",
            "delta": "ery\":\"rust\"}",
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 1,
            "item": {
                "type": "function_call",
                "id": "fc_123",
                "call_id": "call_abc",
                "name": "search",
                "arguments": "{\"query\":\"rust\"}",
            },
        }),
        json!({
            "type": "response.completed",
            "response": {
                "id": "resp_stream",
                "status": "completed",
                "output": [],
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 5,
                    "input_tokens_details": { "cached_tokens": 2 },
                    "output_tokens_details": { "reasoning_tokens": 1 },
                },
            },
        }),
    ];

    let mut events = Vec::new();
    for event in transcript {
        events.extend(decoder.decode(sse(&event))?);
    }
    events.extend(decoder.finish()?);

    assert_block_boundaries(&events)?;
    let starts = events
        .iter()
        .filter(|event| matches!(event, StreamEvent::ContentBlockStart { .. }))
        .count();
    // The message contributes two blocks: the visible text and the whole
    // item, kept for replay.
    assert_eq!(starts, 3);
    assert!(matches!(
        events.first(),
        Some(StreamEvent::Started { id: Some(id) }) if id == "resp_stream"
    ));

    let parts = ended_parts(&events);
    assert_eq!(parts[0], ContentPart::Text {
        text: "Hello".to_owned(),
    });
    assert_eq!(
        parts[1],
        ContentPart::opaque(
            MESSAGE_KIND,
            json!({
                "type": "message",
                "id": "msg_1",
                "content": [{ "type": "output_text", "text": "Hello" }],
            })
        )
    );
    let ContentPart::ToolCall(streamed) = &parts[2] else {
        return Err("expected a streamed tool call".into());
    };
    assert_eq!(streamed.id, "call_abc");
    assert_eq!(streamed.name, "search");
    assert_eq!(streamed.input.wire_value(), json!({ "query": "rust" }));
    assert_eq!(
        streamed.provider_metadata.get("openai"),
        Some(&json!({ "item_id": "fc_123" }))
    );

    let response = completed(&events)?;
    assert_eq!(response.content, parts);
    assert_eq!(response.id.as_deref(), Some("resp_stream"));
    assert_eq!(response.usage.input, 9);
    assert_eq!(response.usage.cache_read, 2);
    assert_eq!(response.usage.reasoning, 1);
    assert!(response.raw.is_some());
    // The terminal document's `output` is empty — trimmed by a middlebox,
    // say — but the stream plainly delivered a tool call, and the
    // assembled blocks decide.
    assert_eq!(response.finish_reason, FinishReason::ToolCall);
    Ok(())
}

#[test]
fn a_streamed_reasoning_item_keeps_its_text_and_its_replay_part() -> Result<(), Box<dyn StdError>> {
    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();
    let mut decoder = codec().stream_decoder(&route);
    let item = json!({
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{ "type": "summary_text", "text": "Checked the files." }],
        "content": [{ "type": "reasoning_text", "text": "checked" }],
        "encrypted_content": "gAAAA",
    });

    let mut events = decoder.decode(sse(&json!({
        "type": "response.output_item.added",
        "output_index": 0,
        "item": { "type": "reasoning", "id": "rs_1", "summary": [] },
    })))?;
    // The summary streams first and the trace after it. Both are shown
    // live; the terminal item then settles the part on its `content`
    // alone, the same text the blocking decode of the item produces.
    events.extend(decoder.decode(sse(&json!({
        "type": "response.reasoning_summary_text.delta",
        "item_id": "rs_1",
        "summary_index": 0,
        "delta": "Checked the files.",
    })))?);
    events.extend(decoder.decode(sse(&json!({
        "type": "response.reasoning_text.delta",
        "item_id": "rs_1",
        "content_index": 0,
        "delta": "checked",
    })))?);
    events.extend(decoder.decode(sse(&json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "item": item.clone(),
    })))?);
    events.extend(decoder.finish()?);

    assert_block_boundaries(&events)?;
    let parts = ended_parts(&events);
    let [
        ContentPart::Reasoning(reasoning),
        ContentPart::Opaque { kind, data },
    ] = parts.as_slice()
    else {
        return Err("expected reasoning text and its replay part".into());
    };
    assert_eq!(reasoning.text, "checked");
    assert_eq!(kind, "openai.reasoning");
    assert_eq!(data, &item);
    assert_eq!(reasoning_deltas(&events), vec![
        "Checked the files.",
        "checked"
    ]);
    assert_eq!(completed(&events)?.content, parts);
    assert_eq!(
        parts,
        codec()
            .decode_response(
                &route,
                json!({ "id": "resp_1", "status": "completed", "output": [item] })
            )?
            .content,
        "the stream and the blocking decode must agree"
    );
    Ok(())
}

#[test]
fn a_streamed_summary_only_reasoning_item_streams_its_summary() -> Result<(), Box<dyn StdError>> {
    // The hosted models' usual shape: a summary and no `content`. The
    // summary streams live, its blocks separated by the blank line the
    // terminal item's join puts there, so reconciliation has nothing to
    // add.
    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();
    let mut decoder = codec().stream_decoder(&route);
    let item = json!({
        "type": "reasoning",
        "id": "rs_1",
        "summary": [
            { "type": "summary_text", "text": "Checked the files." },
            { "type": "summary_text", "text": "Nothing to change." },
        ],
        "encrypted_content": "gAAAA",
    });

    let mut events = decoder.decode(sse(&json!({
        "type": "response.output_item.added",
        "output_index": 0,
        "item": { "type": "reasoning", "id": "rs_1", "summary": [] },
    })))?;
    for (index, delta) in ["Checked the files.", "Nothing to change."]
        .iter()
        .enumerate()
    {
        events.extend(decoder.decode(sse(&json!({
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs_1",
            "summary_index": index,
            "delta": delta,
        })))?);
    }
    events.extend(decoder.decode(sse(&json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "item": item.clone(),
    })))?);
    events.extend(decoder.finish()?);

    assert_block_boundaries(&events)?;
    assert_eq!(reasoning_deltas(&events), vec![
        "Checked the files.",
        "\n\nNothing to change."
    ]);
    let parts = ended_parts(&events);
    assert_eq!(parts, vec![
        ContentPart::Reasoning(ReasoningContent {
            text:             "Checked the files.\n\nNothing to change.".to_owned(),
            signature:        None,
            signature_origin: None,
            redacted:         false,
        }),
        ContentPart::opaque(REASONING_KIND, item.clone()),
    ]);
    assert_eq!(completed(&events)?.content, parts);
    assert_eq!(
        parts,
        codec()
            .decode_response(
                &route,
                json!({ "id": "resp_1", "status": "completed", "output": [item] })
            )?
            .content,
        "the stream and the blocking decode must agree"
    );
    Ok(())
}

#[test]
fn streamed_reasoning_entries_join_the_way_the_terminal_item_does() -> Result<(), Box<dyn StdError>>
{
    // The second `content` entry opens with the blank line the terminal
    // item's join puts there, so the streamed text is a prefix of the
    // whole and reconciliation has nothing to add.
    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();
    let mut decoder = codec().stream_decoder(&route);
    let item = json!({
        "type": "reasoning",
        "id": "rs_1",
        "summary": [],
        "content": [
            { "type": "reasoning_text", "text": "First." },
            { "type": "reasoning_text", "text": "Second." },
        ],
    });

    let mut events = Vec::new();
    for (index, delta) in [(0, "Fir"), (0, "st."), (1, "Sec"), (1, "ond.")] {
        events.extend(decoder.decode(sse(&json!({
            "type": "response.reasoning_text.delta",
            "item_id": "rs_1",
            "content_index": index,
            "delta": delta,
        })))?);
    }
    events.extend(decoder.decode(sse(&json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "item": item.clone(),
    })))?);
    events.extend(decoder.finish()?);

    assert_block_boundaries(&events)?;
    assert_eq!(reasoning_deltas(&events), vec![
        "Fir", "st.", "\n\nSec", "ond."
    ]);
    let parts = ended_parts(&events);
    let [ContentPart::Reasoning(reasoning), _opaque] = parts.as_slice() else {
        return Err("expected reasoning text and its replay part".into());
    };
    assert_eq!(reasoning.text, "First.\n\nSecond.");
    assert_eq!(
        parts,
        codec()
            .decode_response(
                &route,
                json!({ "id": "resp_1", "status": "completed", "output": [item] })
            )?
            .content,
        "the stream and the blocking decode must agree"
    );
    Ok(())
}

#[test]
fn a_stream_error_ends_the_stream_without_completing() -> Result<(), Box<dyn StdError>> {
    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();
    let mut decoder = codec().stream_decoder(&route);

    decoder.decode(sse(
        &json!({ "type": "response.created", "response": { "id": "resp_1" } }),
    ))?;
    let error = decoder
        .decode(sse(&json!({
            "type": "error",
            "error": { "type": "server_error", "message": "upstream failed" },
        })))
        .err()
        .ok_or("expected a stream error")?;

    assert!(error.message().contains("upstream failed"));
    Ok(())
}

#[test]
fn a_failed_event_without_a_response_wrapper_keeps_its_detail() -> Result<(), Box<dyn StdError>> {
    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();
    let mut decoder = codec().stream_decoder(&route);

    let error = decoder
        .decode(sse(&json!({
            "type": "response.failed",
            "error": { "code": "rate_limit_exceeded", "message": "Rate limit reached" },
        })))
        .err()
        .ok_or("expected a stream error")?;

    assert!(error.message().contains("Rate limit reached"));
    assert_eq!(error.provider_code(), Some("rate_limit_exceeded"));
    assert!(error.raw_data().is_some());
    Ok(())
}

#[test]
fn an_item_without_deltas_still_produces_its_content() -> Result<(), Box<dyn StdError>> {
    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();
    let mut decoder = codec().stream_decoder(&route);

    let mut events = decoder.decode(sse(&json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "type": "custom_tool_call",
            "id": "ctc_1",
            "call_id": "call_001",
            "name": "apply_patch",
            "input": "*** Begin Patch",
        },
    })))?;
    events.extend(decoder.finish()?);

    assert_block_boundaries(&events)?;
    let parts = ended_parts(&events);
    let [ContentPart::ToolCall(recovered)] = parts.as_slice() else {
        return Err("expected one recovered tool call".into());
    };
    assert_eq!(recovered.input.kind(), ToolCallKind::Custom);
    assert_eq!(
        recovered.input.wire_value(),
        Value::String("*** Begin Patch".to_owned())
    );
    Ok(())
}

#[test]
fn the_count_tokens_body_keeps_only_allowlisted_fields() -> Result<(), Box<dyn StdError>> {
    let request = Request::builder()
        .model(MODEL)
        .user("Hello")
        .temperature(0.2)
        .max_output_tokens(256)
        .stop_sequence("END")
        .metadata_entry("trace_id", "t789")
        .provider_option("openai", "seed", json!(7))
        .build()?;
    let call = call(request)?;

    let encoded = codec()
        .encode_count_tokens(&call)
        .ok_or("expected a token count request")??;

    assert!(encoded.url.ends_with("/v1/responses/input_tokens"));
    let body = encoded.body.as_object().ok_or("expected a JSON object")?;
    for key in body.keys() {
        assert!(
            COUNT_TOKENS_FIELDS.contains(&key.as_str()),
            "{key} is not accepted by the count endpoint",
        );
    }
    assert!(!body.contains_key("temperature"));
    assert!(!body.contains_key("max_output_tokens"));
    assert!(!body.contains_key("stop"));
    assert!(!body.contains_key("stream"));
    assert!(!body.contains_key("metadata"));
    assert!(!body.contains_key("seed"));
    assert_eq!(body["model"], json!("gpt-5.6-luna"));
    Ok(())
}

#[test]
fn a_token_count_with_the_wrong_object_is_a_decode_error() -> Result<(), Box<dyn StdError>> {
    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();
    let codec = codec();

    let tokens = codec.decode_count_tokens(
        &route,
        json!({ "input_tokens": 123, "object": "response.input_tokens" }),
    )?;
    let error = codec
        .decode_count_tokens(&route, json!({ "input_tokens": 123, "object": "response" }))
        .err()
        .ok_or("expected a decode error")?;

    assert_eq!(tokens, 123);
    assert_eq!(error.kind(), ErrorKind::ResponseDecode);
    assert_eq!(error.retry_classification(), RetryClassification::Safe);
    Ok(())
}

/// The ported round trip from the reference implementation.
///
/// An assistant turn of reasoning, visible text, and a tool call replays as
/// three items in that order. The preserved `message` item is sent with its
/// own `id` and `status`, rather than reconstructed from the text part, so
/// the reasoning item that precedes it still names an item that follows it.
#[test]
fn a_reasoning_message_and_function_call_replay_in_order() -> Result<(), Box<dyn StdError>> {
    let reasoning = json!({
        "type": "reasoning",
        "id": "rs_xyz789",
        "summary": [{ "type": "summary_text", "text": "Let me check." }],
        "encrypted_content": "gAAAA",
    });
    let message = json!({
        "type": "message",
        "id": "msg_abc123",
        "status": "completed",
        "role": "assistant",
        "content": [{ "type": "output_text", "text": "Checking now." }],
    });
    let mut tool_call = ToolCall::function("call_001", "shell", json!({ "cmd": "ls" }));
    tool_call
        .provider_metadata
        .insert("openai".to_owned(), json!({ "item_id": "fc_def456" }));
    let request = Request::builder()
        .model(MODEL)
        .user("List the files")
        .message(Message::new(Role::Assistant, [
            ContentPart::opaque(REASONING_KIND, reasoning.clone()),
            ContentPart::Text {
                text: "Checking now.".to_owned(),
            },
            ContentPart::opaque(MESSAGE_KIND, message.clone()),
            ContentPart::ToolCall(tool_call),
        ]))
        .build()?;

    let encoded = codec().encode(&call(request)?, false)?;

    let input = encoded.body["input"]
        .as_array()
        .ok_or("expected an input array")?;
    assert_eq!(input.len(), 4, "the user turn plus three replayed items");
    assert_eq!(input[1], reasoning);
    assert_eq!(input[2], message);
    assert_eq!(input[3]["type"], json!("function_call"));
    assert_eq!(input[3]["id"], json!("fc_def456"));
    assert_eq!(input[3]["call_id"], json!("call_001"));
    assert_eq!(
        encoded.body.to_string().matches("Checking now.").count(),
        1,
        "the preserved item already carries the assistant text",
    );
    Ok(())
}

/// A history tool call with an empty name encodes to nothing, as the
/// reference client did — `{"name": ""}` draws a provider 400.
#[test]
fn an_empty_name_tool_call_in_history_is_skipped() -> Result<(), Box<dyn StdError>> {
    let request = Request::builder()
        .model(MODEL)
        .user("List the files")
        .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
            ToolCall::function("call_1", "", json!({})),
        )]))
        .build()?;

    let encoded = codec().encode(&call(request)?, false)?;

    let input = encoded.body["input"]
        .as_array()
        .ok_or("expected an input array")?;
    assert_eq!(input.len(), 1, "only the user turn may travel: {input:?}");
    Ok(())
}

/// Media in a system message never reaches the wire: the Responses API
/// accepts only `input_text` inside a system item and 400s on anything
/// else, where the reference client silently dropped it. The text
/// travels, the drop is reported.
#[test]
fn media_in_a_system_message_is_dropped_and_warned() -> Result<(), Box<dyn StdError>> {
    let request = Request::builder()
        .model(MODEL)
        .message(Message::new(Role::System, [
            ContentPart::Text {
                text: "Use the style guide.".to_owned(),
            },
            ContentPart::Image(ImageContent::new(MediaSource::url(
                "https://example.com/guide.png",
            ))),
        ]))
        .user("Hello")
        .build()?;

    let encoded = codec().encode(&call(request)?, false)?;

    let system_content = &encoded.body["input"][0]["content"];
    assert_eq!(
        system_content,
        &json!([{ "type": "input_text", "text": "Use the style guide." }]),
        "only the text may travel inside a system item"
    );
    assert_eq!(
        encoded.warnings.len(),
        1,
        "the dropped image must be reported: {:?}",
        encoded.warnings
    );
    Ok(())
}

/// The decoder emits a readable reasoning part BESIDE the opaque item
/// that replays the same text, so the codec's own round trip must not
/// warn about dropped reasoning — nothing is lost. Reasoning without an
/// opaque sibling in its message still warns.
#[test]
fn a_reasoning_part_beside_its_opaque_item_does_not_warn() -> Result<(), Box<dyn StdError>> {
    let reasoning = ContentPart::Reasoning(ReasoningContent {
        text:             "Let me check.".to_owned(),
        signature:        None,
        signature_origin: None,
        redacted:         false,
    });
    let item = json!({
        "type": "reasoning",
        "id": "rs_1",
        "summary": [],
        "content": [{ "type": "reasoning_text", "text": "Let me check." }],
    });

    let round_trip = Request::builder()
        .model(MODEL)
        .user("List the files")
        .message(Message::new(Role::Assistant, [
            reasoning.clone(),
            ContentPart::opaque(REASONING_KIND, item),
            ContentPart::Text {
                text: "Checking now.".to_owned(),
            },
        ]))
        .build()?;
    let encoded = codec().encode(&call(round_trip)?, false)?;
    assert_eq!(encoded.warnings, vec![], "the round trip loses nothing");

    let orphaned = Request::builder()
        .model(MODEL)
        .user("List the files")
        .message(Message::new(Role::Assistant, [reasoning]))
        .build()?;
    let encoded = codec().encode(&call(orphaned)?, false)?;
    assert_eq!(
        encoded.warnings.len(),
        1,
        "reasoning with no opaque sibling is dropped and must warn"
    );
    Ok(())
}

/// An interleaved turn keeps every reasoning item beside the item it
/// anchors, which hoisting the opaque parts to the front would break.
#[test]
fn interleaved_replay_items_keep_their_pairing() -> Result<(), Box<dyn StdError>> {
    let first = json!({ "type": "reasoning", "id": "rs_1", "encrypted_content": "a" });
    let second = json!({ "type": "reasoning", "id": "rs_2", "encrypted_content": "b" });
    let request = Request::builder()
        .model(MODEL)
        .user("Do both")
        .message(Message::new(Role::Assistant, [
            ContentPart::opaque(REASONING_KIND, first.clone()),
            ContentPart::ToolCall(ToolCall::function("call_1", "one", json!({}))),
            ContentPart::opaque(REASONING_KIND, second.clone()),
            ContentPart::ToolCall(ToolCall::function("call_2", "two", json!({}))),
        ]))
        .build()?;

    let encoded = codec().encode(&call(request)?, false)?;

    let types: Vec<&Value> = encoded.body["input"]
        .as_array()
        .ok_or("expected an input array")?
        .iter()
        .map(|item| &item["type"])
        .collect();
    // The user turn is a plain `{role, content}` item, which carries no
    // `type` of its own.
    assert_eq!(types, vec![
        &Value::Null,
        &json!("reasoning"),
        &json!("function_call"),
        &json!("reasoning"),
        &json!("function_call"),
    ]);
    assert_eq!(encoded.body["input"][1], first);
    assert_eq!(encoded.body["input"][3], second);
    Ok(())
}

/// Without a preserved item, assistant text still builds one.
#[test]
fn assistant_text_without_a_replay_item_builds_a_message() -> Result<(), Box<dyn StdError>> {
    let request = Request::builder()
        .model(MODEL)
        .user("Hi")
        .message(Message::new(Role::Assistant, [ContentPart::Text {
            text: "Hello".to_owned(),
        }]))
        .build()?;

    let encoded = codec().encode(&call(request)?, false)?;

    assert_eq!(
        encoded.body["input"][1],
        json!({
            "role": "assistant",
            "content": [{ "type": "output_text", "text": "Hello" }],
        })
    );
    Ok(())
}

#[test]
fn a_message_item_decodes_to_text_and_a_replay_part() -> Result<(), Box<dyn StdError>> {
    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();
    let item = json!({
        "type": "message",
        "id": "msg_1",
        "status": "completed",
        "role": "assistant",
        "content": [{ "type": "output_text", "text": "hello" }],
    });

    let response = codec().decode_response(
        &route,
        json!({ "id": "resp_1", "status": "completed", "output": [item.clone()] }),
    )?;

    assert_eq!(response.content, vec![
        ContentPart::Text {
            text: "hello".to_owned(),
        },
        ContentPart::opaque(MESSAGE_KIND, item),
    ]);
    Ok(())
}

/// A `function_call` with no name is model-internal: it is not content, and
/// it does not make the turn look like a tool call.
#[test]
fn a_tool_call_cut_at_the_output_limit_is_dropped() -> Result<(), Box<dyn StdError>> {
    // Verified live: a call the limit cut short arrives as an
    // `incomplete` item whose arguments are a JSON prefix. It is not a
    // call, so it leaves the content and a warning takes its place.
    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();

    let response = codec().decode_response(
        &route,
        json!({
            "id": "resp_1",
            "status": "incomplete",
            "incomplete_details": { "reason": "max_output_tokens" },
            "output": [{
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "status": "incomplete",
                "name": "write_note",
                "arguments": "{\"title\":\"Rome\",\"body\":\"Rome began",
            }],
        }),
    )?;

    assert_eq!(response.content, Vec::new());
    assert_eq!(response.finish_reason, FinishReason::Length);
    assert_eq!(response.warnings.len(), 1);
    assert_eq!(response.warnings[0].code, "truncated_tool_call");
    assert!(response.warnings[0].message.contains("write_note"));
    assert!(response.raw.is_some(), "the provider's item stays in raw");
    Ok(())
}

#[test]
fn a_streamed_tool_call_cut_at_the_output_limit_is_dropped_from_the_completed_response()
-> Result<(), Box<dyn StdError>> {
    // The block events deliver the call as it streams; the completed
    // response, which is what a consumer acts on, carries no call.
    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();
    let mut decoder = codec().stream_decoder(&route);
    let item = |status: &str, arguments: &str| {
        json!({
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "status": status,
            "name": "write_note",
            "arguments": arguments,
        })
    };
    let transcript = vec![
        json!({ "type": "response.created", "response": { "id": "resp_1" } }),
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": item("in_progress", ""),
        }),
        json!({
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "delta": "{\"title\":\"Rome\",\"body\":\"Rome began",
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": item("incomplete", "{\"title\":\"Rome\",\"body\":\"Rome began"),
        }),
        json!({
            "type": "response.incomplete",
            "response": {
                "id": "resp_1",
                "status": "incomplete",
                "incomplete_details": { "reason": "max_output_tokens" },
                "output": [item("incomplete", "{\"title\":\"Rome\",\"body\":\"Rome began")],
                "usage": { "input_tokens": 10, "output_tokens": 40 },
            },
        }),
    ];

    let mut events = Vec::new();
    for event in transcript {
        events.extend(decoder.decode(sse(&event))?);
    }
    events.extend(decoder.finish()?);

    assert!(
        matches!(ended_parts(&events).as_slice(), [ContentPart::ToolCall(_)]),
        "the block end still shows what streamed"
    );
    let response = completed(&events)?;
    assert_eq!(response.content, Vec::new());
    assert_eq!(response.finish_reason, FinishReason::Length);
    assert_eq!(response.warnings.len(), 1);
    assert_eq!(response.warnings[0].code, "truncated_tool_call");
    Ok(())
}

#[test]
fn an_unnamed_tool_call_is_dropped() -> Result<(), Box<dyn StdError>> {
    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();

    let response = codec().decode_response(
        &route,
        json!({
            "id": "resp_1",
            "status": "completed",
            "output": [{
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "",
                "arguments": "{}",
            }],
        }),
    )?;

    assert_eq!(response.content, Vec::new());
    assert_eq!(response.finish_reason, FinishReason::Stop);
    Ok(())
}

#[test]
fn an_unnamed_streamed_tool_call_is_dropped() -> Result<(), Box<dyn StdError>> {
    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();
    let mut decoder = codec().stream_decoder(&route);
    let item = json!({
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "",
        "arguments": "{}",
    });

    let mut events = decoder.decode(sse(&json!({
        "type": "response.output_item.added",
        "output_index": 0,
        "item": item,
    })))?;
    events.extend(decoder.decode(sse(&json!({
        "type": "response.function_call_arguments.delta",
        "item_id": "fc_1",
        "delta": "{}",
    })))?);
    events.extend(decoder.decode(sse(&json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "item": item,
    })))?);
    events.extend(decoder.decode(sse(&json!({
        "type": "response.completed",
        "response": { "id": "resp_1", "status": "completed", "output": [item] },
    })))?);

    assert_block_boundaries(&events)?;
    assert_eq!(ended_parts(&events), Vec::new());
    let response = completed(&events)?;
    assert_eq!(response.content, Vec::new());
    assert_eq!(response.finish_reason, FinishReason::Stop);
    Ok(())
}

/// A proxy's keepalive frame is not model output, so it cannot end the
/// stream.
#[test]
fn a_frame_that_is_not_json_is_skipped() -> Result<(), Box<dyn StdError>> {
    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();
    let mut decoder = codec().stream_decoder(&route);

    let ignored = decoder.decode(SseEvent {
        event: None,
        data:  "keepalive".to_owned(),
    })?;
    let events = decoder.decode(sse(&json!({
        "type": "response.created",
        "response": { "id": "resp_1" },
    })))?;

    assert_eq!(ignored, Vec::new());
    assert!(matches!(
        events.first(),
        Some(StreamEvent::Started { id: Some(id) }) if id == "resp_1"
    ));
    Ok(())
}

/// A proxy that drops `response.created` still produces a started stream.
#[test]
fn the_stream_starts_without_a_created_event() -> Result<(), Box<dyn StdError>> {
    let route = call(Request::builder().model(MODEL).user("hi").build()?)?
        .route()
        .clone();
    let mut decoder = codec().stream_decoder(&route);

    let mut events = decoder.decode(sse(&json!({
        "type": "response.output_item.added",
        "output_index": 0,
        "item": { "type": "message", "id": "msg_1", "role": "assistant", "content": [] },
    })))?;
    events.extend(decoder.decode(sse(&json!({
        "type": "response.output_text.delta",
        "item_id": "msg_1",
        "delta": "hi",
    })))?);

    assert!(
        matches!(events.first(), Some(StreamEvent::Started { id: None })),
        "the first event must still be `Started`",
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, StreamEvent::Started { .. }))
            .count(),
        1,
        "a later event must not start the stream twice",
    );
    Ok(())
}

/// A tool result carrying only JSON sends the value itself, because the
/// `ContentPart` envelope is this crate's shape rather than the tool's.
#[test]
fn a_json_tool_result_sends_the_bare_value() -> Result<(), Box<dyn StdError>> {
    let request = Request::builder()
        .model(MODEL)
        .user("Look it up")
        .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
            ToolCall::function("call_1", "search", json!({})),
        )]))
        .message(Message::new(Role::Tool, [ContentPart::ToolResult(
            ToolResult {
                tool_call_id: "call_1".to_owned(),
                name:         Some("search".to_owned()),
                content:      vec![ContentPart::Json {
                    value: json!({ "matches": 2 }),
                }],
                is_error:     false,
            },
        )]))
        .build()?;

    let encoded = codec().encode(&call(request)?, false)?;

    let output = encoded.body["input"][2]["output"]
        .as_str()
        .ok_or("expected a function_call_output")?;
    assert_eq!(
        serde_json::from_str::<Value>(output)?,
        json!({ "matches": 2 })
    );
    Ok(())
}

#[test]
fn an_empty_text_tool_result_sends_an_empty_output() -> Result<(), Box<dyn StdError>> {
    // A command with no stdout answers with nothing. Serializing the
    // ContentPart envelope instead would hand the model spurious JSON as
    // the tool's answer.
    let request = Request::builder()
        .model(MODEL)
        .user("Make the directory")
        .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
            ToolCall::function("call_1", "shell", json!({ "cmd": "mkdir foo" })),
        )]))
        .message(Message::new(Role::Tool, [ContentPart::ToolResult(
            ToolResult {
                tool_call_id: "call_1".to_owned(),
                name:         Some("shell".to_owned()),
                content:      vec![ContentPart::Text {
                    text: String::new(),
                }],
                is_error:     false,
            },
        )]))
        .build()?;

    let encoded = codec().encode(&call(request)?, false)?;

    assert_eq!(encoded.body["input"][2]["output"], json!(""));
    Ok(())
}
