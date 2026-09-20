use std::error::Error as StdError;

use serde_json::{Value, json};

use super::{BedrockConverseCodec, Codec};
use crate::codecs::test_support::{resolved, resolved_in};
use crate::transport::SseEvent;
use crate::types::{
    ContentPart, DocumentContent, ErrorKind, FinishReason, ImageContent, MediaSource, Message,
    ReasoningContent, ReasoningEffort, Request, Response, ResponseFormat, RetryClassification,
    Role, Speed, StreamEvent, ToolCall, ToolChoice, ToolDefinition, ToolResult,
};

const MODEL: &str = "bedrock/anthropic.claude-sonnet-4-6";

fn warnings_for(request: Request) -> Result<Vec<String>, Box<dyn StdError>> {
    Ok(BedrockConverseCodec
        .encode(&resolved(request)?, false)?
        .warnings
        .into_iter()
        .map(|warning| warning.code)
        .collect())
}

fn encoded(request: Request) -> Result<Value, Box<dyn StdError>> {
    Ok(BedrockConverseCodec
        .encode(&resolved(request)?, false)?
        .body)
}

fn decoded(body: Value) -> Result<Response, Box<dyn StdError>> {
    let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;
    Ok(BedrockConverseCodec.decode_response(call.route(), body)?)
}

/// Drives one whole stream, returning every event in order.
fn streamed(frames: &[(&str, Value)]) -> Result<Vec<StreamEvent>, Box<dyn StdError>> {
    let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;
    let mut decoder = BedrockConverseCodec.stream_decoder(call.route());
    let mut events = Vec::new();
    for (name, payload) in frames {
        events.extend(decoder.decode(SseEvent {
            event: Some((*name).to_owned()),
            data:  payload.to_string(),
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

#[test]
fn a_body_without_the_output_message_fails_to_decode() -> Result<(), Box<dyn StdError>> {
    let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;

    // `{}` from a broken proxy and a body whose message lost its content
    // are both indistinguishable from a real empty answer; neither may
    // decode as success.
    for body in [json!({}), json!({ "output": { "message": {} } })] {
        let error = BedrockConverseCodec
            .decode_response(call.route(), body)
            .expect_err("a structureless 200 must fail to decode");
        assert_eq!(error.kind(), ErrorKind::ResponseDecode);
        assert_eq!(error.retry_classification(), RetryClassification::Safe);
    }
    Ok(())
}

#[test]
fn a_stream_event_that_is_not_json_fails_retryably() -> Result<(), Box<dyn StdError>> {
    let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;
    let mut decoder = BedrockConverseCodec.stream_decoder(call.route());

    let error = decoder
        .decode(SseEvent {
            event: Some("contentBlockDelta".to_owned()),
            data:  "not json".to_owned(),
        })
        .expect_err("a garbled event must fail the stream");

    assert_eq!(error.kind(), ErrorKind::StreamDecode);
    assert_eq!(error.retry_classification(), RetryClassification::Safe);
    Ok(())
}

#[test]
fn a_refusal_response_fails_instead_of_decoding() -> Result<(), Box<dyn StdError>> {
    let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;
    let body = json!({
        "output": { "message": { "content": [] } },
        "stopReason": "refusal",
    });

    let error = BedrockConverseCodec
        .decode_response(call.route(), body.clone())
        .expect_err("a refusal must fail the call");

    assert_eq!(error.kind(), ErrorKind::ContentFilter);
    assert_eq!(error.provider_code(), Some("refusal"));
    assert_eq!(error.raw_data(), Some(&body));
    Ok(())
}

#[test]
fn an_input_fragment_for_an_unopened_block_fails_the_stream() -> Result<(), Box<dyn StdError>> {
    // Converse announces every tool call in a contentBlockStart carrying
    // its id and name. When that start is lost, assembling the fragments
    // would fabricate a nameless call whose replay fails identifier
    // validation, so the stream fails retryably instead — the contract
    // R2-24 set for the Chat codec.
    let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;
    let mut decoder = BedrockConverseCodec.stream_decoder(call.route());

    let error = decoder
        .decode(SseEvent {
            event: Some("contentBlockDelta".to_owned()),
            data:  json!({
                "contentBlockIndex": 0,
                "delta": { "toolUse": { "input": "{\"q\":\"rust\"}" } },
            })
            .to_string(),
        })
        .err()
        .ok_or("expected the orphan input fragment to fail the stream")?;

    assert_eq!(error.kind(), ErrorKind::StreamDecode);
    assert_eq!(error.retry_classification(), RetryClassification::Safe);
    Ok(())
}

#[test]
fn a_streamed_refusal_ends_the_stream_as_an_error() -> Result<(), Box<dyn StdError>> {
    let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;
    let mut decoder = BedrockConverseCodec.stream_decoder(call.route());

    decoder.decode(SseEvent {
        event: Some("messageStart".to_owned()),
        data:  json!({ "role": "assistant" }).to_string(),
    })?;
    let error = decoder
        .decode(SseEvent {
            event: Some("messageStop".to_owned()),
            data:  json!({ "stopReason": "refusal" }).to_string(),
        })
        .expect_err("a refusal must fail the stream");

    assert_eq!(error.kind(), ErrorKind::ContentFilter);
    assert_eq!(error.provider_code(), Some("refusal"));
    Ok(())
}

#[test]
fn encodes_document_reasoning_and_performance_fields() -> Result<(), Box<dyn StdError>> {
    let body = encoded(
        Request::builder()
            .model(MODEL)
            .message(Message::new(Role::Assistant, [ContentPart::Reasoning(
                ReasoningContent {
                    text:             "checked".to_owned(),
                    signature:        Some("sig".to_owned()),
                    signature_origin: Some("anthropic".to_owned()),
                    redacted:         false,
                },
            )]))
            .message(Message::new(Role::User, [ContentPart::Document(
                DocumentContent {
                    source: MediaSource::parse("data:application/pdf;base64,aGVsbG8="),
                    name:   Some("source".to_owned()),
                },
            )]))
            .reasoning_effort(ReasoningEffort::High)
            .speed(Speed::Fast)
            .build()?,
    )?;

    assert_eq!(
        body["messages"][0]["content"][0]["reasoningContent"]["reasoningText"]["signature"],
        "sig"
    );
    assert_eq!(
        body["messages"][1]["content"][0]["document"]["format"],
        "pdf"
    );
    assert_eq!(
        body["messages"][1]["content"][0]["document"]["name"],
        "source"
    );
    assert_eq!(
        body["messages"][1]["content"][0]["document"]["source"]["bytes"],
        "aGVsbG8="
    );
    assert_eq!(body["performanceConfig"]["latency"], "optimized");
    assert_eq!(
        body["additionalModelRequestFields"]["output_config"]["effort"],
        "high"
    );
    Ok(())
}

/// A Bedrock model that reasons with a thinking budget, not effort levels.
const BUDGET_CATALOG: &str = r#"
    schema_version = 1

    [providers.bedrock]
    display_name = "Amazon Bedrock"
    adapter = "bedrock"
    codecs = ["bedrock-converse"]
    base_url = "https://bedrock-runtime.us-east-1.amazonaws.com"
    default_model = "older-claude"
    auth = { type = "none" }

    [providers.bedrock.models.older-claude]
    display_name = "Older Claude"
    api_model = "us.anthropic.claude-3-7"
    capabilities = { text = true, tools = true, reasoning = true, tool_choice = { required = true, named = true } }
"#;

/// The budget-model catalog with passthrough allowed.
const PASSTHROUGH_CATALOG: &str = r#"
    schema_version = 1

    [providers.bedrock]
    display_name = "Amazon Bedrock"
    adapter = "bedrock"
    codecs = ["bedrock-converse"]
    base_url = "https://bedrock-runtime.us-east-1.amazonaws.com"
    allow_passthrough = true
    default_model = "older-claude"
    auth = { type = "none" }

    [providers.bedrock.models.older-claude]
    display_name = "Older Claude"
    api_model = "us.anthropic.claude-3-7"
    capabilities = { text = true, tools = true, reasoning = true, tool_choice = { required = true, named = true } }
"#;

#[test]
fn a_passthrough_model_takes_the_effort_dialect() -> Result<(), Box<dyn StdError>> {
    // Passthrough serves models newer than the catalog, and those reject
    // a manual thinking toggle — so the uncataloged guess is the modern
    // dialect, the same one the Anthropic codec makes.
    let call = resolved_in(
        PASSTHROUGH_CATALOG,
        Request::builder()
            .model("bedrock/us.anthropic.claude-next")
            .user("Hello")
            .reasoning_effort(ReasoningEffort::High)
            .build()?,
    )?;

    let body = BedrockConverseCodec.encode(&call, false)?.body;

    assert_eq!(
        body["additionalModelRequestFields"]["output_config"]["effort"],
        "high"
    );
    assert_eq!(
        body["additionalModelRequestFields"]["thinking"],
        json!(null)
    );
    Ok(())
}

#[test]
fn effort_becomes_a_thinking_budget_without_effort_levels() -> Result<(), Box<dyn StdError>> {
    // A reasoning model without `reasoning_effort_levels` rejects
    // `output_config.effort`; it takes an explicit thinking budget, the
    // same translation the Anthropic codec applies. A caller limit the
    // budget would not fit under grows to keep the budget strictly below
    // `maxTokens`.
    let call = resolved_in(
        BUDGET_CATALOG,
        Request::builder()
            .model("bedrock/older-claude")
            .user("Hello")
            .reasoning_effort(ReasoningEffort::High)
            .max_output_tokens(4096)
            .build()?,
    )?;

    let body = BedrockConverseCodec.encode(&call, false)?.body;

    assert_eq!(
        body["additionalModelRequestFields"]["thinking"],
        json!({ "type": "enabled", "budget_tokens": 3072 })
    );
    assert_eq!(
        body["additionalModelRequestFields"]["output_config"],
        json!(null)
    );
    assert_eq!(body["inferenceConfig"]["maxTokens"], 4096);
    Ok(())
}

#[test]
fn a_thinking_budget_without_a_caller_limit_still_sends_max_tokens() -> Result<(), Box<dyn StdError>>
{
    // With no caller limit the budget derives from the fallback output
    // limit, and AWS's per-model default `maxTokens` would be in charge —
    // a default at or below the budget draws a ValidationException. The
    // effective limit therefore always goes on the wire, lifted when the
    // budget equals it.
    let call = resolved_in(
        BUDGET_CATALOG,
        Request::builder()
            .model("bedrock/older-claude")
            .user("Hello")
            .reasoning_effort(ReasoningEffort::Max)
            .build()?,
    )?;

    let body = BedrockConverseCodec.encode(&call, false)?.body;

    assert_eq!(
        body["additionalModelRequestFields"]["thinking"],
        json!({ "type": "enabled", "budget_tokens": 65_536 })
    );
    assert_eq!(body["inferenceConfig"]["maxTokens"], 66_560);
    Ok(())
}

#[test]
fn a_forced_tool_choice_suppresses_the_effort_encoding() -> Result<(), Box<dyn StdError>> {
    // The upstream model rejects thinking alongside a forced tool choice,
    // so neither effort dialect may reach the wire, and the dropped
    // control is reported.
    let encoded = BedrockConverseCodec.encode(
        &resolved(
            Request::builder()
                .model(MODEL)
                .user("Hello")
                .tool(ToolDefinition::function("lookup", "look up", json!({})))
                .tool_choice(ToolChoice::Required)
                .reasoning_effort(ReasoningEffort::High)
                .build()?,
        )?,
        false,
    )?;

    assert_eq!(encoded.body.get("additionalModelRequestFields"), None);
    assert!(
        encoded
            .warnings
            .iter()
            .any(|warning| warning.message.contains("forced tool choice")),
        "{:?}",
        encoded.warnings
    );
    Ok(())
}

#[test]
fn an_anthropic_signed_part_replays_and_a_gemini_one_is_skipped() -> Result<(), Box<dyn StdError>> {
    // Converse carries Claude-minted signatures, so a part signed at the
    // Anthropic provider keeps replaying after a failover to Bedrock; a
    // Gemini thought signature cannot verify and is skipped.
    let anthropic_signed = ReasoningContent {
        text:             "claude thought".to_owned(),
        signature:        Some("claude-sig".to_owned()),
        signature_origin: Some("anthropic".to_owned()),
        redacted:         false,
    };
    let gemini_signed = ReasoningContent {
        text:             "gemini thought".to_owned(),
        signature:        Some("gemini-sig".to_owned()),
        signature_origin: Some("gemini".to_owned()),
        redacted:         false,
    };
    let encoded = BedrockConverseCodec.encode(
        &resolved(
            Request::builder()
                .model(MODEL)
                .user("Hello")
                .message(Message::new(Role::Assistant, [
                    ContentPart::Reasoning(anthropic_signed),
                    ContentPart::Reasoning(gemini_signed),
                    ContentPart::Text {
                        text: "answer".to_owned(),
                    },
                ]))
                .user("Continue")
                .build()?,
        )?,
        false,
    )?;

    let body = encoded.body.to_string();
    assert!(body.contains("claude-sig"), "{body}");
    assert!(!body.contains("gemini-sig"), "{body}");
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
fn tool_choice_none_keeps_the_tools_when_the_history_carries_tool_blocks()
-> Result<(), Box<dyn StdError>> {
    // Converse requires `toolConfig` whenever messages carry toolUse or
    // toolResult blocks, so the agent-loop ending — answer in prose after
    // a tool exchange — must keep the tools on the wire, force nothing,
    // and report the choice it could not express.
    let encoded = BedrockConverseCodec.encode(
        &resolved(
            Request::builder()
                .model(MODEL)
                .user("What is the weather?")
                .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
                    ToolCall::function("call-1", "get_weather", json!({ "city": "Paris" })),
                )]))
                .message(Message::new(Role::Tool, [ContentPart::ToolResult(
                    ToolResult {
                        tool_call_id: "call-1".to_owned(),
                        name:         Some("get_weather".to_owned()),
                        content:      vec![ContentPart::Text {
                            text: "18C".to_owned(),
                        }],
                        is_error:     false,
                    },
                )]))
                .tool(ToolDefinition::function(
                    "get_weather",
                    "weather",
                    json!({}),
                ))
                .tool_choice(ToolChoice::None)
                .build()?,
        )?,
        false,
    )?;

    assert_eq!(
        encoded.body["toolConfig"]["tools"][0]["toolSpec"]["name"],
        "get_weather"
    );
    assert_eq!(encoded.body["toolConfig"].get("toolChoice"), None);
    assert!(
        encoded
            .warnings
            .iter()
            .any(|warning| warning.message.contains("tool_choice none")),
        "{:?}",
        encoded.warnings
    );

    // Without tool blocks in the history, withholding the tools stays the
    // faithful encoding of `none`.
    let clean = BedrockConverseCodec.encode(
        &resolved(
            Request::builder()
                .model(MODEL)
                .user("Hello")
                .tool(ToolDefinition::function(
                    "get_weather",
                    "weather",
                    json!({}),
                ))
                .tool_choice(ToolChoice::None)
                .build()?,
        )?,
        false,
    )?;
    assert_eq!(clean.body.get("toolConfig"), None);
    Ok(())
}

#[test]
fn a_streaming_request_names_the_event_stream_framing() -> Result<(), Box<dyn StdError>> {
    let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;

    let unary = BedrockConverseCodec.encode(&call, false)?;
    let streaming = BedrockConverseCodec.encode(&call, true)?;

    assert_eq!(streaming.headers, vec![(
        "accept".to_owned(),
        "application/vnd.amazon.eventstream".to_owned(),
    )]);
    assert!(unary.headers.is_empty());
    Ok(())
}

#[test]
fn percent_encodes_the_model_id_in_the_path() -> Result<(), Box<dyn StdError>> {
    let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;

    let unary = BedrockConverseCodec.encode(&call, false)?;
    let streaming = BedrockConverseCodec.encode(&call, true)?;

    assert_eq!(
        unary.url,
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-sonnet-4-6/converse"
    );
    assert_eq!(
        streaming.url,
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-sonnet-4-6/converse-stream"
    );
    Ok(())
}

#[test]
fn percent_encodes_an_arn_model_id() -> Result<(), Box<dyn StdError>> {
    let call = resolved(
        Request::builder()
            .model("bedrock/arn:aws:bedrock:us-east-1:1:inference-profile/custom")
            .user("Hello")
            .build()?,
    )?;

    let encoded = BedrockConverseCodec.encode(&call, false)?;

    assert!(
        encoded
            .url
            .ends_with("/model/arn:aws:bedrock:us-east-1:1:inference-profile%2Fcustom/converse"),
        "{}",
        encoded.url
    );
    Ok(())
}

#[test]
fn usage_buckets_stay_disjoint_without_subtraction() -> Result<(), Box<dyn StdError>> {
    let response = decoded(json!({
        "output": { "message": { "content": [{ "text": "hi" }] } },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 30,
            "outputTokens": 628,
            "totalTokens": 658,
            "cacheReadInputTokens": 1024,
            "cacheWriteInputTokens": 512,
        },
    }))?;

    assert_eq!(response.usage.input, 30);
    assert_eq!(response.usage.output, 628);
    assert_eq!(response.usage.reasoning, 0);
    assert_eq!(response.usage.cache_read, 1024);
    assert_eq!(response.usage.cache_write, 512);
    assert!(response.cost.is_none());
    assert!(response.raw.is_some());
    Ok(())
}

#[test]
fn decodes_tool_calls_and_finish_reasons() -> Result<(), Box<dyn StdError>> {
    let response = decoded(json!({
        "output": { "message": { "content": [
            { "toolUse": { "toolUseId": "call-1", "name": "search", "input": { "q": "rust" } } },
        ] } },
        "stopReason": "tool_use",
    }))?;

    assert_eq!(response.finish_reason, FinishReason::ToolCall);
    match response.content.first() {
        Some(ContentPart::ToolCall(call)) => {
            assert_eq!(call.id, "call-1");
            assert_eq!(call.name, "search");
            assert_eq!(call.input.wire_value(), json!({ "q": "rust" }));
        }
        other => return Err(format!("expected a tool call, got {other:?}").into()),
    }

    let stop = |reason: &str| {
        json!({
            "output": { "message": { "content": [] } },
            "stopReason": reason,
        })
    };
    assert_eq!(
        decoded(stop("stop_sequence"))?.finish_reason,
        FinishReason::Stop
    );
    assert_eq!(
        decoded(stop("max_tokens"))?.finish_reason,
        FinishReason::Length
    );
    assert_eq!(
        decoded(stop("guardrail_intervened"))?.finish_reason,
        FinishReason::ContentFilter
    );
    assert_eq!(
        decoded(stop("content_filtered"))?.finish_reason,
        FinishReason::ContentFilter
    );
    Ok(())
}

#[test]
fn raw_options_win_and_controls_never_reach_the_wire() -> Result<(), Box<dyn StdError>> {
    let body = encoded(
        Request::builder()
            .model(MODEL)
            .user("Hello")
            .max_output_tokens(16)
            .provider_option("bedrock", "inferenceConfig", json!({ "maxTokens": 4096 }))
            .provider_option("bedrock", "guardrailConfig", json!({ "trace": "enabled" }))
            .provider_option("bedrock", "auto_cache", json!(false))
            .provider_option("anthropic", "thinking", json!({ "type": "enabled" }))
            .build()?,
    )?;

    assert_eq!(body["inferenceConfig"]["maxTokens"], 4096);
    assert_eq!(body["guardrailConfig"]["trace"], "enabled");
    assert!(body.get("auto_cache").is_none());
    assert!(body.get("thinking").is_none());
    Ok(())
}

#[test]
fn stop_sequences_keep_their_order() -> Result<(), Box<dyn StdError>> {
    let body = encoded(
        Request::builder()
            .model(MODEL)
            .user("Hello")
            .stop_sequences(["END", "STOP"])
            .stop_sequence("HALT")
            .build()?,
    )?;

    assert_eq!(
        body["inferenceConfig"]["stopSequences"],
        json!(["END", "STOP", "HALT"])
    );
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
                json!({
                    "type": "grammar"
                }),
            ))
            .build()?,
    )?;

    let error = BedrockConverseCodec
        .encode(&call, false)
        .err()
        .ok_or("expected a custom tool to be rejected")?;

    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    assert!(error.message().contains("custom tools"));
    Ok(())
}

#[test]
fn a_url_media_source_is_rejected_before_dispatch() -> Result<(), Box<dyn StdError>> {
    let call = resolved(
        Request::builder()
            .model(MODEL)
            .message(Message::new(Role::User, [ContentPart::Image(
                ImageContent::new(MediaSource::url("https://example.com/cat.png")),
            )]))
            .build()?,
    )?;

    let error = BedrockConverseCodec
        .encode(&call, false)
        .err()
        .ok_or("expected URL media to be rejected")?;

    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    assert!(error.message().contains("URL media"));
    Ok(())
}

/// Two user turns, so the conversation prefix has somewhere to land.
fn caching_request(auto_cache: Option<bool>) -> Result<Request, Box<dyn StdError>> {
    let mut builder = Request::builder()
        .model(MODEL)
        .system("Be brief")
        .user("First")
        .message(Message::text(Role::Assistant, "Answer"))
        .user("Second")
        .tool(ToolDefinition::function("search", "Searches", json!({})));
    if let Some(auto_cache) = auto_cache {
        builder = builder.provider_option("bedrock", "auto_cache", json!(auto_cache));
    }
    Ok(builder.build()?)
}

#[test]
fn auto_cache_places_cache_points_by_default() -> Result<(), Box<dyn StdError>> {
    let body = encoded(caching_request(None)?)?;
    let marker = json!({ "cachePoint": { "type": "default" } });

    assert_eq!(body["system"][1], marker);
    assert_eq!(body["toolConfig"]["tools"][1], marker);
    // The second-to-last user turn, which is the first of the two.
    assert_eq!(body["messages"][0]["content"][1], marker);
    assert_eq!(
        body["messages"][2]["content"].as_array().map(Vec::len),
        Some(1)
    );
    Ok(())
}

#[test]
fn auto_cache_false_places_no_cache_points() -> Result<(), Box<dyn StdError>> {
    let body = encoded(caching_request(Some(false))?)?;

    assert!(!body.to_string().contains("cachePoint"), "{body}");
    assert_eq!(body["system"].as_array().map(Vec::len), Some(1));
    assert_eq!(
        body["toolConfig"]["tools"].as_array().map(Vec::len),
        Some(1)
    );
    Ok(())
}

/// A Bedrock provider whose one model declares no prompt caching.
///
/// Bedrock allows passthrough, so a caller can name any hosted family, and
/// most of them cannot cache. The built-in catalog carries no such entry,
/// so this test catalog supplies one.
const UNCACHED_CATALOG: &str = r#"
    schema_version = 1

    [providers.bedrock]
    display_name = "Amazon Bedrock"
    adapter = "bedrock"
    codecs = ["bedrock-converse"]
    base_url = "https://bedrock-runtime.us-east-1.amazonaws.com"
    default_model = "llama"
    auth = { type = "none" }

    [providers.bedrock.models.llama]
    display_name = "Llama 3 70B"
    api_model = "meta.llama3-70b-instruct-v1:0"
    capabilities = { text = true, tools = true, tool_choice = { required = true, named = true } }
"#;

#[test]
fn a_model_that_cannot_cache_gets_no_cache_points() -> Result<(), Box<dyn StdError>> {
    // A `cachePoint` block is not ignored by a family that cannot cache:
    // Converse rejects the whole request with a ValidationException, so
    // every call to such a model would fail.
    let mut builder = Request::builder()
        .model("bedrock/llama")
        .system("Be brief")
        .user("First")
        .message(Message::text(Role::Assistant, "Answer"))
        .user("Second")
        .tool(ToolDefinition::function("search", "Searches", json!({})));
    builder = builder.max_output_tokens(64);
    let call = resolved_in(UNCACHED_CATALOG, builder.build()?)?;

    let body = BedrockConverseCodec.encode(&call, false)?.body;
    assert!(!body.to_string().contains("cachePoint"), "{body}");

    // The count-tokens body describes the same prompt, so it drops the
    // markers with it.
    let counted = BedrockConverseCodec
        .encode_count_tokens(&call)
        .ok_or("expected Bedrock to have a count-tokens endpoint")??;
    assert!(
        !counted.body.to_string().contains("cachePoint"),
        "{}",
        counted.body
    );
    Ok(())
}

#[test]
fn a_tool_use_cut_at_the_output_limit_is_dropped() -> Result<(), Box<dyn StdError>> {
    let response = decoded(json!({
        "output": { "message": { "content": [
            { "toolUse": { "toolUseId": "tool-1", "name": "write_note", "input": {} } }
        ] } },
        "stopReason": "max_tokens",
    }))?;

    assert_eq!(response.content, Vec::new());
    assert_eq!(response.finish_reason, FinishReason::Length);
    assert_eq!(response.warnings.len(), 1);
    assert_eq!(response.warnings[0].code, "truncated_tool_call");
    Ok(())
}

#[test]
fn a_context_window_stop_is_a_length_finish() -> Result<(), Box<dyn StdError>> {
    // Converse's own name for a generation that ran out of context. It is
    // the same outcome as `max_tokens`, so it maps onto the same reason.
    let response = decoded(json!({
        "output": { "message": { "content": [{ "text": "Partial" }] } },
        "stopReason": "model_context_window_exceeded",
    }))?;

    assert_eq!(response.finish_reason, FinishReason::Length);
    Ok(())
}

#[test]
fn redacted_reasoning_round_trips_through_the_blocking_path() -> Result<(), Box<dyn StdError>> {
    let response = decoded(json!({
        "output": { "message": { "content": [
            { "reasoningContent": { "redactedContent": "sealed-blob" } },
        ] } },
        "stopReason": "end_turn",
    }))?;

    let Some(ContentPart::Reasoning(reasoning)) = response.content.first() else {
        return Err(format!("expected reasoning, got {:?}", response.content).into());
    };
    assert!(reasoning.redacted);
    assert_eq!(reasoning.text, "sealed-blob");

    let body = encoded(
        Request::builder()
            .model(MODEL)
            .message(Message::new(
                Role::Assistant,
                response.content.iter().cloned(),
            ))
            .build()?,
    )?;

    assert_eq!(
        body["messages"][0]["content"][0]["reasoningContent"]["redactedContent"],
        "sealed-blob"
    );
    Ok(())
}

#[test]
fn redacted_reasoning_round_trips_through_the_stream() -> Result<(), Box<dyn StdError>> {
    let events = streamed(&[
        ("messageStart", json!({ "role": "assistant" })),
        (
            "contentBlockDelta",
            json!({
                "contentBlockIndex": 0,
                "delta": { "reasoningContent": { "redactedContent": "sealed-blob" } },
            }),
        ),
        ("contentBlockStop", json!({ "contentBlockIndex": 0 })),
        ("messageStop", json!({ "stopReason": "end_turn" })),
    ])?;

    // The sealed blob must not leak through live reasoning deltas.
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, StreamEvent::ReasoningDelta { .. })),
        "a redacted block leaked a reasoning delta: {events:?}"
    );
    let responses = completed(&events);
    let [response] = responses.as_slice() else {
        return Err(format!("expected one Ended, got {}", responses.len()).into());
    };
    let Some(ContentPart::Reasoning(reasoning)) = response.content.first() else {
        return Err(format!("expected reasoning, got {:?}", response.content).into());
    };
    assert!(reasoning.redacted);
    assert_eq!(reasoning.text, "sealed-blob");
    Ok(())
}

#[test]
fn reasoning_text_after_a_redacted_blob_is_dropped() -> Result<(), Box<dyn StdError>> {
    // The blob is the payload the provider verifies on replay; text mixed
    // into the same buffer would corrupt it, so the blob wins.
    let events = streamed(&[
        ("messageStart", json!({ "role": "assistant" })),
        (
            "contentBlockDelta",
            json!({
                "contentBlockIndex": 0,
                "delta": { "reasoningContent": { "redactedContent": "sealed-blob" } },
            }),
        ),
        (
            "contentBlockDelta",
            json!({
                "contentBlockIndex": 0,
                "delta": { "reasoningContent": { "text": "stray text" } },
            }),
        ),
        ("contentBlockStop", json!({ "contentBlockIndex": 0 })),
        ("messageStop", json!({ "stopReason": "end_turn" })),
    ])?;

    let responses = completed(&events);
    let [response] = responses.as_slice() else {
        return Err(format!("expected one Ended, got {}", responses.len()).into());
    };
    let Some(ContentPart::Reasoning(reasoning)) = response.content.first() else {
        return Err(format!("expected reasoning, got {:?}", response.content).into());
    };
    assert!(reasoning.redacted);
    assert_eq!(reasoning.text, "sealed-blob");
    Ok(())
}

#[test]
fn a_redacted_blob_after_reasoning_text_fails_the_stream() -> Result<(), Box<dyn StdError>> {
    // The text was already delivered, so the blob cannot silently win;
    // assembling text and blob together replays a corrupted sealed
    // payload that Bedrock rejects next turn.
    let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;
    let mut decoder = BedrockConverseCodec.stream_decoder(call.route());
    decoder.decode(SseEvent {
        event: Some("messageStart".to_owned()),
        data:  json!({ "role": "assistant" }).to_string(),
    })?;
    decoder.decode(SseEvent {
        event: Some("contentBlockDelta".to_owned()),
        data:  json!({
            "contentBlockIndex": 0,
            "delta": { "reasoningContent": { "text": "step one" } },
        })
        .to_string(),
    })?;

    let error = decoder
        .decode(SseEvent {
            event: Some("contentBlockDelta".to_owned()),
            data:  json!({
                "contentBlockIndex": 0,
                "delta": { "reasoningContent": { "redactedContent": "sealed-blob" } },
            })
            .to_string(),
        })
        .expect_err("a blob landing on a text block must fail the stream");

    assert_eq!(error.kind(), ErrorKind::StreamDecode);
    assert_eq!(error.retry_classification(), RetryClassification::Safe);
    Ok(())
}

#[test]
fn the_content_block_index_becomes_a_stable_block_id() -> Result<(), Box<dyn StdError>> {
    let events = streamed(&[
        ("messageStart", json!({ "role": "assistant" })),
        (
            "contentBlockDelta",
            json!({ "contentBlockIndex": 3, "delta": { "text": "Hel" } }),
        ),
        (
            "contentBlockDelta",
            json!({ "contentBlockIndex": 3, "delta": { "text": "lo" } }),
        ),
        ("contentBlockStop", json!({ "contentBlockIndex": 3 })),
        ("messageStop", json!({ "stopReason": "end_turn" })),
    ])?;

    let ids: Vec<&str> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ContentBlockStart { id, .. }
            | StreamEvent::TextDelta { id, .. }
            | StreamEvent::ContentBlockEnd { id, .. } => Some(id.as_str()),
            _ => None,
        })
        .collect();

    assert_eq!(ids, ["block-3", "block-3", "block-3", "block-3"]);
    let responses = completed(&events);
    let [response] = responses.as_slice() else {
        return Err("expected exactly one Ended".into());
    };
    assert_eq!(response.text(), "Hello");
    Ok(())
}

#[test]
fn interleaved_tool_calls_keep_separate_identities() -> Result<(), Box<dyn StdError>> {
    let events = streamed(&[
        ("messageStart", json!({ "role": "assistant" })),
        (
            "contentBlockStart",
            json!({
                "contentBlockIndex": 0,
                "start": { "toolUse": { "toolUseId": "call-a", "name": "search" } },
            }),
        ),
        (
            "contentBlockStart",
            json!({
                "contentBlockIndex": 1,
                "start": { "toolUse": { "toolUseId": "call-b", "name": "lookup" } },
            }),
        ),
        (
            "contentBlockDelta",
            json!({
                "contentBlockIndex": 0,
                "delta": { "toolUse": { "input": "{\"q\":" } },
            }),
        ),
        (
            "contentBlockDelta",
            json!({
                "contentBlockIndex": 1,
                "delta": { "toolUse": { "input": "{\"id\":1}" } },
            }),
        ),
        (
            "contentBlockDelta",
            json!({
                "contentBlockIndex": 0,
                "delta": { "toolUse": { "input": "\"rust\"}" } },
            }),
        ),
        ("contentBlockStop", json!({ "contentBlockIndex": 0 })),
        ("contentBlockStop", json!({ "contentBlockIndex": 1 })),
        ("messageStop", json!({ "stopReason": "tool_use" })),
        (
            "metadata",
            json!({ "usage": { "inputTokens": 12, "outputTokens": 5 } }),
        ),
    ])?;

    let starts: Vec<(&str, String)> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ContentBlockStart { id, kind } => Some((id.as_str(), format!("{kind:?}"))),
            _ => None,
        })
        .collect();
    assert_eq!(starts.len(), 2);
    assert_eq!(starts[0].0, "block-0");
    assert!(starts[0].1.contains("call-a"), "{}", starts[0].1);
    assert_eq!(starts[1].0, "block-1");
    assert!(starts[1].1.contains("call-b"), "{}", starts[1].1);

    let responses = completed(&events);
    let [response] = responses.as_slice() else {
        return Err("expected exactly one Ended".into());
    };
    let calls: Vec<(&str, Value)> = response
        .content
        .iter()
        .filter_map(|part| match part {
            ContentPart::ToolCall(call) => Some((call.id.as_str(), call.input.wire_value())),
            _ => None,
        })
        .collect();
    assert_eq!(calls, [
        ("call-a", json!({ "q": "rust" })),
        ("call-b", json!({ "id": 1 })),
    ]);
    assert_eq!(response.finish_reason, FinishReason::ToolCall);
    Ok(())
}

#[test]
fn the_terminal_metadata_event_carries_usage_and_leaves_raw_unset() -> Result<(), Box<dyn StdError>>
{
    let events = streamed(&[
        ("messageStart", json!({ "role": "assistant" })),
        (
            "contentBlockDelta",
            json!({ "contentBlockIndex": 0, "delta": { "text": "hi" } }),
        ),
        ("contentBlockStop", json!({ "contentBlockIndex": 0 })),
        ("messageStop", json!({ "stopReason": "end_turn" })),
        (
            "metadata",
            json!({
                "usage": {
                    "inputTokens": 30,
                    "outputTokens": 628,
                    "cacheReadInputTokens": 1024,
                    "cacheWriteInputTokens": 512,
                },
            }),
        ),
    ])?;

    let usages: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::Usage { usage } => Some(*usage),
            _ => None,
        })
        .collect();
    let [usage] = usages.as_slice() else {
        return Err(format!("expected one Usage event, got {}", usages.len()).into());
    };
    assert_eq!(usage.input, 30);
    assert_eq!(usage.output, 628);
    assert_eq!(usage.reasoning, 0);
    assert_eq!(usage.cache_read, 1024);
    assert_eq!(usage.cache_write, 512);

    let responses = completed(&events);
    let [response] = responses.as_slice() else {
        return Err(format!("expected one Ended, got {}", responses.len()).into());
    };
    assert_eq!(response.usage, *usage);
    assert!(response.raw.is_none());
    Ok(())
}

#[test]
fn an_empty_text_delta_opens_no_text_block() -> Result<(), Box<dyn StdError>> {
    // An empty delta says nothing. Passing it on opened a text block that
    // carried no text, and the completed response ended with an empty part
    // that re-encodes to nothing.
    let events = streamed(&[
        ("messageStart", json!({ "role": "assistant" })),
        (
            "contentBlockDelta",
            json!({ "contentBlockIndex": 0, "delta": { "text": "" } }),
        ),
        ("contentBlockStop", json!({ "contentBlockIndex": 0 })),
        ("messageStop", json!({ "stopReason": "end_turn" })),
        ("metadata", json!({ "usage": { "inputTokens": 3 } })),
    ])?;

    assert!(
        !events
            .iter()
            .any(|event| matches!(event, StreamEvent::TextDelta { .. })),
        "{events:?}"
    );
    let responses = completed(&events);
    let [response] = responses.as_slice() else {
        return Err(format!("expected one Ended, got {}", responses.len()).into());
    };
    assert!(response.content.is_empty(), "{:?}", response.content);
    Ok(())
}

#[test]
fn an_unlabelled_frame_falls_back_to_its_single_top_level_key() -> Result<(), Box<dyn StdError>> {
    let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;
    let mut decoder = BedrockConverseCodec.stream_decoder(call.route());

    let events = decoder.decode(SseEvent {
        event: None,
        data:  json!({
            "contentBlockDelta": { "contentBlockIndex": 0, "delta": { "text": "hi" } },
        })
        .to_string(),
    })?;

    assert!(
        events.iter().any(|event| matches!(
            event,
            StreamEvent::TextDelta { text, .. } if text == "hi"
        )),
        "{events:?}"
    );
    Ok(())
}

#[test]
fn a_mid_stream_exception_payload_becomes_an_error() -> Result<(), Box<dyn StdError>> {
    let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;
    let mut decoder = BedrockConverseCodec.stream_decoder(call.route());

    let error = decoder
        .decode(SseEvent {
            event: Some("throttlingException".to_owned()),
            data:  json!({
                "__type": "com.amazon.bedrock#ThrottlingException",
                "message": "Too many requests",
            })
            .to_string(),
        })
        .err()
        .ok_or("expected an exception payload to fail the stream")?;

    assert_eq!(error.kind(), ErrorKind::RateLimit);
    assert_eq!(error.message(), "Too many requests");
    Ok(())
}

#[test]
fn the_count_tokens_request_has_the_documented_shape() -> Result<(), Box<dyn StdError>> {
    let call = resolved(
        Request::builder()
            .model(MODEL)
            .system("Be brief")
            .user("Hello")
            .max_output_tokens(64)
            .temperature(0.5)
            .provider_option("bedrock", "auto_cache", json!(false))
            .build()?,
    )?;

    let encoded = BedrockConverseCodec
        .encode_count_tokens(&call)
        .ok_or("expected Bedrock to have a count-tokens endpoint")??;

    assert!(
        encoded
            .url
            .ends_with("/model/us.anthropic.claude-sonnet-4-6/count-tokens"),
        "{}",
        encoded.url
    );
    assert_eq!(
        encoded.body,
        json!({
            "input": {
                "converse": {
                    "messages": [{ "role": "user", "content": [{ "text": "Hello" }] }],
                    "system": [{ "text": "Be brief" }],
                }
            }
        })
    );

    let tokens =
        BedrockConverseCodec.decode_count_tokens(call.route(), json!({ "inputTokens": 41 }))?;
    assert_eq!(tokens, 41);

    let error = BedrockConverseCodec
        .decode_count_tokens(call.route(), json!({}))
        .err()
        .ok_or("expected a malformed count body to fail")?;
    assert_eq!(error.kind(), ErrorKind::ResponseDecode);
    assert_eq!(error.retry_classification(), RetryClassification::Safe);
    Ok(())
}

#[test]
fn refuses_a_declared_media_type_it_cannot_name() -> Result<(), Box<dyn StdError>> {
    // Converse takes a format enum, so an unmapped type has nowhere to go.
    // Sending the bytes as `png` anyway would tell the provider something
    // the caller never said.
    let request = Request::builder()
        .model(MODEL)
        .message(Message::new(Role::User, [ContentPart::Image(
            ImageContent::new(MediaSource::base64("aW1n", "image/heic")),
        )]))
        .build()?;

    let Err(error) = BedrockConverseCodec.encode(&resolved(request)?, false) else {
        panic!("an unmapped media type should be refused");
    };

    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    assert!(error.message().contains("image/heic"));
    Ok(())
}

#[test]
fn a_declared_media_type_the_enum_names_is_used() -> Result<(), Box<dyn StdError>> {
    // The refusal above must not have swallowed the mapped types too.
    let request = Request::builder()
        .model(MODEL)
        .message(Message::new(Role::User, [ContentPart::Image(
            ImageContent::new(MediaSource::base64("aW1n", "image/png")),
        )]))
        .build()?;
    let body = encoded(request)?;

    assert_eq!(body["messages"][0]["content"][0]["image"]["format"], "png");
    Ok(())
}

#[test]
fn reports_controls_converse_cannot_express() -> Result<(), Box<dyn StdError>> {
    let request = Request::builder()
        .model(MODEL)
        .user("Answer as JSON.")
        .response_format(ResponseFormat::JsonObject)
        .metadata_entry("tenant", "acme")
        .build()?;

    let codes = warnings_for(request)?;

    assert!(codes.iter().all(|code| code == "unsupported_control"));
    assert_eq!(codes.len(), 2, "metadata and response format both warn");
    Ok(())
}

#[test]
fn a_json_tool_result_reaches_the_wire_unflattened() -> Result<(), Box<dyn StdError>> {
    // `toolResult.content` is a block list, and one of its members is
    // `json`, so structured JSON reaches the model as itself. Warning
    // about it said the opposite of what the encoder does, and every
    // agent-loop request with a structured result carried the false
    // warning.
    let request = Request::builder()
        .model(MODEL)
        .user("Chart it.")
        .message(Message::new(Role::Tool, [ContentPart::ToolResult(
            ToolResult {
                tool_call_id: "call-1".to_owned(),
                name:         Some("chart".to_owned()),
                content:      vec![ContentPart::Json {
                    value: json!({ "quarters": [1, 2] }),
                }],
                is_error:     false,
            },
        )]))
        .build()?;

    let codes = warnings_for(request.clone())?;
    assert!(codes.is_empty(), "{codes:?}");

    let body = encoded(request)?;
    assert_eq!(
        body["messages"][0]["content"][1]["toolResult"]["content"][0]["json"],
        json!({ "quarters": [1, 2] })
    );
    Ok(())
}

#[test]
fn a_reasoning_tool_result_part_is_dropped_and_reported() -> Result<(), Box<dyn StdError>> {
    // `reasoningContent` has no member of the tool-result union, so
    // encoding it there made Converse reject the whole request. The part
    // is dropped and the caller is told, which is the rule for content the
    // protocol cannot carry.
    let request = Request::builder()
        .model(MODEL)
        .user("Chart it.")
        .message(Message::new(Role::Tool, [ContentPart::ToolResult(
            ToolResult {
                tool_call_id: "call-1".to_owned(),
                name:         Some("chart".to_owned()),
                content:      vec![
                    ContentPart::Reasoning(ReasoningContent {
                        text:             "the third quarter is the outlier".to_owned(),
                        signature:        None,
                        signature_origin: None,
                        redacted:         false,
                    }),
                    ContentPart::Text {
                        text: "Q3 leads.".to_owned(),
                    },
                ],
                is_error:     false,
            },
        )]))
        .build()?;

    let codes = warnings_for(request.clone())?;
    assert_eq!(codes, ["unsupported_control"]);

    let body = encoded(request)?;
    let content = &body["messages"][0]["content"][1]["toolResult"]["content"];
    assert_eq!(content, &json!([{ "text": "Q3 leads." }]));
    Ok(())
}

#[test]
fn refuses_a_tool_name_converse_rejects() -> Result<(), Box<dyn StdError>> {
    // Rewriting `mcp.server.tool` to `mcp_server_tool` is lossy, so a
    // decoded call could not be mapped back to the registered tool. The
    // caller owns that mapping because only they can undo it.
    let request = Request::builder()
        .model(MODEL)
        .user("Use the tool.")
        .tool(ToolDefinition::function(
            "mcp.server.tool",
            "A dotted MCP-style name",
            json!({ "type": "object" }),
        ))
        .build()?;

    let Err(error) = BedrockConverseCodec.encode(&resolved(request)?, false) else {
        panic!("a tool name Converse rejects should be refused");
    };

    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    assert!(error.message().contains("mcp.server.tool"));
    Ok(())
}

#[test]
fn accepts_a_tool_name_converse_allows() -> Result<(), Box<dyn StdError>> {
    let request = Request::builder()
        .model(MODEL)
        .user("Use the tool.")
        .tool(ToolDefinition::function(
            "mcp_server_tool-2",
            "An allowed name",
            json!({ "type": "object" }),
        ))
        .build()?;

    let body = encoded(request)?;

    assert_eq!(
        body["toolConfig"]["tools"][0]["toolSpec"]["name"],
        "mcp_server_tool-2"
    );
    Ok(())
}

/// A conversation replaying one historical tool call and its result.
fn replayed_tool_request(name: &str, id: &str) -> Result<Request, Box<dyn StdError>> {
    Ok(Request::builder()
        .model(MODEL)
        .user("What is the weather?")
        .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
            ToolCall::function(id, name, json!({ "city": "Paris" })),
        )]))
        .message(Message::new(Role::Tool, [ContentPart::ToolResult(
            ToolResult {
                tool_call_id: id.to_owned(),
                name:         Some(name.to_owned()),
                content:      vec![ContentPart::Text {
                    text: "18C".to_owned(),
                }],
                is_error:     false,
            },
        )]))
        .build()?)
}

#[test]
fn refuses_a_historical_tool_name_converse_rejects() -> Result<(), Box<dyn StdError>> {
    // A conversation that began on another provider carries that
    // provider's identifiers. Sending a dotted name back to Converse dies
    // at AWS with an opaque ValidationException, so it is refused here
    // with the offending value named.
    let request = replayed_tool_request("mcp.server.tool", "call_1")?;

    let Err(error) = BedrockConverseCodec.encode(&resolved(request)?, false) else {
        panic!("a historical tool name Converse rejects should be refused");
    };

    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    assert!(error.message().contains("mcp.server.tool"), "{error}");
    Ok(())
}

#[test]
fn refuses_a_historical_tool_call_id_converse_rejects() -> Result<(), Box<dyn StdError>> {
    // 65 characters, one over the Converse limit. Counting is the whole
    // check here: every character is allowed.
    let long_id = "a".repeat(65);
    let request = replayed_tool_request("get_weather", &long_id)?;
    let call = resolved(request)?;

    let Err(error) = BedrockConverseCodec.encode(&call, false) else {
        panic!("a historical tool call id Converse rejects should be refused");
    };
    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert!(error.message().contains(&long_id), "{error}");

    // Counting sends the same message blocks, so it refuses the same
    // request rather than returning a count for a call that cannot be
    // made.
    let counted = BedrockConverseCodec
        .encode_count_tokens(&call)
        .ok_or("expected Bedrock to have a count-tokens endpoint")?;
    assert_eq!(
        counted.err().map(|error| error.kind()),
        Some(ErrorKind::InvalidRequest)
    );
    Ok(())
}

#[test]
fn accepts_a_tool_call_id_with_dots_and_colons() -> Result<(), Box<dyn StdError>> {
    // `toolUseId` allows `.` and `:` on top of the tool-name set, so an id
    // in that shape replays unchanged.
    let body = encoded(replayed_tool_request(
        "get_weather",
        "functions.get_weather:4",
    )?)?;

    assert_eq!(
        body["messages"][1]["content"][0]["toolUse"]["toolUseId"],
        "functions.get_weather:4"
    );
    assert_eq!(
        body["messages"][2]["content"][0]["toolResult"]["toolUseId"],
        "functions.get_weather:4"
    );
    Ok(())
}

#[test]
fn refuses_a_tool_call_id_on_a_tool_message() -> Result<(), Box<dyn StdError>> {
    // A tool result can ride on the message rather than in a `ToolResult`
    // part, and that id reaches the wire the same way.
    let request = Request::builder()
        .model(MODEL)
        .user("What is the weather?")
        .message(Message::text(Role::Tool, "18C").with_tool_call_id("call one"))
        .build()?;

    let Err(error) = BedrockConverseCodec.encode(&resolved(request)?, false) else {
        panic!("a message tool call id Converse rejects should be refused");
    };

    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert!(error.message().contains("call one"), "{error}");
    Ok(())
}

#[test]
fn a_tool_call_keeps_arguments_that_are_an_object() -> Result<(), Box<dyn StdError>> {
    let request = Request::builder()
        .model(MODEL)
        .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
            ToolCall::function("call-1", "search", json!({ "q": "rust" })),
        )]))
        .build()?;

    let body = encoded(request)?;

    assert_eq!(
        body["messages"][0]["content"][0]["toolUse"]["input"],
        json!({ "q": "rust" })
    );
    Ok(())
}

#[test]
fn a_tool_call_coerces_non_object_arguments_to_an_object() -> Result<(), Box<dyn StdError>> {
    // Converse requires `toolUse.input` to be a JSON object document, so a
    // replayed scalar or array argument value becomes `{}` rather than
    // reaching the wire as a value AWS rejects with a ValidationException.
    for arguments in [json!([1, 2, 3]), json!("rust"), json!(7)] {
        let request = Request::builder()
            .model(MODEL)
            .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
                ToolCall::function("call-1", "sum", arguments),
            )]))
            .build()?;

        let body = encoded(request)?;

        assert_eq!(
            body["messages"][0]["content"][0]["toolUse"]["input"],
            json!({})
        );
    }
    Ok(())
}

#[test]
fn a_tool_call_with_no_arguments_still_sends_an_object() -> Result<(), Box<dyn StdError>> {
    let request = Request::builder()
        .model(MODEL)
        .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
            ToolCall::function("call-1", "ping", Value::Null),
        )]))
        .build()?;

    let body = encoded(request)?;

    assert_eq!(
        body["messages"][0]["content"][0]["toolUse"]["input"],
        json!({})
    );
    Ok(())
}

#[test]
fn non_text_system_content_is_reported() -> Result<(), Box<dyn StdError>> {
    let request = Request::builder()
        .model(MODEL)
        .message(Message::new(Role::System, [
            ContentPart::Text {
                text: "Be brief.".to_owned(),
            },
            ContentPart::Json {
                value: json!({ "style": "terse" }),
            },
        ]))
        .user("Hello")
        .build()?;

    let codes = warnings_for(request)?;

    assert_eq!(codes, ["unsupported_control"]);
    Ok(())
}
