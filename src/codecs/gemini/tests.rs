use std::error::Error as StdError;

use serde_json::{Map, Value, json};

use super::{Codec as _, GeminiGenerateCodec};
use crate::codecs::test_support::resolved;
use crate::transport::SseEvent;
use crate::types::{
    AudioContent, ContentPart, DocumentContent, Error, ErrorKind, FinishReason, ImageContent,
    MediaSource, Message, ReasoningEffort, Request, Response, ResponseFormat, RetryClassification,
    Role, Speed, StreamEvent, ToolCall, ToolDefinition, ToolResult,
};

fn object(value: Value) -> Result<Map<String, Value>, Box<dyn StdError>> {
    match value {
        Value::Object(map) => Ok(map),
        other => Err(format!("expected a JSON object, got {other}").into()),
    }
}

/// Decodes one blocking response from a `generateContent` payload.
fn decode(payload: Value) -> Result<Response, Box<dyn StdError>> {
    let call = resolved(
        Request::builder()
            .model("gemini/gemini-3.1-pro-preview")
            .user("Hello")
            .build()?,
    )?;

    Ok(GeminiGenerateCodec.decode_response(call.route(), payload)?)
}

/// Decodes one payload that must fail, and returns the codec's own error.
fn decode_failure(payload: Value) -> Result<Error, Box<dyn StdError>> {
    let call = resolved(
        Request::builder()
            .model("gemini/gemini-3.1-pro-preview")
            .user("Hello")
            .build()?,
    )?;

    GeminiGenerateCodec
        .decode_response(call.route(), payload)
        .err()
        .ok_or_else(|| "the payload should not decode into a response".into())
}

/// Runs a whole stream and returns every event it produced.
fn stream(chunks: &[Value]) -> Result<Vec<StreamEvent>, Box<dyn StdError>> {
    let call = resolved(
        Request::builder()
            .model("gemini/gemini-3.1-pro-preview")
            .user("Hello")
            .build()?,
    )?;
    let mut decoder = GeminiGenerateCodec.stream_decoder(call.route());

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

#[test]
fn each_streamed_thought_signature_seals_its_own_block() -> Result<(), Box<dyn StdError>> {
    // Two signed thought parts in one contiguous reasoning run. Each
    // signature is a complete blob; concatenating them would produce a
    // signature Gemini rejects on replay, so each signed part must end
    // as its own reasoning part — the blocking decoder's shape.
    let events = stream(&[json!({
        "responseId": "resp-1",
        "candidates": [{
            "content": { "parts": [
                { "text": "step one", "thought": true, "thoughtSignature": "sig-a" },
                { "text": "step two", "thought": true, "thoughtSignature": "sig-b" },
            ] },
            "finishReason": "STOP",
        }],
    })])?;

    let parts: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ContentBlockEnd { part, .. } => Some(part.clone()),
            _ => None,
        })
        .collect();
    let signatures: Vec<_> = parts
        .iter()
        .map(|part| match part {
            ContentPart::Reasoning(reasoning) => {
                Ok((reasoning.text.as_str(), reasoning.signature.as_deref()))
            }
            other => Err(format!("expected a reasoning part, got {other:?}")),
        })
        .collect::<Result<_, _>>()?;
    assert_eq!(signatures, vec![
        ("step one", Some("sig-a")),
        ("step two", Some("sig-b")),
    ]);
    Ok(())
}

#[test]
fn encodes_media_sources_and_function_response_identity() -> Result<(), Box<dyn StdError>> {
    let call = resolved(
        Request::builder()
            .model("gemini/gemini-3.1-pro-preview")
            .message(Message::new(Role::User, [
                ContentPart::Image(ImageContent::new(MediaSource::base64(
                    "aGVsbG8=",
                    "image/png",
                ))),
                ContentPart::Image(ImageContent::new(MediaSource::url(
                    "https://example.com/cat.png",
                ))),
                ContentPart::Document(DocumentContent::new(MediaSource::url_with_media_type(
                    "https://example.com/report.pdf",
                    "application/pdf",
                ))),
            ]))
            .message(Message::new(Role::Tool, [ContentPart::ToolResult(
                ToolResult {
                    tool_call_id: "call-1".to_owned(),
                    name:         Some("weather".to_owned()),
                    content:      vec![ContentPart::Text {
                        text: "sunny".to_owned(),
                    }],
                    is_error:     false,
                },
            )]))
            .build()?,
    )?;

    let encoded = GeminiGenerateCodec.encode(&call, false)?;

    assert_eq!(
        encoded.body["contents"][0]["parts"][0]["inlineData"],
        json!({ "mimeType": "image/png", "data": "aGVsbG8=" })
    );
    assert_eq!(
        encoded.body["contents"][0]["parts"][1]["fileData"],
        json!({ "mimeType": "image/png", "fileUri": "https://example.com/cat.png" })
    );
    // A declared media type reaches the wire: Vertex-style surfaces
    // require `mimeType` on file references.
    assert_eq!(
        encoded.body["contents"][0]["parts"][2]["fileData"],
        json!({
            "fileUri": "https://example.com/report.pdf",
            "mimeType": "application/pdf"
        })
    );
    // The media message and the tool result both map to the `user` role,
    // so they merge into one turn rather than two consecutive ones.
    assert_eq!(encoded.body["contents"].as_array().map(Vec::len), Some(1));
    assert_eq!(
        encoded.body["contents"][0]["parts"][3]["functionResponse"]["id"],
        "call-1"
    );
    assert_eq!(
        encoded.body["contents"][0]["parts"][3]["functionResponse"]["name"],
        "weather"
    );
    Ok(())
}

/// Encodes one user message holding `part` and returns the `fileData`
/// member of the part it produced.
fn encoded_file_data(part: ContentPart) -> Result<Value, Box<dyn StdError>> {
    let call = resolved(
        Request::builder()
            .model("gemini/gemini-3.1-pro-preview")
            .message(Message::new(Role::User, [part]))
            .build()?,
    )?;

    let encoded = GeminiGenerateCodec.encode(&call, false)?;
    Ok(encoded.body["contents"][0]["parts"][0]["fileData"].clone())
}

#[test]
fn a_url_image_defaults_its_mime_type() -> Result<(), Box<dyn StdError>> {
    let file_data = encoded_file_data(ContentPart::Image(ImageContent::new(MediaSource::url(
        "https://example.com/cat.png",
    ))))?;

    assert_eq!(
        file_data,
        json!({ "mimeType": "image/png", "fileUri": "https://example.com/cat.png" })
    );
    Ok(())
}

#[test]
fn a_url_audio_defaults_its_mime_type() -> Result<(), Box<dyn StdError>> {
    let file_data = encoded_file_data(ContentPart::Audio(AudioContent::new(MediaSource::url(
        "https://example.com/note.wav",
    ))))?;

    assert_eq!(
        file_data,
        json!({ "mimeType": "audio/wav", "fileUri": "https://example.com/note.wav" })
    );
    Ok(())
}

#[test]
fn a_url_document_defaults_its_mime_type() -> Result<(), Box<dyn StdError>> {
    let file_data = encoded_file_data(ContentPart::Document(DocumentContent::new(
        MediaSource::url("https://example.com/report.pdf"),
    )))?;

    assert_eq!(
        file_data,
        json!({ "mimeType": "application/pdf", "fileUri": "https://example.com/report.pdf" })
    );
    Ok(())
}

#[test]
fn usage_buckets_are_disjoint() -> Result<(), Box<dyn StdError>> {
    let response = decode(json!({
        "candidates": [{ "content": { "parts": [{ "text": "Done." }] } }],
        "usageMetadata": {
            "promptTokenCount": 200,
            "cachedContentTokenCount": 180,
            "toolUsePromptTokenCount": 400,
            "candidatesTokenCount": 200,
            "thoughtsTokenCount": 300,
        }
    }))?;

    // (200 - 180) + 400: the cached tokens are inside `promptTokenCount`
    // and the tool-use tokens are outside it.
    assert_eq!(response.usage.input, 420);
    assert_eq!(response.usage.cache_read, 180);
    // `candidatesTokenCount` never contained the thoughts, so nothing is
    // subtracted from the output bucket.
    assert_eq!(response.usage.output, 200);
    assert_eq!(response.usage.reasoning, 300);
    assert_eq!(response.usage.cache_write, 0);
    assert_eq!(response.usage.total(), 1100);
    Ok(())
}

/// The synthesized tool-call ids of one decoded response.
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

#[test]
fn synthesized_tool_call_ids_are_deterministic() -> Result<(), Box<dyn StdError>> {
    let payload = json!({
        "responseId": "resp-1",
        "candidates": [{ "content": { "parts": [
            { "functionCall": { "name": "search", "args": { "query": "rust" } } },
            { "text": "and then" },
            { "functionCall": { "name": "search", "args": { "query": "gemini" } } },
        ] } }]
    });

    let first = decode(payload.clone())?;
    let second = decode(payload)?;

    assert_eq!(call_ids(&first), ["search-0-resp-1", "search-1-resp-1"]);
    assert_eq!(first.content, second.content);
    Ok(())
}

#[test]
fn ids_without_a_response_id_do_not_collide_across_decodes() -> Result<(), Box<dyn StdError>> {
    // Without a `responseId` there is nothing deterministic to scope by,
    // and the bare `{name}-{ordinal}` form repeats across turns: a loop
    // calling `search` once per turn carried `search-0` twice. A random
    // nonce per decode keeps each response's ids unique instead.
    let payload = json!({
        "candidates": [{ "content": { "parts": [
            { "functionCall": { "name": "search", "args": { "query": "rust" } } },
        ] } }]
    });

    let first = decode(payload.clone())?;
    let second = decode(payload)?;

    let [first_id] = call_ids(&first)[..] else {
        return Err(format!("unexpected content {:?}", first.content).into());
    };
    let [second_id] = call_ids(&second)[..] else {
        return Err(format!("unexpected content {:?}", second.content).into());
    };
    assert!(
        first_id.starts_with("search-0-"),
        "the readable prefix must survive: {first_id}"
    );
    assert_ne!(
        first_id, second_id,
        "two responses without a responseId must not share ids"
    );
    Ok(())
}

#[test]
fn a_response_id_scopes_the_synthesized_ids() -> Result<(), Box<dyn StdError>> {
    // The bare `{name}-{ordinal}` form repeats across turns, so the same
    // conversation would carry `search-0` twice. The response id is the
    // provider's own name for one response, so it separates them without
    // costing determinism.
    let response = decode(json!({
        "responseId": "resp-1",
        "candidates": [{ "content": { "parts": [
            { "functionCall": { "name": "search", "args": {} } },
        ] } }]
    }))?;

    let [ContentPart::ToolCall(call)] = response.content.as_slice() else {
        return Err(format!("unexpected content {:?}", response.content).into());
    };
    assert_eq!(call.id, "search-0-resp-1");
    Ok(())
}

#[test]
fn a_function_call_turn_finishes_as_a_tool_call() -> Result<(), Box<dyn StdError>> {
    // Gemini says STOP even when the whole turn is a function call.
    let response = decode(json!({
        "candidates": [{
            "content": { "parts": [{ "functionCall": { "name": "search", "args": {} } }] },
            "finishReason": "STOP",
        }]
    }))?;

    assert_eq!(response.finish_reason, FinishReason::ToolCall);
    Ok(())
}

#[test]
fn a_truncated_function_call_keeps_the_provider_reason() -> Result<(), Box<dyn StdError>> {
    // Only `STOP` is corrected. `MAX_TOKENS` is the more specific fact and
    // survives, so a cut-off call is not reported as a complete one — and
    // the cut-off call itself is dropped, with a warning in its place.
    let response = decode(json!({
        "candidates": [{
            "content": { "parts": [{ "functionCall": { "name": "search", "args": {} } }] },
            "finishReason": "MAX_TOKENS",
        }]
    }))?;

    assert_eq!(response.finish_reason, FinishReason::Length);
    assert_eq!(response.content, Vec::new());
    assert_eq!(response.warnings.len(), 1);
    assert_eq!(response.warnings[0].code, "truncated_tool_call");
    Ok(())
}

#[test]
fn a_blocked_prompt_is_a_content_filter_error() -> Result<(), Box<dyn StdError>> {
    let error = decode_failure(json!({ "promptFeedback": { "blockReason": "SAFETY" } }))?;

    assert_eq!(error.kind(), ErrorKind::ContentFilter);
    assert_eq!(error.provider_code(), Some("SAFETY"));
    assert_eq!(error.retry_classification(), RetryClassification::Never);
    Ok(())
}

#[test]
fn every_block_reason_classifies_as_a_content_filter() -> Result<(), Box<dyn StdError>> {
    // The classifier knows `SAFETY` by name; it does not know the rest. The
    // message names the content policy so the whole family lands in one
    // place, whatever Google spells the reason.
    for reason in ["BLOCKLIST", "PROHIBITED_CONTENT", "IMAGE_SAFETY", "OTHER"] {
        let error = decode_failure(json!({ "promptFeedback": { "blockReason": reason } }))?;

        assert_eq!(error.kind(), ErrorKind::ContentFilter, "{reason}");
        assert_eq!(error.provider_code(), Some(reason), "{reason}");
    }
    Ok(())
}

#[test]
fn a_body_with_no_candidates_and_no_block_reason_fails_to_decode() -> Result<(), Box<dyn StdError>>
{
    let error = decode_failure(json!({ "responseId": "resp-1" }))?;

    assert_eq!(error.kind(), ErrorKind::ResponseDecode);
    assert_eq!(error.retry_classification(), RetryClassification::Safe);
    Ok(())
}

#[test]
fn an_explicit_null_error_member_is_not_an_error() -> Result<(), Box<dyn StdError>> {
    // A gateway that spells out `"error": null` on success chunks must
    // not fail every stream it serves.
    let call = resolved(
        Request::builder()
            .model("gemini/gemini-3.1-pro-preview")
            .user("Hello")
            .build()?,
    )?;
    let mut decoder = GeminiGenerateCodec.stream_decoder(call.route());

    let events = decoder.decode(SseEvent {
        event: None,
        data:  json!({
            "error": null,
            "candidates": [{ "content": { "parts": [{ "text": "hi" }] } }],
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
fn a_streamed_blocked_prompt_fails_like_the_blocking_path() -> Result<(), Box<dyn StdError>> {
    // A blocked prompt must not stream as an empty success while the
    // blocking path reports a content filter for the same body.
    let call = resolved(
        Request::builder()
            .model("gemini/gemini-3.1-pro-preview")
            .user("Hello")
            .build()?,
    )?;
    let mut decoder = GeminiGenerateCodec.stream_decoder(call.route());

    let error = decoder
        .decode(SseEvent {
            event: None,
            data:  json!({ "promptFeedback": { "blockReason": "SAFETY" } }).to_string(),
        })
        .expect_err("a blocked prompt must fail the stream");

    assert_eq!(error.kind(), ErrorKind::ContentFilter);
    assert_eq!(error.provider_code(), Some("SAFETY"));
    assert_eq!(error.retry_classification(), RetryClassification::Never);
    Ok(())
}

#[test]
fn a_stream_chunk_that_is_not_json_fails_retryably() -> Result<(), Box<dyn StdError>> {
    let call = resolved(
        Request::builder()
            .model("gemini/gemini-3.1-pro-preview")
            .user("Hello")
            .build()?,
    )?;
    let mut decoder = GeminiGenerateCodec.stream_decoder(call.route());

    let error = decoder
        .decode(SseEvent {
            event: None,
            data:  "not json".to_owned(),
        })
        .expect_err("a garbled chunk must fail the stream");

    assert_eq!(error.kind(), ErrorKind::StreamDecode);
    assert_eq!(error.retry_classification(), RetryClassification::Safe);
    Ok(())
}

#[test]
fn decodes_thought_signatures_onto_reasoning_and_tool_calls() -> Result<(), Box<dyn StdError>> {
    let response = decode(json!({
        "candidates": [{ "content": { "parts": [
            { "text": "step one", "thought": true, "thoughtSignature": "sig-think" },
            { "functionCall": { "name": "search", "args": {} },
              "thoughtSignature": "sig-call" },
        ] } }]
    }))?;

    let [
        ContentPart::Reasoning(reasoning),
        ContentPart::ToolCall(call),
    ] = response.content.as_slice()
    else {
        return Err(format!("unexpected content {:?}", response.content).into());
    };
    assert_eq!(reasoning.signature.as_deref(), Some("sig-think"));
    assert_eq!(
        call.provider_metadata.get("gemini"),
        Some(&json!({ "thoughtSignature": "sig-call" }))
    );
    Ok(())
}

#[test]
fn raw_options_win_and_only_this_namespace_is_read() -> Result<(), Box<dyn StdError>> {
    let call = resolved(
        Request::builder()
            .model("gemini/gemini-3.1-pro-preview")
            .user("Hello")
            .temperature(0.2)
            .max_output_tokens(64)
            .provider_options(
                "gemini",
                object(json!({
                    "generationConfig": { "temperature": 0.9 },
                    "safetySettings": [{ "category": "HARM_CATEGORY_HARASSMENT" }],
                    "auto_cache": false,
                }))?,
            )
            .provider_option("openai", "temperature", json!(0.1))
            .build()?,
    )?;

    let encoded = GeminiGenerateCodec.encode(&call, false)?;

    assert_eq!(encoded.body["generationConfig"]["temperature"], json!(0.9));
    assert_eq!(
        encoded.body["generationConfig"]["maxOutputTokens"],
        json!(64)
    );
    assert_eq!(
        encoded.body["safetySettings"],
        json!([{ "category": "HARM_CATEGORY_HARASSMENT" }])
    );
    let body = object(encoded.body)?;
    assert!(!body.contains_key("auto_cache"));
    Ok(())
}

#[test]
fn stop_sequences_land_in_the_generation_config() -> Result<(), Box<dyn StdError>> {
    let call = resolved(
        Request::builder()
            .model("gemini/gemini-3.1-pro-preview")
            .user("Hello")
            .stop_sequences(["STOP", "END"])
            .metadata_entry("session", "abc")
            .build()?,
    )?;

    let encoded = GeminiGenerateCodec.encode(&call, false)?;

    assert_eq!(
        encoded.body["generationConfig"]["stopSequences"],
        json!(["STOP", "END"])
    );
    // This protocol has no request metadata field, so nothing is sent and
    // the loss is reported as a warning instead.
    let codes: Vec<&str> = encoded
        .warnings
        .iter()
        .map(|warning| warning.code.as_str())
        .collect();
    assert_eq!(codes, ["unsupported_control"]);
    let body = object(encoded.body)?;
    assert!(!body.contains_key("metadata"));
    Ok(())
}

#[test]
fn a_text_response_format_leaves_the_output_mode_alone() -> Result<(), Box<dyn StdError>> {
    let call = resolved(
        Request::builder()
            .model("gemini/gemini-3.1-pro-preview")
            .user("Hello")
            .response_format(ResponseFormat::Text)
            .build()?,
    )?;

    let encoded = GeminiGenerateCodec.encode(&call, false)?;

    // `Text` is the protocol's default. Declaring `application/json` for it
    // would make the model answer in JSON to a caller who asked for prose.
    assert_eq!(
        encoded.body["generationConfig"].get("responseMimeType"),
        None
    );
    Ok(())
}

#[test]
fn controls_an_unclaimed_route_cannot_carry_are_reported() -> Result<(), Box<dyn StdError>> {
    // A model id the catalog does not list resolves as a passthrough
    // route, whose capabilities are unknown rather than declared.
    let call = resolved(
        Request::builder()
            .model("gemini/gemini-unlisted")
            .user("Hello")
            .speed(Speed::Fast)
            .reasoning_effort(ReasoningEffort::High)
            .build()?,
    )?;

    let encoded = GeminiGenerateCodec.encode(&call, false)?;

    let messages: Vec<&str> = encoded
        .warnings
        .iter()
        .map(|warning| warning.message.as_str())
        .collect();
    assert_eq!(messages, [
        "this provider protocol does not support the speed control",
        "this provider protocol does not support the reasoning effort control",
    ]);
    // Neither control may be guessed at for this passthrough route.
    let body = encoded.body.to_string();
    assert!(!body.contains("thinkingConfig"), "{body}");
    assert!(!body.contains("speed"), "{body}");
    Ok(())
}

#[test]
fn maps_normalized_effort_onto_gemini_thinking_levels() -> Result<(), Box<dyn StdError>> {
    let cases = [
        (ReasoningEffort::Minimal, "low"),
        (ReasoningEffort::Low, "low"),
        (ReasoningEffort::Medium, "medium"),
        (ReasoningEffort::High, "high"),
        (ReasoningEffort::Xhigh, "high"),
        (ReasoningEffort::Max, "high"),
    ];

    for (effort, expected) in cases {
        let call = resolved(
            Request::builder()
                .model("gemini/gemini-3.5-flash")
                .user("Hello")
                .reasoning_effort(effort)
                .build()?,
        )?;

        let encoded = GeminiGenerateCodec.encode(&call, false)?;

        assert_eq!(
            encoded.body["generationConfig"]["thinkingConfig"]["thinkingLevel"], expected,
            "{effort:?}"
        );
        assert!(
            encoded.warnings.is_empty(),
            "a supported effort must not warn: {:?}",
            encoded.warnings
        );
    }
    Ok(())
}

#[test]
fn a_raw_thinking_config_overrides_and_extends_the_effort_level() -> Result<(), Box<dyn StdError>> {
    let call = resolved(
        Request::builder()
            .model("gemini/gemini-3.5-flash")
            .user("Hello")
            .reasoning_effort(ReasoningEffort::High)
            .provider_options(
                "gemini",
                object(json!({
                    "generationConfig": {
                        "thinkingConfig": {
                            "thinkingLevel": "low",
                            "includeThoughts": true,
                        },
                    },
                }))?,
            )
            .build()?,
    )?;

    let encoded = GeminiGenerateCodec.encode(&call, false)?;

    assert_eq!(
        encoded.body["generationConfig"]["thinkingConfig"],
        json!({ "thinkingLevel": "low", "includeThoughts": true })
    );
    assert!(encoded.warnings.is_empty());
    Ok(())
}

#[test]
fn a_tool_result_recovers_its_function_name_from_the_call() -> Result<(), Box<dyn StdError>> {
    // A result that kept no name would otherwise send the call id as the
    // function name, which matches no declared function.
    let call = resolved(
        Request::builder()
            .model("gemini/gemini-3.1-pro-preview")
            .user("What is the weather?")
            .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
                ToolCall::function("call-1", "get_weather", json!({ "city": "Paris" })),
            )]))
            .message(Message::new(Role::Tool, [ContentPart::ToolResult(
                ToolResult {
                    tool_call_id: "call-1".to_owned(),
                    name:         None,
                    content:      vec![ContentPart::Text {
                        text: "18C".to_owned(),
                    }],
                    is_error:     false,
                },
            )]))
            .build()?,
    )?;

    let encoded = GeminiGenerateCodec.encode(&call, false)?;

    assert_eq!(
        encoded.body["contents"][2]["parts"][0]["functionResponse"]["name"],
        "get_weather"
    );
    Ok(())
}

#[test]
fn noncanonical_top_level_thought_signatures_are_ignored() -> Result<(), Box<dyn StdError>> {
    let mut replayed = ToolCall::function("call-1", "search", json!({ "q": "rust" }));
    replayed
        .provider_metadata
        .insert("thoughtSignature".to_owned(), json!("sig-legacy"));
    let call = resolved(
        Request::builder()
            .model("gemini/gemini-3.1-pro-preview")
            .user("Search for rust")
            .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
                replayed,
            )]))
            .message(Message::new(Role::Tool, [ContentPart::ToolResult(
                ToolResult {
                    tool_call_id: "call-1".to_owned(),
                    name:         Some("search".to_owned()),
                    content:      vec![ContentPart::Text {
                        text: "found".to_owned(),
                    }],
                    is_error:     false,
                },
            )]))
            .build()?,
    )?;

    let encoded = GeminiGenerateCodec.encode(&call, false)?;

    assert!(
        encoded.body["contents"][1]["parts"][0]
            .get("thoughtSignature")
            .is_none()
    );
    Ok(())
}

#[test]
fn a_caller_without_safety_settings_gets_the_default() -> Result<(), Box<dyn StdError>> {
    let call = resolved(
        Request::builder()
            .model("gemini/gemini-3.1-pro-preview")
            .user("Hello")
            .build()?,
    )?;

    let encoded = GeminiGenerateCodec.encode(&call, false)?;

    assert_eq!(
        encoded.body["safetySettings"],
        json!([{
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_ONLY_HIGH",
        }])
    );
    Ok(())
}

#[test]
fn a_snake_case_safety_setting_is_not_duplicated() -> Result<(), Box<dyn StdError>> {
    // Proto-JSON reads `safety_settings` and `safetySettings` as one field,
    // so adding the default beside a caller's snake_case list would send
    // the same field twice.
    let call = resolved(
        Request::builder()
            .model("gemini/gemini-3.1-pro-preview")
            .user("Hello")
            .provider_option(
                "gemini",
                "safety_settings",
                json!([{ "category": "HARM_CATEGORY_HARASSMENT" }]),
            )
            .build()?,
    )?;

    let encoded = GeminiGenerateCodec.encode(&call, false)?;

    let body = object(encoded.body)?;
    assert!(!body.contains_key("safetySettings"));
    assert_eq!(
        body["safety_settings"],
        json!([{ "category": "HARM_CATEGORY_HARASSMENT" }])
    );
    Ok(())
}

#[test]
fn custom_tools_are_rejected_before_dispatch() -> Result<(), Box<dyn StdError>> {
    let call = resolved(
        Request::builder()
            .model("gemini/gemini-3.1-pro-preview")
            .user("Hello")
            .tool(ToolDefinition::custom(
                "apply_patch",
                "Edits files",
                json!({ "type": "grammar" }),
            ))
            .build()?,
    )?;

    let error = GeminiGenerateCodec
        .encode(&call, false)
        .err()
        .ok_or("expected a custom tool to be rejected")?;

    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    Ok(())
}

#[test]
fn count_tokens_wraps_the_whole_generate_body() -> Result<(), Box<dyn StdError>> {
    let call = resolved(
        Request::builder()
            .model("gemini/gemini-3.1-pro-preview")
            .user("Hello")
            .max_output_tokens(128)
            .tool(ToolDefinition::function(
                "search",
                "Searches",
                json!({ "type": "object" }),
            ))
            .build()?,
    )?;

    let encoded = GeminiGenerateCodec
        .encode_count_tokens(&call)
        .ok_or("expected a count tokens request")??;

    assert!(
        encoded
            .url
            .ends_with("/v1beta/models/gemini-3.1-pro-preview:countTokens")
    );
    let body = object(encoded.body)?;
    assert!(!body.contains_key("contents"));
    let wrapped = &body["generateContentRequest"];
    assert_eq!(wrapped["contents"][0]["parts"][0]["text"], "Hello");
    assert_eq!(wrapped["generationConfig"]["maxOutputTokens"], json!(128));
    assert!(wrapped.get("tools").is_some());

    assert_eq!(
        GeminiGenerateCodec.decode_count_tokens(call.route(), json!({ "totalTokens": 42 }))?,
        42
    );
    Ok(())
}

#[test]
fn stream_assigns_one_block_per_run_and_keeps_the_last_usage() -> Result<(), Box<dyn StdError>> {
    let events = stream(&[
        json!({ "responseId": "resp-stream-1",
                "candidates": [{ "content": { "parts": [{ "text": "Hel" }] } }],
                "usageMetadata": { "promptTokenCount": 10, "candidatesTokenCount": 1 } }),
        json!({ "candidates": [{ "content": { "parts": [{ "text": "lo" }] } }] }),
        json!({ "candidates": [{ "content": { "parts": [
                    { "text": "Let me think", "thought": true }] } }] }),
        json!({ "candidates": [{ "content": { "parts": [
                    { "functionCall": { "name": "search", "args": { "query": "rust" } },
                      "thoughtSignature": "sig-call" }] },
                "finishReason": "STOP" }],
                "usageMetadata": { "promptTokenCount": 20, "candidatesTokenCount": 9,
                                   "thoughtsTokenCount": 3 } }),
    ])?;

    // The stream opens with `Started`, carrying the response id the first
    // chunk named.
    let [StreamEvent::Started { id }, ..] = events.as_slice() else {
        return Err(format!("unexpected first event {:?}", events.first()).into());
    };
    assert_eq!(id.as_deref(), Some("resp-stream-1"));

    let starts: Vec<&str> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ContentBlockStart { id, .. } => Some(id.as_str()),
            _ => None,
        })
        .collect();
    let ends: Vec<&str> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ContentBlockEnd { id, .. } => Some(id.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(starts, ["text-0", "reasoning-0", "tool-0"]);
    assert_eq!(ends, ["text-0", "reasoning-0", "tool-0"]);

    // A function call arrives whole, so its block carries no delta.
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, StreamEvent::ToolCallDelta { .. }))
    );

    let completed: Vec<&Response> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::Ended { response } => Some(response.as_ref()),
            _ => None,
        })
        .collect();
    let [response] = completed.as_slice() else {
        return Err(format!("expected one completed event, got {}", completed.len()).into());
    };
    assert_eq!(response.text(), "Hello");
    assert_eq!(response.id.as_deref(), Some("resp-stream-1"));
    // The last chunk said STOP, and the turn called a tool.
    assert_eq!(response.finish_reason, FinishReason::ToolCall);
    assert_eq!(response.usage.input, 20);
    assert_eq!(response.usage.output, 9);
    assert_eq!(response.usage.reasoning, 3);
    assert!(response.raw.is_none());

    let [
        ContentPart::Text { .. },
        ContentPart::Reasoning(..),
        ContentPart::ToolCall(call),
    ] = response.content.as_slice()
    else {
        return Err(format!("unexpected content {:?}", response.content).into());
    };
    // The response id from the first chunk scopes the synthesized call id
    // for the whole stream.
    assert_eq!(call.id, "search-0-resp-stream-1");
    assert_eq!(call.input.wire_value(), json!({ "query": "rust" }));
    assert_eq!(
        call.provider_metadata.get("gemini"),
        Some(&json!({ "thoughtSignature": "sig-call" }))
    );
    Ok(())
}

#[test]
fn stream_errors_end_the_stream() -> Result<(), Box<dyn StdError>> {
    let call = resolved(
        Request::builder()
            .model("gemini/gemini-3.1-pro-preview")
            .user("Hello")
            .build()?,
    )?;
    let mut decoder = GeminiGenerateCodec.stream_decoder(call.route());

    let error = decoder
        .decode(SseEvent {
            event: None,
            data:  json!({ "error": { "status": "RESOURCE_EXHAUSTED",
                                      "message": "too many requests" } })
            .to_string(),
        })
        .err()
        .ok_or("expected a stream error")?;

    // The gRPC status is classified by the shared provider classifier, so
    // a mid-stream failure lands in the same category an HTTP one would.
    assert_eq!(error.kind(), ErrorKind::RateLimit);
    assert_eq!(error.provider_code(), Some("RESOURCE_EXHAUSTED"));
    Ok(())
}

#[test]
fn a_failed_tool_result_uses_googles_error_key() -> Result<(), Box<dyn StdError>> {
    let request = Request::builder()
        .model("gemini/gemini-3.1-pro-preview")
        .user("Look it up.")
        .message(Message::new(Role::Tool, [ContentPart::ToolResult(
            ToolResult {
                tool_call_id: "call-1".to_owned(),
                name:         Some("weather".to_owned()),
                content:      vec![ContentPart::Text {
                    text: "the city is unknown".to_owned(),
                }],
                is_error:     true,
            },
        )]))
        .build()?;

    let encoded = GeminiGenerateCodec.encode(&resolved(request)?, false)?;

    let response = &encoded.body["contents"][0]["parts"][1]["functionResponse"]["response"];
    assert_eq!(response, &json!({ "error": "the city is unknown" }));
    assert!(
        response.get("is_error").is_none(),
        "a boolean flag beside `output` reads to the model as ordinary output"
    );
    Ok(())
}

/// Encodes a request whose only tool message carries `content` as one
/// result, and returns the `functionResponse.response` it produced.
fn encoded_tool_response(
    content: Vec<ContentPart>,
    is_error: bool,
) -> Result<Value, Box<dyn StdError>> {
    let request = Request::builder()
        .model("gemini/gemini-3.1-pro-preview")
        .user("Look it up.")
        .message(Message::new(Role::Tool, [ContentPart::ToolResult(
            ToolResult {
                tool_call_id: "call-1".to_owned(),
                name: Some("weather".to_owned()),
                content,
                is_error,
            },
        )]))
        .build()?;

    let encoded = GeminiGenerateCodec.encode(&resolved(request)?, false)?;
    assert!(
        !encoded
            .warnings
            .iter()
            .any(|warning| warning.message.contains("tool result")),
        "carried content must not warn"
    );
    Ok(encoded.body["contents"][0]["parts"][1]["functionResponse"]["response"].clone())
}

#[test]
fn an_object_tool_result_is_the_whole_response_struct() -> Result<(), Box<dyn StdError>> {
    // The caller crafted the exact struct the model should see, so it
    // travels verbatim instead of nesting under `output`.
    let response = encoded_tool_response(
        vec![ContentPart::Json {
            value: json!({ "temperature": 21, "unit": "C" }),
        }],
        false,
    )?;

    assert_eq!(response, json!({ "temperature": 21, "unit": "C" }));
    Ok(())
}

#[test]
fn a_non_object_json_tool_result_stays_under_output() -> Result<(), Box<dyn StdError>> {
    // A bare value could not be the free-form `response` struct, so it
    // keeps Google's conventional `output` key — and still reaches the
    // model rather than flattening to empty text.
    let response = encoded_tool_response(
        vec![ContentPart::Json {
            value: json!([21, "C"]),
        }],
        false,
    )?;

    assert_eq!(response, json!({ "output": [21, "C"] }));
    Ok(())
}

#[test]
fn a_failed_object_tool_result_keeps_the_error_key() -> Result<(), Box<dyn StdError>> {
    // A failure must read as one, so even a crafted object nests under
    // `error` when `is_error` is set.
    let response = encoded_tool_response(
        vec![ContentPart::Json {
            value: json!({ "code": "unknown_city" }),
        }],
        true,
    )?;

    assert_eq!(response, json!({ "error": { "code": "unknown_city" } }));
    Ok(())
}

#[test]
fn an_opaque_payload_that_cannot_be_a_part_is_dropped() -> Result<(), Box<dyn StdError>> {
    // Every Gemini `Part` is an object, so a string payload could never be
    // one. Dropping it beats sending the API something it must reject.
    let request = Request::builder()
        .model("gemini/gemini-3.1-pro-preview")
        .user("Hello")
        .message(Message::new(Role::Assistant, [
            ContentPart::opaque("gemini.thought", json!("not a part")),
            ContentPart::Text {
                text: "Hi.".to_owned(),
            },
        ]))
        .build()?;

    let encoded = GeminiGenerateCodec.encode(&resolved(request)?, false)?;

    let parts = &encoded.body["contents"][1]["parts"];
    assert_eq!(parts.as_array().map(Vec::len), Some(1));
    assert_eq!(parts[0]["text"], "Hi.");
    Ok(())
}

#[test]
fn an_opaque_object_still_replays() -> Result<(), Box<dyn StdError>> {
    let request = Request::builder()
        .model("gemini/gemini-3.1-pro-preview")
        .user("Hello")
        .message(Message::new(Role::Assistant, [ContentPart::opaque(
            "gemini.thought",
            json!({ "text": "kept", "thought": true }),
        )]))
        .build()?;

    let encoded = GeminiGenerateCodec.encode(&resolved(request)?, false)?;

    assert_eq!(
        encoded.body["contents"][1]["parts"][0],
        json!({ "text": "kept", "thought": true })
    );
    Ok(())
}
