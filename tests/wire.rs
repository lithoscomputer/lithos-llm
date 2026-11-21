//! Proves that the provider-native wire types are lossless.
//!
//! Each fixture under `tests/fixtures/wire/` is a hand-written request,
//! response, error body, or SSE stream in one provider's format, shaped like
//! the traffic a coding agent produces: thinking blocks with signatures,
//! encrypted reasoning, cache-control markers, beta parameters, `null` fields,
//! and block, item, and event types these types do not model. Every test
//! parses the raw text, writes it back, and compares the two JSON values.

use std::error::Error as StdError;
use std::fs;
use std::path::Path;

use lithos_llm::wire::{Field, anthropic_messages, openai_chat, openai_responses};
use proptest::prelude::*;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};

type TestResult = Result<(), Box<dyn StdError>>;

fn fixture(name: &str) -> Result<String, Box<dyn StdError>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/wire")
        .join(name);
    Ok(fs::read_to_string(path)?)
}

/// The `data` of each event in an SSE fixture, skipping the `[DONE]` marker.
fn stream_data(name: &str) -> Result<Vec<String>, Box<dyn StdError>> {
    Ok(fixture(name)?
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(ToOwned::to_owned)
        .collect())
}

/// Parses `text` as `T` and writes it back as a JSON value.
fn reserialize<T: DeserializeOwned + Serialize>(text: &str) -> Result<Value, serde_json::Error> {
    serde_json::to_value(serde_json::from_str::<T>(text)?)
}

/// Parses `text` as `T` and asserts that writing it back loses nothing.
fn round_trip<T: DeserializeOwned + Serialize>(text: &str) -> Result<T, Box<dyn StdError>> {
    let parsed: T = serde_json::from_str(text)?;
    assert_eq!(
        serde_json::to_value(&parsed)?,
        serde_json::from_str::<Value>(text)?
    );
    Ok(parsed)
}

fn stream<T: DeserializeOwned + Serialize>(name: &str) -> Result<Vec<T>, Box<dyn StdError>> {
    stream_data(name)?
        .iter()
        .map(|data| round_trip(data))
        .collect()
}

#[test]
fn anthropic_request_round_trips_with_thinking_and_unknown_blocks() -> TestResult {
    use anthropic_messages::{Content, ContentBlock, Request};

    let request: Request = round_trip(&fixture("anthropic_messages_request.json")?)?;
    assert_eq!(request.max_tokens, 32000);
    assert_eq!(request.stream, Field::Present(true));
    assert!(request.extra.contains_key("context_management"));
    let Field::Present(Content::Blocks(system)) = &request.system else {
        panic!("system blocks")
    };
    let ContentBlock::Text(cached) = &system[1] else {
        panic!("system text")
    };
    assert_eq!(cached.extra["cache_control"]["ttl"], "1h");

    let Content::Blocks(turn) = &request.messages[1].content else {
        panic!("assistant blocks")
    };
    let [
        ContentBlock::Thinking(thinking),
        ContentBlock::RedactedThinking(_),
        ContentBlock::Text(_),
        ContentBlock::ToolUse(tool_use),
    ] = turn.as_slice()
    else {
        panic!("assistant turn: {turn:?}")
    };
    assert!(
        matches!(&thinking.signature, Field::Present(signature) if signature.starts_with("EqQB"))
    );
    assert_eq!(tool_use.input["line_range"], json!([1, 40]));

    let unknown: Vec<&str> = request
        .messages
        .iter()
        .filter_map(|message| match &message.content {
            Content::Blocks(blocks) => Some(blocks),
            Content::Text(_) => None,
        })
        .flatten()
        .filter_map(|block| match block {
            ContentBlock::Unknown(object) => object["type"].as_str(),
            _ => None,
        })
        .collect();
    assert_eq!(unknown, ["web_search_tool_result"]);
    Ok(())
}

#[test]
fn anthropic_response_keeps_nulls_apart_from_absent_fields() -> TestResult {
    use anthropic_messages::{ContentBlock, Response};

    let response: Response = round_trip(&fixture("anthropic_messages_response.json")?)?;
    assert_eq!(response.stop_reason, Field::Present("tool_use".to_owned()));
    assert_eq!(response.stop_sequence, Field::Null);
    assert_eq!(
        response.usage.cache_read_input_tokens,
        Field::Present(48211)
    );
    assert_eq!(response.extra["container"], Value::Null);
    assert!(matches!(response.content[0], ContentBlock::Thinking(_)));
    Ok(())
}

#[test]
fn anthropic_stream_events_round_trip() -> TestResult {
    use anthropic_messages::{ContentBlock, Delta, StreamEvent};

    let events: Vec<StreamEvent> = stream("anthropic_messages_stream.sse")?;
    assert_eq!(events.len(), 20);
    let StreamEvent::ContentBlockStart(start) = &events[2] else {
        panic!("thinking start")
    };
    let ContentBlock::Thinking(thinking) = &start.content_block else {
        panic!("thinking block")
    };
    assert_eq!(thinking.signature, Field::Absent);
    assert!(matches!(
        &events[4],
        StreamEvent::ContentBlockDelta(delta) if matches!(delta.delta, Delta::SignatureDelta(_))
    ));
    assert!(matches!(
        &events[8],
        StreamEvent::ContentBlockDelta(delta) if matches!(&delta.delta, Delta::Unknown(object) if object["type"] == "citations_delta")
    ));
    assert!(matches!(
        &events[15],
        StreamEvent::ContentBlockStart(start) if matches!(start.content_block, ContentBlock::Unknown(_))
    ));
    assert!(
        matches!(&events[17], StreamEvent::Unknown(object) if object["type"] == "compaction_progress")
    );
    let StreamEvent::MessageDelta(end) = &events[18] else {
        panic!("message delta")
    };
    assert_eq!(end.delta.stop_reason, Field::Present("tool_use".to_owned()));
    assert_eq!(end.usage.input_tokens, Field::Present(12));
    assert_eq!(end.usage.cache_read_input_tokens, Field::Null);
    assert!(matches!(events[19], StreamEvent::MessageStop(_)));
    Ok(())
}

#[test]
fn responses_request_round_trips_every_item_kind() -> TestResult {
    use openai_responses::{Content, ContentPart, Input, Item, MessageType, Request};

    let request: Request = round_trip(&fixture("openai_responses_request.json")?)?;
    assert_eq!(request.model, Field::Present("gpt-6-astra".to_owned()));
    assert_eq!(request.extra["max_output_tokens"], Value::Null);
    let Field::Present(Input::Items(items)) = &request.input else {
        panic!("input items")
    };
    let [
        Item::Message(untyped),
        Item::Message(typed),
        Item::Reasoning(reasoning),
        Item::FunctionCall(call),
        Item::FunctionCallOutput(_),
        Item::CustomToolCall(_),
        Item::CustomToolCallOutput(_),
        Item::Unknown(web_search),
        Item::Message(assistant),
        Item::Unknown(reference),
    ] = items.as_slice()
    else {
        panic!("input items: {items:?}")
    };
    assert_eq!(untyped.kind, Field::Absent);
    assert_eq!(typed.kind, Field::Present(MessageType::Message));
    assert!(matches!(
        &typed.content,
        Content::Parts(parts) if matches!(
            parts.as_slice(),
            [ContentPart::InputText(_), ContentPart::InputImage(_), ContentPart::InputFile(_)]
        )
    ));
    assert!(
        matches!(&reasoning.encrypted_content, Field::Present(blob) if blob.starts_with("gAAAA"))
    );
    assert_eq!(reasoning.extra["content"], Value::Null);
    assert_eq!(call.arguments, r#"{"path":"tests/parse.rs"}"#);
    assert_eq!(web_search["type"], "web_search_call");
    assert!(matches!(
        &assistant.content,
        Content::Parts(parts) if matches!(parts.as_slice(), [ContentPart::OutputText(_), ContentPart::Refusal(_)])
    ));
    assert_eq!(reference["type"], "item_reference");
    Ok(())
}

#[test]
fn responses_response_round_trips() -> TestResult {
    use openai_responses::{Item, Response};

    let response: Response = round_trip(&fixture("openai_responses_response.json")?)?;
    assert!(matches!(response.output.as_slice(), [
        Item::Reasoning(_),
        Item::Message(_),
        Item::FunctionCall(_)
    ]));
    let Field::Present(usage) = &response.usage else {
        panic!("usage")
    };
    assert_eq!(usage.input_tokens_details.cached_tokens, 48128);
    assert_eq!(usage.output_tokens_details.reasoning_tokens, 320);
    Ok(())
}

#[test]
fn responses_stream_events_round_trip() -> TestResult {
    use openai_responses::StreamEvent;

    let events: Vec<StreamEvent> = stream("openai_responses_stream.sse")?;
    assert_eq!(events.len(), 22);
    let StreamEvent::Created(created) = &events[0] else {
        panic!("created")
    };
    assert_eq!(created.response.usage, Field::Null);
    assert!(
        matches!(&events[12], StreamEvent::Unknown(object) if object["type"] == "response.output_text.annotation.added")
    );
    let unknown = events
        .iter()
        .filter(|event| matches!(event, StreamEvent::Unknown(_)))
        .count();
    assert_eq!(unknown, 1);
    let StreamEvent::Completed(completed) = &events[21] else {
        panic!("completed")
    };
    assert_eq!(completed.response.output.len(), 3);
    Ok(())
}

#[test]
fn chat_request_round_trips_with_openrouter_extensions() -> TestResult {
    use openai_chat::{Content, ContentPart, Request, ToolCall};

    let request: Request = round_trip(&fixture("openai_chat_request.json")?)?;
    assert_eq!(request.stream, Field::Present(true));
    assert!(request.extra.contains_key("provider"));
    let assistant = &request.messages[2];
    assert_eq!(assistant.content, Field::Null);
    assert!(assistant.extra.contains_key("reasoning_details"));
    assert!(matches!(
        &assistant.tool_calls,
        Field::Present(calls) if matches!(calls.as_slice(), [ToolCall::Function(_)])
    ));
    assert_eq!(
        request.messages[3].tool_call_id,
        Field::Present("toolu_01G".to_owned())
    );
    assert!(matches!(
        &request.messages[4].content,
        Field::Present(Content::Parts(parts)) if matches!(
            parts.as_slice(),
            [ContentPart::Text(_), ContentPart::ImageUrl(_), ContentPart::Unknown(_), ContentPart::Unknown(_)]
        )
    ));
    assert!(matches!(
        &request.messages[5].tool_calls,
        Field::Present(calls) if matches!(calls.as_slice(), [ToolCall::Unknown(_)])
    ));
    Ok(())
}

#[test]
fn chat_response_round_trips() -> TestResult {
    use openai_chat::Response;

    let response: Response = round_trip(&fixture("openai_chat_response.json")?)?;
    assert_eq!(
        response.choices[0].finish_reason,
        Field::Present("tool_calls".to_owned())
    );
    let Field::Present(usage) = &response.usage else {
        panic!("usage")
    };
    assert!(
        matches!(&usage.prompt_tokens_details, Field::Present(details) if details.cached_tokens == Field::Present(48128))
    );
    assert_eq!(usage.extra["cost"], json!(0.0315));
    Ok(())
}

#[test]
fn chat_stream_events_round_trip() -> TestResult {
    use openai_chat::StreamEvent;

    let events: Vec<StreamEvent> = stream("openai_chat_stream.sse")?;
    assert_eq!(events.len(), 7);
    let first = &events[3].choices[0].delta;
    let later = &events[4].choices[0].delta;
    let (Field::Present(first), Field::Present(later)) = (&first.tool_calls, &later.tool_calls)
    else {
        panic!("tool call deltas")
    };
    assert_eq!(first[0].id, Field::Present("toolu_01J".to_owned()));
    assert_eq!(later[0].id, Field::Null);
    assert_eq!(
        events[5].choices[0].finish_reason,
        Field::Present("tool_calls".to_owned())
    );
    assert!(events[6].choices.is_empty());
    assert!(matches!(&events[6].usage, Field::Present(usage) if usage.total_tokens == 48319));
    Ok(())
}

#[test]
fn error_bodies_round_trip() -> TestResult {
    use openai_chat::{ErrorCode, ErrorResponse};

    let anthropic: anthropic_messages::ErrorResponse =
        round_trip(&fixture("anthropic_messages_error.json")?)?;
    assert_eq!(anthropic.error.kind, "overloaded_error");
    assert_eq!(anthropic.error.message, "Overloaded");
    assert_eq!(
        anthropic.extra.keys().collect::<Vec<_>>(),
        ["request_id"],
        "`type` is the envelope's, not an unknown field"
    );

    let openai: openai_responses::ErrorResponse =
        round_trip(&fixture("openai_responses_error.json")?)?;
    assert_eq!(
        openai.error.kind,
        Field::Present("invalid_request_error".to_owned())
    );
    assert_eq!(openai.error.param, Field::Null);
    assert_eq!(
        openai.error.code,
        Field::Present(ErrorCode::Text("context_length_exceeded".to_owned()))
    );

    let openrouter: ErrorResponse = round_trip(&fixture("openai_chat_error.json")?)?;
    assert_eq!(openrouter.error.kind, Field::Absent);
    assert_eq!(openrouter.error.param, Field::Absent);
    assert_eq!(
        openrouter.error.code,
        Field::Present(ErrorCode::Number(429.into()))
    );
    assert_eq!(
        openrouter.error.extra["metadata"]["provider_name"],
        "Anthropic"
    );
    assert_eq!(openrouter.extra["user_id"], "user_fixture");
    Ok(())
}

/// A proxy that answers a request itself writes the error in the caller's
/// format, the same way the provider would.
#[test]
fn built_error_bodies_serialize_in_the_provider_format() -> TestResult {
    let anthropic =
        anthropic_messages::ErrorResponse::new("authentication_error", "invalid x-api-key");
    assert_eq!(
        serde_json::to_value(&anthropic)?,
        json!({
            "type": "error",
            "error": {"type": "authentication_error", "message": "invalid x-api-key"},
        })
    );

    let openai =
        openai_chat::ErrorResponse::new("invalid_request_error", "Incorrect API key provided.");
    let mut expected = json!({"error": {
        "message": "Incorrect API key provided.",
        "type": "invalid_request_error",
        "param": null,
        "code": null,
    }});
    assert_eq!(serde_json::to_value(&openai)?, expected);

    let coded = openai.with_code("invalid_api_key");
    expected["error"]["code"] = json!("invalid_api_key");
    assert_eq!(serde_json::to_value(&coded)?, expected);
    Ok(())
}

/// The use case the types exist for: change one thing, forward the rest.
#[test]
fn editing_one_field_changes_only_that_field() -> TestResult {
    let text = fixture("anthropic_messages_request.json")?;
    let mut request: anthropic_messages::Request = serde_json::from_str(&text)?;
    request.model = "claude-sonnet-5-5".to_owned();
    let mut expected: Value = serde_json::from_str(&text)?;
    expected["model"] = json!("claude-sonnet-5-5");
    assert_eq!(serde_json::to_value(&request)?, expected);

    let text = fixture("openai_responses_request.json")?;
    let mut request: openai_responses::Request = serde_json::from_str(&text)?;
    request.instructions = Field::Null;
    let mut expected: Value = serde_json::from_str(&text)?;
    expected["instructions"] = Value::Null;
    assert_eq!(serde_json::to_value(&request)?, expected);

    let text = fixture("openai_chat_request.json")?;
    let mut request: openai_chat::Request = serde_json::from_str(&text)?;
    request.stream = Field::Absent;
    let mut expected: Value = serde_json::from_str(&text)?;
    expected.as_object_mut().ok_or("object")?.remove("stream");
    assert_eq!(serde_json::to_value(&request)?, expected);
    Ok(())
}

#[test]
fn malformed_known_types_are_errors_not_unknown() {
    use anthropic_messages::{ContentBlock, Request, StreamEvent};

    for block in [
        json!({"type": "text", "text": 5}),
        json!({"type": "thinking", "signature": "c2ln"}),
        json!({"text": "no type"}),
        json!({"type": 7}),
    ] {
        assert!(
            serde_json::from_value::<ContentBlock>(block.clone()).is_err(),
            "{block}"
        );
    }
    assert!(serde_json::from_value::<StreamEvent>(json!({"type": "content_block_stop"})).is_err());
    // A field the API never sends as `null` does not accept one, because
    // `T` could not write the `null` back.
    assert!(
        serde_json::from_value::<Request>(
            json!({"model": "m", "max_tokens": null, "messages": []})
        )
        .is_err()
    );
    assert!(
        serde_json::from_value::<openai_responses::Item>(
            json!({"type": "message", "content": "no role"})
        )
        .is_err()
    );
    assert!(
        serde_json::from_value::<openai_chat::ToolCall>(
            json!({"type": "function", "id": "call_1"})
        )
        .is_err()
    );
    // The Anthropic envelope's `type` must be `error`.
    let error = json!({"type": "overloaded_error", "message": "Overloaded"});
    for body in [
        json!({"type": "message", "error": error}),
        json!({"error": error}),
        json!({"type": "error"}),
    ] {
        assert!(
            serde_json::from_value::<anthropic_messages::ErrorResponse>(body.clone()).is_err(),
            "{body}"
        );
    }
    for body in [
        json!({"error": {"type": "server_error"}}),
        json!({"error": {"message": "m", "code": true}}),
    ] {
        assert!(
            serde_json::from_value::<openai_chat::ErrorResponse>(body.clone()).is_err(),
            "{body}"
        );
    }
}

/// Every fixture document, paired with the type that parses it.
type Case = (Value, fn(&str) -> Result<Value, serde_json::Error>);

fn cases() -> Result<Vec<Case>, Box<dyn StdError>> {
    let mut cases: Vec<Case> = Vec::new();
    let mut add = |name: &str, parse: fn(&str) -> Result<Value, serde_json::Error>| {
        cases.push((serde_json::from_str(&fixture(name)?)?, parse));
        Ok::<(), Box<dyn StdError>>(())
    };
    add(
        "anthropic_messages_request.json",
        reserialize::<anthropic_messages::Request>,
    )?;
    add(
        "anthropic_messages_response.json",
        reserialize::<anthropic_messages::Response>,
    )?;
    add(
        "openai_responses_request.json",
        reserialize::<openai_responses::Request>,
    )?;
    add(
        "openai_responses_response.json",
        reserialize::<openai_responses::Response>,
    )?;
    add(
        "openai_chat_request.json",
        reserialize::<openai_chat::Request>,
    )?;
    add(
        "openai_chat_response.json",
        reserialize::<openai_chat::Response>,
    )?;
    add(
        "anthropic_messages_error.json",
        reserialize::<anthropic_messages::ErrorResponse>,
    )?;
    add(
        "openai_responses_error.json",
        reserialize::<openai_responses::ErrorResponse>,
    )?;
    add(
        "openai_chat_error.json",
        reserialize::<openai_chat::ErrorResponse>,
    )?;
    for (name, parse) in [
        (
            "anthropic_messages_stream.sse",
            reserialize::<anthropic_messages::StreamEvent> as fn(&str) -> _,
        ),
        (
            "openai_responses_stream.sse",
            reserialize::<openai_responses::StreamEvent>,
        ),
        (
            "openai_chat_stream.sse",
            reserialize::<openai_chat::StreamEvent>,
        ),
    ] {
        for data in stream_data(name)? {
            cases.push((serde_json::from_str(&data)?, parse));
        }
    }
    Ok(cases)
}

/// Adds the next of `fields` to every object in `value`, cycling through them.
fn add_fields(value: &mut Value, fields: &[(String, Value)], next: &mut usize) {
    match value {
        Value::Object(object) => {
            for child in object.values_mut() {
                add_fields(child, fields, next);
            }
            let (key, field) = &fields[*next % fields.len()];
            object.insert(key.clone(), field.clone());
            *next += 1;
        }
        Value::Array(items) => {
            for item in items {
                add_fields(item, fields, next);
            }
        }
        _ => {}
    }
}

fn json_value() -> impl Strategy<Value = Value> {
    let leaf = prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::from),
        any::<i64>().prop_map(Value::from),
        any::<u64>().prop_map(Value::from),
        any::<f64>()
            .prop_filter("finite", |number| number.is_finite())
            .prop_map(Value::from),
        ".{0,12}".prop_map(Value::from),
    ];
    leaf.prop_recursive(3, 24, 4, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..4).prop_map(Value::from),
            prop::collection::vec(("[a-z_]{1,8}", inner), 0..4)
                .prop_map(|fields| Value::Object(fields.into_iter().collect::<Map<_, _>>())),
        ]
    })
}

fn unknown_fields() -> impl Strategy<Value = Vec<(String, Value)>> {
    // The prefix keeps a generated field from colliding with a modeled one.
    prop::collection::vec(("x_[a-z_]{1,10}", json_value()), 1..6)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn unknown_fields_survive_at_every_depth(fields in unknown_fields()) {
        let cases = cases().map_err(|error| TestCaseError::fail(error.to_string()))?;
        for (document, parse) in cases {
            let mut document = document;
            add_fields(&mut document, &fields, &mut 0);
            let text = document.to_string();
            let written = parse(&text).map_err(|error| TestCaseError::fail(format!("{error}: {text}")))?;
            let expected: Value = serde_json::from_str(&text)
                .map_err(|error| TestCaseError::fail(error.to_string()))?;
            prop_assert_eq!(written, expected);
        }
    }
}
