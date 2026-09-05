//! Pins the serialized form of the types applications persist or send over a
//! wire: `Request`, `Response`, `StreamEvent`, `ErrorData`, and `TokenCounts`.
//!
//! An application that stores a response in an event log or streams events to
//! a peer depends on these shapes byte for byte. A changed snapshot here is a
//! compatibility break for every stored document and every peer speaking the
//! format, so review it as one: an intentional change needs a changelog entry
//! and a migration story, and an unexplained one is a regression.
//!
//! Snapshots live under `tests/snapshots/` and are named
//! `serialization__<test_fn>.snap`.

use std::collections::BTreeMap;
use std::error::Error as StdError;
use std::io;
use std::time::Duration;

use insta::assert_snapshot;
use lithos_llm::catalog::{ModelId, ProviderId};
use lithos_llm::types::{
    AudioContent, CacheHint, ContentBlockId, ContentBlockKind, ContentPart, Cost, CostSource,
    DocumentContent, Error, ErrorData, ErrorKind, FinishReason, ImageContent, MediaSource, Message,
    RateLimits, ReasoningContent, ReasoningEffort, Request, Response, ResponseFormat,
    RetryClassification, Role, Speed, StreamEvent, TokenCounts, ToolArguments, ToolCall,
    ToolCallKind, ToolChoice, ToolDefinition, ToolInput, ToolResult, Warning,
};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::json;

/// Renders a value as the pretty JSON a document store or wire would carry.
///
/// This serializes the value directly rather than through `serde_json::Value`:
/// the detour widens `f32` fields such as `temperature` to `f64`, printing
/// `0.20000000298023224` where the wire carries `0.2`.
fn render<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(value)
}

fn round_trip<T: Serialize + DeserializeOwned>(value: &T) -> Result<T, serde_json::Error> {
    serde_json::from_str(&serde_json::to_string(value)?)
}

/// A request that sets every field and uses every content part kind.
fn complete_request() -> Result<Request, Box<dyn StdError>> {
    let mut call = ToolCall::function("call_1", "lookup", json!({ "query": "rust" }));
    call.input = ToolInput::Function(ToolArguments::from_raw("{\"query\":\"rust\"}".to_owned()));
    call.provider_metadata
        .insert("openai".to_owned(), json!({ "id": "fc_1" }));

    Ok(Request::builder()
        .model("provider/model")
        .system("Keep it short.")
        .developer("Prefer tables.")
        .message(Message::new(Role::User, [
            ContentPart::Text {
                text: "Describe these.".to_owned(),
            },
            ContentPart::Image(ImageContent {
                source: MediaSource::url_with_media_type(
                    "https://example.com/cat.png",
                    "image/png",
                ),
                detail: Some("high".to_owned()),
            }),
            ContentPart::Audio(AudioContent::new(MediaSource::base64("QUJD", "audio/wav"))),
            ContentPart::Document(DocumentContent {
                source: MediaSource::base64("REVG", "application/pdf"),
                name:   Some("report.pdf".to_owned()),
            }),
        ]))
        .message(
            Message::new(Role::Assistant, [
                ContentPart::Reasoning(ReasoningContent {
                    text:             "step one".to_owned(),
                    signature:        Some("sig".to_owned()),
                    signature_origin: Some("anthropic".to_owned()),
                    redacted:         false,
                }),
                ContentPart::ToolCall(call),
                ContentPart::ToolCall(ToolCall::custom("call_2", "apply_patch", "*** Begin Patch")),
                ContentPart::Text {
                    text: "Looking that up.".to_owned(),
                },
            ])
            .with_name("assistant-1"),
        )
        .message(
            Message::new(Role::Tool, [
                ContentPart::ToolResult(ToolResult {
                    tool_call_id: "call_1".to_owned(),
                    name:         Some("lookup".to_owned()),
                    content:      vec![ContentPart::Text {
                        text: "found".to_owned(),
                    }],
                    is_error:     true,
                }),
                ContentPart::Json {
                    value: json!({ "score": 0.5 }),
                },
                ContentPart::opaque("openai.reasoning", json!({ "id": "rs_1" })),
            ])
            .with_tool_call_id("call_1"),
        )
        .tool(ToolDefinition::function(
            "lookup",
            "Looks a term up",
            json!({ "type": "object", "properties": { "query": { "type": "string" } } }),
        ))
        .tool(ToolDefinition::custom(
            "apply_patch",
            "Edits files",
            json!({ "type": "grammar", "syntax": "lark" }),
        ))
        .tool_choice(ToolChoice::Tool {
            name: "lookup".to_owned(),
        })
        .response_format(ResponseFormat::JsonSchema {
            name:   "answer".to_owned(),
            schema: json!({ "type": "object", "properties": { "answer": { "type": "string" } } }),
        })
        .max_output_tokens(512)
        .temperature(0.2)
        .top_p(0.9)
        .reasoning_effort(ReasoningEffort::High)
        .cache_hint(CacheHint::Key {
            key: "tenant-acme".to_owned(),
        })
        .speed(Speed::Fast)
        .timeout(Duration::from_millis(1500))
        .stop_sequence("STOP")
        .metadata_entry("tenant", "acme")
        .provider_option("openai", "service_tier", json!("flex"))
        .build()?)
}

fn minimal_request() -> Result<Request, Box<dyn StdError>> {
    Ok(Request::builder()
        .model("provider/model")
        .user("Hello")
        .build()?)
}

fn complete_response() -> Response {
    let mut response = Response::new(ProviderId::new("openai"), ModelId::new("gpt-5"), vec![
        ContentPart::Text {
            text: "hello".to_owned(),
        },
        ContentPart::ToolCall(ToolCall::function(
            "call_1",
            "lookup",
            json!({ "query": "rust" }),
        )),
    ]);
    response.id = Some("resp_1".to_owned());
    response.finish_reason = FinishReason::ToolCall;
    response.usage = TokenCounts::from_inclusive(100, 60, 25, 30, 10);
    response.cost = Some(Cost {
        usd_micros: 1234,
        source:     CostSource::Provider,
    });
    response.rate_limits = Some(RateLimits {
        request_limit:     Some(1000),
        request_remaining: Some(999),
        request_reset:     Some("1s".to_owned()),
        token_limit:       Some(2_000_000),
        token_remaining:   Some(1_999_000),
        token_reset:       Some("6m0s".to_owned()),
    });
    response.warnings = vec![Warning {
        code:    "flattened_tool_result".to_owned(),
        message: "rich tool result content was flattened to text".to_owned(),
    }];
    response.raw = Some(json!({ "id": "resp_1", "object": "response" }));
    response
}

fn minimal_response() -> Response {
    Response::new(ProviderId::new("openai"), ModelId::new("gpt-5"), vec![
        ContentPart::Text {
            text: "hello".to_owned(),
        },
    ])
}

fn every_stream_event() -> Vec<StreamEvent> {
    let text = ContentBlockId::new("block-0");
    let tool = ContentBlockId::new("block-1");
    let reasoning = ContentBlockId::new("block-2");
    vec![
        StreamEvent::Started {
            id: Some("resp_1".to_owned()),
        },
        StreamEvent::Started { id: None },
        StreamEvent::ContentBlockStart {
            id:   text.clone(),
            kind: ContentBlockKind::Text,
        },
        StreamEvent::ContentBlockStart {
            id:   reasoning.clone(),
            kind: ContentBlockKind::Reasoning,
        },
        StreamEvent::ContentBlockStart {
            id:   tool.clone(),
            kind: ContentBlockKind::ToolCall {
                id:   "call_1".to_owned(),
                name: Some("lookup".to_owned()),
                kind: ToolCallKind::Function,
            },
        },
        StreamEvent::ContentBlockStart {
            id:   ContentBlockId::new("block-3"),
            kind: ContentBlockKind::Opaque {
                kind: "openai.reasoning".to_owned(),
            },
        },
        StreamEvent::TextDelta {
            id:   text.clone(),
            text: "hel".to_owned(),
        },
        StreamEvent::ReasoningDelta {
            id:   reasoning,
            text: "thinking".to_owned(),
        },
        StreamEvent::ToolCallDelta {
            id:        tool.clone(),
            arguments: "{\"query\":".to_owned(),
        },
        StreamEvent::ContentBlockEnd {
            id:   text,
            part: ContentPart::Text {
                text: "hello".to_owned(),
            },
        },
        StreamEvent::ContentBlockEnd {
            id:   tool,
            part: ContentPart::ToolCall(ToolCall::function(
                "call_1",
                "lookup",
                json!({ "query": "rust" }),
            )),
        },
        StreamEvent::Usage {
            usage: TokenCounts::from_inclusive(100, 60, 25, 30, 10),
        },
        StreamEvent::RateLimits {
            rate_limits: RateLimits {
                request_reset: Some("1s".to_owned()),
                ..RateLimits::default()
            },
        },
        StreamEvent::Ended {
            response: Box::new(complete_response()),
        },
    ]
}

fn complete_error_data() -> ErrorData {
    Error::new(ErrorKind::RateLimit, "slow down")
        .with_provider(ProviderId::new("openai"))
        .with_status(429)
        .with_provider_code("rate_limit_exceeded")
        .with_retry(RetryClassification::after(Duration::from_millis(1500)))
        .with_provider_retry_after(Duration::from_secs(2))
        .with_raw_data(json!({ "error": { "type": "rate_limit_error" } }))
        .with_source(io::Error::other("connection reset"))
        .data()
}

fn minimal_error_data() -> ErrorData {
    Error::new(ErrorKind::Cancelled, "the call was cancelled").data()
}

/// Every error kind, so a renamed or removed category shows up here.
const EVERY_ERROR_KIND: [ErrorKind; 18] = [
    ErrorKind::Configuration,
    ErrorKind::ModelSelection,
    ErrorKind::Authentication,
    ErrorKind::AccessDenied,
    ErrorKind::NotFound,
    ErrorKind::InvalidRequest,
    ErrorKind::ContextLength,
    ErrorKind::RateLimit,
    ErrorKind::QuotaExceeded,
    ErrorKind::ContentFilter,
    ErrorKind::Server,
    ErrorKind::Provider,
    ErrorKind::Network,
    ErrorKind::Timeout,
    ErrorKind::StreamDecode,
    ErrorKind::ResponseDecode,
    ErrorKind::Middleware,
    ErrorKind::Cancelled,
];

#[test]
fn a_complete_request_serializes_to_the_pinned_shape() -> Result<(), Box<dyn StdError>> {
    assert_snapshot!(render(&complete_request()?)?);
    Ok(())
}

#[test]
fn a_minimal_request_omits_unset_fields() -> Result<(), Box<dyn StdError>> {
    assert_snapshot!(render(&minimal_request()?)?);
    Ok(())
}

#[test]
fn a_complete_response_serializes_to_the_pinned_shape() -> Result<(), Box<dyn StdError>> {
    assert_snapshot!(render(&complete_response())?);
    Ok(())
}

#[test]
fn a_minimal_response_omits_unset_fields() -> Result<(), Box<dyn StdError>> {
    assert_snapshot!(render(&minimal_response())?);
    Ok(())
}

#[test]
fn every_stream_event_serializes_to_the_pinned_shape() -> Result<(), Box<dyn StdError>> {
    assert_snapshot!(render(&every_stream_event())?);
    Ok(())
}

#[test]
fn complete_error_data_serializes_to_the_pinned_shape() -> Result<(), Box<dyn StdError>> {
    assert_snapshot!(render(&complete_error_data())?);
    Ok(())
}

#[test]
fn minimal_error_data_omits_unset_fields() -> Result<(), Box<dyn StdError>> {
    assert_snapshot!(render(&minimal_error_data())?);
    Ok(())
}

#[test]
fn error_kinds_and_retry_classifications_keep_their_names() -> Result<(), Box<dyn StdError>> {
    let mut vocabulary = BTreeMap::new();
    vocabulary.insert("error_kinds", serde_json::to_value(EVERY_ERROR_KIND)?);
    vocabulary.insert(
        "retry_classifications",
        serde_json::to_value([
            RetryClassification::Never,
            RetryClassification::Safe,
            RetryClassification::after(Duration::from_millis(1500)),
        ])?,
    );
    assert_snapshot!(render(&vocabulary)?);
    Ok(())
}

#[test]
fn token_counts_serialize_every_bucket() -> Result<(), Box<dyn StdError>> {
    assert_snapshot!(render(&[
        TokenCounts::from_inclusive(100, 60, 25, 30, 10),
        TokenCounts::default(),
    ])?);
    Ok(())
}

#[test]
fn every_pinned_shape_round_trips() -> Result<(), Box<dyn StdError>> {
    let request = complete_request()?;
    assert_eq!(round_trip(&request)?, request);
    let request = minimal_request()?;
    assert_eq!(round_trip(&request)?, request);

    let response = complete_response();
    assert_eq!(round_trip(&response)?, response);

    for event in every_stream_event() {
        assert_eq!(round_trip(&event)?, event);
    }

    let error = complete_error_data();
    assert_eq!(round_trip(&error)?, error);
    let error = minimal_error_data();
    assert_eq!(round_trip(&error)?, error);

    for kind in EVERY_ERROR_KIND {
        assert_eq!(round_trip(&kind)?, kind);
    }
    Ok(())
}
