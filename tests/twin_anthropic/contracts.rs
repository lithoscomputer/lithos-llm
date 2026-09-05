use std::collections::BTreeMap;

use lithos_llm::Response;
use lithos_llm::types::{
    ContentBlockKind, ContentPart, DocumentContent, FinishReason, ImageContent, MediaSource,
    Message, ReasoningContent, ResponseFormat, Role, StreamEvent, TokenCounts, ToolCall,
    ToolDefinition, ToolResult,
};
use serde_json::{Value, json};

use super::support::{Twin, bounded, collect, config, request, scenario};

pub(super) fn native_content() -> Value {
    json!([
        {"type":"thinking","thinking":"Check both cities.","signature":"signed-α-123"},
        {"type":"redacted_thinking","data":"opaque-encrypted-data"},
        {"type":"text","text":"Hello 🌊\nchecking cities."},
        {"type":"tool_use","id":"toolu_paris","name":"weather","input":{"city":"Paris","units":"°C"}},
        {"type":"tool_use","id":"toolu_tokyo","name":"weather","input":{"city":"東京","units":"°C"}}
    ])
}

pub(super) fn native_usage() -> Value {
    json!({"input_tokens":11,"output_tokens":19,"cache_creation_input_tokens":7,"cache_read_input_tokens":13,"cache_creation":{"ephemeral_5m_input_tokens":3,"ephemeral_1h_input_tokens":4}})
}

pub(super) fn success() -> Value {
    json!({"kind":"success","content":native_content(),"usage":native_usage(),"stop_reason":"tool_use"})
}

pub(super) fn expected_parts() -> Vec<ContentPart> {
    vec![
        ContentPart::Reasoning(ReasoningContent {
            text:             "Check both cities.".to_owned(),
            signature:        Some("signed-α-123".to_owned()),
            signature_origin: Some("anthropic".to_owned()),
            redacted:         false,
        }),
        ContentPart::Reasoning(ReasoningContent {
            text:             "opaque-encrypted-data".to_owned(),
            signature:        None,
            signature_origin: None,
            redacted:         true,
        }),
        ContentPart::Text {
            text: "Hello 🌊\nchecking cities.".to_owned(),
        },
        ContentPart::ToolCall(ToolCall::function(
            "toolu_paris",
            "weather",
            json!({"city":"Paris","units":"°C"}),
        )),
        ContentPart::ToolCall(ToolCall::function(
            "toolu_tokyo",
            "weather",
            json!({"city":"東京","units":"°C"}),
        )),
    ]
}

pub(super) fn expected() -> Value {
    json!({"content":expected_parts(),"finish_reason":FinishReason::ToolCall,"usage":TokenCounts { input:11,output:19,reasoning:0,cache_write:7,cache_read:13 }})
}

pub(super) fn semantic(response: &Response) -> Value {
    json!({"content":response.content,"finish_reason":response.finish_reason,"usage":response.usage})
}

pub(super) fn completed(events: &[StreamEvent]) -> &Response {
    let mut open = BTreeMap::new();
    let mut closed = BTreeMap::new();
    let mut order = Vec::new();
    let mut deltas = BTreeMap::new();
    let mut completion = None;
    for event in events {
        assert!(completion.is_none(), "event after completion: {event:?}");
        match event {
            StreamEvent::ContentBlockStart { id, kind } => {
                assert!(
                    !closed.contains_key(id) && open.insert(id.clone(), kind.clone()).is_none(),
                    "duplicate block id"
                );
                order.push(id.clone());
                deltas.insert(id.clone(), String::new());
            }
            StreamEvent::TextDelta { id, text } => {
                assert!(
                    matches!(open.get(id), Some(ContentBlockKind::Text)),
                    "text delta outside text block"
                );
                deltas.get_mut(id).expect("open text block").push_str(text);
            }
            StreamEvent::ReasoningDelta { id, text } => {
                assert!(
                    matches!(open.get(id), Some(ContentBlockKind::Reasoning)),
                    "reasoning delta outside reasoning block"
                );
                deltas
                    .get_mut(id)
                    .expect("open reasoning block")
                    .push_str(text);
            }
            StreamEvent::ToolCallDelta { id, arguments } => {
                assert!(
                    matches!(open.get(id), Some(ContentBlockKind::ToolCall { .. })),
                    "argument delta outside tool block"
                );
                deltas
                    .get_mut(id)
                    .expect("open tool block")
                    .push_str(arguments);
            }
            StreamEvent::ContentBlockEnd { id, part } => {
                assert!(open.remove(id).is_some(), "end without start");
                let assembled = deltas.remove(id).expect("block accumulator");
                match part {
                    ContentPart::Text { text } => assert_eq!(&assembled, text),
                    ContentPart::Reasoning(reasoning) if !reasoning.redacted => {
                        assert_eq!(assembled, reasoning.text);
                    }
                    ContentPart::ToolCall(call) => assert_eq!(assembled, call.input.raw()),
                    _ => {}
                }
                assert!(closed.insert(id.clone(), part.clone()).is_none());
            }
            StreamEvent::Ended { response } => {
                assert!(open.is_empty(), "completion with open blocks");
                completion = Some(response);
            }
            _ => {}
        }
    }
    let response = completion.expect("one completed response");
    assert_eq!(closed.len(), response.content.len(), "closed block count");
    let parts: Vec<_> = order.iter().map(|id| closed[id].clone()).collect();
    assert_eq!(
        parts, response.content,
        "final content differs from ordered block ends"
    );
    response
}

fn weather() -> ToolDefinition {
    ToolDefinition::function(
        "weather",
        "Read weather",
        json!({"type":"object","properties":{"city":{"type":"string"},"units":{"type":"string"}},"required":["city","units"]}),
    )
}

#[tokio::test]
async fn complete_and_stream_preserve_every_content_and_usage_value() {
    let twin = Twin::start(config()).await;
    for streaming in [false, true] {
        twin.enqueue("contract", json!([scenario(success())])).await;
        let client = twin.client("contract");
        let req = request()
            .into_builder()
            .tool(weather())
            .build()
            .expect("tools");
        let response = if streaming {
            let events: Vec<_> = collect(bounded(client.stream(req)).await.expect("stream"))
                .await
                .into_iter()
                .collect::<Result<_, _>>()
                .expect("stream items");
            completed(&events).clone()
        } else {
            bounded(client.complete(req)).await.expect("complete")
        };
        assert_eq!(semantic(&response), expected());
        assert_eq!(response.model.provider().as_str(), "anthropic");
        if !streaming {
            assert_eq!(
                response.raw.as_ref().expect("native response")["usage"],
                native_usage()
            );
        }
    }
    twin.shutdown().await;
}

#[tokio::test]
async fn parallel_tool_results_and_thinking_are_replayed_without_loss() {
    let twin = Twin::start(config()).await;
    twin.enqueue(
        "roundtrip",
        json!([
            scenario(success()),
            scenario(json!({"kind":"success","response_text":"Both cities checked."}))
        ]),
    )
    .await;
    let client = twin.client("roundtrip");
    let first = bounded(
        client.complete(
            request()
                .into_builder()
                .tool(weather())
                .build()
                .expect("tools"),
        ),
    )
    .await
    .expect("first response");
    assert_eq!(semantic(&first), expected());
    let mut followup = request()
        .into_builder()
        .tool(weather())
        .message(first.into_message());
    for (id, city) in [("toolu_paris", "Paris"), ("toolu_tokyo", "東京")] {
        followup = followup.message(Message::new(Role::Tool, [ContentPart::ToolResult(
            ToolResult {
                tool_call_id: id.to_owned(),
                name:         Some("weather".to_owned()),
                content:      vec![ContentPart::Json {
                    value: json!({"city":city,"temperature":18}),
                }],
                is_error:     false,
            },
        )]));
    }
    let response = bounded(client.complete(followup.build().expect("followup")))
        .await
        .expect("second response");
    assert_eq!(response.content, vec![ContentPart::Text {
        text: "Both cities checked.".to_owned(),
    }]);
    let capture = twin.captures().pop().expect("captured followup");
    assert_eq!(capture.body["messages"][1]["content"], native_content());
    let results: Vec<_> = capture.body["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .flat_map(|message| message["content"].as_array().expect("content blocks"))
        .filter(|block| block["type"] == "tool_result")
        .cloned()
        .collect();
    assert_eq!(results, vec![
        json!({"type":"tool_result","tool_use_id":"toolu_paris","content":[{"type":"text","text":json!({"city":"Paris","temperature":18}).to_string()}],"is_error":false}),
        json!({"type":"tool_result","tool_use_id":"toolu_tokyo","content":[{"type":"text","text":json!({"city":"東京","temperature":18}).to_string()}],"is_error":false}),
    ]);
    assert_eq!(
        twin.logs("roundtrip").await["requests"]
            .as_array()
            .expect("logs")
            .len(),
        2
    );
    twin.shutdown().await;
}

#[tokio::test]
async fn multimodal_input_reaches_the_twin_with_exact_sources() {
    let twin = Twin::start(config()).await;
    let req = request()
        .into_builder()
        .message(Message::new(Role::User, [
            ContentPart::Image(ImageContent {
                source: MediaSource::url("https://example.invalid/image.png"),
                detail: None,
            }),
            ContentPart::Image(ImageContent {
                source: MediaSource::base64("aW1hZ2U=", "image/png"),
                detail: None,
            }),
            ContentPart::Document(DocumentContent {
                source: MediaSource::url("https://example.invalid/doc.pdf"),
                name:   Some("Document".to_owned()),
            }),
            ContentPart::Document(DocumentContent {
                source: MediaSource::base64("cGRm", "application/pdf"),
                name:   None,
            }),
        ]))
        .build()
        .expect("multimodal request");
    let client = twin.client("media");
    for streaming in [false, true] {
        if streaming {
            let events: Vec<_> = collect(client.stream(req.clone()).await.expect("media stream"))
                .await
                .into_iter()
                .collect::<Result<_, _>>()
                .expect("media events");
            assert_eq!(completed(&events).finish_reason, FinishReason::Stop);
        } else {
            assert_eq!(
                client
                    .complete(req.clone())
                    .await
                    .expect("media complete")
                    .finish_reason,
                FinishReason::Stop
            );
        }
        let capture = twin.captures().pop().expect("media capture");
        let sources: Vec<_> = capture.body["messages"]
            .as_array()
            .expect("messages")
            .iter()
            .flat_map(|message| message["content"].as_array().expect("content"))
            .filter_map(|block| block.get("source"))
            .cloned()
            .collect();
        assert_eq!(sources, vec![
            json!({"type":"url","url":"https://example.invalid/image.png"}),
            json!({"type":"base64","data":"aW1hZ2U=","media_type":"image/png"}),
            json!({"type":"url","url":"https://example.invalid/doc.pdf"}),
            json!({"type":"base64","data":"cGRm","media_type":"application/pdf"}),
        ]);
    }
    twin.shutdown().await;
}

#[tokio::test]
async fn output_truncation_never_returns_tools_for_execution() {
    let twin = Twin::start(config()).await;
    let mut script = success();
    script["stop_reason"] = json!("max_tokens");
    for streaming in [false, true] {
        twin.enqueue("truncated", json!([scenario(script.clone())]))
            .await;
        let client = twin.client("truncated");
        let response = if streaming {
            let events: Vec<_> = collect(client.stream(request()).await.expect("stream"))
                .await
                .into_iter()
                .collect::<Result<_, _>>()
                .expect("events");
            events
                .into_iter()
                .find_map(|e| {
                    if let StreamEvent::Ended { response } = e {
                        Some(response)
                    } else {
                        None
                    }
                })
                .expect("completion")
        } else {
            client.complete(request()).await.expect("complete")
        };
        assert_eq!(response.finish_reason, FinishReason::Length);
        assert_eq!(response.tool_calls().count(), 0);
        assert_eq!(
            response
                .warnings
                .iter()
                .filter(|w| w.code == "truncated_tool_call")
                .count(),
            2
        );
    }
    twin.shutdown().await;
}

#[tokio::test]
async fn structured_output_counting_and_beta_headers_use_the_real_client() {
    let twin = Twin::start(config()).await;
    let document = json!({"city":"Paris","population":2_102_650});
    for streaming in [false, true] {
        twin.enqueue(
            "structured",
            json!([scenario(
                json!({"kind":"success","structured_output":document})
            )]),
        )
        .await;
        let req = request().into_builder().response_format(ResponseFormat::JsonSchema {name:"city".to_owned(),schema:json!({"type":"object","properties":{"city":{"type":"string"},"population":{"type":"integer"}},"required":["city","population"],"additionalProperties":false})}).provider_option("anthropic","beta_headers",json!(["test-beta-a","test-beta-b"])).build().expect("schema request");
        let client = twin.client("structured");
        let response = if streaming {
            let events: Vec<_> = collect(client.stream(req).await.expect("stream"))
                .await
                .into_iter()
                .collect::<Result<_, _>>()
                .expect("items");
            completed(&events).clone()
        } else {
            client.complete(req).await.expect("complete")
        };
        let ContentPart::Text { text } = &response.content[0] else {
            panic!("structured text")
        };
        assert_eq!(
            serde_json::from_str::<Value>(text).expect("valid structured JSON"),
            document
        );
    }
    let captures = twin.captures();
    assert!(captures.iter().all(|c| {
        c.headers
            .get("anthropic-beta")
            .is_some_and(|h| h.contains("test-beta-a") && h.contains("test-beta-b"))
    }));
    twin.enqueue("count",json!([{"matcher":{"endpoint":"messages.count_tokens"},"script":{"kind":"success","input_tokens":123}}])).await;
    let count = twin
        .client("count")
        .count_input_tokens(request())
        .await
        .expect("count")
        .expect("supported count");
    assert_eq!(count.tokens(), 123);
    let capture = twin.captures().pop().expect("count capture");
    assert_eq!(capture.path, "/v1/messages/count_tokens");
    assert!(capture.body.get("max_tokens").is_none());
    twin.shutdown().await;
}

#[tokio::test]
async fn exact_contract_detects_corrupted_tool_signature_usage_and_stop_state() {
    let twin = Twin::start(config()).await;
    for (pointer, replacement) in [
        ("/content/3/input/city", json!("wrong city")),
        ("/content/0/signature", json!("wrong signature")),
        ("/content/1/data", json!("wrong redaction")),
        ("/usage/cache_read_input_tokens", json!(999)),
        ("/usage/output_tokens", json!(999)),
        ("/stop_reason", json!("max_tokens")),
    ] {
        let mut corrupted = success();
        *corrupted.pointer_mut(pointer).expect("mutation location") = replacement;
        for streaming in [false, true] {
            twin.enqueue("mutation", json!([scenario(corrupted.clone())]))
                .await;
            let client = twin.client("mutation");
            let response = if streaming {
                let events: Vec<_> = collect(client.stream(request()).await.expect("stream"))
                    .await
                    .into_iter()
                    .collect::<Result<_, _>>()
                    .expect("items");
                events
                    .iter()
                    .find_map(|e| {
                        if let StreamEvent::Ended { response } = e {
                            Some(response.clone())
                        } else {
                            None
                        }
                    })
                    .expect("completion")
            } else {
                client.complete(request()).await.expect("complete")
            };
            assert_ne!(
                semantic(&response),
                expected(),
                "undetected mutation {pointer}, stream={streaming}"
            );
        }
    }
    twin.shutdown().await;
}
