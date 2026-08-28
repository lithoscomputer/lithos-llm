use reqwest::Method;
use serde_json::{Map, Value, json};

use super::Codec;
use super::common::{endpoint, finish_reason, options, plain_text, system_text, usage};
use crate::adapter::ResolvedCall;
use crate::resolver::ResolvedRoute;
use crate::transport::{EncodedRequest, SseEvent};
use crate::types::{
    ContentPart, Error, ErrorKind, ReasoningContent, ReasoningEffort, Response, Role, Speed,
    StreamEvent, ToolCall,
};

#[derive(Clone, Copy, Debug)]
pub(crate) struct BedrockConverseCodec;

impl Codec for BedrockConverseCodec {
    fn encode(&self, call: &ResolvedCall, stream: bool) -> Result<EncodedRequest, Error> {
        let request = call.request();
        let mut body = options(request, "bedrock");
        let system = system_text(request.messages());
        if !system.is_empty() {
            body.insert("system".to_owned(), json!([{ "text": system }]));
        }
        body.insert(
            "messages".to_owned(),
            Value::Array(
                request
                    .messages()
                    .iter()
                    .filter(|message| {
                        !matches!(
                            message.role(),
                            Role::System | Role::Developer
                        )
                    })
                    .map(|message| {
                        let role = if message.role() == Role::Assistant {
                            "assistant"
                        } else {
                            "user"
                        };
                        json!({
                            "role": role,
                            "content": message.content().iter().filter_map(content).collect::<Vec<_>>(),
                        })
                    })
                    .collect(),
            ),
        );
        let mut inference = Map::new();
        if let Some(max_tokens) = request.max_output_tokens() {
            inference.insert("maxTokens".to_owned(), max_tokens.into());
        }
        if let Some(temperature) = request.temperature() {
            inference.insert("temperature".to_owned(), temperature.into());
        }
        if let Some(top_p) = request.top_p() {
            inference.insert("topP".to_owned(), top_p.into());
        }
        if !inference.is_empty() {
            body.insert("inferenceConfig".to_owned(), Value::Object(inference));
        }
        if !request.tools().is_empty() {
            body.insert(
                "toolConfig".to_owned(),
                json!({ "tools": request.tools().iter().map(|tool| json!({
                    "toolSpec": {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": { "json": tool.input_schema },
                    }
                })).collect::<Vec<_>>() }),
            );
        }
        if let Some(effort) = request.reasoning_effort() {
            let additional = body
                .entry("additionalModelRequestFields".to_owned())
                .or_insert_with(|| json!({}));
            if let Some(additional) = additional.as_object_mut() {
                additional.insert(
                    "output_config".to_owned(),
                    json!({ "effort": bedrock_effort(effort) }),
                );
            }
        }
        if let Some(speed) = request.speed() {
            body.insert(
                "performanceConfig".to_owned(),
                json!({
                    "latency": if matches!(speed, Speed::Fast) {
                        "optimized"
                    } else {
                        "standard"
                    }
                }),
            );
        }
        let model = call
            .route()
            .model()
            .api_model()
            .replace('%', "%25")
            .replace('/', "%2F");
        let operation = if stream {
            "converse-stream"
        } else {
            "converse"
        };
        Ok(EncodedRequest {
            method:  Method::POST,
            url:     endpoint(
                call.route().provider().base_url(),
                &format!("/model/{model}/{operation}"),
            ),
            headers: Vec::new(),
            body:    Value::Object(body),
            timeout: request.timeout(),
        })
    }

    fn decode_response(&self, route: &ResolvedRoute, value: Value) -> Result<Response, Error> {
        let content = value
            .pointer("/output/message/content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|part| {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    Some(ContentPart::Text {
                        text: text.to_owned(),
                    })
                } else if let Some(reasoning) = part.pointer("/reasoningContent/reasoningText") {
                    reasoning.get("text").and_then(Value::as_str).map(|text| {
                        ContentPart::Reasoning(ReasoningContent {
                            text:      text.to_owned(),
                            signature: reasoning
                                .get("signature")
                                .and_then(Value::as_str)
                                .map(ToOwned::to_owned),
                        })
                    })
                } else {
                    part.get("toolUse").map(|call| {
                        ContentPart::ToolCall(ToolCall {
                            id:        call
                                .get("toolUseId")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_owned(),
                            name:      call
                                .get("name")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_owned(),
                            arguments: call.get("input").cloned().unwrap_or(Value::Null),
                        })
                    })
                }
            })
            .collect();
        let usage_value = value.get("usage").unwrap_or(&Value::Null);
        Ok(Response {
            id: None,
            model: route.handle(),
            content,
            finish_reason: finish_reason(value.get("stopReason").and_then(Value::as_str)),
            usage: usage(
                usage_value.get("inputTokens").and_then(Value::as_u64),
                usage_value.get("outputTokens").and_then(Value::as_u64),
            ),
            cost: None,
            rate_limits: None,
            warnings: Vec::new(),
        })
    }

    fn decode_sse(
        &self,
        route: &ResolvedRoute,
        event: SseEvent,
    ) -> Result<Vec<StreamEvent>, Error> {
        let value: Value = serde_json::from_str(&event.data).map_err(|source| {
            Error::new(
                ErrorKind::StreamDecode,
                "Bedrock returned an invalid stream event",
            )
            .with_provider(route.provider().id().clone())
            .with_source(source)
        })?;
        let mut events = Vec::new();
        if value.get("messageStart").is_some() {
            events.push(StreamEvent::Started { id: None });
        }
        if let Some(start) = value.pointer("/contentBlockStart/start/toolUse") {
            events.push(StreamEvent::ToolCallDelta {
                id:        start
                    .get("toolUseId")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                name:      start
                    .get("name")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                arguments: String::new(),
            });
        }
        if let Some(delta) = value.pointer("/contentBlockDelta/delta") {
            if let Some(text) = delta.get("text").and_then(Value::as_str) {
                events.push(StreamEvent::TextDelta {
                    text: text.to_owned(),
                });
            }
            if let Some(arguments) = delta.pointer("/toolUse/input").and_then(Value::as_str) {
                events.push(StreamEvent::ToolCallDelta {
                    id:        String::new(),
                    name:      None,
                    arguments: arguments.to_owned(),
                });
            }
            if let Some(text) = delta
                .pointer("/reasoningContent/text")
                .and_then(Value::as_str)
            {
                events.push(StreamEvent::ReasoningDelta {
                    text: text.to_owned(),
                });
            }
        }
        if let Some(reason) = value
            .pointer("/messageStop/stopReason")
            .and_then(Value::as_str)
        {
            events.push(StreamEvent::Finished {
                reason: finish_reason(Some(reason)),
            });
        }
        if let Some(usage_value) = value.pointer("/metadata/usage") {
            let mut tokens = usage(
                usage_value.get("inputTokens").and_then(Value::as_u64),
                usage_value.get("outputTokens").and_then(Value::as_u64),
            );
            tokens.cached_input = usage_value
                .get("cacheReadInputTokens")
                .and_then(Value::as_u64)
                .unwrap_or_default();
            events.push(StreamEvent::Usage { usage: tokens });
        }
        Ok(events)
    }
}

fn content(part: &ContentPart) -> Option<Value> {
    match part {
        ContentPart::Text { text } => Some(json!({ "text": text })),
        ContentPart::Image(image) => Some(json!({
            "image": {
                "format": image.media_type.as_deref().unwrap_or("png").trim_start_matches("image/"),
                "source": { "bytes": encoded_data(&image.source) },
            }
        })),
        ContentPart::Document(document) => Some(json!({
            "document": {
                "format": document_format(&document.media_type),
                "name": document.name.as_deref().unwrap_or("document"),
                "source": { "bytes": encoded_data(&document.data) },
            }
        })),
        ContentPart::ToolCall(call) => Some(json!({
            "toolUse": { "toolUseId": call.id, "name": call.name, "input": call.arguments }
        })),
        ContentPart::ToolResult(result) => Some(json!({
            "toolResult": {
                "toolUseId": result.tool_call_id,
                "content": [{ "text": plain_text(&result.content) }],
                "status": if result.is_error { "error" } else { "success" },
            }
        })),
        ContentPart::Reasoning(reasoning) => Some(json!({
            "reasoningContent": {
                "reasoningText": {
                    "text": reasoning.text,
                    "signature": reasoning.signature,
                }
            }
        })),
        ContentPart::Audio(_) => None,
    }
}

fn encoded_data(value: &str) -> &str {
    value.split_once(',').map_or(value, |(_, encoded)| encoded)
}

fn document_format(media_type: &str) -> &str {
    match media_type {
        "application/pdf" => "pdf",
        "text/csv" => "csv",
        "text/html" => "html",
        "text/markdown" => "md",
        "text/plain" => "txt",
        "application/msword" => "doc",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => "docx",
        "application/vnd.ms-excel" => "xls",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" => "xlsx",
        other => other,
    }
}

fn bedrock_effort(effort: ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::Minimal | ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::Xhigh => "max",
    }
}

#[cfg(all(test, feature = "builtin-catalog"))]
mod tests {
    use std::error::Error as StdError;

    use super::{BedrockConverseCodec, Codec};
    use crate::codecs::test_support::resolved;
    use crate::types::{
        ContentPart, DocumentContent, Message, ReasoningContent, ReasoningEffort, Request, Role,
        Speed,
    };

    #[test]
    fn encodes_document_reasoning_and_performance_fields() -> Result<(), Box<dyn StdError>> {
        let call = resolved(
            Request::builder()
                .model("bedrock/anthropic.claude-sonnet-4-6")
                .message(Message::new(Role::Assistant, [ContentPart::Reasoning(
                    ReasoningContent {
                        text:      "checked".to_owned(),
                        signature: Some("sig".to_owned()),
                    },
                )]))
                .message(Message::new(Role::User, [ContentPart::Document(
                    DocumentContent {
                        data:       "data:application/pdf;base64,aGVsbG8=".to_owned(),
                        media_type: "application/pdf".to_owned(),
                        name:       Some("source".to_owned()),
                    },
                )]))
                .reasoning_effort(ReasoningEffort::High)
                .speed(Speed::Fast)
                .build()?,
        )?;
        let codec = BedrockConverseCodec;

        let encoded = codec.encode(&call, false)?;

        assert_eq!(
            encoded.body["messages"][0]["content"][0]["reasoningContent"]["reasoningText"]["signature"],
            "sig"
        );
        assert_eq!(
            encoded.body["messages"][1]["content"][0]["document"]["format"],
            "pdf"
        );
        assert_eq!(encoded.body["performanceConfig"]["latency"], "optimized");
        assert_eq!(
            encoded.body["additionalModelRequestFields"]["output_config"]["effort"],
            "high"
        );
        Ok(())
    }
}
