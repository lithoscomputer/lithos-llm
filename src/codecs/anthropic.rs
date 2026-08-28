use reqwest::Method;
use serde_json::{Value, json};

use super::Codec;
use super::common::{endpoint, finish_reason, options, plain_text, system_text, usage};
use crate::adapter::ResolvedCall;
use crate::resolver::ResolvedRoute;
use crate::transport::{EncodedRequest, SseEvent};
use crate::types::{
    ContentPart, Error, ErrorKind, ImageContent, ReasoningContent, ReasoningEffort, Response,
    ResponseFormat, Role, Speed, StreamEvent, ToolCall, ToolChoice,
};

#[derive(Clone, Copy, Debug)]
pub(crate) struct AnthropicMessagesCodec;

impl Codec for AnthropicMessagesCodec {
    fn encode(&self, call: &ResolvedCall, stream: bool) -> Result<EncodedRequest, Error> {
        let request = call.request();
        let mut body = options(request, "anthropic");
        body.insert("model".to_owned(), call.route().model().api_model().into());
        body.insert(
            "max_tokens".to_owned(),
            request.max_output_tokens().unwrap_or(4096).into(),
        );
        body.insert("stream".to_owned(), stream.into());
        let system = system_text(request.messages());
        if !system.is_empty() {
            body.insert("system".to_owned(), system.into());
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
        if let Some(temperature) = request.temperature() {
            body.insert("temperature".to_owned(), temperature.into());
        }
        if let Some(top_p) = request.top_p() {
            body.insert("top_p".to_owned(), top_p.into());
        }
        let mut output_config = body
            .remove("output_config")
            .and_then(|value| value.as_object().cloned())
            .unwrap_or_default();
        if let Some(effort) = request.reasoning_effort() {
            output_config.insert("effort".to_owned(), anthropic_effort(effort).into());
        }
        if let Some(format) = request.response_format() {
            if !matches!(format, ResponseFormat::Text) {
                let schema = match format {
                    ResponseFormat::JsonSchema { schema, .. } => schema.clone(),
                    ResponseFormat::JsonObject => json!({
                        "type": "object",
                        "additionalProperties": true,
                    }),
                    ResponseFormat::Text => Value::Null,
                };
                output_config.insert(
                    "format".to_owned(),
                    json!({ "type": "json_schema", "schema": schema }),
                );
            }
        }
        if !output_config.is_empty() {
            body.insert("output_config".to_owned(), Value::Object(output_config));
        }
        if let Some(speed) = request.speed() {
            match speed {
                Speed::Fast => {
                    body.insert("speed".to_owned(), "fast".into());
                }
                Speed::Balanced => {
                    body.insert("service_tier".to_owned(), "auto".into());
                }
                Speed::Economical => {
                    body.insert("service_tier".to_owned(), "standard_only".into());
                }
            }
        }
        if !request.tools().is_empty() {
            body.insert(
                "tools".to_owned(),
                Value::Array(
                    request
                        .tools()
                        .iter()
                        .map(|tool| {
                            json!({
                                "name": tool.name,
                                "description": tool.description,
                                "input_schema": tool.input_schema,
                            })
                        })
                        .collect(),
                ),
            );
        }
        if let Some(choice) = request.tool_choice() {
            let choice = match choice {
                ToolChoice::Auto => json!({ "type": "auto" }),
                ToolChoice::None => json!({ "type": "none" }),
                ToolChoice::Required => json!({ "type": "any" }),
                ToolChoice::Tool { name } => {
                    json!({ "type": "tool", "name": name })
                }
            };
            body.insert("tool_choice".to_owned(), choice);
        }
        Ok(EncodedRequest {
            method:  Method::POST,
            url:     endpoint(call.route().provider().base_url(), "/v1/messages"),
            headers: vec![("anthropic-version".to_owned(), "2023-06-01".to_owned())],
            body:    Value::Object(body),
            timeout: request.timeout(),
        })
    }

    fn decode_response(&self, route: &ResolvedRoute, value: Value) -> Result<Response, Error> {
        let content = value
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|part| match part.get("type").and_then(Value::as_str) {
                Some("text") => {
                    part.get("text")
                        .and_then(Value::as_str)
                        .map(|text| ContentPart::Text {
                            text: text.to_owned(),
                        })
                }
                Some("thinking") => part.get("thinking").and_then(Value::as_str).map(|text| {
                    ContentPart::Reasoning(ReasoningContent {
                        text:      text.to_owned(),
                        signature: part
                            .get("signature")
                            .and_then(Value::as_str)
                            .map(ToOwned::to_owned),
                    })
                }),
                Some("tool_use") => Some(ContentPart::ToolCall(ToolCall {
                    id:        part
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    name:      part
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    arguments: part.get("input").cloned().unwrap_or(Value::Null),
                })),
                _ => None,
            })
            .collect();
        let usage_value = value.get("usage").unwrap_or(&Value::Null);
        let mut token_counts = usage(
            usage_value.get("input_tokens").and_then(Value::as_u64),
            usage_value.get("output_tokens").and_then(Value::as_u64),
        );
        token_counts.cached_input = usage_value
            .get("cache_read_input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        Ok(Response {
            id: value
                .get("id")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            model: route.handle(),
            content,
            finish_reason: finish_reason(value.get("stop_reason").and_then(Value::as_str)),
            usage: token_counts,
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
                "Anthropic returned an invalid stream event",
            )
            .with_provider(route.provider().id().clone())
            .with_source(source)
        })?;
        let event_type = value
            .get("type")
            .and_then(Value::as_str)
            .or(event.event.as_deref());
        let mut events = Vec::new();
        match event_type {
            Some("message_start") => {
                events.push(StreamEvent::Started {
                    id: value
                        .pointer("/message/id")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned),
                });
                if let Some(usage_value) = value.pointer("/message/usage") {
                    events.push(StreamEvent::Usage {
                        usage: usage(
                            usage_value.get("input_tokens").and_then(Value::as_u64),
                            usage_value.get("output_tokens").and_then(Value::as_u64),
                        ),
                    });
                }
            }
            Some("content_block_start")
                if value.pointer("/content_block/type").and_then(Value::as_str)
                    == Some("tool_use") =>
            {
                events.push(StreamEvent::ToolCallDelta {
                    id:        value
                        .pointer("/content_block/id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    name:      value
                        .pointer("/content_block/name")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned),
                    arguments: String::new(),
                });
            }
            Some("content_block_delta") => {
                let delta = &value["delta"];
                match delta.get("type").and_then(Value::as_str) {
                    Some("text_delta") => {
                        if let Some(text) = delta.get("text").and_then(Value::as_str) {
                            events.push(StreamEvent::TextDelta {
                                text: text.to_owned(),
                            });
                        }
                    }
                    Some("thinking_delta") => {
                        if let Some(text) = delta.get("thinking").and_then(Value::as_str) {
                            events.push(StreamEvent::ReasoningDelta {
                                text: text.to_owned(),
                            });
                        }
                    }
                    Some("input_json_delta") => {
                        events.push(StreamEvent::ToolCallDelta {
                            id:        String::new(),
                            name:      None,
                            arguments: delta
                                .get("partial_json")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_owned(),
                        });
                    }
                    _ => {}
                }
            }
            Some("message_delta") => {
                if let Some(reason) = value.pointer("/delta/stop_reason").and_then(Value::as_str) {
                    events.push(StreamEvent::Finished {
                        reason: finish_reason(Some(reason)),
                    });
                }
                if let Some(output) = value
                    .pointer("/usage/output_tokens")
                    .and_then(Value::as_u64)
                {
                    events.push(StreamEvent::Usage {
                        usage: usage(None, Some(output)),
                    });
                }
            }
            Some("error") => {
                return Err(
                    Error::new(ErrorKind::Provider, "Anthropic stream returned an error")
                        .with_provider(route.provider().id().clone())
                        .with_raw_data(value),
                );
            }
            _ => {}
        }
        Ok(events)
    }
}

fn anthropic_effort(effort: ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::Minimal | ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::Xhigh => "max",
    }
}

fn content(part: &ContentPart) -> Option<Value> {
    match part {
        ContentPart::Text { text } => Some(json!({ "type": "text", "text": text })),
        ContentPart::Image(image) => Some(json!({
            "type": "image",
            "source": image_source(image),
        })),
        ContentPart::Document(document) => Some(json!({
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": document.media_type,
                "data": encoded_data(&document.data),
            },
            "title": document.name,
        })),
        ContentPart::ToolCall(call) => Some(json!({
            "type": "tool_use",
            "id": call.id,
            "name": call.name,
            "input": call.arguments,
        })),
        ContentPart::ToolResult(result) => Some(json!({
            "type": "tool_result",
            "tool_use_id": result.tool_call_id,
            "content": plain_text(&result.content),
            "is_error": result.is_error,
        })),
        ContentPart::Reasoning(reasoning) => Some(json!({
            "type": "thinking",
            "thinking": reasoning.text,
            "signature": reasoning.signature,
        })),
        ContentPart::Audio(_) => None,
    }
}

fn image_source(image: &ImageContent) -> Value {
    if let Some((header, data)) = image.source.split_once(',') {
        if let Some(media_type) = header
            .strip_prefix("data:")
            .and_then(|header| header.strip_suffix(";base64"))
        {
            return json!({ "type": "base64", "media_type": media_type, "data": data });
        }
    }
    if let Some(media_type) = &image.media_type {
        if !image.source.contains("://") {
            return json!({ "type": "base64", "media_type": media_type, "data": image.source });
        }
    }
    json!({ "type": "url", "url": image.source })
}

fn encoded_data(value: &str) -> &str {
    value.split_once(',').map_or(value, |(_, encoded)| encoded)
}

#[cfg(all(test, feature = "builtin-catalog"))]
mod tests {
    use std::error::Error as StdError;

    use serde_json::json;

    use super::{AnthropicMessagesCodec, Codec};
    use crate::codecs::test_support::resolved;
    use crate::types::{ContentPart, ReasoningEffort, Request, ResponseFormat, Speed};

    #[test]
    fn encodes_output_controls_and_decodes_tool_calls() -> Result<(), Box<dyn StdError>> {
        let call = resolved(
            Request::builder()
                .model("anthropic/claude-sonnet-4-6")
                .user("Plan a trip")
                .reasoning_effort(ReasoningEffort::High)
                .speed(Speed::Fast)
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

        let response = codec.decode_response(
            call.route(),
            json!({
                "id": "msg-1",
                "content": [
                    { "type": "thinking", "thinking": "checked", "signature": "sig" },
                    { "type": "tool_use", "id": "tool-1", "name": "lookup", "input": { "q": "x" } }
                ],
                "stop_reason": "tool_use",
                "usage": { "input_tokens": 10, "output_tokens": 4, "cache_read_input_tokens": 2 }
            }),
        )?;

        assert!(matches!(
            response.content.as_slice(),
            [ContentPart::Reasoning(_), ContentPart::ToolCall(call)]
                if call.id == "tool-1" && call.name == "lookup"
        ));
        assert_eq!(response.usage.cached_input, 2);
        Ok(())
    }
}
