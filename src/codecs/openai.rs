use reqwest::Method;
use serde_json::{Map, Value, json, to_string};

use super::Codec;
use super::common::{arguments, endpoint, finish_reason, options, plain_text, tool_choice, usage};
use crate::adapter::ResolvedCall;
use crate::resolver::ResolvedRoute;
use crate::transport::{EncodedRequest, SseEvent};
use crate::types::{
    ContentPart, Error, ErrorKind, FinishReason, Message, ReasoningContent, Response,
    ResponseFormat, Role, Speed, StreamEvent, ToolCall,
};

#[derive(Clone, Copy, Debug)]
pub(crate) struct OpenAiResponsesCodec;

impl Codec for OpenAiResponsesCodec {
    fn encode(&self, call: &ResolvedCall, stream: bool) -> Result<EncodedRequest, Error> {
        let request = call.request();
        let mut body = options(request, "openai");
        body.insert(
            "model".to_owned(),
            Value::String(call.route().model().api_model().to_owned()),
        );
        body.insert(
            "input".to_owned(),
            Value::Array(request.messages().iter().flat_map(input_items).collect()),
        );
        body.insert("stream".to_owned(), Value::Bool(stream));
        if let Some(max_tokens) = request.max_output_tokens() {
            body.insert("max_output_tokens".to_owned(), max_tokens.into());
        }
        if let Some(temperature) = request.temperature() {
            body.insert("temperature".to_owned(), temperature.into());
        }
        if let Some(top_p) = request.top_p() {
            body.insert("top_p".to_owned(), top_p.into());
        }
        if let Some(effort) = request.reasoning_effort() {
            body.insert(
                "reasoning".to_owned(),
                json!({ "effort": format!("{effort:?}").to_lowercase() }),
            );
        }
        if let Some(speed) = request.speed() {
            let service_tier = match speed {
                Speed::Fast => "fast",
                Speed::Balanced => "auto",
                Speed::Economical => "flex",
            };
            body.insert("service_tier".to_owned(), service_tier.into());
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
                                "type": "function",
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.input_schema,
                            })
                        })
                        .collect(),
                ),
            );
        }
        if let Some(choice) = request.tool_choice() {
            body.insert("tool_choice".to_owned(), tool_choice(choice));
        }
        if let Some(format) = request.response_format() {
            let format = match format {
                ResponseFormat::Text => json!({ "type": "text" }),
                ResponseFormat::JsonObject => json!({ "type": "json_object" }),
                ResponseFormat::JsonSchema { name, schema } => {
                    json!({ "type": "json_schema", "name": name, "schema": schema, "strict": true })
                }
            };
            body.insert("text".to_owned(), json!({ "format": format }));
        }
        Ok(EncodedRequest {
            method:  Method::POST,
            url:     endpoint(call.route().provider().base_url(), "/v1/responses"),
            headers: Vec::new(),
            body:    Value::Object(body),
            timeout: request.timeout(),
        })
    }

    fn decode_response(&self, route: &ResolvedRoute, value: Value) -> Result<Response, Error> {
        let mut content = Vec::new();
        for item in value
            .get("output")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            match item.get("type").and_then(Value::as_str) {
                Some("message") => {
                    for part in item
                        .get("content")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                    {
                        if matches!(
                            part.get("type").and_then(Value::as_str),
                            Some("output_text")
                        ) {
                            if let Some(text) = part.get("text").and_then(Value::as_str) {
                                content.push(ContentPart::Text {
                                    text: text.to_owned(),
                                });
                            }
                        }
                    }
                }
                Some("function_call") => content.push(ContentPart::ToolCall(ToolCall {
                    id:        item
                        .get("call_id")
                        .or_else(|| item.get("id"))
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    name:      item
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    arguments: arguments(item.get("arguments").unwrap_or(&Value::Null)),
                })),
                Some("reasoning") => {
                    if let Some(text) = item.pointer("/summary/0/text").and_then(Value::as_str) {
                        content.push(ContentPart::Reasoning(ReasoningContent {
                            text:      text.to_owned(),
                            signature: None,
                        }));
                    }
                }
                _ => {}
            }
        }
        let usage_value = value.get("usage").unwrap_or(&Value::Null);
        let mut token_counts = usage(
            usage_value.get("input_tokens").and_then(Value::as_u64),
            usage_value.get("output_tokens").and_then(Value::as_u64),
        );
        token_counts.cached_input = usage_value
            .pointer("/input_tokens_details/cached_tokens")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        token_counts.reasoning_output = usage_value
            .pointer("/output_tokens_details/reasoning_tokens")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        let status = value.get("status").and_then(Value::as_str);
        Ok(Response {
            id: value
                .get("id")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            model: route.handle(),
            content,
            finish_reason: match status {
                Some("incomplete") => FinishReason::Length,
                Some("failed") => FinishReason::Error,
                _ => FinishReason::Stop,
            },
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
                "OpenAI returned an invalid stream event",
            )
            .with_provider(route.provider().id().clone())
            .with_source(source)
        })?;
        let event_type = value
            .get("type")
            .and_then(Value::as_str)
            .or(event.event.as_deref());
        let event = match event_type {
            Some("response.created") => Some(StreamEvent::Started {
                id: value
                    .pointer("/response/id")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
            }),
            Some("response.output_text.delta") => {
                value
                    .get("delta")
                    .and_then(Value::as_str)
                    .map(|text| StreamEvent::TextDelta {
                        text: text.to_owned(),
                    })
            }
            Some("response.reasoning_text.delta" | "response.reasoning_summary_text.delta") => {
                value
                    .get("delta")
                    .and_then(Value::as_str)
                    .map(|text| StreamEvent::ReasoningDelta {
                        text: text.to_owned(),
                    })
            }
            Some("response.function_call_arguments.delta") => Some(StreamEvent::ToolCallDelta {
                id:        value
                    .get("item_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                name:      None,
                arguments: value
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            }),
            Some("response.output_item.done")
                if value.pointer("/item/type").and_then(Value::as_str) == Some("function_call") =>
            {
                let item = &value["item"];
                Some(StreamEvent::ToolCall {
                    call: ToolCall {
                        id:        item
                            .get("call_id")
                            .or_else(|| item.get("id"))
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                        name:      item
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                        arguments: arguments(item.get("arguments").unwrap_or(&Value::Null)),
                    },
                })
            }
            Some("response.completed") => {
                let response = self.decode_response(route, value["response"].clone())?;
                Some(StreamEvent::Completed { response })
            }
            Some("response.incomplete") => Some(StreamEvent::Finished {
                reason: finish_reason(Some("length")),
            }),
            _ => None,
        };
        Ok(event.into_iter().collect())
    }
}

fn input_items(message: &Message) -> Vec<Value> {
    let role = match message.role() {
        Role::System => "system",
        Role::Developer => "developer",
        Role::User | Role::Tool => "user",
        Role::Assistant => "assistant",
    };
    let content = message
        .content()
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text { text } => Some(json!({ "type": "input_text", "text": text })),
            ContentPart::Image(image) => Some(json!({
                "type": "input_image",
                "image_url": image.source,
                "detail": image.detail,
            })),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut items = Vec::new();
    if !content.is_empty() {
        let mut value = Map::new();
        value.insert("role".to_owned(), Value::String(role.to_owned()));
        value.insert("content".to_owned(), Value::Array(content));
        items.push(Value::Object(value));
    }
    items.extend(message.content().iter().filter_map(|part| match part {
        ContentPart::ToolCall(call) => Some(json!({
            "type": "function_call",
            "call_id": call.id,
            "name": call.name,
            "arguments": to_string(&call.arguments).unwrap_or_default(),
        })),
        ContentPart::ToolResult(result) => {
            let text = plain_text(&result.content);
            let output = if text.is_empty() {
                to_string(&result.content).unwrap_or_default()
            } else {
                text
            };
            Some(json!({
                "type": "function_call_output",
                "call_id": result.tool_call_id,
                "output": output,
            }))
        }
        _ => None,
    }));
    items
}

#[cfg(all(test, feature = "builtin-catalog"))]
mod tests {
    use std::error::Error as StdError;

    use serde_json::json;

    use super::{Codec, OpenAiResponsesCodec};
    use crate::codecs::test_support::resolved;
    use crate::types::{ContentPart, Message, Request, Role, Speed, ToolCall, ToolResult};

    #[test]
    fn preserves_tool_protocol_and_decodes_reasoning() -> Result<(), Box<dyn StdError>> {
        let call = resolved(
            Request::builder()
                .model("openai/gpt-5.6-luna")
                .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
                    ToolCall {
                        id:        "call-1".to_owned(),
                        name:      "weather".to_owned(),
                        arguments: json!({ "city": "Boston" }),
                    },
                )]))
                .message(Message::new(Role::Tool, [ContentPart::ToolResult(
                    ToolResult {
                        tool_call_id: "call-1".to_owned(),
                        name:         Some("weather".to_owned()),
                        content:      vec![ContentPart::Text {
                            text: "cold".to_owned(),
                        }],
                        is_error:     false,
                    },
                )]))
                .speed(Speed::Economical)
                .build()?,
        )?;
        let codec = OpenAiResponsesCodec;

        let encoded = codec.encode(&call, false)?;

        assert_eq!(encoded.body["service_tier"], "flex");
        assert_eq!(encoded.body["input"][0]["type"], "function_call");
        assert_eq!(
            encoded.body["input"][0]["arguments"],
            r#"{"city":"Boston"}"#
        );
        assert_eq!(encoded.body["input"][1]["type"], "function_call_output");
        assert_eq!(encoded.body["input"][1]["call_id"], "call-1");

        let response = codec.decode_response(
            call.route(),
            json!({
                "id": "resp-1",
                "status": "completed",
                "output": [
                    { "type": "reasoning", "summary": [{ "text": "checked" }] },
                    {
                        "type": "function_call",
                        "call_id": "call-2",
                        "name": "coat",
                        "arguments": "{\"needed\":true}"
                    }
                ],
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 7,
                    "input_tokens_details": { "cached_tokens": 3 },
                    "output_tokens_details": { "reasoning_tokens": 2 }
                }
            }),
        )?;

        assert!(matches!(
            response.content.as_slice(),
            [ContentPart::Reasoning(_), ContentPart::ToolCall(call)]
                if call.id == "call-2" && call.arguments == json!({ "needed": true })
        ));
        assert_eq!(response.usage.cached_input, 3);
        assert_eq!(response.usage.reasoning_output, 2);
        Ok(())
    }
}
