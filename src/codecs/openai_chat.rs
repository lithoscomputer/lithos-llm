use reqwest::Method;
use serde_json::{Value, json, to_string};

use super::Codec;
use super::common::{arguments, endpoint, finish_reason, options, plain_text, tool_choice, usage};
use crate::adapter::ResolvedCall;
use crate::resolver::ResolvedRoute;
use crate::transport::{EncodedRequest, SseEvent};
use crate::types::{
    ContentPart, Error, ErrorKind, Message, Response, ResponseFormat, Role, StreamEvent, ToolCall,
    ToolChoice,
};

#[derive(Clone, Copy, Debug)]
pub(crate) struct OpenAiChatCodec;

impl Codec for OpenAiChatCodec {
    fn encode(&self, call: &ResolvedCall, stream: bool) -> Result<EncodedRequest, Error> {
        let request = call.request();
        let mut body = options(request, "openai_compatible");
        body.insert("model".to_owned(), call.route().model().api_model().into());
        body.insert(
            "messages".to_owned(),
            Value::Array(request.messages().iter().flat_map(messages).collect()),
        );
        body.insert("stream".to_owned(), stream.into());
        if stream {
            body.insert(
                "stream_options".to_owned(),
                json!({ "include_usage": true }),
            );
        }
        if let Some(max_tokens) = request.max_output_tokens() {
            body.insert("max_tokens".to_owned(), max_tokens.into());
        }
        if let Some(temperature) = request.temperature() {
            body.insert("temperature".to_owned(), temperature.into());
        }
        if let Some(top_p) = request.top_p() {
            body.insert("top_p".to_owned(), top_p.into());
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
                                "function": {
                                    "name": tool.name,
                                    "description": tool.description,
                                    "parameters": tool.input_schema,
                                }
                            })
                        })
                        .collect(),
                ),
            );
        }
        if let Some(choice) = request.tool_choice() {
            let choice = match choice {
                ToolChoice::Tool { name } => {
                    json!({ "type": "function", "function": { "name": name } })
                }
                _ => tool_choice(choice),
            };
            body.insert("tool_choice".to_owned(), choice);
        }
        if let Some(format) = request.response_format() {
            let value = match format {
                ResponseFormat::Text => json!({ "type": "text" }),
                ResponseFormat::JsonObject => json!({ "type": "json_object" }),
                ResponseFormat::JsonSchema { name, schema } => json!({
                    "type": "json_schema",
                    "json_schema": { "name": name, "schema": schema, "strict": true }
                }),
            };
            body.insert("response_format".to_owned(), value);
        }
        Ok(EncodedRequest {
            method:  Method::POST,
            url:     endpoint(call.route().provider().base_url(), "/v1/chat/completions"),
            headers: Vec::new(),
            body:    Value::Object(body),
            timeout: request.timeout(),
        })
    }

    fn decode_response(&self, route: &ResolvedRoute, value: Value) -> Result<Response, Error> {
        let choice = value.pointer("/choices/0").unwrap_or(&Value::Null);
        let message = choice.get("message").unwrap_or(&Value::Null);
        let mut content = Vec::new();
        if let Some(text) = message.get("content").and_then(Value::as_str) {
            if !text.is_empty() {
                content.push(ContentPart::Text {
                    text: text.to_owned(),
                });
            }
        }
        for call in message
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            content.push(ContentPart::ToolCall(ToolCall {
                id:        call
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                name:      call
                    .pointer("/function/name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                arguments: arguments(call.pointer("/function/arguments").unwrap_or(&Value::Null)),
            }));
        }
        let usage_value = value.get("usage").unwrap_or(&Value::Null);
        let mut token_counts = usage(
            usage_value.get("prompt_tokens").and_then(Value::as_u64),
            usage_value.get("completion_tokens").and_then(Value::as_u64),
        );
        token_counts.cached_input = usage_value
            .pointer("/prompt_tokens_details/cached_tokens")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        token_counts.reasoning_output = usage_value
            .pointer("/completion_tokens_details/reasoning_tokens")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        Ok(Response {
            id: value
                .get("id")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            model: route.handle(),
            content,
            finish_reason: finish_reason(choice.get("finish_reason").and_then(Value::as_str)),
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
                "OpenAI-compatible provider returned an invalid stream event",
            )
            .with_provider(route.provider().id().clone())
            .with_source(source)
        })?;
        let mut events = Vec::new();
        if let Some(id) = value.get("id").and_then(Value::as_str) {
            events.push(StreamEvent::Started {
                id: Some(id.to_owned()),
            });
        }
        if let Some(text) = value
            .pointer("/choices/0/delta/content")
            .and_then(Value::as_str)
        {
            events.push(StreamEvent::TextDelta {
                text: text.to_owned(),
            });
        }
        for call in value
            .pointer("/choices/0/delta/tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            events.push(StreamEvent::ToolCallDelta {
                id:        call
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                name:      call
                    .pointer("/function/name")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                arguments: call
                    .pointer("/function/arguments")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            });
        }
        if let Some(reason) = value
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str)
        {
            events.push(StreamEvent::Finished {
                reason: finish_reason(Some(reason)),
            });
        }
        if let Some(usage_value) = value.get("usage") {
            events.push(StreamEvent::Usage {
                usage: usage(
                    usage_value.get("prompt_tokens").and_then(Value::as_u64),
                    usage_value.get("completion_tokens").and_then(Value::as_u64),
                ),
            });
        }
        Ok(events)
    }
}

fn messages(message: &Message) -> Vec<Value> {
    let tool_results = message.content().iter().filter_map(|part| match part {
        ContentPart::ToolResult(result) => {
            let text = plain_text(&result.content);
            let content = if text.is_empty() {
                to_string(&result.content).unwrap_or_default()
            } else {
                text
            };
            Some(json!({
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "content": content,
            }))
        }
        _ => None,
    });
    if message.role() == Role::Tool {
        return tool_results.collect();
    }

    let role = match message.role() {
        Role::System => "system",
        Role::Developer => "developer",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };
    let parts = message
        .content()
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text { text } => Some(json!({ "type": "text", "text": text })),
            ContentPart::Image(image) => Some(json!({
                "type": "image_url",
                "image_url": { "url": image.source, "detail": image.detail },
            })),
            _ => None,
        })
        .collect::<Vec<_>>();
    let content = match parts.as_slice() {
        [part] if part.get("type").and_then(Value::as_str) == Some("text") => {
            part.get("text").cloned().unwrap_or(Value::Null)
        }
        [] => Value::Null,
        _ => Value::Array(parts),
    };
    let tool_calls = message
        .content()
        .iter()
        .filter_map(|part| match part {
            ContentPart::ToolCall(call) => Some(json!({
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": to_string(&call.arguments).unwrap_or_default(),
                }
            })),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut value = json!({ "role": role, "content": content });
    if !tool_calls.is_empty() {
        value["tool_calls"] = Value::Array(tool_calls);
    }
    vec![value]
}

#[cfg(all(test, feature = "builtin-catalog"))]
mod tests {
    use std::error::Error as StdError;

    use serde_json::json;

    use super::{Codec, OpenAiChatCodec};
    use crate::codecs::test_support::resolved;
    use crate::types::{ContentPart, Message, Request, Role, ToolCall, ToolResult};

    #[test]
    fn encodes_chat_tool_calls_and_results() -> Result<(), Box<dyn StdError>> {
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
                            text: "snow".to_owned(),
                        }],
                        is_error:     false,
                    },
                )]))
                .build()?,
        )?;
        let codec = OpenAiChatCodec;

        let encoded = codec.encode(&call, false)?;

        assert_eq!(encoded.body["messages"][0]["role"], "assistant");
        assert_eq!(encoded.body["messages"][0]["tool_calls"][0]["id"], "call-1");
        assert_eq!(encoded.body["messages"][1]["role"], "tool");
        assert_eq!(encoded.body["messages"][1]["tool_call_id"], "call-1");
        Ok(())
    }
}
