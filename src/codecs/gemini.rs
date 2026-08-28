use reqwest::Method;
use serde_json::{Map, Value, json};

use super::Codec;
use super::common::{endpoint, finish_reason, options, plain_text, system_text, usage};
use crate::adapter::ResolvedCall;
use crate::resolver::ResolvedRoute;
use crate::transport::{EncodedRequest, SseEvent};
use crate::types::{
    ContentPart, Error, ErrorKind, FinishReason, ImageContent, ReasoningContent, Response,
    ResponseFormat, Role, StreamEvent, TokenCounts, ToolCall,
};

#[derive(Clone, Copy, Debug)]
pub(crate) struct GeminiGenerateCodec;

impl Codec for GeminiGenerateCodec {
    fn encode(&self, call: &ResolvedCall, stream: bool) -> Result<EncodedRequest, Error> {
        let request = call.request();
        let mut body = options(request, "gemini");
        let system = system_text(request.messages());
        if !system.is_empty() {
            body.insert(
                "systemInstruction".to_owned(),
                json!({ "parts": [{ "text": system }] }),
            );
        }
        body.insert(
            "contents".to_owned(),
            Value::Array(
                request
                    .messages()
                    .iter()
                    .filter(|message| !matches!(message.role(), Role::System | Role::Developer))
                    .map(|message| {
                        let role = if message.role() == Role::Assistant {
                            "model"
                        } else {
                            "user"
                        };
                        json!({
                            "role": role,
                            "parts": message.content().iter().map(content).collect::<Vec<_>>(),
                        })
                    })
                    .collect(),
            ),
        );
        let mut generation = Map::new();
        if let Some(max_tokens) = request.max_output_tokens() {
            generation.insert("maxOutputTokens".to_owned(), max_tokens.into());
        }
        if let Some(temperature) = request.temperature() {
            generation.insert("temperature".to_owned(), temperature.into());
        }
        if let Some(top_p) = request.top_p() {
            generation.insert("topP".to_owned(), top_p.into());
        }
        if let Some(format) = request.response_format() {
            generation.insert("responseMimeType".to_owned(), "application/json".into());
            if let ResponseFormat::JsonSchema { schema, .. } = format {
                generation.insert("responseJsonSchema".to_owned(), schema.clone());
            }
        }
        if !generation.is_empty() {
            body.insert("generationConfig".to_owned(), Value::Object(generation));
        }
        if !request.tools().is_empty() {
            body.insert(
                "tools".to_owned(),
                json!([{ "functionDeclarations": request.tools().iter().map(|tool| json!({
                    "name": tool.name,
                    "description": tool.description,
                    "parametersJsonSchema": tool.input_schema,
                })).collect::<Vec<_>>() }]),
            );
        }
        let operation = if stream {
            "streamGenerateContent?alt=sse"
        } else {
            "generateContent"
        };
        Ok(EncodedRequest {
            method:  Method::POST,
            url:     endpoint(
                call.route().provider().base_url(),
                &format!(
                    "/v1beta/models/{}:{operation}",
                    call.route().model().api_model()
                ),
            ),
            headers: Vec::new(),
            body:    Value::Object(body),
            timeout: request.timeout(),
        })
    }

    fn decode_response(&self, route: &ResolvedRoute, value: Value) -> Result<Response, Error> {
        let candidate = value.pointer("/candidates/0").unwrap_or(&Value::Null);
        let mut content = Vec::new();
        for (index, part) in candidate
            .pointer("/content/parts")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .enumerate()
        {
            if let Some(text) = part.get("text").and_then(Value::as_str) {
                if part.get("thought").and_then(Value::as_bool) == Some(true) {
                    content.push(ContentPart::Reasoning(ReasoningContent {
                        text:      text.to_owned(),
                        signature: part
                            .get("thoughtSignature")
                            .and_then(Value::as_str)
                            .map(ToOwned::to_owned),
                    }));
                } else {
                    content.push(ContentPart::Text {
                        text: text.to_owned(),
                    });
                }
            } else if let Some(call) = part.get("functionCall") {
                content.push(ContentPart::ToolCall(ToolCall {
                    id:        call
                        .get("id")
                        .and_then(Value::as_str)
                        .map_or_else(|| format!("function-{index}"), ToOwned::to_owned),
                    name:      call
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    arguments: call.get("args").cloned().unwrap_or(Value::Null),
                }));
            }
        }
        let usage_value = value.get("usageMetadata").unwrap_or(&Value::Null);
        let mut token_counts = usage(
            usage_value.get("promptTokenCount").and_then(Value::as_u64),
            usage_value
                .get("candidatesTokenCount")
                .and_then(Value::as_u64),
        );
        token_counts.cached_input = usage_value
            .get("cachedContentTokenCount")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        token_counts.reasoning_output = usage_value
            .get("thoughtsTokenCount")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        Ok(Response {
            id: value
                .get("responseId")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            model: route.handle(),
            content,
            finish_reason: finish_reason(candidate.get("finishReason").and_then(Value::as_str)),
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
                "Gemini returned an invalid stream event",
            )
            .with_provider(route.provider().id().clone())
            .with_source(source)
        })?;
        let response = self.decode_response(route, value)?;
        let mut events = response
            .content
            .iter()
            .filter_map(|part| match part {
                ContentPart::Text { text } => Some(StreamEvent::TextDelta { text: text.clone() }),
                ContentPart::Reasoning(reasoning) => Some(StreamEvent::ReasoningDelta {
                    text: reasoning.text.clone(),
                }),
                ContentPart::ToolCall(call) => Some(StreamEvent::ToolCall { call: call.clone() }),
                _ => None,
            })
            .collect::<Vec<_>>();
        if response.usage != TokenCounts::default() {
            events.push(StreamEvent::Usage {
                usage: response.usage,
            });
        }
        if !matches!(response.finish_reason, FinishReason::Stop) || !events.is_empty() {
            events.push(StreamEvent::Finished {
                reason: response.finish_reason,
            });
        }
        Ok(events)
    }
}

fn content(part: &ContentPart) -> Value {
    match part {
        ContentPart::Text { text } => json!({ "text": text }),
        ContentPart::Image(image) => image_content(image),
        ContentPart::Audio(audio) => json!({
            "inlineData": { "mimeType": audio.media_type, "data": audio.data }
        }),
        ContentPart::Document(document) => json!({
            "inlineData": { "mimeType": document.media_type, "data": document.data }
        }),
        ContentPart::ToolCall(call) => json!({
            "functionCall": { "id": call.id, "name": call.name, "args": call.arguments }
        }),
        ContentPart::ToolResult(result) => json!({
            "functionResponse": {
                "id": result.tool_call_id,
                "name": result.name.as_deref().unwrap_or(&result.tool_call_id),
                "response": { "output": plain_text(&result.content), "is_error": result.is_error }
            }
        }),
        ContentPart::Reasoning(reasoning) => json!({
            "text": reasoning.text,
            "thought": true,
            "thoughtSignature": reasoning.signature,
        }),
    }
}

fn image_content(image: &ImageContent) -> Value {
    if let Some((header, data)) = image.source.split_once(',') {
        if let Some(media_type) = header
            .strip_prefix("data:")
            .and_then(|header| header.strip_suffix(";base64"))
        {
            return json!({ "inlineData": { "mimeType": media_type, "data": data } });
        }
    }
    if let Some(media_type) = &image.media_type {
        if !image.source.contains("://") {
            return json!({
                "inlineData": { "mimeType": media_type, "data": image.source }
            });
        }
    }
    json!({
        "fileData": { "mimeType": image.media_type, "fileUri": image.source }
    })
}

#[cfg(all(test, feature = "builtin-catalog"))]
mod tests {
    use std::error::Error as StdError;

    use serde_json::json;

    use super::{Codec, GeminiGenerateCodec};
    use crate::codecs::test_support::resolved;
    use crate::types::{ContentPart, ImageContent, Message, Request, Role, ToolResult};

    #[test]
    fn encodes_inline_images_and_function_response_identity() -> Result<(), Box<dyn StdError>> {
        let call = resolved(
            Request::builder()
                .model("gemini/gemini-2.5-pro")
                .message(Message::new(Role::User, [ContentPart::Image(
                    ImageContent {
                        source:     "data:image/png;base64,aGVsbG8=".to_owned(),
                        media_type: None,
                        detail:     None,
                    },
                )]))
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
        let codec = GeminiGenerateCodec;

        let encoded = codec.encode(&call, false)?;

        assert_eq!(
            encoded.body["contents"][0]["parts"][0]["inlineData"],
            json!({ "mimeType": "image/png", "data": "aGVsbG8=" })
        );
        assert_eq!(
            encoded.body["contents"][1]["parts"][0]["functionResponse"]["id"],
            "call-1"
        );
        assert_eq!(
            encoded.body["contents"][1]["parts"][0]["functionResponse"]["name"],
            "weather"
        );
        Ok(())
    }
}
