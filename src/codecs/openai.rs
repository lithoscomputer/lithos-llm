//! The OpenAI Responses protocol, `POST /v1/responses`.
//!
//! This is the only dialect that carries custom tools, that replays
//! provider-native reasoning items verbatim, and whose stream terminates with a
//! complete response document. It is also the only one with a native input
//! token count endpoint that projects the generation body down to an allowlist.

use std::collections::{BTreeMap, BTreeSet};

use reqwest::Method;
use serde_json::{Map, Value, json};

use super::assembler::StreamAssembler;
use super::common::{
    endpoint, flattens_tool_result_content, merge_options, parse_arguments, plain_text,
    reject_unencodable, sampling, wire_options,
};
use super::{Codec, StreamDecoder};
use crate::adapter::ResolvedCall;
use crate::resolver::ResolvedRoute;
use crate::transport::{EncodedRequest, SseEvent, provider_error};
use crate::types::{
    ContentBlockId, ContentBlockKind, ContentPart, Error, ErrorKind, FinishReason, MediaSource,
    Message, ReasoningContent, Request, Response, ResponseFormat, Role, Speed, StreamEvent,
    TokenCounts, ToolCall, ToolCallKind, ToolChoice, ToolDefinition, ToolDefinitionKind,
    ToolResult,
};

/// The provider namespace this codec owns.
///
/// It claims [`ContentPart::Opaque`] kinds and
/// [`ToolCall::provider_metadata`] entries under this namespace and ignores
/// every other namespace, so a conversation carrying another provider's replay
/// data can still be sent here.
const NAMESPACE: &str = "openai";

/// The opaque kind holding one whole `reasoning` output item.
const REASONING_KIND: &str = "openai.reasoning";

/// The fields `POST /v1/responses/input_tokens` accepts.
///
/// The projection runs after raw provider options are merged, so a stray
/// merged key is stripped as well.
const COUNT_TOKENS_FIELDS: &[&str] = &[
    "conversation",
    "input",
    "instructions",
    "model",
    "parallel_tool_calls",
    "previous_response_id",
    "reasoning",
    "text",
    "tool_choice",
    "tools",
    "truncation",
];

/// Encodes and decodes the OpenAI Responses protocol.
///
/// The Codex deployment speaks the same protocol with a smaller field set: it
/// rejects the sampling controls and takes the system prompt as `instructions`
/// rather than as input items.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct OpenAiResponsesCodec {
    codex: bool,
}

impl Codec for OpenAiResponsesCodec {
    fn encode(&self, call: &ResolvedCall, stream: bool) -> Result<EncodedRequest, Error> {
        let request = call.request();
        // This protocol carries no audio input. Refusing here is deliberate:
        // substituting a text placeholder would put words the caller never
        // wrote into the prompt, which is worse than a clear failure.
        reject_unencodable(call.route(), request, |part| {
            matches!(part, ContentPart::Audio(_)).then_some("audio content")
        })?;
        let (options, _controls) = wire_options(call);
        let mut body = self.generation_body(call, stream);
        merge_options(&mut body, options);

        let mut encoded = EncodedRequest::new(
            Method::POST,
            endpoint(call.route().provider().base_url(), "/v1/responses"),
            Value::Object(body),
        )
        .with_timeout(request.timeout());

        // Reasoning text has no input item in this protocol; only an
        // `openai.reasoning` opaque part replays. Dropping it is correct, but
        // it must not be silent.
        if request
            .messages()
            .iter()
            .flat_map(Message::content)
            .any(|part| matches!(part, ContentPart::Reasoning(_)))
        {
            encoded = encoded.unsupported_control("replaying reasoning text");
        }

        if flattens_tool_result_content(request, |part| matches!(part, ContentPart::Text { .. })) {
            encoded = encoded.unsupported_control("non-text tool result content");
        }

        if self.codex {
            for (present, control) in [
                (request.temperature().is_some(), "temperature in Codex mode"),
                (request.top_p().is_some(), "top_p in Codex mode"),
                (
                    request.max_output_tokens().is_some(),
                    "max_output_tokens in Codex mode",
                ),
            ] {
                if present {
                    encoded = encoded.unsupported_control(control);
                }
            }
        }

        Ok(encoded)
    }

    fn decode_response(&self, route: &ResolvedRoute, value: Value) -> Result<Response, Error> {
        decode_document(route, value)
    }

    fn stream_decoder(&self, route: &ResolvedRoute) -> Box<dyn StreamDecoder> {
        Box::new(ResponsesStream {
            assembler: StreamAssembler::new(route),
            route:     route.clone(),
            delivered: BTreeSet::new(),
        })
    }

    fn encode_count_tokens(&self, call: &ResolvedCall) -> Option<Result<EncodedRequest, Error>> {
        Some(self.count_tokens_request(call))
    }

    fn decode_count_tokens(&self, route: &ResolvedRoute, value: Value) -> Result<u64, Error> {
        let object = value.get("object").and_then(Value::as_str);
        if object != Some("response.input_tokens") {
            return Err(decode_error(
                route,
                format!(
                    "returned an input token count with object {}",
                    object.unwrap_or("<missing>")
                ),
                value,
            ));
        }

        match value.get("input_tokens").and_then(Value::as_u64) {
            Some(tokens) => Ok(tokens),
            None => Err(decode_error(
                route,
                "returned an input token count without an `input_tokens` number",
                value,
            )),
        }
    }
}

impl OpenAiResponsesCodec {
    /// Creates the codec for one provider.
    ///
    /// `codex` selects the Codex deployment's field set.
    pub(crate) fn new(codex: bool) -> Self {
        Self { codex }
    }

    /// Builds the token count request from the full non-streaming body.
    ///
    /// The body is generated and merged exactly as a generation request is, and
    /// only then projected onto [`COUNT_TOKENS_FIELDS`].
    fn count_tokens_request(self, call: &ResolvedCall) -> Result<EncodedRequest, Error> {
        let mut request = self.encode(call, false)?;
        request.url = endpoint(
            call.route().provider().base_url(),
            "/v1/responses/input_tokens",
        );
        if let Value::Object(body) = &mut request.body {
            body.retain(|key, _| COUNT_TOKENS_FIELDS.contains(&key.as_str()));
        }
        Ok(request)
    }

    /// Builds the `/v1/responses` body from typed request fields only.
    ///
    /// Raw provider options are merged over this by the caller, so every field
    /// here is overridable.
    fn generation_body(self, call: &ResolvedCall, stream: bool) -> Map<String, Value> {
        let request = call.request();
        let custom = CustomTools::new(request);
        let mut body = Map::new();

        body.insert(
            "model".to_owned(),
            Value::String(call.route().api_model().to_owned()),
        );
        // Codex takes the system prompt as `instructions` and rejects system
        // input items, so those messages are hoisted out of the input.
        if self.codex {
            body.insert(
                "instructions".to_owned(),
                Value::String(instructions(request)),
            );
        }
        body.insert(
            "input".to_owned(),
            Value::Array(
                request
                    .messages()
                    .iter()
                    .filter(|message| !self.codex || !is_system(message))
                    .flat_map(|message| input_items(message, &custom))
                    .collect(),
            ),
        );
        body.insert("stream".to_owned(), Value::Bool(stream));
        // This client keeps no server-side conversation state. Encrypted
        // reasoning is asked for instead, so a reasoning item can be replayed
        // from the transcript on the next turn.
        body.insert("store".to_owned(), Value::Bool(false));
        if call.route().model().capabilities().reasoning {
            body.insert("include".to_owned(), json!(["reasoning.encrypted_content"]));
        }

        if !self.codex {
            if let Some(tokens) = request.max_output_tokens() {
                body.insert("max_output_tokens".to_owned(), tokens.into());
            }
            if let Some(temperature) = request.temperature() {
                body.insert("temperature".to_owned(), sampling(temperature));
            }
            if let Some(top_p) = request.top_p() {
                body.insert("top_p".to_owned(), sampling(top_p));
            }
        }

        body.extend(shared_body(request));
        body
    }
}

/// Decodes one complete response document.
///
/// The typed view is read from the same value that is then moved into
/// [`Response::raw`], so the response always carries the provider's own body.
fn decode_document(route: &ResolvedRoute, value: Value) -> Result<Response, Error> {
    if !value.is_object() {
        return Err(decode_error(
            route,
            "returned a response body that is not a JSON object",
            value,
        ));
    }

    let content = decode_output(&value);
    let mut response = Response::new(
        route.provider().id().clone(),
        route.model().id().clone(),
        content,
    );
    response.id = value
        .get("id")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    response.finish_reason = decode_finish_reason(&value, &response.content);
    response.usage = decode_usage(value.get("usage"));
    // This protocol reports no in-band cost; the adapter prices the call from
    // the catalog instead.
    response.cost = None;
    response.raw = Some(value);
    Ok(response)
}

/// The fields the Codex deployment and the public API encode identically.
fn shared_body(request: &Request) -> Map<String, Value> {
    let mut body = Map::new();

    // The order the request carries is the order the provider receives. The
    // count endpoint strips this field again, which is why its allowlist has
    // to name every generation field explicitly.
    if !request.stop_sequences().is_empty() {
        body.insert(
            "stop".to_owned(),
            Value::Array(
                request
                    .stop_sequences()
                    .iter()
                    .map(|sequence| Value::String(sequence.clone()))
                    .collect(),
            ),
        );
    }
    if let Some(effort) = request.reasoning_effort() {
        let effort = serde_json::to_value(effort).unwrap_or(Value::Null);
        body.insert("reasoning".to_owned(), json!({ "effort": effort }));
    }
    if let Some(speed) = request.speed() {
        let tier = match speed {
            Speed::Fast => "fast",
            Speed::Balanced => "auto",
            Speed::Economical => "flex",
        };
        body.insert("service_tier".to_owned(), Value::String(tier.to_owned()));
    }
    if !request.tools().is_empty() {
        body.insert(
            "tools".to_owned(),
            Value::Array(request.tools().iter().map(tool_definition).collect()),
        );
    }
    if let Some(choice) = request.tool_choice() {
        body.insert("tool_choice".to_owned(), tool_choice(choice));
    }
    if let Some(format) = request.response_format() {
        body.insert(
            "text".to_owned(),
            json!({ "format": response_format(format) }),
        );
    }
    if !request.metadata().is_empty() {
        let metadata = request
            .metadata()
            .iter()
            .map(|(key, value)| (key.clone(), Value::String(value.clone())))
            .collect::<Map<String, Value>>();
        body.insert("metadata".to_owned(), Value::Object(metadata));
    }

    body
}

/// The joined text of every system and developer message.
fn instructions(request: &Request) -> String {
    request
        .messages()
        .iter()
        .filter(|message| is_system(message))
        .map(|message| plain_text(message.content()))
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Whether a message carries system or developer instructions.
fn is_system(message: &Message) -> bool {
    matches!(message.role(), Role::System | Role::Developer)
}

/// The tool kinds one request declared, used to route tool results.
///
/// A `custom_tool_call_output` is a different wire item from a
/// `function_call_output`, and a result carries no kind of its own. The kind is
/// recovered from the call it answers, from the tool it names, or from the
/// message label the caller set.
struct CustomTools {
    /// Names declared as custom tools by this request.
    names: BTreeSet<String>,
    /// Call ids of custom tool calls earlier in this request.
    calls: BTreeSet<String>,
}

impl CustomTools {
    fn new(request: &Request) -> Self {
        let names = request
            .tools()
            .iter()
            .filter(|tool| tool.is_custom())
            .map(|tool| tool.name.clone())
            .collect();
        let calls = request
            .messages()
            .iter()
            .flat_map(Message::content)
            .filter_map(|part| match part {
                ContentPart::ToolCall(call) if call.kind == ToolCallKind::Custom => {
                    Some(call.id.clone())
                }
                _ => None,
            })
            .collect();

        Self { names, calls }
    }

    /// Whether a tool result answers a custom tool call.
    fn answers_custom(&self, result: &ToolResult, message: &Message) -> bool {
        if self.calls.contains(&result.tool_call_id) {
            return true;
        }
        let named = result.name.as_deref().or_else(|| message.name());
        named.is_some_and(|name| self.names.contains(name))
    }
}

/// Encodes one message as the input items it produces.
///
/// Opaque replay items come first so a reasoning item keeps the following item
/// the protocol requires it to have, then the message itself, then the tool
/// calls and tool results the message carried.
fn input_items(message: &Message, custom: &CustomTools) -> Vec<Value> {
    let mut items: Vec<Value> = message
        .content()
        .iter()
        .filter_map(|part| match part {
            ContentPart::Opaque { data, .. } if part.opaque_namespace() == Some(NAMESPACE) => {
                Some(data.clone())
            }
            _ => None,
        })
        .collect();

    let has_result = message
        .content()
        .iter()
        .any(|part| matches!(part, ContentPart::ToolResult(_)));
    if message.role() == Role::Tool && !has_result {
        // A tool message whose content is only text still answers a call. The
        // message-level label is the only place that id survives.
        if let Some(call_id) = message.tool_call_id() {
            let custom = message
                .name()
                .is_some_and(|name| custom.names.contains(name));
            items.push(tool_output_item(
                call_id,
                &plain_text(message.content()),
                false,
                custom,
            ));
            return items;
        }
    }

    let content = message_content(message);
    if !content.is_empty() {
        items.push(json!({ "role": role(message.role()), "content": content }));
    }

    for part in message.content() {
        match part {
            ContentPart::ToolCall(call) => items.push(tool_call_item(call)),
            ContentPart::ToolResult(result) => items.push(tool_output_item(
                &result.tool_call_id,
                &result_output(result),
                result.is_error,
                custom.answers_custom(result, message),
            )),
            // Every other part belongs inside the message item above.
            ContentPart::Text { .. }
            | ContentPart::Json { .. }
            | ContentPart::Image(_)
            | ContentPart::Audio(_)
            | ContentPart::Document(_)
            | ContentPart::Reasoning(_)
            | ContentPart::Opaque { .. } => {}
        }
    }

    items
}

/// Encodes the content parts that belong inside a message item.
fn message_content(message: &Message) -> Vec<Value> {
    let text_type = match message.role() {
        Role::Assistant => "output_text",
        Role::System | Role::Developer | Role::User | Role::Tool => "input_text",
    };

    message
        .content()
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text { text } => Some(json!({ "type": text_type, "text": text })),
            ContentPart::Json { value } => Some(json!({
                "type": text_type,
                "text": serde_json::to_string(value).unwrap_or_default(),
            })),
            ContentPart::Image(image) => {
                let mut item = json!({
                    "type": "input_image",
                    "image_url": media_url(&image.source),
                });
                if let (Some(object), Some(detail)) = (item.as_object_mut(), image.detail.as_ref())
                {
                    object.insert("detail".to_owned(), Value::String(detail.clone()));
                }
                Some(item)
            }
            // Documents ride inline, either as a base64 data URL or as a
            // fetchable URL.
            ContentPart::Document(document) => {
                let mut item = json!({ "type": "input_file" });
                if let Some(object) = item.as_object_mut() {
                    match &document.source {
                        MediaSource::Url { url } => {
                            object.insert("file_url".to_owned(), Value::String(url.clone()));
                        }
                        MediaSource::Base64 { .. } => {
                            object.insert(
                                "file_data".to_owned(),
                                Value::String(media_url(&document.source)),
                            );
                        }
                    }
                    if let Some(name) = document.name.as_ref() {
                        object.insert("filename".to_owned(), Value::String(name.clone()));
                    }
                }
                Some(item)
            }
            // Audio never reaches here: `encode` rejects it before dispatch
            // rather than substituting a placeholder, because injecting
            // invented text into a caller's prompt is worse than refusing.
            // Reasoning replays through its opaque item, and calls and results
            // are separate input items.
            ContentPart::Audio(_)
            | ContentPart::Reasoning(_)
            | ContentPart::ToolCall(_)
            | ContentPart::ToolResult(_)
            | ContentPart::Opaque { .. } => None,
        })
        .collect()
}

/// The wire role for one message.
fn role(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::Developer => "developer",
        Role::Assistant => "assistant",
        Role::User | Role::Tool => "user",
    }
}

/// Encodes one tool call as its input item.
///
/// The item id is replayed from this codec's own provider metadata namespace
/// when the decoder kept one, because a replayed item without it cannot anchor
/// the reasoning chain.
fn tool_call_item(call: &ToolCall) -> Value {
    let arguments = call
        .raw_arguments
        .clone()
        .unwrap_or_else(|| match &call.arguments {
            Value::String(input) => input.clone(),
            other => serde_json::to_string(other).unwrap_or_default(),
        });

    let mut item = match call.kind {
        ToolCallKind::Function => json!({
            "type": "function_call",
            "call_id": call.id,
            "name": call.name,
            "arguments": arguments,
        }),
        ToolCallKind::Custom => json!({
            "type": "custom_tool_call",
            "call_id": call.id,
            "name": call.name,
            "input": arguments,
        }),
    };

    let item_id = call
        .provider_metadata
        .get(NAMESPACE)
        .and_then(|metadata| metadata.get("item_id"))
        .and_then(Value::as_str);
    if let (Some(object), Some(item_id)) = (item.as_object_mut(), item_id) {
        object.insert("id".to_owned(), Value::String(item_id.to_owned()));
    }

    item
}

/// Encodes one tool result as its input item.
///
/// `custom_tool_call_output` has no error channel, so an error is reported
/// through the item status only for function outputs.
fn tool_output_item(call_id: &str, output: &str, is_error: bool, custom: bool) -> Value {
    if custom {
        return json!({
            "type": "custom_tool_call_output",
            "call_id": call_id,
            "output": output,
        });
    }

    let mut item = json!({
        "type": "function_call_output",
        "call_id": call_id,
        "output": output,
    });
    if let (Some(object), true) = (item.as_object_mut(), is_error) {
        object.insert("status".to_owned(), Value::String("incomplete".to_owned()));
    }
    item
}

/// The text a tool result sends back, falling back to its serialized parts.
fn result_output(result: &ToolResult) -> String {
    let text = plain_text(&result.content);
    if text.is_empty() {
        return serde_json::to_string(&result.content).unwrap_or_default();
    }
    text
}

/// Encodes one image or media source as the URL this protocol accepts.
fn media_url(source: &MediaSource) -> String {
    match source {
        MediaSource::Url { url } => url.clone(),
        MediaSource::Base64 { data, media_type } => format!("data:{media_type};base64,{data}"),
    }
}

/// Encodes one tool definition in the flat Responses shape.
///
/// This is the only codec that encodes a custom tool. A custom tool sends its
/// `format` and no `parameters`.
fn tool_definition(tool: &ToolDefinition) -> Value {
    match &tool.kind {
        ToolDefinitionKind::Function { input_schema } => json!({
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": input_schema,
        }),
        ToolDefinitionKind::Custom { format } => json!({
            "type": "custom",
            "name": tool.name,
            "description": tool.description,
            "format": format,
        }),
    }
}

/// Encodes the tool selection mode.
fn tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::None => json!("none"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Tool { name } => json!({ "type": "function", "name": name }),
    }
}

/// Encodes the requested response shape as a `text.format` value.
fn response_format(format: &ResponseFormat) -> Value {
    match format {
        ResponseFormat::Text => json!({ "type": "text" }),
        ResponseFormat::JsonObject => json!({ "type": "json_object" }),
        ResponseFormat::JsonSchema { name, schema } => json!({
            "type": "json_schema",
            "name": name,
            "schema": schema,
            "strict": true,
        }),
    }
}

/// Decodes the `output` array of a complete response document.
fn decode_output(value: &Value) -> Vec<ContentPart> {
    let mut content = Vec::new();
    let items = value
        .get("output")
        .and_then(Value::as_array)
        .into_iter()
        .flatten();

    for item in items {
        match item.get("type").and_then(Value::as_str) {
            Some("message") => {
                let text = message_text(item);
                if !text.is_empty() {
                    content.push(ContentPart::Text { text });
                }
            }
            Some("function_call") => content.push(ContentPart::ToolCall(decode_tool_call(
                item,
                ToolCallKind::Function,
            ))),
            Some("custom_tool_call") => content.push(ContentPart::ToolCall(decode_tool_call(
                item,
                ToolCallKind::Custom,
            ))),
            Some("reasoning") => content.extend(decode_reasoning(item)),
            // Provider-side items such as `web_search_call` carry no portable
            // content. They stay available in `Response::raw`.
            _ => {}
        }
    }

    content
}

/// The visible text of one `message` output item.
fn message_text(item: &Value) -> String {
    item.get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|part| part.get("type").and_then(Value::as_str) == Some("output_text"))
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect()
}

/// Decodes one `function_call` or `custom_tool_call` output item.
///
/// `call_id` is the identity a tool result answers, so it is the canonical id.
/// The item id is a second identifier the protocol needs when the call is
/// replayed, so it is kept in this codec's provider metadata namespace when it
/// differs.
fn decode_tool_call(item: &Value, kind: ToolCallKind) -> ToolCall {
    let call_id = item.get("call_id").and_then(Value::as_str);
    let item_id = item.get("id").and_then(Value::as_str);
    let raw = match kind {
        ToolCallKind::Function => item.get("arguments"),
        ToolCallKind::Custom => item.get("input"),
    }
    .and_then(Value::as_str)
    .unwrap_or_default();

    let mut call = ToolCall {
        id: call_id.or(item_id).unwrap_or_default().to_owned(),
        name: item
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        arguments: match kind {
            ToolCallKind::Function => parse_arguments(raw),
            ToolCallKind::Custom => Value::String(raw.to_owned()),
        },
        kind,
        // An empty string means the provider sent no arguments, which the
        // streamed form of the same call also reports as absent.
        raw_arguments: (!raw.is_empty()).then(|| raw.to_owned()),
        provider_metadata: BTreeMap::new(),
    };
    if let (Some(call_id), Some(item_id)) = (call_id, item_id) {
        if call_id != item_id {
            call.provider_metadata
                .insert(NAMESPACE.to_owned(), json!({ "item_id": item_id }));
        }
    }

    call
}

/// Decodes one `reasoning` output item into the parts it contributes.
///
/// The visible summary becomes reasoning content. The whole item is kept
/// verbatim whenever it carries encrypted or otherwise opaque state, because
/// only the original item can be replayed on the next turn.
fn decode_reasoning(item: &Value) -> Vec<ContentPart> {
    let text = reasoning_text(item);
    let mut parts = Vec::new();
    if !text.is_empty() {
        parts.push(ContentPart::Reasoning(ReasoningContent {
            text,
            signature: None,
            redacted: false,
        }));
    }
    if is_opaque_reasoning(item) {
        parts.push(ContentPart::opaque(REASONING_KIND, item.clone()));
    }
    parts
}

/// The visible text of one `reasoning` output item.
fn reasoning_text(item: &Value) -> String {
    ["summary", "content"]
        .iter()
        .filter_map(|key| item.get(*key))
        .filter_map(Value::as_array)
        .flatten()
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("")
}

/// Whether a reasoning item must be preserved verbatim for replay.
fn is_opaque_reasoning(item: &Value) -> bool {
    let encrypted = item
        .get("encrypted_content")
        .is_some_and(|value| !value.is_null());
    encrypted || reasoning_text(item).is_empty()
}

/// Normalizes the response status into a finish reason.
///
/// The protocol has no separate stop reason. A completed response that
/// requested tools reports [`FinishReason::ToolCall`] so consumers can branch
/// on the same value every other codec produces.
fn decode_finish_reason(value: &Value, content: &[ContentPart]) -> FinishReason {
    match value.get("status").and_then(Value::as_str) {
        Some("incomplete") => {
            match value
                .pointer("/incomplete_details/reason")
                .and_then(Value::as_str)
            {
                Some("content_filter") => FinishReason::ContentFilter,
                _ => FinishReason::Length,
            }
        }
        Some("failed") => FinishReason::Error,
        _ if content
            .iter()
            .any(|part| matches!(part, ContentPart::ToolCall(_))) =>
        {
            FinishReason::ToolCall
        }
        _ => FinishReason::Stop,
    }
}

/// Normalizes the inclusive usage counters into disjoint buckets.
///
/// `input_tokens` includes the cached tokens and `output_tokens` includes the
/// reasoning tokens. This protocol bills no separate cache write.
fn decode_usage(usage: Option<&Value>) -> TokenCounts {
    let Some(usage) = usage else {
        return TokenCounts::default();
    };
    let count = |pointer: &str| usage.pointer(pointer).and_then(Value::as_u64).unwrap_or(0);

    TokenCounts::from_inclusive(
        count("/input_tokens"),
        count("/output_tokens"),
        count("/output_tokens_details/reasoning_tokens"),
        count("/input_tokens_details/cached_tokens"),
        0,
    )
}

/// The error a malformed success body produces.
fn decode_error(route: &ResolvedRoute, detail: impl Into<String>, raw: Value) -> Error {
    Error::new(
        ErrorKind::ResponseDecode,
        format!("provider {} {}", route.provider().id(), detail.into()),
    )
    .with_provider(route.provider().id().clone())
    .with_raw_data(raw)
}

/// Decodes one `/v1/responses` stream.
struct ResponsesStream {
    assembler: StreamAssembler,
    route:     ResolvedRoute,
    /// Blocks that already received content, so a terminal item does not
    /// duplicate what the deltas delivered.
    delivered: BTreeSet<ContentBlockId>,
}

impl StreamDecoder for ResponsesStream {
    fn decode(&mut self, event: SseEvent) -> Result<Vec<StreamEvent>, Error> {
        let data = event.data.trim();
        if data.is_empty() || data == "[DONE]" {
            return Ok(Vec::new());
        }

        let value: Value = serde_json::from_str(data).map_err(|source| {
            Error::new(
                ErrorKind::StreamDecode,
                format!(
                    "provider {} returned an invalid stream event",
                    self.route.provider().id()
                ),
            )
            .with_provider(self.route.provider().id().clone())
            .with_source(source)
        })?;

        let kind = value
            .get("type")
            .and_then(Value::as_str)
            .or(event.event.as_deref())
            .unwrap_or_default();

        match kind {
            "error" => Err(provider_error(
                self.route.provider(),
                None,
                Some(value),
                None,
            )),
            "response.failed" => Err(provider_error(
                self.route.provider(),
                None,
                value.get("response").cloned(),
                None,
            )),
            "response.created" => {
                let id = value
                    .pointer("/response/id")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                Ok(vec![self.assembler.started(id)])
            }
            "response.output_item.added" => Ok(self.start_item(&block_id(&value), item(&value))),
            "response.output_text.delta" => Ok(self.text_delta(&value)),
            "response.reasoning_text.delta" | "response.reasoning_summary_text.delta" => {
                Ok(self.reasoning_delta(&value))
            }
            "response.function_call_arguments.delta" | "response.custom_tool_call_input.delta" => {
                Ok(self.arguments_delta(&value))
            }
            "response.output_item.done" => Ok(self.end_item(&block_id(&value), item(&value))),
            "response.completed" | "response.incomplete" => self.complete(&value),
            _ => Ok(Vec::new()),
        }
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>, Error> {
        Ok(self.assembler.complete())
    }
}

impl ResponsesStream {
    /// Opens the block for one output item.
    ///
    /// Opening is idempotent, so the terminal event for an item the provider
    /// never announced still opens the right kind of block. A reasoning item is
    /// deliberately not opened here: whether it becomes visible reasoning or an
    /// opaque replay item is only known once the item is done.
    fn start_item(&mut self, id: &ContentBlockId, item: &Value) -> Vec<StreamEvent> {
        match item.get("type").and_then(Value::as_str) {
            Some("message") => self.assembler.start(id.clone(), ContentBlockKind::Text),
            Some(kind @ ("function_call" | "custom_tool_call")) => {
                let call_kind = match kind {
                    "custom_tool_call" => ToolCallKind::Custom,
                    _ => ToolCallKind::Function,
                };
                let call_id = item.get("call_id").and_then(Value::as_str);
                let item_id = item.get("id").and_then(Value::as_str);
                let mut events = self
                    .assembler
                    .start(id.clone(), ContentBlockKind::ToolCall {
                        id:   call_id.or(item_id).unwrap_or_default().to_owned(),
                        name: item
                            .get("name")
                            .and_then(Value::as_str)
                            .map(ToOwned::to_owned),
                        kind: call_kind,
                    });
                if let (Some(call_id), Some(item_id)) = (call_id, item_id) {
                    if call_id != item_id {
                        events.extend(self.assembler.provider_metadata(
                            id,
                            NAMESPACE,
                            json!({ "item_id": item_id }),
                        ));
                    }
                }
                events
            }
            _ => Vec::new(),
        }
    }

    /// Closes the block for one output item, delivering anything the deltas
    /// did not carry.
    fn end_item(&mut self, id: &ContentBlockId, item: &Value) -> Vec<StreamEvent> {
        if item.get("type").and_then(Value::as_str) == Some("reasoning") {
            return self.end_reasoning(id, item);
        }

        let mut events = self.start_item(id, item);
        if !self.delivered.contains(id) {
            events.extend(self.recover_item(id, item));
        }
        events.extend(self.assembler.end(id));
        events
    }

    /// Delivers the content of an item that streamed no deltas of its own.
    fn recover_item(&mut self, id: &ContentBlockId, item: &Value) -> Vec<StreamEvent> {
        match item.get("type").and_then(Value::as_str) {
            Some("message") => {
                let text = message_text(item);
                if text.is_empty() {
                    return Vec::new();
                }
                self.deliver(id, |assembler| assembler.text(id, &text))
            }
            Some(kind @ ("function_call" | "custom_tool_call")) => {
                let key = match kind {
                    "custom_tool_call" => "input",
                    _ => "arguments",
                };
                let arguments = item
                    .get(key)
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                if arguments.is_empty() {
                    return Vec::new();
                }
                self.deliver(id, |assembler| assembler.arguments(id, &arguments))
            }
            _ => Vec::new(),
        }
    }

    /// Closes a reasoning item, keeping the whole item when it must be
    /// replayed.
    ///
    /// The opaque block reuses the item's own block id when no reasoning text
    /// streamed, and takes a derived id when it did, so both parts of one item
    /// keep distinct stable ids.
    fn end_reasoning(&mut self, id: &ContentBlockId, item: &Value) -> Vec<StreamEvent> {
        let text = reasoning_text(item);
        let mut events = Vec::new();
        if !self.delivered.contains(id) && !text.is_empty() {
            events.extend(self.deliver(id, |assembler| assembler.reasoning(id, &text)));
        }
        let visible = self.delivered.contains(id);
        events.extend(self.assembler.end(id));

        if !is_opaque_reasoning(item) {
            return events;
        }

        let opaque = if visible {
            ContentBlockId::new(format!("{}-item", id.as_str()))
        } else {
            id.clone()
        };
        events.extend(
            self.assembler
                .start(opaque.clone(), ContentBlockKind::Opaque {
                    kind: REASONING_KIND.to_owned(),
                }),
        );
        events.extend(self.assembler.set_opaque_data(&opaque, item.clone()));
        events.extend(self.assembler.end(&opaque));
        events
    }

    fn text_delta(&mut self, value: &Value) -> Vec<StreamEvent> {
        let id = block_id(value);
        let text = delta_text(value);
        self.deliver(&id, |assembler| assembler.text(&id, &text))
    }

    fn reasoning_delta(&mut self, value: &Value) -> Vec<StreamEvent> {
        let id = block_id(value);
        let text = delta_text(value);
        self.deliver(&id, |assembler| assembler.reasoning(&id, &text))
    }

    fn arguments_delta(&mut self, value: &Value) -> Vec<StreamEvent> {
        let id = block_id(value);
        let chunk = delta_text(value);
        self.deliver(&id, |assembler| assembler.arguments(&id, &chunk))
    }

    /// Runs one assembler call and records that the block carried content.
    fn deliver(
        &mut self,
        id: &ContentBlockId,
        call: impl FnOnce(&mut StreamAssembler) -> Vec<StreamEvent>,
    ) -> Vec<StreamEvent> {
        let events = call(&mut self.assembler);
        self.delivered.insert(id.clone());
        events
    }

    /// Completes the stream from the provider's own final response document.
    ///
    /// This is the only protocol that sends one, so the finished response keeps
    /// the provider's id, usage, finish reason, and complete raw body rather
    /// than a document assembled from the event log.
    fn complete(&mut self, value: &Value) -> Result<Vec<StreamEvent>, Error> {
        let Some(document) = value.get("response") else {
            return Ok(self.assembler.complete());
        };

        let response = decode_document(&self.route, document.clone())?;
        if let Some(id) = response.id {
            self.assembler.set_id(id);
        }
        self.assembler.set_finish_reason(response.finish_reason);
        self.assembler.set_raw(document.clone());

        let mut events = vec![self.assembler.usage(response.usage)];
        events.extend(self.assembler.complete());
        Ok(events)
    }
}

/// The output item an item event carries.
fn item(value: &Value) -> &Value {
    value.get("item").unwrap_or(&Value::Null)
}

/// The text fragment a delta event carries.
fn delta_text(value: &Value) -> String {
    value
        .get("delta")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

/// The stable block id for one stream event.
///
/// Delta events name their item directly. Item events carry the item instead,
/// whose own id is the same value. Only an event with neither falls back to the
/// output index.
fn block_id(value: &Value) -> ContentBlockId {
    let named = value
        .get("item_id")
        .and_then(Value::as_str)
        .or_else(|| value.pointer("/item/id").and_then(Value::as_str));
    match named {
        Some(id) => ContentBlockId::new(id),
        None => ContentBlockId::new(format!(
            "block-{}",
            value
                .get("output_index")
                .and_then(Value::as_u64)
                .unwrap_or(0)
        )),
    }
}

#[cfg(all(test, feature = "builtin-catalog"))]
mod tests {
    use std::error::Error as StdError;

    use serde_json::{Value, json};

    use super::{COUNT_TOKENS_FIELDS, Codec, OpenAiResponsesCodec};
    use crate::adapter::ResolvedCall;
    use crate::codecs::test_support::resolved;
    use crate::transport::SseEvent;
    use crate::types::{
        ContentBlockId, ContentPart, ErrorKind, Message, ReasoningContent, Request, Response, Role,
        StreamEvent, ToolCall, ToolCallKind, ToolDefinition, ToolResult,
    };

    const MODEL: &str = "openai/gpt-5.6-luna";

    /// The public deployment's codec.
    fn codec() -> OpenAiResponsesCodec {
        OpenAiResponsesCodec::new(false)
    }

    fn call(request: Request) -> Result<ResolvedCall, Box<dyn StdError>> {
        resolved(request)
    }

    fn sse(data: &Value) -> SseEvent {
        SseEvent {
            event: None,
            data:  data.to_string(),
        }
    }

    /// Every content part carried by a block-end event, in order.
    fn ended_parts(events: &[StreamEvent]) -> Vec<ContentPart> {
        events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::ContentBlockEnd { part, .. } => Some(part.clone()),
                _ => None,
            })
            .collect()
    }

    /// The single completed response of a stream.
    fn completed(events: &[StreamEvent]) -> Result<Response, Box<dyn StdError>> {
        let mut responses = events.iter().filter_map(|event| match event {
            StreamEvent::Completed { response } => Some(response.clone()),
            _ => None,
        });
        let response = responses
            .next()
            .ok_or("the stream emitted no completed event")?;
        if responses.next().is_some() {
            return Err("the stream emitted more than one completed event".into());
        }
        Ok(response)
    }

    /// Checks one start, then deltas, then one end for every block.
    fn assert_block_boundaries(events: &[StreamEvent]) -> Result<(), Box<dyn StdError>> {
        let mut open: Vec<&ContentBlockId> = Vec::new();
        let mut closed: Vec<&ContentBlockId> = Vec::new();

        for event in events {
            match event {
                StreamEvent::ContentBlockStart { id, .. } => {
                    if open.contains(&id) || closed.contains(&id) {
                        return Err(format!("{id:?} started twice").into());
                    }
                    open.push(id);
                }
                StreamEvent::TextDelta { id, .. }
                | StreamEvent::ReasoningDelta { id, .. }
                | StreamEvent::ToolCallDelta { id, .. } => {
                    if !open.contains(&id) {
                        return Err(format!("{id:?} sent a delta before its start").into());
                    }
                }
                StreamEvent::ContentBlockEnd { id, .. } => {
                    if !open.contains(&id) {
                        return Err(format!("{id:?} ended without a start").into());
                    }
                    open.retain(|open_id| *open_id != id);
                    closed.push(id);
                }
                StreamEvent::Started { .. }
                | StreamEvent::Usage { .. }
                | StreamEvent::RateLimits { .. }
                | StreamEvent::Completed { .. } => {}
            }
        }

        if !open.is_empty() {
            return Err(format!("blocks left open: {open:?}").into());
        }
        Ok(())
    }

    #[test]
    fn tool_calls_and_results_keep_their_protocol_identity() -> Result<(), Box<dyn StdError>> {
        let mut call_part = ToolCall::function("call_abc", "search", json!({ "query": "rust" }));
        call_part.raw_arguments = Some("{\"query\":\"rust\"}".to_owned());
        call_part
            .provider_metadata
            .insert("openai".to_owned(), json!({ "item_id": "fc_123" }));
        let request = Request::builder()
            .model(MODEL)
            .user("Search for rust")
            .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
                call_part,
            )]))
            .message(Message::new(Role::Tool, [ContentPart::ToolResult(
                ToolResult {
                    tool_call_id: "call_abc".to_owned(),
                    name:         Some("search".to_owned()),
                    content:      vec![ContentPart::Text {
                        text: "2 matches".to_owned(),
                    }],
                    is_error:     false,
                },
            )]))
            .build()?;
        let codec = codec();

        let encoded = codec.encode(&call(request)?, false)?;

        assert_eq!(
            encoded.body["input"][1],
            json!({
                "type": "function_call",
                "id": "fc_123",
                "call_id": "call_abc",
                "name": "search",
                "arguments": "{\"query\":\"rust\"}",
            })
        );
        assert_eq!(
            encoded.body["input"][2],
            json!({
                "type": "function_call_output",
                "call_id": "call_abc",
                "output": "2 matches",
            })
        );

        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();
        let response = codec.decode_response(
            &route,
            json!({
                "id": "resp_1",
                "status": "completed",
                "output": [{
                    "type": "function_call",
                    "id": "fc_123",
                    "call_id": "call_abc",
                    "name": "search",
                    "arguments": "{\"query\":\"rust\"}",
                }],
            }),
        )?;

        let [ContentPart::ToolCall(decoded)] = response.content.as_slice() else {
            return Err("expected one tool call".into());
        };
        assert_eq!(decoded.id, "call_abc");
        assert_eq!(decoded.arguments, json!({ "query": "rust" }));
        assert_eq!(
            decoded.raw_arguments.as_deref(),
            Some("{\"query\":\"rust\"}")
        );
        assert_eq!(
            decoded.provider_metadata.get("openai"),
            Some(&json!({ "item_id": "fc_123" }))
        );
        Ok(())
    }

    #[test]
    fn raw_options_win_and_foreign_namespaces_are_ignored() -> Result<(), Box<dyn StdError>> {
        let request = Request::builder()
            .model(MODEL)
            .user("Hello")
            .temperature(0.2)
            .provider_option("openai", "temperature", json!(0.9))
            .provider_option("openai", "auto_cache", json!(false))
            .provider_option("anthropic", "top_k", json!(40))
            .build()?;

        let encoded = codec().encode(&call(request)?, false)?;

        assert_eq!(encoded.body["temperature"], json!(0.9));
        let body = encoded.body.as_object().ok_or("expected a JSON object")?;
        assert!(!body.contains_key("auto_cache"));
        assert!(!body.contains_key("top_k"));
        Ok(())
    }

    #[test]
    fn stop_sequences_and_metadata_reach_the_wire() -> Result<(), Box<dyn StdError>> {
        let request = Request::builder()
            .model(MODEL)
            .user("Hello")
            .stop_sequences(["END", "STOP"])
            .metadata_entry("trace_id", "t789")
            .build()?;

        let encoded = codec().encode(&call(request)?, false)?;

        assert_eq!(encoded.body["stop"], json!(["END", "STOP"]));
        assert_eq!(encoded.body["metadata"], json!({ "trace_id": "t789" }));
        assert!(
            encoded.warnings.is_empty(),
            "this protocol expresses both controls",
        );
        Ok(())
    }

    #[test]
    fn codex_mode_hoists_instructions_and_drops_sampling_controls() -> Result<(), Box<dyn StdError>>
    {
        let request = Request::builder()
            .model(MODEL)
            .system("Be brief")
            .user("Hello")
            .temperature(0.2)
            .top_p(0.5)
            .max_output_tokens(256)
            .build()?;

        let encoded = OpenAiResponsesCodec::new(true).encode(&call(request)?, true)?;

        assert_eq!(encoded.body["instructions"], json!("Be brief"));
        assert_eq!(
            encoded.body["input"],
            json!([{
                "role": "user",
                "content": [{ "type": "input_text", "text": "Hello" }],
            }])
        );
        assert_eq!(encoded.body["stream"], json!(true));
        let body = encoded.body.as_object().ok_or("expected a JSON object")?;
        assert!(!body.contains_key("temperature"));
        assert!(!body.contains_key("top_p"));
        assert!(!body.contains_key("max_output_tokens"));
        assert_eq!(encoded.warnings.len(), 3);
        Ok(())
    }

    #[test]
    fn inclusive_usage_becomes_disjoint_buckets() -> Result<(), Box<dyn StdError>> {
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();

        let response = codec().decode_response(
            &route,
            json!({
                "id": "resp_1",
                "status": "completed",
                "output": [],
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "input_tokens_details": { "cached_tokens": 80 },
                    "output_tokens_details": { "reasoning_tokens": 20 },
                },
            }),
        )?;

        assert_eq!(response.usage.input, 20);
        assert_eq!(response.usage.output, 30);
        assert_eq!(response.usage.reasoning, 20);
        assert_eq!(response.usage.cache_read, 80);
        assert_eq!(response.usage.cache_write, 0);
        assert_eq!(response.usage.total(), 150);
        assert_eq!(response.cost, None);
        Ok(())
    }

    #[test]
    fn the_raw_success_document_is_preserved() -> Result<(), Box<dyn StdError>> {
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();
        let document = json!({
            "id": "resp_1",
            "status": "completed",
            "service_tier": "default",
            "output": [{
                "type": "message",
                "id": "msg_1",
                "content": [{ "type": "output_text", "text": "hello" }],
            }],
        });

        let response = codec().decode_response(&route, document.clone())?;

        assert_eq!(response.raw, Some(document));
        assert_eq!(response.text(), "hello");
        Ok(())
    }

    #[test]
    fn custom_tools_round_trip() -> Result<(), Box<dyn StdError>> {
        let request = Request::builder()
            .model(MODEL)
            .tool(ToolDefinition::custom(
                "apply_patch",
                "Edits files",
                json!({ "type": "grammar", "syntax": "lark" }),
            ))
            .user("Patch the file")
            .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
                ToolCall::custom("call_001", "apply_patch", "*** Begin Patch"),
            )]))
            .message(Message::new(Role::Tool, [ContentPart::ToolResult(
                ToolResult {
                    tool_call_id: "call_001".to_owned(),
                    name:         Some("apply_patch".to_owned()),
                    content:      vec![ContentPart::Text {
                        text: "Success".to_owned(),
                    }],
                    is_error:     false,
                },
            )]))
            .build()?;
        let codec = codec();
        let call = call(request)?;

        let encoded = codec.encode(&call, false)?;

        assert_eq!(
            encoded.body["tools"][0],
            json!({
                "type": "custom",
                "name": "apply_patch",
                "description": "Edits files",
                "format": { "type": "grammar", "syntax": "lark" },
            })
        );
        assert_eq!(
            encoded.body["input"][1],
            json!({
                "type": "custom_tool_call",
                "call_id": "call_001",
                "name": "apply_patch",
                "input": "*** Begin Patch",
            })
        );
        assert_eq!(
            encoded.body["input"][2],
            json!({
                "type": "custom_tool_call_output",
                "call_id": "call_001",
                "output": "Success",
            })
        );

        let response = codec.decode_response(
            call.route(),
            json!({
                "id": "resp_1",
                "status": "completed",
                "output": [{
                    "type": "custom_tool_call",
                    "id": "ctc_def456",
                    "call_id": "call_001",
                    "name": "apply_patch",
                    "input": "*** Begin Patch",
                }],
            }),
        )?;

        let [ContentPart::ToolCall(decoded)] = response.content.as_slice() else {
            return Err("expected one tool call".into());
        };
        assert_eq!(decoded.kind, ToolCallKind::Custom);
        assert_eq!(
            decoded.arguments,
            Value::String("*** Begin Patch".to_owned())
        );
        assert_eq!(decoded.raw_arguments.as_deref(), Some("*** Begin Patch"));
        Ok(())
    }

    #[test]
    fn opaque_reasoning_replays_and_other_namespaces_are_skipped() -> Result<(), Box<dyn StdError>>
    {
        let item = json!({
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
            "encrypted_content": "gAAAA",
        });
        let request = Request::builder()
            .model(MODEL)
            .user("Hello")
            .message(Message::new(Role::Assistant, [
                ContentPart::opaque("openai.reasoning", item.clone()),
                ContentPart::opaque("anthropic.thinking", json!({ "signature": "sig" })),
                ContentPart::Reasoning(ReasoningContent {
                    text:      "step one".to_owned(),
                    signature: None,
                    redacted:  false,
                }),
                ContentPart::Text {
                    text: "hello".to_owned(),
                },
            ]))
            .build()?;

        let encoded = codec().encode(&call(request)?, false)?;

        assert_eq!(encoded.body["input"][1], item);
        assert_eq!(
            encoded.body["input"][2],
            json!({
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "hello" }],
            })
        );
        assert_eq!(
            encoded.body["input"].as_array().map(Vec::len),
            Some(3),
            "an opaque part for another provider must not reach the wire",
        );
        Ok(())
    }

    #[test]
    fn a_reasoning_item_decodes_to_visible_text_and_a_replay_part() -> Result<(), Box<dyn StdError>>
    {
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();

        let response = codec().decode_response(
            &route,
            json!({
                "id": "resp_1",
                "status": "completed",
                "output": [{
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{ "type": "summary_text", "text": "checked" }],
                    "encrypted_content": "gAAAA",
                }],
            }),
        )?;

        let [ContentPart::Reasoning(reasoning), opaque] = response.content.as_slice() else {
            return Err("expected reasoning text and its replay part".into());
        };
        assert_eq!(reasoning.text, "checked");
        assert_eq!(opaque.opaque_namespace(), Some("openai"));
        Ok(())
    }

    #[test]
    fn a_stream_transcript_produces_one_block_per_item() -> Result<(), Box<dyn StdError>> {
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();
        let mut decoder = codec().stream_decoder(&route);
        let transcript = vec![
            json!({ "type": "response.created", "response": { "id": "resp_stream" } }),
            json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": { "type": "message", "id": "msg_1", "role": "assistant", "content": [] },
            }),
            json!({ "type": "response.output_text.delta", "item_id": "msg_1", "delta": "Hel" }),
            json!({ "type": "response.output_text.delta", "item_id": "msg_1", "delta": "lo" }),
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "msg_1",
                    "content": [{ "type": "output_text", "text": "Hello" }],
                },
            }),
            json!({
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {
                    "type": "function_call",
                    "id": "fc_123",
                    "call_id": "call_abc",
                    "name": "search",
                },
            }),
            json!({
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_123",
                "delta": "{\"qu",
            }),
            json!({
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_123",
                "delta": "ery\":\"rust\"}",
            }),
            json!({
                "type": "response.output_item.done",
                "output_index": 1,
                "item": {
                    "type": "function_call",
                    "id": "fc_123",
                    "call_id": "call_abc",
                    "name": "search",
                    "arguments": "{\"query\":\"rust\"}",
                },
            }),
            json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_stream",
                    "status": "completed",
                    "output": [],
                    "usage": {
                        "input_tokens": 11,
                        "output_tokens": 5,
                        "input_tokens_details": { "cached_tokens": 2 },
                        "output_tokens_details": { "reasoning_tokens": 1 },
                    },
                },
            }),
        ];

        let mut events = Vec::new();
        for event in transcript {
            events.extend(decoder.decode(sse(&event))?);
        }
        events.extend(decoder.finish()?);

        assert_block_boundaries(&events)?;
        let starts = events
            .iter()
            .filter(|event| matches!(event, StreamEvent::ContentBlockStart { .. }))
            .count();
        assert_eq!(starts, 2);
        assert!(matches!(
            events.first(),
            Some(StreamEvent::Started { id: Some(id) }) if id == "resp_stream"
        ));

        let parts = ended_parts(&events);
        assert_eq!(parts[0], ContentPart::Text {
            text: "Hello".to_owned(),
        });
        let ContentPart::ToolCall(streamed) = &parts[1] else {
            return Err("expected a streamed tool call".into());
        };
        assert_eq!(streamed.id, "call_abc");
        assert_eq!(streamed.name, "search");
        assert_eq!(streamed.arguments, json!({ "query": "rust" }));
        assert_eq!(
            streamed.provider_metadata.get("openai"),
            Some(&json!({ "item_id": "fc_123" }))
        );

        let response = completed(&events)?;
        assert_eq!(response.content, parts);
        assert_eq!(response.id.as_deref(), Some("resp_stream"));
        assert_eq!(response.usage.input, 9);
        assert_eq!(response.usage.cache_read, 2);
        assert_eq!(response.usage.reasoning, 1);
        assert!(response.raw.is_some());
        Ok(())
    }

    #[test]
    fn a_streamed_reasoning_item_keeps_its_text_and_its_replay_part()
    -> Result<(), Box<dyn StdError>> {
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();
        let mut decoder = codec().stream_decoder(&route);
        let item = json!({
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{ "type": "summary_text", "text": "checked" }],
            "encrypted_content": "gAAAA",
        });

        let mut events = decoder.decode(sse(&json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": { "type": "reasoning", "id": "rs_1", "summary": [] },
        })))?;
        events.extend(decoder.decode(sse(&json!({
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs_1",
            "delta": "checked",
        })))?);
        events.extend(decoder.decode(sse(&json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": item,
        })))?);
        events.extend(decoder.finish()?);

        assert_block_boundaries(&events)?;
        let parts = ended_parts(&events);
        let [
            ContentPart::Reasoning(reasoning),
            ContentPart::Opaque { kind, data },
        ] = parts.as_slice()
        else {
            return Err("expected reasoning text and its replay part".into());
        };
        assert_eq!(reasoning.text, "checked");
        assert_eq!(kind, "openai.reasoning");
        assert_eq!(data, &item);
        assert_eq!(completed(&events)?.content, parts);
        Ok(())
    }

    #[test]
    fn a_stream_error_ends_the_stream_without_completing() -> Result<(), Box<dyn StdError>> {
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();
        let mut decoder = codec().stream_decoder(&route);

        decoder.decode(sse(
            &json!({ "type": "response.created", "response": { "id": "resp_1" } }),
        ))?;
        let error = decoder
            .decode(sse(&json!({
                "type": "error",
                "error": { "type": "server_error", "message": "upstream failed" },
            })))
            .err()
            .ok_or("expected a stream error")?;

        assert!(error.message().contains("upstream failed"));
        Ok(())
    }

    #[test]
    fn an_item_without_deltas_still_produces_its_content() -> Result<(), Box<dyn StdError>> {
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();
        let mut decoder = codec().stream_decoder(&route);

        let mut events = decoder.decode(sse(&json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "custom_tool_call",
                "id": "ctc_1",
                "call_id": "call_001",
                "name": "apply_patch",
                "input": "*** Begin Patch",
            },
        })))?;
        events.extend(decoder.finish()?);

        assert_block_boundaries(&events)?;
        let parts = ended_parts(&events);
        let [ContentPart::ToolCall(recovered)] = parts.as_slice() else {
            return Err("expected one recovered tool call".into());
        };
        assert_eq!(recovered.kind, ToolCallKind::Custom);
        assert_eq!(
            recovered.arguments,
            Value::String("*** Begin Patch".to_owned())
        );
        Ok(())
    }

    #[test]
    fn the_count_tokens_body_keeps_only_allowlisted_fields() -> Result<(), Box<dyn StdError>> {
        let request = Request::builder()
            .model(MODEL)
            .user("Hello")
            .temperature(0.2)
            .max_output_tokens(256)
            .stop_sequence("END")
            .metadata_entry("trace_id", "t789")
            .provider_option("openai", "seed", json!(7))
            .build()?;
        let call = call(request)?;

        let encoded = codec()
            .encode_count_tokens(&call)
            .ok_or("expected a token count request")??;

        assert!(encoded.url.ends_with("/v1/responses/input_tokens"));
        let body = encoded.body.as_object().ok_or("expected a JSON object")?;
        for key in body.keys() {
            assert!(
                COUNT_TOKENS_FIELDS.contains(&key.as_str()),
                "{key} is not accepted by the count endpoint",
            );
        }
        assert!(!body.contains_key("temperature"));
        assert!(!body.contains_key("max_output_tokens"));
        assert!(!body.contains_key("stop"));
        assert!(!body.contains_key("stream"));
        assert!(!body.contains_key("metadata"));
        assert!(!body.contains_key("seed"));
        assert_eq!(body["model"], json!("gpt-5.6-luna"));
        Ok(())
    }

    #[test]
    fn a_token_count_with_the_wrong_object_is_a_decode_error() -> Result<(), Box<dyn StdError>> {
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();
        let codec = codec();

        let tokens = codec.decode_count_tokens(
            &route,
            json!({ "input_tokens": 123, "object": "response.input_tokens" }),
        )?;
        let error = codec
            .decode_count_tokens(&route, json!({ "input_tokens": 123, "object": "response" }))
            .err()
            .ok_or("expected a decode error")?;

        assert_eq!(tokens, 123);
        assert_eq!(error.kind(), ErrorKind::ResponseDecode);
        Ok(())
    }
}
