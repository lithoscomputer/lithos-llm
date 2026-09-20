//! Request encoding: input items, tools, and the shared body fields.

use std::collections::BTreeSet;

use serde_json::{Map, Value, json};

use super::{NAMESPACE, REASONING_KIND};
use crate::adapter::ResolvedCall;
use crate::codecs::content::{
    data_url, flattens_system_content, flattens_tool_result_content, plain_text, text_or_json_only,
    tool_result_text,
};
use crate::codecs::options::sampling;
use crate::types::{
    ContentPart, MediaSource, Message, Request, ResponseFormat, Role, Speed, ToolCall,
    ToolCallKind, ToolChoice, ToolDefinition, ToolDefinitionKind, ToolResult,
};

/// The part this protocol cannot carry, or `None` for one it can.
///
/// This protocol carries no audio input. Refusing is deliberate: substituting
/// a text placeholder would put words the caller never wrote into the prompt,
/// which is worse than a clear failure.
///
/// An inline document must carry a name: the live API requires `filename`
/// beside `file_data` (verified 2026-08-30, a 400 names the missing
/// parameter), and inventing one would put a name the caller never wrote in
/// front of the model. URL documents need no name — `file_url` stands alone.
pub(super) fn unencodable_part(part: &ContentPart) -> Option<&'static str> {
    match part {
        ContentPart::Audio(_) => Some("audio content"),
        ContentPart::Document(document)
            if document.name.is_none() && matches!(document.source, MediaSource::Base64 { .. }) =>
        {
            Some("an inline document without a file name")
        }
        _ => None,
    }
}

/// The portable controls this request carries that the body did not encode.
///
/// Each is reported as an `unsupported_control` warning: the request still
/// reaches the model, but not as the caller wrote it. `codex` selects the
/// Codex deployment's rules.
pub(super) fn dropped_controls(request: &Request, codex: bool) -> Vec<&'static str> {
    let mut dropped = Vec::new();
    if replays_reasoning_text_without_an_item(request) {
        dropped.push("replaying reasoning text");
    }
    // An all-JSON result travels as the bare value, so only a mix that must
    // flatten is reported.
    if flattens_tool_result_content(request, text_or_json_only) {
        dropped.push("non-text tool result content");
    }
    if flattens_instruction_content(request, codex) {
        dropped.push("non-text system content");
    }
    if codex {
        dropped.extend(codex_sampling_controls(request));
    }
    // `/v1/responses` takes no stop parameter at all: the live API answers
    // one with a 400 "Unknown parameter: 'stop'" on every model, reasoning
    // or not (probed against gpt-5.6-luna, gpt-5.4, and gpt-4o on
    // 2026-08-30). The sequences are dropped with a warning rather than
    // sent or refused; a Responses-compatible skin that does take a stop
    // member can still receive one through raw provider options.
    if !request.stop_sequences().is_empty() {
        dropped.push("stop sequences");
    }
    dropped
}

/// Whether a message carries reasoning text with no opaque item to replay it.
///
/// Reasoning text has no input item in this protocol; only an
/// `openai.reasoning` opaque part replays. Dropping it is correct, but it must
/// not be silent — unless an opaque reasoning item sits in the same message,
/// because this codec's own decoder emits the readable part BESIDE the opaque
/// item that replays the same text, and warning on the codec's own round trip
/// would claim a loss on every turn.
fn replays_reasoning_text_without_an_item(request: &Request) -> bool {
    request.messages().iter().any(|message| {
        let parts = message.content();
        parts
            .iter()
            .any(|part| matches!(part, ContentPart::Reasoning(_)))
            && !parts.iter().any(
                |part| matches!(part, ContentPart::Opaque { kind, .. } if kind == REASONING_KIND),
            )
    })
}

/// Whether a system or developer message carries content the deployment
/// drops.
///
/// Codex hoists system messages into `instructions`, which is a string, so
/// anything but text in one is dropped. Standard mode keeps them as input
/// items, but a system or developer item takes `input_text` and nothing
/// else, so media in one is dropped there too — structured JSON still
/// travels as text. Either way the text still reaches the model: a warning,
/// not a refusal.
fn flattens_instruction_content(request: &Request, codex: bool) -> bool {
    if codex {
        return flattens_system_content(request);
    }
    request
        .messages()
        .iter()
        .filter(|message| message.is_instruction())
        .flat_map(Message::content)
        .any(|part| matches!(part, ContentPart::Image(_) | ContentPart::Document(_)))
}

/// The sampling controls the Codex deployment rejects and this codec drops.
fn codex_sampling_controls(request: &Request) -> Vec<&'static str> {
    [
        (request.temperature().is_some(), "temperature in Codex mode"),
        (request.top_p().is_some(), "top_p in Codex mode"),
        (
            request.max_output_tokens().is_some(),
            "max_output_tokens in Codex mode",
        ),
    ]
    .into_iter()
    .filter_map(|(present, control)| present.then_some(control))
    .collect()
}

/// The typed fields of a `/v1/responses` body, one method per field.
///
/// Raw provider options are merged over the built body by the caller, so
/// every field here is overridable. `codex` selects the Codex deployment's
/// field set: it takes the system prompt as `instructions` rather than as
/// input items and rejects the sampling controls.
pub(super) struct ResponsesBody<'a> {
    codex:  bool,
    call:   &'a ResolvedCall,
    stream: bool,
}

impl<'a> ResponsesBody<'a> {
    pub(super) fn new(codex: bool, call: &'a ResolvedCall, stream: bool) -> Self {
        Self {
            codex,
            call,
            stream,
        }
    }

    fn request(&self) -> &'a Request {
        self.call.request()
    }

    /// Builds the body in wire order.
    pub(super) fn build(&self) -> Map<String, Value> {
        let request = self.request();
        let mut body = Map::new();
        body.insert("model".to_owned(), self.model());
        if let Some(instructions) = self.instructions() {
            body.insert("instructions".to_owned(), instructions);
        }
        body.insert("input".to_owned(), self.input());
        body.insert("stream".to_owned(), Value::Bool(self.stream));
        // This client keeps no server-side conversation state. Encrypted
        // reasoning is asked for instead, so a reasoning item can be replayed
        // from the transcript on the next turn.
        body.insert("store".to_owned(), Value::Bool(false));
        // Requested unconditionally, as the reference client did. Gating it
        // on the catalog's `reasoning` flag silently breaks multi-turn tool
        // calling for an overlay entry that omits the flag on a reasoning
        // model, and a model without reasoning ignores the include.
        body.insert("include".to_owned(), json!(["reasoning.encrypted_content"]));
        body.extend(self.sampling());
        if let Some(reasoning) = self.reasoning() {
            body.insert("reasoning".to_owned(), reasoning);
        }
        if let Some(tier) = self.service_tier() {
            body.insert("service_tier".to_owned(), tier);
        }
        if !request.tools().is_empty() {
            body.insert("tools".to_owned(), self.tools());
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
        if let Some(metadata) = self.metadata() {
            body.insert("metadata".to_owned(), metadata);
        }
        body
    }

    fn model(&self) -> Value {
        Value::String(self.call.route().api_model().to_owned())
    }

    /// Codex takes the system prompt as `instructions` and rejects system
    /// input items, so those messages are hoisted out of the input.
    fn instructions(&self) -> Option<Value> {
        self.codex
            .then(|| Value::String(instructions(self.request())))
    }

    fn input(&self) -> Value {
        let request = self.request();
        let custom = CustomTools::new(request);
        Value::Array(
            request
                .messages()
                .iter()
                .filter(|message| !self.codex || !message.is_instruction())
                .flat_map(|message| input_items(message, &custom))
                .collect(),
        )
    }

    /// The sampling controls, which the Codex deployment rejects.
    fn sampling(&self) -> Map<String, Value> {
        let request = self.request();
        let mut fields = Map::new();
        if self.codex {
            return fields;
        }
        if let Some(tokens) = request.max_output_tokens() {
            fields.insert("max_output_tokens".to_owned(), tokens.into());
        }
        if let Some(temperature) = request.temperature() {
            fields.insert("temperature".to_owned(), sampling(temperature));
        }
        if let Some(top_p) = request.top_p() {
            fields.insert("top_p".to_owned(), sampling(top_p));
        }
        fields
    }

    fn reasoning(&self) -> Option<Value> {
        let effort = self.request().reasoning_effort()?;
        let effort = serde_json::to_value(effort).unwrap_or(Value::Null);
        Some(json!({ "effort": effort }))
    }

    fn service_tier(&self) -> Option<Value> {
        let tier = match self.request().speed()? {
            Speed::Fast => "priority",
            Speed::Balanced => "auto",
            Speed::Economical => "flex",
        };
        Some(Value::String(tier.to_owned()))
    }

    fn tools(&self) -> Value {
        Value::Array(self.request().tools().iter().map(tool_definition).collect())
    }

    fn metadata(&self) -> Option<Value> {
        let metadata = self.request().metadata();
        if metadata.is_empty() {
            return None;
        }
        Some(Value::Object(
            metadata
                .iter()
                .map(|(key, value)| (key.clone(), Value::String(value.clone())))
                .collect(),
        ))
    }
}

/// The joined text of every system and developer message.
///
/// This differs from the shared `system_text` in one case: a message whose
/// text is only whitespace is kept here and dropped there. The reference
/// client sent it, and no probe has shown Codex rejects it, so the behavior
/// is retained and pinned by `codex_mode_keeps_a_whitespace_only_instruction`.
fn instructions(request: &Request) -> String {
    request
        .messages()
        .iter()
        .filter(|message| message.is_instruction())
        .map(|message| plain_text(message.content()))
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
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
                ContentPart::ToolCall(call) if call.input.kind() == ToolCallKind::Custom => {
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
/// Items come out in the order the content parts carry them, because this
/// protocol reads the input list positionally: a `reasoning` item names the
/// item that must follow it, so replaying `[reasoning, message, reasoning,
/// function_call]` in any other order breaks the pairing. The parts that
/// belong inside a single message item — text, JSON, images, documents — are
/// collected into one item emitted where the first of them appeared.
fn input_items(message: &Message, custom: &CustomTools) -> Vec<Value> {
    let has_result = message
        .content()
        .iter()
        .any(|part| matches!(part, ContentPart::ToolResult(_)));
    if message.role() == Role::Tool && !has_result {
        // A tool message whose content is only text still answers a call. The
        // message-level label is the only place that id survives.
        if let Some(call_id) = message.tool_call_id() {
            let mut items: Vec<Value> = opaque_items(message).cloned().collect();
            // The same routing the ToolResult path applies: a result
            // answering a custom call earlier in the request is custom even
            // when the tool is not redeclared and the message carries no
            // name.
            let custom = custom.calls.contains(call_id)
                || message
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

    // A preserved `message` item already carries the assistant text, with the
    // id and status a reconstructed item cannot have. Sending both would
    // repeat the text on the wire.
    let replays_message = opaque_items(message)
        .any(|item| item.get("type").and_then(Value::as_str) == Some("message"));
    let content = message_content(message, replays_message);
    let mut pending = (!content.is_empty()).then(|| {
        json!({
            "role":    role(message.role()),
            "content": content,
        })
    });

    let mut items = Vec::new();
    for part in message.content() {
        match part {
            ContentPart::Opaque { data, .. } if claims_opaque(part) => {
                items.push(data.clone());
            }
            // A nameless call cannot be encoded — the provider rejects an
            // empty function name. This decoder never produces one, but a
            // caller-built or foreign-history call can carry one; the
            // reference client skipped those.
            ContentPart::ToolCall(call) if call.name.is_empty() => {}
            ContentPart::ToolCall(call) => items.push(tool_call_item(call)),
            // A lone JSON string result stays a JSON string literal here, as
            // the reference client sent it; Chat Completions unquotes it.
            ContentPart::ToolResult(result) => items.push(tool_output_item(
                &result.tool_call_id,
                &tool_result_text(result, false),
                result.is_error,
                custom.answers_custom(result, message),
            )),
            // A text part a preserved `message` item already carries places
            // nothing of its own.
            ContentPart::Text { .. } if replays_message => {}
            // The first part that belongs inside the message item emits the
            // whole item, so it lands where the caller put its content.
            ContentPart::Text { .. }
            | ContentPart::Json { .. }
            | ContentPart::Image(_)
            | ContentPart::Document(_) => items.extend(pending.take()),
            // Audio never reaches here, reasoning text has no input item, and
            // another provider's opaque part is not ours to send.
            ContentPart::Audio(_)
            | ContentPart::Reasoning(_)
            | ContentPart::Opaque { .. }
            | ContentPart::Unknown(_) => {}
        }
    }
    // A message whose only content is skipped assistant text still has to send
    // the parts that survived.
    items.extend(pending);

    items
}

/// The verbatim replay items this codec owns, in content order.
fn opaque_items(message: &Message) -> impl Iterator<Item = &Value> {
    message.content().iter().filter_map(|part| match part {
        ContentPart::Opaque { data, .. } if claims_opaque(part) => Some(data),
        _ => None,
    })
}

/// Whether an opaque part belongs to this protocol's namespace.
fn claims_opaque(part: &ContentPart) -> bool {
    part.opaque_namespace() == Some(NAMESPACE)
}

/// Encodes the content parts that belong inside a message item.
///
/// `skip_text` drops the text parts a preserved `message` item already
/// carries, so replay never doubles the assistant's own words.
fn message_content(message: &Message, skip_text: bool) -> Vec<Value> {
    let text_type = match message.role() {
        Role::Assistant => "output_text",
        Role::System | Role::Developer | Role::User | Role::Tool => "input_text",
    };
    let system = matches!(message.role(), Role::System | Role::Developer);

    message
        .content()
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text { .. } if skip_text => None,
            // A system or developer item takes `input_text` and nothing else;
            // media inside one draws a provider 400. Only the text travels,
            // and `encode` reports the drop.
            ContentPart::Image(_) | ContentPart::Document(_) if system => None,
            ContentPart::Text { text } => Some(json!({ "type": text_type, "text": text })),
            ContentPart::Json { value } => Some(json!({
                "type": text_type,
                "text": serde_json::to_string(value).unwrap_or_default(),
            })),
            ContentPart::Image(image) => {
                let mut item = json!({
                    "type": "input_image",
                    "image_url": data_url(&image.source),
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
                        MediaSource::Url { url, .. } => {
                            object.insert("file_url".to_owned(), Value::String(url.clone()));
                        }
                        MediaSource::Base64 { .. } => {
                            object.insert(
                                "file_data".to_owned(),
                                Value::String(data_url(&document.source)),
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
            | ContentPart::Opaque { .. }
            | ContentPart::Unknown(_) => None,
        })
        .collect()
}

/// The wire role for one message.
pub(super) fn role(role: Role) -> &'static str {
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
    let arguments = call.input.raw();

    let mut item = match call.input.kind() {
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
