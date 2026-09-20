//! Request encoding: input items, tools, and the shared body fields.

use std::collections::BTreeSet;

use serde_json::{Map, Value, json};

use super::NAMESPACE;
use crate::codecs::common::plain_text;
use crate::types::{
    ContentPart, MediaSource, Message, Request, ResponseFormat, Role, Speed, ToolCall,
    ToolCallKind, ToolChoice, ToolDefinition, ToolDefinitionKind, ToolResult,
};

/// The fields the Codex deployment and the public API encode identically.
pub(super) fn shared_body(request: &Request) -> Map<String, Value> {
    let mut body = Map::new();

    if let Some(effort) = request.reasoning_effort() {
        let effort = serde_json::to_value(effort).unwrap_or(Value::Null);
        body.insert("reasoning".to_owned(), json!({ "effort": effort }));
    }
    if let Some(speed) = request.speed() {
        let tier = match speed {
            Speed::Fast => "priority",
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
pub(super) fn instructions(request: &Request) -> String {
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
pub(super) fn is_system(message: &Message) -> bool {
    matches!(message.role(), Role::System | Role::Developer)
}

/// The tool kinds one request declared, used to route tool results.
///
/// A `custom_tool_call_output` is a different wire item from a
/// `function_call_output`, and a result carries no kind of its own. The kind is
/// recovered from the call it answers, from the tool it names, or from the
/// message label the caller set.
pub(super) struct CustomTools {
    /// Names declared as custom tools by this request.
    names: BTreeSet<String>,
    /// Call ids of custom tool calls earlier in this request.
    calls: BTreeSet<String>,
}

impl CustomTools {
    pub(super) fn new(request: &Request) -> Self {
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
pub(super) fn input_items(message: &Message, custom: &CustomTools) -> Vec<Value> {
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
            ContentPart::ToolResult(result) => items.push(tool_output_item(
                &result.tool_call_id,
                &result_output(result),
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
                        MediaSource::Url { url, .. } => {
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

/// The text a tool result sends back.
///
/// A result whose content is only structured JSON sends the bare value — one
/// value on its own, several as an array — rather than the `ContentPart`
/// envelope that wraps it, because the tool's own JSON is what the model was
/// promised. Text-only content sends the joined text even when it is empty —
/// a command with no output answered with nothing, not with a serialized
/// envelope. Anything else falls back to the serialized parts, which keeps
/// mixed content readable instead of dropping the half this protocol has no
/// field for.
fn result_output(result: &ToolResult) -> String {
    let text = plain_text(&result.content);
    let text_only = result
        .content
        .iter()
        .all(|part| matches!(part, ContentPart::Text { .. }));
    if !text.is_empty() || text_only {
        return text;
    }

    let values: Vec<&Value> = result
        .content
        .iter()
        .filter_map(|part| match part {
            ContentPart::Json { value } => Some(value),
            _ => None,
        })
        .collect();
    match values.as_slice() {
        [value] if result.content.len() == 1 => serde_json::to_string(value).unwrap_or_default(),
        values if !values.is_empty() && values.len() == result.content.len() => {
            serde_json::to_string(values).unwrap_or_default()
        }
        _ => serde_json::to_string(&result.content).unwrap_or_default(),
    }
}

/// Encodes one image or media source as the URL this protocol accepts.
fn media_url(source: &MediaSource) -> String {
    match source {
        MediaSource::Url { url, .. } => url.clone(),
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
pub(super) fn tool_choice(choice: &ToolChoice) -> Value {
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
