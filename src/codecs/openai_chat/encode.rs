//! Request encoding: messages, tools, and cache breakpoints.

use serde_json::{Value, json};

use super::OPAQUE_PREFIX;
use crate::codecs::content::{data_url, plain_text, tool_result_text};
use crate::types::{
    ContentPart, ImageContent, Message, ReasoningEffort, ResponseFormat, Role, ToolCall,
    ToolChoice, ToolDefinition, ToolDefinitionKind, ToolResult,
};

/// Marks the cacheable prefix of a conversation for an Anthropic upstream.
///
/// Two breakpoints, the same pair the Anthropic codec places: the last system
/// message of the leading run — the system prompt proper, which the tools and
/// instructions precede on the upstream wire — and the second-to-last user
/// turn, so each iteration of an agent loop reads the prefix the previous one
/// wrote. A tool result is its own message here and counts as a user turn,
/// because it rides in a user message upstream.
///
/// A system message appended later in the conversation is an instruction, not
/// the prompt: marking it would move the system breakpoint every time an agent
/// loop appends one, and the prefix written on the previous turn would never
/// be read back.
pub(super) fn mark_cache_breakpoints(messages: &mut [Value]) {
    let leading = messages
        .iter()
        .position(|message| role_of(message) != Some("system"))
        .unwrap_or(messages.len());
    if let Some(system) = messages[..leading].last_mut() {
        mark_cached(system);
    }

    let user_turns: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, message)| matches!(role_of(message), Some("user" | "tool")))
        .map(|(index, _)| index)
        .collect();
    let Some(turn) = user_turns.len().checked_sub(2).map(|last| user_turns[last]) else {
        return;
    };
    mark_cached(&mut messages[turn]);
}

/// The wire role of one encoded message.
fn role_of(message: &Value) -> Option<&str> {
    message.get("role").and_then(Value::as_str)
}

/// Adds an ephemeral cache breakpoint to one wire message.
///
/// The plain-string content form has nowhere to carry the annotation, so it
/// becomes a one-part array; a message already in the array form marks its
/// last part. A message with no content is left alone.
fn mark_cached(message: &mut Value) {
    let Some(content) = message.get_mut("content") else {
        return;
    };
    let breakpoint = json!({ "type": "ephemeral" });
    *content = match content.take() {
        Value::String(text) => {
            json!([{ "type": "text", "text": text, "cache_control": breakpoint }])
        }
        Value::Array(mut parts) => {
            if let Some(part) = parts.last_mut().and_then(Value::as_object_mut) {
                part.insert("cache_control".to_owned(), breakpoint);
            }
            Value::Array(parts)
        }
        other => other,
    };
}

/// Encodes one canonical message into the wire messages it produces.
///
/// A tool result is its own wire message in this protocol, so one canonical
/// message that carries several results expands into several wire messages.
pub(super) fn encode_message(message: &Message) -> Vec<Value> {
    let results: Vec<Value> = message
        .content()
        .iter()
        .filter_map(|part| match part {
            ContentPart::ToolResult(result) => Some(encode_tool_result(result)),
            _ => None,
        })
        .collect();

    if message.role() == Role::Tool {
        if !results.is_empty() {
            return results;
        }
        // A tool message whose result was flattened into plain text still has
        // to answer a specific call.
        let mut value = json!({ "role": "tool", "content": plain_text(message.content()) });
        if let Some(id) = message.tool_call_id() {
            value["tool_call_id"] = id.into();
        }
        return vec![value];
    }

    let mut encoded = vec![encode_chat_message(message)];
    encoded.extend(results);
    encoded
}

/// Encodes one system, developer, user, or assistant message.
fn encode_chat_message(message: &Message) -> Value {
    let parts: Vec<Value> = message
        .content()
        .iter()
        .filter_map(encode_content_part)
        .collect();
    let all_text = !parts.is_empty()
        && parts
            .iter()
            .all(|part| part.get("type").and_then(Value::as_str) == Some("text"));
    let content = match parts.as_slice() {
        // A message with no encodable parts — an assistant turn that only
        // calls tools — omits the member, the shape the reference client
        // sent. Strict skins validate content as string-or-array and reject
        // an explicit null.
        [] => None,
        // Text-only content uses the plain string form every skin accepts —
        // the part-array form is reserved for content only media-capable
        // skins receive, because a strict text-only skin rejects it. The
        // texts join unseparated, as the reference client sent them.
        _ if all_text => Some(Value::String(
            parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join(""),
        )),
        _ => Some(Value::Array(parts)),
    };

    let mut value = json!({ "role": role_name(message.role()) });
    if let Some(content) = content {
        value["content"] = content;
    }
    if let Some(name) = message.name() {
        value["name"] = name.into();
    }

    let tool_calls: Vec<Value> = message
        .content()
        .iter()
        .filter_map(|part| match part {
            ContentPart::ToolCall(call) => Some(encode_tool_call(call)),
            _ => None,
        })
        .collect();
    if !tool_calls.is_empty() {
        value["tool_calls"] = Value::Array(tool_calls);
    }

    // Kimi and DeepSeek require the assistant's own reasoning back when a
    // tool-call turn continues, so replay it rather than dropping it. Several
    // parts join unseparated, byte for byte what the reference client sent —
    // the same rule the text join follows.
    let reasoning: Vec<&str> = message
        .content()
        .iter()
        .filter_map(|part| match part {
            ContentPart::Reasoning(reasoning) if !reasoning.redacted => {
                Some(reasoning.text.as_str())
            }
            _ => None,
        })
        .collect();
    if !reasoning.is_empty() {
        value["reasoning_content"] = reasoning.concat().into();
    }

    // An opaque part of this dialect names the message field it came from, so
    // `openai_compatible.reasoning_details` replays as `reasoning_details`.
    // Parts of another namespace are skipped, which keeps failover working.
    for part in message.content() {
        if let ContentPart::Opaque { kind, data } = part
            && let Some(field) = kind.strip_prefix(OPAQUE_PREFIX)
        {
            value[field] = data.clone();
        }
    }

    value
}

/// Encodes the content parts this protocol carries inside a message.
///
/// Audio and document parts have no Chat Completions representation and are
/// skipped. Reasoning, tool calls, tool results, and opaque parts are encoded
/// by [`encode_chat_message`] as message fields rather than content parts.
fn encode_content_part(part: &ContentPart) -> Option<Value> {
    match part {
        ContentPart::Text { text } => Some(json!({ "type": "text", "text": text })),
        ContentPart::Json { value } => Some(json!({ "type": "text", "text": value.to_string() })),
        ContentPart::Image(image) => Some(encode_image(image)),
        // Audio and documents are rejected before dispatch by
        // `reject_unencodable`.
        ContentPart::Audio(_)
        | ContentPart::Document(_)
        | ContentPart::Reasoning(_)
        | ContentPart::ToolCall(_)
        | ContentPart::ToolResult(_)
        | ContentPart::Opaque { .. }
        | ContentPart::Unknown(_) => None,
    }
}

/// Encodes an image as the `image_url` part this protocol uses for both
/// sources; see `data_url`.
fn encode_image(image: &ImageContent) -> Value {
    let mut image_url = json!({ "url": data_url(&image.source) });
    if let Some(detail) = &image.detail {
        image_url["detail"] = detail.as_str().into();
    }
    json!({ "type": "image_url", "image_url": image_url })
}

/// Encodes one assistant tool call for replay.
fn encode_tool_call(call: &ToolCall) -> Value {
    json!({
        "id": call.id,
        "type": "function",
        "function": { "name": call.name, "arguments": wire_arguments(call) },
    })
}

/// The argument text to send for a tool call.
///
/// The provider's own string is replayed byte for byte when the call carries
/// one, so a re-sent turn matches what the model produced. A custom tool call
/// that reached this dialect through failover holds its input as a JSON string
/// and is sent as that text, not as a quoted JSON literal.
fn wire_arguments(call: &ToolCall) -> String {
    call.input.raw().to_owned()
}

/// Encodes one tool result as its own `tool` message.
///
/// This protocol takes a string here and nothing else, so the content is
/// flattened by `tool_result_text`. A lone JSON string value sends its raw
/// text unquoted, as the reference encoder did — the tool answered with that
/// text, not with a JSON string literal.
fn encode_tool_result(result: &ToolResult) -> Value {
    json!({
        "role": "tool",
        "tool_call_id": result.tool_call_id,
        "content": tool_result_text(result, true),
    })
}

/// Encodes one function tool definition.
///
/// A custom tool has no representation here and is rejected by
/// [`OpenAiChatCodec::encode`] before this runs, so it is never downgraded.
pub(super) fn encode_tool(tool: &ToolDefinition) -> Option<Value> {
    match &tool.kind {
        ToolDefinitionKind::Function { input_schema } => Some(json!({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": input_schema,
            },
        })),
        ToolDefinitionKind::Custom { .. } => None,
    }
}

pub(super) fn encode_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::None => json!("none"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Tool { name } => json!({ "type": "function", "function": { "name": name } }),
    }
}

pub(super) fn encode_response_format(format: &ResponseFormat) -> Value {
    match format {
        ResponseFormat::Text => json!({ "type": "text" }),
        ResponseFormat::JsonObject => json!({ "type": "json_object" }),
        ResponseFormat::JsonSchema { name, schema } => json!({
            "type": "json_schema",
            "json_schema": { "name": name, "schema": schema, "strict": true },
        }),
    }
}

/// The wire role for one canonical role.
///
/// A developer message is sent as a system message. Only OpenAI itself takes
/// `developer` on this endpoint; a strict skin accepts system, user,
/// assistant, and tool and rejects the request over anything else, so the
/// closest role every skin understands is the compatible choice.
fn role_name(role: Role) -> &'static str {
    match role {
        Role::System | Role::Developer => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

/// The wire spelling of one reasoning effort level.
///
/// The canonical names are the wire vocabulary: this dialect passes the level
/// through untranslated, and a skin that knows fewer levels clamps its own.
pub(super) fn effort_name(effort: ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::Minimal => "minimal",
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::Xhigh => "xhigh",
        ReasoningEffort::Max => "max",
    }
}
