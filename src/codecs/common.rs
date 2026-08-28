#[cfg(any(feature = "openai", feature = "openai-compatible"))]
use serde_json::json;
use serde_json::{Map, Value};

#[cfg(any(feature = "openai", feature = "openai-compatible"))]
use crate::types::ToolChoice;
use crate::types::{ContentPart, FinishReason, Request, TokenCounts};
#[cfg(any(feature = "anthropic", feature = "bedrock", feature = "gemini"))]
use crate::types::{Message, Role};

pub(crate) fn endpoint(base_url: &str, path: &str) -> String {
    format!(
        "{}/{}",
        base_url.trim_end_matches('/'),
        path.trim_start_matches('/')
    )
}

pub(crate) fn options(request: &Request, namespace: &str) -> Map<String, Value> {
    request
        .provider_options()
        .get(namespace)
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default()
}

pub(crate) fn plain_text(parts: &[ContentPart]) -> String {
    parts
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

#[cfg(any(feature = "anthropic", feature = "bedrock", feature = "gemini"))]
pub(crate) fn system_text(messages: &[Message]) -> String {
    messages
        .iter()
        .filter(|message| matches!(message.role(), Role::System | Role::Developer))
        .map(|message| plain_text(message.content()))
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[cfg(any(feature = "openai", feature = "openai-compatible"))]
pub(crate) fn tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => Value::String("auto".to_owned()),
        ToolChoice::None => Value::String("none".to_owned()),
        ToolChoice::Required => Value::String("required".to_owned()),
        ToolChoice::Tool { name } => json!({ "type": "function", "name": name }),
    }
}

pub(crate) fn finish_reason(value: Option<&str>) -> FinishReason {
    match value {
        None | Some("stop" | "end_turn" | "STOP") => FinishReason::Stop,
        Some("length" | "max_tokens" | "MAX_TOKENS") => FinishReason::Length,
        Some("tool_calls" | "tool_use") => FinishReason::ToolCall,
        Some("content_filter" | "SAFETY" | "BLOCKLIST" | "PROHIBITED_CONTENT") => {
            FinishReason::ContentFilter
        }
        Some(other) => FinishReason::Other(other.to_owned()),
    }
}

pub(crate) fn usage(input: Option<u64>, output: Option<u64>) -> TokenCounts {
    TokenCounts {
        input:            input.unwrap_or_default(),
        output:           output.unwrap_or_default(),
        cached_input:     0,
        reasoning_output: 0,
    }
}

#[cfg(any(feature = "openai", feature = "openai-compatible"))]
pub(crate) fn arguments(value: &Value) -> Value {
    value
        .as_str()
        .and_then(|arguments| serde_json::from_str(arguments).ok())
        .unwrap_or_else(|| value.clone())
}
