use crate::types::{ContentPart, Message, Request};

pub(crate) fn estimate_input_tokens(request: &Request) -> u64 {
    let bytes = request
        .messages()
        .iter()
        .flat_map(Message::content)
        .map(|part| match part {
            ContentPart::Text { text } => length(text),
            ContentPart::Reasoning(reasoning) => length(&reasoning.text),
            ContentPart::ToolCall(call) => {
                length(&call.name).saturating_add(length(&call.arguments.to_string()))
            }
            ContentPart::ToolResult(result) => result
                .content
                .iter()
                .map(|part| match part {
                    ContentPart::Text { text } => length(text),
                    _ => 0,
                })
                .sum(),
            ContentPart::Image(_) | ContentPart::Audio(_) | ContentPart::Document(_) => 0,
        })
        .sum::<u64>();
    bytes.saturating_add(3) / 4
}

fn length(value: &str) -> u64 {
    u64::try_from(value.len()).unwrap_or(u64::MAX)
}
