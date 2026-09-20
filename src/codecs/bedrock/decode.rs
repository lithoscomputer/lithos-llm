//! Blocking response decoding: content blocks, stop reasons, and usage.

use serde_json::{Value, json};

use crate::codecs::common::{ANTHROPIC_SIGNATURES, finish_reason};
use crate::types::{ContentPart, FinishReason, ReasoningContent, TokenCounts, ToolCall};

/// Maps a Converse `stopReason` onto the normalized finish reason.
///
/// `model_context_window_exceeded` is Converse's own name for a generation that
/// ran out of context, which is the same outcome as `max_tokens` and so maps
/// onto the same reason. `refusal` never reaches here: both decode paths fail
/// the call before they ask for a finish reason.
pub(super) fn stop_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("stop_sequence") => FinishReason::Stop,
        Some("model_context_window_exceeded") => FinishReason::Length,
        Some("content_filtered" | "guardrail_intervened") => FinishReason::ContentFilter,
        other => finish_reason(other),
    }
}

/// Reads a Converse usage object into the five disjoint buckets.
///
/// Converse counters are already disjoint: `inputTokens` excludes both cache
/// counters, so nothing is subtracted. Reasoning tokens are folded into
/// `outputTokens` with no separate counter, so `reasoning` stays zero rather
/// than guessing at a split.
pub(super) fn token_counts(usage: Option<&Value>) -> TokenCounts {
    let Some(usage) = usage else {
        return TokenCounts::default();
    };
    let count = |key: &str| usage.get(key).and_then(Value::as_u64).unwrap_or_default();
    TokenCounts {
        input:       count("inputTokens"),
        output:      count("outputTokens"),
        reasoning:   0,
        cache_read:  count("cacheReadInputTokens"),
        cache_write: count("cacheWriteInputTokens"),
    }
}

/// Decodes one Converse content block.
///
/// Unknown block kinds are skipped, because the Converse content union grows
/// with every model family.
pub(super) fn decode_content_block(block: &Value) -> Option<ContentPart> {
    if let Some(text) = block.get("text").and_then(Value::as_str) {
        if text.is_empty() {
            return None;
        }
        return Some(ContentPart::Text {
            text: text.to_owned(),
        });
    }
    if let Some(tool_use) = block.get("toolUse") {
        let id = tool_use.get("toolUseId").and_then(Value::as_str)?;
        let name = tool_use.get("name").and_then(Value::as_str)?;
        // A no-argument call is canonically `{}`, so it re-encodes to a valid
        // `toolUse.input` document.
        let arguments = match tool_use.get("input") {
            None | Some(Value::Null) => json!({}),
            Some(value) => value.clone(),
        };
        return Some(ContentPart::ToolCall(ToolCall::function(
            id, name, arguments,
        )));
    }
    if let Some(reasoning) = block.get("reasoningContent") {
        return decode_reasoning_block(reasoning);
    }
    None
}

/// Decodes a `reasoningContent` block, redacted or not.
fn decode_reasoning_block(reasoning: &Value) -> Option<ContentPart> {
    if let Some(text_block) = reasoning.get("reasoningText") {
        let signature = text_block
            .get("signature")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let signature_origin = signature.is_some().then(|| ANTHROPIC_SIGNATURES.to_owned());
        return Some(ContentPart::Reasoning(ReasoningContent {
            text: text_block
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            signature,
            signature_origin,
            redacted: false,
        }));
    }
    let redacted = reasoning.get("redactedContent").and_then(Value::as_str)?;
    Some(ContentPart::Reasoning(ReasoningContent {
        text:             redacted.to_owned(),
        signature:        None,
        signature_origin: None,
        redacted:         true,
    }))
}
