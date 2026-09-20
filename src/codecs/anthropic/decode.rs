//! Blocking response decoding: content blocks, usage, and stream-shared
//! field readers.

use std::collections::BTreeMap;

use serde_json::{Value, json};

use super::NAMESPACE;
use crate::codecs::common::ANTHROPIC_SIGNATURES;
use crate::types::{
    ContentBlockId, ContentPart, ReasoningContent, TokenCounts, ToolArguments, ToolCall, ToolInput,
};

/// The first field a Messages response must carry and this body does not.
///
/// Every real response carries all four, so their absence means the body is
/// not one. Fields the API genuinely omits — `stop_reason` on a streamed
/// message, `stop_details` on anything but a refusal — stay optional.
pub(super) fn missing_response_field(value: &Value) -> Option<&'static str> {
    if !value.get("id").is_some_and(Value::is_string) {
        return Some("id");
    }
    if !value.get("model").is_some_and(Value::is_string) {
        return Some("model");
    }
    if !value.get("content").is_some_and(Value::is_array) {
        return Some("content");
    }
    if !value.get("usage").is_some_and(Value::is_object) {
        return Some("usage");
    }
    None
}

/// Decodes one response content block.
pub(super) fn decode_block(block: &Value) -> Option<ContentPart> {
    match block.get("type").and_then(Value::as_str) {
        Some("text") => Some(ContentPart::Text {
            text: field(block, "text").to_owned(),
        }),
        Some("thinking") => {
            let signature = block
                .get("signature")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
            let signature_origin = signature.is_some().then(|| ANTHROPIC_SIGNATURES.to_owned());
            Some(ContentPart::Reasoning(ReasoningContent {
                text: field(block, "thinking").to_owned(),
                signature,
                signature_origin,
                redacted: false,
            }))
        }
        // The encrypted blob is the whole block. It carries no signature and
        // must be replayed exactly as it arrived.
        Some("redacted_thinking") => Some(ContentPart::Reasoning(ReasoningContent {
            text:             field(block, "data").to_owned(),
            signature:        None,
            signature_origin: None,
            redacted:         true,
        })),
        Some("tool_use") => Some(ContentPart::ToolCall(decode_tool_use(block))),
        // A server-side block this codec does not model still has to survive a
        // replay, so it is kept whole under this codec's namespace.
        Some(kind) => Some(ContentPart::opaque(
            format!("{NAMESPACE}.{kind}"),
            block.clone(),
        )),
        None => None,
    }
}

/// Decodes one `tool_use` block.
///
/// Anthropic sends parsed JSON input rather than an argument string, so
/// `raw_arguments` stays empty unless the wire really did carry a string.
fn decode_tool_use(block: &Value) -> ToolCall {
    let arguments = block.get("input").cloned().unwrap_or_else(|| json!({}));

    ToolCall {
        id:                field(block, "id").to_owned(),
        name:              field(block, "name").to_owned(),
        input:             ToolInput::Function(ToolArguments::from_json(arguments)),
        provider_metadata: BTreeMap::new(),
    }
}

/// Normalizes one Anthropic usage object into the five disjoint buckets.
pub(super) fn token_counts(usage: Option<&Value>) -> TokenCounts {
    let mut counts = TokenCounts::default();
    if let Some(usage) = usage {
        fold_usage(&mut counts, usage);
    }
    counts
}

/// Folds whichever usage counters one wire object carries into a snapshot.
///
/// Anthropic's counters are **already disjoint**: `input_tokens` excludes both
/// `cache_read_input_tokens` and `cache_creation_input_tokens`, so nothing is
/// subtracted. Thinking tokens are billed inside `output_tokens` with no
/// separate counter, so `reasoning` stays 0 and `billable_output` still adds
/// up. Each field is assigned, not added, because a repeated counter is the
/// provider restating the same cumulative total.
pub(super) fn fold_usage(counts: &mut TokenCounts, usage: &Value) {
    if let Some(input) = usage.get("input_tokens").and_then(Value::as_u64) {
        counts.input = input;
    }
    if let Some(output) = usage.get("output_tokens").and_then(Value::as_u64) {
        counts.output = output;
    }
    if let Some(read) = usage.get("cache_read_input_tokens").and_then(Value::as_u64) {
        counts.cache_read = read;
    }
    if let Some(write) = usage
        .get("cache_creation_input_tokens")
        .and_then(Value::as_u64)
    {
        counts.cache_write = write;
    }
}

/// One string field of a wire object, or `""` when it is absent.
pub(super) fn field<'a>(value: &'a Value, key: &str) -> &'a str {
    value.get(key).and_then(Value::as_str).unwrap_or_default()
}

/// The stream-local id of the block one event addresses.
///
/// Anthropic identifies blocks by a top-level `index`, so the id is derived
/// from it. A `tool_use` block's own id is the provider's tool-call id and is
/// kept in [`ContentBlockKind::ToolCall`] instead.
pub(super) fn block_id(value: &Value) -> ContentBlockId {
    let index = value
        .get("index")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    ContentBlockId::new(format!("block-{index}"))
}
