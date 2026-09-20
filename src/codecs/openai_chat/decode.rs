//! Blocking response decoding: the choice, tool calls, usage, and the
//! provider-reported cost.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::codecs::common::usd_micros;
use crate::resolver::ResolvedRoute;
use crate::types::{
    ContentPart, Cost, CostSource, Error, ErrorKind, RetryClassification, TokenCounts, ToolCall,
    ToolCallKind, ToolInput,
};

/// The error a 200 with no choices decodes into.
pub(super) fn no_choices(route: &ResolvedRoute, value: Value) -> Error {
    decode_failure(route, "returned no choices in the response", value)
}

pub(super) fn decode_failure(route: &ResolvedRoute, detail: &str, value: Value) -> Error {
    // A structurally malformed 200 is indistinguishable from a garbled or
    // truncated body, so a fresh attempt is safe — the same classification
    // the transport gives a 200 whose body is not JSON at all.
    Error::new(
        ErrorKind::ResponseDecode,
        format!("provider {} {detail}", route.provider().id()),
    )
    .with_provider(route.provider().id().clone())
    .with_raw_data(value)
    .with_retry(RetryClassification::Safe)
}

/// Decodes one complete tool call from a response message.
pub(super) fn decode_tool_call(call: &Value) -> Result<ContentPart, &'static str> {
    // A call without its id cannot be answered, and one without its name
    // cannot be dispatched; replaying either poisons the conversation. The
    // old serde-strict decode failed retryably, and this keeps that contract.
    let id = call.get("id").and_then(Value::as_str).unwrap_or_default();
    if id.is_empty() {
        return Err("returned a tool call without an id");
    }
    let name = call
        .pointer("/function/name")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if name.is_empty() {
        return Err("returned a tool call without a function name");
    }
    let raw = call
        .pointer("/function/arguments")
        .and_then(Value::as_str)
        .unwrap_or_default();
    Ok(ContentPart::ToolCall(ToolCall {
        id:                id.to_owned(),
        name:              name.to_owned(),
        input:             ToolInput::from_wire(ToolCallKind::Function, raw.to_owned()),
        provider_metadata: BTreeMap::new(),
    }))
}

/// The visible text of a response message or stream delta, in either wire
/// shape.
///
/// Most skins send `content` as a string; a few echo the array form the request
/// uses. Both decode to the same text.
pub(super) fn message_text(message: &Value) -> Option<String> {
    let text = match message.get("content") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect(),
        _ => return None,
    };
    (!text.is_empty()).then_some(text)
}

/// The reasoning text of a response message, in either spelling.
pub(super) fn reasoning_text(message: &Value) -> Option<String> {
    non_empty(message, "reasoning")
        .or_else(|| non_empty(message, "reasoning_content"))
        .map(ToOwned::to_owned)
}

/// A non-empty string field, treating an empty string as absent.
pub(super) fn non_empty<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
}

/// Normalizes one `usage` object into the five disjoint token buckets.
///
/// Every counter this protocol reports is inclusive: `prompt_tokens` contains
/// the cache-read and cache-write details, and `completion_tokens` contains the
/// reasoning detail. Each detail also has a flat spelling that a skin may send
/// instead — DeepSeek's `prompt_cache_hit_tokens` and Modal's
/// `reasoning_tokens` — and the nested detail wins when both are present.
/// Cache writes have a third spelling: a skin fronting an Anthropic model
/// passes through `cache_creation_input_tokens`, nested and flat (Venice
/// sends both).
pub(super) fn token_counts(usage: &Value) -> TokenCounts {
    let count = |pointer: &str| usage.pointer(pointer).and_then(Value::as_u64);

    let cache_read = count("/prompt_tokens_details/cached_tokens")
        .or_else(|| count("/prompt_cache_hit_tokens"))
        .unwrap_or_default();
    let cache_write = count("/prompt_tokens_details/cache_write_tokens")
        .or_else(|| count("/prompt_tokens_details/cache_creation_input_tokens"))
        .or_else(|| count("/cache_creation_input_tokens"))
        .unwrap_or_default();
    let reasoning = count("/completion_tokens_details/reasoning_tokens")
        .or_else(|| count("/reasoning_tokens"))
        .unwrap_or_default();

    TokenCounts::from_inclusive(
        count("/prompt_tokens").unwrap_or_default(),
        count("/completion_tokens").unwrap_or_default(),
        reasoning,
        cache_read,
        cache_write,
    )
}

/// The cost a provider reported in-band, when it reported one.
///
/// Two paths in precedence order: OpenRouter's `usage.cost` and Venice's
/// top-level `cost.usd`. Venice's `cost.diem` sibling is a different currency
/// and is ignored, as is `usage.cost_details.upstream_inference_cost`, which is
/// an upstream figure rather than what this call is billed; both survive in
/// [`Response::raw`].
pub(super) fn provider_cost(body: &Value) -> Option<Cost> {
    let usd = body
        .pointer("/usage/cost")
        .and_then(Value::as_f64)
        .or_else(|| body.pointer("/cost/usd").and_then(Value::as_f64))?;

    Some(Cost {
        usd_micros: usd_micros(usd),
        source:     CostSource::Provider,
    })
}
