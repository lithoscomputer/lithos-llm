//! Blocking response decoding: candidates, tool-call identity, and usage.

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher as _, Hasher as _};

use serde_json::{Value, json};

use super::NAMESPACE;
use crate::codecs::content::{GEMINI_SIGNATURES, finish_reason, promote_tool_finish};
use crate::codecs::errors::{content_filter, malformed_success};
use crate::resolver::ResolvedRoute;
use crate::types::{ContentPart, Error, FinishReason, ReasoningContent, TokenCounts, ToolCall};

/// Decodes one candidate into its content and its finish reason.
///
/// `response_id` scopes the ids synthesized for the function calls; see
/// [`tool_call_id`].
pub(super) fn decode_candidate(
    candidate: &Value,
    response_id: &str,
) -> (Vec<ContentPart>, FinishReason) {
    let mut content = Vec::new();
    let mut calls = 0;
    for part in candidate
        .pointer("/content/parts")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if let Some(function_call) = part.get("functionCall") {
            content.push(decode_tool_call(part, function_call, response_id, calls));
            calls += 1;
        } else if let Some(text) = part.get("text").and_then(Value::as_str) {
            content.push(decode_text(part, text));
        }
    }

    // Gemini reports `STOP` even when the candidate is nothing but function
    // calls; see `promote_tool_finish`.
    let finished = promote_tool_finish(
        finish_reason(candidate.get("finishReason").and_then(Value::as_str)),
        calls > 0,
    );
    (content, finished)
}

/// The error for a 200 response that carried no candidates.
///
/// Gemini reports a prompt it refused to answer as `promptFeedback.blockReason`
/// on an otherwise successful body. Decoding that as an empty `Stop` response
/// makes a blocked prompt indistinguishable from a model that had nothing to
/// say, so it becomes a failure instead.
///
/// The block reason becomes the provider code, so the reasons beyond `SAFETY`
/// — `BLOCKLIST`, `PROHIBITED_CONTENT`, and any Google adds — all classify as
/// [`ErrorKind::ContentFilter`](crate::types::ErrorKind::ContentFilter) rather
/// than only the spellings the shared classifier happens to recognize.
///
/// A body with neither candidates nor a block reason is malformed rather than
/// blocked, so it decodes into
/// [`ErrorKind::ResponseDecode`](crate::types::ErrorKind::ResponseDecode).
pub(super) fn no_candidates(route: &ResolvedRoute, value: Value) -> Error {
    let Some(reason) = value
        .pointer("/promptFeedback/blockReason")
        .and_then(Value::as_str)
    else {
        return malformed_success(
            route,
            "Gemini returned no candidates in the response",
            Some(value),
        );
    };

    blocked_prompt(route, reason).with_raw_data(value)
}

/// The classified error for a prompt Gemini blocked, streamed or not.
pub(super) fn blocked_prompt(route: &ResolvedRoute, reason: &str) -> Error {
    content_filter(
        route,
        reason,
        &format!("blocked the prompt under its content policy (block reason {reason})"),
    )
}

/// Decodes a text part into visible text or reasoning.
fn decode_text(part: &Value, text: &str) -> ContentPart {
    if part.get("thought").and_then(Value::as_bool) != Some(true) {
        return ContentPart::Text {
            text: text.to_owned(),
        };
    }

    let signature = thought_signature(part).map(ToOwned::to_owned);
    let signature_origin = signature.is_some().then(|| GEMINI_SIGNATURES.to_owned());
    ContentPart::Reasoning(ReasoningContent {
        text: text.to_owned(),
        signature,
        signature_origin,
        redacted: false,
    })
}

/// Decodes a `functionCall` part into a tool call.
///
/// `ordinal` counts the function calls already decoded from this response, so
/// the synthesized id is stable within the response it came from.
fn decode_tool_call(
    part: &Value,
    function_call: &Value,
    response_id: &str,
    ordinal: usize,
) -> ContentPart {
    let name = function_call
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let mut call = ToolCall::function(
        tool_call_id(function_call, name, response_id, ordinal),
        name,
        function_call
            .get("args")
            .cloned()
            .unwrap_or_else(|| json!({})),
    );
    if let Some(signature) = thought_signature(part) {
        call.provider_metadata.insert(
            NAMESPACE.to_owned(),
            json!({ "thoughtSignature": signature }),
        );
    }
    ContentPart::ToolCall(call)
}

/// The identity of one function call.
///
/// A call Gemini gave an id keeps it. Gemini normally supplies none, so one is
/// synthesized as `{name}-{ordinal}-{response_id}`:
///
/// - `{name}-{ordinal}` alone is deterministic, which a minted UUID is not —
///   decoding one payload twice yields one id — but it repeats across turns, so
///   a conversation with two `search` calls carries `search-0` twice. An
///   application keyed by call id then collides, and the replayed history sends
///   duplicate ids on the wire.
/// - `responseId` is the provider's own name for one response, so appending it
///   makes the id unique per response while staying a pure function of the
///   payload. It is the fallback that is dropped, not the ordinal: the id stays
///   readable, and the plain `{name}-{ordinal}` form remains its prefix.
///
/// A payload that carries no `responseId` — some gateways omit it — is scoped
/// by a [`response_nonce`] minted once per decode instead. That trades the
/// pure-function property for uniqueness: the bare `{name}-{ordinal}` form
/// repeats across turns, so a multi-turn loop calling the same tool once per
/// turn carried `search-0` twice — colliding in id-keyed applications and
/// replaying duplicate `functionCall.id`/`functionResponse.id` values on the
/// wire.
pub(super) fn tool_call_id(
    function_call: &Value,
    name: &str,
    response_id: &str,
    ordinal: usize,
) -> String {
    if let Some(id) = function_call
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
    {
        return id.to_owned();
    }

    format!("{name}-{ordinal}-{response_id}")
}

/// A random scope for the synthesized ids of one response without a
/// `responseId`.
///
/// Every [`RandomState`] carries its own keys, so finishing an empty hasher
/// yields a fresh value per call without a dependency on a randomness crate,
/// which this crate deliberately avoids.
pub(super) fn response_nonce() -> String {
    format!("{:016x}", RandomState::new().build_hasher().finish())
}

/// The thought signature carried alongside a part, when it has one.
pub(super) fn thought_signature(part: &Value) -> Option<&str> {
    part.get("thoughtSignature").and_then(Value::as_str)
}

/// Normalizes `usageMetadata` into the five disjoint token buckets.
///
/// The Gemini counters are inclusive in one direction and exclusive in the
/// other, which is why none of the shared helpers fit:
///
/// - `promptTokenCount` **includes** `cachedContentTokenCount`, so the cached
///   tokens are subtracted out of `input`.
/// - `toolUsePromptTokenCount` sits **outside** `promptTokenCount`, so it is
///   added to `input`.
/// - `candidatesTokenCount` **excludes** `thoughtsTokenCount`, so `output`
///   passes through untouched and `reasoning` is taken as reported.
///
/// `cache_write` is always zero: creating a Gemini cache is a separate
/// `cachedContents` call, not part of `generateContent` usage.
pub(super) fn token_counts(metadata: &Value) -> TokenCounts {
    let count = |key: &str| {
        metadata
            .get(key)
            .and_then(Value::as_u64)
            .unwrap_or_default()
    };
    let cache_read = count("cachedContentTokenCount");

    TokenCounts {
        input: count("promptTokenCount")
            .saturating_sub(cache_read)
            .saturating_add(count("toolUsePromptTokenCount")),
        output: count("candidatesTokenCount"),
        reasoning: count("thoughtsTokenCount"),
        cache_read,
        cache_write: 0,
    }
}
