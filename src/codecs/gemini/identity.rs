//! The tool-call identity invariant this protocol makes the codec keep.
//!
//! Gemini names a function call by function name, usually with no id, and
//! answers a call by name as well. Everything that restores a stable id to a
//! call, scopes it to its response, or recovers a name the application did
//! not keep lives here, so the invariant is findable in one place: the
//! blocking and streaming decoders synthesize ids the same way, and the
//! encoder recovers names from the assistant turn that made the call.

use std::collections::HashMap;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher as _, Hasher as _};

use serde_json::Value;

use crate::types::{ContentPart, Message, Request, Role};

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

/// Maps every tool-call id in the history to the function it called.
///
/// `functionResponse` identifies the call it answers by function name, and a
/// canonical [`ToolResult`] carries the name only when the application kept it.
/// The assistant turn that made the call always carries it, so the history is
/// the reliable source. Without this, a result whose name is missing sends the
/// call id as the function name, which matches no declared function.
pub(super) fn tool_call_names(request: &Request) -> HashMap<&str, &str> {
    request
        .messages()
        .iter()
        .filter(|message| message.role() == Role::Assistant)
        .flat_map(Message::content)
        .filter_map(|part| match part {
            ContentPart::ToolCall(call) => Some((call.id.as_str(), call.name.as_str())),
            _ => None,
        })
        .collect()
}
