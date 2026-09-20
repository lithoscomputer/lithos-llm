//! The `reasoning_details` channel: verbatim on a complete response,
//! coalesced from fragments on a stream.

use std::slice::from_ref;

use serde_json::Value;

use super::{OPAQUE_PREFIX, REASONING_DETAILS};
use crate::types::ContentPart;

/// The opaque content kind the structured reasoning channel replays as.
pub(super) fn details_kind() -> String {
    format!("{OPAQUE_PREFIX}{REASONING_DETAILS}")
}

/// The structured reasoning channel of one response, in wire order.
///
/// A complete response carries the whole array at once. A stream splits each
/// logical detail across chunks, so the fragments are coalesced back into one
/// entry before the opaque part is built — an aggregator that receives half a
/// signature back rejects the turn.
#[derive(Default)]
pub(super) struct ReasoningDetails {
    entries: Vec<Value>,
}

impl ReasoningDetails {
    /// Absorbs one streamed `reasoning_details` payload.
    ///
    /// A fragment continues the most recent entry of the same `type` whose
    /// `index` matches, so details that arrive interleaved still coalesce. A
    /// fragment without an `index` continues the most recent entry of its
    /// type, which is what a skin that omits the field means by it. Anything
    /// that is not an object carries nothing replayable and is dropped.
    pub(super) fn absorb(&mut self, payload: &Value) {
        let incoming: &[Value] = match payload {
            Value::Array(entries) => entries,
            Value::Object(_) => from_ref(payload),
            _ => &[],
        };
        for entry in incoming.iter().filter(|entry| entry.is_object()) {
            match self
                .entries
                .iter_mut()
                .rev()
                .find(|existing| continues_detail(existing, entry))
            {
                Some(existing) => merge_detail(existing, entry),
                None => self.entries.push(entry.clone()),
            }
        }
    }

    /// The accumulated entries, or `None` when nothing usable arrived.
    pub(super) fn entries(&self) -> Option<Value> {
        (!self.entries.is_empty()).then(|| Value::Array(self.entries.clone()))
    }
}

/// The members of a detail whose fragments concatenate across chunks.
///
/// Every other member is written once, by the fragment that first carried it.
const DETAIL_TEXT_MEMBERS: &[&str] = &["text", "summary", "data"];

/// Whether `fragment` continues the logical detail already held in `entry`.
fn continues_detail(entry: &Value, fragment: &Value) -> bool {
    let (Some(entry_type), Some(fragment_type)) = (
        entry.get("type").and_then(Value::as_str),
        fragment.get("type").and_then(Value::as_str),
    ) else {
        return false;
    };
    if entry_type != fragment_type {
        return false;
    }

    match (
        entry.get("index").and_then(Value::as_u64),
        fragment.get("index").and_then(Value::as_u64),
    ) {
        (Some(entry_index), Some(fragment_index)) => entry_index == fragment_index,
        _ => true,
    }
}

/// Appends one fragment's text onto `entry` and fills in members it lacks.
fn merge_detail(entry: &mut Value, fragment: &Value) {
    let (Some(members), Some(fragment)) = (entry.as_object_mut(), fragment.as_object()) else {
        return;
    };
    for (key, value) in fragment {
        match members.get_mut(key) {
            Some(Value::String(text)) if DETAIL_TEXT_MEMBERS.contains(&key.as_str()) => {
                if let Some(fragment) = value.as_str() {
                    text.push_str(fragment);
                }
            }
            Some(_) => {}
            None => {
                members.insert(key.clone(), value.clone());
            }
        }
    }
}

/// The structured reasoning channel of a complete response message.
///
/// Providers document an array of detail objects; a lone object is accepted as
/// a single entry. The entries are preserved exactly as they arrived — order,
/// count, and shape — because only the model that wrote them can read the
/// encrypted members. Only a stream coalesces, and only because it must undo
/// its own fragmenting; a complete payload is already whole, and merging two
/// same-type entries here would discard the second one's signature.
pub(super) fn complete_details(message: &Value) -> Option<ContentPart> {
    let entries: Vec<Value> = match message.get(REASONING_DETAILS)? {
        Value::Array(entries) => entries
            .iter()
            .filter(|entry| entry.is_object())
            .cloned()
            .collect(),
        payload @ Value::Object(_) => vec![payload.clone()],
        _ => Vec::new(),
    };
    (!entries.is_empty()).then(|| ContentPart::opaque(details_kind(), Value::Array(entries)))
}
