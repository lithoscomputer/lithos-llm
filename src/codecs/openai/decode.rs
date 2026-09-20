//! Response document decoding: output items, tool-call identity, and
//! usage.

use std::collections::BTreeMap;

use serde_json::{Value, json};

use super::{MESSAGE_KIND, NAMESPACE, REASONING_KIND};
use crate::codecs::errors::{malformed_success, refusal};
use crate::resolver::ResolvedRoute;
use crate::types::{
    ContentBlockKind, ContentPart, Error, FinishReason, ReasoningContent, Response, TokenCounts,
    ToolCall, ToolCallKind, ToolInput,
};

/// Decodes one complete response document.
///
/// The typed view is read from the same value that is then moved into
/// [`Response::raw`], so the response always carries the provider's own body.
pub(super) fn decode_document(route: &ResolvedRoute, value: Value) -> Result<Response, Error> {
    if !value.is_object() {
        return Err(decode_error(
            route,
            "returned a response body that is not a JSON object",
            value,
        ));
    }
    // A body without an `output` array is not a Responses document. Decoding
    // `{}` or an error-shaped 200 from a broken gateway as a successful empty
    // response would hand the caller an answer the model never wrote.
    if !value.get("output").is_some_and(Value::is_array) {
        return Err(decode_error(
            route,
            "returned a 200 body without a Responses output array",
            value,
        ));
    }
    // A refusal is a failure, not a short answer — the same contract every
    // other codec applies. This protocol reports it as a `refusal` content
    // part inside a message item.
    if let Some(text) = refusal_text(&value) {
        let explanation = text.to_owned();
        return Err(refusal(route, Some(&explanation), Some(value)));
    }

    let content = decode_output(&value);
    let mut response = Response::new(
        route.provider().id().clone(),
        route.model().id().clone(),
        content,
    );
    response.id = value
        .get("id")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    response.finish_reason = decode_finish_reason(&value, &response.content);
    response.usage = token_counts(value.get("usage"));
    // This protocol reports no in-band cost; the adapter prices the call from
    // the catalog instead.
    response.cost = None;
    response.raw = Some(value);
    response.suppress_unfinished_tool_calls();
    Ok(response)
}

/// Decodes the `output` array of a complete response document.
///
/// A `message` item contributes both its visible text and the whole item as an
/// [`MESSAGE_KIND`] opaque part, because only the original item replays.
///
/// A tool call with no name is a model-internal item rather than a call the
/// caller can answer. It is dropped, which also keeps it from turning the
/// finish reason into [`FinishReason::ToolCall`].
fn decode_output(value: &Value) -> Vec<ContentPart> {
    let mut content = Vec::new();
    let items = value
        .get("output")
        .and_then(Value::as_array)
        .into_iter()
        .flatten();

    for item in items {
        match ItemKind::of(item) {
            ItemKind::Message => {
                let text = message_text(item);
                if !text.is_empty() {
                    content.push(ContentPart::Text { text });
                }
                content.push(ContentPart::opaque(MESSAGE_KIND, item.clone()));
            }
            ItemKind::ToolCall(kind) => {
                let identity = CallIdentity::of(item, kind);
                if !identity.is_internal() {
                    content.push(ContentPart::ToolCall(identity.into_call()));
                }
            }
            ItemKind::Reasoning => content.extend(decode_reasoning(item)),
            // Provider-side items such as `web_search_call` carry no portable
            // content. They stay available in `Response::raw`.
            ItemKind::Other => {}
        }
    }

    content
}

/// The kind of one output item, read from its `type`.
///
/// Blocking and streaming decode both dispatch on this, so a new item type is
/// one edit here rather than one per match on the wire string.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ItemKind {
    Message,
    Reasoning,
    ToolCall(ToolCallKind),
    /// A provider-side item such as `web_search_call`, which carries no
    /// portable content.
    Other,
}

impl ItemKind {
    pub(super) fn of(item: &Value) -> Self {
        match item.get("type").and_then(Value::as_str) {
            Some("message") => Self::Message,
            Some("reasoning") => Self::Reasoning,
            Some("function_call") => Self::ToolCall(ToolCallKind::Function),
            Some("custom_tool_call") => Self::ToolCall(ToolCallKind::Custom),
            _ => Self::Other,
        }
    }
}

impl ToolCallKind {
    /// The item field a call of this kind carries its input in.
    pub(super) fn responses_input_key(self) -> &'static str {
        match self {
            Self::Function => "arguments",
            Self::Custom => "input",
        }
    }
}

/// The identity of one tool-call item, read once and agreed on by both decode
/// paths.
///
/// `call_id` is the identity a tool result answers, so it is the canonical
/// id. The item id is a second identifier the protocol needs when the call is
/// replayed, so it is kept in this codec's provider metadata namespace when
/// the two differ.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CallIdentity<'a> {
    call_id: Option<&'a str>,
    item_id: Option<&'a str>,
    name:    &'a str,
    kind:    ToolCallKind,
    input:   &'a str,
}

impl<'a> CallIdentity<'a> {
    pub(super) fn of(item: &'a Value, kind: ToolCallKind) -> Self {
        let text = |key: &str| item.get(key).and_then(Value::as_str);
        Self {
            call_id: text("call_id"),
            item_id: text("id"),
            name: text("name").unwrap_or_default(),
            kind,
            input: text(kind.responses_input_key()).unwrap_or_default(),
        }
    }

    /// The id a tool result answers: the call id, or the item id when the
    /// item carries no separate call id.
    pub(super) fn canonical_id(&self) -> &'a str {
        self.call_id.or(self.item_id).unwrap_or_default()
    }

    pub(super) fn name(&self) -> Option<&'a str> {
        (!self.name.is_empty()).then_some(self.name)
    }

    /// The call's input text, in the field its kind names.
    pub(super) fn input(&self) -> &'a str {
        self.input
    }

    /// Whether this is a tool call the caller cannot answer.
    ///
    /// A call with no name is model-internal. It has no tool to route to and
    /// no result to send back, so it neither becomes content nor turns the
    /// finish reason into [`FinishReason::ToolCall`].
    pub(super) fn is_internal(&self) -> bool {
        self.name.is_empty()
    }

    /// The provider metadata that lets the call be replayed, when the item
    /// id differs from the call id.
    pub(super) fn replay_metadata(&self) -> Option<Value> {
        match (self.call_id, self.item_id) {
            (Some(call_id), Some(item_id)) if call_id != item_id => {
                Some(json!({ "item_id": item_id }))
            }
            _ => None,
        }
    }

    /// The identity as the block kind a stream opens for it.
    pub(super) fn block_kind(&self) -> ContentBlockKind {
        ContentBlockKind::ToolCall {
            id:   self.canonical_id().to_owned(),
            name: self.name().map(ToOwned::to_owned),
            kind: self.kind,
        }
    }

    /// The complete tool call, for blocking decode.
    pub(super) fn into_call(self) -> ToolCall {
        let mut provider_metadata = BTreeMap::new();
        if let Some(metadata) = self.replay_metadata() {
            provider_metadata.insert(NAMESPACE.to_owned(), metadata);
        }
        ToolCall {
            id: self.canonical_id().to_owned(),
            name: self.name.to_owned(),
            input: ToolInput::from_wire(self.kind, self.input.to_owned()),
            provider_metadata,
        }
    }
}

/// Whether an output item is a tool call the caller cannot answer; see
/// [`CallIdentity::is_internal`].
pub(super) fn is_internal_call(item: &Value) -> bool {
    match ItemKind::of(item) {
        ItemKind::ToolCall(kind) => CallIdentity::of(item, kind).is_internal(),
        ItemKind::Message | ItemKind::Reasoning | ItemKind::Other => false,
    }
}

/// The refusal text carried by a document's message items, if any.
///
/// A non-empty `refusal` content part means the model declined instead of
/// answering; an empty one is treated as absent, as the Chat codec treats an
/// empty `refusal` field.
pub(super) fn refusal_text(value: &Value) -> Option<&str> {
    value
        .get("output")?
        .as_array()?
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("message"))
        .filter_map(|item| item.get("content").and_then(Value::as_array))
        .flatten()
        .filter(|part| part.get("type").and_then(Value::as_str) == Some("refusal"))
        .find_map(|part| part.get("refusal").and_then(Value::as_str))
        .filter(|text| !text.is_empty())
}

/// The visible text of one `message` output item.
pub(super) fn message_text(item: &Value) -> String {
    item.get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|part| part.get("type").and_then(Value::as_str) == Some("output_text"))
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect()
}

/// Decodes one `reasoning` output item into the parts it contributes.
///
/// The item's reasoning text — see [`reasoning_text`] — becomes reasoning
/// content, and the whole item is kept verbatim, summary-only items included.
/// Replaying a turn's function calls without their preceding reasoning item
/// draws a provider 400 when the conversation is not stored, and only the
/// original item, its id included, satisfies that pairing.
fn decode_reasoning(item: &Value) -> Vec<ContentPart> {
    let mut parts = Vec::new();
    if let Some(text) = reasoning_text(item) {
        parts.push(ContentPart::Reasoning(ReasoningContent {
            text,
            signature: None,
            signature_origin: None,
            redacted: false,
        }));
    }
    parts.push(ContentPart::opaque(REASONING_KIND, item.clone()));
    parts
}

/// The reasoning text of one `reasoning` output item, or `None` when it
/// carries none.
///
/// The `reasoning_text` entries of `content` are the trace itself and win
/// when present. Otherwise the `summary_text` blocks stand in, so a hosted
/// model that only ever shows its summary still yields readable reasoning.
/// Either list is joined by a blank line — the separator consumers put
/// between summary blocks when they read them off the opaque item, so a
/// consumer comparing the two sees the same text rather than a run-together
/// copy. Entries of any other type are not text and are skipped.
pub(super) fn reasoning_text(item: &Value) -> Option<String> {
    let joined = |list: &str, kind: &str| -> String {
        item.get(list)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|entry| entry.get("type").and_then(Value::as_str) == Some(kind))
            .filter_map(|entry| entry.get("text").and_then(Value::as_str))
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n")
    };
    [("content", "reasoning_text"), ("summary", "summary_text")]
        .into_iter()
        .map(|(list, kind)| joined(list, kind))
        .find(|text| !text.is_empty())
}

/// Normalizes the response status into a finish reason.
///
/// The protocol has no separate stop reason. A completed response that
/// requested tools reports [`FinishReason::ToolCall`] so consumers can branch
/// on the same value every other codec produces.
pub(super) fn decode_finish_reason(value: &Value, content: &[ContentPart]) -> FinishReason {
    match value.get("status").and_then(Value::as_str) {
        Some("incomplete") => {
            match value
                .pointer("/incomplete_details/reason")
                .and_then(Value::as_str)
            {
                Some("content_filter") => FinishReason::ContentFilter,
                _ => FinishReason::Length,
            }
        }
        Some("failed") => FinishReason::Error,
        // A status this crate does not name — `cancelled`, say — keeps the
        // provider's own spelling instead of collapsing into `Stop`.
        Some(other) if other != "completed" => FinishReason::Other(other.to_owned()),
        _ if content
            .iter()
            .any(|part| matches!(part, ContentPart::ToolCall(_))) =>
        {
            FinishReason::ToolCall
        }
        _ => FinishReason::Stop,
    }
}

/// Normalizes the inclusive usage counters into disjoint buckets.
///
/// `input_tokens` includes the cached and cache-written tokens and
/// `output_tokens` includes the reasoning tokens. The GPT-5.6 family bills
/// cache writes at a premium and reports them as
/// `input_tokens_details.cache_write_tokens` (observed live on 2026-08-30);
/// models that bill no write omit the counter or send zero, which decodes to
/// an empty bucket either way.
pub(super) fn token_counts(usage: Option<&Value>) -> TokenCounts {
    let Some(usage) = usage else {
        return TokenCounts::default();
    };
    let count = |pointer: &str| usage.pointer(pointer).and_then(Value::as_u64).unwrap_or(0);

    TokenCounts::from_inclusive(
        count("/input_tokens"),
        count("/output_tokens"),
        count("/output_tokens_details/reasoning_tokens"),
        count("/input_tokens_details/cached_tokens"),
        count("/input_tokens_details/cache_write_tokens"),
    )
}

/// The error a malformed success body produces; see `malformed_success`.
pub(super) fn decode_error(route: &ResolvedRoute, detail: impl Into<String>, raw: Value) -> Error {
    malformed_success(
        route,
        format!("provider {} {}", route.provider().id(), detail.into()),
        Some(raw),
    )
}
