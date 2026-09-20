//! The Chat Completions SSE stream decoder.

use std::collections::BTreeMap;

use serde_json::Value;

use super::decode::{message_text, non_empty, provider_cost, reasoning_text, token_counts};
use super::reasoning_details::ReasoningDetails;
use super::{
    REASONING_BLOCK, REASONING_DETAILS, REASONING_DETAILS_BLOCK, REASONING_DETAILS_KIND, TEXT_BLOCK,
};
use crate::codecs::StreamDecoder;
use crate::codecs::assembler::StreamAssembler;
use crate::codecs::content::{finish_reason, promote_tool_finish};
use crate::codecs::errors::{invalid_stream_event, malformed_stream, refusal};
use crate::resolver::ResolvedRoute;
use crate::transport::{SseEvent, provider_error};
use crate::types::{ContentBlockId, ContentBlockKind, Error, StreamEvent, ToolCallKind};

/// The per-stream state for one Chat Completions response.
///
/// The protocol supplies no block ids and no terminal response document, so
/// this decoder synthesizes the ids and lets the assembler build the completed
/// response from what the chunks carried.
pub(super) struct ChatStreamDecoder {
    assembler: StreamAssembler,
    route:     ResolvedRoute,
    /// The three synthesized block ids of this stream.
    blocks:    SyntheticBlocks,
    /// Whether the `Started` event has been emitted for this stream.
    started:   bool,
    /// The structured reasoning channel, coalesced as its fragments arrive.
    details:   ReasoningDetails,
    /// The identity fragments have carried for each opened tool-call slot.
    slots:     BTreeMap<u64, SlotIdentity>,
    /// Refusal text accumulated across chunks; a non-empty value fails the
    /// stream when it ends.
    refusal:   String,
}

impl StreamDecoder for ChatStreamDecoder {
    fn decode(&mut self, event: SseEvent) -> Result<Vec<StreamEvent>, Error> {
        let chunk = self.error_check(&event)?;
        let mut events = self.start(&chunk);
        let delta = chunk.pointer("/choices/0/delta").unwrap_or(&Value::Null);
        events.extend(self.delta_events(delta)?);
        events.extend(self.chunk_totals(&chunk));
        Ok(events)
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>, Error> {
        if !self.refusal.is_empty() {
            return Err(refusal(&self.route, Some(&self.refusal), None));
        }
        // Opening a tool block does not mean its arguments finished. Without
        // a finish reason, EOF may have cut the call at any fragment, including
        // before its first argument. Never promote that prefix into a call.
        if self.assembler.has_tool_call() && self.assembler.finish_reason().is_none() {
            return Err(malformed_stream(
                &self.route,
                "the tool call stream ended without a finish reason",
                None,
            ));
        }
        // Some skins finish tool calls with `stop` instead of `tool_calls`.
        if let Some(reason) = self.assembler.finish_reason().cloned() {
            self.assembler
                .set_finish_reason(promote_tool_finish(reason, self.assembler.has_tool_call()));
        }
        Ok(self.assembler.complete())
    }
}

impl ChatStreamDecoder {
    /// Creates the decoder for one stream on `route`.
    pub(super) fn new(route: &ResolvedRoute) -> Self {
        Self {
            assembler: StreamAssembler::new(route),
            route:     route.clone(),
            blocks:    SyntheticBlocks::default(),
            started:   false,
            details:   ReasoningDetails::default(),
            slots:     BTreeMap::new(),
            refusal:   String::new(),
        }
    }

    /// Parses one chunk and fails the stream on an error payload.
    ///
    /// A chunk that is not JSON is indistinguishable from mid-stream
    /// corruption, so the failure is retryable like any other garbled stream.
    /// An error payload ends the stream: the same classifier runs here and on
    /// the HTTP error path, so one provider code means one thing. An explicit
    /// `"error": null` member is not an error — a skin spelling out the field
    /// on success chunks must not fail every stream.
    fn error_check(&self, event: &SseEvent) -> Result<Value, Error> {
        let chunk: Value = serde_json::from_str(&event.data).map_err(|source| {
            invalid_stream_event(
                &self.route,
                format!(
                    "provider {} returned an invalid stream chunk",
                    self.route.provider().id()
                ),
                source,
            )
        })?;
        if chunk.get("error").is_some_and(|error| !error.is_null()) {
            return Err(provider_error(
                self.route.provider(),
                None,
                Some(chunk),
                None,
            ));
        }
        Ok(chunk)
    }

    /// Emits the `Started` event on the first chunk and records the id.
    fn start(&mut self, chunk: &Value) -> Vec<StreamEvent> {
        let id = chunk
            .get("id")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let mut events = Vec::new();
        if !self.started {
            self.started = true;
            events.push(self.assembler.started(id.clone()));
        }
        if let Some(id) = id {
            self.assembler.set_id(id);
        }
        events
    }

    /// Translates the first choice's delta into content events.
    fn delta_events(&mut self, delta: &Value) -> Result<Vec<StreamEvent>, Error> {
        let mut events = Vec::new();
        // The structured channel is coalesced across chunks, so the block
        // carries the whole array every time a fragment lands on it.
        if let Some(payload) = delta.get(REASONING_DETAILS) {
            self.details.absorb(payload);
            if let Some(entries) = self.details.entries() {
                let block = &self.blocks.details;
                events.extend(
                    self.assembler
                        .start(block.clone(), ContentBlockKind::Opaque {
                            kind: REASONING_DETAILS_KIND.to_owned(),
                        }),
                );
                events.extend(self.assembler.set_opaque_data(block, entries));
            }
        }
        // `reasoning_text` reads both spellings of the readable channel.
        if let Some(text) = reasoning_text(delta) {
            events.extend(self.assembler.reasoning(&self.blocks.reasoning, &text));
        }
        // `message_text` reads both wire shapes: a delta may carry the
        // part-array form just as a blocking message may, and dropping it
        // would complete the stream as an empty success.
        if let Some(text) = message_text(delta) {
            events.extend(self.assembler.text(&self.blocks.text, &text));
        }
        // Refusal fragments accumulate silently; the whole explanation fails
        // the stream once it ends, matching the blocking decoder's contract.
        if let Some(text) = non_empty(delta, "refusal") {
            self.refusal.push_str(text);
        }
        for call in delta
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            events.extend(self.decode_tool_call_delta(call)?);
        }
        Ok(events)
    }

    /// Records the chunk-level totals: finish reason, cost, and usage.
    ///
    /// A later chunk's cost replaces an earlier one; usage arrives in a final
    /// chunk whose `choices` array is empty.
    fn chunk_totals(&mut self, chunk: &Value) -> Vec<StreamEvent> {
        if let Some(reason) = chunk
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str)
        {
            self.assembler
                .set_finish_reason(finish_reason(Some(reason)));
        }
        if let Some(cost) = provider_cost(chunk) {
            self.assembler.set_cost(cost);
        }
        let mut events = Vec::new();
        if let Some(usage) = chunk.get("usage").filter(|usage| usage.is_object()) {
            events.push(self.assembler.usage(token_counts(usage)));
        }
        events
    }

    /// Applies one `tool_calls` fragment and returns the events it produced.
    ///
    /// The fragment's `index` is the provider's accumulation slot, so it names
    /// the content block. The provider's own tool-call id and the tool name
    /// normally arrive on the first fragment for a slot — but a skin may split
    /// them across fragments, or stream argument text before any identity at
    /// all, so any fragment opens its slot and identity arriving late repairs
    /// the open block instead of being discarded. Identity that never arrives
    /// leaves the fallback the block opened with — the synthesized block id as
    /// the call id and an empty name — which is how the reference decoder
    /// assembled an identity-less call.
    ///
    /// # Errors
    ///
    /// A fragment without a valid `index` fails the stream: defaulting the
    /// slot would silently merge parallel calls into one garbled call, where
    /// the reference decoder failed the chunk. The failure is retryable, like
    /// any other garbled stream.
    fn decode_tool_call_delta(&mut self, call: &Value) -> Result<Vec<StreamEvent>, Error> {
        let Some(index) = call.get("index").and_then(Value::as_u64) else {
            return Err(malformed_stream(
                &self.route,
                format!(
                    "provider {} streamed a tool-call fragment without an index",
                    self.route.provider().id()
                ),
                Some(call.clone()),
            ));
        };
        let block = ContentBlockId::new(format!("tool-{index}"));

        let id = call.get("id").and_then(Value::as_str);
        let name = call.pointer("/function/name").and_then(Value::as_str);
        let slot = self.slots.entry(index).or_default();
        if let Some(id) = id {
            slot.id = Some(id.to_owned());
        }
        if let Some(name) = name {
            slot.name = Some(name.to_owned());
        }
        let identity = ContentBlockKind::ToolCall {
            id:   slot.id.clone().unwrap_or_else(|| block.as_str().to_owned()),
            name: slot.name.clone(),
            kind: ToolCallKind::Function,
        };
        let mut events = self.assembler.start(block.clone(), identity.clone());
        self.assembler.repair_tool_identity(&block, identity);
        if let Some(fragment) = call.pointer("/function/arguments").and_then(Value::as_str)
            && !fragment.is_empty()
        {
            events.extend(self.assembler.arguments(&block, fragment));
        }
        Ok(events)
    }
}

/// The block ids this protocol never supplies, synthesized once per stream.
///
/// Every text fragment of one response belongs to the same block, and so does
/// every reasoning fragment and every `reasoning_details` fragment.
struct SyntheticBlocks {
    text:      ContentBlockId,
    reasoning: ContentBlockId,
    details:   ContentBlockId,
}

impl Default for SyntheticBlocks {
    fn default() -> Self {
        Self {
            text:      ContentBlockId::new(TEXT_BLOCK),
            reasoning: ContentBlockId::new(REASONING_BLOCK),
            details:   ContentBlockId::new(REASONING_DETAILS_BLOCK),
        }
    }
}

/// The tool-call identity accumulated for one streaming slot.
///
/// A skin may split the call id and the tool name across fragments; each
/// member keeps the last value a fragment carried, which is how the reference
/// decoder read them.
#[derive(Default)]
struct SlotIdentity {
    id:   Option<String>,
    name: Option<String>,
}
