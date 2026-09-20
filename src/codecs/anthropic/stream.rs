//! The Messages SSE stream decoder.

use std::collections::BTreeMap;

use serde_json::Value;

use super::NAMESPACE;
use super::decode::{block_id, field, fold_usage};
use crate::codecs::StreamDecoder;
use crate::codecs::assembler::StreamAssembler;
use crate::codecs::content::{ANTHROPIC_SIGNATURES, finish_reason};
use crate::codecs::errors::{invalid_stream_event, lost_tool_start, refusal};
use crate::resolver::ResolvedRoute;
use crate::transport::{SseEvent, provider_error};
use crate::types::{ContentBlockId, ContentBlockKind, Error, StreamEvent, ToolCallKind};

/// Per-stream state for one Anthropic Messages response.
pub(super) struct AnthropicStreamDecoder {
    route:     ResolvedRoute,
    assembler: StreamAssembler,
    /// Unknown server-side blocks that may stream their `input`.
    ///
    /// A `server_tool_use` block arrives as a start snapshot with an empty
    /// `input` and streams the real value through `input_json_delta`. The
    /// fragments accumulate here next to the snapshot and fold back into it
    /// when the block closes, so the replayed block matches what the blocking
    /// decoder keeps whole.
    opaque:    BTreeMap<ContentBlockId, (Value, String)>,
}

impl StreamDecoder for AnthropicStreamDecoder {
    fn decode(&mut self, event: SseEvent) -> Result<Vec<StreamEvent>, Error> {
        // An event that is not JSON is indistinguishable from mid-stream
        // corruption, so the failure is retryable like any other garbled
        // stream.
        let value: Value = serde_json::from_str(&event.data).map_err(|source| {
            invalid_stream_event(
                &self.route,
                "Anthropic returned an invalid stream event",
                source,
            )
        })?;
        let kind = value
            .get("type")
            .and_then(Value::as_str)
            .or(event.event.as_deref());

        let mut events = Vec::new();
        match kind {
            // Usage is split across two events: the input and cache counters
            // arrive here and the output counter arrives on `message_delta`.
            // Both fold into one cumulative snapshot.
            Some("message_start") => {
                let id = value
                    .pointer("/message/id")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                events.push(self.assembler.started(id));
                if let Some(usage) = value.pointer("/message/usage") {
                    events.push(
                        self.assembler
                            .merge_usage(|counts| fold_usage(counts, usage)),
                    );
                }
            }
            Some("content_block_start") => events.extend(self.start_block(&value)),
            Some("content_block_delta") => events.extend(self.block_delta(&value)?),
            Some("content_block_stop") => {
                let id = block_id(&value);
                // No documented event carries a signature here, but the
                // reference decoder preferred one arriving on the stop event
                // over the captured value, and a dialect that sends it only
                // here would otherwise close the block unsigned — a reasoning
                // part Anthropic rejects on replay.
                if let Some(signature) = value
                    .pointer("/content_block/signature")
                    .and_then(Value::as_str)
                {
                    events.extend(self.assembler.signature(&id, signature));
                }
                events.extend(self.flush_opaque(&id));
                events.extend(self.assembler.end(&id));
            }
            Some("message_delta") => {
                if let Some(reason) = value.pointer("/delta/stop_reason").and_then(Value::as_str) {
                    // A refusal fails the stream here, before `message_stop`
                    // can complete it as a success.
                    if reason == "refusal" {
                        let explanation = value
                            .pointer("/delta/stop_details/explanation")
                            .and_then(Value::as_str)
                            .map(ToOwned::to_owned);
                        return Err(refusal(&self.route, explanation.as_deref(), Some(value)));
                    }
                    self.assembler
                        .set_finish_reason(finish_reason(Some(reason)));
                }
                if let Some(usage) = value.get("usage") {
                    events.push(
                        self.assembler
                            .merge_usage(|counts| fold_usage(counts, usage)),
                    );
                }
            }
            // Anthropic sends no terminal response document, so `raw` stays
            // empty rather than being synthesized from accumulated state.
            Some("message_stop") => events.extend(self.assembler.complete()),
            Some("error") => {
                return Err(provider_error(
                    self.route.provider(),
                    None,
                    Some(value),
                    None,
                ));
            }
            // `ping` is a keep-alive; no other event type carries content.
            _ => {}
        }

        Ok(events)
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>, Error> {
        // A stream that ended without `message_stop` was truncated. Completing
        // here keeps the one-`Ended`-per-successful-stream contract; a
        // stream that already saw `message_stop` gets nothing, because
        // completing is idempotent.
        let ids: Vec<ContentBlockId> = self.opaque.keys().cloned().collect();
        let mut events = Vec::new();
        for id in &ids {
            events.extend(self.flush_opaque(id));
        }
        events.extend(self.assembler.complete());
        Ok(events)
    }
}

impl AnthropicStreamDecoder {
    /// Creates the decoder for one stream on `route`.
    pub(super) fn new(route: &ResolvedRoute) -> Self {
        Self {
            route:     route.clone(),
            assembler: StreamAssembler::new(route).with_signatures(ANTHROPIC_SIGNATURES),
            opaque:    BTreeMap::new(),
        }
    }

    /// Opens the block a `content_block_start` event announces.
    fn start_block(&mut self, value: &Value) -> Vec<StreamEvent> {
        let id = block_id(value);
        let Some(block) = value.get("content_block") else {
            return Vec::new();
        };

        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                let mut events = self.assembler.start(id.clone(), ContentBlockKind::Text);
                let seed = field(block, "text");
                if !seed.is_empty() {
                    events.extend(self.assembler.text(&id, seed));
                }
                events
            }
            Some("thinking") => {
                let mut events = self
                    .assembler
                    .start(id.clone(), ContentBlockKind::Reasoning);
                if let Some(signature) = block.get("signature").and_then(Value::as_str) {
                    events.extend(self.assembler.signature(&id, signature));
                }
                let seed = field(block, "thinking");
                if !seed.is_empty() {
                    events.extend(self.assembler.reasoning(&id, seed));
                }
                events
            }
            // The reference drops this block, which loses the redacted payload
            // the blocking decoder keeps. The blob arrives whole on the start
            // event; no delta follows it.
            Some("redacted_thinking") => {
                let mut events = self
                    .assembler
                    .start(id.clone(), ContentBlockKind::Reasoning);
                events.extend(self.assembler.set_redacted(&id));
                let data = field(block, "data");
                if !data.is_empty() {
                    events.extend(self.assembler.reasoning(&id, data));
                }
                events
            }
            // The provider's tool-call id belongs to the call, not to the
            // block, so it stays here and never becomes the block id.
            Some("tool_use") => self.assembler.start(id, ContentBlockKind::ToolCall {
                id:   field(block, "id").to_owned(),
                name: block
                    .get("name")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                kind: ToolCallKind::Function,
            }),
            Some(kind) => {
                let mut events = self.assembler.start(id.clone(), ContentBlockKind::Opaque {
                    kind: format!("{NAMESPACE}.{kind}"),
                });
                events.extend(self.assembler.set_opaque_data(&id, block.clone()));
                self.opaque.insert(id, (block.clone(), String::new()));
                events
            }
            None => Vec::new(),
        }
    }

    /// Applies one `content_block_delta` event.
    fn block_delta(&mut self, value: &Value) -> Result<Vec<StreamEvent>, Error> {
        let id = block_id(value);
        let Some(delta) = value.get("delta") else {
            return Ok(Vec::new());
        };

        Ok(match delta.get("type").and_then(Value::as_str) {
            Some("text_delta") => self.assembler.text(&id, field(delta, "text")),
            Some("thinking_delta") => self.assembler.reasoning(&id, field(delta, "thinking")),
            // A signature is block state, not visible output, so it produces no
            // event of its own.
            Some("signature_delta") => self.assembler.signature(&id, field(delta, "signature")),
            Some("input_json_delta") => {
                let fragment = field(delta, "partial_json");
                // A server-side block streams its `input` the same way a tool
                // call does; the fragments belong to the snapshot, not to a
                // tool-call buffer.
                if let Some((_, buffer)) = self.opaque.get_mut(&id) {
                    buffer.push_str(fragment);
                    return Ok(Vec::new());
                }
                // Anthropic announces every tool call in a
                // `content_block_start`; see `lost_tool_start`.
                if !self.assembler.announced_tool_call(&id) {
                    return Err(lost_tool_start(&self.route));
                }
                self.assembler.arguments(&id, fragment)
            }
            _ => Vec::new(),
        })
    }

    /// Folds a server-side block's streamed input back into its snapshot.
    ///
    /// Runs when the block closes, and again at end of stream for a block a
    /// truncated stream never closed. Fragments that do not parse leave the
    /// start snapshot untouched, which is the pre-accumulation behavior.
    fn flush_opaque(&mut self, id: &ContentBlockId) -> Vec<StreamEvent> {
        let Some((mut payload, buffer)) = self.opaque.remove(id) else {
            return Vec::new();
        };
        if buffer.is_empty() {
            return Vec::new();
        }
        let Ok(input) = serde_json::from_str::<Value>(&buffer) else {
            return Vec::new();
        };
        if let Some(object) = payload.as_object_mut() {
            object.insert("input".to_owned(), input);
        }
        self.assembler.set_opaque_data(id, payload)
    }
}
