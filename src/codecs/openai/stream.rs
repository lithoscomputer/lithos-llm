//! The Responses SSE stream decoder.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Value, json};

use super::decode::{
    decode_document, decode_finish_reason, is_internal_call, message_text, reasoning_text,
    refusal_text, token_counts,
};
use super::{MESSAGE_KIND, NAMESPACE, REASONING_KIND};
use crate::codecs::StreamDecoder;
use crate::codecs::assembler::StreamAssembler;
use crate::codecs::content::promote_tool_finish;
use crate::codecs::errors::refusal;
use crate::resolver::ResolvedRoute;
use crate::transport::{SseEvent, provider_error};
use crate::types::{ContentBlockId, ContentBlockKind, Error, StreamEvent, ToolCallKind};

/// Decodes one `/v1/responses` stream.
pub(super) struct ResponsesStream {
    assembler:         StreamAssembler,
    route:             ResolvedRoute,
    /// Blocks that already received content, so a terminal reasoning item
    /// knows whether visible text streamed and its opaque replay block needs
    /// a derived id.
    delivered:         BTreeSet<ContentBlockId>,
    /// The entry — its index field and index — of the last reasoning fragment
    /// each reasoning block received, so a fragment that opens the next entry
    /// can be separated from the previous one the way the terminal item is.
    reasoning_entries: BTreeMap<ContentBlockId, (&'static str, u64)>,
    /// Blocks for model-internal items, whose deltas open no block at all.
    skipped:           BTreeSet<ContentBlockId>,
    /// Whether the `Started` event has been emitted for this stream.
    started:           bool,
}

impl StreamDecoder for ResponsesStream {
    fn decode(&mut self, event: SseEvent) -> Result<Vec<StreamEvent>, Error> {
        let data = event.data.trim();
        if data.is_empty() || data == "[DONE]" {
            return Ok(Vec::new());
        }

        // A frame this decoder cannot parse is skipped rather than fatal.
        // Proxies inject their own keepalive payloads, and killing a
        // generation over a frame that carries no model output would trade a
        // whole response for a comment.
        let Ok(value) = serde_json::from_str::<Value>(data) else {
            return Ok(Vec::new());
        };

        let kind = value
            .get("type")
            .and_then(Value::as_str)
            .or(event.event.as_deref())
            .unwrap_or_default();

        // `response.created` is where the response id arrives, but a proxy can
        // drop it. Latching on the first event that is not a failure keeps a
        // consumer from seeing deltas before the stream ever started, which is
        // what the Chat dialect does with its first chunk.
        let mut latched = Vec::new();
        if !self.started && !matches!(kind, "error" | "response.failed") {
            self.started = true;
            let id = value
                .pointer("/response/id")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
            latched.push(self.assembler.started(id));
        }
        let events = self.decode_event(kind, &value)?;
        latched.extend(events);
        Ok(latched)
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>, Error> {
        Ok(self.assembler.complete())
    }
}

impl ResponsesStream {
    /// Creates the decoder for one stream on `route`.
    pub(super) fn new(route: &ResolvedRoute) -> Self {
        Self {
            assembler:         StreamAssembler::new(route),
            route:             route.clone(),
            delivered:         BTreeSet::new(),
            reasoning_entries: BTreeMap::new(),
            skipped:           BTreeSet::new(),
            started:           false,
        }
    }

    /// Decodes one recognized stream event.
    fn decode_event(&mut self, kind: &str, value: &Value) -> Result<Vec<StreamEvent>, Error> {
        match kind {
            "error" => Err(provider_error(
                self.route.provider(),
                None,
                Some(value.clone()),
                None,
            )),
            // Some gateways flatten the failure to `{"type":"response.failed",
            // "error":{...}}`. Falling back to the whole event keeps the
            // provider's code and message when the `response` wrapper is gone.
            "response.failed" => Err(provider_error(
                self.route.provider(),
                None,
                Some(value.get("response").unwrap_or(value).clone()),
                None,
            )),
            "response.output_item.added" => Ok(self.start_item(&block_id(value), item(value))),
            "response.output_text.delta" => Ok(self.text_delta(value)),
            "response.reasoning_text.delta" => Ok(self.reasoning_delta(value, "content_index")),
            "response.reasoning_summary_text.delta" => {
                Ok(self.reasoning_delta(value, "summary_index"))
            }
            "response.function_call_arguments.delta" | "response.custom_tool_call_input.delta" => {
                Ok(self.arguments_delta(value))
            }
            "response.output_item.done" => Ok(self.end_item(&block_id(value), item(value))),
            "response.completed" | "response.incomplete" => {
                // A refusal fails the stream instead of completing it as an
                // empty answer — the same contract every other codec applies.
                let document = value.get("response").unwrap_or(value);
                if let Some(text) = refusal_text(document) {
                    return Err(refusal(&self.route, Some(text), Some(document.clone())));
                }
                Ok(self.complete(value))
            }
            // `response.created` lands here: its id already rode the latched
            // `Started` event, so it contributes nothing of its own.
            _ => Ok(Vec::new()),
        }
    }

    /// Opens the block for one output item.
    ///
    /// Opening is idempotent, so the terminal event for an item the provider
    /// never announced still opens the right kind of block. A reasoning item
    /// is deliberately not opened here: whether it becomes visible reasoning
    /// or an opaque replay item is only known once the item is done. A
    /// message item is not opened here either — its text block latches on the
    /// first text that actually arrives, so a message with none (a
    /// refusal-only message, an empty assistant turn) never emits the empty
    /// `Text` part the blocking decode of the same body omits.
    fn start_item(&mut self, id: &ContentBlockId, item: &Value) -> Vec<StreamEvent> {
        if is_internal_call(item) {
            self.skipped.insert(id.clone());
            return Vec::new();
        }

        match item.get("type").and_then(Value::as_str) {
            Some(kind @ ("function_call" | "custom_tool_call")) => {
                let call_kind = match kind {
                    "custom_tool_call" => ToolCallKind::Custom,
                    _ => ToolCallKind::Function,
                };
                let call_id = item.get("call_id").and_then(Value::as_str);
                let item_id = item.get("id").and_then(Value::as_str);
                let identity = ContentBlockKind::ToolCall {
                    id:   call_id.or(item_id).unwrap_or_default().to_owned(),
                    name: item
                        .get("name")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned),
                    kind: call_kind,
                };
                let mut events = self.assembler.start(id.clone(), identity.clone());
                // When a lost `output_item.added` let an early fragment latch
                // the fallback block, the terminal item event lands here and
                // restores the call id and name the fallback lost.
                self.assembler.repair_tool_identity(id, identity);
                if let (Some(call_id), Some(item_id)) = (call_id, item_id)
                    && call_id != item_id
                {
                    events.extend(self.assembler.provider_metadata(
                        id,
                        NAMESPACE,
                        json!({ "item_id": item_id }),
                    ));
                }
                events
            }
            _ => Vec::new(),
        }
    }

    /// Closes the block for one output item, delivering anything the deltas
    /// did not carry.
    fn end_item(&mut self, id: &ContentBlockId, item: &Value) -> Vec<StreamEvent> {
        if self.skipped.contains(id) {
            return Vec::new();
        }
        if is_internal_call(item) {
            self.skipped.insert(id.clone());
            // A lost `output_item.added` lets the argument deltas latch a
            // fallback block before the name is known. The terminal item
            // reveals the call is model-internal: the end event closes what
            // consumers saw open, but the part is discarded — blocking decode
            // drops the item, and a nameless call must not become content or
            // flip the finish reason. Nothing opens on the normal path, so
            // this is usually empty.
            return self.assembler.discard(id);
        }
        if item.get("type").and_then(Value::as_str) == Some("reasoning") {
            return self.end_reasoning(id, item);
        }

        let mut events = self.start_item(id, item);
        events.extend(self.reconcile_item(id, item));
        events.extend(self.assembler.end(id));

        // The message item itself replays; the text block only carries what a
        // reader sees. The opaque block takes a derived id when the item's
        // own id already named a text block, and the item's id when no text
        // arrived — the same pairing reasoning items use.
        if item.get("type").and_then(Value::as_str) == Some("message") {
            let opaque = if self.delivered.contains(id) {
                ContentBlockId::new(format!("{}-item", id.as_str()))
            } else {
                id.clone()
            };
            events.extend(self.opaque_item(&opaque, MESSAGE_KIND, item));
        }
        events
    }

    /// Reconciles a block with its terminal item event, which carries the
    /// item's complete content.
    ///
    /// The terminal event is the ground truth: an item that streamed no
    /// deltas is delivered whole, a delta lost in transit has its missing
    /// tail appended, and a buffer that disagrees is replaced — so streaming
    /// and blocking decode the same response identically.
    fn reconcile_item(&mut self, id: &ContentBlockId, item: &Value) -> Vec<StreamEvent> {
        let (kind, content) = match item.get("type").and_then(Value::as_str) {
            Some("message") => (ContentBlockKind::Text, message_text(item)),
            Some(kind @ ("function_call" | "custom_tool_call")) => {
                let key = match kind {
                    "custom_tool_call" => "input",
                    _ => "arguments",
                };
                let fallback = ContentBlockKind::ToolCall {
                    id:   id.as_str().to_owned(),
                    name: None,
                    kind: ToolCallKind::Function,
                };
                (
                    fallback,
                    item.get(key)
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                )
            }
            _ => return Vec::new(),
        };
        if content.is_empty() {
            return Vec::new();
        }
        self.deliver(id, |assembler| assembler.reconcile(id, kind, &content))
    }

    /// Closes a reasoning item, keeping the whole item when it must be
    /// replayed.
    ///
    /// The opaque block reuses the item's own block id when no reasoning text
    /// streamed, and takes a derived id when it did, so both parts of one item
    /// keep distinct stable ids.
    fn end_reasoning(&mut self, id: &ContentBlockId, item: &Value) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        // The terminal item is the ground truth for the reasoning text, the
        // same reconciliation the other item kinds get.
        if let Some(text) = reasoning_text(item) {
            events.extend(self.deliver(id, |assembler| {
                assembler.reconcile(id, ContentBlockKind::Reasoning, &text)
            }));
        }
        let visible = self.delivered.contains(id);
        self.reasoning_entries.remove(id);
        events.extend(self.assembler.end(id));

        // Every reasoning item is kept whole, summary-only ones included;
        // see `decode_reasoning` for the replay pairing that requires it.
        let opaque = if visible {
            ContentBlockId::new(format!("{}-item", id.as_str()))
        } else {
            id.clone()
        };
        events.extend(self.opaque_item(&opaque, REASONING_KIND, item));
        events
    }

    /// Emits one whole output item as a closed opaque block.
    fn opaque_item(&mut self, id: &ContentBlockId, kind: &str, item: &Value) -> Vec<StreamEvent> {
        let mut events = self.assembler.start(id.clone(), ContentBlockKind::Opaque {
            kind: kind.to_owned(),
        });
        events.extend(self.assembler.set_opaque_data(id, item.clone()));
        events.extend(self.assembler.end(id));
        events
    }

    fn text_delta(&mut self, value: &Value) -> Vec<StreamEvent> {
        let id = block_id(value);
        let text = delta_text(value);
        self.deliver(&id, |assembler| assembler.text(&id, &text))
    }

    /// Appends one reasoning fragment — a `reasoning_text` or a
    /// `summary_text` delta — whose entry within its item `index_field`
    /// names.
    ///
    /// The terminal item joins its entries with a blank line, so the first
    /// fragment of a later entry of the same list carries that separator too.
    /// For a summary-only or a content-only item the streamed text is then a
    /// prefix of the terminal text and the closing reconciliation has nothing
    /// to replace. An item that streams both lists is reconciled instead: the
    /// terminal text — its `content` alone — replaces the mixed buffer.
    fn reasoning_delta(&mut self, value: &Value, index_field: &'static str) -> Vec<StreamEvent> {
        let id = block_id(value);
        let index = value.get(index_field).and_then(Value::as_u64).unwrap_or(0);
        let mut text = delta_text(value);
        if let Some((previous_field, previous)) = self
            .reasoning_entries
            .insert(id.clone(), (index_field, index))
            && previous_field == index_field
            && index > previous
        {
            text.insert_str(0, "\n\n");
        }
        self.deliver(&id, |assembler| assembler.reasoning(&id, &text))
    }

    /// Appends one argument fragment, unless the item it belongs to is
    /// model-internal and has no block of its own.
    fn arguments_delta(&mut self, value: &Value) -> Vec<StreamEvent> {
        let id = block_id(value);
        if self.skipped.contains(&id) {
            return Vec::new();
        }
        let chunk = delta_text(value);
        self.deliver(&id, |assembler| assembler.arguments(&id, &chunk))
    }

    /// Runs one assembler call and records that the block carried content.
    fn deliver(
        &mut self,
        id: &ContentBlockId,
        call: impl FnOnce(&mut StreamAssembler) -> Vec<StreamEvent>,
    ) -> Vec<StreamEvent> {
        let events = call(&mut self.assembler);
        self.delivered.insert(id.clone());
        events
    }

    /// Completes the stream from the provider's own final response document.
    ///
    /// This is the only protocol that sends one, so the finished response keeps
    /// the provider's id, usage, finish reason, and complete raw body rather
    /// than a document assembled from the event log.
    ///
    /// The terminal shape is read tolerantly: a gateway that flattens the
    /// document into the event itself, or trims it below what
    /// [`decode_document`] accepts, must not fail a stream whose answer was
    /// already delivered — the streamed blocks are the response, and whatever
    /// id, usage, and status the terminal event does carry is salvaged.
    fn complete(&mut self, value: &Value) -> Vec<StreamEvent> {
        let document = value.get("response").unwrap_or(value);
        let Ok(response) = decode_document(&self.route, document.clone()) else {
            return self.salvage(document);
        };
        if let Some(id) = response.id {
            self.assembler.set_id(id);
        }
        // The document's own output decides the reason, but a middlebox can
        // trim `output` in the terminal event — the streamed blocks are the
        // ground truth for whether the model called a tool.
        self.assembler.set_finish_reason(promote_tool_finish(
            response.finish_reason,
            self.assembler.has_tool_call(),
        ));
        self.assembler.set_raw(document.clone());

        let mut events = vec![self.assembler.usage(response.usage)];
        events.extend(self.assembler.complete());
        events
    }

    /// Completes from the assembled blocks, keeping what a nonconforming
    /// terminal document does carry.
    ///
    /// The id and usage are folded in when present; the finish reason is set
    /// only when the document reports a status, so a shape with none still
    /// completes as `incomplete` rather than a claimed clean stop.
    fn salvage(&mut self, document: &Value) -> Vec<StreamEvent> {
        if let Some(id) = document.get("id").and_then(Value::as_str) {
            self.assembler.set_id(id);
        }
        if document.get("status").and_then(Value::as_str).is_some() {
            let reason = decode_finish_reason(document, &[]);
            self.assembler
                .set_finish_reason(promote_tool_finish(reason, self.assembler.has_tool_call()));
        }
        self.assembler.set_raw(document.clone());

        let mut events = Vec::new();
        if let Some(usage) = document.get("usage").filter(|usage| usage.is_object()) {
            events.push(self.assembler.usage(token_counts(Some(usage))));
        }
        events.extend(self.assembler.complete());
        events
    }
}

/// The output item an item event carries.
pub(super) fn item(value: &Value) -> &Value {
    value.get("item").unwrap_or(&Value::Null)
}

/// The text fragment a delta event carries.
fn delta_text(value: &Value) -> String {
    value
        .get("delta")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

/// The stable block id for one stream event.
///
/// Delta events name their item directly. Item events carry the item instead,
/// whose own id is the same value. Only an event with neither falls back to the
/// output index.
fn block_id(value: &Value) -> ContentBlockId {
    let named = value
        .get("item_id")
        .and_then(Value::as_str)
        .or_else(|| value.pointer("/item/id").and_then(Value::as_str));
    match named {
        Some(id) => ContentBlockId::new(id),
        None => ContentBlockId::from_index(
            value
                .get("output_index")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        ),
    }
}
