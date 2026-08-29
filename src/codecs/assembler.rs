//! Shared streaming assembly.
//!
//! [`StreamAssembler`] owns the state every provider stream decoder needs:
//! which content blocks are open, what they have accumulated, the latest usage
//! snapshot, and the fields of the final [`Response`]. Codecs translate their
//! wire protocol into assembler calls and forward the events it returns, so the
//! streaming invariants documented on
//! [`StreamEvent`](crate::types::StreamEvent) hold identically for every
//! provider.

use std::collections::BTreeMap;

use serde_json::Value;

use super::common::parse_arguments;
use crate::catalog::{ModelId, ProviderId, codec_ids};
use crate::resolver::ResolvedRoute;
use crate::types::{
    ContentBlockId, ContentBlockKind, ContentPart, Cost, FinishReason, ReasoningContent, Response,
    StreamEvent, TokenCounts, ToolCall, ToolCallKind, Warning,
};

/// The signature family a route's codec mints reasoning signatures in.
///
/// The values match the constants in [`super::common`]; they are inlined here
/// because those constants are feature-gated per codec while the assembler is
/// always built.
fn signature_family(route: &ResolvedRoute) -> Option<&'static str> {
    match route.provider().codec().as_str() {
        codec_ids::ANTHROPIC_MESSAGES | codec_ids::BEDROCK_CONVERSE => Some("anthropic"),
        codec_ids::GEMINI_GENERATE => Some("gemini"),
        _ => None,
    }
}

/// One content block, open or already closed.
#[derive(Clone, Debug)]
struct Block {
    id:                ContentBlockId,
    kind:              ContentBlockKind,
    open:              bool,
    /// Text, reasoning text, or concatenated tool argument fragments.
    buffer:            String,
    signature:         Option<String>,
    /// The signature family of this stream's codec, stamped onto a signed
    /// reasoning part so a later encoder can tell its own signatures from
    /// foreign ones.
    signature_origin:  Option<&'static str>,
    redacted:          bool,
    opaque_data:       Option<Value>,
    provider_metadata: BTreeMap<String, Value>,
}

/// The kind of content a delta carries.
///
/// A delta only appends to a block that holds this kind of content, which
/// keeps one provider's contradictory event from writing into another block's
/// buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BlockShape {
    Text,
    Reasoning,
    ToolCall,
}

impl BlockShape {
    fn matches(self, kind: &ContentBlockKind) -> bool {
        matches!(
            (self, kind),
            (Self::Text, ContentBlockKind::Text)
                | (Self::Reasoning, ContentBlockKind::Reasoning)
                | (Self::ToolCall, ContentBlockKind::ToolCall { .. })
        )
    }

    fn name(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Reasoning => "reasoning",
            Self::ToolCall => "tool_call",
        }
    }
}

impl Block {
    fn new(
        id: ContentBlockId,
        kind: ContentBlockKind,
        signature_origin: Option<&'static str>,
    ) -> Self {
        Self {
            id,
            kind,
            open: true,
            buffer: String::new(),
            signature: None,
            signature_origin,
            redacted: false,
            opaque_data: None,
            provider_metadata: BTreeMap::new(),
        }
    }

    /// Turns the accumulated state into the part the block-end event carries.
    fn into_part(self) -> ContentPart {
        match self.kind {
            ContentBlockKind::Text => ContentPart::Text { text: self.buffer },
            ContentBlockKind::Reasoning => {
                let signature_origin = self
                    .signature
                    .is_some()
                    .then(|| self.signature_origin.map(ToOwned::to_owned))
                    .flatten();
                ContentPart::Reasoning(ReasoningContent {
                    text: self.buffer,
                    signature: self.signature,
                    signature_origin,
                    redacted: self.redacted,
                })
            }
            ContentBlockKind::ToolCall { id, name, kind } => {
                let arguments = match kind {
                    ToolCallKind::Function => parse_arguments(&self.buffer),
                    ToolCallKind::Custom => Value::String(self.buffer.clone()),
                };
                ContentPart::ToolCall(ToolCall {
                    id,
                    name: name.unwrap_or_default(),
                    arguments,
                    kind,
                    raw_arguments: (!self.buffer.is_empty()).then_some(self.buffer),
                    provider_metadata: self.provider_metadata,
                })
            }
            ContentBlockKind::Opaque { kind } => ContentPart::Opaque {
                kind,
                data: self.opaque_data.unwrap_or(Value::Null),
            },
        }
    }
}

/// Builds one normalized stream from a provider's streaming protocol.
///
/// # Block identity
///
/// A [`ContentBlockId`] is unique within one stream. Opening an id that already
/// exists, open or closed, returns no event. Codecs therefore latch a start
/// lazily on the first delta without checking first, and a provider that also
/// sends an explicit start produces no duplicate.
///
/// # Event order
///
/// Every method that touches a block returns the events it produced, so a codec
/// only ever has to `extend` its own event vector. A delta method opens its
/// block first when it is not open yet, which is why these methods return a
/// vector rather than a single event: the lazy latch emits both the start and
/// the delta. A method that changes block state without producing a delta —
/// [`signature`](Self::signature), [`set_redacted`](Self::set_redacted),
/// [`set_opaque_data`](Self::set_opaque_data),
/// [`provider_metadata`](Self::provider_metadata) — returns an empty vector
/// unless it had to open the block.
///
/// A delta addressed to a block that has already been closed is ignored.
pub(crate) struct StreamAssembler {
    provider:      ProviderId,
    model:         ModelId,
    /// The signature family of this stream's codec; see [`Block`].
    signatures:    Option<&'static str>,
    /// Open and closed blocks, in the order they were opened.
    blocks:        Vec<Block>,
    /// Assembled parts, in the order their blocks were closed.
    parts:         Vec<ContentPart>,
    usage:         TokenCounts,
    finish_reason: Option<FinishReason>,
    response_id:   Option<String>,
    cost:          Option<Cost>,
    warnings:      Vec<Warning>,
    raw:           Option<Value>,
    completed:     bool,
}

impl StreamAssembler {
    /// Creates an assembler for one resolved route.
    ///
    /// The route fixes the canonical
    /// [`ModelHandle`](crate::catalog::ModelHandle) on the completed
    /// response, so a provider that echoes a different model string cannot
    /// change the response identity.
    pub(crate) fn new(route: &ResolvedRoute) -> Self {
        Self {
            provider:      route.provider().id().clone(),
            model:         route.model().id().clone(),
            signatures:    signature_family(route),
            blocks:        Vec::new(),
            parts:         Vec::new(),
            usage:         TokenCounts::default(),
            finish_reason: None,
            response_id:   None,
            cost:          None,
            warnings:      Vec::new(),
            raw:           None,
            completed:     false,
        }
    }

    /// Opens a content block and returns its start event.
    ///
    /// Returns an empty vector when a block with this id already exists, so
    /// calling it again for the same id never emits a duplicate start.
    pub(crate) fn start(&mut self, id: ContentBlockId, kind: ContentBlockKind) -> Vec<StreamEvent> {
        if self.find(&id).is_some() {
            return Vec::new();
        }

        let event = StreamEvent::ContentBlockStart {
            id:   id.clone(),
            kind: kind.clone(),
        };
        self.blocks.push(Block::new(id, kind, self.signatures));
        vec![event]
    }

    /// Appends visible text and returns the matching delta event.
    ///
    /// Opens a [`ContentBlockKind::Text`] block first when the id is not open,
    /// in which case the start event precedes the delta in the returned vector.
    ///
    /// A block already open with another kind of content ignores the delta.
    pub(crate) fn text(&mut self, id: &ContentBlockId, text: &str) -> Vec<StreamEvent> {
        let mut events = self.latch(id, ContentBlockKind::Text);
        let Some(block) = self.open_block_of(id, BlockShape::Text) else {
            return events;
        };

        block.buffer.push_str(text);
        events.push(StreamEvent::TextDelta {
            id:   id.clone(),
            text: text.to_owned(),
        });
        events
    }

    /// Appends reasoning text and returns the matching delta event.
    ///
    /// Opens a [`ContentBlockKind::Reasoning`] block first when the id is not
    /// open.
    ///
    /// A block already open with another kind of content ignores the delta.
    pub(crate) fn reasoning(&mut self, id: &ContentBlockId, text: &str) -> Vec<StreamEvent> {
        let mut events = self.latch(id, ContentBlockKind::Reasoning);
        let Some(block) = self.open_block_of(id, BlockShape::Reasoning) else {
            return events;
        };

        block.buffer.push_str(text);
        events.push(StreamEvent::ReasoningDelta {
            id:   id.clone(),
            text: text.to_owned(),
        });
        events
    }

    /// Appends one raw tool-argument fragment and returns its delta event.
    ///
    /// Call [`start`](Self::start) with the provider's own tool-call id and
    /// name before the first fragment. When the block is not open this
    /// falls back to a function tool call whose id is the block id text and
    /// whose name is empty, so a fragment is never silently dropped, but
    /// that fallback loses the provider identity a caller needs to answer
    /// the call.
    pub(crate) fn arguments(&mut self, id: &ContentBlockId, chunk: &str) -> Vec<StreamEvent> {
        let fallback = ContentBlockKind::ToolCall {
            id:   id.as_str().to_owned(),
            name: None,
            kind: ToolCallKind::Function,
        };
        let mut events = self.latch(id, fallback);
        let Some(block) = self.open_block_of(id, BlockShape::ToolCall) else {
            return events;
        };

        block.buffer.push_str(chunk);
        events.push(StreamEvent::ToolCallDelta {
            id:        id.clone(),
            arguments: chunk.to_owned(),
        });
        events
    }

    /// Accumulates a reasoning signature fragment.
    ///
    /// The signature is not a delta and produces no event of its own. The
    /// returned vector is empty unless the block had to be opened first.
    pub(crate) fn signature(&mut self, id: &ContentBlockId, signature: &str) -> Vec<StreamEvent> {
        let events = self.latch(id, ContentBlockKind::Reasoning);
        let Some(block) = self.open_block(id) else {
            return events;
        };

        block
            .signature
            .get_or_insert_with(String::new)
            .push_str(signature);
        events
    }

    /// Replaces the identity of an open tool-call block.
    ///
    /// A fragment that arrives before its item was announced latches the
    /// fallback block, whose call id is the block id text and whose name is
    /// empty. A codec that later learns the real identity — a terminal item
    /// event carries it — repairs the block here so the assembled part does
    /// not misname the call. The buffered fragments are kept, and a block
    /// that is not an open tool call is left alone.
    pub(crate) fn repair_tool_identity(&mut self, id: &ContentBlockId, identity: ContentBlockKind) {
        let Some(block) = self.open_block(id) else {
            return;
        };
        if matches!(block.kind, ContentBlockKind::ToolCall { .. })
            && matches!(identity, ContentBlockKind::ToolCall { .. })
        {
            block.kind = identity;
        }
    }

    /// Marks a reasoning block as redacted by the provider.
    ///
    /// The returned vector is empty unless the block had to be opened first.
    pub(crate) fn set_redacted(&mut self, id: &ContentBlockId) -> Vec<StreamEvent> {
        let events = self.latch(id, ContentBlockKind::Reasoning);
        let Some(block) = self.open_block(id) else {
            return events;
        };

        block.redacted = true;
        events
    }

    /// Sets the verbatim payload an opaque block carries.
    ///
    /// The returned vector is empty unless the block had to be opened first,
    /// which happens only when a codec forgot to name the opaque kind; the
    /// fallback kind is then the block id text.
    pub(crate) fn set_opaque_data(&mut self, id: &ContentBlockId, data: Value) -> Vec<StreamEvent> {
        let fallback = ContentBlockKind::Opaque {
            kind: id.as_str().to_owned(),
        };
        let events = self.latch(id, fallback);
        let Some(block) = self.open_block(id) else {
            return events;
        };

        block.opaque_data = Some(data);
        events
    }

    /// Attaches provider-namespaced replay data to an open tool-call block.
    ///
    /// It lands in [`ToolCall::provider_metadata`] on the assembled part. Only
    /// the codec that owns `namespace` reads it back when the call is replayed.
    /// The returned vector is empty unless the block had to be opened first.
    pub(crate) fn provider_metadata(
        &mut self,
        id: &ContentBlockId,
        namespace: impl Into<String>,
        value: Value,
    ) -> Vec<StreamEvent> {
        let fallback = ContentBlockKind::ToolCall {
            id:   id.as_str().to_owned(),
            name: None,
            kind: ToolCallKind::Function,
        };
        let events = self.latch(id, fallback);
        let Some(block) = self.open_block(id) else {
            return events;
        };

        block.provider_metadata.insert(namespace.into(), value);
        events
    }

    /// Closes a block and returns its end event with the fully assembled part.
    ///
    /// Returns an empty vector when the block is unknown or already closed, so
    /// a duplicate provider stop event cannot produce two end events.
    pub(crate) fn end(&mut self, id: &ContentBlockId) -> Vec<StreamEvent> {
        let Some(index) = self.find(id) else {
            return Vec::new();
        };
        if !self.blocks[index].open {
            return Vec::new();
        }

        self.blocks[index].open = false;
        let part = self.blocks[index].clone().into_part();
        self.parts.push(part.clone());
        vec![StreamEvent::ContentBlockEnd {
            id: id.clone(),
            part,
        }]
    }

    /// Records the response id and returns the stream's `Started` event.
    pub(crate) fn started(&mut self, id: Option<String>) -> StreamEvent {
        self.response_id.clone_from(&id);
        StreamEvent::Started { id }
    }

    /// Records a complete cumulative usage snapshot and returns its event.
    ///
    /// Usage events are snapshots, not deltas, so this replaces whatever was
    /// recorded before. A protocol that reports incremental counters folds them
    /// with [`merge_usage`](Self::merge_usage) instead.
    pub(crate) fn usage(&mut self, usage: TokenCounts) -> StreamEvent {
        self.usage = usage;
        StreamEvent::Usage { usage }
    }

    /// Folds part of a usage snapshot into the recorded one and returns its
    /// event.
    ///
    /// This is for protocols that split usage across several events. Anthropic
    /// sends the input, cache-read, and cache-write counts on `message_start`
    /// and the output count on `message_delta`; each event updates only the
    /// buckets it carries, and the emitted snapshot stays cumulative.
    pub(crate) fn merge_usage(&mut self, update: impl FnOnce(&mut TokenCounts)) -> StreamEvent {
        update(&mut self.usage);
        StreamEvent::Usage { usage: self.usage }
    }

    /// Records why the model stopped.
    pub(crate) fn set_finish_reason(&mut self, reason: FinishReason) {
        self.finish_reason = Some(reason);
    }

    /// The recorded finish reason, when one arrived.
    pub(crate) fn finish_reason(&self) -> Option<&FinishReason> {
        self.finish_reason.as_ref()
    }

    /// Records the provider's response id.
    pub(crate) fn set_id(&mut self, id: impl Into<String>) {
        self.response_id = Some(id.into());
    }

    /// Records the provider's own final response document.
    ///
    /// Only a protocol whose stream terminates with a complete response object
    /// sets this. It never holds an accumulated log of stream events.
    pub(crate) fn set_raw(&mut self, raw: Value) {
        self.raw = Some(raw);
    }

    /// Records a provider-reported cost, keeping the last one seen.
    ///
    /// A protocol that repeats its cost on every chunk can call this each
    /// time: the cost grows with the response, so the latest report is the
    /// accurate one.
    pub(crate) fn set_cost(&mut self, cost: Cost) {
        self.cost = Some(cost);
    }

    /// Closes every still-open block, then emits the single `Completed` event.
    ///
    /// Blocks close in the order they were opened. The completed response
    /// carries the assembled parts in the order their blocks closed, which is
    /// exactly the order of the emitted [`ContentBlockEnd`] events.
    ///
    /// This is idempotent: a second call returns an empty vector, so a codec
    /// that completes on an explicit terminal event and a
    /// [`StreamDecoder::finish`](super::StreamDecoder::finish) that also
    /// completes cannot emit two `Completed` events.
    pub(crate) fn complete(&mut self) -> Vec<StreamEvent> {
        if self.completed {
            return Vec::new();
        }
        self.completed = true;

        let open: Vec<ContentBlockId> = self
            .blocks
            .iter()
            .filter(|block| block.open)
            .map(|block| block.id.clone())
            .collect();
        let mut events = Vec::new();
        for id in &open {
            events.extend(self.end(id));
        }

        let mut response = Response::new(
            self.provider.clone(),
            self.model.clone(),
            self.parts.clone(),
        );
        response.id.clone_from(&self.response_id);
        // A provider that never reported why it stopped did not stop: the
        // stream was cut short. Reporting `Stop` here would be
        // indistinguishable from a model that finished its answer.
        response.finish_reason = self
            .finish_reason
            .clone()
            .unwrap_or_else(|| FinishReason::Other("incomplete".to_owned()));
        response.usage = self.usage;
        response.cost = self.cost;
        response.warnings.clone_from(&self.warnings);
        response.raw.clone_from(&self.raw);

        events.push(StreamEvent::Completed { response });
        events
    }

    /// Whether any block of this stream, open or closed, is a tool call.
    pub(crate) fn has_tool_call(&self) -> bool {
        self.blocks
            .iter()
            .any(|block| matches!(block.kind, ContentBlockKind::ToolCall { .. }))
    }

    /// The position of a block, open or closed.
    fn find(&self, id: &ContentBlockId) -> Option<usize> {
        self.blocks.iter().position(|block| &block.id == id)
    }

    /// The open block for an id, if it exists and is still open.
    fn open_block(&mut self, id: &ContentBlockId) -> Option<&mut Block> {
        self.blocks
            .iter_mut()
            .find(|block| &block.id == id && block.open)
    }

    /// The open block for an id, when it also holds the expected kind of
    /// content.
    ///
    /// A provider that sends, say, a text delta against an open tool-call
    /// block is contradicting itself. Appending the text would corrupt the
    /// tool arguments, so the delta is dropped instead.
    fn open_block_of(&mut self, id: &ContentBlockId, shape: BlockShape) -> Option<&mut Block> {
        let block = self.open_block(id)?;
        if shape.matches(&block.kind) {
            return Some(block);
        }
        tracing::debug!(
            block_id = id.as_str(),
            expected = shape.name(),
            "dropped a stream delta addressed to a block of another kind"
        );
        None
    }

    /// Opens a block lazily for a codec that latches on its first delta.
    fn latch(&mut self, id: &ContentBlockId, kind: ContentBlockKind) -> Vec<StreamEvent> {
        if self.find(id).is_some() {
            return Vec::new();
        }
        self.start(id.clone(), kind)
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use serde_json::{Value, json};

    use super::StreamAssembler;
    use crate::codecs::test_support;
    use crate::types::{
        ContentBlockId, ContentBlockKind, ContentPart, Cost, CostSource, FinishReason, StreamEvent,
        TokenCounts, ToolCallKind,
    };

    fn assembler() -> Result<StreamAssembler, Box<dyn StdError>> {
        Ok(StreamAssembler::new(&test_support::test_route()?))
    }

    fn tool_kind(id: &str, name: &str) -> ContentBlockKind {
        ContentBlockKind::ToolCall {
            id:   id.to_owned(),
            name: Some(name.to_owned()),
            kind: ToolCallKind::Function,
        }
    }

    /// Checks the start/delta/end invariant over a whole event sequence.
    fn assert_block_boundaries(events: &[StreamEvent]) {
        let mut open: Vec<&ContentBlockId> = Vec::new();
        let mut closed: Vec<&ContentBlockId> = Vec::new();

        for event in events {
            match event {
                StreamEvent::ContentBlockStart { id, .. } => {
                    assert!(!open.contains(&id), "{id:?} started twice");
                    assert!(!closed.contains(&id), "{id:?} restarted after its end");
                    open.push(id);
                }
                StreamEvent::TextDelta { id, .. }
                | StreamEvent::ReasoningDelta { id, .. }
                | StreamEvent::ToolCallDelta { id, .. } => {
                    assert!(open.contains(&id), "{id:?} sent a delta before its start");
                }
                StreamEvent::ContentBlockEnd { id, .. } => {
                    assert!(open.contains(&id), "{id:?} ended without a start");
                    assert!(!closed.contains(&id), "{id:?} ended twice");
                    open.retain(|open_id| *open_id != id);
                    closed.push(id);
                }
                StreamEvent::Started { .. }
                | StreamEvent::Usage { .. }
                | StreamEvent::RateLimits { .. }
                | StreamEvent::Completed { .. } => {}
            }
        }

        assert!(open.is_empty(), "blocks left open: {open:?}");
    }

    /// The parts carried by the block-end events, in order.
    fn ended_parts(events: &[StreamEvent]) -> Vec<ContentPart> {
        events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::ContentBlockEnd { part, .. } => Some(part.clone()),
                _ => None,
            })
            .collect()
    }

    fn completed_responses(events: &[StreamEvent]) -> Vec<&StreamEvent> {
        events
            .iter()
            .filter(|event| matches!(event, StreamEvent::Completed { .. }))
            .collect()
    }

    #[test]
    fn a_stream_that_never_reported_a_finish_reason_is_not_reported_as_stop()
    -> Result<(), Box<dyn StdError>> {
        let mut assembler = assembler()?;
        let id = ContentBlockId::new("block-0");
        assembler.text(&id, "partial");

        let events = assembler.complete();

        let Some(StreamEvent::Completed { response }) = events.last() else {
            panic!("the stream should complete once");
        };
        // A truncated stream must stay distinguishable from a model that
        // finished its answer.
        assert_eq!(
            response.finish_reason,
            FinishReason::Other("incomplete".to_owned())
        );
        Ok(())
    }

    #[test]
    fn a_lazy_latch_emits_a_start_before_its_first_delta() -> Result<(), Box<dyn StdError>> {
        let mut assembler = assembler()?;
        let id = ContentBlockId::new("block-0");

        let mut events = assembler.text(&id, "he");
        events.extend(assembler.text(&id, "llo"));
        events.extend(assembler.complete());

        assert_block_boundaries(&events);
        assert!(matches!(
            events.first(),
            Some(StreamEvent::ContentBlockStart {
                kind: ContentBlockKind::Text,
                ..
            })
        ));
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, StreamEvent::ContentBlockStart { .. }))
                .count(),
            1
        );
        assert_eq!(ended_parts(&events), vec![ContentPart::Text {
            text: "hello".to_owned(),
        }]);
        Ok(())
    }

    #[test]
    fn an_explicit_start_is_not_duplicated_by_a_latch() -> Result<(), Box<dyn StdError>> {
        let mut assembler = assembler()?;
        let id = ContentBlockId::new("block-0");

        let mut events = assembler.start(id.clone(), ContentBlockKind::Text);
        events.extend(assembler.start(id.clone(), ContentBlockKind::Text));
        events.extend(assembler.text(&id, "hello"));
        events.extend(assembler.end(&id));
        events.extend(assembler.end(&id));

        assert_block_boundaries(&events);
        assert_eq!(events.len(), 3);
        Ok(())
    }

    #[test]
    fn two_blocks_keep_separate_stable_ids() -> Result<(), Box<dyn StdError>> {
        let mut assembler = assembler()?;
        let first = ContentBlockId::new("block-0");
        let second = ContentBlockId::new("block-1");

        let mut events = assembler.text(&first, "one");
        events.extend(assembler.end(&first));
        events.extend(assembler.text(&second, "two"));
        events.extend(assembler.end(&second));
        events.extend(assembler.complete());

        assert_block_boundaries(&events);
        assert_eq!(ended_parts(&events), vec![
            ContentPart::Text {
                text: "one".to_owned(),
            },
            ContentPart::Text {
                text: "two".to_owned(),
            },
        ]);
        Ok(())
    }

    #[test]
    fn interleaved_tool_calls_keep_separate_block_and_call_ids() -> Result<(), Box<dyn StdError>> {
        let mut assembler = assembler()?;
        let first = ContentBlockId::new("tool-0");
        let second = ContentBlockId::new("tool-1");

        let mut events = assembler.start(first.clone(), tool_kind("call_a", "lookup"));
        events.extend(assembler.start(second.clone(), tool_kind("call_b", "search")));
        events.extend(assembler.arguments(&first, "{\"q\":"));
        events.extend(assembler.arguments(&second, "{\"n\":"));
        events.extend(assembler.arguments(&first, "\"rust\"}"));
        events.extend(assembler.arguments(&second, "2}"));
        events.extend(assembler.end(&second));
        events.extend(assembler.end(&first));
        events.extend(assembler.complete());

        assert_block_boundaries(&events);
        let parts = ended_parts(&events);
        let ContentPart::ToolCall(second_call) = &parts[0] else {
            return Err("expected the second block to end first".into());
        };
        assert_eq!(second_call.id, "call_b");
        assert_eq!(second_call.name, "search");
        assert_eq!(second_call.arguments, json!({ "n": 2 }));
        assert_eq!(second_call.raw_arguments.as_deref(), Some("{\"n\":2}"));

        let ContentPart::ToolCall(first_call) = &parts[1] else {
            return Err("expected the first block to end last".into());
        };
        assert_eq!(first_call.id, "call_a");
        assert_eq!(first_call.arguments, json!({ "q": "rust" }));
        Ok(())
    }

    #[test]
    fn a_tool_call_without_fragments_ends_as_an_empty_object() -> Result<(), Box<dyn StdError>> {
        let mut assembler = assembler()?;
        let id = ContentBlockId::new("tool-0");

        let mut events = assembler.start(id.clone(), tool_kind("call_a", "ping"));
        events.extend(assembler.end(&id));

        let parts = ended_parts(&events);
        let ContentPart::ToolCall(call) = &parts[0] else {
            return Err("expected a tool call part".into());
        };
        assert_eq!(call.arguments, json!({}));
        assert_eq!(call.raw_arguments, None);
        Ok(())
    }

    #[test]
    fn a_block_end_carries_signature_redaction_and_metadata() -> Result<(), Box<dyn StdError>> {
        let mut assembler = assembler()?;
        let reasoning = ContentBlockId::new("block-0");
        let tool = ContentBlockId::new("block-1");

        let mut events = assembler.reasoning(&reasoning, "step ");
        events.extend(assembler.reasoning(&reasoning, "one"));
        events.extend(assembler.signature(&reasoning, "sig-"));
        events.extend(assembler.signature(&reasoning, "tail"));
        events.extend(assembler.set_redacted(&reasoning));
        events.extend(assembler.end(&reasoning));

        events.extend(assembler.start(tool.clone(), tool_kind("call_a", "lookup")));
        events.extend(assembler.arguments(&tool, "{\"q\":\"rust\"}"));
        events.extend(assembler.provider_metadata(
            &tool,
            "gemini",
            json!({ "thoughtSignature": "abc" }),
        ));
        events.extend(assembler.end(&tool));

        assert_block_boundaries(&events);
        let parts = ended_parts(&events);
        let ContentPart::Reasoning(reasoning_part) = &parts[0] else {
            return Err("expected a reasoning part".into());
        };
        assert_eq!(reasoning_part.text, "step one");
        assert_eq!(reasoning_part.signature.as_deref(), Some("sig-tail"));
        assert!(reasoning_part.redacted);

        let ContentPart::ToolCall(call) = &parts[1] else {
            return Err("expected a tool call part".into());
        };
        assert_eq!(call.raw_arguments.as_deref(), Some("{\"q\":\"rust\"}"));
        assert_eq!(
            call.provider_metadata.get("gemini"),
            Some(&json!({ "thoughtSignature": "abc" }))
        );
        Ok(())
    }

    #[test]
    fn a_custom_tool_call_keeps_its_input_as_a_string() -> Result<(), Box<dyn StdError>> {
        let mut assembler = assembler()?;
        let id = ContentBlockId::new("tool-0");

        let mut events = assembler.start(id.clone(), ContentBlockKind::ToolCall {
            id:   "call_a".to_owned(),
            name: Some("apply_patch".to_owned()),
            kind: ToolCallKind::Custom,
        });
        events.extend(assembler.arguments(&id, "*** Begin Patch"));
        events.extend(assembler.end(&id));

        let parts = ended_parts(&events);
        let ContentPart::ToolCall(call) = &parts[0] else {
            return Err("expected a tool call part".into());
        };
        assert_eq!(call.kind, ToolCallKind::Custom);
        assert_eq!(call.arguments, Value::String("*** Begin Patch".to_owned()));
        Ok(())
    }

    #[test]
    fn an_opaque_block_ends_with_its_verbatim_payload() -> Result<(), Box<dyn StdError>> {
        let mut assembler = assembler()?;
        let id = ContentBlockId::new("block-0");

        let mut events = assembler.start(id.clone(), ContentBlockKind::Opaque {
            kind: "openai.reasoning".to_owned(),
        });
        events.extend(assembler.set_opaque_data(&id, json!({ "id": "rs_1" })));
        events.extend(assembler.end(&id));

        assert_eq!(ended_parts(&events), vec![ContentPart::Opaque {
            kind: "openai.reasoning".to_owned(),
            data: json!({ "id": "rs_1" }),
        }]);
        Ok(())
    }

    #[test]
    fn usage_snapshots_replace_rather_than_accumulate() -> Result<(), Box<dyn StdError>> {
        let mut assembler = assembler()?;

        let first = assembler.usage(TokenCounts::from_inclusive(100, 10, 0, 0, 0));
        let second = assembler.usage(TokenCounts::from_inclusive(100, 60, 25, 30, 10));

        assert_eq!(first, StreamEvent::Usage {
            usage: TokenCounts {
                input: 100,
                output: 10,
                ..TokenCounts::default()
            },
        });
        assert_eq!(second, StreamEvent::Usage {
            usage: TokenCounts {
                input:       60,
                output:      35,
                reasoning:   25,
                cache_read:  30,
                cache_write: 10,
            },
        });

        let events = assembler.complete();
        let [StreamEvent::Completed { response }] = events.as_slice() else {
            return Err("expected one completed event".into());
        };
        assert_eq!(response.usage.total(), 160);
        Ok(())
    }

    #[test]
    fn a_split_usage_fold_produces_one_cumulative_snapshot() -> Result<(), Box<dyn StdError>> {
        let mut assembler = assembler()?;

        // Anthropic sends the input side on `message_start`.
        let start = assembler.merge_usage(|usage| {
            usage.input = 11;
            usage.cache_read = 2;
            usage.cache_write = 1;
        });
        // The output side arrives later, on `message_delta`.
        let delta = assembler.merge_usage(|usage| usage.output = 5);

        assert_eq!(start, StreamEvent::Usage {
            usage: TokenCounts {
                input: 11,
                cache_read: 2,
                cache_write: 1,
                ..TokenCounts::default()
            },
        });
        assert_eq!(delta, StreamEvent::Usage {
            usage: TokenCounts {
                input:       11,
                output:      5,
                reasoning:   0,
                cache_read:  2,
                cache_write: 1,
            },
        });
        Ok(())
    }

    #[test]
    fn complete_closes_open_blocks_and_emits_one_completed() -> Result<(), Box<dyn StdError>> {
        let mut assembler = assembler()?;
        let text = ContentBlockId::new("block-0");
        let tool = ContentBlockId::new("tool-0");

        let mut events = vec![assembler.started(Some("resp_1".to_owned()))];
        events.extend(assembler.text(&text, "hello"));
        events.extend(assembler.start(tool.clone(), tool_kind("call_a", "lookup")));
        events.extend(assembler.arguments(&tool, "{}"));
        assembler.set_finish_reason(FinishReason::ToolCall);
        assembler.set_raw(json!({ "id": "resp_1" }));
        events.push(assembler.usage(TokenCounts::from_inclusive(11, 5, 0, 2, 1)));
        events.extend(assembler.complete());
        events.extend(assembler.complete());

        assert_block_boundaries(&events);
        assert_eq!(completed_responses(&events).len(), 1);

        let Some(StreamEvent::Completed { response }) = events.last() else {
            return Err("expected the stream to end with a completed event".into());
        };
        assert_eq!(response.content, ended_parts(&events));
        assert_eq!(response.id.as_deref(), Some("resp_1"));
        assert_eq!(response.finish_reason, FinishReason::ToolCall);
        assert_eq!(response.raw, Some(json!({ "id": "resp_1" })));
        assert_eq!(response.model.provider().as_str(), "alpha");
        assert_eq!(response.model.model().as_str(), "one");
        Ok(())
    }

    #[test]
    fn the_completed_response_keeps_the_block_end_order() -> Result<(), Box<dyn StdError>> {
        let mut assembler = assembler()?;
        let first = ContentBlockId::new("block-0");
        let second = ContentBlockId::new("block-1");

        let mut events = assembler.reasoning(&first, "thinking");
        events.extend(assembler.text(&second, "hello"));
        events.extend(assembler.end(&second));
        events.extend(assembler.end(&first));
        events.extend(assembler.complete());

        let Some(StreamEvent::Completed { response }) = events.last() else {
            return Err("expected a completed event".into());
        };
        assert_eq!(response.content, ended_parts(&events));
        assert!(matches!(response.content[0], ContentPart::Text { .. }));
        assert!(matches!(response.content[1], ContentPart::Reasoning(_)));
        Ok(())
    }

    #[test]
    fn a_delta_of_another_kind_leaves_the_open_block_alone() -> Result<(), Box<dyn StdError>> {
        let mut assembler = assembler()?;
        let tool = ContentBlockId::new("tool-0");

        let mut events = assembler.start(tool.clone(), tool_kind("call_a", "lookup"));
        events.extend(assembler.arguments(&tool, r#"{"city":"#));
        let ignored = assembler.text(&tool, "sorry, I cannot");
        events.extend(assembler.arguments(&tool, r#""Oslo"}"#));
        events.extend(assembler.complete());

        assert!(ignored.is_empty(), "a mismatched delta emits no event");
        let Some(StreamEvent::Completed { response }) = events.last() else {
            return Err("expected a completed event".into());
        };
        let ContentPart::ToolCall(call) = &response.content[0] else {
            return Err("expected the block to assemble as a tool call".into());
        };
        assert_eq!(call.arguments, json!({ "city": "Oslo" }));
        Ok(())
    }

    #[test]
    fn a_later_cost_replaces_an_earlier_one() -> Result<(), Box<dyn StdError>> {
        let mut assembler = assembler()?;

        assembler.set_cost(Cost {
            usd_micros: 10,
            source:     CostSource::Provider,
        });
        assembler.set_cost(Cost {
            usd_micros: 42,
            source:     CostSource::Provider,
        });
        let events = assembler.complete();

        let Some(StreamEvent::Completed { response }) = events.last() else {
            return Err("expected a completed event".into());
        };
        assert_eq!(
            response.cost,
            Some(Cost {
                usd_micros: 42,
                source:     CostSource::Provider,
            })
        );
        Ok(())
    }
}
