//! The `ConverseStream` event decoder, fed already-unframed
//! `vnd.amazon.eventstream` events.

use std::collections::BTreeSet;

use serde_json::Value;

use super::decode::{stop_reason, token_counts};
use crate::codecs::StreamDecoder;
use crate::codecs::assembler::StreamAssembler;
use crate::codecs::common::{ANTHROPIC_SIGNATURES, refusal};
use crate::resolver::ResolvedRoute;
use crate::transport::{SseEvent, classify};
use crate::types::{
    ContentBlockId, ContentBlockKind, Error, ErrorKind, RetryClassification, StreamEvent,
    ToolCallKind,
};

/// Decodes one Bedrock ConverseStream.
pub(super) struct BedrockStreamDecoder {
    route:           ResolvedRoute,
    assembler:       StreamAssembler,
    /// The blocks a `contentBlockStart` opened as tool calls.
    tool_blocks:     BTreeSet<ContentBlockId>,
    /// The blocks that received a sealed `redactedContent` payload.
    redacted_blocks: BTreeSet<ContentBlockId>,
    /// The blocks that received readable reasoning text.
    text_blocks:     BTreeSet<ContentBlockId>,
}

impl StreamDecoder for BedrockStreamDecoder {
    fn decode(&mut self, event: SseEvent) -> Result<Vec<StreamEvent>, Error> {
        // An event payload that is not JSON is indistinguishable from
        // mid-stream corruption, so the failure is retryable like any other
        // garbled stream.
        let value: Value = serde_json::from_str(&event.data).map_err(|source| {
            Error::new(
                ErrorKind::StreamDecode,
                "Bedrock returned an invalid stream event",
            )
            .with_provider(self.route.provider().id().clone())
            .with_source(source)
            .with_retry(RetryClassification::Safe)
        })?;

        if let Some(error) = self.exception(&value) {
            return Err(error);
        }
        let Some((name, payload)) = named_event(event.event.as_deref(), &value) else {
            return Ok(Vec::new());
        };

        match name {
            "messageStart" => Ok(vec![self.assembler.started(None)]),
            "contentBlockStart" => Ok(self.content_block_start(payload)),
            "contentBlockDelta" => self.content_block_delta(payload),
            "contentBlockStop" => Ok(self.assembler.end(&block_id(payload))),
            "messageStop" => {
                let reason = payload.get("stopReason").and_then(Value::as_str);
                // A refusal fails the stream here, before `metadata` can
                // complete it as a success.
                if reason == Some("refusal") {
                    return Err(refusal(&self.route, None, Some(payload.clone())));
                }
                self.assembler.set_finish_reason(stop_reason(reason));
                Ok(Vec::new())
            }
            // `metadata` is the only usage event and it terminates the stream.
            // Bedrock sends no final response document, so `raw` stays unset.
            "metadata" => {
                let mut events = vec![self.assembler.usage(token_counts(payload.get("usage")))];
                events.extend(self.assembler.complete());
                Ok(events)
            }
            _ => Ok(Vec::new()),
        }
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>, Error> {
        Ok(self.assembler.complete())
    }
}

impl BedrockStreamDecoder {
    /// Creates the decoder for one stream on `route`.
    pub(super) fn new(route: &ResolvedRoute) -> Self {
        Self {
            route:           route.clone(),
            assembler:       StreamAssembler::new(route).with_signatures(ANTHROPIC_SIGNATURES),
            tool_blocks:     BTreeSet::new(),
            redacted_blocks: BTreeSet::new(),
            text_blocks:     BTreeSet::new(),
        }
    }

    /// Opens a tool-call block.
    ///
    /// Text and reasoning blocks carry no start event; their first delta
    /// latches them open.
    fn content_block_start(&mut self, payload: &Value) -> Vec<StreamEvent> {
        let Some(tool_use) = payload.pointer("/start/toolUse") else {
            return Vec::new();
        };
        // The provider tool-call id identifies the call, never the block.
        let kind = ContentBlockKind::ToolCall {
            id:   tool_use
                .get("toolUseId")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            name: tool_use
                .get("name")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            kind: ToolCallKind::Function,
        };
        let id = block_id(payload);
        self.tool_blocks.insert(id.clone());
        self.assembler.start(id, kind)
    }

    fn content_block_delta(&mut self, payload: &Value) -> Result<Vec<StreamEvent>, Error> {
        let id = block_id(payload);
        let Some(delta) = payload.get("delta") else {
            return Ok(Vec::new());
        };

        let mut events = Vec::new();
        // An empty text delta says nothing. Passing it on would open a text
        // block that carries no text, and the assembled response would end with
        // an empty part that re-encodes to nothing.
        if let Some(text) = delta.get("text").and_then(Value::as_str)
            && !text.is_empty()
        {
            events.extend(self.assembler.text(&id, text));
        }
        if let Some(chunk) = delta.pointer("/toolUse/input").and_then(Value::as_str) {
            // Converse announces every tool call in a `contentBlockStart`
            // carrying its id and name. An input fragment for a block no
            // start opened means the start was lost in transit; assembling
            // the rest would fabricate a nameless call that poisons the
            // replayed conversation, so the stream fails retryably instead —
            // the same contract the Chat codec applies.
            if !self.tool_blocks.contains(&id) {
                return Err(Error::new(
                    ErrorKind::StreamDecode,
                    format!(
                        "provider {} streamed tool-call input for a block whose start event never \
                         arrived",
                        self.route.provider().id()
                    ),
                )
                .with_provider(self.route.provider().id().clone())
                .with_retry(RetryClassification::Safe));
            }
            events.extend(self.assembler.arguments(&id, chunk));
        }
        if let Some(reasoning) = delta.get("reasoningContent") {
            events.extend(self.reasoning_delta(&id, reasoning)?);
        }
        Ok(events)
    }

    /// Applies one reasoning delta.
    ///
    /// The streaming members are flat, unlike the nested `reasoningText` block
    /// the request side uses. A redacted block arrives as a sealed
    /// `redactedContent` payload and no text; it is kept as the block's text so
    /// the assembled part re-encodes to the `redactedContent` Bedrock expects
    /// on the next turn.
    ///
    /// A block never legitimately mixes the two members, and appending them
    /// into one buffer would assemble a corrupted sealed payload that Bedrock
    /// rejects on the next turn. Text after a blob is dropped — the blob is
    /// the payload the provider verifies, as the reference decoder preferred
    /// it. A blob after text cannot win the same way, because the text was
    /// already delivered; that stream fails retryably instead of replaying
    /// corruption.
    fn reasoning_delta(
        &mut self,
        id: &ContentBlockId,
        reasoning: &Value,
    ) -> Result<Vec<StreamEvent>, Error> {
        let mut events = Vec::new();
        if let Some(text) = reasoning.get("text").and_then(Value::as_str)
            && !self.redacted_blocks.contains(id)
        {
            self.text_blocks.insert(id.clone());
            events.extend(self.assembler.reasoning(id, text));
        }
        if let Some(signature) = reasoning.get("signature").and_then(Value::as_str) {
            events.extend(self.assembler.signature(id, signature));
        }
        if let Some(sealed) = reasoning.get("redactedContent").and_then(Value::as_str) {
            if self.text_blocks.contains(id) {
                return Err(Error::new(
                    ErrorKind::StreamDecode,
                    format!(
                        "provider {} streamed redacted reasoning into a block that already \
                         carried reasoning text",
                        self.route.provider().id()
                    ),
                )
                .with_provider(self.route.provider().id().clone())
                .with_retry(RetryClassification::Safe));
            }
            self.redacted_blocks.insert(id.clone());
            events.extend(self.assembler.set_redacted(id));
            events.extend(self.assembler.reasoning(id, sealed));
        }
        Ok(events)
    }

    /// Classifies a payload that carries a modeled AWS exception.
    ///
    /// The transport already turns an `exception` frame into an error. This
    /// covers the same shape arriving inside a frame that was not labelled,
    /// which AWS marks with the `__type` discriminator.
    fn exception(&self, value: &Value) -> Option<Error> {
        value.get("__type").and_then(Value::as_str)?;
        let (message, code) = classify::extract(Some(value));
        let failure = classify::classify(None, code.as_deref(), message.as_deref(), None);
        let mut error = Error::new(
            failure.kind,
            failure
                .message
                .unwrap_or_else(|| "Bedrock failed mid-stream".to_owned()),
        )
        .with_provider(self.route.provider().id().clone())
        .with_retry(failure.retry)
        .with_raw_data(value.clone());
        if let Some(code) = failure.code {
            error = error.with_provider_code(code);
        }
        Some(error)
    }
}

/// The event name and its payload.
///
/// The transport normally supplies the name from the frame's `:event-type`
/// header, in which case the payload is the whole value. A frame that arrives
/// without one wraps its payload in a single top-level key naming the event,
/// which is the shape the Converse stream uses when it is replayed as JSON.
fn named_event<'a>(name: Option<&'a str>, value: &'a Value) -> Option<(&'a str, &'a Value)> {
    if let Some(name) = name {
        return Some((name, value));
    }
    let object = value.as_object()?;
    if object.len() != 1 {
        return None;
    }
    object
        .iter()
        .next()
        .map(|(name, payload)| (name.as_str(), payload))
}

/// The stable block id for a stream event.
///
/// Converse identifies a block only by its ordinal, so the ordinal becomes the
/// id. A payload missing the field belongs to the first block.
fn block_id(payload: &Value) -> ContentBlockId {
    let index = payload
        .get("contentBlockIndex")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    ContentBlockId::new(format!("block-{index}"))
}
