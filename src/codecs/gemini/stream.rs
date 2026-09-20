//! The `streamGenerateContent` SSE decoder.

use serde_json::{Value, json};

use super::NAMESPACE;
use super::decode::{
    blocked_prompt, response_nonce, thought_signature, token_counts, tool_call_id,
};
use crate::codecs::StreamDecoder;
use crate::codecs::assembler::StreamAssembler;
use crate::codecs::content::{GEMINI_SIGNATURES, finish_reason, promote_tool_finish};
use crate::codecs::errors::invalid_stream_event;
use crate::resolver::ResolvedRoute;
use crate::transport::{SseEvent, provider_error};
use crate::types::{
    ContentBlockId, ContentBlockKind, Error, FinishReason, StreamEvent, ToolCallKind,
};

/// Which kind of run a streamed text part continues.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Run {
    Text,
    Reasoning,
}

impl Run {
    /// The block id prefix for this kind of run.
    fn prefix(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Reasoning => "reasoning",
        }
    }
}

/// Decodes one `streamGenerateContent` response.
///
/// The protocol supplies neither block ids nor argument fragments, so this
/// decoder owns both. Ids are assigned by kind and ordinal within the stream,
/// and a run of text or reasoning parts is closed as soon as the part kind
/// changes, which is the only signal the protocol gives that a block ended.
pub(super) struct GeminiStreamDecoder {
    route:       ResolvedRoute,
    assembler:   StreamAssembler,
    /// The run currently accepting deltas, if any.
    open:        Option<(ContentBlockId, Run)>,
    texts:       usize,
    reasonings:  usize,
    calls:       usize,
    /// Whether the stream's `Started` event has been emitted.
    started:     bool,
    /// The provider's response id, taken from the first chunk that carries one.
    response_id: Option<String>,
    /// The random scope for synthesized call ids when no chunk names the
    /// response; see [`tool_call_id`].
    nonce:       String,
    /// The finish reason of the last chunk that reported one.
    finished:    Option<FinishReason>,
    /// Whether a chunk already failed, which forbids a completed response.
    failed:      bool,
}

impl GeminiStreamDecoder {
    pub(super) fn new(route: &ResolvedRoute) -> Self {
        Self {
            route:       route.clone(),
            assembler:   StreamAssembler::new(route).with_signatures(GEMINI_SIGNATURES),
            open:        None,
            texts:       0,
            reasonings:  0,
            calls:       0,
            started:     false,
            response_id: None,
            nonce:       response_nonce(),
            finished:    None,
            failed:      false,
        }
    }

    /// Translates one part of a chunk into stream events.
    fn part(&mut self, part: &Value) -> Vec<StreamEvent> {
        if let Some(function_call) = part.get("functionCall") {
            return self.function_call(part, function_call);
        }
        let Some(text) = part.get("text").and_then(Value::as_str) else {
            return Vec::new();
        };

        let run = if part.get("thought").and_then(Value::as_bool) == Some(true) {
            Run::Reasoning
        } else {
            Run::Text
        };
        let mut events = Vec::new();
        if self.open.as_ref().is_some_and(|(_, open)| *open != run) {
            events.extend(self.close_run());
        }
        let id = self.run_id(run);

        events.extend(match run {
            Run::Text => self.assembler.text(&id, text),
            Run::Reasoning => self.assembler.reasoning(&id, text),
        });
        if let (Run::Reasoning, Some(signature)) = (run, thought_signature(part)) {
            events.extend(self.assembler.signature(&id, signature));
            // Each signed wire part carries a complete signature blob over
            // the thought text before it; appending a second blob to the same
            // block would corrupt both. The signature therefore seals the
            // run, matching the blocking decoder's one part per signed part.
            events.extend(self.close_run());
        }
        events
    }

    /// Opens and immediately closes the block for a complete function call.
    ///
    /// A `functionCall` arrives whole in one chunk, so the block carries no
    /// argument delta. The arguments are still handed to the assembler, which
    /// needs them for the end event; the delta it returns is dropped.
    fn function_call(&mut self, part: &Value, function_call: &Value) -> Vec<StreamEvent> {
        let mut events = self.close_run();

        let id = ContentBlockId::new(format!("tool-{}", self.calls));
        let name = function_call
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let call_id = tool_call_id(
            function_call,
            name,
            self.response_id.as_deref().unwrap_or(&self.nonce),
            self.calls,
        );
        events.extend(
            self.assembler
                .start(id.clone(), ContentBlockKind::ToolCall {
                    id:   call_id,
                    name: Some(name.to_owned()),
                    kind: ToolCallKind::Function,
                }),
        );
        self.calls += 1;

        if let Some(arguments) = function_call.get("args") {
            let _ = self.assembler.arguments(&id, &arguments.to_string());
        }
        if let Some(signature) = thought_signature(part) {
            events.extend(self.assembler.provider_metadata(
                &id,
                NAMESPACE,
                json!({ "thoughtSignature": signature }),
            ));
        }
        events.extend(self.assembler.end(&id));
        events
    }

    /// The id of the open run, opening a new one when the kind changed.
    fn run_id(&mut self, run: Run) -> ContentBlockId {
        if let Some((id, _)) = &self.open {
            return id.clone();
        }

        let ordinal = match run {
            Run::Text => &mut self.texts,
            Run::Reasoning => &mut self.reasonings,
        };
        let id = ContentBlockId::new(format!("{}-{ordinal}", run.prefix()));
        *ordinal += 1;
        self.open = Some((id.clone(), run));
        id
    }

    /// Closes the open run, if there is one.
    fn close_run(&mut self) -> Vec<StreamEvent> {
        match self.open.take() {
            Some((id, _)) => self.assembler.end(&id),
            None => Vec::new(),
        }
    }

    /// Translates one chunk into stream events.
    fn chunk(&mut self, event: &SseEvent) -> Result<Vec<StreamEvent>, Error> {
        // A chunk that is not JSON is indistinguishable from mid-stream
        // corruption, so the failure is retryable like any other garbled
        // stream.
        let value: Value = serde_json::from_str(&event.data).map_err(|source| {
            invalid_stream_event(
                &self.route,
                "Gemini returned an invalid stream event",
                source,
            )
        })?;

        // An explicit `"error": null` member is not an error — a gateway
        // spelling out the field on success chunks must not fail every
        // stream.
        if value.get("error").is_some_and(|error| !error.is_null()) {
            return Err(provider_error(
                self.route.provider(),
                None,
                Some(value),
                None,
            ));
        }

        // A blocked prompt streams as a chunk carrying `promptFeedback` and
        // no candidates. The blocking path classifies it as a content-policy
        // failure, and an empty successful stream would hide the block, so
        // the stream fails the same way.
        if value.pointer("/candidates/0").is_none()
            && let Some(reason) = value
                .pointer("/promptFeedback/blockReason")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        {
            return Err(blocked_prompt(&self.route, &reason).with_raw_data(value));
        }

        let mut events = Vec::new();
        // Every chunk repeats the response id. The first one that carries it
        // names the response for the rest of the stream, which is also what
        // scopes the synthesized tool-call ids.
        if self.response_id.is_none()
            && let Some(id) = value.get("responseId").and_then(Value::as_str)
        {
            self.response_id = Some(id.to_owned());
            if self.started {
                self.assembler.set_id(id);
            }
        }
        if !self.started {
            self.started = true;
            events.push(self.assembler.started(self.response_id.clone()));
        }

        for part in value
            .pointer("/candidates/0/content/parts")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .cloned()
            .collect::<Vec<_>>()
        {
            events.extend(self.part(&part));
        }
        // Every chunk repeats the running totals, so the last chunk holds the
        // complete count: assign the snapshot rather than accumulating it.
        if let Some(metadata) = value.get("usageMetadata") {
            events.push(self.assembler.usage(token_counts(metadata)));
        }
        // The reason is held rather than recorded: whether it stays `Stop`
        // depends on whether a function call arrives, which the rest of the
        // stream decides. `finish` records the final answer.
        if let Some(reason) = value
            .pointer("/candidates/0/finishReason")
            .and_then(Value::as_str)
        {
            self.finished = Some(finish_reason(Some(reason)));
        }
        Ok(events)
    }
}

impl StreamDecoder for GeminiStreamDecoder {
    fn decode(&mut self, event: SseEvent) -> Result<Vec<StreamEvent>, Error> {
        let decoded = self.chunk(&event);
        if decoded.is_err() {
            self.failed = true;
        }
        decoded
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>, Error> {
        // A failed stream ends on its error and never completes.
        if self.failed {
            return Ok(Vec::new());
        }

        // A stream that reported no reason at all was cut short, and the
        // assembler says so on its own. One that reported a reason gets the
        // same tool-call correction the blocking path applies.
        if let Some(reason) = self.finished.take() {
            self.assembler
                .set_finish_reason(promote_tool_finish(reason, self.calls > 0));
        }

        // The protocol has no terminal event, so the byte-stream end is the
        // only signal that the last run closed. `raw` stays unset because
        // Gemini never sends a final response document.
        let mut events = self.close_run();
        events.extend(self.assembler.complete());
        Ok(events)
    }
}
