//! The OpenAI Chat Completions dialect.
//!
//! This codec speaks `POST /v1/chat/completions` as OpenAI defined it and as
//! every compatible skin re-implements it: OpenRouter, DeepSeek, Venice, Modal,
//! Kimi, and MiniMax. Those skins agree on the envelope and disagree on the
//! details, so the decoder accepts several spellings of the same counter and
//! keeps the untouched success body in [`Response::raw`]. Fields this crate
//! does not model — `cost_details.upstream_inference_cost`,
//! `prompt_tokens_details.audio_tokens`, OpenRouter's top-level `provider`,
//! and `native_finish_reason` — survive only there.
//!
//! It is the one codec that reads a provider-reported cost off the wire. Every
//! other protocol leaves [`Response::cost`] unset for the adapter to fill in
//! from catalog pricing.

use std::collections::BTreeMap;
use std::slice::from_ref;

use reqwest::Method;
use serde_json::{Map, Value, json, to_string};

use super::assembler::StreamAssembler;
use super::common::{
    cache_routing_key, endpoint, finish_reason, flattens_tool_result_content, merge_options,
    parse_arguments, plain_text, refusal, reject_unencodable, sampling, unsupported_capability,
    wire_options,
};
use super::{Codec, StreamDecoder};
use crate::adapter::ResolvedCall;
use crate::resolver::ResolvedRoute;
use crate::transport::{EncodedRequest, SseEvent, provider_error};
use crate::types::{
    ContentBlockId, ContentBlockKind, ContentPart, Cost, CostSource, Error, ErrorKind,
    FinishReason, ImageContent, MediaSource, Message, ReasoningContent, ReasoningEffort, Response,
    ResponseFormat, RetryClassification, Role, StreamEvent, TokenCounts, ToolCall, ToolCallKind,
    ToolChoice, ToolDefinition, ToolDefinitionKind, ToolResult,
};

/// The prefix of an opaque content kind this dialect claims.
///
/// The namespace is `openai_compatible`; the text after the separator names the
/// message field the part replays into.
const OPAQUE_PREFIX: &str = "openai_compatible.";

/// The message field carrying an aggregator's structured reasoning channel.
///
/// OpenRouter sends signed or encrypted reasoning here, and the upstream model
/// rejects a continued turn that does not send it back. It is preserved
/// verbatim as an opaque part named after this field, so
/// [`encode_chat_message`] replays it into the same place.
const REASONING_DETAILS: &str = "reasoning_details";

/// The kind the reference implementation persisted the channel under.
const LEGACY_DETAILS_KIND: &str = "openai_compat_reasoning_details";

/// The id of the single streamed text block.
///
/// The protocol carries no block ids at all, so every text fragment of one
/// response belongs to the same synthesized block.
const TEXT_BLOCK: &str = "block-0";

/// The id of the single streamed reasoning block.
const REASONING_BLOCK: &str = "reasoning-0";

/// The id of the single streamed `reasoning_details` block.
const REASONING_DETAILS_BLOCK: &str = "reasoning-details-0";

#[derive(Clone, Copy, Debug)]
pub(crate) struct OpenAiChatCodec;

impl Codec for OpenAiChatCodec {
    fn encode(&self, call: &ResolvedCall, stream: bool) -> Result<EncodedRequest, Error> {
        let request = call.request();
        let route = call.route();
        if request.tools().iter().any(ToolDefinition::is_custom) {
            return Err(unsupported_capability(route, "custom tools"));
        }
        // This dialect encodes neither audio nor documents. Dropping them
        // silently would return a successful response for a prompt that never
        // carried the caller's attachment.
        reject_unencodable(route, request, |part| match part {
            ContentPart::Audio(_) => Some("audio content"),
            ContentPart::Document(_) => Some("document content"),
            _ => None,
        })?;

        let (options, controls) = wire_options(call);
        let mut body = Map::new();
        body.insert("model".to_owned(), route.api_model().into());

        let mut messages: Vec<Value> = request.messages().iter().flat_map(encode_message).collect();
        // An aggregator fronting an Anthropic model forwards the breakpoints
        // upstream, and the catalog opts such a model in explicitly. A skin
        // that caches automatically (DeepSeek, Moonshot) declares `caching`
        // for its pricing without `cache_breakpoints`, and rewriting its
        // string content into part arrays could get the request rejected. A
        // caller can also turn breakpoints off with the `auto_cache` control.
        let capabilities = route.model().capabilities();
        if controls.auto_cache && capabilities.caching && capabilities.cache_breakpoints {
            mark_cache_breakpoints(&mut messages);
        }
        body.insert("messages".to_owned(), Value::Array(messages));

        // A blocking request omits the `stream` member entirely, matching the
        // reference encoder — a strict skin may reject an explicit
        // `stream: false`.
        if stream {
            body.insert("stream".to_owned(), true.into());
            // Without this the compatible skins never send a usage chunk, and
            // a streamed response would report no tokens at all.
            body.insert(
                "stream_options".to_owned(),
                json!({ "include_usage": true }),
            );
        }
        if let Some(max_tokens) = request.max_output_tokens() {
            body.insert("max_tokens".to_owned(), max_tokens.into());
        }
        if let Some(temperature) = request.temperature() {
            body.insert("temperature".to_owned(), sampling(temperature));
        }
        if let Some(top_p) = request.top_p() {
            body.insert("top_p".to_owned(), sampling(top_p));
        }
        if let Some(effort) = request.reasoning_effort() {
            body.insert("reasoning_effort".to_owned(), effort_name(effort).into());
        }
        if !request.stop_sequences().is_empty() {
            let stop = request
                .stop_sequences()
                .iter()
                .map(|sequence| Value::from(sequence.as_str()))
                .collect();
            body.insert("stop".to_owned(), Value::Array(stop));
        }
        if !request.tools().is_empty() {
            let tools: Vec<Value> = request.tools().iter().filter_map(encode_tool).collect();
            body.insert("tools".to_owned(), Value::Array(tools));
        }
        if let Some(choice) = request.tool_choice() {
            body.insert("tool_choice".to_owned(), encode_tool_choice(choice));
        }
        if let Some(format) = request.response_format() {
            body.insert("response_format".to_owned(), encode_response_format(format));
        }

        if let Some(key) = cache_routing_key(call, controls) {
            body.insert("prompt_cache_key".to_owned(), key.into());
        }

        // Raw provider options are merged last so an application can override
        // anything encoded above.
        merge_options(&mut body, options);

        let mut encoded = EncodedRequest::new(
            Method::POST,
            endpoint(route.provider().base_url(), "/v1/chat/completions"),
            Value::Object(body),
        )
        .with_timeout(request.timeout())
        // Every chunk is one `data:` line of JSON, and lenient compatible
        // skins and proxies separate them with single newlines rather than
        // the blank line the SSE specification requires.
        .with_data_line_framing();
        // Only OpenAI itself documents `metadata` on this endpoint, and a
        // strict skin rejects the whole request over one unknown field. The
        // tags are dropped rather than risking that, and a caller who knows
        // their skin accepts them can send them as a raw provider option.
        if !request.metadata().is_empty() {
            encoded = encoded.unsupported_control("request metadata");
        }
        // Chat Completions has no speed control. The request is served and
        // billed at standard speed, so the dropped control is reported.
        if request.speed().is_some() {
            encoded = encoded.unsupported_control("the speed control");
        }
        // A `tool` message has no error marker in this protocol, so a failed
        // tool result reaches the model looking like a successful one. The
        // content still arrives, so this is a warning rather than a refusal.
        if request
            .messages()
            .iter()
            .flat_map(Message::content)
            .any(|part| matches!(part, ContentPart::ToolResult(result) if result.is_error))
        {
            encoded = encoded.unsupported_control("the tool result error flag");
        }
        // An all-JSON result travels as the bare value, so only a mix that
        // must flatten is reported.
        if flattens_tool_result_content(request, |parts| {
            parts
                .iter()
                .all(|part| matches!(part, ContentPart::Text { .. }))
                || parts
                    .iter()
                    .all(|part| matches!(part, ContentPart::Json { .. }))
        }) {
            encoded = encoded.unsupported_control("non-text tool result content");
        }
        Ok(encoded)
    }

    fn decode_response(&self, route: &ResolvedRoute, value: Value) -> Result<Response, Error> {
        // A 200 whose `choices` array is missing or empty carries no answer at
        // all. Decoding it as an empty success would hand the caller a
        // finished response the model never wrote, so the body fails to
        // decode instead.
        if value.pointer("/choices/0").is_none() {
            return Err(no_choices(route, value));
        }
        let choice = value.pointer("/choices/0").unwrap_or(&Value::Null);
        // A choice without a message object is not a completion. A truncated
        // or error-shaped choice must not decode as an empty success.
        if !choice.get("message").is_some_and(Value::is_object) {
            return Err(decode_failure(
                route,
                "returned a choice without a message object",
                value,
            ));
        }
        let message = choice.get("message").unwrap_or(&Value::Null);
        // A refusal is a failure, not a short answer — the same contract the
        // Anthropic and Bedrock codecs apply. OpenAI reports it in a channel
        // of its own precisely so it cannot be mistaken for content.
        if let Some(text) = message
            .get("refusal")
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
        {
            let explanation = text.to_owned();
            return Err(refusal(route, Some(&explanation), Some(value)));
        }

        let mut content = Vec::new();
        // The structured reasoning channel comes first, ahead of the readable
        // reasoning text it describes.
        if let Some(details) = complete_details(message) {
            content.push(details);
        }
        if let Some(text) = reasoning_text(message) {
            content.push(ContentPart::Reasoning(ReasoningContent {
                text,
                signature: None,
                signature_origin: None,
                redacted: false,
            }));
        }
        if let Some(text) = message_text(message) {
            content.push(ContentPart::Text { text });
        }
        let mut flaw = None;
        for call in message
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            match decode_tool_call(call) {
                Ok(part) => content.push(part),
                Err(detail) => {
                    flaw = Some(detail);
                    break;
                }
            }
        }
        if let Some(detail) = flaw {
            return Err(decode_failure(route, detail, value));
        }

        let mut response = Response::new(
            route.provider().id().clone(),
            route.model().id().clone(),
            content,
        );
        response.id = value
            .get("id")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        response.finish_reason = finish_reason(choice.get("finish_reason").and_then(Value::as_str));
        // Some skins answer a tool call with `finish_reason: "stop"` — qwen
        // on Venice does. The streaming path already lets an assembled tool
        // call win over a stop reason, and both paths must decode one
        // exchange the same way, so the complete path applies the same rule.
        if response.finish_reason == FinishReason::Stop
            && response
                .content
                .iter()
                .any(|part| matches!(part, ContentPart::ToolCall(_)))
        {
            response.finish_reason = FinishReason::ToolCall;
        }
        response.usage = value.get("usage").map(token_counts).unwrap_or_default();
        response.cost = provider_cost(&value);
        response.raw = Some(value);
        Ok(response)
    }

    fn stream_decoder(&self, route: &ResolvedRoute) -> Box<dyn StreamDecoder> {
        Box::new(ChatStreamDecoder {
            assembler: StreamAssembler::new(route),
            route:     route.clone(),
            started:   false,
            details:   ReasoningDetails::default(),
            slots:     BTreeMap::new(),
            refusal:   String::new(),
        })
    }

    /// This dialect has no token count endpoint.
    ///
    /// Estimating the count locally is deliberately out of scope; a consumer
    /// that needs an estimate owns that choice, including which tokenizer to
    /// trust for a given compatible skin.
    fn encode_count_tokens(&self, _call: &ResolvedCall) -> Option<Result<EncodedRequest, Error>> {
        None
    }
}

/// The per-stream state for one Chat Completions response.
///
/// The protocol supplies no block ids and no terminal response document, so
/// this decoder synthesizes the ids and lets the assembler build the completed
/// response from what the chunks carried.
struct ChatStreamDecoder {
    assembler: StreamAssembler,
    route:     ResolvedRoute,
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
        // A chunk that is not JSON is indistinguishable from mid-stream
        // corruption, so the failure is retryable like any other garbled
        // stream.
        let chunk: Value = serde_json::from_str(&event.data).map_err(|source| {
            Error::new(
                ErrorKind::StreamDecode,
                format!(
                    "provider {} returned an invalid stream chunk",
                    self.route.provider().id()
                ),
            )
            .with_provider(self.route.provider().id().clone())
            .with_source(source)
            .with_retry(RetryClassification::Safe)
        })?;

        // An error payload ends the stream. The same classifier runs here and
        // on the HTTP error path, so one provider code means one thing. An
        // explicit `"error": null` member is not an error — a skin spelling
        // out the field on success chunks must not fail every stream.
        if chunk.get("error").is_some_and(|error| !error.is_null()) {
            return Err(provider_error(
                self.route.provider(),
                None,
                Some(chunk),
                None,
            ));
        }

        let mut events = Vec::new();
        let id = chunk
            .get("id")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        if !self.started {
            self.started = true;
            events.push(self.assembler.started(id.clone()));
        }
        if let Some(id) = id {
            self.assembler.set_id(id);
        }

        let delta = chunk.pointer("/choices/0/delta").unwrap_or(&Value::Null);
        // The structured channel is coalesced across chunks, so the block
        // carries the whole array every time a fragment lands on it.
        if let Some(payload) = delta.get(REASONING_DETAILS) {
            self.details.absorb(payload);
            if let Some(entries) = self.details.entries() {
                let block = ContentBlockId::new(REASONING_DETAILS_BLOCK);
                events.extend(
                    self.assembler
                        .start(block.clone(), ContentBlockKind::Opaque {
                            kind: details_kind(),
                        }),
                );
                events.extend(self.assembler.set_opaque_data(&block, entries));
            }
        }
        // OpenRouter spells it `reasoning`, DeepSeek `reasoning_content`.
        if let Some(text) =
            non_empty(delta, "reasoning").or_else(|| non_empty(delta, "reasoning_content"))
        {
            let block = ContentBlockId::new(REASONING_BLOCK);
            events.extend(self.assembler.reasoning(&block, text));
        }
        // `message_text` reads both wire shapes: a delta may carry the
        // part-array form just as a blocking message may, and dropping it
        // would complete the stream as an empty success.
        if let Some(text) = message_text(delta) {
            let block = ContentBlockId::new(TEXT_BLOCK);
            events.extend(self.assembler.text(&block, &text));
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

        if let Some(reason) = chunk
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str)
        {
            self.assembler
                .set_finish_reason(finish_reason(Some(reason)));
        }
        // A later chunk's cost replaces an earlier one; usage arrives in a
        // final chunk whose `choices` array is empty.
        if let Some(cost) = provider_cost(&chunk) {
            self.assembler.set_cost(cost);
        }
        if let Some(usage) = chunk.get("usage").filter(|usage| usage.is_object()) {
            events.push(self.assembler.usage(token_counts(usage)));
        }

        Ok(events)
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>, Error> {
        if !self.refusal.is_empty() {
            return Err(refusal(&self.route, Some(&self.refusal), None));
        }
        // Some skins stream tool calls yet report `stop`, or no reason at
        // all. The streamed blocks are the ground truth for whether the model
        // called a tool — the same rule the Responses codec applies to a
        // trimmed terminal document.
        if self.assembler.has_tool_call()
            && matches!(
                self.assembler.finish_reason(),
                None | Some(FinishReason::Stop)
            )
        {
            self.assembler.set_finish_reason(FinishReason::ToolCall);
        }
        Ok(self.assembler.complete())
    }
}

impl ChatStreamDecoder {
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
            return Err(Error::new(
                ErrorKind::StreamDecode,
                format!(
                    "provider {} streamed a tool-call fragment without an index",
                    self.route.provider().id()
                ),
            )
            .with_provider(self.route.provider().id().clone())
            .with_raw_data(call.clone())
            .with_retry(RetryClassification::Safe));
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

/// The opaque content kind the structured reasoning channel replays as.
fn details_kind() -> String {
    format!("{OPAQUE_PREFIX}{REASONING_DETAILS}")
}

/// The structured reasoning channel of one response, in wire order.
///
/// A complete response carries the whole array at once. A stream splits each
/// logical detail across chunks, so the fragments are coalesced back into one
/// entry before the opaque part is built — an aggregator that receives half a
/// signature back rejects the turn.
#[derive(Default)]
struct ReasoningDetails {
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
    fn absorb(&mut self, payload: &Value) {
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
    fn entries(&self) -> Option<Value> {
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
fn complete_details(message: &Value) -> Option<ContentPart> {
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

/// The error a 200 with no choices decodes into.
fn no_choices(route: &ResolvedRoute, value: Value) -> Error {
    decode_failure(route, "returned no choices in the response", value)
}

fn decode_failure(route: &ResolvedRoute, detail: &str, value: Value) -> Error {
    // A structurally malformed 200 is indistinguishable from a garbled or
    // truncated body, so a fresh attempt is safe — the same classification
    // the transport gives a 200 whose body is not JSON at all.
    Error::new(
        ErrorKind::ResponseDecode,
        format!("provider {} {detail}", route.provider().id()),
    )
    .with_provider(route.provider().id().clone())
    .with_raw_data(value)
    .with_retry(RetryClassification::Safe)
}

/// Marks the cacheable prefix of a conversation for an Anthropic upstream.
///
/// Two breakpoints, the same pair the Anthropic codec places: the last system
/// message of the leading run — the system prompt proper, which the tools and
/// instructions precede on the upstream wire — and the second-to-last user
/// turn, so each iteration of an agent loop reads the prefix the previous one
/// wrote. A tool result is its own message here and counts as a user turn,
/// because it rides in a user message upstream.
///
/// A system message appended later in the conversation is an instruction, not
/// the prompt: marking it would move the system breakpoint every time an agent
/// loop appends one, and the prefix written on the previous turn would never
/// be read back.
fn mark_cache_breakpoints(messages: &mut [Value]) {
    let leading = messages
        .iter()
        .position(|message| role_of(message) != Some("system"))
        .unwrap_or(messages.len());
    if let Some(system) = messages[..leading].last_mut() {
        mark_cached(system);
    }

    let user_turns: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, message)| matches!(role_of(message), Some("user" | "tool")))
        .map(|(index, _)| index)
        .collect();
    let Some(turn) = user_turns.len().checked_sub(2).map(|last| user_turns[last]) else {
        return;
    };
    mark_cached(&mut messages[turn]);
}

/// The wire role of one encoded message.
fn role_of(message: &Value) -> Option<&str> {
    message.get("role").and_then(Value::as_str)
}

/// Adds an ephemeral cache breakpoint to one wire message.
///
/// The plain-string content form has nowhere to carry the annotation, so it
/// becomes a one-part array; a message already in the array form marks its
/// last part. A message with no content is left alone.
fn mark_cached(message: &mut Value) {
    let Some(content) = message.get_mut("content") else {
        return;
    };
    let breakpoint = json!({ "type": "ephemeral" });
    *content = match content.take() {
        Value::String(text) => {
            json!([{ "type": "text", "text": text, "cache_control": breakpoint }])
        }
        Value::Array(mut parts) => {
            if let Some(part) = parts.last_mut().and_then(Value::as_object_mut) {
                part.insert("cache_control".to_owned(), breakpoint);
            }
            Value::Array(parts)
        }
        other => other,
    };
}

/// Encodes one canonical message into the wire messages it produces.
///
/// A tool result is its own wire message in this protocol, so one canonical
/// message that carries several results expands into several wire messages.
fn encode_message(message: &Message) -> Vec<Value> {
    let results: Vec<Value> = message
        .content()
        .iter()
        .filter_map(|part| match part {
            ContentPart::ToolResult(result) => Some(encode_tool_result(result)),
            _ => None,
        })
        .collect();

    if message.role() == Role::Tool {
        if !results.is_empty() {
            return results;
        }
        // A tool message whose result was flattened into plain text still has
        // to answer a specific call.
        let mut value = json!({ "role": "tool", "content": plain_text(message.content()) });
        if let Some(id) = message.tool_call_id() {
            value["tool_call_id"] = id.into();
        }
        return vec![value];
    }

    let mut encoded = vec![encode_chat_message(message)];
    encoded.extend(results);
    encoded
}

/// Encodes one system, developer, user, or assistant message.
fn encode_chat_message(message: &Message) -> Value {
    let parts: Vec<Value> = message
        .content()
        .iter()
        .filter_map(encode_content_part)
        .collect();
    let all_text = !parts.is_empty()
        && parts
            .iter()
            .all(|part| part.get("type").and_then(Value::as_str) == Some("text"));
    let content = match parts.as_slice() {
        // A message with no encodable parts — an assistant turn that only
        // calls tools — omits the member, the shape the reference client
        // sent. Strict skins validate content as string-or-array and reject
        // an explicit null.
        [] => None,
        // Text-only content uses the plain string form every skin accepts —
        // the part-array form is reserved for content only media-capable
        // skins receive, because a strict text-only skin rejects it. The
        // texts join unseparated, as the reference client sent them.
        _ if all_text => Some(Value::String(
            parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join(""),
        )),
        _ => Some(Value::Array(parts)),
    };

    let mut value = json!({ "role": role_name(message.role()) });
    if let Some(content) = content {
        value["content"] = content;
    }
    if let Some(name) = message.name() {
        value["name"] = name.into();
    }

    let tool_calls: Vec<Value> = message
        .content()
        .iter()
        .filter_map(|part| match part {
            ContentPart::ToolCall(call) => Some(encode_tool_call(call)),
            _ => None,
        })
        .collect();
    if !tool_calls.is_empty() {
        value["tool_calls"] = Value::Array(tool_calls);
    }

    // Kimi and DeepSeek require the assistant's own reasoning back when a
    // tool-call turn continues, so replay it rather than dropping it. Several
    // parts join unseparated, byte for byte what the reference client sent —
    // the same rule the text join follows.
    let reasoning: Vec<&str> = message
        .content()
        .iter()
        .filter_map(|part| match part {
            ContentPart::Reasoning(reasoning) if !reasoning.redacted => {
                Some(reasoning.text.as_str())
            }
            _ => None,
        })
        .collect();
    if !reasoning.is_empty() {
        value["reasoning_content"] = reasoning.concat().into();
    }

    // An opaque part of this dialect names the message field it came from, so
    // `openai_compatible.reasoning_details` replays as `reasoning_details`.
    // The reference implementation persisted the same payload under
    // `openai_compat_reasoning_details`; a migrated history replays alike.
    // Parts of another namespace are skipped, which keeps failover working.
    for part in message.content() {
        if let ContentPart::Opaque { kind, data } = part {
            if let Some(field) = kind.strip_prefix(OPAQUE_PREFIX) {
                value[field] = data.clone();
            } else if kind == LEGACY_DETAILS_KIND {
                value[REASONING_DETAILS] = data.clone();
            }
        }
    }

    value
}

/// Encodes the content parts this protocol carries inside a message.
///
/// Audio and document parts have no Chat Completions representation and are
/// skipped. Reasoning, tool calls, tool results, and opaque parts are encoded
/// by [`encode_chat_message`] as message fields rather than content parts.
fn encode_content_part(part: &ContentPart) -> Option<Value> {
    match part {
        ContentPart::Text { text } => Some(json!({ "type": "text", "text": text })),
        ContentPart::Json { value } => Some(json!({ "type": "text", "text": value.to_string() })),
        ContentPart::Image(image) => Some(encode_image(image)),
        // Audio and documents are rejected before dispatch by
        // `reject_unencodable`.
        ContentPart::Audio(_)
        | ContentPart::Document(_)
        | ContentPart::Reasoning(_)
        | ContentPart::ToolCall(_)
        | ContentPart::ToolResult(_)
        | ContentPart::Opaque { .. } => None,
    }
}

/// Encodes an image as the `image_url` part this protocol uses for both
/// sources.
///
/// Inline bytes become a `data:` URL, which is how every compatible skin
/// accepts them; there is no separate base64 shape.
fn encode_image(image: &ImageContent) -> Value {
    let url = match &image.source {
        MediaSource::Url { url, .. } => url.clone(),
        MediaSource::Base64 { data, media_type } => format!("data:{media_type};base64,{data}"),
    };
    let mut image_url = json!({ "url": url });
    if let Some(detail) = &image.detail {
        image_url["detail"] = detail.as_str().into();
    }
    json!({ "type": "image_url", "image_url": image_url })
}

/// Encodes one assistant tool call for replay.
fn encode_tool_call(call: &ToolCall) -> Value {
    json!({
        "id": call.id,
        "type": "function",
        "function": { "name": call.name, "arguments": wire_arguments(call) },
    })
}

/// The argument text to send for a tool call.
///
/// The provider's own string is replayed byte for byte when the call carries
/// one, so a re-sent turn matches what the model produced. A custom tool call
/// that reached this dialect through failover holds its input as a JSON string
/// and is sent as that text, not as a quoted JSON literal.
fn wire_arguments(call: &ToolCall) -> String {
    if let Some(raw) = &call.raw_arguments {
        return raw.clone();
    }
    match &call.arguments {
        Value::String(input) => input.clone(),
        arguments => to_string(arguments).unwrap_or_else(|_| "{}".to_owned()),
    }
}

/// Encodes one tool result as its own `tool` message.
///
/// This protocol takes a string here and nothing else, so the content is
/// flattened. Text wins when there is any, and text-only content sends its
/// joined text even when that is empty — a command with no output answered
/// with nothing, not with a serialized envelope. A result made only of JSON
/// parts sends the bare values instead — a tool that answers with structured
/// data means the data, not the `ContentPart` envelope that carried it.
fn encode_tool_result(result: &ToolResult) -> Value {
    let text = plain_text(&result.content);
    let text_only = result
        .content
        .iter()
        .all(|part| matches!(part, ContentPart::Text { .. }));
    let content = if !text.is_empty() || text_only {
        text
    } else if let Some(json) = json_result_text(&result.content) {
        json
    } else {
        to_string(&result.content).unwrap_or_default()
    };
    json!({
        "role": "tool",
        "tool_call_id": result.tool_call_id,
        "content": content,
    })
}

/// The wire text of a tool result whose content is only JSON parts.
///
/// One part sends its value; several send an array of them. A lone value that
/// is itself a string sends the raw text unquoted, as the reference encoder
/// did — the tool answered with that text, not with a JSON string literal.
/// Returns `None` when any part is something else, which leaves the caller
/// its own fallback.
fn json_result_text(content: &[ContentPart]) -> Option<String> {
    let values: Vec<&Value> = content
        .iter()
        .map(|part| match part {
            ContentPart::Json { value } => Some(value),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;

    match values.as_slice() {
        [] => None,
        [Value::String(text)] => Some(text.clone()),
        [value] => Some(value.to_string()),
        values => Some(Value::Array(values.iter().map(|&v| v.clone()).collect()).to_string()),
    }
}

/// Encodes one function tool definition.
///
/// A custom tool has no representation here and is rejected by
/// [`OpenAiChatCodec::encode`] before this runs, so it is never downgraded.
fn encode_tool(tool: &ToolDefinition) -> Option<Value> {
    match &tool.kind {
        ToolDefinitionKind::Function { input_schema } => Some(json!({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": input_schema,
            },
        })),
        ToolDefinitionKind::Custom { .. } => None,
    }
}

fn encode_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::None => json!("none"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Tool { name } => json!({ "type": "function", "function": { "name": name } }),
    }
}

fn encode_response_format(format: &ResponseFormat) -> Value {
    match format {
        ResponseFormat::Text => json!({ "type": "text" }),
        ResponseFormat::JsonObject => json!({ "type": "json_object" }),
        ResponseFormat::JsonSchema { name, schema } => json!({
            "type": "json_schema",
            "json_schema": { "name": name, "schema": schema, "strict": true },
        }),
    }
}

/// The wire role for one canonical role.
///
/// A developer message is sent as a system message. Only OpenAI itself takes
/// `developer` on this endpoint; a strict skin accepts system, user,
/// assistant, and tool and rejects the request over anything else, so the
/// closest role every skin understands is the compatible choice.
fn role_name(role: Role) -> &'static str {
    match role {
        Role::System | Role::Developer => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

/// The wire spelling of one reasoning effort level.
///
/// The canonical names are the wire vocabulary: this dialect passes the level
/// through untranslated, and a skin that knows fewer levels clamps its own.
fn effort_name(effort: ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::Minimal => "minimal",
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::Xhigh => "xhigh",
        ReasoningEffort::Max => "max",
    }
}

/// Decodes one complete tool call from a response message.
fn decode_tool_call(call: &Value) -> Result<ContentPart, &'static str> {
    // A call without its id cannot be answered, and one without its name
    // cannot be dispatched; replaying either poisons the conversation. The
    // old serde-strict decode failed retryably, and this keeps that contract.
    let id = call.get("id").and_then(Value::as_str).unwrap_or_default();
    if id.is_empty() {
        return Err("returned a tool call without an id");
    }
    let name = call
        .pointer("/function/name")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if name.is_empty() {
        return Err("returned a tool call without a function name");
    }
    let raw = call
        .pointer("/function/arguments")
        .and_then(Value::as_str)
        .unwrap_or_default();
    Ok(ContentPart::ToolCall(ToolCall {
        id:                id.to_owned(),
        name:              name.to_owned(),
        arguments:         parse_arguments(raw),
        kind:              ToolCallKind::Function,
        raw_arguments:     (!raw.is_empty()).then(|| raw.to_owned()),
        provider_metadata: BTreeMap::new(),
    }))
}

/// The visible text of a response message or stream delta, in either wire
/// shape.
///
/// Most skins send `content` as a string; a few echo the array form the request
/// uses. Both decode to the same text.
fn message_text(message: &Value) -> Option<String> {
    let text = match message.get("content") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect(),
        _ => return None,
    };
    (!text.is_empty()).then_some(text)
}

/// The reasoning text of a response message, in either spelling.
fn reasoning_text(message: &Value) -> Option<String> {
    non_empty(message, "reasoning")
        .or_else(|| non_empty(message, "reasoning_content"))
        .map(ToOwned::to_owned)
}

/// A non-empty string field, treating an empty string as absent.
fn non_empty<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
}

/// Normalizes one `usage` object into the five disjoint token buckets.
///
/// Every counter this protocol reports is inclusive: `prompt_tokens` contains
/// the cache-read and cache-write details, and `completion_tokens` contains the
/// reasoning detail. Each detail also has a flat spelling that a skin may send
/// instead — DeepSeek's `prompt_cache_hit_tokens` and Modal's
/// `reasoning_tokens` — and the nested detail wins when both are present.
/// Cache writes have a third spelling: a skin fronting an Anthropic model
/// passes through `cache_creation_input_tokens`, nested and flat (Venice
/// sends both).
fn token_counts(usage: &Value) -> TokenCounts {
    let count = |pointer: &str| usage.pointer(pointer).and_then(Value::as_u64);

    let cache_read = count("/prompt_tokens_details/cached_tokens")
        .or_else(|| count("/prompt_cache_hit_tokens"))
        .unwrap_or_default();
    let cache_write = count("/prompt_tokens_details/cache_write_tokens")
        .or_else(|| count("/prompt_tokens_details/cache_creation_input_tokens"))
        .or_else(|| count("/cache_creation_input_tokens"))
        .unwrap_or_default();
    let reasoning = count("/completion_tokens_details/reasoning_tokens")
        .or_else(|| count("/reasoning_tokens"))
        .unwrap_or_default();

    TokenCounts::from_inclusive(
        count("/prompt_tokens").unwrap_or_default(),
        count("/completion_tokens").unwrap_or_default(),
        reasoning,
        cache_read,
        cache_write,
    )
}

/// The cost a provider reported in-band, when it reported one.
///
/// Two paths in precedence order: OpenRouter's `usage.cost` and Venice's
/// top-level `cost.usd`. Venice's `cost.diem` sibling is a different currency
/// and is ignored, as is `usage.cost_details.upstream_inference_cost`, which is
/// an upstream figure rather than what this call is billed; both survive in
/// [`Response::raw`].
fn provider_cost(body: &Value) -> Option<Cost> {
    let usd = body
        .pointer("/usage/cost")
        .and_then(Value::as_f64)
        .or_else(|| body.pointer("/cost/usd").and_then(Value::as_f64))?;

    Some(Cost {
        usd_micros: usd_micros(usd),
        source:     CostSource::Provider,
    })
}

/// Converts US dollars to the integer micros the cost type carries.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a float-to-integer cast saturates, which is the clamp a provider-reported cost needs"
)]
fn usd_micros(usd: f64) -> u64 {
    (usd * 1_000_000.0).round() as u64
}

#[cfg(all(test, feature = "builtin-catalog"))]
mod tests {
    use std::error::Error as StdError;

    use serde_json::{Map, Value, json, to_string};

    use super::{Codec, OpenAiChatCodec};
    use crate::codecs::test_support::{resolved, resolved_in};
    use crate::resolver::ResolvedRoute;
    use crate::transport::SseEvent;
    use crate::types::{
        CacheHint, ContentBlockKind, ContentPart, CostSource, Error, ErrorKind, FinishReason,
        Message, ReasoningContent, ReasoningEffort, Request, Response, RetryClassification, Role,
        Speed, StreamEvent, ToolCall, ToolDefinition, ToolResult,
    };

    const MODEL: &str = "openai/gpt-5.6-luna";

    /// The catalog provider namespace `MODEL` resolves to.
    const NAMESPACE: &str = "openai";

    fn route() -> Result<ResolvedRoute, Box<dyn StdError>> {
        let request = Request::builder().model(MODEL).user("Hello").build()?;
        Ok(resolved(request)?.route().clone())
    }

    /// Decodes one complete body against the `MODEL` route.
    fn decode(body: Value) -> Result<Response, Box<dyn StdError>> {
        Ok(OpenAiChatCodec.decode_response(&route()?, body)?)
    }

    fn object(value: Value) -> Result<Map<String, Value>, Box<dyn StdError>> {
        match value {
            Value::Object(map) => Ok(map),
            other => Err(format!("expected a JSON object, got {other}").into()),
        }
    }

    /// Feeds chunks through one stream decoder and finishes the stream.
    fn stream(chunks: Vec<Value>) -> Result<Vec<StreamEvent>, Box<dyn StdError>> {
        let mut decoder = OpenAiChatCodec.stream_decoder(&route()?);
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(decoder.decode(SseEvent {
                event: None,
                data:  chunk.to_string(),
            })?);
        }
        events.extend(decoder.finish()?);
        Ok(events)
    }

    fn completed(events: &[StreamEvent]) -> Vec<&Response> {
        events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::Completed { response } => Some(response),
                _ => None,
            })
            .collect()
    }

    fn tool_parts(response: &Response) -> Vec<&ToolCall> {
        response
            .content
            .iter()
            .filter_map(|part| match part {
                ContentPart::ToolCall(call) => Some(call),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn encodes_chat_tool_calls_and_results() -> Result<(), Box<dyn StdError>> {
        let call = resolved(
            Request::builder()
                .model(MODEL)
                .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
                    ToolCall::function("call-1", "weather", json!({ "city": "Boston" })),
                )]))
                .message(Message::new(Role::Tool, [ContentPart::ToolResult(
                    ToolResult {
                        tool_call_id: "call-1".to_owned(),
                        name:         Some("weather".to_owned()),
                        content:      vec![ContentPart::Text {
                            text: "snow".to_owned(),
                        }],
                        is_error:     false,
                    },
                )]))
                .build()?,
        )?;

        let encoded = OpenAiChatCodec.encode(&call, false)?;

        assert_eq!(encoded.body["messages"][0]["role"], "assistant");
        assert_eq!(encoded.body["messages"][0]["tool_calls"][0]["id"], "call-1");
        assert_eq!(
            encoded.body["messages"][0]["tool_calls"][0]["function"]["arguments"],
            json!(r#"{"city":"Boston"}"#)
        );
        assert_eq!(encoded.body["messages"][1]["role"], "tool");
        assert_eq!(encoded.body["messages"][1]["tool_call_id"], "call-1");
        assert_eq!(encoded.body["messages"][1]["content"], "snow");
        Ok(())
    }

    #[test]
    fn inclusive_usage_becomes_disjoint_buckets() -> Result<(), Box<dyn StdError>> {
        let response = decode(json!({
            "choices": [{ "message": { "content": "ok" }, "finish_reason": "stop" }],
            "usage": {
                "prompt_tokens": 200,
                "completion_tokens": 10,
                "prompt_tokens_details": { "cached_tokens": 50, "cache_write_tokens": 100 },
                "completion_tokens_details": { "reasoning_tokens": 4 },
            },
        }))?;

        assert_eq!(response.usage.input, 50);
        assert_eq!(response.usage.cache_read, 50);
        assert_eq!(response.usage.cache_write, 100);
        assert_eq!(response.usage.output, 6);
        assert_eq!(response.usage.reasoning, 4);
        assert_eq!(response.usage.total(), 210);
        Ok(())
    }

    #[test]
    fn flat_usage_spellings_decode_the_same_way() -> Result<(), Box<dyn StdError>> {
        let response = decode(json!({
            "choices": [{ "message": { "content": "ok" } }],
            "usage": {
                "prompt_tokens": 53,
                "completion_tokens": 66,
                "prompt_cache_hit_tokens": 41,
                "reasoning_tokens": 54,
            },
        }))?;

        assert_eq!(response.usage.input, 12);
        assert_eq!(response.usage.cache_read, 41);
        assert_eq!(response.usage.output, 12);
        assert_eq!(response.usage.reasoning, 54);
        Ok(())
    }

    #[test]
    fn anthropic_style_cache_writes_decode_from_either_position() -> Result<(), Box<dyn StdError>> {
        // Venice fronting a Claude model reports cache writes as
        // `cache_creation_input_tokens`, both nested and flat, and never as
        // `cache_write_tokens`. Live calls on 2026-08-29 sent exactly this
        // shape.
        let response = decode(json!({
            "choices": [{ "message": { "content": "ok" } }],
            "usage": {
                "prompt_tokens": 15002,
                "completion_tokens": 12,
                "prompt_tokens_details": {
                    "cached_tokens": 0,
                    "cache_creation_input_tokens": 13204,
                },
                "cache_creation_input_tokens": 13204,
            },
        }))?;

        assert_eq!(response.usage.cache_write, 13204);
        assert_eq!(response.usage.input, 1798);
        assert_eq!(response.usage.total(), 15014);
        Ok(())
    }

    #[test]
    fn a_nested_detail_wins_over_its_flat_spelling() -> Result<(), Box<dyn StdError>> {
        let response = decode(json!({
            "choices": [{ "message": { "content": "ok" } }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_tokens_details": { "cached_tokens": 30 },
                "prompt_cache_hit_tokens": 90,
                "completion_tokens_details": { "reasoning_tokens": 5 },
                "reasoning_tokens": 19,
            },
        }))?;

        assert_eq!(response.usage.cache_read, 30);
        assert_eq!(response.usage.reasoning, 5);
        Ok(())
    }

    #[test]
    fn the_speed_control_is_reported_and_not_billed() -> Result<(), Box<dyn StdError>> {
        let call = resolved(
            Request::builder()
                .model(MODEL)
                .user("Hello")
                .speed(Speed::Fast)
                .build()?,
        )?;

        let encoded = OpenAiChatCodec.encode(&call, false)?;

        let messages: Vec<&str> = encoded
            .warnings
            .iter()
            .map(|warning| warning.message.as_str())
            .collect();
        assert_eq!(messages, [
            "this provider protocol does not support the speed control",
        ]);
        // Nothing speed-shaped may be guessed at on the wire, and cost must
        // not price a fast tier the provider never served.
        let body = encoded.body.to_string();
        assert!(!body.contains("speed"), "{body}");
        assert!(!body.contains("service_tier"), "{body}");
        assert_eq!(encoded.applied_speed, None);
        Ok(())
    }

    #[test]
    fn openrouter_usage_cost_is_provider_reported() -> Result<(), Box<dyn StdError>> {
        let response = decode(json!({
            "choices": [{ "message": { "content": "ok" } }],
            "usage": { "prompt_tokens": 200, "completion_tokens": 10, "cost": 0.0042 },
        }))?;

        let cost = response.cost.ok_or("expected a provider cost")?;
        assert_eq!(cost.usd_micros, 4200);
        assert_eq!(cost.source, CostSource::Provider);
        Ok(())
    }

    #[test]
    fn venice_top_level_cost_is_provider_reported() -> Result<(), Box<dyn StdError>> {
        let response = decode(json!({
            "choices": [{ "message": { "content": "ok" } }],
            "usage": { "prompt_tokens": 4, "completion_tokens": 2 },
            "cost": { "usd": 0.25, "diem": 9.5 },
        }))?;

        let cost = response.cost.ok_or("expected a provider cost")?;
        assert_eq!(cost.usd_micros, 250_000);
        assert_eq!(cost.source, CostSource::Provider);
        Ok(())
    }

    #[test]
    fn usage_cost_wins_over_the_top_level_cost() -> Result<(), Box<dyn StdError>> {
        let response = decode(json!({
            "choices": [{ "message": { "content": "ok" } }],
            "usage": { "prompt_tokens": 4, "completion_tokens": 2, "cost": 0.001 },
            "cost": { "usd": 0.25 },
        }))?;

        let cost = response.cost.ok_or("expected a provider cost")?;
        assert_eq!(cost.usd_micros, 1000);
        Ok(())
    }

    #[test]
    fn upstream_inference_cost_is_not_parsed_but_survives_in_raw() -> Result<(), Box<dyn StdError>>
    {
        let response = decode(json!({
            "provider": "Fireworks",
            "choices": [{ "message": { "content": "ok" }, "native_finish_reason": "eos" }],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "cost_details": { "upstream_inference_cost": 0.5 },
                "prompt_tokens_details": { "audio_tokens": 3 },
            },
        }))?;

        assert!(response.cost.is_none());
        let raw = response.raw.ok_or("expected the raw body")?;
        assert_eq!(raw["usage"]["cost_details"]["upstream_inference_cost"], 0.5);
        assert_eq!(raw["usage"]["prompt_tokens_details"]["audio_tokens"], 3);
        assert_eq!(raw["provider"], "Fireworks");
        assert_eq!(raw["choices"][0]["native_finish_reason"], "eos");
        Ok(())
    }

    #[test]
    fn decodes_reasoning_and_tool_calls() -> Result<(), Box<dyn StdError>> {
        let response = decode(json!({
            "id": "chatcmpl-1",
            "choices": [{
                "message": {
                    "content": "done",
                    "reasoning_content": "thought about it",
                    "tool_calls": [{
                        "id": "call-1",
                        "function": { "name": "weather", "arguments": "{\"city\":\"Boston\"}" },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        }))?;

        assert_eq!(response.id.as_deref(), Some("chatcmpl-1"));
        let reasoning = response.content.first().ok_or("expected reasoning")?;
        assert!(matches!(reasoning, ContentPart::Reasoning(_)));
        let call = tool_parts(&response)
            .first()
            .copied()
            .ok_or("expected a tool call")?
            .clone();
        assert_eq!(call.arguments, json!({ "city": "Boston" }));
        assert_eq!(call.raw_arguments.as_deref(), Some(r#"{"city":"Boston"}"#));
        Ok(())
    }

    #[test]
    fn a_complete_tool_call_wins_over_a_stop_finish_reason() -> Result<(), Box<dyn StdError>> {
        // The complete-path twin of
        // `a_streamed_tool_call_wins_over_a_stop_finish_reason`: qwen on
        // Venice answers a forced tool call with `finish_reason: "stop"`,
        // and both paths must decode that exchange the same way.
        let response = decode(json!({
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "id": "call-1",
                        "function": { "name": "weather", "arguments": "{\"city\":\"Boston\"}" },
                    }],
                },
                "finish_reason": "stop",
            }],
        }))?;

        assert_eq!(response.finish_reason, FinishReason::ToolCall);
        Ok(())
    }

    #[test]
    fn raw_provider_options_win_and_controls_never_reach_the_wire() -> Result<(), Box<dyn StdError>>
    {
        let options = object(json!({
            "temperature": 0.9,
            "seed": 7,
            "auto_cache": false,
        }))?;
        let call = resolved(
            Request::builder()
                .model(MODEL)
                .user("Hello")
                .temperature(0.2)
                .provider_options(NAMESPACE, options)
                .provider_option("anthropic", "top_k", json!(40))
                .build()?,
        )?;

        let encoded = OpenAiChatCodec.encode(&call, false)?;

        assert_eq!(encoded.body["temperature"], json!(0.9));
        assert_eq!(encoded.body["seed"], json!(7));
        assert_eq!(encoded.body.get("auto_cache"), None);
        assert_eq!(encoded.body.get("top_k"), None);
        Ok(())
    }

    #[test]
    fn stop_sequences_encode_in_order_and_metadata_is_reported() -> Result<(), Box<dyn StdError>> {
        let call = resolved(
            Request::builder()
                .model(MODEL)
                .user("Hello")
                .stop_sequences(["END", "STOP"])
                .metadata_entry("trace_id", "t789")
                .build()?,
        )?;

        let encoded = OpenAiChatCodec.encode(&call, false)?;

        assert_eq!(encoded.body["stop"], json!(["END", "STOP"]));
        // Only OpenAI itself takes `metadata` here, so the tags are dropped
        // with a warning rather than risking a strict skin's rejection.
        assert_eq!(encoded.body.get("metadata"), None);
        let warnings: Vec<&str> = encoded
            .warnings
            .iter()
            .map(|warning| warning.code.as_str())
            .collect();
        assert_eq!(warnings, ["unsupported_control"]);
        Ok(())
    }

    #[test]
    fn reasoning_effort_reaches_the_wire_untranslated() -> Result<(), Box<dyn StdError>> {
        let call = resolved(
            Request::builder()
                .model(MODEL)
                .user("Hello")
                .reasoning_effort(ReasoningEffort::Xhigh)
                .build()?,
        )?;

        let encoded = OpenAiChatCodec.encode(&call, false)?;

        assert_eq!(encoded.body["reasoning_effort"], json!("xhigh"));
        Ok(())
    }

    #[test]
    fn a_developer_message_is_sent_as_a_system_message() -> Result<(), Box<dyn StdError>> {
        let call = resolved(
            Request::builder()
                .model(MODEL)
                .message(Message::text(Role::Developer, "Keep it short."))
                .user("Hello")
                .build()?,
        )?;

        let encoded = OpenAiChatCodec.encode(&call, false)?;

        assert_eq!(encoded.body["messages"][0]["role"], "system");
        Ok(())
    }

    #[test]
    fn a_json_only_tool_result_sends_the_bare_value() -> Result<(), Box<dyn StdError>> {
        let call = resolved(
            Request::builder()
                .model(MODEL)
                .message(Message::new(Role::Tool, [ContentPart::ToolResult(
                    ToolResult {
                        tool_call_id: "call-1".to_owned(),
                        name:         Some("weather".to_owned()),
                        content:      vec![ContentPart::Json {
                            value: json!({ "city": "Boston", "temp_c": 4 }),
                        }],
                        is_error:     false,
                    },
                )]))
                .build()?,
        )?;

        let encoded = OpenAiChatCodec.encode(&call, false)?;

        assert_eq!(
            encoded.body["messages"][0]["content"],
            json!(r#"{"city":"Boston","temp_c":4}"#),
            "the value itself, not the ContentPart envelope"
        );
        Ok(())
    }

    #[test]
    fn a_json_string_tool_result_sends_the_raw_text() -> Result<(), Box<dyn StdError>> {
        // A JSON part whose value is a bare string means that text. The
        // reference encoder sent it unquoted; serializing the value would
        // hand the model the quoted JSON literal `"72F and sunny"` instead.
        let call = resolved(
            Request::builder()
                .model(MODEL)
                .message(Message::new(Role::Tool, [ContentPart::ToolResult(
                    ToolResult {
                        tool_call_id: "call-1".to_owned(),
                        name:         Some("weather".to_owned()),
                        content:      vec![ContentPart::Json {
                            value: json!("72F and sunny"),
                        }],
                        is_error:     false,
                    },
                )]))
                .build()?,
        )?;

        let encoded = OpenAiChatCodec.encode(&call, false)?;

        assert_eq!(
            encoded.body["messages"][0]["content"],
            json!("72F and sunny")
        );
        Ok(())
    }

    #[test]
    fn an_empty_text_tool_result_sends_an_empty_string() -> Result<(), Box<dyn StdError>> {
        // A command with no stdout answers with nothing. Serializing the
        // ContentPart envelope instead would hand the model spurious JSON as
        // the tool's answer.
        let call = resolved(
            Request::builder()
                .model(MODEL)
                .message(Message::new(Role::Tool, [ContentPart::ToolResult(
                    ToolResult {
                        tool_call_id: "call-1".to_owned(),
                        name:         Some("shell".to_owned()),
                        content:      vec![ContentPart::Text {
                            text: String::new(),
                        }],
                        is_error:     false,
                    },
                )]))
                .build()?,
        )?;

        let encoded = OpenAiChatCodec.encode(&call, false)?;

        assert_eq!(encoded.body["messages"][0]["content"], json!(""));
        Ok(())
    }

    #[test]
    fn several_text_parts_join_into_the_plain_string_form() -> Result<(), Box<dyn StdError>> {
        // A strict text-only skin accepts only string content; the part-array
        // form is reserved for messages carrying media. Two text parts join
        // unseparated, as the reference client sent them.
        let call = resolved(
            Request::builder()
                .model(MODEL)
                .message(Message::new(Role::User, [
                    ContentPart::Text {
                        text: "First paragraph. ".to_owned(),
                    },
                    ContentPart::Text {
                        text: "Second paragraph.".to_owned(),
                    },
                ]))
                .build()?,
        )?;

        let encoded = OpenAiChatCodec.encode(&call, false)?;

        assert_eq!(
            encoded.body["messages"][0]["content"],
            json!("First paragraph. Second paragraph.")
        );
        Ok(())
    }

    #[test]
    fn a_legacy_reasoning_details_kind_replays_into_the_field() -> Result<(), Box<dyn StdError>> {
        // The reference implementation persisted the channel as
        // `openai_compat_reasoning_details`; a migrated history must keep its
        // signed reasoning on replay, or the aggregator sees an unsigned turn.
        let details = json!([{ "type": "reasoning.encrypted", "id": "rs-1", "data": "AQ==" }]);
        let call = resolved(
            Request::builder()
                .model(MODEL)
                .user("hi")
                .message(Message::new(Role::Assistant, [
                    ContentPart::opaque("openai_compat_reasoning_details", details.clone()),
                    ContentPart::Text {
                        text: "Looking it up.".to_owned(),
                    },
                ]))
                .user("thanks")
                .build()?,
        )?;

        let encoded = OpenAiChatCodec.encode(&call, false)?;

        assert_eq!(encoded.body["messages"][1]["reasoning_details"], details);
        Ok(())
    }

    #[test]
    fn several_reasoning_parts_replay_concatenated() -> Result<(), Box<dyn StdError>> {
        // A multi-block reasoning history — an Anthropic conversation failing
        // over to a Chat route — replays as one `reasoning_content` string
        // joined unseparated, byte for byte what the reference client sent;
        // the same rule the text join follows.
        let call = resolved(
            Request::builder()
                .model(MODEL)
                .user("hi")
                .message(Message::new(Role::Assistant, [
                    ContentPart::Reasoning(ReasoningContent {
                        text:             "First block. ".to_owned(),
                        signature:        None,
                        signature_origin: None,
                        redacted:         false,
                    }),
                    ContentPart::Reasoning(ReasoningContent {
                        text:             "Second block.".to_owned(),
                        signature:        None,
                        signature_origin: None,
                        redacted:         false,
                    }),
                    ContentPart::Text {
                        text: "Answer.".to_owned(),
                    },
                ]))
                .user("thanks")
                .build()?,
        )?;

        let encoded = OpenAiChatCodec.encode(&call, false)?;

        assert_eq!(
            encoded.body["messages"][1]["reasoning_content"],
            json!("First block. Second block.")
        );
        Ok(())
    }

    #[test]
    fn a_body_without_choices_fails_to_decode() -> Result<(), Box<dyn StdError>> {
        let error = OpenAiChatCodec
            .decode_response(&route()?, json!({ "id": "chatcmpl-1", "choices": [] }))
            .err()
            .ok_or("expected an empty choices array to fail")?;

        assert_eq!(error.kind(), ErrorKind::ResponseDecode);
        assert_eq!(error.retry_classification(), RetryClassification::Safe);
        Ok(())
    }

    #[test]
    fn an_invalid_stream_chunk_fails_retryably() -> Result<(), Box<dyn StdError>> {
        // A chunk that is not JSON is indistinguishable from mid-stream
        // corruption; the old client retried it and this keeps that contract.
        let mut decoder = OpenAiChatCodec.stream_decoder(&route()?);

        let error = decoder
            .decode(SseEvent {
                event: None,
                data:  "{\"id\": \"chatcmpl-1\", \"choi".to_owned(),
            })
            .err()
            .ok_or("expected the truncated chunk to fail the stream")?;

        assert_eq!(error.kind(), ErrorKind::StreamDecode);
        assert_eq!(error.retry_classification(), RetryClassification::Safe);
        Ok(())
    }

    #[test]
    fn an_explicit_null_error_member_is_not_an_error() -> Result<(), Box<dyn StdError>> {
        // A skin that spells out `"error": null` on success chunks must not
        // fail every stream it serves.
        let mut decoder = OpenAiChatCodec.stream_decoder(&route()?);

        let events = decoder.decode(SseEvent {
            event: None,
            data:  json!({
                "id": "chatcmpl-1",
                "error": null,
                "choices": [{ "delta": { "content": "hi" } }],
            })
            .to_string(),
        })?;

        assert!(
            events
                .iter()
                .any(|event| matches!(event, StreamEvent::TextDelta { text, .. } if text == "hi")),
            "{events:?}"
        );
        Ok(())
    }

    #[test]
    fn an_array_form_content_delta_decodes_like_the_blocking_path() -> Result<(), Box<dyn StdError>>
    {
        // A skin that streams content as an array of parts must not complete
        // as an empty success; the blocking path already reads both shapes.
        let mut decoder = OpenAiChatCodec.stream_decoder(&route()?);

        let events = decoder.decode(SseEvent {
            event: None,
            data:  json!({
                "id": "chatcmpl-1",
                "choices": [{ "delta": { "content": [
                    { "type": "text", "text": "Hel" },
                    { "type": "text", "text": "lo" },
                ] } }],
            })
            .to_string(),
        })?;

        assert!(
            events.iter().any(
                |event| matches!(event, StreamEvent::TextDelta { text, .. } if text == "Hello")
            ),
            "{events:?}"
        );
        Ok(())
    }

    #[test]
    fn a_tool_call_fragment_without_an_index_fails_retryably() -> Result<(), Box<dyn StdError>> {
        // `index` is the accumulation slot. Defaulting a missing one to slot
        // 0 would silently merge parallel calls into one call with garbled
        // arguments; the reference decoder required the field and failed the
        // chunk retryably.
        let mut decoder = OpenAiChatCodec.stream_decoder(&route()?);

        let error = decoder
            .decode(SseEvent {
                event: None,
                data:  json!({
                    "id": "chatcmpl-1",
                    "choices": [{ "delta": { "tool_calls": [{
                        "id": "call-1",
                        "function": { "name": "weather", "arguments": "{}" },
                    }] } }],
                })
                .to_string(),
            })
            .err()
            .ok_or("expected the index-less fragment to fail the stream")?;

        assert_eq!(error.kind(), ErrorKind::StreamDecode);
        assert_eq!(error.retry_classification(), RetryClassification::Safe);
        Ok(())
    }

    #[test]
    fn arguments_before_identity_assemble_once_identity_arrives() -> Result<(), Box<dyn StdError>> {
        // Some skins deterministically stream `{index, function: {arguments}}`
        // before the fragment that carries the call's identity. The slot
        // opens on the arguments alone, and the late id and name repair the
        // block; failing the stream instead would retry forever against such
        // a skin.
        let events = stream(vec![
            json!({ "id": "chatcmpl-1", "choices": [{ "delta": { "tool_calls": [
                { "index": 0, "function": { "arguments": "{\"q\":" } },
            ] } }] }),
            json!({ "id": "chatcmpl-1", "choices": [{ "delta": { "tool_calls": [
                { "index": 0, "id": "call-1",
                  "function": { "name": "search", "arguments": "\"rust\"}" } },
            ] } }] }),
            json!({ "id": "chatcmpl-1", "choices": [{ "delta": {}, "finish_reason": "tool_calls" }] }),
        ])?;

        let starts = events
            .iter()
            .filter(|event| matches!(event, StreamEvent::ContentBlockStart { .. }))
            .count();
        assert_eq!(starts, 1, "the late identity must not reopen the block");

        let responses = completed(&events);
        assert_eq!(responses.len(), 1);
        let calls = tool_parts(responses[0]);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call-1");
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].arguments, json!({ "q": "rust" }));
        Ok(())
    }

    #[test]
    fn arguments_only_to_stream_end_keep_the_synthesized_identity() -> Result<(), Box<dyn StdError>>
    {
        // When the identity never arrives at all, the call still assembles —
        // the reference decoder emitted an identity-less call — with the
        // synthesized block id standing in for the call id and an empty name.
        let events = stream(vec![
            json!({ "id": "chatcmpl-1", "choices": [{ "delta": { "tool_calls": [
                { "index": 0, "function": { "arguments": "{\"q\":" } },
            ] } }] }),
            json!({ "id": "chatcmpl-1", "choices": [{ "delta": { "tool_calls": [
                { "index": 0, "function": { "arguments": "\"rust\"}" } },
            ] } }] }),
            json!({ "id": "chatcmpl-1", "choices": [{ "delta": {}, "finish_reason": "tool_calls" }] }),
        ])?;

        let responses = completed(&events);
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].finish_reason, FinishReason::ToolCall);
        let calls = tool_parts(responses[0]);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "tool-0");
        assert_eq!(calls[0].name, "");
        assert_eq!(calls[0].arguments, json!({ "q": "rust" }));
        Ok(())
    }

    #[test]
    fn tool_call_identity_arriving_on_a_later_fragment_is_kept() -> Result<(), Box<dyn StdError>> {
        // A skin may open the slot with the name and send the provider call
        // id on a later fragment. Discarding the late id would answer the
        // call with the synthesized block id, which the provider rejects on
        // the next turn.
        let events = stream(vec![
            json!({ "id": "chatcmpl-1", "choices": [{ "delta": { "tool_calls": [
                { "index": 0, "function": { "name": "search", "arguments": "{\"q\":" } },
            ] } }] }),
            json!({ "id": "chatcmpl-1", "choices": [{ "delta": { "tool_calls": [
                { "index": 0, "id": "call-real", "function": { "arguments": "\"rust\"}" } },
            ] } }] }),
            json!({ "id": "chatcmpl-1", "choices": [{ "delta": {}, "finish_reason": "tool_calls" }] }),
        ])?;

        let responses = completed(&events);
        assert_eq!(responses.len(), 1);
        let calls = tool_parts(responses[0]);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call-real");
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].arguments, json!({ "q": "rust" }));
        Ok(())
    }

    #[test]
    fn a_streamed_tool_call_wins_over_a_stop_finish_reason() -> Result<(), Box<dyn StdError>> {
        // Some skins stream tool calls yet report `finish_reason: "stop"`.
        // The streamed blocks are the ground truth — an agent loop keyed on
        // `ToolCall` must see the calls it is meant to execute.
        let events = stream(vec![
            json!({ "id": "chatcmpl-1", "choices": [{ "delta": { "tool_calls": [
                { "index": 0, "id": "call-1",
                  "function": { "name": "search", "arguments": "{\"q\":\"rust\"}" } },
            ] } }] }),
            json!({ "id": "chatcmpl-1", "choices": [{ "delta": {}, "finish_reason": "stop" }] }),
        ])?;

        let responses = completed(&events);
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].finish_reason, FinishReason::ToolCall);
        Ok(())
    }

    #[test]
    fn a_streamed_tool_call_sets_the_finish_reason_when_none_arrived()
    -> Result<(), Box<dyn StdError>> {
        // A skin that never reports a finish reason still called the tool;
        // completing as `incomplete` would end an agent loop mid-round.
        let events = stream(vec![json!({ "id": "chatcmpl-1", "choices": [{ "delta": {
            "tool_calls": [
                { "index": 0, "id": "call-1",
                  "function": { "name": "search", "arguments": "{}" } },
            ],
        } }] })])?;

        let responses = completed(&events);
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].finish_reason, FinishReason::ToolCall);
        Ok(())
    }

    #[test]
    fn a_non_stop_finish_reason_is_kept_despite_streamed_tool_calls()
    -> Result<(), Box<dyn StdError>> {
        // Truncation trumps inference: a `length` stop on a partial call is
        // still a truncated answer.
        let events = stream(vec![
            json!({ "id": "chatcmpl-1", "choices": [{ "delta": { "tool_calls": [
                { "index": 0, "id": "call-1",
                  "function": { "name": "search", "arguments": "{\"q\":" } },
            ] } }] }),
            json!({ "id": "chatcmpl-1", "choices": [{ "delta": {}, "finish_reason": "length" }] }),
        ])?;

        let responses = completed(&events);
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].finish_reason, FinishReason::Length);
        Ok(())
    }

    #[test]
    fn a_refusal_fails_instead_of_decoding() -> Result<(), Box<dyn StdError>> {
        // The structured-output refusal channel: content null, refusal text.
        // Decoding it as an empty success would hide the refusal from the
        // caller and from failover — the contract H8 set for Anthropic and
        // Bedrock extends here.
        let body = json!({
            "id": "chatcmpl-1",
            "choices": [{
                "message": { "role": "assistant", "content": null, "refusal": "I can't do that." },
                "finish_reason": "stop",
            }],
        });

        let error = OpenAiChatCodec
            .decode_response(&route()?, body.clone())
            .expect_err("a refusal must fail the call");

        assert_eq!(error.kind(), ErrorKind::ContentFilter);
        assert_eq!(error.provider_code(), Some("refusal"));
        assert!(error.to_string().contains("I can't do that."), "{error}");
        assert_eq!(error.raw_data(), Some(&body));
        Ok(())
    }

    #[test]
    fn a_streamed_refusal_fails_the_stream_with_the_whole_explanation()
    -> Result<(), Box<dyn StdError>> {
        let mut decoder = OpenAiChatCodec.stream_decoder(&route()?);
        for fragment in ["I can't ", "do that."] {
            decoder.decode(SseEvent {
                event: None,
                data:  json!({
                    "id": "chatcmpl-1",
                    "choices": [{ "delta": { "refusal": fragment } }],
                })
                .to_string(),
            })?;
        }

        let error = decoder
            .finish()
            .expect_err("a streamed refusal must fail the stream");

        assert_eq!(error.kind(), ErrorKind::ContentFilter);
        assert_eq!(error.provider_code(), Some("refusal"));
        assert!(error.to_string().contains("I can't do that."), "{error}");
        Ok(())
    }

    #[test]
    fn a_choice_without_a_message_fails_to_decode() -> Result<(), Box<dyn StdError>> {
        let error = OpenAiChatCodec
            .decode_response(
                &route()?,
                json!({ "id": "chatcmpl-1", "choices": [{ "finish_reason": "stop" }] }),
            )
            .err()
            .ok_or("expected a choice without a message to fail")?;

        assert_eq!(error.kind(), ErrorKind::ResponseDecode);
        assert_eq!(error.retry_classification(), RetryClassification::Safe);
        Ok(())
    }

    #[test]
    fn a_tool_call_without_its_identity_fails_to_decode() -> Result<(), Box<dyn StdError>> {
        // A call missing its id cannot be answered; one missing its name
        // cannot be dispatched. Both must fail retryably instead of decoding
        // into a part that poisons the replayed conversation.
        for tool_call in [
            json!({ "type": "function", "function": { "name": "f", "arguments": "{}" } }),
            json!({ "id": "call-1", "type": "function", "function": { "arguments": "{}" } }),
        ] {
            let error = OpenAiChatCodec
                .decode_response(
                    &route()?,
                    json!({
                        "id": "chatcmpl-1",
                        "choices": [{ "message": { "tool_calls": [tool_call] } }],
                    }),
                )
                .err()
                .ok_or("expected a tool call without id or name to fail")?;
            assert_eq!(error.kind(), ErrorKind::ResponseDecode);
            assert_eq!(error.retry_classification(), RetryClassification::Safe);
        }
        Ok(())
    }

    /// A one-provider catalog whose model opts into `cache_control`
    /// breakpoints, the way an aggregator fronting an Anthropic model does.
    const BREAKPOINT_CATALOG: &str = r#"
        schema_version = 1

        [providers.gateway]
        display_name = "Gateway"
        adapter = "openai-compatible"
        codec = "openai-chat"
        base_url = "http://127.0.0.1"
        default_model = "fronted"
        auth = { type = "bearer" }

        [providers.gateway.models.fronted]
        display_name = "Fronted"
        api_model = "fronted-v1"
        capabilities = { text = true, caching = true, cache_breakpoints = true }
    "#;

    /// A one-provider catalog whose model takes a cache routing hint, the
    /// way Venice and OpenAI take `prompt_cache_key`.
    const ROUTING_CATALOG: &str = r#"
        schema_version = 1

        [providers.gateway]
        display_name = "Gateway"
        adapter = "openai-compatible"
        codec = "openai-chat"
        base_url = "http://127.0.0.1"
        default_model = "routed"
        auth = { type = "bearer" }

        [providers.gateway.models.routed]
        display_name = "Routed"
        api_model = "routed-v1"
        capabilities = { text = true, tools = true, caching = true, cache_routing = true }
    "#;

    fn routed(request: Request) -> Result<Value, Box<dyn StdError>> {
        let call = resolved_in(ROUTING_CATALOG, request)?;
        Ok(OpenAiChatCodec.encode(&call, false)?.body)
    }

    #[test]
    fn the_default_cache_hint_sends_a_stable_fingerprint() -> Result<(), Box<dyn StdError>> {
        let request = || {
            Request::builder()
                .model("gateway/routed")
                .system("Keep it short.")
                .user("Hello")
                .build()
        };
        let first = routed(request()?)?;
        let second = routed(request()?)?;

        let key = first["prompt_cache_key"]
            .as_str()
            .ok_or("no prompt_cache_key was sent")?;
        assert!(key.starts_with("lithos-"), "unexpected key shape: {key}");
        assert_eq!(first["prompt_cache_key"], second["prompt_cache_key"]);

        // A different system prefix routes elsewhere; a different user turn
        // does not, so an agent loop keeps one replica.
        let other_system = routed(
            Request::builder()
                .model("gateway/routed")
                .system("Answer in French.")
                .user("Hello")
                .build()?,
        )?;
        assert_ne!(first["prompt_cache_key"], other_system["prompt_cache_key"]);
        let other_turn = routed(
            Request::builder()
                .model("gateway/routed")
                .system("Keep it short.")
                .user("A different question entirely")
                .build()?,
        )?;
        assert_eq!(first["prompt_cache_key"], other_turn["prompt_cache_key"]);
        Ok(())
    }

    #[test]
    fn an_explicit_cache_key_is_sent_verbatim() -> Result<(), Box<dyn StdError>> {
        let body = routed(
            Request::builder()
                .model("gateway/routed")
                .user("Hello")
                .cache_key("tenant-42")
                .build()?,
        )?;
        assert_eq!(body["prompt_cache_key"], json!("tenant-42"));
        Ok(())
    }

    #[test]
    fn a_disabled_cache_hint_sends_nothing() -> Result<(), Box<dyn StdError>> {
        let body = routed(
            Request::builder()
                .model("gateway/routed")
                .user("Hello")
                .cache_hint(CacheHint::Disabled)
                .build()?,
        )?;
        assert_eq!(body.get("prompt_cache_key"), None);
        Ok(())
    }

    #[test]
    fn auto_cache_off_suppresses_the_fingerprint() -> Result<(), Box<dyn StdError>> {
        let body = routed(
            Request::builder()
                .model("gateway/routed")
                .user("Hello")
                .provider_option("gateway", "auto_cache", json!(false))
                .build()?,
        )?;
        assert_eq!(body.get("prompt_cache_key"), None);
        Ok(())
    }

    #[test]
    fn a_raw_prompt_cache_key_wins_over_the_fingerprint() -> Result<(), Box<dyn StdError>> {
        let body = routed(
            Request::builder()
                .model("gateway/routed")
                .user("Hello")
                .provider_option("gateway", "prompt_cache_key", json!("raw-key"))
                .build()?,
        )?;
        assert_eq!(body["prompt_cache_key"], json!("raw-key"));
        Ok(())
    }

    #[test]
    fn no_routing_capability_means_no_fingerprint() -> Result<(), Box<dyn StdError>> {
        let call = resolved_in(BREAKPOINT_CATALOG, multi_turn("gateway/fronted")?)?;
        let encoded = OpenAiChatCodec.encode(&call, false)?;
        assert_eq!(encoded.body.get("prompt_cache_key"), None);
        Ok(())
    }

    /// A one-provider catalog with provider-level default request options,
    /// the way a Venice row turns off the injected system prompt.
    const DEFAULT_OPTIONS_CATALOG: &str = r#"
        schema_version = 1

        [providers.gateway]
        display_name = "Gateway"
        adapter = "openai-compatible"
        codec = "openai-chat"
        base_url = "http://127.0.0.1"
        default_model = "fronted"
        auth = { type = "bearer" }

        [providers.gateway.default_options]
        venice_parameters = { include_venice_system_prompt = false, strip_thinking_response = false }

        [providers.gateway.models.fronted]
        display_name = "Fronted"
        api_model = "fronted-v1"
        capabilities = { text = true }
    "#;

    #[test]
    fn catalog_default_options_reach_the_wire() -> Result<(), Box<dyn StdError>> {
        let call = resolved_in(
            DEFAULT_OPTIONS_CATALOG,
            Request::builder()
                .model("gateway/fronted")
                .user("Hello")
                .build()?,
        )?;

        let encoded = OpenAiChatCodec.encode(&call, false)?;

        assert_eq!(
            encoded.body["venice_parameters"]["include_venice_system_prompt"],
            json!(false)
        );
        Ok(())
    }

    #[test]
    fn request_options_win_over_catalog_defaults_key_by_key() -> Result<(), Box<dyn StdError>> {
        let call = resolved_in(
            DEFAULT_OPTIONS_CATALOG,
            Request::builder()
                .model("gateway/fronted")
                .user("Hello")
                .provider_option(
                    "gateway",
                    "venice_parameters",
                    json!({ "include_venice_system_prompt": true }),
                )
                .build()?,
        )?;

        let encoded = OpenAiChatCodec.encode(&call, false)?;

        // The request overrides the one key it names; the default's sibling
        // key survives, because option namespaces merge recursively.
        assert_eq!(
            encoded.body["venice_parameters"]["include_venice_system_prompt"],
            json!(true)
        );
        assert_eq!(
            encoded.body["venice_parameters"]["strip_thinking_response"],
            json!(false)
        );
        Ok(())
    }

    #[test]
    fn a_control_key_in_catalog_defaults_is_consumed() -> Result<(), Box<dyn StdError>> {
        let catalog = BREAKPOINT_CATALOG.replace(
            "[providers.gateway.models.fronted]",
            "[providers.gateway.default_options]\n\
             auto_cache = false\n\n\
             [providers.gateway.models.fronted]",
        );
        let call = resolved_in(&catalog, multi_turn("gateway/fronted")?)?;

        let encoded = OpenAiChatCodec.encode(&call, false)?;

        let rendered = to_string(&encoded.body)?;
        assert_eq!(rendered.matches("cache_control").count(), 0);
        assert_eq!(encoded.body.get("auto_cache"), None);
        Ok(())
    }

    fn multi_turn(model: &str) -> Result<Request, Box<dyn StdError>> {
        Ok(Request::builder()
            .model(model)
            .system("Keep it short.")
            .user("What is the capital of France?")
            .message(Message::text(Role::Assistant, "Paris."))
            .user("And of Spain?")
            .build()?)
    }

    #[test]
    fn auto_cache_marks_the_system_message_and_the_prefix() -> Result<(), Box<dyn StdError>> {
        let call = resolved_in(BREAKPOINT_CATALOG, multi_turn("gateway/fronted")?)?;

        let encoded = OpenAiChatCodec.encode(&call, false)?;

        let messages = &encoded.body["messages"];
        assert_eq!(
            messages[0]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );
        assert_eq!(
            messages[1]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );
        assert_eq!(
            messages[1]["content"][0]["text"],
            "What is the capital of France?"
        );
        assert!(
            messages[3]["content"].is_string(),
            "the last user turn keeps the plain string form"
        );
        Ok(())
    }

    #[test]
    fn auto_cache_false_places_no_breakpoints() -> Result<(), Box<dyn StdError>> {
        let call = resolved_in(
            BREAKPOINT_CATALOG,
            Request::builder()
                .model("gateway/fronted")
                .system("Keep it short.")
                .user("What is the capital of France?")
                .message(Message::text(Role::Assistant, "Paris."))
                .user("And of Spain?")
                .provider_option("gateway", "auto_cache", json!(false))
                .build()?,
        )?;

        let encoded = OpenAiChatCodec.encode(&call, false)?;

        assert_eq!(
            to_string(&encoded.body)?.matches("cache_control").count(),
            0
        );
        Ok(())
    }

    #[test]
    fn caching_without_breakpoint_support_places_no_breakpoints() -> Result<(), Box<dyn StdError>> {
        // The builtin model declares `caching` for its pricing but not
        // `cache_breakpoints`. A skin whose caching is automatic can reject
        // the part-array rewrite, so its content must stay untouched.
        let call = resolved(multi_turn(MODEL)?)?;

        let encoded = OpenAiChatCodec.encode(&call, false)?;

        assert_eq!(
            to_string(&encoded.body)?.matches("cache_control").count(),
            0
        );
        assert!(
            encoded.body["messages"][0]["content"].is_string(),
            "unmarked messages keep the plain string form"
        );
        Ok(())
    }

    #[test]
    fn reasoning_details_are_preserved_verbatim() -> Result<(), Box<dyn StdError>> {
        let details = json!([
            { "type": "reasoning.encrypted", "id": "rs-1", "data": "opaque" },
            { "type": "reasoning.text", "text": "step one", "index": 0 },
        ]);
        let response = decode(json!({
            "choices": [{
                "message": { "content": "done", "reasoning_details": details },
                "finish_reason": "stop",
            }],
        }))?;

        let first = response.content.first().ok_or("expected an opaque part")?;
        match first {
            ContentPart::Opaque { kind, data } => {
                assert_eq!(kind, "openai_compatible.reasoning_details");
                assert_eq!(data, &details);
            }
            other => return Err(format!("expected an opaque part, got {other:?}").into()),
        }
        Ok(())
    }

    #[test]
    fn same_type_unindexed_complete_details_stay_separate_entries() -> Result<(), Box<dyn StdError>>
    {
        // Multi-block reasoning through an aggregator: two entries of one
        // type, no indexes, each sealed by its own signature. Coalescing them
        // would discard the second signature and fail verification upstream
        // on replay — only a stream coalesces, to undo its own fragmenting.
        let details = json!([
            { "type": "reasoning.text", "text": "step one", "signature": "sig-1" },
            { "type": "reasoning.text", "text": "step two", "signature": "sig-2" },
        ]);
        let response = decode(json!({
            "choices": [{
                "message": { "content": "done", "reasoning_details": details },
                "finish_reason": "stop",
            }],
        }))?;

        let first = response.content.first().ok_or("expected an opaque part")?;
        match first {
            ContentPart::Opaque { kind, data } => {
                assert_eq!(kind, "openai_compatible.reasoning_details");
                assert_eq!(data, &details);
            }
            other => return Err(format!("expected an opaque part, got {other:?}").into()),
        }
        Ok(())
    }

    #[test]
    fn streamed_detail_fragments_coalesce_by_type_and_index() -> Result<(), Box<dyn StdError>> {
        let events = stream(vec![
            json!({ "id": "chatcmpl-1", "choices": [{ "delta": { "reasoning_details": [
                { "type": "reasoning.text", "index": 0, "text": "first " },
            ] } }] }),
            json!({ "choices": [{ "delta": { "reasoning_details": [
                { "type": "reasoning.text", "index": 1, "text": "second " },
            ] } }] }),
            json!({ "choices": [{ "delta": { "reasoning_details": [
                { "type": "reasoning.text", "index": 0, "text": "half" },
            ] } }] }),
            json!({ "choices": [{ "delta": { "reasoning_details": [
                { "type": "reasoning.text", "index": 1, "text": "half", "signature": "sig" },
            ] } }] }),
            json!({ "choices": [{ "delta": { "content": "done" }, "finish_reason": "stop" }] }),
        ])?;

        let response = completed(&events)
            .first()
            .copied()
            .ok_or("expected a completed response")?;
        let first = response.content.first().ok_or("expected an opaque part")?;
        match first {
            ContentPart::Opaque { kind, data } => {
                assert_eq!(kind, "openai_compatible.reasoning_details");
                assert_eq!(
                    data,
                    &json!([
                        { "type": "reasoning.text", "index": 0, "text": "first half" },
                        {
                            "type": "reasoning.text",
                            "index": 1,
                            "text": "second half",
                            "signature": "sig",
                        },
                    ])
                );
            }
            other => return Err(format!("expected an opaque part, got {other:?}").into()),
        }
        Ok(())
    }

    #[test]
    fn a_custom_tool_is_rejected_before_dispatch() -> Result<(), Box<dyn StdError>> {
        let call = resolved(
            Request::builder()
                .model(MODEL)
                .user("Hello")
                .tool(ToolDefinition::custom(
                    "apply_patch",
                    "Applies a patch",
                    json!({ "type": "grammar" }),
                ))
                .build()?,
        )?;

        let error = OpenAiChatCodec
            .encode(&call, false)
            .err()
            .ok_or("expected a custom tool to be rejected")?;

        assert_eq!(error.provider_code(), Some("unsupported_capability"));
        Ok(())
    }

    #[test]
    fn count_tokens_is_unavailable() -> Result<(), Box<dyn StdError>> {
        let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;

        assert!(OpenAiChatCodec.encode_count_tokens(&call).is_none());
        Ok(())
    }

    #[test]
    fn streaming_always_requests_usage() -> Result<(), Box<dyn StdError>> {
        let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;

        let streamed = OpenAiChatCodec.encode(&call, true)?;
        let complete = OpenAiChatCodec.encode(&call, false)?;

        assert_eq!(streamed.body["stream"], json!(true));
        assert_eq!(
            streamed.body["stream_options"],
            json!({ "include_usage": true })
        );
        // A blocking request omits both members entirely; a strict skin may
        // reject an explicit `stream: false`.
        assert_eq!(complete.body.get("stream"), None);
        assert_eq!(complete.body.get("stream_options"), None);
        Ok(())
    }

    #[test]
    fn interleaved_tool_calls_keep_separate_blocks() -> Result<(), Box<dyn StdError>> {
        let events = stream(vec![
            json!({ "id": "chatcmpl-1", "choices": [{ "delta": {
                "tool_calls": [{ "index": 0, "id": "call-a", "function": { "name": "alpha", "arguments": "" } }],
            } }] }),
            json!({ "choices": [{ "delta": {
                "tool_calls": [{ "index": 1, "id": "call-b", "function": { "name": "beta", "arguments": "{\"b\":" } }],
            } }] }),
            json!({ "choices": [{ "delta": {
                "tool_calls": [{ "index": 0, "function": { "arguments": "{\"a\":1}" } }],
            } }] }),
            json!({ "choices": [{ "delta": {
                "tool_calls": [{ "index": 1, "function": { "arguments": "2}" } }],
            } }] }),
            json!({ "choices": [{ "delta": {}, "finish_reason": "tool_calls" }] }),
        ])?;

        let starts: Vec<(String, ContentBlockKind)> = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::ContentBlockStart { id, kind } => {
                    Some((id.as_str().to_owned(), kind.clone()))
                }
                _ => None,
            })
            .collect();
        assert_eq!(starts.len(), 2);
        assert_eq!(starts[0].0, "tool-0");
        assert_eq!(starts[1].0, "tool-1");

        let response = completed(&events)
            .first()
            .copied()
            .ok_or("expected a completed response")?;
        let calls = tool_parts(response);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "call-a");
        assert_eq!(calls[0].name, "alpha");
        assert_eq!(calls[0].arguments, json!({ "a": 1 }));
        assert_eq!(calls[1].id, "call-b");
        assert_eq!(calls[1].arguments, json!({ "b": 2 }));
        Ok(())
    }

    #[test]
    fn reasoning_deltas_produce_a_reasoning_block() -> Result<(), Box<dyn StdError>> {
        let events = stream(vec![
            json!({ "id": "chatcmpl-1", "choices": [{ "delta": { "reasoning": "step " } }] }),
            json!({ "choices": [{ "delta": { "reasoning_content": "one" } }] }),
            json!({ "choices": [{ "delta": { "content": "answer" }, "finish_reason": "stop" }] }),
        ])?;

        let reasoning: String = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::ReasoningDelta { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(reasoning, "step one");

        let blocks: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::ContentBlockStart { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(blocks, ["reasoning-0", "block-0"]);

        let response = completed(&events)
            .first()
            .copied()
            .ok_or("expected a completed response")?;
        let first = response.content.first().ok_or("expected reasoning")?;
        match first {
            ContentPart::Reasoning(reasoning) => assert_eq!(reasoning.text, "step one"),
            other => return Err(format!("expected reasoning, got {other:?}").into()),
        }
        assert_eq!(response.text(), "answer");
        Ok(())
    }

    #[test]
    fn the_final_empty_choices_chunk_carries_usage() -> Result<(), Box<dyn StdError>> {
        let events = stream(vec![
            json!({ "id": "chatcmpl-1", "choices": [{ "delta": { "content": "Hel" } }] }),
            json!({ "choices": [{ "delta": { "content": "lo" }, "finish_reason": "stop" }] }),
            json!({ "choices": [], "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 5,
                "prompt_tokens_details": { "cached_tokens": 3 },
                "cost": 0.0001,
            } }),
        ])?;

        let usage: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::Usage { usage } => Some(*usage),
                _ => None,
            })
            .collect();
        assert_eq!(usage.len(), 1);
        assert_eq!(usage[0].input, 8);
        assert_eq!(usage[0].cache_read, 3);

        let responses = completed(&events);
        assert_eq!(responses.len(), 1);
        let response = responses[0];
        assert_eq!(response.id.as_deref(), Some("chatcmpl-1"));
        assert_eq!(response.text(), "Hello");
        assert_eq!(response.usage.output, 5);
        let cost = response.cost.ok_or("expected a provider cost")?;
        assert_eq!(cost.usd_micros, 100);
        assert_eq!(cost.source, CostSource::Provider);
        // This protocol supplies no terminal response document.
        assert!(response.raw.is_none());
        Ok(())
    }

    #[test]
    fn a_stream_error_chunk_ends_the_stream() -> Result<(), Box<dyn StdError>> {
        let mut decoder = OpenAiChatCodec.stream_decoder(&route()?);

        let error: Error = decoder
            .decode(SseEvent {
                event: None,
                data:  json!({ "error": {
                    "message": "rate limit reached",
                    "code": "rate_limit_exceeded",
                } })
                .to_string(),
            })
            .err()
            .ok_or("expected an error chunk to fail the stream")?;

        assert_eq!(error.provider_code(), Some("rate_limit_exceeded"));
        Ok(())
    }
}
