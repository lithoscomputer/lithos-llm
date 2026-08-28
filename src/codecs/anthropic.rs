//! The Anthropic Messages wire protocol.
//!
//! Anthropic splits work across three endpoints this codec speaks:
//! `/v1/messages` for generation, the same path with `stream: true` for the SSE
//! protocol, and `/v1/messages/count_tokens` for a provider-authoritative input
//! token count.

use std::collections::BTreeMap;

use reqwest::Method;
use serde_json::{Map, Value, json};

use super::assembler::StreamAssembler;
use super::common::{
    endpoint, finish_reason, flattens_system_content, flattens_tool_result_content, merge_options,
    plain_text, reject_unencodable, sampling, system_text, unsupported_capability, wire_options,
};
use super::{Codec, StreamDecoder};
use crate::adapter::ResolvedCall;
use crate::resolver::ResolvedRoute;
use crate::transport::{EncodedRequest, SseEvent, provider_error};
use crate::types::{
    ContentBlockId, ContentBlockKind, ContentPart, Error, ErrorKind, MediaSource, Message,
    ReasoningContent, ReasoningEffort, Response, ResponseFormat, Role, Speed, StreamEvent,
    TokenCounts, ToolCall, ToolCallKind, ToolChoice, ToolDefinition, ToolDefinitionKind,
};

/// The opaque-part namespace this codec owns.
///
/// A [`ContentPart::Opaque`] whose kind starts with this namespace is replayed
/// verbatim; every other namespace is skipped, so a request built for another
/// provider still encodes after failover.
const NAMESPACE: &str = "anthropic";

/// The API version every request declares.
const API_VERSION: &str = "2023-06-01";

/// The `max_tokens` used when the request sets no output limit.
///
/// Anthropic requires the field, so there is no "omit it" option.
const DEFAULT_MAX_TOKENS: u32 = 4096;

/// The only fields `/v1/messages/count_tokens` accepts.
///
/// The count body is a narrowing of the generation body. Everything else the
/// generation request sends — `max_tokens` above all, which is required there
/// and rejected here — is dropped after raw provider options are merged, so a
/// raw option cannot smuggle a generation-only field onto this endpoint.
const COUNT_TOKENS_FIELDS: &[&str] = &[
    "messages",
    "model",
    "system",
    "thinking",
    "tool_choice",
    "tools",
];

/// The Anthropic Messages codec.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AnthropicMessagesCodec;

impl Codec for AnthropicMessagesCodec {
    fn encode(&self, call: &ResolvedCall, stream: bool) -> Result<EncodedRequest, Error> {
        reject_custom_tools(call)?;
        // Anthropic Messages carries no audio. Dropping it silently would let
        // the model answer a prompt the caller never sent.
        reject_unencodable(call.route(), call.request(), |part| {
            matches!(part, ContentPart::Audio(_)).then_some("audio content")
        })?;

        let request = call.request();
        let (options, controls) = wire_options(call);
        let mut body = message_body(call, controls.auto_cache);
        body.insert(
            "max_tokens".to_owned(),
            request
                .max_output_tokens()
                .unwrap_or(DEFAULT_MAX_TOKENS)
                .into(),
        );
        body.insert("stream".to_owned(), stream.into());
        merge_options(&mut body, options);

        let mut encoded = EncodedRequest::new(
            Method::POST,
            endpoint(call.route().provider().base_url(), "/v1/messages"),
            Value::Object(body),
        )
        .with_headers(version_headers())
        .with_timeout(request.timeout());
        // The system field of this protocol takes text only, so anything else
        // a system message carries is dropped. The text still reaches the
        // model, so it is reported rather than refused.
        if flattens_system_content(request) {
            encoded = encoded.unsupported_control("non-text system content");
        }
        if flattens_tool_result_content(request, |part| {
            matches!(part, ContentPart::Text { .. } | ContentPart::Image(_))
        }) {
            encoded = encoded.unsupported_control("non-text tool result content");
        }
        Ok(encoded)
    }

    fn decode_response(&self, route: &ResolvedRoute, value: Value) -> Result<Response, Error> {
        let content = value
            .get("content")
            .and_then(Value::as_array)
            .map(|blocks| blocks.iter().filter_map(decode_block).collect())
            .unwrap_or_default();

        let mut response = Response::new(
            route.provider().id().clone(),
            route.model().id().clone(),
            content,
        );
        response.id = value
            .get("id")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        response.finish_reason = finish_reason(value.get("stop_reason").and_then(Value::as_str));
        response.usage = token_counts(value.get("usage"));
        response.raw = Some(value);
        Ok(response)
    }

    fn stream_decoder(&self, route: &ResolvedRoute) -> Box<dyn StreamDecoder> {
        Box::new(AnthropicStreamDecoder {
            route:     route.clone(),
            assembler: StreamAssembler::new(route),
        })
    }

    fn encode_count_tokens(&self, call: &ResolvedCall) -> Option<Result<EncodedRequest, Error>> {
        Some(count_tokens_request(call))
    }

    fn decode_count_tokens(&self, route: &ResolvedRoute, value: Value) -> Result<u64, Error> {
        let Some(tokens) = value.get("input_tokens").and_then(Value::as_u64) else {
            return Err(Error::new(
                ErrorKind::ResponseDecode,
                "Anthropic returned a token count without an input_tokens field",
            )
            .with_provider(route.provider().id().clone())
            .with_raw_data(value));
        };
        Ok(tokens)
    }
}

/// The dialect headers both endpoints send.
///
/// Authentication is the transport's job; this is the protocol version only.
fn version_headers() -> Vec<(String, String)> {
    vec![("anthropic-version".to_owned(), API_VERSION.to_owned())]
}

/// Rejects a request the Messages API cannot express.
///
/// Anthropic has no custom tool. Encoding one as a function tool would change
/// what the model is asked to produce, so the call fails here rather than
/// silently downgrading. Both a custom tool definition and a replayed custom
/// tool call in the conversation are rejected.
fn reject_custom_tools(call: &ResolvedCall) -> Result<(), Error> {
    let request = call.request();
    let defined = request.tools().iter().any(ToolDefinition::is_custom);
    let called = request
        .messages()
        .iter()
        .flat_map(Message::content)
        .any(|part| {
            matches!(part, ContentPart::ToolCall(tool_call) if tool_call.kind == ToolCallKind::Custom)
        });

    if defined || called {
        return Err(unsupported_capability(call.route(), "custom tools"));
    }
    Ok(())
}

/// Encodes the count-token request, which narrows the generation body.
fn count_tokens_request(call: &ResolvedCall) -> Result<EncodedRequest, Error> {
    reject_custom_tools(call)?;
    // Counting refuses exactly what completion refuses. A request the provider
    // would not accept must not come back with a token count.
    reject_unencodable(call.route(), call.request(), |part| {
        matches!(part, ContentPart::Audio(_)).then_some("audio content")
    })?;

    let (options, controls) = wire_options(call);
    let mut body = message_body(call, controls.auto_cache);
    merge_options(&mut body, options);
    body.retain(|key, _| COUNT_TOKENS_FIELDS.contains(&key.as_str()));

    Ok(EncodedRequest::new(
        Method::POST,
        endpoint(
            call.route().provider().base_url(),
            "/v1/messages/count_tokens",
        ),
        Value::Object(body),
    )
    .with_headers(version_headers())
    .with_timeout(call.request().timeout()))
}

/// Builds every typed field of a Messages body.
///
/// Raw provider options are merged over the result by the caller, so a field
/// encoded here is only a default the application can replace.
fn message_body(call: &ResolvedCall, auto_cache: bool) -> Map<String, Value> {
    let request = call.request();
    let route = call.route();
    // A model that cannot cache would reject the breakpoints outright.
    let cached = auto_cache && route.model().capabilities().caching;

    let mut body = Map::new();
    body.insert("model".to_owned(), route.api_model().into());

    let system = system_text(request.messages());
    if !system.is_empty() {
        body.insert("system".to_owned(), system_value(system, cached));
    }

    let mut messages = wire_messages(request.messages());
    if cached {
        mark_conversation_prefix(&mut messages);
    }
    body.insert(
        "messages".to_owned(),
        messages
            .into_iter()
            .map(WireMessage::into_value)
            .collect::<Vec<_>>()
            .into(),
    );

    if let Some(temperature) = request.temperature() {
        body.insert("temperature".to_owned(), sampling(temperature));
    }
    if let Some(top_p) = request.top_p() {
        body.insert("top_p".to_owned(), sampling(top_p));
    }
    if !request.stop_sequences().is_empty() {
        body.insert("stop_sequences".to_owned(), json!(request.stop_sequences()));
    }
    if !request.metadata().is_empty() {
        body.insert("metadata".to_owned(), json!(request.metadata()));
    }

    let output_config = output_config(call);
    if !output_config.is_empty() {
        body.insert("output_config".to_owned(), output_config.into());
    }
    if let Some(speed) = request.speed() {
        let (key, value) = match speed {
            Speed::Fast => ("speed", "fast"),
            Speed::Balanced => ("service_tier", "auto"),
            Speed::Economical => ("service_tier", "standard_only"),
        };
        body.insert(key.to_owned(), value.into());
    }

    if let Some(tools) = tools_value(request.tools(), cached) {
        body.insert("tools".to_owned(), tools);
    }
    if let Some(choice) = request.tool_choice() {
        body.insert("tool_choice".to_owned(), tool_choice_value(choice));
    }

    body
}

/// The `output_config` object, which carries effort and structured output.
fn output_config(call: &ResolvedCall) -> Map<String, Value> {
    let request = call.request();
    let mut config = Map::new();

    if let Some(effort) = request.reasoning_effort() {
        config.insert("effort".to_owned(), anthropic_effort(effort).into());
    }
    if let Some(schema) = request.response_format().and_then(json_schema) {
        config.insert(
            "format".to_owned(),
            json!({ "type": "json_schema", "schema": schema }),
        );
    }

    config
}

/// The schema an output format asks the model to follow, if it asks for one.
fn json_schema(format: &ResponseFormat) -> Option<Value> {
    match format {
        ResponseFormat::Text => None,
        ResponseFormat::JsonObject => Some(json!({
            "type": "object",
            "additionalProperties": true,
        })),
        ResponseFormat::JsonSchema { schema, .. } => Some(schema.clone()),
    }
}

/// Maps the normalized reasoning effort onto Anthropic's levels.
fn anthropic_effort(effort: ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::Minimal | ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::Xhigh => "max",
    }
}

/// Encodes the tool definitions, marking the last one as a cache breakpoint.
///
/// Every definition is a function tool: a custom tool would have failed the
/// request before it reached here.
fn tools_value(tools: &[ToolDefinition], cached: bool) -> Option<Value> {
    let mut encoded: Vec<Value> = tools
        .iter()
        .filter_map(|tool| match &tool.kind {
            ToolDefinitionKind::Function { input_schema } => Some(json!({
                "name": tool.name,
                "description": tool.description,
                "input_schema": input_schema,
            })),
            ToolDefinitionKind::Custom { .. } => None,
        })
        .collect();

    let last = encoded.last_mut()?;
    if cached {
        mark_cached(last);
    }
    Some(encoded.into())
}

/// Encodes the tool choice.
fn tool_choice_value(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!({ "type": "auto" }),
        ToolChoice::None => json!({ "type": "none" }),
        ToolChoice::Required => json!({ "type": "any" }),
        ToolChoice::Tool { name } => json!({ "type": "tool", "name": name }),
    }
}

/// Encodes the system prompt, as a cached block when caching is on.
///
/// A plain string and a one-element block array mean the same thing to
/// Anthropic; only the block form can carry a cache breakpoint.
fn system_value(text: String, cached: bool) -> Value {
    if !cached {
        return text.into();
    }

    let mut block = json!({ "type": "text", "text": text });
    mark_cached(&mut block);
    json!([block])
}

/// One conversation turn translated into Anthropic content blocks.
struct WireMessage {
    /// Anthropic accepts `user` and `assistant` only.
    role:   &'static str,
    blocks: Vec<Value>,
}

impl WireMessage {
    fn into_value(self) -> Value {
        json!({ "role": self.role, "content": self.blocks })
    }
}

/// Translates the conversation, dropping the turns Anthropic cannot carry.
///
/// System and developer turns are hoisted into `system`, and every remaining
/// non-assistant role — a tool result above all — is a `user` turn. A turn
/// whose parts all encode to nothing is dropped, because Anthropic rejects a
/// message with empty content.
fn wire_messages(messages: &[Message]) -> Vec<WireMessage> {
    let mut wire: Vec<WireMessage> = Vec::new();
    for message in messages
        .iter()
        .filter(|message| !matches!(message.role(), Role::System | Role::Developer))
    {
        let role = match message.role() {
            Role::Assistant => "assistant",
            _ => "user",
        };
        let blocks: Vec<Value> = message.content().iter().filter_map(content_block).collect();
        if blocks.is_empty() {
            continue;
        }
        // This protocol alternates roles. Several canonical messages can map
        // to one wire role — parallel tool results are the common case, since
        // each result is its own message but they all answer one assistant
        // turn — so consecutive same-role messages merge into one turn rather
        // than being sent as a run the provider rejects.
        match wire.last_mut() {
            Some(last) if last.role == role => last.blocks.extend(blocks),
            _ => wire.push(WireMessage { role, blocks }),
        }
    }
    wire
}

/// Encodes the content a tool returned.
///
/// This protocol takes an array of blocks here, not just a string, and it
/// accepts text and image blocks. Encoding them keeps an image a tool produced
/// instead of flattening the result to its text and losing the picture.
///
/// A text-only result stays a plain string, which this protocol also accepts
/// and which is what the overwhelming majority of results are. The block array
/// appears only when a result carries something a string cannot hold, so the
/// common case is unchanged.
///
/// Anything with no block of its own falls back to the flattened text, and
/// `flattens_tool_result_content` reports whatever this cannot carry.
fn tool_result_content(parts: &[ContentPart]) -> Value {
    if parts
        .iter()
        .all(|part| matches!(part, ContentPart::Text { .. }))
    {
        return Value::String(plain_text(parts));
    }

    let blocks: Vec<Value> = parts
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text { text } => Some(json!({ "type": "text", "text": text })),
            ContentPart::Image(image) => Some(json!({
                "type": "image",
                "source": media_source(&image.source),
            })),
            _ => None,
        })
        .collect();
    if blocks.is_empty() {
        return Value::String(plain_text(parts));
    }
    Value::Array(blocks)
}

/// Marks the conversation prefix so the next agent-loop turn reuses it.
///
/// The breakpoint lands on the **second-to-last** user turn. The prefix that
/// ends there is exactly what the previous iteration wrote, so each iteration
/// reads the cache the one before it created instead of paying to write the
/// whole conversation again. A conversation with fewer than two user turns has
/// no reusable prefix yet and gets no breakpoint.
fn mark_conversation_prefix(messages: &mut [WireMessage]) {
    let user_turns: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, message)| message.role == "user")
        .map(|(index, _)| index)
        .collect();

    let Some(turn) = user_turns.len().checked_sub(2).map(|last| user_turns[last]) else {
        return;
    };
    let Some(block) = messages[turn].blocks.last_mut() else {
        return;
    };
    mark_cached(block);
}

/// Adds an ephemeral cache breakpoint to one wire object.
fn mark_cached(value: &mut Value) {
    let Some(object) = value.as_object_mut() else {
        return;
    };
    object.insert("cache_control".to_owned(), json!({ "type": "ephemeral" }));
}

/// Encodes one content part, or `None` when Anthropic has no block for it.
fn content_block(part: &ContentPart) -> Option<Value> {
    match part {
        ContentPart::Text { text } => Some(json!({ "type": "text", "text": text })),
        // Anthropic has no structured-output block, so the JSON travels as the
        // text the model would have written.
        ContentPart::Json { value } => Some(json!({
            "type": "text",
            "text": value.to_string(),
        })),
        ContentPart::Image(image) => Some(json!({
            "type": "image",
            "source": media_source(&image.source),
        })),
        ContentPart::Document(document) => {
            let mut block = json!({
                "type": "document",
                "source": media_source(&document.source),
            });
            if let (Some(object), Some(name)) = (block.as_object_mut(), document.name.as_deref()) {
                object.insert("title".to_owned(), name.into());
            }
            Some(block)
        }
        ContentPart::Reasoning(reasoning) if reasoning.redacted => Some(json!({
            "type": "redacted_thinking",
            "data": reasoning.text,
        })),
        ContentPart::Reasoning(reasoning) => {
            let mut block = json!({ "type": "thinking", "thinking": reasoning.text });
            if let (Some(object), Some(signature)) =
                (block.as_object_mut(), reasoning.signature.as_deref())
            {
                object.insert("signature".to_owned(), signature.into());
            }
            Some(block)
        }
        ContentPart::ToolCall(tool_call) => Some(json!({
            "type": "tool_use",
            "id": tool_call.id,
            "name": tool_call.name,
            "input": tool_call.arguments,
        })),
        ContentPart::ToolResult(result) => Some(json!({
            "type": "tool_result",
            "tool_use_id": result.tool_call_id,
            "content": tool_result_content(&result.content),
            "is_error": result.is_error,
        })),
        // A part this codec produced replays verbatim; one belonging to another
        // provider is skipped so failover still encodes.
        ContentPart::Opaque { kind, data } => match kind.split_once('.') {
            Some((NAMESPACE, _)) => Some(data.clone()),
            Some(_) | None => None,
        },
        // The Messages API has no audio input.
        // Rejected before dispatch by `reject_unencodable`.
        ContentPart::Audio(_) => None,
    }
}

/// Encodes a media source as an Anthropic `source` object.
fn media_source(source: &MediaSource) -> Value {
    match source {
        MediaSource::Url { url } => json!({ "type": "url", "url": url }),
        MediaSource::Base64 { data, media_type } => json!({
            "type": "base64",
            "media_type": media_type,
            "data": data,
        }),
    }
}

/// Decodes one response content block.
fn decode_block(block: &Value) -> Option<ContentPart> {
    match block.get("type").and_then(Value::as_str) {
        Some("text") => Some(ContentPart::Text {
            text: field(block, "text").to_owned(),
        }),
        Some("thinking") => Some(ContentPart::Reasoning(ReasoningContent {
            text:      field(block, "thinking").to_owned(),
            signature: block
                .get("signature")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            redacted:  false,
        })),
        // The encrypted blob is the whole block. It carries no signature and
        // must be replayed exactly as it arrived.
        Some("redacted_thinking") => Some(ContentPart::Reasoning(ReasoningContent {
            text:      field(block, "data").to_owned(),
            signature: None,
            redacted:  true,
        })),
        Some("tool_use") => Some(ContentPart::ToolCall(decode_tool_use(block))),
        // A server-side block this codec does not model still has to survive a
        // replay, so it is kept whole under this codec's namespace.
        Some(kind) => Some(ContentPart::opaque(
            format!("{NAMESPACE}.{kind}"),
            block.clone(),
        )),
        None => None,
    }
}

/// Decodes one `tool_use` block.
///
/// Anthropic sends parsed JSON input rather than an argument string, so
/// `raw_arguments` stays empty unless the wire really did carry a string.
fn decode_tool_use(block: &Value) -> ToolCall {
    let arguments = block.get("input").cloned().unwrap_or_else(|| json!({}));
    let raw_arguments = arguments.as_str().map(ToOwned::to_owned);

    ToolCall {
        id: field(block, "id").to_owned(),
        name: field(block, "name").to_owned(),
        arguments,
        kind: ToolCallKind::Function,
        raw_arguments,
        provider_metadata: BTreeMap::new(),
    }
}

/// Normalizes one Anthropic usage object into the five disjoint buckets.
fn token_counts(usage: Option<&Value>) -> TokenCounts {
    let mut counts = TokenCounts::default();
    if let Some(usage) = usage {
        fold_usage(&mut counts, usage);
    }
    counts
}

/// Folds whichever usage counters one wire object carries into a snapshot.
///
/// Anthropic's counters are **already disjoint**: `input_tokens` excludes both
/// `cache_read_input_tokens` and `cache_creation_input_tokens`, so nothing is
/// subtracted. Thinking tokens are billed inside `output_tokens` with no
/// separate counter, so `reasoning` stays 0 and `billable_output` still adds
/// up. Each field is assigned, not added, because a repeated counter is the
/// provider restating the same cumulative total.
fn fold_usage(counts: &mut TokenCounts, usage: &Value) {
    if let Some(input) = usage.get("input_tokens").and_then(Value::as_u64) {
        counts.input = input;
    }
    if let Some(output) = usage.get("output_tokens").and_then(Value::as_u64) {
        counts.output = output;
    }
    if let Some(read) = usage.get("cache_read_input_tokens").and_then(Value::as_u64) {
        counts.cache_read = read;
    }
    if let Some(write) = usage
        .get("cache_creation_input_tokens")
        .and_then(Value::as_u64)
    {
        counts.cache_write = write;
    }
}

/// One string field of a wire object, or `""` when it is absent.
fn field<'a>(value: &'a Value, key: &str) -> &'a str {
    value.get(key).and_then(Value::as_str).unwrap_or_default()
}

/// The stream-local id of the block one event addresses.
///
/// Anthropic identifies blocks by a top-level `index`, so the id is derived
/// from it. A `tool_use` block's own id is the provider's tool-call id and is
/// kept in [`ContentBlockKind::ToolCall`] instead.
fn block_id(value: &Value) -> ContentBlockId {
    let index = value
        .get("index")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    ContentBlockId::new(format!("block-{index}"))
}

/// Per-stream state for one Anthropic Messages response.
struct AnthropicStreamDecoder {
    route:     ResolvedRoute,
    assembler: StreamAssembler,
}

impl StreamDecoder for AnthropicStreamDecoder {
    fn decode(&mut self, event: SseEvent) -> Result<Vec<StreamEvent>, Error> {
        let value: Value = serde_json::from_str(&event.data).map_err(|source| {
            Error::new(
                ErrorKind::StreamDecode,
                "Anthropic returned an invalid stream event",
            )
            .with_provider(self.route.provider().id().clone())
            .with_source(source)
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
            Some("content_block_delta") => events.extend(self.block_delta(&value)),
            Some("content_block_stop") => events.extend(self.assembler.end(&block_id(&value))),
            Some("message_delta") => {
                if let Some(reason) = value.pointer("/delta/stop_reason").and_then(Value::as_str) {
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
        // here keeps the one-`Completed`-per-successful-stream contract; a
        // stream that already saw `message_stop` gets nothing, because
        // completing is idempotent.
        Ok(self.assembler.complete())
    }
}

impl AnthropicStreamDecoder {
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
                events
            }
            None => Vec::new(),
        }
    }

    /// Applies one `content_block_delta` event.
    fn block_delta(&mut self, value: &Value) -> Vec<StreamEvent> {
        let id = block_id(value);
        let Some(delta) = value.get("delta") else {
            return Vec::new();
        };

        match delta.get("type").and_then(Value::as_str) {
            Some("text_delta") => self.assembler.text(&id, field(delta, "text")),
            Some("thinking_delta") => self.assembler.reasoning(&id, field(delta, "thinking")),
            // A signature is block state, not visible output, so it produces no
            // event of its own.
            Some("signature_delta") => self.assembler.signature(&id, field(delta, "signature")),
            Some("input_json_delta") => self.assembler.arguments(&id, field(delta, "partial_json")),
            _ => Vec::new(),
        }
    }
}

#[cfg(all(test, feature = "builtin-catalog"))]
mod tests {
    use std::error::Error as StdError;

    use serde_json::{Value, json};

    use super::{AnthropicMessagesCodec, Codec};
    use crate::codecs::test_support::resolved;
    use crate::transport::SseEvent;
    use crate::types::{
        ContentPart, ErrorKind, ImageContent, MediaSource, Message, ReasoningContent,
        ReasoningEffort, Request, ResponseFormat, Role, Speed, StreamEvent, ToolChoice,
        ToolDefinition, ToolResult,
    };

    const MODEL: &str = "anthropic/claude-sonnet-4-6";

    /// One SSE frame as the transport would hand it to the decoder.
    fn sse(event: &str, data: &Value) -> SseEvent {
        SseEvent {
            event: Some(event.to_owned()),
            data:  data.to_string(),
        }
    }

    /// Counts every `cache_control` breakpoint anywhere in a body.
    fn breakpoints(value: &Value) -> usize {
        match value {
            Value::Object(object) => {
                let own = usize::from(object.contains_key("cache_control"));
                own + object.values().map(breakpoints).sum::<usize>()
            }
            Value::Array(items) => items.iter().map(breakpoints).sum(),
            _ => 0,
        }
    }

    /// A request with a system prompt, tools, and three user turns.
    fn cacheable_request() -> Result<Request, Box<dyn StdError>> {
        Ok(Request::builder()
            .model(MODEL)
            .system("You are terse.")
            .user("First")
            .message(Message::text(Role::Assistant, "Answer"))
            .user("Second")
            .message(Message::text(Role::Assistant, "Answer"))
            .user("Third")
            .tool(ToolDefinition::function(
                "lookup",
                "Look something up",
                json!({ "type": "object" }),
            ))
            .build()?)
    }

    /// Drives a decoder over a whole transcript, then finishes it.
    fn stream(events: Vec<SseEvent>) -> Result<Vec<StreamEvent>, Box<dyn StdError>> {
        let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;
        let mut decoder = AnthropicMessagesCodec.stream_decoder(call.route());

        let mut decoded = Vec::new();
        for event in events {
            decoded.extend(decoder.decode(event)?);
        }
        decoded.extend(decoder.finish()?);
        Ok(decoded)
    }

    #[test]
    fn encodes_output_controls_and_decodes_tool_calls() -> Result<(), Box<dyn StdError>> {
        let call = resolved(
            Request::builder()
                .model(MODEL)
                .user("Plan a trip")
                .reasoning_effort(ReasoningEffort::High)
                .speed(Speed::Fast)
                .stop_sequence("END")
                .stop_sequence("STOP")
                .metadata_entry("user_id", "u-1")
                .response_format(ResponseFormat::JsonSchema {
                    name:   "trip".to_owned(),
                    schema: json!({ "type": "object" }),
                })
                .build()?,
        )?;
        let codec = AnthropicMessagesCodec;

        let encoded = codec.encode(&call, false)?;

        assert_eq!(encoded.body["output_config"]["effort"], "high");
        assert_eq!(
            encoded.body["output_config"]["format"]["type"],
            "json_schema"
        );
        assert_eq!(encoded.body["speed"], "fast");
        assert_eq!(encoded.body["stop_sequences"], json!(["END", "STOP"]));
        assert_eq!(encoded.body["metadata"], json!({ "user_id": "u-1" }));
        assert_eq!(encoded.body["max_tokens"], 4096);
        assert_eq!(encoded.body["stream"], false);
        assert!(encoded.url.ends_with("/v1/messages"));
        assert!(
            encoded
                .headers
                .contains(&("anthropic-version".to_owned(), "2023-06-01".to_owned()))
        );

        let response = codec.decode_response(
            call.route(),
            json!({
                "id": "msg-1",
                "content": [
                    { "type": "thinking", "thinking": "checked", "signature": "sig" },
                    { "type": "tool_use", "id": "tool-1", "name": "lookup", "input": { "q": "x" } }
                ],
                "stop_reason": "tool_use",
                "usage": { "input_tokens": 10, "output_tokens": 4 }
            }),
        )?;

        assert!(matches!(
            response.content.as_slice(),
            [ContentPart::Reasoning(_), ContentPart::ToolCall(tool_call)]
                if tool_call.id == "tool-1" && tool_call.name == "lookup"
        ));
        assert!(response.raw.is_some());
        assert_eq!(response.cost, None);
        Ok(())
    }

    #[test]
    fn usage_buckets_stay_disjoint_without_subtraction() -> Result<(), Box<dyn StdError>> {
        let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;

        let response = AnthropicMessagesCodec.decode_response(
            call.route(),
            json!({
                "content": [],
                "usage": {
                    "input_tokens": 50,
                    "output_tokens": 1200,
                    "cache_read_input_tokens": 9000,
                    "cache_creation_input_tokens": 1000
                }
            }),
        )?;

        assert_eq!(response.usage.input, 50);
        assert_eq!(response.usage.output, 1200);
        assert_eq!(response.usage.cache_read, 9000);
        assert_eq!(response.usage.cache_write, 1000);
        assert_eq!(response.usage.reasoning, 0);
        assert_eq!(response.usage.total(), 11_250);
        Ok(())
    }

    #[test]
    fn raw_options_win_and_controls_are_consumed() -> Result<(), Box<dyn StdError>> {
        let call = resolved(
            Request::builder()
                .model(MODEL)
                .user("Hello")
                .max_output_tokens(64)
                .provider_option("anthropic", "max_tokens", json!(2048))
                .provider_option("anthropic", "thinking", json!({ "type": "enabled" }))
                .provider_option("anthropic", "auto_cache", json!(false))
                .provider_option("openai", "max_tokens", json!(9))
                .build()?,
        )?;

        let encoded = AnthropicMessagesCodec.encode(&call, false)?;

        assert_eq!(encoded.body["max_tokens"], 2048);
        assert_eq!(encoded.body["thinking"], json!({ "type": "enabled" }));
        assert_eq!(encoded.body.get("auto_cache"), None);
        assert_eq!(breakpoints(&encoded.body), 0);
        Ok(())
    }

    #[test]
    fn auto_cache_places_breakpoints_by_default() -> Result<(), Box<dyn StdError>> {
        let call = resolved(cacheable_request()?)?;

        let encoded = AnthropicMessagesCodec.encode(&call, false)?;

        assert_eq!(
            encoded.body["system"][0]["cache_control"]["type"],
            "ephemeral"
        );
        assert_eq!(
            encoded.body["tools"][0]["cache_control"]["type"],
            "ephemeral"
        );
        // Three user turns interleaved with two assistant turns: the prefix
        // breakpoint lands on the second one, at message index 2.
        assert_eq!(
            encoded.body["messages"][2]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );
        assert_eq!(breakpoints(&encoded.body), 3);
        Ok(())
    }

    #[test]
    fn auto_cache_false_places_no_breakpoints() -> Result<(), Box<dyn StdError>> {
        let call = resolved(
            Request::builder()
                .model(MODEL)
                .system("You are terse.")
                .user("First")
                .message(Message::text(Role::Assistant, "Answer"))
                .user("Second")
                .message(Message::text(Role::Assistant, "Answer"))
                .user("Third")
                .tool(ToolDefinition::function(
                    "lookup",
                    "Look something up",
                    json!({ "type": "object" }),
                ))
                .provider_option("anthropic", "auto_cache", json!(false))
                .build()?,
        )?;

        let encoded = AnthropicMessagesCodec.encode(&call, false)?;

        assert_eq!(breakpoints(&encoded.body), 0);
        assert!(encoded.body["system"].is_string());
        Ok(())
    }

    #[test]
    fn custom_tools_are_rejected_before_dispatch() -> Result<(), Box<dyn StdError>> {
        let call = resolved(
            Request::builder()
                .model(MODEL)
                .user("Patch it")
                .tool(ToolDefinition::custom(
                    "apply_patch",
                    "Apply a patch",
                    json!({ "type": "grammar" }),
                ))
                .build()?,
        )?;

        let Err(error) = AnthropicMessagesCodec.encode(&call, false) else {
            return Err("a custom tool has no Anthropic encoding".into());
        };

        assert_eq!(error.kind(), ErrorKind::InvalidRequest);
        assert_eq!(error.provider_code(), Some("unsupported_capability"));

        let counted = AnthropicMessagesCodec
            .encode_count_tokens(&call)
            .ok_or("Anthropic has a count-tokens endpoint")?;
        let Err(counted) = counted else {
            return Err("the count endpoint must reject a custom tool too".into());
        };
        assert_eq!(counted.provider_code(), Some("unsupported_capability"));
        Ok(())
    }

    #[test]
    fn media_encodes_base64_and_url_sources() -> Result<(), Box<dyn StdError>> {
        let call = resolved(
            Request::builder()
                .model(MODEL)
                .message(Message::new(Role::User, [
                    ContentPart::Image(ImageContent::new(MediaSource::base64("QUJD", "image/png"))),
                    ContentPart::Image(ImageContent::new(MediaSource::url(
                        "https://example.com/cat.png",
                    ))),
                ]))
                .provider_option("anthropic", "auto_cache", json!(false))
                .build()?,
        )?;

        let encoded = AnthropicMessagesCodec.encode(&call, false)?;
        let blocks = &encoded.body["messages"][0]["content"];

        assert_eq!(
            blocks[0]["source"],
            json!({ "type": "base64", "media_type": "image/png", "data": "QUJD" })
        );
        assert_eq!(
            blocks[1]["source"],
            json!({ "type": "url", "url": "https://example.com/cat.png" })
        );
        Ok(())
    }

    #[test]
    fn redacted_thinking_round_trips_through_the_blocking_path() -> Result<(), Box<dyn StdError>> {
        let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;
        let codec = AnthropicMessagesCodec;

        let response = codec.decode_response(
            call.route(),
            json!({
                "content": [{ "type": "redacted_thinking", "data": "ENCRYPTED" }],
                "stop_reason": "end_turn"
            }),
        )?;

        let [ContentPart::Reasoning(reasoning)] = response.content.as_slice() else {
            return Err("expected one reasoning part".into());
        };
        assert!(reasoning.redacted);
        assert_eq!(reasoning.text, "ENCRYPTED");

        let replay = resolved(
            Request::builder()
                .model(MODEL)
                .user("Hello")
                .message(Message::new(Role::Assistant, [ContentPart::Reasoning(
                    ReasoningContent {
                        text:      "ENCRYPTED".to_owned(),
                        signature: None,
                        redacted:  true,
                    },
                )]))
                .provider_option("anthropic", "auto_cache", json!(false))
                .build()?,
        )?;
        let encoded = codec.encode(&replay, false)?;

        assert_eq!(
            encoded.body["messages"][1]["content"][0],
            json!({ "type": "redacted_thinking", "data": "ENCRYPTED" })
        );
        Ok(())
    }

    #[test]
    fn streaming_folds_split_usage_into_one_snapshot() -> Result<(), Box<dyn StdError>> {
        let events = stream(vec![
            sse(
                "message_start",
                &json!({
                    "type": "message_start",
                    "message": {
                        "id": "msg-1",
                        "usage": {
                            "input_tokens": 11,
                            "cache_read_input_tokens": 2,
                            "cache_creation_input_tokens": 1,
                            "output_tokens": 0
                        }
                    }
                }),
            ),
            sse(
                "message_delta",
                &json!({
                    "type": "message_delta",
                    "delta": { "stop_reason": "end_turn" },
                    "usage": { "output_tokens": 5 }
                }),
            ),
            sse("message_stop", &json!({ "type": "message_stop" })),
        ])?;

        let snapshots: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::Usage { usage } => Some(*usage),
                _ => None,
            })
            .collect();

        let [first, last] = snapshots.as_slice() else {
            return Err(format!("expected two usage snapshots, got {snapshots:?}").into());
        };
        assert_eq!(first.input, 11);
        assert_eq!(first.output, 0);
        assert_eq!(last.input, 11);
        assert_eq!(last.cache_read, 2);
        assert_eq!(last.cache_write, 1);
        assert_eq!(last.output, 5);
        assert_eq!(last.reasoning, 0);
        Ok(())
    }

    #[test]
    fn streaming_assembles_blocks_and_completes_once() -> Result<(), Box<dyn StdError>> {
        let events = stream(vec![
            sse(
                "message_start",
                &json!({ "type": "message_start", "message": { "id": "msg-1" } }),
            ),
            sse("ping", &json!({ "type": "ping" })),
            sse(
                "content_block_start",
                &json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": { "type": "thinking", "thinking": "" }
                }),
            ),
            sse(
                "content_block_delta",
                &json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": { "type": "thinking_delta", "thinking": "step" }
                }),
            ),
            sse(
                "content_block_delta",
                &json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": { "type": "signature_delta", "signature": "sig" }
                }),
            ),
            sse(
                "content_block_stop",
                &json!({ "type": "content_block_stop", "index": 0 }),
            ),
            sse(
                "content_block_start",
                &json!({
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": { "type": "tool_use", "id": "toolu_1", "name": "lookup" }
                }),
            ),
            sse(
                "content_block_delta",
                &json!({
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": { "type": "input_json_delta", "partial_json": "{\"q\":" }
                }),
            ),
            sse(
                "content_block_delta",
                &json!({
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": { "type": "input_json_delta", "partial_json": "\"rust\"}" }
                }),
            ),
            sse(
                "content_block_stop",
                &json!({ "type": "content_block_stop", "index": 1 }),
            ),
            sse(
                "message_delta",
                &json!({
                    "type": "message_delta",
                    "delta": { "stop_reason": "tool_use" },
                    "usage": { "output_tokens": 7 }
                }),
            ),
            sse("message_stop", &json!({ "type": "message_stop" })),
        ])?;

        let starts = events
            .iter()
            .filter(|event| matches!(event, StreamEvent::ContentBlockStart { .. }))
            .count();
        let ends = events
            .iter()
            .filter(|event| matches!(event, StreamEvent::ContentBlockEnd { .. }))
            .count();
        let completions: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::Completed { response } => Some(response),
                _ => None,
            })
            .collect();

        assert_eq!(starts, 2);
        assert_eq!(ends, 2);

        let [response] = completions.as_slice() else {
            return Err("expected exactly one completed event".into());
        };
        assert_eq!(response.id.as_deref(), Some("msg-1"));
        assert_eq!(response.raw, None);
        assert_eq!(response.usage.output, 7);

        let [
            ContentPart::Reasoning(reasoning),
            ContentPart::ToolCall(tool_call),
        ] = response.content.as_slice()
        else {
            return Err(format!("unexpected content {:?}", response.content).into());
        };
        assert_eq!(reasoning.text, "step");
        assert_eq!(reasoning.signature.as_deref(), Some("sig"));
        assert_eq!(tool_call.id, "toolu_1");
        assert_eq!(tool_call.name, "lookup");
        assert_eq!(tool_call.arguments, json!({ "q": "rust" }));
        assert_eq!(tool_call.raw_arguments.as_deref(), Some("{\"q\":\"rust\"}"));
        Ok(())
    }

    #[test]
    fn streaming_keeps_redacted_thinking() -> Result<(), Box<dyn StdError>> {
        let events = stream(vec![
            sse(
                "content_block_start",
                &json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": { "type": "redacted_thinking", "data": "ENCRYPTED" }
                }),
            ),
            sse(
                "content_block_stop",
                &json!({ "type": "content_block_stop", "index": 0 }),
            ),
            sse("message_stop", &json!({ "type": "message_stop" })),
        ])?;

        let ended = events.iter().find_map(|event| match event {
            StreamEvent::ContentBlockEnd { part, .. } => Some(part),
            _ => None,
        });

        let Some(ContentPart::Reasoning(reasoning)) = ended else {
            return Err(format!("expected a reasoning block end, got {events:?}").into());
        };
        assert!(reasoning.redacted);
        assert_eq!(reasoning.text, "ENCRYPTED");
        Ok(())
    }

    #[test]
    fn a_stream_error_event_ends_the_stream() -> Result<(), Box<dyn StdError>> {
        let request = Request::builder().model(MODEL).user("Hello").build()?;
        let call = resolved(request)?;
        let mut decoder = AnthropicMessagesCodec.stream_decoder(call.route());

        let error = decoder
            .decode(sse(
                "error",
                &json!({
                    "type": "error",
                    "error": { "type": "overloaded_error", "message": "Overloaded" }
                }),
            ))
            .expect_err("an error event fails the stream");

        assert_eq!(error.provider_code(), Some("overloaded_error"));
        assert!(error.message().contains("Overloaded"));
        Ok(())
    }

    #[test]
    fn count_tokens_narrows_the_body_and_drops_max_tokens() -> Result<(), Box<dyn StdError>> {
        let call = resolved(
            Request::builder()
                .model(MODEL)
                .system("You are terse.")
                .user("Hello")
                .max_output_tokens(256)
                .temperature(0.5)
                .top_p(0.9)
                .stop_sequence("END")
                .metadata_entry("user_id", "u-1")
                .speed(Speed::Fast)
                .reasoning_effort(ReasoningEffort::High)
                .tool(ToolDefinition::function(
                    "lookup",
                    "Look something up",
                    json!({ "type": "object" }),
                ))
                .tool_choice(ToolChoice::Auto)
                .provider_option("anthropic", "thinking", json!({ "type": "enabled" }))
                .build()?,
        )?;
        let codec = AnthropicMessagesCodec;

        let encoded = codec
            .encode_count_tokens(&call)
            .ok_or("Anthropic has a count-tokens endpoint")??;

        let body = encoded
            .body
            .as_object()
            .ok_or("the count body must be an object")?;
        let mut keys: Vec<&str> = body.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, [
            "messages",
            "model",
            "system",
            "thinking",
            "tool_choice",
            "tools"
        ]);
        assert!(encoded.url.ends_with("/v1/messages/count_tokens"));
        assert!(
            encoded
                .headers
                .contains(&("anthropic-version".to_owned(), "2023-06-01".to_owned()))
        );

        let tokens = codec.decode_count_tokens(call.route(), json!({ "input_tokens": 123 }))?;
        assert_eq!(tokens, 123);
        Ok(())
    }

    #[test]
    fn a_tool_result_image_becomes_a_block_and_does_not_warn() -> Result<(), Box<dyn StdError>> {
        // `tool_result.content` takes an array of blocks here, so an image a
        // tool produced survives instead of being flattened away.
        let request = Request::builder()
            .model(MODEL)
            .user("Chart it.")
            .message(Message::new(Role::Tool, [ContentPart::ToolResult(
                ToolResult {
                    tool_call_id: "call-1".to_owned(),
                    name:         Some("chart".to_owned()),
                    content:      vec![
                        ContentPart::Text {
                            text: "Revenue by quarter.".to_owned(),
                        },
                        ContentPart::Image(ImageContent::new(MediaSource::base64(
                            "aW1n",
                            "image/png",
                        ))),
                    ],
                    is_error:     false,
                },
            )]))
            .build()?;

        let encoded = AnthropicMessagesCodec.encode(&resolved(request)?, false)?;

        let blocks = &encoded.body["messages"][0]["content"][1]["content"];
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[1]["type"], "image");
        assert_eq!(blocks[1]["source"]["data"], "aW1n");
        assert!(
            encoded.warnings.is_empty(),
            "content this codec carries must not warn"
        );
        Ok(())
    }
}
