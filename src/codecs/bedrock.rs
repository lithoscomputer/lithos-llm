//! The Amazon Bedrock Converse wire protocol.
//!
//! One codec serves both Bedrock authentication schemes and both call shapes.
//! Bearer and SigV4 send the identical body to the identical URL, and the
//! unary, streaming, and token-count paths differ only in the operation
//! segment of the path, so nothing here knows how the request is signed.
//!
//! The streaming half is fed AWS `vnd.amazon.eventstream` frames that
//! [`event_stream`](crate::transport::event_stream) has already decoded. The
//! transport hands each frame over as an [`SseEvent`] whose `event` is the
//! frame's `:event-type` header and whose `data` is the event JSON.

use reqwest::Method;
use serde_json::{Map, Value, json};

use super::assembler::StreamAssembler;
use super::common::{
    endpoint, finish_reason, flattens_tool_result_content, merge_options, plain_text,
    reject_unencodable, sampling, system_text, unsupported_capability, wire_options,
};
use super::{Codec, StreamDecoder};
use crate::adapter::ResolvedCall;
use crate::catalog::ProviderId;
use crate::resolver::ResolvedRoute;
use crate::transport::{EncodedRequest, SseEvent, classify};
use crate::types::{
    ContentBlockId, ContentBlockKind, ContentPart, Error, ErrorKind, FinishReason, MediaSource,
    ReasoningContent, ReasoningEffort, Request, Response, Role, Speed, StreamEvent, TokenCounts,
    ToolCall, ToolCallKind, ToolChoice, ToolDefinition, ToolDefinitionKind, ToolResult,
};

/// The opaque replay namespace this codec claims.
const NAMESPACE: &str = "bedrock";

#[derive(Clone, Copy, Debug)]
pub(crate) struct BedrockConverseCodec;

impl Codec for BedrockConverseCodec {
    fn encode(&self, call: &ResolvedCall, stream: bool) -> Result<EncodedRequest, Error> {
        let request = call.request();
        let route = call.route();
        reject_custom_tools(request, route)?;
        // Bedrock Converse carries no audio. Dropping it silently would let
        // the model answer a prompt the caller never sent.
        reject_unencodable(route, request, |part| {
            matches!(part, ContentPart::Audio(_)).then_some("audio content")
        })?;
        let (options, controls) = wire_options(call);

        let mut body = Map::new();
        let system = system_blocks(request, controls.auto_cache);
        if !system.is_empty() {
            body.insert("system".to_owned(), Value::Array(system));
        }
        body.insert(
            "messages".to_owned(),
            Value::Array(conversation(request, route, controls.auto_cache)?),
        );

        let inference = inference_config(request);
        if !inference.is_empty() {
            body.insert("inferenceConfig".to_owned(), Value::Object(inference));
        }
        if let Some(tool_config) = tool_config(request, controls.auto_cache) {
            body.insert("toolConfig".to_owned(), tool_config);
        }
        if let Some(effort) = request.reasoning_effort() {
            body.insert(
                "additionalModelRequestFields".to_owned(),
                json!({ "output_config": { "effort": bedrock_effort(effort) } }),
            );
        }
        if let Some(speed) = request.speed() {
            let latency = if matches!(speed, Speed::Fast) {
                "optimized"
            } else {
                "standard"
            };
            body.insert(
                "performanceConfig".to_owned(),
                json!({ "latency": latency }),
            );
        }

        merge_options(&mut body, options);

        let operation = if stream {
            "converse-stream"
        } else {
            "converse"
        };
        let mut encoded = EncodedRequest::new(
            Method::POST,
            operation_url(route, operation),
            Value::Object(body),
        )
        .with_timeout(request.timeout());
        // Converse has no request-metadata field, so the map is reported rather
        // than folded into some other field where it would change the prompt.
        if !request.metadata().is_empty() {
            encoded = encoded.unsupported_control("request metadata");
        }
        // Converse has no portable structured-output field. A caller who asked
        // for JSON gets prose, so say so rather than letting them discover it
        // by parsing.
        if request.response_format().is_some() {
            encoded = encoded.unsupported_control("response formats");
        }
        if flattens_tool_result_content(request) {
            encoded = encoded.unsupported_control("non-text tool result content");
        }
        Ok(encoded)
    }

    fn decode_response(&self, route: &ResolvedRoute, value: Value) -> Result<Response, Error> {
        let content = value
            .pointer("/output/message/content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(decode_content_block)
            .collect();

        let mut response = Response::new(
            route.provider().id().clone(),
            route.model().id().clone(),
            content,
        );
        response.finish_reason = stop_reason(value.get("stopReason").and_then(Value::as_str));
        response.usage = token_counts(value.get("usage"));
        response.raw = Some(value);
        Ok(response)
    }

    fn stream_decoder(&self, route: &ResolvedRoute) -> Box<dyn StreamDecoder> {
        Box::new(BedrockStreamDecoder {
            provider:  route.provider().id().clone(),
            assembler: StreamAssembler::new(route),
        })
    }

    fn encode_count_tokens(&self, call: &ResolvedCall) -> Option<Result<EncodedRequest, Error>> {
        Some(encode_count_tokens(call))
    }

    fn decode_count_tokens(&self, route: &ResolvedRoute, value: Value) -> Result<u64, Error> {
        value
            .get("inputTokens")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::ResponseDecode,
                    "Bedrock returned a count-tokens body without an inputTokens count",
                )
                .with_provider(route.provider().id().clone())
                .with_raw_data(value)
            })
    }
}

/// Encodes the Bedrock `CountTokens` request for one call.
///
/// The body carries the Converse messages and system blocks and nothing else:
/// `inferenceConfig`, `toolConfig`, and the raw provider options all shape
/// generation rather than the prompt, and `CountTokens` rejects them. Cache
/// points stay, because the body must describe the same prompt the Converse
/// call would send.
fn encode_count_tokens(call: &ResolvedCall) -> Result<EncodedRequest, Error> {
    let request = call.request();
    let route = call.route();
    reject_custom_tools(request, route)?;
    // Counting refuses exactly what completion refuses. A request the provider
    // would not accept must not come back with a token count.
    reject_unencodable(route, request, |part| {
        matches!(part, ContentPart::Audio(_)).then_some("audio content")
    })?;
    let (_, controls) = wire_options(call);

    let mut converse = Map::new();
    converse.insert(
        "messages".to_owned(),
        Value::Array(conversation(request, route, controls.auto_cache)?),
    );
    let system = system_blocks(request, controls.auto_cache);
    if !system.is_empty() {
        converse.insert("system".to_owned(), Value::Array(system));
    }

    Ok(EncodedRequest::new(
        Method::POST,
        operation_url(route, "count-tokens"),
        json!({ "input": { "converse": Value::Object(converse) } }),
    )
    .with_timeout(request.timeout()))
}

/// The URL of one Bedrock runtime operation for a route.
///
/// The model id is percent-encoded before it is interpolated into the path.
/// The reference implementation interpolates it raw, which splits an ARN-style
/// inference-profile id (`arn:aws:bedrock:…:inference-profile/name`) into extra
/// path segments and makes the call unroutable. Encoding here is a deliberate
/// difference: the signer signs the encoded URL, so client and server still
/// apply exactly one encoding pass to the same bytes.
fn operation_url(route: &ResolvedRoute, operation: &str) -> String {
    let model = route.api_model().replace('%', "%25").replace('/', "%2F");
    endpoint(
        route.provider().base_url(),
        &format!("/model/{model}/{operation}"),
    )
}

/// Rejects a request carrying a tool Converse cannot express.
///
/// Converse has function tools only. A custom tool is refused before dispatch
/// rather than downgraded, so a caller never silently loses its grammar.
fn reject_custom_tools(request: &Request, route: &ResolvedRoute) -> Result<(), Error> {
    if request.tools().iter().any(ToolDefinition::is_custom) {
        return Err(unsupported_capability(route, "custom tools"));
    }
    Ok(())
}

/// The `cachePoint` marker that opens a prompt-cache breakpoint.
fn cache_point() -> Value {
    json!({ "cachePoint": { "type": "default" } })
}

/// The system blocks, with a cache point after the prompt when caching is on.
fn system_blocks(request: &Request, auto_cache: bool) -> Vec<Value> {
    let system = system_text(request.messages());
    if system.is_empty() {
        return Vec::new();
    }

    let mut blocks = vec![json!({ "text": system })];
    if auto_cache {
        blocks.push(cache_point());
    }
    blocks
}

/// The conversation messages, with the prefix cache point already placed.
///
/// System and developer messages are excluded; they became the `system`
/// blocks. A message whose every part is unencodable is dropped, because
/// Converse rejects a message with an empty content list.
fn conversation(
    request: &Request,
    route: &ResolvedRoute,
    auto_cache: bool,
) -> Result<Vec<Value>, Error> {
    let mut messages: Vec<Value> = Vec::new();
    for message in request.messages() {
        if matches!(message.role(), Role::System | Role::Developer) {
            continue;
        }
        let mut blocks = Vec::new();
        for part in message.content() {
            if let Some(block) = encode_content_part(part, route)? {
                blocks.push(block);
            }
        }
        // A tool message whose result rides on the message rather than in a
        // `ToolResult` part still has to reach the wire as a `toolResult`.
        if blocks.is_empty() && message.role() == Role::Tool {
            if let Some(tool_call_id) = message.tool_call_id() {
                blocks.push(tool_result_block(
                    tool_call_id,
                    vec![json!({ "text": plain_text(message.content()) })],
                    false,
                ));
            }
        }
        if blocks.is_empty() {
            continue;
        }
        let role = if message.role() == Role::Assistant {
            "assistant"
        } else {
            // Converse has no tool role; tool results ride in user messages.
            "user"
        };
        // Converse alternates roles. Parallel tool results arrive as one
        // canonical message each but all answer a single assistant turn, so
        // consecutive same-role messages merge into one turn rather than
        // being sent as a run the provider rejects.
        match messages.last_mut() {
            Some(last) if last.get("role").and_then(Value::as_str) == Some(role) => {
                if let Some(Value::Array(content)) = last.get_mut("content") {
                    content.extend(blocks);
                }
            }
            _ => messages.push(json!({ "role": role, "content": blocks })),
        }
    }

    if auto_cache {
        place_conversation_cache_point(&mut messages);
    }
    Ok(messages)
}

/// Appends a cache point to the second-to-last user turn.
///
/// Placing it one turn back means each iteration of an agent loop reuses the
/// prefix the previous iteration wrote, instead of paying to write a prefix
/// that the next turn immediately invalidates. This is the same placement the
/// Anthropic codec uses.
fn place_conversation_cache_point(messages: &mut [Value]) {
    let user_turns: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, message)| message.get("role").and_then(Value::as_str) == Some("user"))
        .map(|(index, _)| index)
        .collect();

    let Some(target) = user_turns.len().checked_sub(2).map(|nth| user_turns[nth]) else {
        return;
    };
    if let Some(content) = messages[target]
        .get_mut("content")
        .and_then(Value::as_array_mut)
    {
        content.push(cache_point());
    }
}

/// Encodes one content part as a Converse content block.
///
/// `Ok(None)` is a part this protocol has no block for — audio, structured
/// JSON outside a tool result, and another provider's opaque replay data. Those
/// are skipped so a conversation built for one provider still sends after
/// failover.
///
/// # Errors
///
/// Returns [`ErrorKind::InvalidRequest`] for media held behind a URL. Converse
/// has no URL media source and this crate does not fetch remote media on the
/// caller's behalf.
fn encode_content_part(part: &ContentPart, route: &ResolvedRoute) -> Result<Option<Value>, Error> {
    let block = match part {
        ContentPart::Text { text } if text.is_empty() => None,
        ContentPart::Text { text } => Some(json!({ "text": text })),
        ContentPart::Image(image) => {
            let (data, media_type) = media_bytes(&image.source, route)?;
            let format = media_format(media_type, "png")
                .ok_or_else(|| unsupported_media_type(route, media_type))?;
            Some(json!({
                "image": {
                    "format": format,
                    "source": { "bytes": data },
                }
            }))
        }
        ContentPart::Document(document) => {
            let (data, media_type) = media_bytes(&document.source, route)?;
            let format = media_format(media_type, "pdf")
                .ok_or_else(|| unsupported_media_type(route, media_type))?;
            Some(json!({
                "document": {
                    "format": format,
                    "name": document.name.as_deref().unwrap_or("document"),
                    "source": { "bytes": data },
                }
            }))
        }
        ContentPart::Reasoning(reasoning) => Some(encode_reasoning(reasoning)),
        ContentPart::ToolCall(call) => Some(encode_tool_call(call)),
        ContentPart::ToolResult(result) => Some(encode_tool_result(result, route)?),
        // Bedrock's own replay data goes back on the wire verbatim; another
        // provider's is skipped so a failed-over conversation still sends.
        ContentPart::Opaque { data, .. } if part.opaque_namespace() == Some(NAMESPACE) => {
            Some(data.clone())
        }
        // Structured JSON has no Converse block of its own, so it rides as
        // text the way every other dialect encodes it.
        ContentPart::Json { value } => Some(json!({ "text": value.to_string() })),
        // Audio is rejected before dispatch. An opaque part in another
        // provider's namespace is skipped so failover can still send.
        ContentPart::Audio(_) | ContentPart::Opaque { .. } => None,
    };
    Ok(block)
}

/// The base64 payload and declared media type of an inline media source.
///
/// # Errors
///
/// Returns [`ErrorKind::InvalidRequest`] for a URL source.
fn media_bytes<'a>(
    source: &'a MediaSource,
    route: &ResolvedRoute,
) -> Result<(&'a str, Option<&'a str>), Error> {
    let Some(data) = source.base64_data() else {
        return Err(unsupported_capability(route, "URL media"));
    };
    Ok((data, source.media_type()))
}

/// Encodes a reasoning part, keeping a redacted block opaque.
///
/// Bedrock validates the signature it issued, so it is echoed back unchanged.
/// A redacted block carries the provider's sealed payload instead of text and
/// has no signature of its own.
fn encode_reasoning(reasoning: &ReasoningContent) -> Value {
    if reasoning.redacted {
        return json!({ "reasoningContent": { "redactedContent": reasoning.text } });
    }

    let mut text_block = Map::new();
    text_block.insert("text".to_owned(), json!(reasoning.text));
    if let Some(signature) = &reasoning.signature {
        text_block.insert("signature".to_owned(), json!(signature));
    }
    json!({ "reasoningContent": { "reasoningText": Value::Object(text_block) } })
}

/// Encodes a tool call as a `toolUse` block.
///
/// Converse requires `toolUse.input` to be a JSON object document and rejects
/// a no-argument call whose input is null, so any other value becomes `{}`.
fn encode_tool_call(call: &ToolCall) -> Value {
    let input = match &call.arguments {
        Value::Object(_) => call.arguments.clone(),
        _ => json!({}),
    };
    json!({
        "toolUse": { "toolUseId": call.id, "name": call.name, "input": input }
    })
}

/// Encodes a tool result as a `toolResult` block.
fn encode_tool_result(result: &ToolResult, route: &ResolvedRoute) -> Result<Value, Error> {
    let mut blocks = Vec::new();
    for part in &result.content {
        match part {
            ContentPart::Json { value } => blocks.push(json!({ "json": value })),
            other => {
                if let Some(block) = encode_content_part(other, route)? {
                    blocks.push(block);
                }
            }
        }
    }
    if blocks.is_empty() {
        blocks.push(json!({ "text": plain_text(&result.content) }));
    }
    Ok(tool_result_block(
        &result.tool_call_id,
        blocks,
        result.is_error,
    ))
}

/// Builds a `toolResult` block from already encoded content blocks.
fn tool_result_block(tool_call_id: &str, content: Vec<Value>, is_error: bool) -> Value {
    let mut block = Map::new();
    block.insert("toolUseId".to_owned(), json!(tool_call_id));
    block.insert("content".to_owned(), Value::Array(content));
    if is_error {
        block.insert("status".to_owned(), json!("error"));
    }
    json!({ "toolResult": Value::Object(block) })
}

/// Maps a media type onto Bedrock's media `format` enum.
/// Maps a declared media type onto the Converse format enum.
///
/// `None` means the source declared nothing, and the caller's `default` stands
/// in. A declared type this protocol has no enum member for returns `None`,
/// because labelling the bytes with a format they are not would tell the
/// provider something the caller never said — `image/heic` sent as `png`.
/// Bedrock is the only dialect that must map onto an enum; the others pass the
/// declared type through verbatim, so only this one can misdescribe content.
/// Refuses media whose declared type has no Converse format.
fn unsupported_media_type(route: &ResolvedRoute, media_type: Option<&str>) -> Error {
    unsupported_capability(
        route,
        &format!("the media type {}", media_type.unwrap_or("<undeclared>")),
    )
}

fn media_format<'a>(media_type: Option<&'a str>, default: &'a str) -> Option<&'a str> {
    let Some(media_type) = media_type else {
        return Some(default);
    };
    let format = match Some(media_type) {
        Some("image/png") => "png",
        Some("image/jpeg" | "image/jpg") => "jpeg",
        Some("image/gif") => "gif",
        Some("image/webp") => "webp",
        Some("application/pdf") => "pdf",
        Some("text/plain") => "txt",
        Some("text/markdown") => "md",
        Some("text/html") => "html",
        Some("text/csv") => "csv",
        Some(
            "application/msword"
            | "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ) => "docx",
        Some(
            "application/vnd.ms-excel"
            | "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ) => "xlsx",
        _ => return None,
    };
    Some(format)
}

/// The `inferenceConfig` object, which is omitted when it would be empty.
fn inference_config(request: &Request) -> Map<String, Value> {
    let mut inference = Map::new();
    if let Some(max_tokens) = request.max_output_tokens() {
        inference.insert("maxTokens".to_owned(), max_tokens.into());
    }
    if let Some(temperature) = request.temperature() {
        inference.insert("temperature".to_owned(), sampling(temperature));
    }
    if let Some(top_p) = request.top_p() {
        inference.insert("topP".to_owned(), sampling(top_p));
    }
    if !request.stop_sequences().is_empty() {
        inference.insert("stopSequences".to_owned(), json!(request.stop_sequences()));
    }
    inference
}

/// The `toolConfig` object, with a cache point after the last tool.
///
/// Converse has no "call no tool" choice, so `ToolChoice::None` drops the whole
/// tool configuration: withholding the tools is the only faithful encoding.
fn tool_config(request: &Request, auto_cache: bool) -> Option<Value> {
    if request.tools().is_empty() || matches!(request.tool_choice(), Some(ToolChoice::None)) {
        return None;
    }

    let mut entries: Vec<Value> = request
        .tools()
        .iter()
        .map(|tool| {
            let input_schema = match &tool.kind {
                ToolDefinitionKind::Function { input_schema } => tool_input_schema(input_schema),
                // Custom tools are rejected before dispatch; this arm keeps the
                // match exhaustive without inventing an encoding.
                ToolDefinitionKind::Custom { .. } => tool_input_schema(&Value::Null),
            };
            json!({
                "toolSpec": {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": { "json": input_schema },
                }
            })
        })
        .collect();
    if auto_cache {
        entries.push(cache_point());
    }

    let mut config = Map::new();
    config.insert("tools".to_owned(), Value::Array(entries));
    match request.tool_choice() {
        Some(ToolChoice::Required) => {
            config.insert("toolChoice".to_owned(), json!({ "any": {} }));
        }
        Some(ToolChoice::Tool { name }) => {
            config.insert("toolChoice".to_owned(), json!({ "tool": { "name": name } }));
        }
        // `auto` is the wire default and `none` dropped the config above.
        Some(ToolChoice::Auto | ToolChoice::None) | None => {}
    }
    Some(Value::Object(config))
}

/// Normalizes a tool schema for `toolSpec.inputSchema.json`.
///
/// Converse validates the schema strictly and requires a top-level `type`.
/// Loose schemas — a bare `{}` for a no-argument tool, or one that leaves the
/// type implicit — are accepted by some model families and rejected by others,
/// so the object type is filled in rather than left out.
fn tool_input_schema(schema: &Value) -> Value {
    match schema {
        Value::Object(map) => {
            let mut map = map.clone();
            map.entry("type").or_insert_with(|| json!("object"));
            Value::Object(map)
        }
        _ => json!({ "type": "object", "properties": {} }),
    }
}

fn bedrock_effort(effort: ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::Minimal | ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::Xhigh => "max",
    }
}

/// Maps a Converse `stopReason` onto the normalized finish reason.
fn stop_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("stop_sequence") => FinishReason::Stop,
        Some("content_filtered" | "guardrail_intervened") => FinishReason::ContentFilter,
        other => finish_reason(other),
    }
}

/// Reads a Converse usage object into the five disjoint buckets.
///
/// Converse counters are already disjoint: `inputTokens` excludes both cache
/// counters, so nothing is subtracted. Reasoning tokens are folded into
/// `outputTokens` with no separate counter, so `reasoning` stays zero rather
/// than guessing at a split.
fn token_counts(usage: Option<&Value>) -> TokenCounts {
    let Some(usage) = usage else {
        return TokenCounts::default();
    };
    let count = |key: &str| usage.get(key).and_then(Value::as_u64).unwrap_or_default();
    TokenCounts {
        input:       count("inputTokens"),
        output:      count("outputTokens"),
        reasoning:   0,
        cache_read:  count("cacheReadInputTokens"),
        cache_write: count("cacheWriteInputTokens"),
    }
}

/// Decodes one Converse content block.
///
/// Unknown block kinds are skipped, because the Converse content union grows
/// with every model family.
fn decode_content_block(block: &Value) -> Option<ContentPart> {
    if let Some(text) = block.get("text").and_then(Value::as_str) {
        if text.is_empty() {
            return None;
        }
        return Some(ContentPart::Text {
            text: text.to_owned(),
        });
    }
    if let Some(tool_use) = block.get("toolUse") {
        let id = tool_use.get("toolUseId").and_then(Value::as_str)?;
        let name = tool_use.get("name").and_then(Value::as_str)?;
        // A no-argument call is canonically `{}`, so it re-encodes to a valid
        // `toolUse.input` document.
        let arguments = match tool_use.get("input") {
            None | Some(Value::Null) => json!({}),
            Some(value) => value.clone(),
        };
        return Some(ContentPart::ToolCall(ToolCall::function(
            id, name, arguments,
        )));
    }
    if let Some(reasoning) = block.get("reasoningContent") {
        return decode_reasoning_block(reasoning);
    }
    None
}

/// Decodes a `reasoningContent` block, redacted or not.
fn decode_reasoning_block(reasoning: &Value) -> Option<ContentPart> {
    if let Some(text_block) = reasoning.get("reasoningText") {
        return Some(ContentPart::Reasoning(ReasoningContent {
            text:      text_block
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            signature: text_block
                .get("signature")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            redacted:  false,
        }));
    }
    let redacted = reasoning.get("redactedContent").and_then(Value::as_str)?;
    Some(ContentPart::Reasoning(ReasoningContent {
        text:      redacted.to_owned(),
        signature: None,
        redacted:  true,
    }))
}

/// Decodes one Bedrock ConverseStream.
struct BedrockStreamDecoder {
    provider:  ProviderId,
    assembler: StreamAssembler,
}

impl StreamDecoder for BedrockStreamDecoder {
    fn decode(&mut self, event: SseEvent) -> Result<Vec<StreamEvent>, Error> {
        let value: Value = serde_json::from_str(&event.data).map_err(|source| {
            Error::new(
                ErrorKind::StreamDecode,
                "Bedrock returned an invalid stream event",
            )
            .with_provider(self.provider.clone())
            .with_source(source)
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
            "contentBlockDelta" => Ok(self.content_block_delta(payload)),
            "contentBlockStop" => Ok(self.assembler.end(&block_id(payload))),
            "messageStop" => {
                let reason = payload.get("stopReason").and_then(Value::as_str);
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
        self.assembler.start(block_id(payload), kind)
    }

    fn content_block_delta(&mut self, payload: &Value) -> Vec<StreamEvent> {
        let id = block_id(payload);
        let Some(delta) = payload.get("delta") else {
            return Vec::new();
        };

        let mut events = Vec::new();
        if let Some(text) = delta.get("text").and_then(Value::as_str) {
            events.extend(self.assembler.text(&id, text));
        }
        if let Some(chunk) = delta.pointer("/toolUse/input").and_then(Value::as_str) {
            events.extend(self.assembler.arguments(&id, chunk));
        }
        if let Some(reasoning) = delta.get("reasoningContent") {
            events.extend(self.reasoning_delta(&id, reasoning));
        }
        events
    }

    /// Applies one reasoning delta.
    ///
    /// The streaming members are flat, unlike the nested `reasoningText` block
    /// the request side uses. A redacted block arrives as a sealed
    /// `redactedContent` payload and no text; it is kept as the block's text so
    /// the assembled part re-encodes to the `redactedContent` Bedrock expects
    /// on the next turn.
    fn reasoning_delta(&mut self, id: &ContentBlockId, reasoning: &Value) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        if let Some(text) = reasoning.get("text").and_then(Value::as_str) {
            events.extend(self.assembler.reasoning(id, text));
        }
        if let Some(signature) = reasoning.get("signature").and_then(Value::as_str) {
            events.extend(self.assembler.signature(id, signature));
        }
        if let Some(sealed) = reasoning.get("redactedContent").and_then(Value::as_str) {
            events.extend(self.assembler.set_redacted(id));
            events.extend(self.assembler.reasoning(id, sealed));
        }
        events
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
        .with_provider(self.provider.clone())
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

#[cfg(all(test, feature = "builtin-catalog"))]
mod tests {
    use std::error::Error as StdError;

    use serde_json::{Value, json};

    use super::{BedrockConverseCodec, Codec};
    use crate::codecs::test_support::resolved;
    use crate::transport::SseEvent;
    use crate::types::{
        ContentPart, DocumentContent, ErrorKind, FinishReason, ImageContent, MediaSource, Message,
        ReasoningContent, ReasoningEffort, Request, Response, ResponseFormat, Role, Speed,
        StreamEvent, ToolDefinition, ToolResult,
    };

    const MODEL: &str = "bedrock/anthropic.claude-sonnet-4-6";

    fn warnings_for(request: Request) -> Result<Vec<String>, Box<dyn StdError>> {
        Ok(BedrockConverseCodec
            .encode(&resolved(request)?, false)?
            .warnings
            .into_iter()
            .map(|warning| warning.code)
            .collect())
    }

    fn encoded(request: Request) -> Result<Value, Box<dyn StdError>> {
        Ok(BedrockConverseCodec
            .encode(&resolved(request)?, false)?
            .body)
    }

    fn decoded(body: Value) -> Result<Response, Box<dyn StdError>> {
        let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;
        Ok(BedrockConverseCodec.decode_response(call.route(), body)?)
    }

    /// Drives one whole stream, returning every event in order.
    fn streamed(frames: &[(&str, Value)]) -> Result<Vec<StreamEvent>, Box<dyn StdError>> {
        let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;
        let mut decoder = BedrockConverseCodec.stream_decoder(call.route());
        let mut events = Vec::new();
        for (name, payload) in frames {
            events.extend(decoder.decode(SseEvent {
                event: Some((*name).to_owned()),
                data:  payload.to_string(),
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

    #[test]
    fn encodes_document_reasoning_and_performance_fields() -> Result<(), Box<dyn StdError>> {
        let body = encoded(
            Request::builder()
                .model(MODEL)
                .message(Message::new(Role::Assistant, [ContentPart::Reasoning(
                    ReasoningContent {
                        text:      "checked".to_owned(),
                        signature: Some("sig".to_owned()),
                        redacted:  false,
                    },
                )]))
                .message(Message::new(Role::User, [ContentPart::Document(
                    DocumentContent {
                        source: MediaSource::parse("data:application/pdf;base64,aGVsbG8="),
                        name:   Some("source".to_owned()),
                    },
                )]))
                .reasoning_effort(ReasoningEffort::High)
                .speed(Speed::Fast)
                .build()?,
        )?;

        assert_eq!(
            body["messages"][0]["content"][0]["reasoningContent"]["reasoningText"]["signature"],
            "sig"
        );
        assert_eq!(
            body["messages"][1]["content"][0]["document"]["format"],
            "pdf"
        );
        assert_eq!(
            body["messages"][1]["content"][0]["document"]["name"],
            "source"
        );
        assert_eq!(
            body["messages"][1]["content"][0]["document"]["source"]["bytes"],
            "aGVsbG8="
        );
        assert_eq!(body["performanceConfig"]["latency"], "optimized");
        assert_eq!(
            body["additionalModelRequestFields"]["output_config"]["effort"],
            "high"
        );
        Ok(())
    }

    #[test]
    fn percent_encodes_the_model_id_in_the_path() -> Result<(), Box<dyn StdError>> {
        let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;

        let unary = BedrockConverseCodec.encode(&call, false)?;
        let streaming = BedrockConverseCodec.encode(&call, true)?;

        assert_eq!(
            unary.url,
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-sonnet-4-6/converse"
        );
        assert_eq!(
            streaming.url,
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-sonnet-4-6/converse-stream"
        );
        Ok(())
    }

    #[test]
    fn percent_encodes_an_arn_model_id() -> Result<(), Box<dyn StdError>> {
        let call = resolved(
            Request::builder()
                .model("bedrock/arn:aws:bedrock:us-east-1:1:inference-profile/custom")
                .user("Hello")
                .build()?,
        )?;

        let encoded = BedrockConverseCodec.encode(&call, false)?;

        assert!(
            encoded.url.ends_with(
                "/model/arn:aws:bedrock:us-east-1:1:inference-profile%2Fcustom/converse"
            ),
            "{}",
            encoded.url
        );
        Ok(())
    }

    #[test]
    fn usage_buckets_stay_disjoint_without_subtraction() -> Result<(), Box<dyn StdError>> {
        let response = decoded(json!({
            "output": { "message": { "content": [{ "text": "hi" }] } },
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 30,
                "outputTokens": 628,
                "totalTokens": 658,
                "cacheReadInputTokens": 1024,
                "cacheWriteInputTokens": 512,
            },
        }))?;

        assert_eq!(response.usage.input, 30);
        assert_eq!(response.usage.output, 628);
        assert_eq!(response.usage.reasoning, 0);
        assert_eq!(response.usage.cache_read, 1024);
        assert_eq!(response.usage.cache_write, 512);
        assert!(response.cost.is_none());
        assert!(response.raw.is_some());
        Ok(())
    }

    #[test]
    fn decodes_tool_calls_and_finish_reasons() -> Result<(), Box<dyn StdError>> {
        let response = decoded(json!({
            "output": { "message": { "content": [
                { "toolUse": { "toolUseId": "call-1", "name": "search", "input": { "q": "rust" } } },
            ] } },
            "stopReason": "tool_use",
        }))?;

        assert_eq!(response.finish_reason, FinishReason::ToolCall);
        match response.content.first() {
            Some(ContentPart::ToolCall(call)) => {
                assert_eq!(call.id, "call-1");
                assert_eq!(call.name, "search");
                assert_eq!(call.arguments, json!({ "q": "rust" }));
            }
            other => return Err(format!("expected a tool call, got {other:?}").into()),
        }

        assert_eq!(
            decoded(json!({ "stopReason": "stop_sequence" }))?.finish_reason,
            FinishReason::Stop
        );
        assert_eq!(
            decoded(json!({ "stopReason": "max_tokens" }))?.finish_reason,
            FinishReason::Length
        );
        assert_eq!(
            decoded(json!({ "stopReason": "guardrail_intervened" }))?.finish_reason,
            FinishReason::ContentFilter
        );
        assert_eq!(
            decoded(json!({ "stopReason": "content_filtered" }))?.finish_reason,
            FinishReason::ContentFilter
        );
        Ok(())
    }

    #[test]
    fn raw_options_win_and_controls_never_reach_the_wire() -> Result<(), Box<dyn StdError>> {
        let body = encoded(
            Request::builder()
                .model(MODEL)
                .user("Hello")
                .max_output_tokens(16)
                .provider_option("bedrock", "inferenceConfig", json!({ "maxTokens": 4096 }))
                .provider_option("bedrock", "guardrailConfig", json!({ "trace": "enabled" }))
                .provider_option("bedrock", "auto_cache", json!(false))
                .provider_option("anthropic", "thinking", json!({ "type": "enabled" }))
                .build()?,
        )?;

        assert_eq!(body["inferenceConfig"]["maxTokens"], 4096);
        assert_eq!(body["guardrailConfig"]["trace"], "enabled");
        assert!(body.get("auto_cache").is_none());
        assert!(body.get("thinking").is_none());
        Ok(())
    }

    #[test]
    fn stop_sequences_keep_their_order() -> Result<(), Box<dyn StdError>> {
        let body = encoded(
            Request::builder()
                .model(MODEL)
                .user("Hello")
                .stop_sequences(["END", "STOP"])
                .stop_sequence("HALT")
                .build()?,
        )?;

        assert_eq!(
            body["inferenceConfig"]["stopSequences"],
            json!(["END", "STOP", "HALT"])
        );
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
                    json!({
                        "type": "grammar"
                    }),
                ))
                .build()?,
        )?;

        let error = BedrockConverseCodec
            .encode(&call, false)
            .err()
            .ok_or("expected a custom tool to be rejected")?;

        assert_eq!(error.kind(), ErrorKind::InvalidRequest);
        assert_eq!(error.provider_code(), Some("unsupported_capability"));
        assert!(error.message().contains("custom tools"));
        Ok(())
    }

    #[test]
    fn a_url_media_source_is_rejected_before_dispatch() -> Result<(), Box<dyn StdError>> {
        let call = resolved(
            Request::builder()
                .model(MODEL)
                .message(Message::new(Role::User, [ContentPart::Image(
                    ImageContent::new(MediaSource::url("https://example.com/cat.png")),
                )]))
                .build()?,
        )?;

        let error = BedrockConverseCodec
            .encode(&call, false)
            .err()
            .ok_or("expected URL media to be rejected")?;

        assert_eq!(error.kind(), ErrorKind::InvalidRequest);
        assert_eq!(error.provider_code(), Some("unsupported_capability"));
        assert!(error.message().contains("URL media"));
        Ok(())
    }

    /// Two user turns, so the conversation prefix has somewhere to land.
    fn caching_request(auto_cache: Option<bool>) -> Result<Request, Box<dyn StdError>> {
        let mut builder = Request::builder()
            .model(MODEL)
            .system("Be brief")
            .user("First")
            .message(Message::text(Role::Assistant, "Answer"))
            .user("Second")
            .tool(ToolDefinition::function("search", "Searches", json!({})));
        if let Some(auto_cache) = auto_cache {
            builder = builder.provider_option("bedrock", "auto_cache", json!(auto_cache));
        }
        Ok(builder.build()?)
    }

    #[test]
    fn auto_cache_places_cache_points_by_default() -> Result<(), Box<dyn StdError>> {
        let body = encoded(caching_request(None)?)?;
        let marker = json!({ "cachePoint": { "type": "default" } });

        assert_eq!(body["system"][1], marker);
        assert_eq!(body["toolConfig"]["tools"][1], marker);
        // The second-to-last user turn, which is the first of the two.
        assert_eq!(body["messages"][0]["content"][1], marker);
        assert_eq!(
            body["messages"][2]["content"].as_array().map(Vec::len),
            Some(1)
        );
        Ok(())
    }

    #[test]
    fn auto_cache_false_places_no_cache_points() -> Result<(), Box<dyn StdError>> {
        let body = encoded(caching_request(Some(false))?)?;

        assert!(!body.to_string().contains("cachePoint"), "{body}");
        assert_eq!(body["system"].as_array().map(Vec::len), Some(1));
        assert_eq!(
            body["toolConfig"]["tools"].as_array().map(Vec::len),
            Some(1)
        );
        Ok(())
    }

    #[test]
    fn redacted_reasoning_round_trips_through_the_blocking_path() -> Result<(), Box<dyn StdError>> {
        let response = decoded(json!({
            "output": { "message": { "content": [
                { "reasoningContent": { "redactedContent": "sealed-blob" } },
            ] } },
            "stopReason": "end_turn",
        }))?;

        let Some(ContentPart::Reasoning(reasoning)) = response.content.first() else {
            return Err(format!("expected reasoning, got {:?}", response.content).into());
        };
        assert!(reasoning.redacted);
        assert_eq!(reasoning.text, "sealed-blob");

        let body = encoded(
            Request::builder()
                .model(MODEL)
                .message(Message::new(
                    Role::Assistant,
                    response.content.iter().cloned(),
                ))
                .build()?,
        )?;

        assert_eq!(
            body["messages"][0]["content"][0]["reasoningContent"]["redactedContent"],
            "sealed-blob"
        );
        Ok(())
    }

    #[test]
    fn redacted_reasoning_round_trips_through_the_stream() -> Result<(), Box<dyn StdError>> {
        let events = streamed(&[
            ("messageStart", json!({ "role": "assistant" })),
            (
                "contentBlockDelta",
                json!({
                    "contentBlockIndex": 0,
                    "delta": { "reasoningContent": { "redactedContent": "sealed-blob" } },
                }),
            ),
            ("contentBlockStop", json!({ "contentBlockIndex": 0 })),
            ("messageStop", json!({ "stopReason": "end_turn" })),
        ])?;

        let responses = completed(&events);
        let [response] = responses.as_slice() else {
            return Err(format!("expected one Completed, got {}", responses.len()).into());
        };
        let Some(ContentPart::Reasoning(reasoning)) = response.content.first() else {
            return Err(format!("expected reasoning, got {:?}", response.content).into());
        };
        assert!(reasoning.redacted);
        assert_eq!(reasoning.text, "sealed-blob");
        Ok(())
    }

    #[test]
    fn the_content_block_index_becomes_a_stable_block_id() -> Result<(), Box<dyn StdError>> {
        let events = streamed(&[
            ("messageStart", json!({ "role": "assistant" })),
            (
                "contentBlockDelta",
                json!({ "contentBlockIndex": 3, "delta": { "text": "Hel" } }),
            ),
            (
                "contentBlockDelta",
                json!({ "contentBlockIndex": 3, "delta": { "text": "lo" } }),
            ),
            ("contentBlockStop", json!({ "contentBlockIndex": 3 })),
            ("messageStop", json!({ "stopReason": "end_turn" })),
        ])?;

        let ids: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::ContentBlockStart { id, .. }
                | StreamEvent::TextDelta { id, .. }
                | StreamEvent::ContentBlockEnd { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();

        assert_eq!(ids, ["block-3", "block-3", "block-3", "block-3"]);
        let responses = completed(&events);
        let [response] = responses.as_slice() else {
            return Err("expected exactly one Completed".into());
        };
        assert_eq!(response.text(), "Hello");
        Ok(())
    }

    #[test]
    fn interleaved_tool_calls_keep_separate_identities() -> Result<(), Box<dyn StdError>> {
        let events = streamed(&[
            ("messageStart", json!({ "role": "assistant" })),
            (
                "contentBlockStart",
                json!({
                    "contentBlockIndex": 0,
                    "start": { "toolUse": { "toolUseId": "call-a", "name": "search" } },
                }),
            ),
            (
                "contentBlockStart",
                json!({
                    "contentBlockIndex": 1,
                    "start": { "toolUse": { "toolUseId": "call-b", "name": "lookup" } },
                }),
            ),
            (
                "contentBlockDelta",
                json!({
                    "contentBlockIndex": 0,
                    "delta": { "toolUse": { "input": "{\"q\":" } },
                }),
            ),
            (
                "contentBlockDelta",
                json!({
                    "contentBlockIndex": 1,
                    "delta": { "toolUse": { "input": "{\"id\":1}" } },
                }),
            ),
            (
                "contentBlockDelta",
                json!({
                    "contentBlockIndex": 0,
                    "delta": { "toolUse": { "input": "\"rust\"}" } },
                }),
            ),
            ("contentBlockStop", json!({ "contentBlockIndex": 0 })),
            ("contentBlockStop", json!({ "contentBlockIndex": 1 })),
            ("messageStop", json!({ "stopReason": "tool_use" })),
            (
                "metadata",
                json!({ "usage": { "inputTokens": 12, "outputTokens": 5 } }),
            ),
        ])?;

        let starts: Vec<(&str, String)> = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::ContentBlockStart { id, kind } => {
                    Some((id.as_str(), format!("{kind:?}")))
                }
                _ => None,
            })
            .collect();
        assert_eq!(starts.len(), 2);
        assert_eq!(starts[0].0, "block-0");
        assert!(starts[0].1.contains("call-a"), "{}", starts[0].1);
        assert_eq!(starts[1].0, "block-1");
        assert!(starts[1].1.contains("call-b"), "{}", starts[1].1);

        let responses = completed(&events);
        let [response] = responses.as_slice() else {
            return Err("expected exactly one Completed".into());
        };
        let calls: Vec<(&str, &Value)> = response
            .content
            .iter()
            .filter_map(|part| match part {
                ContentPart::ToolCall(call) => Some((call.id.as_str(), &call.arguments)),
                _ => None,
            })
            .collect();
        assert_eq!(calls, [
            ("call-a", &json!({ "q": "rust" })),
            ("call-b", &json!({ "id": 1 })),
        ]);
        assert_eq!(response.finish_reason, FinishReason::ToolCall);
        Ok(())
    }

    #[test]
    fn the_terminal_metadata_event_carries_usage_and_leaves_raw_unset()
    -> Result<(), Box<dyn StdError>> {
        let events = streamed(&[
            ("messageStart", json!({ "role": "assistant" })),
            (
                "contentBlockDelta",
                json!({ "contentBlockIndex": 0, "delta": { "text": "hi" } }),
            ),
            ("contentBlockStop", json!({ "contentBlockIndex": 0 })),
            ("messageStop", json!({ "stopReason": "end_turn" })),
            (
                "metadata",
                json!({
                    "usage": {
                        "inputTokens": 30,
                        "outputTokens": 628,
                        "cacheReadInputTokens": 1024,
                        "cacheWriteInputTokens": 512,
                    },
                }),
            ),
        ])?;

        let usages: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::Usage { usage } => Some(*usage),
                _ => None,
            })
            .collect();
        let [usage] = usages.as_slice() else {
            return Err(format!("expected one Usage event, got {}", usages.len()).into());
        };
        assert_eq!(usage.input, 30);
        assert_eq!(usage.output, 628);
        assert_eq!(usage.reasoning, 0);
        assert_eq!(usage.cache_read, 1024);
        assert_eq!(usage.cache_write, 512);

        let responses = completed(&events);
        let [response] = responses.as_slice() else {
            return Err(format!("expected one Completed, got {}", responses.len()).into());
        };
        assert_eq!(response.usage, *usage);
        assert!(response.raw.is_none());
        Ok(())
    }

    #[test]
    fn an_unlabelled_frame_falls_back_to_its_single_top_level_key() -> Result<(), Box<dyn StdError>>
    {
        let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;
        let mut decoder = BedrockConverseCodec.stream_decoder(call.route());

        let events = decoder.decode(SseEvent {
            event: None,
            data:  json!({
                "contentBlockDelta": { "contentBlockIndex": 0, "delta": { "text": "hi" } },
            })
            .to_string(),
        })?;

        assert!(
            events.iter().any(|event| matches!(
                event,
                StreamEvent::TextDelta { text, .. } if text == "hi"
            )),
            "{events:?}"
        );
        Ok(())
    }

    #[test]
    fn a_mid_stream_exception_payload_becomes_an_error() -> Result<(), Box<dyn StdError>> {
        let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;
        let mut decoder = BedrockConverseCodec.stream_decoder(call.route());

        let error = decoder
            .decode(SseEvent {
                event: Some("throttlingException".to_owned()),
                data:  json!({
                    "__type": "com.amazon.bedrock#ThrottlingException",
                    "message": "Too many requests",
                })
                .to_string(),
            })
            .err()
            .ok_or("expected an exception payload to fail the stream")?;

        assert_eq!(error.kind(), ErrorKind::RateLimit);
        assert_eq!(error.message(), "Too many requests");
        Ok(())
    }

    #[test]
    fn the_count_tokens_request_has_the_documented_shape() -> Result<(), Box<dyn StdError>> {
        let call = resolved(
            Request::builder()
                .model(MODEL)
                .system("Be brief")
                .user("Hello")
                .max_output_tokens(64)
                .temperature(0.5)
                .provider_option("bedrock", "auto_cache", json!(false))
                .build()?,
        )?;

        let encoded = BedrockConverseCodec
            .encode_count_tokens(&call)
            .ok_or("expected Bedrock to have a count-tokens endpoint")??;

        assert!(
            encoded
                .url
                .ends_with("/model/anthropic.claude-sonnet-4-6/count-tokens"),
            "{}",
            encoded.url
        );
        assert_eq!(
            encoded.body,
            json!({
                "input": {
                    "converse": {
                        "messages": [{ "role": "user", "content": [{ "text": "Hello" }] }],
                        "system": [{ "text": "Be brief" }],
                    }
                }
            })
        );

        let tokens =
            BedrockConverseCodec.decode_count_tokens(call.route(), json!({ "inputTokens": 41 }))?;
        assert_eq!(tokens, 41);

        let error = BedrockConverseCodec
            .decode_count_tokens(call.route(), json!({}))
            .err()
            .ok_or("expected a malformed count body to fail")?;
        assert_eq!(error.kind(), ErrorKind::ResponseDecode);
        Ok(())
    }

    #[test]
    fn refuses_a_declared_media_type_it_cannot_name() -> Result<(), Box<dyn StdError>> {
        // Converse takes a format enum, so an unmapped type has nowhere to go.
        // Sending the bytes as `png` anyway would tell the provider something
        // the caller never said.
        let request = Request::builder()
            .model(MODEL)
            .message(Message::new(Role::User, [ContentPart::Image(
                ImageContent::new(MediaSource::base64("aW1n", "image/heic")),
            )]))
            .build()?;

        let Err(error) = BedrockConverseCodec.encode(&resolved(request)?, false) else {
            panic!("an unmapped media type should be refused");
        };

        assert_eq!(error.kind(), ErrorKind::InvalidRequest);
        assert_eq!(error.provider_code(), Some("unsupported_capability"));
        assert!(error.message().contains("image/heic"));
        Ok(())
    }

    #[test]
    fn a_declared_media_type_the_enum_names_is_used() -> Result<(), Box<dyn StdError>> {
        // The refusal above must not have swallowed the mapped types too.
        let request = Request::builder()
            .model(MODEL)
            .message(Message::new(Role::User, [ContentPart::Image(
                ImageContent::new(MediaSource::base64("aW1n", "image/png")),
            )]))
            .build()?;
        let body = encoded(request)?;

        assert_eq!(body["messages"][0]["content"][0]["image"]["format"], "png");
        Ok(())
    }

    #[test]
    fn reports_controls_converse_cannot_express() -> Result<(), Box<dyn StdError>> {
        let request = Request::builder()
            .model(MODEL)
            .user("Answer as JSON.")
            .response_format(ResponseFormat::JsonObject)
            .metadata_entry("tenant", "acme")
            .build()?;

        let codes = warnings_for(request)?;

        assert!(codes.iter().all(|code| code == "unsupported_control"));
        assert_eq!(codes.len(), 2, "metadata and response format both warn");
        Ok(())
    }

    #[test]
    fn a_json_tool_result_warns_that_it_is_flattened() -> Result<(), Box<dyn StdError>> {
        // Structured JSON is the easy one to miss: it is not media, but the
        // text flattening drops it just the same.
        let request = Request::builder()
            .model(MODEL)
            .user("Chart it.")
            .message(Message::new(Role::Tool, [ContentPart::ToolResult(
                ToolResult {
                    tool_call_id: "call-1".to_owned(),
                    name:         Some("chart".to_owned()),
                    content:      vec![ContentPart::Json {
                        value: json!({ "quarters": [1, 2] }),
                    }],
                    is_error:     false,
                },
            )]))
            .build()?;

        let codes = warnings_for(request)?;

        assert_eq!(codes, ["unsupported_control"]);
        Ok(())
    }
}
