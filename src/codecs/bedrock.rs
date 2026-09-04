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

use std::collections::BTreeSet;

use reqwest::Method;
use serde_json::{Map, Value, json};

use super::assembler::StreamAssembler;
use super::common::{
    ANTHROPIC_SIGNATURES, carries_foreign_signature, drop_truncated_tool_calls, endpoint,
    finish_reason, flattens_system_content, flattens_tool_result_content, foreign_signature,
    merge_options, plain_text, refusal, reject_unencodable, sampling, system_text,
    unsupported_capability, wire_options,
};
use super::{Codec, StreamDecoder};
use crate::adapter::ResolvedCall;
use crate::resolver::ResolvedRoute;
use crate::transport::{EncodedRequest, SseEvent, classify};
use crate::types::{
    ContentBlockId, ContentBlockKind, ContentPart, Error, ErrorKind, FinishReason, MediaSource,
    Message, ReasoningContent, ReasoningEffort, Request, Response, RetryClassification, Role,
    Speed, StreamEvent, TokenCounts, ToolCall, ToolCallKind, ToolChoice, ToolDefinition,
    ToolDefinitionKind, ToolResult,
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
        reject_unnameable_tools(request, route)?;
        let (options, controls) = wire_options(call);
        let cached = caches(route, controls.auto_cache);

        let mut body = Map::new();
        let system = system_blocks(request, cached);
        if !system.is_empty() {
            body.insert("system".to_owned(), Value::Array(system));
        }
        body.insert(
            "messages".to_owned(),
            Value::Array(conversation(request, route, cached)?),
        );

        let mut inference = inference_config(request);
        if let Some(tool_config) = tool_config(request, cached) {
            body.insert("toolConfig".to_owned(), tool_config);
        }
        // Effort has the same two wire dialects as the Anthropic codec,
        // carried through `additionalModelRequestFields`: a model with effort
        // levels takes `output_config.effort`, an older reasoning model takes
        // an explicit thinking budget, and a forced tool choice suppresses
        // both because the upstream model rejects thinking alongside it. The
        // budget must sit strictly below `maxTokens`, and a request that
        // sends none leaves AWS's per-model default in charge — a default at
        // or below the budget draws a ValidationException. So a budget always
        // travels with an explicit `maxTokens`, lifted when the budget would
        // not fit under it, the same way the Anthropic encoder grows it.
        if let Some(effort) = request.reasoning_effort()
            && !forces_tool_use(request.tool_choice())
        {
            // A passthrough model takes the modern effort dialect, like
            // the Anthropic codec: it is uncataloged precisely because it
            // is newer than the catalog, and a guessed thinking budget is
            // a manual toggle the always-adaptive models reject.
            if route.model().protocol_options().reasoning_effort_levels
                || route.model().is_passthrough()
            {
                body.insert(
                    "additionalModelRequestFields".to_owned(),
                    json!({ "output_config": { "effort": bedrock_effort(effort) } }),
                );
            } else {
                let limit = budget_limit(call);
                let budget = thinking_budget(effort, limit);
                let max_tokens = if limit <= budget {
                    budget.saturating_add(MIN_THINKING_BUDGET)
                } else {
                    limit
                };
                inference.insert("maxTokens".to_owned(), max_tokens.into());
                body.insert(
                    "additionalModelRequestFields".to_owned(),
                    json!({ "thinking": { "type": "enabled", "budget_tokens": budget } }),
                );
            }
        }
        if !inference.is_empty() {
            body.insert("inferenceConfig".to_owned(), Value::Object(inference));
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
        // A streaming request names the framing it expects, which is what the
        // reference client always sent; a gateway that negotiates content
        // types answers with the event stream rather than something else.
        let headers = if stream {
            vec![(
                "accept".to_owned(),
                "application/vnd.amazon.eventstream".to_owned(),
            )]
        } else {
            Vec::new()
        };
        let mut encoded = EncodedRequest::new(
            Method::POST,
            operation_url(route, operation),
            Value::Object(body),
        )
        .with_headers(headers)
        .with_applied_speed(request.speed());
        // Converse has no request-metadata field, so the map is reported rather
        // than folded into some other field where it would change the prompt.
        if !request.metadata().is_empty() {
            encoded = encoded.unsupported_control("request metadata");
        }
        // The suppressed effort never reaches the model, so say so — the same
        // report the Anthropic codec makes for this combination.
        if request.reasoning_effort().is_some() && forces_tool_use(request.tool_choice()) {
            encoded = encoded.unsupported_control("reasoning effort with a forced tool choice");
        }
        // The tools stayed on the wire despite `tool_choice: none`, because
        // Converse rejects a request whose history carries tool blocks
        // without a `toolConfig`. The model may therefore still call a tool.
        if matches!(request.tool_choice(), Some(ToolChoice::None))
            && !request.tools().is_empty()
            && history_carries_tool_blocks(request)
        {
            encoded =
                encoded.unsupported_control("tool_choice none alongside historical tool blocks");
        }
        // A skipped foreign-signed reasoning part never reaches the model,
        // so the skip is reported; see `foreign_signature`.
        if carries_foreign_signature(request, ANTHROPIC_SIGNATURES) {
            encoded = encoded.unsupported_control("reasoning signed by another provider");
        }
        // Converse has no portable structured-output field. A caller who asked
        // for JSON gets prose, so say so rather than letting them discover it
        // by parsing.
        if request.response_format().is_some() {
            encoded = encoded.unsupported_control("response formats");
        }
        // The system field of this protocol takes text only, so anything else
        // a system message carries is dropped. The text still reaches the
        // model, so it is reported rather than refused.
        if flattens_system_content(request) {
            encoded = encoded.unsupported_control("non-text system content");
        }
        // `toolResult.content` is a block list of its own, so text, structured
        // JSON, images, and documents all reach the model as themselves. Only
        // a part with no member of that union — reasoning, most of all — is
        // dropped, and only that is worth reporting.
        if flattens_tool_result_content(request, |parts| parts.iter().all(carries_in_tool_result)) {
            encoded =
                encoded.unsupported_control("tool result content outside text, JSON, and media");
        }
        Ok(encoded)
    }

    fn decode_response(&self, route: &ResolvedRoute, value: Value) -> Result<Response, Error> {
        // A refusal is a failure, not a short answer — the same contract the
        // Anthropic codec applies, since Bedrock passes the stop through for
        // Claude models.
        if value.get("stopReason").and_then(Value::as_str) == Some("refusal") {
            return Err(refusal(route, None, Some(value)));
        }

        // A body without the output message is not a Converse response.
        // Decoding `{}` from a broken proxy as a successful empty answer
        // would be indistinguishable from a real empty completion.
        if !value
            .pointer("/output/message/content")
            .is_some_and(Value::is_array)
        {
            // A structurally malformed 200 is indistinguishable from a
            // garbled or truncated body, so a fresh attempt is safe — the
            // same classification the transport gives a 200 whose body is
            // not JSON at all.
            return Err(Error::new(
                ErrorKind::ResponseDecode,
                format!(
                    "provider {} returned a 200 body without a Converse output message",
                    route.provider().id()
                ),
            )
            .with_provider(route.provider().id().clone())
            .with_raw_data(value)
            .with_retry(RetryClassification::Safe));
        }

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
        drop_truncated_tool_calls(&mut response);
        Ok(response)
    }

    fn stream_decoder(&self, route: &ResolvedRoute) -> Box<dyn StreamDecoder> {
        Box::new(BedrockStreamDecoder {
            route:           route.clone(),
            assembler:       StreamAssembler::new(route),
            tool_blocks:     BTreeSet::new(),
            redacted_blocks: BTreeSet::new(),
            text_blocks:     BTreeSet::new(),
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
                // Malformed like any other garbled 200, so retrying is safe.
                Error::new(
                    ErrorKind::ResponseDecode,
                    "Bedrock returned a count-tokens body without an inputTokens count",
                )
                .with_provider(route.provider().id().clone())
                .with_raw_data(value)
                .with_retry(RetryClassification::Safe)
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
    reject_unnameable_tools(request, route)?;
    let (_, controls) = wire_options(call);
    let cached = caches(route, controls.auto_cache);

    let mut converse = Map::new();
    converse.insert(
        "messages".to_owned(),
        Value::Array(conversation(request, route, cached)?),
    );
    let system = system_blocks(request, cached);
    if !system.is_empty() {
        converse.insert("system".to_owned(), Value::Array(system));
    }

    Ok(EncodedRequest::new(
        Method::POST,
        operation_url(route, "count-tokens"),
        json!({ "input": { "converse": Value::Object(converse) } }),
    ))
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

/// Whether this call may carry `cachePoint` markers.
///
/// Bedrock fronts many model families and the catalog lets a caller name any
/// of them, so the model's own declared support decides this and not the
/// provider: a family that cannot cache rejects the whole request with a
/// ValidationException rather than ignoring the marker. `auto_cache` is the
/// caller's separate veto over markers this codec adds on its own.
fn caches(route: &ResolvedRoute, auto_cache: bool) -> bool {
    auto_cache && route.model().capabilities().caching().is_supported()
}

/// Whether `toolResult.content` has a block for this part.
///
/// The Converse tool-result union takes text, structured JSON, images, and
/// documents, so all four reach the model as themselves. Anything else —
/// reasoning above all — has no member of that union and is dropped.
fn carries_in_tool_result(part: &ContentPart) -> bool {
    matches!(
        part,
        ContentPart::Text { .. }
            | ContentPart::Json { .. }
            | ContentPart::Image(_)
            | ContentPart::Document(_)
    )
}

/// The system blocks, with a cache point after the prompt when caching is on.
fn system_blocks(request: &Request, cached: bool) -> Vec<Value> {
    let system = system_text(request.messages());
    if system.is_empty() {
        return Vec::new();
    }

    let mut blocks = vec![json!({ "text": system })];
    if cached {
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
    cached: bool,
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
        if blocks.is_empty()
            && message.role() == Role::Tool
            && let Some(tool_call_id) = message.tool_call_id()
        {
            blocks.push(tool_result_block(
                tool_call_id,
                vec![json!({ "text": plain_text(message.content()) })],
                false,
            ));
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

    if cached {
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
        // A signature another provider family minted cannot verify here and
        // fails the request, so the part is skipped; the encoder reports it.
        ContentPart::Reasoning(reasoning) if foreign_signature(reasoning, ANTHROPIC_SIGNATURES) => {
            None
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
/// Converse requires `toolUse.input` to be a JSON object document. A
/// no-argument tool call carries `Null`, which Bedrock rejects as
/// "toolUse.input is empty", and a scalar or array argument value would die at
/// AWS with a ValidationException. Any non-object is coerced to `{}` so the
/// wire is always valid, regardless of where the replayed call originated.
fn encode_tool_call(call: &ToolCall) -> Value {
    let input = match &call.input.wire_value() {
        Value::Object(_) => call.input.wire_value().clone(),
        _ => json!({}),
    };
    json!({
        "toolUse": { "toolUseId": call.id, "name": call.name, "input": input }
    })
}

/// Encodes a tool result as a `toolResult` block.
///
/// A part the tool-result union has no member for is skipped rather than
/// encoded as the block it would become in a message. A reasoning part would
/// otherwise become a `reasoningContent` block inside `toolResult`, which
/// Converse rejects, and the whole request would fail over one replayed part.
/// The caller is told about the drop by the `unsupported_control` warning
/// [`carries_in_tool_result`] drives.
fn encode_tool_result(result: &ToolResult, route: &ResolvedRoute) -> Result<Value, Error> {
    let mut blocks = Vec::new();
    for part in &result.content {
        match part {
            ContentPart::Json { value } => blocks.push(json!({ "json": value })),
            other if carries_in_tool_result(other) => {
                if let Some(block) = encode_content_part(other, route)? {
                    blocks.push(block);
                }
            }
            _ => {}
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

/// Refuses media whose declared type has no Converse format.
fn unsupported_media_type(route: &ResolvedRoute, media_type: Option<&str>) -> Error {
    unsupported_capability(
        route,
        &format!("the media type {}", media_type.unwrap_or("<undeclared>")),
    )
}

/// Maps a declared media type onto the Converse format enum.
///
/// `None` means the source declared nothing, and the caller's `default` stands
/// in. A declared type this protocol has no enum member for returns `None`,
/// because labelling the bytes with a format they are not would tell the
/// provider something the caller never said — `image/heic` sent as `png`.
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

/// The longest tool name or tool-call id Converse accepts.
const TOOL_IDENTIFIER_MAX: usize = 64;

/// Converse validates a tool name against `^[a-zA-Z0-9_-]{1,64}$` and a
/// `toolUseId` against the same set plus `.` and `:`.
///
/// Every tool identifier the request would send is checked here: the tool
/// definitions, and the names and ids of the tool calls and tool results
/// already in the message history. History matters as much as the definitions,
/// because a conversation that began on another provider carries that
/// provider's identifiers — a dotted Gemini tool name, or an id longer than
/// Converse allows — and the whole replayed request would die at AWS with an
/// opaque ValidationException.
///
/// An identifier outside the set is refused here rather than rewritten.
/// Rewriting is lossy — `mcp.server.tool` and `mcp_server_tool` both become the
/// same thing, so a decoded tool call could not be mapped back to the tool the
/// caller registered, and a rewritten id no longer matches the call the caller
/// holds. A caller with such names owns that mapping, because only they can
/// undo it. This is the deliberate departure from the reference, which
/// rewrote both with a hash suffix.
///
/// # Errors
///
/// Returns [`ErrorKind::InvalidRequest`] naming the first offending identifier.
fn reject_unnameable_tools(request: &Request, route: &ResolvedRoute) -> Result<(), Error> {
    for tool in request.tools() {
        reject_tool_name(&tool.name, route)?;
    }
    for message in request.messages() {
        if let Some(tool_call_id) = message.tool_call_id() {
            reject_tool_call_id(tool_call_id, route)?;
        }
        for part in message.content() {
            match part {
                ContentPart::ToolCall(call) => {
                    reject_tool_name(&call.name, route)?;
                    reject_tool_call_id(&call.id, route)?;
                }
                ContentPart::ToolResult(result) => {
                    reject_tool_call_id(&result.tool_call_id, route)?;
                }
                _ => {}
            }
        }
    }
    Ok(())
}

/// Refuses a tool name outside `^[a-zA-Z0-9_-]{1,64}$`.
fn reject_tool_name(name: &str, route: &ResolvedRoute) -> Result<(), Error> {
    reject_tool_identifier(name, route, "tool name", "^[a-zA-Z0-9_-]{1,64}$", |byte| {
        byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-')
    })
}

/// Refuses a `toolUseId` outside `^[a-zA-Z0-9_.:-]{1,64}$`.
fn reject_tool_call_id(id: &str, route: &ResolvedRoute) -> Result<(), Error> {
    reject_tool_identifier(
        id,
        route,
        "tool call id",
        "^[a-zA-Z0-9_.:-]{1,64}$",
        |byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.' | b':'),
    )
}

/// Refuses one tool identifier, naming the value and the pattern it must match.
fn reject_tool_identifier(
    value: &str,
    route: &ResolvedRoute,
    kind: &str,
    pattern: &str,
    allowed: fn(u8) -> bool,
) -> Result<(), Error> {
    let valid =
        !value.is_empty() && value.len() <= TOOL_IDENTIFIER_MAX && value.bytes().all(allowed);
    if valid {
        return Ok(());
    }
    Err(unsupported_capability(
        route,
        &format!("the {kind} `{value}`, which must match {pattern}"),
    ))
}

/// The `toolConfig` object, with a cache point after the last tool.
///
/// Converse has no "call no tool" choice, so `ToolChoice::None` drops the
/// whole tool configuration: withholding the tools is the only faithful
/// encoding — except when the history already carries toolUse or toolResult
/// blocks. Converse requires `toolConfig` alongside those blocks, so the
/// common agent-loop ending ("now answer in prose") keeps the tools on the
/// wire and the encoder reports the choice it could not express.
fn tool_config(request: &Request, cached: bool) -> Option<Value> {
    if request.tools().is_empty() {
        return None;
    }
    if matches!(request.tool_choice(), Some(ToolChoice::None))
        && !history_carries_tool_blocks(request)
    {
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
    if cached {
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

/// Whether the message history already carries toolUse or toolResult blocks.
fn history_carries_tool_blocks(request: &Request) -> bool {
    request
        .messages()
        .iter()
        .flat_map(Message::content)
        .any(|part| matches!(part, ContentPart::ToolCall(_) | ContentPart::ToolResult(_)))
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
        ReasoningEffort::Xhigh => "xhigh",
        ReasoningEffort::Max => "max",
    }
}

/// The smallest `thinking.budget_tokens` the upstream model accepts, and the
/// headroom kept above the budget when `maxTokens` must grow.
const MIN_THINKING_BUDGET: u32 = 1024;

/// The output limit the thinking budget scales against.
///
/// The request's own limit wins, then the model's catalog limit, then the
/// same fallback the Anthropic codec uses.
fn budget_limit(call: &ResolvedCall) -> u32 {
    if let Some(tokens) = call.request().max_output_tokens() {
        return tokens;
    }
    call.route().model().limits().map_or(65_536, |limits| {
        u32::try_from(limits.max_output_tokens).unwrap_or(u32::MAX)
    })
}

/// Whether the tool choice makes a tool call mandatory.
///
/// The upstream model rejects extended thinking together with a forced tool
/// choice, so a forced choice suppresses the effort encoding entirely.
fn forces_tool_use(choice: Option<&ToolChoice>) -> bool {
    choice.is_some_and(ToolChoice::is_forced)
}

/// The explicit thinking budget for a reasoning model without effort levels,
/// scaling the same shares of the output limit as the Anthropic codec.
fn thinking_budget(effort: ReasoningEffort, limit: u32) -> u32 {
    let limit = u64::from(limit);
    let share = match effort {
        ReasoningEffort::Minimal | ReasoningEffort::Low => limit / 4,
        ReasoningEffort::Medium => limit / 2,
        ReasoningEffort::High => limit * 3 / 4,
        ReasoningEffort::Xhigh => limit * 7 / 8,
        ReasoningEffort::Max => limit,
    };
    let budget = share.max(u64::from(MIN_THINKING_BUDGET));
    u32::try_from(budget).unwrap_or(u32::MAX)
}

/// Maps a Converse `stopReason` onto the normalized finish reason.
///
/// `model_context_window_exceeded` is Converse's own name for a generation that
/// ran out of context, which is the same outcome as `max_tokens` and so maps
/// onto the same reason. `refusal` never reaches here: both decode paths fail
/// the call before they ask for a finish reason.
fn stop_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("stop_sequence") => FinishReason::Stop,
        Some("model_context_window_exceeded") => FinishReason::Length,
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
        let signature = text_block
            .get("signature")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let signature_origin = signature.is_some().then(|| ANTHROPIC_SIGNATURES.to_owned());
        return Some(ContentPart::Reasoning(ReasoningContent {
            text: text_block
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            signature,
            signature_origin,
            redacted: false,
        }));
    }
    let redacted = reasoning.get("redactedContent").and_then(Value::as_str)?;
    Some(ContentPart::Reasoning(ReasoningContent {
        text:             redacted.to_owned(),
        signature:        None,
        signature_origin: None,
        redacted:         true,
    }))
}

/// Decodes one Bedrock ConverseStream.
struct BedrockStreamDecoder {
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

#[cfg(all(test, feature = "builtin-catalog"))]
mod tests {
    use std::error::Error as StdError;

    use serde_json::{Value, json};

    use super::{BedrockConverseCodec, Codec};
    use crate::codecs::test_support::{resolved, resolved_in};
    use crate::transport::SseEvent;
    use crate::types::{
        ContentPart, DocumentContent, ErrorKind, FinishReason, ImageContent, MediaSource, Message,
        ReasoningContent, ReasoningEffort, Request, Response, ResponseFormat, RetryClassification,
        Role, Speed, StreamEvent, ToolCall, ToolChoice, ToolDefinition, ToolResult,
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
    fn a_body_without_the_output_message_fails_to_decode() -> Result<(), Box<dyn StdError>> {
        let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;

        // `{}` from a broken proxy and a body whose message lost its content
        // are both indistinguishable from a real empty answer; neither may
        // decode as success.
        for body in [json!({}), json!({ "output": { "message": {} } })] {
            let error = BedrockConverseCodec
                .decode_response(call.route(), body)
                .expect_err("a structureless 200 must fail to decode");
            assert_eq!(error.kind(), ErrorKind::ResponseDecode);
            assert_eq!(error.retry_classification(), RetryClassification::Safe);
        }
        Ok(())
    }

    #[test]
    fn a_stream_event_that_is_not_json_fails_retryably() -> Result<(), Box<dyn StdError>> {
        let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;
        let mut decoder = BedrockConverseCodec.stream_decoder(call.route());

        let error = decoder
            .decode(SseEvent {
                event: Some("contentBlockDelta".to_owned()),
                data:  "not json".to_owned(),
            })
            .expect_err("a garbled event must fail the stream");

        assert_eq!(error.kind(), ErrorKind::StreamDecode);
        assert_eq!(error.retry_classification(), RetryClassification::Safe);
        Ok(())
    }

    #[test]
    fn a_refusal_response_fails_instead_of_decoding() -> Result<(), Box<dyn StdError>> {
        let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;
        let body = json!({
            "output": { "message": { "content": [] } },
            "stopReason": "refusal",
        });

        let error = BedrockConverseCodec
            .decode_response(call.route(), body.clone())
            .expect_err("a refusal must fail the call");

        assert_eq!(error.kind(), ErrorKind::ContentFilter);
        assert_eq!(error.provider_code(), Some("refusal"));
        assert_eq!(error.raw_data(), Some(&body));
        Ok(())
    }

    #[test]
    fn an_input_fragment_for_an_unopened_block_fails_the_stream() -> Result<(), Box<dyn StdError>> {
        // Converse announces every tool call in a contentBlockStart carrying
        // its id and name. When that start is lost, assembling the fragments
        // would fabricate a nameless call whose replay fails identifier
        // validation, so the stream fails retryably instead — the contract
        // R2-24 set for the Chat codec.
        let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;
        let mut decoder = BedrockConverseCodec.stream_decoder(call.route());

        let error = decoder
            .decode(SseEvent {
                event: Some("contentBlockDelta".to_owned()),
                data:  json!({
                    "contentBlockIndex": 0,
                    "delta": { "toolUse": { "input": "{\"q\":\"rust\"}" } },
                })
                .to_string(),
            })
            .err()
            .ok_or("expected the orphan input fragment to fail the stream")?;

        assert_eq!(error.kind(), ErrorKind::StreamDecode);
        assert_eq!(error.retry_classification(), RetryClassification::Safe);
        Ok(())
    }

    #[test]
    fn a_streamed_refusal_ends_the_stream_as_an_error() -> Result<(), Box<dyn StdError>> {
        let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;
        let mut decoder = BedrockConverseCodec.stream_decoder(call.route());

        decoder.decode(SseEvent {
            event: Some("messageStart".to_owned()),
            data:  json!({ "role": "assistant" }).to_string(),
        })?;
        let error = decoder
            .decode(SseEvent {
                event: Some("messageStop".to_owned()),
                data:  json!({ "stopReason": "refusal" }).to_string(),
            })
            .expect_err("a refusal must fail the stream");

        assert_eq!(error.kind(), ErrorKind::ContentFilter);
        assert_eq!(error.provider_code(), Some("refusal"));
        Ok(())
    }

    #[test]
    fn encodes_document_reasoning_and_performance_fields() -> Result<(), Box<dyn StdError>> {
        let body = encoded(
            Request::builder()
                .model(MODEL)
                .message(Message::new(Role::Assistant, [ContentPart::Reasoning(
                    ReasoningContent {
                        text:             "checked".to_owned(),
                        signature:        Some("sig".to_owned()),
                        signature_origin: Some("anthropic".to_owned()),
                        redacted:         false,
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

    /// A Bedrock model that reasons with a thinking budget, not effort levels.
    const BUDGET_CATALOG: &str = r#"
        schema_version = 1

        [providers.bedrock]
        display_name = "Amazon Bedrock"
        adapter = "bedrock"
        codec = "bedrock-converse"
        base_url = "https://bedrock-runtime.us-east-1.amazonaws.com"
        default_model = "older-claude"
        auth = { type = "none" }

        [providers.bedrock.models.older-claude]
        display_name = "Older Claude"
        api_model = "us.anthropic.claude-3-7"
        capabilities = { text = true, tools = true, reasoning = true, tool_choice = { required = true, named = true } }
    "#;

    /// The budget-model catalog with passthrough allowed.
    const PASSTHROUGH_CATALOG: &str = r#"
        schema_version = 1

        [providers.bedrock]
        display_name = "Amazon Bedrock"
        adapter = "bedrock"
        codec = "bedrock-converse"
        base_url = "https://bedrock-runtime.us-east-1.amazonaws.com"
        allow_passthrough = true
        default_model = "older-claude"
        auth = { type = "none" }

        [providers.bedrock.models.older-claude]
        display_name = "Older Claude"
        api_model = "us.anthropic.claude-3-7"
        capabilities = { text = true, tools = true, reasoning = true, tool_choice = { required = true, named = true } }
    "#;

    #[test]
    fn a_passthrough_model_takes_the_effort_dialect() -> Result<(), Box<dyn StdError>> {
        // Passthrough serves models newer than the catalog, and those reject
        // a manual thinking toggle — so the uncataloged guess is the modern
        // dialect, the same one the Anthropic codec makes.
        let call = resolved_in(
            PASSTHROUGH_CATALOG,
            Request::builder()
                .model("bedrock/us.anthropic.claude-next")
                .user("Hello")
                .reasoning_effort(ReasoningEffort::High)
                .build()?,
        )?;

        let body = BedrockConverseCodec.encode(&call, false)?.body;

        assert_eq!(
            body["additionalModelRequestFields"]["output_config"]["effort"],
            "high"
        );
        assert_eq!(
            body["additionalModelRequestFields"]["thinking"],
            json!(null)
        );
        Ok(())
    }

    #[test]
    fn effort_becomes_a_thinking_budget_without_effort_levels() -> Result<(), Box<dyn StdError>> {
        // A reasoning model without `reasoning_effort_levels` rejects
        // `output_config.effort`; it takes an explicit thinking budget, the
        // same translation the Anthropic codec applies. A caller limit the
        // budget would not fit under grows to keep the budget strictly below
        // `maxTokens`.
        let call = resolved_in(
            BUDGET_CATALOG,
            Request::builder()
                .model("bedrock/older-claude")
                .user("Hello")
                .reasoning_effort(ReasoningEffort::High)
                .max_output_tokens(4096)
                .build()?,
        )?;

        let body = BedrockConverseCodec.encode(&call, false)?.body;

        assert_eq!(
            body["additionalModelRequestFields"]["thinking"],
            json!({ "type": "enabled", "budget_tokens": 3072 })
        );
        assert_eq!(
            body["additionalModelRequestFields"]["output_config"],
            json!(null)
        );
        assert_eq!(body["inferenceConfig"]["maxTokens"], 4096);
        Ok(())
    }

    #[test]
    fn a_thinking_budget_without_a_caller_limit_still_sends_max_tokens()
    -> Result<(), Box<dyn StdError>> {
        // With no caller limit the budget derives from the fallback output
        // limit, and AWS's per-model default `maxTokens` would be in charge —
        // a default at or below the budget draws a ValidationException. The
        // effective limit therefore always goes on the wire, lifted when the
        // budget equals it.
        let call = resolved_in(
            BUDGET_CATALOG,
            Request::builder()
                .model("bedrock/older-claude")
                .user("Hello")
                .reasoning_effort(ReasoningEffort::Max)
                .build()?,
        )?;

        let body = BedrockConverseCodec.encode(&call, false)?.body;

        assert_eq!(
            body["additionalModelRequestFields"]["thinking"],
            json!({ "type": "enabled", "budget_tokens": 65_536 })
        );
        assert_eq!(body["inferenceConfig"]["maxTokens"], 66_560);
        Ok(())
    }

    #[test]
    fn a_forced_tool_choice_suppresses_the_effort_encoding() -> Result<(), Box<dyn StdError>> {
        // The upstream model rejects thinking alongside a forced tool choice,
        // so neither effort dialect may reach the wire, and the dropped
        // control is reported.
        let encoded = BedrockConverseCodec.encode(
            &resolved(
                Request::builder()
                    .model(MODEL)
                    .user("Hello")
                    .tool(ToolDefinition::function("lookup", "look up", json!({})))
                    .tool_choice(ToolChoice::Required)
                    .reasoning_effort(ReasoningEffort::High)
                    .build()?,
            )?,
            false,
        )?;

        assert_eq!(encoded.body.get("additionalModelRequestFields"), None);
        assert!(
            encoded
                .warnings
                .iter()
                .any(|warning| warning.message.contains("forced tool choice")),
            "{:?}",
            encoded.warnings
        );
        Ok(())
    }

    #[test]
    fn an_anthropic_signed_part_replays_and_a_gemini_one_is_skipped()
    -> Result<(), Box<dyn StdError>> {
        // Converse carries Claude-minted signatures, so a part signed at the
        // Anthropic provider keeps replaying after a failover to Bedrock; a
        // Gemini thought signature cannot verify and is skipped.
        let anthropic_signed = ReasoningContent {
            text:             "claude thought".to_owned(),
            signature:        Some("claude-sig".to_owned()),
            signature_origin: Some("anthropic".to_owned()),
            redacted:         false,
        };
        let gemini_signed = ReasoningContent {
            text:             "gemini thought".to_owned(),
            signature:        Some("gemini-sig".to_owned()),
            signature_origin: Some("gemini".to_owned()),
            redacted:         false,
        };
        let encoded = BedrockConverseCodec.encode(
            &resolved(
                Request::builder()
                    .model(MODEL)
                    .user("Hello")
                    .message(Message::new(Role::Assistant, [
                        ContentPart::Reasoning(anthropic_signed),
                        ContentPart::Reasoning(gemini_signed),
                        ContentPart::Text {
                            text: "answer".to_owned(),
                        },
                    ]))
                    .user("Continue")
                    .build()?,
            )?,
            false,
        )?;

        let body = encoded.body.to_string();
        assert!(body.contains("claude-sig"), "{body}");
        assert!(!body.contains("gemini-sig"), "{body}");
        assert!(
            encoded
                .warnings
                .iter()
                .any(|warning| warning.message.contains("signed by another provider")),
            "{:?}",
            encoded.warnings
        );
        Ok(())
    }

    #[test]
    fn tool_choice_none_keeps_the_tools_when_the_history_carries_tool_blocks()
    -> Result<(), Box<dyn StdError>> {
        // Converse requires `toolConfig` whenever messages carry toolUse or
        // toolResult blocks, so the agent-loop ending — answer in prose after
        // a tool exchange — must keep the tools on the wire, force nothing,
        // and report the choice it could not express.
        let encoded = BedrockConverseCodec.encode(
            &resolved(
                Request::builder()
                    .model(MODEL)
                    .user("What is the weather?")
                    .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
                        ToolCall::function("call-1", "get_weather", json!({ "city": "Paris" })),
                    )]))
                    .message(Message::new(Role::Tool, [ContentPart::ToolResult(
                        ToolResult {
                            tool_call_id: "call-1".to_owned(),
                            name:         Some("get_weather".to_owned()),
                            content:      vec![ContentPart::Text {
                                text: "18C".to_owned(),
                            }],
                            is_error:     false,
                        },
                    )]))
                    .tool(ToolDefinition::function(
                        "get_weather",
                        "weather",
                        json!({}),
                    ))
                    .tool_choice(ToolChoice::None)
                    .build()?,
            )?,
            false,
        )?;

        assert_eq!(
            encoded.body["toolConfig"]["tools"][0]["toolSpec"]["name"],
            "get_weather"
        );
        assert_eq!(encoded.body["toolConfig"].get("toolChoice"), None);
        assert!(
            encoded
                .warnings
                .iter()
                .any(|warning| warning.message.contains("tool_choice none")),
            "{:?}",
            encoded.warnings
        );

        // Without tool blocks in the history, withholding the tools stays the
        // faithful encoding of `none`.
        let clean = BedrockConverseCodec.encode(
            &resolved(
                Request::builder()
                    .model(MODEL)
                    .user("Hello")
                    .tool(ToolDefinition::function(
                        "get_weather",
                        "weather",
                        json!({}),
                    ))
                    .tool_choice(ToolChoice::None)
                    .build()?,
            )?,
            false,
        )?;
        assert_eq!(clean.body.get("toolConfig"), None);
        Ok(())
    }

    #[test]
    fn a_streaming_request_names_the_event_stream_framing() -> Result<(), Box<dyn StdError>> {
        let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;

        let unary = BedrockConverseCodec.encode(&call, false)?;
        let streaming = BedrockConverseCodec.encode(&call, true)?;

        assert_eq!(streaming.headers, vec![(
            "accept".to_owned(),
            "application/vnd.amazon.eventstream".to_owned(),
        )]);
        assert!(unary.headers.is_empty());
        Ok(())
    }

    #[test]
    fn percent_encodes_the_model_id_in_the_path() -> Result<(), Box<dyn StdError>> {
        let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;

        let unary = BedrockConverseCodec.encode(&call, false)?;
        let streaming = BedrockConverseCodec.encode(&call, true)?;

        assert_eq!(
            unary.url,
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-sonnet-4-6/converse"
        );
        assert_eq!(
            streaming.url,
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-sonnet-4-6/converse-stream"
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
                assert_eq!(call.input.wire_value(), json!({ "q": "rust" }));
            }
            other => return Err(format!("expected a tool call, got {other:?}").into()),
        }

        let stop = |reason: &str| {
            json!({
                "output": { "message": { "content": [] } },
                "stopReason": reason,
            })
        };
        assert_eq!(
            decoded(stop("stop_sequence"))?.finish_reason,
            FinishReason::Stop
        );
        assert_eq!(
            decoded(stop("max_tokens"))?.finish_reason,
            FinishReason::Length
        );
        assert_eq!(
            decoded(stop("guardrail_intervened"))?.finish_reason,
            FinishReason::ContentFilter
        );
        assert_eq!(
            decoded(stop("content_filtered"))?.finish_reason,
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

    /// A Bedrock provider whose one model declares no prompt caching.
    ///
    /// Bedrock allows passthrough, so a caller can name any hosted family, and
    /// most of them cannot cache. The built-in catalog carries no such entry,
    /// so this test catalog supplies one.
    const UNCACHED_CATALOG: &str = r#"
        schema_version = 1

        [providers.bedrock]
        display_name = "Amazon Bedrock"
        adapter = "bedrock"
        codec = "bedrock-converse"
        base_url = "https://bedrock-runtime.us-east-1.amazonaws.com"
        default_model = "llama"
        auth = { type = "none" }

        [providers.bedrock.models.llama]
        display_name = "Llama 3 70B"
        api_model = "meta.llama3-70b-instruct-v1:0"
        capabilities = { text = true, tools = true, tool_choice = { required = true, named = true } }
    "#;

    #[test]
    fn a_model_that_cannot_cache_gets_no_cache_points() -> Result<(), Box<dyn StdError>> {
        // A `cachePoint` block is not ignored by a family that cannot cache:
        // Converse rejects the whole request with a ValidationException, so
        // every call to such a model would fail.
        let mut builder = Request::builder()
            .model("bedrock/llama")
            .system("Be brief")
            .user("First")
            .message(Message::text(Role::Assistant, "Answer"))
            .user("Second")
            .tool(ToolDefinition::function("search", "Searches", json!({})));
        builder = builder.max_output_tokens(64);
        let call = resolved_in(UNCACHED_CATALOG, builder.build()?)?;

        let body = BedrockConverseCodec.encode(&call, false)?.body;
        assert!(!body.to_string().contains("cachePoint"), "{body}");

        // The count-tokens body describes the same prompt, so it drops the
        // markers with it.
        let counted = BedrockConverseCodec
            .encode_count_tokens(&call)
            .ok_or("expected Bedrock to have a count-tokens endpoint")??;
        assert!(
            !counted.body.to_string().contains("cachePoint"),
            "{}",
            counted.body
        );
        Ok(())
    }

    #[test]
    fn a_tool_use_cut_at_the_output_limit_is_dropped() -> Result<(), Box<dyn StdError>> {
        let response = decoded(json!({
            "output": { "message": { "content": [
                { "toolUse": { "toolUseId": "tool-1", "name": "write_note", "input": {} } }
            ] } },
            "stopReason": "max_tokens",
        }))?;

        assert_eq!(response.content, Vec::new());
        assert_eq!(response.finish_reason, FinishReason::Length);
        assert_eq!(response.warnings.len(), 1);
        assert_eq!(response.warnings[0].code, "truncated_tool_call");
        Ok(())
    }

    #[test]
    fn a_context_window_stop_is_a_length_finish() -> Result<(), Box<dyn StdError>> {
        // Converse's own name for a generation that ran out of context. It is
        // the same outcome as `max_tokens`, so it maps onto the same reason.
        let response = decoded(json!({
            "output": { "message": { "content": [{ "text": "Partial" }] } },
            "stopReason": "model_context_window_exceeded",
        }))?;

        assert_eq!(response.finish_reason, FinishReason::Length);
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

        // The sealed blob must not leak through live reasoning deltas.
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, StreamEvent::ReasoningDelta { .. })),
            "a redacted block leaked a reasoning delta: {events:?}"
        );
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
    fn reasoning_text_after_a_redacted_blob_is_dropped() -> Result<(), Box<dyn StdError>> {
        // The blob is the payload the provider verifies on replay; text mixed
        // into the same buffer would corrupt it, so the blob wins.
        let events = streamed(&[
            ("messageStart", json!({ "role": "assistant" })),
            (
                "contentBlockDelta",
                json!({
                    "contentBlockIndex": 0,
                    "delta": { "reasoningContent": { "redactedContent": "sealed-blob" } },
                }),
            ),
            (
                "contentBlockDelta",
                json!({
                    "contentBlockIndex": 0,
                    "delta": { "reasoningContent": { "text": "stray text" } },
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
    fn a_redacted_blob_after_reasoning_text_fails_the_stream() -> Result<(), Box<dyn StdError>> {
        // The text was already delivered, so the blob cannot silently win;
        // assembling text and blob together replays a corrupted sealed
        // payload that Bedrock rejects next turn.
        let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;
        let mut decoder = BedrockConverseCodec.stream_decoder(call.route());
        decoder.decode(SseEvent {
            event: Some("messageStart".to_owned()),
            data:  json!({ "role": "assistant" }).to_string(),
        })?;
        decoder.decode(SseEvent {
            event: Some("contentBlockDelta".to_owned()),
            data:  json!({
                "contentBlockIndex": 0,
                "delta": { "reasoningContent": { "text": "step one" } },
            })
            .to_string(),
        })?;

        let error = decoder
            .decode(SseEvent {
                event: Some("contentBlockDelta".to_owned()),
                data:  json!({
                    "contentBlockIndex": 0,
                    "delta": { "reasoningContent": { "redactedContent": "sealed-blob" } },
                })
                .to_string(),
            })
            .expect_err("a blob landing on a text block must fail the stream");

        assert_eq!(error.kind(), ErrorKind::StreamDecode);
        assert_eq!(error.retry_classification(), RetryClassification::Safe);
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
        let calls: Vec<(&str, Value)> = response
            .content
            .iter()
            .filter_map(|part| match part {
                ContentPart::ToolCall(call) => Some((call.id.as_str(), call.input.wire_value())),
                _ => None,
            })
            .collect();
        assert_eq!(calls, [
            ("call-a", json!({ "q": "rust" })),
            ("call-b", json!({ "id": 1 })),
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
    fn an_empty_text_delta_opens_no_text_block() -> Result<(), Box<dyn StdError>> {
        // An empty delta says nothing. Passing it on opened a text block that
        // carried no text, and the completed response ended with an empty part
        // that re-encodes to nothing.
        let events = streamed(&[
            ("messageStart", json!({ "role": "assistant" })),
            (
                "contentBlockDelta",
                json!({ "contentBlockIndex": 0, "delta": { "text": "" } }),
            ),
            ("contentBlockStop", json!({ "contentBlockIndex": 0 })),
            ("messageStop", json!({ "stopReason": "end_turn" })),
            ("metadata", json!({ "usage": { "inputTokens": 3 } })),
        ])?;

        assert!(
            !events
                .iter()
                .any(|event| matches!(event, StreamEvent::TextDelta { .. })),
            "{events:?}"
        );
        let responses = completed(&events);
        let [response] = responses.as_slice() else {
            return Err(format!("expected one Completed, got {}", responses.len()).into());
        };
        assert!(response.content.is_empty(), "{:?}", response.content);
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
                .ends_with("/model/us.anthropic.claude-sonnet-4-6/count-tokens"),
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
        assert_eq!(error.retry_classification(), RetryClassification::Safe);
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
    fn a_json_tool_result_reaches_the_wire_unflattened() -> Result<(), Box<dyn StdError>> {
        // `toolResult.content` is a block list, and one of its members is
        // `json`, so structured JSON reaches the model as itself. Warning
        // about it said the opposite of what the encoder does, and every
        // agent-loop request with a structured result carried the false
        // warning.
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

        let codes = warnings_for(request.clone())?;
        assert!(codes.is_empty(), "{codes:?}");

        let body = encoded(request)?;
        assert_eq!(
            body["messages"][0]["content"][1]["toolResult"]["content"][0]["json"],
            json!({ "quarters": [1, 2] })
        );
        Ok(())
    }

    #[test]
    fn a_reasoning_tool_result_part_is_dropped_and_reported() -> Result<(), Box<dyn StdError>> {
        // `reasoningContent` has no member of the tool-result union, so
        // encoding it there made Converse reject the whole request. The part
        // is dropped and the caller is told, which is the rule for content the
        // protocol cannot carry.
        let request = Request::builder()
            .model(MODEL)
            .user("Chart it.")
            .message(Message::new(Role::Tool, [ContentPart::ToolResult(
                ToolResult {
                    tool_call_id: "call-1".to_owned(),
                    name:         Some("chart".to_owned()),
                    content:      vec![
                        ContentPart::Reasoning(ReasoningContent {
                            text:             "the third quarter is the outlier".to_owned(),
                            signature:        None,
                            signature_origin: None,
                            redacted:         false,
                        }),
                        ContentPart::Text {
                            text: "Q3 leads.".to_owned(),
                        },
                    ],
                    is_error:     false,
                },
            )]))
            .build()?;

        let codes = warnings_for(request.clone())?;
        assert_eq!(codes, ["unsupported_control"]);

        let body = encoded(request)?;
        let content = &body["messages"][0]["content"][1]["toolResult"]["content"];
        assert_eq!(content, &json!([{ "text": "Q3 leads." }]));
        Ok(())
    }

    #[test]
    fn refuses_a_tool_name_converse_rejects() -> Result<(), Box<dyn StdError>> {
        // Rewriting `mcp.server.tool` to `mcp_server_tool` is lossy, so a
        // decoded call could not be mapped back to the registered tool. The
        // caller owns that mapping because only they can undo it.
        let request = Request::builder()
            .model(MODEL)
            .user("Use the tool.")
            .tool(ToolDefinition::function(
                "mcp.server.tool",
                "A dotted MCP-style name",
                json!({ "type": "object" }),
            ))
            .build()?;

        let Err(error) = BedrockConverseCodec.encode(&resolved(request)?, false) else {
            panic!("a tool name Converse rejects should be refused");
        };

        assert_eq!(error.kind(), ErrorKind::InvalidRequest);
        assert_eq!(error.provider_code(), Some("unsupported_capability"));
        assert!(error.message().contains("mcp.server.tool"));
        Ok(())
    }

    #[test]
    fn accepts_a_tool_name_converse_allows() -> Result<(), Box<dyn StdError>> {
        let request = Request::builder()
            .model(MODEL)
            .user("Use the tool.")
            .tool(ToolDefinition::function(
                "mcp_server_tool-2",
                "An allowed name",
                json!({ "type": "object" }),
            ))
            .build()?;

        let body = encoded(request)?;

        assert_eq!(
            body["toolConfig"]["tools"][0]["toolSpec"]["name"],
            "mcp_server_tool-2"
        );
        Ok(())
    }

    /// A conversation replaying one historical tool call and its result.
    fn replayed_tool_request(name: &str, id: &str) -> Result<Request, Box<dyn StdError>> {
        Ok(Request::builder()
            .model(MODEL)
            .user("What is the weather?")
            .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
                ToolCall::function(id, name, json!({ "city": "Paris" })),
            )]))
            .message(Message::new(Role::Tool, [ContentPart::ToolResult(
                ToolResult {
                    tool_call_id: id.to_owned(),
                    name:         Some(name.to_owned()),
                    content:      vec![ContentPart::Text {
                        text: "18C".to_owned(),
                    }],
                    is_error:     false,
                },
            )]))
            .build()?)
    }

    #[test]
    fn refuses_a_historical_tool_name_converse_rejects() -> Result<(), Box<dyn StdError>> {
        // A conversation that began on another provider carries that
        // provider's identifiers. Sending a dotted name back to Converse dies
        // at AWS with an opaque ValidationException, so it is refused here
        // with the offending value named.
        let request = replayed_tool_request("mcp.server.tool", "call_1")?;

        let Err(error) = BedrockConverseCodec.encode(&resolved(request)?, false) else {
            panic!("a historical tool name Converse rejects should be refused");
        };

        assert_eq!(error.kind(), ErrorKind::InvalidRequest);
        assert_eq!(error.provider_code(), Some("unsupported_capability"));
        assert!(error.message().contains("mcp.server.tool"), "{error}");
        Ok(())
    }

    #[test]
    fn refuses_a_historical_tool_call_id_converse_rejects() -> Result<(), Box<dyn StdError>> {
        // 65 characters, one over the Converse limit. Counting is the whole
        // check here: every character is allowed.
        let long_id = "a".repeat(65);
        let request = replayed_tool_request("get_weather", &long_id)?;
        let call = resolved(request)?;

        let Err(error) = BedrockConverseCodec.encode(&call, false) else {
            panic!("a historical tool call id Converse rejects should be refused");
        };
        assert_eq!(error.kind(), ErrorKind::InvalidRequest);
        assert!(error.message().contains(&long_id), "{error}");

        // Counting sends the same message blocks, so it refuses the same
        // request rather than returning a count for a call that cannot be
        // made.
        let counted = BedrockConverseCodec
            .encode_count_tokens(&call)
            .ok_or("expected Bedrock to have a count-tokens endpoint")?;
        assert_eq!(
            counted.err().map(|error| error.kind()),
            Some(ErrorKind::InvalidRequest)
        );
        Ok(())
    }

    #[test]
    fn accepts_a_tool_call_id_with_dots_and_colons() -> Result<(), Box<dyn StdError>> {
        // `toolUseId` allows `.` and `:` on top of the tool-name set, so an id
        // in that shape replays unchanged.
        let body = encoded(replayed_tool_request(
            "get_weather",
            "functions.get_weather:4",
        )?)?;

        assert_eq!(
            body["messages"][1]["content"][0]["toolUse"]["toolUseId"],
            "functions.get_weather:4"
        );
        assert_eq!(
            body["messages"][2]["content"][0]["toolResult"]["toolUseId"],
            "functions.get_weather:4"
        );
        Ok(())
    }

    #[test]
    fn refuses_a_tool_call_id_on_a_tool_message() -> Result<(), Box<dyn StdError>> {
        // A tool result can ride on the message rather than in a `ToolResult`
        // part, and that id reaches the wire the same way.
        let request = Request::builder()
            .model(MODEL)
            .user("What is the weather?")
            .message(Message::text(Role::Tool, "18C").with_tool_call_id("call one"))
            .build()?;

        let Err(error) = BedrockConverseCodec.encode(&resolved(request)?, false) else {
            panic!("a message tool call id Converse rejects should be refused");
        };

        assert_eq!(error.kind(), ErrorKind::InvalidRequest);
        assert!(error.message().contains("call one"), "{error}");
        Ok(())
    }

    #[test]
    fn a_tool_call_keeps_arguments_that_are_an_object() -> Result<(), Box<dyn StdError>> {
        let request = Request::builder()
            .model(MODEL)
            .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
                ToolCall::function("call-1", "search", json!({ "q": "rust" })),
            )]))
            .build()?;

        let body = encoded(request)?;

        assert_eq!(
            body["messages"][0]["content"][0]["toolUse"]["input"],
            json!({ "q": "rust" })
        );
        Ok(())
    }

    #[test]
    fn a_tool_call_coerces_non_object_arguments_to_an_object() -> Result<(), Box<dyn StdError>> {
        // Converse requires `toolUse.input` to be a JSON object document, so a
        // replayed scalar or array argument value becomes `{}` rather than
        // reaching the wire as a value AWS rejects with a ValidationException.
        for arguments in [json!([1, 2, 3]), json!("rust"), json!(7)] {
            let request = Request::builder()
                .model(MODEL)
                .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
                    ToolCall::function("call-1", "sum", arguments),
                )]))
                .build()?;

            let body = encoded(request)?;

            assert_eq!(
                body["messages"][0]["content"][0]["toolUse"]["input"],
                json!({})
            );
        }
        Ok(())
    }

    #[test]
    fn a_tool_call_with_no_arguments_still_sends_an_object() -> Result<(), Box<dyn StdError>> {
        let request = Request::builder()
            .model(MODEL)
            .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
                ToolCall::function("call-1", "ping", Value::Null),
            )]))
            .build()?;

        let body = encoded(request)?;

        assert_eq!(
            body["messages"][0]["content"][0]["toolUse"]["input"],
            json!({})
        );
        Ok(())
    }

    #[test]
    fn non_text_system_content_is_reported() -> Result<(), Box<dyn StdError>> {
        let request = Request::builder()
            .model(MODEL)
            .message(Message::new(Role::System, [
                ContentPart::Text {
                    text: "Be brief.".to_owned(),
                },
                ContentPart::Json {
                    value: json!({ "style": "terse" }),
                },
            ]))
            .user("Hello")
            .build()?;

        let codes = warnings_for(request)?;

        assert_eq!(codes, ["unsupported_control"]);
        Ok(())
    }
}
