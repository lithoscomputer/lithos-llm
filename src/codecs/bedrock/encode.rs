//! Request encoding: the Converse body, tool configuration, and the
//! Claude thinking controls under `additionalModelRequestFields`.

use reqwest::Method;
use serde_json::{Map, Value, json};

use super::NAMESPACE;
use crate::adapter::ResolvedCall;
use crate::codecs::common::{
    ANTHROPIC_SIGNATURES, endpoint, plain_text, reject_unencodable, sampling, system_text,
    unsupported_capability, wire_options,
};
use crate::resolver::ResolvedRoute;
use crate::transport::EncodedRequest;
use crate::types::{
    ContentPart, Error, MediaSource, Message, ReasoningContent, ReasoningEffort, Request, Role,
    ToolCall, ToolChoice, ToolDefinition, ToolDefinitionKind, ToolResult,
};

/// Encodes the Bedrock `CountTokens` request for one call.
///
/// The body carries the Converse messages and system blocks and nothing else:
/// `inferenceConfig`, `toolConfig`, and the raw provider options all shape
/// generation rather than the prompt, and `CountTokens` rejects them. Cache
/// points stay, because the body must describe the same prompt the Converse
/// call would send.
pub(super) fn encode_count_tokens(call: &ResolvedCall) -> Result<EncodedRequest, Error> {
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
pub(super) fn operation_url(route: &ResolvedRoute, operation: &str) -> String {
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
pub(super) fn reject_custom_tools(request: &Request, route: &ResolvedRoute) -> Result<(), Error> {
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
pub(super) fn caches(route: &ResolvedRoute, auto_cache: bool) -> bool {
    auto_cache && route.model().capabilities().caching().is_supported()
}

/// Whether `toolResult.content` has a block for this part.
///
/// The Converse tool-result union takes text, structured JSON, images, and
/// documents, so all four reach the model as themselves. Anything else —
/// reasoning above all — has no member of that union and is dropped.
pub(super) fn carries_in_tool_result(part: &ContentPart) -> bool {
    matches!(
        part,
        ContentPart::Text { .. }
            | ContentPart::Json { .. }
            | ContentPart::Image(_)
            | ContentPart::Document(_)
    )
}

/// The system blocks, with a cache point after the prompt when caching is on.
pub(super) fn system_blocks(request: &Request, cached: bool) -> Vec<Value> {
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
pub(super) fn conversation(
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
        ContentPart::Reasoning(reasoning)
            if reasoning.has_foreign_signature(ANTHROPIC_SIGNATURES) =>
        {
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
        ContentPart::Audio(_) | ContentPart::Opaque { .. } | ContentPart::Unknown(_) => None,
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
pub(super) fn inference_config(request: &Request) -> Map<String, Value> {
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
pub(super) fn reject_unnameable_tools(
    request: &Request,
    route: &ResolvedRoute,
) -> Result<(), Error> {
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
pub(super) fn tool_config(request: &Request, cached: bool) -> Option<Value> {
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
pub(super) fn history_carries_tool_blocks(request: &Request) -> bool {
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

pub(super) fn bedrock_effort(effort: ReasoningEffort) -> &'static str {
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
pub(super) const MIN_THINKING_BUDGET: u32 = 1024;

/// The output limit the thinking budget scales against.
///
/// The request's own limit wins, then the model's catalog limit, then the
/// same fallback the Anthropic codec uses.
pub(super) fn budget_limit(call: &ResolvedCall) -> u32 {
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
pub(super) fn forces_tool_use(choice: Option<&ToolChoice>) -> bool {
    choice.is_some_and(ToolChoice::is_forced)
}

/// The explicit thinking budget for a reasoning model without effort levels,
/// scaling the same shares of the output limit as the Anthropic codec.
pub(super) fn thinking_budget(effort: ReasoningEffort, limit: u32) -> u32 {
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
