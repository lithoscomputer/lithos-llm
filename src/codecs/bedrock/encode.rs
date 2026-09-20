//! Request encoding: the Converse body, tool configuration, and the
//! Claude thinking controls under `additionalModelRequestFields`.

use reqwest::Method;
use serde_json::{Map, Value, json};

use super::NAMESPACE;
use crate::adapter::ResolvedCall;
use crate::codecs::claude::{Thinking, ThinkingPlan};
use crate::codecs::content::{
    ANTHROPIC_SIGNATURES, Turns, flattens_system_content, flattens_tool_result_content, plain_text,
    reject_audio, system_text,
};
use crate::codecs::errors::unsupported_capability;
use crate::codecs::options::{endpoint, sampling, wire_options};
use crate::resolver::ResolvedRoute;
use crate::transport::EncodedRequest;
use crate::types::{
    ContentPart, Error, MediaSource, Message, ReasoningContent, Request, Role, Speed, ToolCall,
    ToolChoice, ToolDefinition, ToolDefinitionKind, ToolResult,
};

/// Encodes the Bedrock `CountTokens` request for one call.
///
/// The body carries the Converse messages and system blocks and nothing else:
/// `inferenceConfig`, `toolConfig`, and the raw provider options all shape
/// generation rather than the prompt, and `CountTokens` rejects them. Cache
/// points stay, because the body must describe the same prompt the Converse
/// call would send.
pub(super) fn encode_count_tokens(call: &ResolvedCall) -> Result<EncodedRequest, Error> {
    preflight(call)?;
    let request = call.request();
    let route = call.route();
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
        Operation::CountTokens.url(route),
        json!({ "input": { "converse": Value::Object(converse) } }),
    ))
}

/// Refuses what no Bedrock runtime operation can carry.
///
/// Generation and counting share one pre-flight: a request the provider would
/// not accept must not come back with a token count either.
///
/// # Errors
///
/// Returns [`ErrorKind::InvalidRequest`](crate::types::ErrorKind::InvalidRequest)
/// for a custom tool definition, audio content, or a tool identifier
/// Converse cannot carry.
pub(super) fn preflight(call: &ResolvedCall) -> Result<(), Error> {
    let request = call.request();
    let route = call.route();
    reject_custom_tools(request, route)?;
    reject_audio(route, request)?;
    reject_unnameable_tools(request, route)
}

/// The Bedrock runtime operations this codec speaks.
///
/// The three differ only in the last path segment and in whether the request
/// names the event-stream framing it expects.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Operation {
    Converse,
    ConverseStream,
    CountTokens,
}

impl Operation {
    /// The last segment of the operation path.
    fn path(self) -> &'static str {
        match self {
            Self::Converse => "converse",
            Self::ConverseStream => "converse-stream",
            Self::CountTokens => "count-tokens",
        }
    }

    /// The URL of this operation for a route.
    ///
    /// The model id is percent-encoded before it is interpolated into the
    /// path. The reference implementation interpolates it raw, which splits
    /// an ARN-style inference-profile id
    /// (`arn:aws:bedrock:…:inference-profile/name`) into extra path segments
    /// and makes the call unroutable. Encoding here is a deliberate
    /// difference: the signer signs the encoded URL, so client and server
    /// still apply exactly one encoding pass to the same bytes.
    pub(super) fn url(self, route: &ResolvedRoute) -> String {
        let model = route.api_model().replace('%', "%25").replace('/', "%2F");
        endpoint(
            route.provider().base_url(),
            &format!("/model/{model}/{}", self.path()),
        )
    }

    /// The dialect headers this operation sends.
    ///
    /// A streaming request names the framing it expects, which is what the
    /// reference client always sent; a gateway that negotiates content types
    /// answers with the event stream rather than something else.
    pub(super) fn headers(self) -> Vec<(String, String)> {
        match self {
            Self::ConverseStream => vec![(
                "accept".to_owned(),
                "application/vnd.amazon.eventstream".to_owned(),
            )],
            Self::Converse | Self::CountTokens => Vec::new(),
        }
    }
}

/// Builds every typed field of a Converse body.
///
/// Raw provider options are merged over the result by the caller, so a field
/// encoded here is only a default the application can replace. This mirrors
/// the Anthropic codec's `message_body`. `plan` is the reasoning policy for
/// this call; see [`ThinkingPlan`].
///
/// # Errors
///
/// Returns [`ErrorKind::InvalidRequest`](crate::types::ErrorKind::InvalidRequest)
/// for media held behind a URL, which Converse cannot fetch.
pub(super) fn converse_body(
    call: &ResolvedCall,
    auto_cache: bool,
    plan: &ThinkingPlan,
) -> Result<Map<String, Value>, Error> {
    let request = call.request();
    let route = call.route();
    let cached = caches(route, auto_cache);

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
    // carried through `additionalModelRequestFields`; the shared plan decided
    // which. Two retained differences from the Anthropic encoder: `maxTokens`
    // goes on the wire only when a budget forces it — a request that sends
    // none leaves AWS's per-model default in charge, and a default at or
    // below the budget draws a ValidationException — and Bedrock never sends
    // an adaptive thinking object, only `output_config.effort`.
    match plan.thinking {
        Some(Thinking::Budget(budget)) => {
            inference.insert("maxTokens".to_owned(), plan.max_tokens.into());
            body.insert(
                "additionalModelRequestFields".to_owned(),
                json!({ "thinking": { "type": "enabled", "budget_tokens": budget } }),
            );
        }
        Some(Thinking::Adaptive) | None => {}
    }
    if let Some(effort) = plan.effort {
        body.insert(
            "additionalModelRequestFields".to_owned(),
            json!({ "output_config": { "effort": effort } }),
        );
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
    Ok(body)
}

/// The portable controls this request carries that the body did not encode.
///
/// Each is reported as an `unsupported_control` warning: the request still
/// reaches the model, but not as the caller wrote it.
///
/// - Converse has no request-metadata field, so the map is reported rather than
///   folded into some other field where it would change the prompt.
/// - A forced tool choice suppresses the effort, which the plan reports; the
///   same report the Anthropic codec makes for this combination.
/// - The tools stay on the wire despite `tool_choice: none` when the history
///   carries tool blocks, because Converse rejects such a request without a
///   `toolConfig`; the model may therefore still call a tool.
/// - A reasoning part signed by another provider is skipped; see
///   `ReasoningContent::has_foreign_signature`.
/// - Converse has no portable structured-output field, so a caller who asked
///   for JSON gets prose.
/// - The `system` field takes text only.
/// - `toolResult.content` carries text, JSON, images, and documents as
///   themselves; only a part outside that union — reasoning, most of all — is
///   dropped.
pub(super) fn dropped_controls(request: &Request, plan: &ThinkingPlan) -> Vec<&'static str> {
    let mut dropped = Vec::new();
    if !request.metadata().is_empty() {
        dropped.push("request metadata");
    }
    dropped.extend(plan.suppressed.iter().copied());
    if keeps_tools_despite_none(request) {
        dropped.push("tool_choice none alongside historical tool blocks");
    }
    if request.carries_foreign_signature(ANTHROPIC_SIGNATURES) {
        dropped.push("reasoning signed by another provider");
    }
    if request.response_format().is_some() {
        dropped.push("response formats");
    }
    if flattens_system_content(request) {
        dropped.push("non-text system content");
    }
    if flattens_tool_result_content(request, |parts| parts.iter().all(carries_in_tool_result)) {
        dropped.push("tool result content outside text, JSON, and media");
    }
    dropped
}

/// Whether `tool_choice: none` left the tools on the wire; see
/// [`tool_config`].
fn keeps_tools_despite_none(request: &Request) -> bool {
    matches!(request.tool_choice(), Some(ToolChoice::None))
        && !request.tools().is_empty()
        && history_carries_tool_blocks(request)
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
/// blocks. A message whose every part is unencodable is dropped and
/// consecutive same-role turns merge; see [`Turns`].
fn conversation(
    request: &Request,
    route: &ResolvedRoute,
    cached: bool,
) -> Result<Vec<Value>, Error> {
    let mut turns = Turns::default();
    for message in request.messages() {
        if message.is_instruction() {
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
        let role = if message.role() == Role::Assistant {
            "assistant"
        } else {
            // Converse has no tool role; tool results ride in user messages.
            "user"
        };
        turns.push(role, blocks);
    }

    // The cache point is appended as its own block to the turn
    // `prefix_cache_target` names — the same placement the Anthropic codec
    // uses, with a block instead of a field.
    if cached && let Some(blocks) = turns.prefix_cache_target() {
        blocks.push(cache_point());
    }
    Ok(turns.into_values("content"))
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
