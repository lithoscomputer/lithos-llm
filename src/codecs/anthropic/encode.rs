//! Request encoding: the `/v1/messages` body, its headers, and the
//! `count_tokens` narrowing.

use reqwest::Method;
use serde_json::{Map, Value, json};

use super::{
    API_VERSION, BETA_HEADERS_OPTION, COUNT_TOKENS_FIELDS, DEFAULT_MAX_TOKENS, FAST_MODE_BETA,
    JSON_OBJECT_INSTRUCTION, MIN_THINKING_BUDGET, NAMESPACE,
};
use crate::adapter::ResolvedCall;
use crate::codecs::content::{
    ANTHROPIC_SIGNATURES, Turns, flattens_system_content, flattens_tool_result_content, plain_text,
    reject_audio, system_text,
};
use crate::codecs::errors::unsupported_capability;
use crate::codecs::options::{endpoint, merge_options, sampling, wire_options};
use crate::resolver::ResolvedRoute;
use crate::transport::EncodedRequest;
use crate::types::{
    ContentPart, Error, MediaSource, Message, ReasoningEffort, Request, ResponseFormat, Role,
    Speed, ToolCallKind, ToolChoice, ToolDefinition, ToolDefinitionKind,
};

/// The dialect headers both endpoints send.
///
/// Authentication is the transport's job; this is the protocol version and,
/// when the request needs one, the `anthropic-beta` opt-in list. Anthropic
/// takes several betas in one comma-separated header value.
pub(super) fn headers(betas: &[String]) -> Vec<(String, String)> {
    let mut headers = vec![("anthropic-version".to_owned(), API_VERSION.to_owned())];
    if !betas.is_empty() {
        headers.push(("anthropic-beta".to_owned(), betas.join(",")));
    }
    headers
}

/// The betas this request opts into, consuming the option that names them.
///
/// `beta_headers` is a header control rather than a body field, so it is taken
/// out of the raw options before they are merged: left in, it would land in
/// the JSON body and the endpoint would reject the request. The key is removed
/// whatever its value, so a malformed option cannot reach the wire either.
///
/// The fast tier is added on top, because the body field alone does nothing
/// without its beta. A caller who already listed it does not get it twice.
pub(super) fn beta_headers(options: &mut Map<String, Value>, speed: Option<Speed>) -> Vec<String> {
    let named = options.remove(BETA_HEADERS_OPTION);
    let mut betas: Vec<String> = named
        .as_ref()
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default();

    if speed == Some(Speed::Fast) && !betas.iter().any(|beta| beta == FAST_MODE_BETA) {
        betas.push(FAST_MODE_BETA.to_owned());
    }
    betas
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
            matches!(part, ContentPart::ToolCall(tool_call) if tool_call.input.kind() == ToolCallKind::Custom)
        });

    if defined || called {
        return Err(unsupported_capability(call.route(), "custom tools"));
    }
    Ok(())
}

/// Refuses what neither endpoint of this protocol can carry.
///
/// Generation and counting share one pre-flight: a request the provider would
/// not accept must not come back with a token count either.
///
/// # Errors
///
/// Returns [`ErrorKind::InvalidRequest`](crate::types::ErrorKind::InvalidRequest)
/// for a custom tool, a replayed custom tool call, or audio content.
pub(super) fn preflight(call: &ResolvedCall) -> Result<(), Error> {
    reject_custom_tools(call)?;
    reject_audio(call.route(), call.request())
}

/// The portable controls this request carries that the body did not encode.
///
/// Each is reported as an `unsupported_control` warning: the request still
/// reaches the model, but not as the caller wrote it. `raw_thinking` says a
/// raw `thinking` provider option is in the body.
///
/// - The `system` field takes text only, so anything else a system message
///   carries is dropped; the text still arrives.
/// - A tool result carries text, JSON, and images as blocks; anything else
///   flattens to text.
/// - A reasoning part signed by another provider is skipped; see
///   `ReasoningContent::has_foreign_signature`.
/// - A forced tool choice drops both output controls; see `forces_tool_use`. A
///   raw `thinking` option stays in the body — raw options are authoritative —
///   but Anthropic rejects the pair, so it is reported too.
pub(super) fn dropped_controls(request: &Request, raw_thinking: bool) -> Vec<&'static str> {
    let mut dropped = Vec::new();
    if flattens_system_content(request) {
        dropped.push("non-text system content");
    }
    if flattens_tool_result_content(request, carries_in_tool_result) {
        dropped.push("non-text tool result content");
    }
    if request.carries_foreign_signature(ANTHROPIC_SIGNATURES) {
        dropped.push("reasoning signed by another provider");
    }
    if forces_tool_use(request.tool_choice()) {
        if request.reasoning_effort().is_some() {
            dropped.push("reasoning effort with a forced tool choice");
        }
        if request.response_format().and_then(json_schema).is_some() {
            dropped.push("structured output with a forced tool choice");
        }
        if raw_thinking {
            dropped.push("a thinking provider option with a forced tool choice");
        }
    }
    dropped
}

/// Whether a tool result's parts all have a block of their own; see
/// [`tool_result_content`].
fn carries_in_tool_result(parts: &[ContentPart]) -> bool {
    parts.iter().all(|part| {
        matches!(
            part,
            ContentPart::Text { .. } | ContentPart::Json { .. } | ContentPart::Image(_)
        )
    })
}

/// Encodes the count-token request, which narrows the generation body.
pub(super) fn count_tokens_request(call: &ResolvedCall) -> Result<EncodedRequest, Error> {
    preflight(call)?;

    let (mut options, controls) = wire_options(call);
    // Counting sends the same betas generation sends: a beta can change what
    // the body means, and a count taken under different terms is not the
    // count the generation will be billed for.
    let betas = beta_headers(&mut options, call.request().speed());
    let mut body = message_body(call, controls.auto_cache, options.contains_key("thinking"));
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
    .with_headers(headers(&betas)))
}

/// Builds every typed field of a Messages body.
///
/// Raw provider options are merged over the result by the caller, so a field
/// encoded here is only a default the application can replace.
///
/// `thinking_overridden` says a raw `thinking` option will replace the
/// derived object wholesale. The recursive option merge replaces matching
/// keys only, so a derived budget under a raw `{"type": "disabled"}` would
/// leave a stray `budget_tokens` the API rejects; with the override the codec
/// derives no object and keeps `max_tokens` unlifted, leaving both entirely
/// to the caller.
pub(super) fn message_body(
    call: &ResolvedCall,
    auto_cache: bool,
    thinking_overridden: bool,
) -> Map<String, Value> {
    let request = call.request();
    let route = call.route();
    // A model that cannot cache would reject the breakpoints outright.
    let cached = auto_cache && route.model().capabilities().caching().is_supported();

    let mut body = Map::new();
    body.insert("model".to_owned(), route.api_model().into());

    // A model that takes system turns keeps only the leading system run in
    // the top-level field; a later system message stays in the conversation.
    // Everywhere else every system message is hoisted, the one encoding
    // those models take.
    let system_turns = route.model().protocol_options().system_turns;
    let hoisted = if system_turns {
        leading_system_run(request.messages())
    } else {
        request.messages()
    };
    let mut system = system_text(hoisted);
    // Free-form JSON output rides on the system text rather than a schema;
    // see [`JSON_OBJECT_INSTRUCTION`]. The count endpoint keeps `system`, so
    // counting sees the same instruction generation sends.
    if matches!(request.response_format(), Some(ResponseFormat::JsonObject)) {
        if !system.is_empty() {
            system.push_str("\n\n");
        }
        system.push_str(JSON_OBJECT_INSTRUCTION);
    }
    if !system.is_empty() {
        body.insert("system".to_owned(), system_value(system, cached));
    }

    let mut turns = wire_turns(request.messages(), system_turns);
    if cached {
        mark_conversation_prefix(&mut turns);
    }
    body.insert(
        "messages".to_owned(),
        Value::Array(turns.into_values("content")),
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

    // Effort has two wire dialects. A model with effort levels takes
    // `output_config.effort` and an adaptive thinking object; an older
    // reasoning model takes an explicit `thinking` budget instead. The budget
    // must sit strictly below `max_tokens`, so the limit grows when the budget
    // would not fit under it. The count endpoint drops `max_tokens` but keeps
    // `thinking`, which is why both are encoded here rather than per endpoint.
    let thinking_allowed = !forces_tool_use(request.tool_choice());
    let mut max_tokens = output_limit(call);
    if thinking_allowed && !thinking_overridden {
        if let Some(budget) = thinking_budget(call, max_tokens) {
            if max_tokens <= budget {
                max_tokens = budget.saturating_add(MIN_THINKING_BUDGET);
            }
            body.insert(
                "thinking".to_owned(),
                json!({ "type": "enabled", "budget_tokens": budget }),
            );
        } else if takes_adaptive_thinking(route) {
            body.insert("thinking".to_owned(), json!({ "type": "adaptive" }));
        }
    }
    body.insert("max_tokens".to_owned(), max_tokens.into());

    if thinking_allowed {
        let output_config = output_config(call);
        if !output_config.is_empty() {
            body.insert("output_config".to_owned(), output_config.into());
        }
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
pub(super) fn output_config(call: &ResolvedCall) -> Map<String, Value> {
    let request = call.request();
    let mut config = Map::new();

    // A model without effort levels gets a thinking budget instead; sending
    // `effort` too would ask the provider to honor a control the model does
    // not take.
    if let Some(effort) = request.reasoning_effort()
        && takes_effort_levels(call.route())
    {
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
///
/// `JsonObject` names no schema: it becomes the system-text instruction
/// [`JSON_OBJECT_INSTRUCTION`] instead, because no schema in Anthropic's
/// structured-output subset says "any JSON object".
fn json_schema(format: &ResponseFormat) -> Option<Value> {
    match format {
        ResponseFormat::Text | ResponseFormat::JsonObject => None,
        ResponseFormat::JsonSchema { schema, .. } => Some(schema.clone()),
    }
}

/// Maps the normalized reasoning effort onto Anthropic's levels.
fn anthropic_effort(effort: ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::Minimal | ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::Xhigh => "xhigh",
        ReasoningEffort::Max => "max",
    }
}

/// The output-token limit this request sends as `max_tokens`.
///
/// Anthropic requires the field. The request's own limit wins; a request that
/// names none takes the model's catalog limit, which is the largest answer the
/// model can give, and only a model the catalog records no limit for falls back
/// to [`DEFAULT_MAX_TOKENS`]. A small fixed default here would cut off a long
/// generation with nothing but a `max_tokens` finish reason to show for it.
fn output_limit(call: &ResolvedCall) -> u32 {
    if let Some(tokens) = call.request().max_output_tokens() {
        return tokens;
    }
    // A catalog limit past `u32` is not a real model limit, so a request that
    // meets one keeps the largest value the field can carry.
    call.route()
        .model()
        .limits()
        .map_or(DEFAULT_MAX_TOKENS, |limits| {
            u32::try_from(limits.max_output_tokens).unwrap_or(u32::MAX)
        })
}

/// Whether this model wants an adaptive thinking object on every request.
///
/// A model with effort levels lets the provider size its own thinking, but it
/// has to be told to: without a `thinking` object the model does not reason at
/// all, which changes answer quality, latency, and spend for a caller who asked
/// for nothing unusual. Effort does not replace it — effort guides how the
/// allocation is spent — so a levels model gets both.
///
/// A model without levels is either natively adaptive, and rejects the toggle,
/// or takes the explicit budget [`thinking_budget`] computes.
///
/// A caller who wants something else sets `thinking` in the raw provider
/// options, which is merged over this.
fn takes_adaptive_thinking(route: &ResolvedRoute) -> bool {
    route.model().protocol_options().reasoning_effort_levels
}

/// Whether effort encodes as `output_config.effort` for this model.
///
/// A passthrough model is uncataloged precisely because it is newer than the
/// catalog, so the modern effort dialect is the safer guess — the one the
/// reference client made for unknown models. Guessing a thinking budget
/// instead would send a manual toggle the always-adaptive models reject. The
/// adaptive thinking object stays gated on the declared capability, so a
/// passthrough request without an effort is encoded exactly as before.
fn takes_effort_levels(route: &ResolvedRoute) -> bool {
    let model = route.model();
    model.protocol_options().reasoning_effort_levels || model.is_passthrough()
}

/// Whether the tool choice makes a tool call mandatory.
///
/// Anthropic rejects extended thinking together with a forced tool choice, so
/// a forced choice suppresses both thinking and `output_config`. `auto` and
/// `none` leave the model free to answer in prose and keep them.
fn forces_tool_use(choice: Option<&ToolChoice>) -> bool {
    choice.is_some_and(ToolChoice::is_forced)
}

/// The explicit thinking budget for a model without effort levels.
///
/// `None` when the request sets no effort or the model takes
/// `output_config.effort` directly. The budget scales the same way effort
/// levels scale — a share of the output limit — with the provider floor of
/// [`MIN_THINKING_BUDGET`]. `Minimal` shares `Low`'s budget for the same
/// reason [`anthropic_effort`] collapses them: the dialect has no smaller
/// step.
fn thinking_budget(call: &ResolvedCall, limit: u32) -> Option<u32> {
    let effort = call.request().reasoning_effort()?;
    if takes_effort_levels(call.route()) {
        return None;
    }

    let limit = u64::from(limit);
    let share = match effort {
        ReasoningEffort::Minimal | ReasoningEffort::Low => limit / 4,
        ReasoningEffort::Medium => limit / 2,
        ReasoningEffort::High => limit * 3 / 4,
        ReasoningEffort::Xhigh => limit * 7 / 8,
        ReasoningEffort::Max => limit,
    };
    let budget = share.max(u64::from(MIN_THINKING_BUDGET));
    Some(u32::try_from(budget).unwrap_or(u32::MAX))
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

/// The system and developer messages before the first conversational turn.
///
/// This run is the system prompt proper: it is hoisted into the top-level
/// field on every model. What follows it is the conversation, and a system
/// message inside the conversation is a mid-conversation instruction, which
/// only some models take in place.
fn leading_system_run(messages: &[Message]) -> &[Message] {
    let end = messages
        .iter()
        .position(|message| !message.is_instruction())
        .unwrap_or(messages.len());
    &messages[..end]
}

/// Translates the conversation, dropping the turns Anthropic cannot carry.
///
/// System and developer turns are hoisted into `system` — all of them, or
/// only the leading run when `system_turns` says the model takes a later one
/// in place. A system turn carries text only, the same as the hoisted field,
/// so `flattens_system_content` reports whatever else such a message held.
/// Every remaining non-assistant role — a tool result above all — is a `user`
/// turn. A turn whose parts all encode to nothing is dropped, and consecutive
/// same-role turns merge; see [`Turns`].
///
/// Turns go out in the order given. Anthropic requires a system turn to
/// follow a `user` turn and to be last or followed by an `assistant` turn;
/// the codec does not reorder or fold a misplaced one, so the provider's
/// placement error reaches the caller as an invalid request.
fn wire_turns(messages: &[Message], system_turns: bool) -> Turns {
    let leading = leading_system_run(messages).len();
    let mut turns = Turns::default();
    for (index, message) in messages.iter().enumerate() {
        let role = match message.role() {
            Role::System | Role::Developer if system_turns && index >= leading => "system",
            Role::System | Role::Developer => continue,
            Role::Assistant => "assistant",
            _ => "user",
        };
        let blocks: Vec<Value> = if role == "system" {
            let text = plain_text(message.content());
            if text.trim().is_empty() {
                Vec::new()
            } else {
                vec![json!({ "type": "text", "text": text })]
            }
        } else {
            message.content().iter().filter_map(content_block).collect()
        };
        turns.push(role, blocks);
    }
    turns
}

/// Encodes the content a tool returned.
///
/// This protocol takes an array of blocks here, not just a string, and it
/// accepts text and image blocks. Encoding them keeps an image a tool produced
/// instead of flattening the result to its text and losing the picture.
///
/// Structured JSON a tool returned has no block of its own, so it travels as
/// the text a tool would have printed — the same translation message content
/// uses. Filtering it out instead would send the model a result the tool never
/// produced, and a JSON-only result would arrive empty.
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
            ContentPart::Json { value } => Some(json!({
                "type": "text",
                "text": value.to_string(),
            })),
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
/// The breakpoint goes on the last block of the turn
/// [`Turns::prefix_cache_target`] names, as a `cache_control` field.
fn mark_conversation_prefix(turns: &mut Turns) {
    if let Some(block) = turns
        .prefix_cache_target()
        .and_then(|blocks| blocks.last_mut())
    {
        mark_cached(block);
    }
}

/// Adds an ephemeral cache breakpoint to one wire object.
fn mark_cached(value: &mut Value) {
    let Some(object) = value.as_object_mut() else {
        return;
    };
    object.insert("cache_control".to_owned(), json!({ "type": "ephemeral" }));
}

/// Encodes one content part, or `None` when Anthropic has no block for it.
pub(super) fn content_block(part: &ContentPart) -> Option<Value> {
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
        // A signature another provider family minted cannot verify here and
        // fails the request, so the part is skipped; the encoder reports it.
        ContentPart::Reasoning(reasoning)
            if reasoning.has_foreign_signature(ANTHROPIC_SIGNATURES) =>
        {
            None
        }
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
            "input": tool_call.input.wire_value(),
        })),
        ContentPart::ToolResult(result) => Some(json!({
            "type": "tool_result",
            "tool_use_id": result.tool_call_id,
            "content": tool_result_content(&result.content),
            "is_error": result.is_error,
        })),
        // A part this codec produced replays verbatim; one belonging to another
        // provider is skipped so failover still encodes.
        ContentPart::Opaque { data, .. } if part.opaque_namespace() == Some(NAMESPACE) => {
            Some(data.clone())
        }
        // The Messages API has no audio input; it is rejected before dispatch
        // by `reject_audio`.
        ContentPart::Opaque { .. } | ContentPart::Audio(_) | ContentPart::Unknown(_) => None,
    }
}

/// Encodes a media source as an Anthropic `source` object.
fn media_source(source: &MediaSource) -> Value {
    match source {
        MediaSource::Url { url, .. } => json!({ "type": "url", "url": url }),
        MediaSource::Base64 { data, media_type } => json!({
            "type": "base64",
            "media_type": media_type,
            "data": data,
        }),
    }
}
