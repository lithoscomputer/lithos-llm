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
    ANTHROPIC_SIGNATURES, carries_foreign_signature, endpoint, finish_reason,
    flattens_system_content, flattens_tool_result_content, foreign_signature, merge_options,
    plain_text, refusal, reject_unencodable, sampling, system_text, unsupported_capability,
    wire_options,
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

/// The `max_tokens` used when neither the request nor the catalog sets one.
///
/// Anthropic requires the field, so there is no "omit it" option. This is the
/// last resort: a request that names no limit and a model the catalog records
/// no output limit for. It is deliberately generous, because a limit picked
/// here truncates a long generation silently.
const DEFAULT_MAX_TOKENS: u32 = 65_536;

/// The beta the `speed: "fast"` tier is gated behind.
///
/// The body field alone is not enough: without this beta the endpoint ignores
/// the tier, so the two always travel together.
const FAST_MODE_BETA: &str = "fast-mode-2026-02-01";

/// The raw provider option that names extra `anthropic-beta` values.
///
/// This is a header control, not a body field, so the codec consumes it before
/// the remaining options are merged into the body.
const BETA_HEADERS_OPTION: &str = "beta_headers";

/// The smallest `thinking.budget_tokens` the API accepts.
///
/// Doubles as the headroom kept above the budget when the output limit must
/// grow, because the budget has to sit strictly below `max_tokens`.
const MIN_THINKING_BUDGET: u32 = 1024;

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
        let (mut options, controls) = wire_options(call);
        let betas = beta_headers(&mut options, request.speed());
        let mut body = message_body(call, controls.auto_cache);
        body.insert("stream".to_owned(), stream.into());
        merge_options(&mut body, options);

        let mut encoded = EncodedRequest::new(
            Method::POST,
            endpoint(call.route().provider().base_url(), "/v1/messages"),
            Value::Object(body),
        )
        .with_headers(headers(&betas))
        .with_timeout(request.timeout())
        .with_applied_speed(request.speed());
        // The system field of this protocol takes text only, so anything else
        // a system message carries is dropped. The text still reaches the
        // model, so it is reported rather than refused.
        if flattens_system_content(request) {
            encoded = encoded.unsupported_control("non-text system content");
        }
        if flattens_tool_result_content(request, |parts| {
            parts.iter().all(|part| {
                matches!(
                    part,
                    ContentPart::Text { .. } | ContentPart::Json { .. } | ContentPart::Image(_)
                )
            })
        }) {
            encoded = encoded.unsupported_control("non-text tool result content");
        }
        // A skipped foreign-signed reasoning part never reaches the model,
        // so the skip is reported; see `foreign_signature`.
        if carries_foreign_signature(request, ANTHROPIC_SIGNATURES) {
            encoded = encoded.unsupported_control("reasoning signed by another provider");
        }
        // A forced tool choice drops both output controls; see
        // `forces_tool_use`. Neither reaches the model, so both are reported.
        if forces_tool_use(request.tool_choice()) {
            if request.reasoning_effort().is_some() {
                encoded = encoded.unsupported_control("reasoning effort with a forced tool choice");
            }
            if request.response_format().and_then(json_schema).is_some() {
                encoded =
                    encoded.unsupported_control("structured output with a forced tool choice");
            }
        }
        Ok(encoded)
    }

    fn decode_response(&self, route: &ResolvedRoute, value: Value) -> Result<Response, Error> {
        // A refusal is a failure, not a short answer; failing here keeps it
        // visible to the caller and the retry and failover middleware.
        if value.get("stop_reason").and_then(Value::as_str) == Some("refusal") {
            let explanation = value
                .pointer("/stop_details/explanation")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
            return Err(refusal(route, explanation.as_deref(), Some(value)));
        }
        // A body that carries none of the fields every Messages response has
        // is not a Messages response: a gateway error page, an empty
        // envelope, or another protocol answering on this URL. Decoding it as
        // an empty success would report a model that said nothing.
        if let Some(field) = missing_response_field(&value) {
            return Err(Error::new(
                ErrorKind::ResponseDecode,
                format!("Anthropic returned a response without the {field} field"),
            )
            .with_provider(route.provider().id().clone())
            .with_raw_data(value));
        }

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
            opaque:    BTreeMap::new(),
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
/// Authentication is the transport's job; this is the protocol version and,
/// when the request needs one, the `anthropic-beta` opt-in list. Anthropic
/// takes several betas in one comma-separated header value.
fn headers(betas: &[String]) -> Vec<(String, String)> {
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
fn beta_headers(options: &mut Map<String, Value>, speed: Option<Speed>) -> Vec<String> {
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

/// The first field a Messages response must carry and this body does not.
///
/// Every real response carries all four, so their absence means the body is
/// not one. Fields the API genuinely omits — `stop_reason` on a streamed
/// message, `stop_details` on anything but a refusal — stay optional.
fn missing_response_field(value: &Value) -> Option<&'static str> {
    if !value.get("id").is_some_and(Value::is_string) {
        return Some("id");
    }
    if !value.get("model").is_some_and(Value::is_string) {
        return Some("model");
    }
    if !value.get("content").is_some_and(Value::is_array) {
        return Some("content");
    }
    if !value.get("usage").is_some_and(Value::is_object) {
        return Some("usage");
    }
    None
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

    let (mut options, controls) = wire_options(call);
    // Counting sends the same betas generation sends: a beta can change what
    // the body means, and a count taken under different terms is not the
    // count the generation will be billed for.
    let betas = beta_headers(&mut options, call.request().speed());
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
    .with_headers(headers(&betas))
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

    // Effort has two wire dialects. A model with effort levels takes
    // `output_config.effort` and an adaptive thinking object; an older
    // reasoning model takes an explicit `thinking` budget instead. The budget
    // must sit strictly below `max_tokens`, so the limit grows when the budget
    // would not fit under it. The count endpoint drops `max_tokens` but keeps
    // `thinking`, which is why both are encoded here rather than per endpoint.
    let thinking_allowed = !forces_tool_use(request.tool_choice());
    let mut max_tokens = output_limit(call);
    if thinking_allowed {
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
fn output_config(call: &ResolvedCall) -> Map<String, Value> {
    let request = call.request();
    let mut config = Map::new();

    // A model without effort levels gets a thinking budget instead; sending
    // `effort` too would ask the provider to honor a control the model does
    // not take.
    if let Some(effort) = request.reasoning_effort() {
        if call.route().model().capabilities().reasoning_effort_levels {
            config.insert("effort".to_owned(), anthropic_effort(effort).into());
        }
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
    route.model().capabilities().reasoning_effort_levels
}

/// Whether the tool choice makes a tool call mandatory.
///
/// Anthropic rejects extended thinking together with a forced tool choice, so
/// a forced choice suppresses both thinking and `output_config`. `auto` and
/// `none` leave the model free to answer in prose and keep them.
fn forces_tool_use(choice: Option<&ToolChoice>) -> bool {
    matches!(choice, Some(ToolChoice::Required | ToolChoice::Tool { .. }))
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
    if call.route().model().capabilities().reasoning_effort_levels {
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
        // A signature another provider family minted cannot verify here and
        // fails the request, so the part is skipped; the encoder reports it.
        ContentPart::Reasoning(reasoning) if foreign_signature(reasoning, ANTHROPIC_SIGNATURES) => {
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
        Some("thinking") => {
            let signature = block
                .get("signature")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
            let signature_origin = signature.is_some().then(|| ANTHROPIC_SIGNATURES.to_owned());
            Some(ContentPart::Reasoning(ReasoningContent {
                text: field(block, "thinking").to_owned(),
                signature,
                signature_origin,
                redacted: false,
            }))
        }
        // The encrypted blob is the whole block. It carries no signature and
        // must be replayed exactly as it arrived.
        Some("redacted_thinking") => Some(ContentPart::Reasoning(ReasoningContent {
            text:             field(block, "data").to_owned(),
            signature:        None,
            signature_origin: None,
            redacted:         true,
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
    /// Unknown server-side blocks that may stream their `input`.
    ///
    /// A `server_tool_use` block arrives as a start snapshot with an empty
    /// `input` and streams the real value through `input_json_delta`. The
    /// fragments accumulate here next to the snapshot and fold back into it
    /// when the block closes, so the replayed block matches what the blocking
    /// decoder keeps whole.
    opaque:    BTreeMap<ContentBlockId, (Value, String)>,
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
            Some("content_block_stop") => {
                let id = block_id(&value);
                events.extend(self.flush_opaque(&id));
                events.extend(self.assembler.end(&id));
            }
            Some("message_delta") => {
                if let Some(reason) = value.pointer("/delta/stop_reason").and_then(Value::as_str) {
                    // A refusal fails the stream here, before `message_stop`
                    // can complete it as a success.
                    if reason == "refusal" {
                        let explanation = value
                            .pointer("/delta/stop_details/explanation")
                            .and_then(Value::as_str)
                            .map(ToOwned::to_owned);
                        return Err(refusal(&self.route, explanation.as_deref(), Some(value)));
                    }
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
        let ids: Vec<ContentBlockId> = self.opaque.keys().cloned().collect();
        let mut events = Vec::new();
        for id in &ids {
            events.extend(self.flush_opaque(id));
        }
        events.extend(self.assembler.complete());
        Ok(events)
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
                self.opaque.insert(id, (block.clone(), String::new()));
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
            Some("input_json_delta") => {
                let fragment = field(delta, "partial_json");
                // A server-side block streams its `input` the same way a tool
                // call does; the fragments belong to the snapshot, not to a
                // tool-call buffer.
                if let Some((_, buffer)) = self.opaque.get_mut(&id) {
                    buffer.push_str(fragment);
                    return Vec::new();
                }
                self.assembler.arguments(&id, fragment)
            }
            _ => Vec::new(),
        }
    }

    /// Folds a server-side block's streamed input back into its snapshot.
    ///
    /// Runs when the block closes, and again at end of stream for a block a
    /// truncated stream never closed. Fragments that do not parse leave the
    /// start snapshot untouched, which is the pre-accumulation behavior.
    fn flush_opaque(&mut self, id: &ContentBlockId) -> Vec<StreamEvent> {
        let Some((mut payload, buffer)) = self.opaque.remove(id) else {
            return Vec::new();
        };
        if buffer.is_empty() {
            return Vec::new();
        }
        let Ok(input) = serde_json::from_str::<Value>(&buffer) else {
            return Vec::new();
        };
        if let Some(object) = payload.as_object_mut() {
            object.insert("input".to_owned(), input);
        }
        self.assembler.set_opaque_data(id, payload)
    }
}

#[cfg(all(test, feature = "builtin-catalog"))]
mod tests {
    use std::error::Error as StdError;

    use serde_json::{Value, json};

    use super::{AnthropicMessagesCodec, Codec};
    use crate::codecs::test_support::{resolved, resolved_in};
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
    fn a_streamed_server_tool_use_block_keeps_its_streamed_input() -> Result<(), Box<dyn StdError>>
    {
        // A server-side block opens with an empty `input` and streams the
        // real value through `input_json_delta`. The assembled opaque part
        // must carry the streamed input — the shape the blocking decoder
        // keeps whole — or a replay misrepresents what the model did.
        let events = stream(vec![
            sse(
                "message_start",
                &json!({ "type": "message_start", "message": { "id": "msg_1" } }),
            ),
            sse(
                "content_block_start",
                &json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "server_tool_use",
                        "id": "srvtoolu_1",
                        "name": "web_search",
                        "input": {},
                    },
                }),
            ),
            sse(
                "content_block_delta",
                &json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": { "type": "input_json_delta", "partial_json": "{\"query\":" },
                }),
            ),
            sse(
                "content_block_delta",
                &json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": { "type": "input_json_delta", "partial_json": "\"rust\"}" },
                }),
            ),
            sse(
                "content_block_stop",
                &json!({ "type": "content_block_stop", "index": 0 }),
            ),
            sse("message_stop", &json!({ "type": "message_stop" })),
        ])?;

        let parts: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::ContentBlockEnd { part, .. } => Some(part.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(parts, vec![ContentPart::Opaque {
            kind: "anthropic.server_tool_use".to_owned(),
            data: json!({
                "type": "server_tool_use",
                "id": "srvtoolu_1",
                "name": "web_search",
                "input": { "query": "rust" },
            }),
        }]);
        Ok(())
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
        // The speed reached the wire, so cost estimation may price it.
        assert_eq!(encoded.applied_speed, Some(Speed::Fast));
        assert_eq!(encoded.body["stop_sequences"], json!(["END", "STOP"]));
        assert_eq!(encoded.body["metadata"], json!({ "user_id": "u-1" }));
        // The model takes effort levels, so it also gets the adaptive
        // thinking object, and no request limit means the model's own.
        assert_eq!(encoded.body["thinking"], json!({ "type": "adaptive" }));
        assert_eq!(encoded.body["max_tokens"], 64000);
        assert_eq!(encoded.body["stream"], false);
        assert!(encoded.url.ends_with("/v1/messages"));
        assert!(
            encoded
                .headers
                .contains(&("anthropic-version".to_owned(), "2023-06-01".to_owned()))
        );
        // The fast tier is a beta, so the body field travels with its header.
        assert!(encoded.headers.contains(&(
            "anthropic-beta".to_owned(),
            "fast-mode-2026-02-01".to_owned()
        )));

        let response = codec.decode_response(
            call.route(),
            json!({
                "id": "msg-1",
                "model": "claude-sonnet-4-6",
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
    fn a_foreign_signed_reasoning_part_is_skipped_with_a_warning() -> Result<(), Box<dyn StdError>>
    {
        // A Gemini-minted signature cannot verify here; replaying it fails
        // the request, so the part is dropped and the drop is reported. A
        // signed part with no recorded origin — persisted before origins
        // existed — still replays.
        let foreign = ReasoningContent {
            text:             "thought".to_owned(),
            signature:        Some("gemini-sig".to_owned()),
            signature_origin: Some("gemini".to_owned()),
            redacted:         false,
        };
        let legacy = ReasoningContent {
            text:             "older thought".to_owned(),
            signature:        Some("sig".to_owned()),
            signature_origin: None,
            redacted:         false,
        };
        let call = resolved(
            Request::builder()
                .model(MODEL)
                .user("Hello")
                .message(Message::new(Role::Assistant, [
                    ContentPart::Reasoning(foreign),
                    ContentPart::Reasoning(legacy),
                    ContentPart::Text {
                        text: "answer".to_owned(),
                    },
                ]))
                .user("Continue")
                .build()?,
        )?;

        let encoded = AnthropicMessagesCodec.encode(&call, false)?;

        let body = encoded.body.to_string();
        assert!(!body.contains("gemini-sig"), "{body}");
        assert!(body.contains("older thought"), "{body}");
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
    fn a_whitespace_only_system_prompt_is_omitted() -> Result<(), Box<dyn StdError>> {
        // Templating commonly leaves a system prompt of pure whitespace. The
        // old encoder dropped it, and with auto-cache on it would otherwise
        // become a blank cached text block the provider rejects.
        let call = resolved(
            Request::builder()
                .model(MODEL)
                .system("   \n\t")
                .user("Hello")
                .build()?,
        )?;

        let encoded = AnthropicMessagesCodec.encode(&call, false)?;

        assert_eq!(encoded.body.get("system"), None);
        Ok(())
    }

    #[test]
    fn usage_buckets_stay_disjoint_without_subtraction() -> Result<(), Box<dyn StdError>> {
        let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;

        let response = AnthropicMessagesCodec.decode_response(
            call.route(),
            json!({
                "id": "msg-1",
                "model": "claude-sonnet-4-6",
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
                "id": "msg-1",
                "model": "claude-sonnet-4-6",
                "content": [{ "type": "redacted_thinking", "data": "ENCRYPTED" }],
                "stop_reason": "end_turn",
                "usage": { "input_tokens": 4, "output_tokens": 2 }
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
                        text:             "ENCRYPTED".to_owned(),
                        signature:        None,
                        signature_origin: None,
                        redacted:         true,
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
    fn a_refusal_response_fails_instead_of_decoding() -> Result<(), Box<dyn StdError>> {
        let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;
        let body = json!({
            "id": "msg-1",
            "content": [],
            "stop_reason": "refusal",
            "stop_details": { "explanation": "it asks for malware" }
        });

        let error = AnthropicMessagesCodec
            .decode_response(call.route(), body.clone())
            .expect_err("a refusal must fail the call");

        assert_eq!(error.kind(), ErrorKind::ContentFilter);
        assert_eq!(error.provider_code(), Some("refusal"));
        assert!(error.message().contains("it asks for malware"), "{error}");
        assert_eq!(error.raw_data(), Some(&body));
        Ok(())
    }

    #[test]
    fn a_streamed_refusal_ends_the_stream_as_an_error() -> Result<(), Box<dyn StdError>> {
        let call = resolved(Request::builder().model(MODEL).user("Hello").build()?)?;
        let mut decoder = AnthropicMessagesCodec.stream_decoder(call.route());

        decoder.decode(sse(
            "message_start",
            &json!({
                "type": "message_start",
                "message": { "id": "msg-1", "usage": { "input_tokens": 3 } }
            }),
        ))?;
        let error = decoder
            .decode(sse(
                "message_delta",
                &json!({
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": "refusal",
                        "stop_details": { "explanation": "it asks for malware" }
                    }
                }),
            ))
            .expect_err("a refusal must fail the stream");

        assert_eq!(error.kind(), ErrorKind::ContentFilter);
        assert_eq!(error.provider_code(), Some("refusal"));
        assert!(error.message().contains("it asks for malware"), "{error}");
        Ok(())
    }

    /// A catalog whose model reasons but takes no effort levels, like
    /// claude-sonnet-4-5.
    const BUDGET_MODEL_CATALOG: &str = r#"
        schema_version = 1

        [providers.anthropic]
        display_name = "Anthropic"
        adapter = "anthropic"
        codec = "anthropic-messages"
        base_url = "http://127.0.0.1"
        default_model = "claude-sonnet-4-5"
        auth = { type = "none" }

        [providers.anthropic.models."claude-sonnet-4-5"]
        display_name = "Budget Claude"
        api_model = "claude-sonnet-4-5"
        capabilities = { text = true, tools = true, reasoning = true }
    "#;

    #[test]
    fn a_model_without_effort_levels_takes_a_thinking_budget() -> Result<(), Box<dyn StdError>> {
        let call = resolved_in(
            BUDGET_MODEL_CATALOG,
            Request::builder()
                .model("anthropic/claude-sonnet-4-5")
                .user("Hello")
                .reasoning_effort(ReasoningEffort::High)
                .max_output_tokens(8000)
                .build()?,
        )?;
        let codec = AnthropicMessagesCodec;

        let encoded = codec.encode(&call, false)?;

        assert_eq!(
            encoded.body["thinking"],
            json!({ "type": "enabled", "budget_tokens": 6000 })
        );
        assert_eq!(encoded.body["max_tokens"], 8000);
        // Sending `effort` too would ask for a control the model does not
        // take.
        assert_eq!(encoded.body.get("output_config"), None);

        // The count body keeps the thinking budget: counting must see the
        // same request generation sends.
        let counted = codec
            .encode_count_tokens(&call)
            .ok_or("Anthropic should count tokens")??;
        assert_eq!(
            counted.body["thinking"],
            json!({ "type": "enabled", "budget_tokens": 6000 })
        );
        assert_eq!(counted.body.get("max_tokens"), None);
        Ok(())
    }

    #[test]
    fn a_thinking_budget_keeps_its_floor_and_fits_under_the_limit() -> Result<(), Box<dyn StdError>>
    {
        let codec = AnthropicMessagesCodec;

        // A quarter of 1200 is under the provider floor; the floor wins and
        // still fits under the limit.
        let floored = codec.encode(
            &resolved_in(
                BUDGET_MODEL_CATALOG,
                Request::builder()
                    .model("anthropic/claude-sonnet-4-5")
                    .user("Hello")
                    .reasoning_effort(ReasoningEffort::Low)
                    .max_output_tokens(1200)
                    .build()?,
            )?,
            false,
        )?;
        assert_eq!(floored.body["thinking"]["budget_tokens"], 1024);
        assert_eq!(floored.body["max_tokens"], 1200);

        // Max effort budgets the whole limit, so the limit grows to keep the
        // budget strictly below it.
        let lifted = codec.encode(
            &resolved_in(
                BUDGET_MODEL_CATALOG,
                Request::builder()
                    .model("anthropic/claude-sonnet-4-5")
                    .user("Hello")
                    .reasoning_effort(ReasoningEffort::Max)
                    .max_output_tokens(2048)
                    .build()?,
            )?,
            false,
        )?;
        assert_eq!(lifted.body["thinking"]["budget_tokens"], 2048);
        assert_eq!(lifted.body["max_tokens"], 3072);
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
