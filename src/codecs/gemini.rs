//! The Gemini `generateContent` wire protocol.

use std::collections::HashMap;

use reqwest::Method;
use serde_json::{Map, Value, json};

use super::assembler::StreamAssembler;
use super::common::{
    GEMINI_SIGNATURES, carries_foreign_signature, endpoint, finish_reason, flattens_system_content,
    flattens_tool_result_content, foreign_signature, merge_options, plain_text, sampling,
    system_text, unsupported_capability, wire_options,
};
use super::{Codec, StreamDecoder};
use crate::adapter::ResolvedCall;
use crate::resolver::ResolvedRoute;
use crate::transport::{EncodedRequest, SseEvent, provider_error};
use crate::types::{
    ContentBlockId, ContentBlockKind, ContentPart, Error, ErrorKind, FinishReason, MediaSource,
    Message, ReasoningContent, Request, Response, ResponseFormat, Role, StreamEvent, TokenCounts,
    ToolCall, ToolCallKind, ToolChoice, ToolDefinitionKind, ToolResult,
};

/// The replay namespace this codec claims.
///
/// It matches the canonical catalog provider id, and it is the only
/// [`ContentPart::Opaque`] namespace and [`ToolCall::provider_metadata`] key
/// this codec reads or writes.
const NAMESPACE: &str = "gemini";

/// The Gemini `generateContent` and `streamGenerateContent` codec.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GeminiGenerateCodec;

impl Codec for GeminiGenerateCodec {
    fn encode(&self, call: &ResolvedCall, stream: bool) -> Result<EncodedRequest, Error> {
        let body = generate_body(call)?;
        let operation = if stream {
            "streamGenerateContent?alt=sse"
        } else {
            "generateContent"
        };

        let mut encoded = EncodedRequest::new(
            Method::POST,
            model_endpoint(call.route(), operation),
            Value::Object(body),
        )
        .with_timeout(call.request().timeout());
        if !call.request().metadata().is_empty() {
            encoded = encoded.unsupported_control("request metadata");
        }
        // The system field of this protocol takes text only, so anything else
        // a system message carries is dropped. The text still reaches the
        // model, so it is reported rather than refused.
        if flattens_system_content(call.request()) {
            encoded = encoded.unsupported_control("non-text system content");
        }
        // An all-JSON result rides `functionResponse.response` natively, so
        // only a mix that must flatten is reported.
        if flattens_tool_result_content(call.request(), |parts| {
            parts
                .iter()
                .all(|part| matches!(part, ContentPart::Text { .. }))
                || parts
                    .iter()
                    .all(|part| matches!(part, ContentPart::Json { .. }))
        }) {
            encoded = encoded.unsupported_control("non-text tool result content");
        }
        // This protocol has no latency tier, so the control is reported rather
        // than guessed at.
        if call.request().speed().is_some() {
            encoded = encoded.unsupported_control("the speed control");
        }
        // `generationConfig.thinkingConfig.thinkingBudget` is the natural
        // target, but it takes a token count, and turning an effort level into
        // a defensible budget needs per-model reasoning limits the catalog does
        // not carry. The control is reported until it does; a mapping can
        // replace this warning later.
        if call.request().reasoning_effort().is_some() {
            encoded = encoded.unsupported_control("the reasoning effort control");
        }
        // A skipped foreign-signed reasoning part never reaches the model,
        // so the skip is reported; see `foreign_signature`.
        if carries_foreign_signature(call.request(), GEMINI_SIGNATURES) {
            encoded = encoded.unsupported_control("reasoning signed by another provider");
        }
        Ok(encoded)
    }

    fn decode_response(&self, route: &ResolvedRoute, value: Value) -> Result<Response, Error> {
        let id = value
            .get("responseId")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);

        // The candidate is decoded through a closure so that the borrow of
        // `value` ends here: the no-candidates path below takes the whole
        // document as the error's raw data.
        let decoded = value
            .pointer("/candidates/0")
            .map(|candidate| decode_candidate(candidate, id.as_deref()));
        let Some((content, finished)) = decoded else {
            return Err(no_candidates(route, value));
        };
        let usage = decode_usage(value.get("usageMetadata").unwrap_or(&Value::Null));

        let mut response = Response::new(
            route.provider().id().clone(),
            route.model().id().clone(),
            content,
        );
        response.id = id;
        response.finish_reason = finished;
        response.usage = usage;
        response.raw = Some(value);
        Ok(response)
    }

    fn stream_decoder(&self, route: &ResolvedRoute) -> Box<dyn StreamDecoder> {
        Box::new(GeminiStreamDecoder::new(route))
    }

    fn encode_count_tokens(&self, call: &ResolvedCall) -> Option<Result<EncodedRequest, Error>> {
        Some(count_tokens_request(call))
    }

    fn decode_count_tokens(&self, route: &ResolvedRoute, value: Value) -> Result<u64, Error> {
        value
            .get("totalTokens")
            .and_then(Value::as_u64)
            .ok_or_else(move || {
                Error::new(
                    ErrorKind::ResponseDecode,
                    "Gemini returned a token count without a totalTokens field",
                )
                .with_provider(route.provider().id().clone())
                .with_raw_data(value)
            })
    }
}

/// The URL of one `models/{model}:{operation}` endpoint.
fn model_endpoint(route: &ResolvedRoute, operation: &str) -> String {
    endpoint(
        route.provider().base_url(),
        &format!("/v1beta/models/{}:{operation}", route.api_model()),
    )
}

/// Encodes the `countTokens` request for a resolved call.
///
/// The endpoint takes the complete `generateContent` body wrapped under a
/// single key. Nothing is stripped: the sampling parameters, the tools, and any
/// raw provider options all count toward the reported total, so the count
/// matches what the generation request would actually send.
///
/// The nested request carries its own `model` field: the API reference marks
/// `generateContentRequest.model` required, and the model in the URL path does
/// not populate it.
fn count_tokens_request(call: &ResolvedCall) -> Result<EncodedRequest, Error> {
    let mut body = generate_body(call)?;
    body.insert(
        "model".to_owned(),
        format!("models/{}", call.route().api_model()).into(),
    );

    Ok(EncodedRequest::new(
        Method::POST,
        model_endpoint(call.route(), "countTokens"),
        json!({ "generateContentRequest": Value::Object(body) }),
    )
    .with_timeout(call.request().timeout()))
}

/// Builds the `generateContent` request body.
///
/// Typed request fields are encoded first and the raw provider options are
/// merged last, so an application can override anything encoded here. The
/// default safety settings are applied after that merge, and only when the
/// caller supplied none; see [`apply_default_safety_settings`].
///
/// # Errors
///
/// Returns [`ErrorKind::InvalidRequest`] when the request carries a custom
/// tool. This protocol has function tools only, and a custom tool is never
/// silently downgraded into one.
fn generate_body(call: &ResolvedCall) -> Result<Map<String, Value>, Error> {
    let request = call.request();
    let route = call.route();
    let (options, _) = wire_options(call);

    let mut declarations = Vec::new();
    for tool in request.tools() {
        let ToolDefinitionKind::Function { input_schema } = &tool.kind else {
            return Err(unsupported_capability(route, "custom tools"));
        };
        declarations.push(json!({
            "name": tool.name,
            "description": tool.description,
            "parametersJsonSchema": input_schema,
        }));
    }

    let mut body = Map::new();

    let system = system_text(request.messages());
    if !system.is_empty() {
        body.insert(
            "systemInstruction".to_owned(),
            json!({ "parts": [{ "text": system }] }),
        );
    }

    let names = tool_call_names(request);
    let mut contents: Vec<Value> = Vec::new();
    for message in request.messages() {
        if matches!(message.role(), Role::System | Role::Developer) {
            continue;
        }
        let parts: Vec<Value> = message
            .content()
            .iter()
            .filter_map(|part| encode_part(part, &names))
            .collect();
        if parts.is_empty() {
            continue;
        }
        let role = if message.role() == Role::Assistant {
            "model"
        } else {
            "user"
        };
        // Parallel tool results arrive as one canonical message each but all
        // answer a single model turn, so consecutive same-role messages merge
        // into one turn. Gemini's documented shape puts every
        // `functionResponse` for a turn in one `user` entry.
        match contents.last_mut() {
            Some(last) if last.get("role").and_then(Value::as_str) == Some(role) => {
                if let Some(Value::Array(existing)) = last.get_mut("parts") {
                    existing.extend(parts);
                }
            }
            _ => contents.push(json!({ "role": role, "parts": parts })),
        }
    }
    body.insert("contents".to_owned(), Value::Array(contents));

    let mut generation = Map::new();
    if let Some(max_tokens) = request.max_output_tokens() {
        generation.insert("maxOutputTokens".to_owned(), max_tokens.into());
    }
    if let Some(temperature) = request.temperature() {
        generation.insert("temperature".to_owned(), sampling(temperature));
    }
    if let Some(top_p) = request.top_p() {
        generation.insert("topP".to_owned(), sampling(top_p));
    }
    if !request.stop_sequences().is_empty() {
        generation.insert(
            "stopSequences".to_owned(),
            Value::Array(
                request
                    .stop_sequences()
                    .iter()
                    .map(|sequence| Value::String(sequence.clone()))
                    .collect(),
            ),
        );
    }
    // `Text` is the protocol's own default, so it sets nothing: declaring
    // `application/json` for it would force JSON output for a caller who asked
    // for prose. Only the two JSON formats set the MIME type.
    match request.response_format() {
        Some(ResponseFormat::JsonObject) => {
            generation.insert("responseMimeType".to_owned(), "application/json".into());
        }
        Some(ResponseFormat::JsonSchema { schema, .. }) => {
            generation.insert("responseMimeType".to_owned(), "application/json".into());
            generation.insert("responseJsonSchema".to_owned(), schema.clone());
        }
        Some(ResponseFormat::Text) | None => {}
    }
    if !generation.is_empty() {
        body.insert("generationConfig".to_owned(), Value::Object(generation));
    }

    if !declarations.is_empty() {
        body.insert(
            "tools".to_owned(),
            json!([{ "functionDeclarations": declarations }]),
        );
    }
    if let Some(choice) = request.tool_choice() {
        body.insert("toolConfig".to_owned(), encode_tool_choice(choice));
    }

    // Request metadata has no field in this protocol, so it is never sent.
    // `encode` records an `unsupported_control` warning in its place.
    merge_options(&mut body, options);
    apply_default_safety_settings(&mut body);
    Ok(body)
}

/// Relaxes the dangerous-content filter unless the caller set its own settings.
///
/// Google's own defaults block a good deal of ordinary text, so a caller who
/// asks for nothing gets `BLOCK_ONLY_HIGH` for dangerous content, which is what
/// the reference implementation sent. This runs after the provider options are
/// merged, so a caller who supplies `safetySettings` keeps exactly what they
/// wrote, for every category.
///
/// Both spellings of the field are treated as caller-supplied. Proto-JSON
/// accepts `safety_settings` and `safetySettings` as the same field, so
/// inserting the camelCase default beside a caller's snake_case list would send
/// the field twice.
fn apply_default_safety_settings(body: &mut Map<String, Value>) {
    if body.contains_key("safetySettings") || body.contains_key("safety_settings") {
        return;
    }
    body.insert(
        "safetySettings".to_owned(),
        json!([{
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_ONLY_HIGH",
        }]),
    );
}

/// Maps every tool-call id in the history to the function it called.
///
/// `functionResponse` identifies the call it answers by function name, and a
/// canonical [`ToolResult`] carries the name only when the application kept it.
/// The assistant turn that made the call always carries it, so the history is
/// the reliable source. Without this, a result whose name is missing sends the
/// call id as the function name, which matches no declared function.
fn tool_call_names(request: &Request) -> HashMap<&str, &str> {
    request
        .messages()
        .iter()
        .filter(|message| message.role() == Role::Assistant)
        .flat_map(Message::content)
        .filter_map(|part| match part {
            ContentPart::ToolCall(call) => Some((call.id.as_str(), call.name.as_str())),
            _ => None,
        })
        .collect()
}

/// Encodes the permitted tool-selection behavior.
fn encode_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!({ "functionCallingConfig": { "mode": "AUTO" } }),
        ToolChoice::None => json!({ "functionCallingConfig": { "mode": "NONE" } }),
        ToolChoice::Required => json!({ "functionCallingConfig": { "mode": "ANY" } }),
        ToolChoice::Tool { name } => json!({
            "functionCallingConfig": { "mode": "ANY", "allowedFunctionNames": [name] }
        }),
    }
}

/// Encodes one content part, or `None` when this protocol cannot carry it.
///
/// `names` maps a tool-call id to the function it called; see
/// [`tool_call_names`].
fn encode_part(part: &ContentPart, names: &HashMap<&str, &str>) -> Option<Value> {
    match part {
        ContentPart::Text { text } => Some(json!({ "text": text })),
        ContentPart::Image(image) => Some(encode_media(&image.source)),
        ContentPart::Audio(audio) => Some(encode_media(&audio.source)),
        ContentPart::Document(document) => Some(encode_media(&document.source)),
        // A signature another provider family minted cannot verify here and
        // fails the request, so the part is skipped; the encoder reports it.
        ContentPart::Reasoning(reasoning) if foreign_signature(reasoning, GEMINI_SIGNATURES) => {
            None
        }
        ContentPart::Reasoning(reasoning) => Some(encode_reasoning(reasoning)),
        ContentPart::ToolCall(call) => Some(encode_tool_call(call)),
        ContentPart::ToolResult(result) => Some(encode_tool_result(result, names)),
        // Gemini parts carry text, not structured JSON, so a JSON part is
        // replayed as the text the model originally produced.
        ContentPart::Json { value } => Some(json!({ "text": value.to_string() })),
        // A part this codec wrote is replayed verbatim; one another provider
        // wrote is dropped, so a conversation can still fail over to Gemini.
        //
        // Every Gemini `Part` is a JSON object, so a payload that is not one
        // could never be a `Part` and is dropped rather than sent for the API
        // to reject. This does not validate that an object IS a valid `Part` —
        // only the provider can say that — it rules out the shapes that
        // certainly are not.
        ContentPart::Opaque { kind, data } => kind
            .split_once('.')
            .is_some_and(|(namespace, _)| namespace == NAMESPACE)
            .then(|| data.clone())
            .filter(Value::is_object),
    }
}

/// Encodes media as inline bytes or as a remote file reference.
fn encode_media(source: &MediaSource) -> Value {
    match source {
        MediaSource::Base64 { data, media_type } => json!({
            "inlineData": { "mimeType": media_type, "data": data }
        }),
        // A URL source carries no media type, so none is declared and Gemini
        // determines it from the fetched file.
        MediaSource::Url { url } => json!({ "fileData": { "fileUri": url } }),
    }
}

/// Encodes reasoning as a thought part with its verification signature.
fn encode_reasoning(reasoning: &ReasoningContent) -> Value {
    let mut part = Map::new();
    part.insert("text".to_owned(), Value::String(reasoning.text.clone()));
    part.insert("thought".to_owned(), Value::Bool(true));
    if let Some(signature) = &reasoning.signature {
        part.insert(
            "thoughtSignature".to_owned(),
            Value::String(signature.clone()),
        );
    }
    Value::Object(part)
}

/// Encodes a tool call, re-attaching the thought signature it was decoded with.
///
/// Gemini 3 rejects a replayed function call whose `thoughtSignature` is
/// missing, and the signature is a sibling of `functionCall` inside the same
/// part rather than a field of the call itself.
///
/// The signature is read from the `gemini` metadata namespace first, then
/// from the metadata's top level, where the predecessor library stored it.
/// Without the fallback, a history persisted before the namespacing would
/// silently replay unsigned and draw the rejection this field exists to
/// prevent.
fn encode_tool_call(call: &ToolCall) -> Value {
    let mut part = Map::new();
    part.insert(
        "functionCall".to_owned(),
        json!({ "id": call.id, "name": call.name, "args": call.arguments }),
    );
    let signature = call
        .provider_metadata
        .get(NAMESPACE)
        .and_then(|metadata| metadata.get("thoughtSignature"))
        .or_else(|| call.provider_metadata.get("thoughtSignature"));
    if let Some(signature) = signature {
        part.insert("thoughtSignature".to_owned(), signature.clone());
    }
    Value::Object(part)
}

/// Encodes a tool result as the function response that answers a call.
///
/// `response` is a free-form struct, and Google's guidance is to report a
/// failure under an `error` key and a success under `output`. Following that
/// convention matters because the model reads this payload: a key it
/// recognizes as an error reads as one, where a boolean flag beside an
/// `output` value reads as ordinary output.
///
/// `name` must be the function that was called. It is taken from the result
/// itself, then from the assistant turn that made the call, and only then from
/// the call id, which is a last resort that names no declared function.
fn encode_tool_result(result: &ToolResult, names: &HashMap<&str, &str>) -> Value {
    // Structured results travel natively: `response` is a free-form struct,
    // so a JSON value goes under the conventional key instead of being
    // flattened to the empty text a Json-only result would otherwise yield.
    let payload = match result.content.as_slice() {
        [ContentPart::Json { value }] => value.clone(),
        parts
            if !parts.is_empty()
                && parts
                    .iter()
                    .all(|part| matches!(part, ContentPart::Json { .. })) =>
        {
            Value::Array(
                parts
                    .iter()
                    .filter_map(|part| match part {
                        ContentPart::Json { value } => Some(value.clone()),
                        _ => None,
                    })
                    .collect(),
            )
        }
        _ => Value::String(plain_text(&result.content)),
    };
    let response = if result.is_error {
        json!({ "error": payload })
    } else {
        json!({ "output": payload })
    };
    let name = result
        .name
        .as_deref()
        .or_else(|| names.get(result.tool_call_id.as_str()).copied())
        .unwrap_or(&result.tool_call_id);
    json!({
        "functionResponse": {
            "id": result.tool_call_id,
            "name": name,
            "response": response,
        }
    })
}

/// Decodes one candidate into its content and its finish reason.
///
/// `response_id` scopes the ids synthesized for the function calls; see
/// [`tool_call_id`].
fn decode_candidate(
    candidate: &Value,
    response_id: Option<&str>,
) -> (Vec<ContentPart>, FinishReason) {
    let mut content = Vec::new();
    let mut calls = 0;
    for part in candidate
        .pointer("/content/parts")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if let Some(function_call) = part.get("functionCall") {
            content.push(decode_tool_call(part, function_call, response_id, calls));
            calls += 1;
        } else if let Some(text) = part.get("text").and_then(Value::as_str) {
            content.push(decode_text(part, text));
        }
    }

    let finished = tool_call_finish(
        finish_reason(candidate.get("finishReason").and_then(Value::as_str)),
        calls > 0,
    );
    (content, finished)
}

/// Corrects a finish reason for a turn that called tools.
///
/// Gemini reports `STOP` even when the candidate is nothing but function
/// calls, so a consumer that dispatches tools on [`FinishReason::ToolCall`]
/// would never run them. Only `Stop` is corrected: a call cut off by
/// `MAX_TOKENS`, or a candidate blocked mid-turn, keeps the reason the provider
/// gave, which is the more specific fact. The OpenAI Responses codec infers the
/// same way for the same reason.
fn tool_call_finish(reason: FinishReason, has_calls: bool) -> FinishReason {
    match reason {
        FinishReason::Stop if has_calls => FinishReason::ToolCall,
        other => other,
    }
}

/// The error for a 200 response that carried no candidates.
///
/// Gemini reports a prompt it refused to answer as `promptFeedback.blockReason`
/// on an otherwise successful body. Decoding that as an empty `Stop` response
/// makes a blocked prompt indistinguishable from a model that had nothing to
/// say, so it becomes a failure instead.
///
/// A block reason is restated in the `error` shape the shared classifier reads,
/// so that classifier — not this codec — decides the kind and the retry
/// classification. The block reason becomes the provider code, and the message
/// names the content policy, so the reasons beyond `SAFETY` — `BLOCKLIST`,
/// `PROHIBITED_CONTENT`, and any Google adds — all classify as
/// [`ErrorKind::ContentFilter`] rather than only the one spelling the
/// classifier happens to recognize. That restated payload is thrown away
/// afterward: the raw data is the provider's own document.
///
/// A body with neither candidates nor a block reason is malformed rather than
/// blocked, so it decodes into [`ErrorKind::ResponseDecode`].
fn no_candidates(route: &ResolvedRoute, value: Value) -> Error {
    let Some(reason) = value
        .pointer("/promptFeedback/blockReason")
        .and_then(Value::as_str)
    else {
        return Error::new(
            ErrorKind::ResponseDecode,
            "Gemini returned no candidates in the response",
        )
        .with_provider(route.provider().id().clone())
        .with_raw_data(value);
    };

    let classified = json!({
        "error": {
            "status": reason,
            "message":
                format!("blocked the prompt under its content policy (block reason {reason})"),
        }
    });
    provider_error(route.provider(), None, Some(classified), None).with_raw_data(value)
}

/// Decodes a text part into visible text or reasoning.
fn decode_text(part: &Value, text: &str) -> ContentPart {
    if part.get("thought").and_then(Value::as_bool) != Some(true) {
        return ContentPart::Text {
            text: text.to_owned(),
        };
    }

    let signature = thought_signature(part).map(ToOwned::to_owned);
    let signature_origin = signature.is_some().then(|| GEMINI_SIGNATURES.to_owned());
    ContentPart::Reasoning(ReasoningContent {
        text: text.to_owned(),
        signature,
        signature_origin,
        redacted: false,
    })
}

/// Decodes a `functionCall` part into a tool call.
///
/// `ordinal` counts the function calls already decoded from this response, so
/// the synthesized id is stable across repeated decodes of the same payload.
fn decode_tool_call(
    part: &Value,
    function_call: &Value,
    response_id: Option<&str>,
    ordinal: usize,
) -> ContentPart {
    let name = function_call
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let mut call = ToolCall::function(
        tool_call_id(function_call, name, response_id, ordinal),
        name,
        function_call
            .get("args")
            .cloned()
            .unwrap_or_else(|| json!({})),
    );
    if let Some(signature) = thought_signature(part) {
        call.provider_metadata.insert(
            NAMESPACE.to_owned(),
            json!({ "thoughtSignature": signature }),
        );
    }
    ContentPart::ToolCall(call)
}

/// The identity of one function call.
///
/// A call Gemini gave an id keeps it. Gemini normally supplies none, so one is
/// synthesized as `{name}-{ordinal}-{response_id}`:
///
/// - `{name}-{ordinal}` alone is deterministic, which a minted UUID is not —
///   decoding one payload twice yields one id — but it repeats across turns, so
///   a conversation with two `search` calls carries `search-0` twice. An
///   application keyed by call id then collides, and the replayed history sends
///   duplicate ids on the wire.
/// - `responseId` is the provider's own name for one response, so appending it
///   makes the id unique per response while staying a pure function of the
///   payload. It is the fallback that is dropped, not the ordinal: the id stays
///   readable, and the plain `{name}-{ordinal}` form remains its prefix.
///
/// A payload that carries no `responseId` — Gemini omits it on some routes —
/// keeps the bare `{name}-{ordinal}` form. That is the same identity the ids
/// had before, so nothing is worse for it.
fn tool_call_id(
    function_call: &Value,
    name: &str,
    response_id: Option<&str>,
    ordinal: usize,
) -> String {
    if let Some(id) = function_call
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
    {
        return id.to_owned();
    }

    match response_id {
        Some(response) => format!("{name}-{ordinal}-{response}"),
        None => format!("{name}-{ordinal}"),
    }
}

/// The thought signature carried alongside a part, when it has one.
fn thought_signature(part: &Value) -> Option<&str> {
    part.get("thoughtSignature").and_then(Value::as_str)
}

/// Normalizes `usageMetadata` into the five disjoint token buckets.
///
/// The Gemini counters are inclusive in one direction and exclusive in the
/// other, which is why none of the shared helpers fit:
///
/// - `promptTokenCount` **includes** `cachedContentTokenCount`, so the cached
///   tokens are subtracted out of `input`.
/// - `toolUsePromptTokenCount` sits **outside** `promptTokenCount`, so it is
///   added to `input`.
/// - `candidatesTokenCount` **excludes** `thoughtsTokenCount`, so `output`
///   passes through untouched and `reasoning` is taken as reported.
///
/// `cache_write` is always zero: creating a Gemini cache is a separate
/// `cachedContents` call, not part of `generateContent` usage.
fn decode_usage(metadata: &Value) -> TokenCounts {
    let count = |key: &str| {
        metadata
            .get(key)
            .and_then(Value::as_u64)
            .unwrap_or_default()
    };
    let cache_read = count("cachedContentTokenCount");

    TokenCounts {
        input: count("promptTokenCount")
            .saturating_sub(cache_read)
            .saturating_add(count("toolUsePromptTokenCount")),
        output: count("candidatesTokenCount"),
        reasoning: count("thoughtsTokenCount"),
        cache_read,
        cache_write: 0,
    }
}

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
struct GeminiStreamDecoder {
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
    /// The finish reason of the last chunk that reported one.
    finished:    Option<FinishReason>,
    /// Whether a chunk already failed, which forbids a completed response.
    failed:      bool,
}

impl GeminiStreamDecoder {
    fn new(route: &ResolvedRoute) -> Self {
        Self {
            route:       route.clone(),
            assembler:   StreamAssembler::new(route),
            open:        None,
            texts:       0,
            reasonings:  0,
            calls:       0,
            started:     false,
            response_id: None,
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
        let call_id = tool_call_id(function_call, name, self.response_id.as_deref(), self.calls);
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
        let value: Value = serde_json::from_str(&event.data).map_err(|source| {
            Error::new(
                ErrorKind::StreamDecode,
                "Gemini returned an invalid stream event",
            )
            .with_provider(self.route.provider().id().clone())
            .with_source(source)
        })?;

        if value.get("error").is_some() {
            return Err(provider_error(
                self.route.provider(),
                None,
                Some(value),
                None,
            ));
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
            events.push(self.assembler.usage(decode_usage(metadata)));
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
                .set_finish_reason(tool_call_finish(reason, self.calls > 0));
        }

        // The protocol has no terminal event, so the byte-stream end is the
        // only signal that the last run closed. `raw` stays unset because
        // Gemini never sends a final response document.
        let mut events = self.close_run();
        events.extend(self.assembler.complete());
        Ok(events)
    }
}

#[cfg(all(test, feature = "builtin-catalog"))]
mod tests {
    use std::error::Error as StdError;

    use serde_json::{Map, Value, json};

    use super::{Codec as _, GeminiGenerateCodec};
    use crate::codecs::test_support::resolved;
    use crate::transport::SseEvent;
    use crate::types::{
        ContentPart, Error, ErrorKind, FinishReason, ImageContent, MediaSource, Message,
        ReasoningEffort, Request, Response, ResponseFormat, RetryClassification, Role, Speed,
        StreamEvent, ToolCall, ToolDefinition, ToolResult,
    };

    fn object(value: Value) -> Result<Map<String, Value>, Box<dyn StdError>> {
        match value {
            Value::Object(map) => Ok(map),
            other => Err(format!("expected a JSON object, got {other}").into()),
        }
    }

    /// Decodes one blocking response from a `generateContent` payload.
    fn decode(payload: Value) -> Result<Response, Box<dyn StdError>> {
        let call = resolved(
            Request::builder()
                .model("gemini/gemini-2.5-pro")
                .user("Hello")
                .build()?,
        )?;

        Ok(GeminiGenerateCodec.decode_response(call.route(), payload)?)
    }

    /// Decodes one payload that must fail, and returns the codec's own error.
    fn decode_failure(payload: Value) -> Result<Error, Box<dyn StdError>> {
        let call = resolved(
            Request::builder()
                .model("gemini/gemini-2.5-pro")
                .user("Hello")
                .build()?,
        )?;

        GeminiGenerateCodec
            .decode_response(call.route(), payload)
            .err()
            .ok_or_else(|| "the payload should not decode into a response".into())
    }

    /// Runs a whole stream and returns every event it produced.
    fn stream(chunks: &[Value]) -> Result<Vec<StreamEvent>, Box<dyn StdError>> {
        let call = resolved(
            Request::builder()
                .model("gemini/gemini-2.5-pro")
                .user("Hello")
                .build()?,
        )?;
        let mut decoder = GeminiGenerateCodec.stream_decoder(call.route());

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

    #[test]
    fn each_streamed_thought_signature_seals_its_own_block() -> Result<(), Box<dyn StdError>> {
        // Two signed thought parts in one contiguous reasoning run. Each
        // signature is a complete blob; concatenating them would produce a
        // signature Gemini rejects on replay, so each signed part must end
        // as its own reasoning part — the blocking decoder's shape.
        let events = stream(&[json!({
            "responseId": "resp-1",
            "candidates": [{
                "content": { "parts": [
                    { "text": "step one", "thought": true, "thoughtSignature": "sig-a" },
                    { "text": "step two", "thought": true, "thoughtSignature": "sig-b" },
                ] },
                "finishReason": "STOP",
            }],
        })])?;

        let parts: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::ContentBlockEnd { part, .. } => Some(part.clone()),
                _ => None,
            })
            .collect();
        let signatures: Vec<_> = parts
            .iter()
            .map(|part| match part {
                ContentPart::Reasoning(reasoning) => {
                    Ok((reasoning.text.as_str(), reasoning.signature.as_deref()))
                }
                other => Err(format!("expected a reasoning part, got {other:?}")),
            })
            .collect::<Result<_, _>>()?;
        assert_eq!(signatures, vec![
            ("step one", Some("sig-a")),
            ("step two", Some("sig-b")),
        ]);
        Ok(())
    }

    #[test]
    fn encodes_media_sources_and_function_response_identity() -> Result<(), Box<dyn StdError>> {
        let call = resolved(
            Request::builder()
                .model("gemini/gemini-2.5-pro")
                .message(Message::new(Role::User, [
                    ContentPart::Image(ImageContent::new(MediaSource::base64(
                        "aGVsbG8=",
                        "image/png",
                    ))),
                    ContentPart::Image(ImageContent::new(MediaSource::url(
                        "https://example.com/cat.png",
                    ))),
                ]))
                .message(Message::new(Role::Tool, [ContentPart::ToolResult(
                    ToolResult {
                        tool_call_id: "call-1".to_owned(),
                        name:         Some("weather".to_owned()),
                        content:      vec![ContentPart::Text {
                            text: "sunny".to_owned(),
                        }],
                        is_error:     false,
                    },
                )]))
                .build()?,
        )?;

        let encoded = GeminiGenerateCodec.encode(&call, false)?;

        assert_eq!(
            encoded.body["contents"][0]["parts"][0]["inlineData"],
            json!({ "mimeType": "image/png", "data": "aGVsbG8=" })
        );
        assert_eq!(
            encoded.body["contents"][0]["parts"][1]["fileData"],
            json!({ "fileUri": "https://example.com/cat.png" })
        );
        // The media message and the tool result both map to the `user` role,
        // so they merge into one turn rather than two consecutive ones.
        assert_eq!(encoded.body["contents"].as_array().map(Vec::len), Some(1));
        assert_eq!(
            encoded.body["contents"][0]["parts"][2]["functionResponse"]["id"],
            "call-1"
        );
        assert_eq!(
            encoded.body["contents"][0]["parts"][2]["functionResponse"]["name"],
            "weather"
        );
        Ok(())
    }

    #[test]
    fn usage_buckets_are_disjoint() -> Result<(), Box<dyn StdError>> {
        let response = decode(json!({
            "candidates": [{ "content": { "parts": [{ "text": "Done." }] } }],
            "usageMetadata": {
                "promptTokenCount": 200,
                "cachedContentTokenCount": 180,
                "toolUsePromptTokenCount": 400,
                "candidatesTokenCount": 200,
                "thoughtsTokenCount": 300,
            }
        }))?;

        // (200 - 180) + 400: the cached tokens are inside `promptTokenCount`
        // and the tool-use tokens are outside it.
        assert_eq!(response.usage.input, 420);
        assert_eq!(response.usage.cache_read, 180);
        // `candidatesTokenCount` never contained the thoughts, so nothing is
        // subtracted from the output bucket.
        assert_eq!(response.usage.output, 200);
        assert_eq!(response.usage.reasoning, 300);
        assert_eq!(response.usage.cache_write, 0);
        assert_eq!(response.usage.total(), 1100);
        Ok(())
    }

    #[test]
    fn synthesized_tool_call_ids_are_deterministic() -> Result<(), Box<dyn StdError>> {
        let payload = json!({
            "candidates": [{ "content": { "parts": [
                { "functionCall": { "name": "search", "args": { "query": "rust" } } },
                { "text": "and then" },
                { "functionCall": { "name": "search", "args": { "query": "gemini" } } },
            ] } }]
        });

        let first = decode(payload.clone())?;
        let second = decode(payload)?;

        let ids: Vec<&str> = first
            .content
            .iter()
            .filter_map(|part| match part {
                ContentPart::ToolCall(call) => Some(call.id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(ids, ["search-0", "search-1"]);
        assert_eq!(first.content, second.content);
        Ok(())
    }

    #[test]
    fn a_response_id_scopes_the_synthesized_ids() -> Result<(), Box<dyn StdError>> {
        // The bare `{name}-{ordinal}` form repeats across turns, so the same
        // conversation would carry `search-0` twice. The response id is the
        // provider's own name for one response, so it separates them without
        // costing determinism.
        let response = decode(json!({
            "responseId": "resp-1",
            "candidates": [{ "content": { "parts": [
                { "functionCall": { "name": "search", "args": {} } },
            ] } }]
        }))?;

        let [ContentPart::ToolCall(call)] = response.content.as_slice() else {
            return Err(format!("unexpected content {:?}", response.content).into());
        };
        assert_eq!(call.id, "search-0-resp-1");
        Ok(())
    }

    #[test]
    fn a_function_call_turn_finishes_as_a_tool_call() -> Result<(), Box<dyn StdError>> {
        // Gemini says STOP even when the whole turn is a function call.
        let response = decode(json!({
            "candidates": [{
                "content": { "parts": [{ "functionCall": { "name": "search", "args": {} } }] },
                "finishReason": "STOP",
            }]
        }))?;

        assert_eq!(response.finish_reason, FinishReason::ToolCall);
        Ok(())
    }

    #[test]
    fn a_truncated_function_call_keeps_the_provider_reason() -> Result<(), Box<dyn StdError>> {
        // Only `STOP` is corrected. `MAX_TOKENS` is the more specific fact and
        // survives, so a cut-off call is not reported as a complete one.
        let response = decode(json!({
            "candidates": [{
                "content": { "parts": [{ "functionCall": { "name": "search", "args": {} } }] },
                "finishReason": "MAX_TOKENS",
            }]
        }))?;

        assert_eq!(response.finish_reason, FinishReason::Length);
        Ok(())
    }

    #[test]
    fn a_blocked_prompt_is_a_content_filter_error() -> Result<(), Box<dyn StdError>> {
        let error = decode_failure(json!({ "promptFeedback": { "blockReason": "SAFETY" } }))?;

        assert_eq!(error.kind(), ErrorKind::ContentFilter);
        assert_eq!(error.provider_code(), Some("SAFETY"));
        assert_eq!(error.retry_classification(), RetryClassification::Never);
        Ok(())
    }

    #[test]
    fn every_block_reason_classifies_as_a_content_filter() -> Result<(), Box<dyn StdError>> {
        // The classifier knows `SAFETY` by name; it does not know the rest. The
        // message names the content policy so the whole family lands in one
        // place, whatever Google spells the reason.
        for reason in ["BLOCKLIST", "PROHIBITED_CONTENT", "IMAGE_SAFETY", "OTHER"] {
            let error = decode_failure(json!({ "promptFeedback": { "blockReason": reason } }))?;

            assert_eq!(error.kind(), ErrorKind::ContentFilter, "{reason}");
            assert_eq!(error.provider_code(), Some(reason), "{reason}");
        }
        Ok(())
    }

    #[test]
    fn a_body_with_no_candidates_and_no_block_reason_fails_to_decode()
    -> Result<(), Box<dyn StdError>> {
        let error = decode_failure(json!({ "responseId": "resp-1" }))?;

        assert_eq!(error.kind(), ErrorKind::ResponseDecode);
        Ok(())
    }

    #[test]
    fn decodes_thought_signatures_onto_reasoning_and_tool_calls() -> Result<(), Box<dyn StdError>> {
        let response = decode(json!({
            "candidates": [{ "content": { "parts": [
                { "text": "step one", "thought": true, "thoughtSignature": "sig-think" },
                { "functionCall": { "name": "search", "args": {} },
                  "thoughtSignature": "sig-call" },
            ] } }]
        }))?;

        let [
            ContentPart::Reasoning(reasoning),
            ContentPart::ToolCall(call),
        ] = response.content.as_slice()
        else {
            return Err(format!("unexpected content {:?}", response.content).into());
        };
        assert_eq!(reasoning.signature.as_deref(), Some("sig-think"));
        assert_eq!(
            call.provider_metadata.get("gemini"),
            Some(&json!({ "thoughtSignature": "sig-call" }))
        );
        Ok(())
    }

    #[test]
    fn raw_options_win_and_only_this_namespace_is_read() -> Result<(), Box<dyn StdError>> {
        let call = resolved(
            Request::builder()
                .model("gemini/gemini-2.5-pro")
                .user("Hello")
                .temperature(0.2)
                .max_output_tokens(64)
                .provider_options(
                    "gemini",
                    object(json!({
                        "generationConfig": { "temperature": 0.9 },
                        "safetySettings": [{ "category": "HARM_CATEGORY_HARASSMENT" }],
                        "auto_cache": false,
                    }))?,
                )
                .provider_option("openai", "temperature", json!(0.1))
                .build()?,
        )?;

        let encoded = GeminiGenerateCodec.encode(&call, false)?;

        assert_eq!(encoded.body["generationConfig"]["temperature"], json!(0.9));
        assert_eq!(
            encoded.body["generationConfig"]["maxOutputTokens"],
            json!(64)
        );
        assert_eq!(
            encoded.body["safetySettings"],
            json!([{ "category": "HARM_CATEGORY_HARASSMENT" }])
        );
        let body = object(encoded.body)?;
        assert!(!body.contains_key("auto_cache"));
        Ok(())
    }

    #[test]
    fn stop_sequences_land_in_the_generation_config() -> Result<(), Box<dyn StdError>> {
        let call = resolved(
            Request::builder()
                .model("gemini/gemini-2.5-pro")
                .user("Hello")
                .stop_sequences(["STOP", "END"])
                .metadata_entry("session", "abc")
                .build()?,
        )?;

        let encoded = GeminiGenerateCodec.encode(&call, false)?;

        assert_eq!(
            encoded.body["generationConfig"]["stopSequences"],
            json!(["STOP", "END"])
        );
        // This protocol has no request metadata field, so nothing is sent and
        // the loss is reported as a warning instead.
        let codes: Vec<&str> = encoded
            .warnings
            .iter()
            .map(|warning| warning.code.as_str())
            .collect();
        assert_eq!(codes, ["unsupported_control"]);
        let body = object(encoded.body)?;
        assert!(!body.contains_key("metadata"));
        Ok(())
    }

    #[test]
    fn a_text_response_format_leaves_the_output_mode_alone() -> Result<(), Box<dyn StdError>> {
        let call = resolved(
            Request::builder()
                .model("gemini/gemini-2.5-pro")
                .user("Hello")
                .response_format(ResponseFormat::Text)
                .build()?,
        )?;

        let encoded = GeminiGenerateCodec.encode(&call, false)?;

        // `Text` is the protocol's default. Declaring `application/json` for it
        // would make the model answer in JSON to a caller who asked for prose.
        assert_eq!(
            encoded.body["generationConfig"].get("responseMimeType"),
            None
        );
        Ok(())
    }

    #[test]
    fn controls_this_protocol_cannot_carry_are_reported() -> Result<(), Box<dyn StdError>> {
        let call = resolved(
            Request::builder()
                .model("gemini/gemini-2.5-pro")
                .user("Hello")
                .speed(Speed::Fast)
                .reasoning_effort(ReasoningEffort::High)
                .build()?,
        )?;

        let encoded = GeminiGenerateCodec.encode(&call, false)?;

        let messages: Vec<&str> = encoded
            .warnings
            .iter()
            .map(|warning| warning.message.as_str())
            .collect();
        assert_eq!(messages, [
            "this provider protocol does not support the speed control",
            "this provider protocol does not support the reasoning effort control",
        ]);
        // Neither control may be guessed at on the wire.
        let body = encoded.body.to_string();
        assert!(!body.contains("thinkingConfig"), "{body}");
        assert!(!body.contains("speed"), "{body}");
        Ok(())
    }

    #[test]
    fn a_tool_result_recovers_its_function_name_from_the_call() -> Result<(), Box<dyn StdError>> {
        // A result that kept no name would otherwise send the call id as the
        // function name, which matches no declared function.
        let call = resolved(
            Request::builder()
                .model("gemini/gemini-2.5-pro")
                .user("What is the weather?")
                .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
                    ToolCall::function("call-1", "get_weather", json!({ "city": "Paris" })),
                )]))
                .message(Message::new(Role::Tool, [ContentPart::ToolResult(
                    ToolResult {
                        tool_call_id: "call-1".to_owned(),
                        name:         None,
                        content:      vec![ContentPart::Text {
                            text: "18C".to_owned(),
                        }],
                        is_error:     false,
                    },
                )]))
                .build()?,
        )?;

        let encoded = GeminiGenerateCodec.encode(&call, false)?;

        assert_eq!(
            encoded.body["contents"][2]["parts"][0]["functionResponse"]["name"],
            "get_weather"
        );
        Ok(())
    }

    #[test]
    fn a_legacy_top_level_thought_signature_still_replays() -> Result<(), Box<dyn StdError>> {
        // The predecessor library stored the signature at the metadata's top
        // level rather than under the `gemini` namespace. A migrated history
        // must keep replaying it, or Gemini 3 rejects the unsigned call.
        let mut replayed = ToolCall::function("call-1", "search", json!({ "q": "rust" }));
        replayed
            .provider_metadata
            .insert("thoughtSignature".to_owned(), json!("sig-legacy"));
        let call = resolved(
            Request::builder()
                .model("gemini/gemini-2.5-pro")
                .user("Search for rust")
                .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
                    replayed,
                )]))
                .message(Message::new(Role::Tool, [ContentPart::ToolResult(
                    ToolResult {
                        tool_call_id: "call-1".to_owned(),
                        name:         Some("search".to_owned()),
                        content:      vec![ContentPart::Text {
                            text: "found".to_owned(),
                        }],
                        is_error:     false,
                    },
                )]))
                .build()?,
        )?;

        let encoded = GeminiGenerateCodec.encode(&call, false)?;

        assert_eq!(
            encoded.body["contents"][1]["parts"][0]["thoughtSignature"],
            "sig-legacy"
        );
        Ok(())
    }

    #[test]
    fn a_caller_without_safety_settings_gets_the_default() -> Result<(), Box<dyn StdError>> {
        let call = resolved(
            Request::builder()
                .model("gemini/gemini-2.5-pro")
                .user("Hello")
                .build()?,
        )?;

        let encoded = GeminiGenerateCodec.encode(&call, false)?;

        assert_eq!(
            encoded.body["safetySettings"],
            json!([{
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH",
            }])
        );
        Ok(())
    }

    #[test]
    fn a_snake_case_safety_setting_is_not_duplicated() -> Result<(), Box<dyn StdError>> {
        // Proto-JSON reads `safety_settings` and `safetySettings` as one field,
        // so adding the default beside a caller's snake_case list would send
        // the same field twice.
        let call = resolved(
            Request::builder()
                .model("gemini/gemini-2.5-pro")
                .user("Hello")
                .provider_option(
                    "gemini",
                    "safety_settings",
                    json!([{ "category": "HARM_CATEGORY_HARASSMENT" }]),
                )
                .build()?,
        )?;

        let encoded = GeminiGenerateCodec.encode(&call, false)?;

        let body = object(encoded.body)?;
        assert!(!body.contains_key("safetySettings"));
        assert_eq!(
            body["safety_settings"],
            json!([{ "category": "HARM_CATEGORY_HARASSMENT" }])
        );
        Ok(())
    }

    #[test]
    fn custom_tools_are_rejected_before_dispatch() -> Result<(), Box<dyn StdError>> {
        let call = resolved(
            Request::builder()
                .model("gemini/gemini-2.5-pro")
                .user("Hello")
                .tool(ToolDefinition::custom(
                    "apply_patch",
                    "Edits files",
                    json!({ "type": "grammar" }),
                ))
                .build()?,
        )?;

        let error = GeminiGenerateCodec
            .encode(&call, false)
            .err()
            .ok_or("expected a custom tool to be rejected")?;

        assert_eq!(error.provider_code(), Some("unsupported_capability"));
        Ok(())
    }

    #[test]
    fn count_tokens_wraps_the_whole_generate_body() -> Result<(), Box<dyn StdError>> {
        let call = resolved(
            Request::builder()
                .model("gemini/gemini-2.5-pro")
                .user("Hello")
                .max_output_tokens(128)
                .tool(ToolDefinition::function(
                    "search",
                    "Searches",
                    json!({ "type": "object" }),
                ))
                .build()?,
        )?;

        let encoded = GeminiGenerateCodec
            .encode_count_tokens(&call)
            .ok_or("expected a count tokens request")??;

        assert!(
            encoded
                .url
                .ends_with("/v1beta/models/gemini-2.5-pro:countTokens")
        );
        let body = object(encoded.body)?;
        assert!(!body.contains_key("contents"));
        let wrapped = &body["generateContentRequest"];
        assert_eq!(wrapped["contents"][0]["parts"][0]["text"], "Hello");
        assert_eq!(wrapped["generationConfig"]["maxOutputTokens"], json!(128));
        assert!(wrapped.get("tools").is_some());

        assert_eq!(
            GeminiGenerateCodec.decode_count_tokens(call.route(), json!({ "totalTokens": 42 }))?,
            42
        );
        Ok(())
    }

    #[test]
    fn stream_assigns_one_block_per_run_and_keeps_the_last_usage() -> Result<(), Box<dyn StdError>>
    {
        let events = stream(&[
            json!({ "responseId": "resp-stream-1",
                    "candidates": [{ "content": { "parts": [{ "text": "Hel" }] } }],
                    "usageMetadata": { "promptTokenCount": 10, "candidatesTokenCount": 1 } }),
            json!({ "candidates": [{ "content": { "parts": [{ "text": "lo" }] } }] }),
            json!({ "candidates": [{ "content": { "parts": [
                        { "text": "Let me think", "thought": true }] } }] }),
            json!({ "candidates": [{ "content": { "parts": [
                        { "functionCall": { "name": "search", "args": { "query": "rust" } },
                          "thoughtSignature": "sig-call" }] },
                    "finishReason": "STOP" }],
                    "usageMetadata": { "promptTokenCount": 20, "candidatesTokenCount": 9,
                                       "thoughtsTokenCount": 3 } }),
        ])?;

        // The stream opens with `Started`, carrying the response id the first
        // chunk named.
        let [StreamEvent::Started { id }, ..] = events.as_slice() else {
            return Err(format!("unexpected first event {:?}", events.first()).into());
        };
        assert_eq!(id.as_deref(), Some("resp-stream-1"));

        let starts: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::ContentBlockStart { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        let ends: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::ContentBlockEnd { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(starts, ["text-0", "reasoning-0", "tool-0"]);
        assert_eq!(ends, ["text-0", "reasoning-0", "tool-0"]);

        // A function call arrives whole, so its block carries no delta.
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, StreamEvent::ToolCallDelta { .. }))
        );

        let completed: Vec<&Response> = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::Completed { response } => Some(response),
                _ => None,
            })
            .collect();
        let [response] = completed.as_slice() else {
            return Err(format!("expected one completed event, got {}", completed.len()).into());
        };
        assert_eq!(response.text(), "Hello");
        assert_eq!(response.id.as_deref(), Some("resp-stream-1"));
        // The last chunk said STOP, and the turn called a tool.
        assert_eq!(response.finish_reason, FinishReason::ToolCall);
        assert_eq!(response.usage.input, 20);
        assert_eq!(response.usage.output, 9);
        assert_eq!(response.usage.reasoning, 3);
        assert!(response.raw.is_none());

        let [
            ContentPart::Text { .. },
            ContentPart::Reasoning(..),
            ContentPart::ToolCall(call),
        ] = response.content.as_slice()
        else {
            return Err(format!("unexpected content {:?}", response.content).into());
        };
        // The response id from the first chunk scopes the synthesized call id
        // for the whole stream.
        assert_eq!(call.id, "search-0-resp-stream-1");
        assert_eq!(call.arguments, json!({ "query": "rust" }));
        assert_eq!(
            call.provider_metadata.get("gemini"),
            Some(&json!({ "thoughtSignature": "sig-call" }))
        );
        Ok(())
    }

    #[test]
    fn stream_errors_end_the_stream() -> Result<(), Box<dyn StdError>> {
        let call = resolved(
            Request::builder()
                .model("gemini/gemini-2.5-pro")
                .user("Hello")
                .build()?,
        )?;
        let mut decoder = GeminiGenerateCodec.stream_decoder(call.route());

        let error = decoder
            .decode(SseEvent {
                event: None,
                data:  json!({ "error": { "status": "RESOURCE_EXHAUSTED",
                                          "message": "too many requests" } })
                .to_string(),
            })
            .err()
            .ok_or("expected a stream error")?;

        // The gRPC status is classified by the shared provider classifier, so
        // a mid-stream failure lands in the same category an HTTP one would.
        assert_eq!(error.kind(), ErrorKind::RateLimit);
        assert_eq!(error.provider_code(), Some("RESOURCE_EXHAUSTED"));
        assert!(decoder.finish()?.is_empty());
        Ok(())
    }

    #[test]
    fn a_failed_tool_result_uses_googles_error_key() -> Result<(), Box<dyn StdError>> {
        let request = Request::builder()
            .model("gemini/gemini-2.5-pro")
            .user("Look it up.")
            .message(Message::new(Role::Tool, [ContentPart::ToolResult(
                ToolResult {
                    tool_call_id: "call-1".to_owned(),
                    name:         Some("weather".to_owned()),
                    content:      vec![ContentPart::Text {
                        text: "the city is unknown".to_owned(),
                    }],
                    is_error:     true,
                },
            )]))
            .build()?;

        let encoded = GeminiGenerateCodec.encode(&resolved(request)?, false)?;

        let response = &encoded.body["contents"][0]["parts"][1]["functionResponse"]["response"];
        assert_eq!(response, &json!({ "error": "the city is unknown" }));
        assert!(
            response.get("is_error").is_none(),
            "a boolean flag beside `output` reads to the model as ordinary output"
        );
        Ok(())
    }

    #[test]
    fn a_json_tool_result_rides_the_response_struct() -> Result<(), Box<dyn StdError>> {
        let request = Request::builder()
            .model("gemini/gemini-2.5-pro")
            .user("Look it up.")
            .message(Message::new(Role::Tool, [ContentPart::ToolResult(
                ToolResult {
                    tool_call_id: "call-1".to_owned(),
                    name:         Some("weather".to_owned()),
                    content:      vec![ContentPart::Json {
                        value: json!({ "temperature": 21, "unit": "C" }),
                    }],
                    is_error:     false,
                },
            )]))
            .build()?;

        let encoded = GeminiGenerateCodec.encode(&resolved(request)?, false)?;

        let response = &encoded.body["contents"][0]["parts"][1]["functionResponse"]["response"];
        assert_eq!(
            response,
            &json!({ "output": { "temperature": 21, "unit": "C" } }),
            "a JSON result must reach the model, not flatten to empty text"
        );
        assert!(
            !encoded
                .warnings
                .iter()
                .any(|warning| warning.message.contains("tool result")),
            "carried content must not warn"
        );
        Ok(())
    }

    #[test]
    fn an_opaque_payload_that_cannot_be_a_part_is_dropped() -> Result<(), Box<dyn StdError>> {
        // Every Gemini `Part` is an object, so a string payload could never be
        // one. Dropping it beats sending the API something it must reject.
        let request = Request::builder()
            .model("gemini/gemini-2.5-pro")
            .user("Hello")
            .message(Message::new(Role::Assistant, [
                ContentPart::opaque("gemini.thought", json!("not a part")),
                ContentPart::Text {
                    text: "Hi.".to_owned(),
                },
            ]))
            .build()?;

        let encoded = GeminiGenerateCodec.encode(&resolved(request)?, false)?;

        let parts = &encoded.body["contents"][1]["parts"];
        assert_eq!(parts.as_array().map(Vec::len), Some(1));
        assert_eq!(parts[0]["text"], "Hi.");
        Ok(())
    }

    #[test]
    fn an_opaque_object_still_replays() -> Result<(), Box<dyn StdError>> {
        let request = Request::builder()
            .model("gemini/gemini-2.5-pro")
            .user("Hello")
            .message(Message::new(Role::Assistant, [ContentPart::opaque(
                "gemini.thought",
                json!({ "text": "kept", "thought": true }),
            )]))
            .build()?;

        let encoded = GeminiGenerateCodec.encode(&resolved(request)?, false)?;

        assert_eq!(
            encoded.body["contents"][1]["parts"][0],
            json!({ "text": "kept", "thought": true })
        );
        Ok(())
    }
}
