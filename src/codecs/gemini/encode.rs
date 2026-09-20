//! Request encoding: the `generateContent` body and its parts.

use std::collections::HashMap;

use reqwest::Method;
use serde_json::{Map, Value, json};

use super::NAMESPACE;
use super::identity::tool_call_names;
use crate::adapter::ResolvedCall;
use crate::codecs::content::{
    GEMINI_SIGNATURES, Turns, flattens_system_content, flattens_tool_result_content, plain_text,
    system_text, text_or_json_only,
};
use crate::codecs::errors::unsupported_capability;
use crate::codecs::options::{endpoint, merge_options, sampling, wire_options};
use crate::resolver::ResolvedRoute;
use crate::transport::EncodedRequest;
use crate::types::{
    ContentPart, Error, MediaSource, ReasoningContent, ReasoningEffort, Request, ResponseFormat,
    Role, ToolCall, ToolChoice, ToolDefinitionKind, ToolResult,
};

/// The URL of one `models/{model}:{operation}` endpoint.
pub(super) fn model_endpoint(route: &ResolvedRoute, operation: &str) -> String {
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
pub(super) fn count_tokens_request(call: &ResolvedCall) -> Result<EncodedRequest, Error> {
    let mut body = generate_body(call)?;
    body.insert(
        "model".to_owned(),
        format!("models/{}", call.route().api_model()).into(),
    );

    Ok(EncodedRequest::new(
        Method::POST,
        model_endpoint(call.route(), "countTokens"),
        json!({ "generateContentRequest": Value::Object(body) }),
    ))
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
pub(super) fn generate_body(call: &ResolvedCall) -> Result<Map<String, Value>, Error> {
    let request = call.request();
    let route = call.route();
    let (options, _) = wire_options(call);
    let declarations = function_declarations(request, route)?;

    let mut body = Map::new();
    let system = system_text(request.messages());
    if !system.is_empty() {
        body.insert(
            "systemInstruction".to_owned(),
            json!({ "parts": [{ "text": system }] }),
        );
    }
    body.insert("contents".to_owned(), Value::Array(contents(request)));
    let generation = generation_config(request, route);
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

/// The portable controls this request carries that the body did not encode.
///
/// Each is reported as an `unsupported_control` warning: the request still
/// reaches the model, but not as the caller wrote it.
///
/// - Request metadata has no field in this protocol.
/// - The `systemInstruction` field takes text only.
/// - An all-JSON tool result rides `functionResponse.response` natively, so
///   only a mix that must flatten is reported.
/// - This protocol has no latency tier, so the speed control is reported rather
///   than guessed at.
/// - Gemini 3 takes named thinking levels. Older and passthrough routes do not
///   claim that dialect, so their effort is reported.
/// - A reasoning part signed by another provider is skipped; see
///   `ReasoningContent::has_foreign_signature`.
pub(super) fn dropped_controls(call: &ResolvedCall) -> Vec<&'static str> {
    let request = call.request();
    let mut dropped = Vec::new();
    if !request.metadata().is_empty() {
        dropped.push("request metadata");
    }
    if flattens_system_content(request) {
        dropped.push("non-text system content");
    }
    if flattens_tool_result_content(request, text_or_json_only) {
        dropped.push("non-text tool result content");
    }
    if request.speed().is_some() {
        dropped.push("the speed control");
    }
    if request.reasoning_effort().is_some() && !takes_thinking_levels(call.route()) {
        dropped.push("the reasoning effort control");
    }
    if request.carries_foreign_signature(GEMINI_SIGNATURES) {
        dropped.push("reasoning signed by another provider");
    }
    dropped
}

/// Whether the route's model takes named `thinkingLevel` values.
fn takes_thinking_levels(route: &ResolvedRoute) -> bool {
    route.model().protocol_options().reasoning_effort_levels
}

/// The `functionDeclarations` for the request's tools.
///
/// # Errors
///
/// Returns [`ErrorKind::InvalidRequest`](crate::types::ErrorKind::InvalidRequest)
/// for a custom tool. This protocol has function tools only, and a custom
/// tool is never downgraded.
fn function_declarations(request: &Request, route: &ResolvedRoute) -> Result<Vec<Value>, Error> {
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
    Ok(declarations)
}

/// The `contents` array: every conversational turn, in wire order.
///
/// System and developer messages are excluded; they became the
/// `systemInstruction`. Consecutive same-role turns merge; see `Turns`.
/// Gemini's documented shape puts every `functionResponse` for a turn in
/// one `user` entry.
fn contents(request: &Request) -> Vec<Value> {
    let names = tool_call_names(request);
    let mut contents = Turns::default();
    for message in request.messages() {
        if message.is_instruction() {
            continue;
        }
        let parts: Vec<Value> = message
            .content()
            .iter()
            .filter_map(|part| encode_part(part, &names))
            .collect();
        let role = if message.role() == Role::Assistant {
            "model"
        } else {
            "user"
        };
        contents.push(role, parts);
    }
    contents.into_values("parts")
}

/// The `generationConfig` object: sampling, limits, thinking, and output
/// format.
fn generation_config(request: &Request, route: &ResolvedRoute) -> Map<String, Value> {
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
    if let Some(effort) = request.reasoning_effort()
        && takes_thinking_levels(route)
    {
        generation.insert(
            "thinkingConfig".to_owned(),
            json!({ "thinkingLevel": gemini_effort(effort) }),
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
    generation
}

/// Maps normalized effort onto the levels shared by the Gemini 3 roster.
fn gemini_effort(effort: ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::Minimal | ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High | ReasoningEffort::Xhigh | ReasoningEffort::Max => "high",
    }
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
        // Unknown content is rejected at the client and dispatch boundaries.
        ContentPart::Unknown(_) => None,
        ContentPart::Text { text } => Some(json!({ "text": text })),
        ContentPart::Image(image) => Some(encode_media(&image.source, "image/png")),
        ContentPart::Audio(audio) => Some(encode_media(&audio.source, "audio/wav")),
        ContentPart::Document(document) => Some(encode_media(&document.source, "application/pdf")),
        // A signature another provider family minted cannot verify here and
        // fails the request, so the part is skipped; the encoder reports it.
        ContentPart::Reasoning(reasoning) if reasoning.has_foreign_signature(GEMINI_SIGNATURES) => {
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
        ContentPart::Opaque { data, .. } => (part.opaque_namespace() == Some(NAMESPACE))
            .then(|| data.clone())
            .filter(Value::is_object),
    }
}

/// Encodes media as inline bytes or as a remote file reference.
///
/// A declared media type passes through verbatim. Vertex-style surfaces
/// require `fileData.mimeType`, so a URL source that declares none gets
/// `default_type` by attachment kind — the old library's defaults of
/// `image/png` for images, `audio/wav` for audio, and `application/pdf` for
/// documents.
fn encode_media(source: &MediaSource, default_type: &str) -> Value {
    match source {
        MediaSource::Base64 { data, media_type } => json!({
            "inlineData": { "mimeType": media_type, "data": data }
        }),
        MediaSource::Url { url, media_type } => json!({
            "fileData": {
                "fileUri": url,
                "mimeType": media_type.as_deref().unwrap_or(default_type),
            }
        }),
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
/// The signature is read only from the `gemini` metadata namespace.
fn encode_tool_call(call: &ToolCall) -> Value {
    let mut part = Map::new();
    part.insert(
        "functionCall".to_owned(),
        json!({ "id": call.id, "name": call.name, "args": call.input.wire_value() }),
    );
    let signature = call
        .provider_metadata
        .get(NAMESPACE)
        .and_then(|metadata| metadata.get("thoughtSignature"));
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
/// A successful result that is one JSON object is the exception: the caller
/// crafted the exact struct the model should see, so it becomes the whole
/// `response` verbatim rather than nesting under `output`. Non-object JSON,
/// mixed JSON parts, and text keep the `output` wrapping, and a failure keeps
/// the `error` key whatever shape it carries.
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
    // A lone JSON object is the caller's own `response` struct; see the doc
    // comment for the verbatim/wrapped split.
    let verbatim =
        matches!(result.content.as_slice(), [ContentPart::Json { value }] if value.is_object());
    let response = if result.is_error {
        json!({ "error": payload })
    } else if verbatim {
        payload
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
