//! The Gemini `generateContent` wire protocol.

use reqwest::Method;
use serde_json::{Map, Value, json};

use super::assembler::StreamAssembler;
use super::common::{
    endpoint, finish_reason, flattens_tool_result_content, merge_options, plain_text, sampling,
    system_text, unsupported_capability, wire_options,
};
use super::{Codec, StreamDecoder};
use crate::adapter::ResolvedCall;
use crate::resolver::ResolvedRoute;
use crate::transport::{EncodedRequest, SseEvent, provider_error};
use crate::types::{
    ContentBlockId, ContentBlockKind, ContentPart, Error, ErrorKind, MediaSource, ReasoningContent,
    Response, ResponseFormat, Role, StreamEvent, TokenCounts, ToolCall, ToolCallKind, ToolChoice,
    ToolDefinitionKind, ToolResult,
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
        if flattens_tool_result_content(call.request()) {
            encoded = encoded.unsupported_control("non-text tool result content");
        }
        Ok(encoded)
    }

    fn decode_response(&self, route: &ResolvedRoute, value: Value) -> Result<Response, Error> {
        let candidate = value.pointer("/candidates/0").unwrap_or(&Value::Null);
        let mut content = Vec::new();
        let mut calls = 0;
        for part in candidate
            .pointer("/content/parts")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let Some(function_call) = part.get("functionCall") {
                content.push(decode_tool_call(part, function_call, calls));
                calls += 1;
            } else if let Some(text) = part.get("text").and_then(Value::as_str) {
                content.push(decode_text(part, text));
            }
        }

        let id = value
            .get("responseId")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let finished = finish_reason(candidate.get("finishReason").and_then(Value::as_str));
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
fn count_tokens_request(call: &ResolvedCall) -> Result<EncodedRequest, Error> {
    let body = generate_body(call)?;

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
/// merged last, so an application can override anything encoded here.
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

    let mut contents: Vec<Value> = Vec::new();
    for message in request.messages() {
        if matches!(message.role(), Role::System | Role::Developer) {
            continue;
        }
        let parts: Vec<Value> = message.content().iter().filter_map(encode_part).collect();
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
    if let Some(format) = request.response_format() {
        generation.insert("responseMimeType".to_owned(), "application/json".into());
        if let ResponseFormat::JsonSchema { schema, .. } = format {
            generation.insert("responseJsonSchema".to_owned(), schema.clone());
        }
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
    Ok(body)
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
fn encode_part(part: &ContentPart) -> Option<Value> {
    match part {
        ContentPart::Text { text } => Some(json!({ "text": text })),
        ContentPart::Image(image) => Some(encode_media(&image.source)),
        ContentPart::Audio(audio) => Some(encode_media(&audio.source)),
        ContentPart::Document(document) => Some(encode_media(&document.source)),
        ContentPart::Reasoning(reasoning) => Some(encode_reasoning(reasoning)),
        ContentPart::ToolCall(call) => Some(encode_tool_call(call)),
        ContentPart::ToolResult(result) => Some(encode_tool_result(result)),
        // Gemini parts carry text, not structured JSON, so a JSON part is
        // replayed as the text the model originally produced.
        ContentPart::Json { value } => Some(json!({ "text": value.to_string() })),
        // A part this codec wrote is replayed verbatim; one another provider
        // wrote is dropped, so a conversation can still fail over to Gemini.
        ContentPart::Opaque { kind, data } => kind
            .split_once('.')
            .is_some_and(|(namespace, _)| namespace == NAMESPACE)
            .then(|| data.clone()),
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
fn encode_tool_call(call: &ToolCall) -> Value {
    let mut part = Map::new();
    part.insert(
        "functionCall".to_owned(),
        json!({ "id": call.id, "name": call.name, "args": call.arguments }),
    );
    if let Some(signature) = call
        .provider_metadata
        .get(NAMESPACE)
        .and_then(|metadata| metadata.get("thoughtSignature"))
    {
        part.insert("thoughtSignature".to_owned(), signature.clone());
    }
    Value::Object(part)
}

/// Encodes a tool result as the function response that answers a call.
fn encode_tool_result(result: &ToolResult) -> Value {
    json!({
        "functionResponse": {
            "id": result.tool_call_id,
            "name": result.name.as_deref().unwrap_or(&result.tool_call_id),
            "response": {
                "output": plain_text(&result.content),
                "is_error": result.is_error,
            },
        }
    })
}

/// Decodes a text part into visible text or reasoning.
fn decode_text(part: &Value, text: &str) -> ContentPart {
    if part.get("thought").and_then(Value::as_bool) != Some(true) {
        return ContentPart::Text {
            text: text.to_owned(),
        };
    }

    ContentPart::Reasoning(ReasoningContent {
        text:      text.to_owned(),
        signature: thought_signature(part).map(ToOwned::to_owned),
        redacted:  false,
    })
}

/// Decodes a `functionCall` part into a tool call.
///
/// `ordinal` counts the function calls already decoded from this response, so
/// the synthesized id is stable across repeated decodes of the same payload.
fn decode_tool_call(part: &Value, function_call: &Value, ordinal: usize) -> ContentPart {
    let name = function_call
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let mut call = ToolCall::function(
        tool_call_id(function_call, name, ordinal),
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
/// Gemini normally supplies no call id, so one is synthesized from the tool
/// name and the call's position in the response. That is deterministic, which a
/// minted UUID is not: decoding the same payload twice yields the same id.
fn tool_call_id(function_call: &Value, name: &str, ordinal: usize) -> String {
    function_call
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map_or_else(|| format!("{name}-{ordinal}"), ToOwned::to_owned)
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
    route:      ResolvedRoute,
    assembler:  StreamAssembler,
    /// The run currently accepting deltas, if any.
    open:       Option<(ContentBlockId, Run)>,
    texts:      usize,
    reasonings: usize,
    calls:      usize,
    /// Whether a chunk already failed, which forbids a completed response.
    failed:     bool,
}

impl GeminiStreamDecoder {
    fn new(route: &ResolvedRoute) -> Self {
        Self {
            route:      route.clone(),
            assembler:  StreamAssembler::new(route),
            open:       None,
            texts:      0,
            reasonings: 0,
            calls:      0,
            failed:     false,
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
        events.extend(
            self.assembler
                .start(id.clone(), ContentBlockKind::ToolCall {
                    id:   tool_call_id(function_call, name, self.calls),
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
        if let Some(reason) = value
            .pointer("/candidates/0/finishReason")
            .and_then(Value::as_str)
        {
            self.assembler
                .set_finish_reason(finish_reason(Some(reason)));
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
        ContentPart, ErrorKind, ImageContent, MediaSource, Message, Request, Response, Role,
        StreamEvent, ToolDefinition, ToolResult,
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
            json!({ "candidates": [{ "content": { "parts": [{ "text": "Hel" }] } }],
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
        assert_eq!(call.id, "search-0");
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
}
