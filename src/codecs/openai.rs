//! The OpenAI Responses protocol, `POST /v1/responses`.
//!
//! This is the only dialect that carries custom tools, that replays
//! provider-native reasoning items verbatim, and whose stream terminates with a
//! complete response document. It is also the only one with a native input
//! token count endpoint that projects the generation body down to an allowlist.

use std::collections::{BTreeMap, BTreeSet};

use reqwest::Method;
use serde_json::{Map, Value, json};

use super::assembler::StreamAssembler;
use super::common::{
    cache_routing_key, endpoint, flattens_system_content, flattens_tool_result_content,
    merge_options, parse_arguments, plain_text, refusal, reject_unencodable, sampling,
    wire_options,
};
use super::{Codec, StreamDecoder};
use crate::adapter::ResolvedCall;
use crate::resolver::ResolvedRoute;
use crate::transport::{EncodedRequest, SseEvent, provider_error};
use crate::types::{
    ContentBlockId, ContentBlockKind, ContentPart, Error, ErrorKind, FinishReason, MediaSource,
    Message, ReasoningContent, Request, Response, ResponseFormat, RetryClassification, Role, Speed,
    StreamEvent, TokenCounts, ToolCall, ToolCallKind, ToolChoice, ToolDefinition,
    ToolDefinitionKind, ToolResult,
};

/// The provider namespace this codec owns.
///
/// It claims [`ContentPart::Opaque`] kinds and
/// [`ToolCall::provider_metadata`] entries under this namespace and ignores
/// every other namespace, so a conversation carrying another provider's replay
/// data can still be sent here.
const NAMESPACE: &str = "openai";

/// The opaque kind holding one whole `reasoning` output item.
const REASONING_KIND: &str = "openai.reasoning";

/// The opaque kind holding one whole assistant `message` output item.
///
/// The item's `id` and `status` are what make it replayable: a `reasoning`
/// item names the item that must follow it, so a reconstructed, id-less
/// assistant message breaks the chain. Keeping the item verbatim is the only
/// way to send back the one the provider issued.
const MESSAGE_KIND: &str = "openai.message";

/// The kind the reference implementation persisted a `reasoning` item under.
const LEGACY_REASONING_KIND: &str = "openai_reasoning";

/// The kind the reference implementation persisted a `message` item under.
const LEGACY_MESSAGE_KIND: &str = "openai_message";

/// The fields `POST /v1/responses/input_tokens` accepts.
///
/// The projection runs after raw provider options are merged, so a stray
/// merged key is stripped as well.
const COUNT_TOKENS_FIELDS: &[&str] = &[
    "conversation",
    "input",
    "instructions",
    "model",
    "parallel_tool_calls",
    "previous_response_id",
    "reasoning",
    "text",
    "tool_choice",
    "tools",
    "truncation",
];

/// Encodes and decodes the OpenAI Responses protocol.
///
/// The Codex deployment speaks the same protocol with a smaller field set: it
/// rejects the sampling controls and takes the system prompt as `instructions`
/// rather than as input items.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct OpenAiResponsesCodec {
    codex: bool,
}

impl Codec for OpenAiResponsesCodec {
    fn encode(&self, call: &ResolvedCall, stream: bool) -> Result<EncodedRequest, Error> {
        let request = call.request();
        // This protocol carries no audio input. Refusing here is deliberate:
        // substituting a text placeholder would put words the caller never
        // wrote into the prompt, which is worse than a clear failure.
        //
        // An inline document must carry a name: the live API requires
        // `filename` beside `file_data` (verified 2026-08-30, a 400 names the
        // missing parameter), and inventing one would put a name the caller
        // never wrote in front of the model. URL documents need no name —
        // `file_url` stands alone.
        reject_unencodable(call.route(), request, |part| match part {
            ContentPart::Audio(_) => Some("audio content"),
            ContentPart::Document(document)
                if document.name.is_none()
                    && matches!(document.source, MediaSource::Base64 { .. }) =>
            {
                Some("an inline document without a file name")
            }
            _ => None,
        })?;
        let (options, controls) = wire_options(call);
        let mut body = self.generation_body(call, stream);
        if let Some(key) = cache_routing_key(call, controls) {
            body.insert("prompt_cache_key".to_owned(), key.into());
        }
        merge_options(&mut body, options);

        // The Codex deployment hangs its Responses endpoint directly off the
        // base path with no version segment: `<base>/responses` answers and
        // `<base>/v1/responses` is an HTML 403 (probed against
        // chatgpt.com/backend-api/codex on 2026-08-30).
        let path = if self.codex {
            "/responses"
        } else {
            "/v1/responses"
        };
        let mut encoded = EncodedRequest::new(
            Method::POST,
            endpoint(call.route().provider().base_url(), path),
            Value::Object(body),
        )
        .with_timeout(request.timeout())
        .with_applied_speed(request.speed());

        // Reasoning text has no input item in this protocol; only an
        // `openai.reasoning` opaque part replays. Dropping it is correct, but
        // it must not be silent — unless an opaque reasoning item sits in the
        // same message, because this codec's own decoder emits the readable
        // part BESIDE the opaque item that replays the same text, and warning
        // on the codec's own round trip would claim a loss on every turn.
        if request.messages().iter().any(|message| {
            let parts = message.content();
            parts
                .iter()
                .any(|part| matches!(part, ContentPart::Reasoning(_)))
                && !parts.iter().any(
                    |part| matches!(part, ContentPart::Opaque { kind, .. } if kind == REASONING_KIND),
                )
        }) {
            encoded = encoded.unsupported_control("replaying reasoning text");
        }

        // An all-JSON result travels as the bare value, so only a mix that
        // must flatten is reported.
        if flattens_tool_result_content(request, |parts| {
            parts
                .iter()
                .all(|part| matches!(part, ContentPart::Text { .. }))
                || parts
                    .iter()
                    .all(|part| matches!(part, ContentPart::Json { .. }))
        }) {
            encoded = encoded.unsupported_control("non-text tool result content");
        }

        // Codex hoists system messages into `instructions`, which is a string,
        // so anything but text in one is dropped. Standard mode keeps them as
        // input items, but a system or developer item takes `input_text` and
        // nothing else, so media in one is dropped there too. Either way the
        // text still reaches the model: a warning, not a refusal.
        if self.codex && flattens_system_content(request) {
            encoded = encoded.unsupported_control("non-text system content");
        }
        if !self.codex
            && request
                .messages()
                .iter()
                .filter(|message| matches!(message.role(), Role::System | Role::Developer))
                .flat_map(Message::content)
                .any(|part| matches!(part, ContentPart::Image(_) | ContentPart::Document(_)))
        {
            encoded = encoded.unsupported_control("non-text system content");
        }

        if self.codex {
            for (present, control) in [
                (request.temperature().is_some(), "temperature in Codex mode"),
                (request.top_p().is_some(), "top_p in Codex mode"),
                (
                    request.max_output_tokens().is_some(),
                    "max_output_tokens in Codex mode",
                ),
            ] {
                if present {
                    encoded = encoded.unsupported_control(control);
                }
            }
        }

        // `/v1/responses` takes no stop parameter at all: the live API answers
        // one with a 400 "Unknown parameter: 'stop'" on every model, reasoning
        // or not (probed against gpt-5.6-luna, gpt-5.4, and gpt-4o on
        // 2026-08-30). The sequences are dropped with a warning rather than
        // sent or refused; a Responses-compatible skin that does take a stop
        // member can still receive one through raw provider options.
        if !request.stop_sequences().is_empty() {
            encoded = encoded.unsupported_control("stop sequences");
        }

        Ok(encoded)
    }

    fn decode_response(&self, route: &ResolvedRoute, value: Value) -> Result<Response, Error> {
        decode_document(route, value)
    }

    fn stream_decoder(&self, route: &ResolvedRoute) -> Box<dyn StreamDecoder> {
        Box::new(ResponsesStream {
            assembler: StreamAssembler::new(route),
            route:     route.clone(),
            delivered: BTreeSet::new(),
            skipped:   BTreeSet::new(),
            started:   false,
        })
    }

    fn encode_count_tokens(&self, call: &ResolvedCall) -> Option<Result<EncodedRequest, Error>> {
        // The Codex deployment serves no input-token count: its
        // `/responses/input_tokens` path answers with an HTML 403 (probed
        // 2026-08-30), so codex mode reports no native count instead of
        // sending a request that cannot succeed.
        if self.codex {
            return None;
        }
        Some(self.count_tokens_request(call))
    }

    fn decode_count_tokens(&self, route: &ResolvedRoute, value: Value) -> Result<u64, Error> {
        let object = value.get("object").and_then(Value::as_str);
        if object != Some("response.input_tokens") {
            return Err(decode_error(
                route,
                format!(
                    "returned an input token count with object {}",
                    object.unwrap_or("<missing>")
                ),
                value,
            ));
        }

        match value.get("input_tokens").and_then(Value::as_u64) {
            Some(tokens) => Ok(tokens),
            None => Err(decode_error(
                route,
                "returned an input token count without an `input_tokens` number",
                value,
            )),
        }
    }
}

impl OpenAiResponsesCodec {
    /// Creates the codec for one provider.
    ///
    /// `codex` selects the Codex deployment's field set.
    pub(crate) fn new(codex: bool) -> Self {
        Self { codex }
    }

    /// Builds the token count request from the full non-streaming body.
    ///
    /// The body is generated and merged exactly as a generation request is, and
    /// only then projected onto [`COUNT_TOKENS_FIELDS`].
    fn count_tokens_request(self, call: &ResolvedCall) -> Result<EncodedRequest, Error> {
        let mut request = self.encode(call, false)?;
        request.url = endpoint(
            call.route().provider().base_url(),
            "/v1/responses/input_tokens",
        );
        if let Value::Object(body) = &mut request.body {
            body.retain(|key, _| COUNT_TOKENS_FIELDS.contains(&key.as_str()));
        }
        Ok(request)
    }

    /// Builds the `/v1/responses` body from typed request fields only.
    ///
    /// Raw provider options are merged over this by the caller, so every field
    /// here is overridable.
    fn generation_body(self, call: &ResolvedCall, stream: bool) -> Map<String, Value> {
        let request = call.request();
        let custom = CustomTools::new(request);
        let mut body = Map::new();

        body.insert(
            "model".to_owned(),
            Value::String(call.route().api_model().to_owned()),
        );
        // Codex takes the system prompt as `instructions` and rejects system
        // input items, so those messages are hoisted out of the input.
        if self.codex {
            body.insert(
                "instructions".to_owned(),
                Value::String(instructions(request)),
            );
        }
        body.insert(
            "input".to_owned(),
            Value::Array(
                request
                    .messages()
                    .iter()
                    .filter(|message| !self.codex || !is_system(message))
                    .flat_map(|message| input_items(message, &custom))
                    .collect(),
            ),
        );
        body.insert("stream".to_owned(), Value::Bool(stream));
        // This client keeps no server-side conversation state. Encrypted
        // reasoning is asked for instead, so a reasoning item can be replayed
        // from the transcript on the next turn.
        body.insert("store".to_owned(), Value::Bool(false));
        // Requested unconditionally, as the reference client did. Gating it
        // on the catalog's `reasoning` flag silently breaks multi-turn tool
        // calling for an overlay entry that omits the flag on a reasoning
        // model, and a model without reasoning ignores the include.
        body.insert("include".to_owned(), json!(["reasoning.encrypted_content"]));

        if !self.codex {
            if let Some(tokens) = request.max_output_tokens() {
                body.insert("max_output_tokens".to_owned(), tokens.into());
            }
            if let Some(temperature) = request.temperature() {
                body.insert("temperature".to_owned(), sampling(temperature));
            }
            if let Some(top_p) = request.top_p() {
                body.insert("top_p".to_owned(), sampling(top_p));
            }
        }

        body.extend(shared_body(request));
        body
    }
}

/// Decodes one complete response document.
///
/// The typed view is read from the same value that is then moved into
/// [`Response::raw`], so the response always carries the provider's own body.
fn decode_document(route: &ResolvedRoute, value: Value) -> Result<Response, Error> {
    if !value.is_object() {
        return Err(decode_error(
            route,
            "returned a response body that is not a JSON object",
            value,
        ));
    }
    // A body without an `output` array is not a Responses document. Decoding
    // `{}` or an error-shaped 200 from a broken gateway as a successful empty
    // response would hand the caller an answer the model never wrote.
    if !value.get("output").is_some_and(Value::is_array) {
        return Err(decode_error(
            route,
            "returned a 200 body without a Responses output array",
            value,
        ));
    }
    // A refusal is a failure, not a short answer — the same contract every
    // other codec applies. This protocol reports it as a `refusal` content
    // part inside a message item.
    if let Some(text) = refusal_text(&value) {
        let explanation = text.to_owned();
        return Err(refusal(route, Some(&explanation), Some(value)));
    }

    let content = decode_output(&value);
    let mut response = Response::new(
        route.provider().id().clone(),
        route.model().id().clone(),
        content,
    );
    response.id = value
        .get("id")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    response.finish_reason = decode_finish_reason(&value, &response.content);
    response.usage = decode_usage(value.get("usage"));
    // This protocol reports no in-band cost; the adapter prices the call from
    // the catalog instead.
    response.cost = None;
    response.raw = Some(value);
    Ok(response)
}

/// The fields the Codex deployment and the public API encode identically.
fn shared_body(request: &Request) -> Map<String, Value> {
    let mut body = Map::new();

    if let Some(effort) = request.reasoning_effort() {
        let effort = serde_json::to_value(effort).unwrap_or(Value::Null);
        body.insert("reasoning".to_owned(), json!({ "effort": effort }));
    }
    if let Some(speed) = request.speed() {
        let tier = match speed {
            Speed::Fast => "priority",
            Speed::Balanced => "auto",
            Speed::Economical => "flex",
        };
        body.insert("service_tier".to_owned(), Value::String(tier.to_owned()));
    }
    if !request.tools().is_empty() {
        body.insert(
            "tools".to_owned(),
            Value::Array(request.tools().iter().map(tool_definition).collect()),
        );
    }
    if let Some(choice) = request.tool_choice() {
        body.insert("tool_choice".to_owned(), tool_choice(choice));
    }
    if let Some(format) = request.response_format() {
        body.insert(
            "text".to_owned(),
            json!({ "format": response_format(format) }),
        );
    }
    if !request.metadata().is_empty() {
        let metadata = request
            .metadata()
            .iter()
            .map(|(key, value)| (key.clone(), Value::String(value.clone())))
            .collect::<Map<String, Value>>();
        body.insert("metadata".to_owned(), Value::Object(metadata));
    }

    body
}

/// The joined text of every system and developer message.
fn instructions(request: &Request) -> String {
    request
        .messages()
        .iter()
        .filter(|message| is_system(message))
        .map(|message| plain_text(message.content()))
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Whether a message carries system or developer instructions.
fn is_system(message: &Message) -> bool {
    matches!(message.role(), Role::System | Role::Developer)
}

/// The tool kinds one request declared, used to route tool results.
///
/// A `custom_tool_call_output` is a different wire item from a
/// `function_call_output`, and a result carries no kind of its own. The kind is
/// recovered from the call it answers, from the tool it names, or from the
/// message label the caller set.
struct CustomTools {
    /// Names declared as custom tools by this request.
    names: BTreeSet<String>,
    /// Call ids of custom tool calls earlier in this request.
    calls: BTreeSet<String>,
}

impl CustomTools {
    fn new(request: &Request) -> Self {
        let names = request
            .tools()
            .iter()
            .filter(|tool| tool.is_custom())
            .map(|tool| tool.name.clone())
            .collect();
        let calls = request
            .messages()
            .iter()
            .flat_map(Message::content)
            .filter_map(|part| match part {
                ContentPart::ToolCall(call) if call.kind == ToolCallKind::Custom => {
                    Some(call.id.clone())
                }
                _ => None,
            })
            .collect();

        Self { names, calls }
    }

    /// Whether a tool result answers a custom tool call.
    fn answers_custom(&self, result: &ToolResult, message: &Message) -> bool {
        if self.calls.contains(&result.tool_call_id) {
            return true;
        }
        let named = result.name.as_deref().or_else(|| message.name());
        named.is_some_and(|name| self.names.contains(name))
    }
}

/// Encodes one message as the input items it produces.
///
/// Items come out in the order the content parts carry them, because this
/// protocol reads the input list positionally: a `reasoning` item names the
/// item that must follow it, so replaying `[reasoning, message, reasoning,
/// function_call]` in any other order breaks the pairing. The parts that
/// belong inside a single message item — text, JSON, images, documents — are
/// collected into one item emitted where the first of them appeared.
fn input_items(message: &Message, custom: &CustomTools) -> Vec<Value> {
    let has_result = message
        .content()
        .iter()
        .any(|part| matches!(part, ContentPart::ToolResult(_)));
    if message.role() == Role::Tool && !has_result {
        // A tool message whose content is only text still answers a call. The
        // message-level label is the only place that id survives.
        if let Some(call_id) = message.tool_call_id() {
            let mut items: Vec<Value> = opaque_items(message).cloned().collect();
            // The same routing the ToolResult path applies: a result
            // answering a custom call earlier in the request is custom even
            // when the tool is not redeclared and the message carries no
            // name.
            let custom = custom.calls.contains(call_id)
                || message
                    .name()
                    .is_some_and(|name| custom.names.contains(name));
            items.push(tool_output_item(
                call_id,
                &plain_text(message.content()),
                false,
                custom,
            ));
            return items;
        }
    }

    // A preserved `message` item already carries the assistant text, with the
    // id and status a reconstructed item cannot have. Sending both would
    // repeat the text on the wire.
    let replays_message = opaque_items(message)
        .any(|item| item.get("type").and_then(Value::as_str) == Some("message"));
    let content = message_content(message, replays_message);
    let mut pending = (!content.is_empty()).then(|| {
        json!({
            "role":    role(message.role()),
            "content": content,
        })
    });

    let mut items = Vec::new();
    for part in message.content() {
        match part {
            ContentPart::Opaque { data, .. } if claims_opaque(part) => {
                items.push(data.clone());
            }
            // A nameless call cannot be encoded — the provider rejects an
            // empty function name. This decoder never produces one, but a
            // caller-built or foreign-history call can carry one; the
            // reference client skipped those.
            ContentPart::ToolCall(call) if call.name.is_empty() => {}
            ContentPart::ToolCall(call) => items.push(tool_call_item(call)),
            ContentPart::ToolResult(result) => items.push(tool_output_item(
                &result.tool_call_id,
                &result_output(result),
                result.is_error,
                custom.answers_custom(result, message),
            )),
            // A text part a preserved `message` item already carries places
            // nothing of its own.
            ContentPart::Text { .. } if replays_message => {}
            // The first part that belongs inside the message item emits the
            // whole item, so it lands where the caller put its content.
            ContentPart::Text { .. }
            | ContentPart::Json { .. }
            | ContentPart::Image(_)
            | ContentPart::Document(_) => items.extend(pending.take()),
            // Audio never reaches here, reasoning text has no input item, and
            // another provider's opaque part is not ours to send.
            ContentPart::Audio(_) | ContentPart::Reasoning(_) | ContentPart::Opaque { .. } => {}
        }
    }
    // A message whose only content is skipped assistant text still has to send
    // the parts that survived.
    items.extend(pending);

    items
}

/// The verbatim replay items this codec owns, in content order.
fn opaque_items(message: &Message) -> impl Iterator<Item = &Value> {
    message.content().iter().filter_map(|part| match part {
        ContentPart::Opaque { data, .. } if claims_opaque(part) => Some(data),
        _ => None,
    })
}

/// Whether an opaque part carries one of this protocol's output items.
///
/// The dotted kinds are this crate's own. The underscore spellings are what
/// the reference implementation persisted; both hold the provider's output
/// item verbatim, so a migrated history replays through the same path.
fn claims_opaque(part: &ContentPart) -> bool {
    part.opaque_namespace() == Some(NAMESPACE)
        || matches!(
            part,
            ContentPart::Opaque { kind, .. }
                if kind == LEGACY_REASONING_KIND || kind == LEGACY_MESSAGE_KIND
        )
}

/// Encodes the content parts that belong inside a message item.
///
/// `skip_text` drops the text parts a preserved `message` item already
/// carries, so replay never doubles the assistant's own words.
fn message_content(message: &Message, skip_text: bool) -> Vec<Value> {
    let text_type = match message.role() {
        Role::Assistant => "output_text",
        Role::System | Role::Developer | Role::User | Role::Tool => "input_text",
    };
    let system = matches!(message.role(), Role::System | Role::Developer);

    message
        .content()
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text { .. } if skip_text => None,
            // A system or developer item takes `input_text` and nothing else;
            // media inside one draws a provider 400. Only the text travels,
            // and `encode` reports the drop.
            ContentPart::Image(_) | ContentPart::Document(_) if system => None,
            ContentPart::Text { text } => Some(json!({ "type": text_type, "text": text })),
            ContentPart::Json { value } => Some(json!({
                "type": text_type,
                "text": serde_json::to_string(value).unwrap_or_default(),
            })),
            ContentPart::Image(image) => {
                let mut item = json!({
                    "type": "input_image",
                    "image_url": media_url(&image.source),
                });
                if let (Some(object), Some(detail)) = (item.as_object_mut(), image.detail.as_ref())
                {
                    object.insert("detail".to_owned(), Value::String(detail.clone()));
                }
                Some(item)
            }
            // Documents ride inline, either as a base64 data URL or as a
            // fetchable URL.
            ContentPart::Document(document) => {
                let mut item = json!({ "type": "input_file" });
                if let Some(object) = item.as_object_mut() {
                    match &document.source {
                        MediaSource::Url { url, .. } => {
                            object.insert("file_url".to_owned(), Value::String(url.clone()));
                        }
                        MediaSource::Base64 { .. } => {
                            object.insert(
                                "file_data".to_owned(),
                                Value::String(media_url(&document.source)),
                            );
                        }
                    }
                    if let Some(name) = document.name.as_ref() {
                        object.insert("filename".to_owned(), Value::String(name.clone()));
                    }
                }
                Some(item)
            }
            // Audio never reaches here: `encode` rejects it before dispatch
            // rather than substituting a placeholder, because injecting
            // invented text into a caller's prompt is worse than refusing.
            // Reasoning replays through its opaque item, and calls and results
            // are separate input items.
            ContentPart::Audio(_)
            | ContentPart::Reasoning(_)
            | ContentPart::ToolCall(_)
            | ContentPart::ToolResult(_)
            | ContentPart::Opaque { .. } => None,
        })
        .collect()
}

/// The wire role for one message.
fn role(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::Developer => "developer",
        Role::Assistant => "assistant",
        Role::User | Role::Tool => "user",
    }
}

/// Encodes one tool call as its input item.
///
/// The item id is replayed from this codec's own provider metadata namespace
/// when the decoder kept one, because a replayed item without it cannot anchor
/// the reasoning chain.
fn tool_call_item(call: &ToolCall) -> Value {
    let arguments = call
        .raw_arguments
        .clone()
        .unwrap_or_else(|| match &call.arguments {
            Value::String(input) => input.clone(),
            other => serde_json::to_string(other).unwrap_or_default(),
        });

    let mut item = match call.kind {
        ToolCallKind::Function => json!({
            "type": "function_call",
            "call_id": call.id,
            "name": call.name,
            "arguments": arguments,
        }),
        ToolCallKind::Custom => json!({
            "type": "custom_tool_call",
            "call_id": call.id,
            "name": call.name,
            "input": arguments,
        }),
    };

    let item_id = call
        .provider_metadata
        .get(NAMESPACE)
        .and_then(|metadata| metadata.get("item_id"))
        // The reference implementation stored the `fc_` item id at the
        // metadata top level; a migrated call keeps its item identity.
        .or_else(|| call.provider_metadata.get("id"))
        .and_then(Value::as_str);
    if let (Some(object), Some(item_id)) = (item.as_object_mut(), item_id) {
        object.insert("id".to_owned(), Value::String(item_id.to_owned()));
    }

    item
}

/// Encodes one tool result as its input item.
///
/// `custom_tool_call_output` has no error channel, so an error is reported
/// through the item status only for function outputs.
fn tool_output_item(call_id: &str, output: &str, is_error: bool, custom: bool) -> Value {
    if custom {
        return json!({
            "type": "custom_tool_call_output",
            "call_id": call_id,
            "output": output,
        });
    }

    let mut item = json!({
        "type": "function_call_output",
        "call_id": call_id,
        "output": output,
    });
    if let (Some(object), true) = (item.as_object_mut(), is_error) {
        object.insert("status".to_owned(), Value::String("incomplete".to_owned()));
    }
    item
}

/// The text a tool result sends back.
///
/// A result whose content is only structured JSON sends the bare value — one
/// value on its own, several as an array — rather than the `ContentPart`
/// envelope that wraps it, because the tool's own JSON is what the model was
/// promised. Text-only content sends the joined text even when it is empty —
/// a command with no output answered with nothing, not with a serialized
/// envelope. Anything else falls back to the serialized parts, which keeps
/// mixed content readable instead of dropping the half this protocol has no
/// field for.
fn result_output(result: &ToolResult) -> String {
    let text = plain_text(&result.content);
    let text_only = result
        .content
        .iter()
        .all(|part| matches!(part, ContentPart::Text { .. }));
    if !text.is_empty() || text_only {
        return text;
    }

    let values: Vec<&Value> = result
        .content
        .iter()
        .filter_map(|part| match part {
            ContentPart::Json { value } => Some(value),
            _ => None,
        })
        .collect();
    match values.as_slice() {
        [value] if result.content.len() == 1 => serde_json::to_string(value).unwrap_or_default(),
        values if !values.is_empty() && values.len() == result.content.len() => {
            serde_json::to_string(values).unwrap_or_default()
        }
        _ => serde_json::to_string(&result.content).unwrap_or_default(),
    }
}

/// Encodes one image or media source as the URL this protocol accepts.
fn media_url(source: &MediaSource) -> String {
    match source {
        MediaSource::Url { url, .. } => url.clone(),
        MediaSource::Base64 { data, media_type } => format!("data:{media_type};base64,{data}"),
    }
}

/// Encodes one tool definition in the flat Responses shape.
///
/// This is the only codec that encodes a custom tool. A custom tool sends its
/// `format` and no `parameters`.
fn tool_definition(tool: &ToolDefinition) -> Value {
    match &tool.kind {
        ToolDefinitionKind::Function { input_schema } => json!({
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": input_schema,
        }),
        ToolDefinitionKind::Custom { format } => json!({
            "type": "custom",
            "name": tool.name,
            "description": tool.description,
            "format": format,
        }),
    }
}

/// Encodes the tool selection mode.
fn tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::None => json!("none"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Tool { name } => json!({ "type": "function", "name": name }),
    }
}

/// Encodes the requested response shape as a `text.format` value.
fn response_format(format: &ResponseFormat) -> Value {
    match format {
        ResponseFormat::Text => json!({ "type": "text" }),
        ResponseFormat::JsonObject => json!({ "type": "json_object" }),
        ResponseFormat::JsonSchema { name, schema } => json!({
            "type": "json_schema",
            "name": name,
            "schema": schema,
            "strict": true,
        }),
    }
}

/// Decodes the `output` array of a complete response document.
///
/// A `message` item contributes both its visible text and the whole item as an
/// [`MESSAGE_KIND`] opaque part, because only the original item replays.
///
/// A tool call with no name is a model-internal item rather than a call the
/// caller can answer. It is dropped, which also keeps it from turning the
/// finish reason into [`FinishReason::ToolCall`].
fn decode_output(value: &Value) -> Vec<ContentPart> {
    let mut content = Vec::new();
    let items = value
        .get("output")
        .and_then(Value::as_array)
        .into_iter()
        .flatten();

    for item in items {
        match item.get("type").and_then(Value::as_str) {
            Some("message") => {
                let text = message_text(item);
                if !text.is_empty() {
                    content.push(ContentPart::Text { text });
                }
                content.push(ContentPart::opaque(MESSAGE_KIND, item.clone()));
            }
            Some(kind @ ("function_call" | "custom_tool_call")) => {
                let call_kind = match kind {
                    "custom_tool_call" => ToolCallKind::Custom,
                    _ => ToolCallKind::Function,
                };
                let call = decode_tool_call(item, call_kind);
                if !call.name.is_empty() {
                    content.push(ContentPart::ToolCall(call));
                }
            }
            Some("reasoning") => content.extend(decode_reasoning(item)),
            // Provider-side items such as `web_search_call` carry no portable
            // content. They stay available in `Response::raw`.
            _ => {}
        }
    }

    content
}

/// The refusal text carried by a document's message items, if any.
///
/// A non-empty `refusal` content part means the model declined instead of
/// answering; an empty one is treated as absent, as the Chat codec treats an
/// empty `refusal` field.
fn refusal_text(value: &Value) -> Option<&str> {
    value
        .get("output")?
        .as_array()?
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("message"))
        .filter_map(|item| item.get("content").and_then(Value::as_array))
        .flatten()
        .filter(|part| part.get("type").and_then(Value::as_str) == Some("refusal"))
        .find_map(|part| part.get("refusal").and_then(Value::as_str))
        .filter(|text| !text.is_empty())
}

/// The visible text of one `message` output item.
fn message_text(item: &Value) -> String {
    item.get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|part| part.get("type").and_then(Value::as_str) == Some("output_text"))
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect()
}

/// Decodes one `function_call` or `custom_tool_call` output item.
///
/// `call_id` is the identity a tool result answers, so it is the canonical id.
/// The item id is a second identifier the protocol needs when the call is
/// replayed, so it is kept in this codec's provider metadata namespace when it
/// differs.
fn decode_tool_call(item: &Value, kind: ToolCallKind) -> ToolCall {
    let call_id = item.get("call_id").and_then(Value::as_str);
    let item_id = item.get("id").and_then(Value::as_str);
    let raw = match kind {
        ToolCallKind::Function => item.get("arguments"),
        ToolCallKind::Custom => item.get("input"),
    }
    .and_then(Value::as_str)
    .unwrap_or_default();

    let mut call = ToolCall {
        id: call_id.or(item_id).unwrap_or_default().to_owned(),
        name: item
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        arguments: match kind {
            ToolCallKind::Function => parse_arguments(raw),
            ToolCallKind::Custom => Value::String(raw.to_owned()),
        },
        kind,
        // An empty string means the provider sent no arguments, which the
        // streamed form of the same call also reports as absent.
        raw_arguments: (!raw.is_empty()).then(|| raw.to_owned()),
        provider_metadata: BTreeMap::new(),
    };
    if let (Some(call_id), Some(item_id)) = (call_id, item_id)
        && call_id != item_id
    {
        call.provider_metadata
            .insert(NAMESPACE.to_owned(), json!({ "item_id": item_id }));
    }

    call
}

/// Decodes one `reasoning` output item into the parts it contributes.
///
/// The visible summary becomes reasoning content, and the whole item is kept
/// verbatim — summary-only items included. Replaying a turn's function calls
/// without their preceding reasoning item draws a provider 400 when the
/// conversation is not stored, and only the original item, its id included,
/// satisfies that pairing.
fn decode_reasoning(item: &Value) -> Vec<ContentPart> {
    let text = reasoning_text(item);
    let mut parts = Vec::new();
    if !text.is_empty() {
        parts.push(ContentPart::Reasoning(ReasoningContent {
            text,
            signature: None,
            signature_origin: None,
            redacted: false,
        }));
    }
    parts.push(ContentPart::opaque(REASONING_KIND, item.clone()));
    parts
}

/// The visible text of one `reasoning` output item.
fn reasoning_text(item: &Value) -> String {
    ["summary", "content"]
        .iter()
        .filter_map(|key| item.get(*key))
        .filter_map(Value::as_array)
        .flatten()
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("")
}

/// Normalizes the response status into a finish reason.
///
/// The protocol has no separate stop reason. A completed response that
/// requested tools reports [`FinishReason::ToolCall`] so consumers can branch
/// on the same value every other codec produces.
fn decode_finish_reason(value: &Value, content: &[ContentPart]) -> FinishReason {
    match value.get("status").and_then(Value::as_str) {
        Some("incomplete") => {
            match value
                .pointer("/incomplete_details/reason")
                .and_then(Value::as_str)
            {
                Some("content_filter") => FinishReason::ContentFilter,
                _ => FinishReason::Length,
            }
        }
        Some("failed") => FinishReason::Error,
        // A status this crate does not name — `cancelled`, say — keeps the
        // provider's own spelling instead of collapsing into `Stop`.
        Some(other) if other != "completed" => FinishReason::Other(other.to_owned()),
        _ if content
            .iter()
            .any(|part| matches!(part, ContentPart::ToolCall(_))) =>
        {
            FinishReason::ToolCall
        }
        _ => FinishReason::Stop,
    }
}

/// Normalizes the inclusive usage counters into disjoint buckets.
///
/// `input_tokens` includes the cached and cache-written tokens and
/// `output_tokens` includes the reasoning tokens. The GPT-5.6 family bills
/// cache writes at a premium and reports them as
/// `input_tokens_details.cache_write_tokens` (observed live on 2026-08-30);
/// models that bill no write omit the counter or send zero, which decodes to
/// an empty bucket either way.
fn decode_usage(usage: Option<&Value>) -> TokenCounts {
    let Some(usage) = usage else {
        return TokenCounts::default();
    };
    let count = |pointer: &str| usage.pointer(pointer).and_then(Value::as_u64).unwrap_or(0);

    TokenCounts::from_inclusive(
        count("/input_tokens"),
        count("/output_tokens"),
        count("/output_tokens_details/reasoning_tokens"),
        count("/input_tokens_details/cached_tokens"),
        count("/input_tokens_details/cache_write_tokens"),
    )
}

/// The error a malformed success body produces.
///
/// A structurally malformed 200 is indistinguishable from a garbled or
/// truncated body, so a fresh attempt is safe — the same classification the
/// transport gives a 200 whose body is not JSON at all.
fn decode_error(route: &ResolvedRoute, detail: impl Into<String>, raw: Value) -> Error {
    Error::new(
        ErrorKind::ResponseDecode,
        format!("provider {} {}", route.provider().id(), detail.into()),
    )
    .with_provider(route.provider().id().clone())
    .with_raw_data(raw)
    .with_retry(RetryClassification::Safe)
}

/// Decodes one `/v1/responses` stream.
struct ResponsesStream {
    assembler: StreamAssembler,
    route:     ResolvedRoute,
    /// Blocks that already received content, so a terminal reasoning item
    /// knows whether visible text streamed and its opaque replay block needs
    /// a derived id.
    delivered: BTreeSet<ContentBlockId>,
    /// Blocks for model-internal items, whose deltas open no block at all.
    skipped:   BTreeSet<ContentBlockId>,
    /// Whether the `Started` event has been emitted for this stream.
    started:   bool,
}

impl StreamDecoder for ResponsesStream {
    fn decode(&mut self, event: SseEvent) -> Result<Vec<StreamEvent>, Error> {
        let data = event.data.trim();
        if data.is_empty() || data == "[DONE]" {
            return Ok(Vec::new());
        }

        // A frame this decoder cannot parse is skipped rather than fatal.
        // Proxies inject their own keepalive payloads, and killing a
        // generation over a frame that carries no model output would trade a
        // whole response for a comment.
        let Ok(value) = serde_json::from_str::<Value>(data) else {
            return Ok(Vec::new());
        };

        let kind = value
            .get("type")
            .and_then(Value::as_str)
            .or(event.event.as_deref())
            .unwrap_or_default();

        // `response.created` is where the response id arrives, but a proxy can
        // drop it. Latching on the first event that is not a failure keeps a
        // consumer from seeing deltas before the stream ever started, which is
        // what the Chat dialect does with its first chunk.
        let mut latched = Vec::new();
        if !self.started && !matches!(kind, "error" | "response.failed") {
            self.started = true;
            let id = value
                .pointer("/response/id")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
            latched.push(self.assembler.started(id));
        }
        let events = self.decode_event(kind, &value)?;
        latched.extend(events);
        Ok(latched)
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>, Error> {
        Ok(self.assembler.complete())
    }
}

impl ResponsesStream {
    /// Decodes one recognized stream event.
    fn decode_event(&mut self, kind: &str, value: &Value) -> Result<Vec<StreamEvent>, Error> {
        match kind {
            "error" => Err(provider_error(
                self.route.provider(),
                None,
                Some(value.clone()),
                None,
            )),
            // Some gateways flatten the failure to `{"type":"response.failed",
            // "error":{...}}`. Falling back to the whole event keeps the
            // provider's code and message when the `response` wrapper is gone.
            "response.failed" => Err(provider_error(
                self.route.provider(),
                None,
                Some(value.get("response").unwrap_or(value).clone()),
                None,
            )),
            "response.output_item.added" => Ok(self.start_item(&block_id(value), item(value))),
            "response.output_text.delta" => Ok(self.text_delta(value)),
            "response.reasoning_text.delta" | "response.reasoning_summary_text.delta" => {
                Ok(self.reasoning_delta(value))
            }
            "response.function_call_arguments.delta" | "response.custom_tool_call_input.delta" => {
                Ok(self.arguments_delta(value))
            }
            "response.output_item.done" => Ok(self.end_item(&block_id(value), item(value))),
            "response.completed" | "response.incomplete" => {
                // A refusal fails the stream instead of completing it as an
                // empty answer — the same contract every other codec applies.
                let document = value.get("response").unwrap_or(value);
                if let Some(text) = refusal_text(document) {
                    return Err(refusal(&self.route, Some(text), Some(document.clone())));
                }
                Ok(self.complete(value))
            }
            // `response.created` lands here: its id already rode the latched
            // `Started` event, so it contributes nothing of its own.
            _ => Ok(Vec::new()),
        }
    }

    /// Opens the block for one output item.
    ///
    /// Opening is idempotent, so the terminal event for an item the provider
    /// never announced still opens the right kind of block. A reasoning item
    /// is deliberately not opened here: whether it becomes visible reasoning
    /// or an opaque replay item is only known once the item is done. A
    /// message item is not opened here either — its text block latches on the
    /// first text that actually arrives, so a message with none (a
    /// refusal-only message, an empty assistant turn) never emits the empty
    /// `Text` part the blocking decode of the same body omits.
    fn start_item(&mut self, id: &ContentBlockId, item: &Value) -> Vec<StreamEvent> {
        if is_internal_call(item) {
            self.skipped.insert(id.clone());
            return Vec::new();
        }

        match item.get("type").and_then(Value::as_str) {
            Some(kind @ ("function_call" | "custom_tool_call")) => {
                let call_kind = match kind {
                    "custom_tool_call" => ToolCallKind::Custom,
                    _ => ToolCallKind::Function,
                };
                let call_id = item.get("call_id").and_then(Value::as_str);
                let item_id = item.get("id").and_then(Value::as_str);
                let identity = ContentBlockKind::ToolCall {
                    id:   call_id.or(item_id).unwrap_or_default().to_owned(),
                    name: item
                        .get("name")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned),
                    kind: call_kind,
                };
                let mut events = self.assembler.start(id.clone(), identity.clone());
                // When a lost `output_item.added` let an early fragment latch
                // the fallback block, the terminal item event lands here and
                // restores the call id and name the fallback lost.
                self.assembler.repair_tool_identity(id, identity);
                if let (Some(call_id), Some(item_id)) = (call_id, item_id)
                    && call_id != item_id
                {
                    events.extend(self.assembler.provider_metadata(
                        id,
                        NAMESPACE,
                        json!({ "item_id": item_id }),
                    ));
                }
                events
            }
            _ => Vec::new(),
        }
    }

    /// Closes the block for one output item, delivering anything the deltas
    /// did not carry.
    fn end_item(&mut self, id: &ContentBlockId, item: &Value) -> Vec<StreamEvent> {
        if self.skipped.contains(id) {
            return Vec::new();
        }
        if is_internal_call(item) {
            self.skipped.insert(id.clone());
            // A lost `output_item.added` lets the argument deltas latch a
            // fallback block before the name is known. The terminal item
            // reveals the call is model-internal: the end event closes what
            // consumers saw open, but the part is discarded — blocking decode
            // drops the item, and a nameless call must not become content or
            // flip the finish reason. Nothing opens on the normal path, so
            // this is usually empty.
            return self.assembler.discard(id);
        }
        if item.get("type").and_then(Value::as_str) == Some("reasoning") {
            return self.end_reasoning(id, item);
        }

        let mut events = self.start_item(id, item);
        events.extend(self.reconcile_item(id, item));
        events.extend(self.assembler.end(id));

        // The message item itself replays; the text block only carries what a
        // reader sees. The opaque block takes a derived id when the item's
        // own id already named a text block, and the item's id when no text
        // arrived — the same pairing reasoning items use.
        if item.get("type").and_then(Value::as_str) == Some("message") {
            let opaque = if self.delivered.contains(id) {
                ContentBlockId::new(format!("{}-item", id.as_str()))
            } else {
                id.clone()
            };
            events.extend(self.opaque_item(&opaque, MESSAGE_KIND, item));
        }
        events
    }

    /// Reconciles a block with its terminal item event, which carries the
    /// item's complete content.
    ///
    /// The terminal event is the ground truth: an item that streamed no
    /// deltas is delivered whole, a delta lost in transit has its missing
    /// tail appended, and a buffer that disagrees is replaced — so streaming
    /// and blocking decode the same response identically.
    fn reconcile_item(&mut self, id: &ContentBlockId, item: &Value) -> Vec<StreamEvent> {
        let (kind, content) = match item.get("type").and_then(Value::as_str) {
            Some("message") => (ContentBlockKind::Text, message_text(item)),
            Some(kind @ ("function_call" | "custom_tool_call")) => {
                let key = match kind {
                    "custom_tool_call" => "input",
                    _ => "arguments",
                };
                let fallback = ContentBlockKind::ToolCall {
                    id:   id.as_str().to_owned(),
                    name: None,
                    kind: ToolCallKind::Function,
                };
                (
                    fallback,
                    item.get(key)
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                )
            }
            _ => return Vec::new(),
        };
        if content.is_empty() {
            return Vec::new();
        }
        self.deliver(id, |assembler| assembler.reconcile(id, kind, &content))
    }

    /// Closes a reasoning item, keeping the whole item when it must be
    /// replayed.
    ///
    /// The opaque block reuses the item's own block id when no reasoning text
    /// streamed, and takes a derived id when it did, so both parts of one item
    /// keep distinct stable ids.
    fn end_reasoning(&mut self, id: &ContentBlockId, item: &Value) -> Vec<StreamEvent> {
        let text = reasoning_text(item);
        let mut events = Vec::new();
        // The terminal item is the ground truth for the visible text, the
        // same reconciliation the other item kinds get.
        if !text.is_empty() {
            events.extend(self.deliver(id, |assembler| {
                assembler.reconcile(id, ContentBlockKind::Reasoning, &text)
            }));
        }
        let visible = self.delivered.contains(id);
        events.extend(self.assembler.end(id));

        // Every reasoning item is kept whole, summary-only ones included;
        // see `decode_reasoning` for the replay pairing that requires it.
        let opaque = if visible {
            ContentBlockId::new(format!("{}-item", id.as_str()))
        } else {
            id.clone()
        };
        events.extend(self.opaque_item(&opaque, REASONING_KIND, item));
        events
    }

    /// Emits one whole output item as a closed opaque block.
    fn opaque_item(&mut self, id: &ContentBlockId, kind: &str, item: &Value) -> Vec<StreamEvent> {
        let mut events = self.assembler.start(id.clone(), ContentBlockKind::Opaque {
            kind: kind.to_owned(),
        });
        events.extend(self.assembler.set_opaque_data(id, item.clone()));
        events.extend(self.assembler.end(id));
        events
    }

    fn text_delta(&mut self, value: &Value) -> Vec<StreamEvent> {
        let id = block_id(value);
        let text = delta_text(value);
        self.deliver(&id, |assembler| assembler.text(&id, &text))
    }

    fn reasoning_delta(&mut self, value: &Value) -> Vec<StreamEvent> {
        let id = block_id(value);
        let text = delta_text(value);
        self.deliver(&id, |assembler| assembler.reasoning(&id, &text))
    }

    /// Appends one argument fragment, unless the item it belongs to is
    /// model-internal and has no block of its own.
    fn arguments_delta(&mut self, value: &Value) -> Vec<StreamEvent> {
        let id = block_id(value);
        if self.skipped.contains(&id) {
            return Vec::new();
        }
        let chunk = delta_text(value);
        self.deliver(&id, |assembler| assembler.arguments(&id, &chunk))
    }

    /// Runs one assembler call and records that the block carried content.
    fn deliver(
        &mut self,
        id: &ContentBlockId,
        call: impl FnOnce(&mut StreamAssembler) -> Vec<StreamEvent>,
    ) -> Vec<StreamEvent> {
        let events = call(&mut self.assembler);
        self.delivered.insert(id.clone());
        events
    }

    /// Completes the stream from the provider's own final response document.
    ///
    /// This is the only protocol that sends one, so the finished response keeps
    /// the provider's id, usage, finish reason, and complete raw body rather
    /// than a document assembled from the event log.
    ///
    /// The terminal shape is read tolerantly: a gateway that flattens the
    /// document into the event itself, or trims it below what
    /// [`decode_document`] accepts, must not fail a stream whose answer was
    /// already delivered — the streamed blocks are the response, and whatever
    /// id, usage, and status the terminal event does carry is salvaged.
    fn complete(&mut self, value: &Value) -> Vec<StreamEvent> {
        let document = value.get("response").unwrap_or(value);
        let Ok(response) = decode_document(&self.route, document.clone()) else {
            return self.salvage(document);
        };
        if let Some(id) = response.id {
            self.assembler.set_id(id);
        }
        // The document's own output decides the reason, but a middlebox can
        // trim `output` in the terminal event — the streamed blocks are the
        // ground truth for whether the model called a tool.
        let finish_reason =
            if response.finish_reason == FinishReason::Stop && self.assembler.has_tool_call() {
                FinishReason::ToolCall
            } else {
                response.finish_reason
            };
        self.assembler.set_finish_reason(finish_reason);
        self.assembler.set_raw(document.clone());

        let mut events = vec![self.assembler.usage(response.usage)];
        events.extend(self.assembler.complete());
        events
    }

    /// Completes from the assembled blocks, keeping what a nonconforming
    /// terminal document does carry.
    ///
    /// The id and usage are folded in when present; the finish reason is set
    /// only when the document reports a status, so a shape with none still
    /// completes as `incomplete` rather than a claimed clean stop.
    fn salvage(&mut self, document: &Value) -> Vec<StreamEvent> {
        if let Some(id) = document.get("id").and_then(Value::as_str) {
            self.assembler.set_id(id);
        }
        if document.get("status").and_then(Value::as_str).is_some() {
            let reason = decode_finish_reason(document, &[]);
            let reason = if reason == FinishReason::Stop && self.assembler.has_tool_call() {
                FinishReason::ToolCall
            } else {
                reason
            };
            self.assembler.set_finish_reason(reason);
        }
        self.assembler.set_raw(document.clone());

        let mut events = Vec::new();
        if let Some(usage) = document.get("usage").filter(|usage| usage.is_object()) {
            events.push(self.assembler.usage(decode_usage(Some(usage))));
        }
        events.extend(self.assembler.complete());
        events
    }
}

/// Whether an output item is a tool call the caller cannot answer.
///
/// A `function_call` with no name is model-internal. It has no tool to route
/// to and no result to send back, so it neither becomes content nor turns the
/// finish reason into [`FinishReason::ToolCall`].
fn is_internal_call(item: &Value) -> bool {
    let is_call = matches!(
        item.get("type").and_then(Value::as_str),
        Some("function_call" | "custom_tool_call")
    );
    is_call
        && item
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .is_empty()
}

/// The output item an item event carries.
fn item(value: &Value) -> &Value {
    value.get("item").unwrap_or(&Value::Null)
}

/// The text fragment a delta event carries.
fn delta_text(value: &Value) -> String {
    value
        .get("delta")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

/// The stable block id for one stream event.
///
/// Delta events name their item directly. Item events carry the item instead,
/// whose own id is the same value. Only an event with neither falls back to the
/// output index.
fn block_id(value: &Value) -> ContentBlockId {
    let named = value
        .get("item_id")
        .and_then(Value::as_str)
        .or_else(|| value.pointer("/item/id").and_then(Value::as_str));
    match named {
        Some(id) => ContentBlockId::new(id),
        None => ContentBlockId::new(format!(
            "block-{}",
            value
                .get("output_index")
                .and_then(Value::as_u64)
                .unwrap_or(0)
        )),
    }
}

#[cfg(all(test, feature = "builtin-catalog"))]
mod tests {
    use std::error::Error as StdError;

    use serde_json::{Value, json};

    use super::{COUNT_TOKENS_FIELDS, Codec, MESSAGE_KIND, OpenAiResponsesCodec, REASONING_KIND};
    use crate::adapter::ResolvedCall;
    use crate::codecs::test_support::{resolved, resolved_in};
    use crate::transport::SseEvent;
    use crate::types::{
        ContentBlockId, ContentPart, ErrorKind, FinishReason, ImageContent, MediaSource, Message,
        ReasoningContent, Request, Response, RetryClassification, Role, Speed, StreamEvent,
        ToolCall, ToolCallKind, ToolDefinition, ToolResult,
    };

    const MODEL: &str = "openai/gpt-5.6-luna";

    /// The public deployment's codec.
    fn codec() -> OpenAiResponsesCodec {
        OpenAiResponsesCodec::new(false)
    }

    fn call(request: Request) -> Result<ResolvedCall, Box<dyn StdError>> {
        resolved(request)
    }

    fn sse(data: &Value) -> SseEvent {
        SseEvent {
            event: None,
            data:  data.to_string(),
        }
    }

    /// Every content part carried by a block-end event, in order.
    fn ended_parts(events: &[StreamEvent]) -> Vec<ContentPart> {
        events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::ContentBlockEnd { part, .. } => Some(part.clone()),
                _ => None,
            })
            .collect()
    }

    /// The single completed response of a stream.
    fn completed(events: &[StreamEvent]) -> Result<Response, Box<dyn StdError>> {
        let mut responses = events.iter().filter_map(|event| match event {
            StreamEvent::Completed { response } => Some(response.clone()),
            _ => None,
        });
        let response = responses
            .next()
            .ok_or("the stream emitted no completed event")?;
        if responses.next().is_some() {
            return Err("the stream emitted more than one completed event".into());
        }
        Ok(response)
    }

    /// Checks one start, then deltas, then one end for every block.
    fn assert_block_boundaries(events: &[StreamEvent]) -> Result<(), Box<dyn StdError>> {
        let mut open: Vec<&ContentBlockId> = Vec::new();
        let mut closed: Vec<&ContentBlockId> = Vec::new();

        for event in events {
            match event {
                StreamEvent::ContentBlockStart { id, .. } => {
                    if open.contains(&id) || closed.contains(&id) {
                        return Err(format!("{id:?} started twice").into());
                    }
                    open.push(id);
                }
                StreamEvent::TextDelta { id, .. }
                | StreamEvent::ReasoningDelta { id, .. }
                | StreamEvent::ToolCallDelta { id, .. } => {
                    if !open.contains(&id) {
                        return Err(format!("{id:?} sent a delta before its start").into());
                    }
                }
                StreamEvent::ContentBlockEnd { id, .. } => {
                    if !open.contains(&id) {
                        return Err(format!("{id:?} ended without a start").into());
                    }
                    open.retain(|open_id| *open_id != id);
                    closed.push(id);
                }
                StreamEvent::Started { .. }
                | StreamEvent::Usage { .. }
                | StreamEvent::RateLimits { .. }
                | StreamEvent::Completed { .. } => {}
            }
        }

        if !open.is_empty() {
            return Err(format!("blocks left open: {open:?}").into());
        }
        Ok(())
    }

    #[test]
    fn tool_calls_and_results_keep_their_protocol_identity() -> Result<(), Box<dyn StdError>> {
        let mut call_part = ToolCall::function("call_abc", "search", json!({ "query": "rust" }));
        call_part.raw_arguments = Some("{\"query\":\"rust\"}".to_owned());
        call_part
            .provider_metadata
            .insert("openai".to_owned(), json!({ "item_id": "fc_123" }));
        let request = Request::builder()
            .model(MODEL)
            .user("Search for rust")
            .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
                call_part,
            )]))
            .message(Message::new(Role::Tool, [ContentPart::ToolResult(
                ToolResult {
                    tool_call_id: "call_abc".to_owned(),
                    name:         Some("search".to_owned()),
                    content:      vec![ContentPart::Text {
                        text: "2 matches".to_owned(),
                    }],
                    is_error:     false,
                },
            )]))
            .build()?;
        let codec = codec();

        let encoded = codec.encode(&call(request)?, false)?;

        assert_eq!(
            encoded.body["input"][1],
            json!({
                "type": "function_call",
                "id": "fc_123",
                "call_id": "call_abc",
                "name": "search",
                "arguments": "{\"query\":\"rust\"}",
            })
        );
        assert_eq!(
            encoded.body["input"][2],
            json!({
                "type": "function_call_output",
                "call_id": "call_abc",
                "output": "2 matches",
            })
        );

        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();
        let response = codec.decode_response(
            &route,
            json!({
                "id": "resp_1",
                "status": "completed",
                "output": [{
                    "type": "function_call",
                    "id": "fc_123",
                    "call_id": "call_abc",
                    "name": "search",
                    "arguments": "{\"query\":\"rust\"}",
                }],
            }),
        )?;

        let [ContentPart::ToolCall(decoded)] = response.content.as_slice() else {
            return Err("expected one tool call".into());
        };
        assert_eq!(decoded.id, "call_abc");
        assert_eq!(decoded.arguments, json!({ "query": "rust" }));
        assert_eq!(
            decoded.raw_arguments.as_deref(),
            Some("{\"query\":\"rust\"}")
        );
        assert_eq!(
            decoded.provider_metadata.get("openai"),
            Some(&json!({ "item_id": "fc_123" }))
        );
        Ok(())
    }

    #[test]
    fn raw_options_win_and_foreign_namespaces_are_ignored() -> Result<(), Box<dyn StdError>> {
        let request = Request::builder()
            .model(MODEL)
            .user("Hello")
            .temperature(0.2)
            .provider_option("openai", "temperature", json!(0.9))
            .provider_option("openai", "auto_cache", json!(false))
            .provider_option("anthropic", "top_k", json!(40))
            .build()?;

        let encoded = codec().encode(&call(request)?, false)?;

        assert_eq!(encoded.body["temperature"], json!(0.9));
        let body = encoded.body.as_object().ok_or("expected a JSON object")?;
        assert!(!body.contains_key("auto_cache"));
        assert!(!body.contains_key("top_k"));
        Ok(())
    }

    #[test]
    fn stop_sequences_are_dropped_with_a_warning() -> Result<(), Box<dyn StdError>> {
        let request = Request::builder()
            .model(MODEL)
            .user("Hello")
            .stop_sequences(["END", "STOP"])
            .metadata_entry("trace_id", "t789")
            .build()?;

        let encoded = codec().encode(&call(request)?, false)?;

        // The live API rejects a `stop` member outright (2026-08-30), so the
        // sequences never reach the wire and the caller is warned instead.
        let body = encoded.body.as_object().ok_or("expected a JSON object")?;
        assert!(!body.contains_key("stop"));
        assert_eq!(encoded.body["metadata"], json!({ "trace_id": "t789" }));
        assert_eq!(
            encoded.warnings.len(),
            1,
            "dropping the stop sequences must warn: {:?}",
            encoded.warnings,
        );
        Ok(())
    }

    #[test]
    fn each_speed_selects_its_service_tier() -> Result<(), Box<dyn StdError>> {
        for (speed, tier) in [
            (Speed::Fast, "priority"),
            (Speed::Balanced, "auto"),
            (Speed::Economical, "flex"),
        ] {
            let request = Request::builder()
                .model(MODEL)
                .user("Hello")
                .speed(speed)
                .build()?;

            let encoded = codec().encode(&call(request)?, false)?;

            assert_eq!(encoded.body["service_tier"], json!(tier));
        }
        Ok(())
    }

    #[test]
    fn codex_mode_hoists_instructions_and_drops_sampling_controls() -> Result<(), Box<dyn StdError>>
    {
        let request = Request::builder()
            .model(MODEL)
            .system("Be brief")
            .user("Hello")
            .temperature(0.2)
            .top_p(0.5)
            .max_output_tokens(256)
            .build()?;

        let encoded = OpenAiResponsesCodec::new(true).encode(&call(request)?, true)?;

        assert_eq!(encoded.body["instructions"], json!("Be brief"));
        assert_eq!(
            encoded.body["input"],
            json!([{
                "role": "user",
                "content": [{ "type": "input_text", "text": "Hello" }],
            }])
        );
        assert_eq!(encoded.body["stream"], json!(true));
        let body = encoded.body.as_object().ok_or("expected a JSON object")?;
        assert!(!body.contains_key("temperature"));
        assert!(!body.contains_key("top_p"));
        assert!(!body.contains_key("max_output_tokens"));
        assert_eq!(encoded.warnings.len(), 3);
        // The deployment hangs its endpoint off the base path with no
        // version segment; `/v1/responses` there is an HTML 403.
        assert!(
            encoded.url.ends_with("/responses") && !encoded.url.contains("/v1/"),
            "codex mode must post to the unversioned path: {}",
            encoded.url
        );
        Ok(())
    }

    #[test]
    fn codex_mode_reports_no_native_token_count() -> Result<(), Box<dyn StdError>> {
        let request = Request::builder().model(MODEL).user("Hello").build()?;
        // The Codex deployment serves no `/responses/input_tokens` (an HTML
        // 403 on 2026-08-30), so codex mode must say "no native count"
        // rather than build a request that cannot succeed.
        assert!(
            OpenAiResponsesCodec::new(true)
                .encode_count_tokens(&call(request)?)
                .is_none()
        );
        Ok(())
    }

    #[test]
    fn inclusive_usage_becomes_disjoint_buckets() -> Result<(), Box<dyn StdError>> {
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();

        let response = codec().decode_response(
            &route,
            json!({
                "id": "resp_1",
                "status": "completed",
                "output": [],
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "input_tokens_details": { "cached_tokens": 80, "cache_write_tokens": 15 },
                    "output_tokens_details": { "reasoning_tokens": 20 },
                },
            }),
        )?;

        assert_eq!(response.usage.input, 5);
        assert_eq!(response.usage.output, 30);
        assert_eq!(response.usage.reasoning, 20);
        assert_eq!(response.usage.cache_read, 80);
        assert_eq!(response.usage.cache_write, 15);
        assert_eq!(response.usage.total(), 150);
        assert_eq!(response.cost, None);
        Ok(())
    }

    #[test]
    fn encrypted_reasoning_is_requested_without_the_catalog_flag() -> Result<(), Box<dyn StdError>>
    {
        // An overlay entry that forgets `reasoning = true` on a reasoning
        // model must not silently lose the encrypted content its next turn
        // needs; a model without reasoning ignores the include.
        let call = resolved_in(
            r#"
            schema_version = 1

            [providers.openai]
            display_name = "OpenAI"
            adapter = "openai"
            codec = "openai-responses"
            base_url = "https://api.openai.com"
            default_model = "overlay"
            auth = { type = "bearer" }

            [providers.openai.models.overlay]
            display_name = "Overlay"
            api_model = "overlay-v1"
            capabilities = { text = true }
            "#,
            Request::builder()
                .model("openai/overlay")
                .user("hi")
                .build()?,
        )?;

        let encoded = codec().encode(&call, false)?;

        assert_eq!(
            encoded.body["include"],
            json!(["reasoning.encrypted_content"])
        );
        Ok(())
    }

    #[test]
    fn a_routing_model_sends_the_cache_fingerprint() -> Result<(), Box<dyn StdError>> {
        let call = resolved_in(
            r#"
            schema_version = 1

            [providers.openai]
            display_name = "OpenAI"
            adapter = "openai"
            codec = "openai-responses"
            base_url = "https://api.openai.com"
            default_model = "routed"
            auth = { type = "bearer" }

            [providers.openai.models.routed]
            display_name = "Routed"
            api_model = "routed-v1"
            capabilities = { text = true, caching = true, cache_routing = true }
            "#,
            Request::builder()
                .model("openai/routed")
                .system("Keep it short.")
                .user("hi")
                .build()?,
        )?;

        let encoded = codec().encode(&call, false)?;

        let key = encoded.body["prompt_cache_key"]
            .as_str()
            .ok_or("no prompt_cache_key was sent")?;
        assert!(key.starts_with("lithos-"), "unexpected key shape: {key}");
        Ok(())
    }

    #[test]
    fn a_summary_only_reasoning_item_is_kept_for_replay() -> Result<(), Box<dyn StdError>> {
        // Function calls must replay behind their reasoning item, and only
        // the original item with its id satisfies the pairing — even when it
        // carries a visible summary and no encrypted payload.
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();
        let item = json!({
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{ "type": "summary_text", "text": "checked" }],
        });

        let response = codec().decode_response(
            &route,
            json!({ "id": "resp_1", "status": "completed", "output": [item.clone()] }),
        )?;

        assert_eq!(response.content, vec![
            ContentPart::Reasoning(ReasoningContent {
                text:             "checked".to_owned(),
                signature:        None,
                signature_origin: None,
                redacted:         false,
            }),
            ContentPart::opaque(REASONING_KIND, item),
        ]);
        Ok(())
    }

    #[test]
    fn a_body_without_an_output_array_fails_to_decode() -> Result<(), Box<dyn StdError>> {
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();

        // An empty object and an error-shaped 200 both lack the `output`
        // array every Responses document carries; neither may decode as a
        // successful empty response.
        for body in [json!({}), json!({ "error": { "message": "boom" } })] {
            let error = codec()
                .decode_response(&route, body)
                .err()
                .ok_or("expected a body without output to fail")?;
            assert_eq!(error.kind(), ErrorKind::ResponseDecode);
            assert_eq!(error.retry_classification(), RetryClassification::Safe);
        }
        Ok(())
    }

    #[test]
    fn an_unknown_response_status_keeps_its_spelling() -> Result<(), Box<dyn StdError>> {
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();

        let response = codec().decode_response(
            &route,
            json!({ "id": "resp_1", "status": "cancelled", "output": [] }),
        )?;

        // The old decoder preserved unknown statuses; collapsing `cancelled`
        // into `Stop` would report an answer the model never finished.
        assert_eq!(
            response.finish_reason,
            FinishReason::Other("cancelled".to_owned())
        );
        Ok(())
    }

    #[test]
    fn the_raw_success_document_is_preserved() -> Result<(), Box<dyn StdError>> {
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();
        let document = json!({
            "id": "resp_1",
            "status": "completed",
            "service_tier": "default",
            "output": [{
                "type": "message",
                "id": "msg_1",
                "content": [{ "type": "output_text", "text": "hello" }],
            }],
        });

        let response = codec().decode_response(&route, document.clone())?;

        assert_eq!(response.raw, Some(document));
        assert_eq!(response.text(), "hello");
        Ok(())
    }

    #[test]
    fn custom_tools_round_trip() -> Result<(), Box<dyn StdError>> {
        let request = Request::builder()
            .model(MODEL)
            .tool(ToolDefinition::custom(
                "apply_patch",
                "Edits files",
                json!({ "type": "grammar", "syntax": "lark" }),
            ))
            .user("Patch the file")
            .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
                ToolCall::custom("call_001", "apply_patch", "*** Begin Patch"),
            )]))
            .message(Message::new(Role::Tool, [ContentPart::ToolResult(
                ToolResult {
                    tool_call_id: "call_001".to_owned(),
                    name:         Some("apply_patch".to_owned()),
                    content:      vec![ContentPart::Text {
                        text: "Success".to_owned(),
                    }],
                    is_error:     false,
                },
            )]))
            .build()?;
        let codec = codec();
        let call = call(request)?;

        let encoded = codec.encode(&call, false)?;

        assert_eq!(
            encoded.body["tools"][0],
            json!({
                "type": "custom",
                "name": "apply_patch",
                "description": "Edits files",
                "format": { "type": "grammar", "syntax": "lark" },
            })
        );
        assert_eq!(
            encoded.body["input"][1],
            json!({
                "type": "custom_tool_call",
                "call_id": "call_001",
                "name": "apply_patch",
                "input": "*** Begin Patch",
            })
        );
        assert_eq!(
            encoded.body["input"][2],
            json!({
                "type": "custom_tool_call_output",
                "call_id": "call_001",
                "output": "Success",
            })
        );

        let response = codec.decode_response(
            call.route(),
            json!({
                "id": "resp_1",
                "status": "completed",
                "output": [{
                    "type": "custom_tool_call",
                    "id": "ctc_def456",
                    "call_id": "call_001",
                    "name": "apply_patch",
                    "input": "*** Begin Patch",
                }],
            }),
        )?;

        let [ContentPart::ToolCall(decoded)] = response.content.as_slice() else {
            return Err("expected one tool call".into());
        };
        assert_eq!(decoded.kind, ToolCallKind::Custom);
        assert_eq!(
            decoded.arguments,
            Value::String("*** Begin Patch".to_owned())
        );
        assert_eq!(decoded.raw_arguments.as_deref(), Some("*** Begin Patch"));
        Ok(())
    }

    #[test]
    fn opaque_reasoning_replays_and_other_namespaces_are_skipped() -> Result<(), Box<dyn StdError>>
    {
        let item = json!({
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
            "encrypted_content": "gAAAA",
        });
        let request = Request::builder()
            .model(MODEL)
            .user("Hello")
            .message(Message::new(Role::Assistant, [
                ContentPart::opaque("openai.reasoning", item.clone()),
                ContentPart::opaque("anthropic.thinking", json!({ "signature": "sig" })),
                ContentPart::Reasoning(ReasoningContent {
                    text:             "step one".to_owned(),
                    signature:        None,
                    signature_origin: None,
                    redacted:         false,
                }),
                ContentPart::Text {
                    text: "hello".to_owned(),
                },
            ]))
            .build()?;

        let encoded = codec().encode(&call(request)?, false)?;

        assert_eq!(encoded.body["input"][1], item);
        assert_eq!(
            encoded.body["input"][2],
            json!({
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "hello" }],
            })
        );
        assert_eq!(
            encoded.body["input"].as_array().map(Vec::len),
            Some(3),
            "an opaque part for another provider must not reach the wire",
        );
        Ok(())
    }

    #[test]
    fn legacy_persisted_replay_data_still_replays() -> Result<(), Box<dyn StdError>> {
        // The reference implementation persisted whole output items under the
        // underscore kinds and the `fc_` item id at the metadata top level. A
        // migrated history must keep its reasoning chain and item identity —
        // dropping the reasoning item draws the provider's 400 about a
        // function_call without its required reasoning item.
        let reasoning = json!({
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
            "encrypted_content": "gAAAA",
        });
        let message = json!({
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [{ "type": "output_text", "text": "checking" }],
        });
        let mut tool_call = ToolCall::function("call_abc", "search", json!({ "query": "rust" }));
        tool_call
            .provider_metadata
            .insert("id".to_owned(), json!("fc_123"));
        let request = Request::builder()
            .model(MODEL)
            .user("Hello")
            .message(Message::new(Role::Assistant, [
                ContentPart::opaque("openai_reasoning", reasoning.clone()),
                ContentPart::opaque("openai_message", message.clone()),
                ContentPart::Text {
                    text: "checking".to_owned(),
                },
                ContentPart::ToolCall(tool_call),
            ]))
            .build()?;

        let encoded = codec().encode(&call(request)?, false)?;

        assert_eq!(encoded.body["input"][1], reasoning);
        assert_eq!(encoded.body["input"][2], message);
        assert_eq!(
            encoded.body["input"][3]["id"], "fc_123",
            "the un-namespaced item id must ride the replayed call"
        );
        assert_eq!(
            encoded.body["input"].as_array().map(Vec::len),
            Some(4),
            "the preserved message item already carries the text",
        );
        Ok(())
    }

    #[test]
    fn a_reasoning_item_decodes_to_visible_text_and_a_replay_part() -> Result<(), Box<dyn StdError>>
    {
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();

        let response = codec().decode_response(
            &route,
            json!({
                "id": "resp_1",
                "status": "completed",
                "output": [{
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{ "type": "summary_text", "text": "checked" }],
                    "encrypted_content": "gAAAA",
                }],
            }),
        )?;

        let [ContentPart::Reasoning(reasoning), opaque] = response.content.as_slice() else {
            return Err("expected reasoning text and its replay part".into());
        };
        assert_eq!(reasoning.text, "checked");
        assert_eq!(opaque.opaque_namespace(), Some("openai"));
        Ok(())
    }

    #[test]
    fn a_lost_item_announcement_recovers_the_call_identity() -> Result<(), Box<dyn StdError>> {
        // The argument fragments arrive before any `output_item.added`, so
        // they latch the assembler's fallback block — call id equal to the
        // item id, no name. The terminal item event carries the real
        // identity, and the assembled part must not keep the fallback's.
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();
        let mut decoder = codec().stream_decoder(&route);
        let transcript = vec![
            json!({ "type": "response.created", "response": { "id": "resp_1" } }),
            json!({
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_123",
                "delta": "{\"query\":",
            }),
            json!({
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_123",
                "delta": "\"rust\"}",
            }),
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_123",
                    "call_id": "call_abc",
                    "name": "search",
                    "arguments": "{\"query\":\"rust\"}",
                },
            }),
        ];

        let mut events = Vec::new();
        for event in transcript {
            events.extend(decoder.decode(sse(&event))?);
        }
        events.extend(decoder.finish()?);

        let parts = ended_parts(&events);
        let [ContentPart::ToolCall(recovered)] = parts.as_slice() else {
            return Err(format!("expected one tool call part, got {parts:?}").into());
        };
        assert_eq!(recovered.id, "call_abc");
        assert_eq!(recovered.name, "search");
        assert_eq!(recovered.arguments, json!({ "query": "rust" }));
        Ok(())
    }

    #[test]
    fn a_lost_added_for_an_internal_call_leaves_no_phantom_part() -> Result<(), Box<dyn StdError>> {
        // Argument deltas for an unannounced item latch the fallback block,
        // and the terminal item then reveals a model-internal call with no
        // name. Blocking decode drops the item, and the stream must match:
        // the latched block closes for consumers that saw it open, but no
        // nameless ToolCall joins the content and the finish reason stays
        // the document's.
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();
        let mut decoder = codec().stream_decoder(&route);
        let transcript = vec![
            json!({ "type": "response.created", "response": { "id": "resp_1" } }),
            json!({
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_9",
                "delta": "{\"q\":1}",
            }),
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": { "type": "function_call", "id": "fc_9", "arguments": "{\"q\":1}" },
            }),
            json!({
                "type": "response.completed",
                "response": { "id": "resp_1", "status": "completed", "output": [] },
            }),
        ];

        let mut events = Vec::new();
        for event in transcript {
            events.extend(decoder.decode(sse(&event))?);
        }
        events.extend(decoder.finish()?);

        assert_block_boundaries(&events)?;
        let response = completed(&events)?;
        assert_eq!(response.content, Vec::new());
        assert_eq!(response.finish_reason, FinishReason::Stop);
        Ok(())
    }

    #[test]
    fn a_lost_argument_delta_is_healed_by_the_terminal_item() -> Result<(), Box<dyn StdError>> {
        // One argument fragment never arrives. The terminal item carries the
        // complete arguments, so the assembled call must not keep the
        // truncation — and the missing tail goes out as an ordinary delta so
        // a consumer concatenating fragments stays correct too.
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();
        let mut decoder = codec().stream_decoder(&route);
        let transcript = vec![
            json!({ "type": "response.created", "response": { "id": "resp_1" } }),
            json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_123",
                    "call_id": "call_abc",
                    "name": "search",
                },
            }),
            json!({
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_123",
                "delta": "{\"query\":",
            }),
            // The fragment carrying "\"rust\"}" is lost in transit.
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_123",
                    "call_id": "call_abc",
                    "name": "search",
                    "arguments": "{\"query\":\"rust\"}",
                },
            }),
        ];

        let mut events = Vec::new();
        for event in transcript {
            events.extend(decoder.decode(sse(&event))?);
        }
        events.extend(decoder.finish()?);

        let tails: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::ToolCallDelta { arguments, .. } => Some(arguments.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(tails, ["{\"query\":", "\"rust\"}"]);
        let parts = ended_parts(&events);
        let [ContentPart::ToolCall(healed)] = parts.as_slice() else {
            return Err(format!("expected one tool call part, got {parts:?}").into());
        };
        assert_eq!(healed.arguments, json!({ "query": "rust" }));
        assert_eq!(
            healed.raw_arguments.as_deref(),
            Some("{\"query\":\"rust\"}")
        );
        Ok(())
    }

    #[test]
    fn a_garbled_argument_buffer_is_replaced_by_the_terminal_item() -> Result<(), Box<dyn StdError>>
    {
        // The streamed fragments disagree with the terminal item — not a
        // prefix, so something was mangled in transit. The terminal item is
        // the ground truth and replaces the buffer outright.
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();
        let mut decoder = codec().stream_decoder(&route);
        let transcript = vec![
            json!({ "type": "response.created", "response": { "id": "resp_1" } }),
            json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_123",
                    "call_id": "call_abc",
                    "name": "search",
                },
            }),
            json!({
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_123",
                "delta": "\"rust\"}",
            }),
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_123",
                    "call_id": "call_abc",
                    "name": "search",
                    "arguments": "{\"query\":\"rust\"}",
                },
            }),
        ];

        let mut events = Vec::new();
        for event in transcript {
            events.extend(decoder.decode(sse(&event))?);
        }
        events.extend(decoder.finish()?);

        let parts = ended_parts(&events);
        let [ContentPart::ToolCall(replaced)] = parts.as_slice() else {
            return Err(format!("expected one tool call part, got {parts:?}").into());
        };
        assert_eq!(replaced.arguments, json!({ "query": "rust" }));
        assert_eq!(
            replaced.raw_arguments.as_deref(),
            Some("{\"query\":\"rust\"}")
        );
        Ok(())
    }

    #[test]
    fn a_text_only_tool_message_answering_a_custom_call_routes_as_custom()
    -> Result<(), Box<dyn StdError>> {
        // The custom call rides earlier in the request; the answering tool
        // message carries only text and a call id — no name, and the tool is
        // not redeclared. A function_call_output against a custom_tool_call
        // is rejected by the provider, so the call-id route must apply here
        // exactly as it does on the ToolResult path.
        let request = Request::builder()
            .model(MODEL)
            .user("Patch the file")
            .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
                ToolCall::custom("call_001", "apply_patch", "*** Begin Patch"),
            )]))
            .message(
                Message::new(Role::Tool, [ContentPart::Text {
                    text: "Success".to_owned(),
                }])
                .with_tool_call_id("call_001"),
            )
            .build()?;

        let encoded = codec().encode(&call(request)?, false)?;

        assert_eq!(
            encoded.body["input"][2],
            json!({
                "type": "custom_tool_call_output",
                "call_id": "call_001",
                "output": "Success",
            })
        );
        Ok(())
    }

    #[test]
    fn a_streamed_message_without_text_emits_no_empty_text_part() -> Result<(), Box<dyn StdError>> {
        // An empty assistant message streams no output_text. Blocking decode
        // of the same body pushes no text part, and the streamed response
        // must match: only the opaque replay item, no Text("") and no stray
        // start/end pair.
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();
        let mut decoder = codec().stream_decoder(&route);
        let item = json!({
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [],
        });
        let transcript = vec![
            json!({ "type": "response.created", "response": { "id": "resp_1" } }),
            json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": { "type": "message", "id": "msg_1", "role": "assistant", "content": [] },
            }),
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": item,
            }),
        ];

        let mut events = Vec::new();
        for event in transcript {
            events.extend(decoder.decode(sse(&event))?);
        }
        events.extend(decoder.finish()?);

        assert_block_boundaries(&events)?;
        let response = completed(&events)?;
        let [ContentPart::Opaque { kind, data }] = response.content.as_slice() else {
            return Err(
                format!("expected only the opaque item, got {:?}", response.content).into(),
            );
        };
        assert_eq!(kind, MESSAGE_KIND);
        assert_eq!(data, &item);
        Ok(())
    }

    #[test]
    fn a_refusal_part_fails_instead_of_decoding() -> Result<(), Box<dyn StdError>> {
        // The structured-output refusal channel: a `refusal` content part in
        // the message item. Decoding it as an empty success would hide the
        // refusal from the caller and from failover — the contract H8 set
        // for Anthropic and Bedrock, and R2-29 extended to Chat, extends
        // here.
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();
        let body = json!({
            "id": "resp_1",
            "status": "completed",
            "output": [{
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "content": [{ "type": "refusal", "refusal": "I can't help with that." }],
            }],
        });

        let error = codec()
            .decode_response(&route, body.clone())
            .expect_err("a refusal must fail the call");

        assert_eq!(error.kind(), ErrorKind::ContentFilter);
        assert_eq!(error.provider_code(), Some("refusal"));
        assert!(
            error.to_string().contains("I can't help with that."),
            "{error}"
        );
        assert_eq!(error.raw_data(), Some(&body));
        Ok(())
    }

    #[test]
    fn a_streamed_refusal_fails_at_the_terminal_event() -> Result<(), Box<dyn StdError>> {
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();
        let mut decoder = codec().stream_decoder(&route);
        let item = json!({
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [{ "type": "refusal", "refusal": "I can't help with that." }],
        });
        decoder.decode(sse(
            &json!({ "type": "response.created", "response": { "id": "resp_1" } }),
        ))?;
        decoder.decode(sse(&json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": { "type": "message", "id": "msg_1", "role": "assistant", "content": [] },
        })))?;
        decoder.decode(sse(&json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": item,
        })))?;

        let error = decoder
            .decode(sse(&json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "output": [item],
                },
            })))
            .expect_err("a refusal must fail the stream");

        assert_eq!(error.kind(), ErrorKind::ContentFilter);
        assert_eq!(error.provider_code(), Some("refusal"));
        assert!(
            error.to_string().contains("I can't help with that."),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn a_terminal_document_without_output_completes_from_the_streamed_blocks()
    -> Result<(), Box<dyn StdError>> {
        // A middlebox that strips the terminal document below the decodable
        // shape must not fail a stream whose answer already streamed. The id,
        // usage, and status it does carry are salvaged.
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();
        let mut decoder = codec().stream_decoder(&route);
        let transcript = vec![
            json!({ "type": "response.created", "response": { "id": "resp_1" } }),
            json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": { "type": "message", "id": "msg_1", "role": "assistant", "content": [] },
            }),
            json!({ "type": "response.output_text.delta", "item_id": "msg_1", "delta": "Hello" }),
            json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "usage": { "input_tokens": 7, "output_tokens": 2 },
                },
            }),
        ];

        let mut events = Vec::new();
        for event in transcript {
            events.extend(decoder.decode(sse(&event))?);
        }
        events.extend(decoder.finish()?);

        let response = completed(&events)?;
        assert_eq!(response.id.as_deref(), Some("resp_1"));
        assert_eq!(response.finish_reason, FinishReason::Stop);
        assert_eq!(response.usage.input, 7);
        assert_eq!(response.usage.output, 2);
        let [ContentPart::Text { text }] = response.content.as_slice() else {
            return Err(format!("expected the streamed text, got {:?}", response.content).into());
        };
        assert_eq!(text, "Hello");
        Ok(())
    }

    #[test]
    fn a_flattened_terminal_event_completes_from_its_own_fields() -> Result<(), Box<dyn StdError>> {
        // A gateway may flatten the terminal document into the event itself.
        // The old client read either shape; the fields are salvaged from the
        // event object instead of completing empty-handed.
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();
        let mut decoder = codec().stream_decoder(&route);
        let transcript = vec![
            json!({ "type": "response.created", "response": { "id": "resp_1" } }),
            json!({ "type": "response.output_text.delta", "item_id": "msg_1", "delta": "Hi" }),
            json!({
                "type": "response.completed",
                "id": "resp_1",
                "status": "completed",
                "usage": { "input_tokens": 3, "output_tokens": 1 },
            }),
        ];

        let mut events = Vec::new();
        for event in transcript {
            events.extend(decoder.decode(sse(&event))?);
        }
        events.extend(decoder.finish()?);

        let response = completed(&events)?;
        assert_eq!(response.id.as_deref(), Some("resp_1"));
        assert_eq!(response.finish_reason, FinishReason::Stop);
        assert_eq!(response.usage.input, 3);
        Ok(())
    }

    #[test]
    fn a_stream_transcript_produces_one_block_per_item() -> Result<(), Box<dyn StdError>> {
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();
        let mut decoder = codec().stream_decoder(&route);
        let transcript = vec![
            json!({ "type": "response.created", "response": { "id": "resp_stream" } }),
            json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": { "type": "message", "id": "msg_1", "role": "assistant", "content": [] },
            }),
            json!({ "type": "response.output_text.delta", "item_id": "msg_1", "delta": "Hel" }),
            json!({ "type": "response.output_text.delta", "item_id": "msg_1", "delta": "lo" }),
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "msg_1",
                    "content": [{ "type": "output_text", "text": "Hello" }],
                },
            }),
            json!({
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {
                    "type": "function_call",
                    "id": "fc_123",
                    "call_id": "call_abc",
                    "name": "search",
                },
            }),
            json!({
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_123",
                "delta": "{\"qu",
            }),
            json!({
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_123",
                "delta": "ery\":\"rust\"}",
            }),
            json!({
                "type": "response.output_item.done",
                "output_index": 1,
                "item": {
                    "type": "function_call",
                    "id": "fc_123",
                    "call_id": "call_abc",
                    "name": "search",
                    "arguments": "{\"query\":\"rust\"}",
                },
            }),
            json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_stream",
                    "status": "completed",
                    "output": [],
                    "usage": {
                        "input_tokens": 11,
                        "output_tokens": 5,
                        "input_tokens_details": { "cached_tokens": 2 },
                        "output_tokens_details": { "reasoning_tokens": 1 },
                    },
                },
            }),
        ];

        let mut events = Vec::new();
        for event in transcript {
            events.extend(decoder.decode(sse(&event))?);
        }
        events.extend(decoder.finish()?);

        assert_block_boundaries(&events)?;
        let starts = events
            .iter()
            .filter(|event| matches!(event, StreamEvent::ContentBlockStart { .. }))
            .count();
        // The message contributes two blocks: the visible text and the whole
        // item, kept for replay.
        assert_eq!(starts, 3);
        assert!(matches!(
            events.first(),
            Some(StreamEvent::Started { id: Some(id) }) if id == "resp_stream"
        ));

        let parts = ended_parts(&events);
        assert_eq!(parts[0], ContentPart::Text {
            text: "Hello".to_owned(),
        });
        assert_eq!(
            parts[1],
            ContentPart::opaque(
                MESSAGE_KIND,
                json!({
                    "type": "message",
                    "id": "msg_1",
                    "content": [{ "type": "output_text", "text": "Hello" }],
                })
            )
        );
        let ContentPart::ToolCall(streamed) = &parts[2] else {
            return Err("expected a streamed tool call".into());
        };
        assert_eq!(streamed.id, "call_abc");
        assert_eq!(streamed.name, "search");
        assert_eq!(streamed.arguments, json!({ "query": "rust" }));
        assert_eq!(
            streamed.provider_metadata.get("openai"),
            Some(&json!({ "item_id": "fc_123" }))
        );

        let response = completed(&events)?;
        assert_eq!(response.content, parts);
        assert_eq!(response.id.as_deref(), Some("resp_stream"));
        assert_eq!(response.usage.input, 9);
        assert_eq!(response.usage.cache_read, 2);
        assert_eq!(response.usage.reasoning, 1);
        assert!(response.raw.is_some());
        // The terminal document's `output` is empty — trimmed by a middlebox,
        // say — but the stream plainly delivered a tool call, and the
        // assembled blocks decide.
        assert_eq!(response.finish_reason, FinishReason::ToolCall);
        Ok(())
    }

    #[test]
    fn a_streamed_reasoning_item_keeps_its_text_and_its_replay_part()
    -> Result<(), Box<dyn StdError>> {
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();
        let mut decoder = codec().stream_decoder(&route);
        let item = json!({
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{ "type": "summary_text", "text": "checked" }],
            "encrypted_content": "gAAAA",
        });

        let mut events = decoder.decode(sse(&json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": { "type": "reasoning", "id": "rs_1", "summary": [] },
        })))?;
        events.extend(decoder.decode(sse(&json!({
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs_1",
            "delta": "checked",
        })))?);
        events.extend(decoder.decode(sse(&json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": item,
        })))?);
        events.extend(decoder.finish()?);

        assert_block_boundaries(&events)?;
        let parts = ended_parts(&events);
        let [
            ContentPart::Reasoning(reasoning),
            ContentPart::Opaque { kind, data },
        ] = parts.as_slice()
        else {
            return Err("expected reasoning text and its replay part".into());
        };
        assert_eq!(reasoning.text, "checked");
        assert_eq!(kind, "openai.reasoning");
        assert_eq!(data, &item);
        assert_eq!(completed(&events)?.content, parts);
        Ok(())
    }

    #[test]
    fn a_stream_error_ends_the_stream_without_completing() -> Result<(), Box<dyn StdError>> {
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();
        let mut decoder = codec().stream_decoder(&route);

        decoder.decode(sse(
            &json!({ "type": "response.created", "response": { "id": "resp_1" } }),
        ))?;
        let error = decoder
            .decode(sse(&json!({
                "type": "error",
                "error": { "type": "server_error", "message": "upstream failed" },
            })))
            .err()
            .ok_or("expected a stream error")?;

        assert!(error.message().contains("upstream failed"));
        Ok(())
    }

    #[test]
    fn a_failed_event_without_a_response_wrapper_keeps_its_detail() -> Result<(), Box<dyn StdError>>
    {
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();
        let mut decoder = codec().stream_decoder(&route);

        let error = decoder
            .decode(sse(&json!({
                "type": "response.failed",
                "error": { "code": "rate_limit_exceeded", "message": "Rate limit reached" },
            })))
            .err()
            .ok_or("expected a stream error")?;

        assert!(error.message().contains("Rate limit reached"));
        assert_eq!(error.provider_code(), Some("rate_limit_exceeded"));
        assert!(error.raw_data().is_some());
        Ok(())
    }

    #[test]
    fn an_item_without_deltas_still_produces_its_content() -> Result<(), Box<dyn StdError>> {
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();
        let mut decoder = codec().stream_decoder(&route);

        let mut events = decoder.decode(sse(&json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "custom_tool_call",
                "id": "ctc_1",
                "call_id": "call_001",
                "name": "apply_patch",
                "input": "*** Begin Patch",
            },
        })))?;
        events.extend(decoder.finish()?);

        assert_block_boundaries(&events)?;
        let parts = ended_parts(&events);
        let [ContentPart::ToolCall(recovered)] = parts.as_slice() else {
            return Err("expected one recovered tool call".into());
        };
        assert_eq!(recovered.kind, ToolCallKind::Custom);
        assert_eq!(
            recovered.arguments,
            Value::String("*** Begin Patch".to_owned())
        );
        Ok(())
    }

    #[test]
    fn the_count_tokens_body_keeps_only_allowlisted_fields() -> Result<(), Box<dyn StdError>> {
        let request = Request::builder()
            .model(MODEL)
            .user("Hello")
            .temperature(0.2)
            .max_output_tokens(256)
            .stop_sequence("END")
            .metadata_entry("trace_id", "t789")
            .provider_option("openai", "seed", json!(7))
            .build()?;
        let call = call(request)?;

        let encoded = codec()
            .encode_count_tokens(&call)
            .ok_or("expected a token count request")??;

        assert!(encoded.url.ends_with("/v1/responses/input_tokens"));
        let body = encoded.body.as_object().ok_or("expected a JSON object")?;
        for key in body.keys() {
            assert!(
                COUNT_TOKENS_FIELDS.contains(&key.as_str()),
                "{key} is not accepted by the count endpoint",
            );
        }
        assert!(!body.contains_key("temperature"));
        assert!(!body.contains_key("max_output_tokens"));
        assert!(!body.contains_key("stop"));
        assert!(!body.contains_key("stream"));
        assert!(!body.contains_key("metadata"));
        assert!(!body.contains_key("seed"));
        assert_eq!(body["model"], json!("gpt-5.6-luna"));
        Ok(())
    }

    #[test]
    fn a_token_count_with_the_wrong_object_is_a_decode_error() -> Result<(), Box<dyn StdError>> {
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();
        let codec = codec();

        let tokens = codec.decode_count_tokens(
            &route,
            json!({ "input_tokens": 123, "object": "response.input_tokens" }),
        )?;
        let error = codec
            .decode_count_tokens(&route, json!({ "input_tokens": 123, "object": "response" }))
            .err()
            .ok_or("expected a decode error")?;

        assert_eq!(tokens, 123);
        assert_eq!(error.kind(), ErrorKind::ResponseDecode);
        assert_eq!(error.retry_classification(), RetryClassification::Safe);
        Ok(())
    }

    /// The ported round trip from the reference implementation.
    ///
    /// An assistant turn of reasoning, visible text, and a tool call replays as
    /// three items in that order. The preserved `message` item is sent with its
    /// own `id` and `status`, rather than reconstructed from the text part, so
    /// the reasoning item that precedes it still names an item that follows it.
    #[test]
    fn a_reasoning_message_and_function_call_replay_in_order() -> Result<(), Box<dyn StdError>> {
        let reasoning = json!({
            "type": "reasoning",
            "id": "rs_xyz789",
            "summary": [{ "type": "summary_text", "text": "Let me check." }],
            "encrypted_content": "gAAAA",
        });
        let message = json!({
            "type": "message",
            "id": "msg_abc123",
            "status": "completed",
            "role": "assistant",
            "content": [{ "type": "output_text", "text": "Checking now." }],
        });
        let mut tool_call = ToolCall::function("call_001", "shell", json!({ "cmd": "ls" }));
        tool_call
            .provider_metadata
            .insert("openai".to_owned(), json!({ "item_id": "fc_def456" }));
        let request = Request::builder()
            .model(MODEL)
            .user("List the files")
            .message(Message::new(Role::Assistant, [
                ContentPart::opaque(REASONING_KIND, reasoning.clone()),
                ContentPart::Text {
                    text: "Checking now.".to_owned(),
                },
                ContentPart::opaque(MESSAGE_KIND, message.clone()),
                ContentPart::ToolCall(tool_call),
            ]))
            .build()?;

        let encoded = codec().encode(&call(request)?, false)?;

        let input = encoded.body["input"]
            .as_array()
            .ok_or("expected an input array")?;
        assert_eq!(input.len(), 4, "the user turn plus three replayed items");
        assert_eq!(input[1], reasoning);
        assert_eq!(input[2], message);
        assert_eq!(input[3]["type"], json!("function_call"));
        assert_eq!(input[3]["id"], json!("fc_def456"));
        assert_eq!(input[3]["call_id"], json!("call_001"));
        assert_eq!(
            encoded.body.to_string().matches("Checking now.").count(),
            1,
            "the preserved item already carries the assistant text",
        );
        Ok(())
    }

    /// A history tool call with an empty name encodes to nothing, as the
    /// reference client did — `{"name": ""}` draws a provider 400.
    #[test]
    fn an_empty_name_tool_call_in_history_is_skipped() -> Result<(), Box<dyn StdError>> {
        let request = Request::builder()
            .model(MODEL)
            .user("List the files")
            .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
                ToolCall::function("call_1", "", json!({})),
            )]))
            .build()?;

        let encoded = codec().encode(&call(request)?, false)?;

        let input = encoded.body["input"]
            .as_array()
            .ok_or("expected an input array")?;
        assert_eq!(input.len(), 1, "only the user turn may travel: {input:?}");
        Ok(())
    }

    /// Media in a system message never reaches the wire: the Responses API
    /// accepts only `input_text` inside a system item and 400s on anything
    /// else, where the reference client silently dropped it. The text
    /// travels, the drop is reported.
    #[test]
    fn media_in_a_system_message_is_dropped_and_warned() -> Result<(), Box<dyn StdError>> {
        let request = Request::builder()
            .model(MODEL)
            .message(Message::new(Role::System, [
                ContentPart::Text {
                    text: "Use the style guide.".to_owned(),
                },
                ContentPart::Image(ImageContent::new(MediaSource::url(
                    "https://example.com/guide.png",
                ))),
            ]))
            .user("Hello")
            .build()?;

        let encoded = codec().encode(&call(request)?, false)?;

        let system_content = &encoded.body["input"][0]["content"];
        assert_eq!(
            system_content,
            &json!([{ "type": "input_text", "text": "Use the style guide." }]),
            "only the text may travel inside a system item"
        );
        assert_eq!(
            encoded.warnings.len(),
            1,
            "the dropped image must be reported: {:?}",
            encoded.warnings
        );
        Ok(())
    }

    /// The decoder emits a readable reasoning part BESIDE the opaque item
    /// that replays the same text, so the codec's own round trip must not
    /// warn about dropped reasoning — nothing is lost. Reasoning without an
    /// opaque sibling in its message still warns.
    #[test]
    fn a_reasoning_part_beside_its_opaque_item_does_not_warn() -> Result<(), Box<dyn StdError>> {
        let reasoning = ContentPart::Reasoning(ReasoningContent {
            text:             "Let me check.".to_owned(),
            signature:        None,
            signature_origin: None,
            redacted:         false,
        });
        let item = json!({
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{ "type": "summary_text", "text": "Let me check." }],
        });

        let round_trip = Request::builder()
            .model(MODEL)
            .user("List the files")
            .message(Message::new(Role::Assistant, [
                reasoning.clone(),
                ContentPart::opaque(REASONING_KIND, item),
                ContentPart::Text {
                    text: "Checking now.".to_owned(),
                },
            ]))
            .build()?;
        let encoded = codec().encode(&call(round_trip)?, false)?;
        assert_eq!(encoded.warnings, vec![], "the round trip loses nothing");

        let orphaned = Request::builder()
            .model(MODEL)
            .user("List the files")
            .message(Message::new(Role::Assistant, [reasoning]))
            .build()?;
        let encoded = codec().encode(&call(orphaned)?, false)?;
        assert_eq!(
            encoded.warnings.len(),
            1,
            "reasoning with no opaque sibling is dropped and must warn"
        );
        Ok(())
    }

    /// An interleaved turn keeps every reasoning item beside the item it
    /// anchors, which hoisting the opaque parts to the front would break.
    #[test]
    fn interleaved_replay_items_keep_their_pairing() -> Result<(), Box<dyn StdError>> {
        let first = json!({ "type": "reasoning", "id": "rs_1", "encrypted_content": "a" });
        let second = json!({ "type": "reasoning", "id": "rs_2", "encrypted_content": "b" });
        let request = Request::builder()
            .model(MODEL)
            .user("Do both")
            .message(Message::new(Role::Assistant, [
                ContentPart::opaque(REASONING_KIND, first.clone()),
                ContentPart::ToolCall(ToolCall::function("call_1", "one", json!({}))),
                ContentPart::opaque(REASONING_KIND, second.clone()),
                ContentPart::ToolCall(ToolCall::function("call_2", "two", json!({}))),
            ]))
            .build()?;

        let encoded = codec().encode(&call(request)?, false)?;

        let types: Vec<&Value> = encoded.body["input"]
            .as_array()
            .ok_or("expected an input array")?
            .iter()
            .map(|item| &item["type"])
            .collect();
        // The user turn is a plain `{role, content}` item, which carries no
        // `type` of its own.
        assert_eq!(types, vec![
            &Value::Null,
            &json!("reasoning"),
            &json!("function_call"),
            &json!("reasoning"),
            &json!("function_call"),
        ]);
        assert_eq!(encoded.body["input"][1], first);
        assert_eq!(encoded.body["input"][3], second);
        Ok(())
    }

    /// Without a preserved item, assistant text still builds one.
    #[test]
    fn assistant_text_without_a_replay_item_builds_a_message() -> Result<(), Box<dyn StdError>> {
        let request = Request::builder()
            .model(MODEL)
            .user("Hi")
            .message(Message::new(Role::Assistant, [ContentPart::Text {
                text: "Hello".to_owned(),
            }]))
            .build()?;

        let encoded = codec().encode(&call(request)?, false)?;

        assert_eq!(
            encoded.body["input"][1],
            json!({
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "Hello" }],
            })
        );
        Ok(())
    }

    #[test]
    fn a_message_item_decodes_to_text_and_a_replay_part() -> Result<(), Box<dyn StdError>> {
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();
        let item = json!({
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [{ "type": "output_text", "text": "hello" }],
        });

        let response = codec().decode_response(
            &route,
            json!({ "id": "resp_1", "status": "completed", "output": [item.clone()] }),
        )?;

        assert_eq!(response.content, vec![
            ContentPart::Text {
                text: "hello".to_owned(),
            },
            ContentPart::opaque(MESSAGE_KIND, item),
        ]);
        Ok(())
    }

    /// A `function_call` with no name is model-internal: it is not content, and
    /// it does not make the turn look like a tool call.
    #[test]
    fn an_unnamed_tool_call_is_dropped() -> Result<(), Box<dyn StdError>> {
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();

        let response = codec().decode_response(
            &route,
            json!({
                "id": "resp_1",
                "status": "completed",
                "output": [{
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "",
                    "arguments": "{}",
                }],
            }),
        )?;

        assert_eq!(response.content, Vec::new());
        assert_eq!(response.finish_reason, FinishReason::Stop);
        Ok(())
    }

    #[test]
    fn an_unnamed_streamed_tool_call_is_dropped() -> Result<(), Box<dyn StdError>> {
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();
        let mut decoder = codec().stream_decoder(&route);
        let item = json!({
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "",
            "arguments": "{}",
        });

        let mut events = decoder.decode(sse(&json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": item,
        })))?;
        events.extend(decoder.decode(sse(&json!({
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "delta": "{}",
        })))?);
        events.extend(decoder.decode(sse(&json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": item,
        })))?);
        events.extend(decoder.decode(sse(&json!({
            "type": "response.completed",
            "response": { "id": "resp_1", "status": "completed", "output": [item] },
        })))?);

        assert_block_boundaries(&events)?;
        assert_eq!(ended_parts(&events), Vec::new());
        let response = completed(&events)?;
        assert_eq!(response.content, Vec::new());
        assert_eq!(response.finish_reason, FinishReason::Stop);
        Ok(())
    }

    /// A proxy's keepalive frame is not model output, so it cannot end the
    /// stream.
    #[test]
    fn a_frame_that_is_not_json_is_skipped() -> Result<(), Box<dyn StdError>> {
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();
        let mut decoder = codec().stream_decoder(&route);

        let ignored = decoder.decode(SseEvent {
            event: None,
            data:  "keepalive".to_owned(),
        })?;
        let events = decoder.decode(sse(&json!({
            "type": "response.created",
            "response": { "id": "resp_1" },
        })))?;

        assert_eq!(ignored, Vec::new());
        assert!(matches!(
            events.first(),
            Some(StreamEvent::Started { id: Some(id) }) if id == "resp_1"
        ));
        Ok(())
    }

    /// A proxy that drops `response.created` still produces a started stream.
    #[test]
    fn the_stream_starts_without_a_created_event() -> Result<(), Box<dyn StdError>> {
        let route = call(Request::builder().model(MODEL).user("hi").build()?)?
            .route()
            .clone();
        let mut decoder = codec().stream_decoder(&route);

        let mut events = decoder.decode(sse(&json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": { "type": "message", "id": "msg_1", "role": "assistant", "content": [] },
        })))?;
        events.extend(decoder.decode(sse(&json!({
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "delta": "hi",
        })))?);

        assert!(
            matches!(events.first(), Some(StreamEvent::Started { id: None })),
            "the first event must still be `Started`",
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, StreamEvent::Started { .. }))
                .count(),
            1,
            "a later event must not start the stream twice",
        );
        Ok(())
    }

    /// A tool result carrying only JSON sends the value itself, because the
    /// `ContentPart` envelope is this crate's shape rather than the tool's.
    #[test]
    fn a_json_tool_result_sends_the_bare_value() -> Result<(), Box<dyn StdError>> {
        let request = Request::builder()
            .model(MODEL)
            .user("Look it up")
            .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
                ToolCall::function("call_1", "search", json!({})),
            )]))
            .message(Message::new(Role::Tool, [ContentPart::ToolResult(
                ToolResult {
                    tool_call_id: "call_1".to_owned(),
                    name:         Some("search".to_owned()),
                    content:      vec![ContentPart::Json {
                        value: json!({ "matches": 2 }),
                    }],
                    is_error:     false,
                },
            )]))
            .build()?;

        let encoded = codec().encode(&call(request)?, false)?;

        let output = encoded.body["input"][2]["output"]
            .as_str()
            .ok_or("expected a function_call_output")?;
        assert_eq!(
            serde_json::from_str::<Value>(output)?,
            json!({ "matches": 2 })
        );
        Ok(())
    }

    #[test]
    fn an_empty_text_tool_result_sends_an_empty_output() -> Result<(), Box<dyn StdError>> {
        // A command with no stdout answers with nothing. Serializing the
        // ContentPart envelope instead would hand the model spurious JSON as
        // the tool's answer.
        let request = Request::builder()
            .model(MODEL)
            .user("Make the directory")
            .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
                ToolCall::function("call_1", "shell", json!({ "cmd": "mkdir foo" })),
            )]))
            .message(Message::new(Role::Tool, [ContentPart::ToolResult(
                ToolResult {
                    tool_call_id: "call_1".to_owned(),
                    name:         Some("shell".to_owned()),
                    content:      vec![ContentPart::Text {
                        text: String::new(),
                    }],
                    is_error:     false,
                },
            )]))
            .build()?;

        let encoded = codec().encode(&call(request)?, false)?;

        assert_eq!(encoded.body["input"][2]["output"], json!(""));
        Ok(())
    }
}
