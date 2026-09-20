//! The OpenAI Responses protocol, `POST /v1/responses`.
//!
//! This is the only dialect that carries custom tools, that replays
//! provider-native reasoning items verbatim, and whose stream terminates with a
//! complete response document. It is also the only one with a native input
//! token count endpoint that projects the generation body down to an allowlist.

mod decode;
mod encode;
mod stream;
#[cfg(all(test, feature = "builtin-catalog"))]
mod tests;

use decode::{decode_document, decode_error};
use encode::{CustomTools, input_items, instructions, is_system, shared_body};
use reqwest::Method;
use serde_json::{Map, Value, json};
use stream::ResponsesStream;

use super::common::{
    cache_routing_key, endpoint, flattens_system_content, flattens_tool_result_content,
    merge_options, reject_unencodable, sampling, wire_options,
};
use super::{Codec, StreamDecoder};
use crate::adapter::ResolvedCall;
use crate::resolver::ResolvedRoute;
use crate::transport::EncodedRequest;
use crate::types::{ContentPart, Error, MediaSource, Message, Response, Role};

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
        Box::new(ResponsesStream::new(route))
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
