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
use encode::{ResponsesBody, dropped_controls, unencodable_part};
use reqwest::Method;
use serde_json::Value;
use stream::ResponsesStream;

use super::content::reject_unencodable;
use super::options::{cache_routing_key, endpoint, merge_options, wire_options};
use super::{Codec, StreamDecoder};
use crate::adapter::ResolvedCall;
use crate::resolver::ResolvedRoute;
use crate::transport::EncodedRequest;
use crate::types::{Error, Response};

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
        reject_unencodable(call.route(), call.request(), unencodable_part)?;
        let (options, controls) = wire_options(call);
        let mut body = ResponsesBody::new(self.codex, call, stream).build();
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
        .with_applied_speed(call.request().speed());
        for control in dropped_controls(call.request(), self.codex) {
            encoded = encoded.unsupported_control(control);
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
}
