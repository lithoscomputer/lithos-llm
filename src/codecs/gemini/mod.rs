//! The Gemini `generateContent` wire protocol.

mod decode;
mod encode;
mod identity;
mod stream;
#[cfg(all(test, feature = "builtin-catalog"))]
mod tests;

use decode::{decode_candidate, no_candidates, token_counts};
use encode::{count_tokens_request, dropped_controls, generate_body, model_endpoint};
use identity::response_nonce;
use reqwest::Method;
use serde_json::Value;
use stream::GeminiStreamDecoder;

use super::{Codec, StreamDecoder};
use crate::adapter::ResolvedCall;
use crate::resolver::ResolvedRoute;
use crate::transport::EncodedRequest;
use crate::types::{Error, ErrorKind, Response};

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
        // `?alt=sse` sends one JSON document per `data:` line, so each line
        // decodes on its own even when a proxy drops the blank line between
        // events.
        .with_data_line_framing();
        for control in dropped_controls(call) {
            encoded = encoded.unsupported_control(control);
        }
        Ok(encoded)
    }

    fn decode_response(&self, route: &ResolvedRoute, value: Value) -> Result<Response, Error> {
        let id = value
            .get("responseId")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        // The ids synthesized for function calls need a scope even when the
        // payload names no response; see [`tool_call_id`].
        let scope = id.clone().unwrap_or_else(response_nonce);

        // The candidate is decoded through a closure so that the borrow of
        // `value` ends here: the no-candidates path below takes the whole
        // document as the error's raw data.
        let decoded = value
            .pointer("/candidates/0")
            .map(|candidate| decode_candidate(candidate, &scope));
        let Some((content, finished)) = decoded else {
            return Err(no_candidates(route, value));
        };
        let usage = token_counts(value.get("usageMetadata").unwrap_or(&Value::Null));

        let mut response = Response::new(
            route.provider().id().clone(),
            route.model().id().clone(),
            content,
        );
        response.id = id;
        response.finish_reason = finished;
        response.usage = usage;
        response.raw = Some(value);
        response.suppress_unfinished_tool_calls();
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
