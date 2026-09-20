//! The Gemini `generateContent` wire protocol.

mod decode;
mod encode;
mod stream;
#[cfg(all(test, feature = "builtin-catalog"))]
mod tests;

use decode::{decode_candidate, decode_usage, no_candidates, response_nonce};
use encode::{count_tokens_request, generate_body, model_endpoint};
use reqwest::Method;
use serde_json::Value;
use stream::GeminiStreamDecoder;

use super::common::{GEMINI_SIGNATURES, flattens_system_content, flattens_tool_result_content};
use super::{Codec, StreamDecoder};
use crate::adapter::ResolvedCall;
use crate::resolver::ResolvedRoute;
use crate::transport::EncodedRequest;
use crate::types::{ContentPart, Error, ErrorKind, Response};

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
        // Gemini 3 takes named thinking levels. Older and passthrough routes
        // do not claim that dialect, so they keep the unsupported warning.
        if call.request().reasoning_effort().is_some()
            && !call
                .route()
                .model()
                .protocol_options()
                .reasoning_effort_levels
        {
            encoded = encoded.unsupported_control("the reasoning effort control");
        }
        // A skipped foreign-signed reasoning part never reaches the model,
        // so the skip is reported; see `ReasoningContent::has_foreign_signature`.
        if call.request().carries_foreign_signature(GEMINI_SIGNATURES) {
            encoded = encoded.unsupported_control("reasoning signed by another provider");
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
