//! The Amazon Bedrock Converse wire protocol.
//!
//! One codec serves both Bedrock authentication schemes and both call shapes.
//! Bearer and SigV4 send the identical body to the identical URL, and the
//! unary, streaming, and token-count paths differ only in the operation
//! segment of the path, so nothing here knows how the request is signed.
//!
//! The streaming half is fed AWS `vnd.amazon.eventstream` frames that
//! [`event_stream`](crate::transport::event_stream) has already decoded. The
//! transport hands each frame over as an [`SseEvent`] whose `event` is the
//! frame's `:event-type` header and whose `data` is the event JSON.

mod decode;
mod encode;
mod stream;
#[cfg(all(test, feature = "builtin-catalog"))]
mod tests;

use decode::{decode_content_block, token_counts};
use encode::{Operation, converse_body, dropped_controls, encode_count_tokens, preflight};
use reqwest::Method;
use serde_json::Value;
use stream::BedrockStreamDecoder;

use super::content::finish_reason;
use super::errors::{malformed_success, refusal};
use super::options::{merge_options, wire_options};
use super::{Codec, StreamDecoder};
use crate::adapter::ResolvedCall;
use crate::resolver::ResolvedRoute;
use crate::transport::EncodedRequest;
use crate::types::{Error, Response};

/// The opaque replay namespace this codec claims.
const NAMESPACE: &str = "bedrock";

#[derive(Clone, Copy, Debug)]
pub(crate) struct BedrockConverseCodec;

impl Codec for BedrockConverseCodec {
    fn encode(&self, call: &ResolvedCall, stream: bool) -> Result<EncodedRequest, Error> {
        preflight(call)?;
        let request = call.request();
        let (options, controls) = wire_options(call);
        let mut body = converse_body(call, controls.auto_cache)?;
        merge_options(&mut body, options);

        let operation = if stream {
            Operation::ConverseStream
        } else {
            Operation::Converse
        };
        let mut encoded = EncodedRequest::new(
            Method::POST,
            operation.url(call.route()),
            Value::Object(body),
        )
        .with_headers(operation.headers())
        .with_applied_speed(request.speed());
        for control in dropped_controls(request) {
            encoded = encoded.unsupported_control(control);
        }
        Ok(encoded)
    }

    fn decode_response(&self, route: &ResolvedRoute, value: Value) -> Result<Response, Error> {
        // A refusal is a failure, not a short answer — the same contract the
        // Anthropic codec applies, since Bedrock passes the stop through for
        // Claude models.
        if value.get("stopReason").and_then(Value::as_str) == Some("refusal") {
            return Err(refusal(route, None, Some(value)));
        }

        // A body without the output message is not a Converse response.
        // Decoding `{}` from a broken proxy as a successful empty answer
        // would be indistinguishable from a real empty completion.
        if !value
            .pointer("/output/message/content")
            .is_some_and(Value::is_array)
        {
            return Err(malformed_success(
                route,
                format!(
                    "provider {} returned a 200 body without a Converse output message",
                    route.provider().id()
                ),
                Some(value),
            ));
        }

        let content = value
            .pointer("/output/message/content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(decode_content_block)
            .collect();

        let mut response = Response::new(
            route.provider().id().clone(),
            route.model().id().clone(),
            content,
        );
        response.finish_reason = finish_reason(value.get("stopReason").and_then(Value::as_str));
        response.usage = token_counts(value.get("usage"));
        response.raw = Some(value);
        response.suppress_unfinished_tool_calls();
        Ok(response)
    }

    fn stream_decoder(&self, route: &ResolvedRoute) -> Box<dyn StreamDecoder> {
        Box::new(BedrockStreamDecoder::new(route))
    }

    fn encode_count_tokens(&self, call: &ResolvedCall) -> Option<Result<EncodedRequest, Error>> {
        Some(encode_count_tokens(call))
    }

    fn decode_count_tokens(&self, route: &ResolvedRoute, value: Value) -> Result<u64, Error> {
        value
            .get("inputTokens")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                malformed_success(
                    route,
                    "Bedrock returned a count-tokens body without an inputTokens count",
                    Some(value),
                )
            })
    }
}
