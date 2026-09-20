//! The Anthropic Messages wire protocol.
//!
//! Anthropic splits work across three endpoints this codec speaks:
//! `/v1/messages` for generation, the same path with `stream: true` for the SSE
//! protocol, and `/v1/messages/count_tokens` for a provider-authoritative input
//! token count.

mod decode;
mod encode;
mod stream;
#[cfg(all(test, feature = "builtin-catalog"))]
mod tests;

use decode::{decode_block, missing_response_field, token_counts};
use encode::{
    beta_headers, count_tokens_request, dropped_controls, headers, message_body, preflight,
};
use reqwest::Method;
use serde_json::Value;
use stream::AnthropicStreamDecoder;

use super::claude::ThinkingPlan;
use super::content::finish_reason;
use super::errors::{malformed_success, refusal};
use super::options::{endpoint, merge_options, wire_options};
use super::{Codec, StreamDecoder};
use crate::adapter::ResolvedCall;
use crate::resolver::ResolvedRoute;
use crate::transport::EncodedRequest;
use crate::types::{Error, ErrorKind, Response};

/// The opaque-part namespace this codec owns.
///
/// A [`ContentPart::Opaque`] whose kind starts with this namespace is replayed
/// verbatim; every other namespace is skipped, so a request built for another
/// provider still encodes after failover.
const NAMESPACE: &str = "anthropic";

/// The API version every request declares.
const API_VERSION: &str = "2023-06-01";

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

/// The instruction [`ResponseFormat::JsonObject`] appends to the system text.
///
/// Free-form JSON has no schema: Anthropic's structured-output subset requires
/// `additionalProperties: false`, so a schema loose enough for "any JSON
/// object" is not expressible, and a strict one would pin the model to the
/// empty object. A soft instruction asks for JSON without constraining its
/// shape.
const JSON_OBJECT_INSTRUCTION: &str = "You must respond with valid JSON only, no other text.";

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
        preflight(call)?;
        let request = call.request();
        let (mut options, controls) = wire_options(call);
        let betas = beta_headers(&mut options, request.speed());
        let plan = ThinkingPlan::for_call(call, options.contains_key("thinking"));
        let mut body = message_body(call, controls.auto_cache, &plan);
        body.insert("stream".to_owned(), stream.into());
        merge_options(&mut body, options);

        let mut encoded = EncodedRequest::new(
            Method::POST,
            endpoint(call.route().provider().base_url(), "/v1/messages"),
            Value::Object(body),
        )
        .with_headers(headers(&betas))
        .with_applied_speed(request.speed());
        for control in dropped_controls(request, &plan) {
            encoded = encoded.unsupported_control(control);
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
            return Err(malformed_success(
                route,
                format!("Anthropic returned a response without the {field} field"),
                Some(value),
            ));
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
        response.suppress_unfinished_tool_calls();
        Ok(response)
    }

    fn stream_decoder(&self, route: &ResolvedRoute) -> Box<dyn StreamDecoder> {
        Box::new(AnthropicStreamDecoder::new(route))
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
