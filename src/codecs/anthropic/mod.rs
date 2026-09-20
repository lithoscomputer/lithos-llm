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
    beta_headers, count_tokens_request, forces_tool_use, headers, json_schema, message_body,
    reject_custom_tools,
};
use reqwest::Method;
use serde_json::Value;
use stream::AnthropicStreamDecoder;

use super::common::{
    ANTHROPIC_SIGNATURES, endpoint, finish_reason, flattens_system_content,
    flattens_tool_result_content, merge_options, refusal, reject_unencodable, wire_options,
};
use super::{Codec, StreamDecoder};
use crate::adapter::ResolvedCall;
use crate::resolver::ResolvedRoute;
use crate::transport::EncodedRequest;
use crate::types::{ContentPart, Error, ErrorKind, Response, RetryClassification};

/// The opaque-part namespace this codec owns.
///
/// A [`ContentPart::Opaque`] whose kind starts with this namespace is replayed
/// verbatim; every other namespace is skipped, so a request built for another
/// provider still encodes after failover.
const NAMESPACE: &str = "anthropic";

/// The API version every request declares.
const API_VERSION: &str = "2023-06-01";

/// The `max_tokens` used when neither the request nor the catalog sets one.
///
/// Anthropic requires the field, so there is no "omit it" option. This is the
/// last resort: a request that names no limit and a model the catalog records
/// no output limit for. It is deliberately generous, because a limit picked
/// here truncates a long generation silently.
const DEFAULT_MAX_TOKENS: u32 = 65_536;

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

/// The smallest `thinking.budget_tokens` the API accepts.
///
/// Doubles as the headroom kept above the budget when the output limit must
/// grow, because the budget has to sit strictly below `max_tokens`.
const MIN_THINKING_BUDGET: u32 = 1024;

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
        reject_custom_tools(call)?;
        // Anthropic Messages carries no audio. Dropping it silently would let
        // the model answer a prompt the caller never sent.
        reject_unencodable(call.route(), call.request(), |part| {
            matches!(part, ContentPart::Audio(_)).then_some("audio content")
        })?;

        let request = call.request();
        let (mut options, controls) = wire_options(call);
        let betas = beta_headers(&mut options, request.speed());
        let raw_thinking = options.contains_key("thinking");
        let mut body = message_body(call, controls.auto_cache, raw_thinking);
        body.insert("stream".to_owned(), stream.into());
        merge_options(&mut body, options);

        let mut encoded = EncodedRequest::new(
            Method::POST,
            endpoint(call.route().provider().base_url(), "/v1/messages"),
            Value::Object(body),
        )
        .with_headers(headers(&betas))
        .with_applied_speed(request.speed());
        // The system field of this protocol takes text only, so anything else
        // a system message carries is dropped. The text still reaches the
        // model, so it is reported rather than refused.
        if flattens_system_content(request) {
            encoded = encoded.unsupported_control("non-text system content");
        }
        if flattens_tool_result_content(request, |parts| {
            parts.iter().all(|part| {
                matches!(
                    part,
                    ContentPart::Text { .. } | ContentPart::Json { .. } | ContentPart::Image(_)
                )
            })
        }) {
            encoded = encoded.unsupported_control("non-text tool result content");
        }
        // A skipped foreign-signed reasoning part never reaches the model,
        // so the skip is reported; see `ReasoningContent::has_foreign_signature`.
        if request.carries_foreign_signature(ANTHROPIC_SIGNATURES) {
            encoded = encoded.unsupported_control("reasoning signed by another provider");
        }
        // A forced tool choice drops both output controls; see
        // `forces_tool_use`. Neither reaches the model, so both are reported.
        // A raw `thinking` option stays in the body — raw options are
        // authoritative — but Anthropic rejects the pair, so it is reported
        // too.
        if forces_tool_use(request.tool_choice()) {
            if request.reasoning_effort().is_some() {
                encoded = encoded.unsupported_control("reasoning effort with a forced tool choice");
            }
            if request.response_format().and_then(json_schema).is_some() {
                encoded =
                    encoded.unsupported_control("structured output with a forced tool choice");
            }
            if raw_thinking {
                encoded = encoded
                    .unsupported_control("a thinking provider option with a forced tool choice");
            }
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
            // A structurally malformed 200 is indistinguishable from a
            // garbled or truncated body, so a fresh attempt is safe — the
            // same classification the transport gives a 200 whose body is
            // not JSON at all.
            return Err(Error::new(
                ErrorKind::ResponseDecode,
                format!("Anthropic returned a response without the {field} field"),
            )
            .with_provider(route.provider().id().clone())
            .with_raw_data(value)
            .with_retry(RetryClassification::Safe));
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
