//! The OpenAI Chat Completions dialect.
//!
//! This codec speaks `POST /v1/chat/completions` as OpenAI defined it and as
//! every compatible skin re-implements it: OpenRouter, DeepSeek, Venice, Modal,
//! Kimi, and MiniMax. Those skins agree on the envelope and disagree on the
//! details, so the decoder accepts several spellings of the same counter and
//! keeps the untouched success body in [`Response::raw`]. Fields this crate
//! does not model — `cost_details.upstream_inference_cost`,
//! `prompt_tokens_details.audio_tokens`, OpenRouter's top-level `provider`,
//! and `native_finish_reason` — survive only there.
//!
//! It is the one codec that reads a provider-reported cost off the wire. Every
//! other protocol leaves [`Response::cost`] unset for the adapter to fill in
//! from catalog pricing.

mod decode;
mod encode;
mod reasoning_details;
mod stream;
#[cfg(all(test, feature = "builtin-catalog"))]
mod tests;

use decode::{
    decode_failure, decode_tool_call, message_text, no_choices, provider_cost, reasoning_text,
    token_counts,
};
use encode::{body, dropped_controls, preflight};
use reasoning_details::complete_details;
use reqwest::Method;
use serde_json::Value;
use stream::ChatStreamDecoder;

use super::content::{finish_reason, promote_tool_finish};
use super::errors::refusal;
use super::options::{cache_routing_key, endpoint, merge_options, wire_options};
use super::{Codec, StreamDecoder};
use crate::adapter::ResolvedCall;
use crate::resolver::ResolvedRoute;
use crate::transport::EncodedRequest;
use crate::types::{ContentPart, Error, ReasoningContent, Response};

/// The prefix of an opaque content kind this dialect claims.
///
/// The namespace is `openai_compatible`; the text after the separator names the
/// message field the part replays into.
const OPAQUE_PREFIX: &str = "openai_compatible.";

/// The message field carrying an aggregator's structured reasoning channel.
///
/// OpenRouter sends signed or encrypted reasoning here, and the upstream model
/// rejects a continued turn that does not send it back. It is preserved
/// verbatim as an opaque part named after this field, so
/// [`encode_chat_message`] replays it into the same place.
const REASONING_DETAILS: &str = "reasoning_details";

/// The opaque content kind the structured reasoning channel replays as:
/// [`OPAQUE_PREFIX`] followed by [`REASONING_DETAILS`]. Its inverse is the
/// `strip_prefix` in `encode_chat_message`.
const REASONING_DETAILS_KIND: &str = "openai_compatible.reasoning_details";

/// The id of the single streamed text block.
///
/// The protocol carries no block ids at all, so every text fragment of one
/// response belongs to the same synthesized block.
const TEXT_BLOCK: &str = "block-0";

/// The id of the single streamed reasoning block.
const REASONING_BLOCK: &str = "reasoning-0";

/// The id of the single streamed `reasoning_details` block.
const REASONING_DETAILS_BLOCK: &str = "reasoning-details-0";

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct OpenAiChatCodec {
    base_url_is_api_root: bool,
}

impl OpenAiChatCodec {
    pub(crate) fn at_api_root() -> Self {
        Self {
            base_url_is_api_root: true,
        }
    }
}

impl Codec for OpenAiChatCodec {
    fn encode(&self, call: &ResolvedCall, stream: bool) -> Result<EncodedRequest, Error> {
        preflight(call)?;
        let (options, controls) = wire_options(call);
        let mut body = body(call, controls.auto_cache, stream);
        if let Some(key) = cache_routing_key(call, controls) {
            body.insert("prompt_cache_key".to_owned(), key.into());
        }
        merge_options(&mut body, options);

        let mut encoded = EncodedRequest::new(
            Method::POST,
            endpoint(
                call.route().provider().base_url(),
                if self.base_url_is_api_root {
                    "/chat/completions"
                } else {
                    "/v1/chat/completions"
                },
            ),
            Value::Object(body),
        )
        // Every chunk is one `data:` line of JSON, and lenient compatible
        // skins and proxies separate them with single newlines rather than
        // the blank line the SSE specification requires.
        .with_data_line_framing();
        for control in dropped_controls(call.request()) {
            encoded = encoded.unsupported_control(control);
        }
        Ok(encoded)
    }

    fn decode_response(&self, route: &ResolvedRoute, value: Value) -> Result<Response, Error> {
        // A 200 whose `choices` array is missing or empty carries no answer at
        // all. Decoding it as an empty success would hand the caller a
        // finished response the model never wrote, so the body fails to
        // decode instead.
        if value.pointer("/choices/0").is_none() {
            return Err(no_choices(route, value));
        }
        let choice = value.pointer("/choices/0").unwrap_or(&Value::Null);
        // A choice without a message object is not a completion. A truncated
        // or error-shaped choice must not decode as an empty success.
        if !choice.get("message").is_some_and(Value::is_object) {
            return Err(decode_failure(
                route,
                "returned a choice without a message object",
                value,
            ));
        }
        let message = choice.get("message").unwrap_or(&Value::Null);
        // A refusal is a failure, not a short answer — the same contract the
        // Anthropic and Bedrock codecs apply. OpenAI reports it in a channel
        // of its own precisely so it cannot be mistaken for content.
        if let Some(text) = message
            .get("refusal")
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
        {
            let explanation = text.to_owned();
            return Err(refusal(route, Some(&explanation), Some(value)));
        }

        let mut content = Vec::new();
        // The structured reasoning channel comes first, ahead of the readable
        // reasoning text it describes.
        if let Some(details) = complete_details(message) {
            content.push(details);
        }
        if let Some(text) = reasoning_text(message) {
            content.push(ContentPart::Reasoning(ReasoningContent {
                text,
                signature: None,
                signature_origin: None,
                redacted: false,
            }));
        }
        if let Some(text) = message_text(message) {
            content.push(ContentPart::Text { text });
        }
        let calls: Result<Vec<ContentPart>, &'static str> = message
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(decode_tool_call)
            .collect();
        match calls {
            Ok(calls) => content.extend(calls),
            Err(detail) => return Err(decode_failure(route, detail, value)),
        }

        let mut response = Response::new(
            route.provider().id().clone(),
            route.model().id().clone(),
            content,
        );
        response.id = value
            .get("id")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        // Some skins answer a tool call with `finish_reason: "stop"` — qwen
        // on Venice does. The streaming path already lets an assembled tool
        // call win over a stop reason, and both paths must decode one
        // exchange the same way, so the complete path applies the same rule.
        response.finish_reason = promote_tool_finish(
            finish_reason(choice.get("finish_reason").and_then(Value::as_str)),
            response
                .content
                .iter()
                .any(|part| matches!(part, ContentPart::ToolCall(_))),
        );
        response.usage = value.get("usage").map(token_counts).unwrap_or_default();
        response.cost = provider_cost(&value);
        response.raw = Some(value);
        response.suppress_unfinished_tool_calls();
        Ok(response)
    }

    fn stream_decoder(&self, route: &ResolvedRoute) -> Box<dyn StreamDecoder> {
        Box::new(ChatStreamDecoder::new(route))
    }

    /// This dialect has no token count endpoint.
    ///
    /// Estimating the count locally is deliberately out of scope; a consumer
    /// that needs an estimate owns that choice, including which tokenizer to
    /// trust for a given compatible skin.
    fn encode_count_tokens(&self, _call: &ResolvedCall) -> Option<Result<EncodedRequest, Error>> {
        None
    }
}
