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
use encode::{
    effort_name, encode_message, encode_response_format, encode_tool, encode_tool_choice,
    mark_cache_breakpoints,
};
use reasoning_details::complete_details;
use reqwest::Method;
use serde_json::{Map, Value, json};
use stream::ChatStreamDecoder;

use super::content::{
    finish_reason, flattens_tool_result_content, promote_tool_finish, reject_unencodable,
    text_or_json_only,
};
use super::errors::{refusal, unsupported_capability};
use super::options::{cache_routing_key, endpoint, merge_options, sampling, wire_options};
use super::{Codec, StreamDecoder};
use crate::adapter::ResolvedCall;
use crate::resolver::ResolvedRoute;
use crate::transport::EncodedRequest;
use crate::types::{ContentPart, Error, Message, ReasoningContent, Response, ToolDefinition};

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
        let request = call.request();
        let route = call.route();
        if request.tools().iter().any(ToolDefinition::is_custom) {
            return Err(unsupported_capability(route, "custom tools"));
        }
        // This dialect encodes neither audio nor documents. Dropping them
        // silently would return a successful response for a prompt that never
        // carried the caller's attachment.
        reject_unencodable(route, request, |part| match part {
            ContentPart::Audio(_) => Some("audio content"),
            ContentPart::Document(_) => Some("document content"),
            _ => None,
        })?;

        let (options, controls) = wire_options(call);
        let mut body = Map::new();
        body.insert("model".to_owned(), route.api_model().into());

        let mut messages: Vec<Value> = request.messages().iter().flat_map(encode_message).collect();
        // An aggregator fronting an Anthropic model forwards the breakpoints
        // upstream, and the catalog opts such a model in explicitly. A skin
        // that caches automatically (DeepSeek, Moonshot) declares `caching`
        // for its pricing without `cache_breakpoints`, and rewriting its
        // string content into part arrays could get the request rejected. A
        // caller can also turn breakpoints off with the `auto_cache` control.
        let capabilities = route.model().capabilities();
        if controls.auto_cache
            && capabilities.caching().is_supported()
            && route.model().protocol_options().cache_breakpoints
        {
            mark_cache_breakpoints(&mut messages);
        }
        body.insert("messages".to_owned(), Value::Array(messages));

        // A blocking request omits the `stream` member entirely, matching the
        // reference encoder — a strict skin may reject an explicit
        // `stream: false`.
        if stream {
            body.insert("stream".to_owned(), true.into());
            // Without this the compatible skins never send a usage chunk, and
            // a streamed response would report no tokens at all.
            body.insert(
                "stream_options".to_owned(),
                json!({ "include_usage": true }),
            );
        }
        if let Some(max_tokens) = request.max_output_tokens() {
            body.insert("max_tokens".to_owned(), max_tokens.into());
        }
        if let Some(temperature) = request.temperature() {
            body.insert("temperature".to_owned(), sampling(temperature));
        }
        if let Some(top_p) = request.top_p() {
            body.insert("top_p".to_owned(), sampling(top_p));
        }
        if let Some(effort) = request.reasoning_effort() {
            body.insert("reasoning_effort".to_owned(), effort_name(effort).into());
        }
        if !request.stop_sequences().is_empty() {
            let stop = request
                .stop_sequences()
                .iter()
                .map(|sequence| Value::from(sequence.as_str()))
                .collect();
            body.insert("stop".to_owned(), Value::Array(stop));
        }
        if !request.tools().is_empty() {
            let tools: Vec<Value> = request.tools().iter().filter_map(encode_tool).collect();
            body.insert("tools".to_owned(), Value::Array(tools));
        }
        if let Some(choice) = request.tool_choice() {
            body.insert("tool_choice".to_owned(), encode_tool_choice(choice));
        }
        if let Some(format) = request.response_format() {
            body.insert("response_format".to_owned(), encode_response_format(format));
        }

        if let Some(key) = cache_routing_key(call, controls) {
            body.insert("prompt_cache_key".to_owned(), key.into());
        }

        // Raw provider options are merged last so an application can override
        // anything encoded above.
        merge_options(&mut body, options);

        let mut encoded = EncodedRequest::new(
            Method::POST,
            endpoint(
                route.provider().base_url(),
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
        // Only OpenAI itself documents `metadata` on this endpoint, and a
        // strict skin rejects the whole request over one unknown field. The
        // tags are dropped rather than risking that, and a caller who knows
        // their skin accepts them can send them as a raw provider option.
        if !request.metadata().is_empty() {
            encoded = encoded.unsupported_control("request metadata");
        }
        // Chat Completions has no speed control. The request is served and
        // billed at standard speed, so the dropped control is reported.
        if request.speed().is_some() {
            encoded = encoded.unsupported_control("the speed control");
        }
        // A `tool` message has no error marker in this protocol, so a failed
        // tool result reaches the model looking like a successful one. The
        // content still arrives, so this is a warning rather than a refusal.
        if request
            .messages()
            .iter()
            .flat_map(Message::content)
            .any(|part| matches!(part, ContentPart::ToolResult(result) if result.is_error))
        {
            encoded = encoded.unsupported_control("the tool result error flag");
        }
        // An all-JSON result travels as the bare value, so only a mix that
        // must flatten is reported.
        if flattens_tool_result_content(request, text_or_json_only) {
            encoded = encoded.unsupported_control("non-text tool result content");
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
        let mut flaw = None;
        for call in message
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            match decode_tool_call(call) {
                Ok(part) => content.push(part),
                Err(detail) => {
                    flaw = Some(detail);
                    break;
                }
            }
        }
        if let Some(detail) = flaw {
            return Err(decode_failure(route, detail, value));
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
