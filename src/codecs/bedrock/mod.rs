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

use decode::{decode_content_block, stop_reason, token_counts};
use encode::{
    MIN_THINKING_BUDGET, bedrock_effort, budget_limit, caches, carries_in_tool_result,
    conversation, encode_count_tokens, forces_tool_use, history_carries_tool_blocks,
    inference_config, operation_url, reject_custom_tools, reject_unnameable_tools, system_blocks,
    thinking_budget, tool_config,
};
use reqwest::Method;
use serde_json::{Map, Value, json};
use stream::BedrockStreamDecoder;

use super::common::{
    ANTHROPIC_SIGNATURES, flattens_system_content, flattens_tool_result_content, merge_options,
    refusal, reject_unencodable, wire_options,
};
use super::{Codec, StreamDecoder};
use crate::adapter::ResolvedCall;
use crate::resolver::ResolvedRoute;
use crate::transport::EncodedRequest;
use crate::types::{
    ContentPart, Error, ErrorKind, Response, RetryClassification, Speed, ToolChoice,
};

/// The opaque replay namespace this codec claims.
const NAMESPACE: &str = "bedrock";

#[derive(Clone, Copy, Debug)]
pub(crate) struct BedrockConverseCodec;

impl Codec for BedrockConverseCodec {
    fn encode(&self, call: &ResolvedCall, stream: bool) -> Result<EncodedRequest, Error> {
        let request = call.request();
        let route = call.route();
        reject_custom_tools(request, route)?;
        // Bedrock Converse carries no audio. Dropping it silently would let
        // the model answer a prompt the caller never sent.
        reject_unencodable(route, request, |part| {
            matches!(part, ContentPart::Audio(_)).then_some("audio content")
        })?;
        reject_unnameable_tools(request, route)?;
        let (options, controls) = wire_options(call);
        let cached = caches(route, controls.auto_cache);

        let mut body = Map::new();
        let system = system_blocks(request, cached);
        if !system.is_empty() {
            body.insert("system".to_owned(), Value::Array(system));
        }
        body.insert(
            "messages".to_owned(),
            Value::Array(conversation(request, route, cached)?),
        );

        let mut inference = inference_config(request);
        if let Some(tool_config) = tool_config(request, cached) {
            body.insert("toolConfig".to_owned(), tool_config);
        }
        // Effort has the same two wire dialects as the Anthropic codec,
        // carried through `additionalModelRequestFields`: a model with effort
        // levels takes `output_config.effort`, an older reasoning model takes
        // an explicit thinking budget, and a forced tool choice suppresses
        // both because the upstream model rejects thinking alongside it. The
        // budget must sit strictly below `maxTokens`, and a request that
        // sends none leaves AWS's per-model default in charge — a default at
        // or below the budget draws a ValidationException. So a budget always
        // travels with an explicit `maxTokens`, lifted when the budget would
        // not fit under it, the same way the Anthropic encoder grows it.
        if let Some(effort) = request.reasoning_effort()
            && !forces_tool_use(request.tool_choice())
        {
            // A passthrough model takes the modern effort dialect, like
            // the Anthropic codec: it is uncataloged precisely because it
            // is newer than the catalog, and a guessed thinking budget is
            // a manual toggle the always-adaptive models reject.
            if route.model().protocol_options().reasoning_effort_levels
                || route.model().is_passthrough()
            {
                body.insert(
                    "additionalModelRequestFields".to_owned(),
                    json!({ "output_config": { "effort": bedrock_effort(effort) } }),
                );
            } else {
                let limit = budget_limit(call);
                let budget = thinking_budget(effort, limit);
                let max_tokens = if limit <= budget {
                    budget.saturating_add(MIN_THINKING_BUDGET)
                } else {
                    limit
                };
                inference.insert("maxTokens".to_owned(), max_tokens.into());
                body.insert(
                    "additionalModelRequestFields".to_owned(),
                    json!({ "thinking": { "type": "enabled", "budget_tokens": budget } }),
                );
            }
        }
        if !inference.is_empty() {
            body.insert("inferenceConfig".to_owned(), Value::Object(inference));
        }
        if let Some(speed) = request.speed() {
            let latency = if matches!(speed, Speed::Fast) {
                "optimized"
            } else {
                "standard"
            };
            body.insert(
                "performanceConfig".to_owned(),
                json!({ "latency": latency }),
            );
        }

        merge_options(&mut body, options);

        let operation = if stream {
            "converse-stream"
        } else {
            "converse"
        };
        // A streaming request names the framing it expects, which is what the
        // reference client always sent; a gateway that negotiates content
        // types answers with the event stream rather than something else.
        let headers = if stream {
            vec![(
                "accept".to_owned(),
                "application/vnd.amazon.eventstream".to_owned(),
            )]
        } else {
            Vec::new()
        };
        let mut encoded = EncodedRequest::new(
            Method::POST,
            operation_url(route, operation),
            Value::Object(body),
        )
        .with_headers(headers)
        .with_applied_speed(request.speed());
        // Converse has no request-metadata field, so the map is reported rather
        // than folded into some other field where it would change the prompt.
        if !request.metadata().is_empty() {
            encoded = encoded.unsupported_control("request metadata");
        }
        // The suppressed effort never reaches the model, so say so — the same
        // report the Anthropic codec makes for this combination.
        if request.reasoning_effort().is_some() && forces_tool_use(request.tool_choice()) {
            encoded = encoded.unsupported_control("reasoning effort with a forced tool choice");
        }
        // The tools stayed on the wire despite `tool_choice: none`, because
        // Converse rejects a request whose history carries tool blocks
        // without a `toolConfig`. The model may therefore still call a tool.
        if matches!(request.tool_choice(), Some(ToolChoice::None))
            && !request.tools().is_empty()
            && history_carries_tool_blocks(request)
        {
            encoded =
                encoded.unsupported_control("tool_choice none alongside historical tool blocks");
        }
        // A skipped foreign-signed reasoning part never reaches the model,
        // so the skip is reported; see `ReasoningContent::has_foreign_signature`.
        if request.carries_foreign_signature(ANTHROPIC_SIGNATURES) {
            encoded = encoded.unsupported_control("reasoning signed by another provider");
        }
        // Converse has no portable structured-output field. A caller who asked
        // for JSON gets prose, so say so rather than letting them discover it
        // by parsing.
        if request.response_format().is_some() {
            encoded = encoded.unsupported_control("response formats");
        }
        // The system field of this protocol takes text only, so anything else
        // a system message carries is dropped. The text still reaches the
        // model, so it is reported rather than refused.
        if flattens_system_content(request) {
            encoded = encoded.unsupported_control("non-text system content");
        }
        // `toolResult.content` is a block list of its own, so text, structured
        // JSON, images, and documents all reach the model as themselves. Only
        // a part with no member of that union — reasoning, most of all — is
        // dropped, and only that is worth reporting.
        if flattens_tool_result_content(request, |parts| parts.iter().all(carries_in_tool_result)) {
            encoded =
                encoded.unsupported_control("tool result content outside text, JSON, and media");
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
            // A structurally malformed 200 is indistinguishable from a
            // garbled or truncated body, so a fresh attempt is safe — the
            // same classification the transport gives a 200 whose body is
            // not JSON at all.
            return Err(Error::new(
                ErrorKind::ResponseDecode,
                format!(
                    "provider {} returned a 200 body without a Converse output message",
                    route.provider().id()
                ),
            )
            .with_provider(route.provider().id().clone())
            .with_raw_data(value)
            .with_retry(RetryClassification::Safe));
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
        response.finish_reason = stop_reason(value.get("stopReason").and_then(Value::as_str));
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
                // Malformed like any other garbled 200, so retrying is safe.
                Error::new(
                    ErrorKind::ResponseDecode,
                    "Bedrock returned a count-tokens body without an inputTokens count",
                )
                .with_provider(route.provider().id().clone())
                .with_raw_data(value)
                .with_retry(RetryClassification::Safe)
            })
    }
}
