use std::fmt;

use serde::de::{Error as DeError, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use super::{ContentPart, Message, Role, ToolCall};
use crate::catalog::{ModelHandle, ModelId, ProviderId};

/// Why a model stopped producing output.
///
/// Every variant is one JSON string, [`FinishReason::Other`] included: a reason
/// this crate does not name yet is carried as the provider's own spelling
/// rather than a different JSON shape. So a stored response keeps loading when
/// a later version turns one of those strings into a named variant, and a
/// caller reading the JSON sees one field with one kind of value.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum FinishReason {
    Stop,
    Length,
    ToolCall,
    ContentFilter,
    Error,
    /// The stream ended without a provider terminal reason.
    Incomplete,
    Other(String),
}

impl FinishReason {
    /// The canonical string for this reason.
    fn as_str(&self) -> &str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::ToolCall => "tool_call",
            Self::ContentFilter => "content_filter",
            Self::Error => "error",
            Self::Incomplete => "incomplete",
            Self::Other(reason) => reason,
        }
    }
}

impl From<&str> for FinishReason {
    /// Names the matching variant, keeping any other spelling verbatim.
    fn from(value: &str) -> Self {
        match value {
            "stop" => Self::Stop,
            "length" => Self::Length,
            "tool_call" => Self::ToolCall,
            "content_filter" => Self::ContentFilter,
            "error" => Self::Error,
            "incomplete" => Self::Incomplete,
            other => Self::Other(other.to_owned()),
        }
    }
}

impl Serialize for FinishReason {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for FinishReason {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_str(FinishReasonVisitor)
    }
}

/// Reads the canonical finish reason string.
struct FinishReasonVisitor;

impl Visitor<'_> for FinishReasonVisitor {
    type Value = FinishReason;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a finish reason string")
    }

    fn visit_str<E: DeError>(self, value: &str) -> Result<Self::Value, E> {
        Ok(FinishReason::from(value))
    }
}

/// Token usage reported or estimated for a call.
///
/// The five buckets are **disjoint**: every token is counted in exactly one of
/// them, so [`TokenCounts::total`] is their plain sum. Providers usually report
/// inclusive counters instead — a prompt total that already contains the cached
/// tokens, or a completion total that already contains the reasoning tokens.
/// Codecs are responsible for normalizing those counters before they build a
/// `TokenCounts`, subtracting the detail counters with saturating arithmetic so
/// an inconsistent provider payload can never underflow.
/// [`TokenCounts::from_inclusive`] does exactly that for the common shape.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TokenCounts {
    /// Prompt tokens that were neither read from nor written to a cache.
    #[serde(default)]
    pub input:       u64,
    /// Completion tokens that are not reasoning tokens.
    #[serde(default)]
    pub output:      u64,
    /// Completion tokens spent on reasoning, billed at the output rate.
    #[serde(default)]
    pub reasoning:   u64,
    /// Prompt tokens served from a provider cache.
    #[serde(default)]
    pub cache_read:  u64,
    /// Prompt tokens written into a provider cache.
    #[serde(default)]
    pub cache_write: u64,
}

impl TokenCounts {
    /// Builds disjoint buckets from provider counters where `input` includes
    /// `cache_read` and `cache_write`, and `output` includes `reasoning`.
    ///
    /// Each detail counter is clamped to what remains of its parent total, so
    /// a provider that reports a detail larger than its own total — a skin
    /// counting reasoning exclusive of completion, say — cannot inflate the
    /// summed total or the billable output past what the parent counters
    /// claim.
    pub fn from_inclusive(
        input: u64,
        output: u64,
        reasoning: u64,
        cache_read: u64,
        cache_write: u64,
    ) -> Self {
        let cache_read = cache_read.min(input);
        let cache_write = cache_write.min(input.saturating_sub(cache_read));
        let reasoning = reasoning.min(output);
        Self {
            input: input.saturating_sub(cache_read).saturating_sub(cache_write),
            output: output.saturating_sub(reasoning),
            reasoning,
            cache_read,
            cache_write,
        }
    }

    /// The sum of all five disjoint buckets.
    pub fn total(self) -> u64 {
        self.input
            .saturating_add(self.output)
            .saturating_add(self.reasoning)
            .saturating_add(self.cache_read)
            .saturating_add(self.cache_write)
    }

    /// Tokens billed at the output rate: `output + reasoning`.
    pub fn billable_output(self) -> u64 {
        self.output.saturating_add(self.reasoning)
    }
}

/// The source used for a computed response cost.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CostSource {
    Catalog,
    Provider,
    Application,
}

/// A cost expressed in US dollar micros.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Cost {
    pub usd_micros: u64,
    pub source:     CostSource,
}

/// Rate-limit information returned by a provider.
///
/// The two reset fields stay separate and keep the provider's own formatting,
/// such as `"1s"` or `"6m0s"` from OpenAI and an RFC 3339 timestamp from
/// Anthropic. This crate does not parse them into a duration or an instant.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct RateLimits {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_limit:     Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_remaining: Option<u64>,
    /// When the request allowance resets, in the provider's own format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_reset:     Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_limit:       Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_remaining:   Option<u64>,
    /// When the token allowance resets, in the provider's own format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_reset:       Option<String>,
}

/// A non-fatal provider or normalization warning.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Warning {
    pub code:    String,
    pub message: String,
}

/// A normalized complete model response.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Response {
    pub id:            Option<String>,
    pub model:         ModelHandle,
    pub content:       Vec<ContentPart>,
    pub finish_reason: FinishReason,
    pub usage:         TokenCounts,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost:          Option<Cost>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_limits:   Option<RateLimits>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings:      Vec<Warning>,
    /// The complete provider success payload, when one was available.
    ///
    /// A complete (non-streaming) response holds the whole JSON success body
    /// exactly as the provider sent it. A streamed response holds the final
    /// provider response object when the streaming protocol supplies one, and
    /// is `None` otherwise. It never holds an accumulated log of stream events.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw:           Option<Value>,
}

impl Response {
    #[cfg(any(
        feature = "openai",
        feature = "anthropic",
        feature = "gemini",
        feature = "openai-compatible",
        feature = "bedrock",
        test
    ))]
    /// Removes the tool calls from a response the output limit cut short.
    ///
    /// A call the model never finished is not a call: its arguments are
    /// whatever prefix survived — a truncated JSON string on OpenAI, an
    /// empty object on Anthropic — and running it would act on a guess.
    /// Every provider reports the cut as a length finish, so a consumer
    /// branches on
    /// [`FinishReason::Length`](crate::types::FinishReason::Length) alone and
    /// never sees the half-call. Each dropped call leaves a
    /// [`TRUNCATED_TOOL_CALL`] warning naming its tool; the provider's own item
    /// stays in [`Response::raw`] for anyone who wants the partial arguments.
    ///
    /// Every codec applies this at the end of its blocking decode, and the
    /// stream assembler applies it to the response it completes, so both
    /// paths agree by construction. Only a length finish is touched: a
    /// malformed argument string on any other finish is a provider bug, not
    /// a cut, and keeps its call.
    pub(crate) fn drop_truncated_tool_calls(&mut self) {
        if self.finish_reason != FinishReason::Length {
            return;
        }
        let mut dropped = Vec::new();
        self.content.retain(|part| match part {
            ContentPart::ToolCall(call) => {
                dropped.push(call.name.clone());
                false
            }
            _ => true,
        });
        self.warnings
            .extend(dropped.into_iter().map(|name| Warning {
                code:    "truncated_tool_call".to_owned(),
                message: format!(
                    "the output limit cut off a call to {name} before its arguments were complete"
                ),
            }));
    }

    /// Visits the final response's tool calls in content order.
    pub fn tool_calls(&self) -> impl Iterator<Item = &ToolCall> {
        self.content.iter().filter_map(|part| match part {
            ContentPart::ToolCall(call) => Some(call),
            _ => None,
        })
    }

    /// Moves all content, including provider replay data, into an assistant
    /// message. Record response usage and other accounting fields before
    /// consuming it.
    pub fn into_message(self) -> Message {
        Message::new(Role::Assistant, self.content)
    }

    pub fn new(provider: ProviderId, model: ModelId, content: Vec<ContentPart>) -> Self {
        Self {
            id: None,
            model: ModelHandle::new(provider, model),
            content,
            finish_reason: FinishReason::Stop,
            usage: TokenCounts::default(),
            cost: None,
            rate_limits: None,
            warnings: Vec::new(),
            raw: None,
        }
    }

    /// Concatenates every text part in order.
    ///
    /// Every other part is ignored, including reasoning, tool calls, and the
    /// structured [`ContentPart::Json`] and provider-native
    /// [`ContentPart::Opaque`] parts.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|part| match part {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use serde_json::json;

    use super::{FinishReason, RateLimits, Response, TokenCounts, Warning};
    use crate::catalog::{ModelId, ProviderId};
    use crate::types::{ContentPart, ToolCall};

    const TRUNCATED_TOOL_CALL: &str = "truncated_tool_call";
    fn response_with_a_call(finish_reason: FinishReason) -> Response {
        let mut response = Response::new(ProviderId::new("alpha"), ModelId::new("one"), vec![
            ContentPart::Text {
                text: "Calling".to_owned(),
            },
            ContentPart::ToolCall(ToolCall::function("call_1", "search", json!({}))),
        ]);
        response.finish_reason = finish_reason;
        response
    }

    #[test]
    fn a_length_finish_drops_its_tool_calls_and_warns() {
        let mut response = response_with_a_call(FinishReason::Length);

        response.drop_truncated_tool_calls();

        assert_eq!(response.content, vec![ContentPart::Text {
            text: "Calling".to_owned(),
        }]);
        assert_eq!(response.finish_reason, FinishReason::Length);
        assert_eq!(response.warnings.len(), 1);
        assert_eq!(response.warnings[0].code, TRUNCATED_TOOL_CALL);
        assert!(response.warnings[0].message.contains("search"));
    }

    #[test]
    fn every_other_finish_keeps_its_tool_calls() {
        for finish_reason in [
            FinishReason::Stop,
            FinishReason::ToolCall,
            FinishReason::ContentFilter,
            FinishReason::Incomplete,
            FinishReason::Other("cancelled".to_owned()),
        ] {
            let mut response = response_with_a_call(finish_reason.clone());
            response.drop_truncated_tool_calls();
            assert_eq!(response.content.len(), 2, "{finish_reason:?}");
            assert!(response.warnings.is_empty(), "{finish_reason:?}");
        }
    }

    fn sample_response() -> Response {
        Response::new(ProviderId::new("openai"), ModelId::new("gpt-5"), vec![
            ContentPart::Text {
                text: "hello".to_owned(),
            },
        ])
    }

    #[test]
    fn every_finish_reason_round_trips_as_one_string() -> Result<(), Box<dyn StdError>> {
        let reasons = [
            (FinishReason::Stop, "stop"),
            (FinishReason::Length, "length"),
            (FinishReason::ToolCall, "tool_call"),
            (FinishReason::ContentFilter, "content_filter"),
            (FinishReason::Error, "error"),
            (FinishReason::Incomplete, "incomplete"),
        ];

        for (reason, wire) in reasons {
            assert_eq!(serde_json::to_value(&reason)?, json!(wire));
            assert_eq!(serde_json::from_value::<FinishReason>(json!(wire))?, reason);
        }
        Ok(())
    }

    #[test]
    fn an_unknown_finish_reason_deserializes_as_other() -> Result<(), Box<dyn StdError>> {
        let reason = serde_json::from_value::<FinishReason>(json!("guardrail_intervened"))?;

        assert_eq!(
            reason,
            FinishReason::Other("guardrail_intervened".to_owned())
        );
        Ok(())
    }

    #[test]
    fn noncanonical_finish_reasons_and_warnings_are_not_converted() {
        assert_eq!(
            FinishReason::from("tool_calls"),
            FinishReason::Other("tool_calls".into())
        );
        assert!(serde_json::from_value::<FinishReason>(json!({ "other": "incomplete" })).is_err());
        assert!(
            serde_json::from_value::<Warning>(json!({ "code": null, "message": "dropped" }))
                .is_err()
        );
    }

    #[test]
    fn a_finish_reason_object_naming_another_field_is_rejected() {
        let error = serde_json::from_value::<FinishReason>(json!({ "stop": "yes" })).unwrap_err();

        assert!(
            error.to_string().contains("finish reason string"),
            "{error}"
        );
    }

    #[test]
    fn total_counts_every_bucket_exactly_once() {
        let usage = TokenCounts {
            input:       10,
            output:      20,
            reasoning:   30,
            cache_read:  40,
            cache_write: 50,
        };

        assert_eq!(usage.total(), 150);
    }

    #[test]
    fn detail_counters_are_clamped_to_their_parent_totals() {
        // A skin that counts reasoning exclusive of completion reports 66
        // reasoning tokens against a completion total of 59. The old library
        // pinned this clamp; without it the total and the billable output
        // inflate past what the parent counters claim.
        let usage = TokenCounts::from_inclusive(0, 59, 66, 0, 0);
        assert_eq!(usage.output, 0);
        assert_eq!(usage.reasoning, 59);
        assert_eq!(usage.total(), 59);

        // The cache counters share the input total the same way.
        let usage = TokenCounts::from_inclusive(100, 0, 0, 80, 50);
        assert_eq!(usage.cache_read, 80);
        assert_eq!(usage.cache_write, 20);
        assert_eq!(usage.input, 0);
        assert_eq!(usage.total(), 100);
    }

    #[test]
    fn billable_output_is_output_plus_reasoning() {
        let usage = TokenCounts {
            output: 20,
            reasoning: 30,
            ..TokenCounts::default()
        };

        assert_eq!(usage.billable_output(), 50);
    }

    #[test]
    fn token_counts_reject_noncanonical_fields() {
        assert!(serde_json::from_value::<TokenCounts>(json!({ "input_tokens": 100 })).is_err());
    }

    #[test]
    fn from_inclusive_produces_disjoint_buckets() {
        let usage = TokenCounts::from_inclusive(100, 60, 25, 30, 10);

        assert_eq!(usage, TokenCounts {
            input:       60,
            output:      35,
            reasoning:   25,
            cache_read:  30,
            cache_write: 10,
        });
        assert_eq!(usage.total(), 160);
    }

    #[test]
    fn from_inclusive_cannot_underflow() {
        // Every detail counter exceeds its parent. Nothing wraps, and the
        // clamps keep the total at what the parent counters claim: 5 + 3.
        let usage = TokenCounts::from_inclusive(5, 3, 9, 7, 8);

        assert_eq!(usage, TokenCounts {
            input:       0,
            output:      0,
            reasoning:   3,
            cache_read:  5,
            cache_write: 0,
        });
        assert_eq!(usage.total(), 8);
    }

    #[test]
    fn rate_limits_keep_request_and_token_resets_separate() -> Result<(), Box<dyn StdError>> {
        let limits = RateLimits {
            request_limit:     Some(500),
            request_remaining: Some(499),
            request_reset:     Some("1s".to_owned()),
            token_limit:       Some(30_000),
            token_remaining:   Some(29_000),
            token_reset:       Some("6m0s".to_owned()),
        };

        let encoded = serde_json::to_value(&limits)?;

        assert_eq!(encoded["request_reset"], json!("1s"));
        assert_eq!(encoded["token_reset"], json!("6m0s"));
        assert_eq!(serde_json::from_value::<RateLimits>(encoded)?, limits);
        Ok(())
    }

    #[test]
    fn response_round_trips_without_raw() -> Result<(), Box<dyn StdError>> {
        let response = sample_response();

        let encoded = serde_json::to_value(&response)?;

        assert!(encoded.get("raw").is_none());
        assert_eq!(serde_json::from_value::<Response>(encoded)?, response);
        Ok(())
    }

    #[test]
    fn response_round_trips_with_raw() -> Result<(), Box<dyn StdError>> {
        let mut response = sample_response();
        response.raw = Some(json!({ "id": "resp_1", "object": "response" }));
        response.usage = TokenCounts::from_inclusive(100, 60, 25, 30, 10);

        let encoded = serde_json::to_value(&response)?;

        assert_eq!(encoded["raw"]["id"], json!("resp_1"));
        assert_eq!(serde_json::from_value::<Response>(encoded)?, response);
        Ok(())
    }

    #[test]
    fn text_ignores_json_and_opaque_parts() {
        let mut response = sample_response();
        response.content.push(ContentPart::Json {
            value: json!({ "ok": true }),
        });
        response.content.push(ContentPart::opaque(
            "openai.reasoning",
            json!({ "id": "rs_1" }),
        ));
        response.content.push(ContentPart::Text {
            text: " world".to_owned(),
        });

        assert_eq!(response.text(), "hello world");
    }
}
