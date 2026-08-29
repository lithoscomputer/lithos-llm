use std::fmt;

use serde::de::{Error as DeError, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use super::ContentPart;
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
            Self::Other(reason) => reason,
        }
    }
}

impl From<&str> for FinishReason {
    /// Names the matching variant, keeping any other spelling verbatim.
    ///
    /// `tool_calls` is the spelling the predecessor library persisted, so it
    /// keeps loading as [`FinishReason::ToolCall`]; a stored tool-call
    /// response must not silently stop matching after migration.
    fn from(value: &str) -> Self {
        match value {
            "stop" => Self::Stop,
            "length" => Self::Length,
            "tool_call" | "tool_calls" => Self::ToolCall,
            "content_filter" => Self::ContentFilter,
            "error" => Self::Error,
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
        deserializer.deserialize_any(FinishReasonVisitor)
    }
}

/// Reads a finish reason from the bare string this crate writes, or from the
/// `{"other": "..."}` object an earlier version wrote for
/// [`FinishReason::Other`].
struct FinishReasonVisitor;

impl<'de> Visitor<'de> for FinishReasonVisitor {
    type Value = FinishReason;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a finish reason string")
    }

    fn visit_str<E: DeError>(self, value: &str) -> Result<Self::Value, E> {
        Ok(FinishReason::from(value))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut reason: Option<String> = None;
        while let Some(key) = map.next_key::<String>()? {
            if key != "other" {
                return Err(DeError::unknown_field(&key, &["other"]));
            }
            reason = Some(map.next_value()?);
        }
        reason
            .map(|reason| FinishReason::from(reason.as_str()))
            .ok_or_else(|| DeError::missing_field("other"))
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
pub struct TokenCounts {
    /// Prompt tokens that were neither read from nor written to a cache.
    ///
    /// The aliases load usage the reference implementation persisted under
    /// its `*_tokens` field names; without them an old document deserializes
    /// without error into all-zero buckets.
    #[serde(default, alias = "input_tokens")]
    pub input:       u64,
    /// Completion tokens that are not reasoning tokens.
    #[serde(default, alias = "output_tokens")]
    pub output:      u64,
    /// Completion tokens spent on reasoning, billed at the output rate.
    #[serde(default, alias = "reasoning_tokens")]
    pub reasoning:   u64,
    /// Prompt tokens served from a provider cache.
    #[serde(default, alias = "cache_read_tokens")]
    pub cache_read:  u64,
    /// Prompt tokens written into a provider cache.
    #[serde(default, alias = "cache_write_tokens")]
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
///
/// The predecessor library serialized an absent code as `"code": null`, so
/// deserialization folds a null into the empty string rather than refusing
/// the stored warning.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Warning {
    #[serde(deserialize_with = "null_as_empty")]
    pub code:    String,
    pub message: String,
}

fn null_as_empty<'de, D: Deserializer<'de>>(deserializer: D) -> Result<String, D::Error> {
    Ok(Option::<String>::deserialize(deserializer)?.unwrap_or_default())
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
    use crate::types::ContentPart;

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
            (FinishReason::Other("incomplete".to_owned()), "incomplete"),
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
    fn the_predecessors_tool_calls_spelling_still_deserializes() -> Result<(), Box<dyn StdError>> {
        // The old library persisted "tool_calls"; a stored tool-call response
        // must keep matching `ToolCall` after migration.
        let reason = serde_json::from_value::<FinishReason>(json!("tool_calls"))?;

        assert_eq!(reason, FinishReason::ToolCall);
        Ok(())
    }

    #[test]
    fn a_stored_warning_with_a_null_code_still_deserializes() -> Result<(), Box<dyn StdError>> {
        let warning =
            serde_json::from_value::<Warning>(json!({ "code": null, "message": "dropped" }))?;

        assert_eq!(warning.code, "");
        assert_eq!(warning.message, "dropped");
        Ok(())
    }

    #[test]
    fn the_earlier_other_object_still_deserializes() -> Result<(), Box<dyn StdError>> {
        let reason = serde_json::from_value::<FinishReason>(json!({ "other": "incomplete" }))?;

        assert_eq!(reason, FinishReason::Other("incomplete".to_owned()));
        Ok(())
    }

    #[test]
    fn a_finish_reason_object_naming_another_field_is_rejected() {
        let error = serde_json::from_value::<FinishReason>(json!({ "stop": "yes" })).unwrap_err();

        assert!(error.to_string().contains("stop"), "{error}");
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
    fn usage_persisted_under_the_legacy_field_names_still_loads() {
        // The reference implementation serialized `input_tokens`-style names.
        // Every field defaults, so without the aliases an old document loads
        // without error into all-zero buckets — silent wrong usage and cost.
        let stored = serde_json::json!({
            "input_tokens": 100,
            "output_tokens": 40,
            "reasoning_tokens": 25,
            "cache_read_tokens": 30,
            "cache_write_tokens": 10,
        });

        let usage: TokenCounts = serde_json::from_value(stored).expect("legacy usage loads");

        assert_eq!(usage, TokenCounts {
            input:       100,
            output:      40,
            reasoning:   25,
            cache_read:  30,
            cache_write: 10,
        });
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
