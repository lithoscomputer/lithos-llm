use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::ContentPart;
use crate::catalog::{ModelHandle, ModelId, ProviderId};

/// Why a model stopped producing output.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum FinishReason {
    Stop,
    Length,
    ToolCall,
    ContentFilter,
    Error,
    Other(String),
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
    /// Subtraction saturates, so a provider that reports a detail counter
    /// larger than its own total yields zero rather than wrapping.
    pub fn from_inclusive(
        input: u64,
        output: u64,
        reasoning: u64,
        cache_read: u64,
        cache_write: u64,
    ) -> Self {
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

    use super::{RateLimits, Response, TokenCounts};
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
    fn billable_output_is_output_plus_reasoning() {
        let usage = TokenCounts {
            output: 20,
            reasoning: 30,
            ..TokenCounts::default()
        };

        assert_eq!(usage.billable_output(), 50);
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
        let usage = TokenCounts::from_inclusive(5, 3, 9, 7, 8);

        assert_eq!(usage, TokenCounts {
            input:       0,
            output:      0,
            reasoning:   9,
            cache_read:  7,
            cache_write: 8,
        });
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
