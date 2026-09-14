use std::fmt;

use serde::de::{Error as DeError, IgnoredAny, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use super::{ContentPart, Error, ErrorKind, Message, Role, ToolCall};
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
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
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

impl<'de> Deserialize<'de> for TokenCounts {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct CountsVisitor;
        impl<'de> Visitor<'de> for CountsVisitor {
            type Value = TokenCounts;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("token usage with five disjoint nonnegative buckets")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<TokenCounts, A::Error> {
                let mut counts = TokenCounts::default();
                let mut seen = 0u8;
                while let Some(key) = map.next_key::<String>()? {
                    let (field, bit, name) = match key.as_str() {
                        "input" => (&mut counts.input, 1, "input"),
                        "output" => (&mut counts.output, 2, "output"),
                        "reasoning" => (&mut counts.reasoning, 4, "reasoning"),
                        "cache_read" => (&mut counts.cache_read, 8, "cache_read"),
                        "cache_write" => (&mut counts.cache_write, 16, "cache_write"),
                        "input_tokens" | "output_tokens" | "reasoning_tokens"
                        | "cache_read_tokens" | "cache_write_tokens" => {
                            // These are historical bucket names, not additive metadata.
                            // Ignoring them would turn real usage into zero tokens.
                            return Err(A::Error::custom(
                                "legacy usage fields require application conversion",
                            ));
                        }
                        _ => {
                            map.next_value::<IgnoredAny>()?;
                            continue;
                        }
                    };
                    if seen & bit != 0 {
                        return Err(A::Error::duplicate_field(name));
                    }
                    seen |= bit;
                    *field = map.next_value()?;
                }
                Ok(counts)
            }
        }
        deserializer.deserialize_map(CountsVisitor)
    }
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

    /// Adds the two counts bucket by bucket, saturating at `u64::MAX`.
    #[must_use]
    pub fn saturating_add(self, other: Self) -> Self {
        Self {
            input:       self.input.saturating_add(other.input),
            output:      self.output.saturating_add(other.output),
            reasoning:   self.reasoning.saturating_add(other.reasoning),
            cache_read:  self.cache_read.saturating_add(other.cache_read),
            cache_write: self.cache_write.saturating_add(other.cache_write),
        }
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

/// Token counts and, when known, what they cost.
///
/// `cost` is `None` when there is no cost data, never a price of zero. A
/// response's own [`Response::usage`] and [`Response::cost`] pair up this way
/// through [`Response::usage_with_cost`]; totals across calls come from
/// [`Usage::saturating_add`].
///
/// The wire shape is the two fields side by side, with `cost` omitted when it
/// is `None`:
///
/// ```
/// use lithos_llm::types::{Cost, CostSource, TokenCounts, Usage};
/// use serde_json::json;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let usage = Usage {
///     tokens: TokenCounts {
///         input:       28_640,
///         output:      8_750,
///         reasoning:   1_200,
///         cache_read:  4_800,
///         cache_write: 1_500,
///     },
///     cost:   Some(Cost {
///         usd_micros: 720_000,
///         source:     CostSource::Catalog,
///     }),
/// };
///
/// assert_eq!(
///     serde_json::to_value(usage)?,
///     json!({
///         "tokens": { "input": 28640, "output": 8750, "reasoning": 1200,
///                     "cache_read": 4800, "cache_write": 1500 },
///         "cost": { "usd_micros": 720000, "source": "catalog" }
///     })
/// );
/// assert_eq!(serde_json::from_value::<Usage>(json!({}))?, Usage::default());
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct Usage {
    #[serde(default)]
    pub tokens: TokenCounts,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost:   Option<Cost>,
}

impl Usage {
    /// Adds the two usages: tokens bucket by bucket, saturating, and cost only
    /// when every part that used tokens is priced.
    ///
    /// A total cost is known only when every part is priced, so a part that
    /// used tokens without a cost makes the sum's cost `None` rather than a
    /// figure that under-reports the whole. A part with no tokens and no cost
    /// used nothing and adds nothing, which keeps [`Usage::default`] the
    /// identity for a fold. Two priced parts keep their shared source, or
    /// take [`CostSource::Application`] when the sources differ, because the
    /// caller assembled the sum rather than either source reporting it.
    #[must_use]
    pub fn saturating_add(self, other: Self) -> Self {
        let cost = match (self.cost, other.cost) {
            (Some(left), Some(right)) => Some(Cost {
                usd_micros: left.usd_micros.saturating_add(right.usd_micros),
                source:     if left.source == right.source {
                    left.source
                } else {
                    CostSource::Application
                },
            }),
            (cost, None) if other.is_empty() => cost,
            (None, cost) if self.is_empty() => cost,
            _ => None,
        };
        Self {
            tokens: self.tokens.saturating_add(other.tokens),
            cost,
        }
    }

    /// The sum of the five token buckets, as [`TokenCounts::total`].
    pub fn total_tokens(self) -> u64 {
        self.tokens.total()
    }

    /// No tokens and no cost data: the usage of something that did not run.
    fn is_empty(self) -> bool {
        self.tokens == TokenCounts::default() && self.cost.is_none()
    }
}

impl From<TokenCounts> for Usage {
    /// The counts with no cost data.
    fn from(tokens: TokenCounts) -> Self {
        Self { tokens, cost: None }
    }
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
    pub id:                    Option<String>,
    pub model:                 ModelHandle,
    pub content:               Vec<ContentPart>,
    /// Calls withheld because the turn ended with `Length` or `Incomplete`.
    /// These retain raw arguments and replay metadata for diagnostics only.
    /// They must not be executed or automatically replayed as assistant
    /// content.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub suppressed_tool_calls: Vec<ToolCall>,
    pub finish_reason:         FinishReason,
    pub usage:                 TokenCounts,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost:                  Option<Cost>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_limits:           Option<RateLimits>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings:              Vec<Warning>,
    /// The complete provider success payload, when one was available.
    ///
    /// A complete (non-streaming) response holds the whole JSON success body
    /// exactly as the provider sent it. A streamed response holds the final
    /// provider response object when the streaming protocol supplies one, and
    /// is `None` otherwise. It never holds an accumulated log of stream events.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw:                   Option<Value>,
}

impl Response {
    #[cfg(any(feature = "runtime", test))]
    /// Withholds every tool call from an unfinished turn, preserving
    /// diagnostics. This is idempotent across codec, adapter, and client
    /// boundaries.
    pub(crate) fn suppress_unfinished_tool_calls(&mut self) {
        if !self.is_unfinished() {
            return;
        }
        let mut content = Vec::with_capacity(self.content.len());
        for part in self.content.drain(..) {
            if let ContentPart::ToolCall(call) = part {
                self.warnings.push(Warning {
                    code: "truncated_tool_call".to_owned(),
                    message: format!(
                        "the turn was unfinished ({reason}); the call to {name} was withheld from execution",
                        reason = self.finish_reason.as_str(), name = call.name,
                    ),
                });
                self.suppressed_tool_calls.push(call);
            } else {
                content.push(part);
            }
        }
        self.content = content;
    }

    fn is_unfinished(&self) -> bool {
        matches!(
            self.finish_reason,
            FinishReason::Length | FinishReason::Incomplete
        )
    }

    /// Visits the final response's tool calls in content order.
    ///
    /// An unfinished turn (`Length` or `Incomplete`) yields no calls, even for
    /// an application-constructed response that has not passed client policy.
    /// Callers still validate arguments and the finish reason before execution.
    pub fn tool_calls(&self) -> impl Iterator<Item = &ToolCall> {
        self.content.iter().filter_map(|part| match part {
            ContentPart::ToolCall(call) if !self.is_unfinished() => Some(call),
            _ => None,
        })
    }

    /// Moves all content, including provider replay data, into an assistant
    /// message. Record response usage and other accounting fields before
    /// consuming it.
    pub fn into_message(self) -> Message {
        Message::new(Role::Assistant, self.content)
    }

    /// The response's token counts and cost as one [`Usage`].
    pub fn usage_with_cost(&self) -> Usage {
        Usage {
            tokens: self.usage,
            cost:   self.cost,
        }
    }

    pub fn new(provider: ProviderId, model: ModelId, content: Vec<ContentPart>) -> Self {
        Self {
            id: None,
            model: ModelHandle::new(provider, model),
            content,
            suppressed_tool_calls: Vec::new(),
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

    /// The JSON document a structured-output request asked for.
    ///
    /// A provider that returns structured output as its own content type
    /// yields a [`ContentPart::Json`] part, which is returned as is. Every
    /// other provider returns the document as text, which is parsed here.
    ///
    /// # Errors
    ///
    /// Returns a `ResponseDecode` error when the response carries no JSON part
    /// and its text is not a JSON document.
    pub fn json_object(&self) -> Result<Value, Error> {
        if let Some(value) = self.content.iter().find_map(|part| match part {
            ContentPart::Json { value } => Some(value.clone()),
            _ => None,
        }) {
            return Ok(value);
        }
        let text = self.text();
        serde_json::from_str(text.trim()).map_err(|source| {
            Error::new(
                ErrorKind::ResponseDecode,
                format!("the model did not return a JSON document: {source}"),
            )
            .with_provider(self.model.provider().clone())
            .with_source(source)
        })
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use serde_json::json;

    use super::{
        Cost, CostSource, ErrorKind, FinishReason, RateLimits, Response, TokenCounts, Usage,
        Warning,
    };
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

        response.suppress_unfinished_tool_calls();

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
            FinishReason::Other("cancelled".to_owned()),
        ] {
            let mut response = response_with_a_call(finish_reason.clone());
            response.suppress_unfinished_tool_calls();
            assert_eq!(response.content.len(), 2, "{finish_reason:?}");
            assert!(response.warnings.is_empty(), "{finish_reason:?}");
        }
    }

    #[test]
    fn unfinished_turns_preserve_calls_only_as_diagnostics() {
        for reason in [FinishReason::Length, FinishReason::Incomplete] {
            let mut response = response_with_a_call(reason);
            let call = ToolCall::custom("partial", "edit", "unfinished patch");
            response.content.push(ContentPart::ToolCall(call.clone()));
            assert_eq!(response.tool_calls().count(), 0);
            response.suppress_unfinished_tool_calls();
            response.suppress_unfinished_tool_calls();
            assert_eq!(response.suppressed_tool_calls.len(), 2);
            assert_eq!(response.suppressed_tool_calls[1], call);
            assert_eq!(response.warnings.len(), 2);
            assert!(
                response
                    .warnings
                    .iter()
                    .all(|w| w.message.contains("turn was unfinished"))
            );
            let saved = serde_json::to_value(&response).expect("serialize diagnostics");
            let restored: Response = serde_json::from_value(saved).expect("restore diagnostics");
            assert_eq!(response, restored);
            assert_eq!(restored.tool_calls().count(), 0);
            assert!(
                restored
                    .into_message()
                    .content()
                    .iter()
                    .all(|p| !matches!(p, ContentPart::ToolCall(_)))
            );
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
    fn additional_usage_fields_do_not_change_the_five_buckets() {
        let usage: TokenCounts = serde_json::from_value(json!({
            "input": 10, "output": 20, "reasoning": 3, "cache_read": 4, "cache_write": 5,
            "future_metadata": {"estimated": true}, "new_counter": 100,
        }))
        .expect("additive fields");
        assert_eq!(usage.total(), 42);
        assert_eq!(usage.billable_output(), 23);
        assert!(serde_json::from_str::<TokenCounts>(r#"{"input":1,"input":2}"#).is_err());
        assert!(serde_json::from_value::<TokenCounts>(json!({"input":-1})).is_err());
    }

    #[test]
    fn token_counts_reject_legacy_bucket_names() {
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
    fn saturating_add_adds_every_bucket_without_wrapping() {
        let left = TokenCounts {
            input:       1,
            output:      2,
            reasoning:   3,
            cache_read:  4,
            cache_write: u64::MAX,
        };
        let right = TokenCounts {
            input:       10,
            output:      20,
            reasoning:   30,
            cache_read:  40,
            cache_write: 1,
        };

        assert_eq!(left.saturating_add(right), TokenCounts {
            input:       11,
            output:      22,
            reasoning:   33,
            cache_read:  44,
            cache_write: u64::MAX,
        });
    }

    fn priced(input: u64, usd_micros: u64, source: CostSource) -> Usage {
        Usage {
            tokens: TokenCounts {
                input,
                ..TokenCounts::default()
            },
            cost:   Some(Cost { usd_micros, source }),
        }
    }

    fn unpriced(input: u64) -> Usage {
        Usage::from(TokenCounts {
            input,
            ..TokenCounts::default()
        })
    }

    #[test]
    fn usage_round_trips_with_and_without_cost() -> Result<(), Box<dyn StdError>> {
        let with_cost = priced(10, 250, CostSource::Provider);
        let encoded = serde_json::to_value(with_cost)?;
        assert_eq!(
            encoded,
            json!({
                "tokens": { "input": 10, "output": 0, "reasoning": 0,
                            "cache_read": 0, "cache_write": 0 },
                "cost": { "usd_micros": 250, "source": "provider" }
            })
        );
        assert_eq!(serde_json::from_value::<Usage>(encoded)?, with_cost);

        let without_cost = unpriced(10);
        let encoded = serde_json::to_value(without_cost)?;
        assert!(encoded.get("cost").is_none());
        assert_eq!(serde_json::from_value::<Usage>(encoded)?, without_cost);
        Ok(())
    }

    #[test]
    fn an_empty_usage_object_is_the_default() -> Result<(), Box<dyn StdError>> {
        assert_eq!(
            serde_json::from_value::<Usage>(json!({}))?,
            Usage::default()
        );
        assert_eq!(
            serde_json::from_value::<Usage>(json!({ "tokens": {} }))?,
            Usage::default()
        );
        assert_eq!(
            serde_json::to_value(Usage::default())?,
            json!({
                "tokens": { "input": 0, "output": 0, "reasoning": 0,
                            "cache_read": 0, "cache_write": 0 }
            })
        );
        Ok(())
    }

    #[test]
    fn usage_add_keeps_a_shared_cost_source() {
        let sum = priced(10, 250, CostSource::Catalog).saturating_add(priced(
            5,
            u64::MAX,
            CostSource::Catalog,
        ));

        assert_eq!(sum, priced(15, u64::MAX, CostSource::Catalog));
    }

    #[test]
    fn usage_add_marks_mixed_cost_sources_as_application() {
        let sum = priced(10, 250, CostSource::Catalog).saturating_add(priced(
            5,
            100,
            CostSource::Provider,
        ));

        assert_eq!(sum, priced(15, 350, CostSource::Application));
    }

    #[test]
    fn usage_add_has_no_cost_when_a_part_with_tokens_is_unpriced() {
        assert_eq!(
            priced(10, 250, CostSource::Catalog).saturating_add(unpriced(5)),
            unpriced(15)
        );
        assert_eq!(
            unpriced(5).saturating_add(priced(10, 250, CostSource::Catalog)),
            unpriced(15)
        );
        assert_eq!(unpriced(5).saturating_add(unpriced(10)), unpriced(15));
    }

    #[test]
    fn usage_add_treats_the_default_as_identity() {
        let usage = priced(10, 250, CostSource::Provider);

        assert_eq!(Usage::default().saturating_add(usage), usage);
        assert_eq!(usage.saturating_add(Usage::default()), usage);
        assert_eq!(
            [usage, priced(1, 1, CostSource::Provider)]
                .into_iter()
                .fold(Usage::default(), Usage::saturating_add),
            priced(11, 251, CostSource::Provider)
        );
    }

    #[test]
    fn usage_from_counts_has_no_cost_and_the_same_total() {
        let counts = TokenCounts::from_inclusive(100, 60, 25, 30, 10);

        let usage = Usage::from(counts);

        assert_eq!(usage, Usage {
            tokens: counts,
            cost:   None,
        });
        assert_eq!(usage.total_tokens(), counts.total());
    }

    #[test]
    fn a_response_pairs_its_usage_and_cost() {
        let mut response = sample_response();
        assert_eq!(response.usage_with_cost(), Usage::default());

        response.usage = TokenCounts::from_inclusive(100, 60, 25, 30, 10);
        response.cost = Some(Cost {
            usd_micros: 720_000,
            source:     CostSource::Catalog,
        });

        assert_eq!(response.usage_with_cost(), Usage {
            tokens: response.usage,
            cost:   response.cost,
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

    #[test]
    fn json_object_prefers_a_json_part_and_parses_text_otherwise() {
        let structured = Response::new(ProviderId::new("alpha"), ModelId::new("one"), vec![
            ContentPart::Text {
                text: "not this".to_owned(),
            },
            ContentPart::Json {
                value: json!({"city": "Paris"}),
            },
        ]);
        assert_eq!(structured.json_object().unwrap(), json!({"city": "Paris"}));

        let text = Response::new(ProviderId::new("alpha"), ModelId::new("one"), vec![
            ContentPart::Text {
                text: " {\"city\": \"Paris\"} \n".to_owned(),
            },
        ]);
        assert_eq!(text.json_object().unwrap(), json!({"city": "Paris"}));

        let prose = Response::new(ProviderId::new("alpha"), ModelId::new("one"), vec![
            ContentPart::Text {
                text: "Paris, in France.".to_owned(),
            },
        ]);
        let error = prose.json_object().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ResponseDecode);
        assert_eq!(error.provider().map(ProviderId::as_str), Some("alpha"));
    }
}
