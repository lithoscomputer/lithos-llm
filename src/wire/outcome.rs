//! What a response reported about how it ended, which model served it, and
//! what it was billed for.
//!
//! Each wire format builds an [`Outcome`] from a whole response with its
//! `Response::outcome`, or folds a stream into one event by event with
//! `StreamEvent::fold_into`. The view is provider-neutral, but every value in
//! it is the provider's own: the model as the provider named it, its stop
//! reason as it spelled it, and its own refusal category. The rules that
//! interpret those values — which stop reasons mean a content filter, which
//! service tier the catalog prices as which [`Speed`], what the provider
//! reported it charged, and which passes of a request were billed — are the
//! same ones the client's codecs apply.

use serde_json::{Map, Value};

use crate::cost::usd_micros;
use crate::types::{Cost, CostSource, FinishReason, Speed, TokenCounts, promote_tool_finish};

/// How a response ended, which model served it, and what it was billed for.
///
/// Fields stay `None` until the response or stream reports them. A stream
/// folded only partway, such as one the client abandoned, keeps what it
/// reached.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Outcome {
    model:         Option<String>,
    stop_reason:   Option<String>,
    finish_reason: Option<FinishReason>,
    has_tool_call: bool,
    refusal:       Option<Refusal>,
    fallbacks:     Vec<Fallback>,
    error:         Option<InBandError>,
    usage:         Option<TokenCounts>,
    iterations:    Vec<Iteration>,
    speed:         Option<String>,
    service_tier:  Option<String>,
    provider_cost: Option<Cost>,
    rules:         Rules,
}

/// A provider declining to answer.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub struct Refusal {
    /// The provider's policy category, such as Anthropic's `cyber`, when it
    /// names one.
    pub category:    Option<String>,
    /// The provider's or the model's own account of the refusal. Its wording
    /// is not stable, and it may quote the request.
    pub explanation: Option<String>,
}

/// One handoff of Anthropic's server-side fallback, from a model that
/// declined to the model that continued.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct Fallback {
    pub from:     String,
    pub to:       String,
    /// The declining model's refusal category.
    pub category: Option<String>,
}

/// An error a provider reported inside a successful response or a stream
/// that already started, rather than as an error status.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct InBandError {
    /// Anthropic's error `type` or OpenAI's error `code`, such as
    /// `overloaded_error` or `cyber_policy`.
    pub code:    Option<String>,
    pub message: String,
}

/// One sampling pass of a request, with the tokens it used.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct Pass {
    /// The model that ran the pass, as the provider named it; `None` for a
    /// pass the provider reports without one, which ran on the requested
    /// model.
    pub model:  Option<String>,
    pub usage:  TokenCounts,
    /// Whether the provider bills this pass. Anthropic does not bill some
    /// refusals that arrive before any output.
    pub billed: bool,
}

/// How the catalog prices the service tier a response was served at.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ServedTier {
    /// The base rates.
    Standard,
    /// The rates of a catalog speed.
    Speed(Speed),
    /// A tier no catalog speed stands for, such as OpenAI's `scale` or
    /// Anthropic's Priority Tier, which the catalog cannot price.
    Unpriced,
}

/// Which provider's billing and tier rules apply.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum Rules {
    #[default]
    OpenAi,
    Anthropic,
}

/// One entry of Anthropic's `usage.iterations`.
#[derive(Clone, Debug, Eq, PartialEq)]
struct Iteration {
    kind:  String,
    model: Option<String>,
    usage: TokenCounts,
}

/// Anthropic's refusal categories that are billed even when the refusal
/// arrives before any output, as of September 2026: `bio`, `frontier_llm`,
/// and `reasoning_extraction`. A refusal before any output in another
/// category, or in none, is not billed.
const ANTHROPIC_BILLED_REFUSALS: [&str; 3] = ["bio", "frontier_llm", "reasoning_extraction"];

impl Outcome {
    /// The model that produced the response, as the provider named it. After
    /// a fallback or a provider's reroute it is not the model requested.
    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    /// The provider's own stop reason: Anthropic's `stop_reason`, Chat's
    /// `finish_reason`, or a Responses `status`, with an `incomplete`
    /// response's `incomplete_details.reason` in its place.
    pub fn stop_reason(&self) -> Option<&str> {
        self.stop_reason.as_deref()
    }

    /// The stop reason normalized as the client normalizes it, a turn of
    /// nothing but tool calls included.
    pub fn finish_reason(&self) -> Option<FinishReason> {
        self.finish_reason
            .clone()
            .map(|reason| promote_tool_finish(reason, self.has_tool_call))
    }

    /// The refusal the response ended in, if it ended in one: Anthropic's
    /// `refusal` stop reason, a Responses `refusal` part, or Chat's
    /// `refusal`. A content filter that cut the output short is a
    /// [`FinishReason::ContentFilter`] instead.
    pub fn refusal(&self) -> Option<&Refusal> {
        self.refusal.as_ref()
    }

    /// Each server-side fallback handoff, in order.
    pub fn fallbacks(&self) -> &[Fallback] {
        &self.fallbacks
    }

    /// An error reported inside the response or stream.
    pub fn error(&self) -> Option<&InBandError> {
        self.error.as_ref()
    }

    /// The token counts the response reported beside its passes. After a
    /// fallback or a compaction they leave passes out; see
    /// [`passes`](Self::passes).
    pub fn usage(&self) -> Option<TokenCounts> {
        self.usage
    }

    /// Anthropic's `usage.speed`, such as `fast`.
    pub fn speed(&self) -> Option<&str> {
        self.speed.as_deref()
    }

    /// The service tier the provider reports serving the response at.
    pub fn service_tier(&self) -> Option<&str> {
        self.service_tier.as_deref()
    }

    /// How the catalog prices the tier the response was served at. A
    /// response that reports no tier was served at the standard one.
    pub fn served_tier(&self) -> ServedTier {
        let speed = self.speed.as_deref();
        let tier = self.service_tier.as_deref();
        match self.rules {
            Rules::Anthropic => match (speed, tier) {
                (None | Some("standard"), None | Some("standard")) => ServedTier::Standard,
                (Some("fast"), None | Some("standard")) => ServedTier::Speed(Speed::Fast),
                (Some("ultrafast"), None | Some("standard")) => ServedTier::Speed(Speed::Ultrafast),
                _ => ServedTier::Unpriced,
            },
            // The client requests `priority` for `Speed::Fast` and `flex`
            // for `Speed::Economical`; `fast` is `priority` renamed.
            Rules::OpenAi => match tier {
                None | Some("default") => ServedTier::Standard,
                Some("ultrafast") => ServedTier::Speed(Speed::Ultrafast),
                Some("priority" | "fast") => ServedTier::Speed(Speed::Fast),
                Some("flex") => ServedTier::Speed(Speed::Economical),
                Some(_) => ServedTier::Unpriced,
            },
        }
    }

    /// What the provider reported charging, when it reports a charge:
    /// OpenRouter's `usage.cost`, or Venice's `cost`.
    pub fn provider_cost(&self) -> Option<Cost> {
        self.provider_cost
    }

    /// Each sampling pass the request ran, in order, with whether it was
    /// billed. Together they are everything the request was billed for.
    ///
    /// Anthropic lists its passes in `usage.iterations` when a request falls
    /// back to another model or compacts its context; each declined pass is
    /// matched to its fallback's refusal category. Otherwise the response is
    /// one pass by the model that served it. A refusal before any output is
    /// not billed when Anthropic does not bill its category.
    pub fn passes(&self) -> Vec<Pass> {
        let anthropic = self.rules == Rules::Anthropic;
        let billed = |usage: &TokenCounts, declined: Option<Option<&str>>| match declined {
            Some(category) if anthropic && usage.output == 0 => {
                category.is_some_and(|category| ANTHROPIC_BILLED_REFUSALS.contains(&category))
            }
            _ => true,
        };
        if self.iterations.is_empty() {
            let Some(usage) = self.usage else {
                return Vec::new();
            };
            let declined = self
                .refusal
                .as_ref()
                .map(|refusal| refusal.category.as_deref());
            return vec![Pass {
                model: self.model.clone(),
                usage,
                billed: billed(&usage, declined),
            }];
        }
        let sampled = self
            .iterations
            .iter()
            .filter(|iteration| iteration.kind != "compaction")
            .count();
        let mut position = 0;
        self.iterations
            .iter()
            .map(|iteration| {
                if iteration.kind == "compaction" {
                    return Pass {
                        model:  iteration.model.clone(),
                        usage:  iteration.usage,
                        billed: true,
                    };
                }
                let declined = if let Some(fallback) = self.fallbacks.get(position) {
                    Some(fallback.category.as_deref())
                } else if position + 1 == sampled {
                    self.refusal
                        .as_ref()
                        .map(|refusal| refusal.category.as_deref())
                } else {
                    None
                };
                position += 1;
                Pass {
                    model:  iteration.model.clone(),
                    usage:  iteration.usage,
                    billed: billed(&iteration.usage, declined),
                }
            })
            .collect()
    }

    pub(super) fn anthropic(&mut self) -> &mut Self {
        self.rules = Rules::Anthropic;
        self
    }

    pub(super) fn set_model(&mut self, model: &str) {
        self.model = Some(model.to_owned());
    }

    /// Records a stop reason and its normalized finish reason.
    pub(super) fn set_stop(&mut self, raw: &str, finish: FinishReason) {
        self.stop_reason = Some(raw.to_owned());
        self.finish_reason = Some(finish);
    }

    pub(super) fn saw_tool_call(&mut self) {
        self.has_tool_call = true;
    }

    pub(super) fn set_refusal(&mut self, refusal: Refusal) {
        self.refusal = Some(refusal);
    }

    /// Appends streamed refusal text, as Chat sends it in pieces.
    pub(super) fn append_refusal(&mut self, text: &str) {
        let refusal = self.refusal.get_or_insert_with(Refusal::default);
        refusal
            .explanation
            .get_or_insert_with(String::new)
            .push_str(text);
    }

    pub(super) fn push_fallback(&mut self, fallback: Fallback) {
        self.model = Some(fallback.to.clone());
        self.fallbacks.push(fallback);
    }

    pub(super) fn set_error(&mut self, code: Option<&str>, message: &str) {
        self.error = Some(InBandError {
            code:    code.map(ToOwned::to_owned),
            message: message.to_owned(),
        });
    }

    pub(super) fn usage_mut(&mut self) -> &mut TokenCounts {
        self.usage.get_or_insert_with(TokenCounts::default)
    }

    pub(super) fn set_usage(&mut self, usage: TokenCounts) {
        self.usage = Some(usage);
    }

    pub(super) fn set_iterations(
        &mut self,
        iterations: impl IntoIterator<Item = (String, Option<String>, TokenCounts)>,
    ) {
        self.iterations = iterations
            .into_iter()
            .map(|(kind, model, usage)| Iteration { kind, model, usage })
            .collect();
    }

    /// Records the tier fields an object reported; one it leaves out keeps
    /// the earlier value.
    pub(super) fn set_tier(&mut self, speed: Option<&Value>, service_tier: Option<&Value>) {
        if let Some(speed) = speed.and_then(Value::as_str) {
            self.speed = Some(speed.to_owned());
        }
        if let Some(tier) = service_tier.and_then(Value::as_str) {
            self.service_tier = Some(tier.to_owned());
        }
    }

    /// Records the cost an object carrying `usage` and `cost` reported.
    pub(super) fn set_provider_cost(
        &mut self,
        usage: Option<&Map<String, Value>>,
        cost: Option<&Value>,
    ) {
        if let Some(cost) = provider_cost(usage, cost) {
            self.provider_cost = Some(cost);
        }
    }
}

/// The charge a provider reported in-band, in precedence order:
///
/// - OpenRouter's `usage.cost`, in credits, which are dollars. With the
///   caller's own key (`usage.is_byok`) it is only OpenRouter's fee, so
///   `usage.cost_details.upstream_inference_cost`, the provider's charge, is
///   added.
/// - Venice's top-level `cost`, split by currency: `usd`, and `diem`, which
///   Venice values at one dollar each.
///
/// A negative or non-finite figure is not a cost.
pub(crate) fn provider_cost(
    usage: Option<&Map<String, Value>>,
    cost: Option<&Value>,
) -> Option<Cost> {
    let usd = usage
        .and_then(openrouter_usd)
        .or_else(|| cost.and_then(venice_usd))?;
    (usd.is_finite() && usd >= 0.0).then(|| Cost {
        usd_micros: usd_micros(usd),
        source:     CostSource::Provider,
    })
}

fn openrouter_usd(usage: &Map<String, Value>) -> Option<f64> {
    let mut usd = usage.get("cost")?.as_f64()?;
    if usage.get("is_byok").and_then(Value::as_bool) == Some(true) {
        usd += usage
            .get("cost_details")
            .and_then(|details| details.get("upstream_inference_cost"))
            .and_then(Value::as_f64)
            .unwrap_or_default();
    }
    Some(usd)
}

fn venice_usd(cost: &Value) -> Option<f64> {
    let usd = cost.get("usd").and_then(Value::as_f64);
    let diem = cost.get("diem").and_then(Value::as_f64);
    if usd.is_none() && diem.is_none() {
        return None;
    }
    Some(usd.unwrap_or_default() + diem.unwrap_or_default())
}
