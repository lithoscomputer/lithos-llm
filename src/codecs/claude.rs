//! The Claude reasoning policy the Anthropic and Bedrock codecs share.
//!
//! Both protocols front the same models and take the same two effort
//! dialects, so which one a request gets, how large a thinking budget is,
//! and what a forced tool choice suppresses are decided here once. Each codec
//! then inserts the plan's fields in its own wire shape and reports the
//! plan's suppressed controls, so the body and the warnings cannot disagree.
//!
//! The two protocols differ in two retained ways, both left to the caller:
//! Anthropic sends an adaptive thinking object for an effort-levels model
//! and Bedrock never does, and Bedrock sends `maxTokens` only when a budget
//! forces it.

use crate::adapter::ResolvedCall;
use crate::resolver::ResolvedRoute;
use crate::types::{ReasoningEffort, ToolChoice};

/// The `max_tokens` used when neither the request nor the catalog sets one.
///
/// Anthropic requires the field, so there is no "omit it" option. This is the
/// last resort: a request that names no limit and a model the catalog records
/// no output limit for. It is deliberately generous, because a limit picked
/// here truncates a long generation silently.
pub(crate) const DEFAULT_MAX_TOKENS: u32 = 65_536;

/// The smallest `thinking.budget_tokens` the API accepts.
///
/// Doubles as the headroom kept above the budget when the output limit must
/// grow, because the budget has to sit strictly below `max_tokens`.
pub(crate) const MIN_THINKING_BUDGET: u32 = 1024;

/// The thinking object a request sends, when it sends one.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Thinking {
    /// An explicit `budget_tokens`, for a reasoning model without effort
    /// levels.
    Budget(u32),
    /// `{"type": "adaptive"}`, for a model with effort levels. Only the
    /// Anthropic codec sends it; see the module documentation.
    Adaptive,
}

/// Everything one request's reasoning controls become on the wire.
///
/// Built once per call by [`ThinkingPlan::for_call`]. The fields are plain
/// data for the codec to insert; nothing here knows a wire shape.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ThinkingPlan {
    /// The output limit, already lifted above the budget when one is set.
    pub max_tokens:         u32,
    /// The thinking object to send, if any.
    pub thinking:           Option<Thinking>,
    /// The `output_config.effort` level to send, if any.
    pub effort:             Option<&'static str>,
    /// Whether a forced tool choice suppressed the output controls.
    ///
    /// Claude rejects extended thinking together with a forced tool choice,
    /// so a forced choice suppresses both thinking and `output_config`.
    /// `auto` and `none` leave the model free to answer in prose and keep
    /// them.
    pub forced_tool_choice: bool,
    /// The controls the forced tool choice dropped, in report order.
    ///
    /// Each is the argument of an `unsupported_control` warning. The
    /// suppressed control never reaches the model, so the drop is reported
    /// rather than silent.
    pub suppressed:         Vec<&'static str>,
}

impl ThinkingPlan {
    /// Plans the reasoning controls for one call.
    ///
    /// `raw_thinking` says a raw `thinking` provider option will replace the
    /// derived object wholesale. The recursive option merge replaces matching
    /// keys only, so a derived budget under a raw `{"type": "disabled"}` would
    /// leave a stray `budget_tokens` the API rejects; with the override the
    /// plan derives no object and keeps `max_tokens` unlifted, leaving both
    /// entirely to the caller. Claude rejects the raw option beside a forced
    /// tool choice too, so that pair is reported. Bedrock has no such option
    /// and passes `false`.
    pub(crate) fn for_call(call: &ResolvedCall, raw_thinking: bool) -> Self {
        let request = call.request();
        let route = call.route();
        let forced_tool_choice = forces_tool_use(request.tool_choice());
        let mut max_tokens = output_limit(call);
        let mut thinking = None;
        let mut effort = None;
        let mut suppressed = Vec::new();

        if forced_tool_choice {
            if request.reasoning_effort().is_some() {
                suppressed.push("reasoning effort with a forced tool choice");
            }
            if raw_thinking {
                suppressed.push("a thinking provider option with a forced tool choice");
            }
            return Self {
                max_tokens,
                thinking,
                effort,
                forced_tool_choice,
                suppressed,
            };
        }

        if !raw_thinking {
            if let Some(budget) = thinking_budget(call, max_tokens) {
                if max_tokens <= budget {
                    max_tokens = budget.saturating_add(MIN_THINKING_BUDGET);
                }
                thinking = Some(Thinking::Budget(budget));
            } else if takes_adaptive_thinking(route) {
                thinking = Some(Thinking::Adaptive);
            }
        }
        // A model without effort levels gets a thinking budget instead;
        // sending `effort` too would ask the provider to honor a control the
        // model does not take.
        if let Some(level) = request.reasoning_effort()
            && takes_effort_levels(route)
        {
            effort = Some(claude_effort(level));
        }

        Self {
            max_tokens,
            thinking,
            effort,
            forced_tool_choice,
            suppressed,
        }
    }
}

/// Maps the normalized reasoning effort onto Claude's levels.
fn claude_effort(effort: ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::Minimal | ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::Xhigh => "xhigh",
        ReasoningEffort::Max => "max",
    }
}

/// The output-token limit this request sends as `max_tokens`.
///
/// The request's own limit wins; a request that names none takes the model's
/// catalog limit, which is the largest answer the model can give, and only a
/// model the catalog records no limit for falls back to
/// [`DEFAULT_MAX_TOKENS`]. A small fixed default here would cut off a long
/// generation with nothing but a `max_tokens` finish reason to show for it.
fn output_limit(call: &ResolvedCall) -> u32 {
    if let Some(tokens) = call.request().max_output_tokens() {
        return tokens;
    }
    // A catalog limit past `u32` is not a real model limit, so a request that
    // meets one keeps the largest value the field can carry.
    call.route()
        .model()
        .limits()
        .map_or(DEFAULT_MAX_TOKENS, |limits| {
            u32::try_from(limits.max_output_tokens).unwrap_or(u32::MAX)
        })
}

/// Whether this model wants an adaptive thinking object on every request.
///
/// A model with effort levels lets the provider size its own thinking, but it
/// has to be told to: without a `thinking` object the model does not reason at
/// all, which changes answer quality, latency, and spend for a caller who asked
/// for nothing unusual. Effort does not replace it — effort guides how the
/// allocation is spent — so a levels model gets both.
///
/// A model without effort levels either does not reason or takes the explicit
/// budget [`thinking_budget`] computes.
///
/// A caller who wants something else sets `thinking` in the raw provider
/// options, which is merged over this.
fn takes_adaptive_thinking(route: &ResolvedRoute) -> bool {
    route.model().protocol_options().reasoning_effort_levels
}

/// Whether effort encodes as `output_config.effort` for this model.
///
/// A passthrough model is uncataloged precisely because it is newer than the
/// catalog, so the modern effort dialect is the safer guess — the one the
/// reference client made for unknown models. Guessing a thinking budget
/// instead would send a manual toggle the always-adaptive models reject. The
/// adaptive thinking object stays gated on the declared capability, so a
/// passthrough request without an effort is encoded exactly as before.
fn takes_effort_levels(route: &ResolvedRoute) -> bool {
    let model = route.model();
    model.protocol_options().reasoning_effort_levels || model.is_passthrough()
}

/// Whether the tool choice makes a tool call mandatory; see
/// [`ThinkingPlan::forced_tool_choice`].
fn forces_tool_use(choice: Option<&ToolChoice>) -> bool {
    choice.is_some_and(ToolChoice::is_forced)
}

/// The explicit thinking budget for a model without effort levels.
///
/// `None` when the request sets no effort or the model takes
/// `output_config.effort` directly. The budget scales the same way effort
/// levels scale — a share of the output limit — with the provider floor of
/// [`MIN_THINKING_BUDGET`]. `Minimal` shares `Low`'s budget for the same
/// reason [`claude_effort`] collapses them: the dialect has no smaller step.
fn thinking_budget(call: &ResolvedCall, limit: u32) -> Option<u32> {
    let effort = call.request().reasoning_effort()?;
    if takes_effort_levels(call.route()) {
        return None;
    }

    let limit = u64::from(limit);
    let share = match effort {
        ReasoningEffort::Minimal | ReasoningEffort::Low => limit / 4,
        ReasoningEffort::Medium => limit / 2,
        ReasoningEffort::High => limit * 3 / 4,
        ReasoningEffort::Xhigh => limit * 7 / 8,
        ReasoningEffort::Max => limit,
    };
    let budget = share.max(u64::from(MIN_THINKING_BUDGET));
    Some(u32::try_from(budget).unwrap_or(u32::MAX))
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use serde_json::json;

    use super::{MIN_THINKING_BUDGET, Thinking, ThinkingPlan};
    use crate::codecs::test_support::resolved_in;
    use crate::types::{ReasoningEffort, Request, ToolChoice, ToolDefinition};

    /// One provider with a budget model, an effort-levels model, and
    /// passthrough allowed.
    const CATALOG: &str = r#"
        schema_version = 1

        [providers.alpha]
        display_name = "Alpha"
        adapter = "test-adapter"
        codecs = ["test-codec"]
        base_url = "http://127.0.0.1"
        allow_passthrough = true
        default_model = "budget"
        auth = { type = "none" }

        [providers.alpha.models.budget]
        display_name = "Budget"
        api_model = "budget-v1"
        capabilities = { text = true, tools = true, reasoning = true, tool_choice = { required = true, named = true } }

        [providers.alpha.models.levels]
        display_name = "Levels"
        api_model = "levels-v1"
        capabilities = { text = true, tools = true, reasoning = true, tool_choice = { required = true, named = true } }
        protocol_options = { reasoning_effort_levels = true }
    "#;

    fn plan(request: Request, raw_thinking: bool) -> Result<ThinkingPlan, Box<dyn StdError>> {
        Ok(ThinkingPlan::for_call(
            &resolved_in(CATALOG, request)?,
            raw_thinking,
        ))
    }

    #[test]
    fn a_budget_model_gets_a_budget_that_fits_under_the_limit() -> Result<(), Box<dyn StdError>> {
        let plan = plan(
            Request::builder()
                .model("alpha/budget")
                .user("Hello")
                .reasoning_effort(ReasoningEffort::High)
                .max_output_tokens(8000)
                .build()?,
            false,
        )?;

        assert_eq!(plan.thinking, Some(Thinking::Budget(6000)));
        assert_eq!(plan.max_tokens, 8000);
        assert_eq!(plan.effort, None, "a budget model takes no effort level");
        assert!(plan.suppressed.is_empty());
        Ok(())
    }

    #[test]
    fn a_budget_that_would_not_fit_lifts_the_limit_by_the_floor() -> Result<(), Box<dyn StdError>> {
        let plan = plan(
            Request::builder()
                .model("alpha/budget")
                .user("Hello")
                .reasoning_effort(ReasoningEffort::Max)
                .max_output_tokens(2048)
                .build()?,
            false,
        )?;

        assert_eq!(plan.thinking, Some(Thinking::Budget(2048)));
        assert_eq!(plan.max_tokens, 2048 + MIN_THINKING_BUDGET);
        Ok(())
    }

    #[test]
    fn a_levels_model_gets_adaptive_thinking_and_an_effort_level() -> Result<(), Box<dyn StdError>>
    {
        let plan = plan(
            Request::builder()
                .model("alpha/levels")
                .user("Hello")
                .reasoning_effort(ReasoningEffort::Xhigh)
                .build()?,
            false,
        )?;

        assert_eq!(plan.thinking, Some(Thinking::Adaptive));
        assert_eq!(plan.effort, Some("xhigh"));
        Ok(())
    }

    #[test]
    fn a_passthrough_model_takes_the_effort_dialect_without_adaptive_thinking()
    -> Result<(), Box<dyn StdError>> {
        let plan = plan(
            Request::builder()
                .model("alpha/next")
                .user("Hello")
                .reasoning_effort(ReasoningEffort::Low)
                .build()?,
            false,
        )?;

        assert_eq!(plan.effort, Some("low"));
        assert_eq!(plan.thinking, None);
        Ok(())
    }

    #[test]
    fn a_forced_tool_choice_suppresses_everything_and_says_so() -> Result<(), Box<dyn StdError>> {
        let plan = plan(
            Request::builder()
                .model("alpha/levels")
                .user("Hello")
                .tool(ToolDefinition::function("lookup", "look up", json!({})))
                .tool_choice(ToolChoice::Required)
                .reasoning_effort(ReasoningEffort::High)
                .build()?,
            true,
        )?;

        assert!(plan.forced_tool_choice);
        assert_eq!(plan.thinking, None);
        assert_eq!(plan.effort, None);
        assert_eq!(plan.suppressed, [
            "reasoning effort with a forced tool choice",
            "a thinking provider option with a forced tool choice",
        ]);
        Ok(())
    }

    #[test]
    fn a_raw_thinking_option_leaves_the_object_and_the_limit_to_the_caller()
    -> Result<(), Box<dyn StdError>> {
        let plan = plan(
            Request::builder()
                .model("alpha/budget")
                .user("Hello")
                .reasoning_effort(ReasoningEffort::Max)
                .max_output_tokens(2048)
                .build()?,
            true,
        )?;

        assert_eq!(plan.thinking, None);
        assert_eq!(plan.max_tokens, 2048);
        assert!(plan.suppressed.is_empty());
        Ok(())
    }
}
