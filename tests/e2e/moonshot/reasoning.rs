//! V2 — reasoning.
//!
//! K3 always reasons. The suite checks reasoning evidence, every documented
//! effort level, and a tool round trip that must preserve reasoning content.

use lithos_llm::types::{
    ContentPart, FinishReason, Message, ReasoningEffort, Role, ToolChoice, ToolDefinition,
    ToolResult,
};
use serde_json::{Value, json};

use crate::moonshot::{self, model_tests};
use crate::support::{self, TestResult};

mod presence {
    use super::*;

    model_tests!(super::shows_reasoning_evidence);
}

/// One test per (model, claimed effort level).
mod effort_levels {
    use super::*;

    macro_rules! level_tests {
        ($($name:ident $model:literal $level:ident,)+) => {
            $(
                #[tokio::test]
                #[ignore = "live Moonshot call; run with `mise run test:e2e`"]
                async fn $name() -> TestResult {
                    super::accepts_the_effort_level($model, ReasoningEffort::$level).await
                }
            )+
        };
    }

    level_tests!(
        kimi_k3_low "kimi-k3" Low,
        kimi_k3_high "kimi-k3" High,
        kimi_k3_max "kimi-k3" Max,
    );
}

mod round_trip {
    use super::*;

    macro_rules! round_trip_tests {
        ($($name:ident $model:literal,)+) => {
            $(
                #[tokio::test]
                #[ignore = "live Moonshot call; run with `mise run test:e2e`"]
                async fn $name() -> TestResult {
                    super::replays_reasoning_with_a_tool_result($model).await
                }
            )+
        };
    }

    round_trip_tests!(
        kimi_k3 "kimi-k3",
    );
}

/// A question hard enough that a reasoning model actually reasons.
///
/// Adaptive thinkers skip reasoning on questions they can answer from
/// memory, so an easy prompt ("is 91 prime?") makes a zero reasoning count
/// ambiguous: inert effort, or no effort needed. This one takes a real
/// enumeration — 7n + 3p = 118 with n, p ≥ 1 has six solutions — so a zero
/// count with reasoning requested means the request had no effect.
const HARD_PROMPT: &str = "A bookshop sells notebooks at $7 and pens at $3. Anna spent exactly \
     $118 and bought at least one of each. How many different combinations of notebooks and pens \
     could she have bought? Answer with just the number.";

/// K3 always reasons, so missing evidence is a failure.
const REASONS_BY_DEFAULT: &[&str] = &["kimi-k3"];

async fn shows_reasoning_evidence(model: &str) -> TestResult {
    let Some(client) = moonshot::live_client() else {
        return support::skip("MOONSHOT_API_KEY is unset");
    };
    let request = moonshot::request(model).user(HARD_PROMPT).build()?;
    let stream = client.stream(request).await?;
    let (events, response) = support::checked_stream(stream).await?;

    let saw_delta = events
        .iter()
        .any(|event| event.get("type").and_then(Value::as_str) == Some("reasoning_delta"));
    let saw_part = response
        .content
        .iter()
        .any(|part| matches!(part, ContentPart::Reasoning(_)));
    let evidence = saw_delta || saw_part || response.usage.reasoning > 0;

    if REASONS_BY_DEFAULT.contains(&model) {
        assert!(
            evidence,
            "{model} reasons by default but showed no reasoning deltas, parts, or usage"
        );
    } else if !evidence {
        support::observe(&format!("{model} showed no reasoning evidence by default"));
    }
    Ok(())
}

async fn accepts_the_effort_level(model: &str, effort: ReasoningEffort) -> TestResult {
    let Some(client) = moonshot::live_client() else {
        return support::skip("MOONSHOT_API_KEY is unset");
    };
    let request = moonshot::request(model)
        .user(HARD_PROMPT)
        .reasoning_effort(effort)
        .build()?;
    let response = client.complete(request).await?;
    assert!(
        matches!(
            response.finish_reason,
            FinishReason::Stop | FinishReason::Length
        ),
        "effort {effort:?} was not answered normally: {:?}",
        response.finish_reason
    );
    // Moonshot carries reasoning in both `reasoning_content` and the
    // completion-token detail.
    let reasoning_parts: usize = response
        .content
        .iter()
        .filter(|part| matches!(part, ContentPart::Reasoning(_)))
        .count();
    support::observe(&format!(
        "{model} effort {effort:?}: reasoning tokens {}, reasoning parts {reasoning_parts}",
        response.usage.reasoning
    ));
    // On a question this hard, a model that takes effort levels must show
    // reasoning at the high end of its vocabulary.
    if matches!(
        effort,
        ReasoningEffort::High | ReasoningEffort::Xhigh | ReasoningEffort::Max
    ) {
        assert!(
            response.usage.reasoning > 0 || reasoning_parts > 0,
            "{model} showed no reasoning at effort {effort:?} on a hard question"
        );
    }
    Ok(())
}

/// One live turn produces reasoning and a tool call; the second turn carries
/// both back with the tool result. A codec that drops or mangles carried
/// reasoning breaks here on the provider side, which no mock can prove.
async fn replays_reasoning_with_a_tool_result(model: &str) -> TestResult {
    let Some(client) = moonshot::live_client() else {
        return support::skip("MOONSHOT_API_KEY is unset");
    };
    let first = moonshot::request(model)
        .system("Use the get_weather tool for every weather question.")
        .user("Look up the weather in Paris, then tell me if it suits a picnic.")
        .tool(tools_weather())
        .tool_choice(ToolChoice::Auto)
        .build()?;
    let opening = client.complete(first).await?;
    let calls = support::tool_calls(&opening);
    let call_id = calls
        .first()
        .ok_or("the opening turn made no tool call")?
        .id
        .clone();

    let second = moonshot::request(model)
        .user("Look up the weather in Paris, then tell me if it suits a picnic.")
        .tool(tools_weather())
        .message(Message::new(Role::Assistant, opening.content.clone()))
        .message(Message::new(Role::Tool, [ContentPart::ToolResult(
            ToolResult {
                tool_call_id: call_id,
                name:         Some("get_weather".to_owned()),
                content:      vec![ContentPart::Text {
                    text: "21C, sunny, light breeze".to_owned(),
                }],
                is_error:     false,
            },
        )]))
        .build()?;
    let closing = client.complete(second).await?;
    assert!(
        !closing.text().trim().is_empty(),
        "the second turn gave no answer after the reasoning replay"
    );
    Ok(())
}

fn tools_weather() -> ToolDefinition {
    ToolDefinition::function(
        "get_weather",
        "Reads the current weather for a city",
        json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
        }),
    )
}
