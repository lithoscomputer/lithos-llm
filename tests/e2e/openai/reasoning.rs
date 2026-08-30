//! V2 — reasoning.
//!
//! Three layers:
//!
//! 1. Presence: every roster model reasons, so a high-effort request must show
//!    reasoning evidence — usage tokens, deltas, or parts.
//! 2. Effort levels: the vocabularies were read off the live 400 messages on
//!    2026-08-30 — the 5.6 family takes low/medium/high/xhigh/max, gpt-5.4,
//!    gpt-5.4-mini, and gpt-5.5 take low/medium/high/xhigh, and the pro rows
//!    take medium/high/xhigh. The cells assert acceptance per claimed level;
//!    the full sweep runs on the cheap representatives and the expensive rows
//!    pin only the edges of their vocabulary. `minimal` is in no roster
//!    vocabulary, and one cell pins the live rejection so a vocabulary change
//!    surfaces as drift.
//! 3. The round trip: this protocol replays reasoning as encrypted content
//!    (`store: false` + `reasoning.encrypted_content`), which is the shape
//!    agent replay depends on — no mock can prove the provider accepts it.

use lithos_llm::types::{
    ContentPart, ErrorKind, FinishReason, Message, ReasoningEffort, Role, ToolChoice,
    ToolDefinition, ToolResult,
};
use serde_json::{Value, json};

use crate::openai::{self, model_tests};
use crate::support::{self, TestResult};

mod presence {
    use super::*;

    model_tests!(super::shows_reasoning_evidence_at_high_effort);
}

/// One test per (model, claimed effort level).
mod effort_levels {
    use super::*;

    macro_rules! level_tests {
        ($($name:ident $model:literal $level:ident,)+) => {
            $(
                #[tokio::test]
                #[ignore = "live OpenAI call; run with `mise run test:e2e`"]
                async fn $name() -> TestResult {
                    super::accepts_the_effort_level($model, ReasoningEffort::$level).await
                }
            )+
        };
    }

    level_tests!(
        gpt_5_6_luna_low "gpt-5.6-luna" Low,
        gpt_5_6_luna_medium "gpt-5.6-luna" Medium,
        gpt_5_6_luna_high "gpt-5.6-luna" High,
        gpt_5_6_luna_xhigh "gpt-5.6-luna" Xhigh,
        gpt_5_6_luna_max "gpt-5.6-luna" Max,
        gpt_5_6_sol_max "gpt-5.6-sol" Max,
        gpt_5_6_terra_max "gpt-5.6-terra" Max,
        gpt_5_4_mini_low "gpt-5.4-mini" Low,
        gpt_5_4_mini_medium "gpt-5.4-mini" Medium,
        gpt_5_4_mini_high "gpt-5.4-mini" High,
        gpt_5_4_mini_xhigh "gpt-5.4-mini" Xhigh,
        gpt_5_4_low "gpt-5.4" Low,
        gpt_5_4_xhigh "gpt-5.4" Xhigh,
        gpt_5_5_low "gpt-5.5" Low,
        gpt_5_5_xhigh "gpt-5.5" Xhigh,
        gpt_5_5_pro_medium "gpt-5.5-pro" Medium,
        gpt_5_4_pro_medium "gpt-5.4-pro" Medium,
    );
}

mod round_trip {
    use super::*;

    macro_rules! round_trip_tests {
        ($($name:ident $model:literal,)+) => {
            $(
                #[tokio::test]
                #[ignore = "live OpenAI call; run with `mise run test:e2e`"]
                async fn $name() -> TestResult {
                    super::replays_reasoning_with_a_tool_result($model).await
                }
            )+
        };
    }

    round_trip_tests!(
        gpt_5_6_luna "gpt-5.6-luna",
        gpt_5_6_sol "gpt-5.6-sol",
        gpt_5_4 "gpt-5.4",
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

async fn shows_reasoning_evidence_at_high_effort(model: &str) -> TestResult {
    let Some(client) = openai::live_client() else {
        return support::skip("OPENAI_API_KEY is unset");
    };
    // `high` is in every roster vocabulary, so the request is legal
    // everywhere and the evidence assertion can be unconditional.
    let request = openai::request(model)
        .user(HARD_PROMPT)
        .reasoning_effort(ReasoningEffort::High)
        .build()?;
    let stream = client.stream(request).await?;
    let (events, response) = support::checked_stream(stream).await?;

    let saw_delta = events
        .iter()
        .any(|event| event.get("type").and_then(Value::as_str) == Some("reasoning_delta"));
    let saw_part = response
        .content
        .iter()
        .any(|part| matches!(part, ContentPart::Reasoning(_)));
    assert!(
        saw_delta || saw_part || response.usage.reasoning > 0,
        "{model} showed no reasoning deltas, parts, or usage at high effort"
    );
    Ok(())
}

async fn accepts_the_effort_level(model: &str, effort: ReasoningEffort) -> TestResult {
    let Some(client) = openai::live_client() else {
        return support::skip("OPENAI_API_KEY is unset");
    };
    let request = openai::request(model)
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
    support::observe(&format!(
        "{model} effort {effort:?}: reasoning tokens {}",
        response.usage.reasoning
    ));
    // On a question this hard, a model must show reasoning at the high end
    // of its vocabulary.
    if matches!(
        effort,
        ReasoningEffort::High | ReasoningEffort::Xhigh | ReasoningEffort::Max
    ) {
        assert!(
            response.usage.reasoning > 0,
            "{model} showed no reasoning at effort {effort:?} on a hard question"
        );
    }
    Ok(())
}

/// `minimal` is in no roster model's vocabulary — the live API rejects it
/// with a 400 naming the supported values (2026-08-30). Pinning the
/// rejection keeps the vocabulary drift visible: if this cell starts
/// failing, OpenAI grew the vocabulary and the catalog notes are stale.
#[tokio::test]
#[ignore = "live OpenAI call; run with `mise run test:e2e`"]
async fn minimal_effort_is_rejected_by_the_provider() -> TestResult {
    // The twin records no error responses, so the 400 cannot replay.
    if let Some(skip) = support::live_only("live error classification") {
        return skip;
    }
    let Some(client) = openai::live_client() else {
        return support::skip("OPENAI_API_KEY is unset");
    };
    let request = openai::request("gpt-5.6-luna")
        .user(HARD_PROMPT)
        .reasoning_effort(ReasoningEffort::Minimal)
        .build()?;
    let error = client
        .complete(request)
        .await
        .expect_err("no roster model accepts minimal effort");
    assert!(
        matches!(
            error.kind(),
            ErrorKind::InvalidRequest | ErrorKind::Provider
        ),
        "the minimal-effort rejection classified as {:?}: {}",
        error.kind(),
        error.message()
    );
    Ok(())
}

/// One live turn produces reasoning and a tool call; the second turn carries
/// both back with the tool result. This protocol replays reasoning as an
/// opaque encrypted item, so a codec that drops or mangles it breaks here on
/// the provider side, which no mock can prove.
async fn replays_reasoning_with_a_tool_result(model: &str) -> TestResult {
    let Some(client) = openai::live_client() else {
        return support::skip("OPENAI_API_KEY is unset");
    };
    let first = openai::request(model)
        .user("Look up the weather in Paris, then tell me if it suits a picnic.")
        .tool(tools_weather())
        .tool_choice(ToolChoice::Tool {
            name: "get_weather".to_owned(),
        })
        .build()?;
    let opening = client.complete(first).await?;
    let calls = support::tool_calls(&opening);
    let call_id = calls
        .first()
        .ok_or("the opening turn made no tool call")?
        .id
        .clone();

    let second = openai::request(model)
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
