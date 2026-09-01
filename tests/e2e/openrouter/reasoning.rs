//! V2 — reasoning.
//!
//! Three layers, matching how much the catalog pins down:
//!
//! 1. Presence: fabro's catalog says the two DeepSeek rows reason by default,
//!    so those assert reasoning evidence; the other rows only record what they
//!    observe, until a first nightly run pins them.
//! 2. Effort levels: models with a known level vocabulary assert every level is
//!    accepted. Models whose vocabulary is unknown or contested — the kimi
//!    rows, and all the Claude/GPT additions — get probes that record
//!    accept/reject per level without failing; the probe output is what fixes
//!    the catalog rows.
//! 3. The round trip: reasoning content produced by the model is carried back
//!    in the next turn, which is the shape agent replay depends on.

use lithos_llm::types::{
    ContentPart, ErrorKind, FinishReason, Message, ReasoningEffort, RequestBuilder, Role,
    ToolChoice, ToolDefinition, ToolResult,
};
use serde_json::{Value, json};

use crate::openrouter::{self, model_tests};
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
                #[ignore = "live OpenRouter call; run with `mise run test:e2e`"]
                async fn $name() -> TestResult {
                    super::accepts_the_effort_level($model, ReasoningEffort::$level).await
                }
            )+
        };
    }

    level_tests!(
        grok_4_6_low "grok-4.6" Low,
        grok_4_6_medium "grok-4.6" Medium,
        grok_4_6_high "grok-4.6" High,
        grok_4_6_xhigh "grok-4.6" Xhigh,
        deepseek_v4_flash_low "deepseek-v4-flash" Low,
        deepseek_v4_flash_high "deepseek-v4-flash" High,
        deepseek_v4_flash_max "deepseek-v4-flash" Max,
        qwen_3_8_27b_low "qwen3.8-27b" Low,
        qwen_3_8_27b_medium "qwen3.8-27b" Medium,
        qwen_3_8_27b_xhigh "qwen3.8-27b" Xhigh,
    );
}

/// Effort probes for the contested and still-unpinned rows.
///
/// fabro's catalog and OpenRouter's live listing disagree about the kimi rows
/// and glm-5.3, and nothing pins the Claude/GPT additions yet. A probe sends
/// one low-effort and one high-effort request and records what came back; it
/// fails only on an error that is neither acceptance nor a clean invalid
/// request, so its output settles the catalog without turning a vocabulary
/// question into a red nightly.
mod effort_probes {
    use super::*;

    macro_rules! probe_tests {
        ($($name:ident $model:literal,)+) => {
            $(
                #[tokio::test]
                #[ignore = "live OpenRouter call; run with `mise run test:e2e`"]
                async fn $name() -> TestResult {
                    super::probes_the_effort_vocabulary($model).await
                }
            )+
        };
    }

    probe_tests!(
        kimi_k3 "kimi-k3",
        glm_5_3 "glm-5.3",
        claude_fable_5_1 "claude-fable-5.1",
        claude_fable_5 "claude-fable-5",
        gpt_5_6_sol "gpt-5.6-sol",
        gpt_5_6_terra "gpt-5.6-terra",
        gpt_5_6_luna "gpt-5.6-luna",
        gpt_5_5 "gpt-5.5",
    );
}

mod round_trip {
    use super::*;

    macro_rules! round_trip_tests {
        ($($name:ident $model:literal,)+) => {
            $(
                #[tokio::test]
                #[ignore = "live OpenRouter call; run with `mise run test:e2e`"]
                async fn $name() -> TestResult {
                    super::replays_reasoning_with_a_tool_result($model).await
                }
            )+
        };
    }

    round_trip_tests!(
        claude_fable_5 "claude-fable-5",
        grok_4_6 "grok-4.6",
        deepseek_v4_flash "deepseek-v4-flash",
    );

    /// Fable 5.1 takes no forced tool choice, so its round trip opens under
    /// `auto` with a system instruction naming the tool instead. A separate
    /// cell rather than a change to the shared runner keeps the three
    /// recorded cells above on their request hashes.
    #[tokio::test]
    #[ignore = "live OpenRouter call; run with `mise run test:e2e`"]
    async fn claude_fable_5_1() -> TestResult {
        super::replays_reasoning_with_a_tool_result_under_auto_choice("claude-fable-5.1").await
    }
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

/// The roster rows fabro's catalog marks `reasoning_by_default`, where
/// missing reasoning evidence is a failure rather than an observation.
const REASONS_BY_DEFAULT: &[&str] = &["deepseek-v4-flash", "deepseek-v4-pro"];

async fn shows_reasoning_evidence(model: &str) -> TestResult {
    let Some(client) = openrouter::live_client() else {
        return support::skip("OPENROUTER_API_KEY is unset");
    };
    let request = openrouter::request(model).user(HARD_PROMPT).build()?;
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
    let Some(client) = openrouter::live_client() else {
        return support::skip("OPENROUTER_API_KEY is unset");
    };
    let request = openrouter::request(model)
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
    // Reasoning evidence is either bucket: some skins count reasoning tokens
    // in usage, others return the reasoning text without counting it — qwen
    // on OpenRouter sends 500+ characters of `reasoning_content` with no
    // `reasoning_tokens` detail at all.
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

async fn probes_the_effort_vocabulary(model: &str) -> TestResult {
    let Some(client) = openrouter::live_client() else {
        return support::skip("OPENROUTER_API_KEY is unset");
    };
    for effort in [ReasoningEffort::Low, ReasoningEffort::High] {
        let request = openrouter::request(model)
            .user(HARD_PROMPT)
            .reasoning_effort(effort)
            .build()?;
        match client.complete(request).await {
            Ok(response) => {
                let reasoning_parts = response
                    .content
                    .iter()
                    .filter(|part| matches!(part, ContentPart::Reasoning(_)))
                    .count();
                support::observe(&format!(
                    "{model} accepted effort {effort:?} (reasoning tokens: {}, reasoning parts: \
                     {reasoning_parts})",
                    response.usage.reasoning
                ));
            }
            Err(error)
                if matches!(
                    error.kind(),
                    ErrorKind::InvalidRequest | ErrorKind::Provider
                ) =>
            {
                support::observe(&format!(
                    "{model} rejected effort {effort:?}: {}",
                    error.message()
                ));
            }
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

/// One live turn produces reasoning and a tool call; the second turn carries
/// both back with the tool result. A codec that drops or mangles carried
/// reasoning breaks here on the provider side, which no mock can prove.
async fn replays_reasoning_with_a_tool_result(model: &str) -> TestResult {
    let opening = openrouter::request(model)
        .user("Look up the weather in Paris, then tell me if it suits a picnic.")
        .tool(tools_weather())
        .tool_choice(ToolChoice::Tool {
            name: "get_weather".to_owned(),
        });
    replays_reasoning_from(model, opening).await
}

/// The round trip for a model that takes no forced tool choice: the opening
/// turn asks for the call in the system prompt and leaves the choice `auto`.
async fn replays_reasoning_with_a_tool_result_under_auto_choice(model: &str) -> TestResult {
    let opening = openrouter::request(model)
        .system("Use the get_weather tool for every weather question before answering.")
        .user("Look up the weather in Paris, then tell me if it suits a picnic.")
        .tool(tools_weather())
        .tool_choice(ToolChoice::Auto);
    replays_reasoning_from(model, opening).await
}

async fn replays_reasoning_from(model: &str, opening: RequestBuilder) -> TestResult {
    let Some(client) = openrouter::live_client() else {
        return support::skip("OPENROUTER_API_KEY is unset");
    };
    let first = opening.build()?;
    let opening = client.complete(first).await?;
    let calls = support::tool_calls(&opening);
    let call_id = calls
        .first()
        .ok_or("the opening turn made no tool call")?
        .id
        .clone();

    let second = openrouter::request(model)
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
