//! V2 — Gemini thinking and signed reasoning replay.
//!
//! Normalized reasoning effort reaches Gemini's `thinkingConfig`. Thought
//! summaries remain opt-in because they disclose returned reasoning content;
//! the signed replay cells enable them explicitly.

use lithos_llm::types::{
    ContentPart, Message, ReasoningEffort, Role, ToolChoice, ToolDefinition, ToolResult,
};
use serde_json::{Value, json};

use crate::gemini::{self, model_tests};
use crate::support::{self, TestResult};

mod presence {
    use super::*;

    model_tests!(super::shows_reasoning_evidence);
}

mod effort_low {
    use super::*;

    model_tests!(super::accepts_low_effort);
}

mod effort_medium {
    use super::*;

    model_tests!(super::accepts_medium_effort);
}

mod effort_high {
    use super::*;

    model_tests!(super::accepts_high_effort);
}

const HARD_PROMPT: &str = "A bookshop sells notebooks at $7 and pens at $3. Anna spent exactly \
     $118 and bought at least one of each. How many different combinations of notebooks and pens \
     could she have bought? Answer with just the number.";

fn thinking_config() -> Value {
    json!({
        "thinkingConfig": {
            "thinkingLevel": "high",
            "includeThoughts": true,
        }
    })
}

async fn accepts_low_effort(model: &str) -> TestResult {
    accepts_effort(model, ReasoningEffort::Low).await
}

async fn accepts_medium_effort(model: &str) -> TestResult {
    accepts_effort(model, ReasoningEffort::Medium).await
}

async fn accepts_high_effort(model: &str) -> TestResult {
    accepts_effort(model, ReasoningEffort::High).await
}

async fn accepts_effort(model: &str, effort: ReasoningEffort) -> TestResult {
    let Some(client) = gemini::live_client() else {
        return support::skip("the Gemini suite is live-only or its API keys are unset");
    };
    let response = client
        .complete(
            gemini::request(model)
                .user(HARD_PROMPT)
                .reasoning_effort(effort)
                .build()?,
        )
        .await?;
    assert!(
        response.usage.reasoning > 0,
        "{model} reported no reasoning tokens for {effort:?} effort"
    );
    Ok(())
}

async fn shows_reasoning_evidence(model: &str) -> TestResult {
    if !gemini::capabilities(model).reasoning {
        return support::skip("the catalog does not claim reasoning");
    }
    let Some(client) = gemini::live_client() else {
        return support::skip("the Gemini suite is live-only or its API keys are unset");
    };
    let response = client
        .complete(
            gemini::request(model)
                .user(HARD_PROMPT)
                .provider_option(gemini::PROVIDER, "generationConfig", thinking_config())
                .build()?,
        )
        .await?;
    let reasoning_parts = response
        .content
        .iter()
        .filter(|part| matches!(part, ContentPart::Reasoning(_)))
        .count();
    assert!(
        response.usage.reasoning > 0 || reasoning_parts > 0,
        "{model} showed no reasoning usage or content with thinking enabled"
    );
    Ok(())
}

/// A generated thought signature must survive the tool-result turn.
#[tokio::test]
#[ignore = "live Gemini call; run with `mise run test:e2e:live`"]
async fn reasoning_round_trip() -> TestResult {
    let Some(client) = gemini::live_client() else {
        return support::skip("the Gemini suite is live-only or its API keys are unset");
    };
    let model = "gemini-3.5-flash";
    let opening = client
        .complete(
            gemini::request(model)
                .system("Think carefully, then use get_weather for every weather question.")
                .user("Look up the weather in Paris.")
                .provider_option(gemini::PROVIDER, "generationConfig", thinking_config())
                .tool(weather_tool())
                .tool_choice(ToolChoice::Auto)
                .build()?,
        )
        .await?;
    assert!(
        opening
            .content
            .iter()
            .any(|part| matches!(part, ContentPart::Reasoning(_))),
        "the opening turn carried no reasoning block"
    );
    let call_id = support::tool_calls(&opening)
        .first()
        .ok_or("the opening turn made no tool call")?
        .id
        .clone();

    let closing = client
        .complete(
            gemini::request(model)
                .system("Think carefully, then use get_weather for every weather question.")
                .user("Look up the weather in Paris.")
                .tool(weather_tool())
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
                .build()?,
        )
        .await?;
    assert!(!closing.text().trim().is_empty());
    Ok(())
}

fn weather_tool() -> ToolDefinition {
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
