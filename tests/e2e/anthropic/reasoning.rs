//! V2 — Anthropic thinking and effort.
//!
//! Every roster model gets an explicit reasoning request. The modern models
//! take named effort levels with adaptive thinking; Sonnet 4.5 and Haiku 4.5
//! take a manual thinking budget derived from the same normalized request.
//!
//! The cells that assert reasoning evidence ask for `Max` effort. Adaptive
//! thinkers may skip reasoning when they judge the question easy, and
//! Claude Fable 5.1 does exactly that on the hard prompt at `High` (no
//! thinking block, zero thinking tokens, observed 2026-09-01) while
//! reasoning at `Xhigh` and `Max`. The strongest level is the one request the
//! catalog claims every reasoning model must honor with visible reasoning.

use lithos_llm::types::{
    ContentPart, FinishReason, Message, ReasoningEffort, Role, ToolChoice, ToolDefinition,
    ToolResult,
};
use serde_json::{Value, json};

use crate::anthropic::{self, model_tests};
use crate::support::{self, TestResult};

mod presence {
    use super::*;

    model_tests!(super::shows_reasoning_evidence);
}

mod effort_levels {
    use super::*;

    macro_rules! level_tests {
        ($($name:ident $model:literal $level:ident,)+) => {
            $(
                #[tokio::test]
                #[ignore = "live Anthropic call; run with mise run test:e2e:live"]
                async fn $name() -> TestResult {
                    super::accepts_the_effort_level($model, ReasoningEffort::$level).await
                }
            )+
        };
    }

    level_tests!(
        fable_5_1_low "claude-fable-5.1" Low,
        fable_5_1_medium "claude-fable-5.1" Medium,
        fable_5_1_high "claude-fable-5.1" High,
        fable_5_1_xhigh "claude-fable-5.1" Xhigh,
        fable_5_1_max "claude-fable-5.1" Max,
        fable_low "claude-fable-5" Low,
        fable_medium "claude-fable-5" Medium,
        fable_high "claude-fable-5" High,
        fable_xhigh "claude-fable-5" Xhigh,
        fable_max "claude-fable-5" Max,
        opus_5_low "claude-opus-5" Low,
        opus_5_medium "claude-opus-5" Medium,
        opus_5_high "claude-opus-5" High,
        opus_5_xhigh "claude-opus-5" Xhigh,
        opus_5_max "claude-opus-5" Max,
        sonnet_5_low "claude-sonnet-5" Low,
        sonnet_5_xhigh "claude-sonnet-5" Xhigh,
        sonnet_5_max "claude-sonnet-5" Max,
        opus_4_8_low "claude-opus-4.8" Low,
        opus_4_8_xhigh "claude-opus-4.8" Xhigh,
        opus_4_8_max "claude-opus-4.8" Max,
        opus_4_7_low "claude-opus-4.7" Low,
        opus_4_7_xhigh "claude-opus-4.7" Xhigh,
        opus_4_7_max "claude-opus-4.7" Max,
        opus_4_6_low "claude-opus-4.6" Low,
        opus_4_6_high "claude-opus-4.6" High,
        opus_4_6_max "claude-opus-4.6" Max,
        sonnet_4_6_low "claude-sonnet-4.6" Low,
        sonnet_4_6_high "claude-sonnet-4.6" High,
        sonnet_4_6_max "claude-sonnet-4.6" Max,
    );
}

mod manual_budget {
    use super::*;

    macro_rules! budget_tests {
        ($($name:ident $model:literal,)+) => {
            $(
                #[tokio::test]
                #[ignore = "live Anthropic call; run with mise run test:e2e:live"]
                async fn $name() -> TestResult {
                    super::accepts_the_effort_level($model, ReasoningEffort::High).await
                }
            )+
        };
    }

    budget_tests!(
        sonnet_4_5 "claude-sonnet-4.5",
        haiku_4_5 "claude-haiku-4.5",
    );
}

const HARD_PROMPT: &str = "A bookshop sells notebooks at $7 and pens at $3. Anna spent exactly \
     $118 and bought at least one of each. How many different combinations of notebooks and pens \
     could she have bought? Answer with just the number.";

async fn shows_reasoning_evidence(model: &str) -> TestResult {
    let Some(client) = anthropic::live_client() else {
        return support::skip("the Anthropic suite is live-only or ANTHROPIC_API_KEY is unset");
    };
    let response = client
        .complete(
            anthropic::request(model)
                .user(HARD_PROMPT)
                .reasoning_effort(ReasoningEffort::Max)
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
        "{model} showed no reasoning usage or content for an explicit reasoning request"
    );
    Ok(())
}

async fn accepts_the_effort_level(model: &str, effort: ReasoningEffort) -> TestResult {
    let Some(client) = anthropic::live_client() else {
        return support::skip("the Anthropic suite is live-only or ANTHROPIC_API_KEY is unset");
    };
    let response = client
        .complete(
            anthropic::request(model)
                .user(HARD_PROMPT)
                .reasoning_effort(effort)
                .build()?,
        )
        .await?;
    assert!(
        matches!(
            response.finish_reason,
            FinishReason::Stop | FinishReason::Length
        ),
        "effort {effort:?} was not answered normally: {:?}",
        response.finish_reason
    );
    Ok(())
}

/// Replays a signed thinking block with a tool result on a second turn.
#[tokio::test]
#[ignore = "live Anthropic call; run with mise run test:e2e:live"]
async fn reasoning_round_trip() -> TestResult {
    replays_reasoning_with_a_tool_result("claude-opus-4.6", None).await
}

/// The same replay on Fable 5.1 with preserved-thinking enforcement on.
///
/// Fable 5.1 binds every thinking block to the bytes before it: the `system`
/// prompt, the `tools`, and every earlier message. The codec does three
/// things between the opening and closing turns that must count as no edit:
/// it merges tool results into one `user` turn, it places `cache_control`
/// markers, and it carries a system message appended after the tool result
/// as a `system` turn rather than folding it into the top-level field. Setting
/// `prefix_mismatch_behavior` opts the request into the check on any account,
/// so this cell fails with a 400 naming the change if the encoder ever rewrites
/// the prefix. `error` rather than `drop_block` because a silent drop is the
/// failure mode being guarded against.
#[tokio::test]
#[ignore = "live Anthropic call; run with mise run test:e2e:live"]
async fn reasoning_round_trip_on_fable_5_1_under_enforcement() -> TestResult {
    replays_reasoning_with_a_tool_result("claude-fable-5.1", Some(BindingControls)).await
}

/// The closing turn opts into preserved-thinking enforcement.
struct BindingControls;

/// The beta that unlocks `thinking.block_binding`.
const BINDING_BETA: &str = "thinking-binding-controls-2026-08-01";

async fn replays_reasoning_with_a_tool_result(
    model: &str,
    binding: Option<BindingControls>,
) -> TestResult {
    let Some(client) = anthropic::live_client() else {
        return support::skip("the Anthropic suite is live-only or ANTHROPIC_API_KEY is unset");
    };
    let opening = client
        .complete(
            anthropic::request(model)
                .system("Think carefully, then use get_weather for every weather question.")
                .user("Look up the weather in Paris.")
                .reasoning_effort(ReasoningEffort::Max)
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

    // Dropping the effort between turns is allowed: effort is not part of
    // the bound prefix.
    let mut closing = anthropic::request(model)
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
        )]));
    if binding.is_some() {
        // An instruction appended after the tool result is what an agent loop
        // does between turns. On a model that takes system turns it goes out
        // in place; hoisting it would rewrite the bound top-level prefix and
        // fail the check below.
        closing = closing.message(Message::text(Role::System, "Answer in one short sentence."));
        // The raw `thinking` option replaces the generated adaptive object,
        // so it has to restate the type beside the binding control.
        closing = closing
            .provider_option(anthropic::PROVIDER, "beta_headers", json!([BINDING_BETA]))
            .provider_option(
                anthropic::PROVIDER,
                "thinking",
                json!({
                    "type": "adaptive",
                    "block_binding": { "prefix_mismatch_behavior": "error" },
                }),
            );
    }
    let closing = client.complete(closing.build()?).await?;
    assert!(!closing.text().trim().is_empty());
    if binding.is_some() {
        // The beta also reports every dropped block. Under `error` a prefix
        // mismatch never gets this far, so the list can only name blocks a
        // model switch dropped, and there was none.
        let dropped = closing
            .raw
            .as_ref()
            .and_then(|raw| raw.get("input_transformations"))
            .and_then(Value::as_array)
            .map_or(0, Vec::len);
        assert_eq!(
            dropped, 0,
            "the replay dropped thinking blocks: {:?}",
            closing.raw
        );
    }
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
