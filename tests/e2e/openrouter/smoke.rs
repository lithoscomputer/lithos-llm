//! V1 — smoke, every roster model.
//!
//! These are the tests that must stay green for a model to count as working
//! at all: one completion, one contract-checked stream, and the request
//! shapes every application uses — a system prompt, a multi-turn history, an
//! output cap, and a stop sequence. The completion test also carries the
//! OpenRouter-specific cost assertion, because every response is expected to
//! carry OpenRouter's in-band cost.

use lithos_llm::types::{CostSource, FinishReason, Message, Role};

use crate::openrouter::{self, family_tests, model_tests};
use crate::support::{self, TestResult};

mod basic_completion {
    use super::*;

    model_tests!(super::completes_with_usage_and_provider_cost);
}

mod stream_contract {
    use super::*;

    model_tests!(super::streams_within_the_contract);
}

mod system_prompt {
    use super::*;

    model_tests!(super::honors_the_system_prompt);
}

mod multi_turn {
    use super::*;

    model_tests!(super::carries_the_conversation);
}

mod length_truncation {
    use super::*;

    model_tests!(super::truncates_at_the_output_cap);
}

mod stop_sequence {
    use super::*;

    model_tests!(super::stops_at_the_stop_sequence);
}

mod mid_conversation_system {
    use super::*;

    family_tests!(super::honors_a_mid_conversation_system_message);
}

/// A system message appended after the conversation has started rides the
/// Chat Completions protocol in place, and the gateway translates it for
/// upstreams whose native protocol has no such turn. The instruction must
/// win: an uppercase answer, where the conversation so far was lowercase
/// prose. One representative per family, since honoring it is upstream
/// behavior.
async fn honors_a_mid_conversation_system_message(model: &str) -> TestResult {
    // MiniMax M2.7 answered in lowercase on every attempt through OpenRouter on
    // 2026-09-01: the message is accepted but not obeyed.
    if model == "minimax-m2.7" {
        return support::skip(
            "the OpenRouter route does not obey a mid-conversation system message",
        );
    }
    let Some(client) = openrouter::live_client() else {
        return support::skip("OPENROUTER_API_KEY is unset");
    };
    // Whether the model obeys the instruction is model behavior, so one miss
    // gets one retry before it counts as a failure.
    for attempt in 0..2 {
        let request = openrouter::request(model)
            .system("Answer with just the city name.")
            .user("What is the capital of France?")
            .message(Message::text(Role::Assistant, "Paris."))
            .user("And of Spain?")
            .message(Message::text(
                Role::System,
                "From now on, write every answer in uppercase letters only.",
            ))
            .build()?;
        let response = client.complete(request).await?;
        let text = response.text();
        if text.contains("MADRID") {
            return Ok(());
        }
        support::observe(&format!(
            "{model} did not follow the mid-conversation system message (attempt {attempt}): \
             {text:?}"
        ));
    }
    Err(format!("{model} never followed the mid-conversation system message").into())
}

async fn completes_with_usage_and_provider_cost(model: &str) -> TestResult {
    let Some(client) = openrouter::live_client() else {
        return support::skip("OPENROUTER_API_KEY is unset");
    };
    let request = openrouter::request(model)
        .user("In one short sentence, say hello.")
        .build()?;
    let response = client.complete(request).await?;

    assert!(
        !response.text().trim().is_empty(),
        "the completion carries no text"
    );
    assert_eq!(response.finish_reason, FinishReason::Stop);
    assert!(response.usage.input > 0, "no input tokens were counted");
    assert!(
        response.usage.billable_output() > 0,
        "no output tokens were counted"
    );
    assert!(
        response.raw.is_some(),
        "a complete response must carry the raw provider payload"
    );

    // The OpenRouter-specific half: cost arrives in-band on every response, and
    // the catalog deliberately prices nothing, so a missing or zero cost
    // means the in-band extraction broke.
    let cost = response.cost.ok_or("the response carries no cost")?;
    assert_eq!(cost.source, CostSource::Provider);
    assert!(cost.usd_micros > 0, "the provider-reported cost is zero");
    Ok(())
}

async fn streams_within_the_contract(model: &str) -> TestResult {
    let Some(client) = openrouter::live_client() else {
        return support::skip("OPENROUTER_API_KEY is unset");
    };
    // The emoji requirement makes the stream carry multi-byte characters, so
    // a codec that splits UTF-8 across SSE chunk boundaries corrupts visibly.
    let request = openrouter::request(model)
        .user("Reply with one short greeting that contains at least two emoji.")
        .build()?;
    let stream = client.stream(request).await?;
    let (_, response) = support::checked_stream(stream).await?;

    assert!(
        !response.text().trim().is_empty(),
        "the streamed completion carries no text"
    );
    assert!(
        response.usage.total() > 0,
        "the streamed completion carries no usage"
    );
    Ok(())
}

async fn honors_the_system_prompt(model: &str) -> TestResult {
    let Some(client) = openrouter::live_client() else {
        return support::skip("OPENROUTER_API_KEY is unset");
    };
    let request = openrouter::request(model)
        .system("When the user says ping, reply with exactly the word PONG and nothing else.")
        .user("ping")
        .build()?;
    let response = client.complete(request).await?;
    assert!(
        response.text().to_uppercase().contains("PONG"),
        "the system prompt was not honored: {:?}",
        response.text()
    );
    Ok(())
}

async fn carries_the_conversation(model: &str) -> TestResult {
    let Some(client) = openrouter::live_client() else {
        return support::skip("OPENROUTER_API_KEY is unset");
    };
    let request = openrouter::request(model)
        .system("Answer with just the city name.")
        .user("What is the capital of France?")
        .message(Message::text(Role::Assistant, "Paris."))
        .user("And of Spain?")
        .build()?;
    let response = client.complete(request).await?;
    assert!(
        response.text().contains("Madrid"),
        "the multi-turn answer does not name Madrid: {:?}",
        response.text()
    );
    Ok(())
}

async fn truncates_at_the_output_cap(model: &str) -> TestResult {
    // Both routes stopped normally before reaching the 32-token cap in the
    // unbounded counting probe through OpenRouter on 2026-08-29.
    if matches!(model, "grok-4.6" | "qwen3.8-27b") {
        return support::skip("the OpenRouter route stops before the output cap");
    }
    let Some(client) = openrouter::live_client() else {
        return support::skip("OPENROUTER_API_KEY is unset");
    };
    let request = openrouter::request(model)
        .user("Count upward from one, one number per line, and do not stop.")
        .max_output_tokens(32)
        .build()?;
    let response = client.complete(request).await?;
    assert_eq!(
        response.finish_reason,
        FinishReason::Length,
        "a 32-token cap on an unbounded task must truncate"
    );
    Ok(())
}

async fn stops_at_the_stop_sequence(model: &str) -> TestResult {
    // Verified through OpenRouter on 2026-08-29. The listing does not claim
    // `stop` for the GPT or Laguna rows. It does claim the parameter for
    // DeepSeek V4 Pro and MiMo, but both upstreams ignored it in the same
    // literal enumeration probe.
    const IGNORES_STOP: &[&str] = &[
        "deepseek-v4-pro",
        "gpt-5.4",
        "gpt-5.5",
        "gpt-5.6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "glm-5.3",
        "grok-4.6",
        "kimi-k3",
        "laguna-s-2.1",
        "laguna-xs-2.1",
        "mimo-v2.5-pro",
    ];
    if IGNORES_STOP.contains(&model) {
        return support::skip("the OpenRouter route ignores stop sequences");
    }
    let Some(client) = openrouter::live_client() else {
        return support::skip("OPENROUTER_API_KEY is unset");
    };
    let request = openrouter::request(model)
        .system("Follow the instruction literally, with no extra commentary.")
        .user("Recite the numbers one to nine as lowercase English words, separated by spaces.")
        .stop_sequence("five")
        .build()?;
    let response = client.complete(request).await?;
    assert!(
        !response.text().contains("six"),
        "the output continued past the stop sequence: {:?}",
        response.text()
    );
    Ok(())
}
