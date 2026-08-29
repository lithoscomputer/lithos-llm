//! V1 — smoke, every roster model.
//!
//! These are the tests that must stay green for a model to count as working
//! at all: one completion, one contract-checked stream, and the request
//! shapes every application uses — a system prompt, a multi-turn history, an
//! output cap, and a stop sequence. The completion test also carries the
//! Venice-specific cost assertion, because every response is expected to
//! carry Venice's in-band cost.

use lithos_llm::types::{CostSource, FinishReason, Message, Role};

use crate::support::{self, TestResult};
use crate::venice::{self, model_tests};

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

async fn completes_with_usage_and_provider_cost(model: &str) -> TestResult {
    let Some(client) = venice::live_client() else {
        return support::skip("VENICE_API_KEY is unset");
    };
    let request = venice::request(model)
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

    // The Venice-specific half: cost arrives in-band on every response, and
    // the catalog deliberately prices nothing, so a missing or zero cost
    // means the in-band extraction broke.
    let cost = response.cost.ok_or("the response carries no cost")?;
    assert_eq!(cost.source, CostSource::Provider);
    assert!(cost.usd_micros > 0, "the provider-reported cost is zero");
    Ok(())
}

async fn streams_within_the_contract(model: &str) -> TestResult {
    let Some(client) = venice::live_client() else {
        return support::skip("VENICE_API_KEY is unset");
    };
    // The emoji requirement makes the stream carry multi-byte characters, so
    // a codec that splits UTF-8 across SSE chunk boundaries corrupts visibly.
    let request = venice::request(model)
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
    let Some(client) = venice::live_client() else {
        return support::skip("VENICE_API_KEY is unset");
    };
    let request = venice::request(model)
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
    let Some(client) = venice::live_client() else {
        return support::skip("VENICE_API_KEY is unset");
    };
    let request = venice::request(model)
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
    // grok-4.6 exempts reasoning tokens from `max_tokens`: a raw call with
    // `max_tokens: 32` on 2026-08-29 returned 894 completion tokens, 862 of
    // them reasoning, with the visible output at the cap — and the same
    // request through OpenRouter behaves the same way, so this is xAI
    // semantics, not a Venice defect. The finish reason stays "stop" even
    // when the visible output is truncated, which is what breaks this test.
    if model == "grok-4.6" {
        return support::skip("the upstream model exempts reasoning from the output cap");
    }
    let Some(client) = venice::live_client() else {
        return support::skip("VENICE_API_KEY is unset");
    };
    let request = venice::request(model)
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

/// Roster models whose upstream ignores `stop` outright.
///
/// The first full run showed the OpenAI and xAI reasoning models running
/// straight through the sequence — those APIs dropped stop-sequence support
/// with reasoning — while every other family honored it. The crate has no
/// per-model capability flag for stop sequences yet, so the pin lives here;
/// see the drift notes in `.ai/plans/live-e2e-test-matrix.md`.
const IGNORES_STOP: &[&str] = &[
    "grok-4.6",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
];

async fn stops_at_the_stop_sequence(model: &str) -> TestResult {
    if IGNORES_STOP.contains(&model) {
        return support::skip("the upstream model ignores stop sequences");
    }
    let Some(client) = venice::live_client() else {
        return support::skip("VENICE_API_KEY is unset");
    };
    let request = venice::request(model)
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
