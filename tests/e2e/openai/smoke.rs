//! V1 — smoke, every roster model.
//!
//! These are the tests that must stay green for a model to count as working
//! at all: one completion, one contract-checked stream, and the request
//! shapes every application uses — a system prompt, a multi-turn history,
//! and an output cap. The completion test also carries the catalog cost
//! assertion, because OpenAI reports token usage and the catalog supplies
//! the published rates.
//!
//! Two cells differ from the other suites' smoke shape:
//!
//! - Stop sequences: `/v1/responses` rejects a `stop` member outright, so the
//!   codec drops the sequences with a warning. The pinned behavior is the drop,
//!   not a stop, and one cell covers it.
//! - Token counting: OpenAI has a native `/v1/responses/input_tokens` endpoint.
//!   The twin does not proxy that path, so the count cells are live-only.

use lithos_llm::types::{CostSource, FinishReason, Message, Role};

use crate::openai::{self, model_tests};
use crate::support::{self, TestResult};

mod basic_completion {
    use super::*;

    model_tests!(super::completes_with_usage_and_catalog_cost);
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

mod token_count {
    use super::*;

    model_tests!(super::counts_input_tokens_natively);
}

async fn completes_with_usage_and_catalog_cost(model: &str) -> TestResult {
    let Some(client) = openai::live_client() else {
        return support::skip("OPENAI_API_KEY is unset");
    };
    let request = openai::request(model)
        .user("In one short sentence, say hello.")
        .build()?;
    let response = client.complete(request).await?;

    assert!(
        !response.text().trim().is_empty(),
        "the completion carries no text"
    );
    assert_eq!(response.finish_reason, FinishReason::Stop);
    assert!(
        response
            .usage
            .input
            .saturating_add(response.usage.cache_read)
            .saturating_add(response.usage.cache_write)
            > 0,
        "no prompt tokens were counted"
    );
    assert!(
        response.usage.billable_output() > 0,
        "no output tokens were counted"
    );
    assert!(
        response.raw.is_some(),
        "a complete response must carry the raw provider payload"
    );

    // OpenAI reports token usage rather than in-band cost, so the published
    // catalog rates must produce the estimate.
    let cost = response.cost.ok_or("the response carries no cost")?;
    assert_eq!(cost.source, CostSource::Catalog);
    assert!(cost.usd_micros > 0, "the catalog-estimated cost is zero");
    Ok(())
}

async fn streams_within_the_contract(model: &str) -> TestResult {
    let Some(client) = openai::live_client() else {
        return support::skip("OPENAI_API_KEY is unset");
    };
    // The emoji requirement makes the stream carry multi-byte characters, so
    // a codec that splits UTF-8 across SSE chunk boundaries corrupts visibly.
    let request = openai::request(model)
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
    let Some(client) = openai::live_client() else {
        return support::skip("OPENAI_API_KEY is unset");
    };
    let request = openai::request(model)
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
    let Some(client) = openai::live_client() else {
        return support::skip("OPENAI_API_KEY is unset");
    };
    let request = openai::request(model)
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
    let Some(client) = openai::live_client() else {
        return support::skip("OPENAI_API_KEY is unset");
    };
    // Every roster model reasons by default and reasoning spends from the
    // same allowance, so a 32-token cap truncates during or right after the
    // reasoning phase either way.
    let request = openai::request(model)
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

async fn counts_input_tokens_natively(model: &str) -> TestResult {
    if let Some(skip) = support::live_only("the twin does not proxy /v1/responses/input_tokens") {
        return skip;
    }
    let Some(client) = openai::live_client() else {
        return support::skip("OPENAI_API_KEY is unset");
    };
    let request = openai::request(model)
        .system("Answer briefly.")
        .user("In one short sentence, say hello.")
        .build()?;
    let count = client
        .count_input_tokens(request)
        .await?
        .ok_or("OpenAI has a native input token count endpoint")?;
    assert!(count.tokens() > 0, "the provider counted zero input tokens");
    Ok(())
}

/// Stop sequences never reach this protocol: the live API rejects a `stop`
/// member with a 400 "Unknown parameter" (probed 2026-08-30), so the codec
/// drops the sequences and warns. The pinned behavior is that the request
/// still completes and the warning arrives; the model running through the
/// sequence is expected, not a failure.
#[tokio::test]
#[ignore = "live OpenAI call; run with `mise run test:e2e`"]
async fn stop_sequences_are_dropped_with_a_warning() -> TestResult {
    let Some(client) = openai::live_client() else {
        return support::skip("OPENAI_API_KEY is unset");
    };
    let request = openai::request("gpt-5.6-luna")
        .system("Follow the instruction literally, with no extra commentary.")
        .user("Recite the numbers one to nine as lowercase English words, separated by spaces.")
        .stop_sequence("five")
        .build()?;
    let response = client.complete(request).await?;
    assert!(
        !response.text().trim().is_empty(),
        "the request with a stop sequence must still complete"
    );
    assert!(
        response
            .warnings
            .iter()
            .any(|warning| warning.message.contains("stop")),
        "dropping the stop sequences must warn: {:?}",
        response.warnings
    );
    Ok(())
}
