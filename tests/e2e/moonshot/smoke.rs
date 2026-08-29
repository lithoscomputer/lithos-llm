//! V1 — smoke, every roster model.
//!
//! These are the tests that must stay green for a model to count as working
//! at all: one completion, one contract-checked stream, and the request
//! shapes every application uses — a system prompt, a multi-turn history, an
//! output cap, and a stop sequence. The completion test also carries the
//! Moonshot-specific cost assertion, because Moonshot reports token usage and
//! the catalog supplies the published rates.

use lithos_llm::types::{CostSource, FinishReason, Message, Role};

use crate::moonshot::{self, model_tests};
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

mod stop_sequence {
    use super::*;

    model_tests!(super::stops_at_the_stop_sequence);
}

async fn completes_with_usage_and_catalog_cost(model: &str) -> TestResult {
    let Some(client) = moonshot::live_client() else {
        return support::skip("MOONSHOT_API_KEY is unset");
    };
    let request = moonshot::request(model)
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

    // Moonshot reports token usage rather than in-band cost, so the published
    // catalog rates must produce the estimate.
    let cost = response.cost.ok_or("the response carries no cost")?;
    assert_eq!(cost.source, CostSource::Catalog);
    assert!(cost.usd_micros > 0, "the catalog-estimated cost is zero");
    Ok(())
}

async fn streams_within_the_contract(model: &str) -> TestResult {
    let Some(client) = moonshot::live_client() else {
        return support::skip("MOONSHOT_API_KEY is unset");
    };
    // The emoji requirement makes the stream carry multi-byte characters, so
    // a codec that splits UTF-8 across SSE chunk boundaries corrupts visibly.
    let request = moonshot::request(model)
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
    let Some(client) = moonshot::live_client() else {
        return support::skip("MOONSHOT_API_KEY is unset");
    };
    let request = moonshot::request(model)
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
    let Some(client) = moonshot::live_client() else {
        return support::skip("MOONSHOT_API_KEY is unset");
    };
    let request = moonshot::request(model)
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
    let Some(client) = moonshot::live_client() else {
        return support::skip("MOONSHOT_API_KEY is unset");
    };
    let request = moonshot::request(model)
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
    let Some(client) = moonshot::live_client() else {
        return support::skip("MOONSHOT_API_KEY is unset");
    };
    let request = moonshot::request(model)
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
