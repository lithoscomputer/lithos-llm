//! Core completion and streaming behavior for the discovered Kimi K3 endpoint.

use lithos_llm::types::{FinishReason, Message, Role};

use crate::modal;
use crate::support::{self, TestResult};

#[tokio::test]
#[ignore = "live Modal call; run with `mise run test:e2e:live`"]
async fn completes_with_usage() -> TestResult {
    if let Some(skip) = support::live_only("Modal has no two-header twin") {
        return skip;
    }
    let Some(endpoint) = modal::live_endpoint().await? else {
        return modal::missing_credentials();
    };
    let request = modal::request(&endpoint.model)
        .user("In one short sentence, say hello.")
        .build()?;
    let response = endpoint.client.complete(request).await?;

    assert!(!response.text().trim().is_empty());
    assert_eq!(response.finish_reason, FinishReason::Stop);
    assert!(response.usage.input > 0, "no input tokens were counted");
    assert!(
        response.usage.billable_output() > 0,
        "no output tokens were counted"
    );
    assert!(response.raw.is_some());
    assert!(
        response.cost.is_none(),
        "a workspace-scoped passthrough model must not inherit invented catalog pricing"
    );
    Ok(())
}

#[tokio::test]
#[ignore = "live Modal call; run with `mise run test:e2e:live`"]
async fn streams_within_the_contract() -> TestResult {
    if let Some(skip) = support::live_only("Modal has no two-header twin") {
        return skip;
    }
    let Some(endpoint) = modal::live_endpoint().await? else {
        return modal::missing_credentials();
    };
    let request = modal::request(&endpoint.model)
        .user("Reply with one short greeting that contains at least two emoji.")
        .build()?;
    let stream = endpoint.client.stream(request).await?;
    let (_, response) = support::checked_stream(stream).await?;

    assert!(!response.text().trim().is_empty());
    assert!(response.usage.total() > 0);
    Ok(())
}

#[tokio::test]
#[ignore = "live Modal call; run with `mise run test:e2e:live`"]
async fn honors_a_system_prompt() -> TestResult {
    if let Some(skip) = support::live_only("Modal has no two-header twin") {
        return skip;
    }
    let Some(endpoint) = modal::live_endpoint().await? else {
        return modal::missing_credentials();
    };
    let request = modal::request(&endpoint.model)
        .system("When the user says ping, reply with exactly the word PONG and nothing else.")
        .user("ping")
        .build()?;
    let response = endpoint.client.complete(request).await?;

    assert!(
        response.text().to_uppercase().contains("PONG"),
        "the system prompt was not honored: {:?}",
        response.text()
    );
    Ok(())
}

#[tokio::test]
#[ignore = "live Modal call; run with `mise run test:e2e:live`"]
async fn carries_a_multi_turn_conversation() -> TestResult {
    if let Some(skip) = support::live_only("Modal has no two-header twin") {
        return skip;
    }
    let Some(endpoint) = modal::live_endpoint().await? else {
        return modal::missing_credentials();
    };
    let request = modal::request(&endpoint.model)
        .system("Answer with just the city name.")
        .user("What is the capital of France?")
        .message(Message::text(Role::Assistant, "Paris."))
        .user("And of Spain?")
        .build()?;
    let response = endpoint.client.complete(request).await?;

    assert!(
        response.text().contains("Madrid"),
        "the multi-turn answer does not name Madrid: {:?}",
        response.text()
    );
    Ok(())
}

#[tokio::test]
#[ignore = "live Modal call; run with `mise run test:e2e:live`"]
async fn truncates_at_the_output_cap() -> TestResult {
    if let Some(skip) = support::live_only("Modal has no two-header twin") {
        return skip;
    }
    let Some(endpoint) = modal::live_endpoint().await? else {
        return modal::missing_credentials();
    };
    let request = modal::request(&endpoint.model)
        .user("Count upward from one, one number per line, and do not stop.")
        .max_output_tokens(32)
        .build()?;
    let response = endpoint.client.complete(request).await?;

    assert_eq!(response.finish_reason, FinishReason::Length);
    Ok(())
}

#[tokio::test]
#[ignore = "live Modal call; run with `mise run test:e2e:live`"]
async fn honors_a_stop_sequence() -> TestResult {
    if let Some(skip) = support::live_only("Modal has no two-header twin") {
        return skip;
    }
    let Some(endpoint) = modal::live_endpoint().await? else {
        return modal::missing_credentials();
    };
    let request = modal::request(&endpoint.model)
        .system("Follow the instruction literally, with no extra commentary.")
        .user("Recite the numbers one to nine as lowercase English words, separated by spaces.")
        .stop_sequence("five")
        .build()?;
    let response = endpoint.client.complete(request).await?;

    assert!(
        !response.text().contains("six"),
        "the output continued past the stop sequence: {:?}",
        response.text()
    );
    Ok(())
}
