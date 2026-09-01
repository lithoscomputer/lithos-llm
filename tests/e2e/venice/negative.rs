//! V3 — negative and cross-cutting behavior.
//!
//! Error classification is where OpenAI-compatible vendors drift most, so
//! the live payload behind each classification matters more than the happy
//! path: `classify.rs` is unit-tested against recorded shapes, and these
//! tests verify the shapes Venice sends today still land in the right
//! [`ErrorKind`].

use std::time::Duration;

use futures_util::StreamExt as _;
use lithos_llm::types::{ErrorKind, Message, Role};

use crate::support::{self, TestResult};
use crate::venice;

/// The workhorse for negative tests: the cheapest roster model.
const MODEL: &str = "deepseek-v4-flash";

#[tokio::test]
#[ignore = "live Venice call; run with `mise run test:e2e`"]
async fn a_bad_key_classifies_as_authentication() -> TestResult {
    if let Some(skip) = support::live_only("live 401 classification") {
        return skip;
    }
    let client = venice::client_with_key("lithos-e2e-invalid-key");
    let request = venice::request(MODEL).user("Hello").build()?;
    let error = client
        .complete(request)
        .await
        .expect_err("an invalid key must not complete");
    assert_eq!(
        error.kind(),
        ErrorKind::Authentication,
        "Venice's live 401 shape classified as {:?}: {}",
        error.kind(),
        error.message()
    );
    Ok(())
}

#[tokio::test]
#[ignore = "live Venice call; run with `mise run test:e2e`"]
async fn an_unknown_passthrough_model_classifies_cleanly() -> TestResult {
    if let Some(skip) = support::live_only("live error classification") {
        return skip;
    }
    let Some(client) = venice::live_client() else {
        return support::skip("VENICE_API_KEY is unset");
    };
    // The provider allows passthrough, so this reaches the wire and Venice
    // itself refuses the unknown id.
    let request = lithos_llm::Request::builder()
        .model(venice::selector("lithos-e2e-does-not-exist"))
        .user("Hello")
        .max_output_tokens(64)
        .build()?;
    let error = client
        .complete(request)
        .await
        .expect_err("an unknown wire model must not complete");
    assert!(
        matches!(
            error.kind(),
            ErrorKind::NotFound | ErrorKind::InvalidRequest | ErrorKind::Provider
        ),
        "Venice's unknown-model shape classified as {:?}: {}",
        error.kind(),
        error.message()
    );
    Ok(())
}

#[tokio::test]
#[ignore = "live Venice call; run with `mise run test:e2e`"]
async fn a_tiny_request_timeout_classifies_as_timeout() -> TestResult {
    if let Some(skip) = support::live_only("timing behavior") {
        return skip;
    }
    let Some(client) = venice::live_client() else {
        return support::skip("VENICE_API_KEY is unset");
    };
    let request = venice::request(MODEL)
        .user("Hello")
        .timeout(Duration::from_millis(1))
        .build()?;
    let error = client
        .complete(request)
        .await
        .expect_err("a one-millisecond budget must not complete");
    assert_eq!(error.kind(), ErrorKind::Timeout);
    Ok(())
}

/// Dropping a live stream mid-flight must simply end it: no hang, no panic,
/// and the connection is released. The test's own timeout bounds the "no
/// hang" half.
#[tokio::test]
#[ignore = "live Venice call; run with `mise run test:e2e`"]
async fn dropping_a_stream_mid_flight_is_clean() -> TestResult {
    if let Some(skip) = support::live_only("live connection behavior") {
        return skip;
    }
    let Some(client) = venice::live_client() else {
        return support::skip("VENICE_API_KEY is unset");
    };
    let request = venice::request(MODEL)
        .user("Write three short sentences about rivers.")
        .build()?;
    let mut stream = client.stream(request).await?;
    let first = stream.next().await;
    assert!(first.is_some(), "the stream ended before its first event");
    drop(stream);
    Ok(())
}

/// Records whether Venice populates rate-limit headers. Nothing pins this
/// yet; the probe output is what decides whether `rate_limits` gets a hard
/// assertion here.
/// Venice's Anthropic translation breaks when a system message carries a
/// cache breakpoint and a second system message follows it: the upstream
/// system array gets an empty text block and Anthropic answers 400 `system:
/// text content blocks must contain non-whitespace text`. Without the
/// breakpoint the same conversation succeeds, and OpenRouter accepts the
/// shape as-is (probed 2026-09-01). The smoke suite skips the Claude rows'
/// mid-conversation cell over this; when Venice fixes the translation this
/// cell turns red and that skip comes off.
#[tokio::test]
#[ignore = "live Venice call; run with `mise run test:e2e`"]
async fn a_cached_system_prefix_beside_a_later_system_message_is_rejected() -> TestResult {
    if let Some(skip) = support::live_only("live 400 classification") {
        return skip;
    }
    let Some(client) = venice::live_client() else {
        return support::skip("VENICE_API_KEY is unset");
    };
    // The default auto-cache marks the leading system prompt on a row that
    // claims breakpoints, which is exactly the shape Venice rejects.
    let request = venice::request("claude-fable-5.1")
        .system("Answer with just the city name.")
        .user("What is the capital of France?")
        .message(Message::text(Role::Assistant, "Paris."))
        .user("And of Spain?")
        .message(Message::text(
            Role::System,
            "From now on, write every answer in uppercase letters only.",
        ))
        .build()?;
    let error = client
        .complete(request)
        .await
        .expect_err("Venice rejected this shape on 2026-09-01; a success means the bug is fixed");
    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert!(
        error.message().contains("non-whitespace text"),
        "Venice's rejection changed shape: {}",
        error.message()
    );
    Ok(())
}

#[tokio::test]
#[ignore = "live Venice call; run with `mise run test:e2e`"]
async fn probe_rate_limit_headers() -> TestResult {
    if let Some(skip) = support::live_only("live rate-limit headers") {
        return skip;
    }
    let Some(client) = venice::live_client() else {
        return support::skip("VENICE_API_KEY is unset");
    };
    let request = venice::request(MODEL)
        .user("In one short sentence, say hello.")
        .build()?;
    let response = client.complete(request).await?;
    match &response.rate_limits {
        Some(limits) => support::observe(&format!("venice rate limits: {limits:?}")),
        None => support::observe("venice sent no rate-limit headers"),
    }
    Ok(())
}
