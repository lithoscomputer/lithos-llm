//! V3 — negative and cross-cutting behavior.
//!
//! The live payload behind each classification matters more than the happy
//! path: `classify.rs` is unit-tested against recorded shapes, and these
//! tests verify the shapes OpenAI sends today still land in the right
//! [`ErrorKind`].

use std::time::Duration;

use futures_util::StreamExt as _;
use lithos_llm::types::{ErrorKind, Speed};

use crate::openai;
use crate::support::{self, TestResult};

/// The workhorse for negative tests: the cheapest roster model.
const MODEL: &str = "gpt-5.6-luna";

#[tokio::test]
#[ignore = "live OpenAI call; run with `mise run test:e2e`"]
async fn a_bad_key_classifies_as_authentication() -> TestResult {
    if let Some(skip) = support::live_only("live 401 classification") {
        return skip;
    }
    let client = openai::client_with_key("lithos-e2e-invalid-key");
    let request = openai::request(MODEL).user("Hello").build()?;
    let error = client
        .complete(request)
        .await
        .expect_err("an invalid key must not complete");
    assert_eq!(
        error.kind(),
        ErrorKind::Authentication,
        "OpenAI's live 401 shape classified as {:?}: {}",
        error.kind(),
        error.message()
    );
    Ok(())
}

#[tokio::test]
#[ignore = "live OpenAI call; run with `mise run test:e2e`"]
async fn an_unknown_passthrough_model_classifies_cleanly() -> TestResult {
    if let Some(skip) = support::live_only("live error classification") {
        return skip;
    }
    let Some(client) = openai::live_client() else {
        return support::skip("OPENAI_API_KEY is unset");
    };
    // The provider allows passthrough, so this reaches the wire and OpenAI
    // itself refuses the unknown id.
    let request = lithos_llm::Request::builder()
        .model(openai::selector("lithos-e2e-does-not-exist"))
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
        "OpenAI's unknown-model shape classified as {:?}: {}",
        error.kind(),
        error.message()
    );
    Ok(())
}

#[tokio::test]
#[ignore = "live OpenAI call; run with `mise run test:e2e`"]
async fn a_tiny_request_timeout_classifies_as_timeout() -> TestResult {
    if let Some(skip) = support::live_only("timing behavior") {
        return skip;
    }
    let Some(client) = openai::live_client() else {
        return support::skip("OPENAI_API_KEY is unset");
    };
    let request = openai::request(MODEL)
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
#[ignore = "live OpenAI call; run with `mise run test:e2e`"]
async fn dropping_a_stream_mid_flight_is_clean() -> TestResult {
    if let Some(skip) = support::live_only("live connection behavior") {
        return skip;
    }
    let Some(client) = openai::live_client() else {
        return support::skip("OPENAI_API_KEY is unset");
    };
    let request = openai::request(MODEL)
        .user("Write three short sentences about rivers.")
        .build()?;
    let mut stream = client.stream(request).await?;
    let first = stream.next().await;
    assert!(first.is_some(), "the stream ended before its first event");
    drop(stream);
    Ok(())
}

/// Records whether OpenAI populates rate-limit headers. Nothing pins this
/// yet; the probe output is what decides whether `rate_limits` gets a hard
/// assertion here.
#[tokio::test]
#[ignore = "live OpenAI call; run with `mise run test:e2e`"]
async fn probe_rate_limit_headers() -> TestResult {
    if let Some(skip) = support::live_only("live rate-limit headers") {
        return skip;
    }
    let Some(client) = openai::live_client() else {
        return support::skip("OPENAI_API_KEY is unset");
    };
    let request = openai::request(MODEL)
        .user("In one short sentence, say hello.")
        .build()?;
    let response = client.complete(request).await?;
    match &response.rate_limits {
        Some(limits) => support::observe(&format!("openai rate limits: {limits:?}")),
        None => support::observe("openai sent no rate-limit headers"),
    }
    Ok(())
}

/// A priority request on a pro row is silently downgraded, not rejected —
/// observed live on 2026-08-30 (`service_tier` came back `default`). The
/// catalog therefore claims no speed tier for the pro rows, and the client
/// refuses the speed before dispatch; this cell pins the refusal.
#[tokio::test]
async fn a_fast_request_on_a_pro_row_is_refused_locally() -> TestResult {
    let client = openai::client_with_key("preflight-key");
    let request = openai::request("gpt-5.5-pro")
        .user("Hello")
        .speed(Speed::Fast)
        .build()?;
    let error = client
        .complete(request)
        .await
        .expect_err("gpt-5.5-pro prices no fast tier, so the client must refuse");
    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    Ok(())
}
