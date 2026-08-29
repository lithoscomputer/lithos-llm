//! Live error classification and connection behavior for Modal.

use std::time::Duration;

use futures_util::StreamExt as _;
use lithos_llm::types::ErrorKind;

use crate::modal;
use crate::support::{self, TestResult};

#[tokio::test]
#[ignore = "live Modal call; run with `mise run test:e2e:live`"]
async fn a_bad_proxy_token_classifies_as_authentication() -> TestResult {
    if let Some(skip) = support::live_only("live 401 classification") {
        return skip;
    }
    let Some(endpoint) = modal::live_endpoint().await? else {
        return modal::missing_credentials();
    };
    let client = modal::client_with_proxy_token("wk-lithos-invalid", "ws-lithos-invalid");
    let request = modal::request(&endpoint.model).user("Hello").build()?;
    let error = client
        .complete(request)
        .await
        .expect_err("an invalid proxy token must not complete");

    assert_eq!(error.kind(), ErrorKind::Authentication);
    Ok(())
}

#[tokio::test]
#[ignore = "live Modal call; run with `mise run test:e2e:live`"]
async fn an_unknown_endpoint_classifies_cleanly() -> TestResult {
    if let Some(skip) = support::live_only("live error classification") {
        return skip;
    }
    let Some(endpoint) = modal::live_endpoint().await? else {
        return modal::missing_credentials();
    };
    let request = modal::request("lithos-e2e-does-not-exist.us-west.modal.direct")
        .user("Hello")
        .max_output_tokens(64)
        .build()?;
    let error = endpoint
        .client
        .complete(request)
        .await
        .expect_err("an unknown endpoint hostname must not complete");

    assert!(
        matches!(
            error.kind(),
            ErrorKind::NotFound | ErrorKind::InvalidRequest | ErrorKind::Provider
        ),
        "Modal's unknown-model shape classified as {:?}: {}",
        error.kind(),
        error.message()
    );
    Ok(())
}

#[tokio::test]
#[ignore = "live Modal call; run with `mise run test:e2e:live`"]
async fn a_tiny_request_timeout_classifies_as_timeout() -> TestResult {
    if let Some(skip) = support::live_only("timing behavior") {
        return skip;
    }
    let Some(endpoint) = modal::live_endpoint().await? else {
        return modal::missing_credentials();
    };
    let request = modal::request(&endpoint.model)
        .user("Hello")
        .timeout(Duration::from_millis(1))
        .build()?;
    let error = endpoint
        .client
        .complete(request)
        .await
        .expect_err("a one-millisecond budget must not complete");

    assert_eq!(error.kind(), ErrorKind::Timeout);
    Ok(())
}

#[tokio::test]
#[ignore = "live Modal call; run with `mise run test:e2e:live`"]
async fn dropping_a_stream_mid_flight_is_clean() -> TestResult {
    if let Some(skip) = support::live_only("live connection behavior") {
        return skip;
    }
    let Some(endpoint) = modal::live_endpoint().await? else {
        return modal::missing_credentials();
    };
    let request = modal::request(&endpoint.model)
        .user("Write three short sentences about rivers.")
        .build()?;
    let mut stream = endpoint.client.stream(request).await?;
    let first = stream.next().await;

    assert!(first.is_some(), "the stream ended before its first event");
    drop(stream);
    Ok(())
}

#[tokio::test]
#[ignore = "live Modal call; run with `mise run test:e2e:live`"]
async fn probe_rate_limit_headers() -> TestResult {
    if let Some(skip) = support::live_only("live rate-limit headers") {
        return skip;
    }
    let Some(endpoint) = modal::live_endpoint().await? else {
        return modal::missing_credentials();
    };
    let request = modal::request(&endpoint.model)
        .user("In one short sentence, say hello.")
        .build()?;
    let response = endpoint.client.complete(request).await?;

    match &response.rate_limits {
        Some(limits) => support::observe(&format!("Modal rate limits: {limits:?}")),
        None => support::observe("Modal sent no rate-limit headers"),
    }
    Ok(())
}
