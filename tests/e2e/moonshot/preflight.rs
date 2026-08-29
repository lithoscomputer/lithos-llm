//! V0 — preflight.
//!
//! Everything here is free or near-free. The local tests are not ignored; only
//! the live catalog checks spend network calls.

use std::env;

use lithos_llm::types::{CostSource, ErrorKind};

use crate::moonshot;
use crate::support::{self, TestResult};

#[tokio::test]
async fn sampling_is_rejected_locally_where_unclaimed() -> TestResult {
    let client = moonshot::client_with_key("preflight-key");
    let request = moonshot::request("kimi-k3")
        .user("Hello")
        .temperature(0.0)
        .build()?;
    let error = client
        .complete(request)
        .await
        .expect_err("kimi-k3 claims no sampling, so the client must refuse");
    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    Ok(())
}

#[tokio::test]
async fn an_oversized_output_cap_is_rejected_locally() -> TestResult {
    let client = moonshot::client_with_key("preflight-key");
    let request = moonshot::request("kimi-k3")
        .user("Hello")
        .max_output_tokens(2_000_000)
        .build()?;
    let error = client
        .complete(request)
        .await
        .expect_err("an output cap above the model limit must be refused");
    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(error.provider_code(), Some("max_output_tokens"));
    Ok(())
}

#[test]
fn an_unknown_model_is_a_selection_error() -> TestResult {
    let client = moonshot::client_with_key("preflight-key");
    let request = lithos_llm::Request::builder()
        .model("no-such-model-anywhere")
        .user("Hello")
        .build()?;
    assert!(
        client.resolve_route(&request).is_err(),
        "an unknown bare model id must not resolve"
    );
    Ok(())
}

/// The built-in Moonshot row works end to end: `Client::from_env` resolves
/// the conventional `MOONSHOT_API_KEY` mapping, K3 resolves, and catalog
/// pricing produces a cost.
#[tokio::test]
#[ignore = "live Moonshot call; run with `mise run test:e2e`"]
async fn the_builtin_catalog_reaches_moonshot() -> TestResult {
    if let Some(skip) = support::live_only("the unproxied built-in base URL") {
        return skip;
    }
    if env::var(moonshot::KEY_VARIABLE).is_err() {
        return support::skip("MOONSHOT_API_KEY is unset");
    }
    let client = lithos_llm::Client::from_env()?.client;
    let request = lithos_llm::Request::builder()
        .model("moonshot/kimi-k3")
        .user("In one short sentence, say hello.")
        .max_output_tokens(8192)
        .build()?;
    let response = client.complete(request).await?;
    assert!(!response.text().trim().is_empty());
    let cost = response.cost.ok_or("the response carries no cost")?;
    assert_eq!(cost.source, CostSource::Catalog);
    assert!(cost.usd_micros > 0);
    Ok(())
}

/// Every configured wire id must be available from Moonshot before the E2E
/// suite attempts completions.
#[tokio::test]
#[ignore = "live Moonshot call; run with `mise run test:e2e`"]
async fn the_live_listing_contains_every_configured_wire_id() -> TestResult {
    if let Some(skip) = support::live_only("the model listing endpoint") {
        return skip;
    }
    let Ok(key) = env::var(moonshot::KEY_VARIABLE) else {
        return support::skip("MOONSHOT_API_KEY is unset");
    };
    let listing: serde_json::Value = reqwest::Client::new()
        .get("https://api.moonshot.ai/v1/models")
        .bearer_auth(key)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let live: Vec<&str> = listing["data"]
        .as_array()
        .ok_or("the model listing carries no data array")?
        .iter()
        .filter_map(|model| model["id"].as_str())
        .collect();
    let catalog = moonshot::catalog();
    let provider = catalog.provider(moonshot::PROVIDER)?;
    for model in provider.models() {
        let api_model = model.api_model();
        assert!(
            live.contains(&api_model),
            "catalog model {} is gone from the live listing (wire id {api_model})",
            model.id()
        );
    }
    Ok(())
}
