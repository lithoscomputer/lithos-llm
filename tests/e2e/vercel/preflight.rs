//! V0 — preflight.
//!
//! Everything here is free or near-free. The local tests are not ignored; only
//! the live catalog checks spend network calls.

use std::env;

use lithos_llm::types::{ErrorKind, ResponseFormat};

use crate::support::{self, TestResult};
use crate::vercel;

#[tokio::test]
async fn sampling_is_rejected_locally_where_unclaimed() -> TestResult {
    let client = vercel::client_with_key("preflight-key");
    let request = vercel::request("claude-opus-5")
        .user("Hello")
        .temperature(0.0)
        .build()?;
    let error = client
        .complete(request)
        .await
        .expect_err("claude-opus-5 claims no sampling, so the client must refuse");
    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    Ok(())
}

#[tokio::test]
async fn structured_output_is_rejected_locally_on_laguna() -> TestResult {
    let client = vercel::client_with_key("preflight-key");
    let request = vercel::request("laguna-s-2.1")
        .user("Hello")
        .response_format(ResponseFormat::JsonObject)
        .build()?;
    let error = client
        .complete(request)
        .await
        .expect_err("laguna-s-2.1 claims no structured output, so the client must refuse");
    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    Ok(())
}

#[tokio::test]
async fn an_oversized_output_cap_is_rejected_locally() -> TestResult {
    let client = vercel::client_with_key("preflight-key");
    let request = vercel::request("deepseek-v4-flash")
        .user("Hello")
        .max_output_tokens(1_000_000)
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
    let client = vercel::client_with_key("preflight-key");
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

/// The built-in catalog's Vercel rows work end to end: `Client::from_env`
/// resolves the conventional `AI_GATEWAY_API_KEY` mapping, the merged roster
/// resolves the route, and the gateway answers with its in-band cost.
#[tokio::test]
#[ignore = "live Vercel AI Gateway call; run with `mise run test:e2e`"]
async fn the_builtin_catalog_reaches_vercel() -> TestResult {
    if let Some(skip) = support::live_only("the unproxied built-in base URL") {
        return skip;
    }
    if env::var(vercel::KEY_VARIABLE).is_err() {
        return support::skip("AI_GATEWAY_API_KEY is unset");
    }
    let client = lithos_llm::Client::from_env()?.client;
    let request = lithos_llm::Request::builder()
        .model("vercel/deepseek-v4-flash")
        .user("In one short sentence, say hello.")
        .max_output_tokens(8192)
        .build()?;
    let response = client.complete(request).await?;
    assert!(!response.text().trim().is_empty());
    let cost = response.cost.ok_or("the response carries no cost")?;
    assert!(cost.usd_micros > 0);
    Ok(())
}

/// Every configured wire id must be listed by the gateway before the E2E
/// suite attempts completions. The listing needs no key.
#[tokio::test]
#[ignore = "live Vercel AI Gateway call; run with `mise run test:e2e`"]
async fn the_live_listing_contains_every_configured_wire_id() -> TestResult {
    if let Some(skip) = support::live_only("the model listing endpoint") {
        return skip;
    }
    let listing: serde_json::Value = reqwest::Client::new()
        .get("https://ai-gateway.vercel.sh/v1/models")
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
    let catalog = vercel::catalog();
    let provider = catalog.provider(vercel::PROVIDER)?;
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
