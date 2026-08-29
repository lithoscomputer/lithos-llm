//! V0 — preflight.
//!
//! Everything here is free or near-free. The local tests are not ignored, so
//! the routine suite already guards the E2E catalog against rot; only the
//! live roster check spends a network call, and even that spends no tokens.

use std::env;

use lithos_llm::types::{ErrorKind, ResponseFormat};

use crate::support::{self, TestResult};
use crate::venice;

/// Every roster row and the api_model id it must put on the wire.
///
/// This is the mapping the resolver must produce; a row whose live id drifts
/// fails here before any token is spent.
const ROSTER: &[(&str, &str)] = &[
    ("kimi-k3", "kimi-k3"),
    ("kimi-k3-fast", "kimi-k3-fast-api"),
    ("grok-4.6", "grok-4-6"),
    ("glm-5.3", "z-ai-glm-5-3"),
    ("deepseek-v4-flash", "deepseek-v4-flash-0731"),
    ("deepseek-v4-pro", "deepseek-v4-pro-0813"),
    ("qwen3.8-max", "qwen-3-8-max"),
    ("qwen3.8-27b", "qwen-3-8-27b"),
    ("claude-fable-5", "claude-fable-5"),
    ("claude-opus-5", "claude-opus-5"),
    ("claude-sonnet-5", "claude-sonnet-5"),
    ("claude-opus-4.8", "claude-opus-4-8"),
    ("gpt-5.6-sol", "openai-gpt-56-sol"),
    ("gpt-5.6-terra", "openai-gpt-56-terra"),
    ("gpt-5.6-luna", "openai-gpt-56-luna"),
    ("gpt-5.5", "openai-gpt-55"),
];

#[test]
fn the_catalog_holds_the_whole_roster() -> TestResult {
    let catalog = venice::catalog();
    let provider = catalog.provider(venice::PROVIDER)?;
    assert_eq!(provider.models().len(), ROSTER.len());
    Ok(())
}

#[test]
fn every_roster_row_resolves_to_its_wire_id() -> TestResult {
    let client = venice::client_with_key("preflight-key");
    for (model, api_model) in ROSTER {
        let route = client.resolve_route(&venice::request(model).user("Hello").build()?)?;
        assert_eq!(
            route.api_model(),
            *api_model,
            "roster model {model} resolves to the wrong wire id"
        );
    }
    Ok(())
}

#[tokio::test]
async fn sampling_is_rejected_locally_where_unclaimed() -> TestResult {
    let client = venice::client_with_key("preflight-key");
    let request = venice::request("kimi-k3")
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
async fn structured_output_is_rejected_locally_on_qwen_max() -> TestResult {
    let client = venice::client_with_key("preflight-key");
    let request = venice::request("qwen3.8-max")
        .user("Hello")
        .response_format(ResponseFormat::JsonObject)
        .build()?;
    let error = client
        .complete(request)
        .await
        .expect_err("qwen3.8-max claims no structured output, so the client must refuse");
    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    Ok(())
}

#[tokio::test]
async fn an_oversized_output_cap_is_rejected_locally() -> TestResult {
    let client = venice::client_with_key("preflight-key");
    let request = venice::request("deepseek-v4-flash")
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
    let client = venice::client_with_key("preflight-key");
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

/// The built-in catalog's Venice rows work end to end: `Client::from_env`
/// resolves the conventional `VENICE_API_KEY` mapping, the merged roster
/// resolves the route, and Venice answers with its in-band cost.
#[tokio::test]
#[ignore = "live Venice call; run with `mise run test:e2e`"]
async fn the_builtin_catalog_reaches_venice() -> TestResult {
    if env::var(venice::KEY_VARIABLE).is_err() {
        return support::skip("VENICE_API_KEY is unset");
    }
    let client = lithos_llm::Client::from_env()?.client;
    let request = lithos_llm::Request::builder()
        .model("venice/deepseek-v4-flash")
        .user("In one short sentence, say hello.")
        .max_output_tokens(8192)
        .build()?;
    let response = client.complete(request).await?;
    assert!(!response.text().trim().is_empty());
    let cost = response.cost.ok_or("the response carries no cost")?;
    assert!(cost.usd_micros > 0);
    Ok(())
}

/// The one preflight test that goes to the network: the live model listing
/// must still contain every wire id the roster resolves to. This is the
/// cheapest possible drift alarm — a withdrawn or renamed model fails here
/// before any completion is attempted.
#[tokio::test]
#[ignore = "live Venice call; run with `mise run test:e2e`"]
async fn the_live_listing_contains_every_roster_wire_id() -> TestResult {
    let Ok(key) = env::var(venice::KEY_VARIABLE) else {
        return support::skip("VENICE_API_KEY is unset");
    };
    let listing: serde_json::Value = reqwest::Client::new()
        .get("https://api.venice.ai/api/v1/models")
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
    for (model, api_model) in ROSTER {
        assert!(
            live.contains(api_model),
            "roster model {model} is gone from the live listing (wire id {api_model})"
        );
    }
    Ok(())
}
