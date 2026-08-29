//! V0 — local catalog checks and low-cost live preflight.

use std::collections::BTreeSet;
use std::error::Error as StdError;

use lithos_llm::catalog::Catalog;
use lithos_llm::types::{CostSource, ErrorKind};

use crate::gemini;
use crate::support::{self, TestResult};

#[test]
fn the_e2e_roster_matches_the_builtin_roster() -> TestResult {
    let builtin = Catalog::builder().with_builtin().build()?;
    let e2e = gemini::catalog();
    let ids = |catalog: &Catalog| -> Result<BTreeSet<String>, Box<dyn StdError>> {
        Ok(catalog
            .provider(gemini::PROVIDER)?
            .models()
            .map(|model| model.id().to_string())
            .collect())
    };
    assert_eq!(ids(&builtin)?, ids(&e2e)?);
    Ok(())
}

#[tokio::test]
async fn an_oversized_output_cap_is_rejected_locally() -> TestResult {
    let client = gemini::client_with_key("preflight-key");
    let request = gemini::request("gemini-3.1-flash-lite")
        .user("Hello")
        .max_output_tokens(65_537)
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
    let client = gemini::client_with_key("preflight-key");
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

/// The built-in row, conventional credential fallback, native URL, and
/// catalog pricing work together in one request.
#[tokio::test]
#[ignore = "live Gemini call; run with `mise run test:e2e:live`"]
async fn the_builtin_catalog_reaches_gemini() -> TestResult {
    if let Some(skip) = support::live_only("the native GenerateContent path") {
        return skip;
    }
    if gemini::api_key().is_none() {
        return support::skip("GEMINI_API_KEY and GOOGLE_API_KEY are unset");
    }
    let client = lithos_llm::Client::from_env()?.client;
    let request = lithos_llm::Request::builder()
        .model("gemini/gemini-3.5-flash")
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

/// Every configured wire id must remain available from the authenticated
/// Gemini model listing.
#[tokio::test]
#[ignore = "live Gemini call; run with `mise run test:e2e:live`"]
async fn the_live_listing_contains_every_configured_wire_id() -> TestResult {
    if let Some(skip) = support::live_only("the model listing endpoint") {
        return skip;
    }
    let Some(key) = gemini::api_key() else {
        return support::skip("GEMINI_API_KEY and GOOGLE_API_KEY are unset");
    };
    let listing: serde_json::Value = reqwest::Client::new()
        .get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000")
        .header("x-goog-api-key", key)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let live: BTreeSet<&str> = listing["models"]
        .as_array()
        .ok_or("the model listing carries no models array")?
        .iter()
        .filter(|model| {
            model["supportedGenerationMethods"]
                .as_array()
                .is_some_and(|methods| {
                    methods
                        .iter()
                        .any(|method| method.as_str() == Some("generateContent"))
                })
        })
        .filter_map(|model| model["name"].as_str())
        .filter_map(|name| name.strip_prefix("models/"))
        .collect();
    let catalog = gemini::catalog();
    let provider = catalog.provider(gemini::PROVIDER)?;
    for model in provider.models() {
        let api_model = model.api_model();
        assert!(
            live.contains(api_model),
            "catalog model {} is gone from the live listing (wire id {api_model})",
            model.id()
        );
    }
    Ok(())
}
