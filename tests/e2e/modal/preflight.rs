//! Free catalog and route checks, plus live-only built-in integration checks.

use std::env;

use lithos_llm::catalog::AuthScheme;

use crate::modal;
use crate::support::{self, TestResult};

#[test]
fn catalog_declares_a_portable_passthrough_provider() -> TestResult {
    let catalog = modal::catalog();
    let provider = catalog.provider(modal::PROVIDER)?;

    assert_eq!(
        provider.base_url(),
        "https://inference.us-west.modal.direct/v1"
    );
    assert!(matches!(provider.auth(), AuthScheme::Headers));
    assert!(provider.allows_passthrough());
    assert_eq!(provider.models().len(), 0);
    assert!(provider.default_model().is_none());
    assert_eq!(
        provider
            .metadata()
            .get("fabro")
            .and_then(|value| value["agent_profile"].as_str()),
        Some("kimi")
    );
    Ok(())
}

#[test]
fn a_workspace_endpoint_hostname_resolves_through_passthrough() -> TestResult {
    let client = modal::client_with_proxy_token("preflight-key", "preflight-secret");
    let request = modal::request("example--ep-kimi-k3-server.us-west.modal.direct")
        .user("Hello")
        .build()?;
    let route = client.resolve_route(&request)?;

    assert_eq!(route.provider().id().as_str(), modal::PROVIDER);
    assert!(route.model().is_passthrough());
    assert_eq!(
        route.model().api_model(),
        "example--ep-kimi-k3-server.us-west.modal.direct"
    );
    Ok(())
}

#[test]
fn a_bare_workspace_hostname_is_not_global() -> TestResult {
    let client = modal::client_with_proxy_token("preflight-key", "preflight-secret");
    let request = lithos_llm::Request::builder()
        .model("example--ep-kimi-k3-server.us-west.modal.direct")
        .user("Hello")
        .build()?;
    assert!(
        client.resolve_route(&request).is_err(),
        "a workspace endpoint must be qualified with the Modal provider"
    );
    Ok(())
}

/// The built-in provider, conventional two-header credentials, dynamic model
/// listing, and passthrough route work together end to end.
#[tokio::test]
#[ignore = "live Modal call; run with `mise run test:e2e:live`"]
async fn the_builtin_catalog_reaches_modal() -> TestResult {
    if let Some(skip) = support::live_only("the unproxied built-in base URL") {
        return skip;
    }
    if env::var(modal::KEY_VARIABLE).is_err() || env::var(modal::SECRET_VARIABLE).is_err() {
        return modal::missing_credentials();
    }
    let Some(endpoint) = modal::live_endpoint().await? else {
        return modal::missing_credentials();
    };
    let client = lithos_llm::Client::from_env()?.client;
    let request = modal::request(&endpoint.model)
        .user("In one short sentence, say hello.")
        .build()?;
    let response = client.complete(request).await?;

    assert!(!response.text().trim().is_empty());
    assert!(response.usage.input > 0);
    assert!(response.usage.billable_output() > 0);
    Ok(())
}

#[tokio::test]
#[ignore = "live Modal call; run with `mise run test:e2e:live`"]
async fn the_live_listing_exposes_a_kimi_k3_endpoint() -> TestResult {
    if let Some(skip) = support::live_only("the model listing endpoint") {
        return skip;
    }
    let Some(endpoint) = modal::live_endpoint().await? else {
        return modal::missing_credentials();
    };
    assert!(endpoint.model.to_ascii_lowercase().contains("kimi-k3"));
    assert!(endpoint.model.ends_with(".modal.direct"));
    Ok(())
}
