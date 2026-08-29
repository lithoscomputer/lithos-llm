//! The Modal live suite.
//!
//! Modal's Shared Endpoint router is OpenAI-compatible, but its model ids are
//! endpoint hostnames scoped to a proxy token. The suite therefore discovers
//! the accessible Kimi K3 endpoint through `GET /v1/models` instead of
//! committing a workspace-specific hostname.
//!
//! Every network test is live-only. The current twin accepts one upstream
//! bearer secret, while this suite deliberately covers Modal's separate
//! `Modal-Key` and `Modal-Secret` credential headers.

mod capabilities;
mod negative;
mod preflight;
mod smoke;

use std::env;
use std::error::Error as StdError;

use lithos_llm::catalog::Catalog;
use lithos_llm::credentials::{CredentialHeader, Credentials, SecretValue, StaticCredentials};
use lithos_llm::middleware::{RetryMiddleware, RetryPolicy};
use lithos_llm::types::RequestBuilder;
use lithos_llm::{Client, Request};
use serde_json::Value;

use crate::support;

pub(crate) const PROVIDER: &str = "modal";

/// The environment variables used by the source fabro integration and this
/// repository's local live-test configuration.
pub(crate) const KEY_VARIABLE: &str = "MODAL_TOKEN_ID";
pub(crate) const SECRET_VARIABLE: &str = "MODAL_TOKEN_SECRET";

const PROXY_KEY_VARIABLE: &str = "MODAL_PROXY_TOKEN_ID";
const PROXY_SECRET_VARIABLE: &str = "MODAL_PROXY_TOKEN_SECRET";
const CATALOG_TOML: &str = include_str!("../modal_catalog.toml");
const MODELS_URL: &str = "https://inference.us-west.modal.direct/v1/models";

pub(crate) struct LiveEndpoint {
    pub(crate) client: Client,
    pub(crate) model:  String,
}

pub(crate) fn catalog() -> Catalog {
    Catalog::builder()
        .toml_layer("modal-e2e", CATALOG_TOML)
        .expect("the Modal E2E catalog should parse")
        .build()
        .expect("the Modal E2E catalog should validate")
}

/// Finds the configured proxy-token pair without exposing either value.
///
/// Modal's current documentation uses the `MODAL_PROXY_TOKEN_*` names. The
/// fabro source and this repository's existing `.env` use the shorter names,
/// so the test harness accepts both complete pairs.
pub(crate) fn proxy_token_pair() -> Option<(String, String)> {
    complete_pair(PROXY_KEY_VARIABLE, PROXY_SECRET_VARIABLE)
        .or_else(|| complete_pair(KEY_VARIABLE, SECRET_VARIABLE))
}

fn complete_pair(key_variable: &str, secret_variable: &str) -> Option<(String, String)> {
    let key = env::var(key_variable)
        .ok()
        .filter(|value| !value.is_empty())?;
    let secret = env::var(secret_variable)
        .ok()
        .filter(|value| !value.is_empty())?;
    Some((key, secret))
}

/// Discovers the workspace-scoped Kimi K3 endpoint and builds its client.
pub(crate) async fn live_endpoint() -> Result<Option<LiveEndpoint>, Box<dyn StdError>> {
    let Some((key, secret)) = proxy_token_pair() else {
        return Ok(None);
    };
    let listing: Value = reqwest::Client::new()
        .get(MODELS_URL)
        .bearer_auth(format!("{key}.{secret}"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let models = listing["data"]
        .as_array()
        .ok_or("the Modal model listing carries no data array")?;
    let model = models
        .iter()
        .filter_map(|entry| entry["id"].as_str())
        .find(|id| id.to_ascii_lowercase().contains("kimi-k3"))
        .ok_or("the Modal model listing carries no Kimi K3 endpoint")?
        .to_owned();

    Ok(Some(LiveEndpoint {
        client: client_with_proxy_token(&key, &secret),
        model,
    }))
}

/// A client using Modal's two proxy-token headers, valid or not.
pub(crate) fn client_with_proxy_token(key: &str, secret: &str) -> Client {
    let credentials = Credentials::headers([
        CredentialHeader::new("Modal-Key", SecretValue::new(key)),
        CredentialHeader::new("Modal-Secret", SecretValue::new(secret)),
    ]);
    let build = Client::builder()
        .catalog(catalog())
        .credentials(StaticCredentials::new().with(PROVIDER, credentials))
        .middleware(RetryMiddleware::new(
            RetryPolicy::exponential().max_attempts(3).jitter(true),
        ))
        .build()
        .expect("the Modal E2E client should build");
    assert!(
        build.issues.is_empty(),
        "the Modal E2E client reported provider build issues: {:?}",
        build.issues
    );
    build.client
}

pub(crate) fn selector(model: &str) -> String {
    format!("{PROVIDER}/{model}")
}

pub(crate) fn request(model: &str) -> RequestBuilder {
    Request::builder()
        .model(selector(model))
        .max_output_tokens(8192)
        .timeout(support::LIVE_TIMEOUT)
}

pub(crate) fn missing_credentials() -> support::TestResult {
    support::skip(
        "MODAL_PROXY_TOKEN_ID/MODAL_PROXY_TOKEN_SECRET and \
         MODAL_TOKEN_ID/MODAL_TOKEN_SECRET are unset",
    )
}
