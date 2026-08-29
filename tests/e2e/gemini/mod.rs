//! The Gemini live-only E2E suite.
//!
//! Gemini uses the native GenerateContent adapter and codec. The suite checks
//! the full roster, catalog-priced usage, implicit prompt caching, thinking,
//! multimodal input, and Gemini error payloads.
//!
//! The twin does not proxy `/v1beta/models/...` paths. Every network cell
//! therefore skips under record and replay until native Gemini support exists.

mod caching;
mod negative;
mod preflight;
mod reasoning;
mod sampling;
mod smoke;
mod structured;
mod tools;
mod vision;

use std::env;

use lithos_llm::catalog::{Catalog, ModelCapabilities};
use lithos_llm::credentials::{CredentialHeader, Credentials, SecretValue, StaticCredentials};
use lithos_llm::middleware::{RetryMiddleware, RetryPolicy};
use lithos_llm::types::RequestBuilder;
use lithos_llm::{Client, Request};

use crate::support;

pub(crate) const PROVIDER: &str = "gemini";
pub(crate) const KEY_VARIABLE: &str = "GEMINI_API_KEY";
pub(crate) const FALLBACK_KEY_VARIABLE: &str = "GOOGLE_API_KEY";

const CATALOG_TOML: &str = include_str!("../gemini_catalog.toml");

macro_rules! model_tests {
    ($runner:path) => {
        model_tests!(@expand $runner,
            gemini_3_1_pro_preview "gemini-3.1-pro-preview",
            gemini_3_1_pro_preview_customtools "gemini-3.1-pro-preview-customtools",
            gemini_3_5_flash "gemini-3.5-flash",
            gemini_3_flash_preview "gemini-3-flash-preview",
            gemini_3_1_flash_lite "gemini-3.1-flash-lite",
        );
    };
    (@expand $runner:path, $($name:ident $model:literal,)+) => {
        $(
            #[tokio::test]
            #[ignore = "live Gemini call; run with mise run test:e2e:live"]
            async fn $name() -> crate::support::TestResult {
                $runner($model).await
            }
        )+
    };
}

macro_rules! family_tests {
    ($runner:path) => {
        crate::gemini::model_tests!(@expand $runner,
            gemini_3_5_flash "gemini-3.5-flash",
        );
    };
}

pub(crate) use family_tests;
pub(crate) use model_tests;

pub(crate) fn catalog() -> Catalog {
    Catalog::builder()
        .toml_layer("gemini-e2e", CATALOG_TOML)
        .expect("the Gemini E2E catalog should parse")
        .build()
        .expect("the Gemini E2E catalog should validate")
}

pub(crate) fn capabilities(model: &str) -> ModelCapabilities {
    catalog()
        .model(PROVIDER, model)
        .expect("the roster model should be in the Gemini E2E catalog")
        .capabilities()
}

/// Returns a live client. Native GenerateContent calls skip under replay and
/// record because the twin does not proxy their paths.
pub(crate) fn live_client() -> Option<Client> {
    if support::Backend::from_env() != support::Backend::Live {
        let _ = support::live_only("native Gemini GenerateContent transport");
        return None;
    }
    let key = api_key()?;
    Some(client_with_key(&key))
}

/// Reads Gemini's preferred conventional key and then Google's fallback.
pub(crate) fn api_key() -> Option<String> {
    env::var(KEY_VARIABLE)
        .ok()
        .filter(|key| !key.is_empty())
        .or_else(|| {
            env::var(FALLBACK_KEY_VARIABLE)
                .ok()
                .filter(|key| !key.is_empty())
        })
}

pub(crate) fn client_with_key(key: &str) -> Client {
    let build = Client::builder()
        .catalog(catalog())
        .credentials(StaticCredentials::new().with(
            PROVIDER,
            Credentials::header(CredentialHeader::new(
                "x-goog-api-key",
                SecretValue::new(key),
            )),
        ))
        .middleware(RetryMiddleware::new(
            RetryPolicy::exponential().max_attempts(3).jitter(true),
        ))
        .build()
        .expect("the Gemini E2E client should build");
    assert!(
        build.issues.is_empty(),
        "the Gemini E2E client reported provider build issues: {:?}",
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
