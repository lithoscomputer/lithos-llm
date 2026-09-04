//! The Anthropic live-only E2E suite.
//!
//! Anthropic uses the native Messages adapter and codec. The suite checks the
//! full roster, catalog-priced usage, explicit prompt caching, adaptive and
//! manual thinking, and Anthropic error payloads.
//!
//! The twin does not proxy /v1/messages. Every network cell therefore skips
//! under record and replay until twin-native Messages support exists.

pub(crate) fn protocol_options(model: &str) -> ModelProtocolOptions {
    catalog()
        .model(PROVIDER, model)
        .expect("roster model")
        .protocol_options()
}

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

use lithos_llm::catalog::{Catalog, ModelCapabilities, ModelProtocolOptions};
use lithos_llm::credentials::{CredentialHeader, Credentials, SecretValue, StaticCredentials};
use lithos_llm::middleware::{RetryMiddleware, RetryPolicy};
use lithos_llm::types::RequestBuilder;
use lithos_llm::{Client, Request};

use crate::support;

pub(crate) const PROVIDER: &str = "anthropic";
pub(crate) const KEY_VARIABLE: &str = "ANTHROPIC_API_KEY";

const CATALOG_TOML: &str = include_str!("../anthropic_catalog.toml");

macro_rules! model_tests {
    ($runner:path) => {
        model_tests!(@expand $runner,
            claude_fable_5_1 "claude-fable-5.1",
            claude_fable_5 "claude-fable-5",
            claude_opus_5 "claude-opus-5",
            claude_sonnet_5 "claude-sonnet-5",
            claude_opus_4_8 "claude-opus-4.8",
            claude_opus_4_7 "claude-opus-4.7",
            claude_opus_4_6 "claude-opus-4.6",
            claude_sonnet_4_5 "claude-sonnet-4.5",
            claude_sonnet_4_6 "claude-sonnet-4.6",
            claude_haiku_4_5 "claude-haiku-4.5",
        );
    };
    (@expand $runner:path, $($name:ident $model:literal,)+) => {
        $(
            #[tokio::test]
            #[ignore = "live Anthropic call; run with mise run test:e2e:live"]
            async fn $name() -> crate::support::TestResult {
                $runner($model).await
            }
        )+
    };
}

/// Fable 5.1 sits beside Fable 5 rather than behind it: the family cells
/// exercise tool selection and parallel-call batching, which are exactly the
/// behaviors the release notes say changed between the two.
macro_rules! family_tests {
    ($runner:path) => {
        crate::anthropic::model_tests!(@expand $runner,
            claude_fable_5_1 "claude-fable-5.1",
            claude_fable_5 "claude-fable-5",
            claude_opus_5 "claude-opus-5",
            claude_sonnet_4_6 "claude-sonnet-4.6",
            claude_haiku_4_5 "claude-haiku-4.5",
        );
    };
}

pub(crate) use family_tests;
pub(crate) use model_tests;

pub(crate) fn catalog() -> Catalog {
    Catalog::builder()
        .toml_layer("anthropic-e2e", CATALOG_TOML)
        .expect("the Anthropic E2E catalog should parse")
        .build()
        .expect("the Anthropic E2E catalog should validate")
}

pub(crate) fn capabilities(model: &str) -> ModelCapabilities {
    catalog()
        .model(PROVIDER, model)
        .expect("the roster model should be in the Anthropic E2E catalog")
        .capabilities()
}

/// Returns a live client. Native Messages calls skip under record and replay.
pub(crate) fn live_client() -> Option<Client> {
    if support::Backend::from_env() != support::Backend::Live {
        let _ = support::live_only("native Anthropic Messages transport");
        return None;
    }
    let key = env::var(KEY_VARIABLE).ok().filter(|key| !key.is_empty())?;
    Some(client_with_key(&key))
}

pub(crate) fn client_with_key(key: &str) -> Client {
    let build = Client::builder()
        .catalog(catalog())
        .credentials(StaticCredentials::new().with(
            PROVIDER,
            Credentials::header(CredentialHeader::new("x-api-key", SecretValue::new(key))),
        ))
        .middleware(RetryMiddleware::new(
            RetryPolicy::exponential().max_attempts(3).jitter(true),
        ))
        .build()
        .expect("the Anthropic E2E client should build");
    assert!(
        build.issues.is_empty(),
        "the Anthropic E2E client reported provider build issues: {:?}",
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
