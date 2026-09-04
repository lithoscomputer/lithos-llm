//! The Venice live suite.
//!
//! Venice runs through the `openai-compatible` adapter and the `openai-chat`
//! codec, so this suite is both the endpoint check for Venice itself and the
//! live exercise of that shared code path. The Venice-specific behavior under
//! test is the in-band top-level `cost` field and the classification of
//! Venice error payloads.
//!
//! The roster lives in `venice_catalog.toml`; the suite iterates it through
//! the [`model_tests`] and [`family_tests`] macros. Capability-gated runners
//! consult the catalog row and skip models whose row does not claim the
//! capability, so the catalog stays the single source of truth for which
//! cells exist.

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
use lithos_llm::credentials::{Credentials, SecretValue, StaticCredentials};
use lithos_llm::middleware::{RetryMiddleware, RetryPolicy};
use lithos_llm::types::RequestBuilder;
use lithos_llm::{Client, Request};

use crate::support;

pub(crate) const PROVIDER: &str = "venice";

/// The environment variable holding the Venice API key.
pub(crate) const KEY_VARIABLE: &str = "VENICE_API_KEY";

const CATALOG_TOML: &str = include_str!("../venice_catalog.toml");

/// Expands one live test per roster model.
///
/// `$runner` is an `async fn(&'static str) -> TestResult` taking the catalog
/// model id. Every generated test is ignored, so the roster only runs under
/// `mise run test:e2e`.
macro_rules! model_tests {
    ($runner:path) => {
        model_tests!(@expand $runner,
            kimi_k3 "kimi-k3",
            kimi_k3_fast "kimi-k3-fast",
            grok_4_6 "grok-4.6",
            glm_5_3 "glm-5.3",
            deepseek_v4_flash "deepseek-v4-flash",
            deepseek_v4_pro "deepseek-v4-pro",
            qwen_3_8_max "qwen3.8-max",
            qwen_3_8_27b "qwen3.8-27b",
            claude_fable_5_1 "claude-fable-5.1",
            claude_fable_5 "claude-fable-5",
            claude_opus_5 "claude-opus-5",
            claude_sonnet_5 "claude-sonnet-5",
            claude_opus_4_8 "claude-opus-4.8",
            gpt_5_6_sol "gpt-5.6-sol",
            gpt_5_6_terra "gpt-5.6-terra",
            gpt_5_6_luna "gpt-5.6-luna",
            gpt_5_5 "gpt-5.5",
        );
    };
    (@expand $runner:path, $($name:ident $model:literal,)+) => {
        $(
            #[tokio::test]
            #[ignore = "live Venice call; run with `mise run test:e2e`"]
            async fn $name() -> crate::support::TestResult {
                $runner($model).await
            }
        )+
    };
}

/// Expands one live test per model family.
///
/// One representative per family keeps the model-behavior-heavy tests — tool
/// choice modes, parallel calls, caching — off the full roster, where they
/// would multiply cost without multiplying signal: within a family the
/// upstream behavior is the same.
macro_rules! family_tests {
    ($runner:path) => {
        crate::venice::model_tests!(@expand $runner,
            kimi_k3 "kimi-k3",
            grok_4_6 "grok-4.6",
            glm_5_3 "glm-5.3",
            deepseek_v4_flash "deepseek-v4-flash",
            qwen_3_8_max "qwen3.8-max",
            // Fable 5.1 sits beside Fable 5: tool selection and parallel-call
            // batching are the behaviors that changed between the two.
            claude_fable_5_1 "claude-fable-5.1",
            claude_fable_5 "claude-fable-5",
            gpt_5_6_terra "gpt-5.6-terra",
        );
    };
}

pub(crate) use family_tests;
pub(crate) use model_tests;

/// The twin proxy's base URL for record and replay runs.
fn twin_url() -> String {
    env::var("LITHOS_E2E_TWIN_URL").unwrap_or_else(|_| "http://127.0.0.1:3921".to_owned())
}

/// The Venice E2E catalog.
///
/// Under record and replay, an overlay points the provider at the twin the
/// `test:e2e` task started; everything else about the rows stays identical,
/// so the wire bodies match between backends.
pub(crate) fn catalog() -> Catalog {
    let base = Catalog::builder()
        .toml_layer("venice-e2e", CATALOG_TOML)
        .expect("the Venice E2E catalog should parse");
    let builder = if support::Backend::from_env() == support::Backend::Live {
        base
    } else {
        base.toml_layer(
            "e2e-twin",
            &format!("[providers.venice]\nbase_url = \"{}\"\n", twin_url()),
        )
        .expect("the twin overlay should parse")
    };
    builder
        .build()
        .expect("the Venice E2E catalog should validate")
}

/// The catalog's capability claims for one roster model.
pub(crate) fn capabilities(model: &str) -> ModelCapabilities {
    catalog()
        .model(PROVIDER, model)
        .expect("the roster model should be in the Venice E2E catalog")
        .capabilities()
}

/// A client for this run's backend, or `None` when a live run has no key.
///
/// Live runs authenticate with `VENICE_API_KEY`. Record and replay runs go
/// through the twin instead, authenticating with the test's namespace as a
/// fake bearer token — the twin holds the real key during recording and
/// needs none during replay.
///
/// The retry middleware is on so a transient rate limit or server error does
/// not fail a nightly cell; a persistent failure still surfaces after the
/// attempts are spent. Nextest gives every test its own process, so the cap
/// on concurrent live requests is the runner's `--test-threads`, not
/// anything configured here.
pub(crate) fn live_client() -> Option<Client> {
    match support::Backend::from_env() {
        support::Backend::Live => {
            let key = env::var(KEY_VARIABLE).ok().filter(|key| !key.is_empty())?;
            Some(client_with_key(&key))
        }
        support::Backend::Record | support::Backend::Replay => {
            Some(client_with_key(&support::test_namespace()))
        }
    }
}

/// A client authenticating with `key`, valid or not.
///
/// The negative suite uses this directly to observe how Venice answers a bad
/// credential.
pub(crate) fn client_with_key(key: &str) -> Client {
    let build = Client::builder()
        .catalog(catalog())
        .credentials(
            StaticCredentials::new().with(PROVIDER, Credentials::bearer(SecretValue::new(key))),
        )
        .middleware(RetryMiddleware::new(
            RetryPolicy::exponential().max_attempts(3).jitter(true),
        ))
        .build()
        .expect("the Venice E2E client should build");
    assert!(
        build.issues.is_empty(),
        "the Venice E2E client reported provider build issues: {:?}",
        build.issues
    );
    build.client
}

/// The request selector for one roster model.
pub(crate) fn selector(model: &str) -> String {
    format!("{PROVIDER}/{model}")
}

/// A request builder with the suite's defaults for one roster model.
///
/// The output allowance is generous because most roster models reason by
/// default and reasoning spends from the same allowance; a test that wants
/// truncation lowers it explicitly. The timeout bounds a wedged provider.
pub(crate) fn request(model: &str) -> RequestBuilder {
    Request::builder()
        .model(selector(model))
        .max_output_tokens(8192)
        .timeout(support::LIVE_TIMEOUT)
}
