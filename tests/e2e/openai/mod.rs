//! The OpenAI live suite.
//!
//! OpenAI runs through the native `openai` adapter and the `openai-responses`
//! codec, so this suite is the live exercise of the Responses protocol: the
//! stateless `store: false` + encrypted-reasoning replay shape, the
//! `/v1/responses/input_tokens` token count, service tiers, and the
//! `cache_write_tokens` usage counter the GPT-5.6 family bills. Cost is a
//! catalog estimate (`CostSource::Catalog`); OpenAI reports no in-band cost.
//!
//! The record/replay cells cover the API-key platform path. The
//! ChatGPT-subscription path through the Codex deployment speaks the same
//! protocol with a different envelope and roster (see the built-in
//! catalog's notes); its live-only battery is the [`codex`] submodule,
//! gated on `OPENAI_CODEX_TOKEN` and `CHATGPT_ACCOUNT_ID`.
//!
//! The roster lives in `openai_catalog.toml`; the suite iterates its recorded
//! rows through the [`model_tests`] and [`family_tests`] macros.
//! Capability-gated runners consult the catalog row and skip models whose row
//! does not claim the capability. GPT-6 Astra is present in the catalog from
//! the published docs but stays out of these macros until API access permits a
//! live run and recording.

mod caching;
mod codex;
mod documents;
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
use lithos_llm::credentials::{Credentials, SecretValue, StaticCredentials};
use lithos_llm::middleware::{RetryMiddleware, RetryPolicy};
use lithos_llm::types::RequestBuilder;
use lithos_llm::{Client, Request};

use crate::support;

pub(crate) const PROVIDER: &str = "openai";

/// The environment variable holding the OpenAI API key.
pub(crate) const KEY_VARIABLE: &str = "OPENAI_API_KEY";

const CATALOG_TOML: &str = include_str!("../openai_catalog.toml");

/// Expands one live test per recorded roster model.
///
/// `$runner` is an `async fn(&'static str) -> TestResult` taking the catalog
/// model id. Every generated test is ignored, so the roster only runs under
/// `mise run test:e2e`.
macro_rules! model_tests {
    ($runner:path) => {
        model_tests!(@expand $runner,
            gpt_5_6_sol "gpt-5.6-sol",
            gpt_5_6_terra "gpt-5.6-terra",
            gpt_5_6_luna "gpt-5.6-luna",
            gpt_5_4 "gpt-5.4",
            gpt_5_5 "gpt-5.5",
            gpt_5_5_pro "gpt-5.5-pro",
            gpt_5_4_pro "gpt-5.4-pro",
            gpt_5_4_mini "gpt-5.4-mini",
        );
    };
    (@expand $runner:path, $($name:ident $model:literal,)+) => {
        $(
            #[tokio::test]
            #[ignore = "live OpenAI call; run with `mise run test:e2e`"]
            async fn $name() -> crate::support::TestResult {
                $runner($model).await
            }
        )+
    };
}

/// Expands one live test per model family.
///
/// The cheap end of each recorded generation represents it: luna for the 5.6
/// rows and gpt-5.4-mini for the 5.4/5.5 rows — within a generation the
/// upstream behavior is the same, so more cells would multiply cost without
/// multiplying signal. The pro rows are deliberately absent: they bill 40x
/// the mini rates and a single request can run for minutes, so they stay on
/// the roster-wide cells only. Astra joins this macro after its first live
/// recording.
macro_rules! family_tests {
    ($runner:path) => {
        crate::openai::model_tests!(@expand $runner,
            gpt_5_6_luna "gpt-5.6-luna",
            gpt_5_4_mini "gpt-5.4-mini",
        );
    };
}

pub(crate) use family_tests;
pub(crate) use model_tests;

/// The twin proxy's base URL for record and replay runs.
fn twin_url() -> String {
    env::var("LITHOS_OPENAI_E2E_TWIN_URL").unwrap_or_else(|_| "http://127.0.0.1:3925".to_owned())
}

/// The OpenAI E2E catalog.
///
/// Under record and replay, an overlay points the provider at the twin the
/// `test:e2e` task started; everything else about the rows stays identical,
/// so the wire bodies match between backends.
pub(crate) fn catalog() -> Catalog {
    let base = Catalog::builder()
        .toml_layer("openai-e2e", CATALOG_TOML)
        .expect("the OpenAI E2E catalog should parse");
    let builder = if support::Backend::from_env() == support::Backend::Live {
        base
    } else {
        base.toml_layer(
            "e2e-twin",
            &format!("[providers.openai]\nbase_url = \"{}\"\n", twin_url()),
        )
        .expect("the twin overlay should parse")
    };
    builder
        .build()
        .expect("the OpenAI E2E catalog should validate")
}

/// The catalog's capability claims for one roster model.
pub(crate) fn capabilities(model: &str) -> ModelCapabilities {
    catalog()
        .model(PROVIDER, model)
        .expect("the roster model should be in the OpenAI E2E catalog")
        .capabilities()
}

/// A client for this run's backend, or `None` when a live run has no key.
///
/// Live runs authenticate with `OPENAI_API_KEY`. Record and replay runs go
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
/// The negative suite uses this directly to observe how OpenAI answers a bad
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
        .expect("the OpenAI E2E client should build");
    assert!(
        build.issues.is_empty(),
        "the OpenAI E2E client reported provider build issues: {:?}",
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
/// The output allowance is generous because every roster model reasons by
/// default at medium effort and reasoning spends from the same allowance; a
/// test that wants truncation lowers it explicitly. The timeout bounds a
/// wedged provider — the pro rows in particular can run for minutes.
pub(crate) fn request(model: &str) -> RequestBuilder {
    Request::builder()
        .model(selector(model))
        .max_output_tokens(8192)
        .timeout(support::LIVE_TIMEOUT)
}
