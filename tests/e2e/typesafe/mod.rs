//! The TypeSafe AI live suite.
//!
//! TypeSafe serves one model family, Jev, through its System One protocol,
//! so this suite is the live exercise of the `systemone` codec's TypeSafe
//! dialect: the `/v1/systemone` path, the `noul` question type, the inline
//! confidence and legend, the versioned `model` that becomes `served_by`,
//! and the `x-typesafe-request-id` header that becomes the verdict id. The
//! cells record and replay through the twin's `/v1/systemone` passthrough
//! (twins PR #12), which carries that header in the transcript.
//!
//! The roster lives in `typesafe_catalog.toml`. Every row is an evaluation
//! row, so there are no generation cells; the [`preflight`] module pins that
//! a completion is refused before dispatch.

mod evaluate;
mod negative;
mod preflight;

use std::env;

use lithos_llm::Client;
use lithos_llm::catalog::Catalog;
use lithos_llm::credentials::{Credentials, SecretValue, StaticCredentials};
use lithos_llm::middleware::{RetryMiddleware, RetryPolicy};

use crate::support;

pub(crate) const PROVIDER: &str = "typesafe";

/// The environment variable holding the TypeSafe API key.
pub(crate) const KEY_VARIABLE: &str = "TYPESAFE_API_KEY";

/// The default model, which the cells evaluate against.
pub(crate) const MODEL: &str = "jev-latest";

const CATALOG_TOML: &str = include_str!("../typesafe_catalog.toml");

/// The twin proxy's base URL for record and replay runs.
fn twin_url() -> String {
    env::var("LITHOS_TYPESAFE_E2E_TWIN_URL").unwrap_or_else(|_| "http://127.0.0.1:3928".to_owned())
}

/// The TypeSafe E2E catalog.
///
/// Under record and replay, an overlay points the provider at the twin the
/// `test:e2e` task started; everything else about the rows stays identical,
/// so the wire bodies match between backends.
pub(crate) fn catalog() -> Catalog {
    let base = Catalog::builder()
        .toml_layer("typesafe-e2e", CATALOG_TOML)
        .expect("the TypeSafe E2E catalog should parse");
    let builder = if support::Backend::from_env() == support::Backend::Live {
        base
    } else {
        base.toml_layer(
            "e2e-twin",
            &format!("[providers.typesafe]\nbase_url = \"{}\"\n", twin_url()),
        )
        .expect("the twin overlay should parse")
    };
    builder
        .build()
        .expect("the TypeSafe E2E catalog should validate")
}

/// A client for this run's backend, or `None` when a live run has no key.
///
/// Live runs authenticate with `TYPESAFE_API_KEY`. Record and replay runs go
/// through the twin instead, authenticating with the test's namespace as a
/// fake bearer token; the twin holds the real key during recording and
/// needs none during replay.
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
/// The negative suite uses this directly to observe how the API answers a
/// bad credential.
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
        .expect("the TypeSafe E2E client should build");
    assert!(
        build.issues.is_empty(),
        "the TypeSafe E2E client reported provider build issues: {:?}",
        build.issues
    );
    build.client
}

/// The request selector for one roster model.
pub(crate) fn selector(model: &str) -> String {
    format!("{PROVIDER}/{model}")
}
