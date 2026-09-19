//! T3 — negative behavior.
//!
//! TypeSafe's API is FastAPI, so its error bodies are the `detail` shapes
//! `classify.rs` learned from recorded fixtures. This cell verifies the live
//! 401 still lands as [`ErrorKind::Authentication`].

use lithos_llm::types::ErrorKind;

use crate::support::{self, TestResult};
use crate::typesafe;

#[tokio::test]
#[ignore = "live TypeSafe AI call; run with `mise run test:e2e`"]
async fn a_bad_key_classifies_as_authentication() -> TestResult {
    if let Some(skip) = support::live_only("live 401 classification") {
        return skip;
    }
    let client = typesafe::client_with_key("lithos-e2e-invalid-key");
    let error = client
        .evaluate(support::mixed_evaluation(&typesafe::selector(
            typesafe::MODEL,
        )))
        .await
        .expect_err("an invalid key must not evaluate");
    assert_eq!(
        error.kind(),
        ErrorKind::Authentication,
        "the API's live 401 shape classified as {:?}: {}",
        error.kind(),
        error.message()
    );
    assert_eq!(error.provider_code(), Some("authentication_error"));
    Ok(())
}
