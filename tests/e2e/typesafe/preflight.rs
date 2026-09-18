//! T0 — preflight.
//!
//! Everything here is free: the local tests are not ignored and touch no
//! network.

use lithos_llm::Request;
use lithos_llm::types::ErrorKind;

use crate::support::{self, TestResult};
use crate::typesafe;

/// Every TypeSafe row is an evaluation row, so a completion is refused
/// before it reaches the adapter, with the operation family named.
#[tokio::test]
async fn completion_is_rejected_locally_on_jev() -> TestResult {
    let client = typesafe::client_with_key("preflight-key");
    let request = Request::builder()
        .model(typesafe::selector(typesafe::MODEL))
        .user("Hello")
        .build()?;
    let error = client
        .complete(request)
        .await
        .expect_err("jev claims no text generation, so the client must refuse");
    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    assert!(
        error.message().contains("no generation codec"),
        "{}",
        error.message()
    );
    Ok(())
}

/// Every roster row resolves an evaluation route on the `systemone` codec.
#[test]
fn every_roster_row_resolves_an_evaluation_route() -> TestResult {
    let client = typesafe::client_with_key("preflight-key");
    for model in ["jev-latest", "jev-preview", "jev-1.13.0"] {
        let evaluation = support::mixed_evaluation(&typesafe::selector(model));
        let route = client.resolve_evaluation_route(&evaluation)?;
        assert_eq!(route.provider().id().as_str(), typesafe::PROVIDER);
        assert_eq!(route.api_model(), model);
    }
    Ok(())
}
