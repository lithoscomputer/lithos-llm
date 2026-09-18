//! Wire parity for the `systemone` codec: `Client::evaluate` on a row that
//! speaks TypeSafe's System One protocol, in both dialects.
//!
//! The success body is a real 200 TypeSafe's API returned on 2026-09-18 for
//! the interface plan's three questions, and the 401 is the body it returns
//! for a rejected key. The OpenRouter body is hand-written from the
//! documented Decisions schema: no decisions model was listed when this was
//! written, so there is no live body to pin.

use httpmock::{Method, MockServer};
use lithos_llm::catalog::{Catalog, codec_ids};
use lithos_llm::types::{CostSource, ErrorKind};
use lithos_llm::{Evaluation, Verdict};
use serde_json::{Value, json};

use crate::support::{self, WireCapture, WireProvider};

const TYPESAFE: &str = "typesafe";
const OPENROUTER: &str = "openrouter";
const MODEL: &str = "jev-latest";
const TYPESAFE_PATH: &str = "/v1/systemone";
const OPENROUTER_PATH: &str = "/alpha/decisions";

/// The header TypeSafe names each verdict by.
const REQUEST_ID_HEADER: &str = "x-typesafe-request-id";
const REQUEST_ID: &str = "req_01K5J6R0X3T8V2W7Y9Z1A4B6C8";

/// The row claims the three question kinds and nothing else.
const EVALUATION_CAPABILITIES: &str =
    "{ evaluation = { choice = true, score = true, boolean = true } }";

/// A real 200 body from `POST /v1/systemone`, 2026-09-18.
const LIVE_BODY: &str = r#"{"model":"jev-1.13.0","answers":{"department":{"type":"choice","choice":"billing","confidence":1.0,"probabilities":{"billing":1.0,"other":0.0,"technical":0.0}},"severity":{"type":"score","score":1.08,"confidence":0.47,"legend":{"0":"Cosmetic","1":"Workaround exists","2":"Blocking; no workaround"},"probabilities":{"0":0.14,"1":0.64,"2":0.22}},"requests_refund":{"type":"noul","noul":0.98}},"usage":{"input_tokens":389,"output_tokens":70}}"#;

/// The 401 body TypeSafe returns for a rejected key, 2026-09-18.
const UNAUTHORIZED_BODY: &str = r#"{"detail":{"error_type":"authentication_error","message":"Cannot authenticate with the server. Please check your API key and try again."}}"#;

/// An OpenRouter Decisions body, hand-written from the documented schema:
/// the TypeSafe answers plus `id`, `provider`, and `usage.cost`.
const OPENROUTER_BODY: &str = r#"{"id":"dec_01K5J6R0X3T8V2W7Y9Z1A4B6C8","provider":"TypeSafe","model":"jev-1.13.0","answers":{"department":{"type":"choice","choice":"billing","confidence":1.0,"probabilities":{"billing":1.0,"other":0.0,"technical":0.0}},"severity":{"type":"score","score":1.08,"confidence":0.47,"legend":{"0":"Cosmetic","1":"Workaround exists","2":"Blocking; no workaround"},"probabilities":{"0":0.14,"1":0.64,"2":0.22}},"requests_refund":{"type":"noul","noul":0.98}},"usage":{"input_tokens":389,"output_tokens":70,"cost":0.000016338}}"#;

fn typesafe() -> WireProvider<'static> {
    WireProvider::new(TYPESAFE, codec_ids::SYSTEMONE, MODEL)
        .with_auth("{ type = \"bearer\" }")
        .with_capabilities(EVALUATION_CAPABILITIES)
}

fn openrouter() -> WireProvider<'static> {
    WireProvider::new(OPENROUTER, codec_ids::SYSTEMONE, MODEL)
        .with_auth("{ type = \"bearer\" }")
        .with_capabilities(EVALUATION_CAPABILITIES)
}

/// The OpenRouter provider's catalog: the same row with the codec's dialect
/// switched through `codec_options`.
fn openrouter_catalog(base_url: &str) -> Catalog {
    let toml = format!(
        "{}\n[providers.\"{OPENROUTER}\".codec_options]\n{} = {{ dialect = \"openrouter\" }}\n",
        openrouter().toml(base_url),
        codec_ids::SYSTEMONE
    );
    support::catalog_from_toml("wire", &toml)
}

/// The interface plan's three questions, which [`LIVE_BODY`] answers.
fn evaluation(provider: &WireProvider<'_>) -> Evaluation {
    Evaluation::builder()
        .model(provider.selector())
        .state("I was charged twice. Please refund the duplicate.")
        .choice("department", "Which team should handle this?", [
            ("billing", Some("Charges and refunds")),
            ("technical", Some("Bugs and outages")),
            ("other", None),
        ])
        .score("severity", "How severe is the issue?", [
            "Cosmetic",
            "Workaround exists",
            "Blocking; no workaround",
        ])
        .boolean_with_criteria(
            "requests_refund",
            "Is the customer requesting money back?",
            "The customer asks for money back",
            "Anything else",
        )
        .build()
        .expect("the fixture evaluation should build")
}

fn body(text: &str) -> Value {
    serde_json::from_str(text).expect("the fixture body is JSON")
}

/// Runs the evaluation against a mock that answers with [`LIVE_BODY`] and
/// the request id header, and returns the captured request and the verdict.
async fn typesafe_exchange() -> (WireCapture, Verdict) {
    let server = MockServer::start_async().await;
    // The mock server's URL has no `/v1`, so the codec adds the whole
    // `/v1/systemone`; the codec's unit tests pin the `/v1` base shape.
    let catalog = typesafe().catalog(&server.base_url());
    let client = support::client_for(catalog, TYPESAFE, support::bearer_credentials());
    let (mock, slot) =
        support::mount_capture_with_headers(&server, TYPESAFE_PATH, &body(LIVE_BODY), &[(
            REQUEST_ID_HEADER,
            REQUEST_ID,
        )]);

    let verdict = client
        .evaluate(evaluation(&typesafe()))
        .await
        .expect("the evaluation should succeed");

    mock.assert_async().await;
    (support::captured(&slot), verdict)
}

#[tokio::test]
async fn encodes_the_typesafe_request_and_decodes_the_verdict() {
    let (wire, verdict) = typesafe_exchange().await;

    assert_eq!(wire.path, TYPESAFE_PATH);
    assert_eq!(
        verdict.id.as_deref(),
        Some(REQUEST_ID),
        "the header request id becomes the verdict id"
    );
    assert_eq!(verdict.served_by.as_deref(), Some("jev-1.13.0"));
    crate::json_snapshot!(wire);
    crate::json_snapshot!(verdict);
}

#[tokio::test]
async fn encodes_the_openrouter_request_and_lifts_its_extras() {
    let server = MockServer::start_async().await;
    let catalog = openrouter_catalog(&server.base_url());
    let client = support::client_for(catalog, OPENROUTER, support::bearer_credentials());
    let (mock, slot) = support::mount_capture(&server, OPENROUTER_PATH, &body(OPENROUTER_BODY));
    let evaluation = evaluation(&openrouter())
        .into_builder()
        .provider_option(OPENROUTER, "session_id", json!("sess_42"))
        .provider_option(OPENROUTER, "provider", json!({ "order": ["typesafe"] }))
        .build()
        .expect("the evaluation with extras should build");

    let verdict = client
        .evaluate(evaluation)
        .await
        .expect("the evaluation should succeed");

    mock.assert_async().await;
    let wire = support::captured(&slot);
    assert_eq!(wire.path, OPENROUTER_PATH);
    assert_eq!(
        verdict.id.as_deref(),
        Some("dec_01K5J6R0X3T8V2W7Y9Z1A4B6C8")
    );
    assert_eq!(
        verdict.cost.map(|cost| cost.source),
        Some(CostSource::Provider),
        "the in-band cost wins over the (absent) catalog estimate"
    );
    crate::json_snapshot!(wire);
    crate::json_snapshot!(verdict);
}

#[tokio::test]
async fn a_rejected_key_401_is_an_authentication_error() {
    let server = MockServer::start_async().await;
    let catalog = typesafe().catalog(&server.base_url());
    let client = support::client_for(catalog, TYPESAFE, support::bearer_credentials());
    let mock = server
        .mock_async(|when, then| {
            when.method(Method::POST).path(TYPESAFE_PATH);
            then.status(401)
                .header("content-type", "application/json")
                .body(UNAUTHORIZED_BODY);
        })
        .await;

    let error = match client.evaluate(evaluation(&typesafe())).await {
        Ok(verdict) => panic!("a 401 should fail, got {verdict:?}"),
        Err(error) => error,
    };

    mock.assert_async().await;
    assert_eq!(error.kind(), ErrorKind::Authentication);
    assert_eq!(error.status(), Some(401));
    assert_eq!(error.provider_code(), Some("authentication_error"));
    assert!(
        error.message().contains("Cannot authenticate"),
        "{}",
        error.message()
    );
    assert!(!error.is_retryable());
}
