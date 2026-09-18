//! Wire parity for the `vercel-evaluation` adapter: `Client::evaluate` on a
//! row that speaks the Vercel AI Gateway's evaluation protocol.
//!
//! The success body is a real 200 the gateway returned on 2026-09-17 for the
//! interface plan's three questions, and the 400 is the body it returns when
//! a language model id is sent to the evaluation path. Both are pinned here
//! so the codec is tested against what the gateway sends rather than what a
//! reader guesses it sends.

use httpmock::{Method, MockServer};
use lithos_llm::catalog::{adapter_ids, codec_ids};
use lithos_llm::types::ErrorKind;
use lithos_llm::{Evaluation, Verdict};
use serde_json::{Value, json};

use crate::support::{self, WireCapture, WireProvider};

const PROVIDER: &str = "vercel";
const MODEL: &str = "jev";
const API_MODEL: &str = "typesafe-ai/jev";
const PATH: &str = "/v4/ai/evaluation-model";

/// The row claims the three question kinds and nothing else.
const EVALUATION_CAPABILITIES: &str =
    "{ evaluation = { choice = true, score = true, boolean = true } }";

/// A real 200 body from `POST /v4/ai/evaluation-model`, 2026-09-17.
const LIVE_BODY: &str = r#"{"answers":{"department":{"type":"choice","choice":"billing","probabilities":{"other":0,"technical":0,"billing":1}},"severity":{"type":"score","score":1.05,"probabilities":{"0":0.13,"1":0.69,"2":0.18}},"requests_refund":{"type":"boolean","probability":0.99}},"rounding":{"probabilityDecimals":2,"scoreDecimals":2},"usage":{"inputTokens":389,"outputTokens":70},"warnings":[],"providerMetadata":{"typesafe":{"confidence":{"department":1,"severity":0.54}},"gateway":{"routing":{"originalModelId":"typesafe-ai/jev","resolvedProvider":"typesafe-ai","fallbacksAvailable":[],"planningReasoning":"System credentials planned for: typesafe-ai. Total execution order: typesafe-ai(system)","canonicalSlug":"typesafe-ai/jev","finalProvider":"typesafe-ai","modelAttemptCount":1,"modelAttempts":[{"canonicalSlug":"typesafe-ai/jev","success":true,"providerAttemptCount":1,"providerAttempts":[{"provider":"typesafe-ai","credentialType":"system","success":true,"startTime":1789687661182,"endTime":1789687661468,"statusCode":200}]}],"totalProviderAttemptCount":1},"cost":"0.000016338","marketCost":"0.000016338","surchargeCost":"0","gatewayCost":"0.000016338","inferenceCost":"0.000016338","inputInferenceCost":"0.000016338","outputInferenceCost":"0","generationId":"gen_01M2RV50ENRGT4K2CHJ0WC4N1H"}}}"#;

/// The 400 the gateway returns when a language model id reaches this path.
const MISMATCH_400_BODY: &str = r#"{"error":{"message":"Model 'anthropic/claude-haiku-4.5' is a language model, not an evaluation model. Use the language generation API instead.","type":"invalid_request_error","param":{"error":"Model 'anthropic/claude-haiku-4.5' is a language model, not an evaluation model. Use the language generation API instead.","type":"invalid_request_error","statusCode":400,"name":"ModelTypeMismatchError","message":"Model 'anthropic/claude-haiku-4.5' is a language model, not an evaluation model. Use the language generation API instead."}},"providerMetadata":{"gateway":{"routing":{"originalModelId":"anthropic/claude-haiku-4.5","resolvedProvider":"claudeaws","fallbacksAvailable":["anthropic","bedrock","vertexAnthropic"],"canonicalSlug":"anthropic/claude-haiku-4.5","modelAttemptCount":1,"modelAttempts":[{"canonicalSlug":"anthropic/claude-haiku-4.5","success":false,"providerAttemptCount":0,"providerAttempts":[]}],"totalProviderAttemptCount":0},"generationId":"gen_01M2RV50YV78ZJ714K3YEJ65TV"}}}"#;

/// The provider-level adapter is this adapter, because the per-row override
/// lands in a later step. The codec id is ignored by the factory.
fn provider() -> WireProvider<'static> {
    WireProvider::new(
        PROVIDER,
        adapter_ids::VERCEL_EVALUATION,
        codec_ids::VERCEL_EVALUATION,
        MODEL,
    )
    .with_api_model(API_MODEL)
    .with_auth("{ type = \"bearer\" }")
    .with_capabilities(EVALUATION_CAPABILITIES)
}

/// The interface plan's three questions, which [`LIVE_BODY`] answers.
fn evaluation() -> Evaluation {
    Evaluation::builder()
        .model(provider().selector())
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
/// returns the captured request and the decoded verdict.
async fn exchange() -> (WireCapture, Verdict) {
    let server = MockServer::start_async().await;
    // The mock server's URL has no `/v1`, so the appended URL shape is the
    // one on the wire here; the codec's unit tests pin the replaced shape.
    let catalog = provider().catalog(&server.base_url());
    let client = support::client_for(catalog, PROVIDER, support::bearer_credentials());
    let (mock, slot) = support::mount_capture(&server, PATH, &body(LIVE_BODY));

    let verdict = client
        .evaluate(evaluation())
        .await
        .expect("the evaluation should succeed");

    mock.assert_async().await;
    (support::captured(&slot), verdict)
}

#[tokio::test]
async fn encodes_the_evaluation_request_and_decodes_the_verdict() {
    let (wire, verdict) = exchange().await;

    assert_eq!(wire.path, PATH);
    crate::json_snapshot!(wire);
    crate::json_snapshot!(verdict);
}

#[tokio::test]
async fn a_model_type_mismatch_400_is_an_invalid_request() {
    let server = MockServer::start_async().await;
    let catalog = provider().catalog(&server.base_url());
    let client = support::client_for(catalog, PROVIDER, support::bearer_credentials());
    let mock = server
        .mock_async(|when, then| {
            when.method(Method::POST).path(PATH);
            then.status(400)
                .header("content-type", "application/json")
                .body(MISMATCH_400_BODY);
        })
        .await;

    let error = match client.evaluate(evaluation()).await {
        Ok(verdict) => panic!("a 400 should fail, got {verdict:?}"),
        Err(error) => error,
    };

    mock.assert_async().await;
    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(error.status(), Some(400));
    assert_eq!(error.provider_code(), Some("invalid_request_error"));
    assert!(
        error.message().contains("not an evaluation model"),
        "{}",
        error.message()
    );
    assert_eq!(
        error.raw_data(),
        Some(&body(MISMATCH_400_BODY)),
        "the gateway's routing metadata stays available on the error"
    );
}

#[tokio::test]
async fn a_completion_on_the_evaluation_row_is_refused_before_dispatch() {
    let server = MockServer::start_async().await;
    let catalog = provider().catalog(&server.base_url());
    let client = support::client_for(catalog, PROVIDER, support::bearer_credentials());
    let mock = server
        .mock_async(|when, then| {
            when.method(Method::POST);
            then.status(200).json_body(json!({}));
        })
        .await;

    let error = match client
        .complete(support::base_request(&provider().selector()))
        .await
    {
        Ok(response) => panic!("a completion should be refused, got {response:?}"),
        Err(error) => error,
    };

    // The row claims no `text`, so the client's capability check refuses the
    // request before the adapter's own refusal could.
    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    mock.assert_calls_async(0).await;
}
