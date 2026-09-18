//! Wire parity for the judge: `Client::evaluate` on a generation row.
//!
//! A row without a native evaluation adapter answers an [`Evaluation`]
//! through one structured-output completion. These fixtures pin that
//! completion's exact body — the system prompt bytes, the user message's JSON,
//! the internal `q<i>` and `c<j>` keys, and the JSON Schema — because the
//! prompt is part of the crate's contract with its callers: changing it
//! changes answers. They also pin the decoded [`Verdict`], so a change in how
//! the judge's object maps back onto the caller's ids is visible too.
//!
//! The dialect is OpenAI Chat Completions, the one most judges will speak. The
//! prompt and schema are dialect-independent; what this file pins is the
//! judge's half of the exchange, not the codec's.

use std::time::Duration;

use httpmock::MockServer;
use lithos_llm::catalog::codec_ids;
use lithos_llm::{Evaluation, Verdict};
use serde_json::{Value, json};

use crate::support::{self, WireCapture, WireProvider};

const PROVIDER: &str = "compat";
const MODEL: &str = "compat-judge";
const API_MODEL: &str = "vendor/compat-judge-v1";
const PATH: &str = "/v1/chat/completions";

/// A row that judges because it claims JSON Schema output, and nothing else
/// about reasoning.
const JUDGE_CAPABILITIES: &str = "{ text = true, response_format = { json_schema = true } }";

/// A row that reasons by default and claims two effort levels.
const REASONING_CAPABILITIES: &str = "{ text = true, response_format = { json_schema = true }, reasoning = true, reasoning_effort = { low = true, medium = true } }";

const REASONING_BY_DEFAULT: &str =
    "\n[providers.compat.models.compat-judge.metadata.agent]\nreasoning_by_default = true\n";

fn provider() -> WireProvider<'static> {
    WireProvider::new(PROVIDER, codec_ids::OPENAI_CHAT, MODEL)
        .with_api_model(API_MODEL)
        .with_auth("{ type = \"bearer\" }")
        .with_capabilities(JUDGE_CAPABILITIES)
}

/// The interface plan's three questions plus a timeout and one metadata
/// entry, so the fixture pins that the evaluation's settings carry over to
/// the judge request.
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
        .timeout(Duration::from_secs(30))
        .metadata_entry("tenant", "acme")
        .build()
        .expect("the fixture evaluation should build")
}

/// The judge's reply under the internal keys: `department` is `q0`,
/// `requests_refund` is `q1`, `severity` is `q2`, in the ids' sorted order.
fn judge_reply() -> Value {
    json!({
        "id": "chatcmpl-judge",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": API_MODEL,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "{\"q0\":\"c0\",\"q1\":0.9,\"q2\":1.5}" },
            "finish_reason": "stop",
        }],
        "usage": { "prompt_tokens": 250, "completion_tokens": 20, "total_tokens": 270 },
    })
}

/// Runs one evaluation against `catalog_toml` and returns the captured judge
/// request and the decoded verdict.
async fn exchange(catalog_toml: impl Fn(&str) -> String) -> (WireCapture, Verdict) {
    let server = MockServer::start_async().await;
    let catalog = support::catalog_from_toml("wire-judge", &catalog_toml(&server.base_url()));
    let client = support::client_for(catalog, PROVIDER, support::bearer_credentials());
    let (mock, slot) = support::mount_capture(&server, PATH, &judge_reply());

    let verdict = client
        .evaluate(evaluation())
        .await
        .expect("the evaluation should succeed");

    mock.assert_async().await;
    (support::captured(&slot), verdict)
}

#[tokio::test]
async fn encodes_the_judge_request_and_decodes_the_verdict() {
    let (wire, verdict) = exchange(|base_url| provider().toml(base_url)).await;

    // The system prompt, the user JSON, the internal keys, and the schema
    // are all in this body. A diff here is a contract change.
    crate::json_snapshot!(wire);
    crate::json_snapshot!(verdict);
}

#[tokio::test]
async fn a_reasoning_by_default_row_gets_the_lowest_effort_it_claims() {
    let (wire, _) = exchange(|base_url| {
        format!(
            "{}{REASONING_BY_DEFAULT}",
            provider()
                .with_capabilities(REASONING_CAPABILITIES)
                .toml(base_url)
        )
    })
    .await;

    assert_eq!(wire.body["reasoning_effort"], json!("low"));
}

#[tokio::test]
async fn a_row_without_the_flag_asks_for_no_reasoning() {
    let (wire, _) = exchange(|base_url| {
        provider()
            .with_capabilities(REASONING_CAPABILITIES)
            .toml(base_url)
    })
    .await;

    assert!(wire.body.get("reasoning_effort").is_none(), "{}", wire.body);
}
