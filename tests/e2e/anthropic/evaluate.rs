//! V2 — evaluation with a generation model as the judge.
//!
//! Anthropic has no native evaluation model, so `evaluate` on a roster row
//! runs the judge path: one structured-output completion through the
//! Messages codec whose JSON object answers every question. One cell on
//! Sonnet 5 is enough; within the family the upstream behavior is the same,
//! and the request is small. Like every network cell in this suite it is
//! live-only, because the twin does not proxy `/v1/messages`.

use crate::anthropic;
use crate::support::{self, TestResult};

/// The judge row. Its catalog row claims `json_schema`, which is what a
/// judge needs.
const MODEL: &str = "claude-sonnet-5";

#[tokio::test]
#[ignore = "live Anthropic call; run with mise run test:e2e:live"]
async fn claude_sonnet_5_judges_the_mixed_evaluation() -> TestResult {
    let Some(client) = anthropic::live_client() else {
        return support::skip("ANTHROPIC_API_KEY is unset");
    };
    let verdict = client
        .evaluate(support::mixed_evaluation(&anthropic::selector(MODEL)))
        .await?;
    support::assert_point_estimates(&verdict)
}
