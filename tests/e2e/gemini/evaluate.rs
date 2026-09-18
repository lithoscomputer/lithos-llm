//! V2 — evaluation with a generation model as the judge.
//!
//! Gemini has no native evaluation model, so `evaluate` on a roster row runs
//! the judge path: one structured-output completion through the
//! GenerateContent codec whose JSON object answers every question. One cell
//! on Gemini 3.5 Flash is enough; within the family the upstream behavior is
//! the same, and the request is small. Like every network cell in this suite
//! it is live-only, because the twin does not proxy the GenerateContent
//! paths.

use crate::gemini;
use crate::support::{self, TestResult};

/// The judge row. Its catalog row claims `json_schema`, which is what a
/// judge needs.
const MODEL: &str = "gemini-3.5-flash";

#[tokio::test]
#[ignore = "live Gemini call; run with mise run test:e2e:live"]
async fn gemini_3_5_flash_judges_the_mixed_evaluation() -> TestResult {
    let Some(client) = gemini::live_client() else {
        return support::skip("GEMINI_API_KEY is unset");
    };
    let verdict = client
        .evaluate(support::mixed_evaluation(&gemini::selector(MODEL)))
        .await?;
    support::assert_point_estimates(&verdict)
}
