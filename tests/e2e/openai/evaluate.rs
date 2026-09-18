//! V2 — evaluation with a generation model as the judge.
//!
//! OpenAI has no native evaluation model, so `evaluate` on a roster row runs
//! the judge path: one structured-output completion through the Responses
//! codec whose JSON object answers every question. One cell on the cheap end
//! of the 5.6 family is enough; within a family the upstream behavior is the
//! same, and the request is small. It records through the OpenAI twin like
//! the rest of the suite.

use crate::openai;
use crate::support::{self, TestResult};

/// The judge row. Its catalog row claims `json_schema`, which is what a
/// judge needs.
const MODEL: &str = "gpt-5.6-luna";

#[tokio::test]
#[ignore = "live OpenAI call; run with `mise run test:e2e`"]
async fn gpt_5_6_luna_judges_the_mixed_evaluation() -> TestResult {
    let Some(client) = openai::live_client() else {
        return support::skip("OPENAI_API_KEY is unset");
    };
    let verdict = client
        .evaluate(support::mixed_evaluation(&openai::selector(MODEL)))
        .await?;
    support::assert_point_estimates(&verdict)
}
