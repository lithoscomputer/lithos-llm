//! V2 — evaluation on the gateway's native evaluation model.
//!
//! `vercel/jev` is the one roster row on the `vercel-evaluation` codec, so
//! these cells are the live exercise of that codec: the
//! `/v4/ai/evaluation-model` path, the three protocol headers, and the
//! metadata lifting that turns the gateway's body into a typed [`Verdict`].
//! They record and replay through the twin like the rest of the suite; the
//! twin proxies the evaluation path as a transcript scenario matched by
//! request hash.
//!
//! The assertions pin what the protocol guarantees — kinds, ranges,
//! distributions that sum to one within the declared rounding, in-band
//! cost — never the specific answer, which the model may change on a
//! re-record.
//!
//! [`Verdict`]: lithos_llm::Verdict

use lithos_llm::Evaluation;
use lithos_llm::types::{CostSource, ErrorKind, EvaluationBuildError, Rounding};
use serde_json::json;

use crate::support::{self, TestResult};
use crate::vercel;

const MODEL: &str = "jev";

/// How far a three-outcome distribution may stray from one: half a unit in
/// the second decimal per outcome (Jev rounds to two decimals), plus the
/// validator's base tolerance.
const THREE_WAY_TOLERANCE: f64 = 0.5 * 3.0 * 0.01 + 1e-6;

/// The most options the protocol carries on one choice question.
const MAX_OPTIONS: usize = 255;

/// The option names `o000`, `o001`, ... for a choice with `count` options.
fn option_names(count: usize) -> Vec<String> {
    (0..count).map(|index| format!("o{index:03}")).collect()
}

/// A one-question evaluation whose choice has `count` unnamed options.
fn choice_with_options(count: usize) -> Result<Evaluation, EvaluationBuildError> {
    Evaluation::builder()
        .model(vercel::selector(MODEL))
        .state("The customer's ticket number is 42.")
        .choice(
            "code",
            "Pick the option whose number matches the ticket.",
            option_names(count)
                .into_iter()
                .map(|name| (name, Option::<&str>::None)),
        )
        .timeout(support::LIVE_TIMEOUT)
        .build()
}

#[tokio::test]
#[ignore = "live Vercel AI Gateway call; run with `mise run test:e2e`"]
async fn a_mixed_evaluation_answers_every_question() -> TestResult {
    let Some(client) = vercel::live_client() else {
        return support::skip("AI_GATEWAY_API_KEY is unset");
    };
    let verdict = client
        .evaluate(support::mixed_evaluation(&vercel::selector(MODEL)))
        .await?;
    support::assert_mixed_answers(&verdict)?;

    let department = verdict.choice("department")?;
    let probabilities = department
        .probabilities
        .as_ref()
        .ok_or("the choice carries no distribution")?;
    let options: Vec<&str> = probabilities.keys().map(String::as_str).collect();
    assert_eq!(
        options,
        support::MIXED_OPTIONS,
        "the choice distribution is not in option order"
    );
    let sum: f64 = probabilities.values().sum();
    assert!(
        (sum - 1.0).abs() <= THREE_WAY_TOLERANCE,
        "the choice distribution sums to {sum}"
    );
    assert!(
        department.confidence.is_some(),
        "the choice carries no confidence"
    );

    let severity = verdict.score("severity")?;
    let distribution = severity
        .probabilities
        .as_ref()
        .ok_or("the score carries no distribution")?;
    assert_eq!(
        distribution.len(),
        3,
        "the score distribution has {} entries, not one per level",
        distribution.len()
    );
    let sum: f64 = distribution.iter().sum();
    assert!(
        (sum - 1.0).abs() <= THREE_WAY_TOLERANCE,
        "the score distribution sums to {sum}"
    );
    assert!(
        severity.confidence.is_some(),
        "the score carries no confidence"
    );

    assert_eq!(
        verdict.rounding,
        Some(Rounding {
            probability_decimals: Some(2),
            score_decimals:       Some(2),
        }),
        "the verdict does not declare two-decimal rounding"
    );
    // The E2E catalog prices nothing, so a missing cost means the in-band
    // `gateway.cost` extraction broke.
    let cost = verdict.cost.ok_or("the verdict carries no cost")?;
    assert_eq!(cost.source, CostSource::Provider);
    assert!(cost.usd_micros > 0, "the provider-reported cost is zero");
    assert!(verdict.id.is_some(), "the verdict carries no generation id");
    Ok(())
}

#[tokio::test]
#[ignore = "live Vercel AI Gateway call; run with `mise run test:e2e`"]
async fn a_structured_state_is_accepted() -> TestResult {
    let Some(client) = vercel::live_client() else {
        return support::skip("AI_GATEWAY_API_KEY is unset");
    };
    let evaluation = Evaluation::builder()
        .model(vercel::selector(MODEL))
        .state(json!({
            "ticket": 4821,
            "channel": "email",
            "subject": "Duplicate charge on my invoice",
            "body": "My card was charged twice for the same order. Please refund one.",
        }))
        .choice("department", "Which team should handle this ticket?", [
            ("billing", Some("Charges and refunds")),
            ("technical", Some("Bugs and outages")),
            ("other", None),
        ])
        .timeout(support::LIVE_TIMEOUT)
        .build()?;
    let verdict = client.evaluate(evaluation).await?;
    let department = verdict.choice("department")?;
    assert!(
        support::MIXED_OPTIONS.contains(&department.choice.as_str()),
        "the choice `{}` names no option",
        department.choice
    );
    Ok(())
}

/// Two hundred questions in one call: the provider answers them all, so a
/// caller can batch a whole rubric into one evaluation.
#[tokio::test]
#[ignore = "live Vercel AI Gateway call; run with `mise run test:e2e`"]
async fn two_hundred_questions_are_answered() -> TestResult {
    const COUNT: usize = 200;
    let Some(client) = vercel::live_client() else {
        return support::skip("AI_GATEWAY_API_KEY is unset");
    };
    let ids: Vec<String> = (0..COUNT).map(|index| format!("q{index:03}")).collect();
    let mut builder = Evaluation::builder()
        .model(vercel::selector(MODEL))
        .state("The parcel arrived a day early, sealed, with every item on the packing list.")
        .timeout(support::LIVE_TIMEOUT);
    for (index, id) in ids.iter().enumerate() {
        builder = builder.boolean(
            id.as_str(),
            format!("Reading {index}: does the message report a satisfactory delivery?"),
        );
    }
    let verdict = client.evaluate(builder.build()?).await?;
    assert_eq!(
        verdict.answers.len(),
        COUNT,
        "the verdict answers {} questions, not {COUNT}",
        verdict.answers.len()
    );
    for id in &ids {
        let probability = verdict.boolean(id)?.probability;
        assert!(
            (0.0..=1.0).contains(&probability),
            "question `{id}` answered {probability}, out of range"
        );
    }
    Ok(())
}

/// The protocol's option ceiling is enforced by the codec, not the wire: a
/// choice at the ceiling is dispatched and answered, and one past it is
/// refused before any request is sent, so the second half costs nothing and
/// needs no recording.
#[tokio::test]
#[ignore = "live Vercel AI Gateway call; run with `mise run test:e2e`"]
async fn a_255_option_choice_is_accepted_and_256_is_refused_before_dispatch() -> TestResult {
    let Some(client) = vercel::live_client() else {
        return support::skip("AI_GATEWAY_API_KEY is unset");
    };
    let verdict = client.evaluate(choice_with_options(MAX_OPTIONS)?).await?;
    let code = verdict.choice("code")?;
    assert!(
        option_names(MAX_OPTIONS).contains(&code.choice),
        "the choice `{}` names no option",
        code.choice
    );
    if let Some(probabilities) = &code.probabilities {
        assert_eq!(
            probabilities.len(),
            MAX_OPTIONS,
            "the distribution has {} entries, not one per option",
            probabilities.len()
        );
    }

    let error = client
        .evaluate(choice_with_options(MAX_OPTIONS + 1)?)
        .await
        .expect_err("256 options exceed the protocol's maximum");
    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(error.provider_code(), Some("invalid_evaluation"));
    Ok(())
}
