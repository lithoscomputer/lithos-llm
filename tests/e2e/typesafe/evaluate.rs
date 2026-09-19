//! T1 — evaluation on TypeSafe's own API.
//!
//! The assertions pin what the protocol guarantees — kinds, ranges,
//! distributions that sum to one within the declared rounding, confidence on
//! the choice and score, the versioned model that answered, the request id
//! from the response header — never the specific answer, which the model may
//! change on a re-record.

use lithos_llm::Evaluation;
use lithos_llm::types::Rounding;

use crate::support::{self, TestResult};
use crate::typesafe;

/// How far a three-outcome distribution may stray from one: half a unit in
/// the second decimal per outcome (Jev answers to two decimals), plus the
/// validator's base tolerance.
const THREE_WAY_TOLERANCE: f64 = 0.5 * 3.0 * 0.01 + 1e-6;

#[tokio::test]
#[ignore = "live TypeSafe AI call; run with `mise run test:e2e`"]
async fn a_mixed_evaluation_answers_every_question() -> TestResult {
    let Some(client) = typesafe::live_client() else {
        return support::skip("TYPESAFE_API_KEY is unset");
    };
    let verdict = client
        .evaluate(support::mixed_evaluation(&typesafe::selector(
            typesafe::MODEL,
        )))
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
    let served_by = verdict
        .served_by
        .as_deref()
        .ok_or("the verdict names no serving model")?;
    assert!(
        served_by.starts_with("jev-"),
        "`{served_by}` is not a Jev version"
    );
    assert!(
        verdict.id.is_some(),
        "the verdict carries no request id from the response header"
    );
    // The E2E catalog prices nothing and TypeSafe reports no in-band cost,
    // so a cost here would mean the API started reporting one.
    assert_eq!(verdict.cost, None, "TypeSafe reports no in-band cost");
    Ok(())
}

/// Two hundred questions in one call: the provider answers them all, so a
/// caller can batch a whole rubric into one evaluation.
#[tokio::test]
#[ignore = "live TypeSafe AI call; run with `mise run test:e2e`"]
async fn two_hundred_questions_are_answered() -> TestResult {
    const COUNT: usize = 200;
    let Some(client) = typesafe::live_client() else {
        return support::skip("TYPESAFE_API_KEY is unset");
    };
    let ids: Vec<String> = (0..COUNT).map(|index| format!("q{index:03}")).collect();
    let mut builder = Evaluation::builder()
        .model(typesafe::selector(typesafe::MODEL))
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
