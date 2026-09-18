//! The answer checks every verdict passes before the client returns it.
//!
//! Both paths run these: a native adapter's verdict and the judge's decoded
//! answers. The rules mirror the Vercel AI SDK's `validateEvaluationAnswers`
//! so the two libraries accept and reject the same verdicts. Values are never
//! renormalized or patched; a verdict that fails is refused whole.

use std::collections::BTreeSet;

use indexmap::IndexMap;

use super::{Answer, Evaluation, Question, QuestionId, QuestionKind, Rounding, Verdict};
use crate::types::{Error, ErrorKind};

/// Absolute tolerance for sums and means, before rounding allowances.
const TOLERANCE: f64 = 1e-6;

/// The most decimal places a provider may declare it rounds to.
const MAX_DECIMALS: u8 = 15;

/// Checks that `verdict` answers exactly the questions in `evaluation`, each
/// with the matching kind and values in range.
///
/// # Errors
///
/// `ResponseDecode`, never retried, naming the offending question, when an
/// answer is missing or unasked, has the wrong kind, names an unknown option,
/// is out of range, or carries a distribution that is incomplete, does not
/// sum to one within the declared rounding, or disagrees with the answer.
/// Rounding decimals past 15 are refused before any answer is read.
pub(crate) fn validate_verdict(evaluation: &Evaluation, verdict: &Verdict) -> Result<(), Error> {
    let invalid = |message: String| {
        Error::new(ErrorKind::ResponseDecode, message)
            .with_provider(verdict.model.provider().clone())
    };
    let rounding = verdict.rounding.unwrap_or_default();
    let probability_error = rounding_error(rounding.probability_decimals).map_err(&invalid)?;
    let score_error = rounding_error(rounding.score_decimals).map_err(&invalid)?;

    for id in evaluation.questions().keys() {
        if !verdict.answers.contains_key(id) {
            return Err(invalid(format!("question `{id}` has no answer")));
        }
    }
    for id in verdict.answers.keys() {
        if !evaluation.questions().contains_key(id) {
            return Err(invalid(format!("answer `{id}` matches no question")));
        }
    }

    for (id, question) in evaluation.questions() {
        let answer = &verdict.answers[id];
        if answer.kind() != question.kind() {
            return Err(invalid(format!(
                "question `{id}` was answered as {}, not {}",
                kind_name(answer.kind()),
                kind_name(question.kind())
            )));
        }
        match (question, answer) {
            (Question::Choice { options, .. }, Answer::Choice(answer)) => {
                if !options.contains_key(&answer.choice) {
                    return Err(invalid(format!(
                        "question `{id}` selected unknown option `{}`",
                        answer.choice
                    )));
                }
                if let Some(probabilities) = &answer.probabilities {
                    validate_choice_distribution(id, options, probabilities, probability_error)
                        .map_err(&invalid)?;
                    let selected = probabilities[&answer.choice];
                    if probabilities
                        .values()
                        .any(|probability| *probability > selected + TOLERANCE)
                    {
                        return Err(invalid(format!(
                            "question `{id}` did not select the most probable option"
                        )));
                    }
                }
            }
            (Question::Score { levels, .. }, Answer::Score(answer)) => {
                let top = (levels.len() - 1) as f64;
                if !answer.score.is_finite() || answer.score < 0.0 || answer.score > top {
                    return Err(invalid(format!(
                        "question `{id}` score must be in [0, {top}]"
                    )));
                }
                if let Some(probabilities) = &answer.probabilities {
                    validate_distribution(
                        id,
                        probabilities.len() == levels.len(),
                        probabilities.iter().copied(),
                        probability_error,
                    )
                    .map_err(&invalid)?;
                    let mean: f64 = probabilities
                        .iter()
                        .enumerate()
                        .map(|(index, probability)| index as f64 * probability)
                        .sum();
                    let mean_error: f64 = (0..levels.len())
                        .map(|index| index as f64 * probability_error)
                        .sum();
                    if (mean - answer.score).abs() > TOLERANCE + mean_error + score_error {
                        return Err(invalid(format!(
                            "question `{id}` score must equal the probability-weighted mean \
                             within the declared rounding"
                        )));
                    }
                }
            }
            (Question::Boolean { .. }, Answer::Boolean(answer))
                if !is_probability(answer.probability) =>
            {
                return Err(invalid(format!(
                    "question `{id}` must return P(true) as a finite probability in [0, 1]"
                )));
            }
            // A boolean in range, or a pair whose kinds were compared above.
            _ => {}
        }
    }
    Ok(())
}

/// Half a unit in the last declared decimal place, or zero when the provider
/// declared nothing.
fn rounding_error(decimals: Option<u8>) -> Result<f64, String> {
    match decimals {
        None => Ok(0.0),
        Some(decimals) if decimals > MAX_DECIMALS => Err(format!(
            "rounding decimals must be at most {MAX_DECIMALS}, not {decimals}"
        )),
        Some(decimals) => Ok(0.5 * 10f64.powi(-i32::from(decimals))),
    }
}

fn validate_choice_distribution(
    id: &QuestionId,
    options: &IndexMap<String, Option<super::Instructions>>,
    probabilities: &IndexMap<String, f64>,
    probability_error: f64,
) -> Result<(), String> {
    let expected: BTreeSet<&str> = options.keys().map(String::as_str).collect();
    let actual: BTreeSet<&str> = probabilities.keys().map(String::as_str).collect();
    validate_distribution(
        id,
        expected == actual,
        probabilities.values().copied(),
        probability_error,
    )
}

/// Checks that a distribution covers exactly the expected outcomes, is made
/// of finite probabilities, and sums to one within tolerance.
fn validate_distribution(
    id: &QuestionId,
    complete: bool,
    probabilities: impl Iterator<Item = f64> + Clone,
    probability_error: f64,
) -> Result<(), String> {
    if !complete || !probabilities.clone().all(is_probability) {
        return Err(format!(
            "question `{id}` must have a complete distribution of finite probabilities in [0, 1]"
        ));
    }
    let count = probabilities.clone().count() as f64;
    let sum: f64 = probabilities.sum();
    if (sum - 1.0).abs() > TOLERANCE + count * probability_error {
        return Err(format!(
            "question `{id}` probabilities must sum to 1 within the declared rounding"
        ));
    }
    Ok(())
}

fn is_probability(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn kind_name(kind: QuestionKind) -> &'static str {
    match kind {
        QuestionKind::Choice => "a choice",
        QuestionKind::Score => "a score",
        QuestionKind::Boolean => "a boolean",
    }
}

impl Rounding {
    /// Both decimal counts, for tests that build a rounding in one call.
    #[cfg(test)]
    fn decimals(probability: Option<u8>, score: Option<u8>) -> Self {
        Self {
            probability_decimals: probability,
            score_decimals:       score,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::error::Error as StdError;

    use indexmap::IndexMap;

    use super::validate_verdict;
    use crate::catalog::{ModelId, ProviderId};
    use crate::evaluation::{
        Answer, BooleanAnswer, ChoiceAnswer, Evaluation, QuestionId, Rounding, ScoreAnswer, Verdict,
    };
    use crate::types::{ErrorKind, RetryClassification};

    /// The interface plan's three questions: `department` (choice, three
    /// options), `requests_refund` (boolean), `severity` (score, three
    /// levels).
    fn evaluation() -> Result<Evaluation, Box<dyn StdError>> {
        Ok(Evaluation::builder()
            .model("alpha/judge")
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
            .boolean("requests_refund", "Is the customer requesting money back?")
            .build()?)
    }

    fn choice(name: &str, probabilities: Option<&[(&str, f64)]>) -> Answer {
        Answer::Choice(ChoiceAnswer {
            choice:        name.to_owned(),
            probabilities: probabilities.map(|pairs| {
                pairs
                    .iter()
                    .map(|(name, probability)| ((*name).to_owned(), *probability))
                    .collect::<IndexMap<_, _>>()
            }),
            confidence:    None,
        })
    }

    fn score(value: f64, probabilities: Option<&[f64]>) -> Answer {
        Answer::Score(ScoreAnswer {
            score:         value,
            probabilities: probabilities.map(<[f64]>::to_vec),
            confidence:    None,
        })
    }

    fn boolean(probability: f64) -> Answer {
        Answer::Boolean(BooleanAnswer { probability })
    }

    /// A verdict with the given answers under the plan's question ids, in
    /// the order `department`, `requests_refund`, `severity`.
    fn verdict(answers: [Answer; 3]) -> Verdict {
        let [department, requests_refund, severity] = answers;
        let answers: BTreeMap<QuestionId, Answer> = [
            ("department", department),
            ("requests_refund", requests_refund),
            ("severity", severity),
        ]
        .into_iter()
        .map(|(id, answer)| (QuestionId::new(id), answer))
        .collect();
        Verdict::new(ProviderId::new("alpha"), ModelId::new("judge"), answers)
    }

    fn plain_verdict() -> Verdict {
        verdict([choice("billing", None), boolean(0.9), score(1.5, None)])
    }

    fn expect_rejection(verdict: &Verdict, needle: &str) -> Result<(), Box<dyn StdError>> {
        let error = validate_verdict(&evaluation()?, verdict).expect_err("the verdict is invalid");
        assert_eq!(error.kind(), ErrorKind::ResponseDecode);
        assert_eq!(error.retry_classification(), RetryClassification::Never);
        assert_eq!(error.provider().map(ProviderId::as_str), Some("alpha"));
        assert!(
            error.message().contains(needle),
            "expected `{needle}` in `{}`",
            error.message()
        );
        Ok(())
    }

    #[test]
    fn a_plain_verdict_passes() -> Result<(), Box<dyn StdError>> {
        validate_verdict(&evaluation()?, &plain_verdict())?;
        Ok(())
    }

    #[test]
    fn a_missing_answer_fails() -> Result<(), Box<dyn StdError>> {
        let mut verdict = plain_verdict();
        verdict.answers.remove("severity");
        expect_rejection(&verdict, "question `severity` has no answer")
    }

    #[test]
    fn an_extra_answer_fails() -> Result<(), Box<dyn StdError>> {
        let mut verdict = plain_verdict();
        verdict
            .answers
            .insert(QuestionId::new("unasked"), boolean(0.5));
        expect_rejection(&verdict, "answer `unasked` matches no question")
    }

    #[test]
    fn a_kind_mismatch_fails() -> Result<(), Box<dyn StdError>> {
        let verdict = verdict([choice("billing", None), score(1.0, None), score(1.5, None)]);
        expect_rejection(
            &verdict,
            "question `requests_refund` was answered as a score, not a boolean",
        )
    }

    #[test]
    fn an_unknown_choice_fails() -> Result<(), Box<dyn StdError>> {
        let verdict = verdict([choice("legal", None), boolean(0.9), score(1.5, None)]);
        expect_rejection(
            &verdict,
            "question `department` selected unknown option `legal`",
        )
    }

    #[test]
    fn a_two_decimal_distribution_summing_to_0_99_passes() -> Result<(), Box<dyn StdError>> {
        let mut verdict = verdict([
            choice(
                "billing",
                Some(&[("billing", 0.5), ("technical", 0.25), ("other", 0.24)]),
            ),
            boolean(0.9),
            score(1.5, None),
        ]);
        verdict.rounding = Some(Rounding::decimals(Some(2), None));
        validate_verdict(&evaluation()?, &verdict)?;
        Ok(())
    }

    #[test]
    fn a_two_decimal_distribution_summing_to_0_97_fails() -> Result<(), Box<dyn StdError>> {
        let mut verdict = verdict([
            choice(
                "billing",
                Some(&[("billing", 0.5), ("technical", 0.25), ("other", 0.22)]),
            ),
            boolean(0.9),
            score(1.5, None),
        ]);
        verdict.rounding = Some(Rounding::decimals(Some(2), None));
        expect_rejection(&verdict, "probabilities must sum to 1")
    }

    #[test]
    fn an_undeclared_rounding_allows_only_the_base_tolerance() -> Result<(), Box<dyn StdError>> {
        let verdict = verdict([
            choice(
                "billing",
                Some(&[("billing", 0.5), ("technical", 0.25), ("other", 0.24)]),
            ),
            boolean(0.9),
            score(1.5, None),
        ]);
        expect_rejection(&verdict, "probabilities must sum to 1")
    }

    #[test]
    fn a_distribution_over_the_wrong_options_fails() -> Result<(), Box<dyn StdError>> {
        let verdict = verdict([
            choice("billing", Some(&[("billing", 0.6), ("legal", 0.4)])),
            boolean(0.9),
            score(1.5, None),
        ]);
        expect_rejection(&verdict, "complete distribution")
    }

    #[test]
    fn a_distribution_in_another_order_passes() -> Result<(), Box<dyn StdError>> {
        let verdict = verdict([
            choice(
                "billing",
                Some(&[("other", 0.1), ("billing", 0.6), ("technical", 0.3)]),
            ),
            boolean(0.9),
            score(1.5, None),
        ]);
        validate_verdict(&evaluation()?, &verdict)?;
        Ok(())
    }

    #[test]
    fn a_choice_that_is_not_maximal_fails() -> Result<(), Box<dyn StdError>> {
        let verdict = verdict([
            choice(
                "other",
                Some(&[("billing", 0.6), ("technical", 0.3), ("other", 0.1)]),
            ),
            boolean(0.9),
            score(1.5, None),
        ]);
        expect_rejection(&verdict, "did not select the most probable option")
    }

    #[test]
    fn a_tied_choice_passes() -> Result<(), Box<dyn StdError>> {
        let verdict = verdict([
            choice(
                "other",
                Some(&[("billing", 0.45), ("technical", 0.1), ("other", 0.45)]),
            ),
            boolean(0.9),
            score(1.5, None),
        ]);
        validate_verdict(&evaluation()?, &verdict)?;
        Ok(())
    }

    #[test]
    fn the_live_jev_score_answer_passes_with_its_rounding() -> Result<(), Box<dyn StdError>> {
        // Jev's own answer to the plan's severity question: the mean of the
        // two-decimal distribution is 1.05 exactly, and the rounding
        // allowances absorb the last-place error in each probability.
        let mut verdict = verdict([
            choice("billing", None),
            boolean(0.9),
            score(1.05, Some(&[0.13, 0.69, 0.18])),
        ]);
        verdict.rounding = Some(Rounding::decimals(Some(2), Some(2)));
        validate_verdict(&evaluation()?, &verdict)?;
        Ok(())
    }

    #[test]
    fn a_score_off_its_mean_fails_without_rounding() -> Result<(), Box<dyn StdError>> {
        // 0.69 + 2 * 0.18 = 1.05 in decimal, but not in binary floating
        // point, so without a declared rounding the base tolerance decides.
        // Move the score by a hair more than 1e-6 to make the mean rule fire
        // on every platform.
        let verdict = verdict([
            choice("billing", None),
            boolean(0.9),
            score(1.05 + 2e-6, Some(&[0.13, 0.69, 0.18])),
        ]);
        expect_rejection(&verdict, "probability-weighted mean")
    }

    #[test]
    fn a_score_distribution_of_the_wrong_length_fails() -> Result<(), Box<dyn StdError>> {
        let verdict = verdict([
            choice("billing", None),
            boolean(0.9),
            score(1.0, Some(&[0.5, 0.5])),
        ]);
        expect_rejection(&verdict, "complete distribution")
    }

    #[test]
    fn a_score_out_of_range_fails() -> Result<(), Box<dyn StdError>> {
        for value in [-0.1, 2.1, f64::NAN, f64::INFINITY] {
            let verdict = verdict([choice("billing", None), boolean(0.9), score(value, None)]);
            expect_rejection(&verdict, "score must be in [0, 2]")?;
        }
        Ok(())
    }

    #[test]
    fn a_boolean_out_of_range_fails() -> Result<(), Box<dyn StdError>> {
        for value in [-0.1, 1.1, f64::NAN, f64::NEG_INFINITY] {
            let verdict = verdict([choice("billing", None), boolean(value), score(1.5, None)]);
            expect_rejection(&verdict, "P(true)")?;
        }
        Ok(())
    }

    #[test]
    fn rounding_decimals_past_fifteen_fail() -> Result<(), Box<dyn StdError>> {
        let mut verdict = plain_verdict();
        verdict.rounding = Some(Rounding::decimals(Some(16), None));
        expect_rejection(&verdict, "rounding decimals must be at most 15")?;
        verdict.rounding = Some(Rounding::decimals(None, Some(16)));
        expect_rejection(&verdict, "rounding decimals must be at most 15")?;
        verdict.rounding = Some(Rounding::decimals(Some(15), Some(0)));
        validate_verdict(&evaluation()?, &verdict)?;
        Ok(())
    }
}
