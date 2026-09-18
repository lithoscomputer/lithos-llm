use std::collections::BTreeMap;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use super::{QuestionId, QuestionKind};
use crate::catalog::{ModelHandle, ModelId, ProviderId};
use crate::types::{Cost, RateLimits, TokenCounts, Usage, Warning};

/// The answer to one question.
///
/// The serialized form is this crate's own shape, tagged by `type`; a codec
/// translates each provider's wire format into it.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Answer {
    Choice(ChoiceAnswer),
    Score(ScoreAnswer),
    Boolean(BooleanAnswer),
}

impl Answer {
    pub fn kind(&self) -> QuestionKind {
        match self {
            Self::Choice(_) => QuestionKind::Choice,
            Self::Score(_) => QuestionKind::Score,
            Self::Boolean(_) => QuestionKind::Boolean,
        }
    }
}

/// The answer to a [`Question::Choice`](super::Question::Choice).
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ChoiceAnswer {
    /// One of the question's option names, and the most probable one when a
    /// distribution is present.
    pub choice:        String,
    /// Probability per option, in the question's option order. Sums to one
    /// within the provider's declared rounding. `None` when the provider
    /// reports only the choice.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probabilities: Option<IndexMap<String, f64>>,
    /// How concentrated the distribution is, in `[0, 1]`, when the provider
    /// reports it. Not the selected option's probability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence:    Option<f64>,
}

/// The answer to a [`Question::Score`](super::Question::Score).
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ScoreAnswer {
    /// A position on the question's scale: `0.0` is the first level and
    /// `levels.len() - 1` the last. When a distribution is present this is
    /// its probability-weighted mean.
    pub score:         f64,
    /// Probability per level, indexed from zero. `None` when the provider
    /// reports only the score.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probabilities: Option<Vec<f64>>,
    /// As on [`ChoiceAnswer::confidence`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence:    Option<f64>,
}

impl ScoreAnswer {
    /// The nearest whole level, for code that needs one outcome.
    ///
    /// A score exactly between two levels rounds to the higher one, so `1.5`
    /// is level `2`. A negative or NaN score, which a validated verdict never
    /// carries, is level `0`; a score past the scale's end is not clamped
    /// here because the answer does not know how many levels the question
    /// had.
    pub fn nearest_level(&self) -> usize {
        if self.score.is_nan() || self.score < 0.0 {
            return 0;
        }
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "the score is finite and nonnegative here, and `as` saturates past usize::MAX"
        )]
        {
            // `round` on a nonnegative value rounds a half up.
            self.score.round() as usize
        }
    }
}

/// The answer to a [`Question::Boolean`](super::Question::Boolean).
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct BooleanAnswer {
    /// The probability that the statement holds, in `[0, 1]`.
    pub probability: f64,
}

impl BooleanAnswer {
    /// Whether the statement more likely holds than not: `probability >= 0.5`.
    ///
    /// A NaN probability, which a validated verdict never carries, is not
    /// likely.
    pub fn is_likely(&self) -> bool {
        self.probability >= 0.5
    }
}

/// Decimal places a provider rounds its probabilities and scores to.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct Rounding {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probability_decimals: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score_decimals:       Option<u8>,
}

/// The typed answers to one evaluation, one per question, with the same
/// accounting fields as a [`Response`](crate::Response).
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Verdict {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id:                Option<String>,
    /// The canonical route, as on `Response::model`.
    pub model:             ModelHandle,
    /// The model the provider says answered, when it names one. An alias
    /// such as `jev-latest` resolves to a versioned id here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub served_by:         Option<String>,
    pub answers:           BTreeMap<QuestionId, Answer>,
    /// Decimal places the provider rounds probabilities and scores to, when
    /// it says. Validation allows half a unit in the last place per rounded
    /// value; the values themselves are never renormalized.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rounding:          Option<Rounding>,
    #[serde(default)]
    pub usage:             TokenCounts,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost:              Option<Cost>,
    /// The provider's rate-limit headers on this response, as on
    /// [`Response::rate_limits`](crate::Response::rate_limits).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_limits:       Option<RateLimits>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings:          Vec<Warning>,
    /// The complete provider success payload, when one was available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw:               Option<Value>,
    /// Provider-specific extras the codec did not model, keyed by provider
    /// namespace. Confidence is modeled on the answers, so it is not here.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub provider_metadata: BTreeMap<String, Value>,
}

impl Verdict {
    /// A verdict with `answers` and every other field at its default.
    pub fn new(
        provider: ProviderId,
        model: ModelId,
        answers: BTreeMap<QuestionId, Answer>,
    ) -> Self {
        Self {
            id: None,
            model: ModelHandle::new(provider, model),
            served_by: None,
            answers,
            rounding: None,
            usage: TokenCounts::default(),
            cost: None,
            rate_limits: None,
            warnings: Vec::new(),
            raw: None,
            provider_metadata: BTreeMap::new(),
        }
    }

    /// The answer to `id`, whatever its kind.
    ///
    /// # Errors
    ///
    /// [`AnswerError::Unknown`] when no question has that id.
    pub fn answer(&self, id: &str) -> Result<&Answer, AnswerError> {
        self.answers.get(id).ok_or_else(|| AnswerError::Unknown {
            id: QuestionId::new(id),
        })
    }

    /// The answer to the choice question `id`.
    ///
    /// # Errors
    ///
    /// [`AnswerError::Unknown`] when no question has that id, and
    /// [`AnswerError::WrongKind`] when the question is not a choice.
    pub fn choice(&self, id: &str) -> Result<&ChoiceAnswer, AnswerError> {
        match self.answer(id)? {
            Answer::Choice(answer) => Ok(answer),
            other => Err(wrong_kind(id, QuestionKind::Choice, other)),
        }
    }

    /// The answer to the score question `id`.
    ///
    /// # Errors
    ///
    /// [`AnswerError::Unknown`] when no question has that id, and
    /// [`AnswerError::WrongKind`] when the question is not a score.
    pub fn score(&self, id: &str) -> Result<&ScoreAnswer, AnswerError> {
        match self.answer(id)? {
            Answer::Score(answer) => Ok(answer),
            other => Err(wrong_kind(id, QuestionKind::Score, other)),
        }
    }

    /// The answer to the boolean question `id`.
    ///
    /// # Errors
    ///
    /// [`AnswerError::Unknown`] when no question has that id, and
    /// [`AnswerError::WrongKind`] when the question is not a boolean.
    pub fn boolean(&self, id: &str) -> Result<&BooleanAnswer, AnswerError> {
        match self.answer(id)? {
            Answer::Boolean(answer) => Ok(answer),
            other => Err(wrong_kind(id, QuestionKind::Boolean, other)),
        }
    }

    /// The verdict's token counts and cost as one [`Usage`].
    pub fn usage_with_cost(&self) -> Usage {
        Usage {
            tokens: self.usage,
            cost:   self.cost,
        }
    }
}

fn wrong_kind(id: &str, requested: QuestionKind, actual: &Answer) -> AnswerError {
    AnswerError::WrongKind {
        id: QuestionId::new(id),
        requested,
        actual: actual.kind(),
    }
}

/// A typed answer lookup on a [`Verdict`] did not match the questions asked.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
#[non_exhaustive]
pub enum AnswerError {
    #[error("no question `{id}` was asked")]
    Unknown { id: QuestionId },
    #[error("question `{id}` is a {actual:?} question, not {requested:?}")]
    WrongKind {
        id:        QuestionId,
        requested: QuestionKind,
        actual:    QuestionKind,
    },
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::error::Error as StdError;

    use indexmap::IndexMap;
    use serde_json::json;

    use super::{Answer, AnswerError, BooleanAnswer, ChoiceAnswer, Rounding, ScoreAnswer, Verdict};
    use crate::catalog::{ModelId, ProviderId};
    use crate::evaluation::{QuestionId, QuestionKind};
    use crate::types::{Cost, CostSource, TokenCounts, Usage, Warning};

    fn choice() -> ChoiceAnswer {
        let mut probabilities = IndexMap::new();
        probabilities.insert("billing".to_owned(), 0.91);
        probabilities.insert("technical".to_owned(), 0.06);
        probabilities.insert("other".to_owned(), 0.03);
        ChoiceAnswer {
            choice:        "billing".to_owned(),
            probabilities: Some(probabilities),
            confidence:    Some(0.88),
        }
    }

    fn score() -> ScoreAnswer {
        ScoreAnswer {
            score:         1.3,
            probabilities: Some(vec![0.1, 0.5, 0.4]),
            confidence:    None,
        }
    }

    fn boolean() -> BooleanAnswer {
        BooleanAnswer { probability: 0.97 }
    }

    fn mixed_answers() -> BTreeMap<QuestionId, Answer> {
        let mut answers = BTreeMap::new();
        answers.insert(QuestionId::new("department"), Answer::Choice(choice()));
        answers.insert(QuestionId::new("severity"), Answer::Score(score()));
        answers.insert(
            QuestionId::new("requests_refund"),
            Answer::Boolean(boolean()),
        );
        answers
    }

    fn minimal_verdict() -> Verdict {
        Verdict::new(
            ProviderId::new("vercel"),
            ModelId::new("jev"),
            mixed_answers(),
        )
    }

    fn complete_verdict() -> Verdict {
        let mut verdict = minimal_verdict();
        verdict.id = Some("gen_1".to_owned());
        verdict.served_by = Some("jev-1.13.0".to_owned());
        verdict.rounding = Some(Rounding {
            probability_decimals: Some(2),
            score_decimals:       Some(1),
        });
        verdict.usage = TokenCounts {
            input: 380,
            ..TokenCounts::default()
        };
        verdict.cost = Some(Cost {
            usd_micros: 16,
            source:     CostSource::Provider,
        });
        verdict.warnings = vec![Warning {
            code:    "unsupported_setting".to_owned(),
            message: "temperature is ignored".to_owned(),
        }];
        verdict.raw = Some(json!({ "answers": {} }));
        verdict
            .provider_metadata
            .insert("gateway".to_owned(), json!({ "routing": "direct" }));
        verdict
    }

    #[test]
    fn typed_accessors_return_the_matching_answer() -> Result<(), Box<dyn StdError>> {
        let verdict = minimal_verdict();

        assert_eq!(verdict.choice("department")?, &choice());
        assert_eq!(verdict.score("severity")?, &score());
        assert_eq!(verdict.boolean("requests_refund")?, &boolean());
        assert_eq!(verdict.answer("severity")?.kind(), QuestionKind::Score);
        Ok(())
    }

    #[test]
    fn every_accessor_reports_an_unknown_id() {
        let verdict = minimal_verdict();
        let unknown = AnswerError::Unknown {
            id: QuestionId::new("missing"),
        };

        assert_eq!(verdict.answer("missing"), Err(unknown.clone()));
        assert_eq!(verdict.choice("missing"), Err(unknown.clone()));
        assert_eq!(verdict.score("missing"), Err(unknown.clone()));
        assert_eq!(verdict.boolean("missing"), Err(unknown.clone()));
        assert_eq!(unknown.to_string(), "no question `missing` was asked");
    }

    #[test]
    fn every_accessor_reports_the_kind_it_wanted_and_the_kind_it_found() {
        let verdict = minimal_verdict();
        let cases: [(&str, QuestionKind, QuestionKind); 6] = [
            ("severity", QuestionKind::Choice, QuestionKind::Score),
            (
                "requests_refund",
                QuestionKind::Choice,
                QuestionKind::Boolean,
            ),
            ("department", QuestionKind::Score, QuestionKind::Choice),
            (
                "requests_refund",
                QuestionKind::Score,
                QuestionKind::Boolean,
            ),
            ("department", QuestionKind::Boolean, QuestionKind::Choice),
            ("severity", QuestionKind::Boolean, QuestionKind::Score),
        ];

        for (id, requested, actual) in cases {
            let error = match requested {
                QuestionKind::Choice => verdict.choice(id).map(|_| ()),
                QuestionKind::Score => verdict.score(id).map(|_| ()),
                QuestionKind::Boolean => verdict.boolean(id).map(|_| ()),
            }
            .expect_err("the kinds differ");
            assert_eq!(error, AnswerError::WrongKind {
                id: QuestionId::new(id),
                requested,
                actual,
            });
            assert_eq!(
                error.to_string(),
                format!("question `{id}` is a {actual:?} question, not {requested:?}")
            );
        }
    }

    #[test]
    fn nearest_level_rounds_a_half_up() {
        let level = |score: f64| {
            ScoreAnswer {
                score,
                probabilities: None,
                confidence: None,
            }
            .nearest_level()
        };

        assert_eq!(level(0.0), 0);
        assert_eq!(level(0.49), 0);
        assert_eq!(level(0.5), 1);
        assert_eq!(level(1.5), 2);
        assert_eq!(level(2.5), 3);
        assert_eq!(level(2.0), 2);
        // Out-of-range inputs a validated verdict never carries still land on
        // a level.
        assert_eq!(level(-0.3), 0);
        assert_eq!(level(f64::NAN), 0);
        assert_eq!(level(f64::INFINITY), usize::MAX);
    }

    #[test]
    fn is_likely_starts_at_one_half() {
        let likely = |probability: f64| BooleanAnswer { probability }.is_likely();

        assert!(likely(0.5));
        assert!(likely(1.0));
        assert!(!likely(0.499_999_999));
        assert!(!likely(0.0));
        assert!(!likely(f64::NAN));
    }

    #[test]
    fn answers_serialize_to_the_tagged_shape() -> Result<(), Box<dyn StdError>> {
        assert_eq!(
            serde_json::to_value(Answer::Choice(choice()))?,
            json!({
                "type": "choice",
                "choice": "billing",
                "probabilities": { "billing": 0.91, "technical": 0.06, "other": 0.03 },
                "confidence": 0.88
            })
        );
        assert_eq!(
            serde_json::to_value(Answer::Score(score()))?,
            json!({ "type": "score", "score": 1.3, "probabilities": [0.1, 0.5, 0.4] })
        );
        assert_eq!(
            serde_json::to_value(Answer::Boolean(boolean()))?,
            json!({ "type": "boolean", "probability": 0.97 })
        );
        let bare = Answer::Choice(ChoiceAnswer {
            choice:        "other".to_owned(),
            probabilities: None,
            confidence:    None,
        });
        assert_eq!(
            serde_json::to_value(&bare)?,
            json!({ "type": "choice", "choice": "other" })
        );
        Ok(())
    }

    #[test]
    fn every_answer_round_trips_and_its_tag_is_its_kind() -> Result<(), Box<dyn StdError>> {
        for answer in mixed_answers().into_values() {
            let encoded = serde_json::to_value(&answer)?;
            assert_eq!(encoded["type"], serde_json::to_value(answer.kind())?);
            assert_eq!(serde_json::from_value::<Answer>(encoded)?, answer);
        }
        Ok(())
    }

    #[test]
    fn choice_probabilities_keep_the_option_order() -> Result<(), Box<dyn StdError>> {
        let decoded =
            serde_json::from_value::<Answer>(serde_json::to_value(Answer::Choice(choice()))?)?;
        let Answer::Choice(ChoiceAnswer {
            probabilities: Some(probabilities),
            ..
        }) = decoded
        else {
            return Err("expected a choice with a distribution".into());
        };

        assert_eq!(
            probabilities.keys().map(String::as_str).collect::<Vec<_>>(),
            ["billing", "technical", "other"]
        );
        Ok(())
    }

    #[test]
    fn a_complete_verdict_round_trips() -> Result<(), Box<dyn StdError>> {
        let verdict = complete_verdict();

        let encoded = serde_json::to_value(&verdict)?;

        assert_eq!(
            encoded["rounding"],
            json!({ "probability_decimals": 2, "score_decimals": 1 })
        );
        assert_eq!(encoded["answers"]["department"]["type"], json!("choice"));
        assert_eq!(serde_json::from_value::<Verdict>(encoded)?, verdict);
        Ok(())
    }

    #[test]
    fn a_minimal_verdict_omits_every_unset_field() -> Result<(), Box<dyn StdError>> {
        let verdict = minimal_verdict();

        let encoded = serde_json::to_value(&verdict)?;

        let mut keys = encoded
            .as_object()
            .ok_or("a verdict is an object")?
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>();
        keys.sort_unstable();
        assert_eq!(keys, ["answers", "model", "usage"]);
        assert_eq!(serde_json::from_value::<Verdict>(encoded)?, verdict);
        Ok(())
    }

    #[test]
    fn a_verdict_pairs_its_usage_and_cost() {
        assert_eq!(minimal_verdict().usage_with_cost(), Usage::default());

        let verdict = complete_verdict();
        assert_eq!(verdict.usage_with_cost(), Usage {
            tokens: verdict.usage,
            cost:   verdict.cost,
        });
    }

    #[test]
    fn rounding_omits_unset_decimals() -> Result<(), Box<dyn StdError>> {
        assert_eq!(serde_json::to_value(Rounding::default())?, json!({}));
        assert_eq!(
            serde_json::from_value::<Rounding>(json!({}))?,
            Rounding::default()
        );
        Ok(())
    }
}
