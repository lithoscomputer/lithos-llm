use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::time::Duration;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

use super::{Instructions, Question, QuestionId, State};
use crate::catalog::ProviderId;
use crate::types::duration_millis;

/// Typed questions to ask about one piece of state.
///
/// `Evaluation` is to the client's `evaluate` what [`Request`](crate::Request)
/// is to `complete`: the whole input, built once
/// and validated on [`EvaluationBuilder::build`]. Its serialized form is this
/// crate's own shape, not any provider's.
///
/// ```
/// use lithos_llm::Evaluation;
/// use lithos_llm::types::QuestionKind;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let evaluation = Evaluation::builder()
///     .model("vercel/jev")
///     .state("I was charged twice. Please refund the duplicate.")
///     .choice("department", "Which team should handle this?", [
///         ("billing", Some("Charges and refunds")),
///         ("technical", Some("Bugs and outages")),
///         ("other", None),
///     ])
///     .score("severity", "How severe is the issue?", [
///         "Cosmetic",
///         "Workaround exists",
///         "Blocking; no workaround",
///     ])
///     .boolean("requests_refund", "Is the customer requesting money back?")
///     .build()?;
///
/// assert_eq!(evaluation.questions().len(), 3);
/// assert_eq!(
///     evaluation.questions()["severity"].kind(),
///     QuestionKind::Score
/// );
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(try_from = "EvaluationBuilder")]
pub struct Evaluation {
    model:            String,
    state:            State,
    questions:        BTreeMap<QuestionId, Question>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "duration_millis"
    )]
    timeout:          Option<Duration>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    metadata:         BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    provider_options: BTreeMap<ProviderId, Map<String, Value>>,
}

impl Evaluation {
    pub fn builder() -> EvaluationBuilder {
        EvaluationBuilder::default()
    }

    /// Returns a builder that preserves every setting in this evaluation.
    pub fn into_builder(self) -> EvaluationBuilder {
        EvaluationBuilder {
            model:            Some(self.model),
            state:            Some(self.state),
            questions:        self.questions,
            duplicate:        None,
            timeout:          self.timeout,
            metadata:         self.metadata,
            provider_options: self.provider_options,
        }
    }

    /// The model selector, resolved through the catalog like a request's.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The shared state every question is asked about.
    pub fn state(&self) -> &State {
        &self.state
    }

    /// The questions by id. A `BTreeMap`, so the order is deterministic.
    pub fn questions(&self) -> &BTreeMap<QuestionId, Question> {
        &self.questions
    }

    pub fn timeout(&self) -> Option<Duration> {
        self.timeout
    }

    /// Free-form metadata, forwarded where a protocol supports it.
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    /// Raw provider options, keyed by canonical catalog provider id.
    pub fn provider_options(&self) -> &BTreeMap<ProviderId, Map<String, Value>> {
        &self.provider_options
    }
}

/// Builds and validates an [`Evaluation`].
#[derive(Default, Deserialize)]
#[serde(default)]
#[must_use]
pub struct EvaluationBuilder {
    model:            Option<String>,
    state:            Option<State>,
    questions:        BTreeMap<QuestionId, Question>,
    /// The first id added twice, reported by `build`.
    #[serde(skip)]
    duplicate:        Option<QuestionId>,
    #[serde(with = "duration_millis")]
    timeout:          Option<Duration>,
    metadata:         BTreeMap<String, String>,
    provider_options: BTreeMap<ProviderId, Map<String, Value>>,
}

impl TryFrom<EvaluationBuilder> for Evaluation {
    type Error = EvaluationBuildError;

    fn try_from(builder: EvaluationBuilder) -> Result<Self, Self::Error> {
        builder.build()
    }
}

impl EvaluationBuilder {
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// The shared state every question is asked about: a string, or any
    /// JSON value (an object or an array of records, say).
    pub fn state(mut self, state: impl Into<State>) -> Self {
        self.state = Some(state.into());
        self
    }

    /// Adds one question under `id`. The shorthands below cover the common
    /// shapes; this one takes a fully built [`Question`].
    ///
    /// Adding a second question under an id already taken is reported by
    /// [`build`](Self::build) as [`EvaluationBuildError::DuplicateQuestion`];
    /// the first question is never silently replaced.
    pub fn question(mut self, id: impl Into<QuestionId>, question: Question) -> Self {
        match self.questions.entry(id.into()) {
            Entry::Vacant(slot) => {
                slot.insert(question);
            }
            Entry::Occupied(taken) => {
                self.duplicate.get_or_insert_with(|| taken.key().clone());
            }
        }
        self
    }

    /// Adds a [`Question::Choice`]: pick one of `options`, each an option
    /// name with an optional description.
    ///
    /// When every description is `None` the description type cannot be
    /// inferred; write `Option::<&str>::None` for one of them, or add the
    /// question through [`question`](Self::question).
    pub fn choice<K, D>(
        self,
        id: impl Into<QuestionId>,
        instructions: impl Into<Instructions>,
        options: impl IntoIterator<Item = (K, Option<D>)>,
    ) -> Self
    where
        K: Into<String>,
        D: Into<Instructions>,
    {
        let options: IndexMap<String, Option<Instructions>> = options
            .into_iter()
            .map(|(name, description)| (name.into(), description.map(Into::into)))
            .collect();
        self.question(id, Question::Choice {
            instructions: instructions.into(),
            options,
        })
    }

    /// Adds a [`Question::Score`] with `levels` from lowest to highest.
    ///
    /// Each level is anything that converts into an optional description
    /// (see [`IntoLevel`]), so `["Cosmetic", "Blocking"]` and
    /// `[Some("Cosmetic"), None]` both read as a scale.
    pub fn score<L>(
        self,
        id: impl Into<QuestionId>,
        instructions: impl Into<Instructions>,
        levels: impl IntoIterator<Item = L>,
    ) -> Self
    where
        L: IntoLevel,
    {
        self.question(id, Question::Score {
            instructions: instructions.into(),
            levels:       levels.into_iter().map(IntoLevel::into_level).collect(),
        })
    }

    /// Adds a [`Question::Boolean`] with no criteria.
    pub fn boolean(self, id: impl Into<QuestionId>, instructions: impl Into<Instructions>) -> Self {
        self.question(id, Question::Boolean {
            instructions: instructions.into(),
            when_true:    None,
            when_false:   None,
        })
    }

    /// [`boolean`](Self::boolean) with what `true` and `false` mean spelled
    /// out.
    pub fn boolean_with_criteria(
        self,
        id: impl Into<QuestionId>,
        instructions: impl Into<Instructions>,
        when_true: impl Into<Instructions>,
        when_false: impl Into<Instructions>,
    ) -> Self {
        self.question(id, Question::Boolean {
            instructions: instructions.into(),
            when_true:    Some(when_true.into()),
            when_false:   Some(when_false.into()),
        })
    }

    /// Sets the total call budget, as on
    /// [`RequestBuilder::timeout`](crate::types::RequestBuilder::timeout).
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Sets one metadata entry, replacing any earlier value for the key.
    pub fn metadata_entry(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Sets one raw option inside a provider namespace.
    ///
    /// `provider` is the canonical catalog provider id, never a codec or
    /// adapter id.
    pub fn provider_option(
        mut self,
        provider: impl Into<ProviderId>,
        key: impl Into<String>,
        value: Value,
    ) -> Self {
        self.provider_options
            .entry(provider.into())
            .or_default()
            .insert(key.into(), value);
        self
    }

    /// Replaces the complete raw options namespace for one provider.
    ///
    /// `provider` is the canonical catalog provider id, never a codec or
    /// adapter id.
    pub fn provider_options(
        mut self,
        provider: impl Into<ProviderId>,
        options: Map<String, Value>,
    ) -> Self {
        self.provider_options.insert(provider.into(), options);
        self
    }

    /// Validates the evaluation.
    ///
    /// # Errors
    ///
    /// [`EvaluationBuildError`] when the model is missing or blank, the state
    /// is absent or JSON `null`, there are no questions, a question id
    /// repeats or is blank, a choice has no options or a blank option name,
    /// or a score has fewer than two levels. Provider maxima (Jev's 255
    /// options and 10 levels) are not checked here; they belong to the model,
    /// and the codec refuses them before dispatch.
    pub fn build(self) -> Result<Evaluation, EvaluationBuildError> {
        let model = self
            .model
            .filter(|model| !model.trim().is_empty())
            .ok_or(EvaluationBuildError::MissingModel)?;
        let state = self
            .state
            .filter(|state| !matches!(state, State::Json(Value::Null)))
            .ok_or(EvaluationBuildError::MissingState)?;
        if self.questions.is_empty() {
            return Err(EvaluationBuildError::NoQuestions);
        }
        if let Some(id) = self.duplicate {
            return Err(EvaluationBuildError::DuplicateQuestion { id });
        }
        for (id, question) in &self.questions {
            if id.as_str().trim().is_empty() {
                return Err(EvaluationBuildError::EmptyQuestionId);
            }
            match question {
                Question::Choice { options, .. } => {
                    if options.is_empty() {
                        return Err(EvaluationBuildError::NoOptions { id: id.clone() });
                    }
                    if options.keys().any(|name| name.trim().is_empty()) {
                        return Err(EvaluationBuildError::EmptyOptionName { id: id.clone() });
                    }
                }
                Question::Score { levels, .. } => {
                    if levels.len() < 2 {
                        return Err(EvaluationBuildError::TooFewLevels { id: id.clone() });
                    }
                }
                Question::Boolean { .. } => {}
            }
        }

        Ok(Evaluation {
            model,
            state,
            questions: self.questions,
            timeout: self.timeout,
            metadata: self.metadata,
            provider_options: self.provider_options,
        })
    }
}

/// One level of a score scale, as [`EvaluationBuilder::score`] accepts it.
///
/// Implemented for `&str`, `String`, [`Value`], [`Instructions`], and
/// `Option<T>` for any `T` that converts into [`Instructions`]. `None` is a
/// level with no description. This trait is sealed.
pub trait IntoLevel: sealed::Sealed {
    fn into_level(self) -> Option<Instructions>;
}

mod sealed {
    use serde_json::Value;

    use crate::evaluation::Instructions;

    /// Bounds [`IntoLevel`](super::IntoLevel) so nothing outside the crate
    /// can implement it.
    pub trait Sealed {}

    impl Sealed for &str {}
    impl Sealed for String {}
    impl Sealed for Value {}
    impl Sealed for Instructions {}
    impl<T: Into<Instructions>> Sealed for Option<T> {}
}

impl IntoLevel for &str {
    fn into_level(self) -> Option<Instructions> {
        Some(self.into())
    }
}

impl IntoLevel for String {
    fn into_level(self) -> Option<Instructions> {
        Some(self.into())
    }
}

impl IntoLevel for Value {
    fn into_level(self) -> Option<Instructions> {
        Some(self.into())
    }
}

impl IntoLevel for Instructions {
    fn into_level(self) -> Option<Instructions> {
        Some(self)
    }
}

impl<T: Into<Instructions>> IntoLevel for Option<T> {
    fn into_level(self) -> Option<Instructions> {
        self.map(Into::into)
    }
}

/// An evaluation failed local construction checks.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
#[non_exhaustive]
pub enum EvaluationBuildError {
    #[error("an evaluation needs a model")]
    MissingModel,
    #[error("an evaluation needs state to ask about")]
    MissingState,
    #[error("an evaluation needs at least one question")]
    NoQuestions,
    #[error("question `{id}` is declared twice")]
    DuplicateQuestion { id: QuestionId },
    #[error("question ids must not be empty")]
    EmptyQuestionId,
    #[error("choice question `{id}` has no options")]
    NoOptions { id: QuestionId },
    #[error("score question `{id}` needs at least two levels")]
    TooFewLevels { id: QuestionId },
    #[error("question `{id}` has an empty option name")]
    EmptyOptionName { id: QuestionId },
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;
    use std::time::Duration;

    use indexmap::IndexMap;
    use serde_json::{Map, Value, json};

    use super::{Evaluation, EvaluationBuildError, EvaluationBuilder};
    use crate::catalog::ProviderId;
    use crate::evaluation::{Question, QuestionId, QuestionKind, State};

    fn base() -> EvaluationBuilder {
        Evaluation::builder()
            .model("vercel/jev")
            .state("I was charged twice.")
            .boolean("requests_refund", "Is the customer requesting money back?")
    }

    /// One question of each kind, exercising every optional part.
    fn mixed() -> EvaluationBuilder {
        let mut gateway = Map::new();
        gateway.insert("order".to_owned(), json!(["typesafe"]));
        Evaluation::builder()
            .model("vercel/jev")
            .state(json!({ "ticket": 41, "body": "I was charged twice." }))
            .choice("department", "Which team should handle this?", [
                ("billing", Some("Charges and refunds")),
                ("technical", Some("Bugs and outages")),
                ("other", None),
            ])
            .score("severity", "How severe is the issue?", [
                Some("Cosmetic"),
                None,
                Some("Blocking; no workaround"),
            ])
            .boolean_with_criteria(
                "requests_refund",
                "Is the customer requesting money back?",
                "Asks for a refund or chargeback",
                json!({ "anything": "else" }),
            )
            .timeout(Duration::from_millis(2500))
            .metadata_entry("tenant", "acme")
            .provider_option("vercel", "user", json!("u-1"))
            .provider_options("openai", gateway)
    }

    #[test]
    fn a_mixed_evaluation_round_trips() -> Result<(), Box<dyn StdError>> {
        let evaluation = mixed().build()?;

        let encoded = serde_json::to_value(&evaluation)?;
        let decoded = serde_json::from_value::<Evaluation>(encoded.clone())?;

        assert_eq!(decoded, evaluation);
        // The timeout is whole milliseconds, as on a request.
        assert_eq!(encoded["timeout"], json!(2500));
        assert_eq!(decoded.timeout(), Some(Duration::from_millis(2500)));
        assert_eq!(
            decoded.state(),
            &State::Json(json!({ "ticket": 41, "body": "I was charged twice." }))
        );
        assert_eq!(
            decoded
                .questions()
                .keys()
                .map(QuestionId::as_str)
                .collect::<Vec<_>>(),
            ["department", "requests_refund", "severity"]
        );
        assert_eq!(
            decoded.metadata().get("tenant").map(String::as_str),
            Some("acme")
        );
        assert_eq!(decoded.provider_options().len(), 2);
        assert_eq!(
            decoded
                .provider_options()
                .get(&ProviderId::new("vercel"))
                .and_then(|options| options.get("user")),
            Some(&json!("u-1"))
        );
        Ok(())
    }

    #[test]
    fn a_minimal_evaluation_omits_unset_fields() -> Result<(), Box<dyn StdError>> {
        let encoded = serde_json::to_value(base().build()?)?;

        let mut keys = encoded
            .as_object()
            .ok_or("an evaluation is an object")?
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>();
        keys.sort_unstable();
        assert_eq!(keys, ["model", "questions", "state"]);
        Ok(())
    }

    #[test]
    fn deserializing_runs_the_build_checks() {
        let document = json!({ "model": "vercel/jev", "state": "text", "questions": {} });

        let error = serde_json::from_value::<Evaluation>(document).expect_err("no questions");

        assert!(
            error.to_string().contains("at least one question"),
            "{error}"
        );
    }

    #[test]
    fn into_builder_preserves_every_setting() -> Result<(), Box<dyn StdError>> {
        let evaluation = mixed().build()?;

        let rebuilt = evaluation.clone().into_builder().build()?;

        assert_eq!(rebuilt, evaluation);
        Ok(())
    }

    #[test]
    fn the_shorthands_build_the_questions_they_name() -> Result<(), Box<dyn StdError>> {
        let evaluation = mixed().build()?;
        let questions = evaluation.questions();

        let Some(Question::Choice { options, .. }) = questions.get("department") else {
            return Err("department is a choice".into());
        };
        assert_eq!(options.keys().map(String::as_str).collect::<Vec<_>>(), [
            "billing",
            "technical",
            "other"
        ]);
        assert_eq!(options["other"], None);
        assert_eq!(options["billing"], Some(State::from("Charges and refunds")));

        let Some(Question::Score { levels, .. }) = questions.get("severity") else {
            return Err("severity is a score".into());
        };
        assert_eq!(levels.len(), 3);
        assert_eq!(levels[1], None);

        let Some(Question::Boolean {
            when_true,
            when_false,
            ..
        }) = questions.get("requests_refund")
        else {
            return Err("requests_refund is a boolean".into());
        };
        assert_eq!(
            when_true,
            &Some(State::from("Asks for a refund or chargeback"))
        );
        assert_eq!(
            when_false,
            &Some(State::Json(json!({ "anything": "else" })))
        );
        Ok(())
    }

    #[test]
    fn score_levels_accept_plain_and_optional_descriptions() -> Result<(), Box<dyn StdError>> {
        let levels = |evaluation: Evaluation| match evaluation.questions().get("severity") {
            Some(Question::Score { levels, .. }) => Ok(levels.clone()),
            _ => Err("severity is a score"),
        };

        let plain = levels(
            base()
                .score("severity", "How severe?", ["Low", "High"])
                .build()?,
        )?;
        assert_eq!(plain, vec![
            Some(State::from("Low")),
            Some(State::from("High"))
        ]);

        let owned = levels(
            base()
                .score("severity", "How severe?", [
                    String::from("Low"),
                    String::from("High"),
                ])
                .build()?,
        )?;
        assert_eq!(owned, plain);

        let json_levels = levels(
            base()
                .score("severity", "How severe?", [
                    json!({ "rank": 1 }),
                    json!({ "rank": 2 }),
                ])
                .build()?,
        )?;
        assert_eq!(json_levels, vec![
            Some(State::Json(json!({ "rank": 1 }))),
            Some(State::Json(json!({ "rank": 2 }))),
        ]);

        let optional = levels(
            base()
                .score("severity", "How severe?", [Some("Low"), None, Some("High")])
                .build()?,
        )?;
        assert_eq!(optional, vec![
            Some(State::from("Low")),
            None,
            Some(State::from("High"))
        ]);

        let states = levels(
            base()
                .score("severity", "How severe?", [
                    State::from("Low"),
                    State::from("High"),
                ])
                .build()?,
        )?;
        assert_eq!(states, plain);
        Ok(())
    }

    #[test]
    fn a_boolean_without_criteria_has_none() -> Result<(), Box<dyn StdError>> {
        let evaluation = base().build()?;

        assert_eq!(
            evaluation.questions().get("requests_refund"),
            Some(&Question::Boolean {
                instructions: State::from("Is the customer requesting money back?"),
                when_true:    None,
                when_false:   None,
            })
        );
        assert_eq!(
            evaluation.questions()["requests_refund"].kind(),
            QuestionKind::Boolean
        );
        Ok(())
    }

    #[test]
    fn rejects_a_missing_or_blank_model() {
        let error = Evaluation::builder()
            .state("text")
            .boolean("q", "?")
            .build()
            .unwrap_err();
        assert_eq!(error, EvaluationBuildError::MissingModel);

        let error = base().model("  ").build().unwrap_err();
        assert_eq!(error, EvaluationBuildError::MissingModel);
        assert_eq!(error.to_string(), "an evaluation needs a model");
    }

    #[test]
    fn rejects_a_missing_or_null_state() {
        let error = Evaluation::builder()
            .model("vercel/jev")
            .boolean("q", "?")
            .build()
            .unwrap_err();
        assert_eq!(error, EvaluationBuildError::MissingState);

        // A JSON `null` state would read back as no state at all.
        let error = base().state(Value::Null).build().unwrap_err();
        assert_eq!(error, EvaluationBuildError::MissingState);
    }

    #[test]
    fn rejects_no_questions() {
        let error = Evaluation::builder()
            .model("vercel/jev")
            .state("text")
            .build()
            .unwrap_err();

        assert_eq!(error, EvaluationBuildError::NoQuestions);
    }

    #[test]
    fn rejects_a_repeated_question_id_and_keeps_the_first() {
        let builder = base()
            .boolean("requests_refund", "Asked again?")
            .boolean("other", "?")
            .boolean("other", "Asked again?");

        let error = builder.build().unwrap_err();

        assert_eq!(error, EvaluationBuildError::DuplicateQuestion {
            id: QuestionId::new("requests_refund"),
        });
        assert_eq!(
            error.to_string(),
            "question `requests_refund` is declared twice"
        );
    }

    #[test]
    fn rejects_a_blank_question_id() {
        let error = base().boolean(" ", "?").build().unwrap_err();

        assert_eq!(error, EvaluationBuildError::EmptyQuestionId);
    }

    #[test]
    fn rejects_a_choice_without_options() {
        let error = base()
            .question("department", Question::Choice {
                instructions: State::from("Which team?"),
                options:      IndexMap::new(),
            })
            .build()
            .unwrap_err();

        assert_eq!(error, EvaluationBuildError::NoOptions {
            id: QuestionId::new("department"),
        });
    }

    #[test]
    fn rejects_a_blank_option_name() {
        let error = base()
            .choice("department", "Which team?", [
                ("billing", None),
                ("", Some("unnamed")),
            ])
            .build()
            .unwrap_err();

        assert_eq!(error, EvaluationBuildError::EmptyOptionName {
            id: QuestionId::new("department"),
        });
    }

    #[test]
    fn rejects_a_scale_with_fewer_than_two_levels() {
        let error = base()
            .score("severity", "How severe?", ["Only"])
            .build()
            .unwrap_err();
        assert_eq!(error, EvaluationBuildError::TooFewLevels {
            id: QuestionId::new("severity"),
        });

        let error = base()
            .score("severity", "How severe?", Vec::<&str>::new())
            .build()
            .unwrap_err();
        assert_eq!(error, EvaluationBuildError::TooFewLevels {
            id: QuestionId::new("severity"),
        });

        assert!(
            base()
                .score("severity", "How severe?", ["Low", "High"])
                .build()
                .is_ok()
        );
    }
}
