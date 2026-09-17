use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

crate::catalog::string_id!(
    /// The caller-chosen name of one question in an evaluation.
    ///
    /// Ids are never shown to the model; they only key the answers.
    QuestionId
);

/// What the questions are about: text, or any JSON value.
///
/// A text state and a JSON string are the same on the wire, so a
/// `State::Json(Value::String(..))` reads back as [`State::Text`].
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum State {
    Text(String),
    Json(Value),
}

impl From<&str> for State {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

impl From<String> for State {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<Value> for State {
    fn from(value: Value) -> Self {
        Self::Json(value)
    }
}

/// How a question, option, or level is described to the model: text, or any
/// JSON value.
///
/// An alias of [`State`] because the two serialize identically. It becomes
/// its own type if they ever diverge.
pub type Instructions = State;

/// One judgment to make about the state.
///
/// The serialized form is this crate's own shape, tagged by `type`; a codec
/// translates it to each provider's wire format.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Question {
    /// Pick one option from a named set.
    Choice {
        instructions: Instructions,
        /// Option name to description; `None` is an option with no
        /// description. Nonempty. Insertion order is preserved and is the
        /// order a [`ChoiceAnswer`](super::ChoiceAnswer) reports
        /// probabilities in.
        options:      IndexMap<String, Option<Instructions>>,
    },
    /// Rate the state on an ordered scale, lowest first.
    Score {
        instructions: Instructions,
        /// At least two levels; `None` is a level with no description. A
        /// [`ScoreAnswer`](super::ScoreAnswer) is a position on this scale,
        /// indexed from zero.
        levels:       Vec<Option<Instructions>>,
    },
    /// The probability that something holds.
    Boolean {
        instructions: Instructions,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        when_true:    Option<Instructions>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        when_false:   Option<Instructions>,
    },
}

impl Question {
    pub fn kind(&self) -> QuestionKind {
        match self {
            Self::Choice { .. } => QuestionKind::Choice,
            Self::Score { .. } => QuestionKind::Score,
            Self::Boolean { .. } => QuestionKind::Boolean,
        }
    }
}

/// The shapes a question can take, for capability checks and answer
/// matching.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum QuestionKind {
    Choice,
    Score,
    Boolean,
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use indexmap::IndexMap;
    use serde_json::{Value, json};

    use super::{Question, QuestionKind, State};

    #[test]
    fn state_converts_from_text_and_json() {
        assert_eq!(State::from("hello"), State::Text("hello".to_owned()));
        assert_eq!(
            State::from(String::from("hello")),
            State::Text("hello".to_owned())
        );
        assert_eq!(
            State::from(json!({ "order": 7 })),
            State::Json(json!({ "order": 7 }))
        );
    }

    #[test]
    fn state_serializes_as_its_bare_value() -> Result<(), Box<dyn StdError>> {
        let text = State::from("hello");
        assert_eq!(
            serde_json::to_value(&text)?,
            Value::String("hello".to_owned())
        );
        assert_eq!(serde_json::from_value::<State>(json!("hello"))?, text);

        let records = State::from(json!([{ "id": 1 }, { "id": 2 }]));
        assert_eq!(
            serde_json::to_value(&records)?,
            json!([{ "id": 1 }, { "id": 2 }])
        );
        assert_eq!(
            serde_json::from_value::<State>(json!([{ "id": 1 }, { "id": 2 }]))?,
            records
        );
        Ok(())
    }

    fn every_question() -> Vec<Question> {
        let mut options = IndexMap::new();
        options.insert(
            "billing".to_owned(),
            Some(State::from("Charges and refunds")),
        );
        options.insert("other".to_owned(), None);
        vec![
            Question::Choice {
                instructions: State::from("Which team?"),
                options,
            },
            Question::Score {
                instructions: State::from("How severe?"),
                levels:       vec![
                    Some(State::from("Cosmetic")),
                    None,
                    Some(State::from("Blocking")),
                ],
            },
            Question::Boolean {
                instructions: State::from("Refund requested?"),
                when_true:    None,
                when_false:   None,
            },
            Question::Boolean {
                instructions: State::from("Refund requested?"),
                when_true:    Some(State::from("Asks for money back")),
                when_false:   Some(json!({ "note": "anything else" }).into()),
            },
        ]
    }

    #[test]
    fn questions_serialize_to_the_tagged_shape() -> Result<(), Box<dyn StdError>> {
        let questions = every_question();
        assert_eq!(
            serde_json::to_value(&questions[0])?,
            json!({
                "type": "choice",
                "instructions": "Which team?",
                "options": { "billing": "Charges and refunds", "other": null }
            })
        );
        assert_eq!(
            serde_json::to_value(&questions[1])?,
            json!({
                "type": "score",
                "instructions": "How severe?",
                "levels": ["Cosmetic", null, "Blocking"]
            })
        );
        assert_eq!(
            serde_json::to_value(&questions[2])?,
            json!({ "type": "boolean", "instructions": "Refund requested?" })
        );
        assert_eq!(
            serde_json::to_value(&questions[3])?,
            json!({
                "type": "boolean",
                "instructions": "Refund requested?",
                "when_true": "Asks for money back",
                "when_false": { "note": "anything else" }
            })
        );
        Ok(())
    }

    #[test]
    fn every_question_round_trips_and_its_tag_is_its_kind() -> Result<(), Box<dyn StdError>> {
        for question in every_question() {
            let encoded = serde_json::to_value(&question)?;
            assert_eq!(encoded["type"], serde_json::to_value(question.kind())?);
            assert_eq!(serde_json::from_value::<Question>(encoded)?, question);
        }
        Ok(())
    }

    #[test]
    fn choice_options_keep_insertion_order() -> Result<(), Box<dyn StdError>> {
        let mut options = IndexMap::new();
        for name in ["zebra", "apple", "mango"] {
            options.insert(name.to_owned(), None);
        }
        let question = Question::Choice {
            instructions: State::from("Pick"),
            options,
        };

        let decoded = serde_json::from_value::<Question>(serde_json::to_value(&question)?)?;
        let Question::Choice { options, .. } = decoded else {
            return Err("expected a choice question".into());
        };

        assert_eq!(options.keys().map(String::as_str).collect::<Vec<_>>(), [
            "zebra", "apple", "mango"
        ]);
        Ok(())
    }

    #[test]
    fn kinds_serialize_in_snake_case() -> Result<(), Box<dyn StdError>> {
        for (kind, wire) in [
            (QuestionKind::Choice, "choice"),
            (QuestionKind::Score, "score"),
            (QuestionKind::Boolean, "boolean"),
        ] {
            assert_eq!(serde_json::to_value(kind)?, json!(wire));
            assert_eq!(serde_json::from_value::<QuestionKind>(json!(wire))?, kind);
        }
        Ok(())
    }
}
