//! Helpers shared by the evaluation codecs.
//!
//! The Vercel AI Gateway's evaluation protocol and TypeSafe's System One
//! protocol descend from the same SDK, so they encode a state, a description,
//! and a question the same way, and they report a choice or score
//! distribution in the same shapes. What differs between them — the yes/no
//! type word, the option ceiling, the answer envelope — is a parameter here
//! or stays in the codec.

use indexmap::IndexMap;
use serde_json::{Map, Value};

use crate::evaluation::{ChoiceAnswer, Question, QuestionId, ScoreAnswer, State};
use crate::resolver::ResolvedRoute;
use crate::types::{Error, ErrorKind};

/// The provider code an evaluation the protocol cannot carry is refused with.
const INVALID_EVALUATION_CODE: &str = "invalid_evaluation";

/// The most levels a score question may carry: Jev's documented maximum,
/// which both protocols that reach it share.
pub(super) const MAX_LEVELS: usize = 10;

/// The ceilings one protocol enforces before dispatch.
#[derive(Clone, Copy, Debug)]
pub(super) struct QuestionLimits {
    /// The most options a choice question may carry, or `None` when the
    /// protocol documents no ceiling.
    pub max_options: Option<usize>,
    /// The most levels a score question may carry.
    pub max_levels:  usize,
}

/// A state or instruction as the bare JSON value the protocols expect.
pub(super) fn encode_state(state: &State) -> Value {
    match state {
        State::Text(text) => Value::String(text.clone()),
        State::Json(value) => value.clone(),
    }
}

/// An optional description as the value or JSON `null`.
pub(super) fn encode_description(description: Option<&State>) -> Value {
    description.map_or(Value::Null, encode_state)
}

/// One question as the SDK's wire shape.
///
/// `boolean_type` is the type word for a yes/no question: `boolean` on the
/// gateway, `noul` on TypeSafe's own API. Choice options and score levels
/// go under `criteria`; a boolean's optional descriptions go under
/// `criteria.true` and `criteria.false`, and the key is omitted when neither
/// is set.
///
/// # Errors
///
/// [`ErrorKind::InvalidRequest`] with code `invalid_evaluation` when the
/// question exceeds one of `limits`.
pub(super) fn encode_question(
    route: &ResolvedRoute,
    id: &QuestionId,
    question: &Question,
    boolean_type: &str,
    limits: QuestionLimits,
) -> Result<Value, Error> {
    let mut encoded = Map::new();
    match question {
        Question::Choice {
            instructions,
            options,
        } => {
            if let Some(max) = limits.max_options
                && options.len() > max
            {
                return Err(invalid_evaluation(
                    route,
                    format!(
                        "question `{id}` has {} options; this protocol takes at most {max}",
                        options.len()
                    ),
                ));
            }
            let criteria: Map<String, Value> = options
                .iter()
                .map(|(name, description)| (name.clone(), encode_description(description.as_ref())))
                .collect();
            encoded.insert("type".to_owned(), "choice".into());
            encoded.insert("instructions".to_owned(), encode_state(instructions));
            encoded.insert("criteria".to_owned(), Value::Object(criteria));
        }
        Question::Score {
            instructions,
            levels,
        } => {
            if levels.len() > limits.max_levels {
                return Err(invalid_evaluation(
                    route,
                    format!(
                        "question `{id}` has {} levels; this protocol takes at most {}",
                        levels.len(),
                        limits.max_levels
                    ),
                ));
            }
            let criteria: Vec<Value> = levels
                .iter()
                .map(|level| encode_description(level.as_ref()))
                .collect();
            encoded.insert("type".to_owned(), "score".into());
            encoded.insert("instructions".to_owned(), encode_state(instructions));
            encoded.insert("criteria".to_owned(), Value::Array(criteria));
        }
        Question::Boolean {
            instructions,
            when_true,
            when_false,
        } => {
            encoded.insert("type".to_owned(), boolean_type.into());
            encoded.insert("instructions".to_owned(), encode_state(instructions));
            let mut criteria = Map::new();
            if let Some(description) = when_true {
                criteria.insert("true".to_owned(), encode_state(description));
            }
            if let Some(description) = when_false {
                criteria.insert("false".to_owned(), encode_state(description));
            }
            if !criteria.is_empty() {
                encoded.insert("criteria".to_owned(), Value::Object(criteria));
            }
        }
    }
    Ok(Value::Object(encoded))
}

/// An evaluation this protocol cannot carry, refused before dispatch.
pub(super) fn invalid_evaluation(route: &ResolvedRoute, message: String) -> Error {
    Error::new(ErrorKind::InvalidRequest, message)
        .with_provider(route.provider().id().clone())
        .with_provider_code(INVALID_EVALUATION_CODE)
}

/// A success body that does not match the protocol.
///
/// A malformed answer is never retried: a model that returned a bad shape
/// once tends to return it again, so the error keeps [`Error::new`]'s
/// default classification.
pub(super) fn decode_failure(route: &ResolvedRoute, detail: &str, body: Value) -> Error {
    Error::new(
        ErrorKind::ResponseDecode,
        format!("provider {} {detail}", route.provider().id()),
    )
    .with_provider(route.provider().id().clone())
    .with_raw_data(body)
}

/// The wire answer object for question `id`, or the detail of why it is not
/// an object.
pub(super) fn answer_object<'a>(
    id: &str,
    wire: &'a Value,
) -> Result<&'a Map<String, Value>, String> {
    wire.as_object()
        .ok_or_else(|| format!("answered question `{id}` with something other than an object"))
}

/// A choice answer from its wire object: the `choice` and, when present,
/// its `probabilities` in the question's option order. Confidence is left
/// empty; each protocol carries it differently.
pub(super) fn decode_choice(
    id: &str,
    question: Option<&Question>,
    object: &Map<String, Value>,
) -> Result<ChoiceAnswer, String> {
    let choice = object
        .get("choice")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("answered choice question `{id}` without a choice"))?;
    let probabilities = distribution(object)
        .map(|wire| choice_probabilities(id, question, wire))
        .transpose()?;
    Ok(ChoiceAnswer {
        choice: choice.to_owned(),
        probabilities,
        confidence: None,
    })
}

/// A score answer from its wire object: the `score` and, when present, its
/// `probabilities` as a vector over the levels. Confidence is left empty.
pub(super) fn decode_score(
    id: &str,
    question: Option<&Question>,
    object: &Map<String, Value>,
) -> Result<ScoreAnswer, String> {
    let score = object
        .get("score")
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("answered score question `{id}` without a number"))?;
    let probabilities = distribution(object)
        .map(|wire| score_probabilities(id, question, wire))
        .transpose()?;
    Ok(ScoreAnswer {
        score,
        probabilities,
        confidence: None,
    })
}

/// The `probabilities` value, treating JSON `null` as absent.
fn distribution(object: &Map<String, Value>) -> Option<&Value> {
    object.get("probabilities").filter(|value| !value.is_null())
}

/// The wire's option-keyed probabilities in the question's option order.
///
/// Without a matching choice question there is no order to follow, so the
/// wire order is kept and the client's validator judges the answer.
fn choice_probabilities(
    id: &str,
    question: Option<&Question>,
    wire: &Value,
) -> Result<IndexMap<String, f64>, String> {
    let object = wire.as_object().ok_or_else(|| {
        format!("reported probabilities for question `{id}` that are not keyed by option")
    })?;
    let probability = |option: &str, value: &Value| {
        value.as_f64().ok_or_else(|| {
            format!("reported a non-numeric probability for option `{option}` of question `{id}`")
        })
    };

    let Some(Question::Choice { options, .. }) = question else {
        return object
            .iter()
            .map(|(option, value)| Ok((option.clone(), probability(option, value)?)))
            .collect();
    };
    let mut probabilities = IndexMap::with_capacity(options.len());
    for option in options.keys() {
        let value = object.get(option).ok_or_else(|| {
            format!("reported no probability for option `{option}` of question `{id}`")
        })?;
        probabilities.insert(option.clone(), probability(option, value)?);
    }
    if let Some(foreign) = object
        .keys()
        .find(|option| !options.contains_key(option.as_str()))
    {
        return Err(format!(
            "reported a probability for `{foreign}`, which is not an option of question `{id}`"
        ));
    }
    Ok(probabilities)
}

/// The wire's index-keyed probabilities as a vector over the levels.
///
/// Without a matching score question the wire's own size sets the level
/// count, and the client's validator judges the answer.
fn score_probabilities(
    id: &str,
    question: Option<&Question>,
    wire: &Value,
) -> Result<Vec<f64>, String> {
    let object = wire.as_object().ok_or_else(|| {
        format!("reported probabilities for question `{id}` that are not keyed by level")
    })?;
    let levels = match question {
        Some(Question::Score { levels, .. }) => levels.len(),
        _ => object.len(),
    };
    let mut probabilities = Vec::with_capacity(levels);
    for index in 0..levels {
        let value = object.get(&index.to_string()).ok_or_else(|| {
            format!("reported no probability for level {index} of question `{id}`")
        })?;
        probabilities.push(value.as_f64().ok_or_else(|| {
            format!("reported a non-numeric probability for level {index} of question `{id}`")
        })?);
    }
    if object.len() != levels {
        let extra = object
            .keys()
            .find(|key| !key.parse::<usize>().is_ok_and(|index| index < levels))
            .map_or("?", String::as_str);
        return Err(format!(
            "reported a probability for level `{extra}`, which question `{id}` does not have"
        ));
    }
    Ok(probabilities)
}
