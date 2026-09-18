//! Typed questions about one piece of state, and the typed answers.
//!
//! An [`Evaluation`] names a model, a [`State`] to ask about, and a set of
//! [`Question`]s keyed by [`QuestionId`]. A [`Verdict`] answers every question
//! with an [`Answer`] of the matching kind. These types are the crate's own
//! shapes; codecs translate them to and from each provider's wire format.

mod answer;
mod builder;
mod question;
#[cfg(any(feature = "runtime", test))]
pub(crate) mod validate;

pub use answer::{
    Answer, AnswerError, BooleanAnswer, ChoiceAnswer, Rounding, ScoreAnswer, Verdict,
};
pub use builder::{Evaluation, EvaluationBuildError, EvaluationBuilder, IntoLevel};
pub use question::{Instructions, Question, QuestionId, QuestionKind, State};
