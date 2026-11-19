//! Lossless provider-native wire types.
//!
//! Each submodule mirrors one provider API exactly, in the provider's own
//! field names and nesting: its request body, its response body, the JSON
//! payload of each event in its SSE stream, and its error body. Unlike
//! [`Request`](crate::Request), which every codec converts to and from, these
//! types never normalize. They serve callers that must stay in a provider's
//! format, such as a proxy that reads a request, changes one field, and
//! forwards it, or renders its own errors the way the provider would.
//!
//! | Module | API | Codec |
//! | --- | --- | --- |
//! | [`anthropic_messages`] | Anthropic Messages | `anthropic-messages` |
//! | [`openai_responses`] | OpenAI Responses | `openai-responses` |
//! | [`openai_chat`] | OpenAI Chat Completions | `openai-chat` |
//!
//! # Losslessness
//!
//! Serializing a parsed value gives back the same JSON value it was parsed
//! from. Key order and whitespace are not preserved; everything else is:
//!
//! - Every object keeps the fields its type does not name in an `extra` map, so
//!   a new provider field, a beta parameter, or a cache-control marker survives
//!   a round trip untouched.
//! - Every tagged union has an `Unknown` variant that keeps a block, item, or
//!   event of a type it does not model as its full JSON object. A known type
//!   with malformed fields is an error, not an `Unknown`.
//! - A field is `T` when the API always sends it and never sends `null`, and a
//!   `null` there is an error. Every other field is a [`Field<T>`], which tells
//!   an absent field apart from a `null` one, so neither collapses into the
//!   other.
//! - Values the provider treats as opaque, such as tool input schemas, thinking
//!   signatures, and encrypted reasoning, are kept exactly as they arrived.
//!   Numeric request parameters such as `temperature` stay in `extra`, because
//!   typing them as `f64` would turn `1` into `1.0`.
//!
//! Parse with `serde_json` and edit the fields directly:
//!
//! ```
//! use lithos_llm::wire::anthropic_messages::Request;
//! use serde_json::json;
//!
//! let body = json!({
//!     "model": "claude-opus-5-5",
//!     "max_tokens": 1024,
//!     "messages": [{"role": "user", "content": "Hello"}],
//!     "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
//! });
//! let mut request: Request = serde_json::from_value(body.clone())?;
//! request.max_tokens = 256;
//!
//! let mut expected = body;
//! expected["max_tokens"] = json!(256);
//! assert_eq!(serde_json::to_value(&request)?, expected);
//! # Ok::<(), serde_json::Error>(())
//! ```
//!
//! SSE framing is not part of these types: parse the `data` of each event.
//! The Chat Completions stream ends with a `data: [DONE]` line that is not
//! JSON and has no type here.

pub mod anthropic_messages;
pub mod openai_chat;
pub mod openai_responses;

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

/// The fields of a JSON object that its type does not name.
pub type Extra = Map<String, Value>;

/// A field that the API may omit or send as `null`.
///
/// `Option<T>` would read both as `None` and write both back the same way,
/// so a `null` would disappear or an absent field would appear. A struct
/// field of this type is declared with
/// `#[serde(default, skip_serializing_if = "Field::is_absent")]`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Field<T> {
    /// The key is not in the object.
    #[default]
    Absent,
    /// The key is present with the value `null`.
    Null,
    /// The key is present with a value.
    Present(T),
}

impl<T> Field<T> {
    pub fn is_absent(&self) -> bool {
        matches!(self, Self::Absent)
    }

    /// The value, if the field is present and not `null`.
    pub fn as_option(&self) -> Option<&T> {
        match self {
            Self::Present(value) => Some(value),
            Self::Absent | Self::Null => None,
        }
    }

    /// The value, if the field is present and not `null`.
    pub fn into_option(self) -> Option<T> {
        match self {
            Self::Present(value) => Some(value),
            Self::Absent | Self::Null => None,
        }
    }
}

impl<T: Serialize> Serialize for Field<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Present(value) => value.serialize(serializer),
            // An absent field is skipped before it gets here.
            Self::Absent | Self::Null => serializer.serialize_none(),
        }
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Field<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // A missing key never reaches here: `#[serde(default)]` supplies
        // `Absent`.
        Ok(Option::deserialize(deserializer)?.map_or(Self::Null, Self::Present))
    }
}

/// A tagged union whose variants are selected by a string `type` field.
trait Tagged {
    /// The `type` values of the modeled variants.
    const TYPES: &'static [&'static str];
}

/// Deserializes the `Unknown` variant of a tagged union.
///
/// Serde tries an untagged variant after the tagged ones fail, both for an
/// unrecognized `type` and for a recognized one with malformed fields. This
/// accepts only the first, so a malformed known block stays an error.
fn unknown<'de, T, D>(deserializer: D) -> Result<Extra, D::Error>
where
    T: Tagged,
    D: Deserializer<'de>,
{
    let object = Extra::deserialize(deserializer)?;
    match object.get("type").and_then(Value::as_str) {
        Some(kind) if T::TYPES.contains(&kind) => {
            Err(D::Error::custom(format_args!("invalid `{kind}` object")))
        }
        Some(_) => Ok(object),
        None => Err(D::Error::custom("expected an object with a string `type`")),
    }
}

#[cfg(test)]
fn assert_types_are_modeled<T>(examples: &[Value])
where
    T: Tagged + for<'de> Deserialize<'de> + Serialize,
{
    let mut tags: Vec<&str> = examples
        .iter()
        .filter_map(|example| example.get("type").and_then(Value::as_str))
        .collect();
    tags.sort_unstable();
    let mut types = T::TYPES.to_vec();
    types.sort_unstable();
    assert_eq!(tags, types, "every listed type needs one example");

    for example in examples {
        let parsed: T = serde_json::from_value(example.clone()).expect("example parses");
        // A modeled variant accepts the example. If the type were listed
        // without a variant, parsing would have failed above instead.
        assert_eq!(serde_json::to_value(parsed).expect("serializes"), *example);
    }
}
