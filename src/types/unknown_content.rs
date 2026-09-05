//! Preservation of content types introduced by newer writers.

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use thiserror::Error;

/// An unknown transcript content type, preserved as its full JSON object.
///
/// This is stored data, not provider-native `Opaque` content. Clients refuse
/// to dispatch it until the application explicitly converts or removes it.
/// Known types with malformed fields remain errors rather than falling back.
#[derive(Clone, Debug, PartialEq)]
pub struct UnknownContent {
    kind: String,
    raw:  Value,
}

impl UnknownContent {
    pub fn kind(&self) -> &str {
        &self.kind
    }

    /// The original object, including its `type` and all additional fields.
    pub fn as_value(&self) -> &Value {
        &self.raw
    }
}

/// Unknown content must be an object with an unrecognized string `type`.
#[derive(Debug, Error)]
#[error("expected an object with an unknown string content type")]
pub struct UnknownContentError;

impl TryFrom<Value> for UnknownContent {
    type Error = UnknownContentError;

    fn try_from(raw: Value) -> Result<Self, Self::Error> {
        let kind = raw
            .get("type")
            .and_then(Value::as_str)
            .ok_or(UnknownContentError)?;
        if matches!(
            kind,
            "text"
                | "image"
                | "audio"
                | "document"
                | "reasoning"
                | "tool_call"
                | "tool_result"
                | "json"
                | "opaque"
        ) {
            return Err(UnknownContentError);
        }
        Ok(Self {
            kind: kind.to_owned(),
            raw,
        })
    }
}

impl Serialize for UnknownContent {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.raw.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for UnknownContent {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::try_from(Value::deserialize(deserializer)?).map_err(D::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::types::{ContentPart, Message};

    #[test]
    fn future_content_round_trips_without_becoming_provider_opaque() {
        let raw = json!({"role":"assistant","content":[
            {"type":"future_video","source":{"frames":[1,2]},"revision":7},
            {"type":"text","text":"caption"}
        ]});
        let message: Message = serde_json::from_value(raw.clone()).expect("future transcript");
        let ContentPart::Unknown(content) = &message.content()[0] else {
            panic!("unknown content")
        };
        assert_eq!(content.kind(), "future_video");
        assert_eq!(content.as_value(), &raw["content"][0]);
        assert_eq!(serde_json::to_value(message).expect("preserve"), raw);
    }

    #[test]
    fn malformed_known_content_does_not_become_unknown() {
        for raw in [
            json!({"type":"text","text":42}),
            json!({"type":"tool_call","id":"missing-input"}),
            json!({"type":"opaque","kind":42}),
            json!({"type":42}),
            json!({"text":"missing type"}),
            json!([]),
        ] {
            assert!(
                serde_json::from_value::<ContentPart>(raw.clone()).is_err(),
                "{raw}"
            );
        }
    }
}
