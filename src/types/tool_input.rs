use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use super::ToolCallKind;

/// A tool's input, with its kind determined by its representation.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ToolInput {
    Function(ToolArguments),
    Custom(String),
}

impl ToolInput {
    pub fn kind(&self) -> ToolCallKind {
        match self {
            Self::Function(_) => ToolCallKind::Function,
            Self::Custom(_) => ToolCallKind::Custom,
        }
    }

    /// The exact text used by protocols that replay arguments as a string.
    pub fn raw(&self) -> &str {
        match self {
            Self::Function(arguments) => arguments.raw(),
            Self::Custom(text) => text,
        }
    }

    /// Projects parsed function arguments or custom text to an owned JSON
    /// value. This checks JSON syntax, not the tool's schema.
    pub fn to_value(&self) -> Result<Value, ToolArgumentError> {
        match self {
            Self::Function(arguments) => arguments.json().cloned().map_err(Clone::clone),
            Self::Custom(text) => Ok(Value::String(text.clone())),
        }
    }

    pub(crate) fn from_wire(kind: ToolCallKind, raw: String) -> Self {
        match kind {
            ToolCallKind::Function if raw.is_empty() => {
                Self::Function(ToolArguments::from_json(serde_json::json!({})))
            }
            ToolCallKind::Function => Self::Function(ToolArguments::from_raw(raw)),
            ToolCallKind::Custom => Self::Custom(raw),
        }
    }

    /// JSON protocols retain malformed argument text as a string on replay.
    pub(crate) fn wire_value(&self) -> Value {
        self.to_value()
            .unwrap_or_else(|_| Value::String(self.raw().to_owned()))
    }
}

/// Function arguments whose original bytes and parsed value stay consistent.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(from = "String", into = "String")]
pub struct ToolArguments {
    raw:    String,
    parsed: Result<Value, ToolArgumentError>,
}

impl ToolArguments {
    pub fn from_json(value: Value) -> Self {
        Self {
            raw:    value.to_string(),
            parsed: Ok(value),
        }
    }

    /// Preserves even malformed JSON so the application can report and replay
    /// it.
    pub fn from_raw(raw: String) -> Self {
        let parsed = serde_json::from_str(&raw).map_err(|source| ToolArgumentError {
            source: Arc::new(source),
        });
        Self { raw, parsed }
    }

    pub fn json(&self) -> Result<&Value, &ToolArgumentError> {
        self.parsed.as_ref()
    }
    pub fn raw(&self) -> &str {
        &self.raw
    }
    pub fn replace(&mut self, value: Value) {
        *self = Self::from_json(value);
    }
}

impl PartialEq for ToolArguments {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}

impl From<String> for ToolArguments {
    fn from(raw: String) -> Self {
        Self::from_raw(raw)
    }
}

impl From<ToolArguments> for String {
    fn from(arguments: ToolArguments) -> Self {
        arguments.raw
    }
}

/// Function arguments were not syntactically valid JSON.
#[derive(Clone, Debug, Error)]
#[error("tool arguments are not valid JSON")]
pub struct ToolArgumentError {
    #[source]
    source: Arc<serde_json::Error>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::ToolArguments;

    #[test]
    fn preserves_bytes_until_replaced() {
        let mut arguments = ToolArguments::from_raw("{ \"z\": 1, \"a\": 2 }".to_owned());
        assert_eq!(arguments.json().expect("valid"), &json!({"z": 1, "a": 2}));
        assert_eq!(arguments.raw(), "{ \"z\": 1, \"a\": 2 }");
        arguments.replace(json!({"path": "new.txt"}));
        assert_eq!(arguments.raw(), "{\"path\":\"new.txt\"}");
    }

    #[test]
    fn malformed_arguments_survive_serialization() {
        let arguments = ToolArguments::from_raw("{\"broken\":".to_owned());
        assert!(arguments.json().is_err());
        let encoded = serde_json::to_string(&arguments).expect("serialize");
        let decoded: ToolArguments = serde_json::from_str(&encoded).expect("deserialize");
        assert_eq!(decoded, arguments);
        assert!(decoded.json().is_err());
    }
}
