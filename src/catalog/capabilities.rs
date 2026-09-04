//! Model support queries, independent of pricing and protocol encoding.

use serde::{Deserialize, Serialize};

use crate::types::{ReasoningEffort, ResponseFormat, Speed, ToolChoice};

/// Whether the catalog knows that a model supports a feature.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Deserialize, Serialize)]
#[serde(from = "SupportValue", into = "SupportValue")]
#[non_exhaustive]
pub enum Support {
    Supported,
    #[default]
    Unsupported,
    Unknown,
}

#[derive(Deserialize, Serialize)]
#[serde(untagged)]
enum SupportValue {
    Known(bool),
    Unknown(UnknownSupport),
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum UnknownSupport {
    Unknown,
}

impl From<SupportValue> for Support {
    fn from(value: SupportValue) -> Self {
        match value {
            SupportValue::Known(true) => Self::Supported,
            SupportValue::Known(false) => Self::Unsupported,
            SupportValue::Unknown(_) => Self::Unknown,
        }
    }
}
impl From<Support> for SupportValue {
    fn from(value: Support) -> Self {
        match value {
            Support::Supported => Self::Known(true),
            Support::Unsupported => Self::Known(false),
            Support::Unknown => Self::Unknown(UnknownSupport::Unknown),
        }
    }
}
impl Support {
    pub fn is_supported(self) -> bool {
        self == Self::Supported
    }
    pub fn is_unsupported(self) -> bool {
        self == Self::Unsupported
    }
}

/// Catalog support for individual toolchoice choices.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ToolChoiceSupport {
    pub required: Support,
    pub named:    Support,
}
impl Default for ToolChoiceSupport {
    fn default() -> Self {
        Self {
            required: Support::Unsupported,
            named:    Support::Unsupported,
        }
    }
}

/// Catalog support for individual responseformat choices.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ResponseFormatSupport {
    pub json_object: Support,
    pub json_schema: Support,
}
impl Default for ResponseFormatSupport {
    fn default() -> Self {
        Self {
            json_object: Support::Unsupported,
            json_schema: Support::Unsupported,
        }
    }
}

/// Catalog support for individual reasoningeffort choices.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReasoningEffortSupport {
    pub minimal: Support,
    pub low:     Support,
    pub medium:  Support,
    pub high:    Support,
    pub xhigh:   Support,
    pub max:     Support,
}
impl Default for ReasoningEffortSupport {
    fn default() -> Self {
        Self {
            minimal: Support::Unknown,
            low:     Support::Unknown,
            medium:  Support::Unknown,
            high:    Support::Unknown,
            xhigh:   Support::Unknown,
            max:     Support::Unknown,
        }
    }
}

/// Catalog support for individual speed choices.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct SpeedSupport {
    pub fast:       Support,
    pub balanced:   Support,
    pub economical: Support,
}
impl Default for SpeedSupport {
    fn default() -> Self {
        Self {
            fast:       Support::Unknown,
            balanced:   Support::Supported,
            economical: Support::Unknown,
        }
    }
}

/// Portable capability facts. Omitted basic features are unsupported.
/// Effort and speed choices remain unknown until declared by the catalog.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelCapabilities {
    text:             Support,
    images:           Support,
    audio:            Support,
    documents:        Support,
    tools:            Support,
    reasoning:        Support,
    caching:          Support,
    cache_routing:    Support,
    sampling:         Support,
    tool_choice:      ToolChoiceSupport,
    response_format:  ResponseFormatSupport,
    reasoning_effort: ReasoningEffortSupport,
    speed:            SpeedSupport,
}
impl ModelCapabilities {
    pub fn text(self) -> Support {
        self.text
    }
    pub fn images(self) -> Support {
        self.images
    }
    pub fn audio(self) -> Support {
        self.audio
    }
    pub fn documents(self) -> Support {
        self.documents
    }
    pub fn tools(self) -> Support {
        self.tools
    }
    pub fn reasoning(self) -> Support {
        self.reasoning
    }
    pub fn caching(self) -> Support {
        self.caching
    }
    pub fn cache_routing(self) -> Support {
        self.cache_routing
    }
    pub fn sampling(self) -> Support {
        self.sampling
    }
    pub fn tool_choice(self, choice: &ToolChoice) -> Support {
        if self.tools.is_unsupported() {
            return Support::Unsupported;
        }
        match choice {
            ToolChoice::Auto | ToolChoice::None => self.tools,
            ToolChoice::Required => self.tool_choice.required,
            ToolChoice::Tool { .. } => self.tool_choice.named,
        }
    }
    pub fn response_format(self, format: &ResponseFormat) -> Support {
        match format {
            ResponseFormat::Text => self.text,
            ResponseFormat::JsonObject => self.response_format.json_object,
            ResponseFormat::JsonSchema { .. } => self.response_format.json_schema,
        }
    }
    pub fn reasoning_effort(self, effort: ReasoningEffort) -> Support {
        if self.reasoning.is_unsupported() {
            return Support::Unsupported;
        }
        match effort {
            ReasoningEffort::Minimal => self.reasoning_effort.minimal,
            ReasoningEffort::Low => self.reasoning_effort.low,
            ReasoningEffort::Medium => self.reasoning_effort.medium,
            ReasoningEffort::High => self.reasoning_effort.high,
            ReasoningEffort::Xhigh => self.reasoning_effort.xhigh,
            ReasoningEffort::Max => self.reasoning_effort.max,
        }
    }
    pub fn speed(self, speed: Speed) -> Support {
        match speed {
            Speed::Fast => self.speed.fast,
            Speed::Balanced => self.speed.balanced,
            Speed::Economical => self.speed.economical,
        }
    }
    pub(crate) fn unknown() -> Self {
        Self {
            text:             Support::Unknown,
            images:           Support::Unknown,
            audio:            Support::Unknown,
            documents:        Support::Unknown,
            tools:            Support::Unknown,
            reasoning:        Support::Unknown,
            caching:          Support::Unknown,
            cache_routing:    Support::Unknown,
            sampling:         Support::Unknown,
            tool_choice:      ToolChoiceSupport {
                required: Support::Unknown,
                named:    Support::Unknown,
            },
            response_format:  ResponseFormatSupport {
                json_object: Support::Unknown,
                json_schema: Support::Unknown,
            },
            reasoning_effort: ReasoningEffortSupport::default(),
            speed:            SpeedSupport {
                balanced: Support::Unknown,
                ..SpeedSupport::default()
            },
        }
    }
}

/// Protocol encoding options, separate from portable model support.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelProtocolOptions {
    /// Encode effort names instead of converting effort to a token budget.
    pub reasoning_effort_levels: bool,
    /// Accept Anthropic-style cache breakpoints on a compatible endpoint.
    pub cache_breakpoints:       bool,
    /// Accept system-role turns after the leading system messages.
    pub system_turns:            bool,
}

#[cfg(test)]
mod tests {
    use super::{ModelCapabilities, Support};
    use crate::types::{ReasoningEffort, ResponseFormat, Speed, ToolChoice};

    #[test]
    fn queries_individual_choices_and_preserves_unknown() {
        let caps: ModelCapabilities = toml::from_str(
            r#"
            text = true
            images = "unknown"
            tools = true
            reasoning = true
            tool_choice = { required = true, named = false }
            response_format = { json_object = true, json_schema = false }
            reasoning_effort = { low = true, max = false }
            speed = { fast = true, economical = false }
        "#,
        )
        .expect("valid capabilities");
        assert_eq!(caps.images(), Support::Unknown);
        assert_eq!(caps.audio(), Support::Unsupported);
        assert_eq!(caps.tool_choice(&ToolChoice::Required), Support::Supported);
        assert_eq!(
            caps.tool_choice(&ToolChoice::Tool {
                name: "read".into(),
            }),
            Support::Unsupported
        );
        assert_eq!(
            caps.response_format(&ResponseFormat::JsonObject),
            Support::Supported
        );
        assert_eq!(
            caps.reasoning_effort(ReasoningEffort::Low),
            Support::Supported
        );
        assert_eq!(
            caps.reasoning_effort(ReasoningEffort::Max),
            Support::Unsupported
        );
        assert_eq!(
            caps.reasoning_effort(ReasoningEffort::Medium),
            Support::Unknown
        );
        assert_eq!(caps.speed(Speed::Fast), Support::Supported);
        assert_eq!(caps.speed(Speed::Economical), Support::Unsupported);
        let encoded = toml::to_string(&caps).expect("serializable");
        assert_eq!(
            toml::from_str::<ModelCapabilities>(&encoded).expect("round trip"),
            caps
        );
    }

    #[test]
    fn passthrough_capabilities_are_unknown_including_text_and_balanced_speed() {
        let caps = ModelCapabilities::unknown();
        assert_eq!(caps.text(), Support::Unknown);
        assert_eq!(caps.tools(), Support::Unknown);
        assert_eq!(caps.tool_choice(&ToolChoice::Required), Support::Unknown);
        assert_eq!(caps.speed(Speed::Balanced), Support::Unknown);
    }
}
