//! Model support queries, independent of pricing and protocol encoding.

use std::cmp::Reverse;

use serde::{Deserialize, Serialize};

use crate::evaluation::QuestionKind;
use crate::types::{ContentPart, ReasoningEffort, ResponseFormat, Speed, ToolChoice};

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

/// Catalog support for each evaluation question kind.
///
/// Every field defaults to [`Support::Unknown`], which means "derive it":
/// [`ModelCapabilities::evaluation`] then answers from the row's JSON Schema
/// support, because any model that produces schema-bound JSON can act as a
/// judge. A row writes a field explicitly to narrow that claim or to claim
/// evaluation natively.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct EvaluationSupport {
    pub choice:  Support,
    pub score:   Support,
    pub boolean: Support,
}
impl Default for EvaluationSupport {
    fn default() -> Self {
        Self {
            choice:  Support::Unknown,
            score:   Support::Unknown,
            boolean: Support::Unknown,
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
    evaluation:       EvaluationSupport,
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

    /// Whether the model takes a content part of this kind.
    ///
    /// Text, images, audio, and documents map onto their own claims. A
    /// reasoning part needs the `reasoning` claim, and a tool call or tool
    /// result needs `tools`. Structured JSON, an opaque replay part, and an
    /// unknown part make no catalog claim, so they answer `Unknown`; the
    /// resolver refuses unknown content separately, and a codec decides what
    /// it can carry of the rest.
    pub fn content_part(self, part: &ContentPart) -> Support {
        match part {
            ContentPart::Text { .. } => self.text,
            ContentPart::Image(_) => self.images,
            ContentPart::Audio(_) => self.audio,
            ContentPart::Document(_) => self.documents,
            ContentPart::Reasoning(_) => self.reasoning,
            ContentPart::ToolCall(_) | ContentPart::ToolResult(_) => self.tools,
            ContentPart::Json { .. } | ContentPart::Opaque { .. } | ContentPart::Unknown(_) => {
                Support::Unknown
            }
        }
    }

    /// Whether the model answers evaluation questions of `kind`.
    ///
    /// The row's explicit `evaluation` claim wins when it has one for this
    /// kind. Otherwise the answer is the row's JSON Schema support: a model
    /// that produces schema-bound JSON judges every kind through structured
    /// output, so a structured-output row evaluates without saying so.
    pub fn evaluation(self, kind: QuestionKind) -> Support {
        let explicit = match kind {
            QuestionKind::Choice => self.evaluation.choice,
            QuestionKind::Score => self.evaluation.score,
            QuestionKind::Boolean => self.evaluation.boolean,
        };
        if explicit == Support::Unknown {
            self.response_format.json_schema
        } else {
            explicit
        }
    }

    /// Whether the model answers any evaluation question at all.
    ///
    /// `Supported` when at least one kind is supported, `Unsupported` when
    /// every kind is unsupported, and `Unknown` otherwise.
    pub fn evaluates(self) -> Support {
        let kinds = [
            QuestionKind::Choice,
            QuestionKind::Score,
            QuestionKind::Boolean,
        ]
        .map(|kind| self.evaluation(kind));
        if kinds.iter().any(|support| support.is_supported()) {
            Support::Supported
        } else if kinds.iter().all(|support| support.is_unsupported()) {
            Support::Unsupported
        } else {
            Support::Unknown
        }
    }

    /// The `evaluation` claim exactly as the row wrote it, with `Unknown`
    /// for every kind the row left to derivation.
    ///
    /// [`evaluation`](Self::evaluation) hides whether a claim was written,
    /// and codec derivation needs that fact: only a row that writes the
    /// claim reaches a native evaluation codec.
    pub(crate) fn explicit_evaluation(self) -> EvaluationSupport {
        self.evaluation
    }

    /// Whether the row wrote `evaluation` with at least one kind `true`.
    pub(crate) fn claims_native_evaluation(self) -> bool {
        let explicit = self.explicit_evaluation();
        [explicit.choice, explicit.score, explicit.boolean]
            .iter()
            .any(|support| support.is_supported())
    }

    /// Whether any generation claim (`text`, `tools`, `images`, `audio`,
    /// `documents`, or a `response_format` field) is not `false`.
    ///
    /// `Unknown` counts as a claim: an unstated feature is left to the
    /// provider to accept or reject, not refused before dispatch.
    pub(crate) fn claims_generation(self) -> bool {
        [
            self.text,
            self.tools,
            self.images,
            self.audio,
            self.documents,
            self.response_format.json_object,
            self.response_format.json_schema,
        ]
        .iter()
        .any(|support| !support.is_unsupported())
    }

    /// The supported reasoning effort nearest to `requested`.
    ///
    /// A request written for one model often names an effort the fallback
    /// model does not offer; this picks the level to send instead. Distance
    /// is counted in levels along [`ReasoningEffort::ALL`], and when two
    /// supported levels are equally near, the higher one wins. `None` when no
    /// level is known to be supported.
    pub fn closest_supported_effort(self, requested: ReasoningEffort) -> Option<ReasoningEffort> {
        let rank = |effort: ReasoningEffort| {
            ReasoningEffort::ALL
                .iter()
                .position(|candidate| *candidate == effort)
                .unwrap_or(ReasoningEffort::ALL.len())
        };
        let target = rank(requested);
        ReasoningEffort::ALL
            .into_iter()
            .filter(|effort| self.reasoning_effort(*effort).is_supported())
            .min_by_key(|effort| {
                let position = rank(*effort);
                (position.abs_diff(target), Reverse(position))
            })
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
            evaluation:       EvaluationSupport::default(),
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
    use serde_json::json;

    use super::{ModelCapabilities, Support};
    use crate::evaluation::QuestionKind;
    use crate::types::{
        ContentPart, ImageContent, MediaSource, ReasoningEffort, ResponseFormat, Speed, ToolChoice,
        ToolResult,
    };

    #[test]
    fn content_parts_map_onto_their_claims_and_the_rest_are_unknown() {
        let capabilities: ModelCapabilities =
            toml::from_str("text = true\nimages = false\ntools = true").expect("a valid row");

        assert_eq!(
            capabilities.content_part(&ContentPart::Text {
                text: "hi".to_owned(),
            }),
            Support::Supported
        );
        assert_eq!(
            capabilities.content_part(&ContentPart::Image(ImageContent::new(MediaSource::url(
                "https://x/y.png"
            )))),
            Support::Unsupported
        );
        // Tool results ride on the `tools` claim, not a claim of their own.
        assert_eq!(
            capabilities.content_part(&ContentPart::ToolResult(ToolResult {
                tool_call_id: "c".to_owned(),
                name:         None,
                content:      Vec::new(),
                is_error:     false,
            })),
            Support::Supported
        );
        // No claim exists for structured JSON or an opaque replay part.
        assert_eq!(
            capabilities.content_part(&ContentPart::Json { value: json!(1) }),
            Support::Unknown
        );
        assert_eq!(
            capabilities.content_part(&ContentPart::opaque("openai.reasoning", json!({}))),
            Support::Unknown
        );
    }

    const KINDS: [QuestionKind; 3] = [
        QuestionKind::Choice,
        QuestionKind::Score,
        QuestionKind::Boolean,
    ];

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

    #[test]
    fn a_structured_output_row_evaluates_every_kind_without_saying_so() {
        let caps: ModelCapabilities =
            toml::from_str("text = true\nresponse_format = { json_schema = true }")
                .expect("valid capabilities");
        for kind in KINDS {
            assert_eq!(caps.evaluation(kind), Support::Supported, "{kind:?}");
        }
        assert_eq!(caps.evaluates(), Support::Supported);

        let plain: ModelCapabilities = toml::from_str("text = true").expect("valid capabilities");
        for kind in KINDS {
            assert_eq!(plain.evaluation(kind), Support::Unsupported, "{kind:?}");
        }
        assert_eq!(plain.evaluates(), Support::Unsupported);
    }

    #[test]
    fn an_explicit_evaluation_claim_narrows_or_overrides_the_derived_one() {
        let narrowed: ModelCapabilities = toml::from_str(
            "response_format = { json_schema = true }\nevaluation = { score = false }",
        )
        .expect("valid capabilities");
        assert_eq!(
            narrowed.evaluation(QuestionKind::Score),
            Support::Unsupported
        );
        assert_eq!(
            narrowed.evaluation(QuestionKind::Choice),
            Support::Supported
        );
        assert_eq!(
            narrowed.evaluation(QuestionKind::Boolean),
            Support::Supported
        );
        assert_eq!(narrowed.evaluates(), Support::Supported);

        let native: ModelCapabilities = toml::from_str(
            "response_format = { json_schema = false }\nevaluation = { choice = true, score = true, boolean = true }",
        )
        .expect("valid capabilities");
        for kind in KINDS {
            assert_eq!(native.evaluation(kind), Support::Supported, "{kind:?}");
        }
    }

    #[test]
    fn evaluation_stays_unknown_when_nothing_is_claimed() {
        let caps = ModelCapabilities::unknown();
        for kind in KINDS {
            assert_eq!(caps.evaluation(kind), Support::Unknown, "{kind:?}");
        }
        assert_eq!(caps.evaluates(), Support::Unknown);

        // One kind denied, the rest unknown: the whole is still unknown.
        let partial: ModelCapabilities = toml::from_str(
            "response_format = { json_schema = \"unknown\" }\nevaluation = { score = false }",
        )
        .expect("valid capabilities");
        assert_eq!(partial.evaluates(), Support::Unknown);
    }

    #[test]
    fn evaluation_round_trips_and_rejects_a_misspelled_key() {
        let caps: ModelCapabilities =
            toml::from_str("evaluation = { choice = true, score = false, boolean = \"unknown\" }")
                .expect("valid capabilities");
        let encoded = toml::to_string(&caps).expect("serializable");
        assert_eq!(
            toml::from_str::<ModelCapabilities>(&encoded).expect("round trip"),
            caps
        );

        let error = toml::from_str::<ModelCapabilities>("evaluation = { scores = true }")
            .expect_err("an unknown evaluation key is rejected");
        assert!(error.to_string().contains("scores"), "{error}");
    }

    #[test]
    fn closest_supported_effort_prefers_the_higher_neighbor_on_ties() {
        let capabilities: ModelCapabilities = toml::from_str(
            r"reasoning = true
reasoning_effort = { low = true, high = true }",
        )
        .expect("parses");
        assert_eq!(
            capabilities.closest_supported_effort(ReasoningEffort::Medium),
            Some(ReasoningEffort::High)
        );
        assert_eq!(
            capabilities.closest_supported_effort(ReasoningEffort::Max),
            Some(ReasoningEffort::High)
        );
        assert_eq!(
            capabilities.closest_supported_effort(ReasoningEffort::Minimal),
            Some(ReasoningEffort::Low)
        );
        assert_eq!(
            capabilities.closest_supported_effort(ReasoningEffort::Low),
            Some(ReasoningEffort::Low)
        );
        let none: ModelCapabilities = toml::from_str("text = true").expect("parses");
        assert_eq!(none.closest_supported_effort(ReasoningEffort::Medium), None);
    }
}
