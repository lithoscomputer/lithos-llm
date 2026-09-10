//! Readable model reasoning, normalized out of provider-shaped content.
//!
//! Every provider that returns readable reasoning does it differently, and
//! several return more than one channel at once: Anthropic and Gemini send
//! [`ReasoningContent`] blocks, the OpenAI Responses protocol sends
//! `reasoning` items with a `summary` and sometimes `content`, and
//! OpenAI-compatible gateways send `reasoning_details` arrays. This module
//! reduces a response's content to two fields, a summary and a verbatim trace,
//! without reading opaque material (signatures, item ids, encrypted payloads)
//! and without failing a response it cannot classify.
//!
//! Parsing is tolerant on purpose: provider payloads are read as
//! `serde_json::Value` with optional lookups, so unknown detail variants,
//! missing members, extra members, and unexpected member types are ignored
//! rather than surfaced as errors.

use serde::{Deserialize, Serialize, de};
use serde_json::Value;

use super::{ContentPart, ReasoningContent, Response};

/// OpenAI Responses reasoning items, as the codec stores them.
pub const OPENAI_REASONING_KIND: &str = "openai.reasoning";
/// OpenAI Responses message items, as the codec stores them.
pub const OPENAI_MESSAGE_KIND: &str = "openai.message";
/// OpenAI-compatible `reasoning_details` arrays, as the codec stores them.
pub const OPENAI_COMPAT_REASONING_DETAILS_KIND: &str = "openai_compatible.reasoning_details";

/// Separator between distinct complete reasoning blocks.
const BLOCK_SEPARATOR: &str = "\n\n";

/// Readable model reasoning in a provider-neutral shape.
///
/// Both fields may be present for one response. A value always carries at
/// least one of them: the constructors cannot build an empty one and
/// deserialization rejects an object with neither. Absent fields are omitted
/// when serializing, so the wire form is `{"summary": …}`, `{"trace": …}`, or
/// both. Opaque provider material never appears here.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ReasoningOutput {
    /// The model-authored summary of its reasoning, safe to show to users.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    summary: Option<String>,
    /// Verbatim readable reasoning text, when the provider returns it in
    /// addition to, or instead of, a summary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    trace:   Option<String>,
}

impl ReasoningOutput {
    /// Reasoning with both a summary and a verbatim trace.
    pub fn new(summary: impl Into<String>, trace: impl Into<String>) -> Self {
        Self {
            summary: Some(summary.into()),
            trace:   Some(trace.into()),
        }
    }

    /// Reasoning with only a model-authored summary.
    pub fn from_summary(summary: impl Into<String>) -> Self {
        Self {
            summary: Some(summary.into()),
            trace:   None,
        }
    }

    /// Reasoning with only a verbatim trace.
    pub fn from_trace(trace: impl Into<String>) -> Self {
        Self {
            summary: None,
            trace:   Some(trace.into()),
        }
    }

    /// Normalizes the content of a final response.
    ///
    /// Returns `None` when the content carries no readable reasoning.
    /// Structured channels win: an explicit summary or trace from an OpenAI
    /// reasoning item or a `reasoning_details` entry is used as is, and a
    /// flattened [`ReasoningContent`] block only fills a trace no structured
    /// channel produced. A flattened block equal to the summary is dropped,
    /// because gateways commonly duplicate the summary there.
    pub fn from_content(content: &[ContentPart]) -> Option<Self> {
        let mut blocks = Blocks::default();
        for part in content {
            match part {
                ContentPart::Reasoning(ReasoningContent {
                    text,
                    redacted: false,
                    ..
                }) => {
                    push_block(&mut blocks.fallback_trace, text);
                }
                ContentPart::Opaque { kind, data } if kind == OPENAI_REASONING_KIND => {
                    collect_openai_reasoning_item(data, &mut blocks);
                }
                ContentPart::Opaque { kind, data }
                    if kind == OPENAI_COMPAT_REASONING_DETAILS_KIND =>
                {
                    collect_reasoning_details(data, &mut blocks);
                }
                _ => {}
            }
        }
        blocks.into_output()
    }

    /// The model-authored summary, when present.
    pub fn summary(&self) -> Option<&str> {
        self.summary.as_deref()
    }

    /// The verbatim readable trace, when present.
    pub fn trace(&self) -> Option<&str> {
        self.trace.as_deref()
    }
}

impl<'de> Deserialize<'de> for ReasoningOutput {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Fields {
            #[serde(default)]
            summary: Option<String>,
            #[serde(default)]
            trace:   Option<String>,
        }

        let Fields { summary, trace } = Fields::deserialize(deserializer)?;
        match (summary, trace) {
            (Some(summary), Some(trace)) => Ok(Self::new(summary, trace)),
            (Some(summary), None) => Ok(Self::from_summary(summary)),
            (None, Some(trace)) => Ok(Self::from_trace(trace)),
            (None, None) => Err(de::Error::custom(
                "reasoning output requires a summary or trace",
            )),
        }
    }
}

impl Response {
    /// The readable reasoning in this response, normalized across providers.
    ///
    /// See [`ReasoningOutput::from_content`].
    pub fn reasoning(&self) -> Option<ReasoningOutput> {
        ReasoningOutput::from_content(&self.content)
    }
}

impl ContentPart {
    /// Whether this part is provider material a conversation keeps so the
    /// next request can replay it, as opposed to content a person reads.
    ///
    /// That is every [`Reasoning`](Self::Reasoning) block and every
    /// [`Opaque`](Self::Opaque) item, `openai.message` included: an OpenAI
    /// reasoning item and the message item it precedes must stay paired until
    /// compaction discards both.
    pub fn is_replay_material(&self) -> bool {
        matches!(self, Self::Reasoning(_) | Self::Opaque { .. })
    }

    /// Whether this part is an OpenAI Responses item tied to one specific API
    /// response.
    ///
    /// Such items become invalid once compaction replaces their surrounding
    /// context, and a codec that replays a reasoning item without its message
    /// item violates the protocol.
    pub fn is_opaque_openai(&self) -> bool {
        matches!(
            self,
            Self::Opaque { kind, .. }
                if kind == OPENAI_REASONING_KIND || kind == OPENAI_MESSAGE_KIND
        )
    }
}

/// Readable blocks collected per normalized field.
///
/// Explicit blocks come from a channel with documented reasoning semantics.
/// Fallback blocks come from flattened provider reasoning text, which
/// gateways commonly duplicate beside a structured channel. They only fill a
/// trace that no explicit trace produced.
#[derive(Default)]
struct Blocks<'a> {
    explicit_summary: Vec<&'a str>,
    explicit_trace:   Vec<&'a str>,
    fallback_trace:   Vec<&'a str>,
}

impl Blocks<'_> {
    fn into_output(self) -> Option<ReasoningOutput> {
        let summary = join_blocks(&self.explicit_summary);
        let trace = join_blocks(&self.explicit_trace)
            .or_else(|| join_blocks(&self.fallback_trace))
            .filter(|trace| summary.as_ref() != Some(trace));

        match (summary, trace) {
            (Some(summary), Some(trace)) => Some(ReasoningOutput::new(summary, trace)),
            (Some(summary), None) => Some(ReasoningOutput::from_summary(summary)),
            (None, Some(trace)) => Some(ReasoningOutput::from_trace(trace)),
            (None, None) => None,
        }
    }
}

/// Joins complete blocks in provider order. Text is never trimmed or
/// rewritten.
fn join_blocks(blocks: &[&str]) -> Option<String> {
    (!blocks.is_empty()).then(|| blocks.join(BLOCK_SEPARATOR))
}

fn push_block<'a>(blocks: &mut Vec<&'a str>, block: &'a str) {
    if !block.trim().is_empty() {
        blocks.push(block);
    }
}

fn readable_member<'a>(entry: &'a Value, member: &str) -> Option<&'a str> {
    entry.get(member).and_then(Value::as_str)
}

/// Reads an OpenAI Responses `reasoning` output item.
///
/// `summary[].text` is the model-authored summary; `content[]` entries typed
/// `reasoning_text` are the verbatim trace. `encrypted_content`, `id`, and
/// `status` are opaque and ignored.
fn collect_openai_reasoning_item<'a>(item: &'a Value, blocks: &mut Blocks<'a>) {
    if let Some(entries) = item.get("summary").and_then(Value::as_array) {
        for entry in entries {
            if let Some(text) = entry.as_str() {
                push_block(&mut blocks.explicit_summary, text);
            } else if let Some(text) = readable_member(entry, "text") {
                push_block(&mut blocks.explicit_summary, text);
            }
        }
    }
    if let Some(entries) = item.get("content").and_then(Value::as_array) {
        for entry in entries {
            let Some(text) = readable_member(entry, "text") else {
                continue;
            };
            if readable_member(entry, "type").unwrap_or_default() == "reasoning_text" {
                push_block(&mut blocks.explicit_trace, text);
            }
        }
    }
}

/// Reads OpenAI-compatible `reasoning_details` entries.
fn collect_reasoning_details<'a>(details: &'a Value, blocks: &mut Blocks<'a>) {
    let Some(entries) = details.as_array() else {
        return;
    };
    for entry in entries {
        match readable_member(entry, "type").unwrap_or_default() {
            "reasoning.text" => {
                if let Some(text) = readable_member(entry, "text") {
                    push_block(&mut blocks.explicit_trace, text);
                }
            }
            "reasoning.summary" => {
                if let Some(text) = readable_member(entry, "summary") {
                    push_block(&mut blocks.explicit_summary, text);
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        OPENAI_COMPAT_REASONING_DETAILS_KIND, OPENAI_MESSAGE_KIND, OPENAI_REASONING_KIND,
        ReasoningOutput,
    };
    use crate::catalog::{ModelId, ProviderId};
    use crate::types::{ContentPart, ReasoningContent, Response};

    fn reasoning(text: &str) -> ContentPart {
        ContentPart::Reasoning(ReasoningContent {
            text:             text.to_owned(),
            signature:        None,
            signature_origin: None,
            redacted:         false,
        })
    }

    fn openai_reasoning(item: serde_json::Value) -> ContentPart {
        ContentPart::opaque(OPENAI_REASONING_KIND, item)
    }

    fn reasoning_details(details: serde_json::Value) -> ContentPart {
        ContentPart::opaque(OPENAI_COMPAT_REASONING_DETAILS_KIND, details)
    }

    fn normalize(content: &[ContentPart]) -> Option<ReasoningOutput> {
        ReasoningOutput::from_content(content)
    }

    // --- Wire shape ---

    #[test]
    fn each_shape_round_trips_and_omits_absent_fields() {
        for (output, expected) in [
            (
                ReasoningOutput::from_summary("checked the parser first"),
                json!({"summary": "checked the parser first"}),
            ),
            (
                ReasoningOutput::from_trace("step one, step two"),
                json!({"trace": "step one, step two"}),
            ),
            (
                ReasoningOutput::new("summary", "trace"),
                json!({"summary": "summary", "trace": "trace"}),
            ),
        ] {
            let value = serde_json::to_value(&output).expect("serializes");
            assert_eq!(value, expected);
            assert_eq!(
                serde_json::from_value::<ReasoningOutput>(value).expect("parses"),
                output
            );
        }
    }

    #[test]
    fn an_empty_object_and_explicit_nulls_are_rejected() {
        for value in [json!({}), json!({"summary": null, "trace": null})] {
            let error = serde_json::from_value::<ReasoningOutput>(value)
                .expect_err("no reasoning is not reasoning");
            assert!(error.to_string().contains("requires a summary or trace"));
        }
    }

    // --- Normalization ---

    #[test]
    fn non_redacted_reasoning_becomes_a_trace() {
        let output = normalize(&[reasoning("weighing the options")]).expect("readable");
        assert!(output.summary().is_none());
        assert_eq!(output.trace(), Some("weighing the options"));
    }

    #[test]
    fn redacted_reasoning_yields_nothing_readable() {
        let redacted = ContentPart::Reasoning(ReasoningContent {
            text:             "AAAAopaque".to_owned(),
            signature:        Some("sig".to_owned()),
            signature_origin: Some("anthropic".to_owned()),
            redacted:         true,
        });
        assert!(normalize(&[redacted]).is_none());
    }

    #[test]
    fn a_responses_item_with_summary_and_reasoning_text_produces_both() {
        let output = normalize(&[openai_reasoning(json!({
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "gAAAAA",
            "summary": [{"type": "summary_text", "text": "inspect first"}],
            "content": [{"type": "reasoning_text", "text": "step one"}],
        }))])
        .expect("readable");
        assert_eq!(output.summary(), Some("inspect first"));
        assert_eq!(output.trace(), Some("step one"));
    }

    #[test]
    fn responses_blocks_join_in_provider_order() {
        let output = normalize(&[openai_reasoning(json!({
            "summary": [
                {"type": "summary_text", "text": "first"},
                {"type": "summary_text", "text": "second"},
            ],
        }))])
        .expect("readable");
        assert_eq!(output.summary(), Some("first\n\nsecond"));
    }

    #[test]
    fn a_flattened_copy_of_the_responses_summary_is_deduplicated() {
        // What the OpenAI Responses codec decodes for a `reasoning` item with
        // two `summary_text` blocks and no `content[]`: a `Reasoning` part
        // holding the summaries joined by a blank line, then the opaque item.
        // The fallback matches the joined summary exactly and never becomes a
        // trace.
        let output = normalize(&[
            reasoning("A\n\nB"),
            openai_reasoning(json!({
                "summary": [
                    {"type": "summary_text", "text": "A"},
                    {"type": "summary_text", "text": "B"},
                ],
            })),
        ])
        .expect("readable");
        assert_eq!(output, ReasoningOutput::from_summary("A\n\nB"));
    }

    #[test]
    fn a_flattened_copy_joined_differently_survives_as_a_trace() {
        // Deduplication is by equality alone.
        let output = normalize(&[
            reasoning("AB"),
            openai_reasoning(json!({
                "summary": [
                    {"type": "summary_text", "text": "A"},
                    {"type": "summary_text", "text": "B"},
                ],
            })),
        ])
        .expect("readable");
        assert_eq!(output, ReasoningOutput::new("A\n\nB", "AB"));
    }

    #[test]
    fn a_responses_trace_is_not_duplicated_by_its_flattened_part() {
        let output = normalize(&[
            reasoning("step one"),
            openai_reasoning(json!({
                "summary": [{"type": "summary_text", "text": "A"}],
                "content": [{"type": "reasoning_text", "text": "step one"}],
            })),
        ])
        .expect("readable");
        assert_eq!(output, ReasoningOutput::new("A", "step one"));
    }

    #[test]
    fn unknown_responses_content_types_stay_opaque() {
        assert!(
            normalize(&[openai_reasoning(json!({
                "content": [{"type": "reasoning_future", "text": "not classified"}],
            }))])
            .is_none()
        );
    }

    #[test]
    fn structured_details_produce_summary_and_trace() {
        let output = normalize(&[reasoning_details(json!([
            {"type": "reasoning.summary", "summary": "checked the parser"},
            {"type": "reasoning.text", "text": "read convert.rs", "signature": "sig"},
            {"type": "reasoning.encrypted", "data": "gAAAAAsecret"},
        ]))])
        .expect("readable");
        assert_eq!(output.summary(), Some("checked the parser"));
        assert_eq!(output.trace(), Some("read convert.rs"));
    }

    #[test]
    fn encrypted_or_unknown_details_alone_produce_nothing() {
        assert!(
            normalize(&[reasoning_details(json!([
                {"type": "reasoning.encrypted", "data": "gAAAAAsecret"},
            ]))])
            .is_none()
        );
        assert!(
            normalize(&[reasoning_details(json!([
                {"type": "reasoning.future", "text": "new channel"},
            ]))])
            .is_none()
        );
    }

    #[test]
    fn malformed_details_are_ignored_without_failing() {
        assert!(normalize(&[reasoning_details(json!("not-an-array"))]).is_none());
        assert!(
            normalize(&[reasoning_details(json!([
                42,
                {"type": "reasoning.summary", "summary": 7},
                {"no_type": true},
            ]))])
            .is_none()
        );
    }

    #[test]
    fn structured_channels_win_over_flattened_text() {
        let output = normalize(&[
            reasoning_details(json!([
                {"type": "reasoning.summary", "summary": "checked the parser"},
            ])),
            reasoning("checked the parser"),
        ])
        .expect("readable");
        assert_eq!(output.summary(), Some("checked the parser"));
        assert!(
            output.trace().is_none(),
            "a duplicate of the summary is dropped"
        );

        let output = normalize(&[
            reasoning_details(json!([{"type": "reasoning.text", "text": "verbatim"}])),
            reasoning("flattened"),
        ])
        .expect("readable");
        assert!(output.summary().is_none());
        assert_eq!(output.trace(), Some("verbatim"));

        let output = normalize(&[
            reasoning_details(json!([
                {"type": "reasoning.summary", "summary": "short summary"},
            ])),
            reasoning("full verbatim trace"),
        ])
        .expect("readable");
        assert_eq!(output.summary(), Some("short summary"));
        assert_eq!(output.trace(), Some("full verbatim trace"));
    }

    #[test]
    fn whitespace_only_text_is_nothing_and_other_text_is_verbatim() {
        assert!(normalize(&[reasoning("   \n ")]).is_none());
        let output = normalize(&[reasoning("  indented thought\n")]).expect("readable");
        assert_eq!(output.trace(), Some("  indented thought\n"));
    }

    #[test]
    fn unrelated_parts_are_ignored_and_the_response_reads_the_same() {
        let parts = vec![
            ContentPart::Text {
                text: "answer".to_owned(),
            },
            ContentPart::opaque(
                OPENAI_MESSAGE_KIND,
                json!({"type": "message", "content": [{"text": "answer"}]}),
            ),
        ];
        assert!(normalize(&parts).is_none());
        let response = Response::new(ProviderId::new("openai"), ModelId::new("m"), parts);
        assert!(response.reasoning().is_none());
    }

    // --- Replay predicates ---

    #[test]
    fn replay_material_is_every_reasoning_and_opaque_part() {
        let message = ContentPart::opaque(OPENAI_MESSAGE_KIND, json!({}));
        assert!(reasoning("x").is_replay_material());
        assert!(openai_reasoning(json!({})).is_replay_material());
        assert!(
            message.is_replay_material(),
            "message items stay paired with reasoning"
        );
        assert!(reasoning_details(json!([])).is_replay_material());
        assert!(
            !ContentPart::Text {
                text: "x".to_owned(),
            }
            .is_replay_material()
        );

        assert!(openai_reasoning(json!({})).is_opaque_openai());
        assert!(message.is_opaque_openai());
        assert!(!reasoning_details(json!([])).is_opaque_openai());
        assert!(!reasoning("x").is_opaque_openai());
    }
}
