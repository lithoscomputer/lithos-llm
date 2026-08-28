use std::pin::Pin;

use futures_core::Stream;
use serde::{Deserialize, Serialize};

use super::{ContentPart, Error, RateLimits, Response, TokenCounts, ToolCallKind};

/// A stable identifier for one content block within one response stream.
///
/// Block ids are unique inside a single stream and stay the same across every
/// event for that block. They are stream-local: two streams may reuse the same
/// text, and an id carries no meaning once its stream has completed.
///
/// A provider tool-call id is a different thing and is never used as a block
/// id. It stays in [`ContentBlockKind::ToolCall`] and in the final
/// [`ToolCall`](super::ToolCall).
#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct ContentBlockId(String);

impl ContentBlockId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// What kind of content one streamed block carries.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ContentBlockKind {
    /// Visible assistant text, delivered as [`StreamEvent::TextDelta`].
    Text,
    /// Model reasoning, delivered as [`StreamEvent::ReasoningDelta`].
    Reasoning,
    /// A tool call whose arguments arrive as [`StreamEvent::ToolCallDelta`].
    ToolCall {
        /// The provider's tool-call id, which is not the block id.
        id:   String,
        /// The tool name, when the provider supplies it before the first delta.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(default)]
        kind: ToolCallKind,
    },
    /// Provider-native content kept verbatim, with a namespaced kind such as
    /// `"openai.reasoning"`.
    Opaque { kind: String },
}

/// One normalized event from a streaming response.
///
/// # Invariants
///
/// Every codec upholds the following, so consumers can rely on them:
///
/// - Each content block produces exactly one [`ContentBlockStart`], then zero
///   or more matching deltas, then exactly one [`ContentBlockEnd`].
/// - A [`ContentBlockId`] is unique within one stream and stays the same across
///   every event for its block.
/// - Provider tool-call ids live in [`ContentBlockKind::ToolCall`] and in the
///   final [`ToolCall`](super::ToolCall). They are never reused as block ids.
/// - [`ContentBlockEnd`] carries the fully assembled part: complete text, the
///   accumulated reasoning signature and redaction state, parsed tool arguments
///   with the provider's original argument string, and provider metadata.
/// - [`Usage`] events are cumulative snapshots for the current provider call,
///   not deltas. A codec that receives incremental counters accumulates them.
/// - Every successful stream ends with exactly one [`Completed`] event whose
///   response content is the ordered sequence of [`ContentBlockEnd`] parts.
/// - A stream failure is an `Err(Error)` item and produces no [`Completed`]
///   event.
///
/// [`ContentBlockStart`]: StreamEvent::ContentBlockStart
/// [`ContentBlockEnd`]: StreamEvent::ContentBlockEnd
/// [`Usage`]: StreamEvent::Usage
/// [`Completed`]: StreamEvent::Completed
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum StreamEvent {
    /// The provider accepted the call and began the response.
    Started {
        id: Option<String>,
    },
    /// A new content block opened.
    ContentBlockStart {
        id:   ContentBlockId,
        kind: ContentBlockKind,
    },
    /// More visible text for an open text block.
    TextDelta {
        id:   ContentBlockId,
        text: String,
    },
    /// More reasoning text for an open reasoning block.
    ReasoningDelta {
        id:   ContentBlockId,
        text: String,
    },
    /// More raw argument text for an open tool-call block.
    ToolCallDelta {
        id:        ContentBlockId,
        arguments: String,
    },
    /// A content block closed, carrying its fully assembled part.
    ContentBlockEnd {
        id:   ContentBlockId,
        part: ContentPart,
    },
    /// A cumulative usage snapshot for the current provider call.
    Usage {
        usage: TokenCounts,
    },
    RateLimits {
        rate_limits: RateLimits,
    },
    /// The single terminal event of a successful stream.
    Completed {
        response: Response,
    },
}

impl StreamEvent {
    /// Whether this event carries content or accounting a user-facing consumer
    /// normally shows.
    ///
    /// Protocol bookkeeping — [`Started`](Self::Started),
    /// [`RateLimits`](Self::RateLimits), and
    /// [`ContentBlockStart`](Self::ContentBlockStart) — is not visible.
    pub fn is_visible(&self) -> bool {
        !matches!(
            self,
            Self::Started { .. } | Self::RateLimits { .. } | Self::ContentBlockStart { .. }
        )
    }
}

/// A cancel-on-drop stream of normalized provider events.
pub type ResponseStream = Pin<Box<dyn Stream<Item = Result<StreamEvent, Error>> + Send + 'static>>;

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use serde_json::json;

    use super::{ContentBlockId, ContentBlockKind, StreamEvent};
    use crate::catalog::{ModelId, ProviderId};
    use crate::types::{ContentPart, RateLimits, Response, TokenCounts, ToolCall, ToolCallKind};

    fn every_variant() -> Vec<StreamEvent> {
        let id = ContentBlockId::new("block-0");
        let tool = ContentBlockId::new("tool-1");

        vec![
            StreamEvent::Started {
                id: Some("resp_1".to_owned()),
            },
            StreamEvent::ContentBlockStart {
                id:   id.clone(),
                kind: ContentBlockKind::Text,
            },
            StreamEvent::ContentBlockStart {
                id:   tool.clone(),
                kind: ContentBlockKind::ToolCall {
                    id:   "call_abc".to_owned(),
                    name: Some("lookup".to_owned()),
                    kind: ToolCallKind::Function,
                },
            },
            StreamEvent::ContentBlockStart {
                id:   ContentBlockId::new("opaque-2"),
                kind: ContentBlockKind::Opaque {
                    kind: "openai.reasoning".to_owned(),
                },
            },
            StreamEvent::TextDelta {
                id:   id.clone(),
                text: "hello".to_owned(),
            },
            StreamEvent::ReasoningDelta {
                id:   ContentBlockId::new("reasoning-3"),
                text: "thinking".to_owned(),
            },
            StreamEvent::ToolCallDelta {
                id:        tool.clone(),
                arguments: "{\"q\":".to_owned(),
            },
            StreamEvent::ContentBlockEnd {
                id:   tool,
                part: ContentPart::ToolCall(ToolCall::function(
                    "call_abc",
                    "lookup",
                    json!({ "q": "rust" }),
                )),
            },
            StreamEvent::Usage {
                usage: TokenCounts::from_inclusive(100, 60, 25, 30, 10),
            },
            StreamEvent::RateLimits {
                rate_limits: RateLimits {
                    request_reset: Some("1s".to_owned()),
                    ..RateLimits::default()
                },
            },
            StreamEvent::Completed {
                response: Response::new(ProviderId::new("openai"), ModelId::new("gpt-5"), vec![
                    ContentPart::Text {
                        text: "hello".to_owned(),
                    },
                ]),
            },
        ]
    }

    #[test]
    fn every_variant_round_trips() -> Result<(), Box<dyn StdError>> {
        for event in every_variant() {
            let encoded = serde_json::to_value(&event)?;

            assert_eq!(serde_json::from_value::<StreamEvent>(encoded)?, event);
        }
        Ok(())
    }

    #[test]
    fn only_protocol_events_are_hidden() {
        for event in every_variant() {
            let hidden = matches!(
                event,
                StreamEvent::Started { .. }
                    | StreamEvent::RateLimits { .. }
                    | StreamEvent::ContentBlockStart { .. }
            );

            assert_eq!(event.is_visible(), !hidden, "{event:?}");
        }
    }

    #[test]
    fn block_ids_keep_their_text() {
        assert_eq!(ContentBlockId::new("tool-1").as_str(), "tool-1");
    }
}
