use std::pin::Pin;
use std::task::{Context, Poll};

use futures_core::Stream;
use futures_core::stream::FusedStream;
use serde::{Deserialize, Serialize};

use super::{
    ContentPart, Error, ErrorKind, RateLimits, Response, RetryClassification, TokenCounts,
    ToolCallKind,
};

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
/// - Every successful stream ends with exactly one [`Completed`] event. Its
///   response is authoritative. Block-end parts are provisional: a truncated
///   tool call may be omitted from the final response. Inspect the final finish
///   reason before executing tools.
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
    /// A content block closed, carrying its provisional assembled part.
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
    /// [`RateLimits`](Self::RateLimits),
    /// [`ContentBlockStart`](Self::ContentBlockStart), and
    /// [`Usage`](Self::Usage) — is not visible. Usage events are cumulative
    /// snapshots a reconnected attempt replaces wholesale, and Anthropic
    /// reports one before any content exists, so counting them as visible
    /// would close the stream-retry window at `message_start`.
    pub fn is_visible(&self) -> bool {
        !matches!(
            self,
            Self::Started { .. }
                | Self::RateLimits { .. }
                | Self::ContentBlockStart { .. }
                | Self::Usage { .. }
        )
    }
}

/// A cancel-on-drop stream with exactly one terminal event or error.
///
/// Completion or failure immediately releases the inner stream. All later
/// polls return `None`. An inner stream that ends without `Completed` produces
/// a `StreamDecode` error, rather than silently appearing successful.
pub struct ResponseStream {
    inner: Option<BoxedEvents>,
}

type BoxedEvents = Pin<Box<dyn Stream<Item = Result<StreamEvent, Error>> + Send + 'static>>;

impl ResponseStream {
    pub fn new(stream: impl Stream<Item = Result<StreamEvent, Error>> + Send + 'static) -> Self {
        Self {
            inner: Some(Box::pin(stream)),
        }
    }
}

impl Stream for ResponseStream {
    type Item = Result<StreamEvent, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let Some(inner) = self.inner.as_mut() else {
            return Poll::Ready(None);
        };
        let result = inner.as_mut().poll_next(cx);
        match result {
            Poll::Ready(Some(Ok(StreamEvent::Completed { .. }) | Err(_))) => {
                self.inner = None;
                result
            }
            Poll::Ready(None) => {
                self.inner = None;
                Poll::Ready(Some(Err(Error::new(
                    ErrorKind::StreamDecode,
                    "the response stream ended without completion",
                )
                .with_retry(RetryClassification::Safe))))
            }
            _ => result,
        }
    }
}

impl FusedStream for ResponseStream {
    fn is_terminated(&self) -> bool {
        self.inner.is_none()
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use futures_util::StreamExt as _;
    use futures_util::stream::iter;
    use serde_json::json;

    use super::{ContentBlockId, ContentBlockKind, ResponseStream, StreamEvent};
    use crate::catalog::{ModelId, ProviderId};
    use crate::middleware::{finalize_stream, map_stream};
    use crate::types::{
        ContentPart, Error, ErrorKind, RateLimits, Response, TokenCounts, ToolCall, ToolCallKind,
    };

    #[tokio::test]
    async fn terminal_events_release_resources_and_fuse() {
        for terminal in [
            Ok(StreamEvent::Completed {
                response: Response::new(ProviderId::new("p"), ModelId::new("m"), vec![]),
            }),
            Err(Error::new(ErrorKind::Network, "failed")),
        ] {
            let drops = Arc::new(AtomicUsize::new(0));
            let counter = drops.clone();
            let inner = finalize_stream(
                ResponseStream::new(iter([terminal, Ok(StreamEvent::Started { id: None })])),
                move || {
                    counter.fetch_add(1, Ordering::SeqCst);
                },
            );
            let mut stream = ResponseStream::new(inner);
            assert!(stream.next().await.is_some());
            assert_eq!(drops.load(Ordering::SeqCst), 1);
            assert!(stream.next().await.is_none());
            assert!(stream.next().await.is_none());
        }
    }

    #[tokio::test]
    async fn premature_eof_and_mapping_errors_are_terminal() {
        let mut stream = ResponseStream::new(iter([]));
        let error = stream
            .next()
            .await
            .expect("error")
            .expect_err("missing completion");
        assert_eq!(error.kind(), ErrorKind::StreamDecode);
        assert_eq!(
            error.retry_classification(),
            super::RetryClassification::Safe
        );
        assert!(stream.next().await.is_none());
        let mut mapped = map_stream(
            ResponseStream::new(iter([
                Ok(StreamEvent::Started { id: None }),
                Ok(StreamEvent::Started { id: None }),
            ])),
            |_| Err(Error::new(ErrorKind::Middleware, "mapping failed")),
        );
        assert!(mapped.next().await.expect("mapping error").is_err());
        assert!(mapped.next().await.is_none());
    }

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
                    | StreamEvent::Usage { .. }
            );

            assert_eq!(event.is_visible(), !hidden, "{event:?}");
        }
    }

    #[test]
    fn block_ids_keep_their_text() {
        assert_eq!(ContentBlockId::new("tool-1").as_str(), "tool-1");
    }
}
