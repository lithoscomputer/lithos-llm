use std::pin::Pin;

use futures_core::Stream;
use serde::{Deserialize, Serialize};

use super::{ContentPart, Error, FinishReason, RateLimits, Response, TokenCounts, ToolCall};

/// One normalized event from a streaming response.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum StreamEvent {
    Started {
        id: Option<String>,
    },
    TextDelta {
        text: String,
    },
    ReasoningDelta {
        text: String,
    },
    ToolCallDelta {
        id:        String,
        name:      Option<String>,
        arguments: String,
    },
    ToolCall {
        call: ToolCall,
    },
    Content {
        part: ContentPart,
    },
    Usage {
        usage: TokenCounts,
    },
    RateLimits {
        rate_limits: RateLimits,
    },
    Finished {
        reason: FinishReason,
    },
    Completed {
        response: Response,
    },
}

impl StreamEvent {
    pub fn is_visible(&self) -> bool {
        !matches!(self, Self::Started { .. } | Self::RateLimits { .. })
    }
}

/// A cancel-on-drop stream of normalized provider events.
pub type ResponseStream = Pin<Box<dyn Stream<Item = Result<StreamEvent, Error>> + Send + 'static>>;
