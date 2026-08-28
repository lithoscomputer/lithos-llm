//! Provider-neutral request, response, content, stream, and error types.

mod content;
mod error;
mod request;
mod response;
#[cfg(feature = "runtime")]
mod stream;

pub use content::{
    AudioContent, ContentPart, DocumentContent, ImageContent, MediaSource, Message,
    ReasoningContent, Role, ToolCall, ToolCallKind, ToolChoice, ToolDefinition, ToolDefinitionKind,
    ToolResult,
};
pub use error::{Error, ErrorData, ErrorKind, RetryClassification};
pub use request::{
    ReasoningEffort, Request, RequestBuildError, RequestBuilder, ResponseFormat, Speed,
};
pub use response::{Cost, CostSource, FinishReason, RateLimits, Response, TokenCounts, Warning};
#[cfg(feature = "runtime")]
pub use stream::{ContentBlockId, ContentBlockKind, ResponseStream, StreamEvent};
