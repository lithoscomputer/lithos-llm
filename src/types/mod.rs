//! Provider-neutral request, response, content, stream, and error types.

mod content;
mod error;
mod request;
mod response;
#[cfg(feature = "runtime")]
mod stream;

pub use content::{
    AudioContent, ContentPart, DocumentContent, ImageContent, Message, ReasoningContent, Role,
    ToolCall, ToolChoice, ToolDefinition, ToolResult,
};
pub use error::{Error, ErrorKind, RetryClassification};
pub use request::{
    ReasoningEffort, Request, RequestBuildError, RequestBuilder, ResponseFormat, Speed,
};
pub use response::{Cost, CostSource, FinishReason, RateLimits, Response, TokenCounts, Warning};
#[cfg(feature = "runtime")]
pub use stream::{ResponseStream, StreamEvent};
