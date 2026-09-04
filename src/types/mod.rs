//! Provider-neutral request, response, content, stream, and error types.

mod content;
mod error;
#[cfg(feature = "runtime")]
mod limits;
mod request;
mod response;
#[cfg(feature = "runtime")]
mod stream;
mod tool_input;

pub use content::{
    AudioContent, ContentPart, DocumentContent, ImageContent, MediaSource, Message,
    ReasoningContent, Role, ToolCall, ToolCallKind, ToolChoice, ToolDefinition, ToolDefinitionKind,
    ToolResult,
};
pub use error::{Error, ErrorData, ErrorKind, RetryClassification};
#[cfg(feature = "runtime")]
pub use limits::ResponseLimits;
#[cfg(feature = "runtime")]
pub(crate) use limits::{ResponsePolicy, limit_error};
pub use request::{
    CacheHint, ReasoningEffort, Request, RequestBuildError, RequestBuilder, ResponseFormat, Speed,
};
pub use response::{Cost, CostSource, FinishReason, RateLimits, Response, TokenCounts, Warning};
#[cfg(feature = "runtime")]
pub use stream::{ContentBlockId, ContentBlockKind, ResponseStream, StreamEvent};
pub use tool_input::{ToolArgumentError, ToolArguments, ToolInput};
