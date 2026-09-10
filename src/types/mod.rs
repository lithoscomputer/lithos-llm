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
mod unknown_content;

pub use content::{
    AudioContent, ContentPart, DocumentContent, ImageContent, MediaSource, Message,
    ReasoningContent, Role, ToolCall, ToolCallKind, ToolChoice, ToolDefinition, ToolDefinitionKind,
    ToolResult,
};
pub use error::{Error, ErrorData, ErrorKind, RetryClassification};
#[cfg(feature = "runtime")]
pub use limits::ResponseLimits;
#[cfg(feature = "runtime")]
pub(crate) use limits::ResponsePolicy;
#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "openai-compatible",
    feature = "bedrock"
))]
pub(crate) use limits::limit_error;
pub use request::{
    CacheHint, ReasoningEffort, Request, RequestBuildError, RequestBuilder, ResponseFormat, Speed,
    UnknownControlValue,
};
pub use response::{Cost, CostSource, FinishReason, RateLimits, Response, TokenCounts, Warning};
#[cfg(feature = "runtime")]
pub use stream::{ContentBlockId, ContentBlockKind, ResponseStream, StreamEvent};
pub use tool_input::{ToolArgumentError, ToolArguments, ToolInput};
pub use unknown_content::{UnknownContent, UnknownContentError};
