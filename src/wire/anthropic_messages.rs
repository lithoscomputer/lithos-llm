//! The Anthropic Messages API: `POST /v1/messages`.
//!
//! See the [module documentation](super) for how these types stay lossless.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{Extra, Field, Tagged, unknown};

/// A request body for `POST /v1/messages`.
///
/// Parameters without a field here, such as `tools`, `thinking`,
/// `temperature`, and `metadata`, are kept in `extra`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Request {
    pub model:      String,
    pub messages:   Vec<InputMessage>,
    pub max_tokens: u64,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub system:     Field<Content>,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub stream:     Field<bool>,
    #[serde(flatten)]
    pub extra:      Extra,
}

/// One turn in [`Request::messages`].
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct InputMessage {
    /// `user` or `assistant`.
    pub role:    String,
    pub content: Content,
    #[serde(flatten)]
    pub extra:   Extra,
}

/// Message, system, or tool result content: a string or a list of blocks.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// A content block in a request or a response.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ContentBlock {
    Text(TextBlock),
    Image(SourceBlock),
    Document(SourceBlock),
    ToolUse(ToolUseBlock),
    ToolResult(ToolResultBlock),
    Thinking(ThinkingBlock),
    RedactedThinking(RedactedThinkingBlock),
    ServerToolUse(ToolUseBlock),
    /// A block type not modeled here, kept as its full JSON object.
    #[serde(untagged, deserialize_with = "unknown::<ContentBlock, _>")]
    Unknown(Extra),
}

impl Tagged for ContentBlock {
    const TYPES: &'static [&'static str] = &[
        "text",
        "image",
        "document",
        "tool_use",
        "tool_result",
        "thinking",
        "redacted_thinking",
        "server_tool_use",
    ];
}

/// A `text` block. `cache_control` and `citations` are kept in `extra`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TextBlock {
    pub text:  String,
    #[serde(flatten)]
    pub extra: Extra,
}

/// An `image` or `document` block, whose `source` is kept as sent.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SourceBlock {
    pub source: Value,
    #[serde(flatten)]
    pub extra:  Extra,
}

/// A `tool_use` or `server_tool_use` block.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ToolUseBlock {
    pub id:    String,
    pub name:  String,
    pub input: Value,
    #[serde(flatten)]
    pub extra: Extra,
}

/// A `tool_result` block. `is_error` and `cache_control` are kept in `extra`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ToolResultBlock {
    pub tool_use_id: String,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub content:     Field<Content>,
    #[serde(flatten)]
    pub extra:       Extra,
}

/// A `thinking` block.
///
/// Anthropic rejects a replayed block whose text or signature changed, so a
/// rewrite should leave both alone. A block opened by `content_block_start`
/// has no signature yet; a `signature_delta` supplies it.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ThinkingBlock {
    pub thinking:  String,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub signature: Field<String>,
    #[serde(flatten)]
    pub extra:     Extra,
}

/// A `redacted_thinking` block, whose `data` is encrypted and opaque.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RedactedThinkingBlock {
    pub data:  String,
    #[serde(flatten)]
    pub extra: Extra,
}

/// A response body from `POST /v1/messages`, also carried by
/// [`StreamEvent::MessageStart`].
///
/// `type`, `role`, `container`, and `stop_details` are kept in `extra`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Response {
    pub id:            String,
    pub model:         String,
    pub content:       Vec<ContentBlock>,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub stop_reason:   Field<String>,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub stop_sequence: Field<String>,
    pub usage:         Usage,
    #[serde(flatten)]
    pub extra:         Extra,
}

/// Token counts on a [`Response`].
///
/// `cache_creation`, `server_tool_use`, and `service_tier` are kept in
/// `extra`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub cache_creation_input_tokens: Field<u64>,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub cache_read_input_tokens: Field<u64>,
    #[serde(flatten)]
    pub extra: Extra,
}

/// The JSON payload of one event in a streamed response.
///
/// The SSE `event` name repeats the payload's `type`, so parsing the `data`
/// alone loses nothing.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum StreamEvent {
    MessageStart(MessageStart),
    ContentBlockStart(ContentBlockStart),
    ContentBlockDelta(ContentBlockDelta),
    ContentBlockStop(ContentBlockStop),
    MessageDelta(MessageDelta),
    MessageStop(Empty),
    Ping(Empty),
    Error(ErrorEvent),
    /// An event type not modeled here, kept as its full JSON object.
    #[serde(untagged, deserialize_with = "unknown::<StreamEvent, _>")]
    Unknown(Extra),
}

impl Tagged for StreamEvent {
    const TYPES: &'static [&'static str] = &[
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
        "ping",
        "error",
    ];
}

/// An event that carries nothing but its type, such as `ping`.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Empty {
    #[serde(flatten)]
    pub extra: Extra,
}

/// `message_start`: the response so far, with empty content.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MessageStart {
    pub message: Response,
    #[serde(flatten)]
    pub extra:   Extra,
}

/// `content_block_start`: a block opens at `index`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ContentBlockStart {
    pub index:         u64,
    pub content_block: ContentBlock,
    #[serde(flatten)]
    pub extra:         Extra,
}

/// `content_block_delta`: an increment to the block at `index`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ContentBlockDelta {
    pub index: u64,
    pub delta: Delta,
    #[serde(flatten)]
    pub extra: Extra,
}

/// `content_block_stop`: the block at `index` is complete.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ContentBlockStop {
    pub index: u64,
    #[serde(flatten)]
    pub extra: Extra,
}

/// The increment inside a [`ContentBlockDelta`].
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Delta {
    TextDelta(TextDelta),
    InputJsonDelta(InputJsonDelta),
    ThinkingDelta(ThinkingDelta),
    SignatureDelta(SignatureDelta),
    /// A delta type not modeled here, such as `citations_delta`, kept as its
    /// full JSON object.
    #[serde(untagged, deserialize_with = "unknown::<Delta, _>")]
    Unknown(Extra),
}

impl Tagged for Delta {
    const TYPES: &'static [&'static str] = &[
        "text_delta",
        "input_json_delta",
        "thinking_delta",
        "signature_delta",
    ];
}

/// `text_delta`: more text for a `text` block.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TextDelta {
    pub text:  String,
    #[serde(flatten)]
    pub extra: Extra,
}

/// `input_json_delta`: a fragment of a tool use block's JSON input.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct InputJsonDelta {
    pub partial_json: String,
    #[serde(flatten)]
    pub extra:        Extra,
}

/// `thinking_delta`: more text for a `thinking` block.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ThinkingDelta {
    pub thinking: String,
    #[serde(flatten)]
    pub extra:    Extra,
}

/// `signature_delta`: the signature of a `thinking` block.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SignatureDelta {
    pub signature: String,
    #[serde(flatten)]
    pub extra:     Extra,
}

/// `message_delta`: top-level changes and cumulative usage near the end.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MessageDelta {
    pub delta: MessageDeltaFields,
    pub usage: DeltaUsage,
    #[serde(flatten)]
    pub extra: Extra,
}

/// The changed top-level fields inside a [`MessageDelta`].
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MessageDeltaFields {
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub stop_reason:   Field<String>,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub stop_sequence: Field<String>,
    #[serde(flatten)]
    pub extra:         Extra,
}

/// Cumulative token counts on a [`MessageDelta`].
///
/// Unlike [`Usage`], `input_tokens` may be absent or `null` here.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DeltaUsage {
    pub output_tokens: u64,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub input_tokens: Field<u64>,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub cache_creation_input_tokens: Field<u64>,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub cache_read_input_tokens: Field<u64>,
    #[serde(flatten)]
    pub extra: Extra,
}

/// `error`: the stream failed after it started.
///
/// An error response body is the same object; [`ErrorResponse`] parses it.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ErrorEvent {
    pub error: ApiError,
    #[serde(flatten)]
    pub extra: Extra,
}

/// An error response body: `{"type": "error", "error": {...}}`.
///
/// It writes `"type": "error"` itself, and parsing rejects any other `type`.
/// `request_id` is kept in `extra`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(from = "TaggedError", into = "TaggedError")]
pub struct ErrorResponse {
    pub error: ApiError,
    pub extra: Extra,
}

/// The wire form of an [`ErrorResponse`]. A tagged struct would neither
/// check its `type` nor keep it out of `extra`; an enum does both.
#[derive(Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum TaggedError {
    Error(ErrorEvent),
}

impl From<TaggedError> for ErrorResponse {
    fn from(TaggedError::Error(ErrorEvent { error, extra }): TaggedError) -> Self {
        Self { error, extra }
    }
}

impl From<ErrorResponse> for TaggedError {
    fn from(ErrorResponse { error, extra }: ErrorResponse) -> Self {
        Self::Error(ErrorEvent { error, extra })
    }
}

/// The error inside an [`ErrorEvent`] or an [`ErrorResponse`].
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ApiError {
    /// The error type, such as `overloaded_error`.
    #[serde(rename = "type")]
    pub kind:    String,
    pub message: String,
    #[serde(flatten)]
    pub extra:   Extra,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{ContentBlock, Delta, StreamEvent};
    use crate::wire::assert_types_are_modeled;

    #[test]
    fn every_listed_type_has_a_variant() {
        assert_types_are_modeled::<ContentBlock>(&[
            json!({"type": "text", "text": "hi"}),
            json!({"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}),
            json!({"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "x"}}),
            json!({"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}}),
            json!({"type": "tool_result", "tool_use_id": "toolu_1"}),
            json!({"type": "thinking", "thinking": ""}),
            json!({"type": "redacted_thinking", "data": "ZW5j"}),
            json!({"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}}),
        ]);
        assert_types_are_modeled::<StreamEvent>(&[
            json!({"type": "message_start", "message": {
                "id": "msg_1", "model": "m", "content": [],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }}),
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "a"}}),
            json!({"type": "content_block_stop", "index": 0}),
            json!({"type": "message_delta", "delta": {}, "usage": {"output_tokens": 1}}),
            json!({"type": "message_stop"}),
            json!({"type": "ping"}),
            json!({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}),
        ]);
        assert_types_are_modeled::<Delta>(&[
            json!({"type": "text_delta", "text": "a"}),
            json!({"type": "input_json_delta", "partial_json": "{"}),
            json!({"type": "thinking_delta", "thinking": "a"}),
            json!({"type": "signature_delta", "signature": "c2ln"}),
        ]);
    }
}
