//! The OpenAI Chat Completions API: `POST /v1/chat/completions`.
//!
//! OpenRouter and most OpenAI-compatible providers speak this format with
//! their own additions, such as OpenRouter's `reasoning_details`; those are
//! kept in `extra` like any other unknown field.
//!
//! See the [module documentation](super) for how these types stay lossless.

use serde::{Deserialize, Serialize};
use serde_json::Number;

use super::{Extra, Field, Tagged, unknown};

/// A request body for `POST /v1/chat/completions`.
///
/// Parameters without a field here, such as `tools`, `max_tokens`,
/// `temperature`, `stream_options`, and provider-specific options, are kept
/// in `extra`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Request {
    pub model:    String,
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub stream:   Field<bool>,
    #[serde(flatten)]
    pub extra:    Extra,
}

/// A message in [`Request::messages`] or a [`Choice`].
///
/// One struct covers every role, because the roles differ only in which
/// optional fields they carry. `name`, `refusal`, `annotations`, and `audio`
/// are kept in `extra`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Message {
    /// `system`, `developer`, `user`, `assistant`, or `tool`.
    pub role:         String,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub content:      Field<Content>,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub tool_calls:   Field<Vec<ToolCall>>,
    /// The call a `tool` message answers.
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub tool_call_id: Field<String>,
    #[serde(flatten)]
    pub extra:        Extra,
}

/// Message content: a string or a list of parts.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// A part of message content.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ContentPart {
    /// A `text` part. `cache_control`, which some providers accept here, is
    /// kept in `extra`.
    Text(TextPart),
    ImageUrl(ImageUrlPart),
    Refusal(RefusalPart),
    /// A part type not modeled here, such as `input_audio` or `file`, kept as
    /// its full JSON object.
    #[serde(untagged, deserialize_with = "unknown::<ContentPart, _>")]
    Unknown(Extra),
}

impl Tagged for ContentPart {
    const TYPES: &'static [&'static str] = &["text", "image_url", "refusal"];
}

/// A `text` part.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TextPart {
    pub text:  String,
    #[serde(flatten)]
    pub extra: Extra,
}

/// An `image_url` part.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ImageUrlPart {
    pub image_url: ImageUrl,
    #[serde(flatten)]
    pub extra:     Extra,
}

/// The image inside an [`ImageUrlPart`]. `detail` is kept in `extra`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ImageUrl {
    /// An `https` URL or a `data:` URL.
    pub url:   String,
    #[serde(flatten)]
    pub extra: Extra,
}

/// A `refusal` part.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RefusalPart {
    pub refusal: String,
    #[serde(flatten)]
    pub extra:   Extra,
}

/// A tool call on an assistant [`Message`].
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ToolCall {
    Function(FunctionToolCall),
    /// A tool call type not modeled here, such as `custom`, kept as its full
    /// JSON object.
    #[serde(untagged, deserialize_with = "unknown::<ToolCall, _>")]
    Unknown(Extra),
}

impl Tagged for ToolCall {
    const TYPES: &'static [&'static str] = &["function"];
}

/// A `function` tool call.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FunctionToolCall {
    pub id:       String,
    pub function: FunctionCall,
    #[serde(flatten)]
    pub extra:    Extra,
}

/// The function and arguments of a [`FunctionToolCall`].
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FunctionCall {
    pub name:      String,
    /// The arguments as the JSON text the model wrote.
    pub arguments: String,
    #[serde(flatten)]
    pub extra:     Extra,
}

/// A response body from `POST /v1/chat/completions`.
///
/// `object`, `created`, `system_fingerprint`, and `service_tier` are kept in
/// `extra`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Response {
    pub id:      String,
    pub model:   String,
    pub choices: Vec<Choice>,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub usage:   Field<Usage>,
    #[serde(flatten)]
    pub extra:   Extra,
}

/// One completion in [`Response::choices`]. `logprobs` is kept in `extra`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Choice {
    pub index:         u64,
    pub message:       Message,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub finish_reason: Field<String>,
    #[serde(flatten)]
    pub extra:         Extra,
}

/// Token counts on a [`Response`] or the last [`StreamEvent`].
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Usage {
    pub prompt_tokens:             u64,
    pub completion_tokens:         u64,
    pub total_tokens:              u64,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub prompt_tokens_details:     Field<PromptTokensDetails>,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub completion_tokens_details: Field<CompletionTokensDetails>,
    #[serde(flatten)]
    pub extra:                     Extra,
}

/// The prompt token breakdown on a [`Usage`].
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PromptTokensDetails {
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub cached_tokens: Field<u64>,
    #[serde(flatten)]
    pub extra:         Extra,
}

/// The completion token breakdown on a [`Usage`].
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CompletionTokensDetails {
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub reasoning_tokens: Field<u64>,
    #[serde(flatten)]
    pub extra:            Extra,
}

/// The JSON payload of one event in a streamed response: a
/// `chat.completion.chunk` object.
///
/// The stream ends with `data: [DONE]`, which is not JSON and has no type.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct StreamEvent {
    pub id:      String,
    pub model:   String,
    pub choices: Vec<ChunkChoice>,
    /// Present on the last chunk when the request set
    /// `stream_options.include_usage`.
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub usage:   Field<Usage>,
    #[serde(flatten)]
    pub extra:   Extra,
}

/// One choice in a [`StreamEvent`]. `logprobs` is kept in `extra`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ChunkChoice {
    pub index:         u64,
    pub delta:         Delta,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub finish_reason: Field<String>,
    #[serde(flatten)]
    pub extra:         Extra,
}

/// The increment inside a [`ChunkChoice`]. `refusal` is kept in `extra`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Delta {
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub role:       Field<String>,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub content:    Field<String>,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub tool_calls: Field<Vec<ToolCallDelta>>,
    #[serde(flatten)]
    pub extra:      Extra,
}

/// A fragment of the tool call at `index`. `type` is kept in `extra`.
///
/// The first fragment of a call carries its `id` and function name; later
/// fragments carry more argument text.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ToolCallDelta {
    pub index:    u64,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub id:       Field<String>,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub function: Field<FunctionCallDelta>,
    #[serde(flatten)]
    pub extra:    Extra,
}

/// The function fragment inside a [`ToolCallDelta`].
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FunctionCallDelta {
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub name:      Field<String>,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub arguments: Field<String>,
    #[serde(flatten)]
    pub extra:     Extra,
}

/// An error response body: `{"error": {...}}`.
///
/// OpenAI Responses returns the same body, so
/// [`openai_responses`](super::openai_responses) re-exports this type.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ErrorResponse {
    pub error: ApiError,
    #[serde(flatten)]
    pub extra: Extra,
}

impl ErrorResponse {
    /// An error of type `kind`, such as `invalid_request_error`, with `param`
    /// and `code` set to `null`, as OpenAI writes an error that has neither.
    pub fn new(kind: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            error: ApiError {
                message: message.into(),
                kind:    Field::Present(kind.into()),
                param:   Field::Null,
                code:    Field::Null,
                extra:   Extra::new(),
            },
            extra: Extra::new(),
        }
    }
}

/// The error inside an [`ErrorResponse`].
///
/// OpenAI sends `type`, `param`, and `code`, often as `null`. OpenRouter
/// omits `type` and `param`, sends the HTTP status as `code`, and adds
/// `metadata`, which is kept in `extra`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ApiError {
    pub message: String,
    /// The error type, such as `invalid_request_error`.
    #[serde(rename = "type", default, skip_serializing_if = "Field::is_absent")]
    pub kind:    Field<String>,
    /// The request parameter the error is about.
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub param:   Field<String>,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub code:    Field<ErrorCode>,
    #[serde(flatten)]
    pub extra:   Extra,
}

/// [`ApiError::code`]: a string on OpenAI, a number on OpenRouter.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ErrorCode {
    /// A code such as `context_length_exceeded`.
    Text(String),
    /// A number, such as OpenRouter's HTTP status, kept exactly as sent.
    Number(Number),
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{ContentPart, ToolCall};
    use crate::wire::assert_types_are_modeled;

    #[test]
    fn every_listed_type_has_a_variant() {
        assert_types_are_modeled::<ContentPart>(&[
            json!({"type": "text", "text": "hi"}),
            json!({"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}),
            json!({"type": "refusal", "refusal": "no"}),
        ]);
        assert_types_are_modeled::<ToolCall>(&[json!({
            "type": "function", "id": "call_1",
            "function": {"name": "f", "arguments": "{}"},
        })]);
    }
}
