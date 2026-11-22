//! The OpenAI Responses API: `POST /v1/responses`.
//!
//! Input and output share one [`Item`] type, because a response's output
//! items are replayed as the next request's input.
//!
//! See the [module documentation](super) for how these types stay lossless.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Chat Completions and Responses return the same error body.
pub use super::openai_chat::{ApiError, ErrorCode, ErrorResponse};
use super::{Extra, Field, Tagged, to_json, unknown};
use crate::types::TokenCounts;

/// A request body for `POST /v1/responses`.
///
/// Parameters without a field here, such as `tools`, `reasoning`, `include`,
/// `store`, and `previous_response_id`, are kept in `extra`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Request {
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub model:        Field<String>,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub input:        Field<Input>,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub instructions: Field<String>,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub stream:       Field<bool>,
    #[serde(flatten)]
    pub extra:        Extra,
}

/// [`Request::input`]: a string or a list of items.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Input {
    Text(String),
    Items(Vec<Item>),
}

/// An input or output item.
///
/// A message may omit its `type`; [`Message::kind`] records whether it was
/// present, so it is written back the way it arrived.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Item {
    FunctionCall(FunctionCall),
    FunctionCallOutput(CallOutput),
    CustomToolCall(CustomToolCall),
    CustomToolCallOutput(CallOutput),
    Reasoning(Reasoning),
    #[serde(untagged)]
    Message(Message),
    /// An item type not modeled here, such as `web_search_call`, kept as its
    /// full JSON object.
    #[serde(untagged, deserialize_with = "unknown::<Item, _>")]
    Unknown(Extra),
}

impl Tagged for Item {
    const TYPES: &'static [&'static str] = &[
        "message",
        "function_call",
        "function_call_output",
        "custom_tool_call",
        "custom_tool_call_output",
        "reasoning",
    ];
}

/// A `message` item. `id`, `status`, and `phase` are kept in `extra`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Message {
    /// `Some` when the item carried `"type": "message"`.
    #[serde(rename = "type", default, skip_serializing_if = "Field::is_absent")]
    pub kind:    Field<MessageType>,
    /// `user`, `assistant`, `system`, or `developer`.
    pub role:    String,
    pub content: Content,
    #[serde(flatten)]
    pub extra:   Extra,
}

/// The only `type` a [`Message`] item carries.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageType {
    Message,
}

/// Message or tool output content: a string or a list of parts.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// A part of message or tool output content.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ContentPart {
    InputText(Text),
    InputImage(InputImage),
    InputFile(InputFile),
    /// Model output. `annotations` and `logprobs` are kept in `extra`.
    OutputText(Text),
    Refusal(Refusal),
    /// A part type not modeled here, kept as its full JSON object.
    #[serde(untagged, deserialize_with = "unknown::<ContentPart, _>")]
    Unknown(Extra),
}

impl Tagged for ContentPart {
    const TYPES: &'static [&'static str] = &[
        "input_text",
        "input_image",
        "input_file",
        "output_text",
        "refusal",
    ];
}

/// A part that carries text: `input_text`, `output_text`, or `summary_text`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Text {
    pub text:  String,
    #[serde(flatten)]
    pub extra: Extra,
}

/// An `input_image` part. `file_id` and `detail` are kept in `extra`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct InputImage {
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub image_url: Field<String>,
    #[serde(flatten)]
    pub extra:     Extra,
}

/// An `input_file` part, whose `file_id`, `file_data`, `file_url`, and
/// `filename` are kept in `extra`.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct InputFile {
    #[serde(flatten)]
    pub extra: Extra,
}

/// A `refusal` part.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Refusal {
    pub refusal: String,
    #[serde(flatten)]
    pub extra:   Extra,
}

/// A `function_call` item. `id` and `status` are kept in `extra`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FunctionCall {
    pub call_id:   String,
    pub name:      String,
    /// The arguments as the JSON text the model wrote.
    pub arguments: String,
    #[serde(flatten)]
    pub extra:     Extra,
}

/// A `custom_tool_call` item, whose `input` is free-form text.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CustomToolCall {
    pub call_id: String,
    pub name:    String,
    pub input:   String,
    #[serde(flatten)]
    pub extra:   Extra,
}

/// A `function_call_output` or `custom_tool_call_output` item.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CallOutput {
    pub call_id: String,
    pub output:  Content,
    #[serde(flatten)]
    pub extra:   Extra,
}

/// A `reasoning` item.
///
/// OpenAI rejects a replayed item whose `encrypted_content` changed, so a
/// rewrite should leave it alone. `id`, `content`, and `status` are kept in
/// `extra`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Reasoning {
    pub summary:           Vec<SummaryPart>,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub encrypted_content: Field<String>,
    #[serde(flatten)]
    pub extra:             Extra,
}

/// A part of [`Reasoning::summary`].
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum SummaryPart {
    SummaryText(Text),
    /// A part type not modeled here, kept as its full JSON object.
    #[serde(untagged, deserialize_with = "unknown::<SummaryPart, _>")]
    Unknown(Extra),
}

impl Tagged for SummaryPart {
    const TYPES: &'static [&'static str] = &["summary_text"];
}

/// A response object: the body of `POST /v1/responses`, and the `response`
/// of each lifecycle [`StreamEvent`].
///
/// Most response fields echo the request, such as `instructions`, `tools`,
/// and `reasoning`, and are kept in `extra`.
///
/// `model` is the model that generated the response, which OpenAI may route
/// to another model than the one requested.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Response {
    pub id:                 String,
    pub model:              String,
    /// `completed`, `failed`, `in_progress`, `cancelled`, `queued`, or
    /// `incomplete`.
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub status:             Field<String>,
    /// Why a `failed` response failed.
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub error:              Field<ResponseError>,
    /// Why an `incomplete` response stopped.
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub incomplete_details: Field<IncompleteDetails>,
    pub output:             Vec<Item>,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub usage:              Field<Usage>,
    #[serde(flatten)]
    pub extra:              Extra,
}

/// The [`Response::error`] of a `failed` response.
///
/// Unlike an error body, this arrives inside a successful response or a
/// `response.failed` event. `misalignment` is kept in `extra`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ResponseError {
    /// Such as `server_error`, `invalid_prompt`, or a policy block such as
    /// `bio_policy`.
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub code:    Field<String>,
    pub message: String,
    #[serde(flatten)]
    pub extra:   Extra,
}

/// The [`Response::incomplete_details`] of an `incomplete` response.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct IncompleteDetails {
    /// Such as `max_output_tokens`, or `content_filter` when the output was
    /// cut off by OpenAI's content filters.
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub reason: Field<String>,
    #[serde(flatten)]
    pub extra:  Extra,
}

/// Token counts on a [`Response`].
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Usage {
    pub input_tokens:          u64,
    pub input_tokens_details:  InputTokensDetails,
    pub output_tokens:         u64,
    pub output_tokens_details: OutputTokensDetails,
    pub total_tokens:          u64,
    #[serde(flatten)]
    pub extra:                 Extra,
}

impl Usage {
    /// These counts in the disjoint buckets that
    /// [`Offering::estimate_cost`](crate::catalog::Offering::estimate_cost)
    /// prices, normalized as the client normalizes Responses usage.
    ///
    /// A stream reports usage once, on the `response` of its
    /// `response.completed` event.
    pub fn token_counts(&self) -> TokenCounts {
        token_counts(&to_json(self))
    }
}

/// Normalizes the inclusive usage counters into disjoint buckets.
///
/// `input_tokens` includes the cached and cache-written tokens and
/// `output_tokens` includes the reasoning tokens. The GPT-5.6 family bills
/// cache writes at a premium and reports them as
/// `input_tokens_details.cache_write_tokens` (observed live on 2026-08-30);
/// models that bill no write omit the counter or send zero, which decodes to
/// an empty bucket either way. Models that offer a 12-hour cache lifetime
/// split the writes in `cache_write_tokens_details`; its `ttl_12h_tokens` is
/// the long-lived share.
pub(crate) fn token_counts(usage: &Value) -> TokenCounts {
    let count = |pointer: &str| usage.pointer(pointer).and_then(Value::as_u64).unwrap_or(0);

    TokenCounts::from_inclusive(
        count("/input_tokens"),
        count("/output_tokens"),
        count("/output_tokens_details/reasoning_tokens"),
        count("/input_tokens_details/cached_tokens"),
        count("/input_tokens_details/cache_write_tokens"),
    )
    .with_long_cache_write(count(
        "/input_tokens_details/cache_write_tokens_details/ttl_12h_tokens",
    ))
}

/// The input token breakdown on a [`Usage`].
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct InputTokensDetails {
    pub cached_tokens: u64,
    #[serde(flatten)]
    pub extra:         Extra,
}

/// The output token breakdown on a [`Usage`].
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct OutputTokensDetails {
    pub reasoning_tokens: u64,
    #[serde(flatten)]
    pub extra:            Extra,
}

/// The JSON payload of one event in a streamed response.
///
/// Every event's `sequence_number` is kept in its `extra`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum StreamEvent {
    #[serde(rename = "response.created")]
    Created(ResponseEvent),
    #[serde(rename = "response.queued")]
    Queued(ResponseEvent),
    #[serde(rename = "response.in_progress")]
    InProgress(ResponseEvent),
    #[serde(rename = "response.completed")]
    Completed(ResponseEvent),
    #[serde(rename = "response.failed")]
    Failed(ResponseEvent),
    #[serde(rename = "response.incomplete")]
    Incomplete(ResponseEvent),
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded(OutputItemEvent),
    #[serde(rename = "response.output_item.done")]
    OutputItemDone(OutputItemEvent),
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded(ContentPartEvent),
    #[serde(rename = "response.content_part.done")]
    ContentPartDone(ContentPartEvent),
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta(TextDeltaEvent),
    #[serde(rename = "response.output_text.done")]
    OutputTextDone(TextDoneEvent),
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta(ArgumentsDeltaEvent),
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone(ArgumentsDoneEvent),
    #[serde(rename = "response.reasoning_summary_part.added")]
    ReasoningSummaryPartAdded(SummaryPartEvent),
    #[serde(rename = "response.reasoning_summary_part.done")]
    ReasoningSummaryPartDone(SummaryPartEvent),
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ReasoningSummaryTextDelta(SummaryTextDeltaEvent),
    #[serde(rename = "response.reasoning_summary_text.done")]
    ReasoningSummaryTextDone(SummaryTextDoneEvent),
    #[serde(rename = "error")]
    Error(ErrorEvent),
    /// An event type not modeled here, kept as its full JSON object.
    #[serde(untagged, deserialize_with = "unknown::<StreamEvent, _>")]
    Unknown(Extra),
}

impl Tagged for StreamEvent {
    const TYPES: &'static [&'static str] = &[
        "response.created",
        "response.queued",
        "response.in_progress",
        "response.completed",
        "response.failed",
        "response.incomplete",
        "response.output_item.added",
        "response.output_item.done",
        "response.content_part.added",
        "response.content_part.done",
        "response.output_text.delta",
        "response.output_text.done",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_part.done",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.done",
        "error",
    ];
}

/// A lifecycle event carrying the whole response so far.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ResponseEvent {
    pub response: Response,
    #[serde(flatten)]
    pub extra:    Extra,
}

/// `response.output_item.added` or `response.output_item.done`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct OutputItemEvent {
    pub output_index: u64,
    pub item:         Item,
    #[serde(flatten)]
    pub extra:        Extra,
}

/// `response.content_part.added` or `response.content_part.done`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ContentPartEvent {
    pub item_id:       String,
    pub output_index:  u64,
    pub content_index: u64,
    pub part:          ContentPart,
    #[serde(flatten)]
    pub extra:         Extra,
}

/// `response.output_text.delta`. `logprobs` is kept in `extra`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TextDeltaEvent {
    pub item_id:       String,
    pub output_index:  u64,
    pub content_index: u64,
    pub delta:         String,
    #[serde(flatten)]
    pub extra:         Extra,
}

/// `response.output_text.done`. `logprobs` is kept in `extra`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TextDoneEvent {
    pub item_id:       String,
    pub output_index:  u64,
    pub content_index: u64,
    pub text:          String,
    #[serde(flatten)]
    pub extra:         Extra,
}

/// `response.function_call_arguments.delta`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ArgumentsDeltaEvent {
    pub item_id:      String,
    pub output_index: u64,
    pub delta:        String,
    #[serde(flatten)]
    pub extra:        Extra,
}

/// `response.function_call_arguments.done`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ArgumentsDoneEvent {
    pub item_id:      String,
    pub output_index: u64,
    pub arguments:    String,
    #[serde(flatten)]
    pub extra:        Extra,
}

/// `response.reasoning_summary_part.added` or
/// `response.reasoning_summary_part.done`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SummaryPartEvent {
    pub item_id:       String,
    pub output_index:  u64,
    pub summary_index: u64,
    pub part:          SummaryPart,
    #[serde(flatten)]
    pub extra:         Extra,
}

/// `response.reasoning_summary_text.delta`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SummaryTextDeltaEvent {
    pub item_id:       String,
    pub output_index:  u64,
    pub summary_index: u64,
    pub delta:         String,
    #[serde(flatten)]
    pub extra:         Extra,
}

/// `response.reasoning_summary_text.done`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SummaryTextDoneEvent {
    pub item_id:       String,
    pub output_index:  u64,
    pub summary_index: u64,
    pub text:          String,
    #[serde(flatten)]
    pub extra:         Extra,
}

/// `error`: the stream failed after it started.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ErrorEvent {
    pub message: String,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub code:    Field<String>,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub param:   Field<String>,
    #[serde(flatten)]
    pub extra:   Extra,
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{ContentPart, Item, StreamEvent, SummaryPart};
    use crate::wire::assert_types_are_modeled;

    #[test]
    fn every_listed_type_has_a_variant() {
        assert_types_are_modeled::<Item>(&[
            json!({"type": "message", "role": "user", "content": "hi"}),
            json!({"type": "function_call", "call_id": "call_1", "name": "f", "arguments": "{}"}),
            json!({"type": "function_call_output", "call_id": "call_1", "output": "ok"}),
            json!({"type": "custom_tool_call", "call_id": "call_2", "name": "g", "input": "x"}),
            json!({"type": "custom_tool_call_output", "call_id": "call_2", "output": "ok"}),
            json!({"type": "reasoning", "summary": []}),
        ]);
        assert_types_are_modeled::<ContentPart>(&[
            json!({"type": "input_text", "text": "hi"}),
            json!({"type": "input_image", "image_url": "https://example.com/a.png"}),
            json!({"type": "input_file", "file_id": "file_1"}),
            json!({"type": "output_text", "text": "hi", "annotations": []}),
            json!({"type": "refusal", "refusal": "no"}),
        ]);
        assert_types_are_modeled::<SummaryPart>(&[json!({"type": "summary_text", "text": "s"})]);

        let response = json!({"id": "resp_1", "model": "m", "output": []});
        let events: Vec<Value> = [
            "response.created",
            "response.queued",
            "response.in_progress",
            "response.completed",
            "response.failed",
            "response.incomplete",
        ]
        .into_iter()
        .map(|kind| json!({"type": kind, "response": response}))
        .chain([
            json!({"type": "response.output_item.added", "output_index": 0, "item": {"type": "reasoning", "summary": []}}),
            json!({"type": "response.output_item.done", "output_index": 0, "item": {"type": "reasoning", "summary": []}}),
            json!({"type": "response.content_part.added", "item_id": "msg_1", "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": ""}}),
            json!({"type": "response.content_part.done", "item_id": "msg_1", "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": "a"}}),
            json!({"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0, "content_index": 0, "delta": "a"}),
            json!({"type": "response.output_text.done", "item_id": "msg_1", "output_index": 0, "content_index": 0, "text": "a"}),
            json!({"type": "response.function_call_arguments.delta", "item_id": "fc_1", "output_index": 0, "delta": "{"}),
            json!({"type": "response.function_call_arguments.done", "item_id": "fc_1", "output_index": 0, "arguments": "{}"}),
            json!({"type": "response.reasoning_summary_part.added", "item_id": "rs_1", "output_index": 0, "summary_index": 0, "part": {"type": "summary_text", "text": ""}}),
            json!({"type": "response.reasoning_summary_part.done", "item_id": "rs_1", "output_index": 0, "summary_index": 0, "part": {"type": "summary_text", "text": "s"}}),
            json!({"type": "response.reasoning_summary_text.delta", "item_id": "rs_1", "output_index": 0, "summary_index": 0, "delta": "s"}),
            json!({"type": "response.reasoning_summary_text.done", "item_id": "rs_1", "output_index": 0, "summary_index": 0, "text": "s"}),
            json!({"type": "error", "message": "boom"}),
        ])
        .collect();
        assert_types_are_modeled::<StreamEvent>(&events);
    }
}
