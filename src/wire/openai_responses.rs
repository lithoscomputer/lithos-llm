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
use super::{Extra, Field, Outcome, Refusal as OutcomeRefusal, Tagged, to_json, unknown};
use crate::types::{FinishReason, TokenCounts};

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
    /// A bare string, as Venice writes the summary of a model it serves by
    /// translating Chat Completions.
    #[serde(untagged)]
    Plain(String),
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
///
/// OpenAI always sends both details objects. Venice omits
/// `input_tokens_details` for a model it serves by translating Chat
/// Completions, so either may be absent.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Usage {
    pub input_tokens:          u64,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub input_tokens_details:  Field<InputTokensDetails>,
    pub output_tokens:         u64,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub output_tokens_details: Field<OutputTokensDetails>,
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

impl Response {
    /// How this response ended, which model served it, and what it was
    /// billed for.
    pub fn outcome(&self) -> Outcome {
        let mut outcome = Outcome::default();
        fold_response(self, &mut outcome);
        outcome
    }
}

impl StreamEvent {
    /// Folds what this event reports into `outcome`: everything a lifecycle
    /// event's response carries, a finished `refusal` part, and an `error`.
    /// Fold every event of a stream, in order.
    pub fn fold_into(&self, outcome: &mut Outcome) {
        match self {
            Self::Created(event)
            | Self::Queued(event)
            | Self::InProgress(event)
            | Self::Completed(event)
            | Self::Failed(event)
            | Self::Incomplete(event) => fold_response(&event.response, outcome),
            Self::ContentPartDone(event) => {
                if let ContentPart::Refusal(part) = &event.part
                    && !part.refusal.is_empty()
                {
                    outcome.set_refusal(OutcomeRefusal {
                        category:    None,
                        explanation: Some(part.refusal.clone()),
                    });
                }
            }
            Self::Error(event) => {
                outcome.set_error(event.code.as_option().map(String::as_str), &event.message);
            }
            _ => {}
        }
    }
}

fn fold_response(response: &Response, outcome: &mut Outcome) {
    outcome.set_model(&response.model);
    let reason = response
        .incomplete_details
        .as_option()
        .and_then(|details| details.reason.as_option())
        .map(String::as_str);
    // A response still being generated has not stopped.
    if let Some(status) = response
        .status
        .as_option()
        .filter(|status| !matches!(status.as_str(), "queued" | "in_progress"))
    {
        let has_tool_call = response
            .output
            .iter()
            .any(|item| matches!(item, Item::FunctionCall(_) | Item::CustomToolCall(_)));
        let raw = match reason {
            Some(reason) if status == "incomplete" => reason,
            _ => status,
        };
        outcome.set_stop(raw, finish_reason(Some(status), reason, has_tool_call));
    }
    if let Field::Present(error) = &response.error {
        outcome.set_error(error.code.as_option().map(String::as_str), &error.message);
    }
    if let Some(text) = refusal_in_output(&to_json(&response.output)) {
        outcome.set_refusal(OutcomeRefusal {
            category:    None,
            explanation: Some(text.to_owned()),
        });
    }
    let usage = response.usage.as_option();
    if let Some(usage) = usage {
        outcome.set_usage(usage.token_counts());
    }
    outcome.set_tier(None, response.extra.get("service_tier"));
    outcome.set_provider_cost(usage.map(|usage| &usage.extra), response.extra.get("cost"));
}

/// Normalizes a response's `status` into a finish reason.
///
/// The protocol has no separate stop reason. An `incomplete` response stopped
/// at a limit, or at a content filter when `incomplete_reason` says so. A
/// completed response that called tools reports [`FinishReason::ToolCall`],
/// the value every other protocol produces for one. A status this crate does
/// not name, such as `cancelled`, keeps the provider's own spelling.
pub(crate) fn finish_reason(
    status: Option<&str>,
    incomplete_reason: Option<&str>,
    has_tool_call: bool,
) -> FinishReason {
    match status {
        Some("incomplete") => match incomplete_reason {
            Some("content_filter") => FinishReason::ContentFilter,
            _ => FinishReason::Length,
        },
        Some("failed") => FinishReason::Error,
        Some(other) if other != "completed" => FinishReason::Other(other.to_owned()),
        _ if has_tool_call => FinishReason::ToolCall,
        _ => FinishReason::Stop,
    }
}

/// The refusal text carried by a response document's message items, if any.
pub(crate) fn refusal_text(document: &Value) -> Option<&str> {
    refusal_in_output(document.get("output")?)
}

/// The refusal text in a response's `output` items, if any.
///
/// A non-empty `refusal` content part means the model declined instead of
/// answering; an empty one is treated as absent, as an empty Chat `refusal`
/// is.
fn refusal_in_output(output: &Value) -> Option<&str> {
    output
        .as_array()?
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("message"))
        .filter_map(|item| item.get("content").and_then(Value::as_array))
        .flatten()
        .filter(|part| part.get("type").and_then(Value::as_str) == Some("refusal"))
        .find_map(|part| part.get("refusal").and_then(Value::as_str))
        .filter(|text| !text.is_empty())
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

    use super::{ContentPart, Item, Response, StreamEvent, SummaryPart};
    use crate::types::TokenCounts;
    use crate::wire::{Outcome, assert_types_are_modeled};

    /// Venice's Responses body for a model it serves by translating Chat
    /// Completions, as returned for GLM 5.3 Flash on 2026-11-26: string
    /// summaries and no `input_tokens_details`.
    fn venice_translated() -> Value {
        json!({
            "id": "resp_chatcmpl-1",
            "object": "response",
            "created_at": 1,
            "model": "z-ai-glm-5-3-flash",
            "status": "completed",
            "output": [
                {"type": "reasoning", "id": "rs_1", "summary": ["Simple request."]},
                {"type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
                 "content": [{"type": "output_text", "text": "ok.", "annotations": []}]}
            ],
            "usage": {"input_tokens": 21, "output_tokens": 12,
                      "output_tokens_details": {"reasoning_tokens": 8}, "total_tokens": 33}
        })
    }

    #[test]
    fn reads_venices_translated_responses_without_loss() {
        let body = venice_translated();
        let response: Response = serde_json::from_value(body.clone()).unwrap();

        assert_eq!(serde_json::to_value(&response).unwrap(), body);
        let counts = response.outcome().usage().unwrap();
        assert_eq!(counts, TokenCounts::from_inclusive(21, 12, 8, 0, 0));

        // The same response completes a stream.
        let event = json!({"type": "response.completed", "response": body, "sequence_number": 11});
        let event: StreamEvent = serde_json::from_value(event).unwrap();
        let mut outcome = Outcome::default();
        event.fold_into(&mut outcome);
        assert_eq!(outcome.usage(), Some(counts));
    }

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
