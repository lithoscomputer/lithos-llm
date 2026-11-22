//! The Anthropic Messages API: `POST /v1/messages`.
//!
//! See the [module documentation](super) for how these types stay lossless.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{Extra, Fallback, Field, Outcome, Refusal, Tagged, to_json, unknown};
use crate::types::{TokenCounts, finish_reason};

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
    Fallback(FallbackBlock),
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
        "fallback",
    ];
}

/// A `fallback` block: where server-side fallback handed the turn from a
/// model that declined to the next one. A request that set `fallbacks` gets
/// one at each handoff, and must send it back in place on the next turn.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FallbackBlock {
    /// The model that declined: the requested model as the request spelled
    /// it, or an earlier fallback.
    pub from:    FallbackModel,
    /// The model that continues, by its resolved id.
    pub to:      FallbackModel,
    pub trigger: FallbackTrigger,
    #[serde(flatten)]
    pub extra:   Extra,
}

/// One side of a [`FallbackBlock`].
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FallbackModel {
    pub model: String,
    #[serde(flatten)]
    pub extra: Extra,
}

/// Why the model in [`FallbackBlock::from`] declined.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FallbackTrigger {
    /// `refusal`.
    #[serde(rename = "type")]
    pub kind:     String,
    /// The refusal's [`StopDetails::category`].
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub category: Field<String>,
    #[serde(flatten)]
    pub extra:    Extra,
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
/// `type`, `role`, and `container` are kept in `extra`.
///
/// `model` is the model that produced the message. After server-side
/// fallback, or a later turn that fallback's sticky routing sent straight to
/// the fallback model, that is not the model the request named.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Response {
    pub id:            String,
    pub model:         String,
    pub content:       Vec<ContentBlock>,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub stop_reason:   Field<String>,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub stop_sequence: Field<String>,
    /// Present on a `refusal`, and `null` for every other stop reason.
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub stop_details:  Field<StopDetails>,
    pub usage:         Usage,
    #[serde(flatten)]
    pub extra:         Extra,
}

/// Why a response stopped, on a [`Response`] or a [`MessageDeltaFields`].
/// A `refusal` from Anthropic's safety classifiers carries one; a model that
/// declines in its own words does not.
///
/// `explanation` is human-readable text whose wording is not stable.
/// `fallback_credit_token` is kept in `extra`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct StopDetails {
    /// `refusal`.
    #[serde(rename = "type")]
    pub kind:              String,
    /// The policy area, such as `cyber` or `bio`; `null` when the refusal
    /// maps to no named category.
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub category:          Field<String>,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub explanation:       Field<String>,
    /// A model to retry on, present only when the request set `fallbacks`.
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub recommended_model: Field<String>,
    #[serde(flatten)]
    pub extra:             Extra,
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
    /// Each sampling pass the request ran. Requests that set `fallbacks` or
    /// use the compaction beta get them.
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub iterations: Field<Vec<Iteration>>,
    #[serde(flatten)]
    pub extra: Extra,
}

/// One sampling pass in [`Usage::iterations`], each billed on its own.
///
/// Server-side fallback lists a model that declined as `message` and the one
/// that served the turn as `fallback_message`, each with its `model`.
/// Compaction lists its summary pass as `compaction`, without a `model`. The
/// counters beside `iterations` leave passes out: compaction passes always,
/// and after a fallback every pass but the one that produced the message. A
/// request's whole cost is the sum over its iterations, each at its own
/// model's rates.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Iteration {
    /// `message`, `fallback_message`, or `compaction`.
    #[serde(rename = "type")]
    pub kind:  String,
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub model: Field<String>,
    /// This pass's counters; [`Usage::token_counts`] normalizes them.
    #[serde(flatten)]
    pub usage: Usage,
}

impl Usage {
    /// These counts in the disjoint buckets that
    /// [`Offering::estimate_cost`](crate::catalog::Offering::estimate_cost)
    /// prices, normalized as the client normalizes Anthropic usage.
    ///
    /// A stream splits its usage: `message_start` carries this object with
    /// the input and cache counters, and each `message_delta` restates the
    /// cumulative counters in a [`DeltaUsage`]. Start from this and fold each
    /// delta in with [`DeltaUsage::fold_into`].
    pub fn token_counts(&self) -> TokenCounts {
        token_counts(&to_json(self))
    }
}

/// Normalizes one Anthropic usage object into the five disjoint buckets.
pub(crate) fn token_counts(usage: &Value) -> TokenCounts {
    let mut counts = TokenCounts::default();
    fold_usage(&mut counts, usage);
    counts
}

/// Folds whichever usage counters one wire object carries into a snapshot.
///
/// Anthropic's counters are **already disjoint**: `input_tokens` excludes both
/// `cache_read_input_tokens` and `cache_creation_input_tokens`, so nothing is
/// subtracted. Thinking tokens are billed inside `output_tokens` with no
/// separate counter, so `reasoning` stays 0 and `billable_output` still adds
/// up. Each field is assigned, not added, because a repeated counter is the
/// provider restating the same cumulative total. The cache writes' 1-hour
/// share comes from `cache_creation.ephemeral_1h_input_tokens`, which only
/// `message_start` and a whole response carry.
pub(crate) fn fold_usage(counts: &mut TokenCounts, usage: &Value) {
    if let Some(input) = usage.get("input_tokens").and_then(Value::as_u64) {
        counts.input = input;
    }
    if let Some(output) = usage.get("output_tokens").and_then(Value::as_u64) {
        counts.output = output;
    }
    if let Some(read) = usage.get("cache_read_input_tokens").and_then(Value::as_u64) {
        counts.cache_read = read;
    }
    if let Some(write) = usage
        .get("cache_creation_input_tokens")
        .and_then(Value::as_u64)
    {
        counts.cache_write = write;
    }
    if let Some(long) = usage
        .pointer("/cache_creation/ephemeral_1h_input_tokens")
        .and_then(Value::as_u64)
    {
        counts.long_cache_write = long;
    }
    counts.long_cache_write = counts.long_cache_write.min(counts.cache_write);
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
    /// Present with a `refusal` stop reason.
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub stop_details:  Field<StopDetails>,
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
    /// As [`Usage::iterations`]; a stream reports them on its final
    /// `message_delta`.
    #[serde(default, skip_serializing_if = "Field::is_absent")]
    pub iterations: Field<Vec<Iteration>>,
    #[serde(flatten)]
    pub extra: Extra,
}

impl DeltaUsage {
    /// Folds the counters this event carries into `counts`, which starts as
    /// `message_start`'s [`Usage::token_counts`].
    ///
    /// Each counter present here replaces the one in `counts`, because the
    /// event restates cumulative totals; an absent or `null` counter leaves
    /// it as it was.
    pub fn fold_into(&self, counts: &mut TokenCounts) {
        fold_usage(counts, &to_json(self));
    }
}

impl Response {
    /// How this response ended, which model served it, and what it was
    /// billed for.
    pub fn outcome(&self) -> Outcome {
        let mut outcome = Outcome::default();
        fold_message(self, outcome.anthropic());
        outcome
    }
}

impl StreamEvent {
    /// Folds what this event reports into `outcome`: the serving model from
    /// `message_start` and each `fallback` block, the stop reason, refusal,
    /// cumulative usage, and iterations from `message_delta`, and an `error`.
    /// Fold every event of a stream, in order.
    pub fn fold_into(&self, outcome: &mut Outcome) {
        let outcome = outcome.anthropic();
        match self {
            Self::MessageStart(start) => fold_message(&start.message, outcome),
            Self::ContentBlockStart(start) => {
                if let ContentBlock::Fallback(block) = &start.content_block {
                    outcome.push_fallback(fallback(block));
                }
            }
            Self::MessageDelta(delta) => {
                fold_stop(
                    delta.delta.stop_reason.as_option(),
                    &delta.delta.stop_details,
                    outcome,
                );
                delta.usage.fold_into(outcome.usage_mut());
                fold_iterations(&delta.usage.iterations, outcome);
                let extra = &delta.usage.extra;
                outcome.set_tier(extra.get("speed"), extra.get("service_tier"));
                outcome.set_provider_cost(Some(extra), None);
            }
            Self::Error(event) => outcome.set_error(Some(&event.error.kind), &event.error.message),
            _ => {}
        }
    }
}

/// Folds a whole response, or the start of a stream, into `outcome`.
fn fold_message(message: &Response, outcome: &mut Outcome) {
    for block in &message.content {
        if let ContentBlock::Fallback(block) = block {
            outcome.push_fallback(fallback(block));
        }
    }
    outcome.set_model(&message.model);
    fold_stop(
        message.stop_reason.as_option(),
        &message.stop_details,
        outcome,
    );
    outcome.set_usage(message.usage.token_counts());
    fold_iterations(&message.usage.iterations, outcome);
    let extra = &message.usage.extra;
    outcome.set_tier(
        extra.get("speed"),
        extra
            .get("service_tier")
            .or_else(|| message.extra.get("service_tier")),
    );
    outcome.set_provider_cost(Some(extra), message.extra.get("cost"));
}

fn fold_stop(reason: Option<&String>, details: &Field<StopDetails>, outcome: &mut Outcome) {
    let Some(reason) = reason else {
        return;
    };
    outcome.set_stop(reason, finish_reason(Some(reason)));
    if reason == "refusal" {
        let details = details.as_option();
        outcome.set_refusal(Refusal {
            category:    details.and_then(|details| details.category.as_option().cloned()),
            explanation: details.and_then(|details| details.explanation.as_option().cloned()),
        });
    }
}

fn fold_iterations(iterations: &Field<Vec<Iteration>>, outcome: &mut Outcome) {
    if let Field::Present(iterations) = iterations {
        outcome.set_iterations(iterations.iter().map(|iteration| {
            (
                iteration.kind.clone(),
                iteration.model.as_option().cloned(),
                iteration.usage.token_counts(),
            )
        }));
    }
}

fn fallback(block: &FallbackBlock) -> Fallback {
    Fallback {
        from:     block.from.model.clone(),
        to:       block.to.model.clone(),
        category: block.trigger.category.as_option().cloned(),
    }
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

impl ErrorResponse {
    /// An error of type `kind`, such as `authentication_error`, with no
    /// other fields.
    pub fn new(kind: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            error: ApiError {
                kind:    kind.into(),
                message: message.into(),
                extra:   Extra::new(),
            },
            extra: Extra::new(),
        }
    }
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
            json!({"type": "fallback", "from": {"model": "a"}, "to": {"model": "b"}, "trigger": {"type": "refusal", "category": null}}),
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
