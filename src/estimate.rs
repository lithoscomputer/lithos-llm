//! Local, provider-neutral input token estimation.
//!
//! Every function here is a synchronous heuristic: it sizes text at about four
//! characters per token, JSON and binary payloads at four bytes per token,
//! and media at a fixed floor when only a URL is known. The same input always
//! produces the same estimate. Use it where a count is needed before a call
//! is made or where no provider count exists — context-window projections,
//! truncation budgets, progress displays.
//!
//! A [`TokenEstimate`] is a local estimate by construction. Where a provider
//! offers a native count, `Client::count_input_tokens` is authoritative and
//! the client never substitutes an estimate for it.

use std::collections::BTreeSet;
use std::fmt;
use std::ops::{Add, AddAssign};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::types::{
    AudioContent, ContentPart, DocumentContent, ImageContent, MediaSource, Message, Request, Role,
    ToolDefinition, ToolDefinitionKind, ToolResult,
};

/// Tokens assumed for media whose size is unknown, or as an image's floor.
const MEDIA_FLOOR_TOKENS: u64 = 2000;

/// Something the estimator could only approximate.
///
/// An estimate is approximate everywhere; these kinds mark the inputs where it
/// is markedly less precise than the text heuristic, so an application can
/// tell a user why a projected total may be off.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum EstimateWarning {
    /// An image, audio, or document part was sized from its bytes or given a
    /// fixed floor. Providers tokenize media by their own rules.
    Media,
    /// A provider-native opaque part was measured as its JSON text.
    OpaqueContent,
    /// Provider options were measured as their JSON text.
    ProviderOptions,
}

impl EstimateWarning {
    /// The stable snake_case code, identical to the serialized form.
    pub fn code(self) -> &'static str {
        match self {
            Self::Media => "media",
            Self::OpaqueContent => "opaque_content",
            Self::ProviderOptions => "provider_options",
        }
    }
}

impl fmt::Display for EstimateWarning {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Media => "media content was sized from bytes, not tokenized",
            Self::OpaqueContent => "provider-native content was measured as JSON text",
            Self::ProviderOptions => "provider options were measured as JSON text",
        })
    }
}

/// A local token estimate with the reasons it may be imprecise.
///
/// Estimates add: summing the estimates of a request's parts gives the
/// estimate of the request, with the warnings of every part.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
#[must_use]
pub struct TokenEstimate {
    tokens:   u64,
    warnings: BTreeSet<EstimateWarning>,
}

impl TokenEstimate {
    /// An estimate of `tokens` with no warnings.
    pub fn exact(tokens: u64) -> Self {
        Self {
            tokens,
            warnings: BTreeSet::new(),
        }
    }

    pub fn tokens(&self) -> u64 {
        self.tokens
    }

    /// The distinct warnings, in a stable order.
    pub fn warnings(&self) -> impl Iterator<Item = EstimateWarning> + '_ {
        self.warnings.iter().copied()
    }

    pub fn has_warning(&self, warning: EstimateWarning) -> bool {
        self.warnings.contains(&warning)
    }

    fn add_tokens(&mut self, tokens: u64) {
        self.tokens = self.tokens.saturating_add(tokens);
    }

    fn warn(&mut self, warning: EstimateWarning) {
        self.warnings.insert(warning);
    }
}

impl AddAssign for TokenEstimate {
    fn add_assign(&mut self, other: Self) {
        self.add_tokens(other.tokens);
        self.warnings.extend(other.warnings);
    }
}

impl Add for TokenEstimate {
    type Output = Self;

    fn add(mut self, other: Self) -> Self {
        self += other;
        self
    }
}

/// Estimates tokens in plain text at four characters per token.
#[must_use]
pub fn text_tokens(text: &str) -> u64 {
    to_u64(text.chars().count()).div_ceil(4)
}

/// Estimates tokens in a binary payload at four bytes per token.
#[must_use]
pub fn byte_tokens(byte_len: usize) -> u64 {
    to_u64(byte_len).div_ceil(4)
}

/// Estimates tokens in a JSON value from its compact serialized form.
#[must_use]
pub fn json_tokens(value: &Value) -> u64 {
    serde_json::to_string(value).map_or(0, |json| byte_tokens(json.len()))
}

/// Estimates one message: its role and labels, plus every content part with
/// a small per-part framing cost.
pub fn message_tokens(message: &Message) -> TokenEstimate {
    let mut estimate = TokenEstimate::exact(4 + text_tokens(role_name(message.role())));
    estimate.add_tokens(message.name().map_or(0, text_tokens));
    estimate.add_tokens(message.tool_call_id().map_or(0, text_tokens));
    for part in message.content() {
        estimate.add_tokens(1);
        estimate += content_part_tokens(part);
    }
    estimate
}

/// Estimates one content part.
pub fn content_part_tokens(part: &ContentPart) -> TokenEstimate {
    match part {
        ContentPart::Text { text } => TokenEstimate::exact(text_tokens(text)),
        ContentPart::Image(image) => image_tokens(image),
        ContentPart::Audio(audio) => audio_tokens(audio),
        ContentPart::Document(document) => document_tokens(document),
        ContentPart::Reasoning(reasoning) => TokenEstimate::exact(
            text_tokens(&reasoning.text)
                + reasoning.signature.as_deref().map_or(0, text_tokens)
                + u64::from(reasoning.redacted),
        ),
        ContentPart::ToolCall(call) => {
            // The raw argument text is what a codec replays, and `arguments`
            // repeats it in parsed form, so only one of them counts.
            let arguments = call
                .raw_arguments
                .as_deref()
                .map_or_else(|| json_tokens(&call.arguments), text_tokens);
            TokenEstimate::exact(text_tokens(&call.id) + text_tokens(&call.name) + arguments)
        }
        ContentPart::ToolResult(result) => tool_result_tokens(result),
        ContentPart::Json { value } => TokenEstimate::exact(json_tokens(value)),
        ContentPart::Opaque { kind, data } => {
            let mut estimate = TokenEstimate::exact(text_tokens(kind) + json_tokens(data));
            estimate.warn(EstimateWarning::OpaqueContent);
            estimate
        }
    }
}

/// Estimates one tool definition: its name, description, and schema or
/// format, plus a small framing cost.
#[must_use]
pub fn tool_definition_tokens(tool: &ToolDefinition) -> u64 {
    let shape = match &tool.kind {
        ToolDefinitionKind::Function { input_schema } => json_tokens(input_schema),
        ToolDefinitionKind::Custom { format } => json_tokens(format),
    };
    8 + text_tokens(&tool.name) + text_tokens(&tool.description) + shape
}

/// Estimates the request controls that reach the prompt: tool choice,
/// response format, reasoning effort, and provider options.
///
/// Sampling settings, limits, timeouts, stop sequences, metadata, and cache
/// hints steer the call without adding prompt tokens, so they count nothing.
pub fn request_control_tokens(request: &Request) -> TokenEstimate {
    let mut estimate = TokenEstimate::default();
    if let Some(choice) = request.tool_choice() {
        estimate.add_tokens(serialized_tokens(choice));
    }
    if let Some(format) = request.response_format() {
        estimate.add_tokens(serialized_tokens(format));
    }
    if let Some(effort) = request.reasoning_effort() {
        estimate.add_tokens(serialized_tokens(&effort));
    }
    if !request.provider_options().is_empty() {
        estimate.add_tokens(serialized_tokens(request.provider_options()));
        estimate.warn(EstimateWarning::ProviderOptions);
    }
    estimate
}

/// Estimates a whole request: every message, every tool definition, and the
/// request controls.
pub fn request_tokens(request: &Request) -> TokenEstimate {
    let mut estimate = TokenEstimate::default();
    for message in request.messages() {
        estimate += message_tokens(message);
    }
    for tool in request.tools() {
        estimate.add_tokens(tool_definition_tokens(tool));
    }
    estimate += request_control_tokens(request);
    estimate
}

fn image_tokens(image: &ImageContent) -> TokenEstimate {
    let mut estimate = media_source_tokens(&image.source);
    estimate.add_tokens(image.detail.as_deref().map_or(0, text_tokens));
    // Inline images are sized from their bytes but never below the floor,
    // because providers charge for the rendered image, not the file.
    estimate.add_tokens(
        inline_byte_len(&image.source).map_or(MEDIA_FLOOR_TOKENS, |len| {
            byte_tokens(len).max(MEDIA_FLOOR_TOKENS)
        }),
    );
    estimate
}

fn audio_tokens(audio: &AudioContent) -> TokenEstimate {
    let mut estimate = media_source_tokens(&audio.source);
    estimate.add_tokens(inline_byte_len(&audio.source).map_or(MEDIA_FLOOR_TOKENS, byte_tokens));
    estimate
}

fn document_tokens(document: &DocumentContent) -> TokenEstimate {
    let mut estimate = media_source_tokens(&document.source);
    estimate.add_tokens(document.name.as_deref().map_or(0, text_tokens));
    estimate.add_tokens(inline_byte_len(&document.source).map_or(MEDIA_FLOOR_TOKENS, byte_tokens));
    estimate
}

/// The tokens a media source's own labels cost, with the media warning.
fn media_source_tokens(source: &MediaSource) -> TokenEstimate {
    let mut estimate = TokenEstimate::exact(source.media_type().map_or(0, text_tokens));
    if let MediaSource::Url { url, .. } = source {
        estimate.add_tokens(text_tokens(url));
    }
    estimate.warn(EstimateWarning::Media);
    estimate
}

/// The decoded size of an inline payload, or `None` for a URL.
fn inline_byte_len(source: &MediaSource) -> Option<usize> {
    // Base64 spends four characters on every three bytes.
    source.base64_data().map(|data| data.len() / 4 * 3)
}

fn tool_result_tokens(result: &ToolResult) -> TokenEstimate {
    let mut estimate = TokenEstimate::exact(
        text_tokens(&result.tool_call_id)
            + result.name.as_deref().map_or(0, text_tokens)
            + u64::from(result.is_error),
    );
    for part in &result.content {
        estimate += content_part_tokens(part);
    }
    estimate
}

/// Measures a control by its serialized form, so the estimate follows the
/// wire shape rather than a hand-kept table of variant sizes.
fn serialized_tokens<T: Serialize>(value: &T) -> u64 {
    serde_json::to_value(value).map_or(0, |value| json_tokens(&value))
}

fn role_name(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::Developer => "developer",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

fn to_u64(len: usize) -> u64 {
    u64::try_from(len).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        EstimateWarning, TokenEstimate, byte_tokens, content_part_tokens, message_tokens,
        request_control_tokens, request_tokens, text_tokens, tool_definition_tokens,
    };
    use crate::Request;
    use crate::types::{
        AudioContent, ContentPart, DocumentContent, ImageContent, MediaSource, Message,
        ReasoningEffort, RequestBuildError, ResponseFormat, Role, ToolCall, ToolChoice,
        ToolDefinition, ToolResult,
    };

    fn request(messages: Vec<Message>) -> Result<Request, RequestBuildError> {
        let mut builder = Request::builder().model("test/model");
        for message in messages {
            builder = builder.message(message);
        }
        builder.build()
    }

    fn user(text: &str) -> Message {
        Message::text(Role::User, text)
    }

    #[test]
    fn text_rounds_up_to_whole_tokens() {
        assert_eq!(text_tokens(""), 0);
        assert_eq!(text_tokens("abcd"), 1);
        assert_eq!(text_tokens("abcde"), 2);
        assert_eq!(byte_tokens(9), 3);
    }

    #[test]
    fn text_counts_characters_not_bytes() {
        assert_eq!(
            text_tokens("éééé"),
            1,
            "four two-byte characters are one token"
        );
    }

    #[test]
    fn a_text_message_costs_its_framing_plus_its_text() {
        let estimate = message_tokens(&user("hello world"));

        // 4 framing + "user" (1) + 1 per part + "hello world" (3).
        assert_eq!(estimate.tokens(), 9);
        assert_eq!(estimate.warnings().count(), 0);
    }

    #[test]
    fn message_labels_add_to_the_estimate() {
        let plain = message_tokens(&Message::text(Role::Tool, "done"));
        let labelled = message_tokens(
            &Message::text(Role::Tool, "done")
                .with_name("lookup")
                .with_tool_call_id("call_1234"),
        );

        assert!(labelled.tokens() > plain.tokens());
    }

    #[test]
    fn a_tool_definition_increases_the_request_estimate() -> Result<(), RequestBuildError> {
        let without = request_tokens(&request(vec![user("hello")])?);
        let with = request_tokens(
            &Request::builder()
                .model("test/model")
                .user("hello")
                .tool(ToolDefinition::function(
                    "search",
                    "Search files",
                    json!({"type": "object", "properties": {"query": {"type": "string"}}}),
                ))
                .build()?,
        );

        assert!(with.tokens() > without.tokens());
        Ok(())
    }

    #[test]
    fn custom_tools_measure_their_format() {
        let function = ToolDefinition::function("edit", "Edits", json!({"type": "object"}));
        let custom = ToolDefinition::custom(
            "edit",
            "Edits",
            json!({"type": "grammar", "syntax": "lark", "definition": "start: line+"}),
        );

        assert!(tool_definition_tokens(&custom) > tool_definition_tokens(&function));
    }

    #[test]
    fn controls_that_reach_the_prompt_add_tokens() -> Result<(), RequestBuildError> {
        let plain = request_control_tokens(&request(vec![user("hello")])?);
        let controlled = request_control_tokens(
            &Request::builder()
                .model("test/model")
                .user("hello")
                .tool(ToolDefinition::function("search", "Search", json!({})))
                .tool_choice(ToolChoice::Required)
                .response_format(ResponseFormat::JsonSchema {
                    name:   "answer".to_owned(),
                    schema: json!({"type": "object", "properties": {"answer": {"type": "string"}}}),
                })
                .reasoning_effort(ReasoningEffort::High)
                .build()?,
        );

        assert_eq!(plain.tokens(), 0);
        assert!(controlled.tokens() > 0);
        assert_eq!(controlled.warnings().count(), 0);
        Ok(())
    }

    #[test]
    fn provider_options_are_measured_with_a_warning() -> Result<(), RequestBuildError> {
        let estimate = request_control_tokens(
            &Request::builder()
                .model("test/model")
                .user("hello")
                .provider_option("gemini", "cached_content", json!("cachedContents/1"))
                .build()?,
        );

        assert!(estimate.tokens() > 0);
        assert!(estimate.has_warning(EstimateWarning::ProviderOptions));
        Ok(())
    }

    #[test]
    fn media_is_sized_with_a_floor_and_a_warning() {
        let estimate = message_tokens(&Message::new(Role::User, [
            ContentPart::Image(ImageContent {
                source: MediaSource::url_with_media_type(
                    "https://example.test/image.png",
                    "image/png",
                ),
                detail: Some("high".to_owned()),
            }),
            ContentPart::Document(DocumentContent {
                source: MediaSource::base64("A".repeat(5464), "application/pdf"),
                name:   Some("doc.pdf".to_owned()),
            }),
        ]));

        // A URL image takes the 2000-token floor; a 4098-byte document takes
        // its byte estimate of 1025.
        assert!(estimate.tokens() >= 3025);
        assert!(estimate.has_warning(EstimateWarning::Media));
    }

    #[test]
    fn a_small_inline_image_is_never_below_the_floor() {
        let estimate = content_part_tokens(&ContentPart::Image(ImageContent::new(
            MediaSource::base64("QUJD", "image/png"),
        )));

        assert!(estimate.tokens() >= 2000);
    }

    #[test]
    fn opaque_content_is_measured_with_a_warning() {
        let estimate = content_part_tokens(&ContentPart::opaque(
            "openai.reasoning",
            json!({"id": "rs_123", "summary": []}),
        ));

        assert!(estimate.tokens() > 0);
        assert!(estimate.has_warning(EstimateWarning::OpaqueContent));
    }

    #[test]
    fn a_tool_call_counts_its_arguments_once() {
        let parsed = ToolCall::function("call_1", "lookup", json!({"query": "rust"}));
        let mut replayed = parsed.clone();
        replayed.raw_arguments = Some("{\"query\":\"rust\"}".to_owned());

        assert_eq!(
            content_part_tokens(&ContentPart::ToolCall(parsed)).tokens(),
            content_part_tokens(&ContentPart::ToolCall(replayed)).tokens(),
        );
    }

    #[test]
    fn media_inside_a_tool_result_carries_its_warning() {
        let estimate = content_part_tokens(&ContentPart::ToolResult(ToolResult {
            tool_call_id: "call_1".to_owned(),
            name:         None,
            content:      vec![ContentPart::Image(ImageContent::new(MediaSource::base64(
                "QUJD",
                "image/png",
            )))],
            is_error:     true,
        }));

        assert!(estimate.has_warning(EstimateWarning::Media));
    }

    #[test]
    fn estimates_add_and_merge_their_warnings() {
        let mut total = TokenEstimate::exact(1);
        total += content_part_tokens(&ContentPart::opaque("x.y", json!(1)));
        total += content_part_tokens(&ContentPart::opaque("x.z", json!(2)));
        let media = content_part_tokens(&ContentPart::Audio(AudioContent::new(MediaSource::url(
            "https://example.test/a.wav",
        ))));
        let total = total + media;

        assert_eq!(total.warnings().collect::<Vec<_>>(), [
            EstimateWarning::Media,
            EstimateWarning::OpaqueContent,
        ]);
        assert!(total.tokens() > 2000);
    }

    #[test]
    fn the_estimate_is_deterministic() -> Result<(), RequestBuildError> {
        let request = Request::builder()
            .model("test/model")
            .system("Be brief.")
            .user("repeatable")
            .tool(ToolDefinition::function("t", "d", json!({"a": 1, "b": 2})))
            .build()?;

        assert_eq!(request_tokens(&request), request_tokens(&request));
        Ok(())
    }

    #[test]
    fn warning_codes_match_their_serialized_form() -> Result<(), serde_json::Error> {
        for warning in [
            EstimateWarning::Media,
            EstimateWarning::OpaqueContent,
            EstimateWarning::ProviderOptions,
        ] {
            assert_eq!(
                serde_json::to_value(warning)?,
                json!(warning.code()),
                "{warning:?}"
            );
        }
        Ok(())
    }
}
