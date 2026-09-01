use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The author of a message.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Role {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

/// One message in a model conversation.
///
/// `name` and `tool_call_id` are optional message-level labels that a few
/// providers accept. They are omitted from the serialized form when unset, so
/// documents written before they existed still parse.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Message {
    role:         Role,
    content:      Vec<ContentPart>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name:         Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

impl Message {
    /// Creates a message from a role and its content parts.
    pub fn new(role: Role, content: impl IntoIterator<Item = ContentPart>) -> Self {
        Self {
            role,
            content: content.into_iter().collect(),
            name: None,
            tool_call_id: None,
        }
    }

    /// Creates a message holding a single text part.
    pub fn text(role: Role, text: impl Into<String>) -> Self {
        Self::new(role, [ContentPart::Text { text: text.into() }])
    }

    /// Sets the participant name carried alongside the message.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the tool call this message answers.
    ///
    /// This mirrors the identifier inside a [`ToolResult`] part. Codecs use it
    /// only when the content itself carries no tool result.
    #[must_use]
    pub fn with_tool_call_id(mut self, id: impl Into<String>) -> Self {
        self.tool_call_id = Some(id.into());
        self
    }

    pub fn role(&self) -> Role {
        self.role
    }

    pub fn content(&self) -> &[ContentPart] {
        &self.content
    }

    /// The participant name carried alongside the message, when one was set.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// The tool call this message answers, when one was set.
    pub fn tool_call_id(&self) -> Option<&str> {
        self.tool_call_id.as_deref()
    }
}

/// One provider-neutral part of a message or response.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ContentPart {
    Text {
        text: String,
    },
    Image(ImageContent),
    Audio(AudioContent),
    Document(DocumentContent),
    Reasoning(ReasoningContent),
    ToolCall(ToolCall),
    ToolResult(ToolResult),
    /// Structured JSON output that is not text.
    Json {
        value: Value,
    },
    /// Provider-native content preserved verbatim for lossless replay.
    ///
    /// `kind` MUST be provider-namespaced as `<namespace>.<name>`, for example
    /// `openai.reasoning`. A codec consumes only the kinds in its own provider
    /// namespace and re-emits them on the wire. Every other codec ignores an
    /// unrecognized opaque part instead of failing, so a conversation carrying
    /// one provider's replay data can still fail over to another provider.
    Opaque {
        kind: String,
        data: Value,
    },
}

impl ContentPart {
    /// Creates a provider-native part preserved verbatim for replay.
    ///
    /// `kind` must be provider-namespaced as `<namespace>.<name>`, such as
    /// `openai.reasoning`. An un-namespaced kind is accepted but no codec will
    /// claim it, so it is dropped on every wire format.
    pub fn opaque(kind: impl Into<String>, data: Value) -> Self {
        Self::Opaque {
            kind: kind.into(),
            data,
        }
    }

    /// The namespace of an opaque part, that is the text before the first `.`.
    ///
    /// Returns `None` for any other part and for an opaque kind that carries no
    /// namespace.
    pub fn opaque_namespace(&self) -> Option<&str> {
        match self {
            Self::Opaque { kind, .. } => kind.split_once('.').map(|(namespace, _)| namespace),
            _ => None,
        }
    }
}

/// Where binary or remote media content comes from.
///
/// A provider either fetches the media itself from a URL or receives the bytes
/// inline. Codecs that can only send bytes read [`MediaSource::base64_data`]
/// and report an invalid request when handed a URL they cannot fetch.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum MediaSource {
    Url {
        url:        String,
        /// The declared media type of the file behind the URL.
        ///
        /// Some protocols require it — Gemini's `fileData.mimeType` — and no
        /// provider can be trusted to sniff one. Codecs send it when the
        /// protocol has a field for it and otherwise ignore it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        media_type: Option<String>,
    },
    Base64 {
        data:       String,
        media_type: String,
    },
}

impl MediaSource {
    /// Creates a source the provider fetches from a URL.
    pub fn url(url: impl Into<String>) -> Self {
        Self::Url {
            url:        url.into(),
            media_type: None,
        }
    }

    /// Creates a URL source with the media type of the file behind it.
    pub fn url_with_media_type(url: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self::Url {
            url:        url.into(),
            media_type: Some(media_type.into()),
        }
    }

    /// Creates an inline source from base64 data and its media type.
    pub fn base64(data: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self::Base64 {
            data:       data.into(),
            media_type: media_type.into(),
        }
    }

    /// Parses a `data:<media-type>;base64,<data>` URI into
    /// [`MediaSource::Base64`].
    ///
    /// Every other input, including a bare base64 payload, becomes
    /// [`MediaSource::Url`].
    pub fn parse(source: &str) -> Self {
        if let Some(rest) = source.strip_prefix("data:")
            && let Some((parameters, data)) = rest.split_once(',')
            && let Some(media_type) = parameters.strip_suffix(";base64")
        {
            return Self::base64(data, media_type);
        }
        Self::url(source)
    }

    /// The declared media type, when the source carries one.
    ///
    /// An inline source always has one; a URL source has one only when the
    /// caller declared it.
    pub fn media_type(&self) -> Option<&str> {
        match self {
            Self::Url { media_type, .. } => media_type.as_deref(),
            Self::Base64 { media_type, .. } => Some(media_type),
        }
    }

    /// The base64 payload of an inline source.
    pub fn base64_data(&self) -> Option<&str> {
        match self {
            Self::Url { .. } => None,
            Self::Base64 { data, .. } => Some(data),
        }
    }
}

/// An image supplied to or returned by a model.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ImageContent {
    pub source: MediaSource,
    /// The provider-specific detail hint, such as `low` or `high`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl ImageContent {
    /// Creates image content with no detail hint.
    pub fn new(source: MediaSource) -> Self {
        Self {
            source,
            detail: None,
        }
    }
}

/// Audio supplied to or returned by a model.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct AudioContent {
    pub source: MediaSource,
}

impl AudioContent {
    /// Creates audio content.
    pub fn new(source: MediaSource) -> Self {
        Self { source }
    }
}

/// A document supplied to a model.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DocumentContent {
    pub source: MediaSource,
    /// The document file name, which only some providers send on the wire.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name:   Option<String>,
}

impl DocumentContent {
    /// Creates document content with no file name.
    pub fn new(source: MediaSource) -> Self {
        Self { source, name: None }
    }
}

/// Reasoning content returned by a provider.
///
/// `signature` is the provider's verification token for the reasoning block.
/// `redacted` marks reasoning whose text the provider withheld; the remaining
/// `text` is then an opaque payload that must be echoed back unchanged.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ReasoningContent {
    pub text:             String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature:        Option<String>,
    /// The signature family that minted `signature`, when one did.
    ///
    /// A verification signature is only valid at the provider family that
    /// produced it — `anthropic` covers the Anthropic and Bedrock Converse
    /// protocols, which both carry Claude-minted signatures, and `gemini`
    /// covers thought signatures. Codecs record the family at decode time and
    /// skip a foreign signature at encode time, so a conversation that failed
    /// over between providers does not replay a signature the target rejects.
    /// `None` on a signed part means the origin is unknown — a history
    /// persisted before this field existed — and the signature replays as
    /// before.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature_origin: Option<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub redacted:         bool,
}

/// A tool exposed to the model.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ToolDefinition {
    pub name:        String,
    pub description: String,
    pub kind:        ToolDefinitionKind,
}

impl ToolDefinition {
    /// Creates a function tool described by a JSON Schema.
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
    ) -> Self {
        Self {
            name:        name.into(),
            description: description.into(),
            kind:        ToolDefinitionKind::Function { input_schema },
        }
    }

    /// Creates a custom tool described by a provider-specific format.
    pub fn custom(name: impl Into<String>, description: impl Into<String>, format: Value) -> Self {
        Self {
            name:        name.into(),
            description: description.into(),
            kind:        ToolDefinitionKind::Custom { format },
        }
    }

    /// Whether this definition is a custom tool.
    pub fn is_custom(&self) -> bool {
        matches!(self.kind, ToolDefinitionKind::Custom { .. })
    }
}

/// What kind of tool a [`ToolDefinition`] describes.
///
/// The kind is a typed field. It is never carried by a reserved or magic key
/// inside the JSON Schema, so nothing leaks onto the wire.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ToolDefinitionKind {
    /// A tool whose arguments are JSON validated by `input_schema`.
    Function { input_schema: Value },
    /// A tool whose input is free-form text described by a provider format,
    /// such as a grammar.
    Custom { format: Value },
}

/// The model's permitted tool-selection behavior.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
    Tool {
        name: String,
    },
}

impl ToolChoice {
    /// Whether this choice makes a tool call mandatory.
    ///
    /// `Required` and a named `Tool` force a call; `Auto` and `None` leave
    /// the model free to answer in prose. Some models take no forced choice
    /// at all, which the catalog records as
    /// [`ModelCapabilities::forced_tool_choice`](crate::catalog::ModelCapabilities::forced_tool_choice).
    pub fn is_forced(&self) -> bool {
        matches!(self, Self::Required | Self::Tool { .. })
    }
}

/// A tool invocation requested by the model.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ToolCall {
    pub id:                String,
    pub name:              String,
    /// The parsed arguments. A custom call holds its free-form input as a JSON
    /// string.
    pub arguments:         Value,
    #[serde(default)]
    pub kind:              ToolCallKind,
    /// The provider's original argument string, when it supplied one.
    ///
    /// Codecs replay this text verbatim rather than re-serializing
    /// `arguments`, because reordered keys break provider prompt caches and
    /// because a malformed argument string must survive parsing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_arguments:     Option<String>,
    /// Provider-namespaced replay data, keyed by namespace.
    ///
    /// A codec reads only the entry matching its own provider namespace and
    /// ignores every other entry, so one call can carry replay data for
    /// several failover candidates.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub provider_metadata: BTreeMap<String, Value>,
}

impl ToolCall {
    /// Creates a function call from parsed JSON arguments.
    pub fn function(id: impl Into<String>, name: impl Into<String>, arguments: Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments,
            kind: ToolCallKind::Function,
            raw_arguments: None,
            provider_metadata: BTreeMap::new(),
        }
    }

    /// Creates a custom call from free-form input text.
    ///
    /// The input is stored both as a JSON string in `arguments` and verbatim in
    /// `raw_arguments`, because custom tool input is not JSON.
    pub fn custom(
        id: impl Into<String>,
        name: impl Into<String>,
        input: impl Into<String>,
    ) -> Self {
        let input = input.into();
        Self {
            id:                id.into(),
            name:              name.into(),
            arguments:         Value::String(input.clone()),
            kind:              ToolCallKind::Custom,
            raw_arguments:     Some(input),
            provider_metadata: BTreeMap::new(),
        }
    }
}

/// What kind of tool a [`ToolCall`] invokes.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ToolCallKind {
    #[default]
    Function,
    Custom,
}

/// The result of an application-executed tool call.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name:         Option<String>,
    pub content:      Vec<ContentPart>,
    #[serde(default)]
    pub is_error:     bool,
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use serde::Serialize;
    use serde::de::DeserializeOwned;
    use serde_json::{Value, json};

    use super::{
        AudioContent, ContentPart, DocumentContent, ImageContent, MediaSource, Message,
        ReasoningContent, Role, ToolCall, ToolCallKind, ToolDefinition, ToolDefinitionKind,
        ToolResult,
    };

    fn round_trip<T>(value: &T) -> Result<T, Box<dyn StdError>>
    where
        T: DeserializeOwned + Serialize,
    {
        let encoded = serde_json::to_string(value)?;
        Ok(serde_json::from_str(&encoded)?)
    }

    fn every_variant() -> Vec<ContentPart> {
        vec![
            ContentPart::Text {
                text: "hello".to_owned(),
            },
            ContentPart::Image(ImageContent {
                source: MediaSource::url("https://example.com/cat.png"),
                detail: Some("high".to_owned()),
            }),
            ContentPart::Audio(AudioContent::new(MediaSource::base64("QUJD", "audio/wav"))),
            ContentPart::Document(DocumentContent {
                source: MediaSource::base64("REVG", "application/pdf"),
                name:   Some("report.pdf".to_owned()),
            }),
            ContentPart::Reasoning(ReasoningContent {
                text:             "step one".to_owned(),
                signature:        Some("sig".to_owned()),
                signature_origin: None,
                redacted:         false,
            }),
            ContentPart::ToolCall(ToolCall::function(
                "call_1",
                "lookup",
                json!({ "query": "rust" }),
            )),
            ContentPart::ToolResult(ToolResult {
                tool_call_id: "call_1".to_owned(),
                name:         Some("lookup".to_owned()),
                content:      vec![ContentPart::Text {
                    text: "found".to_owned(),
                }],
                is_error:     false,
            }),
            ContentPart::Json {
                value: json!({ "score": 0.5 }),
            },
            ContentPart::opaque("openai.reasoning", json!({ "id": "rs_1" })),
        ]
    }

    #[test]
    fn every_content_part_variant_survives_a_round_trip() -> Result<(), Box<dyn StdError>> {
        for part in every_variant() {
            assert_eq!(round_trip(&part)?, part);
        }

        Ok(())
    }

    #[test]
    fn message_preserves_name_and_tool_call_id() -> Result<(), Box<dyn StdError>> {
        let message = Message::text(Role::Tool, "done")
            .with_name("apply_patch")
            .with_tool_call_id("call_7");

        let decoded = round_trip(&message)?;

        assert_eq!(decoded, message);
        assert_eq!(decoded.name(), Some("apply_patch"));
        assert_eq!(decoded.tool_call_id(), Some("call_7"));
        Ok(())
    }

    #[test]
    fn message_without_labels_omits_those_keys() -> Result<(), Box<dyn StdError>> {
        let message = Message::text(Role::User, "hello");

        let encoded: Value = serde_json::to_value(&message)?;
        let object = encoded.as_object().ok_or("expected a JSON object")?;

        assert!(!object.contains_key("name"));
        assert!(!object.contains_key("tool_call_id"));
        assert_eq!(round_trip(&message)?, message);
        Ok(())
    }

    #[test]
    fn url_media_preserves_image_detail() -> Result<(), Box<dyn StdError>> {
        let image = ImageContent {
            source: MediaSource::url("https://example.com/cat.png"),
            detail: Some("low".to_owned()),
        };

        let decoded = round_trip(&image)?;

        assert_eq!(decoded, image);
        assert_eq!(decoded.detail.as_deref(), Some("low"));
        assert_eq!(decoded.source.media_type(), None);
        assert_eq!(decoded.source.base64_data(), None);
        Ok(())
    }

    #[test]
    fn base64_media_preserves_media_type_and_document_name() -> Result<(), Box<dyn StdError>> {
        let document = DocumentContent {
            source: MediaSource::base64("REVG", "application/pdf"),
            name:   Some("report.pdf".to_owned()),
        };

        let decoded = round_trip(&document)?;

        assert_eq!(decoded, document);
        assert_eq!(decoded.name.as_deref(), Some("report.pdf"));
        assert_eq!(decoded.source.media_type(), Some("application/pdf"));
        assert_eq!(decoded.source.base64_data(), Some("REVG"));
        Ok(())
    }

    #[test]
    fn parses_data_uris_urls_and_bare_payloads() {
        assert_eq!(
            MediaSource::parse("data:image/png;base64,QUJD"),
            MediaSource::base64("QUJD", "image/png")
        );
        assert_eq!(
            MediaSource::parse("https://example.com/cat.png"),
            MediaSource::url("https://example.com/cat.png")
        );
        assert_eq!(MediaSource::parse("QUJD"), MediaSource::url("QUJD"));
    }

    #[test]
    fn reasoning_signature_and_redaction_survive_a_round_trip() -> Result<(), Box<dyn StdError>> {
        let signed = ReasoningContent {
            text:             "step one".to_owned(),
            signature:        Some("sig".to_owned()),
            signature_origin: None,
            redacted:         false,
        };
        let redacted = ReasoningContent {
            text:             "opaque payload".to_owned(),
            signature:        None,
            signature_origin: None,
            redacted:         true,
        };

        assert_eq!(round_trip(&signed)?, signed);
        assert_eq!(round_trip(&redacted)?, redacted);

        let encoded: Value = serde_json::to_value(&signed)?;
        let object = encoded.as_object().ok_or("expected a JSON object")?;
        assert!(!object.contains_key("redacted"));
        Ok(())
    }

    #[test]
    fn tool_calls_preserve_kind_arguments_and_metadata() -> Result<(), Box<dyn StdError>> {
        let mut call = ToolCall::function("call_1", "lookup", json!({ "query": "rust" }));
        call.raw_arguments = Some("{\"query\":\"rust\"}".to_owned());
        call.provider_metadata
            .insert("openai".to_owned(), json!({ "id": "fc_1" }));

        let decoded = round_trip(&call)?;

        assert_eq!(decoded, call);
        assert_eq!(decoded.kind, ToolCallKind::Function);
        assert_eq!(decoded.arguments, json!({ "query": "rust" }));
        assert_eq!(
            decoded.raw_arguments.as_deref(),
            Some("{\"query\":\"rust\"}")
        );
        assert_eq!(
            decoded.provider_metadata.get("openai"),
            Some(&json!({ "id": "fc_1" }))
        );
        Ok(())
    }

    #[test]
    fn custom_tool_calls_keep_their_free_form_input() -> Result<(), Box<dyn StdError>> {
        let call = ToolCall::custom("call_2", "apply_patch", "*** Begin Patch");

        let decoded = round_trip(&call)?;

        assert_eq!(decoded, call);
        assert_eq!(decoded.kind, ToolCallKind::Custom);
        assert_eq!(
            decoded.arguments,
            Value::String("*** Begin Patch".to_owned())
        );
        assert_eq!(decoded.raw_arguments.as_deref(), Some("*** Begin Patch"));
        Ok(())
    }

    #[test]
    fn tool_definitions_round_trip_both_kinds() -> Result<(), Box<dyn StdError>> {
        let function = ToolDefinition::function(
            "lookup",
            "Looks a term up",
            json!({ "type": "object", "properties": {} }),
        );
        let custom = ToolDefinition::custom(
            "apply_patch",
            "Edits files",
            json!({ "type": "grammar", "syntax": "lark" }),
        );

        assert_eq!(round_trip(&function)?, function);
        assert_eq!(round_trip(&custom)?, custom);

        assert!(!function.is_custom());
        assert!(custom.is_custom());
        assert_eq!(function.kind, ToolDefinitionKind::Function {
            input_schema: json!({ "type": "object", "properties": {} }),
        });
        assert_eq!(custom.kind, ToolDefinitionKind::Custom {
            format: json!({ "type": "grammar", "syntax": "lark" }),
        });
        Ok(())
    }

    #[test]
    fn tool_definitions_carry_no_reserved_schema_keys() -> Result<(), Box<dyn StdError>> {
        let custom = ToolDefinition::custom("apply_patch", "Edits files", json!({ "a": 1 }));

        let encoded = serde_json::to_string(&custom)?;

        assert!(!encoded.contains("x-fabro"));
        Ok(())
    }

    #[test]
    fn opaque_namespace_reads_the_kind_prefix() {
        let opaque = ContentPart::opaque("openai.reasoning", json!({}));
        let text = ContentPart::Text {
            text: "hello".to_owned(),
        };

        assert_eq!(opaque.opaque_namespace(), Some("openai"));
        assert_eq!(text.opaque_namespace(), None);
        assert_eq!(
            ContentPart::opaque("unnamespaced", json!({})).opaque_namespace(),
            None
        );
    }
}
