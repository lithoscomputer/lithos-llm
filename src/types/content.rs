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
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Message {
    role:    Role,
    content: Vec<ContentPart>,
}

impl Message {
    pub fn new(role: Role, content: impl IntoIterator<Item = ContentPart>) -> Self {
        Self {
            role,
            content: content.into_iter().collect(),
        }
    }

    pub fn text(role: Role, text: impl Into<String>) -> Self {
        Self::new(role, [ContentPart::Text { text: text.into() }])
    }

    pub fn role(&self) -> Role {
        self.role
    }

    pub fn content(&self) -> &[ContentPart] {
        &self.content
    }
}

/// One provider-neutral part of a message or response.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ContentPart {
    Text { text: String },
    Image(ImageContent),
    Audio(AudioContent),
    Document(DocumentContent),
    Reasoning(ReasoningContent),
    ToolCall(ToolCall),
    ToolResult(ToolResult),
}

/// An image URL or encoded image payload.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ImageContent {
    pub source:     String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail:     Option<String>,
}

/// Encoded audio content.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct AudioContent {
    pub data:       String,
    pub media_type: String,
}

/// Encoded document content.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DocumentContent {
    pub data:       String,
    pub media_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name:       Option<String>,
}

/// Reasoning content returned by a provider.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ReasoningContent {
    pub text:      String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

/// A tool exposed to the model.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ToolDefinition {
    pub name:         String,
    pub description:  String,
    pub input_schema: Value,
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

/// A tool invocation requested by the model.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ToolCall {
    pub id:        String,
    pub name:      String,
    pub arguments: Value,
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
