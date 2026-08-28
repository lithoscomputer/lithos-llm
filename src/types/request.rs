use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use super::{Message, Role, ToolChoice, ToolDefinition};

/// Requested reasoning depth, when a provider supports it.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ReasoningEffort {
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
}

/// Requested latency or cost preference.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Speed {
    Fast,
    Balanced,
    Economical,
}

/// Requested response shape.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema { name: String, schema: Value },
}

/// A provider-neutral inference request.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Request {
    model:             String,
    messages:          Vec<Message>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    tools:             Vec<ToolDefinition>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_choice:       Option<ToolChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    response_format:   Option<ResponseFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    temperature:       Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    top_p:             Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reasoning_effort:  Option<ReasoningEffort>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    speed:             Option<Speed>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "duration_millis"
    )]
    timeout:           Option<Duration>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    provider_options:  BTreeMap<String, Value>,
}

impl Request {
    pub fn builder() -> RequestBuilder {
        RequestBuilder::default()
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn tools(&self) -> &[ToolDefinition] {
        &self.tools
    }

    pub fn tool_choice(&self) -> Option<&ToolChoice> {
        self.tool_choice.as_ref()
    }

    pub fn response_format(&self) -> Option<&ResponseFormat> {
        self.response_format.as_ref()
    }

    pub fn max_output_tokens(&self) -> Option<u32> {
        self.max_output_tokens
    }

    pub fn temperature(&self) -> Option<f32> {
        self.temperature
    }

    pub fn top_p(&self) -> Option<f32> {
        self.top_p
    }

    pub fn reasoning_effort(&self) -> Option<ReasoningEffort> {
        self.reasoning_effort
    }

    pub fn speed(&self) -> Option<Speed> {
        self.speed
    }

    pub fn timeout(&self) -> Option<Duration> {
        self.timeout
    }

    pub fn provider_options(&self) -> &BTreeMap<String, Value> {
        &self.provider_options
    }
}

/// Builds and validates an inference request.
#[derive(Default)]
#[must_use]
pub struct RequestBuilder {
    model:             Option<String>,
    messages:          Vec<Message>,
    tools:             Vec<ToolDefinition>,
    tool_choice:       Option<ToolChoice>,
    response_format:   Option<ResponseFormat>,
    max_output_tokens: Option<u32>,
    temperature:       Option<f32>,
    top_p:             Option<f32>,
    reasoning_effort:  Option<ReasoningEffort>,
    speed:             Option<Speed>,
    timeout:           Option<Duration>,
    provider_options:  BTreeMap<String, Value>,
}

impl RequestBuilder {
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    pub fn system(self, text: impl Into<String>) -> Self {
        self.message(Message::text(Role::System, text))
    }

    pub fn developer(self, text: impl Into<String>) -> Self {
        self.message(Message::text(Role::Developer, text))
    }

    pub fn user(self, text: impl Into<String>) -> Self {
        self.message(Message::text(Role::User, text))
    }

    pub fn tool(mut self, tool: ToolDefinition) -> Self {
        self.tools.push(tool);
        self
    }

    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    pub fn response_format(mut self, format: ResponseFormat) -> Self {
        self.response_format = Some(format);
        self
    }

    pub fn max_output_tokens(mut self, tokens: u32) -> Self {
        self.max_output_tokens = Some(tokens);
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }

    pub fn speed(mut self, speed: Speed) -> Self {
        self.speed = Some(speed);
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    pub fn provider_option(mut self, namespace: impl Into<String>, value: Value) -> Self {
        self.provider_options.insert(namespace.into(), value);
        self
    }

    pub fn build(self) -> Result<Request, RequestBuildError> {
        let model = self.model.ok_or(RequestBuildError::MissingModel)?;
        if model.trim().is_empty() {
            return Err(RequestBuildError::EmptyModel);
        }
        if self.messages.is_empty() {
            return Err(RequestBuildError::NoMessages);
        }
        if self
            .messages
            .iter()
            .any(|message| message.content().is_empty())
        {
            return Err(RequestBuildError::EmptyMessage);
        }
        if self.max_output_tokens == Some(0) {
            return Err(RequestBuildError::ZeroOutputTokens);
        }
        if self
            .temperature
            .is_some_and(|value| !(0.0..=2.0).contains(&value))
        {
            return Err(RequestBuildError::InvalidTemperature);
        }
        if self
            .top_p
            .is_some_and(|value| !(0.0..=1.0).contains(&value))
        {
            return Err(RequestBuildError::InvalidTopP);
        }
        if self.timeout == Some(Duration::ZERO) {
            return Err(RequestBuildError::ZeroTimeout);
        }
        let mut tool_names = BTreeSet::new();
        for tool in &self.tools {
            if tool.name.trim().is_empty() {
                return Err(RequestBuildError::EmptyToolName);
            }
            if !tool_names.insert(tool.name.as_str()) {
                return Err(RequestBuildError::DuplicateToolName);
            }
        }
        if let Some(ToolChoice::Tool { name }) = &self.tool_choice {
            if !tool_names.contains(name.as_str()) {
                return Err(RequestBuildError::UnknownToolChoice);
            }
        }
        if let Some(ResponseFormat::JsonSchema { name, .. }) = &self.response_format {
            if name.trim().is_empty() {
                return Err(RequestBuildError::EmptySchemaName);
            }
        }
        if self
            .provider_options
            .keys()
            .any(|namespace| namespace.trim().is_empty())
        {
            return Err(RequestBuildError::EmptyProviderNamespace);
        }

        Ok(Request {
            model,
            messages: self.messages,
            tools: self.tools,
            tool_choice: self.tool_choice,
            response_format: self.response_format,
            max_output_tokens: self.max_output_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            reasoning_effort: self.reasoning_effort,
            speed: self.speed,
            timeout: self.timeout,
            provider_options: self.provider_options,
        })
    }
}

/// A request failed local construction checks.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
#[non_exhaustive]
pub enum RequestBuildError {
    #[error("a model selector is required")]
    MissingModel,
    #[error("the model selector must not be empty")]
    EmptyModel,
    #[error("at least one message is required")]
    NoMessages,
    #[error("messages must contain at least one content part")]
    EmptyMessage,
    #[error("max_output_tokens must be greater than zero")]
    ZeroOutputTokens,
    #[error("temperature must be between 0 and 2")]
    InvalidTemperature,
    #[error("top_p must be between 0 and 1")]
    InvalidTopP,
    #[error("timeout must be greater than zero")]
    ZeroTimeout,
    #[error("tool names must not be empty")]
    EmptyToolName,
    #[error("tool names must be unique")]
    DuplicateToolName,
    #[error("the selected tool must be present in the request tools")]
    UnknownToolChoice,
    #[error("JSON schema names must not be empty")]
    EmptySchemaName,
    #[error("provider option namespaces must not be empty")]
    EmptyProviderNamespace,
}

mod duration_millis {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serialize as _, Serializer};

    #[expect(
        clippy::ref_option,
        reason = "Serde's field serializer passes a reference to the Option"
    )]
    pub(super) fn serialize<S>(value: &Option<Duration>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        value
            .map(|duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX))
            .serialize(serializer)
    }

    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<Option<Duration>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Option::<u64>::deserialize(deserializer).map(|value| value.map(Duration::from_millis))
    }
}
