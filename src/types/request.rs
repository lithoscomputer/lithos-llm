use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

use super::{Message, Role, ToolChoice, ToolDefinition, ToolDefinitionKind};
use crate::catalog::ProviderId;

/// Requested reasoning depth, when a provider supports it.
///
/// The variants run from least to most reasoning. A codec maps them onto the
/// levels its provider names, which is why `Xhigh` and `Max` are separate:
/// providers that offer both treat them as different levels, and collapsing
/// them would make the higher one unreachable.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ReasoningEffort {
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
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

/// How a request steers provider cache routing.
///
/// A backend that shards requests across replicas takes a routing hint —
/// `prompt_cache_key` on the OpenAI-style protocols — so a repeated prompt
/// lands on the replica that holds its cache entry. Without the hint, a
/// gateway such as Venice can write the same cache entry on every call and
/// never read one.
///
/// The catalog declares support per model with the `cache_routing`
/// capability. An unset hint behaves as [`CacheHint::Auto`].
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum CacheHint {
    /// Send a stable fingerprint of the cacheable prefix — the system
    /// messages and the tool definitions — where the catalog claims
    /// `cache_routing`. This is the default, and `auto_cache: false` turns
    /// it off along with the other automatic cache behavior.
    Auto,
    /// Send exactly this key, for callers that already partition their
    /// prompts, for example per tenant or per conversation.
    Key { key: String },
    /// Send no routing hint, even where the backend takes one.
    Disabled,
}

/// A provider-neutral inference request.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(try_from = "RequestBuilder")]
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
    cache_hint:        Option<CacheHint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    speed:             Option<Speed>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "duration_millis"
    )]
    timeout:           Option<Duration>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    stop_sequences:    Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    metadata:          BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    provider_options:  BTreeMap<ProviderId, Map<String, Value>>,
}

impl Request {
    /// Returns a builder that preserves every setting in this request.
    pub fn into_builder(self) -> RequestBuilder {
        RequestBuilder {
            model:             Some(self.model),
            messages:          self.messages,
            tools:             self.tools,
            tool_choice:       self.tool_choice,
            response_format:   self.response_format,
            max_output_tokens: self.max_output_tokens,
            temperature:       self.temperature,
            top_p:             self.top_p,
            reasoning_effort:  self.reasoning_effort,
            cache_hint:        self.cache_hint,
            speed:             self.speed,
            timeout:           self.timeout,
            stop_sequences:    self.stop_sequences,
            metadata:          self.metadata,
            provider_options:  self.provider_options,
        }
    }

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

    pub fn cache_hint(&self) -> Option<&CacheHint> {
        self.cache_hint.as_ref()
    }

    pub fn speed(&self) -> Option<Speed> {
        self.speed
    }

    pub fn timeout(&self) -> Option<Duration> {
        self.timeout
    }

    /// The sequences that stop generation, in the order they were added.
    pub fn stop_sequences(&self) -> &[String] {
        &self.stop_sequences
    }

    /// Free-form request metadata, forwarded where a protocol supports it.
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    /// Raw provider options, keyed by canonical catalog provider id.
    ///
    /// Each namespace is a JSON object of wire fields for one provider. A
    /// request can carry namespaces for several failover candidates; a codec
    /// reads only the namespace of the provider it was routed to.
    pub fn provider_options(&self) -> &BTreeMap<ProviderId, Map<String, Value>> {
        &self.provider_options
    }

    /// The raw options namespace for one provider, or `None` when the request
    /// carries no options for it.
    pub fn options_for(&self, provider: &ProviderId) -> Option<&Map<String, Value>> {
        self.provider_options.get(provider)
    }
}

/// Builds and validates an inference request.
#[derive(Default, Deserialize)]
#[serde(default)]
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
    cache_hint:        Option<CacheHint>,
    speed:             Option<Speed>,
    #[serde(with = "duration_millis")]
    timeout:           Option<Duration>,
    stop_sequences:    Vec<String>,
    metadata:          BTreeMap<String, String>,
    provider_options:  BTreeMap<ProviderId, Map<String, Value>>,
}

impl TryFrom<RequestBuilder> for Request {
    type Error = RequestBuildError;

    fn try_from(builder: RequestBuilder) -> Result<Self, Self::Error> {
        builder.build()
    }
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

    /// Sets how the request steers provider cache routing.
    pub fn cache_hint(mut self, hint: CacheHint) -> Self {
        self.cache_hint = Some(hint);
        self
    }

    /// Sets an exact cache routing key. Shorthand for [`CacheHint::Key`].
    pub fn cache_key(self, key: impl Into<String>) -> Self {
        self.cache_hint(CacheHint::Key { key: key.into() })
    }

    pub fn speed(mut self, speed: Speed) -> Self {
        self.speed = Some(speed);
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Appends one stop sequence. Order is preserved.
    pub fn stop_sequence(mut self, sequence: impl Into<String>) -> Self {
        self.stop_sequences.push(sequence.into());
        self
    }

    /// Appends several stop sequences. Order is preserved.
    pub fn stop_sequences(
        mut self,
        sequences: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.stop_sequences
            .extend(sequences.into_iter().map(Into::into));
        self
    }

    /// Sets one metadata entry, replacing any earlier value for the key.
    pub fn metadata_entry(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Sets one raw option inside a provider namespace.
    ///
    /// `provider` is the canonical catalog provider id, never a codec or
    /// adapter id.
    pub fn provider_option(
        mut self,
        provider: impl Into<ProviderId>,
        key: impl Into<String>,
        value: Value,
    ) -> Self {
        self.provider_options
            .entry(provider.into())
            .or_default()
            .insert(key.into(), value);
        self
    }

    /// Replaces the complete raw options namespace for one provider.
    ///
    /// `provider` is the canonical catalog provider id, never a codec or
    /// adapter id.
    pub fn provider_options(
        mut self,
        provider: impl Into<ProviderId>,
        options: Map<String, Value>,
    ) -> Self {
        self.provider_options.insert(provider.into(), options);
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
        if self
            .stop_sequences
            .iter()
            .any(|sequence| sequence.trim().is_empty())
        {
            return Err(RequestBuildError::EmptyStopSequence);
        }
        if self.metadata.keys().any(|key| key.trim().is_empty()) {
            return Err(RequestBuildError::EmptyMetadataKey);
        }
        let mut tool_names = BTreeSet::new();
        for tool in &self.tools {
            if tool.name.trim().is_empty() {
                return Err(RequestBuildError::EmptyToolName);
            }
            if !tool_names.insert(tool.name.as_str()) {
                return Err(RequestBuildError::DuplicateToolName);
            }
            if let ToolDefinitionKind::Custom { format } = &tool.kind
                && format.is_null()
            {
                return Err(RequestBuildError::CustomToolFormatRequired);
            }
        }
        if let Some(ToolChoice::Tool { name }) = &self.tool_choice
            && !tool_names.contains(name.as_str())
        {
            return Err(RequestBuildError::UnknownToolChoice);
        }
        if let Some(ResponseFormat::JsonSchema { name, .. }) = &self.response_format
            && name.trim().is_empty()
        {
            return Err(RequestBuildError::EmptySchemaName);
        }
        if self
            .provider_options
            .keys()
            .any(|namespace| namespace.as_str().trim().is_empty())
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
            cache_hint: self.cache_hint,
            speed: self.speed,
            timeout: self.timeout,
            stop_sequences: self.stop_sequences,
            metadata: self.metadata,
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
    #[error("stop sequences must not be empty")]
    EmptyStopSequence,
    #[error("metadata keys must not be empty")]
    EmptyMetadataKey,
    #[error("tool names must not be empty")]
    EmptyToolName,
    #[error("tool names must be unique")]
    DuplicateToolName,
    #[error("custom tools must define a format")]
    CustomToolFormatRequired,
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

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use serde_json::{Map, json};

    use super::{ReasoningEffort, Request, RequestBuildError, RequestBuilder};
    use crate::catalog::ProviderId;
    use crate::types::{ToolChoice, ToolDefinition};

    fn base() -> RequestBuilder {
        Request::builder().model("test-model").user("hello")
    }

    #[test]
    fn every_reasoning_effort_has_its_own_wire_value() -> Result<(), Box<dyn StdError>> {
        let levels = [
            (ReasoningEffort::Minimal, "minimal"),
            (ReasoningEffort::Low, "low"),
            (ReasoningEffort::Medium, "medium"),
            (ReasoningEffort::High, "high"),
            (ReasoningEffort::Xhigh, "xhigh"),
            (ReasoningEffort::Max, "max"),
        ];

        for (effort, wire) in levels {
            assert_eq!(serde_json::to_value(effort)?, json!(wire));
            assert_eq!(
                serde_json::from_value::<ReasoningEffort>(json!(wire))?,
                effort
            );
        }
        Ok(())
    }

    #[test]
    fn stop_sequences_round_trip_in_order() -> Result<(), Box<dyn StdError>> {
        let request = base()
            .stop_sequence("zebra")
            .stop_sequences(["alpha", "END"])
            .build()?;
        assert_eq!(request.stop_sequences(), ["zebra", "alpha", "END"]);

        let encoded = serde_json::to_string(&request)?;
        let decoded = serde_json::from_str::<Request>(&encoded)?;

        assert_eq!(decoded.stop_sequences(), ["zebra", "alpha", "END"]);
        assert_eq!(decoded, request);
        Ok(())
    }

    #[test]
    fn rejects_an_empty_stop_sequence() {
        let error = base().stop_sequence("   ").build().unwrap_err();

        assert_eq!(error, RequestBuildError::EmptyStopSequence);
    }

    #[test]
    fn rejects_an_empty_metadata_key() {
        let error = base().metadata_entry("", "value").build().unwrap_err();

        assert_eq!(error, RequestBuildError::EmptyMetadataKey);
    }

    #[test]
    fn round_trips_metadata_and_provider_options() -> Result<(), Box<dyn StdError>> {
        let mut anthropic = Map::new();
        anthropic.insert("top_k".to_owned(), json!(5));
        anthropic.insert("auto_cache".to_owned(), json!(false));

        let request = base()
            .metadata_entry("user_id", "u-1")
            .metadata_entry("trace_id", "t-789")
            .provider_option("openai", "seed", json!(7))
            .provider_options("anthropic", anthropic)
            .stop_sequence("END")
            .build()?;

        let encoded = serde_json::to_string(&request)?;
        let decoded = serde_json::from_str::<Request>(&encoded)?;

        assert_eq!(decoded, request);
        assert_eq!(
            decoded.metadata().get("trace_id").map(String::as_str),
            Some("t-789")
        );
        assert_eq!(decoded.provider_options().len(), 2);
        Ok(())
    }

    #[test]
    fn options_for_reads_only_the_requested_namespace() -> Result<(), Box<dyn StdError>> {
        let request = base()
            .provider_option("openai", "seed", json!(7))
            .provider_option("anthropic", "top_k", json!(5))
            .build()?;

        let openai = request
            .options_for(&ProviderId::new("openai"))
            .ok_or("the openai namespace is missing")?;

        assert_eq!(openai.get("seed"), Some(&json!(7)));
        assert!(!openai.contains_key("top_k"));
        assert!(request.options_for(&ProviderId::new("gemini")).is_none());
        Ok(())
    }

    #[test]
    fn rejects_a_non_object_provider_namespace() {
        let document = r#"{"model":"m","messages":[],"provider_options":{"openai":"seed"}}"#;

        assert!(serde_json::from_str::<Request>(document).is_err());
    }

    #[test]
    fn rejects_duplicate_names_across_tool_kinds() {
        let error = base()
            .tool(ToolDefinition::function(
                "patch",
                "apply a patch",
                json!({}),
            ))
            .tool(ToolDefinition::custom(
                "patch",
                "apply a patch",
                json!({"type": "text"}),
            ))
            .build()
            .unwrap_err();

        assert_eq!(error, RequestBuildError::DuplicateToolName);
    }

    #[test]
    fn accepts_a_named_choice_of_a_custom_tool() -> Result<(), Box<dyn StdError>> {
        let request = base()
            .tool(ToolDefinition::custom(
                "apply_patch",
                "apply a patch",
                json!({"type": "text"}),
            ))
            .tool_choice(ToolChoice::Tool {
                name: "apply_patch".to_owned(),
            })
            .build()?;

        assert_eq!(request.tools().len(), 1);
        assert!(request.tools()[0].is_custom());
        Ok(())
    }

    #[test]
    fn rejects_a_custom_tool_without_a_format() {
        let error = base()
            .tool(ToolDefinition::custom(
                "apply_patch",
                "apply a patch",
                json!(null),
            ))
            .build()
            .unwrap_err();

        assert_eq!(error, RequestBuildError::CustomToolFormatRequired);
    }
}
