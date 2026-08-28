use serde::{Deserialize, Serialize};

use super::ContentPart;
use crate::catalog::{ModelHandle, ModelId, ProviderId};

/// Why a model stopped producing output.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum FinishReason {
    Stop,
    Length,
    ToolCall,
    ContentFilter,
    Error,
    Other(String),
}

/// Token usage reported or estimated for a call.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct TokenCounts {
    pub input:            u64,
    pub output:           u64,
    #[serde(default)]
    pub cached_input:     u64,
    #[serde(default)]
    pub reasoning_output: u64,
}

impl TokenCounts {
    pub fn total(self) -> u64 {
        self.input.saturating_add(self.output)
    }
}

/// The source used for a computed response cost.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CostSource {
    Catalog,
    Provider,
    Application,
}

/// A cost expressed in US dollar micros.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Cost {
    pub usd_micros: u64,
    pub source:     CostSource,
}

/// Rate-limit information returned by a provider.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct RateLimits {
    pub request_limit:     Option<u64>,
    pub request_remaining: Option<u64>,
    pub token_limit:       Option<u64>,
    pub token_remaining:   Option<u64>,
}

/// A non-fatal provider or normalization warning.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Warning {
    pub code:    String,
    pub message: String,
}

/// A normalized complete model response.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Response {
    pub id:            Option<String>,
    pub model:         ModelHandle,
    pub content:       Vec<ContentPart>,
    pub finish_reason: FinishReason,
    pub usage:         TokenCounts,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost:          Option<Cost>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_limits:   Option<RateLimits>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings:      Vec<Warning>,
}

impl Response {
    pub fn new(provider: ProviderId, model: ModelId, content: Vec<ContentPart>) -> Self {
        Self {
            id: None,
            model: ModelHandle::new(provider, model),
            content,
            finish_reason: FinishReason::Stop,
            usage: TokenCounts::default(),
            cost: None,
            rate_limits: None,
            warnings: Vec::new(),
        }
    }

    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|part| match part {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }
}
