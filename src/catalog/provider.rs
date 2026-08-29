use std::collections::BTreeMap;
use std::fmt;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use super::CatalogModel;

macro_rules! string_id {
    ($name:ident) => {
        #[derive(
            Clone, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
        )]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }

            pub fn into_string(self) -> String {
                self.0
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self::new(value)
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self::new(value)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(self.as_str())
            }
        }

        impl std::borrow::Borrow<str> for $name {
            fn borrow(&self) -> &str {
                self.as_str()
            }
        }
    };
}

string_id!(ProviderId);
string_id!(ModelId);
string_id!(AdapterId);
string_id!(CodecId);

/// Built-in adapter identifiers.
pub mod adapter_ids {
    pub const ANTHROPIC: &str = "anthropic";
    pub const BEDROCK: &str = "bedrock";
    pub const GEMINI: &str = "gemini";
    pub const OPENAI: &str = "openai";
    pub const OPENAI_COMPATIBLE: &str = "openai-compatible";
}

/// Built-in wire codec identifiers.
pub mod codec_ids {
    pub const ANTHROPIC_MESSAGES: &str = "anthropic-messages";
    pub const BEDROCK_CONVERSE: &str = "bedrock-converse";
    pub const GEMINI_GENERATE: &str = "gemini-generate";
    pub const OPENAI_CHAT: &str = "openai-chat";
    pub const OPENAI_RESPONSES: &str = "openai-responses";
}

/// A resolved provider and model identity.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ModelHandle {
    provider: ProviderId,
    model:    ModelId,
}

impl ModelHandle {
    pub fn new(provider: ProviderId, model: ModelId) -> Self {
        Self { provider, model }
    }

    pub fn provider(&self) -> &ProviderId {
        &self.provider
    }

    pub fn model(&self) -> &ModelId {
        &self.model
    }
}

impl fmt::Display for ModelHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}/{}", self.provider, self.model)
    }
}

/// Deterministic application-owned extension metadata.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(transparent)]
pub struct Metadata(BTreeMap<String, Value>);

impl Metadata {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn namespace<T: DeserializeOwned>(&self, name: &str) -> Result<Option<T>, MetadataError> {
        self.0
            .get(name)
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|source| MetadataError {
                namespace: name.to_owned(),
                source,
            })
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.0.get(name)
    }
}

/// An application metadata namespace has the wrong shape.
#[derive(Debug, Error)]
#[error("metadata namespace `{namespace}` is invalid")]
pub struct MetadataError {
    namespace: String,
    #[source]
    source:    serde_json::Error,
}

/// The authentication shape declared by a provider.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum AuthScheme {
    None,
    Bearer {
        #[serde(default = "authorization_header")]
        header: String,
        #[serde(default = "bearer_prefix")]
        prefix: String,
    },
    Header {
        name: String,
    },
    Headers,
    Aws {
        region: Option<String>,
    },
    BedrockBearer,
}

fn authorization_header() -> String {
    "authorization".to_owned()
}

fn bearer_prefix() -> String {
    "Bearer ".to_owned()
}

/// Provider-level catalog facts.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogProvider {
    #[serde(skip)]
    id:                ProviderId,
    display_name:      String,
    #[serde(default)]
    aliases:           Vec<String>,
    adapter:           AdapterId,
    codec:             CodecId,
    base_url:          String,
    auth:              AuthScheme,
    #[serde(default)]
    priority:          i32,
    #[serde(default)]
    allow_passthrough: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    default_model:     Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    default_headers:   BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    adapter_options:   Value,
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    default_options:   serde_json::Map<String, Value>,
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    metadata:          Metadata,
    #[serde(default)]
    models:            BTreeMap<ModelId, CatalogModel>,
}

impl CatalogProvider {
    pub fn id(&self) -> &ProviderId {
        &self.id
    }

    pub fn display_name(&self) -> &str {
        &self.display_name
    }

    pub fn aliases(&self) -> &[String] {
        &self.aliases
    }

    pub fn adapter(&self) -> &AdapterId {
        &self.adapter
    }

    pub fn codec(&self) -> &CodecId {
        &self.codec
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn auth(&self) -> &AuthScheme {
        &self.auth
    }

    pub fn priority(&self) -> i32 {
        self.priority
    }

    pub fn allows_passthrough(&self) -> bool {
        self.allow_passthrough
    }

    pub fn default_model(&self) -> Option<&str> {
        self.default_model.as_deref()
    }

    /// Headers applied to every request for this provider.
    ///
    /// These are ordinary catalog data. They are not secret, they are not
    /// redacted, and they can appear in debug output and logs. Keep API keys
    /// and other secrets in credentials instead. Credential headers are applied
    /// after these and win a collision.
    ///
    /// Names and values are validated when the catalog is built.
    pub fn default_headers(&self) -> &BTreeMap<String, String> {
        &self.default_headers
    }

    /// Raw adapter options for this provider.
    ///
    /// The catalog does not interpret these. Each adapter factory deserializes
    /// its own typed shape and reports its own errors. The value is
    /// [`Value::Null`] when the catalog declares no options.
    pub fn adapter_options(&self) -> &Value {
        &self.adapter_options
    }

    /// Default request options for this provider's own namespace.
    ///
    /// Codecs treat these exactly like request-level
    /// [`provider_options`](crate::Request::provider_options) for this
    /// provider, except that request-level values win a collision. This is
    /// where a catalog turns off a gateway behavior for every request, such
    /// as Venice's injected system prompt:
    ///
    /// ```toml
    /// [providers.venice.default_options]
    /// venice_parameters = { include_venice_system_prompt = false }
    /// ```
    ///
    /// Control keys such as `auto_cache` work here too and are consumed, not
    /// sent.
    pub fn default_options(&self) -> &serde_json::Map<String, Value> {
        &self.default_options
    }

    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    pub fn models(&self) -> impl ExactSizeIterator<Item = &CatalogModel> {
        self.models.values()
    }

    pub fn model(&self, id_or_alias: &str) -> Option<&CatalogModel> {
        self.models.get(id_or_alias).or_else(|| {
            self.models
                .values()
                .find(|model| model.aliases().iter().any(|alias| alias == id_or_alias))
        })
    }

    pub(crate) fn set_id(&mut self, id: ProviderId) {
        self.id = id;
    }

    pub(crate) fn models_mut(
        &mut self,
    ) -> impl ExactSizeIterator<Item = (&ModelId, &mut CatalogModel)> {
        self.models.iter_mut()
    }
}
