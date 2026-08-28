use toml::Value;
use toml::map::Map;

use super::overlay::merge;
use super::{Catalog, CatalogError};

#[cfg(feature = "builtin-catalog")]
const BUILTIN_CATALOG: &str = r#"
schema_version = 1

[providers.openai]
display_name = "OpenAI"
aliases = ["oa"]
adapter = "openai"
codec = "openai-responses"
base_url = "https://api.openai.com"
priority = 100
default_model = "gpt-5.6-luna"

[providers.openai.auth]
type = "bearer"

[providers.openai.models."gpt-5.6-luna"]
display_name = "GPT-5.6 Luna"
aliases = ["luna"]
api_model = "gpt-5.6-luna"
limits = { context_tokens = 1050000, max_output_tokens = 128000 }
capabilities = { text = true, images = true, tools = true, structured_output = true, reasoning = true, caching = true, sampling = true }
pricing = { input_usd_micros_per_million = 200000, output_usd_micros_per_million = 1200000, cached_input_usd_micros_per_million = 20000 }

[providers.anthropic]
display_name = "Anthropic"
aliases = ["claude"]
adapter = "anthropic"
codec = "anthropic-messages"
base_url = "https://api.anthropic.com"
priority = 90
default_model = "claude-sonnet-4-6"

[providers.anthropic.auth]
type = "header"
name = "x-api-key"

[providers.anthropic.models."claude-sonnet-4-6"]
display_name = "Claude Sonnet 4.6"
aliases = ["sonnet"]
api_model = "claude-sonnet-4-6"
limits = { context_tokens = 1000000, max_output_tokens = 128000 }
capabilities = { text = true, images = true, documents = true, tools = true, structured_output = true, reasoning = true, caching = true, sampling = true }
pricing = { input_usd_micros_per_million = 3000000, output_usd_micros_per_million = 15000000, cached_input_usd_micros_per_million = 300000 }

[providers.gemini]
display_name = "Google Gemini"
aliases = ["google"]
adapter = "gemini"
codec = "gemini-generate"
base_url = "https://generativelanguage.googleapis.com"
priority = 80
default_model = "gemini-2.5-pro"

[providers.gemini.auth]
type = "header"
name = "x-goog-api-key"

[providers.gemini.models."gemini-2.5-pro"]
display_name = "Gemini 2.5 Pro"
aliases = ["gemini-pro"]
api_model = "gemini-2.5-pro"
limits = { context_tokens = 1048576, max_output_tokens = 65536 }
capabilities = { text = true, images = true, audio = true, documents = true, tools = true, structured_output = true, reasoning = true, caching = true, sampling = true }
pricing = { input_usd_micros_per_million = 1250000, output_usd_micros_per_million = 10000000, cached_input_usd_micros_per_million = 125000, long_context = { above_input_tokens = 200000, input_usd_micros_per_million = 2500000, output_usd_micros_per_million = 15000000, cached_input_usd_micros_per_million = 250000 } }

[providers.bedrock]
display_name = "Amazon Bedrock"
aliases = ["aws-bedrock"]
adapter = "bedrock"
codec = "bedrock-converse"
base_url = "https://bedrock-runtime.us-east-1.amazonaws.com"
priority = 50
allow_passthrough = true
default_model = "anthropic.claude-sonnet-4-6"

[providers.bedrock.auth]
type = "aws"
region = "us-east-1"

[providers.bedrock.models."anthropic.claude-sonnet-4-6"]
display_name = "Claude Sonnet 4.6 on Bedrock"
aliases = ["bedrock-sonnet"]
api_model = "anthropic.claude-sonnet-4-6"
limits = { context_tokens = 1000000, max_output_tokens = 128000 }
capabilities = { text = true, images = true, documents = true, tools = true, reasoning = true, sampling = true }
"#;

enum Layer {
    Builtin,
    Overlay(toml::Value),
}

/// Builds an immutable catalog from ordered layers.
#[derive(Default)]
#[must_use]
pub struct CatalogBuilder {
    layers: Vec<Layer>,
}

impl CatalogBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_builtin(mut self) -> Self {
        self.layers.push(Layer::Builtin);
        self
    }

    pub fn overlay_toml(mut self, source: &str) -> Result<Self, CatalogError> {
        let overlay = toml::from_str(source).map_err(CatalogError::Parse)?;
        self.layers.push(Layer::Overlay(overlay));
        Ok(self)
    }

    pub fn build(self) -> Result<Catalog, CatalogError> {
        let mut merged = Value::Table(Map::new());
        for layer in self.layers {
            match layer {
                Layer::Builtin => merge(&mut merged, builtin_value()?),
                Layer::Overlay(value) => merge(&mut merged, value),
            }
        }
        let document = merged.try_into().map_err(CatalogError::Parse)?;
        Catalog::from_document(document)
    }
}

#[cfg(feature = "builtin-catalog")]
fn builtin_value() -> Result<toml::Value, CatalogError> {
    toml::from_str(BUILTIN_CATALOG).map_err(CatalogError::Parse)
}

#[cfg(not(feature = "builtin-catalog"))]
fn builtin_value() -> Result<toml::Value, CatalogError> {
    Err(CatalogError::BuiltinCatalogDisabled)
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use serde::Deserialize;

    use super::{Catalog, CatalogError};

    #[derive(Debug, Deserialize, Eq, PartialEq)]
    struct AppMetadata {
        latency: String,
        nested:  Nested,
    }

    #[derive(Debug, Deserialize, Eq, PartialEq)]
    struct Nested {
        enabled: bool,
        count:   u64,
    }

    #[cfg(feature = "builtin-catalog")]
    #[test]
    fn overlays_catalog_and_recursively_merges_metadata() -> Result<(), Box<dyn StdError>> {
        let first = r#"
            [providers.openai.models."gpt-5.6-luna".metadata.app]
            latency = "fast"
            nested = { enabled = true }
        "#;
        let second = r#"
            [providers.openai.models."gpt-5.6-luna".metadata.app.nested]
            count = 2
        "#;
        let catalog = Catalog::builder()
            .with_builtin()
            .overlay_toml(first)?
            .overlay_toml(second)?
            .build()?;

        let metadata = catalog
            .model("openai", "luna")?
            .metadata()
            .namespace::<AppMetadata>("app")?
            .expect("app metadata should be present");
        assert_eq!(metadata, AppMetadata {
            latency: "fast".to_owned(),
            nested:  Nested {
                enabled: true,
                count:   2,
            },
        });
        Ok(())
    }

    #[test]
    fn rejects_provider_alias_that_collides_with_an_id() -> Result<(), Box<dyn StdError>> {
        let source = r#"
            schema_version = 1
            [providers.first]
            display_name = "First"
            aliases = ["second"]
            adapter = "custom"
            codec = "custom"
            base_url = "https://example.com"
            auth = { type = "none" }

            [providers.second]
            display_name = "Second"
            adapter = "custom"
            codec = "custom"
            base_url = "https://example.com"
            auth = { type = "none" }
        "#;

        let result = Catalog::builder().overlay_toml(source)?.build();

        assert!(matches!(
            result,
            Err(CatalogError::DuplicateProviderAlias { alias }) if alias == "second"
        ));
        Ok(())
    }

    #[test]
    fn rejects_duplicate_model_selector_within_provider() -> Result<(), Box<dyn StdError>> {
        let source = r#"
            schema_version = 1
            [providers.test]
            display_name = "Test"
            adapter = "custom"
            codec = "custom"
            base_url = "https://example.com"
            auth = { type = "none" }

            [providers.test.models.first]
            display_name = "First"
            aliases = ["shared"]
            api_model = "first"

            [providers.test.models.second]
            display_name = "Second"
            aliases = ["shared"]
            api_model = "second"
        "#;

        let result = Catalog::builder().overlay_toml(source)?.build();

        assert!(matches!(
            result,
            Err(CatalogError::DuplicateModelSelector { provider, selector })
                if provider.as_str() == "test" && selector == "shared"
        ));
        Ok(())
    }
}
