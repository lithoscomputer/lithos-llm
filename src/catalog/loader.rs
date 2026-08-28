use std::collections::BTreeMap;

use serde::Deserialize;
use toml::Value;
use toml::map::Map;

use super::overlay::merge;
use super::{Catalog, CatalogDocument, CatalogError, CatalogProvider, LayerOrigins, ProviderId};

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
limits = { context_tokens = 272000, max_output_tokens = 128000 }
capabilities = { text = true, images = true, tools = true, structured_output = true, reasoning = true, reasoning_effort_levels = true, caching = true, sampling = true }
pricing = { input_usd_micros_per_million = 1000000, output_usd_micros_per_million = 6000000, cached_input_usd_micros_per_million = 100000 }

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
limits = { context_tokens = 200000, max_output_tokens = 64000 }
capabilities = { text = true, images = true, documents = true, tools = true, structured_output = true, reasoning = true, reasoning_effort_levels = true, caching = true, sampling = true }

# Cache writes bill at 1.25x input. The fast speed tier doubles every rate.
[providers.anthropic.models."claude-sonnet-4-6".pricing]
input_usd_micros_per_million = 3000000
output_usd_micros_per_million = 15000000
cached_input_usd_micros_per_million = 300000
cache_write_usd_micros_per_million = 3750000
speed = { fast = { input_usd_micros_per_million = 6000000, output_usd_micros_per_million = 30000000, cached_input_usd_micros_per_million = 600000, cache_write_usd_micros_per_million = 7500000 } }

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
# The `us.` cross-region inference profile: Bedrock on-demand access to this
# model needs the profile rather than the bare model id.
api_model = "us.anthropic.claude-sonnet-4-6"
limits = { context_tokens = 200000, max_output_tokens = 64000 }
capabilities = { text = true, images = true, documents = true, tools = true, reasoning = true, reasoning_effort_levels = true, caching = true, sampling = true }
pricing = { input_usd_micros_per_million = 3000000, output_usd_micros_per_million = 15000000, cached_input_usd_micros_per_million = 300000, cache_write_usd_micros_per_million = 3750000 }
"#;

/// The layer name reported for the built-in catalog.
const BUILTIN_LAYER: &str = "built-in";

/// The layer name reported when a failure belongs to no single layer.
const MERGED_LAYER: &str = "<merged catalog>";

struct Layer {
    name:   String,
    source: LayerSource,
}

enum LayerSource {
    Builtin,
    Toml(Value),
}

/// The merged document before provider entries are given their schema.
///
/// Providers stay as raw TOML so a schema failure can name the layer that
/// last wrote the offending provider.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDocument {
    schema_version: u32,
    providers:      BTreeMap<ProviderId, Value>,
}

/// Builds an immutable catalog from ordered layers.
///
/// Layers merge in call order and later layers win. Each layer carries a name
/// that appears in parse and validation errors. No layer is added implicitly:
/// a catalog built only from application TOML contains no built-in entry.
#[derive(Default)]
#[must_use]
pub struct CatalogBuilder {
    layers:   Vec<Layer>,
    overlays: usize,
}

impl CatalogBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds the minimal built-in catalog as the next layer.
    ///
    /// The layer is named `built-in`.
    pub fn with_builtin(mut self) -> Self {
        self.layers.push(Layer {
            name:   BUILTIN_LAYER.to_owned(),
            source: LayerSource::Builtin,
        });
        self
    }

    /// Adds a named TOML layer.
    ///
    /// `name` identifies the layer in parse and validation errors. A file path
    /// such as `providers/openai.toml` reads well in a message.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::Parse`] when `source` is not valid TOML.
    pub fn toml_layer(
        mut self,
        name: impl Into<String>,
        source: &str,
    ) -> Result<Self, CatalogError> {
        let name = name.into();
        let value = parse_layer(&name, source)?;
        self.layers.push(Layer {
            name,
            source: LayerSource::Toml(value),
        });
        Ok(self)
    }

    /// Adds an unnamed TOML layer.
    ///
    /// The layer is named `overlay 1`, `overlay 2`, and so on by position. Use
    /// [`CatalogBuilder::toml_layer`] to choose a more useful name.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::Parse`] when `source` is not valid TOML.
    pub fn overlay_toml(mut self, source: &str) -> Result<Self, CatalogError> {
        self.overlays += 1;
        let name = format!("overlay {}", self.overlays);
        self.toml_layer(name, source)
    }

    /// Merges every layer and validates the result.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::Parse`] when a merged provider does not match
    /// the catalog schema, [`CatalogError::Layer`] when a provider fails
    /// validation and its layer is known, and the underlying
    /// [`CatalogError`] otherwise.
    pub fn build(self) -> Result<Catalog, CatalogError> {
        let mut merged = Value::Table(Map::new());
        let mut origins = LayerOrigins::new();
        for layer in self.layers {
            let value = match layer.source {
                LayerSource::Builtin => builtin_value(&layer.name)?,
                LayerSource::Toml(value) => value,
            };
            record_origins(&mut origins, &layer.name, &value);
            merge(&mut merged, value);
        }

        let raw: RawDocument = merged.try_into().map_err(|source| CatalogError::Parse {
            layer: MERGED_LAYER.to_owned(),
            source,
        })?;

        let mut providers = BTreeMap::new();
        for (provider_id, value) in raw.providers {
            let layer = origins
                .get(&provider_id)
                .cloned()
                .unwrap_or_else(|| MERGED_LAYER.to_owned());
            let provider: CatalogProvider = value
                .try_into()
                .map_err(|source| CatalogError::Parse { layer, source })?;
            providers.insert(provider_id, provider);
        }

        Catalog::from_document(
            CatalogDocument {
                schema_version: raw.schema_version,
                providers,
            },
            &origins,
        )
    }
}

fn parse_layer(layer: &str, source: &str) -> Result<Value, CatalogError> {
    toml::from_str(source).map_err(|error| CatalogError::Parse {
        layer:  layer.to_owned(),
        source: error,
    })
}

/// Records `layer` as the most recent writer of every provider it declares.
fn record_origins(origins: &mut LayerOrigins, layer: &str, value: &Value) {
    let Some(Value::Table(providers)) = value.get("providers") else {
        return;
    };
    for name in providers.keys() {
        origins.insert(ProviderId::new(name.clone()), layer.to_owned());
    }
}

#[cfg(feature = "builtin-catalog")]
fn builtin_value(layer: &str) -> Result<Value, CatalogError> {
    parse_layer(layer, BUILTIN_CATALOG)
}

#[cfg(not(feature = "builtin-catalog"))]
fn builtin_value(_layer: &str) -> Result<Value, CatalogError> {
    Err(CatalogError::BuiltinCatalogDisabled)
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use serde::Deserialize;

    use super::{Catalog, CatalogError};
    #[cfg(feature = "builtin-catalog")]
    use crate::types::Speed;

    const BASE: &str = r#"
        schema_version = 1

        [providers.first]
        display_name = "First"
        adapter = "custom"
        codec = "custom"
        base_url = "https://example.com"
        auth = { type = "none" }

        [providers.first.models.model]
        display_name = "Model"
        api_model = "model"
    "#;

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

    #[test]
    fn external_layers_add_no_built_in_entry() -> Result<(), Box<dyn StdError>> {
        let catalog = Catalog::builder()
            .toml_layer("catalog.toml", BASE)?
            .build()?;

        assert_eq!(catalog.providers().len(), 1);
        assert!(catalog.provider("openai").is_err());
        assert!(catalog.provider("anthropic").is_err());
        assert!(catalog.models_matching("gpt-5.6-luna").is_empty());
        Ok(())
    }

    #[test]
    fn merges_named_layers_in_call_order() -> Result<(), Box<dyn StdError>> {
        let priority = r"
            [providers.first]
            priority = 10
        ";
        let second = r#"
            [providers.second]
            display_name = "Second"
            adapter = "custom"
            codec = "custom"
            base_url = "https://second.example.com"
            auth = { type = "none" }
        "#;
        let later = r"
            [providers.first]
            priority = 20
        ";

        let catalog = Catalog::builder()
            .toml_layer("catalog.toml", BASE)?
            .toml_layer("providers/first.toml", priority)?
            .toml_layer("providers/second.toml", second)?
            .toml_layer("overrides.toml", later)?
            .build()?;

        let ids = catalog
            .providers()
            .map(|provider| provider.id().as_str().to_owned())
            .collect::<Vec<_>>();

        assert_eq!(ids, vec!["first".to_owned(), "second".to_owned()]);
        assert_eq!(catalog.provider("first")?.priority(), 20);
        Ok(())
    }

    #[test]
    fn reports_the_layer_name_for_invalid_toml() -> Result<(), Box<dyn StdError>> {
        let result = Catalog::builder().toml_layer("providers/broken.toml", "not = = toml");

        let Err(CatalogError::Parse { layer, .. }) = result.map(|_| ()) else {
            return Err("expected a parse failure".into());
        };
        assert_eq!(layer, "providers/broken.toml");
        Ok(())
    }

    #[test]
    fn reports_the_layer_name_for_a_validation_failure() -> Result<(), Box<dyn StdError>> {
        let broken = r#"
            [providers.first]
            default_model = "missing"
        "#;

        let result = Catalog::builder()
            .toml_layer("catalog.toml", BASE)?
            .toml_layer("providers/first.toml", broken)?
            .build();

        let Err(CatalogError::Layer { layer, source }) = result else {
            return Err("expected a layer failure".into());
        };
        assert_eq!(layer, "providers/first.toml");
        assert!(matches!(
            *source,
            CatalogError::UnknownDefaultModel { model, .. } if model == "missing"
        ));
        Ok(())
    }

    #[test]
    fn reports_the_layer_name_for_an_unknown_provider_field() -> Result<(), Box<dyn StdError>> {
        let unknown = r#"
            [providers.first]
            base_urls = "https://example.com"
        "#;

        let result = Catalog::builder()
            .toml_layer("catalog.toml", BASE)?
            .toml_layer("providers/first.toml", unknown)?
            .build();

        let Err(CatalogError::Parse { layer, source }) = result else {
            return Err("expected a schema failure".into());
        };
        assert_eq!(layer, "providers/first.toml");
        assert!(source.to_string().contains("base_urls"));
        Ok(())
    }

    #[test]
    fn rejects_an_unknown_model_field() -> Result<(), Box<dyn StdError>> {
        let unknown = r"
            [providers.first.models.model]
            context_window = 4096
        ";

        let result = Catalog::builder()
            .toml_layer("catalog.toml", BASE)?
            .toml_layer("providers/first.toml", unknown)?
            .build();

        let Err(CatalogError::Parse { layer, source }) = result else {
            return Err("expected a schema failure".into());
        };
        assert_eq!(layer, "providers/first.toml");
        assert!(source.to_string().contains("context_window"));
        Ok(())
    }

    #[test]
    fn round_trips_namespaced_metadata() -> Result<(), Box<dyn StdError>> {
        let first = r#"
            [providers.first.models.model.metadata.app]
            latency = "fast"
            nested = { enabled = true }
        "#;
        let second = r"
            [providers.first.models.model.metadata.app.nested]
            count = 2
        ";

        let catalog = Catalog::builder()
            .toml_layer("catalog.toml", BASE)?
            .toml_layer("app/first.toml", first)?
            .toml_layer("app/second.toml", second)?
            .build()?;

        let metadata = catalog
            .model("first", "model")?
            .metadata()
            .namespace::<AppMetadata>("app")?
            .ok_or("app metadata should be present")?;

        assert_eq!(metadata, AppMetadata {
            latency: "fast".to_owned(),
            nested:  Nested {
                enabled: true,
                count:   2,
            },
        });
        Ok(())
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
            .ok_or("app metadata should be present")?;
        assert_eq!(metadata, AppMetadata {
            latency: "fast".to_owned(),
            nested:  Nested {
                enabled: true,
                count:   2,
            },
        });
        Ok(())
    }

    #[cfg(feature = "builtin-catalog")]
    #[test]
    fn names_unnamed_overlays_by_position() -> Result<(), Box<dyn StdError>> {
        let broken = r#"
            [providers.openai]
            default_model = "missing"
        "#;

        let result = Catalog::builder()
            .with_builtin()
            .overlay_toml("")?
            .overlay_toml(broken)?
            .build();

        let Err(CatalogError::Layer { layer, .. }) = result else {
            return Err("expected a layer failure".into());
        };
        assert_eq!(layer, "overlay 2");
        Ok(())
    }

    #[cfg(feature = "builtin-catalog")]
    #[test]
    fn the_builtin_catalog_carries_the_published_rates_and_limits() -> Result<(), Box<dyn StdError>>
    {
        let catalog = Catalog::builder().with_builtin().build()?;

        let luna = catalog.model("openai", "gpt-5.6-luna")?;
        let pricing = luna.pricing().ok_or("gpt-5.6-luna should be priced")?;
        assert_eq!(pricing.input_usd_micros_per_million, Some(1_000_000));
        assert_eq!(pricing.output_usd_micros_per_million, Some(6_000_000));
        assert_eq!(pricing.cached_input_usd_micros_per_million, Some(100_000));
        assert_eq!(
            luna.limits().map(|limits| limits.context_tokens),
            Some(272_000)
        );

        let sonnet = catalog.model("anthropic", "claude-sonnet-4-6")?;
        assert_eq!(
            sonnet.limits().map(|limits| limits.context_tokens),
            Some(200_000)
        );
        assert_eq!(
            sonnet.limits().map(|limits| limits.max_output_tokens),
            Some(64_000)
        );
        let pricing = sonnet
            .pricing()
            .ok_or("claude-sonnet-4-6 should be priced")?;
        // Anthropic bills a cache write at 1.25x input, and the fast tier
        // doubles every rate.
        assert_eq!(pricing.cache_write_usd_micros_per_million, Some(3_750_000));
        let fast = pricing.for_speed(Some(Speed::Fast));
        assert_eq!(fast.input_usd_micros_per_million, Some(6_000_000));
        assert_eq!(fast.output_usd_micros_per_million, Some(30_000_000));
        assert_eq!(fast.cached_input_usd_micros_per_million, Some(600_000));
        assert_eq!(fast.cache_write_usd_micros_per_million, Some(7_500_000));
        assert!(sonnet.capabilities().reasoning_effort_levels);

        // Bedrock on-demand access needs the `us.` inference profile, and the
        // model caches, which the codec gates on.
        let bedrock = catalog.model("bedrock", "anthropic.claude-sonnet-4-6")?;
        assert_eq!(bedrock.api_model(), "us.anthropic.claude-sonnet-4-6");
        assert!(bedrock.capabilities().caching);
        assert!(bedrock.capabilities().reasoning_effort_levels);
        assert_eq!(
            bedrock.limits().map(|limits| limits.max_output_tokens),
            Some(64_000)
        );
        let pricing = bedrock
            .pricing()
            .ok_or("the Bedrock model should be priced")?;
        assert_eq!(pricing.input_usd_micros_per_million, Some(3_000_000));
        assert_eq!(pricing.output_usd_micros_per_million, Some(15_000_000));
        assert_eq!(pricing.cached_input_usd_micros_per_million, Some(300_000));
        assert_eq!(pricing.cache_write_usd_micros_per_million, Some(3_750_000));
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

        let Err(CatalogError::Layer { source, .. }) = result else {
            return Err("expected a layer failure".into());
        };
        assert!(matches!(
            *source,
            CatalogError::DuplicateProviderAlias { alias } if alias == "second"
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

        let Err(CatalogError::Layer { layer, source }) = result else {
            return Err("expected a layer failure".into());
        };
        assert_eq!(layer, "overlay 1");
        assert!(matches!(
            *source,
            CatalogError::DuplicateModelSelector { provider, selector }
                if provider.as_str() == "test" && selector == "shared"
        ));
        Ok(())
    }
}
