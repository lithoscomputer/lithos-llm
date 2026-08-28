use serde::{Deserialize, Serialize};

use super::{Metadata, ModelId, ProviderId};

/// Portable model capabilities.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelCapabilities {
    #[serde(default)]
    pub text:              bool,
    #[serde(default)]
    pub images:            bool,
    #[serde(default)]
    pub audio:             bool,
    #[serde(default)]
    pub documents:         bool,
    #[serde(default)]
    pub tools:             bool,
    #[serde(default)]
    pub structured_output: bool,
    #[serde(default)]
    pub reasoning:         bool,
    #[serde(default)]
    pub caching:           bool,
    #[serde(default)]
    pub sampling:          bool,
}

/// Context and output token limits.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelLimits {
    pub context_tokens:    u64,
    pub max_output_tokens: u64,
}

/// Catalog pricing in US dollar micros per million tokens.
///
/// `cache_write_usd_micros_per_million` prices tokens written into a provider
/// cache. Callers that do not find it should fall back to the input rate.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Pricing {
    pub input_usd_micros_per_million: Option<u64>,
    pub output_usd_micros_per_million: Option<u64>,
    pub cached_input_usd_micros_per_million: Option<u64>,
    pub cache_write_usd_micros_per_million: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub long_context: Option<LongContextPricing>,
}

/// Alternate rates selected when input crosses a model billing threshold.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LongContextPricing {
    pub above_input_tokens:                  u64,
    pub input_usd_micros_per_million:        Option<u64>,
    pub output_usd_micros_per_million:       Option<u64>,
    pub cached_input_usd_micros_per_million: Option<u64>,
    pub cache_write_usd_micros_per_million:  Option<u64>,
}

impl Pricing {
    /// Selects the rates that apply to a request with `input_tokens` of input.
    ///
    /// Long-context rates replace every base rate, including the cache-write
    /// rate, so a long-context block that omits one rate reports no rate.
    #[must_use]
    pub fn for_input_tokens(self, input_tokens: u64) -> Self {
        let Some(long_context) = self
            .long_context
            .filter(|rates| input_tokens > rates.above_input_tokens)
        else {
            return self;
        };
        Self {
            input_usd_micros_per_million: long_context.input_usd_micros_per_million,
            output_usd_micros_per_million: long_context.output_usd_micros_per_million,
            cached_input_usd_micros_per_million: long_context.cached_input_usd_micros_per_million,
            cache_write_usd_micros_per_million: long_context.cache_write_usd_micros_per_million,
            long_context: self.long_context,
        }
    }
}

/// Model-level catalog facts.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogModel {
    #[serde(skip)]
    provider:     ProviderId,
    #[serde(skip)]
    id:           ModelId,
    display_name: String,
    #[serde(default)]
    aliases:      Vec<String>,
    api_model:    String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    limits:       Option<ModelLimits>,
    #[serde(default)]
    capabilities: ModelCapabilities,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pricing:      Option<Pricing>,
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    metadata:     Metadata,
}

impl CatalogModel {
    pub fn provider_id(&self) -> &ProviderId {
        &self.provider
    }

    pub fn id(&self) -> &ModelId {
        &self.id
    }

    pub fn display_name(&self) -> &str {
        &self.display_name
    }

    pub fn aliases(&self) -> &[String] {
        &self.aliases
    }

    pub fn api_model(&self) -> &str {
        &self.api_model
    }

    pub fn limits(&self) -> Option<ModelLimits> {
        self.limits
    }

    pub fn capabilities(&self) -> ModelCapabilities {
        self.capabilities
    }

    pub fn pricing(&self) -> Option<Pricing> {
        self.pricing
    }

    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    pub(crate) fn set_identity(&mut self, provider: ProviderId, id: ModelId) {
        self.provider = provider;
        self.id = id;
    }

    pub(crate) fn passthrough(provider: ProviderId, model: ModelId) -> Self {
        Self {
            provider,
            display_name: model.to_string(),
            api_model: model.to_string(),
            id: model,
            aliases: Vec::new(),
            limits: None,
            capabilities: ModelCapabilities {
                text: true,
                ..ModelCapabilities::default()
            },
            pricing: None,
            metadata: Metadata::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use super::{LongContextPricing, Pricing};

    fn sample_pricing() -> Pricing {
        Pricing {
            input_usd_micros_per_million: Some(1),
            output_usd_micros_per_million: Some(2),
            cached_input_usd_micros_per_million: Some(3),
            cache_write_usd_micros_per_million: Some(7),
            long_context: Some(LongContextPricing {
                above_input_tokens:                  200_000,
                input_usd_micros_per_million:        Some(4),
                output_usd_micros_per_million:       Some(5),
                cached_input_usd_micros_per_million: Some(6),
                cache_write_usd_micros_per_million:  Some(8),
            }),
        }
    }

    #[test]
    fn selects_long_context_rates_above_the_threshold() {
        let pricing = sample_pricing();

        assert_eq!(
            pricing
                .for_input_tokens(200_000)
                .input_usd_micros_per_million,
            Some(1)
        );
        assert_eq!(
            pricing
                .for_input_tokens(200_001)
                .input_usd_micros_per_million,
            Some(4)
        );
    }

    #[test]
    fn carries_the_cache_write_rate_across_the_threshold() {
        let pricing = sample_pricing();

        assert_eq!(
            pricing
                .for_input_tokens(200_000)
                .cache_write_usd_micros_per_million,
            Some(7)
        );
        assert_eq!(
            pricing
                .for_input_tokens(200_001)
                .cache_write_usd_micros_per_million,
            Some(8)
        );
    }

    #[test]
    fn reads_the_cache_write_rate_from_toml() -> Result<(), Box<dyn StdError>> {
        let pricing = toml::from_str::<Pricing>(
            r"
            input_usd_micros_per_million = 1
            output_usd_micros_per_million = 2
            cached_input_usd_micros_per_million = 3
            cache_write_usd_micros_per_million = 7
            long_context = { above_input_tokens = 100, cache_write_usd_micros_per_million = 8 }
            ",
        )?;

        assert_eq!(pricing.cache_write_usd_micros_per_million, Some(7));
        assert_eq!(
            pricing
                .long_context
                .and_then(|rates| rates.cache_write_usd_micros_per_million),
            Some(8)
        );
        Ok(())
    }
}
