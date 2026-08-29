use serde::{Deserialize, Serialize};

use super::{Metadata, ModelId, ProviderId};
use crate::types::Speed;

/// Portable model capabilities.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelCapabilities {
    #[serde(default)]
    pub text:                    bool,
    #[serde(default)]
    pub images:                  bool,
    #[serde(default)]
    pub audio:                   bool,
    #[serde(default)]
    pub documents:               bool,
    #[serde(default)]
    pub tools:                   bool,
    #[serde(default)]
    pub structured_output:       bool,
    #[serde(default)]
    pub reasoning:               bool,
    /// The model takes a named reasoning effort level.
    ///
    /// A reasoning model without this capability takes only a reasoning token
    /// budget, so a codec converts a requested effort into a budget.
    #[serde(default)]
    pub reasoning_effort_levels: bool,
    #[serde(default)]
    pub caching:                 bool,
    #[serde(default)]
    pub sampling:                bool,
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
/// `cached_input_usd_micros_per_million` prices cache reads and
/// `cache_write_usd_micros_per_million` prices tokens written into a provider
/// cache. Callers that do not find either should fall back to the input rate.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Pricing {
    pub input_usd_micros_per_million: Option<u64>,
    pub output_usd_micros_per_million: Option<u64>,
    pub cached_input_usd_micros_per_million: Option<u64>,
    pub cache_write_usd_micros_per_million: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub long_context: Option<LongContextPricing>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<SpeedPricing>,
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

/// Rate overrides for each requested speed.
///
/// A speed the model does not price keeps the base rates.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpeedPricing {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fast:       Option<SpeedRates>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub balanced:   Option<SpeedRates>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub economical: Option<SpeedRates>,
}

impl SpeedPricing {
    /// The overrides for one speed, if the model prices that speed.
    #[must_use]
    pub fn for_speed(self, speed: Speed) -> Option<SpeedRates> {
        match speed {
            Speed::Fast => self.fast,
            Speed::Balanced => self.balanced,
            Speed::Economical => self.economical,
        }
    }
}

/// The rates one speed replaces.
///
/// Each rate is optional, so a speed that changes only some rates states only
/// those and the rest stay at their base value.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpeedRates {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_usd_micros_per_million:        Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_usd_micros_per_million:       Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_input_usd_micros_per_million: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
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
            speed: self.speed,
        }
    }

    /// Applies the rate overrides for the request's speed.
    ///
    /// A request with no speed, a model with no speed rates, and a speed the
    /// model does not price all keep the rates unchanged. Speed overrides
    /// apply after [`Pricing::for_input_tokens`], so a speed rate wins over a
    /// long-context rate.
    #[must_use]
    pub fn for_speed(self, speed: Option<Speed>) -> Self {
        let Some(rates) = speed
            .zip(self.speed)
            .and_then(|(speed, pricing)| pricing.for_speed(speed))
        else {
            return self;
        };
        Self {
            input_usd_micros_per_million: rates
                .input_usd_micros_per_million
                .or(self.input_usd_micros_per_million),
            output_usd_micros_per_million: rates
                .output_usd_micros_per_million
                .or(self.output_usd_micros_per_million),
            cached_input_usd_micros_per_million: rates
                .cached_input_usd_micros_per_million
                .or(self.cached_input_usd_micros_per_million),
            cache_write_usd_micros_per_million: rates
                .cache_write_usd_micros_per_million
                .or(self.cache_write_usd_micros_per_million),
            long_context: self.long_context,
            speed: self.speed,
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
    #[serde(skip)]
    passthrough:  bool,
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

    /// Whether this model was synthesized for a passthrough route.
    ///
    /// A passthrough model is not described by the catalog, so its
    /// capabilities are unknown rather than absent. Request validation trusts
    /// the caller and lets the provider reject what the model cannot do.
    pub fn is_passthrough(&self) -> bool {
        self.passthrough
    }

    pub(crate) fn set_identity(&mut self, provider: ProviderId, id: ModelId) {
        self.provider = provider;
        self.id = id;
    }

    pub(crate) fn passthrough(provider: ProviderId, model: ModelId) -> Self {
        Self {
            provider,
            passthrough: true,
            display_name: model.to_string(),
            api_model: model.to_string(),
            id: model,
            aliases: Vec::new(),
            limits: None,
            // Conservative flags: codecs that gate wire behavior on a
            // capability (cache points, effort levels) stay on their safe
            // path. Validation is skipped instead via `is_passthrough`.
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

    use super::{LongContextPricing, ModelCapabilities, Pricing};
    use crate::types::Speed;

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
            speed: None,
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

    #[test]
    fn reads_speed_rates_from_toml_and_applies_them() -> Result<(), Box<dyn StdError>> {
        let pricing = toml::from_str::<Pricing>(
            r"
            input_usd_micros_per_million = 100
            output_usd_micros_per_million = 200
            cached_input_usd_micros_per_million = 10
            cache_write_usd_micros_per_million = 125
            speed = { fast = { input_usd_micros_per_million = 200, output_usd_micros_per_million = 400 } }
            ",
        )?;

        let fast = pricing.for_speed(Some(Speed::Fast));

        assert_eq!(fast.input_usd_micros_per_million, Some(200));
        assert_eq!(fast.output_usd_micros_per_million, Some(400));
        // A rate the speed leaves out keeps its base value.
        assert_eq!(fast.cached_input_usd_micros_per_million, Some(10));
        assert_eq!(fast.cache_write_usd_micros_per_million, Some(125));
        Ok(())
    }

    #[test]
    fn keeps_the_base_rates_without_a_matching_speed() {
        let pricing = sample_pricing();

        assert_eq!(
            pricing.for_speed(None).input_usd_micros_per_million,
            Some(1)
        );
        assert_eq!(
            pricing
                .for_speed(Some(Speed::Fast))
                .input_usd_micros_per_million,
            Some(1)
        );
    }

    #[test]
    fn reads_the_effort_level_capability_from_toml() -> Result<(), Box<dyn StdError>> {
        let capabilities = toml::from_str::<ModelCapabilities>(
            r"
            reasoning = true
            reasoning_effort_levels = true
            ",
        )?;

        assert!(capabilities.reasoning_effort_levels);
        // The capability defaults to false, so a reasoning model states it.
        assert!(!toml::from_str::<ModelCapabilities>("reasoning = true")?.reasoning_effort_levels);
        Ok(())
    }
}
