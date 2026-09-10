use serde::{Deserialize, Serialize};

use super::{
    CatalogError, Metadata, ModelCapabilities, ModelHandle, ModelId, ModelProtocolOptions,
    ProviderId,
};
use crate::types::Speed;

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
    /// Each rate is optional, so a long-context block that changes only some
    /// rates states only those and the rest stay at their base value.
    #[must_use]
    pub fn for_input_tokens(self, input_tokens: u64) -> Self {
        let Some(long_context) = self
            .long_context
            .filter(|rates| input_tokens > rates.above_input_tokens)
        else {
            return self;
        };
        Self {
            input_usd_micros_per_million: long_context
                .input_usd_micros_per_million
                .or(self.input_usd_micros_per_million),
            output_usd_micros_per_million: long_context
                .output_usd_micros_per_million
                .or(self.output_usd_micros_per_million),
            cached_input_usd_micros_per_million: long_context
                .cached_input_usd_micros_per_million
                .or(self.cached_input_usd_micros_per_million),
            cache_write_usd_micros_per_million: long_context
                .cache_write_usd_micros_per_million
                .or(self.cache_write_usd_micros_per_million),
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

/// Parsing record without catalog identity. Never exposed as a validated model.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ModelRecord {
    display_name:         String,
    #[serde(default)]
    aliases:              Vec<String>,
    api_model:            String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    family:               Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    training_cutoff:      Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    knowledge_cutoff:     Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    estimated_output_tps: Option<f64>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    small_default:        bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    probe:                bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    limits:               Option<ModelLimits>,
    #[serde(default)]
    capabilities:         ModelCapabilities,
    #[serde(default)]
    protocol_options:     ModelProtocolOptions,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pricing:              Option<Pricing>,
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    metadata:             Metadata,
}

/// Model-level catalog facts.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogModel {
    #[serde(skip)]
    provider:             ProviderId,
    #[serde(skip)]
    id:                   ModelId,
    #[serde(skip)]
    passthrough:          bool,
    display_name:         String,
    #[serde(default)]
    aliases:              Vec<String>,
    api_model:            String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    family:               Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    training_cutoff:      Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    knowledge_cutoff:     Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    estimated_output_tps: Option<f64>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    small_default:        bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    probe:                bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    limits:               Option<ModelLimits>,
    #[serde(default)]
    capabilities:         ModelCapabilities,
    #[serde(default)]
    protocol_options:     ModelProtocolOptions,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pricing:              Option<Pricing>,
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    metadata:             Metadata,
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

    /// The model family label a picker groups this model under, such as
    /// `claude-5` or `gpt-5`.
    pub fn family(&self) -> Option<&str> {
        self.family.as_deref()
    }

    /// The training data cutoff as the provider states it, usually an ISO
    /// date.
    pub fn training_cutoff(&self) -> Option<&str> {
        self.training_cutoff.as_deref()
    }

    /// The knowledge cutoff as a person would write it, for display.
    pub fn knowledge_cutoff(&self) -> Option<&str> {
        self.knowledge_cutoff.as_deref()
    }

    /// A rough sustained output rate in tokens per second, for display and
    /// for choosing between otherwise equivalent models.
    pub fn estimated_output_tps(&self) -> Option<f64> {
        self.estimated_output_tps
    }

    /// Whether this is the provider's model for cheap utility calls such as
    /// title generation. At most one model per provider should say so.
    pub fn is_small_default(&self) -> bool {
        self.small_default
    }

    /// Whether this is the provider's model for connectivity probes: cheap,
    /// fast, and available on every account tier.
    pub fn is_probe(&self) -> bool {
        self.probe
    }

    pub fn limits(&self) -> Option<ModelLimits> {
        self.limits
    }

    pub fn capabilities(&self) -> ModelCapabilities {
        self.capabilities
    }

    pub fn protocol_options(&self) -> ModelProtocolOptions {
        self.protocol_options
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

    pub(super) fn try_from_record(
        provider: ProviderId,
        id: ModelId,
        record: ModelRecord,
    ) -> Result<Self, CatalogError> {
        super::validate_identifier("model", id.as_str(), true)?;
        if record.display_name.trim().is_empty() {
            return Err(CatalogError::EmptyDisplayName {
                item: format!("{provider}/{id}"),
            });
        }
        for alias in &record.aliases {
            super::validate_identifier("model alias", alias, false)?;
        }
        if record
            .limits
            .is_some_and(|limits| limits.max_output_tokens > limits.context_tokens)
        {
            return Err(CatalogError::InvalidModelLimits {
                model: ModelHandle::new(provider, id),
            });
        }
        Ok(Self {
            provider,
            id,
            passthrough: false,
            display_name: record.display_name,
            aliases: record.aliases,
            api_model: record.api_model,
            family: record.family,
            training_cutoff: record.training_cutoff,
            knowledge_cutoff: record.knowledge_cutoff,
            estimated_output_tps: record.estimated_output_tps,
            small_default: record.small_default,
            probe: record.probe,
            limits: record.limits,
            capabilities: record.capabilities,
            protocol_options: record.protocol_options,
            pricing: record.pricing,
            metadata: record.metadata,
        })
    }

    pub(crate) fn passthrough(provider: ProviderId, model: ModelId) -> Self {
        Self {
            provider,
            passthrough: true,
            display_name: model.to_string(),
            api_model: model.to_string(),
            id: model,
            aliases: Vec::new(),
            family: None,
            training_cutoff: None,
            knowledge_cutoff: None,
            estimated_output_tps: None,
            small_default: false,
            probe: false,
            limits: None,
            capabilities: ModelCapabilities::unknown(),
            protocol_options: ModelProtocolOptions::default(),
            pricing: None,
            metadata: Metadata::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use super::{LongContextPricing, Pricing};
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
    fn long_context_rates_inherit_missing_rates_from_the_base() {
        let pricing = Pricing {
            long_context: Some(LongContextPricing {
                above_input_tokens:                  200_000,
                input_usd_micros_per_million:        Some(4),
                output_usd_micros_per_million:       None,
                cached_input_usd_micros_per_million: None,
                cache_write_usd_micros_per_million:  None,
            }),
            ..sample_pricing()
        };

        let tiered = pricing.for_input_tokens(200_001);

        assert_eq!(tiered.input_usd_micros_per_million, Some(4));
        // Rates the tier leaves out keep their base value.
        assert_eq!(tiered.output_usd_micros_per_million, Some(2));
        assert_eq!(tiered.cached_input_usd_micros_per_million, Some(3));
        assert_eq!(tiered.cache_write_usd_micros_per_million, Some(7));
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
}
