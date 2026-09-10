//! Queries over a built catalog: which providers are on, which model a
//! selector names, and which model to pick for a job.
//!
//! These answer the questions an application asks before it has a request:
//! when it lists models, chooses a default, picks the model for a probe or a
//! cheap utility call, or canonicalizes a name an operator typed. At request
//! time the [resolver](crate::resolver) makes the same decisions from a
//! [`Request`](crate::types::Request); the two agree on ranking.
//!
//! Every query skips disabled providers. A provider the catalog marks
//! `enabled = false` is still visible through [`Catalog::providers`] and
//! [`Catalog::provider`], but it offers nothing here.

use std::collections::BTreeSet;

use super::{Catalog, CatalogModel, CatalogProvider, ModelHandle, ModelId, ProviderId};
use crate::resolver::ResolvedRoute;
use crate::types::{Cost, Speed, TokenCounts};

/// One provider's offering of one model, borrowed from the catalog.
///
/// [`ResolvedRoute`] is the owned form a request carries; this is the borrowed
/// form a listing or a picker works with.
#[derive(Clone, Copy, Debug)]
pub struct Offering<'a> {
    pub provider: &'a CatalogProvider,
    pub model:    &'a CatalogModel,
}

impl Offering<'_> {
    pub fn handle(&self) -> ModelHandle {
        ModelHandle::new(self.provider.id().clone(), self.model.id().clone())
    }

    /// The owned route for this offering.
    pub fn into_route(self) -> ResolvedRoute {
        ResolvedRoute::try_new(self.provider.clone(), self.model.clone())
            .expect("an offering pairs a model with its own provider")
    }

    /// Estimates the catalog cost of `usage` on this offering, when the
    /// catalog prices it.
    pub fn estimate_cost(&self, usage: TokenCounts, speed: Option<Speed>) -> Option<Cost> {
        self.into_route().estimate_cost(usage, speed)
    }
}

impl CatalogProvider {
    /// This provider's offering of `selector`, by canonical id, alias, or
    /// wire id.
    pub fn offering(&self, selector: &str) -> Option<Offering<'_>> {
        self.model(selector).map(|model| Offering {
            provider: self,
            model,
        })
    }

    /// Every model this provider offers, in catalog order.
    pub fn offerings(&self) -> impl ExactSizeIterator<Item = Offering<'_>> {
        self.models().map(|model| Offering {
            provider: self,
            model,
        })
    }

    /// The provider's default model, when it names one the catalog knows.
    pub fn default_offering(&self) -> Option<Offering<'_>> {
        self.default_model()
            .and_then(|selector| self.offering(selector))
    }

    /// The model to probe this provider with: the row marked `probe`, else
    /// the default.
    pub fn probe_offering(&self) -> Option<Offering<'_>> {
        self.offerings()
            .find(|offering| offering.model.is_probe())
            .or_else(|| self.default_offering())
    }

    /// This provider's model closest to `reference`: the same tools, image,
    /// and reasoning support, and the nearest input price. For provider-level
    /// fallbacks, where the reference model lives on another provider.
    pub fn closest_offering(&self, reference: &CatalogModel) -> Option<Offering<'_>> {
        let reference_caps = reference.capabilities();
        let reference_price = input_price(reference);
        self.offerings()
            .filter(|offering| {
                let caps = offering.model.capabilities();
                caps.tools().is_supported() == reference_caps.tools().is_supported()
                    && caps.images().is_supported() == reference_caps.images().is_supported()
                    && caps.reasoning().is_supported() == reference_caps.reasoning().is_supported()
            })
            .min_by_key(|offering| input_price(offering.model).abs_diff(reference_price))
    }
}

fn input_price(model: &CatalogModel) -> u64 {
    model
        .pricing()
        .and_then(|pricing| pricing.input_usd_micros_per_million)
        .unwrap_or(0)
}

impl Catalog {
    /// Enabled providers, highest priority first, ties broken by id.
    pub fn enabled_providers(&self) -> Vec<&CatalogProvider> {
        let mut providers: Vec<_> = self.providers().filter(|p| p.is_enabled()).collect();
        providers.sort_by(|left, right| {
            right
                .priority()
                .cmp(&left.priority())
                .then_with(|| left.id().cmp(right.id()))
        });
        providers
    }

    /// Enabled providers an application lists as offerings of their own.
    /// A provider that stands in for another routes requests but is not
    /// listed.
    pub fn listed_providers(&self) -> Vec<&CatalogProvider> {
        self.enabled_providers()
            .into_iter()
            .filter(|provider| provider.stands_in_for().is_none())
            .collect()
    }

    /// The ids of every enabled provider.
    pub fn enabled_provider_ids(&self) -> BTreeSet<ProviderId> {
        self.providers()
            .filter(|provider| provider.is_enabled())
            .map(|provider| provider.id().clone())
            .collect()
    }

    /// An enabled provider by id or alias.
    pub fn enabled_provider(&self, selector: &str) -> Option<&CatalogProvider> {
        self.find_provider(selector)
            .filter(|provider| provider.is_enabled())
    }

    /// Every enabled offering of `selector`, ranked as the resolver ranks
    /// them: a model whose canonical id is `selector` before any alias for
    /// it, then higher provider priority, then provider id.
    pub fn offerings_matching(&self, selector: &str) -> Vec<Offering<'_>> {
        let mut matches: Vec<Offering<'_>> = self
            .enabled_providers()
            .into_iter()
            .flat_map(CatalogProvider::offerings)
            .filter(|offering| {
                offering.model.id().as_str() == selector
                    || offering
                        .model
                        .aliases()
                        .iter()
                        .any(|alias| alias == selector)
            })
            .collect();
        // `enabled_providers` already orders by priority then id, and the
        // sort is stable, so ranking by alias-ness alone keeps that order
        // within each group.
        matches.sort_by_key(|offering| offering.model.id().as_str() != selector);
        matches
    }

    /// Whether `selector` names a model on any enabled provider.
    pub fn is_model_selector(&self, selector: &str) -> bool {
        !self.offerings_matching(selector).is_empty()
    }

    /// The canonical id for `selector`: the named provider's offering when
    /// one is given and has it, else the best-ranked offering anywhere.
    /// `None` when no enabled provider offers it.
    pub fn canonical_model_id(
        &self,
        provider: Option<&ProviderId>,
        selector: &str,
    ) -> Option<&ModelId> {
        provider
            .and_then(|id| self.enabled_provider(id.as_str()))
            .and_then(|provider| provider.offering(selector))
            .or_else(|| self.offerings_matching(selector).into_iter().next())
            .map(|offering| offering.model.id())
    }

    /// The default offering across `ready` providers: the highest-priority
    /// ready provider's default. Falls back to any enabled provider's default
    /// when none is ready, so a caller always has a model to name.
    pub fn default_offering_for<'a>(
        &self,
        ready: impl IntoIterator<Item = &'a ProviderId>,
    ) -> Option<Offering<'_>> {
        let ready: BTreeSet<&ProviderId> = ready.into_iter().collect();
        let providers = self.enabled_providers();
        providers
            .iter()
            .filter(|provider| ready.contains(provider.id()))
            .chain(providers.iter())
            .find_map(|provider| provider.default_offering())
    }

    /// The model for cheap utility calls across `ready` providers: the first
    /// `small_default` row in provider priority order, else the ready
    /// default.
    pub fn small_default_for<'a>(
        &self,
        ready: impl IntoIterator<Item = &'a ProviderId>,
    ) -> Option<Offering<'_>> {
        let ready: BTreeSet<&ProviderId> = ready.into_iter().collect();
        self.enabled_providers()
            .into_iter()
            .filter(|provider| ready.contains(provider.id()))
            .flat_map(CatalogProvider::offerings)
            .find(|offering| offering.model.is_small_default())
            .or_else(|| self.default_offering_for(ready))
    }

    /// Estimates the catalog cost of `usage` on `handle`, when the catalog
    /// knows and prices that offering.
    pub fn estimate_cost(
        &self,
        handle: &ModelHandle,
        usage: TokenCounts,
        speed: Option<Speed>,
    ) -> Option<Cost> {
        self.enabled_provider(handle.provider().as_str())?
            .offering(handle.model().as_str())?
            .estimate_cost(usage, speed)
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use super::super::{Catalog, ProviderId};
    use crate::types::TokenCounts;

    const CATALOG: &str = r#"
        schema_version = 1

        [providers.high]
        display_name = "High"
        adapter = "openai-compatible"
        codec = "openai-chat"
        base_url = "http://127.0.0.1"
        priority = 100
        default_model = "big"
        auth = { type = "bearer" }

        [providers.high.models.big]
        display_name = "Big"
        api_model = "big"
        aliases = ["shared"]
        capabilities = { text = true, tools = true, reasoning = true }
        pricing = { input_usd_micros_per_million = 10000000, output_usd_micros_per_million = 20000000 }

        [providers.high.models.small]
        display_name = "Small"
        api_model = "small"
        capabilities = { text = true, tools = true }
        pricing = { input_usd_micros_per_million = 1000000, output_usd_micros_per_million = 2000000 }
        small_default = true
        probe = true

        [providers.low]
        display_name = "Low"
        aliases = ["lo"]
        adapter = "openai-compatible"
        codec = "openai-chat"
        base_url = "http://127.0.0.1"
        priority = 10
        default_model = "shared"
        auth = { type = "bearer" }

        [providers.low.models.shared]
        display_name = "Shared, by name"
        api_model = "shared"
        capabilities = { text = true, tools = true }
        pricing = { input_usd_micros_per_million = 3000000, output_usd_micros_per_million = 6000000 }

        [providers.low.models.cheap]
        display_name = "Cheap"
        api_model = "cheap"
        capabilities = { text = true, tools = true, reasoning = true }
        pricing = { input_usd_micros_per_million = 500000, output_usd_micros_per_million = 1000000 }

        [providers.off]
        display_name = "Off"
        adapter = "openai-compatible"
        codec = "openai-chat"
        base_url = "http://127.0.0.1"
        priority = 200
        enabled = false
        default_model = "shared"
        auth = { type = "bearer" }

        [providers.off.models.shared]
        display_name = "Shared but off"
        api_model = "shared"
        capabilities = { text = true }

        [providers.seat]
        display_name = "Seat"
        adapter = "openai-compatible"
        codec = "openai-chat"
        base_url = "http://127.0.0.1"
        priority = 99
        stands_in_for = "high"
        default_model = "big"
        auth = { type = "bearer" }

        [providers.seat.models.big]
        display_name = "Big, by seat"
        api_model = "big"
        capabilities = { text = true }
    "#;

    fn catalog() -> Result<Catalog, Box<dyn StdError>> {
        Ok(Catalog::builder().overlay_toml(CATALOG)?.build()?)
    }

    fn ids<'a>(providers: &[&'a super::CatalogProvider]) -> Vec<&'a str> {
        providers.iter().map(|p| p.id().as_str()).collect()
    }

    #[test]
    fn enabled_providers_rank_by_priority_and_skip_disabled() -> Result<(), Box<dyn StdError>> {
        let catalog = catalog()?;
        assert_eq!(ids(&catalog.enabled_providers()), ["high", "seat", "low"]);
        assert_eq!(ids(&catalog.listed_providers()), ["high", "low"]);
        assert_eq!(
            catalog.enabled_provider_ids(),
            ["high", "low", "seat"].map(ProviderId::new).into()
        );
        assert!(catalog.enabled_provider("lo").is_some(), "aliases resolve");
        assert!(catalog.enabled_provider("off").is_none());
        assert!(
            catalog.provider("off").is_ok(),
            "disabled stays in the catalog"
        );
        Ok(())
    }

    #[test]
    fn offerings_matching_puts_a_named_model_before_an_alias_for_it()
    -> Result<(), Box<dyn StdError>> {
        let catalog = catalog()?;
        let handles: Vec<String> = catalog
            .offerings_matching("shared")
            .iter()
            .map(|offering| offering.handle().to_string())
            .collect();
        assert_eq!(handles, ["low/shared", "high/big"]);
        assert!(catalog.is_model_selector("shared"));
        assert!(!catalog.is_model_selector("nothing"));
        Ok(())
    }

    #[test]
    fn canonical_model_id_prefers_the_named_provider() -> Result<(), Box<dyn StdError>> {
        let catalog = catalog()?;
        let high = ProviderId::new("high");
        assert_eq!(
            catalog
                .canonical_model_id(Some(&high), "shared")
                .map(ToString::to_string),
            Some("big".to_owned())
        );
        assert_eq!(
            catalog
                .canonical_model_id(None, "shared")
                .map(ToString::to_string),
            Some("shared".to_owned())
        );
        assert!(catalog.canonical_model_id(None, "nothing").is_none());
        Ok(())
    }

    #[test]
    fn defaults_probe_and_small_default_follow_readiness() -> Result<(), Box<dyn StdError>> {
        let catalog = catalog()?;
        let high = ProviderId::new("high");
        let low = ProviderId::new("low");
        let handle = |offering: Option<super::Offering<'_>>| {
            offering.map(|offering| offering.handle().to_string())
        };

        assert_eq!(
            handle(catalog.default_offering_for([&low])),
            Some("low/shared".to_owned())
        );
        assert_eq!(
            handle(catalog.default_offering_for([&high, &low])),
            Some("high/big".to_owned())
        );
        assert_eq!(
            handle(catalog.default_offering_for([])),
            Some("high/big".to_owned()),
            "no ready provider falls back to the best enabled default"
        );
        assert_eq!(
            handle(catalog.small_default_for([&high])),
            Some("high/small".to_owned())
        );
        assert_eq!(
            handle(catalog.small_default_for([&low])),
            Some("low/shared".to_owned()),
            "a provider without a small default lends its default"
        );
        let high_provider = catalog.enabled_provider("high").unwrap();
        assert_eq!(
            handle(high_provider.probe_offering()),
            Some("high/small".to_owned())
        );
        let low_provider = catalog.enabled_provider("low").unwrap();
        assert_eq!(
            handle(low_provider.probe_offering()),
            Some("low/shared".to_owned()),
            "no probe row falls back to the default"
        );
        Ok(())
    }

    #[test]
    fn closest_offering_matches_capabilities_then_price() -> Result<(), Box<dyn StdError>> {
        let catalog = catalog()?;
        let reference = catalog.model("high", "big")?;
        let low = catalog.enabled_provider("low").unwrap();
        let closest = low.closest_offering(reference).unwrap();
        assert_eq!(
            closest.model.id().as_str(),
            "cheap",
            "the only low model with reasoning support"
        );
        let reference = catalog.model("high", "small")?;
        let closest = low.closest_offering(reference).unwrap();
        assert_eq!(closest.model.id().as_str(), "shared");
        Ok(())
    }

    #[test]
    fn estimate_cost_reads_the_offering_pricing() -> Result<(), Box<dyn StdError>> {
        let catalog = catalog()?;
        let usage = TokenCounts {
            input: 1_000_000,
            output: 0,
            ..TokenCounts::default()
        };
        let handle = super::ModelHandle::new(ProviderId::new("high"), super::ModelId::new("small"));
        let cost = catalog.estimate_cost(&handle, usage, None).unwrap();
        assert_eq!(cost.usd_micros, 1_000_000);
        let off = super::ModelHandle::new(ProviderId::new("off"), super::ModelId::new("shared"));
        assert!(catalog.estimate_cost(&off, usage, None).is_none());
        Ok(())
    }
}
