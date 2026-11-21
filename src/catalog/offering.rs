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
use std::ptr;

use super::{
    Catalog, CatalogModel, CatalogProvider, CodecId, ModelHandle, ModelId, ProviderId,
    SelectorMatch,
};
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
    /// them.
    ///
    /// A `provider/model` selector whose prefix is an enabled provider's id
    /// or alias names that provider's offering of `model`, by canonical id,
    /// alias, or wire id, as it does for the resolver, and ranks first:
    /// `anthropic/claude-opus-5-5`. Then come the offerings that `selector`
    /// names whole: a model whose canonical id is `selector`, then one with
    /// `selector` as an alias, then one with `selector` as its wire id
    /// (`api_model`), such as Anthropic's `claude-opus-5-5` or OpenRouter's
    /// `anthropic/claude-opus-5.5`. Each group is ordered by higher provider
    /// priority, then provider id.
    ///
    /// [`selector_matches`](Self::selector_matches) returns the same list
    /// with how `selector` named each offering.
    pub fn offerings_matching(&self, selector: &str) -> Vec<Offering<'_>> {
        self.selector_matches(selector)
            .into_iter()
            .map(|(offering, _)| offering)
            .collect()
    }

    /// [`offerings_matching`](Self::offerings_matching), in the same order,
    /// with how `selector` named each offering.
    ///
    /// The kinds come in [`SelectorMatch`] order, so the first entry's kind
    /// is the best match. An offering appears once, under its best kind: when
    /// `anthropic/claude-fable-5` names Anthropic's `claude-fable-5` and is
    /// also OpenRouter's wire id for the same model, Anthropic's row is
    /// [`ProviderQualified`](SelectorMatch::ProviderQualified) and
    /// OpenRouter's follows as [`WireId`](SelectorMatch::WireId).
    ///
    /// A caller that accepts only provider ids as qualifiers skips
    /// [`ProviderAliasQualified`](SelectorMatch::ProviderAliasQualified)
    /// entries; the offerings the selector names whole still follow.
    pub fn selector_matches(&self, selector: &str) -> Vec<(Offering<'_>, SelectorMatch)> {
        let qualified = selector.split_once('/').and_then(|(prefix, model)| {
            let provider = self.enabled_provider(prefix)?;
            let kind = if provider.id().as_str() == prefix {
                SelectorMatch::ProviderQualified
            } else {
                SelectorMatch::ProviderAliasQualified
            };
            provider.offering(model).map(|offering| (offering, kind))
        });
        let mut matches: Vec<_> = self
            .enabled_providers()
            .into_iter()
            .flat_map(CatalogProvider::offerings)
            .filter(|offering| {
                qualified.is_none_or(|(qualified, _)| !ptr::eq(qualified.model, offering.model))
            })
            .filter_map(|offering| {
                offering
                    .model
                    .selector_match(selector)
                    .map(|kind| (offering, kind))
            })
            .collect();
        // `enabled_providers` already orders by priority then id, and the
        // sort is stable, so ranking by match kind alone keeps that order
        // within each group.
        matches.sort_by_key(|(_, kind)| *kind);
        qualified.into_iter().chain(matches).collect()
    }

    /// Every enabled provider's offering of the model whose canonical id is
    /// `model`, highest provider priority first, ties broken by provider id.
    ///
    /// Unlike [`offerings_matching`](Self::offerings_matching), this matches
    /// the canonical id only, so every entry is the same model and none is a
    /// different model that happens to carry `model` as an alias. A provider
    /// that stands in for another is included, because it routes requests.
    pub fn offerings(&self, model: &ModelId) -> Vec<Offering<'_>> {
        self.enabled_providers()
            .into_iter()
            .filter_map(|provider| {
                provider
                    .model_by_id(model)
                    .map(|model| Offering { provider, model })
            })
            .collect()
    }

    /// The [`offerings`](Self::offerings) of `model` that a call through
    /// `codec` reaches, in the same order: the candidates a caller tries in
    /// turn when it fails over between providers.
    pub fn offerings_for_codec(&self, model: &ModelId, codec: &CodecId) -> Vec<Offering<'_>> {
        let mut offerings = self.offerings(model);
        offerings.retain(|offering| offering.model.codecs().contains(codec));
        offerings
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

    use super::super::{Catalog, ProviderId, SelectorMatch};
    use crate::types::TokenCounts;

    const CATALOG: &str = r#"
        schema_version = 1

        [providers.high]
        display_name = "High"
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
    fn offerings_list_one_model_by_priority_and_filter_by_codec() -> Result<(), Box<dyn StdError>> {
        let catalog = Catalog::builder()
            .overlay_toml(
                r#"
                schema_version = 1

                [providers.alpha]
                display_name = "Alpha"
                adapter = "test-adapter"
                codecs = ["openai-chat", "anthropic-messages"]
                base_url = "http://127.0.0.1"
                priority = 10
                auth = { type = "bearer" }

                [providers.alpha.models.one]
                display_name = "One, by messages only"
                api_model = "alpha-one"
                codecs = ["anthropic-messages"]

                [providers.beta]
                display_name = "Beta"
                adapter = "test-adapter"
                codecs = ["openai-chat"]
                base_url = "http://127.0.0.1"
                priority = 50
                auth = { type = "bearer" }

                [providers.beta.models.one]
                display_name = "One"
                api_model = "beta/one"

                [providers.delta]
                display_name = "Delta"
                adapter = "test-adapter"
                codecs = ["openai-chat"]
                base_url = "http://127.0.0.1"
                priority = 60
                auth = { type = "bearer" }

                [providers.delta.models.two]
                display_name = "Two, aliased as one"
                api_model = "two"
                aliases = ["one"]

                [providers.gamma]
                display_name = "Gamma"
                adapter = "test-adapter"
                codecs = ["openai-chat"]
                base_url = "http://127.0.0.1"
                priority = 50
                stands_in_for = "beta"
                auth = { type = "bearer" }

                [providers.gamma.models.one]
                display_name = "One, by seat"
                api_model = "one"

                [providers.off]
                display_name = "Off"
                adapter = "test-adapter"
                codecs = ["openai-chat"]
                base_url = "http://127.0.0.1"
                priority = 100
                enabled = false
                auth = { type = "bearer" }

                [providers.off.models.one]
                display_name = "One but off"
                api_model = "one"
                "#,
            )?
            .build()?;
        let handles = |offerings: Vec<super::Offering<'_>>| -> Vec<String> {
            offerings
                .iter()
                .map(|offering| offering.handle().to_string())
                .collect()
        };
        let one = super::ModelId::new("one");

        assert_eq!(
            handles(catalog.offerings(&one)),
            ["beta/one", "gamma/one", "alpha/one"],
            "priority first, then id; a stand-in is included; an alias for another model and a \
             disabled provider are not"
        );
        assert_eq!(
            handles(catalog.offerings_for_codec(&one, &super::CodecId::new("openai-chat"))),
            ["beta/one", "gamma/one"],
            "alpha's row narrows itself to anthropic-messages"
        );
        assert_eq!(
            handles(catalog.offerings_for_codec(&one, &super::CodecId::new("anthropic-messages"))),
            ["alpha/one"]
        );
        assert!(
            catalog
                .offerings(&super::ModelId::new("beta/one"))
                .is_empty(),
            "a wire id is not a canonical id"
        );
        Ok(())
    }

    #[test]
    fn selector_matches_names_how_each_offering_matched() -> Result<(), Box<dyn StdError>> {
        // The provider whose wire id collides has the highest priority, so
        // only the match kind can put it last.
        let catalog = Catalog::builder()
            .overlay_toml(
                r#"
                schema_version = 1

                [providers.wire]
                display_name = "Wire"
                base_url = "http://127.0.0.1"
                priority = 100
                auth = { type = "bearer" }

                [providers.wire.models."model-1.0"]
                display_name = "Model 1.0"
                api_model = "model-1-0"

                [providers.wire.models.unrelated]
                display_name = "Unrelated"
                api_model = "named"

                [providers.wire.models.other]
                display_name = "Other"
                api_model = "nickname"

                [providers.wire.models.routed]
                display_name = "Routed"
                api_model = "low/named-v1"

                [providers.low]
                display_name = "Low"
                aliases = ["lo"]
                base_url = "http://127.0.0.1"
                priority = 1
                auth = { type = "bearer" }

                [providers.low.models.named]
                display_name = "Named"
                aliases = ["nickname"]
                api_model = "named-v1"
            "#,
            )?
            .build()?;
        let matches = |selector| -> Vec<(String, SelectorMatch)> {
            catalog
                .selector_matches(selector)
                .into_iter()
                .map(|(offering, kind)| (offering.handle().to_string(), kind))
                .collect()
        };
        let expected = |rows: &[(&str, SelectorMatch)]| -> Vec<(String, SelectorMatch)> {
            rows.iter()
                .map(|&(handle, kind)| (handle.to_owned(), kind))
                .collect()
        };

        assert_eq!(
            matches("model-1-0"),
            expected(&[("wire/model-1.0", SelectorMatch::WireId)])
        );
        assert_eq!(
            matches("named"),
            expected(&[
                ("low/named", SelectorMatch::Id),
                ("wire/unrelated", SelectorMatch::WireId)
            ])
        );
        assert_eq!(
            matches("nickname"),
            expected(&[
                ("low/named", SelectorMatch::Alias),
                ("wire/other", SelectorMatch::WireId)
            ])
        );
        // A prefix that is a provider names that provider's model first,
        // ahead of another provider's wire id that is the whole selector, and
        // the kind says whether the prefix was the provider's id or an alias.
        assert_eq!(
            matches("low/named-v1"),
            expected(&[
                ("low/named", SelectorMatch::ProviderQualified),
                ("wire/routed", SelectorMatch::WireId)
            ])
        );
        assert_eq!(
            matches("lo/named-v1"),
            expected(&[("low/named", SelectorMatch::ProviderAliasQualified)])
        );
        assert_eq!(
            matches("wire/model-1-0"),
            expected(&[("wire/model-1.0", SelectorMatch::ProviderQualified)])
        );
        assert!(matches("low/model-1-0").is_empty());
        assert_eq!(
            catalog
                .offerings_matching("low/named-v1")
                .iter()
                .map(|offering| offering.handle().to_string())
                .collect::<Vec<_>>(),
            ["low/named", "wire/routed"]
        );
        assert_eq!(
            catalog
                .canonical_model_id(None, "model-1-0")
                .map(ToString::to_string),
            Some("model-1.0".to_owned())
        );
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
