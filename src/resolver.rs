//! Model route selection.

#[cfg(feature = "runtime")]
mod validation;

use std::collections::BTreeSet;

use thiserror::Error;

use crate::catalog::{Catalog, CatalogModel, CatalogProvider, ModelHandle, ModelId, ProviderId};
use crate::cost::estimate_catalog_cost;
#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "openai-compatible",
    feature = "bedrock"
))]
use crate::types::Response;
use crate::types::{Cost, Error as LlmError, ErrorKind, Request, Speed, TokenCounts};

/// Providers with runtime adapters available to the client.
#[derive(Clone, Debug, Default)]
pub struct AvailableProviders {
    providers: BTreeSet<ProviderId>,
}

impl AvailableProviders {
    pub fn new(providers: impl IntoIterator<Item = ProviderId>) -> Self {
        Self {
            providers: providers.into_iter().collect(),
        }
    }

    /// Every enabled provider in the catalog.
    ///
    /// A provider the catalog marks `enabled = false` is never available, so
    /// it is left out here just as the client leaves it out when it builds
    /// adapters.
    pub fn all(catalog: &Catalog) -> Self {
        Self::new(
            catalog
                .providers()
                .filter(|provider| provider.is_enabled())
                .map(|provider| provider.id().clone()),
        )
    }

    pub fn contains(&self, provider: &ProviderId) -> bool {
        self.providers.contains(provider)
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &ProviderId> {
        self.providers.iter()
    }
}

/// An immutable route selected before middleware runs.
#[derive(Clone, Debug)]
pub struct ResolvedRoute {
    provider: CatalogProvider,
    model:    CatalogModel,
}

impl ResolvedRoute {
    /// Adds catalog pricing only when the provider did not supply a cost.
    #[cfg(any(
        feature = "openai",
        feature = "anthropic",
        feature = "gemini",
        feature = "openai-compatible",
        feature = "bedrock"
    ))]
    pub(crate) fn apply_catalog_cost(&self, response: &mut Response, speed: Option<Speed>) {
        if response.cost.is_none() {
            response.cost = self.estimate_cost(response.usage, speed);
        }
    }

    /// Builds a route whose model belongs to the selected provider.
    ///
    /// # Errors
    ///
    /// Returns a provider mismatch when the model belongs to another provider.
    pub fn try_new(
        provider: CatalogProvider,
        model: CatalogModel,
    ) -> Result<Self, ModelSelectionError> {
        if provider.id() != model.provider_id() {
            return Err(ModelSelectionError::ModelProviderMismatch {
                provider: provider.id().clone(),
                model:    ModelHandle::new(model.provider_id().clone(), model.id().clone()),
            });
        }
        Ok(Self { provider, model })
    }

    pub fn provider(&self) -> &CatalogProvider {
        &self.provider
    }

    pub fn model(&self) -> &CatalogModel {
        &self.model
    }

    /// The provider API model identifier for this route.
    ///
    /// This is the value sent on the wire. It differs from
    /// [`ResolvedRoute::handle`], which is the catalog identity.
    pub fn api_model(&self) -> &str {
        self.model.api_model()
    }

    pub fn handle(&self) -> ModelHandle {
        ModelHandle::new(self.provider.id().clone(), self.model.id().clone())
    }

    /// Estimates the catalog cost for supplied token usage on this route.
    ///
    /// The token buckets must be disjoint as described by [`TokenCounts`].
    /// The estimate selects long-context rates from the whole prompt and then
    /// applies rates for `speed`. It returns `None` when the model has no
    /// catalog pricing or a non-empty cache bucket has no applicable rate.
    ///
    /// This calculation has no state. It does not reserve or enforce a budget.
    #[must_use]
    pub fn estimate_cost(&self, usage: TokenCounts, speed: Option<Speed>) -> Option<Cost> {
        estimate_catalog_cost(self, usage, speed)
    }
}

/// Selects one provider and model for a request.
///
/// This is an open application extension point. Resolution is synchronous and
/// side-effect free, including when called without dispatch through the
/// client's route-inspection API. Do not read credentials, perform network I/O,
/// or block. Implementations may be called concurrently.
pub trait ModelResolver: Send + Sync {
    /// Selects an available provider and a model that belongs to it.
    ///
    /// Honor `available`, not just catalog membership. Construct custom routes
    /// with [`ResolvedRoute::try_new`]. The client validates request
    /// capabilities separately; the resolver owns selection policy, not
    /// request execution.
    ///
    /// # Errors
    ///
    /// Return [`ModelSelectionError`] when no requested route is available or
    /// route construction fails. Do not silently select an unrelated provider
    /// unless that fallback is part of the resolver's documented policy.
    fn resolve(
        &self,
        request: &Request,
        catalog: &Catalog,
        available: &AvailableProviders,
    ) -> Result<ResolvedRoute, ModelSelectionError>;
}

/// Catalog-based resolution using explicit routes, aliases, priority,
/// defaults, and stand-in providers.
///
/// A provider the catalog marks `enabled = false` resolves no route, whatever
/// the selector shape. A provider that is enabled but not available, because
/// no adapter was built for it, is served by the provider that
/// [`stands_in_for`](CatalogProvider::stands_in_for) it when that one is
/// available: `openai/gpt-5.6-sol` reaches `openai-codex/gpt-5.6-sol` on a
/// client that holds a ChatGPT credential but no platform API key.
#[derive(Clone, Copy, Debug, Default)]
pub struct CatalogResolver;

impl ModelResolver for CatalogResolver {
    fn resolve(
        &self,
        request: &Request,
        catalog: &Catalog,
        available: &AvailableProviders,
    ) -> Result<ResolvedRoute, ModelSelectionError> {
        let selector = request.model();
        if let Some((provider_selector, model_selector)) = selector.split_once('/') {
            let provider = catalog.find_provider(provider_selector).ok_or_else(|| {
                ModelSelectionError::ProviderNotFound {
                    provider: provider_selector.to_owned(),
                }
            })?;
            let provider = serving_provider(catalog, provider, available)?;
            return resolve_explicit(provider, model_selector);
        }

        if selector == "default" {
            return resolve_default(catalog, available);
        }

        if let Some(provider) = catalog.find_provider(selector) {
            let provider = serving_provider(catalog, provider, available)?;
            return resolve_provider_default(provider);
        }

        let mut matches = catalog.models_matching(selector);
        matches.retain(|model| {
            available.contains(model.provider_id())
                && catalog
                    .provider_by_id(model.provider_id())
                    .is_some_and(CatalogProvider::is_enabled)
        });
        // A model actually named `selector` wins over any provider's alias for
        // it, whatever the provider priorities are. Priority only separates
        // matches of the same kind.
        matches.sort_by(|left, right| {
            let left_priority = catalog
                .find_provider(left.provider_id().as_str())
                .map_or(i32::MIN, CatalogProvider::priority);
            let right_priority = catalog
                .find_provider(right.provider_id().as_str())
                .map_or(i32::MIN, CatalogProvider::priority);
            let left_alias = left.id().as_str() != selector;
            let right_alias = right.id().as_str() != selector;
            left_alias
                .cmp(&right_alias)
                .then_with(|| right_priority.cmp(&left_priority))
                .then_with(|| left.provider_id().cmp(right.provider_id()))
        });
        let model =
            matches
                .into_iter()
                .next()
                .ok_or_else(|| ModelSelectionError::ModelNotFound {
                    selector: selector.to_owned(),
                })?;
        let provider = catalog
            .find_provider(model.provider_id().as_str())
            .ok_or_else(|| ModelSelectionError::ProviderNotFound {
                provider: model.provider_id().to_string(),
            })?;
        ResolvedRoute::try_new(provider.clone(), model.clone())
    }
}

/// The provider that serves routes addressed to `provider`.
///
/// That is `provider` itself when it is enabled and available. When it is
/// enabled but has no adapter, the available provider that stands in for it
/// serves instead. A disabled provider is refused outright: disabling is a
/// catalog decision, and no stand-in overrides it.
fn serving_provider<'a>(
    catalog: &'a Catalog,
    provider: &'a CatalogProvider,
    available: &AvailableProviders,
) -> Result<&'a CatalogProvider, ModelSelectionError> {
    if !provider.is_enabled() {
        return Err(ModelSelectionError::ProviderDisabled {
            provider: provider.id().clone(),
        });
    }
    if available.contains(provider.id()) {
        return Ok(provider);
    }
    catalog
        .providers()
        .find(|candidate| {
            candidate.is_enabled()
                && available.contains(candidate.id())
                && candidate.stands_in_for() == Some(provider.id())
        })
        .ok_or_else(|| ModelSelectionError::ProviderUnavailable {
            provider: provider.id().clone(),
        })
}

fn resolve_explicit(
    provider: &CatalogProvider,
    model_selector: &str,
) -> Result<ResolvedRoute, ModelSelectionError> {
    let model = if let Some(model) = provider.model(model_selector) {
        model.clone()
    } else if provider.allows_passthrough() && !model_selector.trim().is_empty() {
        CatalogModel::passthrough(provider.id().clone(), ModelId::new(model_selector))
    } else {
        return Err(ModelSelectionError::ModelNotFound {
            selector: format!("{}/{model_selector}", provider.id()),
        });
    };
    ResolvedRoute::try_new(provider.clone(), model)
}

fn resolve_default(
    catalog: &Catalog,
    available: &AvailableProviders,
) -> Result<ResolvedRoute, ModelSelectionError> {
    let mut providers: Vec<_> = catalog
        .providers()
        .filter(|provider| {
            provider.is_enabled()
                && available.contains(provider.id())
                && provider.default_model().is_some()
        })
        .collect();
    providers.sort_by(|left, right| {
        right
            .priority()
            .cmp(&left.priority())
            .then_with(|| left.id().cmp(right.id()))
    });
    let provider = providers
        .into_iter()
        .next()
        .ok_or(ModelSelectionError::NoDefaultModel)?;
    resolve_provider_default(provider)
}

fn resolve_provider_default(
    provider: &CatalogProvider,
) -> Result<ResolvedRoute, ModelSelectionError> {
    let default_model =
        provider
            .default_model()
            .ok_or_else(|| ModelSelectionError::ProviderHasNoDefault {
                provider: provider.id().clone(),
            })?;
    let model =
        provider
            .model(default_model)
            .ok_or_else(|| ModelSelectionError::ModelNotFound {
                selector: format!("{}/{default_model}", provider.id()),
            })?;
    ResolvedRoute::try_new(provider.clone(), model.clone())
}

/// A request could not be mapped to an available route.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
#[non_exhaustive]
pub enum ModelSelectionError {
    #[error("model {model} does not belong to provider {provider}")]
    ModelProviderMismatch {
        provider: ProviderId,
        model:    ModelHandle,
    },
    #[error("provider `{provider}` was not found")]
    ProviderNotFound { provider: String },
    #[error("provider {provider} has no registered adapter")]
    ProviderUnavailable { provider: ProviderId },
    #[error("provider {provider} is disabled in the catalog")]
    ProviderDisabled { provider: ProviderId },
    #[error("model selector `{selector}` was not found")]
    ModelNotFound { selector: String },
    #[error("provider {provider} has no default model")]
    ProviderHasNoDefault { provider: ProviderId },
    #[error("no available provider has a default model")]
    NoDefaultModel,
}

impl From<ModelSelectionError> for LlmError {
    fn from(error: ModelSelectionError) -> Self {
        Self::new(ErrorKind::ModelSelection, error.to_string()).with_source(error)
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use super::{AvailableProviders, CatalogResolver, ModelResolver};
    use crate::catalog::Catalog;
    use crate::types::{CostSource, Request, TokenCounts};

    const TEST_CATALOG: &str = r#"
        schema_version = 1

        [providers.alpha]
        display_name = "Alpha"
        adapter = "test-adapter"
        codec = "test-codec"
        base_url = "http://127.0.0.1"
        allow_passthrough = true
        default_model = "one"
        auth = { type = "none" }

        [providers.alpha.models.one]
        display_name = "One"
        aliases = ["uno"]
        api_model = "alpha-one-v1"
        capabilities = { text = true }
        pricing = { input_usd_micros_per_million = 1000000, output_usd_micros_per_million = 2000000, cached_input_usd_micros_per_million = 100000, cache_write_usd_micros_per_million = 500000 }
    "#;

    #[test]
    fn route_construction_rejects_a_model_from_another_provider() -> Result<(), Box<dyn StdError>> {
        let catalog = Catalog::builder().overlay_toml(TEST_CATALOG)?.build()?;
        let other = Catalog::builder()
            .overlay_toml(&TEST_CATALOG.replace("providers.alpha", "providers.beta"))?
            .build()?;
        let error = super::ResolvedRoute::try_new(
            catalog.provider("alpha")?.clone(),
            other.model("beta", "one")?.clone(),
        )
        .expect_err("provider mismatch");
        assert!(
            matches!(error, super::ModelSelectionError::ModelProviderMismatch { provider, model }
            if provider.as_str() == "alpha" && model.provider().as_str() == "beta")
        );
        let route = super::ResolvedRoute::try_new(
            catalog.provider("alpha")?.clone(),
            catalog.model("alpha", "one")?.clone(),
        )?;
        assert_eq!(route.model().provider_id(), route.provider().id());
        Ok(())
    }

    #[test]
    fn a_resolved_route_estimates_catalog_cost() -> Result<(), Box<dyn StdError>> {
        let catalog = Catalog::builder().overlay_toml(TEST_CATALOG)?.build()?;
        let available = AvailableProviders::all(&catalog);
        let request = Request::builder()
            .model("alpha/one")
            .user("Hello")
            .build()?;
        let route = CatalogResolver.resolve(&request, &catalog, &available)?;

        let cost = route.estimate_cost(
            TokenCounts {
                input:       1_000,
                output:      1_000,
                reasoning:   1_000,
                cache_read:  1_000,
                cache_write: 1_000,
            },
            None,
        );

        assert_eq!(cost.map(|cost| cost.usd_micros), Some(5_600));
        assert_eq!(cost.map(|cost| cost.source), Some(CostSource::Catalog));
        Ok(())
    }

    #[test]
    fn reports_the_api_model_for_catalog_and_passthrough_routes() -> Result<(), Box<dyn StdError>> {
        let catalog = Catalog::builder().overlay_toml(TEST_CATALOG)?.build()?;
        let available = AvailableProviders::all(&catalog);

        let request = Request::builder()
            .model("alpha/uno")
            .user("Hello")
            .build()?;
        let route = CatalogResolver.resolve(&request, &catalog, &available)?;
        assert_eq!(route.model().id().as_str(), "one");
        assert_eq!(route.api_model(), "alpha-one-v1");

        let request = Request::builder()
            .model("alpha/not-in-catalog")
            .user("Hello")
            .build()?;
        let route = CatalogResolver.resolve(&request, &catalog, &available)?;
        assert_eq!(route.api_model(), "not-in-catalog");
        Ok(())
    }

    #[test]
    fn a_wire_id_resolves_to_its_catalog_entry_before_passthrough() -> Result<(), Box<dyn StdError>>
    {
        // A selector copied from provider documentation is the wire id. It
        // must land on the catalog entry that prices and describes the model;
        // falling through to passthrough would silently drop pricing and
        // capabilities on a passthrough provider, and fail outright elsewhere.
        let catalog = Catalog::builder().overlay_toml(TEST_CATALOG)?.build()?;
        let available = AvailableProviders::all(&catalog);

        let request = Request::builder()
            .model("alpha/alpha-one-v1")
            .user("Hello")
            .build()?;
        let route = CatalogResolver.resolve(&request, &catalog, &available)?;

        assert_eq!(route.model().id().as_str(), "one");
        assert_eq!(route.api_model(), "alpha-one-v1");
        Ok(())
    }

    #[cfg(feature = "builtin-catalog")]
    #[test]
    fn resolves_explicit_aliases_and_defaults() -> Result<(), Box<dyn StdError>> {
        let catalog = Catalog::builder().with_builtin().build()?;
        let available = AvailableProviders::all(&catalog);
        let request = Request::builder().model("oa/luna").user("Hello").build()?;
        let route = CatalogResolver.resolve(&request, &catalog, &available)?;
        assert_eq!(route.provider().id().as_str(), "openai");
        assert_eq!(route.model().id().as_str(), "gpt-5.6-luna");

        let request = Request::builder().model("default").user("Hello").build()?;
        let route = CatalogResolver.resolve(&request, &catalog, &available)?;
        assert_eq!(route.provider().id().as_str(), "anthropic");
        assert_eq!(route.model().id().as_str(), "claude-sonnet-5");
        Ok(())
    }

    const STAND_IN_CATALOG: &str = r#"
        schema_version = 1

        [providers.platform]
        display_name = "Platform"
        adapter = "test-adapter"
        codec = "test-codec"
        base_url = "http://127.0.0.1"
        priority = 90
        default_model = "one"
        auth = { type = "bearer" }

        [providers.platform.models.one]
        display_name = "One"
        aliases = ["uno"]
        api_model = "one"

        [providers.seat]
        display_name = "Seat"
        adapter = "test-adapter"
        codec = "test-codec"
        base_url = "http://127.0.0.1/seat"
        priority = 89
        stands_in_for = "platform"
        default_model = "one"
        auth = { type = "bearer" }

        [providers.seat.models.one]
        display_name = "One"
        aliases = ["uno"]
        api_model = "one"

        [providers.parked]
        display_name = "Parked"
        adapter = "test-adapter"
        codec = "test-codec"
        base_url = "http://127.0.0.1/parked"
        priority = 100
        enabled = false
        default_model = "one"
        auth = { type = "none" }

        [providers.parked.models.one]
        display_name = "One"
        aliases = ["shared"]
        api_model = "one"
    "#;

    fn stand_in_catalog() -> Result<Catalog, Box<dyn StdError>> {
        Ok(Catalog::builder().overlay_toml(STAND_IN_CATALOG)?.build()?)
    }

    fn resolve(
        catalog: &Catalog,
        available: &AvailableProviders,
        selector: &str,
    ) -> Result<super::ResolvedRoute, super::ModelSelectionError> {
        let request = Request::builder()
            .model(selector)
            .user("Hello")
            .build()
            .expect("the test request builds");
        CatalogResolver.resolve(&request, catalog, available)
    }

    #[test]
    fn a_disabled_provider_resolves_no_route() -> Result<(), Box<dyn StdError>> {
        let catalog = stand_in_catalog()?;
        // Even a caller that lists the provider as available cannot reach it.
        let available =
            AvailableProviders::new(["platform", "seat", "parked"].map(super::ProviderId::new));

        for selector in ["parked/one", "parked"] {
            assert!(
                matches!(
                    resolve(&catalog, &available, selector),
                    Err(super::ModelSelectionError::ProviderDisabled { provider })
                        if provider.as_str() == "parked"
                ),
                "`{selector}` should be refused"
            );
        }
        // The disabled provider's alias never wins a bare-selector match, and
        // the default skips it despite its higher priority.
        assert!(matches!(
            resolve(&catalog, &available, "shared"),
            Err(super::ModelSelectionError::ModelNotFound { .. })
        ));
        assert_eq!(
            resolve(&catalog, &available, "default")?
                .provider()
                .id()
                .as_str(),
            "platform"
        );
        assert!(
            !AvailableProviders::all(&catalog).contains(&super::ProviderId::new("parked")),
            "`all` leaves disabled providers out"
        );
        Ok(())
    }

    #[test]
    fn a_stand_in_serves_an_unavailable_provider() -> Result<(), Box<dyn StdError>> {
        let catalog = stand_in_catalog()?;
        let only_seat = AvailableProviders::new([super::ProviderId::new("seat")]);

        for selector in ["platform/one", "platform/uno", "platform"] {
            let route = resolve(&catalog, &only_seat, selector)?;
            assert_eq!(route.provider().id().as_str(), "seat", "`{selector}`");
            assert_eq!(route.model().id().as_str(), "one");
        }
        // A passthrough-style selector the stand-in cannot serve fails on the
        // stand-in, not on the provider it stands in for.
        assert!(matches!(
            resolve(&catalog, &only_seat, "platform/missing"),
            Err(super::ModelSelectionError::ModelNotFound { selector }) if selector == "seat/missing"
        ));

        // When the provider itself is available, it serves its own routes.
        let both = AvailableProviders::new(["platform", "seat"].map(super::ProviderId::new));
        assert_eq!(
            resolve(&catalog, &both, "platform/one")?
                .provider()
                .id()
                .as_str(),
            "platform"
        );

        // With neither available, the error names the provider that was asked for.
        let none = AvailableProviders::new(Vec::<super::ProviderId>::new());
        assert!(matches!(
            resolve(&catalog, &none, "platform/one"),
            Err(super::ModelSelectionError::ProviderUnavailable { provider })
                if provider.as_str() == "platform"
        ));
        Ok(())
    }

    #[test]
    fn a_canonical_model_id_outranks_an_alias_from_a_higher_priority_provider()
    -> Result<(), Box<dyn StdError>> {
        let catalog = Catalog::builder()
            .overlay_toml(
                r#"
                schema_version = 1

                [providers.high]
                display_name = "High"
                adapter = "test-adapter"
                codec = "test-codec"
                base_url = "http://127.0.0.1"
                priority = 100
                auth = { type = "none" }

                [providers.high.models.other]
                display_name = "Other"
                aliases = ["shared"]
                api_model = "other-v1"

                [providers.low]
                display_name = "Low"
                adapter = "test-adapter"
                codec = "test-codec"
                base_url = "http://127.0.0.1"
                priority = 1
                auth = { type = "none" }

                [providers.low.models.shared]
                display_name = "Shared"
                api_model = "shared-v1"
            "#,
            )?
            .build()?;
        let available = AvailableProviders::all(&catalog);

        let request = Request::builder().model("shared").user("Hello").build()?;
        let route = CatalogResolver.resolve(&request, &catalog, &available)?;

        assert_eq!(route.provider().id().as_str(), "low");
        assert_eq!(route.model().id().as_str(), "shared");
        Ok(())
    }
}
