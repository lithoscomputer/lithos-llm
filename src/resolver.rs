//! Model route selection.

#[cfg(feature = "runtime")]
mod validation;

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[cfg(feature = "runtime")]
use crate::catalog::CodecId;
use crate::catalog::{Catalog, CatalogModel, CatalogProvider, ModelHandle, ModelId, ProviderId};
use crate::cost::estimate_catalog_cost;
#[cfg(feature = "runtime")]
use crate::types::Response;
use crate::types::{Cost, Error as LlmError, ErrorKind, Request, Speed, TokenCounts};

/// Selection intent, independent of a conversation or credentials.
///
/// Raw selectors use `provider/model` or `provider:model` qualification. A
/// scoped model is opaque: slashes, colons, and the word `default` are model
/// text, never reparsed. Use `None` for a scoped provider's default. Strings
/// retain the legacy JSON representation; structured forms serialize as objects
/// and `Default` as null.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(untagged, deny_unknown_fields)]
#[non_exhaustive]
pub enum ModelSelection {
    /// The highest-priority eligible provider's configured default.
    #[default]
    Default,
    /// A raw selector: `default`, a model, or `provider/model`
    /// (also `provider:model`).
    Selector(String),
    /// A provider id or alias and an optional opaque model id or alias.
    Provider {
        provider: String,
        model:    Option<String>,
    },
    /// Canonical catalog identity. Neither aliases nor stand-ins are consulted.
    /// Missing rows (including passthrough handles) return `ModelNotFound`.
    Resolved { handle: ModelHandle },
}

impl ModelSelection {
    /// Keeps a separate provider pin structured; does not parse or concatenate.
    pub fn from_parts(model: Option<&str>, provider: Option<&str>) -> Self {
        match (model, provider) {
            (model, Some(provider)) => Self::Provider {
                provider: provider.to_owned(),
                model:    model.map(str::to_owned),
            },
            (Some(model), None) => Self::Selector(model.to_owned()),
            (None, None) => Self::Default,
        }
    }

    /// Model text for legacy payload inspection. This omits provider scope;
    /// use the full selection for routing, persistence, or equality checks.
    pub(crate) fn model_text(&self) -> &str {
        match self {
            Self::Default => "default",
            Self::Selector(model) => model,
            Self::Provider { model, .. } => model.as_deref().unwrap_or("default"),
            Self::Resolved { handle } => handle.model().as_str(),
        }
    }
}

// An explicit JSON null means Default, while an absent builder field still
// means MissingModel. Option's standard deserializer would conflate them.
pub(crate) fn deserialize_selection<'de, D>(
    deserializer: D,
) -> Result<Option<ModelSelection>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    ModelSelection::deserialize(deserializer).map(Some)
}

impl From<ModelHandle> for ModelSelection {
    fn from(handle: ModelHandle) -> Self {
        Self::Resolved { handle }
    }
}

/// How a catalog miss may be sent to a provider. Provider passthrough
/// permission is required in both permissive modes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum UnknownModelPolicy {
    /// Only explicitly provider-scoped unknown models may pass through.
    #[default]
    ExplicitProvider,
    /// Also send unknown bare models to the default provider. Known but
    /// ineligible models and unknown qualified providers never fall back.
    DefaultProvider,
    /// Require a catalog model, including for explicitly scoped selections.
    CatalogOnly,
}

/// Whether an enabled but unavailable provider may use its catalog stand-in.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum ProviderMatchPolicy {
    #[default]
    AllowStandIns,
    Exact,
}

/// Configurable catalog resolver. Install with `ClientBuilder::resolver` to
/// use the same policy for inspection and every dispatch operation.
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct SelectionPolicy {
    pub unknown_models: UnknownModelPolicy,
    pub providers:      ProviderMatchPolicy,
}

impl SelectionPolicy {
    #[must_use]
    pub fn with_unknown_models(mut self, policy: UnknownModelPolicy) -> Self {
        self.unknown_models = policy;
        self
    }

    #[must_use]
    pub fn with_providers(mut self, policy: ProviderMatchPolicy) -> Self {
        self.providers = policy;
        self
    }
}

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
    ///
    /// `codec` is the codec the response came back through; it selects the
    /// rate family, as [`estimate_cost_for`](Self::estimate_cost_for) does.
    #[cfg(feature = "runtime")]
    pub(crate) fn apply_catalog_cost(
        &self,
        response: &mut Response,
        codec: Option<&CodecId>,
        speed: Option<Speed>,
    ) {
        if response.cost.is_none() {
            response.cost = self.estimate_cost_for(codec, response.usage, speed);
        }
    }

    /// [`estimate_cost`](Self::estimate_cost) at the rates of one codec.
    ///
    /// The built-in adapters call this with the codec the call was
    /// dispatched on, so a provider that lists several codecs is priced for
    /// the protocol that answered. `None` prices as `estimate_cost` does.
    #[cfg(feature = "runtime")]
    pub(crate) fn estimate_cost_for(
        &self,
        codec: Option<&CodecId>,
        usage: TokenCounts,
        speed: Option<Speed>,
    ) -> Option<Cost> {
        estimate_catalog_cost(self, codec, usage, speed)
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
    /// The rate family follows the route's first generation codec, which is
    /// the codec a call on this route selects: an Anthropic Messages or
    /// Bedrock Converse route derives a missing cache-write rate from the
    /// input rate as Anthropic bills it. The built-in adapters price a
    /// response at the codec that answered instead, which differs only on a
    /// provider whose row lists several generation codecs.
    ///
    /// This calculation has no state. It does not reserve or enforce a budget.
    #[must_use]
    pub fn estimate_cost(&self, usage: TokenCounts, speed: Option<Speed>) -> Option<Cost> {
        estimate_catalog_cost(self, None, usage, speed)
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
    fn resolve_selection(
        &self,
        selection: &ModelSelection,
        catalog: &Catalog,
        available: &AvailableProviders,
    ) -> Result<ResolvedRoute, ModelSelectionError>;

    /// Compatibility convenience for callers with a request already built.
    ///
    /// # Errors
    /// Returns the selection error from [`Self::resolve_selection`].
    fn resolve(
        &self,
        request: &Request,
        catalog: &Catalog,
        available: &AvailableProviders,
    ) -> Result<ResolvedRoute, ModelSelectionError> {
        self.resolve_selection(request.selection(), catalog, available)
    }
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
    fn resolve_selection(
        &self,
        selection: &ModelSelection,
        catalog: &Catalog,
        available: &AvailableProviders,
    ) -> Result<ResolvedRoute, ModelSelectionError> {
        SelectionPolicy::default().resolve_selection(selection, catalog, available)
    }
}

impl ModelResolver for SelectionPolicy {
    fn resolve_selection(
        &self,
        selection: &ModelSelection,
        catalog: &Catalog,
        available: &AvailableProviders,
    ) -> Result<ResolvedRoute, ModelSelectionError> {
        let selector = match selection {
            ModelSelection::Default => return resolve_default(catalog, available),
            ModelSelection::Provider { provider, model } => {
                let provider = find_serving_provider(catalog, provider, available, self.providers)?;
                return match model {
                    Some(model) => resolve_explicit(provider, model, self.unknown_models),
                    None => resolve_provider_default(provider),
                };
            }
            ModelSelection::Resolved { handle } => {
                // Canonical ids only: a renamed row must never turn into an
                // alias for a different route on resume.
                let provider = catalog.provider_by_id(handle.provider()).ok_or_else(|| {
                    ModelSelectionError::ProviderNotFound {
                        provider: handle.provider().to_string(),
                    }
                })?;
                let provider =
                    serving_provider(catalog, provider, available, ProviderMatchPolicy::Exact)?;
                let model = provider
                    .models()
                    .find(|model| model.id() == handle.model())
                    .ok_or_else(|| ModelSelectionError::ModelNotFound {
                        selector: handle.to_string(),
                    })?;
                return ResolvedRoute::try_new(provider.clone(), model.clone());
            }
            ModelSelection::Selector(selector) => selector.as_str(),
        };
        if let Some((provider_selector, model_selector)) = selector.split_once(['/', ':']) {
            let provider =
                find_serving_provider(catalog, provider_selector, available, self.providers)?;
            return resolve_explicit(provider, model_selector, self.unknown_models);
        }

        if selector == "default" {
            return resolve_default(catalog, available);
        }

        let mut matches = catalog.models_matching(selector);
        let known = !matches.is_empty();
        // A real model id or alias remains a model even when it shares a
        // provider's name. A provider-only name must never become a default
        // selection or an unrelated provider's passthrough wire id.
        if !known && catalog.find_provider(selector).is_some() {
            return Err(ModelSelectionError::ProviderInModel {
                selector: selector.to_owned(),
            });
        }
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
        let Some(model) = matches.into_iter().next() else {
            if !known && self.unknown_models == UnknownModelPolicy::DefaultProvider {
                let default = resolve_default(catalog, available)?;
                return resolve_explicit(default.provider(), selector, self.unknown_models);
            }
            return Err(ModelSelectionError::ModelNotFound {
                selector: selector.to_owned(),
            });
        };
        let provider = catalog
            .find_provider(model.provider_id().as_str())
            .ok_or_else(|| ModelSelectionError::ProviderNotFound {
                provider: model.provider_id().to_string(),
            })?;
        ResolvedRoute::try_new(provider.clone(), model.clone())
    }
}

fn find_serving_provider<'a>(
    catalog: &'a Catalog,
    selector: &str,
    available: &AvailableProviders,
    policy: ProviderMatchPolicy,
) -> Result<&'a CatalogProvider, ModelSelectionError> {
    let provider =
        catalog
            .find_provider(selector)
            .ok_or_else(|| ModelSelectionError::ProviderNotFound {
                provider: selector.to_owned(),
            })?;
    serving_provider(catalog, provider, available, policy)
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
    policy: ProviderMatchPolicy,
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
            policy == ProviderMatchPolicy::AllowStandIns
                && candidate.is_enabled()
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
    policy: UnknownModelPolicy,
) -> Result<ResolvedRoute, ModelSelectionError> {
    let model = if let Some(model) = provider.model(model_selector) {
        model.clone()
    } else if policy != UnknownModelPolicy::CatalogOnly
        && provider.allows_passthrough()
        && !model_selector.trim().is_empty()
    {
        CatalogModel::passthrough(provider, ModelId::new(model_selector))
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
    #[error(
        "`{selector}` names a provider, not a model; set provider to `{selector}` and omit model to use its default"
    )]
    ProviderInModel { selector: String },
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
        codecs = ["test-codec"]
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
        codecs = ["test-codec"]
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
        codecs = ["test-codec"]
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
        codecs = ["test-codec"]
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

        for selection in [
            super::ModelSelection::from_parts(Some("parked/one"), None),
            super::ModelSelection::from_parts(None, Some("parked")),
        ] {
            assert!(matches!(
                CatalogResolver.resolve_selection(&selection, &catalog, &available),
                Err(super::ModelSelectionError::ProviderDisabled { provider })
                    if provider.as_str() == "parked"
            ));
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

        for selector in ["platform/one", "platform/uno"] {
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
                codecs = ["test-codec"]
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
                codecs = ["test-codec"]
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
