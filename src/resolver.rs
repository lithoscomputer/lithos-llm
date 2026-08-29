//! Model route selection.

use std::collections::BTreeSet;

use thiserror::Error;

use crate::catalog::{Catalog, CatalogModel, CatalogProvider, ModelHandle, ModelId, ProviderId};
use crate::types::{Error as LlmError, ErrorKind, Request};

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

    pub fn all(catalog: &Catalog) -> Self {
        Self::new(catalog.providers().map(|provider| provider.id().clone()))
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
    pub fn new(provider: CatalogProvider, model: CatalogModel) -> Self {
        Self { provider, model }
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
}

/// Selects one provider and model for a request.
pub trait ModelResolver: Send + Sync {
    fn resolve(
        &self,
        request: &Request,
        catalog: &Catalog,
        available: &AvailableProviders,
    ) -> Result<ResolvedRoute, ModelSelectionError>;
}

/// Catalog-based resolution using explicit routes, aliases, priority, and
/// defaults.
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
            return resolve_explicit(catalog, available, provider_selector, model_selector);
        }

        if selector == "default" {
            return resolve_default(catalog, available);
        }

        if let Some(provider) = catalog.find_provider(selector) {
            return resolve_provider_default(provider, available);
        }

        let mut matches = catalog.models_matching(selector);
        matches.retain(|model| available.contains(model.provider_id()));
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
        Ok(ResolvedRoute::new(provider.clone(), model.clone()))
    }
}

fn resolve_explicit(
    catalog: &Catalog,
    available: &AvailableProviders,
    provider_selector: &str,
    model_selector: &str,
) -> Result<ResolvedRoute, ModelSelectionError> {
    let provider = catalog.find_provider(provider_selector).ok_or_else(|| {
        ModelSelectionError::ProviderNotFound {
            provider: provider_selector.to_owned(),
        }
    })?;
    require_available(provider, available)?;
    let model = if let Some(model) = provider.model(model_selector) {
        model.clone()
    } else if provider.allows_passthrough() && !model_selector.trim().is_empty() {
        CatalogModel::passthrough(provider.id().clone(), ModelId::new(model_selector))
    } else {
        return Err(ModelSelectionError::ModelNotFound {
            selector: format!("{provider_selector}/{model_selector}"),
        });
    };
    Ok(ResolvedRoute::new(provider.clone(), model))
}

fn resolve_default(
    catalog: &Catalog,
    available: &AvailableProviders,
) -> Result<ResolvedRoute, ModelSelectionError> {
    let mut providers: Vec<_> = catalog
        .providers()
        .filter(|provider| available.contains(provider.id()) && provider.default_model().is_some())
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
    resolve_provider_default(provider, available)
}

fn resolve_provider_default(
    provider: &CatalogProvider,
    available: &AvailableProviders,
) -> Result<ResolvedRoute, ModelSelectionError> {
    require_available(provider, available)?;
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
    Ok(ResolvedRoute::new(provider.clone(), model.clone()))
}

fn require_available(
    provider: &CatalogProvider,
    available: &AvailableProviders,
) -> Result<(), ModelSelectionError> {
    if available.contains(provider.id()) {
        Ok(())
    } else {
        Err(ModelSelectionError::ProviderUnavailable {
            provider: provider.id().clone(),
        })
    }
}

/// A request could not be mapped to an available route.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
#[non_exhaustive]
pub enum ModelSelectionError {
    #[error("provider `{provider}` was not found")]
    ProviderNotFound { provider: String },
    #[error("provider {provider} has no registered adapter")]
    ProviderUnavailable { provider: ProviderId },
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
    use crate::types::Request;

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
    "#;

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
        assert_eq!(route.provider().id().as_str(), "openai");
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
