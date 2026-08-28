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
        matches.sort_by(|left, right| {
            let left_priority = catalog
                .find_provider(left.provider_id().as_str())
                .map_or(i32::MIN, CatalogProvider::priority);
            let right_priority = catalog
                .find_provider(right.provider_id().as_str())
                .map_or(i32::MIN, CatalogProvider::priority);
            right_priority
                .cmp(&left_priority)
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
}
