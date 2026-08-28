//! Immutable provider and model catalog data.

mod loader;
mod model;
mod overlay;
mod provider;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

pub use loader::CatalogBuilder;
pub use model::{CatalogModel, LongContextPricing, ModelCapabilities, ModelLimits, Pricing};
pub use provider::{
    AdapterId, AuthScheme, CatalogProvider, CodecId, Metadata, MetadataError, ModelHandle, ModelId,
    ProviderId, adapter_ids, codec_ids,
};
use serde::Deserialize;
use thiserror::Error;
use toml::de::Error as TomlError;

/// The supported catalog document schema.
pub const CATALOG_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct CatalogDocument {
    pub schema_version: u32,
    pub providers:      BTreeMap<ProviderId, CatalogProvider>,
}

/// A validated immutable provider and model catalog.
#[derive(Clone, Debug)]
pub struct Catalog {
    inner: Arc<CatalogDocument>,
}

impl Catalog {
    pub fn builder() -> CatalogBuilder {
        CatalogBuilder::new()
    }

    pub(crate) fn from_document(mut document: CatalogDocument) -> Result<Self, CatalogError> {
        if document.schema_version != CATALOG_SCHEMA_VERSION {
            return Err(CatalogError::UnsupportedSchemaVersion {
                found: document.schema_version,
            });
        }
        if document.providers.is_empty() {
            return Err(CatalogError::NoProviders);
        }

        let mut provider_selectors = document
            .providers
            .keys()
            .map(ToString::to_string)
            .collect::<BTreeSet<_>>();
        for (provider_id, provider) in &mut document.providers {
            validate_identifier("provider", provider_id.as_str(), false)?;
            provider.set_id(provider_id.clone());
            if provider.display_name().trim().is_empty() {
                return Err(CatalogError::EmptyDisplayName {
                    item: provider_id.to_string(),
                });
            }
            if provider.base_url().trim().is_empty() {
                return Err(CatalogError::EmptyBaseUrl {
                    provider: provider_id.clone(),
                });
            }
            for alias in provider.aliases() {
                validate_identifier("provider alias", alias, false)?;
                if !provider_selectors.insert(alias.clone()) {
                    return Err(CatalogError::DuplicateProviderAlias {
                        alias: alias.clone(),
                    });
                }
            }

            let mut model_selectors = provider
                .models_mut()
                .map(|(model_id, _)| model_id.to_string())
                .collect::<BTreeSet<_>>();
            for (model_id, model) in provider.models_mut() {
                validate_identifier("model", model_id.as_str(), true)?;
                model.set_identity(provider_id.clone(), model_id.clone());
                if model.display_name().trim().is_empty() {
                    return Err(CatalogError::EmptyDisplayName {
                        item: format!("{provider_id}/{model_id}"),
                    });
                }
                for alias in model.aliases() {
                    validate_identifier("model alias", alias, false)?;
                    if !model_selectors.insert(alias.clone()) {
                        return Err(CatalogError::DuplicateModelSelector {
                            provider: provider_id.clone(),
                            selector: alias.clone(),
                        });
                    }
                }
                if let Some(limits) = model.limits() {
                    if limits.max_output_tokens > limits.context_tokens {
                        return Err(CatalogError::InvalidModelLimits {
                            model: ModelHandle::new(provider_id.clone(), model_id.clone()),
                        });
                    }
                }
            }

            if let Some(default_model) = provider.default_model() {
                if !provider
                    .models()
                    .any(|model| model.id().as_str() == default_model)
                {
                    return Err(CatalogError::UnknownDefaultModel {
                        provider: provider_id.clone(),
                        model:    default_model.to_owned(),
                    });
                }
            }
        }

        Ok(Self {
            inner: Arc::new(document),
        })
    }

    pub fn schema_version(&self) -> u32 {
        self.inner.schema_version
    }

    pub fn providers(&self) -> impl ExactSizeIterator<Item = &CatalogProvider> {
        self.inner.providers.values()
    }

    pub fn provider(&self, id_or_alias: &str) -> Result<&CatalogProvider, CatalogError> {
        self.find_provider(id_or_alias)
            .ok_or_else(|| CatalogError::ProviderNotFound {
                provider: id_or_alias.to_owned(),
            })
    }

    pub fn model(&self, provider: &str, model: &str) -> Result<&CatalogModel, CatalogError> {
        let provider = self.provider(provider)?;
        provider
            .model(model)
            .ok_or_else(|| CatalogError::ModelNotFound {
                provider: provider.id().clone(),
                model:    model.to_owned(),
            })
    }

    pub(crate) fn find_provider(&self, id_or_alias: &str) -> Option<&CatalogProvider> {
        self.inner.providers.get(id_or_alias).or_else(|| {
            self.inner
                .providers
                .values()
                .find(|provider| provider.aliases().iter().any(|alias| alias == id_or_alias))
        })
    }

    pub(crate) fn models_matching(&self, selector: &str) -> Vec<&CatalogModel> {
        self.providers()
            .flat_map(CatalogProvider::models)
            .filter(|model| {
                model.id().as_str() == selector
                    || model.aliases().iter().any(|alias| alias == selector)
            })
            .collect()
    }
}

fn validate_identifier(
    kind: &'static str,
    value: &str,
    allow_route_characters: bool,
) -> Result<(), CatalogError> {
    if value.is_empty()
        || !value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(byte, b'-' | b'_' | b'.')
                || allow_route_characters && matches!(byte, b':' | b'/')
        })
    {
        return Err(CatalogError::InvalidIdentifier {
            kind,
            value: value.to_owned(),
        });
    }
    Ok(())
}

/// A catalog could not be parsed or validated.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CatalogError {
    #[error("catalog TOML is invalid")]
    Parse(#[source] TomlError),
    #[error("catalog schema version {found} is not supported")]
    UnsupportedSchemaVersion { found: u32 },
    #[error("the catalog has no providers")]
    NoProviders,
    #[error("{kind} identifier `{value}` is invalid")]
    InvalidIdentifier { kind: &'static str, value: String },
    #[error("{item} has an empty display name")]
    EmptyDisplayName { item: String },
    #[error("provider {provider} has an empty base URL")]
    EmptyBaseUrl { provider: ProviderId },
    #[error("provider alias `{alias}` is defined more than once")]
    DuplicateProviderAlias { alias: String },
    #[error("model selector `{selector}` is defined more than once for provider {provider}")]
    DuplicateModelSelector {
        provider: ProviderId,
        selector: String,
    },
    #[error("provider {provider} names unknown default model `{model}`")]
    UnknownDefaultModel {
        provider: ProviderId,
        model:    String,
    },
    #[error("model {model} has an output limit above its context limit")]
    InvalidModelLimits { model: ModelHandle },
    #[error("provider `{provider}` was not found")]
    ProviderNotFound { provider: String },
    #[error("model `{model}` was not found for provider {provider}")]
    ModelNotFound {
        provider: ProviderId,
        model:    String,
    },
    #[error("the built-in catalog feature is disabled")]
    BuiltinCatalogDisabled,
}
