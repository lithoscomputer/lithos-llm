//! Immutable provider and model catalog data.

mod loader;
mod model;
mod overlay;
mod provider;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

pub use loader::CatalogBuilder;
pub use model::{
    CatalogModel, LongContextPricing, ModelCapabilities, ModelLimits, Pricing, SpeedPricing,
    SpeedRates,
};
pub use provider::{
    AdapterId, AuthScheme, CatalogProvider, CodecId, Metadata, MetadataError, ModelHandle, ModelId,
    ProviderId, adapter_ids, codec_ids,
};
use serde::Deserialize;
use thiserror::Error;
use toml::de::Error as TomlError;

/// The supported catalog document schema.
pub const CATALOG_SCHEMA_VERSION: u32 = 1;

/// The layer that most recently wrote each provider entry.
pub(crate) type LayerOrigins = BTreeMap<ProviderId, String>;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
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

    pub(crate) fn from_document(
        mut document: CatalogDocument,
        origins: &LayerOrigins,
    ) -> Result<Self, CatalogError> {
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
            validate_provider(provider_id, provider, &mut provider_selectors)
                .map_err(|source| in_layer(origins, provider_id, source))?;
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

    /// Looks up a provider by canonical identifier.
    ///
    /// Provider aliases are ignored. Use [`Catalog::provider`] to accept an
    /// alias as well.
    pub fn provider_by_id(&self, id: &ProviderId) -> Option<&CatalogProvider> {
        self.inner.providers.get(id)
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

    /// Every model whose canonical identifier or alias is `selector`.
    ///
    /// Canonical and alias matches are pooled, so a caller that prefers one
    /// kind ranks the result itself.
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

fn validate_provider(
    provider_id: &ProviderId,
    provider: &mut CatalogProvider,
    provider_selectors: &mut BTreeSet<String>,
) -> Result<(), CatalogError> {
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
    validate_default_headers(provider_id, provider.default_headers())?;

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
        if let Some(limits) = model.limits()
            && limits.max_output_tokens > limits.context_tokens
        {
            return Err(CatalogError::InvalidModelLimits {
                model: ModelHandle::new(provider_id.clone(), model_id.clone()),
            });
        }
    }

    if let Some(default_model) = provider.default_model()
        && !provider
            .models()
            .any(|model| model.id().as_str() == default_model)
    {
        return Err(CatalogError::UnknownDefaultModel {
            provider: provider_id.clone(),
            model:    default_model.to_owned(),
        });
    }
    Ok(())
}

fn in_layer(origins: &LayerOrigins, provider: &ProviderId, source: CatalogError) -> CatalogError {
    match origins.get(provider) {
        Some(layer) => CatalogError::Layer {
            layer:  layer.clone(),
            source: Box::new(source),
        },
        None => source,
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

fn validate_default_headers(
    provider: &ProviderId,
    headers: &BTreeMap<String, String>,
) -> Result<(), CatalogError> {
    for (name, value) in headers {
        if !is_header_name(name) {
            return Err(CatalogError::InvalidHeaderName {
                provider: provider.clone(),
                name:     name.clone(),
            });
        }
        if !is_header_value(value) {
            return Err(CatalogError::InvalidHeaderValue {
                provider: provider.clone(),
                name:     name.clone(),
            });
        }
    }
    Ok(())
}

/// Reports whether `name` is a non-empty RFC 7230 field name.
fn is_header_name(name: &str) -> bool {
    !name.is_empty()
        && name.bytes().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(
                    byte,
                    b'!' | b'#'
                        | b'$'
                        | b'%'
                        | b'&'
                        | b'\''
                        | b'*'
                        | b'+'
                        | b'-'
                        | b'.'
                        | b'^'
                        | b'_'
                        | b'`'
                        | b'|'
                        | b'~'
                )
        })
}

/// Reports whether `value` is a non-empty RFC 7230 field value.
///
/// Visible ASCII, spaces, and horizontal tabs are allowed. Control bytes,
/// including carriage return, line feed, and NUL, are not.
fn is_header_value(value: &str) -> bool {
    !value.trim().is_empty()
        && value
            .bytes()
            .all(|byte| matches!(byte, 0x21..=0x7E | b' ' | b'\t'))
}

/// A catalog could not be parsed or validated.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CatalogError {
    #[error("catalog layer `{layer}` is invalid TOML")]
    Parse {
        layer:  String,
        #[source]
        source: TomlError,
    },
    #[error("catalog layer `{layer}` produced an invalid catalog")]
    Layer {
        layer:  String,
        #[source]
        source: Box<Self>,
    },
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
    #[error("provider {provider} declares invalid default header name `{name}`")]
    InvalidHeaderName {
        provider: ProviderId,
        name:     String,
    },
    #[error("provider {provider} declares an invalid value for default header `{name}`")]
    InvalidHeaderValue {
        provider: ProviderId,
        name:     String,
    },
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

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use serde_json::json;

    use super::{Catalog, CatalogError, ProviderId};

    const HEADER_CATALOG: &str = r#"
        schema_version = 1

        [providers.test]
        display_name = "Test"
        aliases = ["alias"]
        adapter = "custom"
        codec = "custom"
        base_url = "https://example.com"
        auth = { type = "none" }
    "#;

    fn catalog_with(extra: &str) -> Result<Catalog, CatalogError> {
        let source = format!("{HEADER_CATALOG}{extra}");
        Catalog::builder().overlay_toml(&source)?.build()
    }

    #[test]
    fn provider_by_id_ignores_aliases() -> Result<(), Box<dyn StdError>> {
        let catalog = catalog_with("")?;

        assert!(
            catalog
                .provider_by_id(&ProviderId::new("test"))
                .is_some_and(|provider| provider.id().as_str() == "test")
        );
        assert!(catalog.provider_by_id(&ProviderId::new("alias")).is_none());
        assert!(catalog.provider("alias").is_ok());
        Ok(())
    }

    #[test]
    fn keeps_valid_default_headers_in_order() -> Result<(), Box<dyn StdError>> {
        let catalog = catalog_with(
            r#"
            default_headers = { "x-zone" = "eu", "OpenAI-Beta" = "responses=v1", "x-app" = "lithos" }
        "#,
        )?;

        let headers = catalog
            .provider_by_id(&ProviderId::new("test"))
            .ok_or("provider test should be present")?
            .default_headers()
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect::<Vec<_>>();

        assert_eq!(headers, vec![
            ("OpenAI-Beta", "responses=v1"),
            ("x-app", "lithos"),
            ("x-zone", "eu"),
        ]);
        Ok(())
    }

    fn cause(result: Result<Catalog, CatalogError>) -> Result<CatalogError, Box<dyn StdError>> {
        match result {
            Err(CatalogError::Layer { source, .. }) => Ok(*source),
            Err(other) => Ok(other),
            Ok(_) => Err("expected the catalog to be rejected".into()),
        }
    }

    #[test]
    fn rejects_an_invalid_default_header_name() -> Result<(), Box<dyn StdError>> {
        let result = cause(catalog_with(
            r#"default_headers = { "x api key" = "value" }"#,
        ))?;

        assert!(matches!(
            result,
            CatalogError::InvalidHeaderName { provider, name }
                if provider.as_str() == "test" && name == "x api key"
        ));
        Ok(())
    }

    #[test]
    fn rejects_an_invalid_default_header_value() -> Result<(), Box<dyn StdError>> {
        let control = cause(catalog_with(
            r#"default_headers = { "x-trace" = "one\ntwo" }"#,
        ))?;

        assert!(matches!(
            control,
            CatalogError::InvalidHeaderValue { provider, name }
                if provider.as_str() == "test" && name == "x-trace"
        ));

        let blank = cause(catalog_with(r#"default_headers = { "x-trace" = "   " }"#))?;

        assert!(matches!(
            blank,
            CatalogError::InvalidHeaderValue { name, .. } if name == "x-trace"
        ));
        Ok(())
    }

    #[test]
    fn converts_adapter_options_into_json() -> Result<(), Box<dyn StdError>> {
        let catalog = catalog_with(
            r#"
            [providers.test.adapter_options]
            mode = "codex"
            retries = 3
            flags = ["a", "b"]

            [providers.test.adapter_options.nested]
            enabled = true
        "#,
        )?;

        let options = catalog
            .provider_by_id(&ProviderId::new("test"))
            .ok_or("provider test should be present")?
            .adapter_options();

        assert_eq!(
            options,
            &json!({
                "mode": "codex",
                "retries": 3,
                "flags": ["a", "b"],
                "nested": { "enabled": true },
            })
        );
        Ok(())
    }

    #[test]
    fn adapter_options_default_to_null() -> Result<(), Box<dyn StdError>> {
        let catalog = catalog_with("")?;

        assert!(
            catalog
                .provider_by_id(&ProviderId::new("test"))
                .ok_or("provider test should be present")?
                .adapter_options()
                .is_null()
        );
        Ok(())
    }
}
