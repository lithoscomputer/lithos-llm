//! Fabro's catalog contract, pinned without depending on a Fabro checkout.
#![cfg(feature = "builtin-catalog")]

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error as StdError;

use lithos_llm::Request;
use lithos_llm::catalog::Catalog;
use lithos_llm::resolver::{AvailableProviders, CatalogResolver, ModelResolver as _};
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Inventory {
    source_revision: String,
    providers:       BTreeMap<String, Provider>,
    routes:          Vec<Route>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Provider {
    fabro_default:  Option<String>,
    lithos_default: Option<String>,
    fabro_priority: i32,
    fabro_enabled:  bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Route {
    provider:        String,
    fabro_model:     String,
    fabro_aliases:   Vec<String>,
    status:          Status,
    canonical_model: String,
    api_model:       Option<String>,
    reason:          Option<String>,
    fabro_metadata:  BTreeMap<String, serde_json::Value>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Status {
    Catalog,
    Excluded,
    Deployment,
}

fn inventory() -> Result<Inventory, serde_json::Error> {
    serde_json::from_str(include_str!("fixtures/fabro-model-parity.json"))
}

#[test]
fn fabro_names_resolve_to_full_catalog_models() -> Result<(), Box<dyn StdError>> {
    let inventory = inventory()?;
    assert_eq!(
        inventory.source_revision,
        "775b62b500c957fe319710fe34e6327ae1eb1bbf"
    );
    assert_eq!(inventory.routes.len(), 99);
    let catalog = Catalog::builder().with_builtin().build()?;
    let available = AvailableProviders::all(&catalog);
    let mut seen = BTreeSet::new();
    let mut excluded = 0;
    let mut deployments = 0;
    for route in inventory.routes {
        assert!(seen.insert((route.provider.clone(), route.fabro_model.clone())));
        match route.status {
            Status::Excluded => {
                excluded += 1;
                assert!(route.fabro_model.starts_with("gpt-oss-"));
                assert!(
                    catalog
                        .provider(&route.provider)?
                        .model(&route.canonical_model)
                        .is_none()
                );
                assert!(!route.reason.as_deref().unwrap_or_default().is_empty());
                continue;
            }
            Status::Deployment => {
                deployments += 1;
                assert_eq!(route.provider, "modal");
                assert!(!route.reason.as_deref().unwrap_or_default().is_empty());
                continue;
            }
            Status::Catalog => {}
        }
        let expected = catalog.model(&route.provider, &route.canonical_model)?;
        assert_eq!(Some(expected.api_model()), route.api_model.as_deref());
        for name in [route.fabro_model].into_iter().chain(route.fabro_aliases) {
            let request = Request::builder()
                .model(format!("{}/{name}", route.provider))
                .user("parity")
                .build()?;
            let resolved = CatalogResolver.resolve(&request, &catalog, &available)?;
            assert!(!resolved.model().is_passthrough(), "{}", request.model());
            assert_eq!(
                resolved.model(),
                expected,
                "{} must retain capabilities, pricing, and metadata",
                request.model()
            );
        }
    }
    // A growing exception list is a regression.
    assert_eq!(excluded, 4);
    assert_eq!(deployments, 1);
    Ok(())
}

#[test]
fn provider_defaults_stay_explicit_during_import() -> Result<(), Box<dyn StdError>> {
    let catalog = Catalog::builder().with_builtin().build()?;
    let inventory = inventory()?;
    assert_eq!(inventory.providers.len(), 17);
    for (id, provider) in inventory.providers {
        assert_eq!(
            catalog.provider(&id)?.default_model(),
            provider.lithos_default.as_deref()
        );
        if let Some(default) = provider.fabro_default {
            assert!(
                inventory
                    .routes
                    .iter()
                    .any(|route| { route.provider == id && route.fabro_model == default })
            );
        }
        assert!(provider.fabro_priority >= 0);
        if id == "bedrock" {
            assert!(!provider.fabro_enabled);
        }
    }
    Ok(())
}

#[test]
fn fabro_policy_is_opt_in_and_preserves_model_metadata() -> Result<(), Box<dyn StdError>> {
    let builtin = Catalog::builder().with_builtin().build()?;
    let catalog = Catalog::builder()
        .with_builtin()
        .overlay_toml(include_str!("../docs/catalogs/fabro-policy.toml"))?
        .build()?;
    let inventory = inventory()?;
    for (id, policy) in &inventory.providers {
        let provider = catalog.provider(id)?;
        assert_eq!(provider.priority(), policy.fabro_priority);
        assert_eq!(
            provider
                .metadata()
                .get("fabro")
                .ok_or("missing provider policy")?["enabled"],
            policy.fabro_enabled
        );
        if id == "modal" {
            assert!(provider.default_model().is_none());
        } else if let Some(default) = &policy.fabro_default {
            assert_eq!(
                provider.model(default),
                provider.default_model().and_then(|id| provider.model(id))
            );
        } else {
            assert!(provider.default_model().is_none());
        }
    }
    assert_eq!(
        builtin.provider("bedrock")?.default_model(),
        Some("anthropic.claude-sonnet-4-6")
    );
    assert_eq!(
        catalog.provider("bedrock")?.default_model(),
        Some("claude-sonnet-5")
    );
    for route in inventory.routes {
        if !matches!(route.status, Status::Catalog) {
            continue;
        }
        let model = catalog.model(&route.provider, &route.canonical_model)?;
        let original = builtin.model(&route.provider, &route.canonical_model)?;
        let metadata = model
            .metadata()
            .get("fabro")
            .ok_or("missing model policy")?;
        for (key, value) in route.fabro_metadata {
            assert_eq!(
                metadata[&key], value,
                "{}/{}: {key}",
                route.provider, route.fabro_model
            );
        }
        assert_eq!(model.api_model(), original.api_model());
        assert_eq!(model.capabilities(), original.capabilities());
        assert_eq!(model.limits(), original.limits());
        assert_eq!(model.pricing(), original.pricing());
        assert_eq!(
            model.metadata().get("pebble"),
            original.metadata().get("pebble")
        );
    }
    Ok(())
}

#[test]
fn application_enablement_excludes_opt_in_providers() -> Result<(), Box<dyn StdError>> {
    let catalog = Catalog::builder()
        .with_builtin()
        .overlay_toml(include_str!("../docs/catalogs/fabro-policy.toml"))?
        .build()?;
    let enabled = catalog.providers().filter_map(|provider| {
        let enabled = provider
            .metadata()
            .get("fabro")?
            .get("enabled")?
            .as_bool()?;
        enabled.then(|| provider.id().clone())
    });
    let available = AvailableProviders::new(enabled);
    for (id, policy) in inventory()?.providers {
        assert_eq!(
            available.contains(catalog.provider(&id)?.id()),
            policy.fabro_enabled
        );
        if !policy.fabro_enabled {
            let request = Request::builder()
                .model(format!("{id}/test-model"))
                .user("Ping")
                .build()?;
            assert!(
                CatalogResolver
                    .resolve(&request, &catalog, &available)
                    .is_err()
            );
        }
    }
    Ok(())
}

#[test]
fn all_in_scope_routes_resolve_after_deployment_configuration() -> Result<(), Box<dyn StdError>> {
    for template in [
        include_str!("../docs/catalogs/modal-dedicated.toml"),
        include_str!("../docs/catalogs/modal-shared.toml"),
    ] {
        let overlay = template
            .replace(
                "https://REPLACE_WITH_DEDICATED_ENDPOINT.invalid/v1",
                "https://example.invalid/v1",
            )
            .replace("REPLACE_WITH_ACCEPTED_MODEL_ID", "deployed/kimi-k3")
            .replace(
                "REPLACE_WITH_SHARED_ENDPOINT_HOSTNAME",
                "deployed-kimi.example.invalid",
            );
        let catalog = Catalog::builder()
            .with_builtin()
            .overlay_toml(include_str!("../docs/catalogs/fabro-policy.toml"))?
            .overlay_toml(&overlay)?
            .build()?;
        let available = AvailableProviders::all(&catalog);
        let mut resolved_count = 0;
        for route in inventory()?.routes {
            if matches!(route.status, Status::Excluded) {
                assert!(
                    catalog
                        .provider(&route.provider)?
                        .model(&route.canonical_model)
                        .is_none()
                );
                continue;
            }
            let request = Request::builder()
                .model(format!("{}/{}", route.provider, route.fabro_model))
                .user("Ping")
                .build()?;
            let resolved = CatalogResolver.resolve(&request, &catalog, &available)?;
            assert!(!resolved.model().is_passthrough());
            let metadata = resolved
                .model()
                .metadata()
                .get("fabro")
                .ok_or("missing policy")?;
            for (key, value) in route.fabro_metadata {
                assert_eq!(metadata[&key], value);
            }
            assert!(resolved.model().limits().is_some());
            resolved_count += 1;
        }
        assert_eq!(resolved_count, 95);
        assert_eq!(catalog.provider("modal")?.default_model(), Some("kimi-k3"));
    }
    Ok(())
}
