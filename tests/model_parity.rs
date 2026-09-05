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
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Status {
    Catalog,
    Pending,
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
    let mut pending = 0;
    let mut deployments = 0;
    for route in inventory.routes {
        assert!(seen.insert((route.provider.clone(), route.fabro_model.clone())));
        match route.status {
            Status::Pending => {
                pending += 1;
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
    // Removed as imports land; a growing exception list is a regression.
    assert_eq!(pending, 20);
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
