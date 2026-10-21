#![cfg(feature = "builtin-catalog")]

use std::error::Error as StdError;

use lithos_llm::Request;
use lithos_llm::catalog::{Catalog, ProviderId};
use lithos_llm::resolver::{AvailableProviders, CatalogResolver, ModelResolver};

const CLAUDE_ALIASES: [(&str, &str); 3] = [
    ("claude-sonnet-4-6", "claude-sonnet-4.6"),
    ("claude-opus-4-7", "claude-opus-4.7"),
    ("claude-opus-4-8", "claude-opus-4.8"),
];

#[test]
fn claude_alias_catalog_matches_prefer_anthropic_over_gateways() -> Result<(), Box<dyn StdError>> {
    let catalog = Catalog::builder().with_builtin().build()?;

    for (alias, canonical) in CLAUDE_ALIASES {
        let matches: Vec<_> = catalog
            .offerings_matching(alias)
            .into_iter()
            .map(|offering| {
                (
                    offering.provider.id().to_string(),
                    offering.model.id().to_string(),
                )
            })
            .collect();
        let expected: Vec<_> = ["anthropic", "vercel", "openrouter"]
            .into_iter()
            .map(|provider| (provider.to_owned(), canonical.to_owned()))
            .collect();
        assert_eq!(matches, expected, "catalog matches for {alias}");
    }
    Ok(())
}

#[test]
fn claude_aliases_route_like_canonical_names_with_available_providers()
-> Result<(), Box<dyn StdError>> {
    let catalog = Catalog::builder().with_builtin().build()?;
    let configurations: &[(&[&str], &str)] = &[
        (&["anthropic"], "anthropic"),
        (&["anthropic", "openrouter"], "anthropic"),
        (&["anthropic", "openrouter", "vercel"], "anthropic"),
        (&["openrouter"], "openrouter"),
        (&["openrouter", "vercel"], "vercel"),
    ];

    for (alias, canonical) in CLAUDE_ALIASES {
        for &(providers, expected_provider) in configurations {
            let available = AvailableProviders::new(providers.iter().copied().map(ProviderId::new));
            for selector in [alias, canonical] {
                let request = Request::builder().model(selector).user("Hello").build()?;
                let route = CatalogResolver.resolve(&request, &catalog, &available)?;
                assert_eq!(
                    route.provider().id().as_str(),
                    expected_provider,
                    "{selector}"
                );
                assert_eq!(route.model().id().as_str(), canonical, "{selector}");
                let expected_wire_model = if expected_provider == "anthropic" {
                    alias.to_owned()
                } else {
                    format!("anthropic/{canonical}")
                };
                assert_eq!(route.api_model(), expected_wire_model, "{selector}");
            }
        }

        let available = AvailableProviders::all(&catalog);
        for provider in ["anthropic", "openrouter", "vercel"] {
            let request = Request::builder()
                .model(format!("{provider}/{alias}"))
                .user("Hello")
                .build()?;
            let route = CatalogResolver.resolve(&request, &catalog, &available)?;
            assert_eq!(route.provider().id().as_str(), provider);
            assert_eq!(route.model().id().as_str(), canonical);
        }
    }
    Ok(())
}
