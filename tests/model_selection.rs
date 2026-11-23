#![cfg(feature = "builtin-catalog")]

use std::error::Error as StdError;

use lithos_llm::Request;
use lithos_llm::catalog::{Catalog, ProviderId, SelectorMatch};
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
        let mut providers = vec!["anthropic", "vercel", "openrouter"];
        // Venice's wire id for Opus 4.8 is the dashed form, so it matches
        // after the aliases.
        if alias == "claude-opus-4-8" {
            providers.push("venice");
        }
        let expected: Vec<_> = providers
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

/// Each line's unversioned aliases with the newest generally available model
/// of the line, from Anthropic's models overview and OpenAI's models page,
/// and versioned aliases that must stay on their version.
const LINE_ALIASES: [(&str, &str); 19] = [
    ("fable", "claude-fable-5.2"),
    ("claude-fable", "claude-fable-5.2"),
    ("opus", "claude-opus-5.5"),
    ("claude-opus", "claude-opus-5.5"),
    ("sonnet", "claude-sonnet-5.5"),
    ("claude-sonnet", "claude-sonnet-5.5"),
    ("haiku", "claude-haiku-5.5"),
    ("claude-haiku", "claude-haiku-5.5"),
    ("astra", "gpt-6.2-astra"),
    ("gpt-astra", "gpt-6.2-astra"),
    ("sol", "gpt-6.2-sol"),
    ("gpt-sol", "gpt-6.2-sol"),
    ("terra", "gpt-6.1-terra"),
    ("gpt-terra", "gpt-6.1-terra"),
    ("luna", "gpt-6-luna"),
    ("gpt-luna", "gpt-6-luna"),
    ("claude-haiku-4-5", "claude-haiku-4.5"),
    ("gpt-5.6", "gpt-5.6-sol"),
    ("gpt-6", "gpt-6-astra"),
];

#[test]
fn a_line_alias_names_the_same_model_on_every_provider() -> Result<(), Box<dyn StdError>> {
    let catalog = Catalog::builder().with_builtin().build()?;
    for (alias, canonical) in LINE_ALIASES {
        let matches = catalog.selector_matches(alias);
        assert!(!matches.is_empty(), "{alias} matches no offering");
        for (offering, _) in matches {
            assert_eq!(
                offering.model.id().as_str(),
                canonical,
                "{alias} on {}",
                offering.provider.id()
            );
        }
    }
    Ok(())
}

#[test]
fn a_provider_wire_id_matches_and_routes_like_its_canonical_name() -> Result<(), Box<dyn StdError>>
{
    let catalog = Catalog::builder().with_builtin().build()?;
    let matches = |selector| -> Vec<String> {
        catalog
            .offerings_matching(selector)
            .into_iter()
            .map(|offering| offering.handle().to_string())
            .collect()
    };

    assert_eq!(matches("claude-opus-5-5"), [
        "anthropic/claude-opus-5.5",
        "venice/claude-opus-5.5"
    ]);
    // Prefixed with its provider, as Amp's Custom URL connections send it,
    // the wire id names that provider's offering.
    assert_eq!(matches("anthropic/claude-opus-5-5"), [
        "anthropic/claude-opus-5.5"
    ]);
    // OpenRouter's and Vercel's wire id has the same shape. Its prefix names
    // Anthropic's offering, which ranks first as the resolver would route
    // it, and the gateways' own rows follow.
    assert_eq!(matches("anthropic/claude-opus-5.5"), [
        "anthropic/claude-opus-5.5",
        "vercel/claude-opus-5.5",
        "openrouter/claude-opus-5.5"
    ]);
    assert!(matches("openai/claude-opus-5-5").is_empty());

    // The kind says which form matched, so a caller can tell a provider id
    // prefix from a provider alias (`claude` is Anthropic's) and see that
    // the qualified reading of OpenRouter's wire id ranked first.
    let kinds = |selector| -> Vec<SelectorMatch> {
        catalog
            .selector_matches(selector)
            .into_iter()
            .map(|(_, kind)| kind)
            .collect()
    };
    assert_eq!(matches("anthropic/claude-fable-5"), [
        "anthropic/claude-fable-5",
        "vercel/claude-fable-5",
        "openrouter/claude-fable-5"
    ]);
    assert_eq!(kinds("anthropic/claude-fable-5"), [
        SelectorMatch::ProviderQualified,
        SelectorMatch::WireId,
        SelectorMatch::WireId
    ]);
    assert_eq!(kinds("claude/claude-opus-5-5"), [
        SelectorMatch::ProviderAliasQualified
    ]);
    assert_eq!(kinds("claude-opus-5.5")[0], SelectorMatch::Id);
    assert_eq!(kinds("opus")[0], SelectorMatch::Alias);

    let available = AvailableProviders::new(["openrouter", "venice"].map(ProviderId::new));
    let request = Request::builder()
        .model("claude-opus-5-5")
        .user("Hello")
        .build()?;
    let route = CatalogResolver.resolve(&request, &catalog, &available)?;
    assert_eq!(route.provider().id().as_str(), "venice");
    assert_eq!(route.model().id().as_str(), "claude-opus-5.5");
    Ok(())
}
