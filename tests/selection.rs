//! Public, runtime-independent selection contract.
use std::error::Error;

use lithos_llm::catalog::{Catalog, ModelHandle, ModelId, ProviderId};
use lithos_llm::resolver::{
    AvailableProviders, CatalogResolver, ModelResolver, ModelSelectionError, ProviderMatchPolicy,
    SelectionPolicy, UnknownModelPolicy,
};
use lithos_llm::{Evaluation, ModelSelection, Request};

const CATALOG: &str = r#"
schema_version = 1
[providers.alpha]
display_name = "Alpha"
aliases = ["a"]
base_url = "http://127.0.0.1"
priority = 100
allow_passthrough = true
default_model = "one"
auth = { type = "none" }
[providers.alpha.models.one]
display_name = "One"
api_model = "wire-one"
aliases = ["uno", "shared"]
[providers.alpha.models.two]
display_name = "Two"
api_model = "vendor/two:latest"
[providers.beta]
display_name = "Beta"
aliases = ["b"]
base_url = "http://127.0.0.1"
priority = 10
default_model = "shared"
auth = { type = "none" }
[providers.beta.models.shared]
display_name = "Shared"
api_model = "wire-shared"
[providers.beta.models.alpha]
display_name = "A model also named Alpha"
api_model = "beta-alpha-wire"
[providers.seat]
display_name = "Seat"
base_url = "http://127.0.0.1"
stands_in_for = "alpha"
default_model = "one"
auth = { type = "none" }
[providers.seat.models.one]
display_name = "One"
api_model = "seat-one"
[providers.parked]
display_name = "Parked"
base_url = "http://127.0.0.1"
enabled = false
auth = { type = "none" }
[providers.parked.models.parked_only]
display_name = "Parked only"
api_model = "parked-wire"
[providers.defaultless]
display_name = "Defaultless"
base_url = "http://127.0.0.1"
auth = { type = "none" }
"#;

fn catalog() -> Result<Catalog, Box<dyn Error>> {
    Ok(Catalog::builder().overlay_toml(CATALOG)?.build()?)
}
fn eligible(ids: &[&str]) -> AvailableProviders {
    AvailableProviders::new(ids.iter().map(|id| ProviderId::new(*id)))
}
fn exact() -> SelectionPolicy {
    SelectionPolicy::default().with_providers(ProviderMatchPolicy::Exact)
}
fn permissive() -> SelectionPolicy {
    exact().with_unknown_models(UnknownModelPolicy::DefaultProvider)
}

#[test]
fn defaults_aliases_qualification_and_priorities_share_one_algorithm() -> Result<(), Box<dyn Error>>
{
    let catalog = catalog()?;
    let ready = AvailableProviders::all(&catalog);
    for (model, provider, expected) in [
        (None, None, "alpha/one"),
        (Some("default"), None, "alpha/one"),
        (Some("alpha"), None, "alpha/one"),
        (Some("alpha"), Some("beta"), "beta/alpha"),
        (Some("b"), None, "beta/shared"),
        (Some("uno"), None, "alpha/one"),
        (Some("shared"), None, "beta/shared"),
        (Some("a/uno"), None, "alpha/one"),
        (Some("a:uno"), None, "alpha/one"),
        (None, Some("a"), "alpha/one"),
        (Some("wire-one"), Some("a"), "alpha/one"),
    ] {
        let selection = ModelSelection::from_parts(model, provider);
        let route = exact().resolve_selection(&selection, &catalog, &ready)?;
        assert_eq!(route.handle().to_string(), expected);
    }
    Ok(())
}

#[test]
fn configured_defaults_override_builtin_defaults() -> Result<(), Box<dyn Error>> {
    let catalog = Catalog::builder()
        .overlay_toml(CATALOG)?
        .overlay_toml("schema_version = 1\n[providers.alpha]\ndefault_model = 'two'")?
        .build()?;
    for selection in [
        ModelSelection::Default,
        ModelSelection::from_parts(Some("a"), None),
        ModelSelection::from_parts(None, Some("alpha")),
    ] {
        let route = CatalogResolver.resolve_selection(
            &selection,
            &catalog,
            &AvailableProviders::all(&catalog),
        )?;
        assert_eq!(route.handle().to_string(), "alpha/two");
        assert_eq!(route.api_model(), "vendor/two:latest");
    }
    Ok(())
}

#[test]
fn explicit_scope_keeps_opaque_model_text_whole() -> Result<(), Box<dyn Error>> {
    let catalog = catalog()?;
    let ready = AvailableProviders::all(&catalog);
    for (model, canonical) in [
        ("vendor/two:latest", "two"),
        ("beta/shared", "beta/shared"),
        ("ghost:model", "ghost:model"),
        ("default", "default"),
        ("alpha", "alpha"),
    ] {
        let selection = ModelSelection::from_parts(Some(model), Some("a"));
        let route = exact().resolve_selection(&selection, &catalog, &ready)?;
        assert_eq!(route.provider().id().as_str(), "alpha");
        assert_eq!(route.handle().model().as_str(), canonical);
        assert_eq!(route.api_model(), model);
    }
    Ok(())
}

#[test]
fn passthrough_is_explicit_policy_and_never_overrides_provider_permission()
-> Result<(), Box<dyn Error>> {
    let catalog = catalog()?;
    let ready = AvailableProviders::all(&catalog);
    let bare = ModelSelection::from_parts(Some("new-model"), None);
    assert!(matches!(
        exact().resolve_selection(&bare, &catalog, &ready),
        Err(ModelSelectionError::ModelNotFound { .. })
    ));
    let route = permissive().resolve_selection(&bare, &catalog, &ready)?;
    assert_eq!(route.handle().to_string(), "alpha/new-model");
    assert!(route.model().is_passthrough());
    assert!(
        permissive()
            .resolve_selection(&bare, &catalog, &eligible(&["beta"]))
            .is_err()
    );
    for policy in [
        exact(),
        permissive(),
        exact().with_unknown_models(UnknownModelPolicy::CatalogOnly),
    ] {
        assert!(
            policy
                .resolve_selection(
                    &ModelSelection::from_parts(Some("new-model"), Some("beta")),
                    &catalog,
                    &ready
                )
                .is_err()
        );
    }
    let scoped = ModelSelection::from_parts(Some("new-model"), Some("alpha"));
    assert!(exact().resolve_selection(&scoped, &catalog, &ready).is_ok());
    assert!(
        exact()
            .with_unknown_models(UnknownModelPolicy::CatalogOnly)
            .resolve_selection(&scoped, &catalog, &ready)
            .is_err()
    );
    Ok(())
}

#[test]
fn provider_failures_never_fall_back_to_an_unrelated_provider() -> Result<(), Box<dyn Error>> {
    let catalog = catalog()?;
    let ready = eligible(&["beta", "parked", "defaultless"]);
    for selector in ["alpha", "a", "alpha/one", "alpha:one"] {
        assert!(matches!(
            permissive().resolve_selection(
                &ModelSelection::from_parts(Some(selector), None),
                &catalog,
                &ready
            ),
            Err(ModelSelectionError::ProviderUnavailable { .. })
        ));
    }
    for selector in ["ghost/one", "ghost:one"] {
        assert!(matches!(
            permissive().resolve_selection(
                &ModelSelection::from_parts(Some(selector), None),
                &catalog,
                &ready
            ),
            Err(ModelSelectionError::ProviderNotFound { .. })
        ));
    }
    for model in [None, Some("one")] {
        assert!(matches!(
            permissive().resolve_selection(
                &ModelSelection::from_parts(model, Some("ghost")),
                &catalog,
                &ready
            ),
            Err(ModelSelectionError::ProviderNotFound { .. })
        ));
        assert!(matches!(
            permissive().resolve_selection(
                &ModelSelection::from_parts(model, Some("parked")),
                &catalog,
                &ready
            ),
            Err(ModelSelectionError::ProviderDisabled { .. })
        ));
    }
    assert!(matches!(
        permissive().resolve_selection(
            &ModelSelection::from_parts(Some("defaultless"), None),
            &catalog,
            &ready
        ),
        Err(ModelSelectionError::ProviderHasNoDefault { .. })
    ));
    assert!(matches!(
        permissive().resolve_selection(
            &ModelSelection::Default,
            &catalog,
            &eligible(&["defaultless"])
        ),
        Err(ModelSelectionError::NoDefaultModel)
    ));
    // A known model on an ineligible or disabled provider is not "unknown".
    for selector in ["uno", "parked_only"] {
        assert!(matches!(
            permissive().resolve_selection(
                &ModelSelection::from_parts(Some(selector), None),
                &catalog,
                &ready
            ),
            Err(ModelSelectionError::ModelNotFound { .. })
        ));
    }
    Ok(())
}

#[test]
fn stand_ins_are_opt_in_policy_and_never_apply_to_resolved_handles() -> Result<(), Box<dyn Error>> {
    let catalog = catalog()?;
    let ready = eligible(&["seat"]);
    for selection in [
        ModelSelection::from_parts(Some("a"), None),
        ModelSelection::from_parts(Some("a/one"), None),
        ModelSelection::from_parts(None, Some("a")),
    ] {
        assert_eq!(
            CatalogResolver
                .resolve_selection(&selection, &catalog, &ready)?
                .handle()
                .to_string(),
            "seat/one"
        );
        assert!(matches!(
            exact().resolve_selection(&selection, &catalog, &ready),
            Err(ModelSelectionError::ProviderUnavailable { .. })
        ));
    }
    let selection = ModelSelection::from(ModelHandle::new(
        ProviderId::new("alpha"),
        ModelId::new("one"),
    ));
    assert!(matches!(
        CatalogResolver.resolve_selection(&selection, &catalog, &ready),
        Err(ModelSelectionError::ProviderUnavailable { .. })
    ));
    Ok(())
}

#[test]
fn a_saved_handle_ignores_new_defaults_and_aliases_and_refuses_removed_rows()
-> Result<(), Box<dyn Error>> {
    let original = catalog()?;
    let route = exact().resolve_selection(
        &ModelSelection::from_parts(Some("uno"), None),
        &original,
        &AvailableProviders::all(&original),
    )?;
    let handle: ModelHandle = serde_json::from_str(&serde_json::to_string(&route.handle())?)?;
    let changed = Catalog::builder()
        .overlay_toml(CATALOG)?
        .overlay_toml("schema_version = 1\n[providers.alpha]\ndefault_model = \"two\"")?
        .build()?;
    let resolved = exact().resolve_selection(
        &handle.clone().into(),
        &changed,
        &AvailableProviders::all(&changed),
    )?;
    assert_eq!(resolved.handle(), handle);
    assert_eq!(resolved.api_model(), "wire-one");
    let removed = Catalog::builder()
        .overlay_toml(
            &CATALOG
                .replace("models.one", "models.renamed")
                .replace("default_model = \"one\"", "default_model = \"renamed\"")
                .replace(
                    "aliases = [\"uno\", \"shared\"]",
                    "aliases = [\"one\", \"uno\", \"shared\"]",
                ),
        )?
        .build()?;
    assert!(matches!(
        permissive().resolve_selection(
            &handle.into(),
            &removed,
            &AvailableProviders::all(&removed)
        ),
        Err(ModelSelectionError::ModelNotFound { .. })
    ));
    Ok(())
}

#[test]
fn structured_selection_survives_request_and_evaluation_round_trips() -> Result<(), Box<dyn Error>>
{
    for selection in [
        ModelSelection::Default,
        ModelSelection::from_parts(Some("a/uno"), None),
        ModelSelection::from_parts(None, Some("a")),
        ModelSelection::from_parts(Some("vendor/two:latest"), Some("a")),
        ModelHandle::new(ProviderId::new("alpha"), ModelId::new("one")).into(),
    ] {
        let request = Request::builder()
            .selection(selection.clone())
            .user("hi")
            .build()?;
        let restored: Request = serde_json::from_value(serde_json::to_value(&request)?)?;
        assert_eq!(restored.selection(), &selection);
        assert_eq!(request.clone().into_builder().build()?, request);
        let evaluation = Evaluation::builder()
            .selection(selection.clone())
            .state("hi")
            .boolean("ok", "OK?")
            .build()?;
        let restored: Evaluation = serde_json::from_value(serde_json::to_value(&evaluation)?)?;
        assert_eq!(restored.selection(), &selection);
        assert_eq!(evaluation.clone().into_builder().build()?, evaluation);
    }
    let request = Request::builder().model("a/uno").user("hi").build()?;
    assert_eq!(serde_json::to_value(request)?["model"], "a/uno");
    Ok(())
}

#[test]
fn malformed_structured_inputs_are_not_silently_reinterpreted() {
    for value in [
        serde_json::json!({"provider":"alpha", "model":"one", "handle":{"provider":"beta","model":"shared"}}),
        serde_json::json!({"provider":5}),
        serde_json::json!({"provider":"alpha", "model":false}),
        serde_json::json!({"selector":"uno"}),
    ] {
        assert!(serde_json::from_value::<ModelSelection>(value).is_err());
    }
    assert!(Request::builder().user("hi").build().is_err());
    assert!(
        Evaluation::builder()
            .state("hi")
            .boolean("ok", "OK?")
            .build()
            .is_err()
    );
}
