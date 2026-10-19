//! Regression for applications sending a provider name as an unrelated wire
//! model.
use std::error::Error;

use httpmock::{Method, MockServer};
use lithos_llm::catalog::Catalog;
use lithos_llm::credentials::StaticCredentials;
use lithos_llm::middleware::{Call, Middleware, Next, Output};
use lithos_llm::resolver::{
    AvailableProviders, ModelResolver, ModelSelectionError, ProviderMatchPolicy, ResolvedRoute,
    SelectionPolicy, UnknownModelPolicy,
};
use lithos_llm::types::{Error as LlmError, ErrorKind};
use lithos_llm::{Client, ModelSelection, Request};
use serde_json::json;

use crate::support;

fn catalog(url: &str) -> Catalog {
    catalog_with_default(url, "flash")
}

fn catalog_with_default(url: &str, default: &str) -> Catalog {
    support::catalog_from_toml(
        "selection",
        &format!(
            r#"
        schema_version = 1
        [providers.anthropic]
        display_name = "Anthropic"
        codecs = ["anthropic-messages"]
        base_url = "{url}"
        priority = 100
        default_model = "sonnet"
        auth = {{ type = "bearer" }}
        [providers.anthropic.models.sonnet]
        display_name = "Sonnet"
        api_model = "claude-wire"
        capabilities = {{ text = true }}
        [providers.gemini]
        display_name = "Gemini"
        aliases = ["google"]
        codecs = ["gemini-generate"]
        base_url = "{url}"
        priority = 10
        allow_passthrough = true
        default_model = "{default}"
        auth = {{ type = "header", name = "x-goog-api-key" }}
        [providers.gemini.models.flash]
        display_name = "Flash"
        aliases = ["fast"]
        api_model = "gemini-wire"
        capabilities = {{ text = true }}
        [providers.gemini.models.pro]
        display_name = "Pro"
        api_model = "gemini-pro-wire"
        capabilities = {{ text = true }}
    "#
        ),
    )
}

fn credentials() -> StaticCredentials {
    StaticCredentials::new()
        .with("anthropic", support::bearer_credentials())
        .with("gemini", support::header_credentials("x-goog-api-key"))
}

fn policy() -> SelectionPolicy {
    SelectionPolicy::default()
        .with_providers(ProviderMatchPolicy::Exact)
        .with_unknown_models(UnknownModelPolicy::DefaultProvider)
}

#[tokio::test]
async fn provider_only_aliases_and_scoped_models_dispatch_the_inspected_gemini_route()
-> Result<(), Box<dyn Error>> {
    let server = MockServer::start_async().await;
    let client = Client::builder()
        .catalog(self::catalog(&server.base_url()))
        .credentials(credentials())
        .resolver(policy())
        .build()?
        .client;
    let wrong = server.mock(|when, then| {
        when.method(Method::POST).path("/v1/messages");
        then.status(500);
    });
    let reply = json!({
        "responseId": "selection",
        "candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}
    });
    let generate = "/v1beta/models/gemini-wire:generateContent";
    let stream_path = "/v1beta/models/gemini-wire:streamGenerateContent";
    let count_path = "/v1beta/models/gemini-wire:countTokens";
    let (complete, complete_slot) = support::mount_capture(&server, generate, &reply);
    let transcript = support::sse_data_transcript(&[&reply.to_string()]);
    let (stream, stream_slot) = support::mount_capture_sse(&server, stream_path, &transcript);
    let (count, count_slot) =
        support::mount_capture(&server, count_path, &json!({"totalTokens": 1}));
    let cases = [
        ModelSelection::from_parts(Some("gemini"), None),
        ModelSelection::from_parts(Some("google"), None),
        ModelSelection::from_parts(Some("fast"), None),
        ModelSelection::from_parts(Some("google/fast"), None),
        ModelSelection::from_parts(Some("google:fast"), None),
        ModelSelection::from_parts(None, Some("google")),
        ModelSelection::from_parts(Some("gemini-wire"), Some("gemini")),
    ];
    for selection in cases {
        let route = client.resolve_selection(&selection)?;
        assert_eq!(route.handle().to_string(), "gemini/flash");
        assert_eq!(route.api_model(), "gemini-wire");
        let request = Request::builder().selection(selection).user("hi").build()?;
        assert_eq!(client.resolve_route(&request)?.handle(), route.handle());
        let response = client.complete(request.clone()).await?;
        assert_eq!(response.model, route.handle());
        assert_eq!(support::captured(&complete_slot).path, generate);
        let events = support::collect_stream_events(client.stream(request.clone()).await?).await;
        support::assert_stream_contract(&events);
        let ended = events
            .iter()
            .find(|event| event["type"] == "ended")
            .expect("ended");
        assert_eq!(
            ended["response"]["model"],
            serde_json::to_value(route.handle())?
        );
        assert!(
            support::captured(&stream_slot)
                .path
                .starts_with(stream_path)
        );
        let tokens = client
            .count_input_tokens(request)
            .await?
            .expect("Gemini token count");
        assert_eq!(tokens.model(), &route.handle());
        assert_eq!(support::captured(&count_slot).path, count_path);
    }
    complete.assert_calls_async(7).await;
    stream.assert_calls_async(7).await;
    count.assert_calls_async(7).await;
    wrong.assert_calls_async(0).await;
    Ok(())
}

#[tokio::test]
async fn configured_default_and_saved_handle_use_distinct_wire_models() -> Result<(), Box<dyn Error>>
{
    let server = MockServer::start_async().await;
    let source = catalog(&server.base_url());
    let old = policy().resolve_selection(
        &ModelSelection::from_parts(Some("gemini"), None),
        &source,
        &AvailableProviders::all(&source),
    )?;
    let changed = catalog_with_default(&server.base_url(), "pro");
    let client = Client::builder()
        .catalog(changed)
        .credentials(credentials())
        .resolver(policy())
        .build()?
        .client;
    for (selection, wire) in [
        (
            ModelSelection::from_parts(Some("gemini"), None),
            "gemini-pro-wire",
        ),
        (old.handle().into(), "gemini-wire"),
    ] {
        let path = format!("/v1beta/models/{wire}:generateContent");
        let (mock, slot) = support::mount_capture(
            &server,
            &path,
            &json!({"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}),
        );
        let route = client.resolve_selection(&selection)?;
        let response = client
            .complete(Request::builder().selection(selection).user("hi").build()?)
            .await?;
        assert_eq!(response.model, route.handle());
        assert_eq!(route.api_model(), wire);
        assert_eq!(support::captured(&slot).path, path);
        mock.assert_async().await;
    }
    Ok(())
}

/// A custom resolver deliberately redirects all selection intent.
struct AlwaysGemini;
impl ModelResolver for AlwaysGemini {
    fn resolve_selection(
        &self,
        _selection: &ModelSelection,
        catalog: &Catalog,
        available: &AvailableProviders,
    ) -> Result<ResolvedRoute, ModelSelectionError> {
        policy().resolve_selection(
            &ModelSelection::from_parts(None, Some("gemini")),
            catalog,
            available,
        )
    }
}

#[tokio::test]
async fn custom_resolvers_drive_inspection_and_dispatch() -> Result<(), Box<dyn Error>> {
    let server = MockServer::start_async().await;
    let client = Client::builder()
        .catalog(catalog(&server.base_url()))
        .credentials(credentials())
        .resolver(AlwaysGemini)
        .build()?
        .client;
    let selection = ModelSelection::from_parts(Some("anthropic"), None);
    let route = client.resolve_selection(&selection)?;
    let (mock, _) = support::mount_capture(
        &server,
        "/v1beta/models/gemini-wire:generateContent",
        &json!({"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}),
    );
    let response = client
        .complete(Request::builder().selection(selection).user("hi").build()?)
        .await?;
    assert_eq!(route.handle().to_string(), "gemini/flash");
    assert_eq!(response.model, route.handle());
    mock.assert_async().await;
    Ok(())
}

struct ChangeScope;

#[async_trait::async_trait]
impl Middleware for ChangeScope {
    async fn handle(&self, call: Call, next: Next) -> Result<Output, LlmError> {
        let call = call.map_request(|request| {
            // Both legacy model text getters say "flash"; only full selection
            // comparison catches the provider change.
            request
                .into_builder()
                .selection(ModelSelection::from_parts(Some("flash"), Some("anthropic")))
                .build()
        })?;
        next.run(call).await
    }
}

#[tokio::test]
async fn middleware_cannot_change_only_the_provider_scope() -> Result<(), Box<dyn Error>> {
    let server = MockServer::start_async().await;
    let receiver = server.mock(|when, then| {
        when.method(Method::POST);
        then.status(500);
    });
    let client = Client::builder()
        .catalog(catalog(&server.base_url()))
        .credentials(credentials())
        .middleware(ChangeScope)
        .build()?
        .client;
    let request = Request::builder()
        .selection(ModelSelection::from_parts(Some("flash"), Some("gemini")))
        .user("hi")
        .build()?;
    let error = client
        .complete(request)
        .await
        .expect_err("scope changes are forbidden");
    assert_eq!(error.kind(), ErrorKind::Middleware);
    receiver.assert_calls_async(0).await;
    Ok(())
}

#[tokio::test]
async fn unknown_bare_models_use_the_configured_policy_for_inspection_and_wire_dispatch()
-> Result<(), Box<dyn Error>> {
    let server = MockServer::start_async().await;
    let client = Client::builder()
        .catalog(catalog(&server.base_url()))
        .credentials(credentials())
        .enabled_providers(["gemini"])
        .resolver(policy())
        .build()?
        .client;
    let selection = ModelSelection::from_parts(Some("brand-new"), None);
    let route = client.resolve_selection(&selection)?;
    assert_eq!(route.handle().to_string(), "gemini/brand-new");
    let (mock, slot) = support::mount_capture(
        &server,
        "/v1beta/models/brand-new:generateContent",
        &json!({"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}),
    );
    let response = client
        .complete(Request::builder().selection(selection).user("hi").build()?)
        .await?;
    assert_eq!(response.model, route.handle());
    assert_eq!(
        support::captured(&slot).path,
        "/v1beta/models/brand-new:generateContent"
    );
    mock.assert_async().await;
    Ok(())
}
