//! Client construction and inference behavior.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use futures_util::StreamExt as _;
use futures_util::stream::unfold;
use thiserror::Error;
use tokio::time::{Instant as TokioInstant, sleep_until};

use crate::adapter::{
    AdapterBuildError, AdapterContext, AdapterFactory, AdapterRegistry, InputTokenCount,
    ProviderAdapter, ResolvedCall,
};
use crate::catalog::{AdapterId, Catalog, CatalogError, ProviderId, adapter_ids};
#[cfg(all(
    feature = "builtin-catalog",
    feature = "environment-credentials",
    any(
        feature = "openai",
        feature = "anthropic",
        feature = "gemini",
        feature = "openai-compatible",
        feature = "bedrock"
    )
))]
use crate::credentials::EnvironmentCredentials;
use crate::credentials::{CredentialProvider, NoCredentials};
use crate::middleware::{Call, CallContext, CancellationToken, Middleware, Mode, Output, Pipeline};
use crate::providers::register_builtin;
use crate::resolver::{
    AvailableProviders, CatalogResolver, ModelResolver, ModelSelectionError, ResolvedRoute,
};
use crate::types::{
    ContentPart, Error, ErrorKind, Message, Request, Response, ResponseFormat, ResponseStream,
};

/// An immutable provider-neutral client.
#[derive(Clone)]
pub struct Client {
    catalog:   Catalog,
    resolver:  Arc<dyn ModelResolver>,
    available: AvailableProviders,
    pipeline:  Arc<Pipeline>,
}

impl fmt::Debug for Client {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Client")
            .field("available_providers", &self.available)
            .finish_non_exhaustive()
    }
}

impl Client {
    pub fn builder() -> ClientBuilder {
        ClientBuilder::new()
    }

    /// Builds a client with the built-in catalog and conventional environment
    /// variable names.
    ///
    /// Credentials are resolved for each provider attempt. This constructor
    /// does not read credentials while it builds the client.
    ///
    /// Providers that cannot create an adapter are reported through
    /// [`ClientBuild::issues`] rather than failing the call.
    ///
    /// # Errors
    ///
    /// Returns an error if the built-in catalog is invalid or the default HTTP
    /// client cannot be built.
    #[cfg(all(
        feature = "builtin-catalog",
        feature = "environment-credentials",
        any(
            feature = "openai",
            feature = "anthropic",
            feature = "gemini",
            feature = "openai-compatible",
            feature = "bedrock"
        )
    ))]
    pub fn from_env() -> Result<ClientBuild, ClientBuildError> {
        let catalog = Catalog::builder()
            .with_builtin()
            .build()
            .map_err(|source| ClientBuildError::BuiltInCatalog { source })?;
        Self::builder()
            .catalog(catalog)
            .credentials(EnvironmentCredentials::conventional())
            .build()
    }

    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    /// The providers whose adapters were constructed successfully.
    ///
    /// This is the set the resolver may select from. It can be smaller than
    /// [`Client::catalog`], and it can be empty.
    pub fn available_providers(&self) -> &AvailableProviders {
        &self.available
    }

    /// Resolves the provider and model this request would use.
    ///
    /// The client's configured resolver, catalog, and available-provider set
    /// decide the route, so the result matches what [`Client::complete`],
    /// [`Client::stream`], and [`Client::count_input_tokens`] would dispatch.
    ///
    /// This is synchronous and side effect free. It does not read credentials,
    /// run middleware, dispatch an adapter, or validate that the model
    /// supports the request.
    ///
    /// # Errors
    ///
    /// Returns an error if the request selector names no provider or model
    /// that is both in the catalog and available.
    pub fn resolve_route(&self, request: &Request) -> Result<ResolvedRoute, ModelSelectionError> {
        self.resolver
            .resolve(request, &self.catalog, &self.available)
    }

    pub async fn complete(&self, request: Request) -> Result<Response, Error> {
        self.complete_with_context(request, CallContext::new())
            .await
    }

    pub async fn complete_with_context(
        &self,
        request: Request,
        context: CallContext,
    ) -> Result<Response, Error> {
        match self.call_guarded(request, context, Mode::Complete).await? {
            Output::Complete(response) => Ok(response),
            Output::Stream(_) => Err(Error::new(
                ErrorKind::Middleware,
                "complete middleware returned a stream",
            )),
        }
    }

    pub async fn stream(&self, request: Request) -> Result<ResponseStream, Error> {
        self.stream_with_context(request, CallContext::new()).await
    }

    pub async fn stream_with_context(
        &self,
        request: Request,
        context: CallContext,
    ) -> Result<ResponseStream, Error> {
        let cancellation = context.cancellation().clone();
        let deadline = context.deadline();
        match self.call_guarded(request, context, Mode::Stream).await? {
            Output::Stream(stream) => Ok(guard_stream(stream, cancellation, deadline)),
            Output::Complete(_) => Err(Error::new(
                ErrorKind::Middleware,
                "stream middleware returned a complete response",
            )),
        }
    }

    pub async fn count_input_tokens(
        &self,
        request: Request,
    ) -> Result<Option<InputTokenCount>, Error> {
        let route = self.resolve_route(&request)?;
        validate_request(&request, &route)?;
        let adapter = self
            .pipeline
            .adapters
            .get(route.provider().id())
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::Configuration,
                    format!(
                        "no adapter is registered for provider {}",
                        route.provider().id()
                    ),
                )
            })?;
        adapter
            .count_input_tokens(&ResolvedCall::new(request, route, CallContext::new()))
            .await
    }

    async fn call(
        &self,
        request: Request,
        context: CallContext,
        mode: Mode,
    ) -> Result<Output, Error> {
        let route = self.resolve_route(&request)?;
        validate_request(&request, &route)?;
        self.pipeline
            .start()
            .run(Call {
                request,
                route,
                mode,
                context,
            })
            .await
    }

    async fn call_guarded(
        &self,
        request: Request,
        context: CallContext,
        mode: Mode,
    ) -> Result<Output, Error> {
        let cancellation = context.cancellation().clone();
        if let Some(deadline) = context.deadline() {
            tokio::select! {
                () = cancellation.cancelled() => Err(cancelled_error()),
                () = sleep_until(TokioInstant::from_std(deadline)) => Err(deadline_error()),
                result = self.call(request, context, mode) => result,
            }
        } else {
            tokio::select! {
                () = cancellation.cancelled() => Err(cancelled_error()),
                result = self.call(request, context, mode) => result,
            }
        }
    }
}

fn validate_request(request: &Request, route: &ResolvedRoute) -> Result<(), Error> {
    let capabilities = route.model().capabilities();
    if !request.tools().is_empty() && !capabilities.tools {
        return Err(unsupported_capability(route, "tools"));
    }
    if matches!(
        request.response_format(),
        Some(ResponseFormat::JsonObject | ResponseFormat::JsonSchema { .. })
    ) && !capabilities.structured_output
    {
        return Err(unsupported_capability(route, "structured output"));
    }
    if request.reasoning_effort().is_some() && !capabilities.reasoning {
        return Err(unsupported_capability(route, "reasoning"));
    }
    if (request.temperature().is_some() || request.top_p().is_some()) && !capabilities.sampling {
        return Err(unsupported_capability(route, "sampling"));
    }
    if let Some(limits) = route.model().limits() {
        if request
            .max_output_tokens()
            .is_some_and(|tokens| u64::from(tokens) > limits.max_output_tokens)
        {
            return Err(Error::new(
                ErrorKind::InvalidRequest,
                format!(
                    "model {} allows at most {} output tokens",
                    route.handle(),
                    limits.max_output_tokens
                ),
            )
            .with_provider(route.provider().id().clone())
            .with_provider_code("max_output_tokens"));
        }
    }
    for part in request.messages().iter().flat_map(Message::content) {
        let capability = match part {
            ContentPart::Text { .. } if !capabilities.text => Some("text"),
            ContentPart::Image(_) if !capabilities.images => Some("images"),
            ContentPart::Audio(_) if !capabilities.audio => Some("audio"),
            ContentPart::Document(_) if !capabilities.documents => Some("documents"),
            ContentPart::Reasoning(_) if !capabilities.reasoning => Some("reasoning"),
            ContentPart::ToolCall(_) | ContentPart::ToolResult(_) if !capabilities.tools => {
                Some("tools")
            }
            _ => None,
        };
        if let Some(capability) = capability {
            return Err(unsupported_capability(route, capability));
        }
    }
    Ok(())
}

fn unsupported_capability(route: &ResolvedRoute, capability: &str) -> Error {
    Error::new(
        ErrorKind::InvalidRequest,
        format!("model {} does not support {capability}", route.handle()),
    )
    .with_provider(route.provider().id().clone())
    .with_provider_code("unsupported_capability")
}

fn guard_stream(
    stream: ResponseStream,
    cancellation: CancellationToken,
    deadline: Option<Instant>,
) -> ResponseStream {
    let stream = cancellation_stream(stream, cancellation);
    match deadline {
        Some(deadline) => deadline_stream(stream, deadline),
        None => stream,
    }
}

fn cancellation_stream(stream: ResponseStream, cancellation: CancellationToken) -> ResponseStream {
    Box::pin(unfold(
        (stream, cancellation, false),
        |(mut stream, cancellation, finished)| async move {
            if finished {
                return None;
            }
            tokio::select! {
                () = cancellation.cancelled() => {
                    Some((Err(cancelled_error()), (stream, cancellation, true)))
                }
                item = stream.next() => {
                    item.map(|item| (item, (stream, cancellation, false)))
                }
            }
        },
    ))
}

fn deadline_stream(stream: ResponseStream, deadline: Instant) -> ResponseStream {
    Box::pin(unfold(
        (stream, false),
        move |(mut stream, finished)| async move {
            if finished {
                return None;
            }
            tokio::select! {
                () = sleep_until(TokioInstant::from_std(deadline)) => {
                    Some((Err(deadline_error()), (stream, true)))
                }
                item = stream.next() => item.map(|item| (item, (stream, false))),
            }
        },
    ))
}

fn cancelled_error() -> Error {
    Error::new(ErrorKind::Cancelled, "the call was cancelled")
}

fn deadline_error() -> Error {
    Error::new(ErrorKind::Timeout, "the call deadline expired")
}

/// Builds a client from immutable catalog data and runtime extensions.
#[must_use]
pub struct ClientBuilder {
    catalog:     Option<Catalog>,
    resolver:    Arc<dyn ModelResolver>,
    credentials: Arc<dyn CredentialProvider>,
    http:        Option<reqwest::Client>,
    middleware:  Vec<Arc<dyn Middleware>>,
    registry:    AdapterRegistry,
    enabled:     Option<BTreeSet<ProviderId>>,
}

impl Default for ClientBuilder {
    fn default() -> Self {
        let mut registry = AdapterRegistry::new();
        register_builtin(&mut registry);
        Self {
            catalog: None,
            resolver: Arc::new(CatalogResolver),
            credentials: Arc::new(NoCredentials),
            http: None,
            middleware: Vec::new(),
            registry,
            enabled: None,
        }
    }
}

impl ClientBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn catalog(mut self, catalog: Catalog) -> Self {
        self.catalog = Some(catalog);
        self
    }

    pub fn resolver(mut self, resolver: impl ModelResolver + 'static) -> Self {
        self.resolver = Arc::new(resolver);
        self
    }

    pub fn resolver_arc(mut self, resolver: Arc<dyn ModelResolver>) -> Self {
        self.resolver = resolver;
        self
    }

    pub fn credentials(mut self, credentials: impl CredentialProvider + 'static) -> Self {
        self.credentials = Arc::new(credentials);
        self
    }

    pub fn credentials_arc(mut self, credentials: Arc<dyn CredentialProvider>) -> Self {
        self.credentials = credentials;
        self
    }

    /// Injects an application-configured HTTP client.
    ///
    /// Every built-in adapter uses this client. Without it the builder creates
    /// a default client that sends the Lithos user agent.
    pub fn http(mut self, client: reqwest::Client) -> Self {
        self.http = Some(client);
        self
    }

    /// Builds adapters only for the named providers.
    ///
    /// Ids must be canonical [`ProviderId`] values; provider aliases are not
    /// accepted. The last call replaces any earlier selection. Without a
    /// selection the builder attempts every provider in the catalog.
    ///
    /// The client keeps the complete catalog either way, so unselected
    /// providers stay visible through [`Client::catalog`] even though they
    /// cannot be resolved as routes.
    pub fn enabled_providers(
        mut self,
        providers: impl IntoIterator<Item = impl Into<ProviderId>>,
    ) -> Self {
        self.enabled = Some(providers.into_iter().map(Into::into).collect());
        self
    }

    /// Adds middleware. The first layer added is the outermost layer.
    pub fn middleware(mut self, middleware: impl Middleware) -> Self {
        self.middleware.push(Arc::new(middleware));
        self
    }

    pub fn middleware_arc(mut self, middleware: Arc<dyn Middleware>) -> Self {
        self.middleware.push(middleware);
        self
    }

    pub fn adapter_factory(
        mut self,
        id: impl Into<AdapterId>,
        factory: impl AdapterFactory + 'static,
    ) -> Self {
        self.registry.register_factory(id, factory);
        self
    }

    pub fn adapter(
        mut self,
        provider: impl Into<ProviderId>,
        adapter: impl ProviderAdapter + 'static,
    ) -> Self {
        self.registry.register_adapter(provider, adapter);
        self
    }

    pub fn adapter_arc(
        mut self,
        provider: impl Into<ProviderId>,
        adapter: Arc<dyn ProviderAdapter>,
    ) -> Self {
        self.registry.register_adapter_arc(provider, adapter);
        self
    }

    /// Builds the client and reports every provider that could not be built.
    ///
    /// Each enabled provider is attempted in catalog order. Successful
    /// adapters go to the client and provider-local failures go to
    /// [`ClientBuild::issues`] in that same order. A client with no available
    /// provider is a valid outcome. Credentials are never read here.
    ///
    /// # Errors
    ///
    /// Returns an error only for failures that prevent coherent construction:
    /// a missing catalog, a default HTTP client that cannot be built, or an
    /// enabled provider id that is not a canonical catalog provider.
    pub fn build(self) -> Result<ClientBuild, ClientBuildError> {
        let catalog = self.catalog.ok_or(ClientBuildError::MissingCatalog)?;
        let http = match self.http {
            Some(http) => http,
            None => reqwest::Client::builder()
                .user_agent(concat!("lithos-llm/", env!("CARGO_PKG_VERSION")))
                .build()
                .map_err(ClientBuildError::HttpClient)?,
        };
        if let Some(enabled) = &self.enabled {
            for provider in enabled {
                if catalog.provider_by_id(provider).is_none() {
                    return Err(ClientBuildError::UnknownEnabledProvider {
                        provider: provider.clone(),
                    });
                }
            }
        }
        let context = AdapterContext::new(http, self.credentials);
        let mut adapters = BTreeMap::new();
        let mut issues = Vec::new();
        for provider in catalog.providers() {
            if self
                .enabled
                .as_ref()
                .is_some_and(|enabled| !enabled.contains(provider.id()))
            {
                continue;
            }
            let outcome = if let Some(adapter) = self.registry.explicit(provider.id()) {
                Ok(adapter)
            } else if let Some(factory) = self.registry.factory(provider.adapter()) {
                factory
                    .create(provider, &context)
                    .map_err(ProviderBuildCause::Adapter)
            } else if is_disabled_builtin_adapter(provider.adapter()) {
                Err(ProviderBuildCause::AdapterFeatureDisabled {
                    adapter: provider.adapter().clone(),
                })
            } else {
                Err(ProviderBuildCause::MissingAdapterFactory {
                    adapter: provider.adapter().clone(),
                })
            };
            match outcome {
                Ok(adapter) => {
                    adapters.insert(provider.id().clone(), adapter);
                }
                Err(cause) => issues.push(ProviderBuildIssue {
                    provider: provider.id().clone(),
                    adapter: provider.adapter().clone(),
                    cause,
                }),
            }
        }
        let available = AvailableProviders::new(adapters.keys().cloned());
        Ok(ClientBuild {
            client: Client {
                catalog,
                resolver: self.resolver,
                available,
                pipeline: Arc::new(Pipeline {
                    middleware: self.middleware,
                    adapters,
                }),
            },
            issues,
        })
    }
}

fn is_disabled_builtin_adapter(adapter: &AdapterId) -> bool {
    match adapter.as_str() {
        adapter_ids::OPENAI => !cfg!(feature = "openai"),
        adapter_ids::ANTHROPIC => !cfg!(feature = "anthropic"),
        adapter_ids::GEMINI => !cfg!(feature = "gemini"),
        adapter_ids::OPENAI_COMPATIBLE => !cfg!(feature = "openai-compatible"),
        adapter_ids::BEDROCK => !cfg!(feature = "bedrock"),
        _ => false,
    }
}

/// A constructed client plus every provider-local construction issue.
///
/// The client is usable even when `issues` is not empty. Applications that
/// need to report degraded startup read `issues`; applications that do not can
/// take `client` and continue.
#[derive(Debug)]
#[non_exhaustive]
pub struct ClientBuild {
    pub client: Client,
    pub issues: Vec<ProviderBuildIssue>,
}

/// One provider that could not be constructed.
#[derive(Debug)]
#[non_exhaustive]
pub struct ProviderBuildIssue {
    pub provider: ProviderId,
    pub adapter:  AdapterId,
    pub cause:    ProviderBuildCause,
}

/// Why one provider could not be constructed.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ProviderBuildCause {
    #[error("no adapter factory is registered for adapter {adapter}")]
    MissingAdapterFactory { adapter: AdapterId },
    #[error("the feature for adapter {adapter} is disabled")]
    AdapterFeatureDisabled { adapter: AdapterId },
    #[error("adapter construction failed")]
    Adapter(#[source] AdapterBuildError),
}

/// Client construction failed before any provider was attempted.
///
/// Provider-local failures are reported through [`ProviderBuildIssue`]
/// instead, so this error means no coherent client could be built at all.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ClientBuildError {
    #[error("a catalog is required")]
    MissingCatalog,
    #[error("the built-in catalog could not be loaded")]
    BuiltInCatalog {
        #[source]
        source: CatalogError,
    },
    #[error("the default HTTP client could not be built")]
    HttpClient(#[source] reqwest::Error),
    #[error("enabled provider `{provider}` is not in the catalog")]
    UnknownEnabledProvider { provider: ProviderId },
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use futures_util::stream::empty;

    use super::{Client, ClientBuildError, ProviderBuildCause};
    use crate::adapter::{
        AdapterBuildError, AdapterContext, AdapterFactory, ProviderAdapter, ResolvedCall,
    };
    use crate::catalog::{AdapterId, Catalog, CatalogProvider, ProviderId};
    use crate::resolver::{AvailableProviders, ModelResolver, ModelSelectionError, ResolvedRoute};
    use crate::types::{ContentPart, Error, Request, Response, ResponseStream};

    const TEST_CATALOG: &str = r#"
        schema_version = 1

        [providers.alpha]
        display_name = "Alpha"
        aliases = ["a"]
        adapter = "alpha-adapter"
        codec = "test-codec"
        base_url = "http://127.0.0.1"
        priority = 10
        allow_passthrough = true
        default_model = "one"
        auth = { type = "none" }

        [providers.alpha.models.one]
        display_name = "One"
        aliases = ["uno"]
        api_model = "alpha-one-v1"
        capabilities = { text = true }

        [providers.beta]
        display_name = "Beta"
        adapter = "beta-adapter"
        codec = "test-codec"
        base_url = "http://127.0.0.1"
        priority = 5
        default_model = "two"
        auth = { type = "none" }

        [providers.beta.models.two]
        display_name = "Two"
        api_model = "beta-two-v1"
        capabilities = { text = true }

        [providers.gamma]
        display_name = "Gamma"
        adapter = "gamma-adapter"
        codec = "test-codec"
        base_url = "http://127.0.0.1"
        priority = 1
        default_model = "three"
        auth = { type = "none" }

        [providers.gamma.models.three]
        display_name = "Three"
        api_model = "gamma-three-v1"
        capabilities = { text = true }
    "#;

    struct FakeAdapter {
        id: AdapterId,
    }

    #[async_trait]
    impl ProviderAdapter for FakeAdapter {
        fn id(&self) -> &AdapterId {
            &self.id
        }

        async fn complete(&self, call: &ResolvedCall) -> Result<Response, Error> {
            Ok(Response::new(
                call.route().provider().id().clone(),
                call.route().model().id().clone(),
                vec![ContentPart::Text {
                    text: call.route().api_model().to_owned(),
                }],
            ))
        }

        async fn stream(&self, _call: &ResolvedCall) -> Result<ResponseStream, Error> {
            Ok(Box::pin(empty()))
        }
    }

    /// Counts how many adapters it created so tests can assert that
    /// unselected providers are never constructed.
    #[derive(Clone, Default)]
    struct CountingFactory {
        created: Arc<AtomicUsize>,
    }

    impl AdapterFactory for CountingFactory {
        fn create(
            &self,
            provider: &CatalogProvider,
            _context: &AdapterContext,
        ) -> Result<Arc<dyn ProviderAdapter>, AdapterBuildError> {
            self.created.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(FakeAdapter {
                id: provider.adapter().clone(),
            }))
        }
    }

    struct FailingFactory;

    impl AdapterFactory for FailingFactory {
        fn create(
            &self,
            provider: &CatalogProvider,
            _context: &AdapterContext,
        ) -> Result<Arc<dyn ProviderAdapter>, AdapterBuildError> {
            Err(AdapterBuildError::UnsupportedCodec {
                provider: provider.id().clone(),
                codec:    provider.codec().clone(),
            })
        }
    }

    /// Always resolves to `gamma/three`, whatever the request asked for.
    struct FixedResolver;

    impl ModelResolver for FixedResolver {
        fn resolve(
            &self,
            _request: &Request,
            catalog: &Catalog,
            _available: &AvailableProviders,
        ) -> Result<ResolvedRoute, ModelSelectionError> {
            let provider = catalog
                .provider_by_id(&ProviderId::new("gamma"))
                .ok_or_else(|| ModelSelectionError::ProviderNotFound {
                    provider: "gamma".to_owned(),
                })?;
            let model =
                provider
                    .model("three")
                    .ok_or_else(|| ModelSelectionError::ModelNotFound {
                        selector: "gamma/three".to_owned(),
                    })?;
            Ok(ResolvedRoute::new(provider.clone(), model.clone()))
        }
    }

    fn catalog() -> Result<Catalog, Box<dyn StdError>> {
        Ok(Catalog::builder().overlay_toml(TEST_CATALOG)?.build()?)
    }

    fn available_ids(client: &Client) -> Vec<String> {
        client
            .available_providers()
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    #[test]
    fn a_clean_build_returns_a_client_and_no_issues() -> Result<(), Box<dyn StdError>> {
        let factory = CountingFactory::default();
        let build = Client::builder()
            .catalog(catalog()?)
            .adapter_factory("alpha-adapter", factory.clone())
            .adapter_factory("beta-adapter", factory.clone())
            .adapter_factory("gamma-adapter", factory.clone())
            .build()?;

        assert!(build.issues.is_empty());
        assert_eq!(available_ids(&build.client), ["alpha", "beta", "gamma"]);
        assert_eq!(factory.created.load(Ordering::SeqCst), 3);
        Ok(())
    }

    #[test]
    fn one_failing_provider_keeps_the_others() -> Result<(), Box<dyn StdError>> {
        let build = Client::builder()
            .catalog(catalog()?)
            .adapter_factory("alpha-adapter", CountingFactory::default())
            .adapter_factory("beta-adapter", FailingFactory)
            .adapter_factory("gamma-adapter", CountingFactory::default())
            .build()?;

        assert_eq!(available_ids(&build.client), ["alpha", "gamma"]);
        assert_eq!(build.issues.len(), 1);
        assert_eq!(build.issues[0].provider.as_str(), "beta");
        assert_eq!(build.issues[0].adapter.as_str(), "beta-adapter");
        assert!(matches!(
            build.issues[0].cause,
            ProviderBuildCause::Adapter(AdapterBuildError::UnsupportedCodec { .. })
        ));
        Ok(())
    }

    #[test]
    fn issues_come_back_in_catalog_order() -> Result<(), Box<dyn StdError>> {
        let build = Client::builder()
            .catalog(catalog()?)
            .adapter_factory("alpha-adapter", CountingFactory::default())
            .adapter_factory("gamma-adapter", FailingFactory)
            .build()?;

        assert_eq!(available_ids(&build.client), ["alpha"]);
        let issues: Vec<_> = build
            .issues
            .iter()
            .map(|issue| issue.provider.to_string())
            .collect();
        assert_eq!(issues, ["beta", "gamma"]);
        assert!(matches!(
            build.issues[0].cause,
            ProviderBuildCause::MissingAdapterFactory { .. }
        ));
        assert!(matches!(
            build.issues[1].cause,
            ProviderBuildCause::Adapter(_)
        ));
        Ok(())
    }

    #[test]
    fn every_provider_failing_yields_an_empty_client() -> Result<(), Box<dyn StdError>> {
        let build = Client::builder().catalog(catalog()?).build()?;

        assert!(available_ids(&build.client).is_empty());
        let issues: Vec<_> = build
            .issues
            .iter()
            .map(|issue| issue.provider.to_string())
            .collect();
        assert_eq!(issues, ["alpha", "beta", "gamma"]);
        assert_eq!(build.client.catalog().providers().len(), 3);
        Ok(())
    }

    #[test]
    fn no_enabled_providers_yields_an_empty_client_and_no_issues() -> Result<(), Box<dyn StdError>>
    {
        let factory = CountingFactory::default();
        let build = Client::builder()
            .catalog(catalog()?)
            .adapter_factory("alpha-adapter", factory.clone())
            .enabled_providers(Vec::<ProviderId>::new())
            .build()?;

        assert!(available_ids(&build.client).is_empty());
        assert!(build.issues.is_empty());
        assert_eq!(factory.created.load(Ordering::SeqCst), 0);
        assert_eq!(build.client.catalog().providers().len(), 3);
        Ok(())
    }

    #[test]
    fn a_missing_catalog_is_a_build_error() {
        let result = Client::builder().build();
        assert!(matches!(result, Err(ClientBuildError::MissingCatalog)));
    }

    #[test]
    fn selecting_one_provider_builds_only_that_provider() -> Result<(), Box<dyn StdError>> {
        let factory = CountingFactory::default();
        let build = Client::builder()
            .catalog(catalog()?)
            .adapter_factory("alpha-adapter", factory.clone())
            .adapter_factory("beta-adapter", factory.clone())
            .adapter_factory("gamma-adapter", factory.clone())
            .enabled_providers(["alpha"])
            .build()?;

        assert_eq!(factory.created.load(Ordering::SeqCst), 1);
        assert_eq!(available_ids(&build.client), ["alpha"]);
        assert!(build.issues.is_empty());
        assert_eq!(build.client.catalog().providers().len(), 3);
        assert!(build.client.catalog().provider("beta").is_ok());

        let request = Request::builder().model("beta/two").user("hi").build()?;
        assert!(matches!(
            build.client.resolve_route(&request),
            Err(ModelSelectionError::ProviderUnavailable { .. })
        ));
        Ok(())
    }

    #[test]
    fn a_provider_alias_is_not_an_enabled_provider() -> Result<(), Box<dyn StdError>> {
        let result = Client::builder()
            .catalog(catalog()?)
            .adapter_factory("alpha-adapter", CountingFactory::default())
            .enabled_providers(["a"])
            .build();

        match result {
            Err(ClientBuildError::UnknownEnabledProvider { provider }) => {
                assert_eq!(provider.as_str(), "a");
            }
            _ => panic!("an alias must not be accepted as an enabled provider"),
        }
        Ok(())
    }

    #[test]
    fn resolve_route_canonicalizes_aliases_defaults_and_api_models() -> Result<(), Box<dyn StdError>>
    {
        let factory = CountingFactory::default();
        let build = Client::builder()
            .catalog(catalog()?)
            .adapter_factory("alpha-adapter", factory.clone())
            .adapter_factory("beta-adapter", factory.clone())
            .build()?;
        let client = build.client;

        let request = Request::builder().model("a/uno").user("hi").build()?;
        let route = client.resolve_route(&request)?;
        assert_eq!(route.handle().to_string(), "alpha/one");
        assert_eq!(route.api_model(), "alpha-one-v1");

        let request = Request::builder().model("default").user("hi").build()?;
        let route = client.resolve_route(&request)?;
        assert_eq!(route.handle().to_string(), "alpha/one");

        let request = Request::builder()
            .model("alpha/not-in-catalog")
            .user("hi")
            .build()?;
        let route = client.resolve_route(&request)?;
        assert_eq!(route.api_model(), "not-in-catalog");
        Ok(())
    }

    #[test]
    fn resolve_route_honors_a_custom_resolver() -> Result<(), Box<dyn StdError>> {
        let build = Client::builder()
            .catalog(catalog()?)
            .resolver(FixedResolver)
            .adapter_factory("gamma-adapter", CountingFactory::default())
            .build()?;

        let request = Request::builder().model("a/uno").user("hi").build()?;
        let route = build.client.resolve_route(&request)?;
        assert_eq!(route.handle().to_string(), "gamma/three");
        Ok(())
    }

    #[tokio::test]
    async fn resolve_route_agrees_with_complete_dispatch() -> Result<(), Box<dyn StdError>> {
        let factory = CountingFactory::default();
        let build = Client::builder()
            .catalog(catalog()?)
            .adapter_factory("alpha-adapter", factory.clone())
            .adapter_factory("beta-adapter", factory.clone())
            .build()?;
        let client = build.client;

        for selector in ["default", "a/uno", "beta/two", "alpha/not-in-catalog"] {
            let request = Request::builder().model(selector).user("hi").build()?;
            let route = client.resolve_route(&request)?;
            let response = client.complete(request).await?;
            assert_eq!(response.model, route.handle());
            assert_eq!(response.content, vec![ContentPart::Text {
                text: route.api_model().to_owned(),
            }]);
        }
        Ok(())
    }
}
