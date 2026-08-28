//! Client construction and inference behavior.

use std::collections::BTreeMap;
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
use crate::resolver::{AvailableProviders, CatalogResolver, ModelResolver, ResolvedRoute};
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

impl Client {
    pub fn builder() -> ClientBuilder {
        ClientBuilder::new()
    }

    /// Builds a client with the built-in catalog and conventional environment
    /// variable names.
    ///
    /// Credentials are resolved for each provider attempt. This constructor
    /// does not require credentials to be present when it builds the client.
    ///
    /// # Errors
    ///
    /// Returns an error if the built-in catalog is invalid, the default HTTP
    /// client cannot be built, or no enabled provider can create an adapter.
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
    pub fn from_env() -> Result<Self, ClientBuildError> {
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

    pub fn available_providers(&self) -> &AvailableProviders {
        &self.available
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
        let route = self
            .resolver
            .resolve(&request, &self.catalog, &self.available)?;
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
        let route = self
            .resolver
            .resolve(&request, &self.catalog, &self.available)?;
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

    pub fn http_client(mut self, client: reqwest::Client) -> Self {
        self.http = Some(client);
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

    pub fn build(self) -> Result<Client, ClientBuildError> {
        let catalog = self.catalog.ok_or(ClientBuildError::MissingCatalog)?;
        let http = match self.http {
            Some(http) => http,
            None => reqwest::Client::builder()
                .user_agent(concat!("lithos-llm/", env!("CARGO_PKG_VERSION")))
                .build()
                .map_err(ClientBuildError::HttpClient)?,
        };
        let context = AdapterContext::new(http, self.credentials);
        let mut adapters = BTreeMap::new();
        for provider in catalog.providers() {
            let adapter = if let Some(adapter) = self.registry.explicit(provider.id()) {
                Some(adapter)
            } else if let Some(factory) = self.registry.factory(provider.adapter()) {
                Some(factory.create(provider, &context).map_err(|source| {
                    ClientBuildError::Adapter {
                        provider: provider.id().clone(),
                        source,
                    }
                })?)
            } else if is_disabled_builtin_adapter(provider.adapter()) {
                None
            } else {
                return Err(ClientBuildError::MissingAdapterFactory {
                    provider: provider.id().clone(),
                    adapter:  provider.adapter().clone(),
                });
            };
            if let Some(adapter) = adapter {
                adapters.insert(provider.id().clone(), adapter);
            }
        }
        if adapters.is_empty() {
            return Err(ClientBuildError::NoAdapters);
        }
        let available = AvailableProviders::new(adapters.keys().cloned());
        Ok(Client {
            catalog,
            resolver: self.resolver,
            available,
            pipeline: Arc::new(Pipeline {
                middleware: self.middleware,
                adapters,
            }),
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

/// Client construction failed before any request ran.
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
    #[error("no catalog provider has a registered adapter")]
    NoAdapters,
    #[error("provider {provider} uses unregistered adapter {adapter}")]
    MissingAdapterFactory {
        provider: ProviderId,
        adapter:  AdapterId,
    },
    #[error("the default HTTP client could not be built")]
    HttpClient(#[source] reqwest::Error),
    #[error("adapter construction failed for provider {provider}")]
    Adapter {
        provider: ProviderId,
        #[source]
        source:   AdapterBuildError,
    },
}
