//! Client construction and inference behavior.

mod probe;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

pub use probe::{ProbeOptions, ProbeOutcome, ProbeReport};
use thiserror::Error;

use crate::adapter::{
    AdapterBuildError, AdapterContext, AdapterFactory, AdapterRegistry,
    DEFAULT_STREAM_IDLE_TIMEOUT, InputTokenCount, ProviderAdapter,
};
use crate::catalog::{AdapterId, Catalog, CatalogError, CatalogProvider, ProviderId, adapter_ids};
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
use crate::credentials::ConventionalCredentials;
use crate::credentials::{self, CredentialError, CredentialProvider, NoCredentials};
use crate::middleware::{Call, CallContext, CallGuard, Middleware, Operation, Output, Pipeline};
use crate::providers::register_builtin;
use crate::resolver::{
    AvailableProviders, CatalogResolver, ModelResolver, ModelSelectionError, ResolvedRoute,
};
use crate::types::{
    Error, ErrorKind, Request, Response, ResponseLimits, ResponsePolicy, ResponseStream,
};

/// How long the default HTTP client waits to establish a connection.
///
/// A provider whose endpoint accepts no connection fails here rather than
/// waiting for the operating system's own limit.
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// An immutable provider-neutral client.
#[derive(Clone)]
pub struct Client {
    default_timeout: Option<Duration>,
    catalog:         Catalog,
    resolver:        Arc<dyn ModelResolver>,
    available:       AvailableProviders,
    pipeline:        Arc<Pipeline>,
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
            .credentials(ConventionalCredentials::new())
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
        let context = self.prepare_context(&request, context)?;
        match self
            .call_guarded(request, context, Operation::Complete)
            .await?
        {
            Output::Complete(response) => Ok(response),
            _ => Err(Error::new(
                ErrorKind::Middleware,
                "complete middleware returned an incompatible output",
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
        let context = self.prepare_context(&request, context)?;
        let guard = CallGuard::new(&context);
        match self
            .call_guarded(request, context, Operation::Stream)
            .await?
        {
            Output::Stream(stream) => Ok(guard.stream(stream)),
            _ => Err(Error::new(
                ErrorKind::Middleware,
                "stream middleware returned an incompatible output",
            )),
        }
    }

    pub async fn count_input_tokens(
        &self,
        request: Request,
    ) -> Result<Option<InputTokenCount>, Error> {
        self.count_input_tokens_with_context(request, CallContext::new())
            .await
    }

    /// Counts tokens through the same deadlines, cancellation and middleware
    /// as inference. None means that the adapter has no native count endpoint.
    pub async fn count_input_tokens_with_context(
        &self,
        request: Request,
        context: CallContext,
    ) -> Result<Option<InputTokenCount>, Error> {
        let context = self.prepare_context(&request, context)?;
        match self
            .call_guarded(request, context, Operation::CountInputTokens)
            .await?
        {
            Output::InputTokenCount(count) => Ok(count),
            _ => Err(Error::new(
                ErrorKind::Middleware,
                "token count middleware returned an incompatible output",
            )),
        }
    }

    fn prepare_context(
        &self,
        request: &Request,
        mut context: CallContext,
    ) -> Result<CallContext, Error> {
        if let Some(timeout) = request.timeout().or(self.default_timeout) {
            let guard = CallGuard::new(&context).with_timeout(timeout)?;
            if let Some(deadline) = guard.deadline() {
                context.set_deadline(deadline);
            }
        }
        Ok(context)
    }

    async fn call(
        &self,
        request: Request,
        context: CallContext,
        mode: Operation,
    ) -> Result<Output, Error> {
        let route = self.resolve_route(&request)?;
        route.validate_request(&request)?;
        let output = self
            .pipeline
            .start()
            .run(Call {
                request,
                route,
                mode,
                context,
            })
            .await?;
        // Middleware can return cached responses or transform adapter output.
        // Apply the policy again at the client boundary so those paths obey
        // the same output limits and raw-body retention setting.
        match output {
            Output::Complete(response) => self
                .pipeline
                .policy
                .response(response)
                .map(Output::Complete),
            Output::Stream(stream) => Ok(Output::Stream(self.pipeline.policy.stream(stream))),
            Output::InputTokenCount(count) => Ok(Output::InputTokenCount(count)),
        }
    }

    async fn call_guarded(
        &self,
        request: Request,
        context: CallContext,
        mode: Operation,
    ) -> Result<Output, Error> {
        CallGuard::new(&context)
            .run(Box::pin(self.call(request, context, mode)))
            .await
    }
}

/// Builds a client from immutable catalog data and runtime extensions.
#[must_use]
pub struct ClientBuilder {
    policy:              ResponsePolicy,
    default_timeout:     Option<Duration>,
    catalog:             Option<Catalog>,
    resolver:            Arc<dyn ModelResolver>,
    credentials:         Arc<dyn CredentialProvider>,
    http:                Option<reqwest::Client>,
    connect_timeout:     Option<Duration>,
    stream_idle_timeout: Option<Duration>,
    middleware:          Vec<Arc<dyn Middleware>>,
    registry:            AdapterRegistry,
    enabled:             Option<BTreeSet<ProviderId>>,
    application:         Option<String>,
}

impl Default for ClientBuilder {
    fn default() -> Self {
        let mut registry = AdapterRegistry::new();
        register_builtin(&mut registry);
        Self {
            default_timeout: None,
            policy: ResponsePolicy::default(),
            catalog: None,
            resolver: Arc::new(CatalogResolver),
            credentials: Arc::new(NoCredentials),
            http: None,
            connect_timeout: Some(DEFAULT_CONNECT_TIMEOUT),
            stream_idle_timeout: Some(DEFAULT_STREAM_IDLE_TIMEOUT),
            middleware: Vec::new(),
            registry,
            enabled: None,
            application: None,
        }
    }
}

impl ClientBuilder {
    /// Sets body and frame limits for built-in adapters, and normalized-output
    /// limits for all responses, including middleware output and cache hits.
    /// Custom adapters also receive these limits through AdapterContext.
    pub fn response_limits(mut self, limits: ResponseLimits) -> Self {
        self.policy.limits = limits;
        self
    }

    /// Retains raw provider bodies on successful responses. Defaults to true.
    /// Disabling this does not redact content, replay metadata, or errors.
    pub fn retain_raw_response(mut self, retain: bool) -> Self {
        self.policy.retain_raw = retain;
        self
    }

    /// Sets the total budget for a call, including middleware, credentials,
    /// retries, and stream consumption. A request timeout overrides this
    /// default; an earlier context deadline still wins. Unset means no budget.
    pub fn default_timeout(mut self, timeout: Duration) -> Self {
        self.default_timeout = Some(timeout);
        self
    }

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

    /// Names the application to providers that expect a client to identify
    /// itself.
    ///
    /// The OpenAI Codex deployment expects every client to send an
    /// `originator` header naming the calling application; its adapter sends
    /// this value there. Other built-in adapters ignore it. Custom adapters
    /// read it through [`AdapterContext::application`].
    pub fn application(mut self, name: impl Into<String>) -> Self {
        self.application = Some(name.into());
        self
    }

    /// Injects an application-configured HTTP client.
    ///
    /// Every built-in adapter uses this client. Without it the builder creates
    /// a default client that sends the Lithos user agent and applies
    /// [`connect_timeout`](Self::connect_timeout). An injected client carries
    /// its own connect timeout, so this builder does not change it.
    pub fn http(mut self, client: reqwest::Client) -> Self {
        self.http = Some(client);
        self
    }

    /// Replaces the connect timeout of the default HTTP client.
    ///
    /// The default is 30 seconds. `None` waits as long as the operating
    /// system allows, which can be minutes.
    ///
    /// This applies only to the client this builder creates. A client passed
    /// to [`http`](Self::http) keeps the timeouts it was built with.
    pub fn connect_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.connect_timeout = timeout;
        self
    }

    /// Replaces the longest a response stream may stall between two chunks.
    ///
    /// The default is 300 seconds. A provider that stops sending mid-stream
    /// then fails with a retryable timeout instead of hanging. `None` waits
    /// forever, which suits an application that bounds the call some other
    /// way, such as
    /// [`ClientBuilder::default_timeout`](crate::ClientBuilder::default_timeout).
    ///
    /// This reaches every built-in adapter through
    /// [`AdapterContext::stream_idle_timeout`].
    pub fn stream_idle_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.stream_idle_timeout = timeout;
        self
    }

    /// Builds adapters only for the named providers.
    ///
    /// Ids must be canonical [`ProviderId`] values; provider aliases are not
    /// accepted. The last call replaces any earlier selection. Without a
    /// selection the builder attempts every enabled provider in the catalog.
    ///
    /// This selection narrows the catalog and never widens it: a provider the
    /// catalog marks `enabled = false` builds no adapter even when named
    /// here. Turn such a provider on with a catalog overlay instead.
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
    /// Each enabled provider is attempted in catalog order; a provider the
    /// catalog disables is skipped without an issue. Successful
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
        self.build_with_credential_issues(Vec::new())
    }

    /// Builds a client over the providers whose credentials resolve right now.
    ///
    /// Resolves credentials once for every provider in the builder's
    /// selection (the [`enabled_providers`](Self::enabled_providers) set when
    /// one was given, else every enabled provider in the catalog) and builds
    /// adapters only for the ones that resolved. The selection is narrowed,
    /// never widened: a provider left out of `enabled_providers` stays out
    /// however good its credentials are. A provider with an explicit
    /// [`adapter`](Self::adapter) counts as ready without a lookup, because
    /// that adapter owns its own authentication.
    ///
    /// Providers the store holds nothing for are left out silently. Providers
    /// with material that cannot be used come back in
    /// [`ClientBuild::credential_issues`]. Providers that resolved but whose
    /// adapter could not be constructed come back in
    /// [`ClientBuild::issues`], as they do from [`build`](Self::build).
    ///
    /// # Errors
    ///
    /// The same failures as [`build`](Self::build).
    pub async fn build_ready(mut self) -> Result<ClientBuild, ClientBuildError> {
        let catalog = self
            .catalog
            .as_ref()
            .ok_or(ClientBuildError::MissingCatalog)?;
        let selection: Vec<&CatalogProvider> = catalog
            .providers()
            .filter(|provider| provider.is_enabled())
            .filter(|provider| {
                self.enabled
                    .as_ref()
                    .is_none_or(|enabled| enabled.contains(provider.id()))
            })
            .collect();
        let (with_adapter, needs_credentials): (Vec<_>, Vec<_>) = selection
            .into_iter()
            .partition(|provider| self.registry.explicit(provider.id()).is_some());
        let readiness = credentials::readiness(needs_credentials, self.credentials.as_ref()).await;
        let ready: BTreeSet<ProviderId> = with_adapter
            .into_iter()
            .map(|provider| provider.id().clone())
            .chain(readiness.ready)
            .collect();
        self.enabled = Some(ready);
        self.build_with_credential_issues(readiness.issues)
    }

    fn build_with_credential_issues(
        self,
        credential_issues: Vec<(ProviderId, CredentialError)>,
    ) -> Result<ClientBuild, ClientBuildError> {
        let catalog = self.catalog.ok_or(ClientBuildError::MissingCatalog)?;
        let http = if let Some(http) = self.http {
            http
        } else {
            let mut builder = reqwest::Client::builder()
                .user_agent(concat!("lithos-llm/", env!("CARGO_PKG_VERSION")));
            if let Some(connect_timeout) = self.connect_timeout {
                builder = builder.connect_timeout(connect_timeout);
            }
            builder.build().map_err(ClientBuildError::HttpClient)?
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
        let context = AdapterContext::new(http, self.credentials)
            .with_stream_idle_timeout(self.stream_idle_timeout)
            .with_response_limits(self.policy.limits)
            .with_retain_raw_response(self.policy.retain_raw)
            .with_application(self.application);
        let mut adapters = BTreeMap::new();
        let mut issues = Vec::new();
        let mut ready = Vec::new();
        for provider in catalog.providers() {
            // A catalog-disabled provider builds no adapter, whether or not
            // the application named it. `enabled_providers` narrows the
            // catalog; it does not override the catalog.
            if !provider.is_enabled()
                || self
                    .enabled
                    .as_ref()
                    .is_some_and(|enabled| !enabled.contains(provider.id()))
            {
                continue;
            }
            ready.push(provider.id().clone());
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
            ready,
            credential_issues,
            client: Client {
                default_timeout: self.default_timeout,
                catalog,
                resolver: self.resolver,
                available,
                pipeline: Arc::new(Pipeline {
                    policy: self.policy,
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
    pub client:            Client,
    /// The providers the build attempted, in catalog order: the builder's
    /// selection, narrowed to credential-ready providers by
    /// [`ClientBuilder::build_ready`]. A provider here that also appears in
    /// `issues` has credentials but no adapter, so
    /// [`Client::available_providers`] is this list minus `issues`.
    pub ready:             Vec<ProviderId>,
    /// Providers [`ClientBuilder::build_ready`] left out because their stored
    /// credential material could not be used. Always empty after
    /// [`ClientBuilder::build`], which reads no credentials.
    pub credential_issues: Vec<(ProviderId, CredentialError)>,
    /// Providers that could not be constructed.
    pub issues:            Vec<ProviderBuildIssue>,
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
    use serde_json::json;

    use super::{Client, ClientBuildError, ProviderBuildCause};
    use crate::adapter::{
        AdapterBuildError, AdapterContext, AdapterFactory, ProviderAdapter, ResolvedCall,
    };
    use crate::catalog::{AdapterId, Catalog, CatalogProvider, ProviderId};
    use crate::credentials::{CredentialError, CredentialProvider, Credentials, StaticCredentials};
    use crate::resolver::{AvailableProviders, ModelResolver, ModelSelectionError, ResolvedRoute};
    use crate::types::{
        ContentPart, Error, Request, Response, ResponseStream, Speed, ToolDefinition,
    };

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
        # Prices a fast tier, so the model declares fast speed support.
        pricing = { input_usd_micros_per_million = 100, output_usd_micros_per_million = 200, speed = { fast = { input_usd_micros_per_million = 150 } } }

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
        capabilities = { text = true, speed = { fast = false, economical = false } }
        # This model explicitly supports only the balanced speed.
        pricing = { input_usd_micros_per_million = 100, output_usd_micros_per_million = 200 }

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
            Ok(ResponseStream::new(empty()))
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
            ResolvedRoute::try_new(provider.clone(), model.clone())
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
    fn a_catalog_disabled_provider_builds_no_adapter() -> Result<(), Box<dyn StdError>> {
        let catalog = Catalog::builder()
            .overlay_toml(TEST_CATALOG)?
            .overlay_toml("[providers.beta]\nenabled = false\n")?
            .build()?;
        let factory = CountingFactory::default();
        let build = Client::builder()
            .catalog(catalog.clone())
            .adapter_factory("alpha-adapter", factory.clone())
            .adapter_factory("beta-adapter", factory.clone())
            .adapter_factory("gamma-adapter", factory.clone())
            .build()?;

        assert!(build.issues.is_empty(), "disabled is not an issue");
        assert_eq!(available_ids(&build.client), ["alpha", "gamma"]);
        assert_eq!(factory.created.load(Ordering::SeqCst), 2);

        // Naming it in the allow-list does not resurrect it.
        let build = Client::builder()
            .catalog(catalog)
            .adapter_factory("beta-adapter", CountingFactory::default())
            .enabled_providers(["beta"])
            .build()?;
        assert!(build.issues.is_empty());
        assert!(available_ids(&build.client).is_empty());
        Ok(())
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
    async fn a_passthrough_route_skips_capability_validation() -> Result<(), Box<dyn StdError>> {
        let build = Client::builder()
            .catalog(catalog()?)
            .adapter_factory("alpha-adapter", CountingFactory::default())
            .build()?;
        let client = build.client;

        // The catalog cannot describe a passthrough model, so a request using
        // tools and sampling must reach the provider instead of failing here.
        let request = Request::builder()
            .model("alpha/not-in-catalog")
            .user("hi")
            .tool(ToolDefinition::function(
                "patch",
                "apply a patch",
                json!({}),
            ))
            .temperature(0.7)
            .build()?;
        let response = client.complete(request).await?;
        assert_eq!(response.content, vec![ContentPart::Text {
            text: "not-in-catalog".to_owned(),
        }]);

        // A cataloged model keeps its declared capability limits.
        let request = Request::builder()
            .model("alpha/one")
            .user("hi")
            .tool(ToolDefinition::function(
                "patch",
                "apply a patch",
                json!({}),
            ))
            .build()?;
        let error = client
            .complete(request)
            .await
            .expect_err("a cataloged model without tools must reject them");
        assert!(error.to_string().contains("does not support tools"));
        Ok(())
    }

    #[tokio::test]
    async fn sampling_controls_on_a_pinned_sampling_model_are_refused()
    -> Result<(), Box<dyn StdError>> {
        // The reference client silently stripped temperature and top_p when
        // the catalog said the model pins its sampling; this crate refuses
        // before dispatch instead. The cutover review names this the refusal
        // most likely to fire on migrated traffic, so the gate is pinned.
        let build = Client::builder()
            .catalog(catalog()?)
            .adapter_factory("alpha-adapter", CountingFactory::default())
            .build()?;

        let request = Request::builder()
            .model("alpha/one")
            .user("hi")
            .temperature(0.7)
            .build()?;
        let error = build
            .client
            .complete(request)
            .await
            .expect_err("a pinned-sampling model must refuse sampling controls");
        assert!(error.to_string().contains("sampling"), "{error}");
        Ok(())
    }

    #[tokio::test]
    async fn an_undeclared_speed_is_rejected_before_dispatch() -> Result<(), Box<dyn StdError>> {
        let build = Client::builder()
            .catalog(catalog()?)
            .adapter_factory("beta-adapter", CountingFactory::default())
            .build()?;
        let request = Request::builder()
            .model("beta/two")
            .user("hi")
            .speed(Speed::Fast)
            .build()?;

        // The fake adapter would answer successfully, so an error proves the
        // request never dispatched.
        let error = build
            .client
            .complete(request.clone())
            .await
            .expect_err("a priced model with no fast tier must reject fast");
        assert!(
            error
                .to_string()
                .contains("model beta/two does not support speed 'fast'"),
            "{error}"
        );

        let Err(error) = build.client.stream(request).await else {
            panic!("streaming applies the same speed gate");
        };
        assert!(error.to_string().contains("speed 'fast'"), "{error}");
        Ok(())
    }

    #[tokio::test]
    async fn a_priced_speed_tier_passes_validation() -> Result<(), Box<dyn StdError>> {
        let build = Client::builder()
            .catalog(catalog()?)
            .adapter_factory("alpha-adapter", CountingFactory::default())
            .build()?;

        // alpha/one prices a fast tier, so the catalog declares the speed.
        let request = Request::builder()
            .model("alpha/one")
            .user("hi")
            .speed(Speed::Fast)
            .build()?;
        build.client.complete(request).await?;
        Ok(())
    }

    #[tokio::test]
    async fn balanced_needs_no_speed_declaration() -> Result<(), Box<dyn StdError>> {
        let build = Client::builder()
            .catalog(catalog()?)
            .adapter_factory("beta-adapter", CountingFactory::default())
            .build()?;

        // `Balanced` is the default tier, so a priced model that declares no
        // speed tiers still takes it.
        let request = Request::builder()
            .model("beta/two")
            .user("hi")
            .speed(Speed::Balanced)
            .build()?;
        build.client.complete(request).await?;
        Ok(())
    }

    #[tokio::test]
    async fn a_passthrough_model_may_request_any_speed() -> Result<(), Box<dyn StdError>> {
        let build = Client::builder()
            .catalog(catalog()?)
            .adapter_factory("alpha-adapter", CountingFactory::default())
            .build()?;

        // A passthrough model carries no pricing, so its speeds are unknown
        // rather than absent, and the provider judges the request.
        for speed in [Speed::Fast, Speed::Balanced, Speed::Economical] {
            let request = Request::builder()
                .model("alpha/not-in-catalog")
                .user("hi")
                .speed(speed)
                .build()?;
            build.client.complete(request).await?;
        }
        Ok(())
    }

    #[tokio::test]
    async fn an_unpriced_model_leaves_speed_to_the_provider() -> Result<(), Box<dyn StdError>> {
        let build = Client::builder()
            .catalog(catalog()?)
            .adapter_factory("gamma-adapter", CountingFactory::default())
            .build()?;

        // gamma/three declares no pricing at all, so the catalog says nothing
        // about its speeds.
        let request = Request::builder()
            .model("gamma/three")
            .user("hi")
            .speed(Speed::Economical)
            .build()?;
        build.client.complete(request).await?;
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

    /// Credentials for exactly the named providers; every other lookup says
    /// the store holds nothing.
    fn credentials_for(providers: &[&str]) -> StaticCredentials {
        providers
            .iter()
            .fold(StaticCredentials::new(), |store, id| {
                store.with(*id, Credentials::none())
            })
    }

    /// A store whose material for one provider is present but unusable.
    struct UnusableFor(&'static str);

    #[async_trait]
    impl CredentialProvider for UnusableFor {
        async fn credentials(
            &self,
            provider: &CatalogProvider,
        ) -> Result<Credentials, CredentialError> {
            if provider.id().as_str() == self.0 {
                Err(CredentialError::Unusable {
                    provider: provider.id().clone(),
                    reason:   "the stored token has expired and has no refresh token".to_owned(),
                    source:   None,
                })
            } else {
                Ok(Credentials::none())
            }
        }
    }

    #[tokio::test]
    async fn build_ready_keeps_only_credentialed_providers() -> Result<(), Box<dyn StdError>> {
        let factory = CountingFactory::default();
        let build = Client::builder()
            .catalog(catalog()?)
            .credentials(credentials_for(&["alpha", "gamma"]))
            .adapter_factory("alpha-adapter", factory.clone())
            .adapter_factory("beta-adapter", factory.clone())
            .adapter_factory("gamma-adapter", factory.clone())
            .build_ready()
            .await?;

        assert_eq!(available_ids(&build.client), ["alpha", "gamma"]);
        assert_eq!(build.ready, ["alpha", "gamma"].map(ProviderId::new));
        assert!(
            build.credential_issues.is_empty(),
            "unconfigured is silence"
        );
        assert!(build.issues.is_empty());
        assert_eq!(
            factory.created.load(Ordering::SeqCst),
            2,
            "no adapter is built for a provider without credentials"
        );
        Ok(())
    }

    #[tokio::test]
    async fn build_ready_never_widens_the_enabled_selection() -> Result<(), Box<dyn StdError>> {
        let build = Client::builder()
            .catalog(catalog()?)
            .credentials(credentials_for(&["alpha", "beta", "gamma"]))
            .enabled_providers(["alpha"])
            .adapter_factory("alpha-adapter", CountingFactory::default())
            .adapter_factory("beta-adapter", CountingFactory::default())
            .adapter_factory("gamma-adapter", CountingFactory::default())
            .build_ready()
            .await?;

        assert_eq!(available_ids(&build.client), ["alpha"]);
        assert_eq!(build.ready, [ProviderId::new("alpha")]);
        Ok(())
    }

    #[tokio::test]
    async fn build_ready_counts_an_explicit_adapter_as_ready_without_credentials()
    -> Result<(), Box<dyn StdError>> {
        let build = Client::builder()
            .catalog(catalog()?)
            .credentials(credentials_for(&[]))
            .adapter("beta", FakeAdapter {
                id: AdapterId::new("beta-adapter"),
            })
            .adapter_factory("alpha-adapter", CountingFactory::default())
            .build_ready()
            .await?;

        assert_eq!(available_ids(&build.client), ["beta"]);
        assert_eq!(build.ready, [ProviderId::new("beta")]);
        assert!(build.credential_issues.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn build_ready_separates_credential_issues_from_build_issues()
    -> Result<(), Box<dyn StdError>> {
        let build = Client::builder()
            .catalog(catalog()?)
            .credentials(UnusableFor("beta"))
            .adapter_factory("alpha-adapter", CountingFactory::default())
            .adapter_factory("beta-adapter", CountingFactory::default())
            .adapter_factory("gamma-adapter", FailingFactory)
            .build_ready()
            .await?;

        assert_eq!(available_ids(&build.client), ["alpha"]);
        assert_eq!(build.ready, ["alpha", "gamma"].map(ProviderId::new));
        assert_eq!(build.credential_issues.len(), 1);
        assert_eq!(build.credential_issues[0].0.as_str(), "beta");
        assert!(matches!(
            build.credential_issues[0].1,
            CredentialError::Unusable { .. }
        ));
        assert_eq!(build.issues.len(), 1);
        assert_eq!(build.issues[0].provider.as_str(), "gamma");
        Ok(())
    }

    #[test]
    fn build_reads_no_credentials_and_reports_the_attempted_set() -> Result<(), Box<dyn StdError>> {
        let build = Client::builder()
            .catalog(catalog()?)
            .credentials(UnusableFor("beta"))
            .adapter_factory("alpha-adapter", CountingFactory::default())
            .adapter_factory("beta-adapter", CountingFactory::default())
            .adapter_factory("gamma-adapter", CountingFactory::default())
            .build()?;

        assert_eq!(build.ready, ["alpha", "beta", "gamma"].map(ProviderId::new));
        assert!(build.credential_issues.is_empty());
        Ok(())
    }
}
