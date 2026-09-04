//! Public provider adapter extension points.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use thiserror::Error;

use crate::catalog::{AdapterId, CatalogProvider, CodecId, ModelHandle, ProviderId};
use crate::credentials::CredentialProvider;
use crate::middleware::CallContext;
use crate::resolver::ResolvedRoute;
use crate::types::{Error, Request, Response, ResponseLimits, ResponsePolicy, ResponseStream};

/// How long a response stream may stall between two chunks by default.
///
/// A provider that stops sending bytes mid-generation would otherwise hold a
/// call open forever, because nothing in the HTTP layer bounds a read.
pub(crate) const DEFAULT_STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(300);

/// A request with a fixed provider and model route.
#[derive(Clone, Debug)]
pub struct ResolvedCall {
    request: Request,
    route:   ResolvedRoute,
    context: CallContext,
}

impl ResolvedCall {
    pub fn new(request: Request, route: ResolvedRoute, context: CallContext) -> Self {
        Self {
            request,
            route,
            context,
        }
    }

    pub fn request(&self) -> &Request {
        &self.request
    }

    pub fn route(&self) -> &ResolvedRoute {
        &self.route
    }

    pub fn context(&self) -> &CallContext {
        &self.context
    }
}

/// A provider-authoritative input token count for a resolved route.
///
/// This crate never estimates. A value of this type always came from a
/// provider count endpoint, and it carries the canonical model the provider
/// counted for, which is the catalog identity rather than any alias the
/// request used.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InputTokenCount {
    tokens: u64,
    model:  ModelHandle,
}

impl InputTokenCount {
    pub fn new(tokens: u64, model: ModelHandle) -> Self {
        Self { tokens, model }
    }

    pub fn tokens(&self) -> u64 {
        self.tokens
    }

    /// The canonical model this count was produced for.
    pub fn model(&self) -> &ModelHandle {
        &self.model
    }
}

/// Implements normalized complete and streaming calls for one protocol.
#[async_trait]
pub trait ProviderAdapter: Send + Sync {
    fn id(&self) -> &AdapterId;

    async fn complete(&self, call: &ResolvedCall) -> Result<Response, Error>;

    async fn stream(&self, call: &ResolvedCall) -> Result<ResponseStream, Error>;

    /// Counts the input tokens this call would send, using the provider.
    ///
    /// `Some` is a provider-authoritative count. `None` means this adapter has
    /// no native count endpoint; it never means the count was guessed. This
    /// crate does not estimate token counts locally, so a caller that wants an
    /// estimate for a `None` adapter supplies its own.
    ///
    /// # Errors
    ///
    /// Returns the normal classified provider error when a native count
    /// request fails. A failure is never reported as `None`.
    async fn count_input_tokens(
        &self,
        _call: &ResolvedCall,
    ) -> Result<Option<InputTokenCount>, Error> {
        Ok(None)
    }
}

/// Dependencies available when a catalog provider creates an adapter.
#[derive(Clone)]
pub struct AdapterContext {
    policy:              ResponsePolicy,
    http:                reqwest::Client,
    credentials:         Arc<dyn CredentialProvider>,
    stream_idle_timeout: Option<Duration>,
}

impl AdapterContext {
    /// Builds a context with the default stream-idle timeout.
    pub fn new(http: reqwest::Client, credentials: Arc<dyn CredentialProvider>) -> Self {
        Self {
            http,
            policy: ResponsePolicy::default(),
            credentials,
            stream_idle_timeout: Some(DEFAULT_STREAM_IDLE_TIMEOUT),
        }
    }

    /// Replaces the longest a stream may stall between two chunks.
    ///
    /// `None` waits forever, which only an application that bounds the call
    /// some other way should choose.
    #[must_use]
    pub fn with_stream_idle_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.stream_idle_timeout = timeout;
        self
    }

    #[must_use]
    pub fn with_response_limits(mut self, limits: ResponseLimits) -> Self {
        self.policy.limits = limits;
        self
    }

    #[must_use]
    pub fn with_retain_raw_response(mut self, retain: bool) -> Self {
        self.policy.retain_raw = retain;
        self
    }

    pub fn response_limits(&self) -> ResponseLimits {
        self.policy.limits
    }

    pub fn retain_raw_response(&self) -> bool {
        self.policy.retain_raw
    }

    #[cfg(any(
        feature = "openai",
        feature = "anthropic",
        feature = "gemini",
        feature = "openai-compatible"
    ))]
    pub(crate) fn response_policy(&self) -> ResponsePolicy {
        self.policy
    }

    pub fn http(&self) -> &reqwest::Client {
        &self.http
    }

    pub fn credentials(&self) -> &Arc<dyn CredentialProvider> {
        &self.credentials
    }

    /// The longest a stream may stall between two chunks.
    pub fn stream_idle_timeout(&self) -> Option<Duration> {
        self.stream_idle_timeout
    }
}

/// Creates an adapter for a catalog provider that names its adapter ID.
pub trait AdapterFactory: Send + Sync {
    fn create(
        &self,
        provider: &CatalogProvider,
        context: &AdapterContext,
    ) -> Result<Arc<dyn ProviderAdapter>, AdapterBuildError>;
}

/// Runtime registrations for factories and provider-specific instances.
#[derive(Default)]
pub struct AdapterRegistry {
    factories: BTreeMap<AdapterId, Arc<dyn AdapterFactory>>,
    adapters:  BTreeMap<ProviderId, Arc<dyn ProviderAdapter>>,
}

impl AdapterRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_factory(
        &mut self,
        id: impl Into<AdapterId>,
        factory: impl AdapterFactory + 'static,
    ) {
        self.factories.insert(id.into(), Arc::new(factory));
    }

    pub fn register_factory_arc(
        &mut self,
        id: impl Into<AdapterId>,
        factory: Arc<dyn AdapterFactory>,
    ) {
        self.factories.insert(id.into(), factory);
    }

    pub fn register_adapter(
        &mut self,
        provider: impl Into<ProviderId>,
        adapter: impl ProviderAdapter + 'static,
    ) {
        self.adapters.insert(provider.into(), Arc::new(adapter));
    }

    pub fn register_adapter_arc(
        &mut self,
        provider: impl Into<ProviderId>,
        adapter: Arc<dyn ProviderAdapter>,
    ) {
        self.adapters.insert(provider.into(), adapter);
    }

    pub(crate) fn explicit(&self, provider: &ProviderId) -> Option<Arc<dyn ProviderAdapter>> {
        self.adapters.get(provider).cloned()
    }

    pub(crate) fn factory(&self, id: &AdapterId) -> Option<Arc<dyn AdapterFactory>> {
        self.factories.get(id).cloned()
    }
}

/// A provider-local reason one adapter could not be constructed.
///
/// Each variant names one catalog provider. A build failure here affects only
/// that provider; the client keeps every adapter that was built successfully.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AdapterBuildError {
    #[error("provider {provider} uses unsupported codec {codec}")]
    UnsupportedCodec {
        provider: ProviderId,
        codec:    CodecId,
    },
    #[error("provider {provider} has invalid adapter configuration: {message}")]
    InvalidConfiguration {
        provider: ProviderId,
        message:  String,
    },
    /// The provider's `adapter_options` catalog table did not match the typed
    /// shape the adapter factory expects.
    #[error("provider {provider} has invalid adapter options")]
    InvalidAdapterOptions {
        provider: ProviderId,
        #[source]
        source:   serde_json::Error,
    },
}
