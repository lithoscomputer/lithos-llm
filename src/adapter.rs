//! Public provider adapter extension points.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use thiserror::Error;

use crate::catalog::{AdapterId, CatalogProvider, CodecId, ProviderId};
use crate::credentials::CredentialProvider;
use crate::middleware::CallContext;
use crate::resolver::ResolvedRoute;
use crate::types::{Error, Request, Response, ResponseStream};

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

/// A provider-reported or locally counted input token total.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InputTokenCount(u64);

impl InputTokenCount {
    pub fn new(tokens: u64) -> Self {
        Self(tokens)
    }

    pub fn tokens(self) -> u64 {
        self.0
    }
}

/// Implements normalized complete and streaming calls for one protocol.
#[async_trait]
pub trait ProviderAdapter: Send + Sync {
    fn id(&self) -> &AdapterId;

    async fn complete(&self, call: &ResolvedCall) -> Result<Response, Error>;

    async fn stream(&self, call: &ResolvedCall) -> Result<ResponseStream, Error>;

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
    http:        reqwest::Client,
    credentials: Arc<dyn CredentialProvider>,
}

impl AdapterContext {
    pub fn new(http: reqwest::Client, credentials: Arc<dyn CredentialProvider>) -> Self {
        Self { http, credentials }
    }

    pub fn http(&self) -> &reqwest::Client {
        &self.http
    }

    pub fn credentials(&self) -> &Arc<dyn CredentialProvider> {
        &self.credentials
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

/// A catalog provider could not create its runtime adapter.
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
}
