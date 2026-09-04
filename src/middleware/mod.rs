//! Runtime-installed middleware for one resolved logical call.

mod concurrency;
mod observer;
mod retry;
mod tracing_layer;

use std::any::{Any, TypeId};
use std::collections::BTreeMap;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};
use std::time::Instant;

use async_trait::async_trait;
pub use concurrency::ConcurrencyLimitMiddleware;
use futures_core::Stream;
pub use observer::{CallOutcome, Observer, ObserverMiddleware, RetryStage};
pub use retry::{RetryMiddleware, RetryPolicy};
use tokio::sync::Notify;
pub use tracing_layer::TracingMiddleware;

use crate::adapter::{InputTokenCount, ProviderAdapter, ResolvedCall};
use crate::catalog::ProviderId;
use crate::resolver::ResolvedRoute;
use crate::types::{
    Error, ErrorKind, Request, RequestBuildError, Response, ResponseStream, StreamEvent,
};

/// The operation executed through the middleware pipeline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Operation {
    Complete,
    Stream,
    CountInputTokens,
}

/// A clone-safe application cancellation signal.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    state: Arc<CancellationState>,
}

#[derive(Debug, Default)]
struct CancellationState {
    cancelled: AtomicBool,
    notify:    Notify,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.state.cancelled.store(true, Ordering::Release);
        self.state.notify.notify_waiters();
    }

    pub fn is_cancelled(&self) -> bool {
        self.state.cancelled.load(Ordering::Acquire)
    }

    /// Waits until this token is cancelled.
    pub async fn cancelled(&self) {
        loop {
            if self.is_cancelled() {
                return;
            }
            let notified = self.state.notify.notified();
            if self.is_cancelled() {
                return;
            }
            notified.await;
        }
    }
}

/// A clone-safe typed map for application runtime data.
#[derive(Clone, Default)]
pub struct Extensions {
    values: BTreeMap<TypeId, Arc<dyn Any + Send + Sync>>,
}

impl Extensions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert<T: Send + Sync + 'static>(&mut self, value: T) -> Option<Arc<T>> {
        self.values
            .insert(TypeId::of::<T>(), Arc::new(value))
            .and_then(|previous| previous.downcast::<T>().ok())
    }

    pub fn get<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.values
            .get(&TypeId::of::<T>())
            .and_then(|value| value.downcast_ref())
    }

    pub fn remove<T: Send + Sync + 'static>(&mut self) -> Option<Arc<T>> {
        self.values
            .remove(&TypeId::of::<T>())
            .and_then(|value| value.downcast::<T>().ok())
    }
}

impl fmt::Debug for Extensions {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Extensions")
            .field("len", &self.values.len())
            .finish()
    }
}

/// Runtime context shared by middleware and the provider adapter.
#[derive(Clone, Debug)]
pub struct CallContext {
    call_id:      uuid::Uuid,
    attempt:      u32,
    extensions:   Extensions,
    deadline:     Option<Instant>,
    cancellation: CancellationToken,
}

impl Default for CallContext {
    fn default() -> Self {
        Self {
            call_id:      uuid::Uuid::new_v4(),
            attempt:      1,
            extensions:   Extensions::default(),
            deadline:     None,
            cancellation: CancellationToken::default(),
        }
    }
}

impl CallContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn call_id(&self) -> uuid::Uuid {
        self.call_id
    }

    pub fn attempt(&self) -> u32 {
        self.attempt
    }

    pub fn set_attempt(&mut self, attempt: u32) {
        self.attempt = attempt.max(1);
    }

    pub fn extensions(&self) -> &Extensions {
        &self.extensions
    }

    pub fn extensions_mut(&mut self) -> &mut Extensions {
        &mut self.extensions
    }

    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    pub fn set_deadline(&mut self, deadline: Instant) {
        self.deadline = Some(deadline);
    }

    pub fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }
}

/// One resolved logical inference call.
#[derive(Clone, Debug)]
pub struct Call {
    pub(crate) request: Request,
    pub(crate) route:   ResolvedRoute,
    pub(crate) mode:    Operation,
    pub(crate) context: CallContext,
}

impl Call {
    pub fn request(&self) -> &Request {
        &self.request
    }
    pub fn route(&self) -> &ResolvedRoute {
        &self.route
    }
    pub fn operation(&self) -> Operation {
        self.mode
    }
    pub fn context(&self) -> &CallContext {
        &self.context
    }
    pub fn context_mut(&mut self) -> &mut CallContext {
        &mut self.context
    }

    /// Transforms the payload without changing its resolved model selector.
    pub fn map_request(
        mut self,
        transform: impl FnOnce(Request) -> Result<Request, RequestBuildError>,
    ) -> Result<Self, Error> {
        let selector = self.request.model().to_owned();
        let request = transform(self.request).map_err(|source| {
            Error::new(
                ErrorKind::InvalidRequest,
                "middleware produced an invalid request",
            )
            .with_source(source)
        })?;
        if request.model() != selector {
            return Err(Error::new(
                ErrorKind::Middleware,
                "middleware cannot change the model selector",
            ));
        }
        self.request = request;
        Ok(self)
    }
}

/// A complete response or an accepted response stream.
#[expect(
    clippy::large_enum_variant,
    reason = "the public middleware contract keeps complete responses directly accessible"
)]
pub enum Output {
    Complete(Response),
    Stream(ResponseStream),
    InputTokenCount(Option<InputTokenCount>),
}

impl fmt::Debug for Output {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Complete(response) => formatter.debug_tuple("Complete").field(response).finish(),
            Self::Stream(_) => formatter.write_str("Stream(<response stream>)"),
            Self::InputTokenCount(count) => formatter
                .debug_tuple("InputTokenCount")
                .field(count)
                .finish(),
        }
    }
}

/// Wraps one resolved logical call.
#[async_trait]
pub trait Middleware: Send + Sync + 'static {
    async fn handle(&self, call: Call, next: Next) -> Result<Output, Error>;
}

/// The remaining middleware stack and provider endpoint.
#[derive(Clone)]
pub struct Next {
    pipeline: Arc<Pipeline>,
    index:    usize,
}

impl Next {
    pub async fn run(self, call: Call) -> Result<Output, Error> {
        if call.context.cancellation().is_cancelled() {
            return Err(Error::new(ErrorKind::Cancelled, "the call was cancelled"));
        }
        if call
            .context
            .deadline()
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(Error::new(ErrorKind::Timeout, "the call deadline expired"));
        }

        if let Some(middleware) = self.pipeline.middleware.get(self.index).cloned() {
            return middleware
                .handle(call, Self {
                    pipeline: self.pipeline,
                    index:    self.index + 1,
                })
                .await;
        }

        crate::client::validate_request(&call.request, &call.route)?;
        let adapter = self
            .pipeline
            .adapters
            .get(call.route.provider().id())
            .cloned()
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::Configuration,
                    format!(
                        "no adapter is registered for provider {}",
                        call.route.provider().id()
                    ),
                )
            })?;
        let resolved = ResolvedCall::new(call.request, call.route, call.context);
        match call.mode {
            Operation::Complete => adapter
                .complete(&resolved)
                .await
                .and_then(|response| self.pipeline.policy.response(response))
                .map(Output::Complete),
            Operation::Stream => adapter
                .stream(&resolved)
                .await
                .map(|stream| Output::Stream(self.pipeline.policy.stream(stream))),
            Operation::CountInputTokens => adapter
                .count_input_tokens(&resolved)
                .await
                .map(Output::InputTokenCount),
        }
    }
}

pub(crate) struct Pipeline {
    pub policy:     crate::types::ResponsePolicy,
    pub middleware: Vec<Arc<dyn Middleware>>,
    pub adapters:   BTreeMap<ProviderId, Arc<dyn ProviderAdapter>>,
}

impl Pipeline {
    pub(crate) fn start(self: &Arc<Self>) -> Next {
        Next {
            pipeline: self.clone(),
            index:    0,
        }
    }
}

/// Maps every successful stream event with a fallible application function.
pub fn map_stream(
    stream: ResponseStream,
    map: impl Fn(StreamEvent) -> Result<StreamEvent, Error> + Send + Sync + 'static,
) -> ResponseStream {
    use futures_util::StreamExt as _;

    let map = Arc::new(map);
    ResponseStream::new(stream.map(move |item| item.and_then(|event| map(event))))
}

/// Inspects stream events and errors without changing them.
pub fn inspect_stream(
    stream: ResponseStream,
    inspect: impl Fn(&Result<StreamEvent, Error>) + Send + Sync + 'static,
) -> ResponseStream {
    use futures_util::StreamExt as _;

    ResponseStream::new(stream.inspect(inspect))
}

/// Runs one callback when a stream ends or is dropped.
pub fn finalize_stream(
    stream: ResponseStream,
    finalize: impl FnOnce() + Send + 'static,
) -> ResponseStream {
    ResponseStream::new(FinalizeStream {
        stream,
        finalize: Some(Box::new(finalize)),
    })
}

struct FinalizeStream {
    stream:   ResponseStream,
    finalize: Option<Box<dyn FnOnce() + Send>>,
}

impl Stream for FinalizeStream {
    type Item = Result<StreamEvent, Error>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let result = Pin::new(&mut self.stream).poll_next(context);
        if matches!(result, Poll::Ready(None))
            && let Some(finalize) = self.finalize.take()
        {
            finalize();
        }
        result
    }
}

impl Drop for FinalizeStream {
    fn drop(&mut self) {
        if let Some(finalize) = self.finalize.take() {
            finalize();
        }
    }
}
