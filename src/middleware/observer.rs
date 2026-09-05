use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures_util::StreamExt as _;

use super::{Call, CallGuard, Middleware, Next, Output};
use crate::adapter::InputTokenCount;
use crate::types::{Error, ErrorKind, Response, ResponseStream, StreamEvent};

/// The final outcome of one observed middleware invocation.
#[derive(Debug)]
#[non_exhaustive]
pub enum CallOutcome<'a> {
    Response(&'a Response),
    InputTokenCount(Option<&'a InputTokenCount>),
    Failed(&'a Error),
    Cancelled,
    Dropped,
}

/// Where in a call's life a retry was decided.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RetryStage {
    /// The attempt failed before a stream existed: the provider rejected the
    /// request, or the stream could not be opened.
    Request,
    /// The attempt's stream failed before it delivered visible output.
    Stream,
}

/// Receives synchronous lifecycle observations without changing a call.
///
/// Every hook runs inline on the call path. An implementation must return
/// quickly, must not block, and must not panic. Hand off slow work, such as
/// persisting an event, to a channel or task the application owns.
pub trait Observer: Send + Sync + 'static {
    fn on_start(&self, _call: &Call) {}

    /// Called once when an observed invocation finishes or is abandoned.
    /// Install outside retry to observe one logical call.
    fn on_finish(&self, _call: &Call, _outcome: CallOutcome<'_>) {}

    fn on_stream_event(&self, _call: &Call, _event: Result<&StreamEvent, &Error>) {}

    /// Observes one retry decided by
    /// [`RetryMiddleware`](super::RetryMiddleware).
    ///
    /// `attempt` is the attempt that just failed with `error`, counted from 1
    /// as [`CallContext::attempt`](super::CallContext::attempt) does. The
    /// middleware waits `delay` and then runs attempt `attempt + 1`. `stage`
    /// names how far the failed attempt got: a request the provider never
    /// answered with a stream, or a stream that failed before it delivered
    /// visible output. Failures the policy refuses to retry are not reported
    /// here; that error reaches the caller instead.
    ///
    /// This hook fires only for observers given to the retry middleware
    /// through [`RetryMiddleware::observer`](super::RetryMiddleware::observer).
    /// [`ObserverMiddleware`] never calls it, because a middleware outside the
    /// retry layer sees one logical call, not its attempts.
    fn on_retry(
        &self,
        _call: &Call,
        _error: &Error,
        _attempt: u32,
        _delay: Duration,
        _stage: RetryStage,
    ) {
    }
}

/// Adapts an observer into middleware.
pub struct ObserverMiddleware {
    observer: Arc<dyn Observer>,
}

impl ObserverMiddleware {
    pub fn new(observer: impl Observer) -> Self {
        Self {
            observer: Arc::new(observer),
        }
    }

    pub fn from_arc(observer: Arc<dyn Observer>) -> Self {
        Self { observer }
    }
}

#[async_trait]
impl Middleware for ObserverMiddleware {
    async fn handle(&self, call: Call, next: Next) -> Result<Output, Error> {
        let mut finish = FinishGuard {
            observer: self.observer.clone(),
            call:     call.clone(),
            finished: false,
        };
        self.observer.on_start(&call);
        match next.run(call.clone()).await {
            Ok(Output::InputTokenCount(count)) => {
                finish.finish(CallOutcome::InputTokenCount(count.as_ref()));
                Ok(Output::InputTokenCount(count))
            }
            Ok(Output::Complete(response)) => {
                finish.finish(CallOutcome::Response(&response));
                Ok(Output::Complete(response))
            }
            Ok(Output::Stream(stream)) => {
                let observer = self.observer.clone();
                Ok(Output::Stream(ResponseStream::new(stream.inspect(
                    move |event| {
                        observer.on_stream_event(&call, event.as_ref());
                        match event {
                            Ok(StreamEvent::Ended { response }) => {
                                finish.finish(CallOutcome::Response(response));
                            }
                            Err(error) => finish.error(error),
                            _ => {}
                        }
                    },
                ))))
            }
            Err(error) => {
                finish.error(&error);
                Err(error)
            }
        }
    }
}

struct FinishGuard {
    observer: Arc<dyn Observer>,
    call:     Call,
    finished: bool,
}

impl FinishGuard {
    fn finish(&mut self, outcome: CallOutcome<'_>) {
        if !self.finished {
            self.finished = true;
            self.observer.on_finish(&self.call, outcome);
        }
    }

    fn error(&mut self, error: &Error) {
        self.finish(if error.kind() == ErrorKind::Cancelled {
            CallOutcome::Cancelled
        } else {
            CallOutcome::Failed(error)
        });
    }
}

impl Drop for FinishGuard {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        match CallGuard::new(self.call.context()).check() {
            Err(error) => self.error(&error),
            Ok(()) => self.finish(CallOutcome::Dropped),
        }
    }
}
