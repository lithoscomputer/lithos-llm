use std::sync::Arc;

use async_trait::async_trait;
use futures_util::StreamExt as _;

use super::{Call, Middleware, Next, Output};
use crate::types::{Error, Response, StreamEvent};

/// Receives synchronous lifecycle observations without changing a call.
pub trait Observer: Send + Sync + 'static {
    fn on_start(&self, _call: &Call) {}

    fn on_complete(&self, _call: &Call, _result: Result<&Response, &Error>) {}

    fn on_stream_event(&self, _call: &Call, _event: Result<&StreamEvent, &Error>) {}
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
        self.observer.on_start(&call);
        let result = next.run(call.clone()).await;
        match result {
            Ok(Output::Complete(response)) => {
                self.observer.on_complete(&call, Ok(&response));
                Ok(Output::Complete(response))
            }
            Ok(Output::Stream(stream)) => {
                let observer = self.observer.clone();
                Ok(Output::Stream(Box::pin(stream.inspect(move |event| {
                    observer.on_stream_event(&call, event.as_ref());
                }))))
            }
            Err(error) => {
                self.observer.on_complete(&call, Err(&error));
                Err(error)
            }
        }
    }
}
