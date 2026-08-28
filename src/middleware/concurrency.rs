use std::num::NonZeroUsize;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::StreamExt as _;
use tokio::sync::Semaphore;

use super::{Call, Middleware, Next, Output};
use crate::types::{Error, ErrorKind};

/// Limits the number of calls active inside this middleware layer.
#[derive(Clone, Debug)]
pub struct ConcurrencyLimitMiddleware {
    semaphore: Arc<Semaphore>,
}

impl ConcurrencyLimitMiddleware {
    pub fn new(limit: NonZeroUsize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(limit.get())),
        }
    }
}

#[async_trait]
impl Middleware for ConcurrencyLimitMiddleware {
    async fn handle(&self, call: Call, next: Next) -> Result<Output, Error> {
        let permit = self
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|source| {
                Error::new(ErrorKind::Cancelled, "the concurrency limiter was closed")
                    .with_source(source)
            })?;
        match next.run(call).await? {
            Output::Complete(response) => Ok(Output::Complete(response)),
            Output::Stream(stream) => Ok(Output::Stream(Box::pin(stream.map(move |item| {
                let _permit = &permit;
                item
            })))),
        }
    }
}
