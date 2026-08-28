use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures_util::StreamExt as _;
use futures_util::stream::unfold;
use tokio::time::{Instant as TokioInstant, timeout_at};

use super::{Call, Middleware, Next, Output};
use crate::types::{Error, ErrorKind, ResponseStream};

/// Applies a call deadline and a maximum wait between stream events.
#[derive(Clone, Copy, Debug)]
pub struct TimeoutMiddleware {
    timeout: Duration,
}

impl TimeoutMiddleware {
    pub fn new(timeout: Duration) -> Self {
        Self { timeout }
    }
}

#[async_trait]
impl Middleware for TimeoutMiddleware {
    async fn handle(&self, mut call: Call, next: Next) -> Result<Output, Error> {
        let deadline = Instant::now().checked_add(self.timeout).ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidRequest,
                "the requested timeout is too large for this platform",
            )
        })?;
        let deadline = call
            .context
            .deadline()
            .map_or(deadline, |current| current.min(deadline));
        if call.context.deadline() != Some(deadline) {
            call.context.set_deadline(deadline);
        }
        let result = timeout_at(TokioInstant::from_std(deadline), next.run(call)).await;
        match result {
            Ok(Ok(Output::Stream(stream))) => Ok(Output::Stream(timeout_stream(
                stream,
                self.timeout,
                deadline,
            ))),
            Ok(result) => result,
            Err(source) => {
                Err(Error::new(ErrorKind::Timeout, "the LLM call timed out").with_source(source))
            }
        }
    }
}

fn timeout_stream(stream: ResponseStream, max_wait: Duration, deadline: Instant) -> ResponseStream {
    Box::pin(unfold(
        (stream, false),
        move |(mut stream, finished)| async move {
            if finished {
                return None;
            }
            let read_deadline = Instant::now()
                .checked_add(max_wait)
                .map_or(deadline, |read_deadline| read_deadline.min(deadline));
            match timeout_at(TokioInstant::from_std(read_deadline), stream.next()).await {
                Ok(Some(item)) => Some((item, (stream, false))),
                Ok(None) => None,
                Err(source) => Some((
                    Err(Error::new(
                        ErrorKind::Timeout,
                        if read_deadline == deadline {
                            "the LLM call deadline expired"
                        } else {
                            "the LLM stream read timed out"
                        },
                    )
                    .with_source(source)),
                    (stream, true),
                )),
            }
        },
    ))
}
