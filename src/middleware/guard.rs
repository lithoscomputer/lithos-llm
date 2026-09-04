use std::future::{Future, pending};
use std::time::{Duration, Instant};

use futures_util::StreamExt as _;
use futures_util::stream::unfold;
use tokio::time::{Instant as TokioInstant, sleep_until};

use super::{CallContext, CancellationToken};
use crate::types::{Error, ErrorKind, ResponseStream};

/// Owns the cancellation signal and deadline at one call boundary.
///
/// Each middleware boundary takes its own snapshot. Inner calls can tighten
/// the budget but cannot replace the guard retained by an enclosing call.
#[derive(Clone)]
pub(crate) struct CallGuard {
    cancellation: CancellationToken,
    deadline:     Option<Instant>,
}

impl CallGuard {
    pub(crate) fn new(context: &CallContext) -> Self {
        Self {
            cancellation: context.cancellation().clone(),
            deadline:     context.deadline(),
        }
    }

    pub(crate) fn with_timeout(mut self, timeout: Duration) -> Result<Self, Error> {
        let deadline = Instant::now().checked_add(timeout).ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidRequest,
                "the requested timeout is too large for this platform",
            )
        })?;
        self.deadline = Some(
            self.deadline
                .map_or(deadline, |current| current.min(deadline)),
        );
        Ok(self)
    }

    pub(crate) fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    pub(crate) fn check(&self) -> Result<(), Error> {
        if self.cancellation.is_cancelled() {
            return Err(Self::cancelled());
        }
        if self
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(Self::expired());
        }
        Ok(())
    }

    pub(crate) fn permits_retry_after(&self, delay: Duration) -> bool {
        self.check().is_ok()
            && self.deadline.is_none_or(|deadline| {
                Instant::now()
                    .checked_add(delay)
                    .is_some_and(|next| next < deadline)
            })
    }

    pub(crate) async fn run<T>(
        &self,
        future: impl Future<Output = Result<T, Error>>,
    ) -> Result<T, Error> {
        self.check()?;
        tokio::select! {
            biased;
            () = self.cancellation.cancelled() => Err(Self::cancelled()),
            () = self.wait_for_deadline() => Err(Self::expired()),
            result = future => result,
        }
    }

    pub(crate) fn stream(self, stream: ResponseStream) -> ResponseStream {
        ResponseStream::new(unfold((self, stream), |(guard, mut stream)| async move {
            let item = guard.run(async { stream.next().await.transpose() }).await;
            match item {
                Ok(Some(event)) => Some((Ok(event), (guard, stream))),
                Err(error) => Some((Err(error), (guard, stream))),
                Ok(None) => None,
            }
        }))
    }

    async fn wait_for_deadline(&self) {
        match self.deadline {
            Some(deadline) => sleep_until(TokioInstant::from_std(deadline)).await,
            None => pending().await,
        }
    }

    fn cancelled() -> Error {
        Error::new(ErrorKind::Cancelled, "the call was cancelled")
    }

    fn expired() -> Error {
        Error::new(ErrorKind::Timeout, "the call deadline expired")
    }
}

#[cfg(test)]
mod tests {
    use std::future::ready;
    use std::time::{Duration, Instant};

    use futures_util::{StreamExt as _, stream};

    use super::CallGuard;
    use crate::middleware::CallContext;
    use crate::types::{ErrorKind, ResponseStream, StreamEvent};

    #[tokio::test]
    async fn cancellation_wins_over_ready_calls_and_stream_events() {
        let context = CallContext::new();
        context.cancellation().cancel();
        let guard = CallGuard::new(&context);
        assert_eq!(
            guard
                .run(ready(Ok(())))
                .await
                .expect_err("cancelled")
                .kind(),
            ErrorKind::Cancelled
        );
        let mut events = guard.stream(ResponseStream::new(stream::iter([Ok(
            StreamEvent::Started { id: None },
        )])));
        assert_eq!(
            events
                .next()
                .await
                .expect("terminal error")
                .expect_err("cancelled")
                .kind(),
            ErrorKind::Cancelled
        );
        assert!(events.next().await.is_none());
    }

    #[test]
    fn a_timeout_cannot_extend_an_expired_context_deadline() {
        let mut context = CallContext::new();
        let deadline = Instant::now();
        context.set_deadline(deadline);
        let guard = CallGuard::new(&context)
            .with_timeout(Duration::from_secs(60))
            .expect("budget");
        assert_eq!(guard.deadline(), Some(deadline));
        assert_eq!(
            guard.check().expect_err("expired").kind(),
            ErrorKind::Timeout
        );
        assert!(!guard.permits_retry_after(Duration::ZERO));
    }
}
