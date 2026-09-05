use std::future::{Future, pending};
use std::time::{Duration, Instant};

use futures_util::StreamExt as _;
use futures_util::stream::unfold;
use tokio::time::{Instant as TokioInstant, sleep_until};

use super::{CallContext, CancellationToken};
use crate::types::{Error, ErrorKind, ResponseStream};

/// Owns the cancellation signal and deadline at one call boundary.
///
/// Deadlines use Tokio's clock internally, matching its timers. The public
/// context still accepts standard-library instants.
/// Each middleware boundary takes its own snapshot. Inner calls can tighten
/// the budget but cannot replace the guard retained by an enclosing call.
#[derive(Clone)]
pub(crate) struct CallGuard {
    cancellation: CancellationToken,
    deadline:     Option<TokioInstant>,
}

impl CallGuard {
    pub(crate) fn new(context: &CallContext) -> Self {
        Self {
            cancellation: context.cancellation().clone(),
            deadline:     context.deadline().map(TokioInstant::from_std),
        }
    }

    pub(crate) fn with_timeout(mut self, timeout: Duration) -> Result<Self, Error> {
        let deadline = TokioInstant::now().checked_add(timeout).ok_or_else(|| {
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
        self.deadline.map(TokioInstant::into_std)
    }

    pub(crate) fn check(&self) -> Result<(), Error> {
        if self.cancellation.is_cancelled() {
            return Err(Self::cancelled());
        }
        if self
            .deadline
            .is_some_and(|deadline| TokioInstant::now() >= deadline)
        {
            return Err(Self::expired());
        }
        Ok(())
    }

    pub(crate) fn permits_retry_after(&self, delay: Duration) -> bool {
        self.check().is_ok()
            && self.deadline.is_none_or(|deadline| {
                TokioInstant::now()
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
            Some(deadline) => sleep_until(deadline).await,
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
    use std::time::Duration;

    use futures_util::{StreamExt as _, stream};
    use tokio::time::{Instant, advance};

    use super::CallGuard;
    use crate::middleware::CallContext;
    use crate::types::{ErrorKind, Response, ResponseStream, StreamEvent};

    #[tokio::test(start_paused = true)]
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

    #[tokio::test(start_paused = true)]
    async fn a_timeout_cannot_extend_an_expired_context_deadline() {
        let mut context = CallContext::new();
        let deadline = Instant::now().into_std();
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

    #[tokio::test(start_paused = true)]
    async fn retry_windows_and_expiry_share_the_timer_clock() {
        let guard = CallGuard::new(&CallContext::new())
            .with_timeout(Duration::from_secs(10))
            .expect("budget");
        assert!(guard.permits_retry_after(Duration::from_secs(9)));
        assert!(!guard.permits_retry_after(Duration::from_secs(10)));
        advance(Duration::from_secs(8)).await;
        assert!(guard.permits_retry_after(Duration::from_secs(1)));
        assert!(!guard.permits_retry_after(Duration::from_secs(2)));
        advance(Duration::from_secs(2)).await;
        assert_eq!(
            guard.run(ready(Ok(()))).await.expect_err("expired").kind(),
            ErrorKind::Timeout
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_stream_keeps_its_budget_between_polls() {
        let guard = CallGuard::new(&CallContext::new())
            .with_timeout(Duration::from_secs(10))
            .expect("budget");
        let mut events = guard.stream(ResponseStream::new(stream::iter([
            Ok(StreamEvent::Started { id: None }),
            Ok(StreamEvent::Ended {
                response: Box::new(Response::new("test".into(), "model".into(), Vec::new())),
            }),
        ])));
        advance(Duration::from_secs(9)).await;
        assert!(events.next().await.expect("within budget").is_ok());
        advance(Duration::from_secs(1)).await;
        assert_eq!(
            events
                .next()
                .await
                .expect("terminal")
                .expect_err("expired")
                .kind(),
            ErrorKind::Timeout
        );
        assert!(events.next().await.is_none());
    }
}
