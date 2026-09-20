use std::collections::hash_map::RandomState;
use std::fmt;
use std::hash::{BuildHasher as _, Hasher as _};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures_util::StreamExt as _;
use futures_util::stream::{empty, iter, unfold};
use tokio::time::sleep;

use super::{
    Call, CallGuard, Middleware, Next, Observer, Operation, Output, RetryEvent, RetryStage,
};
use crate::types::{
    Error, ErrorKind, FinishReason, Response, ResponseStream, RetryClassification, StreamEvent,
};

/// The longest `Retry-After` this policy honors by default.
const DEFAULT_RETRY_AFTER_CAP: Duration = Duration::from_secs(60);

/// Same-route retry settings.
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct RetryPolicy {
    max_attempts:    u32,
    initial_delay:   Duration,
    max_delay:       Duration,
    retry_after_cap: Duration,
    jitter:          bool,
}

impl RetryPolicy {
    pub fn exponential() -> Self {
        Self {
            max_attempts:    3,
            initial_delay:   Duration::from_millis(100),
            max_delay:       Duration::from_secs(5),
            retry_after_cap: DEFAULT_RETRY_AFTER_CAP,
            jitter:          false,
        }
    }

    pub fn max_attempts(mut self, attempts: u32) -> Self {
        self.max_attempts = attempts.max(1);
        self
    }

    pub fn initial_delay(mut self, delay: Duration) -> Self {
        self.initial_delay = delay;
        self
    }

    /// Caps the computed backoff delay.
    ///
    /// This bounds the policy's own exponential delays only. A provider that
    /// sends `Retry-After` is answered through
    /// [`retry_after_cap`](Self::retry_after_cap) instead.
    pub fn max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    /// Sets the longest `Retry-After` the policy honors.
    ///
    /// A provider that asks for a wait within this cap gets exactly the wait
    /// it asked for. A longer one stops the retries: the provider has said
    /// the call cannot succeed soon, so the caller decides what to do rather
    /// than the policy re-sending against explicit guidance. The default cap
    /// is 60 seconds.
    pub fn retry_after_cap(mut self, cap: Duration) -> Self {
        self.retry_after_cap = cap;
        self
    }

    /// Spreads computed delays over a random range, ending at the delay.
    ///
    /// This decorrelates many callers that failed together. It never changes
    /// a `Retry-After` wait, which is the provider's own instruction. Jitter
    /// is off by default.
    pub fn jitter(mut self, jitter: bool) -> Self {
        self.jitter = jitter;
        self
    }

    /// The wait before the next attempt, or `None` when this failure ends the
    /// retries.
    ///
    /// `attempt` is the attempt that just failed, counted from 1. `None`
    /// covers all three refusals: a spent attempt budget, an error that
    /// repeating cannot fix, and a `Retry-After` longer than the cap.
    ///
    /// This is the whole retry decision, so an application that must own its
    /// own retry loop — for example to replay a turn after a stream already
    /// delivered visible output, which no middleware can do — can drive that
    /// loop with the same policy the [`RetryMiddleware`] uses.
    pub fn next_delay(self, attempt: u32, error: &Error) -> Option<Duration> {
        if attempt >= self.max_attempts
            || matches!(error.retry_classification(), RetryClassification::Never)
        {
            return None;
        }
        if let Some(retry_after) = error.retry_after() {
            return (retry_after <= self.retry_after_cap).then_some(retry_after);
        }
        let exponent = attempt.saturating_sub(1).min(31);
        let delay = self
            .initial_delay
            .saturating_mul(1_u32 << exponent)
            .min(self.max_delay);
        Some(if self.jitter { jittered(delay) } else { delay })
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::exponential()
    }
}

/// Picks a delay between half of `delay` and `delay`.
///
/// `RandomState` seeds itself from the operating system, which is entropy
/// enough to separate concurrent callers without a random-number dependency.
fn jittered(delay: Duration) -> Duration {
    let nanos = u64::try_from(delay.as_nanos()).unwrap_or(u64::MAX);
    let floor = nanos / 2;
    let span = nanos - floor;
    if span == 0 {
        return delay;
    }
    let random = RandomState::new().build_hasher().finish();
    Duration::from_nanos(floor + random % (span + 1))
}

/// The shared attempt loop: decides, reports, and waits out one retry.
///
/// Both the complete path and the pre-visible stream path retry the same
/// way; only what they do after the wait differs. Holding the policy and the
/// observer together keeps the report and the sleep in one place.
#[derive(Clone)]
struct Retrier {
    policy:   RetryPolicy,
    observer: Option<Arc<dyn Observer>>,
}

impl Retrier {
    /// Waits before the next attempt, or answers `false` when this failure
    /// ends the retries.
    ///
    /// On `true`, `call` already names the next attempt and the backoff has
    /// elapsed, so the caller runs `next` immediately. The
    /// [`CallGuard`] veto applies here: a wait the deadline could not survive
    /// ends the retries the same way a spent budget does.
    async fn wait_before_retry(&self, call: &mut Call, error: &Error, stage: RetryStage) -> bool {
        let attempt = call.context.attempt();
        let Some(delay) = self
            .policy
            .next_delay(attempt, error)
            .filter(|delay| CallGuard::new(call.context()).permits_retry_after(*delay))
        else {
            return false;
        };
        self.report(call, RetryEvent::new(error, attempt, delay, stage));
        sleep(delay).await;
        call.context.set_attempt(attempt.saturating_add(1));
        true
    }

    /// Reports one retry to the trace and to the observer, when one is set.
    fn report(&self, call: &Call, retry: RetryEvent<'_>) {
        tracing::warn!(
            attempt = retry.attempt,
            delay_secs = retry.delay.as_secs_f64(),
            stage = ?retry.stage,
            error_kind = ?retry.error.kind(),
            status = retry.error.status(),
            provider_code = retry.error.provider_code(),
            error = ?retry.error,
            "the provider call failed and will be retried"
        );
        if let Some(observer) = &self.observer {
            observer.on_retry(call, retry);
        }
    }
}

/// An attempt the retry loop gave up on, kept until the loop decides whether
/// another attempt replaces it.
///
/// A response that ended incomplete is retried as if it had failed, but if the
/// retries run out the caller gets that response back rather than the error
/// it was classified as. A new attempt supersedes the abandoned one entirely,
/// including when opening that attempt fails.
enum Abandoned {
    Failed(Error),
    /// The response the provider gave, and the error the retry policy
    /// classifies it by. Boxed so the common failed attempt does not grow by
    /// the response's size.
    Incomplete {
        response: Box<Response>,
        error:    Error,
    },
}

impl Abandoned {
    fn incomplete(response: Box<Response>, call: &Call) -> Self {
        Self::Incomplete {
            response,
            error: incomplete_error(call),
        }
    }

    /// The error the retry policy classifies this attempt by.
    fn error(&self) -> &Error {
        match self {
            Self::Failed(error) | Self::Incomplete { error, .. } => error,
        }
    }

    /// What the caller receives when no further attempt is made.
    fn into_output(self) -> Result<Box<Response>, Error> {
        match self {
            Self::Failed(error) => Err(error),
            Self::Incomplete { response, .. } => Ok(response),
        }
    }

    /// The stream item the caller receives when no further attempt is made.
    fn into_item(self) -> Result<StreamEvent, Error> {
        self.into_output()
            .map(|response| StreamEvent::Ended { response })
    }
}

/// Retries retryable failures on the same resolved route.
#[derive(Clone)]
#[must_use]
pub struct RetryMiddleware {
    retrier: Retrier,
}

impl RetryMiddleware {
    pub fn new(policy: RetryPolicy) -> Self {
        Self {
            retrier: Retrier {
                policy,
                observer: None,
            },
        }
    }

    /// Reports every retried attempt to `observer` through
    /// [`Observer::on_retry`].
    ///
    /// The retry layer must hold the observer itself: an observer installed
    /// as ordinary middleware sees one logical call, not its attempts.
    pub fn observer(self, observer: impl Observer) -> Self {
        self.observer_arc(Arc::new(observer))
    }

    pub fn observer_arc(mut self, observer: Arc<dyn Observer>) -> Self {
        self.retrier.observer = Some(observer);
        self
    }
}

impl fmt::Debug for RetryMiddleware {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RetryMiddleware")
            .field("policy", &self.retrier.policy)
            .field("observer", &self.retrier.observer.is_some())
            .finish()
    }
}

#[async_trait]
impl Middleware for RetryMiddleware {
    async fn handle(&self, call: Call, next: Next) -> Result<Output, Error> {
        let mut current = call.clone();
        loop {
            let abandoned = match next.clone().run(current.clone()).await {
                Ok(Output::Stream(stream)) if call.mode == Operation::Stream => {
                    return Ok(Output::Stream(retry_stream(
                        stream,
                        current,
                        next,
                        self.retrier.clone(),
                    )));
                }
                // An incomplete response is retried as if it had failed; see
                // `Abandoned`.
                Ok(Output::Complete(response))
                    if response.finish_reason == FinishReason::Incomplete =>
                {
                    Abandoned::incomplete(Box::new(response), &current)
                }
                Ok(output) => return Ok(output),
                Err(error) => Abandoned::Failed(error),
            };
            if !self
                .retrier
                .wait_before_retry(&mut current, abandoned.error(), RetryStage::Request)
                .await
            {
                return abandoned
                    .into_output()
                    .map(|response| Output::Complete(*response));
            }
        }
    }
}

fn incomplete_error(call: &Call) -> Error {
    Error::new(
        ErrorKind::StreamDecode,
        "the provider ended without a complete response",
    )
    .with_provider(call.route().provider().id().clone())
    .with_provider_code("incomplete_response")
    .with_retry(RetryClassification::Safe)
}

/// Retries a stream that fails or ends incomplete before it delivers content.
///
/// Protocol bookkeeping — `Started`, `RateLimits`, `ContentBlockStart` — is
/// held back until the stream produces its first visible event. A reconnect
/// therefore discards the abandoned attempt's bookkeeping and replaces it with
/// the new attempt's, so one logical stream delivers one `Started` and no
/// repeated block start, however many attempts it took. The cost is that the
/// consumer learns the stream opened at the same moment it learns what the
/// stream says.
fn retry_stream(
    stream: ResponseStream,
    call: Call,
    next: Next,
    retrier: Retrier,
) -> ResponseStream {
    let state = StreamRetry {
        stream,
        call,
        next,
        retrier,
        phase: Phase::Holding(Vec::new()),
    };
    ResponseStream::new(
        unfold(state, |mut state| async move {
            Box::pin(state.step()).await.map(|batch| (batch, state))
        })
        .flat_map(iter),
    )
}

/// Where one logical stream is in its life.
enum Phase {
    /// Before the first visible event: bookkeeping is held so a reconnect can
    /// discard it.
    Holding(Vec<StreamEvent>),
    /// After the first visible event: everything passes through, and an
    /// error is final because a replay would duplicate what the caller saw.
    Delivering,
}

/// The state of one retrying stream, driven a step at a time.
///
/// Each step reads one item from the current attempt and answers with the
/// batch of items the caller receives for it — usually one, none while
/// bookkeeping is held, and several when held bookkeeping is released.
struct StreamRetry {
    stream:  ResponseStream,
    call:    Call,
    next:    Next,
    retrier: Retrier,
    phase:   Phase,
}

impl StreamRetry {
    /// Reads one item and decides what the caller receives for it.
    ///
    /// `None` ends the stream. An exhausted retry swaps in an empty stream, so
    /// the step after the terminal item hits EOF.
    async fn step(&mut self) -> Option<Vec<Result<StreamEvent, Error>>> {
        let item = self.stream.next().await;
        let Phase::Holding(held) = &mut self.phase else {
            // Delivering: errors and events alike pass through, and EOF ends
            // the stream.
            return item.map(|item| vec![item]);
        };

        let abandoned = match item {
            Some(Ok(StreamEvent::Ended { response }))
                if response.finish_reason == FinishReason::Incomplete =>
            {
                Abandoned::incomplete(response, &self.call)
            }
            // A decoder may synthesize closing blocks at EOF. Hold those
            // until actual deltas or the terminal outcome reach the caller.
            Some(Ok(event))
                if !event.is_visible() || matches!(event, StreamEvent::ContentBlockEnd { .. }) =>
            {
                held.push(event);
                return Some(Vec::new());
            }
            Some(Ok(event)) => {
                let mut released: Vec<_> = held.drain(..).map(Ok).collect();
                released.push(Ok(event));
                self.phase = Phase::Delivering;
                return Some(released);
            }
            Some(Err(error)) => Abandoned::Failed(error),
            None => {
                let released: Vec<_> = held.drain(..).map(Ok).collect();
                self.stream = ResponseStream::new(empty());
                return Some(released);
            }
        };
        Some(Box::pin(self.reconnect(abandoned)).await)
    }

    /// Replaces the abandoned attempt with a new one, or delivers the
    /// abandoned outcome when the retries are spent.
    async fn reconnect(&mut self, mut abandoned: Abandoned) -> Vec<Result<StreamEvent, Error>> {
        // The failed attempt's stream must be dropped before the backoff
        // sleep and the reconnect: a layer below may hold a resource — a
        // concurrency permit, a connection — for exactly as long as its
        // stream lives, and the reconnect re-enters that layer to acquire the
        // same resource.
        self.stream = ResponseStream::new(empty());
        loop {
            if !self
                .retrier
                .wait_before_retry(&mut self.call, abandoned.error(), RetryStage::Stream)
                .await
            {
                break;
            }
            // A new attempt supersedes the abandoned response, including when
            // opening that attempt fails.
            match self.next.clone().run(self.call.clone()).await {
                Ok(Output::Stream(stream)) => {
                    // The abandoned attempt produced no visible output, so its
                    // bookkeeping describes a stream nobody saw.
                    self.stream = stream;
                    self.phase = Phase::Holding(Vec::new());
                    return Vec::new();
                }
                Ok(_) => {
                    abandoned = Abandoned::Failed(Error::new(
                        ErrorKind::Middleware,
                        "stream retry returned a complete response",
                    ));
                    break;
                }
                Err(next_error) => abandoned = Abandoned::Failed(next_error),
            }
        }
        let Phase::Holding(held) = &mut self.phase else {
            unreachable!("a reconnect only runs while holding");
        };
        let mut released: Vec<_> = held.drain(..).map(Ok).collect();
        released.push(abandoned.into_item());
        released
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{RetryPolicy, jittered};
    use crate::types::{Error, ErrorKind, RetryClassification};

    fn retryable(retry: RetryClassification) -> Error {
        Error::new(ErrorKind::Server, "provider failure").with_retry(retry)
    }

    #[test]
    fn a_retry_after_within_the_cap_is_used_exactly() {
        let policy = RetryPolicy::exponential().max_delay(Duration::from_millis(1));
        let error = retryable(RetryClassification::after(Duration::from_secs(30)));

        assert_eq!(
            policy.next_delay(1, &error),
            Some(Duration::from_secs(30)),
            "the backoff cap must not clamp a Retry-After value"
        );
    }

    #[test]
    fn a_retry_after_beyond_the_cap_ends_the_retries() {
        let policy = RetryPolicy::exponential().retry_after_cap(Duration::from_secs(60));
        let error = retryable(RetryClassification::after(Duration::from_secs(120)));

        assert_eq!(policy.next_delay(1, &error), None);
    }

    #[test]
    fn a_computed_delay_grows_up_to_the_maximum() {
        let policy = RetryPolicy::exponential()
            .initial_delay(Duration::from_millis(100))
            .max_delay(Duration::from_millis(250));
        let error = retryable(RetryClassification::Safe);

        assert_eq!(
            policy.next_delay(1, &error),
            Some(Duration::from_millis(100))
        );
        assert_eq!(
            policy.next_delay(2, &error),
            Some(Duration::from_millis(200))
        );
        assert_eq!(policy.next_delay(3, &error), None, "attempts are spent");
    }

    #[test]
    fn an_unretryable_error_ends_the_retries() {
        let policy = RetryPolicy::exponential();
        let error = retryable(RetryClassification::Never);

        assert_eq!(policy.next_delay(1, &error), None);
    }

    #[test]
    fn jitter_stays_between_half_the_delay_and_the_delay() {
        let delay = Duration::from_millis(400);

        for _ in 0..100 {
            let jittered = jittered(delay);
            assert!(
                jittered >= delay / 2 && jittered <= delay,
                "jittered delay out of range: {jittered:?}"
            );
        }
    }

    #[test]
    fn jitter_leaves_a_zero_delay_alone() {
        assert_eq!(jittered(Duration::ZERO), Duration::ZERO);
    }
}
