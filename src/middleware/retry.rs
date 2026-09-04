use std::collections::VecDeque;
use std::collections::hash_map::RandomState;
use std::fmt;
use std::hash::{BuildHasher as _, Hasher as _};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures_util::StreamExt as _;
use futures_util::stream::{empty, unfold};
use tokio::time::sleep;

use super::{Call, Middleware, Next, Observer, Operation, Output, RetryStage};
use crate::types::{Error, ErrorKind, ResponseStream, RetryClassification, StreamEvent};

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

/// Reports one retry to the trace and to the observer, when one is set.
fn report_retry(
    observer: Option<&Arc<dyn Observer>>,
    call: &Call,
    attempt: u32,
    delay: Duration,
    error: &Error,
    stage: RetryStage,
) {
    tracing::warn!(
        attempt,
        delay_secs = delay.as_secs_f64(),
        stage = ?stage,
        error_kind = ?error.kind(),
        status = error.status(),
        provider_code = error.provider_code(),
        error = ?error,
        "the provider call failed and will be retried"
    );
    if let Some(observer) = observer {
        observer.on_retry(call, error, attempt, delay, stage);
    }
}

/// Retries retryable failures on the same resolved route.
#[derive(Clone)]
#[must_use]
pub struct RetryMiddleware {
    policy:   RetryPolicy,
    observer: Option<Arc<dyn Observer>>,
}

impl RetryMiddleware {
    pub fn new(policy: RetryPolicy) -> Self {
        Self {
            policy,
            observer: None,
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
        self.observer = Some(observer);
        self
    }
}

impl fmt::Debug for RetryMiddleware {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RetryMiddleware")
            .field("policy", &self.policy)
            .field("observer", &self.observer.is_some())
            .finish()
    }
}

#[async_trait]
impl Middleware for RetryMiddleware {
    async fn handle(&self, call: Call, next: Next) -> Result<Output, Error> {
        let mut attempt = call.context.attempt();
        let mut current = call.clone();
        loop {
            current.context.set_attempt(attempt);
            match next.clone().run(current.clone()).await {
                Ok(Output::Stream(stream)) if call.mode == Operation::Stream => {
                    return Ok(Output::Stream(retry_stream(
                        stream,
                        current,
                        next,
                        self.policy,
                        self.observer.clone(),
                    )));
                }
                Ok(output) => return Ok(output),
                Err(error) => {
                    let Some(delay) = self.policy.next_delay(attempt, &error) else {
                        return Err(error);
                    };
                    if deadline_prevents_retry(&current, delay) {
                        return Err(error);
                    }
                    report_retry(
                        self.observer.as_ref(),
                        &current,
                        attempt,
                        delay,
                        &error,
                        RetryStage::Request,
                    );
                    sleep(delay).await;
                    attempt = attempt.saturating_add(1);
                }
            }
        }
    }
}

/// Retries a stream that fails before it produces visible output.
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
    policy: RetryPolicy,
    observer: Option<Arc<dyn Observer>>,
) -> ResponseStream {
    struct State {
        stream:   ResponseStream,
        call:     Call,
        next:     Next,
        policy:   RetryPolicy,
        observer: Option<Arc<dyn Observer>>,
        attempt:  u32,
        visible:  bool,
        /// Bookkeeping from the current attempt, not yet delivered.
        held:     Vec<StreamEvent>,
        /// Items already decided on, waiting for the consumer to poll.
        ready:    VecDeque<Result<StreamEvent, Error>>,
        ended:    bool,
    }

    impl State {
        /// Moves the held bookkeeping into the delivery queue.
        fn release_held(&mut self) {
            self.ready.extend(self.held.drain(..).map(Ok));
        }
    }

    let state = State {
        stream,
        attempt: call.context.attempt(),
        call,
        next,
        policy,
        observer,
        visible: false,
        held: Vec::new(),
        ready: VecDeque::new(),
        ended: false,
    };
    ResponseStream::new(unfold(state, |mut state| async move {
        'read: loop {
            if let Some(item) = state.ready.pop_front() {
                return Some((item, state));
            }
            if state.ended {
                return None;
            }
            match state.stream.next().await {
                Some(Ok(event)) if !state.visible && !event.is_visible() => {
                    state.held.push(event);
                }
                Some(Ok(event)) => {
                    state.visible = true;
                    state.release_held();
                    state.ready.push_back(Ok(event));
                }
                Some(Err(mut error)) if !state.visible => {
                    // The failed attempt's stream must be dropped before the
                    // backoff sleep and the reconnect: a layer below may hold
                    // a resource — a concurrency permit, a connection — for
                    // exactly as long as its stream lives, and the reconnect
                    // re-enters that layer to acquire the same resource.
                    state.stream = ResponseStream::new(empty());
                    while let Some(delay) = state.policy.next_delay(state.attempt, &error) {
                        if deadline_prevents_retry(&state.call, delay) {
                            break;
                        }
                        report_retry(
                            state.observer.as_ref(),
                            &state.call,
                            state.attempt,
                            delay,
                            &error,
                            RetryStage::Stream,
                        );
                        sleep(delay).await;
                        state.attempt = state.attempt.saturating_add(1);
                        state.call.context.set_attempt(state.attempt);
                        match state.next.clone().run(state.call.clone()).await {
                            Ok(Output::Stream(stream)) => {
                                // The abandoned attempt produced no visible
                                // output, so its bookkeeping describes a
                                // stream nobody saw.
                                state.held.clear();
                                state.stream = stream;
                                continue 'read;
                            }
                            Ok(_) => {
                                error = Error::new(
                                    ErrorKind::Middleware,
                                    "stream retry returned a complete response",
                                );
                                break;
                            }
                            Err(next_error) => error = next_error,
                        }
                    }
                    state.release_held();
                    state.ready.push_back(Err(error));
                    state.ended = true;
                }
                Some(Err(error)) => {
                    state.ready.push_back(Err(error));
                }
                None => {
                    state.release_held();
                    state.ended = true;
                }
            }
        }
    }))
}

fn deadline_prevents_retry(call: &Call, delay: Duration) -> bool {
    call.context.deadline().is_some_and(|deadline| {
        Instant::now()
            .checked_add(delay)
            .is_none_or(|next| next >= deadline)
    })
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
