use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures_util::StreamExt as _;
use futures_util::stream::unfold;
use tokio::time::sleep;

use super::{Call, Middleware, Mode, Next, Output};
use crate::types::{Error, ErrorKind, ResponseStream, RetryClassification};

/// Same-route retry settings.
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct RetryPolicy {
    max_attempts:  u32,
    initial_delay: Duration,
    max_delay:     Duration,
}

impl RetryPolicy {
    pub fn exponential() -> Self {
        Self {
            max_attempts:  3,
            initial_delay: Duration::from_millis(100),
            max_delay:     Duration::from_secs(5),
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

    pub fn max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    fn delay(self, attempt: u32, error: &Error) -> Duration {
        if let Some(retry_after) = error.retry_after() {
            return retry_after.min(self.max_delay);
        }
        let exponent = attempt.saturating_sub(1).min(31);
        self.initial_delay
            .saturating_mul(1_u32 << exponent)
            .min(self.max_delay)
    }

    fn can_retry(self, attempt: u32, error: &Error) -> bool {
        attempt < self.max_attempts
            && !matches!(error.retry_classification(), RetryClassification::Never)
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::exponential()
    }
}

/// Retries retryable failures on the same resolved route.
#[derive(Clone, Copy, Debug)]
pub struct RetryMiddleware {
    policy: RetryPolicy,
}

impl RetryMiddleware {
    pub fn new(policy: RetryPolicy) -> Self {
        Self { policy }
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
                Ok(Output::Stream(stream)) if call.mode == Mode::Stream => {
                    return Ok(Output::Stream(retry_stream(
                        stream,
                        current,
                        next,
                        self.policy,
                    )));
                }
                Ok(output) => return Ok(output),
                Err(error) if self.policy.can_retry(attempt, &error) => {
                    let delay = self.policy.delay(attempt, &error);
                    if deadline_prevents_retry(&current, delay) {
                        return Err(error);
                    }
                    sleep(delay).await;
                    attempt = attempt.saturating_add(1);
                }
                Err(error) => return Err(error),
            }
        }
    }
}

fn retry_stream(
    stream: ResponseStream,
    call: Call,
    next: Next,
    policy: RetryPolicy,
) -> ResponseStream {
    struct State {
        stream:  ResponseStream,
        call:    Call,
        next:    Next,
        policy:  RetryPolicy,
        attempt: u32,
        visible: bool,
    }

    let state = State {
        stream,
        attempt: call.context.attempt(),
        call,
        next,
        policy,
        visible: false,
    };
    Box::pin(unfold(state, |mut state| async move {
        'read: loop {
            match state.stream.next().await {
                Some(Ok(event)) => {
                    state.visible |= event.is_visible();
                    return Some((Ok(event), state));
                }
                Some(Err(mut error)) if !state.visible => {
                    while state.policy.can_retry(state.attempt, &error) {
                        let delay = state.policy.delay(state.attempt, &error);
                        if deadline_prevents_retry(&state.call, delay) {
                            return Some((Err(error), state));
                        }
                        sleep(delay).await;
                        state.attempt = state.attempt.saturating_add(1);
                        state.call.context.set_attempt(state.attempt);
                        match state.next.clone().run(state.call.clone()).await {
                            Ok(Output::Stream(stream)) => {
                                state.stream = stream;
                                continue 'read;
                            }
                            Ok(Output::Complete(_)) => {
                                return Some((
                                    Err(Error::new(
                                        ErrorKind::Middleware,
                                        "stream retry returned a complete response",
                                    )),
                                    state,
                                ));
                            }
                            Err(next_error) => error = next_error,
                        }
                    }
                    return Some((Err(error), state));
                }
                Some(Err(error)) => return Some((Err(error), state)),
                None => return None,
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
