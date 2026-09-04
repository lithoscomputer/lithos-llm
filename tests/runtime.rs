#![cfg(feature = "runtime")]

use std::error::Error as StdError;
use std::future::pending;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures_util::StreamExt as _;
use futures_util::stream::{iter, pending as pending_stream};
use lithos_llm::adapter::{InputTokenCount, ProviderAdapter, ResolvedCall};
use lithos_llm::catalog::{AdapterId, Catalog, CatalogError, ModelId, ProviderId};
use lithos_llm::client::{ClientBuildError, ProviderBuildCause};
use lithos_llm::middleware::{
    Call, CallContext, CallOutcome, ConcurrencyLimitMiddleware, Middleware, Next, Observer,
    ObserverMiddleware, Output, RetryMiddleware, RetryPolicy, RetryStage,
};
use lithos_llm::types::{
    ContentBlockId, ContentBlockKind, ContentPart, Error, ErrorKind, ImageContent, MediaSource,
    Message, RequestBuildError, Response, ResponseStream, RetryClassification, Role, StreamEvent,
    TokenCounts, ToolChoice, ToolDefinition,
};
use lithos_llm::{Client, Request};
use tokio::spawn;
use tokio::task::yield_now;
use tokio::time::timeout;

const TEST_CATALOG: &str = r#"
schema_version = 1

[providers.test]
display_name = "Test"
adapter = "test-adapter"
codec = "test-codec"
base_url = "http://127.0.0.1"
default_model = "model"

[providers.test.auth]
type = "none"

[providers.test.models.model]
display_name = "Test model"
api_model = "model"
capabilities = { text = true }
"#;

struct FakeAdapter {
    id:                   AdapterId,
    complete_calls:       Arc<AtomicUsize>,
    complete_failures:    usize,
    stream_calls:         Arc<AtomicUsize>,
    /// How many `stream` calls fail to open before one returns a stream.
    stream_open_failures: usize,
    stream_fails_initial: bool,
    stream_fails_visible: bool,
}

impl FakeAdapter {
    fn successful() -> Self {
        Self {
            id:                   AdapterId::new("test-adapter"),
            complete_calls:       Arc::new(AtomicUsize::new(0)),
            complete_failures:    0,
            stream_calls:         Arc::new(AtomicUsize::new(0)),
            stream_open_failures: 0,
            stream_fails_initial: false,
            stream_fails_visible: false,
        }
    }
}

#[async_trait]
impl ProviderAdapter for FakeAdapter {
    fn id(&self) -> &AdapterId {
        &self.id
    }

    async fn complete(&self, call: &ResolvedCall) -> Result<Response, Error> {
        let call_index = self.complete_calls.fetch_add(1, Ordering::SeqCst);
        if call_index < self.complete_failures {
            return Err(retryable_error());
        }
        Ok(success_response(call, "done"))
    }

    async fn stream(&self, call: &ResolvedCall) -> Result<ResponseStream, Error> {
        let call_index = self.stream_calls.fetch_add(1, Ordering::SeqCst);
        if call_index < self.stream_open_failures {
            return Err(retryable_error());
        }
        let events = if self.stream_fails_initial && call_index == self.stream_open_failures {
            vec![Err(retryable_error())]
        } else if self.stream_fails_visible {
            vec![
                Ok(StreamEvent::TextDelta {
                    id:   ContentBlockId::new("block-0"),
                    text: "visible".to_owned(),
                }),
                Err(retryable_error()),
            ]
        } else {
            vec![
                Ok(StreamEvent::TextDelta {
                    id:   ContentBlockId::new("block-0"),
                    text: "done".to_owned(),
                }),
                Ok(StreamEvent::Completed {
                    response: success_response(call, "done"),
                }),
            ]
        };
        Ok(ResponseStream::new(iter(events)))
    }
}

fn retryable_error() -> Error {
    Error::new(ErrorKind::Network, "temporary failure").with_retry(RetryClassification::Safe)
}

fn success_response(call: &ResolvedCall, text: &str) -> Response {
    Response::new(
        call.route().provider().id().clone(),
        call.route().model().id().clone(),
        vec![ContentPart::Text {
            text: text.to_owned(),
        }],
    )
}

fn catalog() -> Result<Catalog, CatalogError> {
    Catalog::builder().overlay_toml(TEST_CATALOG)?.build()
}

fn request() -> Result<Request, RequestBuildError> {
    Request::builder().model("test/model").user("hello").build()
}

struct Recorder {
    name: &'static str,
    log:  Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl Middleware for Recorder {
    async fn handle(&self, call: Call, next: Next) -> Result<Output, Error> {
        self.log
            .lock()
            .expect("recorder mutex should not be poisoned")
            .push(format!("{} request", self.name));
        let result = next.run(call).await;
        self.log
            .lock()
            .expect("recorder mutex should not be poisoned")
            .push(format!("{} response", self.name));
        result
    }
}

#[tokio::test]
async fn first_middleware_is_outermost() -> Result<(), Box<dyn StdError>> {
    let log = Arc::new(Mutex::new(Vec::new()));
    let client = Client::builder()
        .catalog(catalog()?)
        .adapter("test", FakeAdapter::successful())
        .middleware(Recorder {
            name: "first",
            log:  log.clone(),
        })
        .middleware(Recorder {
            name: "second",
            log:  log.clone(),
        })
        .build()?
        .client;

    client.complete(request()?).await?;

    assert_eq!(
        *log.lock().expect("recorder mutex should not be poisoned"),
        [
            "first request",
            "second request",
            "second response",
            "first response",
        ]
    );
    Ok(())
}

struct ShortCircuit;

struct ChangeRequest {
    change_model: bool,
}

#[async_trait]
impl Middleware for ChangeRequest {
    async fn handle(&self, call: Call, next: Next) -> Result<Output, Error> {
        let call = call.map_request(|request| {
            let builder = request.into_builder();
            if self.change_model {
                builder.model("test/another").build()
            } else {
                builder.temperature(0.5).build()
            }
        })?;
        next.run(call).await
    }
}

#[tokio::test]
async fn middleware_cannot_change_routing_or_bypass_limits() -> Result<(), Box<dyn StdError>> {
    for change_model in [true, false] {
        let client = Client::builder()
            .catalog(catalog()?)
            .adapter("test", FakeAdapter::successful())
            .middleware(ChangeRequest { change_model })
            .build()?
            .client;
        let error = client
            .complete(request()?)
            .await
            .expect_err("invalid transformation");
        assert_eq!(
            error.kind(),
            if change_model {
                ErrorKind::Middleware
            } else {
                ErrorKind::InvalidRequest
            }
        );
    }
    Ok(())
}

#[async_trait]
impl Middleware for ShortCircuit {
    async fn handle(&self, call: Call, _next: Next) -> Result<Output, Error> {
        Ok(Output::Complete(Response::new(
            call.route().provider().id().clone(),
            call.route().model().id().clone(),
            vec![ContentPart::Text {
                text: "cached".to_owned(),
            }],
        )))
    }
}

#[tokio::test]
async fn middleware_can_short_circuit() -> Result<(), Box<dyn StdError>> {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut adapter = FakeAdapter::successful();
    adapter.complete_calls = calls.clone();
    let client = Client::builder()
        .catalog(catalog()?)
        .adapter("test", adapter)
        .middleware(ShortCircuit)
        .build()?
        .client;

    let response = client.complete(request()?).await?;

    assert_eq!(response.text(), "cached");
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    Ok(())
}

#[tokio::test]
async fn retry_repeats_complete_calls_on_the_same_route() -> Result<(), Box<dyn StdError>> {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut adapter = FakeAdapter::successful();
    adapter.complete_calls = calls.clone();
    adapter.complete_failures = 2;
    let client = Client::builder()
        .catalog(catalog()?)
        .adapter("test", adapter)
        .middleware(RetryMiddleware::new(
            RetryPolicy::exponential()
                .max_attempts(3)
                .initial_delay(Duration::ZERO),
        ))
        .build()?
        .client;

    let response = client.complete(request()?).await?;

    assert_eq!(response.text(), "done");
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    Ok(())
}

#[tokio::test]
async fn retry_restarts_stream_before_visible_output() -> Result<(), Box<dyn StdError>> {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut adapter = FakeAdapter::successful();
    adapter.stream_calls = calls.clone();
    adapter.stream_fails_initial = true;
    let client = Client::builder()
        .catalog(catalog()?)
        .adapter("test", adapter)
        .middleware(RetryMiddleware::new(
            RetryPolicy::exponential()
                .max_attempts(2)
                .initial_delay(Duration::ZERO),
        ))
        .build()?
        .client;

    let events = client.stream(request()?).await?.collect::<Vec<_>>().await;

    assert!(matches!(
        events.as_slice(),
        [Ok(StreamEvent::TextDelta { text, .. }), Ok(StreamEvent::Completed { .. })] if text == "done"
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    Ok(())
}

#[tokio::test]
async fn a_stream_retry_does_not_deadlock_behind_the_concurrency_limiter()
-> Result<(), Box<dyn StdError>> {
    // The failed attempt's stream holds the limiter's only permit until it is
    // dropped. The retry must release that stream before reconnecting, or the
    // reconnect waits forever on its own abandoned attempt.
    let calls = Arc::new(AtomicUsize::new(0));
    let mut adapter = FakeAdapter::successful();
    adapter.stream_calls = calls.clone();
    adapter.stream_fails_initial = true;
    let client = Client::builder()
        .catalog(catalog()?)
        .adapter("test", adapter)
        .middleware(RetryMiddleware::new(
            RetryPolicy::exponential()
                .max_attempts(2)
                .initial_delay(Duration::ZERO),
        ))
        .middleware(ConcurrencyLimitMiddleware::new(NonZeroUsize::MIN))
        .build()?
        .client;

    let stream = client.stream(request()?).await?;
    let events = timeout(Duration::from_secs(5), stream.collect::<Vec<_>>())
        .await
        .expect("the stream retry should not hang on the limiter's permit");

    assert!(matches!(
        events.as_slice(),
        [Ok(StreamEvent::TextDelta { text, .. }), Ok(StreamEvent::Completed { .. })] if text == "done"
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    Ok(())
}

/// Fails after a usage snapshot on the first call, then streams normally.
struct UsageThenFailAdapter {
    id:    AdapterId,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl ProviderAdapter for UsageThenFailAdapter {
    fn id(&self) -> &AdapterId {
        &self.id
    }

    async fn complete(&self, call: &ResolvedCall) -> Result<Response, Error> {
        Ok(success_response(call, "done"))
    }

    async fn stream(&self, _call: &ResolvedCall) -> Result<ResponseStream, Error> {
        let call_index = self.calls.fetch_add(1, Ordering::SeqCst);
        let events = if call_index == 0 {
            vec![
                Ok(StreamEvent::Usage {
                    usage: TokenCounts {
                        input: 12,
                        ..TokenCounts::default()
                    },
                }),
                Err(retryable_error()),
            ]
        } else {
            vec![
                Ok(StreamEvent::TextDelta {
                    id:   ContentBlockId::new("block-0"),
                    text: "done".to_owned(),
                }),
                Ok(StreamEvent::Completed {
                    response: success_response(_call, "done"),
                }),
            ]
        };
        Ok(ResponseStream::new(iter(events)))
    }
}

#[tokio::test]
async fn a_usage_snapshot_does_not_close_the_stream_retry_window() -> Result<(), Box<dyn StdError>>
{
    // Anthropic reports usage at message_start, before any content exists. A
    // failure right after it must still retry, and the abandoned attempt's
    // snapshot must not leak into the surviving stream.
    let calls = Arc::new(AtomicUsize::new(0));
    let client = Client::builder()
        .catalog(catalog()?)
        .adapter("test", UsageThenFailAdapter {
            id:    AdapterId::new("test-adapter"),
            calls: calls.clone(),
        })
        .middleware(RetryMiddleware::new(
            RetryPolicy::exponential()
                .max_attempts(2)
                .initial_delay(Duration::ZERO),
        ))
        .build()?
        .client;

    let events = client.stream(request()?).await?.collect::<Vec<_>>().await;

    assert!(matches!(
        events.as_slice(),
        [Ok(StreamEvent::TextDelta { text, .. }), Ok(StreamEvent::Completed { .. })] if text == "done"
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    Ok(())
}

#[tokio::test]
async fn retry_does_not_replay_after_visible_output() -> Result<(), Box<dyn StdError>> {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut adapter = FakeAdapter::successful();
    adapter.stream_calls = calls.clone();
    adapter.stream_fails_visible = true;
    let client = Client::builder()
        .catalog(catalog()?)
        .adapter("test", adapter)
        .middleware(RetryMiddleware::new(
            RetryPolicy::exponential()
                .max_attempts(3)
                .initial_delay(Duration::ZERO),
        ))
        .build()?
        .client;

    let events = client.stream(request()?).await?.collect::<Vec<_>>().await;

    assert_eq!(events.len(), 2);
    assert!(events[0].is_ok());
    assert!(events[1].is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    Ok(())
}

fn throttled_error(retry_after: Duration) -> Error {
    Error::new(ErrorKind::RateLimit, "slow down")
        .with_retry(RetryClassification::after(retry_after))
}

/// An adapter whose complete calls fail with a fixed error until the given
/// attempt succeeds.
struct ThrottledAdapter {
    id:       AdapterId,
    calls:    Arc<AtomicUsize>,
    failures: usize,
    error:    Arc<dyn Fn() -> Error + Send + Sync>,
}

#[async_trait]
impl ProviderAdapter for ThrottledAdapter {
    fn id(&self) -> &AdapterId {
        &self.id
    }

    async fn complete(&self, call: &ResolvedCall) -> Result<Response, Error> {
        let call_index = self.calls.fetch_add(1, Ordering::SeqCst);
        if call_index < self.failures {
            return Err((self.error)());
        }
        Ok(success_response(call, "done"))
    }

    async fn stream(&self, _call: &ResolvedCall) -> Result<ResponseStream, Error> {
        Err(Error::new(ErrorKind::Middleware, "not used"))
    }
}

fn throttled_client(
    calls: &Arc<AtomicUsize>,
    failures: usize,
    error: impl Fn() -> Error + Send + Sync + 'static,
    policy: RetryPolicy,
) -> Result<Client, Box<dyn StdError>> {
    Ok(Client::builder()
        .catalog(catalog()?)
        .adapter("test", ThrottledAdapter {
            id: AdapterId::new("test-adapter"),
            calls: calls.clone(),
            failures,
            error: Arc::new(error),
        })
        .middleware(RetryMiddleware::new(policy))
        .build()?
        .client)
}

#[tokio::test]
async fn retry_waits_the_exact_retry_after_within_the_cap() -> Result<(), Box<dyn StdError>> {
    let calls = Arc::new(AtomicUsize::new(0));
    let client = throttled_client(
        &calls,
        1,
        || throttled_error(Duration::from_millis(60)),
        // A backoff cap far below the header value: the header wins.
        RetryPolicy::exponential()
            .max_attempts(2)
            .initial_delay(Duration::ZERO)
            .max_delay(Duration::from_millis(1)),
    )?;

    let started = Instant::now();
    let response = client.complete(request()?).await?;

    assert_eq!(response.text(), "done");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(
        started.elapsed() >= Duration::from_millis(60),
        "the exact Retry-After value should be honored"
    );
    Ok(())
}

#[tokio::test]
async fn retry_stops_when_retry_after_exceeds_the_cap() -> Result<(), Box<dyn StdError>> {
    let calls = Arc::new(AtomicUsize::new(0));
    let client = throttled_client(
        &calls,
        5,
        || throttled_error(Duration::from_secs(120)),
        RetryPolicy::exponential()
            .max_attempts(5)
            .initial_delay(Duration::ZERO)
            .retry_after_cap(Duration::from_secs(60)),
    )?;

    let error = client
        .complete(request()?)
        .await
        .expect_err("a long Retry-After should end the retries");

    assert_eq!(error.kind(), ErrorKind::RateLimit);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    Ok(())
}

/// Records every retry the retry middleware reports.
#[derive(Default)]
struct RetryRecorder {
    retries: Mutex<Vec<RecordedRetry>>,
}

#[derive(Clone, Debug, PartialEq)]
struct RecordedRetry {
    attempt:         u32,
    context_attempt: u32,
    delay:           Duration,
    kind:            ErrorKind,
    stage:           RetryStage,
}

impl RetryRecorder {
    fn retries(&self) -> Vec<RecordedRetry> {
        self.retries
            .lock()
            .expect("recorder mutex should not be poisoned")
            .clone()
    }
}

impl Observer for RetryRecorder {
    fn on_retry(
        &self,
        call: &Call,
        error: &Error,
        attempt: u32,
        delay: Duration,
        stage: RetryStage,
    ) {
        self.retries
            .lock()
            .expect("recorder mutex should not be poisoned")
            .push(RecordedRetry {
                attempt,
                context_attempt: call.context().attempt(),
                delay,
                kind: error.kind(),
                stage,
            });
    }
}

#[tokio::test]
async fn retry_reports_each_retried_attempt_to_the_observer() -> Result<(), Box<dyn StdError>> {
    let recorder = Arc::new(RetryRecorder::default());
    let mut adapter = FakeAdapter::successful();
    adapter.complete_failures = 2;
    let client = Client::builder()
        .catalog(catalog()?)
        .adapter("test", adapter)
        .middleware(
            RetryMiddleware::new(
                RetryPolicy::exponential()
                    .max_attempts(3)
                    .initial_delay(Duration::from_millis(1)),
            )
            .observer_arc(recorder.clone()),
        )
        .build()?
        .client;

    let response = client.complete(request()?).await?;

    assert_eq!(response.text(), "done");
    assert_eq!(recorder.retries(), [
        RecordedRetry {
            attempt:         1,
            context_attempt: 1,
            delay:           Duration::from_millis(1),
            kind:            ErrorKind::Network,
            stage:           RetryStage::Request,
        },
        RecordedRetry {
            attempt:         2,
            context_attempt: 2,
            delay:           Duration::from_millis(2),
            kind:            ErrorKind::Network,
            stage:           RetryStage::Request,
        },
    ]);
    Ok(())
}

#[tokio::test]
async fn a_pre_visible_stream_retry_is_reported() -> Result<(), Box<dyn StdError>> {
    let recorder = Arc::new(RetryRecorder::default());
    let mut adapter = FakeAdapter::successful();
    adapter.stream_fails_initial = true;
    let client = Client::builder()
        .catalog(catalog()?)
        .adapter("test", adapter)
        .middleware(
            RetryMiddleware::new(
                RetryPolicy::exponential()
                    .max_attempts(2)
                    .initial_delay(Duration::ZERO),
            )
            .observer_arc(recorder.clone()),
        )
        .build()?
        .client;

    let events = client.stream(request()?).await?.collect::<Vec<_>>().await;

    assert!(matches!(
        events.as_slice(),
        [Ok(StreamEvent::TextDelta { text, .. }), Ok(StreamEvent::Completed { .. })] if text == "done"
    ));
    assert_eq!(recorder.retries(), [RecordedRetry {
        attempt:         1,
        context_attempt: 1,
        delay:           Duration::ZERO,
        kind:            ErrorKind::Network,
        stage:           RetryStage::Stream,
    }]);
    Ok(())
}

#[tokio::test]
async fn a_retry_names_the_stage_that_failed() -> Result<(), Box<dyn StdError>> {
    let recorder = Arc::new(RetryRecorder::default());
    let mut adapter = FakeAdapter::successful();
    // The first call never opens a stream, the second opens one that fails
    // before any visible event, and the third streams the answer.
    adapter.stream_open_failures = 1;
    adapter.stream_fails_initial = true;
    let client = Client::builder()
        .catalog(catalog()?)
        .adapter("test", adapter)
        .middleware(
            RetryMiddleware::new(
                RetryPolicy::exponential()
                    .max_attempts(3)
                    .initial_delay(Duration::ZERO),
            )
            .observer_arc(recorder.clone()),
        )
        .build()?
        .client;

    let events = client.stream(request()?).await?.collect::<Vec<_>>().await;

    assert!(matches!(
        events.as_slice(),
        [Ok(StreamEvent::TextDelta { text, .. }), Ok(StreamEvent::Completed { .. })] if text == "done"
    ));
    assert_eq!(recorder.retries(), [
        RecordedRetry {
            attempt:         1,
            context_attempt: 1,
            delay:           Duration::ZERO,
            kind:            ErrorKind::Network,
            stage:           RetryStage::Request,
        },
        RecordedRetry {
            attempt:         2,
            context_attempt: 2,
            delay:           Duration::ZERO,
            kind:            ErrorKind::Network,
            stage:           RetryStage::Stream,
        },
    ]);
    Ok(())
}

#[tokio::test]
async fn a_refused_retry_is_not_reported() -> Result<(), Box<dyn StdError>> {
    let recorder = Arc::new(RetryRecorder::default());
    let calls = Arc::new(AtomicUsize::new(0));
    let client = Client::builder()
        .catalog(catalog()?)
        .adapter("test", ThrottledAdapter {
            id:       AdapterId::new("test-adapter"),
            calls:    calls.clone(),
            failures: 5,
            error:    Arc::new(|| throttled_error(Duration::from_secs(120))),
        })
        .middleware(
            RetryMiddleware::new(
                RetryPolicy::exponential()
                    .max_attempts(5)
                    .initial_delay(Duration::ZERO)
                    .retry_after_cap(Duration::from_secs(60)),
            )
            .observer_arc(recorder.clone()),
        )
        .build()?
        .client;

    let error = client
        .complete(request()?)
        .await
        .expect_err("a long Retry-After should end the retries");

    assert_eq!(error.kind(), ErrorKind::RateLimit);
    assert!(
        recorder.retries().is_empty(),
        "a refused retry reaches the caller as an error, not the observer"
    );
    Ok(())
}

#[test]
fn an_external_retry_driver_reads_the_policy_delay() {
    let policy = RetryPolicy::exponential()
        .initial_delay(Duration::from_millis(10))
        .max_delay(Duration::from_millis(15));
    let fatal =
        Error::new(ErrorKind::Authentication, "bad key").with_retry(RetryClassification::Never);

    assert_eq!(
        policy.next_delay(1, &retryable_error()),
        Some(Duration::from_millis(10))
    );
    assert_eq!(
        policy.next_delay(2, &retryable_error()),
        Some(Duration::from_millis(15)),
        "the computed delay is capped"
    );
    assert_eq!(
        policy.next_delay(3, &retryable_error()),
        None,
        "the default budget is three attempts"
    );
    assert_eq!(policy.next_delay(1, &fatal), None);
}

/// An adapter whose stream sends bookkeeping before it fails, then succeeds.
struct BookkeepingAdapter {
    id:    AdapterId,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl ProviderAdapter for BookkeepingAdapter {
    fn id(&self) -> &AdapterId {
        &self.id
    }

    async fn complete(&self, _call: &ResolvedCall) -> Result<Response, Error> {
        Err(Error::new(ErrorKind::Middleware, "not used"))
    }

    async fn stream(&self, _call: &ResolvedCall) -> Result<ResponseStream, Error> {
        let call_index = self.calls.fetch_add(1, Ordering::SeqCst);
        let mut events = vec![
            Ok(StreamEvent::Started {
                id: Some(format!("resp-{call_index}")),
            }),
            Ok(StreamEvent::ContentBlockStart {
                id:   ContentBlockId::new("block-0"),
                kind: ContentBlockKind::Text,
            }),
        ];
        if call_index == 0 {
            events.push(Err(retryable_error()));
        } else {
            events.push(Ok(StreamEvent::TextDelta {
                id:   ContentBlockId::new("block-0"),
                text: "done".to_owned(),
            }));
            events.push(Ok(StreamEvent::Completed {
                response: success_response(_call, "done"),
            }));
        }
        Ok(ResponseStream::new(iter(events)))
    }
}

#[tokio::test]
async fn a_retried_stream_starts_once() -> Result<(), Box<dyn StdError>> {
    let calls = Arc::new(AtomicUsize::new(0));
    let client = Client::builder()
        .catalog(catalog()?)
        .adapter("test", BookkeepingAdapter {
            id:    AdapterId::new("test-adapter"),
            calls: calls.clone(),
        })
        .middleware(RetryMiddleware::new(
            RetryPolicy::exponential()
                .max_attempts(2)
                .initial_delay(Duration::ZERO),
        ))
        .build()?
        .client;

    let events = client.stream(request()?).await?.collect::<Vec<_>>().await;

    assert_eq!(calls.load(Ordering::SeqCst), 2);
    let starts = events
        .iter()
        .filter(|event| matches!(event, Ok(StreamEvent::Started { .. })))
        .count();
    let block_starts = events
        .iter()
        .filter(|event| matches!(event, Ok(StreamEvent::ContentBlockStart { .. })))
        .count();
    assert_eq!(starts, 1, "one logical stream, one Started");
    assert_eq!(block_starts, 1, "a block starts once");
    // The surviving Started is the attempt the consumer actually reads.
    assert!(matches!(
        events.first(),
        Some(Ok(StreamEvent::Started { id })) if id.as_deref() == Some("resp-1")
    ));
    assert!(matches!(
        events.last(),
        Some(Ok(StreamEvent::Completed { response })) if response.text() == "done"
    ));
    Ok(())
}

#[tokio::test]
async fn a_call_timeout_is_never_retried() -> Result<(), Box<dyn StdError>> {
    let calls = Arc::new(AtomicUsize::new(0));
    let client = Client::builder()
        .catalog(catalog()?)
        .adapter("test", CountingPendingAdapter {
            id:    AdapterId::new("test-adapter"),
            calls: calls.clone(),
        })
        .middleware(RetryMiddleware::new(
            RetryPolicy::exponential()
                .max_attempts(3)
                .initial_delay(Duration::ZERO),
        ))
        .default_timeout(Duration::from_millis(5))
        .build()?
        .client;

    let error = client
        .complete(request()?)
        .await
        .expect_err("the call should time out");

    assert_eq!(error.kind(), ErrorKind::Timeout);
    assert_eq!(error.retry_classification(), RetryClassification::Never);
    assert_eq!(calls.load(Ordering::SeqCst), 1, "a timeout is not repeated");
    Ok(())
}

struct CountingPendingAdapter {
    id:    AdapterId,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl ProviderAdapter for CountingPendingAdapter {
    fn id(&self) -> &AdapterId {
        &self.id
    }

    async fn complete(&self, _call: &ResolvedCall) -> Result<Response, Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        pending().await
    }

    async fn stream(&self, _call: &ResolvedCall) -> Result<ResponseStream, Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(ResponseStream::new(pending_stream()))
    }
}

#[test]
fn timeout_knobs_are_configurable() -> Result<(), Box<dyn StdError>> {
    let build = Client::builder()
        .catalog(catalog()?)
        .connect_timeout(Some(Duration::from_secs(5)))
        .stream_idle_timeout(None)
        .adapter("test", FakeAdapter::successful())
        .build()?;

    assert!(build.client.available_providers().iter().next().is_some());
    Ok(())
}

#[test]
fn public_identity_types_remain_open() {
    assert_eq!(ProviderId::new("custom").as_str(), "custom");
    assert_eq!(ModelId::new("custom-model").as_str(), "custom-model");
}

#[cfg(all(
    feature = "builtin-catalog",
    feature = "environment-credentials",
    any(
        feature = "openai",
        feature = "anthropic",
        feature = "gemini",
        feature = "openai-compatible",
        feature = "bedrock"
    )
))]
#[test]
fn from_env_builds_without_reading_credentials() -> Result<(), Box<dyn StdError>> {
    let build = Client::from_env()?;

    assert!(build.client.available_providers().iter().next().is_some());
    // The built-in catalog names every built-in provider, including ones whose
    // adapter feature this build disables. Those are reported rather than
    // hidden, so the contract is that no OTHER kind of issue appears.
    for issue in &build.issues {
        assert!(
            matches!(
                issue.cause,
                ProviderBuildCause::AdapterFeatureDisabled { .. }
            ),
            "unexpected provider build issue: {issue:?}"
        );
    }
    Ok(())
}

#[test]
fn unknown_adapter_factory_becomes_a_provider_issue() -> Result<(), Box<dyn StdError>> {
    let source = TEST_CATALOG.replace("test-adapter", "custom-protocol");
    let catalog = Catalog::builder().overlay_toml(&source)?.build()?;

    let build = Client::builder().catalog(catalog).build()?;

    assert!(build.client.available_providers().iter().next().is_none());
    assert!(matches!(
        build.issues.as_slice(),
        [issue] if issue.provider.as_str() == "test"
            && issue.adapter.as_str() == "custom-protocol"
            && matches!(&issue.cause, ProviderBuildCause::MissingAdapterFactory { .. })
    ));
    Ok(())
}

#[test]
fn a_missing_catalog_is_still_a_fatal_build_error() {
    let result = Client::builder().build();

    assert!(matches!(result, Err(ClientBuildError::MissingCatalog)));
}

#[tokio::test]
async fn catalog_capabilities_reject_unsupported_content() -> Result<(), Box<dyn StdError>> {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut adapter = FakeAdapter::successful();
    adapter.complete_calls = calls.clone();
    let client = Client::builder()
        .catalog(catalog()?)
        .adapter("test", adapter)
        .build()?
        .client;
    let request = Request::builder()
        .model("test/model")
        .message(Message::new(Role::User, [ContentPart::Image(
            ImageContent::new(MediaSource::url("https://example.com/image.png")),
        )]))
        .build()?;

    let error = client
        .complete(request)
        .await
        .expect_err("the catalog does not permit images");

    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    Ok(())
}

/// A catalog whose model takes tools but no forced tool choice, as Claude
/// Fable 5.1 does.
const NO_FORCED_CHOICE_CATALOG: &str = r#"
schema_version = 1

[providers.test]
display_name = "Test"
adapter = "test-adapter"
codec = "test-codec"
base_url = "http://127.0.0.1"
default_model = "model"

[providers.test.auth]
type = "none"

[providers.test.models.model]
display_name = "Test model"
api_model = "model"
capabilities = { text = true, tools = true, tool_choice = { required = false, named = false } }
"#;

#[tokio::test]
async fn a_forced_tool_choice_needs_the_capability() -> Result<(), Box<dyn StdError>> {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut adapter = FakeAdapter::successful();
    adapter.complete_calls = calls.clone();
    let client = Client::builder()
        .catalog(
            Catalog::builder()
                .overlay_toml(NO_FORCED_CHOICE_CATALOG)?
                .build()?,
        )
        .adapter("test", adapter)
        .build()?
        .client;
    let with_choice = |choice: ToolChoice| {
        Request::builder()
            .model("test/model")
            .user("What is the weather in Paris?")
            .tool(ToolDefinition::function(
                "get_weather",
                "Reads the current weather for a city",
                serde_json::json!({ "type": "object" }),
            ))
            .tool_choice(choice)
            .build()
    };

    // `required` and a named tool both force a call the model cannot take,
    // so both are refused before the adapter sees them.
    for choice in [ToolChoice::Required, ToolChoice::Tool {
        name: "get_weather".to_owned(),
    }] {
        let error = client
            .complete(with_choice(choice)?)
            .await
            .expect_err("the catalog denies forced tool choice");
        assert_eq!(error.kind(), ErrorKind::InvalidRequest);
        assert_eq!(error.provider_code(), Some("unsupported_capability"));
        assert!(error.message().contains("forced tool choice"));
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    // `auto` and `none` leave the model free to answer, so they still go out.
    for choice in [ToolChoice::Auto, ToolChoice::None] {
        client.complete(with_choice(choice)?).await?;
    }
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    Ok(())
}

#[tokio::test]
async fn an_explicit_cache_hint_needs_the_routing_capability() -> Result<(), Box<dyn StdError>> {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut adapter = FakeAdapter::successful();
    adapter.complete_calls = calls.clone();
    let client = Client::builder()
        .catalog(catalog()?)
        .adapter("test", adapter)
        .build()?
        .client;
    let request = Request::builder()
        .model("test/model")
        .user("Hello")
        .cache_key("tenant-42")
        .build()?;

    let error = client
        .complete(request)
        .await
        .expect_err("the catalog does not claim cache routing");

    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    Ok(())
}

struct PendingAdapter {
    id: AdapterId,
}

struct PendingMiddleware;

struct OutcomeObserver(Arc<Mutex<Vec<&'static str>>>);

impl Observer for OutcomeObserver {
    fn on_finish(&self, _call: &Call, outcome: CallOutcome<'_>) {
        use CallOutcome;
        self.0.lock().expect("outcomes").push(match outcome {
            CallOutcome::Response(_) => "response",
            CallOutcome::InputTokenCount(_) => "count",
            CallOutcome::Failed(_) => "failed",
            CallOutcome::Cancelled => "cancelled",
            CallOutcome::Dropped => "dropped",
            _ => "other",
        });
    }
}

#[tokio::test]
async fn observer_finishes_complete_stream_and_count_once() -> Result<(), Box<dyn StdError>> {
    let outcomes = Arc::new(Mutex::new(Vec::new()));
    let client = Client::builder()
        .catalog(catalog()?)
        .adapter("test", FakeAdapter::successful())
        .middleware(ObserverMiddleware::new(OutcomeObserver(outcomes.clone())))
        .build()?
        .client;
    client.complete(request()?).await?;
    let mut stream = client.stream(request()?).await?;
    while stream.next().await.is_some() {}
    assert_eq!(*outcomes.lock().expect("outcomes"), [
        "response", "response"
    ]);
    drop(stream);
    assert!(client.count_input_tokens(request()?).await?.is_none());
    assert_eq!(*outcomes.lock().expect("outcomes"), [
        "response", "response", "count"
    ]);
    Ok(())
}

#[tokio::test]
async fn observer_finishes_dropped_and_cancelled_calls_once() -> Result<(), Box<dyn StdError>> {
    let outcomes = Arc::new(Mutex::new(Vec::new()));
    let client = Client::builder()
        .catalog(catalog()?)
        .adapter("test", PendingAdapter {
            id: AdapterId::new("test-adapter"),
        })
        .middleware(ObserverMiddleware::new(OutcomeObserver(outcomes.clone())))
        .build()?
        .client;
    let mut future = Box::pin(client.complete(request()?));
    assert!(futures_util::poll!(&mut future).is_pending());
    drop(future);
    let stream = client.stream(request()?).await?;
    drop(stream);
    let context = CallContext::new();
    let cancel = context.cancellation().clone();
    let mut stream = client.stream_with_context(request()?, context).await?;
    cancel.cancel();
    assert!(stream.next().await.expect("cancelled").is_err());
    drop(stream);
    let timed = request()?
        .into_builder()
        .timeout(Duration::from_millis(5))
        .build()?;
    assert!(client.complete(timed).await.is_err());
    assert_eq!(*outcomes.lock().expect("outcomes"), [
        "dropped",
        "dropped",
        "cancelled",
        "failed"
    ]);
    Ok(())
}

struct CountingTokensAdapter {
    id:    AdapterId,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl ProviderAdapter for CountingTokensAdapter {
    fn id(&self) -> &AdapterId {
        &self.id
    }
    async fn complete(&self, _call: &ResolvedCall) -> Result<Response, Error> {
        unreachable!("count only")
    }
    async fn stream(&self, _call: &ResolvedCall) -> Result<ResponseStream, Error> {
        unreachable!("count only")
    }
    async fn count_input_tokens(
        &self,
        call: &ResolvedCall,
    ) -> Result<Option<InputTokenCount>, Error> {
        assert_eq!(call.context().extensions().get::<u32>(), Some(&42));
        assert!(call.context().deadline().is_some());
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            return Err(retryable_error());
        }
        Ok(Some(InputTokenCount::new(12, call.route().handle())))
    }
}

#[tokio::test]
async fn token_counting_uses_context_and_retry_middleware() -> Result<(), Box<dyn StdError>> {
    let calls = Arc::new(AtomicUsize::new(0));
    let log = Arc::new(Mutex::new(Vec::new()));
    let client = Client::builder()
        .catalog(catalog()?)
        .adapter("test", CountingTokensAdapter {
            id:    AdapterId::new("test-adapter"),
            calls: calls.clone(),
        })
        .middleware(Recorder {
            name: "count",
            log:  log.clone(),
        })
        .middleware(RetryMiddleware::new(
            RetryPolicy::exponential().initial_delay(Duration::ZERO),
        ))
        .default_timeout(Duration::from_secs(2))
        .build()?
        .client;
    let mut context = CallContext::new();
    context.extensions_mut().insert(42_u32);
    let count = client
        .count_input_tokens_with_context(request()?, context)
        .await?
        .expect("native count");
    assert_eq!(count.tokens(), 12);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(*log.lock().expect("log"), [
        "count request",
        "count response"
    ]);
    Ok(())
}

#[async_trait]
impl Middleware for PendingMiddleware {
    async fn handle(&self, _call: Call, _next: Next) -> Result<Output, Error> {
        pending().await
    }
}

#[tokio::test]
async fn request_budget_includes_middleware_waits() -> Result<(), Box<dyn StdError>> {
    let client = Client::builder()
        .catalog(catalog()?)
        .adapter("test", FakeAdapter::successful())
        .middleware(PendingMiddleware)
        .default_timeout(Duration::from_secs(60))
        .build()?
        .client;
    let request = request()?
        .into_builder()
        .timeout(Duration::from_millis(5))
        .build()?;
    let error = timeout(Duration::from_secs(1), client.complete(request))
        .await
        .expect("request budget must bound middleware")
        .expect_err("budget expires");
    assert_eq!(error.kind(), ErrorKind::Timeout);
    Ok(())
}

#[async_trait]
impl ProviderAdapter for PendingAdapter {
    fn id(&self) -> &AdapterId {
        &self.id
    }

    async fn complete(&self, _call: &ResolvedCall) -> Result<Response, Error> {
        pending().await
    }

    async fn stream(&self, _call: &ResolvedCall) -> Result<ResponseStream, Error> {
        Ok(ResponseStream::new(pending_stream()))
    }
}

fn pending_client() -> Result<Client, Box<dyn StdError>> {
    Ok(Client::builder()
        .catalog(catalog()?)
        .adapter("test", PendingAdapter {
            id: AdapterId::new("test-adapter"),
        })
        .build()?
        .client)
}

#[tokio::test]
async fn cancellation_stops_an_active_complete_call() -> Result<(), Box<dyn StdError>> {
    let client = pending_client()?;
    let context = CallContext::new();
    let cancellation = context.cancellation().clone();
    let request = request()?;
    let task = spawn(async move { client.complete_with_context(request, context).await });
    yield_now().await;

    cancellation.cancel();
    let error = task.await?.expect_err("the call should be cancelled");

    assert_eq!(error.kind(), ErrorKind::Cancelled);
    Ok(())
}

#[tokio::test]
async fn cancellation_ends_an_active_stream() -> Result<(), Box<dyn StdError>> {
    let client = pending_client()?;
    let context = CallContext::new();
    let cancellation = context.cancellation().clone();
    let mut stream = client.stream_with_context(request()?, context).await?;

    cancellation.cancel();
    let error = stream
        .next()
        .await
        .expect("cancellation should produce one event")
        .expect_err("the event should be an error");

    assert_eq!(error.kind(), ErrorKind::Cancelled);
    assert!(stream.next().await.is_none());
    Ok(())
}

#[tokio::test]
async fn context_deadline_stops_an_active_call() -> Result<(), Box<dyn StdError>> {
    let client = pending_client()?;
    let mut context = CallContext::new();
    context.set_deadline(Instant::now() + Duration::from_millis(5));

    let error = client
        .complete_with_context(request()?, context)
        .await
        .expect_err("the deadline should expire");

    assert_eq!(error.kind(), ErrorKind::Timeout);
    Ok(())
}

#[tokio::test]
async fn timeout_stream_emits_one_terminal_error() -> Result<(), Box<dyn StdError>> {
    let client = Client::builder()
        .catalog(catalog()?)
        .adapter("test", PendingAdapter {
            id: AdapterId::new("test-adapter"),
        })
        .default_timeout(Duration::from_millis(5))
        .build()?
        .client;
    let mut stream = client.stream(request()?).await?;

    let error = stream
        .next()
        .await
        .expect("the timeout should produce one event")
        .expect_err("the event should be an error");

    assert_eq!(error.kind(), ErrorKind::Timeout);
    assert!(stream.next().await.is_none());
    Ok(())
}
