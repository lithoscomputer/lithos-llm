#![cfg(feature = "runtime")]

use std::collections::BTreeMap;
use std::error::Error as StdError;
use std::future::pending;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures_util::StreamExt as _;
use futures_util::stream::{iter, pending as pending_stream};
use lithos_llm::adapter::{
    AdapterBuildError, AdapterContext, AdapterFactory, InputTokenCount, ProviderAdapter,
    ResolvedCall, ResolvedEvaluation,
};
use lithos_llm::catalog::{AdapterId, Catalog, CatalogError, CatalogProvider, ModelId, ProviderId};
use lithos_llm::client::{ClientBuildError, ProviderBuildCause};
use lithos_llm::middleware::{
    Call, CallContext, CallOutcome, ConcurrencyLimitMiddleware, Middleware, Next, Observer,
    ObserverMiddleware, Operation, Output, RetryMiddleware, RetryPolicy, RetryStage, map_stream,
};
use lithos_llm::types::{
    Answer, BooleanAnswer, ChoiceAnswer, ContentBlockId, ContentBlockKind, ContentPart, Error,
    ErrorKind, FinishReason, ImageContent, MediaSource, Message, QuestionId, QuestionKind,
    RequestBuildError, Response, ResponseFormat, ResponseLimits, ResponseStream,
    RetryClassification, Role, ScoreAnswer, StreamEvent, TokenCounts, ToolArguments, ToolCall,
    ToolChoice, ToolDefinition, ToolInput,
};
use lithos_llm::{Client, Evaluation, Request, Verdict};
use serde_json::json;
use tokio::spawn;
use tokio::task::yield_now;
use tokio::time::{Instant, timeout};

const TEST_CATALOG: &str = r#"
schema_version = 1

[providers.test]
display_name = "Test"
adapter = "test-adapter"
codecs = ["test-codec"]
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
                Ok(StreamEvent::Ended {
                    response: Box::new(success_response(call, "done")),
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

#[tokio::test(start_paused = true)]
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

#[tokio::test(start_paused = true)]
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

#[tokio::test(start_paused = true)]
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

/// Supplies cache hits or transforms real adapter output after its policy ran.
struct ResponsePolicyMiddleware {
    cached:         bool,
    response_bytes: usize,
    delta_bytes:    usize,
}

#[async_trait]
impl Middleware for ResponsePolicyMiddleware {
    async fn handle(&self, call: Call, next: Next) -> Result<Output, Error> {
        let output = if self.cached {
            let response = Response::new(
                call.route().provider().id().clone(),
                call.route().model().id().clone(),
                Vec::new(),
            );
            if call.operation() == Operation::Stream {
                Output::Stream(ResponseStream::new(iter([
                    Ok(StreamEvent::TextDelta {
                        id:   ContentBlockId::new("text"),
                        text: String::new(),
                    }),
                    Ok(StreamEvent::Ended {
                        response: Box::new(response),
                    }),
                ])))
            } else {
                Output::Complete(response)
            }
        } else {
            next.run(call).await?
        };
        let response_bytes = self.response_bytes;
        let rewrite = move |mut response: Response| {
            response.content = vec![ContentPart::Text {
                text: "x".repeat(response_bytes),
            }];
            response.raw = Some(serde_json::json!({ "provider_body": "retained" }));
            response
        };
        Ok(match output {
            Output::Complete(response) => Output::Complete(rewrite(response)),
            Output::Stream(stream) => {
                let delta_bytes = self.delta_bytes;
                Output::Stream(map_stream(stream, move |event| {
                    Ok(match event {
                        StreamEvent::TextDelta { id, .. } => StreamEvent::TextDelta {
                            id,
                            text: "x".repeat(delta_bytes),
                        },
                        StreamEvent::Ended { response } => StreamEvent::Ended {
                            response: Box::new(rewrite(*response)),
                        },
                        event => event,
                    })
                }))
            }
            output @ (Output::InputTokenCount(_) | Output::Verdict(_)) => output,
        })
    }
}

fn response_policy_client(
    cached: bool,
    response_bytes: usize,
    delta_bytes: usize,
    retain_raw: bool,
) -> Result<Client, Box<dyn StdError>> {
    Ok(Client::builder()
        .catalog(catalog()?)
        .adapter("test", FakeAdapter::successful())
        .response_limits(ResponseLimits::default().max_output_bytes(1024))
        .retain_raw_response(retain_raw)
        .middleware(ResponsePolicyMiddleware {
            cached,
            response_bytes,
            delta_bytes,
        })
        .build()?
        .client)
}

#[tokio::test(start_paused = true)]
async fn response_limits_apply_to_cached_and_transformed_final_responses()
-> Result<(), Box<dyn StdError>> {
    for cached in [true, false] {
        let client = response_policy_client(cached, 4096, 4, true)?;
        let error = client.complete(request()?).await.expect_err("output limit");
        assert_eq!(error.kind(), ErrorKind::ResourceLimit);
        let mut stream = client.stream(request()?).await?;
        assert!(matches!(
            stream.next().await,
            Some(Ok(StreamEvent::TextDelta { .. }))
        ));
        let error = stream
            .next()
            .await
            .expect("terminal event")
            .expect_err("output limit");
        assert_eq!(error.kind(), ErrorKind::ResourceLimit);
        assert!(stream.next().await.is_none());
    }
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn response_limits_apply_to_cached_and_transformed_deltas() -> Result<(), Box<dyn StdError>> {
    for cached in [true, false] {
        let client = response_policy_client(cached, 4, 4096, true)?;
        let mut stream = client.stream(request()?).await?;
        let error = stream
            .next()
            .await
            .expect("terminal event")
            .expect_err("output limit");
        assert_eq!(error.kind(), ErrorKind::ResourceLimit);
        assert!(stream.next().await.is_none());
    }
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn raw_retention_applies_to_cached_and_transformed_responses() -> Result<(), Box<dyn StdError>>
{
    for cached in [true, false] {
        for retain_raw in [true, false] {
            let client = response_policy_client(cached, 4, 4, retain_raw)?;
            let response = client.complete(request()?).await?;
            assert_eq!(response.text(), "xxxx");
            assert_eq!(response.raw.is_some(), retain_raw);
            let mut stream = client.stream(request()?).await?;
            assert!(matches!(
                stream.next().await,
                Some(Ok(StreamEvent::TextDelta { .. }))
            ));
            let StreamEvent::Ended { response } =
                stream.next().await.ok_or("missing completion")??
            else {
                return Err("expected completion".into());
            };
            assert_eq!(response.text(), "xxxx");
            assert_eq!(response.raw.is_some(), retain_raw);
            assert!(stream.next().await.is_none());
        }
    }
    Ok(())
}

#[tokio::test(start_paused = true)]
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

#[tokio::test(start_paused = true)]
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
        [Ok(StreamEvent::TextDelta { text, .. }), Ok(StreamEvent::Ended { .. })] if text == "done"
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    Ok(())
}

#[tokio::test(start_paused = true)]
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
        [Ok(StreamEvent::TextDelta { text, .. }), Ok(StreamEvent::Ended { .. })] if text == "done"
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
                Ok(StreamEvent::Ended {
                    response: Box::new(success_response(_call, "done")),
                }),
            ]
        };
        Ok(ResponseStream::new(iter(events)))
    }
}

#[tokio::test(start_paused = true)]
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
        [Ok(StreamEvent::TextDelta { text, .. }), Ok(StreamEvent::Ended { .. })] if text == "done"
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    Ok(())
}

#[tokio::test(start_paused = true)]
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

#[tokio::test(start_paused = true)]
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

#[tokio::test(start_paused = true)]
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

#[tokio::test(start_paused = true)]
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

#[tokio::test(start_paused = true)]
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
        [Ok(StreamEvent::TextDelta { text, .. }), Ok(StreamEvent::Ended { .. })] if text == "done"
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

#[tokio::test(start_paused = true)]
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
        [Ok(StreamEvent::TextDelta { text, .. }), Ok(StreamEvent::Ended { .. })] if text == "done"
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

#[tokio::test(start_paused = true)]
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
            events.push(Ok(StreamEvent::Ended {
                response: Box::new(success_response(_call, "done")),
            }));
        }
        Ok(ResponseStream::new(iter(events)))
    }
}

#[tokio::test(start_paused = true)]
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
        Some(Ok(StreamEvent::Ended { response })) if response.text() == "done"
    ));
    Ok(())
}

/// Streams bookkeeping and closes a block, then ends without an `Ended`.
struct TruncatedAdapter {
    id:    AdapterId,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl ProviderAdapter for TruncatedAdapter {
    fn id(&self) -> &AdapterId {
        &self.id
    }

    async fn complete(&self, _call: &ResolvedCall) -> Result<Response, Error> {
        Err(Error::new(ErrorKind::Middleware, "not used"))
    }

    async fn stream(&self, _call: &ResolvedCall) -> Result<ResponseStream, Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let id = ContentBlockId::new("block-0");
        Ok(ResponseStream::new(iter(vec![
            Ok(StreamEvent::Started { id: None }),
            Ok(StreamEvent::ContentBlockStart {
                id:   id.clone(),
                kind: ContentBlockKind::Text,
            }),
            Ok(StreamEvent::ContentBlockEnd {
                id,
                part: ContentPart::Text {
                    text: String::new(),
                },
            }),
        ])))
    }
}

/// Answers every attempt after the first with a complete response, whatever
/// the operation asked for.
struct CompletesOnRetry;

#[async_trait]
impl Middleware for CompletesOnRetry {
    async fn handle(&self, call: Call, next: Next) -> Result<Output, Error> {
        if call.context().attempt() < 2 {
            return next.run(call).await;
        }
        Ok(Output::Complete(Response::new(
            call.route().provider().id().clone(),
            call.route().model().id().clone(),
            vec![ContentPart::Text {
                text: "complete".to_owned(),
            }],
        )))
    }
}

#[tokio::test(start_paused = true)]
async fn exhausted_incomplete_retries_deliver_the_original_response_not_an_error()
-> Result<(), Box<dyn StdError>> {
    // An incomplete turn is retried as if it were a failure, but when the
    // retries run out the caller gets the provider's actual answer back — the
    // partial text with its `Incomplete` finish reason — never the synthetic
    // error the retry loop classified it as. Both the complete path and the
    // pre-visible stream path keep that rule.
    for streaming in [false, true] {
        let calls = Arc::new(AtomicUsize::new(0));
        let client = Client::builder()
            .catalog(catalog()?)
            .adapter("test", UnfinishedAdapter {
                calls:               calls.clone(),
                unfinished_attempts: usize::MAX,
                reason:              FinishReason::Incomplete,
                visible:             false,
                id:                  AdapterId::new("test-adapter"),
            })
            .middleware(RetryMiddleware::new(
                RetryPolicy::exponential()
                    .max_attempts(2)
                    .initial_delay(Duration::ZERO),
            ))
            .build()?
            .client;

        let response = if streaming {
            let events = client.stream(request()?).await?.collect::<Vec<_>>().await;
            assert!(
                events.iter().all(Result::is_ok),
                "no error reaches the caller: {events:?}"
            );
            let Some(Ok(StreamEvent::Ended { response })) = events.into_iter().last() else {
                panic!("the stream ends with the provider's own response")
            };
            *response
        } else {
            client.complete(request()?).await?
        };

        assert_eq!(response.finish_reason, FinishReason::Incomplete);
        assert_eq!(response.text(), "partial", "the original body survives");
        assert_eq!(calls.load(Ordering::SeqCst), 2, "every attempt was spent");
    }
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn a_stream_retry_that_yields_a_complete_response_is_a_middleware_error()
-> Result<(), Box<dyn StdError>> {
    // A reconnect must produce a stream. A layer below the retry that answers
    // the second attempt with a complete response has broken the middleware
    // contract, and the stream reports that as a `Middleware` error rather
    // than fabricating events from the response or hanging.
    let calls = Arc::new(AtomicUsize::new(0));
    let mut adapter = FakeAdapter::successful();
    adapter.stream_calls = calls.clone();
    adapter.stream_fails_initial = true;
    let client = Client::builder()
        .catalog(catalog()?)
        .adapter("test", adapter)
        .middleware(RetryMiddleware::new(
            RetryPolicy::exponential()
                .max_attempts(3)
                .initial_delay(Duration::ZERO),
        ))
        .middleware(CompletesOnRetry)
        .build()?
        .client;

    let events = client.stream(request()?).await?.collect::<Vec<_>>().await;

    let [Err(error)] = events.as_slice() else {
        panic!("one terminal error and nothing else: {events:?}")
    };
    assert_eq!(error.kind(), ErrorKind::Middleware);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "the second attempt never reached the adapter"
    );
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn bookkeeping_held_before_visible_output_is_released_when_the_stream_ends()
-> Result<(), Box<dyn StdError>> {
    // Protocol bookkeeping is held back until the first visible event so a
    // reconnect can discard it. A stream that ends before anything visible
    // arrives, and cannot be retried, still owes the caller that bookkeeping:
    // the `ContentBlockEnd` a decoder synthesized at EOF is delivered ahead
    // of the terminal error, not dropped with the attempt.
    let calls = Arc::new(AtomicUsize::new(0));
    let client = Client::builder()
        .catalog(catalog()?)
        .adapter("test", TruncatedAdapter {
            id:    AdapterId::new("test-adapter"),
            calls: calls.clone(),
        })
        .middleware(RetryMiddleware::new(
            RetryPolicy::exponential().max_attempts(1),
        ))
        .build()?
        .client;

    let events = client.stream(request()?).await?.collect::<Vec<_>>().await;

    assert!(
        matches!(events.as_slice(), [
            Ok(StreamEvent::Started { .. }),
            Ok(StreamEvent::ContentBlockStart { .. }),
            Ok(StreamEvent::ContentBlockEnd { .. }),
            Err(_),
        ]),
        "held bookkeeping, then the terminal error: {events:?}"
    );
    let Some(Err(error)) = events.last() else {
        unreachable!("matched above")
    };
    assert_eq!(error.kind(), ErrorKind::StreamDecode);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    Ok(())
}

#[tokio::test(start_paused = true)]
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

#[cfg(all(feature = "builtin-catalog", feature = "environment-credentials"))]
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

#[tokio::test(start_paused = true)]
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
codecs = ["test-codec"]
base_url = "http://127.0.0.1"
default_model = "model"

[providers.test.auth]
type = "none"

[providers.test.models.model]
display_name = "Test model"
api_model = "model"
capabilities = { text = true, tools = true, tool_choice = { required = false, named = false } }
"#;

#[tokio::test(start_paused = true)]
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

#[tokio::test(start_paused = true)]
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

#[tokio::test(start_paused = true)]
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

#[tokio::test(start_paused = true)]
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

#[tokio::test(start_paused = true)]
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

#[tokio::test(start_paused = true)]
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

    async fn count_input_tokens(
        &self,
        _call: &ResolvedCall,
    ) -> Result<Option<InputTokenCount>, Error> {
        pending().await
    }
}

struct SetDeadline(Duration);

#[async_trait]
impl Middleware for SetDeadline {
    async fn handle(&self, mut call: Call, next: Next) -> Result<Output, Error> {
        call.context_mut()
            .set_deadline((Instant::now() + self.0).into_std());
        next.run(call).await
    }
}

#[tokio::test(start_paused = true)]
async fn middleware_deadlines_bound_completion_and_token_counting() -> Result<(), Box<dyn StdError>>
{
    for default_timeout in [None, Some(Duration::from_secs(60))] {
        let mut builder = Client::builder()
            .catalog(catalog()?)
            .adapter("test", PendingAdapter {
                id: AdapterId::new("test-adapter"),
            })
            .middleware(SetDeadline(Duration::from_millis(5)));
        if let Some(budget) = default_timeout {
            builder = builder.default_timeout(budget);
        }
        let client = builder.build()?.client;
        let error = timeout(Duration::from_secs(1), client.complete(request()?))
            .await
            .expect("middleware deadline must stop completion")
            .expect_err("deadline expires");
        assert_eq!(error.kind(), ErrorKind::Timeout);
        let error = timeout(
            Duration::from_secs(1),
            client.count_input_tokens(request()?),
        )
        .await
        .expect("middleware deadline must stop counting")
        .expect_err("deadline expires");
        assert_eq!(error.kind(), ErrorKind::Timeout);
    }
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn middleware_deadline_bounds_downstream_stream_setup() -> Result<(), Box<dyn StdError>> {
    let client = Client::builder()
        .catalog(catalog()?)
        .adapter("test", FakeAdapter::successful())
        .middleware(SetDeadline(Duration::from_millis(5)))
        .middleware(PendingMiddleware)
        .build()?
        .client;
    let error = timeout(Duration::from_secs(1), client.stream(request()?))
        .await
        .expect("middleware deadline must stop stream setup")
        .map(|_| ())
        .expect_err("deadline expires");
    assert_eq!(error.kind(), ErrorKind::Timeout);
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn middleware_deadline_survives_stream_setup_and_later_extension()
-> Result<(), Box<dyn StdError>> {
    let outcomes = Arc::new(Mutex::new(Vec::new()));
    let client = Client::builder()
        .catalog(catalog()?)
        .adapter("test", PendingAdapter {
            id: AdapterId::new("test-adapter"),
        })
        .middleware(ObserverMiddleware::new(OutcomeObserver(outcomes.clone())))
        .middleware(SetDeadline(Duration::from_millis(20)))
        .middleware(SetDeadline(Duration::from_secs(60)))
        .build()?
        .client;
    let mut stream = client.stream(request()?).await?;
    let error = timeout(Duration::from_secs(1), stream.next())
        .await
        .expect("earlier deadline must bound consumption")
        .expect("terminal event")
        .expect_err("deadline expires");
    assert_eq!(error.kind(), ErrorKind::Timeout);
    assert!(stream.next().await.is_none());
    assert_eq!(*outcomes.lock().expect("outcomes"), ["failed"]);
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn middleware_cannot_extend_the_client_budget() -> Result<(), Box<dyn StdError>> {
    let client = Client::builder()
        .catalog(catalog()?)
        .adapter("test", PendingAdapter {
            id: AdapterId::new("test-adapter"),
        })
        .default_timeout(Duration::from_millis(5))
        .middleware(SetDeadline(Duration::from_secs(60)))
        .build()?
        .client;
    let error = timeout(Duration::from_secs(1), client.complete(request()?))
        .await
        .expect("client budget still applies")
        .expect_err("deadline expires");
    assert_eq!(error.kind(), ErrorKind::Timeout);
    Ok(())
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

#[tokio::test(start_paused = true)]
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

#[tokio::test(start_paused = true)]
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

#[tokio::test(start_paused = true)]
async fn context_deadline_stops_an_active_call() -> Result<(), Box<dyn StdError>> {
    let client = pending_client()?;
    let mut context = CallContext::new();
    context.set_deadline((Instant::now() + Duration::from_millis(5)).into_std());

    let error = client
        .complete_with_context(request()?, context)
        .await
        .expect_err("the deadline should expire");

    assert_eq!(error.kind(), ErrorKind::Timeout);
    Ok(())
}

#[tokio::test(start_paused = true)]
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

/// Supplies unfinished turns, including block ends synthesized at EOF.
struct UnfinishedAdapter {
    calls:               Arc<AtomicUsize>,
    unfinished_attempts: usize,
    reason:              FinishReason,
    visible:             bool,
    id:                  AdapterId,
}

#[async_trait]
impl ProviderAdapter for UnfinishedAdapter {
    fn id(&self) -> &AdapterId {
        &self.id
    }

    async fn complete(&self, call: &ResolvedCall) -> Result<Response, Error> {
        let attempt = self.calls.fetch_add(1, Ordering::SeqCst);
        let mut response = success_response(call, "partial");
        if attempt < self.unfinished_attempts {
            response.finish_reason = self.reason.clone();
        } else {
            response = success_response(call, "done");
        }
        Ok(response)
    }

    async fn stream(&self, call: &ResolvedCall) -> Result<ResponseStream, Error> {
        let response = self.complete(call).await?;
        let id = ContentBlockId::new("unfinished");
        let mut events = vec![Ok(StreamEvent::Started { id: None })];
        if self.visible {
            events.push(Ok(StreamEvent::TextDelta {
                id:   id.clone(),
                text: response.text(),
            }));
        }
        events.push(Ok(StreamEvent::ContentBlockEnd {
            id,
            part: ContentPart::Text {
                text: response.text(),
            },
        }));
        events.push(Ok(StreamEvent::Ended {
            response: Box::new(response),
        }));
        Ok(ResponseStream::new(iter(events)))
    }
}

#[tokio::test(start_paused = true)]
async fn incomplete_turns_retry_only_before_delivery() -> Result<(), Box<dyn StdError>> {
    for (streaming, visible, reason, failures, expected_calls, expected_reason) in [
        (
            true,
            false,
            FinishReason::Incomplete,
            1,
            2,
            FinishReason::Stop,
        ),
        (
            false,
            false,
            FinishReason::Incomplete,
            1,
            2,
            FinishReason::Stop,
        ),
        (
            true,
            true,
            FinishReason::Incomplete,
            1,
            1,
            FinishReason::Incomplete,
        ),
        (
            true,
            false,
            FinishReason::Incomplete,
            10,
            3,
            FinishReason::Incomplete,
        ),
        (
            false,
            false,
            FinishReason::Incomplete,
            10,
            3,
            FinishReason::Incomplete,
        ),
        (
            true,
            false,
            FinishReason::Length,
            10,
            1,
            FinishReason::Length,
        ),
        (
            false,
            false,
            FinishReason::Length,
            10,
            1,
            FinishReason::Length,
        ),
    ] {
        let calls = Arc::new(AtomicUsize::new(0));
        let observer = RetryRecorder::default();
        let observations = Arc::new(observer);
        let client = Client::builder()
            .catalog(catalog()?)
            .adapter("test", UnfinishedAdapter {
                calls: calls.clone(),
                unfinished_attempts: failures,
                reason,
                visible,
                id: AdapterId::new("test-adapter"),
            })
            .middleware(
                RetryMiddleware::new(RetryPolicy::exponential().max_attempts(3))
                    .observer_arc(observations.clone()),
            )
            .middleware(ConcurrencyLimitMiddleware::new(NonZeroUsize::MIN))
            .build()?
            .client;
        let start = Instant::now();
        let response = if streaming {
            let events = client
                .stream(request()?)
                .await?
                .collect::<Vec<_>>()
                .await
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?;
            assert_eq!(
                events
                    .iter()
                    .filter(|e| matches!(e, StreamEvent::Started { .. }))
                    .count(),
                1
            );
            assert_eq!(
                events
                    .iter()
                    .filter(|e| matches!(e, StreamEvent::Ended { .. }))
                    .count(),
                1
            );
            let Some(StreamEvent::Ended { response }) = events.into_iter().last() else {
                panic!("terminal response missing")
            };
            response
        } else {
            Box::new(client.complete(request()?).await?)
        };
        assert_eq!(response.finish_reason, expected_reason);
        assert_eq!(calls.load(Ordering::SeqCst), expected_calls);
        let retries = observations.retries();
        assert_eq!(retries.len(), expected_calls - 1);
        for (index, retry) in retries.iter().enumerate() {
            assert_eq!(retry.attempt, u32::try_from(index + 1)?);
            assert_eq!(
                retry.stage,
                if streaming {
                    RetryStage::Stream
                } else {
                    RetryStage::Request
                }
            );
        }
        assert_eq!(start.elapsed(), match expected_calls {
            1 => Duration::ZERO,
            2 => Duration::from_millis(100),
            _ => Duration::from_millis(300),
        });
    }
    Ok(())
}

#[tokio::test]
async fn unknown_transcript_content_is_rejected_before_dispatch() -> Result<(), Box<dyn StdError>> {
    let adapter = FakeAdapter::successful();
    let calls = adapter.complete_calls.clone();
    let streams = adapter.stream_calls.clone();
    let client = Client::builder()
        .catalog(catalog()?)
        .adapter("test", adapter)
        .build()?
        .client;
    for content in [
        json!({"type":"future_image","pixels":[1,2]}),
        json!({"type":"tool_result","tool_call_id":"call_1","content":[
            {"type":"tool_result","tool_call_id":"nested","content":[
                {"type":"future_image","pixels":[1,2]}
            ]}
        ]}),
    ] {
        let message: Message = serde_json::from_value(json!({"role":"user","content":[content]}))?;
        let request = Request::builder()
            .model("test/model")
            .message(message)
            .build()?;
        let error = client
            .complete(request.clone())
            .await
            .expect_err("unknown content refused");
        assert_eq!(error.kind(), ErrorKind::InvalidRequest);
        assert_eq!(error.provider_code(), Some("unknown_content_type"));
        assert!(client.stream(request.clone()).await.is_err());
        assert!(client.count_input_tokens(request).await.is_err());
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(streams.load(Ordering::SeqCst), 0);
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn incomplete_retry_returns_the_partial_response_when_backoff_exceeds_deadline()
-> Result<(), Box<dyn StdError>> {
    for streaming in [false, true] {
        let calls = Arc::new(AtomicUsize::new(0));
        let observations = Arc::new(RetryRecorder::default());
        let client = Client::builder()
            .catalog(catalog()?)
            .adapter("test", UnfinishedAdapter {
                calls:               calls.clone(),
                unfinished_attempts: usize::MAX,
                reason:              FinishReason::Incomplete,
                visible:             false,
                id:                  AdapterId::new("test-adapter"),
            })
            .middleware(
                RetryMiddleware::new(RetryPolicy::default()).observer_arc(observations.clone()),
            )
            .build()?
            .client;
        let start = Instant::now();
        let mut context = CallContext::new();
        context.set_deadline((start + Duration::from_millis(50)).into_std());
        let response = if streaming {
            let events = client
                .stream_with_context(request()?, context)
                .await?
                .collect::<Vec<_>>()
                .await
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?;
            let Some(StreamEvent::Ended { response }) = events.into_iter().last() else {
                panic!("missing terminal response")
            };
            response
        } else {
            Box::new(client.complete_with_context(request()?, context).await?)
        };
        assert_eq!(response.finish_reason, FinishReason::Incomplete);
        assert_eq!(response.text(), "partial");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(observations.retries().is_empty());
        assert_eq!(start.elapsed(), Duration::ZERO);
    }
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn cancellation_during_incomplete_backoff_prevents_another_attempt()
-> Result<(), Box<dyn StdError>> {
    for streaming in [false, true] {
        let calls = Arc::new(AtomicUsize::new(0));
        let observations = Arc::new(RetryRecorder::default());
        let client = Client::builder()
            .catalog(catalog()?)
            .adapter("test", UnfinishedAdapter {
                calls:               calls.clone(),
                unfinished_attempts: usize::MAX,
                reason:              FinishReason::Incomplete,
                visible:             false,
                id:                  AdapterId::new("test-adapter"),
            })
            .middleware(
                RetryMiddleware::new(RetryPolicy::default()).observer_arc(observations.clone()),
            )
            .build()?
            .client;
        let context = CallContext::new();
        let cancellation = context.cancellation().clone();
        let request = request()?;
        let task = spawn(async move {
            if streaming {
                let mut stream = client.stream_with_context(request, context).await?;
                let result = stream
                    .next()
                    .await
                    .expect("terminal cancellation")
                    .map(|_| ());
                assert!(stream.next().await.is_none());
                result
            } else {
                client
                    .complete_with_context(request, context)
                    .await
                    .map(|_| ())
            }
        });
        yield_now().await;
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(observations.retries().len(), 1);
        cancellation.cancel();
        assert_eq!(
            task.await?.expect_err("cancelled backoff").kind(),
            ErrorKind::Cancelled
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
    Ok(())
}

/// Models a cache or response-transforming middleware returning tool calls
/// on an unfinished turn. The public response policy must still suppress them.
struct UnfinishedToolsMiddleware {
    reason:        FinishReason,
    payload_bytes: usize,
}

fn diagnostic_calls(payload_bytes: usize) -> Vec<ToolCall> {
    let mut partial = ToolCall::function("partial", "lookup", json!({}));
    partial.input = ToolInput::Function(ToolArguments::from_raw(format!(
        "{{\"query\":\"{}",
        "x".repeat(payload_bytes)
    )));
    partial
        .provider_metadata
        .insert("openai".to_owned(), json!({"id":"fc_1"}));
    vec![
        ToolCall::function("valid", "lookup", json!({"query":"ok"})),
        partial,
    ]
}

#[async_trait]
impl Middleware for UnfinishedToolsMiddleware {
    async fn handle(&self, call: Call, _next: Next) -> Result<Output, Error> {
        let mut response = Response::new(
            call.route().provider().id().clone(),
            call.route().model().id().clone(),
            vec![ContentPart::Text {
                text: "partial".to_owned(),
            }],
        );
        response.finish_reason = self.reason.clone();
        response.content.extend(
            diagnostic_calls(self.payload_bytes)
                .into_iter()
                .map(ContentPart::ToolCall),
        );
        response.raw = Some(json!({"provider_payload":"discard"}));
        Ok(if call.operation() == Operation::Stream {
            Output::Stream(ResponseStream::new(iter([Ok(StreamEvent::Ended {
                response: Box::new(response),
            })])))
        } else {
            Output::Complete(response)
        })
    }
}

#[tokio::test]
async fn unfinished_call_diagnostics_survive_raw_removal_and_obey_output_limits()
-> Result<(), Box<dyn StdError>> {
    for reason in [FinishReason::Length, FinishReason::Incomplete] {
        for payload_bytes in [8, 4096] {
            let client = Client::builder()
                .catalog(catalog()?)
                .adapter("test", FakeAdapter::successful())
                .retain_raw_response(false)
                .response_limits(ResponseLimits::default().max_output_bytes(2048))
                .middleware(UnfinishedToolsMiddleware {
                    reason: reason.clone(),
                    payload_bytes,
                })
                .build()?
                .client;
            for streaming in [false, true] {
                let result = if streaming {
                    let mut stream = client.stream(request()?).await?;
                    let result = stream
                        .next()
                        .await
                        .expect("terminal response")
                        .map(|event| {
                            let StreamEvent::Ended { response } = event else {
                                panic!("expected Ended")
                            };
                            *response
                        });
                    assert!(stream.next().await.is_none());
                    result
                } else {
                    client.complete(request()?).await
                };
                if payload_bytes == 4096 {
                    assert_eq!(
                        result.expect_err("diagnostics exceed output limit").kind(),
                        ErrorKind::ResourceLimit
                    );
                    continue;
                }
                let response = result?;
                assert_eq!(response.finish_reason, reason);
                assert_eq!(response.tool_calls().count(), 0);
                assert_eq!(
                    response.suppressed_tool_calls,
                    diagnostic_calls(payload_bytes)
                );
                assert_eq!(response.warnings.len(), 2);
                assert!(
                    response
                        .warnings
                        .iter()
                        .all(|warning| warning.message.contains("unfinished"))
                );
                assert!(response.raw.is_none());
                let stored: Response = serde_json::from_value(serde_json::to_value(&response)?)?;
                assert_eq!(stored, response);
                assert!(
                    stored
                        .into_message()
                        .content()
                        .iter()
                        .all(|part| !matches!(part, ContentPart::ToolCall(_)))
                );
            }
        }
    }
    Ok(())
}

#[test]
fn unfamiliar_error_categories_never_authorize_automatic_retries() {
    for hint in [
        RetryClassification::Never,
        RetryClassification::Safe,
        RetryClassification::after(Duration::from_secs(1)),
    ] {
        let error = Error::new(
            ErrorKind::Unknown("future_error".to_owned()),
            "new category",
        )
        .with_retry(hint)
        .with_provider_retry_after(Duration::from_secs(1));
        assert_eq!(RetryPolicy::default().next_delay(1, &error), None);
    }
}

// ===========================================================================
// Evaluation
// ===========================================================================

/// One provider whose `judge` row claims JSON Schema output and whose
/// `plain` row claims only text.
const EVALUATION_CATALOG: &str = r#"
schema_version = 1

[providers.test]
display_name = "Test"
adapter = "test-adapter"
codecs = ["test-codec"]
base_url = "http://127.0.0.1"
default_model = "judge"

[providers.test.auth]
type = "none"

[providers.test.models.judge]
display_name = "Judge"
api_model = "judge"
capabilities = { text = true, response_format = { json_schema = true } }

[providers.test.models.plain]
display_name = "Plain"
api_model = "plain"
capabilities = { text = true }
"#;

fn evaluation_catalog() -> Result<Catalog, CatalogError> {
    Catalog::builder().overlay_toml(EVALUATION_CATALOG)?.build()
}

/// A choice, a boolean, and a score, under ids that sort in that order.
fn evaluation(model: &str) -> Evaluation {
    Evaluation::builder()
        .model(model)
        .state("I was charged twice.")
        .choice("department", "Which team?", [
            ("billing", Some("Charges and refunds")),
            ("technical", None),
        ])
        .boolean("requests_refund", "Refund requested?")
        .score("severity", "How severe?", ["Cosmetic", "Blocking"])
        .build()
        .expect("the evaluation should build")
}

/// Answers every judge request with one canned text body.
struct JudgeAdapter {
    id:    AdapterId,
    calls: Arc<AtomicUsize>,
    /// The text the judge returns, under the internal keys.
    text:  &'static str,
}

impl JudgeAdapter {
    fn answering(text: &'static str) -> Self {
        Self {
            id: AdapterId::new("test-adapter"),
            calls: Arc::new(AtomicUsize::new(0)),
            text,
        }
    }
}

#[async_trait]
impl ProviderAdapter for JudgeAdapter {
    fn id(&self) -> &AdapterId {
        &self.id
    }

    async fn complete(&self, call: &ResolvedCall) -> Result<Response, Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        assert!(
            matches!(
                call.request().response_format(),
                Some(ResponseFormat::JsonSchema { name, .. }) if name == "evaluation"
            ),
            "the judge request carries the evaluation schema"
        );
        Ok(success_response(call, self.text))
    }

    async fn stream(&self, _call: &ResolvedCall) -> Result<ResponseStream, Error> {
        Ok(ResponseStream::new(pending_stream()))
    }
}

/// An adapter with its own evaluation protocol.
struct NativeEvaluator {
    id:              AdapterId,
    complete_calls:  Arc<AtomicUsize>,
    evaluate_calls:  Arc<AtomicUsize>,
    /// Whether the returned verdict skips the boolean answer.
    omits_an_answer: bool,
}

impl NativeEvaluator {
    fn new(omits_an_answer: bool) -> Self {
        Self {
            id: AdapterId::new("test-adapter"),
            complete_calls: Arc::new(AtomicUsize::new(0)),
            evaluate_calls: Arc::new(AtomicUsize::new(0)),
            omits_an_answer,
        }
    }
}

#[async_trait]
impl ProviderAdapter for NativeEvaluator {
    fn id(&self) -> &AdapterId {
        &self.id
    }

    async fn complete(&self, call: &ResolvedCall) -> Result<Response, Error> {
        self.complete_calls.fetch_add(1, Ordering::SeqCst);
        Ok(success_response(call, "unexpected"))
    }

    async fn stream(&self, _call: &ResolvedCall) -> Result<ResponseStream, Error> {
        Ok(ResponseStream::new(pending_stream()))
    }

    async fn evaluate(&self, call: &ResolvedEvaluation) -> Result<Verdict, Error> {
        self.evaluate_calls.fetch_add(1, Ordering::SeqCst);
        let mut answers = BTreeMap::new();
        for (id, question) in call.evaluation().questions() {
            let answer = match question.kind() {
                QuestionKind::Choice => Answer::Choice(ChoiceAnswer {
                    choice:        "billing".to_owned(),
                    probabilities: None,
                    confidence:    None,
                }),
                QuestionKind::Score => Answer::Score(ScoreAnswer {
                    score:         1.0,
                    probabilities: None,
                    confidence:    None,
                }),
                QuestionKind::Boolean if self.omits_an_answer => continue,
                QuestionKind::Boolean => Answer::Boolean(BooleanAnswer { probability: 0.75 }),
                _ => unreachable!("the evaluation asks only the three known kinds"),
            };
            answers.insert(id.clone(), answer);
        }
        Ok(Verdict::new(
            call.route().provider().id().clone(),
            call.route().model().id().clone(),
            answers,
        ))
    }

    fn evaluates_natively(&self) -> bool {
        true
    }
}

#[tokio::test(start_paused = true)]
async fn evaluate_runs_one_judge_completion_on_a_structured_output_row()
-> Result<(), Box<dyn StdError>> {
    let adapter = JudgeAdapter::answering(r#"{"q0":"c1","q1":0.9,"q2":1}"#);
    let calls = adapter.calls.clone();
    let client = Client::builder()
        .catalog(evaluation_catalog()?)
        .adapter("test", adapter)
        .build()?
        .client;

    let verdict = client.evaluate(evaluation("test/judge")).await?;

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(verdict.model.to_string(), "test/judge");
    assert_eq!(verdict.choice("department")?.choice, "technical");
    assert!(verdict.boolean("requests_refund")?.is_likely());
    assert_eq!(verdict.score("severity")?.nearest_level(), 1);
    assert!(verdict.rounding.is_none());
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn evaluate_refuses_a_row_without_structured_output_before_dispatch()
-> Result<(), Box<dyn StdError>> {
    let adapter = JudgeAdapter::answering("{}");
    let calls = adapter.calls.clone();
    let client = Client::builder()
        .catalog(evaluation_catalog()?)
        .adapter("test", adapter)
        .build()?
        .client;

    let error = client
        .evaluate(evaluation("test/plain"))
        .await
        .expect_err("a text-only row cannot judge");

    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn a_malformed_judge_answer_is_a_decode_error_and_is_never_retried()
-> Result<(), Box<dyn StdError>> {
    let adapter = JudgeAdapter::answering(r#"{"q0":"c9","q1":0.9,"q2":1}"#);
    let calls = adapter.calls.clone();
    let client = Client::builder()
        .catalog(evaluation_catalog()?)
        .adapter("test", adapter)
        .middleware(RetryMiddleware::new(
            RetryPolicy::exponential()
                .max_attempts(3)
                .initial_delay(Duration::ZERO),
        ))
        .build()?
        .client;

    let error = client
        .evaluate(evaluation("test/judge"))
        .await
        .expect_err("`c9` names no option");

    assert_eq!(error.kind(), ErrorKind::ResponseDecode);
    assert_eq!(error.retry_classification(), RetryClassification::Never);
    assert!(
        error.message().contains("department"),
        "{}",
        error.message()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn a_native_adapter_receives_evaluate_and_its_verdict_is_validated()
-> Result<(), Box<dyn StdError>> {
    let adapter = NativeEvaluator::new(false);
    let complete_calls = adapter.complete_calls.clone();
    let evaluate_calls = adapter.evaluate_calls.clone();
    let client = Client::builder()
        .catalog(evaluation_catalog()?)
        .adapter("test", adapter)
        .build()?
        .client;

    let verdict = client.evaluate(evaluation("test/judge")).await?;

    assert_eq!(evaluate_calls.load(Ordering::SeqCst), 1);
    assert_eq!(complete_calls.load(Ordering::SeqCst), 0);
    assert_eq!(verdict.choice("department")?.choice, "billing");
    assert!((verdict.boolean("requests_refund")?.probability - 0.75).abs() < f64::EPSILON);
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn a_native_verdict_missing_an_answer_is_a_decode_error() -> Result<(), Box<dyn StdError>> {
    let client = Client::builder()
        .catalog(evaluation_catalog()?)
        .adapter("test", NativeEvaluator::new(true))
        .build()?
        .client;

    let error = client
        .evaluate(evaluation("test/judge"))
        .await
        .expect_err("the verdict skips the boolean");

    assert_eq!(error.kind(), ErrorKind::ResponseDecode);
    assert!(
        error.message().contains("`requests_refund` has no answer"),
        "{}",
        error.message()
    );
    Ok(())
}

/// Records the operation and payload shape of every call it sees.
struct OperationRecorder {
    seen: Arc<Mutex<Vec<(Operation, bool, String)>>>,
}

#[async_trait]
impl Middleware for OperationRecorder {
    async fn handle(&self, call: Call, next: Next) -> Result<Output, Error> {
        self.seen.lock().expect("recorder").push((
            call.operation(),
            call.evaluation().is_some(),
            call.request().model().to_owned(),
        ));
        next.run(call).await
    }
}

#[tokio::test(start_paused = true)]
async fn a_native_evaluation_reaches_middleware_as_operation_evaluate()
-> Result<(), Box<dyn StdError>> {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let client = Client::builder()
        .catalog(evaluation_catalog()?)
        .adapter("test", NativeEvaluator::new(false))
        .middleware(OperationRecorder { seen: seen.clone() })
        .build()?
        .client;

    client.evaluate(evaluation("test/judge")).await?;

    assert_eq!(*seen.lock().expect("recorder"), [(
        Operation::Evaluate,
        true,
        "test/judge".to_owned()
    )]);
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn a_judge_evaluation_reaches_middleware_as_a_complete_call() -> Result<(), Box<dyn StdError>>
{
    let seen = Arc::new(Mutex::new(Vec::new()));
    let client = Client::builder()
        .catalog(evaluation_catalog()?)
        .adapter(
            "test",
            JudgeAdapter::answering(r#"{"q0":"c0","q1":0.5,"q2":0}"#),
        )
        .middleware(OperationRecorder { seen: seen.clone() })
        .build()?
        .client;

    client.evaluate(evaluation("test/judge")).await?;

    assert_eq!(*seen.lock().expect("recorder"), [(
        Operation::Complete,
        false,
        "test/judge".to_owned()
    )]);
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn resolve_evaluation_route_agrees_with_evaluate() -> Result<(), Box<dyn StdError>> {
    let client = Client::builder()
        .catalog(evaluation_catalog()?)
        .adapter(
            "test",
            JudgeAdapter::answering(r#"{"q0":"c0","q1":0.5,"q2":0}"#),
        )
        .build()?
        .client;

    // The provider default is the judge row, so a bare provider selector
    // resolves there and the verdict names the canonical route.
    let evaluation = evaluation("test");
    let route = client.resolve_evaluation_route(&evaluation)?;
    let verdict = client.evaluate(evaluation).await?;

    assert_eq!(route.handle().to_string(), "test/judge");
    assert_eq!(verdict.model, route.handle());
    assert!(verdict.answers.contains_key(&QuestionId::new("severity")));
    Ok(())
}

// ===========================================================================
// Per-call codec selection
// ===========================================================================

/// A provider on the default `http` adapter that lists the Chat codec and
/// the Vercel evaluation codec, with one generation row and one native
/// evaluation row, as the built-in `vercel` provider does.
const MIXED_HTTP_CATALOG: &str = r#"
schema_version = 1

[providers.vercel]
display_name = "Vercel"
codecs = ["openai-chat", "vercel-evaluation"]
base_url = "http://127.0.0.1:1"
default_model = "chat"

[providers.vercel.auth]
type = "none"

[providers.vercel.models.chat]
display_name = "Chat"
api_model = "chat"
capabilities = { text = true }

[providers.vercel.models.jev]
display_name = "Jev"
api_model = "typesafe-ai/jev"
capabilities = { evaluation = { choice = true, score = true, boolean = true } }
"#;

/// Records the codec each call carried when it reached the middleware.
struct CodecRecorder {
    seen: Arc<Mutex<Vec<Option<String>>>>,
}

#[async_trait]
impl Middleware for CodecRecorder {
    async fn handle(&self, call: Call, next: Next) -> Result<Output, Error> {
        self.seen
            .lock()
            .expect("recorder")
            .push(call.codec().map(ToString::to_string));
        next.run(call).await
    }
}

/// An evaluation-only row has no generation codec, so `complete` is refused
/// before the pipeline runs, and the refusal names the family.
#[tokio::test(start_paused = true)]
async fn a_completion_on_an_evaluation_only_row_is_refused_before_dispatch()
-> Result<(), Box<dyn StdError>> {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let client = Client::builder()
        .catalog(
            Catalog::builder()
                .overlay_toml(MIXED_HTTP_CATALOG)?
                .build()?,
        )
        .middleware(CodecRecorder { seen: seen.clone() })
        .build()?
        .client;
    // The row claims no `text` either, but the codec refusal comes first
    // and names the family, which is the more useful message.
    let request = Request::builder()
        .model("vercel/jev")
        .user("Hello")
        .build()?;

    let error = client
        .complete(request.clone())
        .await
        .expect_err("no generation codec reaches the evaluation row");

    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    assert_eq!(
        error.message(),
        "no generation codec reaches model vercel/jev"
    );
    assert!(
        client.stream(request.clone()).await.is_err(),
        "streaming applies the same gate"
    );
    assert!(client.count_input_tokens(request).await.is_err());
    assert!(
        seen.lock().expect("recorder").is_empty(),
        "the refusal happens before any middleware runs"
    );
    Ok(())
}

/// The generation row on the same provider carries its codec into the
/// pipeline, where middleware can read it.
#[tokio::test(start_paused = true)]
async fn middleware_sees_the_selected_codec_on_a_builtin_provider() -> Result<(), Box<dyn StdError>>
{
    let seen = Arc::new(Mutex::new(Vec::new()));
    let client = Client::builder()
        .catalog(
            Catalog::builder()
                .overlay_toml(MIXED_HTTP_CATALOG)?
                .build()?,
        )
        .middleware(CodecRecorder { seen: seen.clone() })
        .build()?
        .client;

    // Port 1 refuses every connection, so the call fails at the transport,
    // after the middleware has already recorded the codec.
    let request = Request::builder()
        .model("vercel/chat")
        .user("Hello")
        .build()?;
    let error = client
        .complete(request)
        .await
        .expect_err("nothing listens on port 1");
    assert_eq!(error.kind(), ErrorKind::Network, "{error}");

    let evaluation = Evaluation::builder()
        .model("vercel/jev")
        .state("I was charged twice.")
        .boolean("requests_refund", "Refund requested?")
        .build()?;
    let error = client
        .evaluate(evaluation)
        .await
        .expect_err("nothing listens on port 1");
    assert_eq!(error.kind(), ErrorKind::Network, "{error}");

    assert_eq!(*seen.lock().expect("recorder"), [
        Some("openai-chat".to_owned()),
        Some("vercel-evaluation".to_owned()),
    ]);
    Ok(())
}

/// An explicit adapter installed with `ClientBuilder::adapter` owns its own
/// wire handling, so the client selects no codec for it.
#[tokio::test(start_paused = true)]
async fn middleware_sees_no_codec_on_an_explicit_adapter() -> Result<(), Box<dyn StdError>> {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let client = Client::builder()
        .catalog(catalog()?)
        .adapter("test", FakeAdapter::successful())
        .middleware(CodecRecorder { seen: seen.clone() })
        .build()?
        .client;

    client.complete(request()?).await?;

    assert_eq!(*seen.lock().expect("recorder"), [None]);
    Ok(())
}

/// An adapter a custom factory builds, which records the codec it received.
struct CodecObservingAdapter {
    id:     AdapterId,
    codecs: Arc<Mutex<Vec<Option<String>>>>,
}

#[async_trait]
impl ProviderAdapter for CodecObservingAdapter {
    fn id(&self) -> &AdapterId {
        &self.id
    }

    async fn complete(&self, call: &ResolvedCall) -> Result<Response, Error> {
        self.codecs
            .lock()
            .expect("codecs")
            .push(call.codec().map(ToString::to_string));
        Ok(success_response(call, "done"))
    }

    async fn stream(&self, _call: &ResolvedCall) -> Result<ResponseStream, Error> {
        Ok(ResponseStream::new(pending_stream()))
    }

    async fn evaluate(&self, call: &ResolvedEvaluation) -> Result<Verdict, Error> {
        self.codecs
            .lock()
            .expect("codecs")
            .push(call.codec().map(ToString::to_string));
        let answers = call
            .evaluation()
            .questions()
            .keys()
            .map(|id| {
                (
                    id.clone(),
                    Answer::Boolean(BooleanAnswer { probability: 0.5 }),
                )
            })
            .collect();
        Ok(Verdict::new(
            call.route().provider().id().clone(),
            call.route().model().id().clone(),
            answers,
        ))
    }

    fn evaluates_natively(&self) -> bool {
        true
    }
}

struct CodecObservingFactory {
    codecs: Arc<Mutex<Vec<Option<String>>>>,
}

impl AdapterFactory for CodecObservingFactory {
    fn create(
        &self,
        provider: &CatalogProvider,
        _context: &AdapterContext,
    ) -> Result<Arc<dyn ProviderAdapter>, AdapterBuildError> {
        Ok(Arc::new(CodecObservingAdapter {
            id:     provider.adapter().clone(),
            codecs: self.codecs.clone(),
        }))
    }
}

/// A custom factory registered under an opaque adapter id builds, serves
/// `complete` and `evaluate`, and sees no codec on either: codecs are how
/// the built-in adapters organize their protocols, not a contract every
/// adapter must adopt.
#[tokio::test(start_paused = true)]
async fn a_custom_factory_serves_complete_and_evaluate_with_no_codec()
-> Result<(), Box<dyn StdError>> {
    let codecs = Arc::new(Mutex::new(Vec::new()));
    let build = Client::builder()
        .catalog(evaluation_catalog()?)
        .adapter_factory("test-adapter", CodecObservingFactory {
            codecs: codecs.clone(),
        })
        .build()?;
    assert!(build.issues.is_empty(), "{:?}", build.issues);
    let client = build.client;

    let response = client
        .complete(Request::builder().model("test/judge").user("hi").build()?)
        .await?;
    assert_eq!(response.text(), "done");

    let evaluation = Evaluation::builder()
        .model("test/judge")
        .state("I was charged twice.")
        .boolean("requests_refund", "Refund requested?")
        .build()?;
    let verdict = client.evaluate(evaluation).await?;
    assert!(
        verdict
            .answers
            .contains_key(&QuestionId::new("requests_refund"))
    );

    assert_eq!(*codecs.lock().expect("codecs"), [None, None]);
    Ok(())
}

/// A provider whose codec has options that do not parse is one build issue
/// naming the codec, and the provider builds no adapter.
#[test]
fn invalid_codec_options_are_one_issue_naming_the_codec() -> Result<(), Box<dyn StdError>> {
    let catalog = Catalog::builder()
        .overlay_toml(MIXED_HTTP_CATALOG)?
        .overlay_toml(
            r#"
            [providers.vercel]
            codec_options = { openai-chat = { base_url_is_api_root = "yes" } }
            "#,
        )?
        .build()?;

    let build = Client::builder().catalog(catalog).build()?;

    assert!(build.client.available_providers().iter().next().is_none());
    assert!(
        matches!(
            build.issues.as_slice(),
            [issue] if issue.provider.as_str() == "vercel"
                && issue.adapter.as_str() == "http"
                && issue.codec.as_ref().map(ToString::to_string).as_deref() == Some("openai-chat")
                && matches!(&issue.cause, ProviderBuildCause::Adapter(_))
        ),
        "{:?}",
        build.issues
    );
    Ok(())
}

/// The E2E Vercel catalog's `jev` row writes no `adapter` line and no
/// `codecs` line; its evaluation claim alone selects the native codec, and
/// its sibling structured-output rows keep judging through Chat.
#[test]
fn the_vercel_catalog_derives_native_and_judge_rows_without_row_adapters()
-> Result<(), Box<dyn StdError>> {
    let catalog = Catalog::builder()
        .toml_layer("vercel", include_str!("e2e/vercel_catalog.toml"))?
        .build()?;
    let build = Client::builder().catalog(catalog).build()?;
    assert!(build.issues.is_empty(), "{:?}", build.issues);
    let client = build.client;

    let jev = client.resolve_evaluation_route(
        &Evaluation::builder()
            .model("vercel/jev")
            .state("state")
            .boolean("q", "Q?")
            .build()?,
    )?;
    assert_eq!(
        jev.model()
            .codecs()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        ["vercel-evaluation"]
    );

    let sonnet = client.resolve_evaluation_route(
        &Evaluation::builder()
            .model("vercel/claude-sonnet-5")
            .state("state")
            .boolean("q", "Q?")
            .build()?,
    )?;
    assert_eq!(
        sonnet
            .model()
            .codecs()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        ["openai-chat"]
    );
    Ok(())
}
