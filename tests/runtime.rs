#![cfg(feature = "runtime")]

use std::error::Error as StdError;
use std::future::pending;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures_util::StreamExt as _;
use futures_util::stream::{iter, pending as pending_stream};
use lithos_llm::adapter::{ProviderAdapter, ResolvedCall};
use lithos_llm::catalog::{AdapterId, Catalog, CatalogError, ModelId, ProviderId};
use lithos_llm::client::{ClientBuildError, ProviderBuildCause};
use lithos_llm::middleware::{
    Call, CallContext, Middleware, Next, Output, RetryMiddleware, RetryPolicy, TimeoutMiddleware,
};
use lithos_llm::types::{
    ContentBlockId, ContentPart, Error, ErrorKind, ImageContent, MediaSource, Message,
    RequestBuildError, Response, ResponseStream, RetryClassification, Role, StreamEvent,
};
use lithos_llm::{Client, Request};
use tokio::spawn;
use tokio::task::yield_now;

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

    async fn stream(&self, _call: &ResolvedCall) -> Result<ResponseStream, Error> {
        let call_index = self.stream_calls.fetch_add(1, Ordering::SeqCst);
        let events = if self.stream_fails_initial && call_index == 0 {
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
            vec![Ok(StreamEvent::TextDelta {
                id:   ContentBlockId::new("block-0"),
                text: "done".to_owned(),
            })]
        };
        Ok(Box::pin(iter(events)))
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

#[async_trait]
impl Middleware for ShortCircuit {
    async fn handle(&self, call: Call, _next: Next) -> Result<Output, Error> {
        Ok(Output::Complete(Response::new(
            call.route.provider().id().clone(),
            call.route.model().id().clone(),
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
        [Ok(StreamEvent::TextDelta { text, .. })] if text == "done"
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

struct PendingAdapter {
    id: AdapterId,
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
        Ok(Box::pin(pending_stream()))
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
        .middleware(TimeoutMiddleware::new(Duration::from_millis(5)))
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
