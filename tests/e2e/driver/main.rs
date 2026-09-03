use std::collections::VecDeque;
use std::error::Error as StdError;
use std::io::{self, Write as _};
use std::num::NonZeroUsize;
use std::path::Path;
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};
use std::{env, fs};

use async_trait::async_trait;
use futures_util::StreamExt as _;
use futures_util::stream::unfold;
use lithos_llm::Client;
use lithos_llm::adapter::{
    AdapterBuildError, AdapterContext, AdapterFactory, ProviderAdapter, ResolvedCall,
};
use lithos_llm::catalog::{AdapterId, Catalog, CatalogProvider, ModelId, ProviderId};
use lithos_llm::credentials::{EnvironmentCredentials, NoCredentials};
use lithos_llm::estimate::{EstimateWarning, request_tokens};
use lithos_llm::middleware::{
    Call, CallContext, ConcurrencyLimitMiddleware, Middleware, Next, Observer, ObserverMiddleware,
    Output, RetryMiddleware, RetryPolicy, RetryStage, TimeoutMiddleware, finalize_stream,
    inspect_stream, map_stream,
};
use lithos_llm::resolver::CatalogResolver;
use lithos_llm::types::{
    ContentPart, Error, ErrorKind, Request, Response, ResponseStream, RetryClassification,
    StreamEvent, TokenCounts,
};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::time::sleep;

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            let _ignored = writeln!(io::stderr().lock(), "error: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<(), Box<dyn StdError>> {
    let mut arguments = env::args().skip(1);
    let action = arguments.next().ok_or("an action is required")?;
    let output = match action.as_str() {
        "estimate" => {
            let request = read_request(required_argument(&mut arguments, "request path")?)?;
            estimate(&request)
        }
        "count" => {
            let catalog = required_argument(&mut arguments, "catalog path")?;
            let request = read_request(required_argument(&mut arguments, "request path")?)?;
            count(Path::new(&catalog), request).await?
        }
        "simulate" => {
            let simulation = read_json(required_argument(&mut arguments, "simulation path")?)?;
            simulate(simulation).await?
        }
        other => return Err(format!("unknown action `{other}`").into()),
    };
    writeln!(
        io::stdout().lock(),
        "{}",
        serde_json::to_string_pretty(&output)?
    )?;
    Ok(())
}

fn required_argument(
    arguments: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<String, Box<dyn StdError>> {
    arguments
        .next()
        .ok_or_else(|| format!("{name} is required").into())
}

fn read_request(path: String) -> Result<Request, Box<dyn StdError>> {
    read_json(path)
}

fn read_json<T: for<'de> Deserialize<'de>>(path: String) -> Result<T, Box<dyn StdError>> {
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

fn estimate(request: &Request) -> Value {
    let estimate = request_tokens(request);
    let warnings = estimate
        .warnings()
        .map(|warning| {
            json!({
                "code": warning.code(),
                "message": warning.to_string(),
                "present": estimate.has_warning(warning),
            })
        })
        .collect::<Vec<_>>();
    json!({
        "kind": "estimate",
        "tokens": estimate.tokens(),
        "warnings": warnings,
        "has_media_warning": estimate.has_warning(EstimateWarning::Media),
    })
}

async fn count(catalog_path: &Path, request: Request) -> Result<Value, Box<dyn StdError>> {
    let source = fs::read_to_string(catalog_path)?;
    let catalog = Catalog::builder()
        .with_builtin()
        .toml_layer(catalog_path.display().to_string(), &source)?
        .build()?;
    let build = Client::builder()
        .catalog(catalog)
        .credentials(EnvironmentCredentials::conventional())
        .build()?;
    match build.client.count_input_tokens(request).await {
        Ok(Some(count)) => Ok(json!({
            "kind": "provider_count",
            "tokens": count.tokens(),
            "model": count.model().to_string(),
        })),
        Ok(None) => Ok(json!({ "kind": "unsupported" })),
        Err(error) => Ok(render_error(&error)),
    }
}

fn render_error(error: &Error) -> Value {
    json!({
        "kind": "error",
        "error_kind": format!("{:?}", error.kind()),
        "message": error.message(),
        "provider": error.provider().map(ToString::to_string),
        "status": error.status(),
        "provider_code": error.provider_code(),
        "retry": format!("{:?}", error.retry_classification()),
    })
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SimulationMode {
    Complete,
    Stream,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Registration {
    Direct,
    Arc,
    Factory,
}

#[derive(Debug, Deserialize)]
struct Simulation {
    mode:                SimulationMode,
    request:             Request,
    steps:               Vec<AdapterStep>,
    #[serde(default = "direct_registration")]
    registration:        Registration,
    #[serde(default)]
    builder_variants:    bool,
    #[serde(default)]
    retry:               Option<RetrySpec>,
    #[serde(default)]
    timeout_ms:          Option<u64>,
    #[serde(default)]
    concurrency:         Option<usize>,
    #[serde(default)]
    observer:            bool,
    #[serde(default)]
    cancel_before:       bool,
    #[serde(default)]
    deadline_ms:         Option<i64>,
    #[serde(default)]
    stream_hooks:        bool,
    #[serde(default)]
    initial_attempt:     Option<u32>,
    #[serde(default)]
    exercise_extensions: bool,
}

const fn direct_registration() -> Registration {
    Registration::Direct
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct RetrySpec {
    max_attempts:       u32,
    initial_delay_ms:   u64,
    max_delay_ms:       u64,
    retry_after_cap_ms: u64,
    #[serde(default)]
    jitter:             bool,
    #[serde(default)]
    observer:           bool,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AdapterStep {
    Complete {
        text:     String,
        #[serde(default)]
        delay_ms: u64,
    },
    Stream {
        items:    Vec<StreamItem>,
        #[serde(default)]
        delay_ms: u64,
    },
    Error {
        message:        String,
        retry:          RetryKind,
        #[serde(default)]
        retry_after_ms: Option<u64>,
        #[serde(default)]
        delay_ms:       u64,
    },
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RetryKind {
    Never,
    Safe,
    After,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StreamItem {
    Event {
        event:    Box<StreamEvent>,
        #[serde(default)]
        delay_ms: u64,
    },
    Error {
        message:  String,
        retry:    RetryKind,
        #[serde(default)]
        delay_ms: u64,
    },
}

#[derive(Clone, Default)]
struct ScriptedAdapter {
    steps:    Arc<Mutex<VecDeque<AdapterStep>>>,
    calls:    Arc<AtomicUsize>,
    attempts: Arc<Mutex<Vec<u32>>>,
}

impl ScriptedAdapter {
    fn new(steps: Vec<AdapterStep>) -> Self {
        Self {
            steps:    Arc::new(Mutex::new(steps.into())),
            calls:    Arc::new(AtomicUsize::new(0)),
            attempts: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn next(&self, call: &ResolvedCall) -> Result<AdapterStep, Error> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.attempts
            .lock()
            .expect("attempt lock should not be poisoned")
            .push(call.context().attempt());
        self.steps
            .lock()
            .expect("step lock should not be poisoned")
            .pop_front()
            .ok_or_else(|| Error::new(ErrorKind::Provider, "the fixture script is exhausted"))
    }
}

#[async_trait]
impl ProviderAdapter for ScriptedAdapter {
    fn id(&self) -> &AdapterId {
        static ID: LazyLock<AdapterId> = LazyLock::new(|| AdapterId::new("fixture-adapter"));
        &ID
    }

    async fn complete(&self, call: &ResolvedCall) -> Result<Response, Error> {
        match self.next(call)? {
            AdapterStep::Complete { text, delay_ms } => {
                sleep(Duration::from_millis(delay_ms)).await;
                Ok(fixture_response(text))
            }
            AdapterStep::Error {
                message,
                retry,
                retry_after_ms,
                delay_ms,
            } => {
                sleep(Duration::from_millis(delay_ms)).await;
                Err(script_error(message, retry, retry_after_ms))
            }
            AdapterStep::Stream { .. } => Err(Error::new(
                ErrorKind::Middleware,
                "a stream script was used for a complete call",
            )),
        }
    }

    async fn stream(&self, call: &ResolvedCall) -> Result<ResponseStream, Error> {
        match self.next(call)? {
            AdapterStep::Stream { items, delay_ms } => {
                sleep(Duration::from_millis(delay_ms)).await;
                Ok(scripted_stream(items))
            }
            AdapterStep::Error {
                message,
                retry,
                retry_after_ms,
                delay_ms,
            } => {
                sleep(Duration::from_millis(delay_ms)).await;
                Err(script_error(message, retry, retry_after_ms))
            }
            AdapterStep::Complete { .. } => Err(Error::new(
                ErrorKind::Middleware,
                "a complete script was used for a stream call",
            )),
        }
    }
}

#[derive(Clone)]
struct FixtureFactory(ScriptedAdapter);

impl AdapterFactory for FixtureFactory {
    fn create(
        &self,
        _provider: &CatalogProvider,
        context: &AdapterContext,
    ) -> Result<Arc<dyn ProviderAdapter>, AdapterBuildError> {
        let _http = context.http();
        let _credentials = context.credentials();
        let _idle = context.stream_idle_timeout();
        Ok(Arc::new(self.0.clone()))
    }
}

#[derive(Default)]
struct RecordingObserver {
    starts:          AtomicUsize,
    completes:       AtomicUsize,
    events:          AtomicUsize,
    errors:          AtomicUsize,
    retries:         AtomicUsize,
    request_retries: AtomicUsize,
    stream_retries:  AtomicUsize,
}

impl Observer for RecordingObserver {
    fn on_start(&self, _call: &Call) {
        self.starts.fetch_add(1, Ordering::Relaxed);
    }

    fn on_complete(&self, _call: &Call, result: Result<&Response, &Error>) {
        self.completes.fetch_add(1, Ordering::Relaxed);
        if result.is_err() {
            self.errors.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn on_stream_event(&self, _call: &Call, event: Result<&StreamEvent, &Error>) {
        self.events.fetch_add(1, Ordering::Relaxed);
        if event.is_err() {
            self.errors.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn on_retry(
        &self,
        _call: &Call,
        _error: &Error,
        _attempt: u32,
        _delay: Duration,
        stage: RetryStage,
    ) {
        self.retries.fetch_add(1, Ordering::Relaxed);
        match stage {
            RetryStage::Request => self.request_retries.fetch_add(1, Ordering::Relaxed),
            RetryStage::Stream => self.stream_retries.fetch_add(1, Ordering::Relaxed),
            _ => 0,
        };
    }
}

#[derive(Clone, Copy, Debug)]
struct PassThroughMiddleware;

#[async_trait]
impl Middleware for PassThroughMiddleware {
    async fn handle(&self, call: Call, next: Next) -> Result<Output, Error> {
        next.run(call).await
    }
}

async fn simulate(simulation: Simulation) -> Result<Value, Box<dyn StdError>> {
    let adapter = ScriptedAdapter::new(simulation.steps);
    let observer = Arc::new(RecordingObserver::default());
    let catalog = Catalog::builder()
        .toml_layer("simulation", simulation_catalog())?
        .build()?;
    let mut builder = Client::builder().catalog(catalog);
    if simulation.builder_variants {
        builder = builder
            .resolver(CatalogResolver)
            .resolver_arc(Arc::new(CatalogResolver))
            .credentials(NoCredentials)
            .credentials_arc(Arc::new(NoCredentials))
            .http(reqwest::Client::new())
            .connect_timeout(None)
            .stream_idle_timeout(None)
            .enabled_providers([ProviderId::new("fixture")])
            .middleware_arc(Arc::new(PassThroughMiddleware));
    }
    builder = match simulation.registration {
        Registration::Direct => builder.adapter("fixture", adapter.clone()),
        Registration::Arc => builder.adapter_arc("fixture", Arc::new(adapter.clone())),
        Registration::Factory => {
            builder.adapter_factory("fixture-adapter", FixtureFactory(adapter.clone()))
        }
    };
    if let Some(retry) = simulation.retry {
        let policy = RetryPolicy::exponential()
            .max_attempts(retry.max_attempts)
            .initial_delay(Duration::from_millis(retry.initial_delay_ms))
            .max_delay(Duration::from_millis(retry.max_delay_ms))
            .retry_after_cap(Duration::from_millis(retry.retry_after_cap_ms))
            .jitter(retry.jitter);
        let middleware = if retry.observer {
            RetryMiddleware::new(policy).observer_arc(observer.clone())
        } else {
            RetryMiddleware::new(policy)
        };
        builder = builder.middleware(middleware);
    }
    if let Some(timeout_ms) = simulation.timeout_ms {
        builder = builder.middleware(TimeoutMiddleware::new(Duration::from_millis(timeout_ms)));
    }
    if let Some(limit) = simulation.concurrency.and_then(NonZeroUsize::new) {
        builder = builder.middleware(ConcurrencyLimitMiddleware::new(limit));
    }
    if simulation.observer {
        builder = builder.middleware(ObserverMiddleware::from_arc(observer.clone()));
    }
    let client = builder.build()?.client;
    let mut context = CallContext::new();
    if let Some(attempt) = simulation.initial_attempt {
        context.set_attempt(attempt);
    }
    let extension_exercised = if simulation.exercise_extensions {
        let replaced = context
            .extensions_mut()
            .insert::<String>("first".to_owned());
        let missing = context.extensions().get::<u64>().is_none();
        let present = context.extensions().get::<String>().map(String::as_str) == Some("first");
        let removed = context.extensions_mut().remove::<String>().is_some();
        replaced.is_none() && missing && present && removed
    } else {
        false
    };
    if let Some(deadline_ms) = simulation.deadline_ms {
        let deadline = if deadline_ms < 0 {
            Instant::now()
                .checked_sub(Duration::from_millis(deadline_ms.unsigned_abs()))
                .unwrap_or_else(Instant::now)
        } else {
            Instant::now()
                .checked_add(Duration::from_millis(deadline_ms.unsigned_abs()))
                .unwrap_or_else(Instant::now)
        };
        context.set_deadline(deadline);
    }
    if simulation.cancel_before {
        context.cancellation().cancel();
    }
    let result = match simulation.mode {
        SimulationMode::Complete => match client
            .complete_with_context(simulation.request, context)
            .await
        {
            Ok(response) => json!({
                "kind": "complete",
                "text": response.text(),
                "tokens": response.usage.total(),
            }),
            Err(error) => render_error(&error),
        },
        SimulationMode::Stream => match client
            .stream_with_context(simulation.request, context)
            .await
        {
            Ok(mut stream) => {
                let inspected = Arc::new(AtomicUsize::new(0));
                let finalized = Arc::new(AtomicUsize::new(0));
                if simulation.stream_hooks {
                    stream = map_stream(stream, Ok);
                    let count = inspected.clone();
                    stream = inspect_stream(stream, move |_| {
                        count.fetch_add(1, Ordering::Relaxed);
                    });
                    let count = finalized.clone();
                    stream = finalize_stream(stream, move || {
                        count.fetch_add(1, Ordering::Relaxed);
                    });
                }
                let mut events = Vec::new();
                while let Some(item) = stream.next().await {
                    match item {
                        Ok(event) => events.push(event_name(&event)),
                        Err(error) => {
                            events.push(format!("error:{:?}", error.kind()));
                            break;
                        }
                    }
                }
                drop(stream);
                json!({
                    "kind": "stream",
                    "events": events,
                    "inspected": inspected.load(Ordering::Relaxed),
                    "finalized": finalized.load(Ordering::Relaxed),
                })
            }
            Err(error) => render_error(&error),
        },
    };
    let attempts = adapter
        .attempts
        .lock()
        .expect("attempt lock should not be poisoned")
        .clone();
    Ok(json!({
        "kind": "simulation",
        "result": result,
        "adapter_calls": adapter.calls.load(Ordering::Relaxed),
        "attempts": attempts,
        "extension_exercised": extension_exercised,
        "observer": {
            "starts": observer.starts.load(Ordering::Relaxed),
            "completes": observer.completes.load(Ordering::Relaxed),
            "events": observer.events.load(Ordering::Relaxed),
            "errors": observer.errors.load(Ordering::Relaxed),
            "retries": observer.retries.load(Ordering::Relaxed),
            "request_retries": observer.request_retries.load(Ordering::Relaxed),
            "stream_retries": observer.stream_retries.load(Ordering::Relaxed),
        }
    }))
}

fn simulation_catalog() -> &'static str {
    r#"
schema_version = 1

[providers.fixture]
display_name = "Fixture"
adapter = "fixture-adapter"
codec = "fixture-codec"
base_url = "http://127.0.0.1"
auth = { type = "none" }
default_model = "model"

[providers.fixture.models.model]
display_name = "Fixture model"
api_model = "fixture-model"
capabilities = { text = true, images = true, audio = true, documents = true, tools = true, forced_tool_choice = true, structured_output = true, reasoning = true, reasoning_effort_levels = true, caching = true, cache_routing = true, sampling = true }
"#
}

fn fixture_response(text: String) -> Response {
    let mut response = Response::new(ProviderId::new("fixture"), ModelId::new("model"), vec![
        ContentPart::Text { text },
    ]);
    response.usage = TokenCounts {
        input:       1,
        output:      2,
        reasoning:   3,
        cache_read:  4,
        cache_write: 5,
    };
    response
}

fn scripted_stream(items: Vec<StreamItem>) -> ResponseStream {
    Box::pin(unfold(items.into_iter(), |mut items| async move {
        let item = items.next()?;
        let (result, delay_ms) = match item {
            StreamItem::Event { event, delay_ms } => (Ok(*event), delay_ms),
            StreamItem::Error {
                message,
                retry,
                delay_ms,
            } => (Err(script_error(message, retry, None)), delay_ms),
        };
        sleep(Duration::from_millis(delay_ms)).await;
        Some((result, items))
    }))
}

fn script_error(message: String, retry: RetryKind, retry_after_ms: Option<u64>) -> Error {
    let classification = match retry {
        RetryKind::Never => RetryClassification::Never,
        RetryKind::Safe => RetryClassification::Safe,
        RetryKind::After => {
            RetryClassification::after(Duration::from_millis(retry_after_ms.unwrap_or_default()))
        }
    };
    let mut error = Error::new(ErrorKind::Server, message)
        .with_provider(ProviderId::new("fixture"))
        .with_status(503)
        .with_provider_code("fixture_error")
        .with_retry(classification)
        .with_raw_data(json!({ "fixture": true }));
    if let Some(delay) = retry_after_ms {
        error = error.with_provider_retry_after(Duration::from_millis(delay));
    }
    error
}

fn event_name(event: &StreamEvent) -> String {
    let visible = event.is_visible();
    let name = match event {
        StreamEvent::Started { .. } => "started",
        StreamEvent::ContentBlockStart { .. } => "block_start",
        StreamEvent::TextDelta { .. } => "text",
        StreamEvent::ReasoningDelta { .. } => "reasoning",
        StreamEvent::ToolCallDelta { .. } => "tool",
        StreamEvent::ContentBlockEnd { .. } => "block_end",
        StreamEvent::Usage { .. } => "usage",
        StreamEvent::RateLimits { .. } => "rate_limits",
        StreamEvent::Completed { .. } => "completed",
        _ => "unknown",
    };
    format!("{name}:visible={visible}")
}
