use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;

use async_trait::async_trait;
use futures_core::Stream;
use tracing::Instrument as _;
use tracing::field::Empty;

use super::{Call, CancellationToken, Middleware, Mode, Next, Output};
use crate::types::{
    CostSource, Error, ErrorKind, FinishReason, Response, ResponseStream, StreamEvent, TokenCounts,
};

const OUTCOME_COMPLETED: &str = "completed";
const OUTCOME_INCOMPLETE: &str = "incomplete";
const OUTCOME_FAILED: &str = "failed";
const OUTCOME_CANCELLED: &str = "cancelled";
const OUTCOME_DROPPED: &str = "dropped";

/// Emits secret-safe tracing spans without installing a subscriber.
///
/// Each `llm.call` span carries stable `call_id`, `attempt`, `provider`,
/// `model`, and `mode` fields. Before the span closes, `outcome` is set to one
/// of `completed`, `incomplete`, `failed`, `cancelled`, or `dropped`.
/// Successful and incomplete calls also record token counts and cost when
/// available. Failures record `error_kind`, HTTP `status`, and `provider_code`
/// when available.
///
/// A stream owns its span until it produces [`StreamEvent::Completed`],
/// produces an error, ends without a terminal event, or is dropped. The span
/// is entered while each stream poll runs, but not while the stream is idle.
///
/// Place this middleware outside [`super::RetryMiddleware`] to trace one
/// logical call. Place it inside retry middleware to trace each attempt.
#[derive(Clone, Copy, Debug, Default)]
pub struct TracingMiddleware;

#[async_trait]
impl Middleware for TracingMiddleware {
    async fn handle(&self, call: Call, next: Next) -> Result<Output, Error> {
        let mut trace = CallTrace::new(&call);
        let span = trace.span.clone();
        let result = next.run(call).instrument(span).await;
        match result {
            Ok(Output::Complete(response)) => {
                trace.finish_completed(&response);
                Ok(Output::Complete(response))
            }
            Ok(Output::Stream(stream)) => {
                tracing::debug!(parent: &trace.span, "LLM stream accepted");
                Ok(Output::Stream(trace_stream(stream, trace)))
            }
            Err(error) => {
                trace.finish_error(&error);
                Err(error)
            }
        }
    }
}

struct CallTrace {
    span:         tracing::Span,
    cancellation: CancellationToken,
    deadline:     Option<Instant>,
    finished:     bool,
}

impl CallTrace {
    fn new(call: &Call) -> Self {
        let span = tracing::info_span!(
            "llm.call",
            call_id = %call.context.call_id(),
            attempt = call.context.attempt(),
            provider = %call.route.provider().id(),
            model = %call.route.model().id(),
            mode = mode_name(call.mode),
            outcome = Empty,
            error_kind = Empty,
            status = Empty,
            provider_code = Empty,
            input_tokens = Empty,
            output_tokens = Empty,
            reasoning_tokens = Empty,
            cache_read_tokens = Empty,
            cache_write_tokens = Empty,
            cost_usd_micros = Empty,
            cost_source = Empty,
        );
        Self {
            span,
            cancellation: call.context.cancellation().clone(),
            deadline: call.context.deadline(),
            finished: false,
        }
    }

    fn record_usage(&self, usage: TokenCounts) {
        self.span.record("input_tokens", usage.input);
        self.span.record("output_tokens", usage.output);
        self.span.record("reasoning_tokens", usage.reasoning);
        self.span.record("cache_read_tokens", usage.cache_read);
        self.span.record("cache_write_tokens", usage.cache_write);
    }

    fn finish_completed(&mut self, response: &Response) {
        if self.finished {
            return;
        }
        self.record_usage(response.usage);
        if let Some(cost) = response.cost {
            self.span.record("cost_usd_micros", cost.usd_micros);
            self.span
                .record("cost_source", cost_source_name(cost.source));
        }
        if response.finish_reason == FinishReason::Incomplete {
            self.record_terminal(OUTCOME_INCOMPLETE, None);
            tracing::warn!(
                parent: &self.span,
                outcome = OUTCOME_INCOMPLETE,
                "LLM call finished without a provider finish reason"
            );
        } else {
            self.record_terminal(OUTCOME_COMPLETED, None);
            tracing::debug!(
                parent: &self.span,
                outcome = OUTCOME_COMPLETED,
                "LLM call finished"
            );
        }
    }

    fn finish_error(&mut self, error: &Error) {
        let kind = error_kind_name(error.kind());
        let outcome = if error.kind() == ErrorKind::Cancelled {
            OUTCOME_CANCELLED
        } else {
            OUTCOME_FAILED
        };
        if !self.record_terminal(outcome, Some(kind)) {
            return;
        }
        if let Some(status) = error.status() {
            self.span.record("status", u64::from(status));
        }
        if let Some(code) = error.provider_code() {
            self.span.record("provider_code", code);
        }
        if error.kind() == ErrorKind::Cancelled {
            tracing::debug!(
                parent: &self.span,
                outcome,
                error_kind = kind,
                "LLM call finished"
            );
        } else {
            tracing::warn!(
                parent: &self.span,
                outcome,
                error_kind = kind,
                "LLM call finished"
            );
        }
    }

    fn finish_incomplete_stream(&mut self) {
        let kind = error_kind_name(ErrorKind::StreamDecode);
        if !self.record_terminal(OUTCOME_FAILED, Some(kind)) {
            return;
        }
        tracing::warn!(
            parent: &self.span,
            outcome = OUTCOME_FAILED,
            error_kind = kind,
            "LLM stream ended without a terminal event"
        );
    }

    fn finish_dropped(&mut self) {
        if self.cancellation.is_cancelled() {
            let kind = error_kind_name(ErrorKind::Cancelled);
            if self.record_terminal(OUTCOME_CANCELLED, Some(kind)) {
                tracing::debug!(
                    parent: &self.span,
                    outcome = OUTCOME_CANCELLED,
                    error_kind = kind,
                    "LLM call finished"
                );
            }
            return;
        }
        if self
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            let kind = error_kind_name(ErrorKind::Timeout);
            if self.record_terminal(OUTCOME_FAILED, Some(kind)) {
                tracing::warn!(
                    parent: &self.span,
                    outcome = OUTCOME_FAILED,
                    error_kind = kind,
                    "LLM call finished"
                );
            }
            return;
        }
        if self.record_terminal(OUTCOME_DROPPED, None) {
            tracing::debug!(
                parent: &self.span,
                outcome = OUTCOME_DROPPED,
                "LLM call finished"
            );
        }
    }

    fn record_terminal(&mut self, outcome: &'static str, error_kind: Option<&'static str>) -> bool {
        if self.finished {
            return false;
        }
        self.finished = true;
        self.span.record("outcome", outcome);
        if let Some(kind) = error_kind {
            self.span.record("error_kind", kind);
        }
        true
    }
}

impl Drop for CallTrace {
    fn drop(&mut self) {
        if !self.finished {
            self.finish_dropped();
        }
    }
}

fn trace_stream(stream: ResponseStream, trace: CallTrace) -> ResponseStream {
    Box::pin(TracedStream {
        stream,
        trace: Some(trace),
    })
}

struct TracedStream {
    stream: ResponseStream,
    trace:  Option<CallTrace>,
}

impl TracedStream {
    fn finish(&mut self, finish: impl FnOnce(&mut CallTrace)) {
        if let Some(mut trace) = self.trace.take() {
            finish(&mut trace);
        }
    }
}

impl Stream for TracedStream {
    type Item = Result<StreamEvent, Error>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let span = self.trace.as_ref().map(|trace| trace.span.clone());
        let result = if let Some(span) = span.as_ref() {
            let _entered = span.enter();
            self.stream.as_mut().poll_next(context)
        } else {
            self.stream.as_mut().poll_next(context)
        };

        match &result {
            Poll::Ready(Some(Ok(StreamEvent::Completed { response }))) => {
                self.finish(|trace| trace.finish_completed(response));
            }
            Poll::Ready(Some(Ok(StreamEvent::Usage { usage }))) => {
                if let Some(trace) = self.trace.as_ref() {
                    trace.record_usage(*usage);
                }
            }
            Poll::Ready(Some(Err(error))) => {
                self.finish(|trace| trace.finish_error(error));
            }
            Poll::Ready(None) => {
                self.finish(CallTrace::finish_incomplete_stream);
            }
            Poll::Pending | Poll::Ready(Some(Ok(_))) => {}
        }
        result
    }
}

fn mode_name(mode: Mode) -> &'static str {
    match mode {
        Mode::Complete => "complete",
        Mode::Stream => "stream",
    }
}

fn cost_source_name(source: CostSource) -> &'static str {
    match source {
        CostSource::Catalog => "catalog",
        CostSource::Provider => "provider",
        CostSource::Application => "application",
    }
}

fn error_kind_name(kind: ErrorKind) -> &'static str {
    match kind {
        ErrorKind::Configuration => "configuration",
        ErrorKind::ModelSelection => "model_selection",
        ErrorKind::Authentication => "authentication",
        ErrorKind::AccessDenied => "access_denied",
        ErrorKind::NotFound => "not_found",
        ErrorKind::InvalidRequest => "invalid_request",
        ErrorKind::ContextLength => "context_length",
        ErrorKind::RateLimit => "rate_limit",
        ErrorKind::QuotaExceeded => "quota_exceeded",
        ErrorKind::ContentFilter => "content_filter",
        ErrorKind::Server => "server",
        ErrorKind::Provider => "provider",
        ErrorKind::Network => "network",
        ErrorKind::Timeout => "timeout",
        ErrorKind::StreamDecode => "stream_decode",
        ErrorKind::ResponseDecode => "response_decode",
        ErrorKind::Middleware => "middleware",
        ErrorKind::Cancelled => "cancelled",
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::error::Error as StdError;
    use std::fmt;
    use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

    use futures_util::StreamExt as _;
    use futures_util::stream::{iter, pending};
    use tracing::dispatcher::set_default;
    use tracing::field::{Field, Visit};
    use tracing::span::{Attributes, Id, Record};
    use tracing::{Dispatch, Event, Subscriber};
    use tracing_subscriber::layer::{Context, SubscriberExt as _};
    use tracing_subscriber::{Layer, registry};

    use super::{
        CallTrace, OUTCOME_CANCELLED, OUTCOME_COMPLETED, OUTCOME_DROPPED, OUTCOME_FAILED,
        OUTCOME_INCOMPLETE, trace_stream,
    };
    use crate::catalog::{Catalog, ModelId, ProviderId};
    use crate::middleware::{Call, CallContext, Mode};
    use crate::resolver::{AvailableProviders, CatalogResolver, ModelResolver};
    use crate::types::{
        Cost, CostSource, Error, ErrorKind, FinishReason, Request, Response, StreamEvent,
        TokenCounts,
    };

    const TEST_CATALOG: &str = r#"
        schema_version = 1

        [providers.test]
        display_name = "Test"
        adapter = "test-adapter"
        codec = "test-codec"
        base_url = "http://127.0.0.1"
        auth = { type = "none" }

        [providers.test.models.model]
        display_name = "Model"
        api_model = "model-v1"
    "#;

    #[derive(Clone, Debug, Default)]
    struct Capture {
        active: Arc<Mutex<BTreeMap<u64, CapturedSpan>>>,
        closed: Arc<Mutex<Vec<CapturedSpan>>>,
        events: Arc<Mutex<Vec<BTreeMap<String, String>>>>,
    }

    #[derive(Clone, Debug, Default)]
    struct CapturedSpan {
        name:   String,
        fields: BTreeMap<String, String>,
    }

    impl Capture {
        fn closed(&self) -> Vec<CapturedSpan> {
            lock(&self.closed).clone()
        }

        fn events(&self) -> Vec<BTreeMap<String, String>> {
            lock(&self.events).clone()
        }
    }

    impl<S: Subscriber> Layer<S> for Capture {
        fn on_new_span(&self, attributes: &Attributes<'_>, id: &Id, _context: Context<'_, S>) {
            if attributes.metadata().name() != "llm.call" {
                return;
            }
            let mut span = CapturedSpan {
                name: attributes.metadata().name().to_owned(),
                ..CapturedSpan::default()
            };
            attributes.record(&mut FieldVisitor(&mut span.fields));
            lock(&self.active).insert(id.clone().into_u64(), span);
        }

        fn on_record(&self, id: &Id, values: &Record<'_>, _context: Context<'_, S>) {
            if let Some(span) = lock(&self.active).get_mut(&id.clone().into_u64()) {
                values.record(&mut FieldVisitor(&mut span.fields));
            }
        }

        fn on_event(&self, event: &Event<'_>, _context: Context<'_, S>) {
            let mut fields = BTreeMap::new();
            event.record(&mut FieldVisitor(&mut fields));
            lock(&self.events).push(fields);
        }

        fn on_close(&self, id: Id, _context: Context<'_, S>) {
            if let Some(span) = lock(&self.active).remove(&id.into_u64()) {
                lock(&self.closed).push(span);
            }
        }
    }

    struct FieldVisitor<'a>(&'a mut BTreeMap<String, String>);

    impl Visit for FieldVisitor<'_> {
        fn record_u64(&mut self, field: &Field, value: u64) {
            self.0.insert(field.name().to_owned(), value.to_string());
        }

        fn record_str(&mut self, field: &Field, value: &str) {
            self.0.insert(field.name().to_owned(), value.to_owned());
        }

        fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
            self.0.insert(field.name().to_owned(), format!("{value:?}"));
        }
    }

    fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
        mutex.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn call(mode: Mode) -> Result<Call, Box<dyn StdError>> {
        let catalog = Catalog::builder().overlay_toml(TEST_CATALOG)?.build()?;
        let available = AvailableProviders::all(&catalog);
        let request = Request::builder()
            .model("test/model")
            .user("Hello")
            .build()?;
        let route = CatalogResolver.resolve(&request, &catalog, &available)?;
        Ok(Call {
            request,
            route,
            mode,
            context: CallContext::new(),
        })
    }

    fn response() -> Response {
        let mut response =
            Response::new(ProviderId::new("test"), ModelId::new("model"), Vec::new());
        response.usage = TokenCounts {
            input:       10,
            output:      20,
            reasoning:   30,
            cache_read:  40,
            cache_write: 50,
        };
        response.cost = Some(Cost {
            usd_micros: 60,
            source:     CostSource::Catalog,
        });
        response
    }

    fn field<'a>(span: &'a CapturedSpan, name: &str) -> Option<&'a str> {
        span.fields.get(name).map(String::as_str)
    }

    fn emitted(capture: &Capture, outcome: &str) -> bool {
        capture
            .events()
            .iter()
            .any(|fields| fields.get("outcome").is_some_and(|value| value == outcome))
    }

    #[test]
    fn a_complete_call_records_stable_fields_and_closes() -> Result<(), Box<dyn StdError>> {
        let capture = Capture::default();
        let dispatch = Dispatch::new(registry().with(capture.clone()));
        let _guard = set_default(&dispatch);
        let mut trace = CallTrace::new(&call(Mode::Complete)?);

        trace.finish_completed(&response());
        drop(trace);

        let closed = capture.closed();
        assert_eq!(closed.len(), 1);
        assert_eq!(closed[0].name, "llm.call");
        assert!(field(&closed[0], "call_id").is_some());
        assert_eq!(field(&closed[0], "attempt"), Some("1"));
        assert_eq!(field(&closed[0], "provider"), Some("test"));
        assert_eq!(field(&closed[0], "model"), Some("model"));
        assert_eq!(field(&closed[0], "mode"), Some("complete"));
        assert_eq!(field(&closed[0], "outcome"), Some(OUTCOME_COMPLETED));
        assert_eq!(field(&closed[0], "input_tokens"), Some("10"));
        assert_eq!(field(&closed[0], "output_tokens"), Some("20"));
        assert_eq!(field(&closed[0], "reasoning_tokens"), Some("30"));
        assert_eq!(field(&closed[0], "cache_read_tokens"), Some("40"));
        assert_eq!(field(&closed[0], "cache_write_tokens"), Some("50"));
        assert_eq!(field(&closed[0], "cost_usd_micros"), Some("60"));
        assert_eq!(field(&closed[0], "cost_source"), Some("catalog"));
        assert!(emitted(&capture, OUTCOME_COMPLETED));
        Ok(())
    }

    #[test]
    fn an_incomplete_call_records_its_own_outcome() -> Result<(), Box<dyn StdError>> {
        let capture = Capture::default();
        let dispatch = Dispatch::new(registry().with(capture.clone()));
        let _guard = set_default(&dispatch);
        let mut trace = CallTrace::new(&call(Mode::Complete)?);
        let mut response = response();
        response.finish_reason = FinishReason::Incomplete;

        trace.finish_completed(&response);
        drop(trace);

        let closed = capture.closed();
        assert_eq!(closed.len(), 1);
        assert_eq!(field(&closed[0], "outcome"), Some(OUTCOME_INCOMPLETE));
        assert!(emitted(&capture, OUTCOME_INCOMPLETE));
        assert!(!emitted(&capture, OUTCOME_COMPLETED));
        Ok(())
    }

    #[tokio::test]
    async fn a_stream_span_stays_open_until_completed() -> Result<(), Box<dyn StdError>> {
        let capture = Capture::default();
        let dispatch = Dispatch::new(registry().with(capture.clone()));
        let _guard = set_default(&dispatch);
        let trace = CallTrace::new(&call(Mode::Stream)?);
        let mut stream = trace_stream(
            Box::pin(iter([
                Ok(StreamEvent::Started {
                    id: Some("response-1".to_owned()),
                }),
                Ok(StreamEvent::Usage {
                    usage: TokenCounts {
                        input: 7,
                        ..TokenCounts::default()
                    },
                }),
                Ok(StreamEvent::Completed {
                    response: response(),
                }),
            ])),
            trace,
        );

        assert!(capture.closed().is_empty());
        assert!(matches!(
            stream.next().await,
            Some(Ok(StreamEvent::Started { .. }))
        ));
        assert!(capture.closed().is_empty());
        assert!(matches!(
            stream.next().await,
            Some(Ok(StreamEvent::Usage { .. }))
        ));
        assert!(capture.closed().is_empty());
        assert!(matches!(
            stream.next().await,
            Some(Ok(StreamEvent::Completed { .. }))
        ));

        let closed = capture.closed();
        assert_eq!(closed.len(), 1);
        assert_eq!(field(&closed[0], "mode"), Some("stream"));
        assert_eq!(field(&closed[0], "outcome"), Some(OUTCOME_COMPLETED));
        assert_eq!(field(&closed[0], "input_tokens"), Some("10"));
        assert!(emitted(&capture, OUTCOME_COMPLETED));
        Ok(())
    }

    #[tokio::test]
    async fn a_stream_error_records_failure_and_closes() -> Result<(), Box<dyn StdError>> {
        let capture = Capture::default();
        let dispatch = Dispatch::new(registry().with(capture.clone()));
        let _guard = set_default(&dispatch);
        let trace = CallTrace::new(&call(Mode::Stream)?);
        let error = Error::new(ErrorKind::RateLimit, "slow down")
            .with_status(429)
            .with_provider_code("rate_limited");
        let mut stream = trace_stream(Box::pin(iter([Err(error)])), trace);

        let error = stream
            .next()
            .await
            .expect("the stream should yield an error")
            .expect_err("the event should fail");

        assert_eq!(error.kind(), ErrorKind::RateLimit);
        let closed = capture.closed();
        assert_eq!(closed.len(), 1);
        assert_eq!(field(&closed[0], "outcome"), Some(OUTCOME_FAILED));
        assert_eq!(field(&closed[0], "error_kind"), Some("rate_limit"));
        assert_eq!(field(&closed[0], "status"), Some("429"));
        assert_eq!(field(&closed[0], "provider_code"), Some("rate_limited"));
        assert!(emitted(&capture, OUTCOME_FAILED));
        Ok(())
    }

    #[tokio::test]
    async fn a_stream_without_a_terminal_event_records_failure() -> Result<(), Box<dyn StdError>> {
        let capture = Capture::default();
        let dispatch = Dispatch::new(registry().with(capture.clone()));
        let _guard = set_default(&dispatch);
        let trace = CallTrace::new(&call(Mode::Stream)?);
        let events: Vec<Result<StreamEvent, Error>> = Vec::new();
        let mut stream = trace_stream(Box::pin(iter(events)), trace);

        assert!(stream.next().await.is_none());

        let closed = capture.closed();
        assert_eq!(closed.len(), 1);
        assert_eq!(field(&closed[0], "outcome"), Some(OUTCOME_FAILED));
        assert_eq!(field(&closed[0], "error_kind"), Some("stream_decode"));
        assert!(emitted(&capture, OUTCOME_FAILED));
        Ok(())
    }

    #[test]
    fn dropping_an_unfinished_stream_records_drop() -> Result<(), Box<dyn StdError>> {
        let capture = Capture::default();
        let dispatch = Dispatch::new(registry().with(capture.clone()));
        let _guard = set_default(&dispatch);
        let trace = CallTrace::new(&call(Mode::Stream)?);
        let stream = trace_stream(Box::pin(pending()), trace);

        assert!(capture.closed().is_empty());
        drop(stream);

        let closed = capture.closed();
        assert_eq!(closed.len(), 1);
        assert_eq!(field(&closed[0], "outcome"), Some(OUTCOME_DROPPED));
        assert!(emitted(&capture, OUTCOME_DROPPED));
        Ok(())
    }

    #[test]
    fn cancelling_an_unfinished_stream_records_cancellation() -> Result<(), Box<dyn StdError>> {
        let capture = Capture::default();
        let dispatch = Dispatch::new(registry().with(capture.clone()));
        let _guard = set_default(&dispatch);
        let call = call(Mode::Stream)?;
        let cancellation = call.context.cancellation().clone();
        let trace = CallTrace::new(&call);
        let stream = trace_stream(Box::pin(pending()), trace);

        cancellation.cancel();
        drop(stream);

        let closed = capture.closed();
        assert_eq!(closed.len(), 1);
        assert_eq!(field(&closed[0], "outcome"), Some(OUTCOME_CANCELLED));
        assert_eq!(field(&closed[0], "error_kind"), Some("cancelled"));
        assert!(emitted(&capture, OUTCOME_CANCELLED));
        Ok(())
    }
}
