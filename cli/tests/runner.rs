use std::io::{self, Cursor, Write};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use futures_util::stream;
use lithos_llm::adapter::{ProviderAdapter, ResolvedCall};
use lithos_llm::catalog::{AdapterId, Catalog, ModelId, ProviderId};
use lithos_llm::middleware::CancellationToken;
use lithos_llm::types::{
    CacheHint, ContentBlockId, ContentPart, Error, ErrorKind, ReasoningEffort, Request, Response,
    ResponseFormat, ResponseStream, Speed, StreamEvent,
};
use lithos_llm::{Client, ClientBuild};
use lithos_llm_cli::{ExitStatus, ProcessIo, TerminalState, run};

const CATALOG: &str = r#"
schema_version = 1

[providers.alpha]
display_name = "Alpha"
aliases = ["a"]
adapter = "fake"
codec = "fake"
base_url = "http://127.0.0.1"
priority = 10
default_model = "one"
auth = { type = "none" }

[providers.alpha.models.one]
display_name = "One Model"
aliases = ["uno"]
api_model = "one-v1"
capabilities = { text = true, images = true, audio = true, documents = true, tools = true, structured_output = true, reasoning = true, reasoning_effort_levels = true, caching = true, cache_routing = true, sampling = true }

[providers.beta]
display_name = "Beta"
adapter = "missing"
codec = "fake"
base_url = "http://127.0.0.1"
priority = 1
default_model = "two"
auth = { type = "none" }

[providers.beta.models.two]
display_name = "Two Model"
aliases = ["dos"]
api_model = "two-v1"
capabilities = { text = true }
"#;

#[derive(Clone, Default)]
struct RecordingAdapter {
    requests: Arc<Mutex<Vec<Request>>>,
}

#[async_trait]
impl ProviderAdapter for RecordingAdapter {
    fn id(&self) -> &AdapterId {
        static ID: OnceLock<AdapterId> = OnceLock::new();
        ID.get_or_init(|| AdapterId::new("fake"))
    }

    async fn complete(&self, call: &ResolvedCall) -> Result<Response, Error> {
        self.requests
            .lock()
            .expect("request lock should work")
            .push(call.request().clone());
        Ok(response("complete output"))
    }

    async fn stream(&self, call: &ResolvedCall) -> Result<ResponseStream, Error> {
        self.requests
            .lock()
            .expect("request lock should work")
            .push(call.request().clone());
        let id = ContentBlockId::new("text-0");
        Ok(Box::pin(stream::iter([
            Ok(StreamEvent::TextDelta {
                id:   id.clone(),
                text: "stream ".to_owned(),
            }),
            Ok(StreamEvent::TextDelta {
                id,
                text: "output".to_owned(),
            }),
            Ok(StreamEvent::Completed {
                response: response("stream output"),
            }),
        ])))
    }
}

struct FailingAdapter;

#[async_trait]
impl ProviderAdapter for FailingAdapter {
    fn id(&self) -> &AdapterId {
        static ID: OnceLock<AdapterId> = OnceLock::new();
        ID.get_or_init(|| AdapterId::new("fake"))
    }

    async fn complete(&self, _call: &ResolvedCall) -> Result<Response, Error> {
        Err(provider_error())
    }

    async fn stream(&self, _call: &ResolvedCall) -> Result<ResponseStream, Error> {
        Err(provider_error())
    }
}

fn provider_error() -> Error {
    Error::new(ErrorKind::RateLimit, "request was limited")
        .with_provider(ProviderId::new("alpha"))
        .with_status(429)
        .with_provider_code("rate_limit")
        .with_provider_retry_after(Duration::from_secs(2))
}

fn response(text: &str) -> Response {
    Response::new(ProviderId::new("alpha"), ModelId::new("one"), vec![
        ContentPart::Text {
            text: text.to_owned(),
        },
    ])
}

fn client(adapter: impl ProviderAdapter + 'static) -> ClientBuild {
    let catalog = Catalog::builder()
        .overlay_toml(CATALOG)
        .expect("catalog overlay should parse")
        .build()
        .expect("catalog should build");
    Client::builder()
        .catalog(catalog)
        .adapter("alpha", adapter)
        .build()
        .expect("client should build")
}

async fn invoke(
    client: &Client,
    arguments: &[&str],
    stdin: Vec<u8>,
    stdin_is_terminal: bool,
    cancellation: CancellationToken,
) -> (ExitStatus, Vec<u8>, Vec<u8>) {
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let status = run(
        arguments.iter().copied(),
        client,
        ProcessIo {
            stdin:  Cursor::new(stdin),
            stdout: &mut stdout,
            stderr: &mut stderr,
        },
        TerminalState {
            stdin:  stdin_is_terminal,
            stdout: false,
            stderr: false,
        },
        cancellation,
    )
    .await;
    (status, stdout, stderr)
}

#[tokio::test]
async fn maps_every_request_control_and_provider_option() {
    let adapter = RecordingAdapter::default();
    let build = client(adapter.clone());
    let (status, stdout, stderr) = invoke(
        &build.client,
        &[
            "lithos",
            "prompt",
            "hello",
            "world",
            "--no-stream",
            "--model",
            "alpha/one",
            "--system",
            "system text",
            "--max-output-tokens",
            "42",
            "--temperature",
            "0.5",
            "--top-p",
            "0.8",
            "--reasoning-effort",
            "high",
            "--speed",
            "economical",
            "--timeout",
            "2s",
            "--stop",
            "END",
            "--stop",
            "DONE",
            "--cache-key",
            "cache-one",
            "--metadata",
            "trace=yes",
            "--metadata",
            "trace=last",
            "--option",
            "count=2",
            "--option",
            "count=3",
            "--option",
            "label=plain",
        ],
        Vec::new(),
        true,
        CancellationToken::new(),
    )
    .await;

    assert_eq!(status, ExitStatus::Success);
    assert_eq!(stdout, b"complete output\n");
    assert!(stderr.is_empty());
    let requests = adapter.requests.lock().expect("request lock should work");
    let request = requests.last().expect("adapter should receive a request");
    assert_eq!(request.model(), "alpha/one");
    assert_eq!(request.max_output_tokens(), Some(42));
    assert_eq!(request.temperature(), Some(0.5));
    assert_eq!(request.top_p(), Some(0.8));
    assert_eq!(request.reasoning_effort(), Some(ReasoningEffort::High));
    assert_eq!(request.speed(), Some(Speed::Economical));
    assert_eq!(request.timeout(), Some(Duration::from_secs(2)));
    assert_eq!(request.stop_sequences(), ["END", "DONE"]);
    assert_eq!(
        request.cache_hint(),
        Some(&CacheHint::Key {
            key: "cache-one".to_owned(),
        })
    );
    assert_eq!(
        request.metadata().get("trace").map(String::as_str),
        Some("last")
    );
    let options = request
        .options_for(&ProviderId::new("alpha"))
        .expect("provider options should use the canonical provider");
    assert_eq!(options["count"], 3);
    assert_eq!(options["label"], "plain");
    assert_eq!(request.messages().len(), 2);
    assert!(matches!(
        request.messages()[1].content(),
        [ContentPart::Text { text }] if text == "hello world"
    ));
}

#[tokio::test]
async fn streams_only_text_deltas_and_adds_one_newline() {
    let build = client(RecordingAdapter::default());
    let (status, stdout, stderr) = invoke(
        &build.client,
        &["lithos", "hello", "--model", "alpha/one"],
        Vec::new(),
        true,
        CancellationToken::new(),
    )
    .await;

    assert_eq!(status, ExitStatus::Success);
    assert_eq!(stdout, b"stream output\n");
    assert!(stderr.is_empty());
}

#[tokio::test]
async fn no_cache_maps_to_a_disabled_cache_hint() {
    let adapter = RecordingAdapter::default();
    let build = client(adapter.clone());
    let (status, _, _) = invoke(
        &build.client,
        &[
            "lithos",
            "hello",
            "--model",
            "alpha/one",
            "--no-stream",
            "--no-cache",
        ],
        Vec::new(),
        true,
        CancellationToken::new(),
    )
    .await;

    assert_eq!(status, ExitStatus::Success);
    let requests = adapter.requests.lock().expect("request lock should work");
    assert_eq!(
        requests.last().and_then(Request::cache_hint),
        Some(&CacheHint::Disabled)
    );
}

#[tokio::test]
async fn json_output_buffers_and_contains_the_request_and_response() {
    let build = client(RecordingAdapter::default());
    let (status, stdout, stderr) = invoke(
        &build.client,
        &["lithos", "hello", "--model", "alpha/one", "--json"],
        Vec::new(),
        true,
        CancellationToken::new(),
    )
    .await;

    assert_eq!(status, ExitStatus::Success);
    let envelope: serde_json::Value =
        serde_json::from_slice(&stdout).expect("output should be JSON");
    assert_eq!(envelope["version"], 1);
    assert_eq!(envelope["request"]["model"], "alpha/one");
    assert_eq!(envelope["response"]["model"]["provider"], "alpha");
    assert!(stderr.is_empty());
}

#[tokio::test]
async fn schema_flags_map_to_a_named_json_schema() {
    let adapter = RecordingAdapter::default();
    let build = client(adapter.clone());
    let (status, _, _) = invoke(
        &build.client,
        &[
            "lithos",
            "hello",
            "--model",
            "alpha/one",
            "--no-stream",
            "--schema",
            r#"{"type":"object"}"#,
            "--schema-name",
            "answer",
        ],
        Vec::new(),
        true,
        CancellationToken::new(),
    )
    .await;

    assert_eq!(status, ExitStatus::Success);
    let requests = adapter.requests.lock().expect("request lock should work");
    assert!(matches!(
        requests.last().and_then(Request::response_format),
        Some(ResponseFormat::JsonSchema { name, schema })
            if name == "answer" && schema["type"] == "object"
    ));
}

#[tokio::test]
async fn reports_safe_provider_error_fields_on_standard_error() {
    let build = client(FailingAdapter);
    let (status, stdout, stderr) = invoke(
        &build.client,
        &["lithos", "hello", "--model", "alpha/one", "--no-stream"],
        Vec::new(),
        true,
        CancellationToken::new(),
    )
    .await;

    assert_eq!(status, ExitStatus::Failure);
    assert!(stdout.is_empty());
    let diagnostic = String::from_utf8(stderr).expect("diagnostic should be UTF-8");
    assert!(diagnostic.contains("error: rate_limit: request was limited"));
    assert!(diagnostic.contains("provider=alpha"));
    assert!(diagnostic.contains("status=429"));
    assert!(diagnostic.contains("code=rate_limit"));
    assert!(diagnostic.contains("retry_after=2s"));
}

#[tokio::test]
async fn a_cancelled_signal_returns_status_130() {
    let build = client(RecordingAdapter::default());
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let (status, stdout, stderr) = invoke(
        &build.client,
        &["lithos", "hello", "--model", "alpha/one"],
        Vec::new(),
        true,
        cancellation,
    )
    .await;

    assert_eq!(status, ExitStatus::Interrupted);
    assert!(stdout.is_empty());
    assert!(String::from_utf8_lossy(&stderr).contains("interrupted"));
}

#[tokio::test]
async fn model_listing_marks_runtime_availability_and_filters_it() {
    let build = client(RecordingAdapter::default());
    let (status, stdout, stderr) = invoke(
        &build.client,
        &["lithos", "models", "--json"],
        Vec::new(),
        true,
        CancellationToken::new(),
    )
    .await;
    assert_eq!(status, ExitStatus::Success);
    let listing: serde_json::Value =
        serde_json::from_slice(&stdout).expect("output should be JSON");
    assert_eq!(listing["models"][0]["selector"], "alpha/one");
    assert_eq!(listing["models"][0]["available"], true);
    assert_eq!(listing["models"][1]["selector"], "beta/two");
    assert_eq!(listing["models"][1]["available"], false);
    assert!(stderr.is_empty());

    let (_, filtered, _) = invoke(
        &build.client,
        &["lithos", "models", "--json", "--available"],
        Vec::new(),
        true,
        CancellationToken::new(),
    )
    .await;
    let listing: serde_json::Value =
        serde_json::from_slice(&filtered).expect("output should be JSON");
    assert_eq!(listing["models"].as_array().map(Vec::len), Some(1));
}

#[tokio::test]
async fn model_search_is_case_insensitive_and_reports_no_match() {
    let adapter = RecordingAdapter::default();
    let build = client(adapter.clone());
    let (status, _, _) = invoke(
        &build.client,
        &["lithos", "hello", "--model-query", "UNO", "--no-stream"],
        Vec::new(),
        true,
        CancellationToken::new(),
    )
    .await;
    assert_eq!(status, ExitStatus::Success);
    {
        let requests = adapter.requests.lock().expect("request lock should work");
        assert_eq!(requests.last().map(Request::model), Some("alpha/one"));
    }

    let (status, stdout, stderr) = invoke(
        &build.client,
        &["lithos", "hello", "-q", "missing"],
        Vec::new(),
        true,
        CancellationToken::new(),
    )
    .await;
    assert_eq!(status, ExitStatus::Usage);
    assert!(stdout.is_empty());
    assert!(String::from_utf8_lossy(&stderr).contains("missing"));
}

struct BrokenWriter;

impl Write for BrokenWriter {
    fn write(&mut self, _buffer: &[u8]) -> io::Result<usize> {
        Err(io::Error::new(io::ErrorKind::BrokenPipe, "closed"))
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn a_closed_output_pipe_is_success() {
    let build = client(RecordingAdapter::default());
    let mut stderr = Vec::new();
    let status = run(
        ["lithos", "hello", "--model", "alpha/one"],
        &build.client,
        ProcessIo {
            stdin:  Cursor::new(Vec::new()),
            stdout: BrokenWriter,
            stderr: &mut stderr,
        },
        TerminalState {
            stdin:  true,
            stdout: false,
            stderr: false,
        },
        CancellationToken::new(),
    )
    .await;

    assert_eq!(status, ExitStatus::Success);
    assert!(stderr.is_empty());
}

#[tokio::test]
async fn non_utf8_standard_input_is_a_usage_error() {
    let build = client(RecordingAdapter::default());
    let (status, stdout, stderr) = invoke(
        &build.client,
        &["lithos", "--model", "alpha/one"],
        vec![0xff],
        false,
        CancellationToken::new(),
    )
    .await;

    assert_eq!(status, ExitStatus::Usage);
    assert!(stdout.is_empty());
    assert!(String::from_utf8_lossy(&stderr).contains("UTF-8"));
}
