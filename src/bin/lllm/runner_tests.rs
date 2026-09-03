use std::error::Error as StdError;
use std::io::{self, Cursor, Write};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use std::{env, fmt, fs, process};

use async_trait::async_trait;
use futures_util::stream;
use lithos_llm::adapter::{ProviderAdapter, ResolvedCall};
use lithos_llm::catalog::{AdapterId, Catalog, ModelId, ProviderId};
use lithos_llm::credentials::CredentialError;
use lithos_llm::middleware::CancellationToken;
use lithos_llm::types::{
    CacheHint, ContentBlockId, ContentPart, Cost, CostSource, Error, ErrorKind, ReasoningEffort,
    Request, Response, ResponseFormat, ResponseStream, Speed, StreamEvent, TokenCounts,
};
use lithos_llm::{Client, ClientBuild};

use crate::app::{CliEnvironment, ExitStatus, ProcessIo, TerminalState, run};

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

struct CredentialFailingAdapter;

#[derive(Debug)]
struct ProviderSource;

impl fmt::Display for ProviderSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("the upstream connection closed")
    }
}

impl StdError for ProviderSource {}

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

#[async_trait]
impl ProviderAdapter for CredentialFailingAdapter {
    fn id(&self) -> &AdapterId {
        static ID: OnceLock<AdapterId> = OnceLock::new();
        ID.get_or_init(|| AdapterId::new("fake"))
    }

    async fn complete(&self, _call: &ResolvedCall) -> Result<Response, Error> {
        Err(credential_error())
    }

    async fn stream(&self, _call: &ResolvedCall) -> Result<ResponseStream, Error> {
        Err(credential_error())
    }
}

fn provider_error() -> Error {
    Error::new(ErrorKind::RateLimit, "request was limited")
        .with_provider(ProviderId::new("alpha"))
        .with_status(429)
        .with_provider_code("rate_limit")
        .with_provider_retry_after(Duration::from_secs(2))
        .with_source(ProviderSource)
}

fn credential_error() -> Error {
    Error::new(ErrorKind::Authentication, "credentials are unavailable")
        .with_provider(ProviderId::new("alpha"))
        .with_source(CredentialError::Environment {
            provider: ProviderId::new("alpha"),
            variable: "ALPHA_API_KEY".to_owned(),
            source:   env::VarError::NotPresent,
        })
}

fn response(text: &str) -> Response {
    let mut response = Response::new(ProviderId::new("alpha"), ModelId::new("one"), vec![
        ContentPart::Text {
            text: text.to_owned(),
        },
    ]);
    response.usage = TokenCounts {
        input:       1_000,
        output:      150,
        reasoning:   33,
        cache_read:  200,
        cache_write: 40,
    };
    response.cost = Some(Cost {
        usd_micros: 4_310,
        source:     CostSource::Catalog,
    });
    response
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
    invoke_with_environment(
        client,
        arguments,
        stdin,
        stdin_is_terminal,
        &test_environment(None),
        cancellation,
    )
    .await
}

fn test_environment(model: Option<&str>) -> CliEnvironment {
    CliEnvironment::new(model.map(Into::into), [
        ProviderId::new("alpha"),
        ProviderId::new("beta"),
    ])
}

async fn invoke_with_environment(
    client: &Client,
    arguments: &[&str],
    stdin: Vec<u8>,
    stdin_is_terminal: bool,
    environment: &CliEnvironment,
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
            stdin: stdin_is_terminal,
        },
        environment,
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
            "lllm",
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
async fn local_fragments_become_user_and_system_context() {
    let directory = env::temp_dir();
    let user_one = directory.join(format!("lllm-user-one-{}.txt", process::id()));
    let user_two = directory.join(format!("lllm-user-two-{}.txt", process::id()));
    let system = directory.join(format!("lllm-system-{}.txt", process::id()));
    fs::write(&user_one, "first context").expect("fixture should be written");
    fs::write(&user_two, "second context").expect("fixture should be written");
    fs::write(&system, "review policy").expect("fixture should be written");

    let adapter = RecordingAdapter::default();
    let build = client(adapter.clone());
    let arguments = [
        "lllm",
        "Review",
        "--model",
        "alpha/one",
        "--no-stream",
        "--system",
        "Be precise",
        "--system-fragment",
        system.to_str().expect("path should be UTF-8"),
        "-f",
        user_one.to_str().expect("path should be UTF-8"),
        "-f",
        user_two.to_str().expect("path should be UTF-8"),
    ];
    let (status, _, stderr) = invoke(
        &build.client,
        &arguments,
        Vec::new(),
        true,
        CancellationToken::new(),
    )
    .await;

    fs::remove_file(user_one).expect("fixture should be removed");
    fs::remove_file(user_two).expect("fixture should be removed");
    fs::remove_file(system).expect("fixture should be removed");
    assert_eq!(status, ExitStatus::Success);
    assert!(stderr.is_empty());
    let requests = adapter.requests.lock().expect("request lock should work");
    let request = requests.last().expect("adapter should receive a request");
    assert!(matches!(
        request.messages()[0].content(),
        [ContentPart::Text { text }] if text == "Be precise\n\nreview policy"
    ));
    assert!(matches!(
        request.messages()[1].content(),
        [
            ContentPart::Text { text: first },
            ContentPart::Text { text: second },
            ContentPart::Text { text: prompt },
        ] if first == "first context" && second == "second context" && prompt == "Review"
    ));
}

#[tokio::test]
async fn streams_only_text_deltas_and_adds_one_newline() {
    let build = client(RecordingAdapter::default());
    let (status, stdout, stderr) = invoke(
        &build.client,
        &["lllm", "hello", "--model", "alpha/one"],
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
async fn usage_is_written_to_standard_error_for_streaming() {
    let build = client(RecordingAdapter::default());
    let (status, stdout, stderr) = invoke(
        &build.client,
        &["lllm", "hello", "--model", "alpha/one", "--usage"],
        Vec::new(),
        true,
        CancellationToken::new(),
    )
    .await;

    assert_eq!(status, ExitStatus::Success);
    assert_eq!(stdout, b"stream output\n");
    let usage = String::from_utf8(stderr).expect("usage should be UTF-8");
    assert!(usage.starts_with("alpha/one · 1,240 input · 183 output · $0.00431 · "));
    assert!(usage.ends_with("ms\n") || usage.ends_with("s\n"));
}

#[tokio::test]
async fn usage_is_written_to_standard_error_for_non_streaming() {
    let build = client(RecordingAdapter::default());
    let (status, stdout, stderr) = invoke(
        &build.client,
        &[
            "lllm",
            "hello",
            "--model",
            "alpha/one",
            "--no-stream",
            "--usage",
        ],
        Vec::new(),
        true,
        CancellationToken::new(),
    )
    .await;

    assert_eq!(status, ExitStatus::Success);
    assert_eq!(stdout, b"complete output\n");
    let usage = String::from_utf8(stderr).expect("usage should be UTF-8");
    assert!(usage.starts_with("alpha/one · 1,240 input · 183 output · $0.00431 · "));
    assert!(usage.ends_with("ms\n") || usage.ends_with("s\n"));
}

#[tokio::test]
async fn no_cache_maps_to_a_disabled_cache_hint() {
    let adapter = RecordingAdapter::default();
    let build = client(adapter.clone());
    let (status, _, _) = invoke(
        &build.client,
        &[
            "lllm",
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
        &["lllm", "hello", "--model", "alpha/one", "--json"],
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
            "lllm",
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
async fn schema_multi_maps_the_shorthand_to_an_array_schema() {
    let adapter = RecordingAdapter::default();
    let build = client(adapter.clone());
    let (status, _, stderr) = invoke(
        &build.client,
        &[
            "lllm",
            "hello",
            "--model",
            "alpha/one",
            "--no-stream",
            "--schema-multi",
            "name, age int",
        ],
        Vec::new(),
        true,
        CancellationToken::new(),
    )
    .await;

    assert_eq!(status, ExitStatus::Success);
    assert!(stderr.is_empty());
    let requests = adapter.requests.lock().expect("request lock should work");
    assert!(matches!(
        requests.last().and_then(Request::response_format),
        Some(ResponseFormat::JsonSchema { schema, .. })
            if schema["type"] == "array"
                && schema["items"]["properties"]["age"]["type"] == "integer"
    ));
}

#[tokio::test]
async fn reports_safe_provider_error_fields_on_standard_error() {
    let build = client(FailingAdapter);
    let (status, stdout, stderr) = invoke(
        &build.client,
        &["lllm", "hello", "--model", "alpha/one", "--no-stream"],
        Vec::new(),
        true,
        CancellationToken::new(),
    )
    .await;

    assert_eq!(status, ExitStatus::Failure);
    assert!(stdout.is_empty());
    let diagnostic = String::from_utf8(stderr).expect("diagnostic should be UTF-8");
    assert!(diagnostic.contains("error: rate_limit: request was limited"));
    assert!(diagnostic.contains("route=alpha/one"));
    assert!(diagnostic.contains("provider=alpha"));
    assert!(diagnostic.contains("status=429"));
    assert!(diagnostic.contains("code=rate_limit"));
    assert!(diagnostic.contains("retry_after=2s"));
    assert!(diagnostic.contains("caused by: the upstream connection closed"));
}

#[tokio::test]
async fn credential_errors_name_the_route_and_environment_variable() {
    let build = client(CredentialFailingAdapter);
    let (status, stdout, stderr) = invoke(
        &build.client,
        &["lllm", "hello", "--model", "alpha/one", "--no-stream"],
        Vec::new(),
        true,
        CancellationToken::new(),
    )
    .await;

    assert_eq!(status, ExitStatus::Failure);
    assert!(stdout.is_empty());
    let diagnostic = String::from_utf8(stderr).expect("diagnostic should be UTF-8");
    assert!(diagnostic.contains("route=alpha/one"));
    assert!(diagnostic.contains("hint: set ALPHA_API_KEY"));
}

#[tokio::test]
async fn unknown_models_suggest_the_closest_canonical_selector() {
    let build = client(RecordingAdapter::default());
    let (status, stdout, stderr) = invoke(
        &build.client,
        &["lllm", "resolve", "--model", "alpha/onn"],
        Vec::new(),
        true,
        CancellationToken::new(),
    )
    .await;

    assert_eq!(status, ExitStatus::Usage);
    assert!(stdout.is_empty());
    assert!(String::from_utf8_lossy(&stderr).contains("Did you mean `alpha/one`?"));
}

#[tokio::test]
async fn top_level_help_shows_the_implicit_prompt_form_and_examples() {
    let build = client(RecordingAdapter::default());
    let (status, stdout, stderr) = invoke(
        &build.client,
        &["lllm", "--help"],
        Vec::new(),
        true,
        CancellationToken::new(),
    )
    .await;

    assert_eq!(status, ExitStatus::Success);
    assert!(stderr.is_empty());
    let help = String::from_utf8(stdout).expect("help should be UTF-8");
    assert!(help.contains("Usage: lllm [OPTIONS] [PROMPT]..."));
    assert!(help.contains("Examples:"));
}

#[tokio::test]
async fn prompt_help_shows_aliases_and_key_value_syntax() {
    let build = client(RecordingAdapter::default());
    let (status, stdout, stderr) = invoke(
        &build.client,
        &["lllm", "prompt", "--help"],
        Vec::new(),
        true,
        CancellationToken::new(),
    )
    .await;

    assert_eq!(status, ExitStatus::Success);
    assert!(stderr.is_empty());
    let help = String::from_utf8(stdout).expect("help should be UTF-8");
    assert!(help.contains("--xl"));
    assert!(help.contains("--metadata <KEY=VALUE>"));
    assert!(help.contains("--option <KEY=VALUE>"));
}

#[tokio::test]
async fn reports_input_error_source_chains() {
    let build = client(RecordingAdapter::default());
    let (status, stdout, stderr) = invoke(
        &build.client,
        &["lllm", "hello", "--schema", "{"],
        Vec::new(),
        true,
        CancellationToken::new(),
    )
    .await;

    assert_eq!(status, ExitStatus::Usage);
    assert!(stdout.is_empty());
    let diagnostic = String::from_utf8(stderr).expect("diagnostic should be UTF-8");
    assert!(diagnostic.contains("error: schema is not valid JSON"));
    assert!(diagnostic.contains("caused by:"));
    assert!(diagnostic.contains("line 1 column 1"), "{diagnostic}");
}

#[tokio::test]
async fn a_cancelled_signal_returns_status_130() {
    let build = client(RecordingAdapter::default());
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let (status, stdout, stderr) = invoke(
        &build.client,
        &["lllm", "hello", "--model", "alpha/one"],
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
        &["lllm", "models", "--json"],
        Vec::new(),
        true,
        CancellationToken::new(),
    )
    .await;
    assert_eq!(status, ExitStatus::Success);
    let listing: serde_json::Value =
        serde_json::from_slice(&stdout).expect("output should be JSON");
    assert_eq!(listing["models"][0]["selector"], "alpha/one");
    assert_eq!(listing["version"], 2);
    assert_eq!(listing["effective_default"], "alpha/one");
    assert_eq!(listing["models"][0]["adapter_compiled"], true);
    assert_eq!(listing["models"][0]["credentials_configured"], true);
    assert_eq!(listing["models"][0]["effective_default"], true);
    assert_eq!(listing["models"][1]["selector"], "beta/two");
    assert_eq!(listing["models"][1]["adapter_compiled"], false);
    assert_eq!(listing["models"][1]["credentials_configured"], true);
    assert!(stderr.is_empty());

    let (_, filtered, _) = invoke(
        &build.client,
        &["lllm", "models", "--json", "--adapter-compiled"],
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
async fn model_listing_filters_compose() {
    let build = client(RecordingAdapter::default());
    let environment = CliEnvironment::new(None, [ProviderId::new("alpha")]);
    let (status, stdout, stderr) = invoke_with_environment(
        &build.client,
        &[
            "lllm",
            "models",
            "--json",
            "--provider",
            "a",
            "--capability",
            "structured-output",
            "--configured",
            "--default",
        ],
        Vec::new(),
        true,
        &environment,
        CancellationToken::new(),
    )
    .await;

    assert_eq!(status, ExitStatus::Success);
    assert!(stderr.is_empty());
    let listing: serde_json::Value =
        serde_json::from_slice(&stdout).expect("output should be JSON");
    assert_eq!(listing["models"].as_array().map(Vec::len), Some(1));
    assert_eq!(listing["models"][0]["selector"], "alpha/one");

    let (_, stdout, _) = invoke(
        &build.client,
        &[
            "lllm",
            "models",
            "--json",
            "--provider",
            "beta",
            "--capability",
            "tools",
        ],
        Vec::new(),
        true,
        CancellationToken::new(),
    )
    .await;
    let listing: serde_json::Value =
        serde_json::from_slice(&stdout).expect("output should be JSON");
    assert_eq!(listing["models"].as_array().map(Vec::len), Some(0));

    let (_, stdout, _) = invoke(
        &build.client,
        &["lllm", "models", "--json", "--capability", "tools"],
        Vec::new(),
        true,
        CancellationToken::new(),
    )
    .await;
    let listing: serde_json::Value =
        serde_json::from_slice(&stdout).expect("output should be JSON");
    assert_eq!(listing["models"].as_array().map(Vec::len), Some(1));
    assert_eq!(listing["models"][0]["selector"], "alpha/one");
}

#[tokio::test]
async fn model_search_is_case_insensitive_and_reports_no_match() {
    let adapter = RecordingAdapter::default();
    let build = client(adapter.clone());
    let (status, _, _) = invoke(
        &build.client,
        &["lllm", "hello", "--model-query", "UNO", "--no-stream"],
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
        &["lllm", "hello", "-q", "missing"],
        Vec::new(),
        true,
        CancellationToken::new(),
    )
    .await;
    assert_eq!(status, ExitStatus::Usage);
    assert!(stdout.is_empty());
    assert!(String::from_utf8_lossy(&stderr).contains("missing"));
}

#[tokio::test]
async fn model_selection_uses_the_documented_precedence() {
    let adapter = RecordingAdapter::default();
    let build = client(adapter.clone());
    let environment = test_environment(Some("alpha/uno"));

    let (status, stdout, stderr) = invoke_with_environment(
        &build.client,
        &["lllm", "resolve"],
        Vec::new(),
        true,
        &environment,
        CancellationToken::new(),
    )
    .await;
    assert_eq!(status, ExitStatus::Success);
    assert_eq!(stdout, b"alpha/one\n");
    assert!(stderr.is_empty());

    let (_, stdout, _) = invoke_with_environment(
        &build.client,
        &["lllm", "resolve", "--model-query", "one"],
        Vec::new(),
        true,
        &environment,
        CancellationToken::new(),
    )
    .await;
    assert_eq!(stdout, b"alpha/one\n");

    let (_, stdout, _) = invoke_with_environment(
        &build.client,
        &[
            "lllm",
            "resolve",
            "--model",
            "alpha/one",
            "--model-query",
            "missing",
        ],
        Vec::new(),
        true,
        &environment,
        CancellationToken::new(),
    )
    .await;
    assert_eq!(stdout, b"alpha/one\n");
    assert!(
        adapter
            .requests
            .lock()
            .expect("request lock should work")
            .is_empty()
    );
}

#[tokio::test]
async fn probe_uses_model_selection_and_emits_a_json_report() {
    let adapter = RecordingAdapter::default();
    let build = client(adapter.clone());
    let environment = test_environment(Some("beta/two"));
    let (status, stdout, stderr) = invoke_with_environment(
        &build.client,
        &[
            "lllm",
            "probe",
            "--model-query",
            "one",
            "--reasoning-effort",
            "high",
            "--timeout",
            "2s",
            "--json",
        ],
        Vec::new(),
        true,
        &environment,
        CancellationToken::new(),
    )
    .await;

    assert_eq!(status, ExitStatus::Success);
    assert!(stderr.is_empty());
    let report: serde_json::Value =
        serde_json::from_slice(&stdout).expect("probe output should be JSON");
    assert_eq!(report["version"], 1);
    assert_eq!(report["route"], "alpha/one");
    assert_eq!(report["outcome"]["status"], "passed");
    assert_eq!(report["usage"]["input"], 1_000);
    let requests = adapter.requests.lock().expect("request lock should work");
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].model(), "alpha/one");
    assert_eq!(requests[0].max_output_tokens(), Some(1_024));
    assert_eq!(requests[0].reasoning_effort(), Some(ReasoningEffort::High));
}

#[tokio::test]
async fn a_failed_probe_prints_the_finding_and_returns_status_one() {
    let build = client(FailingAdapter);
    let (status, stdout, stderr) = invoke(
        &build.client,
        &["lllm", "probe", "--model", "alpha/one"],
        Vec::new(),
        true,
        CancellationToken::new(),
    )
    .await;

    assert_eq!(status, ExitStatus::Failure);
    assert!(stderr.is_empty());
    let output = String::from_utf8(stdout).expect("probe output should be UTF-8");
    assert!(output.starts_with(
        "failed alpha/one · rate_limit · request was limited · status=429 · code=rate_limit"
    ));
    assert!(output.contains(" · 0 input · 0 output · "));
    assert!(output.ends_with("ms\n") || output.ends_with("s\n"));
}

#[tokio::test]
async fn an_incorrect_tool_probe_prints_the_finding_and_returns_status_one() {
    let build = client(RecordingAdapter::default());
    let (status, stdout, stderr) = invoke(
        &build.client,
        &["lllm", "probe", "--model", "alpha/one", "--tools"],
        Vec::new(),
        true,
        CancellationToken::new(),
    )
    .await;

    assert_eq!(status, ExitStatus::Failure);
    assert!(stderr.is_empty());
    let output = String::from_utf8(stdout).expect("probe output should be UTF-8");
    assert!(
        output.starts_with("incorrect alpha/one · the model answered without calling the tool")
    );
}

#[tokio::test]
async fn environment_model_is_used_for_prompts_and_marked_in_models() {
    let adapter = RecordingAdapter::default();
    let build = client(adapter.clone());
    let environment = test_environment(Some("uno"));
    let (status, _, _) = invoke_with_environment(
        &build.client,
        &["lllm", "hello", "--no-stream"],
        Vec::new(),
        true,
        &environment,
        CancellationToken::new(),
    )
    .await;
    assert_eq!(status, ExitStatus::Success);
    assert_eq!(
        adapter
            .requests
            .lock()
            .expect("request lock should work")
            .last()
            .map(Request::model),
        Some("uno")
    );

    let (_, stdout, _) = invoke_with_environment(
        &build.client,
        &["lllm", "models", "--json"],
        Vec::new(),
        true,
        &environment,
        CancellationToken::new(),
    )
    .await;
    let listing: serde_json::Value =
        serde_json::from_slice(&stdout).expect("output should be JSON");
    assert_eq!(listing["effective_default"], "alpha/one");
    assert_eq!(listing["models"][0]["effective_default"], true);
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
        ["lllm", "hello", "--model", "alpha/one"],
        &build.client,
        ProcessIo {
            stdin:  Cursor::new(Vec::new()),
            stdout: BrokenWriter,
            stderr: &mut stderr,
        },
        TerminalState { stdin: true },
        &test_environment(None),
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
        &["lllm", "--model", "alpha/one"],
        vec![0xff],
        false,
        CancellationToken::new(),
    )
    .await;

    assert_eq!(status, ExitStatus::Usage);
    assert!(stdout.is_empty());
    assert!(String::from_utf8_lossy(&stderr).contains("UTF-8"));
}
