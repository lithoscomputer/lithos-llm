#![cfg(feature = "runtime")]
//! Secrets never leak into what the library returns, formats, or logs.
//!
//! Each case builds a real client against a mock provider, with random
//! secrets in the credentials, and drives one call to success or to one of
//! several failures. Everything a caller or an operator can see — the result,
//! every error in its source chain, `Debug` output, retry reports, and every
//! trace event — is collected, and no secret may appear in it. A positive
//! control checks that a well-formed secret really reached the provider, so
//! the property cannot pass by never sending one.

use std::error::Error as StdError;
use std::fmt::Write as _;
use std::io;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use futures_util::StreamExt as _;
use httpmock::{Method, MockServer, Then};
use lithos_llm::catalog::Catalog;
use lithos_llm::credentials::{
    CredentialHeader, Credentials, HttpAuthentication, HttpCredentials, SecretValue,
    StaticCredentials,
};
use lithos_llm::middleware::{
    Call, Observer, RetryEvent, RetryMiddleware, RetryPolicy, TracingMiddleware,
};
use lithos_llm::types::Error;
use lithos_llm::{Client, Request};
use proptest::bool::weighted;
use proptest::option;
use proptest::prelude::*;
use serde_json::json;
use tokio::runtime::Builder;
use tracing::subscriber::set_default;
use tracing_subscriber::{EnvFilter, fmt};

const PATH: &str = "/v1/chat/completions";

/// The provider's declared authentication scheme.
#[derive(Clone, Copy, Debug)]
enum Scheme {
    Bearer,
    Header,
    Headers,
}

impl Scheme {
    fn toml(self) -> &'static str {
        match self {
            Self::Bearer => r#"{ type = "bearer" }"#,
            Self::Header => r#"{ type = "header", name = "x-api-key" }"#,
            Self::Headers => r#"{ type = "headers" }"#,
        }
    }
}

/// The shape of the primary credential the application supplies.
#[derive(Clone, Copy, Debug)]
enum Shape {
    Bearer,
    Header,
}

/// What the mock provider answers.
#[derive(Clone, Copy, Debug)]
enum Outcome {
    Completed,
    Streamed,
    Unauthorized,
    RateLimited,
    ServerError,
    StreamFailure,
}

impl Outcome {
    fn respond(self, then: Then) {
        match self {
            Self::Completed => {
                then.status(200).json_body(json!({
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "model": "api-model",
                    "choices": [{
                        "index": 0,
                        "message": { "role": "assistant", "content": "ok" },
                        "finish_reason": "stop",
                    }],
                    "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 },
                }));
            }
            Self::Streamed => {
                then.status(200)
                    .header("content-type", "text/event-stream")
                    .body(
                        "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":\
                         {\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
                    );
            }
            Self::Unauthorized => {
                then.status(401).json_body(json!({
                    "error": {
                        "message": "Incorrect API key provided.",
                        "type": "invalid_request_error",
                        "code": "invalid_api_key",
                    },
                }));
            }
            Self::RateLimited => {
                then.status(429)
                    .header("retry-after", "0")
                    .json_body(json!({ "error": { "message": "Rate limit reached." } }));
            }
            Self::ServerError => {
                then.status(500).body("upstream failure");
            }
            Self::StreamFailure => {
                then.status(200)
                    .header("content-type", "text/event-stream")
                    .body("data: {\"error\":{\"message\":\"overloaded\",\"code\":529}}\n\n");
            }
        }
    }
}

/// One generated case.
#[derive(Clone, Debug)]
struct Case {
    scheme:    Scheme,
    shape:     Shape,
    secret:    String,
    /// An extra credential header, such as a project id, with its own secret.
    extra:     Option<String>,
    /// Whether the primary secret carries a trailing line break, which no
    /// HTTP header value may hold.
    malformed: bool,
    outcome:   Outcome,
    streaming: bool,
}

impl Case {
    /// Whether the credentials fit the provider's scheme.
    fn matches_scheme(&self) -> bool {
        !matches!(
            (self.scheme, self.shape),
            (Scheme::Bearer, Shape::Header) | (Scheme::Header, Shape::Bearer)
        )
    }

    /// Whether the call reaches the provider at all.
    fn sends(&self) -> bool {
        self.matches_scheme() && !self.malformed
    }

    fn sent_secret(&self) -> String {
        if self.malformed {
            format!("{}\n", self.secret)
        } else {
            self.secret.clone()
        }
    }

    /// The primary header as the provider should receive it.
    fn primary_header(&self) -> (&'static str, String) {
        match self.shape {
            Shape::Bearer => ("authorization", format!("Bearer {}", self.secret)),
            Shape::Header => ("x-api-key", self.secret.clone()),
        }
    }

    fn credentials(&self) -> Credentials {
        let secret = SecretValue::new(self.sent_secret());
        let auth = match self.shape {
            Shape::Bearer => HttpAuthentication::Bearer(secret),
            Shape::Header => HttpAuthentication::Header(CredentialHeader::new("x-api-key", secret)),
        };
        let mut credentials = HttpCredentials::new(auth);
        if let Some(extra) = &self.extra {
            credentials = credentials.with_header(CredentialHeader::new(
                "x-project",
                SecretValue::new(extra.clone()),
            ));
        }
        Credentials::Http(credentials)
    }

    /// Every secret this case must never reveal.
    fn secrets(&self) -> Vec<&str> {
        let mut secrets = vec![self.secret.as_str()];
        secrets.extend(self.extra.as_deref());
        secrets
    }
}

fn case() -> impl Strategy<Value = Case> {
    let secret = "[A-Za-z0-9]{24,40}";
    (
        prop_oneof![
            Just(Scheme::Bearer),
            Just(Scheme::Header),
            Just(Scheme::Headers)
        ],
        prop_oneof![Just(Shape::Bearer), Just(Shape::Header)],
        secret,
        option::of(secret),
        weighted(0.15),
        prop_oneof![
            Just(Outcome::Completed),
            Just(Outcome::Streamed),
            Just(Outcome::Unauthorized),
            Just(Outcome::RateLimited),
            Just(Outcome::ServerError),
            Just(Outcome::StreamFailure),
        ],
        any::<bool>(),
    )
        .prop_map(
            |(scheme, shape, secret, extra, malformed, outcome, streaming)| Case {
                scheme,
                shape,
                secret,
                extra,
                malformed,
                outcome,
                streaming,
            },
        )
}

/// A log sink the test reads back after the call.
#[derive(Clone, Default)]
struct Sink(Arc<Mutex<Vec<u8>>>);

impl Sink {
    fn text(&self) -> String {
        let bytes = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

impl io::Write for Sink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Records every retry report as an application observer would.
#[derive(Clone, Default)]
struct RetryReports(Arc<Mutex<String>>);

impl Observer for RetryReports {
    fn on_retry(&self, call: &Call, retry: RetryEvent<'_>) {
        let mut reports = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        let _ = writeln!(reports, "{call:?} {retry:?}");
        describe_error(&mut reports, retry.error);
    }
}

/// Writes an error, and every error in its source chain, as a caller or a
/// logger would print them.
fn describe_error(out: &mut String, error: &(dyn StdError + 'static)) {
    let mut current = Some(error);
    while let Some(error) = current {
        let _ = writeln!(out, "{error} | {error:?}");
        current = error.source();
    }
}

fn catalog(scheme: Scheme, base_url: &str) -> Result<Catalog, Box<dyn StdError>> {
    let toml = format!(
        "schema_version = 1\n\n\
         [providers.wire]\n\
         display_name = \"Wire\"\n\
         codecs = [\"openai-chat\"]\n\
         base_url = \"{base_url}\"\n\
         default_model = \"model\"\n\
         auth = {auth}\n\n\
         [providers.wire.models.model]\n\
         display_name = \"Model\"\n\
         api_model = \"api-model\"\n\
         capabilities = {{ text = true }}\n",
        auth = scheme.toml(),
    );
    Ok(Catalog::builder().toml_layer("secrets", &toml)?.build()?)
}

/// Runs one case and returns everything visible, and how often the provider
/// was called with the expected credentials.
async fn run(case: &Case) -> Result<(String, usize), Box<dyn StdError>> {
    let server = MockServer::start_async().await;
    let mock = server
        .mock_async(|when, then| {
            let mut when = when.method(Method::POST).path(PATH);
            if case.sends() {
                let (name, value) = case.primary_header();
                when = when.header(name, value);
                if let Some(extra) = &case.extra {
                    when = when.header("x-project", extra);
                }
            }
            let _ = when;
            case.outcome.respond(then);
        })
        .await;

    let reports = RetryReports::default();
    let credentials = StaticCredentials::new().with("wire", case.credentials());
    let mut seen = format!("{credentials:?}\n");
    let client = Client::builder()
        .catalog(catalog(case.scheme, &server.url("/v1"))?)
        .credentials(credentials)
        .middleware(TracingMiddleware)
        .middleware(
            RetryMiddleware::new(
                RetryPolicy::exponential()
                    .max_attempts(2)
                    .initial_delay(Duration::ZERO),
            )
            .observer(reports.clone()),
        )
        .build()?
        .client;
    let _ = writeln!(seen, "{client:?}");

    let request = Request::builder()
        .model("wire/model")
        .user("hello")
        .build()?;
    if case.streaming {
        match client.stream(request).await {
            Ok(mut stream) => {
                while let Some(item) = stream.next().await {
                    match item {
                        Ok(event) => {
                            let _ = writeln!(seen, "{event:?}");
                        }
                        Err(error) => describe(&mut seen, &error),
                    }
                }
            }
            Err(error) => describe(&mut seen, &error),
        }
    } else {
        match client.complete(request).await {
            Ok(response) => {
                let _ = writeln!(seen, "{response:?}");
            }
            Err(error) => describe(&mut seen, &error),
        }
    }
    seen.push_str(&reports.0.lock().unwrap_or_else(PoisonError::into_inner));
    Ok((seen, mock.calls_async().await))
}

fn describe(out: &mut String, error: &Error) {
    describe_error(out, error);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn no_secret_reaches_results_errors_or_logs(case in case()) {
        let sink = Sink::default();
        let writer = sink.clone();
        let subscriber = fmt()
            .with_env_filter(EnvFilter::new("trace,httpmock=off"))
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .finish();
        let _guard = set_default(subscriber);
        let runtime = Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| TestCaseError::fail(error.to_string()))?;

        let (mut seen, calls) = runtime
            .block_on(run(&case))
            .map_err(|error| TestCaseError::fail(error.to_string()))?;
        seen.push_str(&sink.text());

        for secret in case.secrets() {
            prop_assert!(
                !seen.contains(secret),
                "a secret leaked into what the caller or the logs see:\n{seen}"
            );
        }
        if case.sends() {
            prop_assert!(calls >= 1, "the provider never received the secret");
        } else {
            prop_assert_eq!(calls, 0, "a call that cannot authenticate was sent");
        }
    }
}
