use std::collections::BTreeMap;
use std::future::Future;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::{env, fs};

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::extract::Request as HttpRequest;
use axum::middleware::{self, Next};
use futures_util::StreamExt as _;
use lithos_llm::catalog::Catalog;
use lithos_llm::credentials::{CredentialHeader, Credentials, SecretValue, StaticCredentials};
use lithos_llm::middleware::{RetryMiddleware, RetryPolicy};
use lithos_llm::types::{Error, ResponseStream, StreamEvent};
use lithos_llm::{Client, Request};
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tokio::time::timeout;
use twin_anthropic::config::Config;
use uuid::Uuid;

pub(super) const MODEL: &str = "anthropic/claude";

pub(super) fn config() -> Config {
    Config::from_lookup(&|_| None).expect("explicit twin defaults")
}

pub(super) async fn bounded<T>(future: impl Future<Output = T>) -> T {
    timeout(Duration::from_secs(20), future)
        .await
        .expect("local operation deadline")
}

#[derive(Clone, Debug)]
pub(super) struct Capture {
    pub(super) path:    String,
    pub(super) headers: BTreeMap<String, String>,
    pub(super) body:    Value,
}

pub(super) struct Twin {
    pub(super) url:  String,
    pub(super) http: reqwest::Client,
    captures:        Arc<Mutex<Vec<Capture>>>,
    task:            JoinHandle<()>,
}

impl Drop for Twin {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl Twin {
    pub(super) async fn start(config: Config) -> Self {
        Self::router(twin_anthropic::build_app_with_config(config).expect("twin app")).await
    }

    pub(super) async fn router(router: Router) -> Self {
        let captures = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&captures);
        let router = router.layer(middleware::from_fn(
            move |request: HttpRequest, next: Next| {
                let sink = Arc::clone(&sink);
                async move {
                    let (parts, body) = request.into_parts();
                    let bytes = to_bytes(body, 32 * 1024 * 1024)
                        .await
                        .expect("request body");
                    if parts.uri.path().starts_with("/v1/") {
                        sink.lock().expect("capture lock").push(Capture {
                            path:    parts.uri.path().to_owned(),
                            headers: parts
                                .headers
                                .iter()
                                .filter(|(name, _)| {
                                    !matches!(name.as_str(), "x-api-key" | "authorization")
                                })
                                .map(|(name, value)| {
                                    (name.to_string(), value.to_str().expect("header").to_owned())
                                })
                                .collect(),
                            body:    serde_json::from_slice(&bytes).unwrap_or(Value::Null),
                        });
                    }
                    next.run(HttpRequest::from_parts(parts, Body::from(bytes)))
                        .await
                }
            },
        ));
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("loopback listener");
        let url = format!("http://{}", listener.local_addr().expect("bound address"));
        let task = tokio::spawn(async move {
            axum::serve(listener, router).await.expect("twin server");
        });
        Self {
            url,
            http: reqwest::Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(10))
                .build()
                .expect("HTTP client"),
            captures,
            task,
        }
    }

    pub(super) fn client(&self, namespace: &str) -> Client {
        self.client_with_policy(namespace, None, None)
    }

    pub(super) fn client_with_policy(
        &self,
        namespace: &str,
        retry: Option<RetryPolicy>,
        idle: Option<Duration>,
    ) -> Client {
        let catalog = Catalog::builder().toml_layer("twin", &format!(r#"
schema_version = 1
[providers.anthropic]
display_name = "Anthropic twin"
adapter = "anthropic"
codec = "anthropic-messages"
base_url = "{}"
default_model = "claude"
auth = {{ type = "header", name = "x-api-key" }}
[providers.anthropic.models.claude]
display_name = "Claude test"
api_model = "claude-test"
capabilities = {{ text = true, images = true, documents = true, tools = true, response_format = {{ json_object = true, json_schema = true }}, reasoning = true, caching = true, sampling = true, tool_choice = {{ required = true, named = true }} }}
protocol_options = {{ cache_breakpoints = true, system_turns = true }}
"#, self.url)).expect("catalog TOML").build().expect("catalog");
        let mut builder = Client::builder()
            .catalog(catalog)
            .credentials(StaticCredentials::new().with(
                "anthropic",
                Credentials::header(CredentialHeader::new(
                    "x-api-key",
                    SecretValue::new(namespace),
                )),
            ))
            .http(self.http.clone())
            .stream_idle_timeout(idle);
        if let Some(policy) = retry {
            builder = builder.middleware(RetryMiddleware::new(policy));
        }
        let built = builder.build().expect("client");
        assert!(built.issues.is_empty(), "{:?}", built.issues);
        built.client
    }

    pub(super) async fn enqueue(&self, key: &str, scenarios: Value) {
        let response = self
            .http
            .post(format!("{}/__admin/scenarios", self.url))
            .header("x-api-key", key)
            .json(&json!({"scenarios":scenarios}))
            .send()
            .await
            .expect("enqueue");
        let status = response.status();
        assert_eq!(
            status,
            200,
            "{}",
            response.text().await.expect("enqueue body")
        );
    }

    pub(super) async fn reset(&self, key: &str) {
        assert_eq!(
            self.http
                .post(format!("{}/__admin/reset", self.url))
                .header("x-api-key", key)
                .send()
                .await
                .expect("reset")
                .status(),
            200
        );
    }

    pub(super) async fn logs(&self, key: &str) -> Value {
        self.http
            .get(format!("{}/__admin/requests", self.url))
            .header("x-api-key", key)
            .send()
            .await
            .expect("logs")
            .json()
            .await
            .expect("log JSON")
    }

    pub(super) fn captures(&self) -> Vec<Capture> {
        self.captures.lock().expect("capture lock").clone()
    }

    pub(super) async fn shutdown(self) {
        self.task.abort();
        // Drop remains the panic-path guard; borrow the handle to await it.
        let mut owned = self;
        let error = (&mut owned.task).await.expect_err("server cancelled");
        assert!(error.is_cancelled());
    }
}

pub(super) fn request() -> Request {
    Request::builder()
        .model(MODEL)
        .user("hello")
        .max_output_tokens(2048)
        .timeout(Duration::from_secs(5))
        .build()
        .expect("request")
}

pub(super) fn scenario(script: Value) -> Value {
    {
        let mut value = json!({"matcher":{"endpoint":"messages"}});
        value["script"] = script;
        value
    }
}

pub(super) async fn collect(stream: ResponseStream) -> Vec<Result<StreamEvent, Error>> {
    bounded(stream.collect()).await
}

pub(super) struct TempDir(pub(super) PathBuf);
impl TempDir {
    pub(super) fn new() -> Self {
        let path = env::temp_dir().join(format!("lithos-anthropic-{}", Uuid::new_v4()));
        fs::create_dir(&path).expect("temporary directory");
        Self(path)
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
