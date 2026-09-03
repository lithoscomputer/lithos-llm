#![cfg(feature = "cli")]

use std::collections::BTreeMap;
use std::fs;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Router;
use axum::body::{Body, Bytes, to_bytes};
use axum::extract::{Request, State};
use axum::http::{HeaderName, HeaderValue, Method, StatusCode};
use axum::response::Response;
use axum::routing::any;
use serde::Deserialize;
use serde_json::Value;
use tokio::net::TcpListener;
use twin_openai::config::Config;

const FIXTURE_ADDRESS: &str = "127.0.0.1:3931";
const PROTOCOL_FIXTURE_ADDRESS: &str = "127.0.0.1:3932";

#[derive(Clone, Debug, Deserialize)]
struct ProtocolScenario {
    id:            String,
    method:        String,
    path_contains: String,
    body_contains: Option<String>,
    status:        u16,
    content_type:  String,
    #[serde(default)]
    headers:       BTreeMap<String, String>,
    body:          Value,
    #[serde(default)]
    delay_ms:      u64,
}

#[derive(Debug, Deserialize)]
struct ProtocolScenarios {
    scenarios: Vec<ProtocolScenario>,
}

#[derive(Clone)]
struct ProtocolState {
    scenarios: Arc<Mutex<Vec<ProtocolScenario>>>,
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cli_contract() {
    let address: SocketAddr = FIXTURE_ADDRESS
        .parse()
        .expect("fixture address should parse");
    let mut config = Config::from_lookup(&|_| None).expect("fixture configuration should build");
    config.bind_addr = address;
    config.scenarios_path =
        Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/e2e/cli/scenarios.json"));
    let app = twin_openai::build_app_with_config(config).expect("fixture app should build");
    let listener = TcpListener::bind(address)
        .await
        .expect("fixture address should be available");
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("fixture server should run");
    });
    let protocol_server = protocol_server().await;

    let cases = trycmd::TestCases::new();
    if let Ok(profile_file) = std::env::var("LLVM_PROFILE_FILE") {
        cases.env("LLVM_PROFILE_FILE", profile_file);
    }
    cases
        .env("RUST_LOG", "")
        .case("tests/e2e/cli/*.trycmd")
        .case("tests/e2e/cli/*.toml")
        .run();

    server.abort();
    protocol_server.abort();
    let error = server
        .await
        .expect_err("fixture server should stop by cancellation");
    assert!(error.is_cancelled(), "fixture server should be cancelled");
    let error = protocol_server
        .await
        .expect_err("protocol fixture server should stop by cancellation");
    assert!(
        error.is_cancelled(),
        "protocol fixture server should be cancelled"
    );
}

async fn protocol_server() -> tokio::task::JoinHandle<()> {
    let source = fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/e2e/cli/http_scenarios.json"),
    )
    .expect("protocol scenarios should be readable");
    let scenarios: ProtocolScenarios =
        serde_json::from_str(&source).expect("protocol scenarios should be valid JSON");
    let state = ProtocolState {
        scenarios: Arc::new(Mutex::new(scenarios.scenarios)),
    };
    let app = Router::new()
        .route("/{*path}", any(protocol_response))
        .with_state(state);
    let address: SocketAddr = PROTOCOL_FIXTURE_ADDRESS
        .parse()
        .expect("protocol fixture address should parse");
    let listener = TcpListener::bind(address)
        .await
        .expect("protocol fixture address should be available");
    tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("protocol fixture server should run");
    })
}

async fn protocol_response(State(state): State<ProtocolState>, request: Request) -> Response<Body> {
    let method = request.method().clone();
    let path = request
        .uri()
        .path_and_query()
        .map_or_else(|| request.uri().path().to_owned(), ToString::to_string);
    let body = to_bytes(request.into_body(), 16 * 1024 * 1024)
        .await
        .expect("fixture request body should be readable");
    let scenario = take_scenario(&state, &method, &path, &body);
    match scenario {
        Some(scenario) => scenario_response(scenario).await,
        None => Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .header("content-type", "application/json")
            .body(Body::from(format!(
                "{{\"error\":{{\"message\":\"no protocol scenario matched {} {}\"}}}}",
                method, path
            )))
            .expect("unmatched fixture response should build"),
    }
}

fn take_scenario(
    state: &ProtocolState,
    method: &Method,
    path: &str,
    body: &Bytes,
) -> Option<ProtocolScenario> {
    let text = String::from_utf8_lossy(body);
    let mut scenarios = state
        .scenarios
        .lock()
        .expect("protocol scenario lock should not be poisoned");
    let position = scenarios.iter().position(|scenario| {
        scenario.method == method.as_str()
            && path.contains(&scenario.path_contains)
            && scenario
                .body_contains
                .as_deref()
                .is_none_or(|needle| text.contains(needle))
    })?;
    Some(scenarios.remove(position))
}

async fn scenario_response(scenario: ProtocolScenario) -> Response<Body> {
    if scenario.delay_ms > 0 {
        tokio::time::sleep(Duration::from_millis(scenario.delay_ms)).await;
    }
    let status = StatusCode::from_u16(scenario.status).expect("fixture status should be valid");
    let mut response = Response::builder()
        .status(status)
        .header("content-type", scenario.content_type)
        .header("x-fixture-scenario", scenario.id);
    for (name, value) in scenario.headers {
        response = response.header(
            HeaderName::try_from(name).expect("fixture header name should be valid"),
            HeaderValue::try_from(value).expect("fixture header value should be valid"),
        );
    }
    let body = match scenario.body {
        Value::String(text) => text,
        value => serde_json::to_string(&value).expect("fixture JSON body should encode"),
    };
    response
        .body(Body::from(body))
        .expect("fixture response should build")
}
