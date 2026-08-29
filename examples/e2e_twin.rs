//! The record-and-replay proxy for the live E2E suite.
//!
//! The `test:e2e` mise tasks start this before Nextest and stop it after.
//! One process serves every test: per-test bearer tokens isolate the
//! recording namespaces, so concurrent tests never contend for a queue.
//!
//! - `LITHOS_E2E_TWIN_MODE=replay` (default): serve the committed recording in
//!   strict fixture mode. No keys, no network.
//! - `LITHOS_E2E_TWIN_MODE=record`: proxy to the live Venice API with
//!   `VENICE_API_KEY` and rewrite the recording as verbatim transcripts.
//!
//! The bind address and recording path have fixed defaults the mise tasks
//! and the test harness share; `LITHOS_E2E_TWIN_ADDR` and
//! `LITHOS_E2E_RECORDING_PATH` override them together when needed.

use std::env;
use std::error::Error as StdError;
use std::net::SocketAddr;

use tokio::net::TcpListener;
use twin_openai::config::{Config, Mode, RecordFormat};

const DEFAULT_ADDR: &str = "127.0.0.1:3921";
const DEFAULT_RECORDING: &str = "tests/e2e/recordings/venice.json";
const VENICE_UPSTREAM: &str = "https://api.venice.ai/api";

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn StdError>> {
    let addr: SocketAddr = env::var("LITHOS_E2E_TWIN_ADDR")
        .unwrap_or_else(|_| DEFAULT_ADDR.to_owned())
        .parse()?;
    let recording =
        env::var("LITHOS_E2E_RECORDING_PATH").unwrap_or_else(|_| DEFAULT_RECORDING.to_owned());
    let mode = env::var("LITHOS_E2E_TWIN_MODE").unwrap_or_else(|_| "replay".to_owned());

    let config = match mode.as_str() {
        "replay" => Config {
            scenarios_path: Some(recording.into()),
            allow_unmatched: false,
            enable_admin: false,
            ..Config::default()
        },
        "record" => Config {
            mode: Mode::ProxyRecord,
            upstream_url: VENICE_UPSTREAM.to_owned(),
            upstream_api_key: Some(env::var("VENICE_API_KEY")?),
            recording_path: Some(recording.into()),
            record_format: RecordFormat::Transcript,
            ..Config::default()
        },
        other => {
            return Err(
                format!("LITHOS_E2E_TWIN_MODE must be replay or record, got {other}").into(),
            );
        }
    };

    let app = twin_openai::build_app_with_config(config)?;
    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
