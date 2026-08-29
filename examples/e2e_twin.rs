//! The record-and-replay proxy for the live E2E suite.
//!
//! The `test:e2e` mise tasks start this before Nextest and stop it after.
//! One process serves every test: per-test bearer tokens isolate the
//! recording namespaces, so concurrent tests never contend for a queue.
//!
//! - `LITHOS_E2E_TWIN_MODE=replay` (default): serve the committed recording in
//!   strict fixture mode. No keys, no network.
//! - `LITHOS_E2E_TWIN_MODE=record`: proxy to the configured live API and
//!   rewrite the recording as verbatim transcripts.
//!
//! The bind address and recording path have fixed defaults the mise tasks
//! and the test harness share; `LITHOS_E2E_TWIN_ADDR` and
//! `LITHOS_E2E_RECORDING_PATH`, `LITHOS_E2E_UPSTREAM_URL`, and
//! `LITHOS_E2E_API_KEY_VARIABLE` override them when needed. Set
//! `LITHOS_E2E_RECORDING_APPEND=1` to add focused reruns to a recording.

use std::env;
use std::error::Error as StdError;
use std::net::SocketAddr;

use tokio::net::TcpListener;
use twin_openai::config::{Config, Mode, RecordFormat};

const DEFAULT_ADDR: &str = "127.0.0.1:3921";
const DEFAULT_RECORDING: &str = "tests/e2e/recordings/venice.json";
const VENICE_UPSTREAM: &str = "https://api.venice.ai/api";
const VENICE_KEY_VARIABLE: &str = "VENICE_API_KEY";

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
        "record" => {
            let upstream =
                env::var("LITHOS_E2E_UPSTREAM_URL").unwrap_or_else(|_| VENICE_UPSTREAM.to_owned());
            let key_variable = env::var("LITHOS_E2E_API_KEY_VARIABLE")
                .unwrap_or_else(|_| VENICE_KEY_VARIABLE.to_owned());
            Config {
                mode: Mode::ProxyRecord,
                upstream_url: upstream,
                upstream_api_key: Some(env::var(key_variable)?),
                recording_path: Some(recording.into()),
                record_format: RecordFormat::Transcript,
                recording_append: env::var("LITHOS_E2E_RECORDING_APPEND").as_deref() == Ok("1"),
                ..Config::default()
            }
        }
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
