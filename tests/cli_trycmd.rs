#![cfg(feature = "cli")]

use std::net::SocketAddr;
use std::path::PathBuf;

use tokio::net::TcpListener;
use twin_openai::config::Config;

const FIXTURE_ADDRESS: &str = "127.0.0.1:3931";

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

    trycmd::TestCases::new()
        .env("RUST_LOG", "")
        .case("tests/e2e/cli/*.trycmd")
        .case("tests/e2e/cli/*.toml");

    server.abort();
    let error = server
        .await
        .expect_err("fixture server should stop by cancellation");
    assert!(error.is_cancelled(), "fixture server should be cancelled");
}
