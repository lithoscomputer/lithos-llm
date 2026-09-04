//! V3 — negative and cross-cutting behavior.
//!
//! These tests verify that Anthropic's live error envelopes still land in the
//! right [`ErrorKind`].

use std::env;
use std::error::Error as StdError;
use std::time::Duration;

use futures_util::StreamExt as _;
use lithos_llm::Client;
use lithos_llm::catalog::Catalog;
use lithos_llm::credentials::{CredentialHeader, Credentials, SecretValue, StaticCredentials};
use lithos_llm::types::{ErrorKind, Message, Role, ToolChoice, ToolDefinition};
use serde_json::json;

use crate::anthropic;
use crate::support::{self, TestResult};

/// The workhorse for negative tests: the cheapest roster model.
const MODEL: &str = "claude-haiku-4.5";

#[tokio::test]
#[ignore = "live Anthropic call; run with `mise run test:e2e`"]
async fn a_bad_key_classifies_as_authentication() -> TestResult {
    if let Some(skip) = support::live_only("live 401 classification") {
        return skip;
    }
    let client = anthropic::client_with_key("lithos-e2e-invalid-key");
    let request = anthropic::request(MODEL).user("Hello").build()?;
    let error = client
        .complete(request)
        .await
        .expect_err("an invalid key must not complete");
    assert_eq!(
        error.kind(),
        ErrorKind::Authentication,
        "Anthropic's live 401 shape classified as {:?}: {}",
        error.kind(),
        error.message()
    );
    Ok(())
}

#[tokio::test]
#[ignore = "live Anthropic call; run with `mise run test:e2e`"]
async fn an_unknown_passthrough_model_classifies_cleanly() -> TestResult {
    if let Some(skip) = support::live_only("live error classification") {
        return skip;
    }
    let Some(client) = anthropic::live_client() else {
        return support::skip("ANTHROPIC_API_KEY is unset");
    };
    // The provider allows passthrough, so this reaches the wire and Anthropic
    // itself refuses the unknown id.
    let request = lithos_llm::Request::builder()
        .model(anthropic::selector("lithos-e2e-does-not-exist"))
        .user("Hello")
        .max_output_tokens(64)
        .build()?;
    let error = client
        .complete(request)
        .await
        .expect_err("an unknown wire model must not complete");
    assert!(
        matches!(
            error.kind(),
            ErrorKind::NotFound | ErrorKind::InvalidRequest | ErrorKind::Provider
        ),
        "Anthropic's unknown-model shape classified as {:?}: {}",
        error.kind(),
        error.message()
    );
    Ok(())
}

#[tokio::test]
#[ignore = "live Anthropic call; run with `mise run test:e2e`"]
async fn a_tiny_request_timeout_classifies_as_timeout() -> TestResult {
    if let Some(skip) = support::live_only("timing behavior") {
        return skip;
    }
    let Some(client) = anthropic::live_client() else {
        return support::skip("ANTHROPIC_API_KEY is unset");
    };
    let request = anthropic::request(MODEL)
        .user("Hello")
        .timeout(Duration::from_millis(1))
        .build()?;
    let error = client
        .complete(request)
        .await
        .expect_err("a one-millisecond budget must not complete");
    assert_eq!(error.kind(), ErrorKind::Timeout);
    Ok(())
}

/// Dropping a live stream mid-flight must simply end it: no hang, no panic,
/// and the connection is released. The test's own timeout bounds the "no
/// hang" half.
#[tokio::test]
#[ignore = "live Anthropic call; run with `mise run test:e2e`"]
async fn dropping_a_stream_mid_flight_is_clean() -> TestResult {
    if let Some(skip) = support::live_only("live connection behavior") {
        return skip;
    }
    let Some(client) = anthropic::live_client() else {
        return support::skip("ANTHROPIC_API_KEY is unset");
    };
    let request = anthropic::request(MODEL)
        .user("Write three short sentences about rivers.")
        .build()?;
    let mut stream = client.stream(request).await?;
    let first = stream.next().await;
    assert!(first.is_some(), "the stream ended before its first event");
    drop(stream);
    Ok(())
}

/// A live client over a one-row catalog whose claims differ from the roster's.
///
/// The roster rows keep the client from sending what a model rejects. These
/// cells send it anyway, to pin the upstream rejection that justifies the
/// roster's claim.
fn client_for_row(row: &str, key: &str) -> Result<Client, Box<dyn StdError>> {
    let catalog = Catalog::builder()
        .toml_layer("anthropic-e2e-unrestricted", row)?
        .build()?;
    Ok(Client::builder()
        .catalog(catalog)
        .credentials(StaticCredentials::new().with(
            anthropic::PROVIDER,
            Credentials::header(CredentialHeader::new(
                "x-api-key",
                SecretValue::new(key.to_owned()),
            )),
        ))
        .build()?
        .client)
}

/// A Haiku 4.5 row that claims system turns, which the model rejects.
const SYSTEM_TURN_ROW: &str = r#"
schema_version = 1

[providers.anthropic]
display_name = "Anthropic"
adapter = "anthropic"
codec = "anthropic-messages"
base_url = "https://api.anthropic.com"
default_model = "claude-haiku-4.5"

[providers.anthropic.auth]
type = "header"
name = "x-api-key"

[providers.anthropic.models."claude-haiku-4.5"]
display_name = "Claude Haiku 4.5, system turns claimed"
api_model = "claude-haiku-4-5-20251001"
capabilities = { text = true }
protocol_options = { system_turns = true }
"#;

/// The older rows claim no system turns, so the client hoists their
/// mid-conversation system messages. This cell claims the capability on Haiku
/// anyway and pins the 400 that justifies the omission. When Anthropic starts
/// taking `system` turns on Haiku, this turns red and the row gains the claim.
#[tokio::test]
#[ignore = "live Anthropic call; run with `mise run test:e2e`"]
async fn a_system_turn_is_refused_upstream_where_unclaimed() -> TestResult {
    if let Some(skip) = support::live_only("live 400 classification") {
        return skip;
    }
    let Ok(key) = env::var(anthropic::KEY_VARIABLE) else {
        return support::skip("ANTHROPIC_API_KEY is unset");
    };
    let client = client_for_row(SYSTEM_TURN_ROW, &key)?;
    let request = anthropic::request(MODEL)
        .system("Answer with just the city name.")
        .user("What is the capital of France?")
        .message(Message::text(Role::Assistant, "Paris."))
        .user("And of Spain?")
        .message(Message::text(Role::System, "Write in uppercase."))
        .build()?;
    let error = client
        .complete(request)
        .await
        .expect_err("Haiku 4.5 must reject a system turn");
    assert_eq!(
        error.kind(),
        ErrorKind::InvalidRequest,
        "the system-turn 400 classified as {:?}: {}",
        error.kind(),
        error.message()
    );
    assert!(
        error.message().contains("not supported on this model"),
        "the rejection no longer names the system role: {}",
        error.message()
    );
    Ok(())
}

/// A Fable 5.1 row that leaves `forced_tool_choice` at its default, so a
/// forced choice reaches the wire.
const UNRESTRICTED_ROW: &str = r#"
schema_version = 1

[providers.anthropic]
display_name = "Anthropic"
adapter = "anthropic"
codec = "anthropic-messages"
base_url = "https://api.anthropic.com"
default_model = "claude-fable-5.1"

[providers.anthropic.auth]
type = "header"
name = "x-api-key"

[providers.anthropic.models."claude-fable-5.1"]
display_name = "Claude Fable 5.1, forced choice unrestricted"
api_model = "claude-fable-5-1"
capabilities = { text = true, tools = true, reasoning = true, tool_choice = { required = true, named = true } }
protocol_options = { reasoning_effort_levels = true }
"#;

/// The catalog row for Fable 5.1 denies forced tool choice, so the client
/// never sends one. This cell sends one anyway, through a one-row catalog that
/// leaves the flag at its default, and pins the upstream 400 that justifies
/// the denial. If Anthropic starts accepting forced choice on Fable 5.1, this
/// turns red and the restriction comes off the row.
#[tokio::test]
#[ignore = "live Anthropic call; run with `mise run test:e2e`"]
async fn a_forced_tool_choice_is_refused_upstream_on_fable_5_1() -> TestResult {
    if let Some(skip) = support::live_only("live 400 classification") {
        return skip;
    }
    let Ok(key) = env::var(anthropic::KEY_VARIABLE) else {
        return support::skip("ANTHROPIC_API_KEY is unset");
    };
    let client = client_for_row(UNRESTRICTED_ROW, &key)?;
    let request = anthropic::request("claude-fable-5.1")
        .user("What is the weather in Paris?")
        .tool(ToolDefinition::function(
            "get_weather",
            "Reads the current weather for a city",
            json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"],
            }),
        ))
        .tool_choice(ToolChoice::Tool {
            name: "get_weather".to_owned(),
        })
        .build()?;
    let error = client
        .complete(request)
        .await
        .expect_err("Fable 5.1 must reject a forced tool choice");
    assert_eq!(
        error.kind(),
        ErrorKind::InvalidRequest,
        "the forced-choice 400 classified as {:?}: {}",
        error.kind(),
        error.message()
    );
    assert!(
        error.message().contains("not supported for this model"),
        "the rejection no longer names the tool choice: {}",
        error.message()
    );
    Ok(())
}

/// Records whether Anthropic populates rate-limit headers. Nothing pins this
/// yet; the probe output is what decides whether `rate_limits` gets a hard
/// assertion here.
#[tokio::test]
#[ignore = "live Anthropic call; run with `mise run test:e2e`"]
async fn probe_rate_limit_headers() -> TestResult {
    if let Some(skip) = support::live_only("live rate-limit headers") {
        return skip;
    }
    let Some(client) = anthropic::live_client() else {
        return support::skip("ANTHROPIC_API_KEY is unset");
    };
    let request = anthropic::request(MODEL)
        .user("In one short sentence, say hello.")
        .build()?;
    let response = client.complete(request).await?;
    match &response.rate_limits {
        Some(limits) => support::observe(&format!("anthropic rate limits: {limits:?}")),
        None => support::observe("anthropic sent no rate-limit headers"),
    }
    Ok(())
}
