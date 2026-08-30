//! The Codex deployment — OpenAI's ChatGPT-subscription access path.
//!
//! The same Responses protocol as the platform API with a different
//! envelope, verified live on 2026-08-30: the unversioned `/responses` path
//! under `chatgpt.com/backend-api/codex`, forced streaming, hoisted
//! instructions, rejected sampling fields, a roster without the pro rows,
//! `detail`-spelled errors, and a terminal `response.completed` document
//! whose `output` array is empty — the streamed items are the only content.
//!
//! Credentials are application state, not environment convention: the cells
//! read `OPENAI_CODEX_TOKEN` (a ChatGPT OAuth access token, e.g. from the
//! Codex CLI's `~/.codex/auth.json`) and `CHATGPT_ACCOUNT_ID`, and skip when
//! either is unset. Every network cell is live-only — the twin proxies
//! neither this host nor its path shape — and spends the seat's quota, so
//! the suite is one small battery on the cheap rows rather than a roster
//! sweep.

use std::collections::BTreeSet;
use std::env;
use std::error::Error as StdError;

use lithos_llm::catalog::Catalog;
use lithos_llm::credentials::{Credentials, SecretValue, StaticCredentials};
use lithos_llm::types::{
    ContentPart, ErrorKind, FinishReason, Message, ReasoningEffort, RequestBuilder, Role,
    ToolChoice, ToolDefinition, ToolResult,
};
use lithos_llm::{Client, Request};
use serde_json::json;

use crate::support::{self, TestResult};

const PROVIDER: &str = "openai-codex";
const TOKEN_VARIABLE: &str = "OPENAI_CODEX_TOKEN";
const ACCOUNT_VARIABLE: &str = "CHATGPT_ACCOUNT_ID";

const CATALOG_TOML: &str = include_str!("../openai_codex_catalog.toml");

/// The Codex catalog with the per-user headers overlaid at runtime.
///
/// The account id is per-user state, so it rides a runtime overlay rather
/// than the checked-in file. The `originator` header names this suite, as
/// the deployment expects every client to identify itself.
fn catalog(account_id: &str) -> Catalog {
    Catalog::builder()
        .toml_layer("openai-codex-e2e", CATALOG_TOML)
        .expect("the Codex E2E catalog should parse")
        .toml_layer(
            "codex-headers",
            &format!(
                "[providers.openai-codex]\ndefault_headers = {{ \"ChatGPT-Account-Id\" = \
                 \"{account_id}\", originator = \"lithos-llm-e2e\" }}\n"
            ),
        )
        .expect("the Codex header overlay should parse")
        .build()
        .expect("the Codex E2E catalog should validate")
}

/// A client for the live Codex deployment, or `None` without the credential.
fn live_client() -> Option<Client> {
    let token = env::var(TOKEN_VARIABLE).ok().filter(|t| !t.is_empty())?;
    let account = env::var(ACCOUNT_VARIABLE).ok().filter(|a| !a.is_empty())?;
    Some(client_with(&token, &account))
}

fn client_with(token: &str, account_id: &str) -> Client {
    let build = Client::builder()
        .catalog(catalog(account_id))
        .credentials(
            StaticCredentials::new().with(PROVIDER, Credentials::bearer(SecretValue::new(token))),
        )
        .build()
        .expect("the Codex E2E client should build");
    assert!(
        build.issues.is_empty(),
        "the Codex E2E client reported provider build issues: {:?}",
        build.issues
    );
    build.client
}

/// A request for one Codex model.
///
/// No output cap: the deployment rejects `max_output_tokens`, so the codec
/// would drop a configured one with a warning on every cell.
fn request(model: &str) -> RequestBuilder {
    Request::builder()
        .model(format!("{PROVIDER}/{model}"))
        .timeout(support::LIVE_TIMEOUT)
}

/// Skips a cell when the Codex credential is absent, and under record and
/// replay always: the twin proxies neither this host nor its path shape.
fn guard() -> Option<TestResult> {
    if let Some(skip) = support::live_only("the Codex deployment is not proxied") {
        return Some(skip);
    }
    None
}

/// The workhorse: the cheapest row the deployment serves.
const MODEL: &str = "gpt-5.6-luna";

/// The Codex roster is the platform roster minus the pro rows, which the
/// deployment refuses for a ChatGPT account (400, 2026-08-30).
#[test]
fn the_codex_roster_is_the_platform_roster_minus_the_pro_rows() -> TestResult {
    let builtin = Catalog::builder().with_builtin().build()?;
    let ids = |catalog: &Catalog, provider: &str| -> Result<BTreeSet<String>, Box<dyn StdError>> {
        Ok(catalog
            .provider(provider)?
            .models()
            .map(|model| model.id().to_string())
            .collect())
    };
    let mut expected = ids(&builtin, "openai")?;
    assert!(expected.remove("gpt-5.5-pro"));
    assert!(expected.remove("gpt-5.4-pro"));
    assert_eq!(ids(&catalog("acct_preflight"), PROVIDER)?, expected);
    Ok(())
}

/// Codex mode reports no native token count: the deployment's
/// `/responses/input_tokens` path is an HTML 403, so the client must answer
/// `None` locally instead of sending a request that cannot succeed.
#[tokio::test]
async fn token_counting_reports_no_native_count() -> TestResult {
    let client = client_with("preflight-token", "acct_preflight");
    let request = request(MODEL).user("Hello").build()?;
    assert!(client.count_input_tokens(request).await?.is_none());
    Ok(())
}

/// A blocking `complete` runs through the forced stream the deployment
/// requires, and a seat-billed response carries usage but no cost estimate.
#[tokio::test]
#[ignore = "live Codex call; spends the ChatGPT seat's quota"]
async fn a_blocking_complete_runs_through_the_forced_stream() -> TestResult {
    if let Some(skip) = guard() {
        return skip;
    }
    let Some(client) = live_client() else {
        return support::skip("OPENAI_CODEX_TOKEN or CHATGPT_ACCOUNT_ID is unset");
    };
    let request = request(MODEL)
        .system("When the user says ping, reply with exactly the word PONG and nothing else.")
        .user("ping")
        .build()?;
    let response = client.complete(request).await?;
    assert!(
        response.text().to_uppercase().contains("PONG"),
        "the hoisted instructions were not honored: {:?}",
        response.text()
    );
    assert_eq!(response.finish_reason, FinishReason::Stop);
    assert!(response.usage.total() > 0, "the seat reported no usage");
    assert!(
        response.cost.is_none(),
        "a seat-billed response must carry no cost estimate: {:?}",
        response.cost
    );
    Ok(())
}

#[tokio::test]
#[ignore = "live Codex call; spends the ChatGPT seat's quota"]
async fn streams_within_the_contract() -> TestResult {
    if let Some(skip) = guard() {
        return skip;
    }
    let Some(client) = live_client() else {
        return support::skip("OPENAI_CODEX_TOKEN or CHATGPT_ACCOUNT_ID is unset");
    };
    let request = request(MODEL)
        .user("Reply with one short greeting that contains at least two emoji.")
        .build()?;
    let stream = client.stream(request).await?;
    let (_, response) = support::checked_stream(stream).await?;
    assert!(!response.text().trim().is_empty());
    assert!(response.usage.total() > 0);
    Ok(())
}

/// The forced tool call also pins the deployment's empty terminal document:
/// its `response.completed` carries an empty `output` array, so the
/// assembled stream blocks are the only source of the call — and the finish
/// reason must still read `ToolCall`, not the document's bare "completed".
#[tokio::test]
#[ignore = "live Codex call; spends the ChatGPT seat's quota"]
async fn calls_the_forced_tool_despite_the_empty_terminal_document() -> TestResult {
    if let Some(skip) = guard() {
        return skip;
    }
    let Some(client) = live_client() else {
        return support::skip("OPENAI_CODEX_TOKEN or CHATGPT_ACCOUNT_ID is unset");
    };
    let request = request(MODEL)
        .user("What is the weather in Paris?")
        .tool(weather_tool())
        .tool_choice(ToolChoice::Tool {
            name: "get_weather".to_owned(),
        })
        .build()?;
    let response = client.complete(request).await?;
    let calls = support::tool_calls(&response);
    let call = calls.first().ok_or("the response carries no tool call")?;
    assert_eq!(call.name, "get_weather");
    assert!(!call.id.is_empty());
    assert_eq!(response.finish_reason, FinishReason::ToolCall);
    Ok(())
}

/// The stateless replay shape on the seat path: encrypted reasoning and the
/// tool call ride back with the result, exactly as on the platform API.
#[tokio::test]
#[ignore = "live Codex call; spends the ChatGPT seat's quota"]
async fn replays_reasoning_with_a_tool_result() -> TestResult {
    if let Some(skip) = guard() {
        return skip;
    }
    let Some(client) = live_client() else {
        return support::skip("OPENAI_CODEX_TOKEN or CHATGPT_ACCOUNT_ID is unset");
    };
    let opening = client
        .complete(
            request(MODEL)
                .user("Look up the weather in Paris, then tell me if it suits a picnic.")
                .tool(weather_tool())
                .tool_choice(ToolChoice::Tool {
                    name: "get_weather".to_owned(),
                })
                .reasoning_effort(ReasoningEffort::Low)
                .build()?,
        )
        .await?;
    let call_id = support::tool_calls(&opening)
        .first()
        .ok_or("the opening turn made no tool call")?
        .id
        .clone();

    let closing = client
        .complete(
            request(MODEL)
                .user("Look up the weather in Paris, then tell me if it suits a picnic.")
                .tool(weather_tool())
                .message(Message::new(Role::Assistant, opening.content.clone()))
                .message(Message::new(Role::Tool, [ContentPart::ToolResult(
                    ToolResult {
                        tool_call_id: call_id,
                        name:         Some("get_weather".to_owned()),
                        content:      vec![ContentPart::Text {
                            text: "21C, sunny, light breeze".to_owned(),
                        }],
                        is_error:     false,
                    },
                )]))
                .build()?,
        )
        .await?;
    assert!(
        !closing.text().trim().is_empty(),
        "the second turn gave no answer after the reasoning replay"
    );
    Ok(())
}

/// The deployment rejects `temperature` and `top_p` outright, and no Codex
/// row claims `sampling`, so the local capability gate refuses the request
/// before the codec's codex-mode drop is even reached.
#[tokio::test]
async fn sampling_is_rejected_locally() -> TestResult {
    let client = client_with("preflight-token", "acct_preflight");
    let request = request(MODEL).user("Hello").temperature(0.2).build()?;
    let error = client
        .complete(request)
        .await
        .expect_err("no Codex row claims sampling, so the client must refuse");
    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    Ok(())
}

/// `max_output_tokens` has no capability gate, so it reaches the codec,
/// which drops it in codex mode — the deployment rejects the field — and
/// the request completes with the one warning.
#[tokio::test]
#[ignore = "live Codex call; spends the ChatGPT seat's quota"]
async fn an_output_cap_is_dropped_with_a_warning() -> TestResult {
    if let Some(skip) = guard() {
        return skip;
    }
    let Some(client) = live_client() else {
        return support::skip("OPENAI_CODEX_TOKEN or CHATGPT_ACCOUNT_ID is unset");
    };
    let request = request(MODEL)
        .user("Reply with exactly PONG")
        .max_output_tokens(256)
        .build()?;
    let response = client.complete(request).await?;
    assert!(!response.text().trim().is_empty());
    assert_eq!(
        response.warnings.len(),
        1,
        "the dropped output cap must be the one warning: {:?}",
        response.warnings
    );
    Ok(())
}

/// A pro model reaches the wire through passthrough and the deployment
/// itself refuses it — the roster restriction is upstream policy, and its
/// `detail`-spelled error body must classify cleanly.
#[tokio::test]
#[ignore = "live Codex call; spends the ChatGPT seat's quota"]
async fn a_pro_model_is_refused_upstream() -> TestResult {
    if let Some(skip) = guard() {
        return skip;
    }
    let Some(client) = live_client() else {
        return support::skip("OPENAI_CODEX_TOKEN or CHATGPT_ACCOUNT_ID is unset");
    };
    let request = Request::builder()
        .model(format!("{PROVIDER}/gpt-5.5-pro"))
        .user("Hello")
        .timeout(support::LIVE_TIMEOUT)
        .build()?;
    let error = client
        .complete(request)
        .await
        .expect_err("a ChatGPT seat must not serve a pro model");
    assert!(
        matches!(
            error.kind(),
            ErrorKind::InvalidRequest | ErrorKind::NotFound | ErrorKind::Provider
        ),
        "the Codex roster refusal classified as {:?}: {}",
        error.kind(),
        error.message()
    );
    assert!(
        error.message().contains("not supported"),
        "the detail-spelled refusal lost its message: {}",
        error.message()
    );
    Ok(())
}

fn weather_tool() -> ToolDefinition {
    ToolDefinition::function(
        "get_weather",
        "Reads the current weather for a city",
        json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
        }),
    )
}
