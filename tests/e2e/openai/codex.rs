//! The Codex deployment — OpenAI's ChatGPT-subscription access path.
//!
//! The same Responses protocol as the platform API with a different
//! envelope, verified live on 2026-08-30: the unversioned `/responses` path
//! under `chatgpt.com/backend-api/codex`, forced streaming, hoisted
//! instructions, rejected sampling fields, a roster without the pro rows,
//! `detail`-spelled errors, and a terminal `response.completed` document
//! whose `output` array is empty — the streamed items are the only content.
//!
//! The suite runs on all three backends. Replay needs nothing: the sixth
//! twin serves the committed recording with a placeholder account id, since
//! the recording matches on body hashes and the seat headers ride outside
//! the body. Recording needs `CHATGPT_ACCOUNT_ID` here and
//! `OPENAI_CODEX_TOKEN` in the twin's environment. Live needs both
//! variables here — a ChatGPT OAuth access token (e.g. from the Codex CLI's
//! `~/.codex/auth.json`) and the seat's account id — and spends the seat's
//! quota, so the roster cells stay small and cheap.

use std::collections::BTreeSet;
use std::env;
use std::error::Error as StdError;

use lithos_llm::catalog::Catalog;
use lithos_llm::credentials::{Credentials, SecretValue, StaticCredentials};
use lithos_llm::types::{
    ContentPart, ErrorKind, FinishReason, ImageContent, MediaSource, Message, ReasoningEffort,
    RequestBuilder, ResponseFormat, Role, ToolChoice, ToolDefinition, ToolResult,
};
use lithos_llm::{Client, Request};
use serde_json::{Value, json};

use crate::support::{self, TestResult};

const PROVIDER: &str = "openai-codex";
const TOKEN_VARIABLE: &str = "OPENAI_CODEX_TOKEN";
const ACCOUNT_VARIABLE: &str = "CHATGPT_ACCOUNT_ID";

const CATALOG_TOML: &str = include_str!("../openai_codex_catalog.toml");

/// Expands one live test per Codex roster model.
macro_rules! codex_model_tests {
    ($runner:path) => {
        codex_model_tests!(@expand $runner,
            gpt_5_6_sol "gpt-5.6-sol",
            gpt_5_6_terra "gpt-5.6-terra",
            gpt_5_6_luna "gpt-5.6-luna",
            gpt_5_4 "gpt-5.4",
            gpt_5_5 "gpt-5.5",
            gpt_5_4_mini "gpt-5.4-mini",
        );
    };
    (@expand $runner:path, $($name:ident $model:literal,)+) => {
        $(
            #[tokio::test]
            #[ignore = "live Codex call; spends the ChatGPT seat's quota"]
            async fn $name() -> crate::support::TestResult {
                $runner($model).await
            }
        )+
    };
}

/// The twin proxy's base URL for record and replay runs.
///
/// The `/v1` suffix matters: the codec posts to `<base>/responses` in codex
/// mode, and the twin's client-facing route is `/v1/responses` — only its
/// upstream half is rebased to the deployment's unversioned path.
fn twin_url() -> String {
    env::var("LITHOS_CODEX_E2E_TWIN_URL").unwrap_or_else(|_| "http://127.0.0.1:3926/v1".to_owned())
}

/// The Codex catalog with the per-user headers overlaid at runtime.
///
/// The account id is per-user state, so it rides a runtime overlay rather
/// than the checked-in file. The `originator` header names this suite, as
/// the deployment expects every client to identify itself. Under record and
/// replay a second overlay points the provider at the sixth twin.
fn catalog(account_id: &str) -> Catalog {
    let base = Catalog::builder()
        .toml_layer("openai-codex-e2e", CATALOG_TOML)
        .expect("the Codex E2E catalog should parse")
        .toml_layer(
            "codex-headers",
            &format!(
                "[providers.openai-codex]\ndefault_headers = {{ \"ChatGPT-Account-Id\" = \
                 \"{account_id}\", originator = \"lithos-llm-e2e\" }}\n"
            ),
        )
        .expect("the Codex header overlay should parse");
    let builder = if support::Backend::from_env() == support::Backend::Live {
        base
    } else {
        base.toml_layer(
            "e2e-twin",
            &format!("[providers.openai-codex]\nbase_url = \"{}\"\n", twin_url()),
        )
        .expect("the twin overlay should parse")
    };
    builder
        .build()
        .expect("the Codex E2E catalog should validate")
}

/// A client for this run's backend, or `None` when its inputs are missing.
///
/// Live authenticates with the OAuth token and the real account id. Record
/// goes through the twin — which holds the token — but still needs the real
/// account id, because the seat header reaches the deployment verbatim.
/// Replay needs nothing: the recording matches on body hashes, so a
/// placeholder account id serves.
fn live_client() -> Option<Client> {
    let account = || env::var(ACCOUNT_VARIABLE).ok().filter(|a| !a.is_empty());
    match support::Backend::from_env() {
        support::Backend::Live => {
            let token = env::var(TOKEN_VARIABLE).ok().filter(|t| !t.is_empty())?;
            Some(client_with(&token, &account()?))
        }
        support::Backend::Record => Some(client_with(&support::test_namespace(), &account()?)),
        support::Backend::Replay => Some(client_with(&support::test_namespace(), "acct_replay")),
    }
}

/// The one skip message every network cell shares.
fn missing_credential() -> TestResult {
    support::skip("OPENAI_CODEX_TOKEN or CHATGPT_ACCOUNT_ID is unset")
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

/// The workhorse: the cheapest row the deployment serves.
const MODEL: &str = "gpt-5.6-luna";

// ---------------------------------------------------------------------------
// Local preflight
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Roster cells
// ---------------------------------------------------------------------------

mod basic_completion {
    codex_model_tests!(super::completes_through_the_forced_stream);
}

mod stream_contract {
    codex_model_tests!(super::streams_within_the_contract);
}

mod multi_turn {
    codex_model_tests!(super::carries_the_conversation);
}

mod forced_call {
    codex_model_tests!(super::calls_the_forced_tool_despite_the_empty_terminal_document);
}

mod structured {
    codex_model_tests!(super::conforms_to_the_schema);
}

mod vision {
    codex_model_tests!(super::describes_an_inline_image);
}

/// One test per (model, effort level): the floor of every vocabulary and
/// its top — `max` on the 5.6 rows, `xhigh` elsewhere (live 400 messages,
/// 2026-08-30).
mod effort_levels {
    use super::*;

    macro_rules! level_tests {
        ($($name:ident $model:literal $level:ident,)+) => {
            $(
                #[tokio::test]
                #[ignore = "live Codex call; spends the ChatGPT seat's quota"]
                async fn $name() -> TestResult {
                    super::accepts_the_effort_level($model, ReasoningEffort::$level).await
                }
            )+
        };
    }

    level_tests!(
        gpt_5_6_sol_low "gpt-5.6-sol" Low,
        gpt_5_6_sol_max "gpt-5.6-sol" Max,
        gpt_5_6_terra_low "gpt-5.6-terra" Low,
        gpt_5_6_terra_max "gpt-5.6-terra" Max,
        gpt_5_6_luna_low "gpt-5.6-luna" Low,
        gpt_5_6_luna_max "gpt-5.6-luna" Max,
        gpt_5_4_low "gpt-5.4" Low,
        gpt_5_4_xhigh "gpt-5.4" Xhigh,
        gpt_5_5_low "gpt-5.5" Low,
        gpt_5_5_xhigh "gpt-5.5" Xhigh,
        gpt_5_4_mini_low "gpt-5.4-mini" Low,
        gpt_5_4_mini_xhigh "gpt-5.4-mini" Xhigh,
    );
}

/// The stateless replay shape on the seat path, one representative per
/// generation.
mod round_trip {
    use super::*;

    macro_rules! round_trip_tests {
        ($($name:ident $model:literal,)+) => {
            $(
                #[tokio::test]
                #[ignore = "live Codex call; spends the ChatGPT seat's quota"]
                async fn $name() -> TestResult {
                    super::replays_reasoning_with_a_tool_result($model).await
                }
            )+
        };
    }

    round_trip_tests!(
        gpt_5_6_sol "gpt-5.6-sol",
        gpt_5_6_luna "gpt-5.6-luna",
        gpt_5_4 "gpt-5.4",
    );
}

async fn completes_through_the_forced_stream(model: &str) -> TestResult {
    let Some(client) = live_client() else {
        return missing_credential();
    };
    let request = request(model)
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
    // Seat billing prices no tokens, so a response carries no cost estimate.
    assert!(
        response.cost.is_none(),
        "a seat-billed response must carry no cost estimate: {:?}",
        response.cost
    );
    Ok(())
}

async fn streams_within_the_contract(model: &str) -> TestResult {
    let Some(client) = live_client() else {
        return missing_credential();
    };
    let request = request(model)
        .user("Reply with one short greeting that contains at least two emoji.")
        .build()?;
    let stream = client.stream(request).await?;
    let (_, response) = support::checked_stream(stream).await?;
    assert!(!response.text().trim().is_empty());
    assert!(response.usage.total() > 0);
    Ok(())
}

async fn carries_the_conversation(model: &str) -> TestResult {
    let Some(client) = live_client() else {
        return missing_credential();
    };
    let request = request(model)
        .system("Answer with just the city name.")
        .user("What is the capital of France?")
        .message(Message::text(Role::Assistant, "Paris."))
        .user("And of Spain?")
        .build()?;
    let response = client.complete(request).await?;
    assert!(
        response.text().contains("Madrid"),
        "the multi-turn answer does not name Madrid: {:?}",
        response.text()
    );
    Ok(())
}

/// The forced tool call also pins the deployment's empty terminal document:
/// its `response.completed` carries an empty `output` array, so the
/// assembled stream blocks are the only source of the call — and the finish
/// reason must still read `ToolCall`, not the document's bare "completed".
async fn calls_the_forced_tool_despite_the_empty_terminal_document(model: &str) -> TestResult {
    let Some(client) = live_client() else {
        return missing_credential();
    };
    let request = request(model)
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

async fn conforms_to_the_schema(model: &str) -> TestResult {
    let Some(client) = live_client() else {
        return missing_credential();
    };
    let request = request(model)
        .user("Report the city of the Eiffel Tower and its approximate population.")
        .response_format(ResponseFormat::JsonSchema {
            name:   "city_report".to_owned(),
            schema: json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "city":       { "type": "string" },
                    "population": { "type": "integer" },
                },
                "required": ["city", "population"],
            }),
        })
        .build()?;
    let response = client.complete(request).await?;
    let payload = support::json_payload(&response)?;
    assert!(
        payload
            .get("city")
            .and_then(Value::as_str)
            .is_some_and(|city| !city.is_empty()),
        "the report carries no city: {payload}"
    );
    assert!(
        payload.get("population").is_some_and(Value::is_i64),
        "the report carries no integer population: {payload}"
    );
    Ok(())
}

/// A 64x64 solid red PNG, shared with the platform vision suite.
const RED_SQUARE_PNG_BASE64: &str = "iVBORw0KGgoAAAAN\
SUhEUgAAAEAAAABACAIAAAAlC+aJAAAAS0lEQVR42u3PQQkAAAgAsetfWiP4FgYrsKZeS0BAQEBA\
QEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEDgsqnc8OJg6Ln3AAAAAElF\
TkSuQmCC";

async fn describes_an_inline_image(model: &str) -> TestResult {
    let Some(client) = live_client() else {
        return missing_credential();
    };
    let request = request(model)
        .message(Message::new(Role::User, [
            ContentPart::Text {
                text: "What single color fills this image? Answer with the color name.".to_owned(),
            },
            ContentPart::Image(ImageContent {
                source: MediaSource::base64(RED_SQUARE_PNG_BASE64, "image/png"),
                detail: None,
            }),
        ]))
        .build()?;
    let response = client.complete(request).await?;
    assert!(
        response.text().to_lowercase().contains("red"),
        "the model did not see a red image: {:?}",
        response.text()
    );
    Ok(())
}

/// A question hard enough that a reasoning model actually reasons; shared
/// with the platform reasoning suite.
const HARD_PROMPT: &str = "A bookshop sells notebooks at $7 and pens at $3. Anna spent exactly \
     $118 and bought at least one of each. How many different combinations of notebooks and pens \
     could she have bought? Answer with just the number.";

async fn accepts_the_effort_level(model: &str, effort: ReasoningEffort) -> TestResult {
    let Some(client) = live_client() else {
        return missing_credential();
    };
    let request = request(model)
        .user(HARD_PROMPT)
        .reasoning_effort(effort)
        .build()?;
    let response = client.complete(request).await?;
    assert!(
        matches!(
            response.finish_reason,
            FinishReason::Stop | FinishReason::Length
        ),
        "effort {effort:?} was not answered normally: {:?}",
        response.finish_reason
    );
    support::observe(&format!(
        "{model} effort {effort:?}: reasoning tokens {}",
        response.usage.reasoning
    ));
    if matches!(effort, ReasoningEffort::Xhigh | ReasoningEffort::Max) {
        assert!(
            response.usage.reasoning > 0,
            "{model} showed no reasoning at effort {effort:?} on a hard question"
        );
    }
    Ok(())
}

async fn replays_reasoning_with_a_tool_result(model: &str) -> TestResult {
    let Some(client) = live_client() else {
        return missing_credential();
    };
    let opening = client
        .complete(
            request(model)
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
            request(model)
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

// ---------------------------------------------------------------------------
// Envelope cells
// ---------------------------------------------------------------------------

/// `max_output_tokens` has no capability gate, so it reaches the codec,
/// which drops it in codex mode — the deployment rejects the field — and
/// the request completes with the one warning.
#[tokio::test]
#[ignore = "live Codex call; spends the ChatGPT seat's quota"]
async fn an_output_cap_is_dropped_with_a_warning() -> TestResult {
    let Some(client) = live_client() else {
        return missing_credential();
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
/// `detail`-spelled error body must classify cleanly. Live-only: the twin
/// records no error responses.
#[tokio::test]
#[ignore = "live Codex call; spends the ChatGPT seat's quota"]
async fn a_pro_model_is_refused_upstream() -> TestResult {
    if let Some(skip) = support::live_only("live error classification") {
        return skip;
    }
    let Some(client) = live_client() else {
        return missing_credential();
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
