#![cfg(feature = "runtime")]

use std::error::Error as StdError;
use std::future::pending;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use lithos_llm::adapter::{ProviderAdapter, ResolvedCall};
use lithos_llm::catalog::{AdapterId, Catalog, CatalogError};
use lithos_llm::client::{ProbeOptions, ProbeOutcome};
use lithos_llm::middleware::CallContext;
use lithos_llm::types::{
    ContentPart, Error, ErrorKind, FinishReason, Message, ReasoningEffort, Request, Response,
    ResponseStream, TokenCounts, ToolCall,
};
use lithos_llm::{Client, ClientBuild};
use serde_json::json;

/// Every request the fake provider received, in order.
type RequestLog = Arc<Mutex<Vec<Request>>>;

const TEST_CATALOG: &str = r#"
schema_version = 1

[providers.test]
display_name = "Test"
adapter = "test-adapter"
codec = "test-codec"
base_url = "http://127.0.0.1"
default_model = "chat"

[providers.test.auth]
type = "none"

[providers.test.models.chat]
display_name = "Chat"
api_model = "chat"
capabilities = { text = true }

[providers.test.models.agent]
display_name = "Agent"
api_model = "agent"
capabilities = { text = true, tools = true, reasoning = true }
"#;

/// How the fake provider behaves.
#[derive(Clone, Copy)]
enum Script {
    Answers,
    RejectsCredentials,
    NeverAnswers,
    /// Calls `add` twice with the results the probe hands back, then reports
    /// the last result as the total.
    UsesTheTool,
    /// Answers with the right number without ever calling the tool.
    SkipsTheTool,
    /// Calls the tool once, then answers without the total.
    MissesTheTotal,
}

struct ScriptedAdapter {
    id:       AdapterId,
    script:   Script,
    requests: RequestLog,
}

impl ScriptedAdapter {
    fn new(script: Script) -> Self {
        Self {
            id: AdapterId::new("test-adapter"),
            script,
            requests: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

/// The text of every tool result the probe has sent so far, in order.
fn tool_result_texts(request: &Request) -> Vec<String> {
    request
        .messages()
        .iter()
        .flat_map(Message::content)
        .filter_map(|part| match part {
            ContentPart::ToolResult(result) => Some(result),
            _ => None,
        })
        .flat_map(|result| result.content.iter())
        .filter_map(|part| match part {
            ContentPart::Text { text } => Some(text.clone()),
            _ => None,
        })
        .collect()
}

fn respond(call: &ResolvedCall, content: Vec<ContentPart>, finish: FinishReason) -> Response {
    let mut response = Response::new(
        call.route().provider().id().clone(),
        call.route().model().id().clone(),
        content,
    );
    response.finish_reason = finish;
    response.usage = TokenCounts {
        input: 10,
        output: 5,
        ..TokenCounts::default()
    };
    response
}

fn text(call: &ResolvedCall, text: &str) -> Response {
    respond(
        call,
        vec![ContentPart::Text {
            text: text.to_owned(),
        }],
        FinishReason::Stop,
    )
}

fn add_call(call: &ResolvedCall, id: &str, a: i64, b: i64) -> Response {
    respond(
        call,
        vec![ContentPart::ToolCall(ToolCall::function(
            id,
            "add",
            json!({ "a": a, "b": b }),
        ))],
        FinishReason::ToolCall,
    )
}

#[async_trait]
impl ProviderAdapter for ScriptedAdapter {
    fn id(&self) -> &AdapterId {
        &self.id
    }

    async fn complete(&self, call: &ResolvedCall) -> Result<Response, Error> {
        self.requests
            .lock()
            .expect("request log mutex should not be poisoned")
            .push(call.request().clone());
        let results = tool_result_texts(call.request());
        match (self.script, results.as_slice()) {
            (Script::Answers, _) => Ok(text(call, "OK")),
            (Script::RejectsCredentials, _) => {
                Err(Error::new(ErrorKind::Authentication, "bad key").with_status(401))
            }
            (Script::NeverAnswers, _) => pending().await,
            (Script::UsesTheTool | Script::MissesTheTotal, []) => {
                Ok(add_call(call, "call_1", 15, 27))
            }
            (Script::UsesTheTool, [first]) => {
                Ok(add_call(call, "call_2", first.parse().unwrap_or(0), 42))
            }
            (Script::UsesTheTool, [.., last]) => Ok(text(
                call,
                &format!("The grand total is {last}, which is even."),
            )),
            (Script::SkipsTheTool, _) => Ok(text(call, "The grand total is 84, which is even.")),
            (Script::MissesTheTotal, _) => Ok(text(call, "Done.")),
        }
    }

    async fn stream(&self, _call: &ResolvedCall) -> Result<ResponseStream, Error> {
        Err(Error::new(ErrorKind::Middleware, "not used"))
    }
}

fn catalog() -> Result<Catalog, CatalogError> {
    Catalog::builder().overlay_toml(TEST_CATALOG)?.build()
}

fn client(script: Script) -> Result<(Client, RequestLog), Box<dyn StdError>> {
    let adapter = ScriptedAdapter::new(script);
    let requests = adapter.requests.clone();
    let ClientBuild { client, .. } = Client::builder()
        .catalog(catalog()?)
        .adapter("test", adapter)
        .build()?;
    Ok((client, requests))
}

fn requests(log: &RequestLog) -> Vec<Request> {
    log.lock()
        .expect("request log mutex should not be poisoned")
        .clone()
}

#[tokio::test]
async fn a_one_word_probe_passes_and_reports_route_and_usage() -> Result<(), Box<dyn StdError>> {
    let (client, log) = client(Script::Answers)?;

    let report = client.probe("test/chat", ProbeOptions::new()).await;

    assert!(report.passed(), "{report:?}");
    assert_eq!(
        report.route.map(|route| route.to_string()),
        Some("test/chat".to_owned())
    );
    assert_eq!(report.usage.input, 10);
    assert_eq!(report.usage.output, 5);
    let requests = requests(&log);
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].max_output_tokens(), Some(16));
    assert!(requests[0].tools().is_empty());
    assert_eq!(requests[0].messages()[0].content(), [ContentPart::Text {
        text: "Say OK".to_owned(),
    }]);
    Ok(())
}

#[tokio::test]
async fn reasoning_effort_widens_the_output_budget() -> Result<(), Box<dyn StdError>> {
    let (client, log) = client(Script::Answers)?;

    let report = client
        .probe(
            "test/agent",
            ProbeOptions::new().reasoning_effort(ReasoningEffort::High),
        )
        .await;

    assert!(report.passed(), "{report:?}");
    let requests = requests(&log);
    assert_eq!(requests[0].max_output_tokens(), Some(1024));
    assert_eq!(requests[0].reasoning_effort(), Some(ReasoningEffort::High));
    Ok(())
}

#[tokio::test]
async fn an_unknown_model_fails_before_any_request() -> Result<(), Box<dyn StdError>> {
    let (client, log) = client(Script::Answers)?;

    let report = client.probe("test/missing", ProbeOptions::new()).await;

    let ProbeOutcome::Failed(data) = &report.outcome else {
        panic!("an unknown model must fail the probe: {report:?}");
    };
    assert_eq!(data.kind, ErrorKind::ModelSelection);
    assert!(report.route.is_none());
    assert!(requests(&log).is_empty());
    Ok(())
}

#[tokio::test]
async fn bad_credentials_fail_with_the_authentication_kind() -> Result<(), Box<dyn StdError>> {
    let (client, _) = client(Script::RejectsCredentials)?;

    let report = client.probe("test/chat", ProbeOptions::new()).await;

    let ProbeOutcome::Failed(data) = &report.outcome else {
        panic!("a rejected key must fail the probe: {report:?}");
    };
    assert_eq!(data.kind, ErrorKind::Authentication);
    assert_eq!(data.status, Some(401));
    assert_eq!(
        report.route.map(|route| route.to_string()),
        Some("test/chat".to_owned())
    );
    Ok(())
}

#[tokio::test]
async fn a_tool_probe_on_a_model_without_tools_fails_locally() -> Result<(), Box<dyn StdError>> {
    let (client, log) = client(Script::UsesTheTool)?;

    let report = client
        .probe("test/chat", ProbeOptions::new().tools(true))
        .await;

    let ProbeOutcome::Failed(data) = &report.outcome else {
        panic!("a no-tools model must fail the tool probe: {report:?}");
    };
    assert_eq!(data.kind, ErrorKind::InvalidRequest);
    assert_eq!(
        data.provider_code.as_deref(),
        Some("unsupported_capability")
    );
    assert!(
        requests(&log).is_empty(),
        "the catalog rejection must happen before any request"
    );
    Ok(())
}

#[tokio::test]
async fn a_tool_probe_passes_when_the_model_uses_the_tool() -> Result<(), Box<dyn StdError>> {
    let (client, log) = client(Script::UsesTheTool)?;

    let report = client
        .probe("test/agent", ProbeOptions::new().tools(true))
        .await;

    assert!(report.passed(), "{report:?}");
    let requests = requests(&log);
    assert_eq!(requests.len(), 3, "two tool turns and one final answer");
    assert_eq!(
        tool_result_texts(&requests[2]),
        ["42", "84"],
        "the probe answers the tool calls itself"
    );
    assert_eq!(requests[0].tools().len(), 1);
    assert_eq!(requests[0].max_output_tokens(), Some(1024));
    assert_eq!(report.usage.input, 30, "usage sums across turns");
    Ok(())
}

#[tokio::test]
async fn a_tool_probe_fails_when_the_model_skips_the_tool() -> Result<(), Box<dyn StdError>> {
    let (client, _) = client(Script::SkipsTheTool)?;

    let report = client
        .probe("test/agent", ProbeOptions::new().tools(true))
        .await;

    let ProbeOutcome::Incorrect { detail } = &report.outcome else {
        panic!("a skipped tool must be an incorrect outcome: {report:?}");
    };
    assert!(detail.contains("without calling the tool"), "{detail}");
    Ok(())
}

#[tokio::test]
async fn a_tool_probe_fails_on_the_wrong_total() -> Result<(), Box<dyn StdError>> {
    let (client, _) = client(Script::MissesTheTotal)?;

    let report = client
        .probe("test/agent", ProbeOptions::new().tools(true))
        .await;

    let ProbeOutcome::Incorrect { detail } = &report.outcome else {
        panic!("a wrong total must be an incorrect outcome: {report:?}");
    };
    assert!(detail.contains("84"), "{detail}");
    Ok(())
}

#[tokio::test]
async fn a_probe_that_runs_out_of_time_reports_a_timeout() -> Result<(), Box<dyn StdError>> {
    let (client, _) = client(Script::NeverAnswers)?;

    let report = client
        .probe(
            "test/chat",
            ProbeOptions::new().timeout(Duration::from_millis(10)),
        )
        .await;

    let ProbeOutcome::Failed(data) = &report.outcome else {
        panic!("an unanswered probe must time out: {report:?}");
    };
    assert_eq!(data.kind, ErrorKind::Timeout);
    assert!(report.latency >= Duration::from_millis(10));
    Ok(())
}

#[tokio::test]
async fn an_outer_cancellation_ends_a_probe() -> Result<(), Box<dyn StdError>> {
    let (client, log) = client(Script::NeverAnswers)?;
    let context = CallContext::new();
    context.cancellation().cancel();

    let report = client
        .probe_with_context("test/chat", ProbeOptions::new(), context)
        .await;

    let ProbeOutcome::Failed(data) = &report.outcome else {
        panic!("a cancelled probe must fail: {report:?}");
    };
    assert_eq!(data.kind, ErrorKind::Cancelled);
    assert!(requests(&log).is_empty());
    Ok(())
}
