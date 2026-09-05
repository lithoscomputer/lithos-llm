use std::fmt::Write as _;
use std::io::Write;
use std::time::Duration;

use lithos_llm::Client;
use lithos_llm::client::{ProbeOptions, ProbeOutcome, ProbeReport};
use lithos_llm::middleware::CancellationToken;
use lithos_llm::types::{ErrorData, ReasoningEffort, TokenCounts};
use serde::Serialize;

use crate::app::args::{ProbeArgs, ReasoningEffortArg};
use crate::app::models::select_model;
use crate::app::output::write_text;
use crate::app::usage::{duration, grouped};
use crate::app::{CliEnvironment, CliError, CliResult, ExitStatus, error_kind};

#[derive(Serialize)]
struct JsonReport<'a> {
    version:    u32,
    route:      Option<String>,
    outcome:    JsonOutcome<'a>,
    latency_ms: u64,
    usage:      TokenCounts,
}

#[derive(Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum JsonOutcome<'a> {
    Passed,
    Failed { error: &'a ErrorData },
    Incorrect { detail: &'a str },
    Unknown,
}

pub(crate) async fn run(
    client: &Client,
    args: &ProbeArgs,
    environment: &CliEnvironment,
    output: &mut impl Write,
    cancellation: &CancellationToken,
) -> CliResult<ExitStatus> {
    let selector = select_model(
        client,
        args.model.as_deref(),
        &args.model_query,
        environment,
    )?;
    let mut options = ProbeOptions::new().tools(args.tools);
    if let Some(effort) = args.reasoning_effort {
        options = options.reasoning_effort(reasoning_effort(effort));
    }
    if let Some(timeout) = args.timeout {
        options = options.timeout(timeout);
    }
    let report = tokio::select! {
        biased;
        () = cancellation.cancelled() => return Err(CliError::Interrupted),
        report = client.probe(selector.clone(), options) => report,
    };
    let rendered = if args.json {
        render_json(&report)?
    } else {
        render_text(&selector, &report)
    };
    write_text(output, &rendered)?;
    Ok(if report.passed() {
        ExitStatus::Success
    } else {
        ExitStatus::Failure
    })
}

fn render_text(selector: &str, report: &ProbeReport) -> String {
    let route = report
        .route
        .as_ref()
        .map_or_else(|| selector.to_owned(), ToString::to_string);
    let finding = match &report.outcome {
        ProbeOutcome::Passed => format!("passed {route}"),
        ProbeOutcome::Failed(error) => failed_text(&route, error),
        ProbeOutcome::Incorrect { detail } => format!("incorrect {route} · {detail}"),
        _ => format!("failed {route} · unknown probe outcome"),
    };
    format!(
        "{finding} · {} input · {} output · {}",
        grouped(input_tokens(report.usage)),
        grouped(report.usage.billable_output()),
        duration(report.latency)
    )
}

fn failed_text(route: &str, error: &ErrorData) -> String {
    let mut rendered = format!(
        "failed {route} · {} · {}",
        error_kind(&error.kind),
        error.message
    );
    if let Some(status) = error.status {
        let _ignored = write!(rendered, " · status={status}");
    }
    if let Some(code) = &error.provider_code {
        let _ignored = write!(rendered, " · code={code}");
    }
    if let Some(source) = error
        .source_message
        .as_deref()
        .filter(|source| *source != error.message.as_str())
    {
        let _ignored = write!(rendered, " · {source}");
    }
    rendered
}

fn render_json(report: &ProbeReport) -> CliResult<String> {
    let outcome = match &report.outcome {
        ProbeOutcome::Passed => JsonOutcome::Passed,
        ProbeOutcome::Failed(error) => JsonOutcome::Failed { error },
        ProbeOutcome::Incorrect { detail } => JsonOutcome::Incorrect { detail },
        _ => JsonOutcome::Unknown,
    };
    serde_json::to_string_pretty(&JsonReport {
        version: 1,
        route: report.route.as_ref().map(ToString::to_string),
        outcome,
        latency_ms: milliseconds(report.latency),
        usage: report.usage,
    })
    .map_err(CliError::Json)
}

const fn input_tokens(usage: TokenCounts) -> u64 {
    usage
        .input
        .saturating_add(usage.cache_read)
        .saturating_add(usage.cache_write)
}

fn milliseconds(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

const fn reasoning_effort(value: ReasoningEffortArg) -> ReasoningEffort {
    match value {
        ReasoningEffortArg::Minimal => ReasoningEffort::Minimal,
        ReasoningEffortArg::Low => ReasoningEffort::Low,
        ReasoningEffortArg::Medium => ReasoningEffort::Medium,
        ReasoningEffortArg::High => ReasoningEffort::High,
        ReasoningEffortArg::Xhigh => ReasoningEffort::Xhigh,
        ReasoningEffortArg::Max => ReasoningEffort::Max,
    }
}
