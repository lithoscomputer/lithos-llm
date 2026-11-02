use std::fmt::Write as _;
use std::io::Write;

use lithos_llm::Client;
use lithos_llm::client::{ProbeOptions, ProbeOutcome, ProbeReport};
use lithos_llm::middleware::CancellationToken;
use lithos_llm::types::{ErrorData, TokenCounts};

use crate::app::args::ProbeArgs;
use crate::app::models::select_model;
use crate::app::output::write_text;
use crate::app::usage::{duration, grouped};
use crate::app::{CliEnvironment, CliError, CliResult, ExitStatus, cancellable, error_kind};

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
        options = options.reasoning_effort(effort.into());
    }
    if let Some(timeout) = args.timeout {
        options = options.timeout(timeout);
    }
    let report = cancellable(cancellation, client.probe(selector.clone(), options)).await?;
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
    serde_json::to_string_pretty(report).map_err(CliError::Json)
}

const fn input_tokens(usage: TokenCounts) -> u64 {
    usage
        .input
        .saturating_add(usage.cache_read)
        .saturating_add(usage.cache_write)
}
