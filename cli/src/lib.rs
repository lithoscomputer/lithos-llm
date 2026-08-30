use std::error::Error as StdError;
use std::ffi::OsString;
use std::fmt::Write as _;
use std::io::{Error as IoError, Read, Write};
use std::time::Duration;

use lithos_llm::Client;
use lithos_llm::middleware::CancellationToken;
use lithos_llm::types::{Error as LlmError, ErrorKind};
use thiserror::Error;

pub mod args;
mod attachment;
mod input;
mod models;
mod output;
mod prompt;

use args::Command;

pub type CliResult<T> = Result<T, CliError>;

#[derive(Debug, Error)]
pub enum CliError {
    #[error("{message}")]
    Input { message: String },
    #[error("{message}")]
    InputSource {
        message: String,
        #[source]
        source:  Box<dyn StdError + Send + Sync>,
    },
    #[error("{0}")]
    Llm(#[source] LlmError),
    #[error("the operation was interrupted")]
    Interrupted,
    #[error("could not encode JSON output")]
    Json(#[source] serde_json::Error),
    #[error("could not write output")]
    Output(#[source] IoError),
}

impl CliError {
    pub(crate) fn input_source(
        message: impl Into<String>,
        source: impl StdError + Send + Sync + 'static,
    ) -> Self {
        Self::InputSource {
            message: message.into(),
            source:  Box::new(source),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TerminalState {
    pub stdin:  bool,
    pub stdout: bool,
    pub stderr: bool,
}

pub struct ProcessIo<R, W, E> {
    pub stdin:  R,
    pub stdout: W,
    pub stderr: E,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExitStatus {
    Success,
    Failure,
    Usage,
    Interrupted,
}

impl ExitStatus {
    pub const fn code(self) -> u8 {
        match self {
            Self::Success => 0,
            Self::Failure => 1,
            Self::Usage => 2,
            Self::Interrupted => 130,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OutputState {
    Written,
    Closed,
}

pub async fn run<I, T, R, W, E>(
    arguments: I,
    client: &Client,
    mut io: ProcessIo<R, W, E>,
    terminal: TerminalState,
    cancellation: CancellationToken,
) -> ExitStatus
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
    R: Read,
    W: Write,
    E: Write,
{
    let parsed = match args::parse_from(arguments) {
        Ok(parsed) => parsed,
        Err(error) => {
            let status = if error.use_stderr() {
                ExitStatus::Usage
            } else {
                ExitStatus::Success
            };
            let destination: &mut dyn Write = if error.use_stderr() {
                &mut io.stderr
            } else {
                &mut io.stdout
            };
            let _ignored = destination.write_all(error.to_string().as_bytes());
            return status;
        }
    };

    let result = match parsed.cli.command {
        Command::Models(arguments) => models::render(client, &arguments, &mut io.stdout),
        Command::Prompt(arguments) => {
            prompt::run(
                client,
                &arguments,
                &parsed.attachments,
                terminal,
                &mut io.stdin,
                &mut io.stdout,
                &cancellation,
            )
            .await
        }
    };
    match result {
        Ok(OutputState::Written | OutputState::Closed) => ExitStatus::Success,
        Err(error) => {
            let status = status_for(&error);
            let diagnostic = render_error(&error);
            let _ignored = writeln!(io.stderr, "{diagnostic}");
            status
        }
    }
}

fn status_for(error: &CliError) -> ExitStatus {
    match error {
        CliError::Input { .. } | CliError::InputSource { .. } => ExitStatus::Usage,
        CliError::Interrupted => ExitStatus::Interrupted,
        CliError::Llm(error) => match error.kind() {
            ErrorKind::ModelSelection | ErrorKind::InvalidRequest => ExitStatus::Usage,
            ErrorKind::Cancelled => ExitStatus::Interrupted,
            _ => ExitStatus::Failure,
        },
        CliError::Json(_) | CliError::Output(_) => ExitStatus::Failure,
    }
}

fn render_error(error: &CliError) -> String {
    let CliError::Llm(error) = error else {
        return format!("error: {}", format_error_chain(error));
    };
    let mut rendered = format!("error: {}: {}", error_kind(error.kind()), error.message());
    if let Some(provider) = error.provider() {
        let _ignored = write!(rendered, " provider={provider}");
    }
    if let Some(status) = error.status() {
        let _ignored = write!(rendered, " status={status}");
    }
    if let Some(code) = error.provider_code() {
        let _ignored = write!(rendered, " code={code}");
    }
    if let Some(delay) = error.provider_retry_after() {
        let _ignored = write!(rendered, " retry_after={}", duration_text(delay));
    }
    append_sources(&mut rendered, error);
    rendered
}

/// Formats an error and each preserved source for command-line diagnostics.
pub fn format_error_chain(error: &(dyn StdError + 'static)) -> String {
    let mut rendered = error.to_string();
    append_sources(&mut rendered, error);
    rendered
}

fn append_sources(rendered: &mut String, error: &(dyn StdError + 'static)) {
    let mut source = error.source();
    while let Some(cause) = source {
        let _ignored = write!(rendered, "\n  caused by: {cause}");
        source = cause.source();
    }
}

const fn error_kind(kind: ErrorKind) -> &'static str {
    match kind {
        ErrorKind::Configuration => "configuration",
        ErrorKind::ModelSelection => "model_selection",
        ErrorKind::Authentication => "authentication",
        ErrorKind::AccessDenied => "access_denied",
        ErrorKind::NotFound => "not_found",
        ErrorKind::InvalidRequest => "invalid_request",
        ErrorKind::ContextLength => "context_length",
        ErrorKind::RateLimit => "rate_limit",
        ErrorKind::QuotaExceeded => "quota_exceeded",
        ErrorKind::ContentFilter => "content_filter",
        ErrorKind::Server => "server",
        ErrorKind::Provider => "provider",
        ErrorKind::Network => "network",
        ErrorKind::Timeout => "timeout",
        ErrorKind::StreamDecode => "stream_decode",
        ErrorKind::ResponseDecode => "response_decode",
        ErrorKind::Middleware => "middleware",
        ErrorKind::Cancelled => "cancelled",
        _ => "unknown",
    }
}

fn duration_text(duration: Duration) -> String {
    humantime::format_duration(duration).to_string()
}
