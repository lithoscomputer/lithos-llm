use std::collections::BTreeSet;
use std::error::Error as StdError;
use std::ffi::OsString;
use std::fmt::Write as _;
use std::io::{Error as IoError, Read, Write};
use std::time::Duration;

use lithos_llm::Client;
use lithos_llm::catalog::ProviderId;
use lithos_llm::credentials::CredentialError;
use lithos_llm::middleware::CancellationToken;
use lithos_llm::types::{Error as LlmError, ErrorKind};
use thiserror::Error;

pub(crate) mod args;
mod attachment;
mod input;
mod models;
mod output;
mod probe;
mod prompt;
mod usage;

use args::Command;

pub(crate) type CliResult<T> = Result<T, CliError>;

#[derive(Clone, Debug, Default)]
pub(crate) struct CliEnvironment {
    model:                Option<OsString>,
    configured_providers: BTreeSet<ProviderId>,
}

impl CliEnvironment {
    pub(crate) fn new(
        model: Option<OsString>,
        configured_providers: impl IntoIterator<Item = ProviderId>,
    ) -> Self {
        Self {
            model,
            configured_providers: configured_providers.into_iter().collect(),
        }
    }

    pub(crate) fn model(&self) -> CliResult<Option<&str>> {
        let Some(model) = &self.model else {
            return Ok(None);
        };
        let model = model.to_str().ok_or_else(|| CliError::Input {
            message: "LLLM_MODEL must contain valid UTF-8".to_owned(),
        })?;
        if model.trim().is_empty() {
            return Err(CliError::Input {
                message: "LLLM_MODEL must not be empty".to_owned(),
            });
        }
        Ok(Some(model))
    }

    pub(crate) fn credentials_configured(&self, provider: &ProviderId) -> bool {
        self.configured_providers.contains(provider)
    }
}

#[derive(Debug, Error)]
pub(crate) enum CliError {
    #[error("{message}")]
    Input { message: String },
    #[error("{message}")]
    InputSource {
        message: String,
        #[source]
        source:  Box<dyn StdError + Send + Sync>,
    },
    #[error("{source}")]
    Call {
        route:  String,
        #[source]
        source: Box<LlmError>,
    },
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
pub(crate) struct TerminalState {
    pub(crate) stdin: bool,
}

pub(crate) struct ProcessIo<R, W, E> {
    pub(crate) stdin:  R,
    pub(crate) stdout: W,
    pub(crate) stderr: E,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExitStatus {
    Success,
    Failure,
    Usage,
    Interrupted,
}

impl ExitStatus {
    pub(crate) const fn code(self) -> u8 {
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

pub(crate) async fn run<I, T, R, W, E>(
    arguments: I,
    client: &Client,
    mut io: ProcessIo<R, W, E>,
    terminal: TerminalState,
    environment: &CliEnvironment,
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
        Command::Models(arguments) => {
            models::render(client, &arguments, environment, &mut io.stdout)
                .map(|_| ExitStatus::Success)
        }
        Command::Probe(arguments) => {
            probe::run(
                client,
                &arguments,
                environment,
                &mut io.stdout,
                &cancellation,
            )
            .await
        }
        Command::Resolve(arguments) => {
            models::render_resolution(client, &arguments, environment, &mut io.stdout)
                .map(|_| ExitStatus::Success)
        }
        Command::Prompt(arguments) => prompt::run(
            client,
            &arguments,
            &parsed.attachments,
            terminal,
            &mut io.stdin,
            &mut io.stdout,
            &mut io.stderr,
            environment,
            &cancellation,
        )
        .await
        .map(|_| ExitStatus::Success),
    };
    match result {
        Ok(status) => status,
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
        CliError::Call { source: error, .. } => match error.kind() {
            ErrorKind::ModelSelection | ErrorKind::InvalidRequest => ExitStatus::Usage,
            ErrorKind::Cancelled => ExitStatus::Interrupted,
            _ => ExitStatus::Failure,
        },
        CliError::Json(_) | CliError::Output(_) => ExitStatus::Failure,
    }
}

fn render_error(error: &CliError) -> String {
    let (error, route) = match error {
        CliError::Call { route, source } => (source.as_ref(), route.as_str()),
        _ => return format!("error: {}", format_error_chain(error)),
    };
    let mut rendered = format!("error: {}: {}", error_kind(&error.kind()), error.message());
    let _ignored = write!(rendered, " route={route}");
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
    if let Some(CredentialError::MissingSecret { name, .. }) = find_source::<CredentialError>(error)
    {
        let _ignored = write!(rendered, "\n  hint: set {name}");
    }
    rendered
}

fn find_source<'a, T: StdError + 'static>(error: &'a (dyn StdError + 'static)) -> Option<&'a T> {
    let mut current = Some(error);
    while let Some(source) = current {
        if let Some(found) = source.downcast_ref::<T>() {
            return Some(found);
        }
        current = source.source();
    }
    None
}

pub(crate) fn format_error_chain(error: &(dyn StdError + 'static)) -> String {
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

pub(crate) fn error_kind(kind: &ErrorKind) -> &'static str {
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
