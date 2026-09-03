use std::env;
use std::future::Future;
use std::io::{IsTerminal as _, Write as _, stderr, stdin, stdout};
use std::process::ExitCode;

mod app;
#[cfg(test)]
mod runner_tests;

use app::{CliEnvironment, ExitStatus, ProcessIo, TerminalState, args, format_error_chain};
use lithos_llm::catalog::{AuthScheme, Catalog};
use lithos_llm::client::ClientBuildError;
use lithos_llm::credentials::{CredentialProvider as _, EnvironmentCredentials};
use lithos_llm::middleware::{CancellationToken, TracingMiddleware};
use lithos_llm::{Client, ClientBuild};
use tokio::signal::ctrl_c;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> ExitCode {
    // The subscriber must exist before the client builds, but `run` owns
    // argument parsing and clap error rendering. This early parse only reads
    // the flag; a parse failure is reported by `run`.
    let verbose = args::parse_from(env::args_os()).is_ok_and(|parsed| parsed.cli.verbose);
    init_tracing(verbose);
    let (build, environment) = match build_client().await {
        Ok(build) => build,
        Err(error) => {
            let diagnostic = format_error_chain(&error);
            let _ignored = writeln!(stderr().lock(), "error: configuration: {diagnostic}");
            return ExitCode::from(ExitStatus::Failure.code());
        }
    };
    let cancellation = CancellationToken::new();

    let stdin = stdin();
    let stdout = stdout();
    let stderr = stderr();
    let terminal = TerminalState {
        stdin: stdin.is_terminal(),
    };
    let status = match Box::pin(run_until_signal(
        app::run(
            env::args_os(),
            &build.client,
            ProcessIo {
                stdin:  stdin.lock(),
                stdout: stdout.lock(),
                stderr: stderr.lock(),
            },
            terminal,
            &environment,
            cancellation.clone(),
        ),
        ctrl_c(),
        &cancellation,
    ))
    .await
    {
        Ok(status) => status,
        Err(error) => {
            let diagnostic = format_error_chain(&error);
            let _ignored = writeln!(stderr.lock(), "error: signal handling: {diagnostic}");
            ExitStatus::Failure
        }
    };
    ExitCode::from(status.code())
}

/// Builds the environment-configured client with call tracing attached.
///
/// This mirrors [`Client::from_env`] and adds [`TracingMiddleware`], so the
/// library's call spans reach the subscriber [`init_tracing`] installs.
async fn build_client() -> Result<(ClientBuild, CliEnvironment), ClientBuildError> {
    let catalog = Catalog::builder()
        .with_builtin()
        .build()
        .map_err(|source| ClientBuildError::BuiltInCatalog { source })?;
    let credentials = EnvironmentCredentials::conventional();
    let mut configured = Vec::new();
    for provider in catalog.providers() {
        if matches!(provider.auth(), AuthScheme::None)
            || credentials.credentials(provider).await.is_ok()
        {
            configured.push(provider.id().clone());
        }
    }
    let environment = CliEnvironment::new(env::var_os("LLLM_MODEL"), configured);
    let build = Client::builder()
        .catalog(catalog)
        .credentials(credentials)
        .middleware(TracingMiddleware)
        .build()?;
    Ok((build, environment))
}

/// Installs a stderr tracing subscriber when `RUST_LOG` or `--verbose` asks
/// for one.
///
/// A set `RUST_LOG` wins over `--verbose`. Without either, no subscriber is
/// installed, so normal use prints no telemetry and stdout stays reserved
/// for command output.
fn init_tracing(verbose: bool) {
    let filter = if env::var_os("RUST_LOG").is_some() {
        match EnvFilter::try_from_default_env() {
            Ok(filter) => filter,
            Err(error) => {
                let _ignored = writeln!(
                    stderr().lock(),
                    "warning: RUST_LOG is not a valid filter: {error}"
                );
                return;
            }
        }
    } else if verbose {
        EnvFilter::new("lithos_llm=debug")
    } else {
        return;
    };
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(stderr)
        .init();
}

async fn run_until_signal<R, S, E>(
    run: R,
    signal: S,
    cancellation: &CancellationToken,
) -> Result<ExitStatus, E>
where
    R: Future<Output = ExitStatus>,
    S: Future<Output = Result<(), E>>,
{
    tokio::pin!(run);
    tokio::select! {
        status = &mut run => Ok(status),
        result = signal => {
            result?;
            cancellation.cancel();
            Ok(run.await)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::future::{pending, ready};
    use std::io;

    use lithos_llm::middleware::CancellationToken;

    use super::{ExitStatus, run_until_signal};

    #[tokio::test]
    async fn a_finished_command_does_not_wait_for_a_signal() {
        let cancellation = CancellationToken::new();

        let status = run_until_signal(
            ready(ExitStatus::Success),
            pending::<io::Result<()>>(),
            &cancellation,
        )
        .await
        .expect("the command should finish");

        assert_eq!(status, ExitStatus::Success);
        assert!(!cancellation.is_cancelled());
    }

    #[tokio::test]
    async fn a_signal_cancels_and_waits_for_the_command() {
        let cancellation = CancellationToken::new();
        let observed = cancellation.clone();
        let run = async move {
            observed.cancelled().await;
            ExitStatus::Interrupted
        };

        let status = run_until_signal(run, ready(Ok::<(), io::Error>(())), &cancellation)
            .await
            .expect("the signal should be handled");

        assert_eq!(status, ExitStatus::Interrupted);
        assert!(cancellation.is_cancelled());
    }

    #[tokio::test]
    async fn a_signal_error_is_returned() {
        let cancellation = CancellationToken::new();
        let error = run_until_signal(
            pending::<ExitStatus>(),
            ready(Err(io::Error::other("signal unavailable"))),
            &cancellation,
        )
        .await
        .expect_err("the signal error should be returned");

        assert_eq!(error.kind(), io::ErrorKind::Other);
        assert!(!cancellation.is_cancelled());
    }
}
