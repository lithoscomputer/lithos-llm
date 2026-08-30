use std::env;
use std::future::Future;
use std::io::{IsTerminal as _, Write as _, stderr, stdin, stdout};
use std::process::ExitCode;

use lithos_llm::Client;
use lithos_llm::middleware::CancellationToken;
use lithos_llm_cli::{ExitStatus, ProcessIo, TerminalState, format_error_chain};
use tokio::signal::ctrl_c;

#[tokio::main]
async fn main() -> ExitCode {
    let build = match Client::from_env() {
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
        stdin:  stdin.is_terminal(),
        stdout: stdout.is_terminal(),
        stderr: stderr.is_terminal(),
    };
    let status = match Box::pin(run_until_signal(
        lithos_llm_cli::run(
            env::args_os(),
            &build.client,
            ProcessIo {
                stdin:  stdin.lock(),
                stdout: stdout.lock(),
                stderr: stderr.lock(),
            },
            terminal,
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
    use lithos_llm_cli::ExitStatus;

    use super::run_until_signal;

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
