use std::env;
use std::io::{IsTerminal as _, Write as _, stderr, stdin, stdout};
use std::process::ExitCode;

use lithos_llm::Client;
use lithos_llm::middleware::CancellationToken;
use lithos_llm_cli::{ExitStatus, ProcessIo, TerminalState};
use tokio::signal::ctrl_c;

#[tokio::main]
async fn main() -> ExitCode {
    let build = match Client::from_env() {
        Ok(build) => build,
        Err(error) => {
            let _ignored = writeln!(stderr().lock(), "error: configuration: {error}");
            return ExitCode::from(ExitStatus::Failure.code());
        }
    };
    let cancellation = CancellationToken::new();
    let signal = cancellation.clone();
    let _signal_task = tokio::spawn(async move {
        if ctrl_c().await.is_ok() {
            signal.cancel();
        }
    });

    let stdin = stdin();
    let stdout = stdout();
    let stderr = stderr();
    let terminal = TerminalState {
        stdin:  stdin.is_terminal(),
        stdout: stdout.is_terminal(),
        stderr: stderr.is_terminal(),
    };
    let status = lithos_llm_cli::run(
        env::args_os(),
        &build.client,
        ProcessIo {
            stdin:  stdin.lock(),
            stdout: stdout.lock(),
            stderr: stderr.lock(),
        },
        terminal,
        cancellation,
    )
    .await;
    ExitCode::from(status.code())
}
