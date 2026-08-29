//! The shared harness for the live E2E suites.
//!
//! The stream-contract machinery and the catalog loader are reused from the
//! wire-parity target verbatim, so a live stream is held to exactly the same
//! invariants as a mocked one and the two targets can never drift apart on
//! what a legal stream is.

use std::error::Error as StdError;
use std::time::Duration;

use lithos_llm::types::{ContentPart, Response, ResponseStream, ToolCall};
use serde_json::Value;

/// The wire-parity harness, included by path rather than copied.
///
/// Only its provider-independent pieces are meaningful here — the stream
/// contract, the stream collector, and the catalog loader. The mock-mounting
/// half is unused in a live target.
#[path = "../it/support.rs"]
pub(crate) mod it_support;

pub(crate) use it_support::{assert_stream_contract, catalog_from_toml, collect_stream_events};

pub(crate) type TestResult = Result<(), Box<dyn StdError>>;

/// How long any single live request may take before the test fails.
///
/// High-effort reasoning calls are the slow end; everything else finishes in
/// seconds. The bound exists so a wedged provider fails a test instead of
/// hanging the nightly run.
pub(crate) const LIVE_TIMEOUT: Duration = Duration::from_secs(300);

/// Passes a test while noting on stderr why it did no work.
///
/// Used when a provider's key variable is unset. A missing credential must
/// not fail a local run, and must not pass silently either, so the reason
/// lands in the captured test output.
#[allow(
    clippy::print_stderr,
    reason = "a skipped live test must say why in the captured test output"
)]
#[allow(
    clippy::unnecessary_wraps,
    reason = "the result type lets a caller end its test with `return support::skip(..)`"
)]
pub(crate) fn skip(reason: &str) -> TestResult {
    eprintln!("skipped: {reason}");
    Ok(())
}

/// Notes a probe observation on stderr without failing anything.
///
/// Probe tests record live behavior the catalog does not pin yet — see the
/// effort and rate-limit probes. Their output is the deliverable.
#[allow(
    clippy::print_stderr,
    reason = "a probe's observation is its deliverable and belongs in the test output"
)]
pub(crate) fn observe(note: &str) {
    eprintln!("probe: {note}");
}

/// Drives a live stream to its end, asserts the stream contract, and returns
/// the raw events with the completed response.
///
/// # Errors
///
/// Fails when the stream never completes, carrying the embedded stream error
/// when one was emitted.
pub(crate) async fn checked_stream(
    stream: ResponseStream,
) -> Result<(Vec<Value>, Response), Box<dyn StdError>> {
    let events = collect_stream_events(stream).await;
    assert_stream_contract(&events);
    let completed = events
        .iter()
        .rev()
        .find(|event| event.get("type").and_then(Value::as_str) == Some("completed"));
    let Some(completed) = completed else {
        let failure = events
            .iter()
            .rev()
            .find(|event| event.get("type").and_then(Value::as_str) == Some("error"))
            .map_or_else(|| "no error event".to_owned(), ToString::to_string);
        return Err(format!("the stream never completed: {failure}").into());
    };
    let response: Response = serde_json::from_value(
        completed
            .get("response")
            .ok_or("the completed event carries no response")?
            .clone(),
    )?;
    Ok((events, response))
}

/// The tool calls in a response, in content order.
pub(crate) fn tool_calls(response: &Response) -> Vec<&ToolCall> {
    response
        .content
        .iter()
        .filter_map(|part| match part {
            ContentPart::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect()
}

/// The response's JSON payload for a structured-output request.
///
/// Codecs differ on whether structured output arrives as a text part holding
/// JSON or as a decoded [`ContentPart::Json`] part, and both are legal, so a
/// structured-output test accepts either.
pub(crate) fn json_payload(response: &Response) -> Result<Value, Box<dyn StdError>> {
    if let Some(value) = response.content.iter().find_map(|part| match part {
        ContentPart::Json { value } => Some(value.clone()),
        _ => None,
    }) {
        return Ok(value);
    }
    let text = response.text();
    Ok(serde_json::from_str(text.trim())?)
}
