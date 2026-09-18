//! The shared harness for the live E2E suites.
//!
//! The stream-contract machinery and the catalog loader are reused from the
//! wire-parity target verbatim, so a live stream is held to exactly the same
//! invariants as a mocked one and the two targets can never drift apart on
//! what a legal stream is.

use std::error::Error as StdError;
use std::time::Duration;
use std::{env, thread};

use lithos_llm::types::{ContentPart, Response, ResponseStream, ToolCall};
use lithos_llm::{Evaluation, Verdict};
use serde_json::Value;

/// The wire-parity harness, included by path rather than copied.
///
/// Only its provider-independent pieces are meaningful here — the stream
/// contract, the stream collector, and the catalog loader. The mock-mounting
/// half is unused in a live target.
#[path = "../it/support.rs"]
pub(crate) mod it_support;

pub(crate) use it_support::{assert_stream_contract, collect_stream_events};

pub(crate) type TestResult = Result<(), Box<dyn StdError>>;

/// Which backend the suite runs against.
///
/// The default is `Replay`: offline, keyless, and deterministic against the
/// committed recording, served by the twin the `test:e2e` mise task starts.
/// `Record` proxies live traffic through the twin and rewrites the
/// recording; `Live` is the unproxied nightly battery.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Backend {
    Live,
    Record,
    Replay,
}

impl Backend {
    pub(crate) fn from_env() -> Self {
        match env::var("LITHOS_E2E_BACKEND").as_deref() {
            Ok("live") => Self::Live,
            Ok("record") => Self::Record,
            Ok("replay") | Err(_) => Self::Replay,
            Ok(other) => panic!("LITHOS_E2E_BACKEND must be live, record, or replay, got {other}"),
        }
    }
}

/// Skips a test that only makes sense against the live API.
///
/// Returns `Some(skip)` under record and replay for behavior the proxy
/// cannot carry: endpoints it does not forward, error responses it does not
/// record, and timing behavior.
pub(crate) fn live_only(what: &str) -> Option<TestResult> {
    (Backend::from_env() != Backend::Live)
        .then(|| skip(&format!("live-only ({what}) under record/replay")))
}

/// The recording namespace for the current test: its full test path.
///
/// The test harness names each test's thread after the test, and Nextest
/// runs one test per process, so the thread name is a stable, unique
/// namespace. It becomes the fake bearer token the twin records and replays
/// under.
pub(crate) fn test_namespace() -> String {
    let thread = thread::current();
    let name = thread.name().unwrap_or("main");
    assert_ne!(
        name, "main",
        "the recording namespace needs the test thread's name; run under the test harness"
    );
    name.to_owned()
}

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
        .find(|event| event.get("type").and_then(Value::as_str) == Some("ended"));
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
/// structured-output test accepts either. A Markdown code fence around the
/// document is tolerated too: `json_object` mode promises valid JSON, not
/// bare JSON, and Claude models on Venice wrap it.
pub(crate) fn json_payload(response: &Response) -> Result<Value, Box<dyn StdError>> {
    if let Some(value) = response.content.iter().find_map(|part| match part {
        ContentPart::Json { value } => Some(value.clone()),
        _ => None,
    }) {
        return Ok(value);
    }
    let text = response.text();
    let mut text = text.trim();
    if let Some(fenced) = text.strip_prefix("```") {
        let fenced = fenced.strip_prefix("json").unwrap_or(fenced);
        text = fenced.strip_suffix("```").unwrap_or(fenced).trim();
    }
    // Some models append prose after the document — Claude Fable 5 on Venice
    // does — so the payload is the first complete JSON value, not the whole
    // text.
    let mut values = serde_json::Deserializer::from_str(text).into_iter::<Value>();
    Ok(values
        .next()
        .ok_or("the response text holds no JSON value")??)
}

/// The option names of the mixed evaluation's choice question, in order.
pub(crate) const MIXED_OPTIONS: [&str; 3] = ["billing", "technical", "other"];

/// The interface plan's three-question evaluation against `model`: one
/// question of each kind about one support message.
///
/// Every suite judges this same case, so a native evaluation model and a
/// generation model used as a judge are held to the same shape. The score
/// scale has three levels, so a valid score lies in `[0, 2]`.
pub(crate) fn mixed_evaluation(model: &str) -> Evaluation {
    Evaluation::builder()
        .model(model)
        .state("I was charged twice. Please refund the duplicate.")
        .choice("department", "Which team should handle this?", [
            (MIXED_OPTIONS[0], Some("Charges and refunds")),
            (MIXED_OPTIONS[1], Some("Bugs and outages")),
            (MIXED_OPTIONS[2], None),
        ])
        .score("severity", "How severe is the issue?", [
            "Cosmetic",
            "Workaround exists",
            "Blocking; no workaround",
        ])
        .boolean("requests_refund", "Is the customer requesting money back?")
        .timeout(LIVE_TIMEOUT)
        .build()
        .expect("the mixed evaluation builds")
}

/// Asserts what any verdict on [`mixed_evaluation`] must satisfy: three
/// answers of the asked kinds, the choice naming one of the options, the
/// score on its scale, the boolean a probability, and input tokens counted.
///
/// The specific answer is never asserted; a model may change its mind on a
/// re-record.
///
/// # Errors
///
/// Fails when an answer is missing or has the wrong kind.
pub(crate) fn assert_mixed_answers(verdict: &Verdict) -> TestResult {
    assert_eq!(
        verdict.answers.len(),
        3,
        "the verdict answers {} questions, not 3",
        verdict.answers.len()
    );
    let department = verdict.choice("department")?;
    assert!(
        MIXED_OPTIONS.contains(&department.choice.as_str()),
        "the choice `{}` names no option",
        department.choice
    );
    let severity = verdict.score("severity")?;
    // Three levels: positions 0, 1, and 2.
    assert!(
        (0.0..=2.0).contains(&severity.score),
        "the score {} is off the three-level scale",
        severity.score
    );
    let refund = verdict.boolean("requests_refund")?;
    assert!(
        (0.0..=1.0).contains(&refund.probability),
        "the boolean probability {} is out of range",
        refund.probability
    );
    assert!(
        verdict.usage.input > 0,
        "the verdict counts no input tokens"
    );
    Ok(())
}

/// Asserts a judge's verdict on [`mixed_evaluation`]: everything in
/// [`assert_mixed_answers`], plus point estimates only. A generation model
/// judging through structured output reports no distribution, no
/// confidence, and no rounding; those fields are the native path's.
///
/// # Errors
///
/// Fails when an answer is missing or has the wrong kind.
pub(crate) fn assert_point_estimates(verdict: &Verdict) -> TestResult {
    assert_mixed_answers(verdict)?;
    let department = verdict.choice("department")?;
    assert!(
        department.probabilities.is_none() && department.confidence.is_none(),
        "a judge's choice carries a distribution or confidence: {department:?}"
    );
    let severity = verdict.score("severity")?;
    assert!(
        severity.probabilities.is_none() && severity.confidence.is_none(),
        "a judge's score carries a distribution or confidence: {severity:?}"
    );
    assert!(
        verdict.rounding.is_none(),
        "a judge's verdict declares rounding: {:?}",
        verdict.rounding
    );
    Ok(())
}
