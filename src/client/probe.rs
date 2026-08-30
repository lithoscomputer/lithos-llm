//! Model probes: one cheap exchange that answers "can this client serve this
//! model right now" with a classified report instead of an error.

use std::time::{Duration, Instant};

use serde_json::{Value, json};

use super::Client;
use crate::catalog::ModelHandle;
use crate::middleware::CallContext;
use crate::types::{
    ContentPart, Error, ErrorData, ErrorKind, Message, ReasoningEffort, Request, Response, Role,
    TokenCounts, ToolCall, ToolDefinition, ToolResult,
};

/// The default bound on a probe that asks for one word.
const BASIC_TIMEOUT: Duration = Duration::from_secs(30);
/// The default bound on a probe that runs a tool exchange.
const TOOL_TIMEOUT: Duration = Duration::from_secs(90);
/// Output budget for a probe whose whole answer is one word.
const BASIC_OUTPUT_TOKENS: u32 = 16;
/// Output budget when reasoning or tool rounds spend completion tokens before
/// the answer.
const EXPANDED_OUTPUT_TOKENS: u32 = 1024;
/// The most model turns a tool probe takes before it gives up.
const MAX_TOOL_TURNS: u32 = 5;
const BASIC_PROMPT: &str = "Say OK";
const TOOL_PROMPT: &str = "Use the add tool twice: first add 15 and 27, then add that result to \
                           42. Finally, tell me whether the grand total is even or odd and why.";
const TOOL_NAME: &str = "add";
/// The grand total a correct tool exchange arrives at: (15 + 27) + 42.
const TOOL_ANSWER: &str = "84";

/// What a probe exercises and how long it may take.
///
/// The default probe sends one short prompt and accepts any successful
/// response. A tool probe additionally hands the model one `add` tool and
/// checks that it calls the tool, uses the results, and reaches the right
/// total.
#[derive(Clone, Debug, Default)]
#[must_use]
pub struct ProbeOptions {
    tools:            bool,
    reasoning_effort: Option<ReasoningEffort>,
    timeout:          Option<Duration>,
}

impl ProbeOptions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Runs a tool exchange instead of a one-word prompt.
    ///
    /// A model whose catalog entry declares no tool support fails the probe
    /// locally, before any request is sent.
    pub fn tools(mut self, tools: bool) -> Self {
        self.tools = tools;
        self
    }

    /// Requests this reasoning effort, and widens the output budget so the
    /// reasoning has room before the answer.
    pub fn reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }

    /// Bounds the whole probe, every turn of a tool exchange included.
    ///
    /// The default is 30 seconds for the one-word prompt and 90 seconds for a
    /// tool exchange. A probe that runs out of time reports a `Timeout`
    /// failure.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    fn effective_timeout(&self) -> Duration {
        self.timeout.unwrap_or(if self.tools {
            TOOL_TIMEOUT
        } else {
            BASIC_TIMEOUT
        })
    }
}

/// How a probe ended.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ProbeOutcome {
    /// Every request succeeded and, for a tool probe, the model used the tool
    /// and reached the right total.
    Passed,
    /// A request failed. The data's [`kind`](ErrorData::kind) says why:
    /// `ModelSelection` for an unknown model, `Authentication`, `NotFound`,
    /// `InvalidRequest` with the `unsupported_capability` provider code for a
    /// catalog rejection, `Timeout` when the probe ran out of time, and so on.
    Failed(ErrorData),
    /// Every request succeeded, but the model did not behave as a working
    /// model would: it never called the tool, gave the wrong total, or was
    /// still calling tools when the turn budget ran out.
    Incorrect { detail: String },
}

/// The result of one probe.
#[derive(Clone, Debug)]
#[must_use]
#[non_exhaustive]
pub struct ProbeReport {
    /// The canonical route the probe resolved, or `None` when selection
    /// itself failed.
    pub route:   Option<ModelHandle>,
    pub outcome: ProbeOutcome,
    /// Wall-clock time from the start of the probe to its report.
    pub latency: Duration,
    /// The usage every successful request reported, summed.
    pub usage:   TokenCounts,
}

impl ProbeReport {
    pub fn passed(&self) -> bool {
        matches!(self.outcome, ProbeOutcome::Passed)
    }
}

impl Client {
    /// Probes whether this client can serve `selector` right now.
    ///
    /// The probe resolves the route exactly as [`complete`](Self::complete)
    /// does, so it tests the same provider, model, credentials, and headers a
    /// real request would use, and it runs through the same middleware. It
    /// never returns an error: every failure is a finding in the report.
    pub async fn probe(&self, selector: impl Into<String>, options: ProbeOptions) -> ProbeReport {
        self.probe_with_context(selector, options, CallContext::new())
            .await
    }

    /// Probes with an application call context, so the probe can be cancelled
    /// or bounded by an outer deadline.
    ///
    /// The probe's own timeout is added to the context as a deadline; a
    /// sooner deadline already on the context is kept.
    pub async fn probe_with_context(
        &self,
        selector: impl Into<String>,
        options: ProbeOptions,
        mut context: CallContext,
    ) -> ProbeReport {
        let started = Instant::now();
        let selector = selector.into();
        if let Some(deadline) = started.checked_add(options.effective_timeout())
            && context
                .deadline()
                .is_none_or(|existing| deadline < existing)
        {
            context.set_deadline(deadline);
        }
        let mut probe = Probe {
            client: self,
            context,
            route: None,
            usage: TokenCounts::default(),
        };
        let outcome = if options.tools {
            probe.tool_exchange(&selector, &options).await
        } else {
            probe.one_word(&selector, &options).await
        };
        ProbeReport {
            route: probe.route,
            outcome,
            latency: started.elapsed(),
            usage: probe.usage,
        }
    }
}

/// One probe in progress: the route and usage it has seen so far.
struct Probe<'a> {
    client:  &'a Client,
    context: CallContext,
    route:   Option<ModelHandle>,
    usage:   TokenCounts,
}

impl Probe<'_> {
    async fn one_word(&mut self, selector: &str, options: &ProbeOptions) -> ProbeOutcome {
        let request = match build_request(
            selector,
            options,
            vec![Message::text(Role::User, BASIC_PROMPT)],
            false,
        ) {
            Ok(request) => request,
            Err(error) => return ProbeOutcome::Failed(error.data()),
        };
        match self.turn(request).await {
            Ok(_) => ProbeOutcome::Passed,
            Err(outcome) => outcome,
        }
    }

    async fn tool_exchange(&mut self, selector: &str, options: &ProbeOptions) -> ProbeOutcome {
        let mut messages = vec![Message::text(Role::User, TOOL_PROMPT)];
        let mut calls_made = 0_usize;
        for _ in 0..MAX_TOOL_TURNS {
            let request = match build_request(selector, options, messages.clone(), true) {
                Ok(request) => request,
                Err(error) => return ProbeOutcome::Failed(error.data()),
            };
            let response = match self.turn(request).await {
                Ok(response) => response,
                Err(outcome) => return outcome,
            };
            let calls: Vec<&ToolCall> = response
                .content
                .iter()
                .filter_map(|part| match part {
                    ContentPart::ToolCall(call) => Some(call),
                    _ => None,
                })
                .collect();
            if calls.is_empty() {
                return judge_final_answer(&response, calls_made);
            }
            calls_made += calls.len();
            let results: Vec<Message> = calls.into_iter().map(tool_result_message).collect();
            messages.push(Message::new(Role::Assistant, response.content));
            messages.extend(results);
        }
        ProbeOutcome::Incorrect {
            detail: format!("the model was still calling tools after {MAX_TOOL_TURNS} turns"),
        }
    }

    /// Runs one request, recording the route it resolved and the usage it
    /// reported.
    async fn turn(&mut self, request: Request) -> Result<Response, ProbeOutcome> {
        if self.route.is_none()
            && let Ok(route) = self.client.resolve_route(&request)
        {
            self.route = Some(route.handle());
        }
        match self
            .client
            .complete_with_context(request, self.context.clone())
            .await
        {
            Ok(response) => {
                self.usage = add_usage(self.usage, response.usage);
                Ok(response)
            }
            Err(error) => Err(ProbeOutcome::Failed(error.data())),
        }
    }
}

fn judge_final_answer(response: &Response, calls_made: usize) -> ProbeOutcome {
    if calls_made == 0 {
        return ProbeOutcome::Incorrect {
            detail: "the model answered without calling the tool".to_owned(),
        };
    }
    if response.text().contains(TOOL_ANSWER) {
        ProbeOutcome::Passed
    } else {
        ProbeOutcome::Incorrect {
            detail: format!("the model's final answer did not contain the total {TOOL_ANSWER}"),
        }
    }
}

fn build_request(
    selector: &str,
    options: &ProbeOptions,
    messages: Vec<Message>,
    tools: bool,
) -> Result<Request, Error> {
    let mut builder = Request::builder().model(selector);
    for message in messages {
        builder = builder.message(message);
    }
    let expanded = tools || options.reasoning_effort.is_some();
    builder = builder.max_output_tokens(if expanded {
        EXPANDED_OUTPUT_TOKENS
    } else {
        BASIC_OUTPUT_TOKENS
    });
    if let Some(effort) = options.reasoning_effort {
        builder = builder.reasoning_effort(effort);
    }
    if tools {
        builder = builder.tool(add_tool());
    }
    builder.build().map_err(|error| {
        Error::new(
            ErrorKind::InvalidRequest,
            "the probe request could not be built",
        )
        .with_source(error)
    })
}

fn add_tool() -> ToolDefinition {
    ToolDefinition::function(
        TOOL_NAME,
        "Add two integers and return the sum",
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "integer", "description": "First number" },
                "b": { "type": "integer", "description": "Second number" }
            },
            "required": ["a", "b"]
        }),
    )
}

/// Answers one tool call the way the real `add` tool would.
fn tool_result_message(call: &ToolCall) -> Message {
    let (text, is_error) = if call.name == TOOL_NAME {
        let sum = integer(&call.arguments, "a").saturating_add(integer(&call.arguments, "b"));
        (sum.to_string(), false)
    } else {
        (format!("unknown tool {}", call.name), true)
    };
    Message::new(Role::Tool, [ContentPart::ToolResult(ToolResult {
        tool_call_id: call.id.clone(),
        name: Some(call.name.clone()),
        content: vec![ContentPart::Text { text }],
        is_error,
    })])
    .with_tool_call_id(call.id.clone())
}

fn integer(arguments: &Value, key: &str) -> i64 {
    arguments.get(key).and_then(Value::as_i64).unwrap_or(0)
}

fn add_usage(total: TokenCounts, usage: TokenCounts) -> TokenCounts {
    TokenCounts {
        input:       total.input.saturating_add(usage.input),
        output:      total.output.saturating_add(usage.output),
        reasoning:   total.reasoning.saturating_add(usage.reasoning),
        cache_read:  total.cache_read.saturating_add(usage.cache_read),
        cache_write: total.cache_write.saturating_add(usage.cache_write),
    }
}
