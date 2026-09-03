use std::io::{Read, Write};
use std::time::Instant;

use futures_util::StreamExt as _;
use lithos_llm::Client;
use lithos_llm::catalog::ProviderId;
use lithos_llm::middleware::{CallContext, CancellationToken};
use lithos_llm::types::{
    CacheHint, ContentPart, ErrorKind, Message, ReasoningEffort, Request, RequestBuilder, Response,
    ResponseFormat, Role, Speed, StreamEvent,
};
use serde_json::Value;

use crate::app::args::{AttachmentArg, PromptArgs, ReasoningEffortArg, SpeedArg, read_schema};
use crate::app::output::{self, write_delta, write_text};
use crate::app::{CliEnvironment, CliError, CliResult, OutputState, TerminalState, input, models};

pub(crate) async fn run(
    client: &Client,
    args: &PromptArgs,
    attachments: &[AttachmentArg],
    terminal: TerminalState,
    stdin: &mut impl Read,
    stdout: &mut impl Write,
    stderr: &mut impl Write,
    environment: &CliEnvironment,
    cancellation: &CancellationToken,
) -> CliResult<OutputState> {
    let input = input::prepare(&args.prompt, attachments, terminal.stdin, stdin)?;
    let model = models::select_model(
        client,
        args.model.as_deref(),
        &args.model_query,
        environment,
    )?;
    let format = response_format(args)?;
    let (request, route) = request(client, args, &model, &input.parts, format)?;
    let buffered = args.json || args.extract || args.extract_last;
    if !args.no_stream && !buffered {
        stream(
            client,
            request,
            &route,
            stdout,
            stderr,
            args.usage,
            cancellation,
        )
        .await
    } else {
        complete(client, request, &route, args, stdout, stderr, cancellation).await
    }
}

fn response_format(args: &PromptArgs) -> CliResult<Option<ResponseFormat>> {
    if args.json_object {
        Ok(Some(ResponseFormat::JsonObject))
    } else if let Some(schema) = &args.schema {
        Ok(Some(ResponseFormat::JsonSchema {
            name:   args
                .schema_name
                .clone()
                .unwrap_or_else(|| "response".to_owned()),
            schema: read_schema(schema)?,
        }))
    } else {
        Ok(None)
    }
}

fn request(
    client: &Client,
    args: &PromptArgs,
    model: &str,
    parts: &[ContentPart],
    format: Option<ResponseFormat>,
) -> CliResult<(Request, String)> {
    let initial = build_request(args, model, parts, format.clone(), None)?;
    let route = client
        .resolve_route(&initial)
        .map_err(|error| models::selection_error(client, model, error))?;
    let route_name = route.handle().to_string();
    let request = if args.options.is_empty() {
        initial
    } else {
        build_request(args, model, parts, format, Some(route.provider().id()))?
    };
    Ok((request, route_name))
}

fn build_request(
    args: &PromptArgs,
    model: &str,
    parts: &[ContentPart],
    format: Option<ResponseFormat>,
    option_provider: Option<&ProviderId>,
) -> CliResult<Request> {
    let mut builder = Request::builder().model(model);
    if let Some(system) = &args.system {
        builder = builder.system(system);
    }
    builder = builder.message(Message::new(Role::User, parts.iter().cloned()));
    builder = apply_controls(builder, args, format, option_provider);
    builder
        .build()
        .map_err(|source| CliError::input_source("request is invalid", source))
}

fn apply_controls(
    mut builder: RequestBuilder,
    args: &PromptArgs,
    format: Option<ResponseFormat>,
    option_provider: Option<&ProviderId>,
) -> RequestBuilder {
    if let Some(tokens) = args.max_output_tokens {
        builder = builder.max_output_tokens(tokens);
    }
    if let Some(temperature) = args.temperature {
        builder = builder.temperature(temperature);
    }
    if let Some(top_p) = args.top_p {
        builder = builder.top_p(top_p);
    }
    if let Some(effort) = args.reasoning_effort {
        builder = builder.reasoning_effort(reasoning_effort(effort));
    }
    if let Some(speed) = args.speed {
        builder = builder.speed(speed_value(speed));
    }
    if let Some(timeout) = args.timeout {
        builder = builder.timeout(timeout);
    }
    builder = builder.stop_sequences(args.stop.iter().cloned());
    if let Some(key) = &args.cache_key {
        builder = builder.cache_key(key);
    } else if args.no_cache {
        builder = builder.cache_hint(CacheHint::Disabled);
    }
    for pair in &args.metadata {
        builder = builder.metadata_entry(&pair.key, &pair.value);
    }
    if let Some(format) = format {
        builder = builder.response_format(format);
    }
    if let Some(provider) = option_provider {
        for pair in &args.options {
            let value = serde_json::from_str(&pair.value)
                .unwrap_or_else(|_| Value::String(pair.value.clone()));
            builder = builder.provider_option(provider.clone(), &pair.key, value);
        }
    }
    builder
}

async fn complete(
    client: &Client,
    request: Request,
    route: &str,
    args: &PromptArgs,
    output_writer: &mut impl Write,
    error_writer: &mut impl Write,
    cancellation: &CancellationToken,
) -> CliResult<OutputState> {
    let started = Instant::now();
    let result = tokio::select! {
        biased;
        () = cancellation.cancelled() => return Err(CliError::Interrupted),
        result = client.complete_with_context(request.clone(), CallContext::new()) => result,
    };
    let response = result.map_err(|source| CliError::Call {
        route: route.to_owned(),
        source,
    })?;
    let text = buffered_text(&request, &response, args)?;
    let state = write_text(output_writer, &text)?;
    if args.usage {
        crate::app::usage::write(error_writer, &response, started.elapsed())?;
    }
    Ok(state)
}

fn buffered_text(request: &Request, response: &Response, args: &PromptArgs) -> CliResult<String> {
    if args.json {
        output::envelope(request, response)
    } else {
        let text = output::response_text(response)?;
        if args.extract {
            Ok(output::extract_first(&text).to_owned())
        } else if args.extract_last {
            Ok(output::extract_last(&text).to_owned())
        } else {
            Ok(text)
        }
    }
}

async fn stream(
    client: &Client,
    request: Request,
    route: &str,
    output_writer: &mut impl Write,
    error_writer: &mut impl Write,
    show_usage: bool,
    cancellation: &CancellationToken,
) -> CliResult<OutputState> {
    let started = Instant::now();
    let result = tokio::select! {
        biased;
        () = cancellation.cancelled() => return Err(CliError::Interrupted),
        result = client.stream_with_context(request, CallContext::new()) => result,
    };
    let mut stream = result.map_err(|source| CliError::Call {
        route: route.to_owned(),
        source,
    })?;
    let mut response = None;
    let mut wrote_delta = false;
    let mut ends_with_newline = false;
    loop {
        let item = tokio::select! {
            biased;
            () = cancellation.cancelled() => return Err(CliError::Interrupted),
            item = stream.next() => item,
        };
        let Some(item) = item else {
            break;
        };
        match item.map_err(|source| CliError::Call {
            route: route.to_owned(),
            source,
        })? {
            StreamEvent::TextDelta { text, .. } => {
                wrote_delta = true;
                if !text.is_empty() {
                    ends_with_newline = text.ends_with('\n');
                }
                if write_delta(output_writer, &text)? == OutputState::Closed {
                    return Ok(OutputState::Closed);
                }
            }
            StreamEvent::Completed {
                response: completed,
            } => response = Some(completed),
            _ => {}
        }
    }
    let response = response.ok_or_else(|| CliError::Call {
        route:  route.to_owned(),
        source: lithos_llm::Error::new(
            ErrorKind::ResponseDecode,
            "the response stream ended without a completed response",
        ),
    })?;
    let state = if wrote_delta {
        if ends_with_newline {
            OutputState::Written
        } else {
            write_delta(output_writer, "\n")?
        }
    } else {
        write_text(output_writer, &output::response_text(&response)?)?
    };
    if show_usage {
        crate::app::usage::write(error_writer, &response, started.elapsed())?;
    }
    Ok(state)
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

const fn speed_value(value: SpeedArg) -> Speed {
    match value {
        SpeedArg::Fast => Speed::Fast,
        SpeedArg::Balanced => Speed::Balanced,
        SpeedArg::Economical => Speed::Economical,
    }
}
