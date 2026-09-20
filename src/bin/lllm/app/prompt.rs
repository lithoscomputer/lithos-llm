use std::io::{Read, Write};
use std::time::Instant;

use futures_util::StreamExt as _;
use lithos_llm::Client;
use lithos_llm::catalog::ProviderId;
use lithos_llm::middleware::{CallContext, CancellationToken};
use lithos_llm::types::{
    CacheHint, ContentPart, ErrorKind, Message, Request, RequestBuilder, Response, ResponseFormat,
    Role, StreamEvent,
};
use serde_json::Value;

use crate::app::args::{AttachmentArg, PromptArgs, read_multi_schema, read_schema};
use crate::app::output::{self, DeltaWriter, write_text};
use crate::app::{
    CliEnvironment, CliError, CliResult, OutputState, TerminalState, cancellable, input, models,
    usage,
};

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
    let input = input::prepare(
        &args.prompt,
        &args.fragments,
        attachments,
        terminal.stdin,
        stdin,
    )?;
    let system = input::system_text(args.system.as_deref(), &args.system_fragments)?;
    let model = models::select_model(
        client,
        args.model.as_deref(),
        &args.model_query,
        environment,
    )?;
    // The route is resolved from a stand-in first — the resolver reads only
    // the model selector — so the `--option` pairs can attach to the resolved
    // provider and the real request is built exactly once.
    let route = client
        .resolve_route(
            &Request::builder()
                .model(&model)
                .user("resolve")
                .build()
                .map_err(|source| CliError::input_source("request is invalid", source))?,
        )
        .map_err(|error| models::selection_error(client, &model, error))?;
    let request = build_request(
        args,
        &model,
        &input.parts,
        system.as_deref(),
        response_format(args)?,
        route.provider().id(),
    )?;
    let session = PromptSession {
        client,
        args,
        route: route.handle().to_string(),
        stdout,
        stderr,
        cancellation,
        started: Instant::now(),
    };
    let buffered = args.json || args.extract || args.extract_last;
    if !args.no_stream && !buffered {
        session.stream(request).await
    } else {
        session.complete(request).await
    }
}

/// The values every step of one prompt shares.
///
/// `probe` and `models` each have one entry `run`; this is the prompt
/// command's, with its ambient state named once instead of threaded through
/// every free function.
struct PromptSession<'a, W, E> {
    client:       &'a Client,
    args:         &'a PromptArgs,
    /// The resolved route, as diagnostics name it.
    route:        String,
    stdout:       &'a mut W,
    stderr:       &'a mut E,
    cancellation: &'a CancellationToken,
    started:      Instant,
}

impl<W: Write, E: Write> PromptSession<'_, W, E> {
    /// Wraps a library failure with the route it happened on.
    fn call_error(&self, source: lithos_llm::Error) -> CliError {
        CliError::Call {
            route:  self.route.clone(),
            source: Box::new(source),
        }
    }

    /// Runs the request to completion and prints the buffered text.
    async fn complete(mut self, request: Request) -> CliResult<OutputState> {
        let response = cancellable(
            self.cancellation,
            self.client
                .complete_with_context(request.clone(), CallContext::new()),
        )
        .await?
        .map_err(|source| self.call_error(source))?;
        let text = self.buffered_text(&request, &response)?;
        let state = write_text(self.stdout, &text)?;
        self.print_usage(&response)?;
        Ok(state)
    }

    /// Streams text deltas as they arrive, then finishes the response.
    ///
    /// A failure after deltas were written ends the open line first, so the
    /// diagnostic on standard error does not start mid-line and a piped
    /// stdout ends on a newline.
    async fn stream(self, request: Request) -> CliResult<OutputState> {
        let mut stream = cancellable(
            self.cancellation,
            self.client.stream_with_context(request, CallContext::new()),
        )
        .await?
        .map_err(|source| self.call_error(source))?;
        let mut deltas = DeltaWriter::default();
        let mut response = None;
        while let Some(item) = cancellable(self.cancellation, stream.next()).await? {
            match item {
                Ok(StreamEvent::TextDelta { text, .. }) => {
                    if deltas.write(self.stdout, &text)? == OutputState::Closed {
                        return Ok(OutputState::Closed);
                    }
                }
                Ok(StreamEvent::Ended {
                    response: completed,
                }) => response = Some(completed),
                Ok(_) => {}
                Err(source) => {
                    if deltas.finish_line(self.stdout)? == OutputState::Closed {
                        return Ok(OutputState::Closed);
                    }
                    return Err(self.call_error(source));
                }
            }
        }
        let response = response.ok_or_else(|| {
            self.call_error(lithos_llm::Error::new(
                ErrorKind::ResponseDecode,
                "the response stream ended without a completed response",
            ))
        })?;
        self.finish(&response, &mut deltas)
    }

    /// Writes what the stream did not: the whole text when nothing streamed,
    /// the closing newline otherwise, and the usage line.
    fn finish(mut self, response: &Response, deltas: &mut DeltaWriter) -> CliResult<OutputState> {
        let state = if deltas.wrote_anything() {
            deltas.finish_line(self.stdout)?
        } else {
            write_text(self.stdout, &output::response_text(response)?)?
        };
        self.print_usage(response)?;
        Ok(state)
    }

    fn print_usage(&mut self, response: &Response) -> CliResult<()> {
        if self.args.usage {
            usage::write(self.stderr, response, self.started.elapsed())?;
        }
        Ok(())
    }

    /// The text a non-streaming run prints: the JSON envelope, an extracted
    /// code block, or the response text.
    fn buffered_text(&self, request: &Request, response: &Response) -> CliResult<String> {
        if self.args.json {
            return output::envelope(request, response);
        }
        let text = output::response_text(response)?;
        if self.args.extract {
            Ok(output::extract_first(&text).to_owned())
        } else if self.args.extract_last {
            Ok(output::extract_last(&text).to_owned())
        } else {
            Ok(text)
        }
    }
}

/// The response format the flags ask for.
///
/// clap's `structured_schema` group already guarantees at most one of the
/// schema flags, so this is a plain three-way choice.
fn response_format(args: &PromptArgs) -> CliResult<Option<ResponseFormat>> {
    let name = || {
        args.schema_name
            .clone()
            .unwrap_or_else(|| "response".to_owned())
    };
    if args.json_object {
        Ok(Some(ResponseFormat::JsonObject))
    } else if let Some(schema) = &args.schema {
        Ok(Some(ResponseFormat::JsonSchema {
            name:   name(),
            schema: read_schema(schema)?,
        }))
    } else if let Some(schema) = &args.schema_multi {
        Ok(Some(ResponseFormat::JsonSchema {
            name:   name(),
            schema: read_multi_schema(schema)?,
        }))
    } else {
        Ok(None)
    }
}

fn build_request(
    args: &PromptArgs,
    model: &str,
    parts: &[ContentPart],
    system: Option<&str>,
    format: Option<ResponseFormat>,
    provider: &ProviderId,
) -> CliResult<Request> {
    let mut builder = Request::builder().model(model);
    if let Some(system) = system {
        builder = builder.system(system);
    }
    builder = builder.message(Message::new(Role::User, parts.iter().cloned()));
    builder = apply_controls(builder, args, format, provider);
    builder
        .build()
        .map_err(|source| CliError::input_source("request is invalid", source))
}

fn apply_controls(
    mut builder: RequestBuilder,
    args: &PromptArgs,
    format: Option<ResponseFormat>,
    provider: &ProviderId,
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
        builder = builder.reasoning_effort(effort.into());
    }
    if let Some(speed) = args.speed {
        builder = builder.speed(speed.into());
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
    for pair in &args.options {
        let value =
            serde_json::from_str(&pair.value).unwrap_or_else(|_| Value::String(pair.value.clone()));
        builder = builder.provider_option(provider.clone(), &pair.key, value);
    }
    builder
}
