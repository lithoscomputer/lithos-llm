# lithos-llm

`lithos-llm` is a provider-neutral Rust library for language model access. It
separates catalog data, model resolution, credentials, middleware, provider
adapters, wire codecs, and HTTP transport.

The optional `cli` feature provides the stateless `lllm` command-line
application. See the [CLI documentation](docs/cli.md) for installation and
usage.

Applications can use built-in OpenAI, Anthropic, Gemini,
OpenAI-compatible, and optional Amazon Bedrock adapters. Applications can also
register their own `ProviderAdapter` implementations.

The library does not execute tools or own an agent loop. It carries tool calls
and tool results between an application and a provider.

## Example

```rust
use std::error::Error;

use lithos_llm::{Client, Request};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let build = Client::from_env()?;
    for issue in &build.issues {
        eprintln!("provider {} is unavailable: {}", issue.provider, issue.cause);
    }
    let client = build.client;

    let request = Request::builder()
        .model("openai/gpt-5.6-luna")
        .user("Why is the sky blue?")
        .build()?;
    let response = client.complete(request).await?;
    assert!(!response.text().is_empty());
    Ok(())
}
```

`Client::from_env()` uses the built-in catalog, the default HTTP client, and
conventional provider environment variables such as `OPENAI_API_KEY`.

Building a client returns a `ClientBuild`, which carries the client together
with one issue for every provider that could not be constructed. A provider
that fails does not remove the providers that succeeded, so an application can
start in a degraded state and report why. Client construction never reads
credentials.

Applications own the Tokio runtime and tracing subscriber. Credential lookup
runs for each provider attempt, so tokens can refresh without rebuilding the
client. Retry and tracing middleware remain opt-in.

`RetryMiddleware::observer` reports each retried attempt to an `Observer`
through `on_retry`, with the failed attempt, its error, the wait before the
next one, and a `RetryStage` naming what failed: a request that never became a
stream, or a stream that failed before it delivered visible output. The retry
layer retries only until a stream delivers visible output; an application that
must replay a turn after that point can drive its own loop with
`RetryPolicy::next_delay`, the same decision the middleware uses.

An `Incomplete` response also retries while no content has reached the caller.
Closing blocks that precede any content are held until the terminal outcome.
When the attempt or deadline budget prevents a retry, the partial response is
returned with its original finish reason. `Length` does not trigger a retry.

Use `ClientBuilder::http` to inject an application-configured
`reqwest::Client`, and `ClientBuilder::enabled_providers` to build adapters for
only the providers a deployment has configured. The complete catalog stays
available for inspection either way.

`Client::resolve_route` reports the provider and model a request would use
without dispatching it, which is the same resolution `complete`, `stream`, and
`count_input_tokens` perform.

## Call controls

`Request::into_builder()` preserves settings while changing a request.
Deserialization runs the same validation as the builder.

`ClientBuilder::default_timeout(Duration)` sets a total call budget, including
middleware, credentials, retries, and stream consumption. A request timeout
replaces the default; an earlier context deadline still wins. All three
operations have a `*_with_context` method and pass through middleware.

`ClientBuilder::response_limits(ResponseLimits)` controls HTTP body, stream
frame, and output sizes. Defaults are 32 MiB, 4 MiB, and 32 MiB. Output limits
also bound cumulative provider-event data before decoding. Exceeding a limit
returns a non-retryable `ErrorKind::ResourceLimit`.
`retain_raw_response(false)` drops raw success bodies. It does not redact
normalized content, replay metadata, or errors.

Middleware reads `Call::request()`, `route()`, `operation()`, and `context()`.
Use `map_request` to change the payload without changing its resolved model.
`Observer::on_finish` reports one final outcome per observed invocation,
including cancellation and dropped futures or streams.

Capability queries return `Support::Supported`, `Unsupported`, or `Unknown`.
Use `tool_choice`, `response_format`, `reasoning_effort`, and `speed` to check
a specific setting. Unknown support does not reject a request locally.
Protocol flags live in `CatalogModel::protocol_options()`. Pricing does not
determine capability support.

See the [API migration notes](docs/api-migration.md) for breaking changes and
canonical stored formats. Applications own conversion of historical records.

## Streaming

Streaming uses the same request type and model resolution as `complete`. Add
`futures-util` to the application to use `StreamExt`.

```rust
use std::error::Error;
use std::io::{self, Write as _};

use futures_util::StreamExt as _;
use lithos_llm::types::StreamEvent;
use lithos_llm::{Client, Request};

async fn stream_answer(client: &Client) -> Result<(), Box<dyn Error>> {
    let request = Request::builder()
        .model("openai/gpt-5.6-luna")
        .user("Why is the sky blue?")
        .build()?;
    let mut stream = client.stream(request).await?;

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::TextDelta { text, .. } => {
                print!("{text}");
                io::stdout().flush()?;
            }
            StreamEvent::Ended { response } => {
                eprintln!("\n{} tokens", response.usage.total());
            }
            _ => {}
        }
    }
    Ok(())
}
```

Every successful stream ends with one `StreamEvent::Ended`. It contains
the assembled `Response`. An ended stream can carry an incomplete answer;
inspect `response.finish_reason` before accepting the turn or executing tools.
Streams can also carry reasoning, tool-call, usage,
and rate-limit events. Dropping the stream cancels the operation as far as the
provider and transport permit.

Use `ResponseStream::new` for custom adapter streams. Completion or an error
releases the inner stream immediately. Ending without a terminal event is an
error. Block-end content is provisional; the final response is authoritative.

## Tool calling

`ToolCall::input` is either `ToolInput::Function(ToolArguments)` or
`ToolInput::Custom(String)`. Function arguments keep their raw text and parsed
result together. `ToolArguments::json()` returns an error for malformed JSON;
it does not replace malformed input with an empty object.

Lithos carries tool definitions, calls, and results. The application executes
each tool and decides whether to make another model request.

```rust
use std::error::Error;

use lithos_llm::types::{
    ContentPart, Message, Role, ToolDefinition, ToolResult,
};
use lithos_llm::{Client, Request};
use serde_json::json;

async fn answer_with_weather(client: &Client) -> Result<String, Box<dyn Error>> {
    let model = "openai/gpt-5.6-luna";
    let prompt = "What is the weather in Paris?";
    let tool = ToolDefinition::function(
        "get_weather",
        "Reads the current weather for a city",
        json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
            "additionalProperties": false,
        }),
    );

    let request = Request::builder()
        .model(model)
        .user(prompt)
        .tool(tool.clone())
        .build()?;
    let response = client.complete(request).await?;
    let Some(call) = response.tool_calls().next() else {
        return Err("the model did not call get_weather".into());
    };

    let result = ToolResult {
        tool_call_id: call.id.clone(),
        name: Some(call.name.clone()),
        content: vec![ContentPart::Text {
            text: "18 C and clear".to_owned(),
        }],
        is_error: false,
    };
    let follow_up = Request::builder()
        .model(model)
        .user(prompt)
        .tool(tool)
        .message(response.into_message())
        .message(Message::new(Role::Tool, [ContentPart::ToolResult(result)]))
        .build()?;

    Ok(client.complete(follow_up).await?.text())
}
```

A turn cut short by the output limit (`Length`) or an incomplete stream
(`Incomplete`) exposes no calls through `tool_calls()`. Withheld calls remain
in `suppressed_tool_calls` for diagnostics, with their raw arguments and replay
metadata. Do not execute these calls. A `truncated_tool_call` warning names each
withheld tool. Inspect the finish reason to distinguish a finished turn.

Replay the complete assistant content in the follow-up request. It can contain
provider data that a later request needs. One response can contain more than
one `ToolCall`; applications must return a result for each call they execute.

## Structured output

Use `ResponseFormat` to request a JSON object or a document that follows a JSON
Schema.

```rust
use std::error::Error;

use lithos_llm::types::ResponseFormat;
use lithos_llm::{Client, Request};
use serde_json::{Value, json};

async fn extract_city(client: &Client) -> Result<Value, Box<dyn Error>> {
    let request = Request::builder()
        .model("openai/gpt-5.6-luna")
        .user("Extract the city and country from: I live in Paris, France.")
        .response_format(ResponseFormat::JsonSchema {
            name: "location".to_owned(),
            schema: json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" },
                    "country": { "type": "string" },
                },
                "required": ["city", "country"],
                "additionalProperties": false,
            }),
        })
        .build()?;
    let response = client.complete(request).await?;
    Ok(serde_json::from_str(&response.text())?)
}
```

The built-in adapters return the JSON document as response text. The
application parses it into `serde_json::Value` or its own type. Lithos rejects
structured-output requests before dispatch when the selected model does not
declare that capability.

## Multimodal input

A message can contain text, images, audio, and documents. Media can use a
provider-accessible URL or inline base64 data.

```rust
use std::error::Error;

use lithos_llm::types::{ContentPart, ImageContent, MediaSource, Message, Role};
use lithos_llm::{Client, Request};

async fn describe_image(client: &Client) -> Result<String, Box<dyn Error>> {
    let request = Request::builder()
        .model("openai/gpt-5.6-luna")
        .message(Message::new(Role::User, [
            ContentPart::Text {
                text: "Describe this image.".to_owned(),
            },
            ContentPart::Image(ImageContent::new(MediaSource::url(
                "https://example.com/image.png",
            ))),
        ]))
        .build()?;

    Ok(client.complete(request).await?.text())
}
```

Use `MediaSource::base64(data, media_type)` for inline bytes. `AudioContent`
and `DocumentContent` use the same source type. Provider URL rules differ.

## Model probes

`Client::probe` answers "can this client serve this model right now" with a
report instead of an error:

The optional CLI exposes the same diagnostic:

```sh
lllm probe --model anthropic/claude-sonnet-5
lllm probe --model-query sonnet --tools --json
```

```rust
use lithos_llm::Client;
use lithos_llm::client::{ProbeOptions, ProbeOutcome};

async fn check(client: &Client) {
    let report = client
        .probe("anthropic/claude-sonnet-5", ProbeOptions::new().tools(true))
        .await;
    match &report.outcome {
        ProbeOutcome::Passed => println!("ok in {:?}", report.latency),
        ProbeOutcome::Failed(data) => println!("{:?}: {}", data.kind, data.message),
        ProbeOutcome::Incorrect { detail } => println!("answered, but {detail}"),
    }
}
```

The probe resolves the route, credentials, and middleware exactly as
`complete` does. The default probe sends one short prompt; `tools(true)` runs
an `add`-tool exchange and checks the model calls the tool and reaches the
right total. A catalog rejection — a tool probe against a model that declares
no tools — fails before any request is sent. `Failed` carries the classified
`ErrorData`, so an unknown model, bad credentials, a missing model, and a
timeout are told apart by its `kind`.

## Token estimation

`lithos_llm::estimate` sizes a request locally, before any call is made:

```rust
use lithos_llm::Request;
use lithos_llm::estimate::{EstimateWarning, message_tokens, request_tokens};

fn budget(request: &Request) -> u64 {
    let estimate = request_tokens(request);
    if estimate.has_warning(EstimateWarning::Media) {
        // Media is sized from bytes or given a floor; the total is rough.
    }
    let largest = request
        .messages()
        .iter()
        .map(|message| message_tokens(message).tokens())
        .max()
        .unwrap_or(0);
    estimate.tokens().max(largest)
}
```

Estimates are deterministic heuristics with no provider call and no feature
flag. Each `TokenEstimate` carries typed warnings for the inputs it could only
approximate — media, opaque provider content, and provider options — and
estimates add together. `Client::count_input_tokens` is the authoritative path
where a provider offers a native count; the client never substitutes an
estimate for it.
Lithos rejects media that the selected model does not support before it sends
the request.

## Catalog overlays

Catalog documents use schema version `1`. Later overlays win. Tables merge
recursively. Arrays and scalar values replace earlier values.

```rust
use std::error::Error;

use lithos_llm::catalog::Catalog;

fn main() -> Result<(), Box<dyn Error>> {
    let catalog = Catalog::builder()
        .with_builtin()
        .toml_layer(
            "example_app.toml",
            r#"
        [providers.openai.models."gpt-5.6-luna".metadata.example_app]
        latency_class = "fast"
        "#,
        )?
        .build()?;
    assert_eq!(catalog.schema_version(), 1);
    Ok(())
}
```

Name each layer with `toml_layer`. The name appears in parse and validation
errors, which matters when a catalog is assembled from several fragments.

Unknown core provider and model fields are rejected. Application extensions
belong under a namespaced `metadata` table, which Lithos preserves without
interpreting.

An application can supply its whole catalog and use none of the built-in
entries. Building from external TOML never adds built-in providers implicitly;
only `with_builtin()` does that.

```rust
use std::error::Error;

use lithos_llm::catalog::Catalog;

fn build(root: &str, openai: &str, anthropic: &str) -> Result<(), Box<dyn Error>> {
    let catalog = Catalog::builder()
        .toml_layer("catalog.toml", root)?
        .toml_layer("providers/openai.toml", openai)?
        .toml_layer("providers/anthropic.toml", anthropic)?
        .build()?;
    assert_eq!(catalog.schema_version(), 1);
    Ok(())
}
```

## Features

| Feature | Default | Purpose |
| --- | --- | --- |
| `builtin-catalog` | yes | Embedded provider and model catalog |
| `runtime` | through provider features | Client and public runtime extension points |
| `openai` | yes | OpenAI Responses adapter |
| `anthropic` | yes | Anthropic Messages adapter |
| `gemini` | yes | Gemini Generate Content adapter |
| `openai-compatible` | yes | Chat Completions-compatible adapter |
| `environment-credentials` | yes | Environment-backed credential provider |
| `bedrock` | no | Bedrock Converse adapter with bearer-token authentication |
| `bedrock-aws` | no | AWS credential chain and SigV4 signing for Bedrock |

Bedrock uses one HTTP transport for both authentication paths. `bedrock` alone
adds no AWS crates; `bedrock-aws` adds the AWS credential chain and SigV4.

A catalog-only consumer can avoid Tokio, reqwest, and provider SDKs:

```toml
lithos-llm = { version = "0.1", default-features = false, features = ["builtin-catalog"] }
```

A custom adapter consumer can enable `runtime` without a built-in provider.

## Setup

Install the locked tools and prepare the repository:

```sh
mise trust
mise install --locked
mise run setup
```

Use the repository tasks for development and verification:

```sh
mise run dev
mise run test
mise run check
```

See [DEVELOPING.md](DEVELOPING.md) for the complete development workflow.
