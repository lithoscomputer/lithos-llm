# Library guide

Detailed examples and behavior for applications using `lithos-llm`.
Start with the [README](https://github.com/lithoscomputer/lithos-llm/blob/main/README.md) for installation.

The library separates catalog data, model resolution, credentials, middleware,
provider adapters, wire codecs, and HTTP transport. An adapter owns transport
and signing; a codec translates one provider wire protocol.

The built-in `http` adapter supports OpenAI Chat Completions, OpenAI Responses,
Anthropic Messages, Gemini, Vercel evaluation, and TypeSafe System One.
The optional `bedrock` adapter supports Converse. Applications can register
custom `ProviderAdapter` implementations.

## Build a client


```rust,no_run
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
conventional provider environment variables such as `OPENAI_API_KEY`. The
built-in catalog covers the first-party APIs (Anthropic, OpenAI, Gemini,
Bedrock, and TypeSafe AI with `TYPESAFE_API_KEY`) and the routers that
front many models behind one key: OpenRouter (`OPENROUTER_API_KEY`), Venice
(`VENICE_API_KEY`), and Vercel AI Gateway (`AI_GATEWAY_API_KEY`).

`ConventionalCredentials` is the table behind that: which named secrets each
provider expects and how they shape into its authentication. Where a named
secret lives is the application's decision. The default reads the process
environment; `with_lookup` reads the same names from any other store, and
`ChainedCredentials` tries several stores in order:

```rust
use lithos_llm::credentials::{ChainedCredentials, ConventionalCredentials};

fn main() {
    let secrets = std::collections::HashMap::<String, String>::new();
    let _credentials = ChainedCredentials::new()
        .then(ConventionalCredentials::new().with_lookup(move |name| secrets.get(name).cloned()))
        .then(ConventionalCredentials::new());
}
```

`ConventionalCredentials::secret_names` lists the names a provider reads, so
an install flow can tell an operator what to set and write under the first. A
provider the table does not list reads a name derived from its id: a catalog
overlay that adds `acme` with bearer auth is served by `ACME_API_KEY`.

`catalog::builtin::ids` names every provider the built-in catalog ships, so an
install flow or a login command refers to `ids::OPENAI` rather than a string.

The catalog also includes provisional imports for DeepSeek, Inception, MiniMax,
Z.ai, Poolside, LiteLLM, Ollama, and Bedrock OpenAI. Their credential names,
configuration notes, and pending live checks are in
[provider live-test TODOs](https://github.com/lithoscomputer/lithos-llm/blob/main/docs/provider-live-tests.md).

Modal deployment templates live in [docs/catalogs](https://github.com/lithoscomputer/lithos-llm/tree/main/docs/catalogs). GPT-OSS
models are excluded from the built-in catalog.

Building a client returns a `ClientBuild`, which carries the client together
with one issue for every provider that could not be constructed. A provider
that fails does not remove the providers that succeeded, so an application can
start in a degraded state and report why. Client construction never reads
credentials.

`ClientBuilder::build_ready` is the alternative for applications that want to
know up front which providers they can serve: it resolves credentials once for
every selected provider and builds adapters only for the ones that resolved.
Providers the store holds nothing for are left out silently; providers whose
stored material cannot be used (an expired token with no refresh token, an
entry of the wrong type) come back in `ClientBuild::credential_issues` with an
operator-facing reason. `credentials::readiness` answers the same question
without building a client, for listings and diagnostics.

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

`ClientBuilder::application` names the calling application to providers that
expect a client to identify itself; the OpenAI Codex deployment reads it from
an `originator` header. Use `ClientBuilder::http` to inject an
application-configured `reqwest::Client`, and `ClientBuilder::enabled_providers`
to build adapters for only the providers a deployment has configured. That selection narrows the
catalog; a provider the catalog marks `enabled = false` builds no adapter
either way, and a catalog overlay is what turns it on. The complete catalog
stays available for inspection.

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
a specific setting; `closest_supported_effort` picks the nearest supported
reasoning level when a request written for one model goes to another. Unknown
support does not reject a request locally.
Protocol flags live in `CatalogModel::protocol_options()`. Pricing does not
determine capability support.

See the [API migration notes](https://github.com/lithoscomputer/lithos-llm/blob/main/docs/api-migration.md) for breaking changes and
canonical stored formats. Applications own conversion of historical records.

## Streaming

Streaming uses the same request type and model resolution as `complete`. Add
`futures-util` to the application to use `StreamExt`.

```rust,no_run
use std::error::Error;
use std::io::{self, Write as _};

use futures_util::StreamExt as _;
use lithos_llm::types::StreamEvent;
use lithos_llm::{Client, Request};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let client = Client::from_env()?.client;
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

## Readable reasoning

Providers return readable reasoning through unrelated channels: Anthropic and
Gemini as `ReasoningContent` blocks, the OpenAI Responses protocol as
`reasoning` items with a summary and sometimes a verbatim trace, and
OpenAI-compatible gateways as `reasoning_details`. `Response::reasoning()`
reduces whichever arrived to one `ReasoningOutput` with a `summary` and a
`trace`, at least one of them present, without reading signatures, item ids, or
encrypted payloads. Keep the original parts for the next request:
`ContentPart::is_replay_material()` marks every reasoning and opaque part a
conversation replays, and `is_opaque_openai()` marks the OpenAI items that stop
being valid once compaction replaces the turn they belong to.

## Tool calling

`ToolCall::input` is either `ToolInput::Function(ToolArguments)` or
`ToolInput::Custom(String)`. Function arguments keep their raw text and parsed
result together. `ToolArguments::json()` returns an error for malformed JSON;
it does not replace malformed input with an empty object.

Lithos carries tool definitions, calls, and results. The application executes
each tool and decides whether to make another model request.

```rust,no_run
use std::error::Error;

use lithos_llm::types::{
    ContentPart, Message, Role, ToolDefinition, ToolResult,
};
use lithos_llm::{Client, Request};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let client = Client::from_env()?.client;
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

    println!("{}", client.complete(follow_up).await?.text());
    Ok(())
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

```rust,no_run
use std::error::Error;

use lithos_llm::types::ResponseFormat;
use lithos_llm::{Client, Request};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let client = Client::from_env()?.client;
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
    let location: serde_json::Value = serde_json::from_str(&response.text())?;
    println!("{location}");
    Ok(())
}
```

The built-in adapters return the JSON document as response text. The
application parses it into `serde_json::Value` or its own type, or lets
`Response::json_object` do it: a `Json` part is taken as is, otherwise the text
is parsed. `Client::complete_object(request, "location", schema)` does the
whole exchange, attaching the schema and returning a `StructuredCompletion`
with the response and the parsed document. Lithos rejects structured-output
requests before dispatch when the selected model does not declare that
capability.

## Evaluate

An evaluation asks typed questions about one piece of state. Each answer is
a chosen option, a position on an ordered scale, or a probability that a
statement holds, and every answer comes back as its own type, not as prose
to parse. `Evaluation` is to `Client::evaluate` what `Request` is to
`complete`: the whole input, built once and checked on `build`.

```rust,no_run
use std::error::Error;

use lithos_llm::{Client, Evaluation};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let client = Client::from_env()?.client;
    let evaluation = Evaluation::builder()
        .model("vercel/jev")
        .state("I was charged twice. Please refund the duplicate.")
        .choice("department", "Which team should handle this?", [
            ("billing", Some("Charges and refunds")),
            ("technical", Some("Bugs and outages")),
            ("other", None),
        ])
        .score("severity", "How severe is the issue?", [
            "Cosmetic",
            "Workaround exists",
            "Blocking; no workaround",
        ])
        .boolean("requests_refund", "Is the customer requesting money back?")
        .build()?;

    let verdict = client.evaluate(evaluation).await?;

    let department = verdict.choice("department")?;
    let severity = verdict.score("severity")?;
    let refund = verdict.boolean("requests_refund")?;

    let needs_refund = department.choice == "billing"
        && severity.nearest_level() >= 1
        && refund.is_likely();
    println!("Needs refund review: {needs_refund}");
    Ok(())
}
```

Two kinds of model answer the same `Evaluation`. A row that reaches an
evaluation codec (today `vercel/jev`, TypeSafe's Jev through the
`vercel-evaluation` codec) answers natively and returns a probability per
option or level, a `confidence` on each choice and score answer, and the
provider's declared rounding. `typesafe/jev-latest` evaluates natively too,
through TypeSafe's own API on the `systemone` codec, and names the
versioned model that answered in `Verdict::served_by`. `openrouter/jev-latest`
and `openrouter/jev-1.13` reach the same model through OpenRouter's Decisions
API on the `systemone` codec's OpenRouter dialect. Every other row that claims
`response_format.json_schema` acts as a judge: the client runs one
structured-output completion with a fixed system prompt and reads the JSON
object back into answers. On a provider served by a built-in adapter the
row's codec set makes that choice; on a provider served by an adapter the
application registered, the adapter's own `evaluates_natively()` does. A
judge returns point estimates, so `probabilities`, `confidence`, and
`rounding` are absent on its `Verdict`. The same `Evaluation` and `Verdict`
types serve both, so an application can run one question set against a
native model and a judge and compare them.

The client validates every verdict before it returns it: exactly the
questions asked are answered, each with the matching kind, choices name a
known option, scores and probabilities are in range, and distributions sum to
one within the declared rounding. A malformed verdict is a `ResponseDecode`
error and is never retried. The judge's system prompt is part of the crate's
contract, because changing it changes answers; any change to it is called out
in the changelog. See [Catalog overlays](https://github.com/lithoscomputer/lithos-llm/blob/main/docs/catalog.md) for the
`evaluation` claim and how a row's codecs follow from it.

## Multimodal input

A message can contain text, images, audio, and documents. Media can use a
provider-accessible URL or inline base64 data. With the `local-files` feature,
the `InlineLocalFiles` middleware accepts a local path as well (`/…`, `./…`,
`~/…`, or `file://`) and reads the file into inline base64 before the request
reaches a codec, so a caller on the same machine as its files can point at
them directly.

`InlineLocalFiles::new([dir, …])` reads only files that resolve inside the
given directories, checked both by name and after symlinks resolve. Paths in
tool results, and in a server's user messages, are not always the
application's own, so a path outside, a file that is not a regular file, or
one over the size limit (32 MiB by default, set with `max_file_bytes`) fails
the call with `InvalidRequest`. `InlineLocalFiles::unrestricted()` reads any
path and is meant for trusted local use, such as a command-line tool run by
the owner of the files.

```rust,no_run
use std::error::Error;

use lithos_llm::types::{ContentPart, ImageContent, MediaSource, Message, Role};
use lithos_llm::{Client, Request};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let client = Client::from_env()?.client;
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

    println!("{}", client.complete(request).await?.text());
    Ok(())
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

```rust,no_run
use lithos_llm::Client;
use lithos_llm::client::{ProbeOptions, ProbeOutcome};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::from_env()?.client;
    let report = client
        .probe("anthropic/claude-sonnet-5", ProbeOptions::new().tools(true))
        .await;
    match &report.outcome {
        ProbeOutcome::Passed => println!("ok in {:?}", report.latency),
        ProbeOutcome::Failed(data) => println!("{:?}: {}", data.kind, data.message),
        ProbeOutcome::Incorrect { detail } => println!("answered, but {detail}"),
        outcome => println!("{outcome:?}"),
    }
    Ok(())
}
```

The probe resolves the route, credentials, and middleware exactly as
`complete` does. The default probe sends one short prompt; `tools(true)` runs
an `add`-tool exchange and checks the model calls the tool and reaches the
right total. A catalog rejection — a tool probe against a model that declares
no tools — fails before any request is sent. `Failed` carries the classified
`ErrorData`, so an unknown model, bad credentials, a missing model, and a
timeout are told apart by its `kind`.

## Error policy

Every failure is an `Error` while it is being handled and an `ErrorData` once
it is stored or sent somewhere. Both answer the questions a retry loop, a
failover chain, or a diagnostic asks: `is_retryable` (repeating the same call
may succeed), `is_auth_error`, `is_cancelled`, and `failover_eligible`
(another provider is worth trying: everything retryable, plus failures local to
this provider such as credentials, model inventory, quota, a timeout, or a
refusal). An invalid request or a context overflow follows the request to the
next provider, so it is not failover-eligible. `ErrorData` prints its message,
implements `std::error::Error`, and reads like `Error`, so an application can
keep one code path for both forms.

## Token estimation

`lithos_llm::estimate` sizes a request locally, before any call is made:

```rust
use lithos_llm::Request;
use lithos_llm::estimate::{EstimateWarning, message_tokens, request_tokens};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let request = Request::builder()
        .model("openai/gpt-5.6-luna")
        .user("Why is the sky blue?")
        .build()?;
    let estimate = request_tokens(&request);
    if estimate.has_warning(EstimateWarning::Media) {
        eprintln!("Media token counts are approximate.");
    }
    let largest = request
        .messages()
        .iter()
        .map(|message| message_tokens(message).tokens())
        .max()
        .unwrap_or(0);
    println!("Estimated tokens: {}; largest message: {largest}", estimate.tokens());
    Ok(())
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

## Client-scoped HTTP headers

Derive a client for a session, run, or tenant without changing the shared
client:

```rust,no_run
use std::error::Error;

use lithos_llm::{Client, Request};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let client = Client::from_env()?.client;
    let session = client.with_default_headers([("x-session-id", "session-123")])?;
    let request = Request::builder()
        .model("openai/gpt-5.6-luna")
        .user("Why is the sky blue?")
        .build()?;
    println!("{}", session.complete(request).await?.text());
    Ok(())
}
```

The derived client shares adapters, middleware, and the HTTP pool. Additional
calls to `with_default_headers` retain previous defaults and replace matching
names case-insensitively.

Client defaults never override a header Lithos sets: codec headers, provider
`default_headers` from the catalog, and credential headers all take
precedence. To override a protocol header deliberately, set it in the
provider's `default_headers` in a catalog overlay.

All built-in HTTP operations carry the defaults, including retries, token
counting, and evaluation; Bedrock applies them before signing. Custom adapters
can read `call.context().default_headers()` and must apply them themselves.
Header values are redacted in Debug output and are not serialized into the
model request. Authentication belongs in credential providers.
