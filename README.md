# lithos-llm

A provider-neutral LLM client for Rust: one request type for OpenAI, Anthropic, Gemini, and more.

[![CI](https://github.com/lithoscomputer/lithos-llm/actions/workflows/routine.yml/badge.svg)](https://github.com/lithoscomputer/lithos-llm/actions/workflows/routine.yml)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

```rust,no_run
use std::error::Error;

use lithos_llm::{Client, Request};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let client = Client::from_env()?.client;
    let request = Request::builder()
        .model("openai/gpt-5.6-luna")
        .user("Why is the sky blue?")
        .build()?;
    println!("{}", client.complete(request).await?.text());
    Ok(())
}
```

Part of [Lithos](#part-of-lithos), a family of Rust building blocks for AI tools.

## Why does this exist?

Provider APIs differ in request formats, streaming events, and model
capabilities. `lithos-llm` pairs a shared request API with a TOML catalog of
providers, models, capabilities, and prices. Applications can customize that
catalog in layers while keeping control of tool execution and their agent loop.

## Who is it for?

**Use it if** you are building an agent, developer tool, or service that needs
multiple LLM providers, or a model picker that needs a local catalog.

**Look elsewhere if** you need a ready-made agent loop or coding agent;
[pebble](https://github.com/lithoscomputer/pebble) adds those on top of this crate.

## What does it do?

- Send completions and stream text, reasoning, tool calls, and usage through one API.
- Carry images, audio, documents, and tool results where the selected model supports them.
- Request structured JSON output and evaluate state with typed choice, score, and boolean questions.
- Resolve models using a built-in catalog of aliases, capabilities, limits, and prices; customize it with TOML overlays.
- Estimate tokens locally, request native token counts where available, and probe provider connectivity.
- Extend credentials, middleware, and provider adapters for your application.

Built-in integrations cover OpenAI, Anthropic, Gemini, TypeSafe, and gateways
such as OpenRouter, Venice, and Vercel. Amazon Bedrock support is optional.
See [provider coverage and pending live checks](https://github.com/lithoscomputer/lithos-llm/blob/main/docs/provider-live-tests.md) for verification details.

**Out of scope:** tool execution and agent loops. Your application supplies the
Tokio runtime, credentials, tracing subscriber, and conversation storage.
Retry and tracing middleware are opt-in.

| Feature | Default | Enables |
| --- | --- | --- |
| `builtin-catalog` | yes | Embedded provider and model catalog |
| `runtime` | yes | Async client, HTTP transport, and built-in HTTP wire protocols |
| `environment-credentials` | yes | Credentials from conventional environment variables |
| `cli` | no | The `lllm` command-line application |
| `local-files` | no | Middleware that reads local media from allowed directories |
| `bedrock` | no | Bedrock Converse with bearer-token authentication |
| `bedrock-aws` | no | Bedrock with the AWS credential chain and SigV4 signing |

A catalog-only build needs no Tokio, reqwest, or provider SDKs. The `bedrock`
feature adds no AWS SDK crates; `bedrock-aws` adds them.

## How do I use it?

### Install

The crate is not yet published on crates.io. Add it from Git:

```sh
cargo add lithos-llm --git https://github.com/lithoscomputer/lithos-llm
cargo add tokio --features macros,rt-multi-thread
```

Set `OPENAI_API_KEY` to run the opening example. For other providers, use
`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, or the credential names in the
[library guide](https://github.com/lithoscomputer/lithos-llm/blob/main/docs/library-guide.md#build-a-client).
Change `.model(...)` to select a different provider and model.

`Client::from_env()` returns a client and any provider construction issues.
Inspect `ClientBuild::issues` to report unavailable providers. Credentials are
resolved when a call runs; building the client does not validate your keys.

For catalog access without the runtime:

```sh
cargo add lithos-llm --git https://github.com/lithoscomputer/lithos-llm --no-default-features --features builtin-catalog
```

### Use it as a library

Stream text as it arrives. Add `futures-util` for `StreamExt`:

```sh
cargo add futures-util
```

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
        .user("Explain ownership in Rust.")
        .build()?;
    let mut stream = client.stream(request).await?;
    while let Some(event) = stream.next().await {
        if let StreamEvent::TextDelta { text, .. } = event? {
            print!("{text}");
            io::stdout().flush()?;
        }
    }
    Ok(())
}
```

Successful streams end with `StreamEvent::Ended`, which carries the assembled
response. Check its `finish_reason` before accepting the turn or executing tools.
See the [streaming guide](https://github.com/lithoscomputer/lithos-llm/blob/main/docs/library-guide.md#streaming).

Inspect a model's catalog entry offline, without credentials or a runtime:

```rust
use std::error::Error;

use lithos_llm::catalog::Catalog;

fn main() -> Result<(), Box<dyn Error>> {
    let catalog = Catalog::builder().with_builtin().build()?;
    let model = catalog.model("openai", "gpt-5.6-luna")?;
    println!("{}", model.id());
    Ok(())
}
```

See the [library guide](https://github.com/lithoscomputer/lithos-llm/blob/main/docs/library-guide.md)
for tool calling, structured output, evaluation, media, and call controls.
See [catalog overlays](https://github.com/lithoscomputer/lithos-llm/blob/main/docs/catalog.md)
for model selection and custom providers.
Build the API reference locally with `cargo doc -p lithos-llm --no-deps --open`
from a project that depends on the crate.

### Use it from the command line

Install the optional `lllm` binary from Git. It sends stateless prompts using
the same catalog and provider credentials:

```sh
cargo install --git https://github.com/lithoscomputer/lithos-llm --locked --features cli lithos-llm
lllm models --provider openai
lllm --model openai/gpt-5.6-luna 'Explain ownership in Rust'
lllm probe --model openai/gpt-5.6-luna
```

See the [CLI guide](https://github.com/lithoscomputer/lithos-llm/blob/main/docs/cli.md).

## Status and compatibility

- **Maturity:** pre-1.0 and unreleased. The changelog records breaking API changes; review it when updating your Git dependency.
- **MSRV:** Rust 1.88, edition 2024.
- **CI platforms:** Linux x86_64, Linux arm64, and macOS arm64.
- **Public API beyond Rust types:** serialized requests, responses, content, and catalog data; CLI JSON output; the evaluation judge prompt.
- **Compatibility notes:** [API migration](https://github.com/lithoscomputer/lithos-llm/blob/main/docs/api-migration.md).
- **Changelog:** [CHANGELOG.md](https://github.com/lithoscomputer/lithos-llm/blob/main/CHANGELOG.md).

## Part of Lithos

| Crate | Role |
| --- | --- |
| **[lithos-llm](https://github.com/lithoscomputer/lithos-llm)** | **Model access: one client and catalog across LLM providers** |
| [pebble](https://github.com/lithoscomputer/pebble) | Agent loop, plus a coding agent and CLI built on it |
| [sandbox-driver](https://github.com/lithoscomputer/sandbox-driver) | Sandboxes (host, Docker, cloud VMs) behind one interface |
| [petri](https://github.com/lithoscomputer/petri) | Workflow engine that runs CI and agent workflows on the crates above |

## Contributing

Development setup, tests, and conventions are in
[DEVELOPING.md](https://github.com/lithoscomputer/lithos-llm/blob/main/DEVELOPING.md).
Read [CONTRIBUTING.md](https://github.com/lithoscomputer/lithos-llm/blob/main/CONTRIBUTING.md)
before submitting a change. Report vulnerabilities through
[SECURITY.md](https://github.com/lithoscomputer/lithos-llm/blob/main/SECURITY.md).

## License

Licensed under either [MIT](https://github.com/lithoscomputer/lithos-llm/blob/main/LICENSE-MIT)
or [Apache-2.0](https://github.com/lithoscomputer/lithos-llm/blob/main/LICENSE-APACHE), at your option.
