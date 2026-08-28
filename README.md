# lithos-llm

`lithos-llm` is a provider-neutral Rust library for language model access. It
separates catalog data, model resolution, credentials, middleware, provider
adapters, wire codecs, and HTTP transport.

Applications can use built-in OpenAI, Anthropic, Gemini,
OpenAI-compatible, and optional Amazon Bedrock adapters. Applications can also
register their own `ProviderAdapter` implementations.

The library does not execute tools or own an agent loop. It carries tool calls
and tool results between an application and a provider.

## Example

```rust
use std::error::Error;

use lithos_llm::{
    Client, Request,
    catalog::Catalog,
    credentials::EnvironmentCredentials,
    middleware::{RetryMiddleware, RetryPolicy, TracingMiddleware},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let catalog = Catalog::builder().with_builtin().build()?;
    let credentials = EnvironmentCredentials::builder()
        .bearer("openai", "OPENAI_API_KEY")
        .build();
    let client = Client::builder()
        .catalog(catalog)
        .credentials(credentials)
        .middleware(TracingMiddleware)
        .middleware(RetryMiddleware::new(
            RetryPolicy::exponential().max_attempts(3),
        ))
        .build()?;

    let request = Request::builder()
        .model("openai/gpt-5.6-luna")
        .system("Answer in one short paragraph.")
        .user("Why is the sky blue?")
        .max_output_tokens(300)
        .build()?;
    let response = client.complete(request).await?;
    assert!(!response.text().is_empty());
    Ok(())
}
```

Applications own the Tokio runtime and tracing subscriber. Credential lookup
runs for each provider attempt, so application credential providers can refresh
tokens without rebuilding the client.

## Catalog overlays

Catalog documents use schema version `1`. Later overlays win. Tables merge
recursively. Arrays and scalar values replace earlier values.

```rust
use std::error::Error;

use lithos_llm::catalog::Catalog;

fn main() -> Result<(), Box<dyn Error>> {
    let catalog = Catalog::builder()
        .with_builtin()
        .overlay_toml(
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

Metadata stays application-owned. Lithos preserves unknown namespaces without
interpreting them.

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
| `bedrock` | no | Bedrock Converse adapter and AWS SDK |

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
