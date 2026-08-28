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

Use `ClientBuilder::http` to inject an application-configured
`reqwest::Client`, and `ClientBuilder::enabled_providers` to build adapters for
only the providers a deployment has configured. The complete catalog stays
available for inspection either way.

`Client::resolve_route` reports the provider and model a request would use
without dispatching it, which is the same resolution `complete`, `stream`, and
`count_input_tokens` perform.

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
