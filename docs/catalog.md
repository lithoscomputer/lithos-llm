# Catalog overlays

Start with the [README](https://github.com/lithoscomputer/lithos-llm/blob/main/README.md) for installation.


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

Provider rows carry routing facts (`adapter`, `codecs`, `codec_options`,
`base_url`, `auth`, `priority`, `default_model`, `allow_passthrough`,
`default_headers`, `adapter_options`, `default_options`) and offering policy:
`enabled` (default
`true`; a disabled provider stays in the catalog but builds no adapter and
resolves no route), `stands_in_for` (a provider this one answers for when that
provider has no adapter), and `api_key_url`. Model rows carry the wire id,
limits, capabilities, protocol options, pricing, and display facts (`family`,
`training_cutoff`, `knowledge_cutoff`, `estimated_output_tps`), plus
`small_default` and `probe`, which name the provider's model for cheap utility
calls and for connectivity probes. A built-in provider that needs only an API
key ships enabled; credentials decide whether it is ready. A provider that
needs deployment-specific setup or has no portable model roster (Bedrock,
LiteLLM, Modal, Ollama) ships with `enabled = false`; an overlay turns one on:

```toml
[providers.ollama]
enabled = true
```

An adapter owns transport and signing; a codec owns one wire protocol. A
provider names one adapter and lists the codecs its host speaks. `adapter`
defaults to `"http"`, which holds every codec the provider lists and
dispatches each call on the codec the client selected; `"bedrock"` adds
SigV4 signing and event-stream framing for the Converse codec; any other id
names a factory the application registered with
`ClientBuilder::adapter_factory`, whose adapter owns its own wire handling
and receives no codec selection. `codecs` defaults to `["openai-chat"]`, so
a Chat Completions host needs neither line. OpenAI lists
`["openai-responses"]`, Anthropic `["anthropic-messages"]`, Gemini
`["gemini-generate"]`, Bedrock `["bedrock-converse"]`, Vercel
`["openai-chat", "vercel-evaluation"]`, and TypeSafe `["systemone"]`. `codec_options` is a table keyed by
codec id that carries one codec's own options, such as
`codec_options = { openai-chat = { base_url_is_api_root = true } }` for a host
whose base URL is already versioned; `adapter_options` carries transport
concerns such as Codex's `force_streaming_complete` and
`identify_application`. The old `codec` field and the four protocol-named
adapter ids (`openai-compatible`, `openai`, `anthropic`, `gemini`) are
rejected with an error that names the replacement.

Each model row reaches a subset of its provider's codecs, and the client
picks one per call by the operation's family, in provider order. A row may
write `codecs = [...]` to narrow the set. Without that line the set follows
the row's own capability claims: a generation claim (`text`, `tools`,
`images`, `audio`, `documents`, or a `response_format` field) that is not
`false` reaches every generation codec the provider lists; an explicit
`capabilities.evaluation` with at least one kind `true` reaches every
evaluation codec the provider lists; a row that claims neither keeps the
generation codecs. A passthrough model gets the generation codecs only.

A model row that claims `response_format.json_schema` also answers evaluation
questions of every kind (`choice`, `score`, `boolean`): it judges by producing
one schema-bound JSON object, so the catalog does not repeat the claim per
row. That derived claim never reaches an evaluation codec. A row writes
`capabilities.evaluation = { choice = true, score = false, boolean = true }`
to narrow the judge claim, or to claim evaluation on a model that speaks an
evaluation protocol natively; only such a row derives its provider's
evaluation codec, as `vercel/jev` does. `ModelCapabilities::evaluation` and
`evaluates` answer the question a caller asks.

Before it has a request, an application asks the catalog which providers are
on and which model to pick for a job. `Catalog::enabled_providers` lists them
in priority order and `listed_providers` drops the ones that stand in for
another; `offerings_matching` ranks every enabled offering of a selector the
way the resolver would; `default_offering_for` and `small_default_for` pick the
default and the cheap utility model across the providers that are ready;
`CatalogProvider::probe_offering` names the row to probe with and
`closest_offering` the nearest model to one on another provider. Each returns
an `Offering`, the borrowed provider-and-model pair, with `handle`,
`into_route`, and `estimate_cost`.

An application can supply its whole catalog and use none of the built-in
entries. Building from external TOML never adds built-in providers implicitly;
only `with_builtin()` does that.

```rust
use std::error::Error;

use lithos_llm::catalog::Catalog;

fn main() -> Result<(), Box<dyn Error>> {
    let root = "schema_version = 1";
    let openai = r#"
        [providers.openai]
        display_name = "OpenAI"
        base_url = "https://api.openai.com"
        auth = { type = "bearer" }
        codecs = ["openai-responses"]
        [providers.openai.models."gpt-5.6-luna"]
        display_name = "GPT-5.6 Luna"
        api_model = "gpt-5.6-luna"
        capabilities = { text = true }
    "#;
    let anthropic = r#"
        [providers.anthropic]
        display_name = "Anthropic"
        base_url = "https://api.anthropic.com"
        auth = { type = "header", name = "x-api-key" }
        codecs = ["anthropic-messages"]
        [providers.anthropic.models."claude-sonnet-5"]
        display_name = "Claude Sonnet 5"
        api_model = "claude-sonnet-5"
        capabilities = { text = true }
    "#;
    let catalog = Catalog::builder()
        .toml_layer("catalog.toml", root)?
        .toml_layer("providers/openai.toml", openai)?
        .toml_layer("providers/anthropic.toml", anthropic)?
        .build()?;
    assert_eq!(catalog.schema_version(), 1);
    Ok(())
}
```

