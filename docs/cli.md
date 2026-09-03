# lllm

The `cli` feature installs the `lllm` binary. It sends one stateless prompt
through the provider-neutral `lithos-llm` library.

Install it from a checkout:

```sh
cargo install --locked --path . --features cli
```

Run it without installing:

```sh
cargo run --features cli -- --help
```

Send prompt text with the default streaming output:

```sh
lllm 'Explain ownership in Rust'
lllm prompt 'Explain ownership in Rust' --model openai/gpt-5
printf 'Explain this input' | lllm
```

Add `-u` or `--usage` to print a compact model, token, cost, and latency
summary to standard error. Standard output remains safe to pipe:

```text
anthropic/claude-sonnet-5 · 1,240 input · 183 output · $0.00431 · 1.8s
```

Model selection uses this precedence:

1. `--model`
2. `--model-query`
3. `LLLM_MODEL`
4. The highest-priority catalog default with a compiled adapter

Print the canonical provider and model without making a provider request:

```sh
lllm resolve
lllm resolve --model claude-sonnet
lllm resolve --model-query sonnet
```

Use `--option KEY=VALUE` and `--metadata KEY=VALUE` for raw request values.
Use `--xl` as the short form of `--extract-last`.

List or search the built-in model catalog without credentials or network
access:

```sh
lllm models
lllm models claude --adapter-compiled
lllm models --provider anthropic
lllm models --capability tools
lllm models --capability structured-output
lllm models --configured
lllm models --default
lllm models --json
```

The table marks the effective default. It reports adapter and credential
status separately. `ADAPTER` means that this CLI contains the provider
adapter. `CREDENTIALS` means that the required environment credentials are
configured. An unauthenticated provider is always configured.
Filters can be combined. Use `--configured --adapter-compiled` to list routes
that have both credentials and an adapter.

The JSON model list has this CLI-owned shape:

```json
{
  "version": 2,
  "effective_default": "provider/model",
  "models": [
    {
      "selector": "provider/model",
      "display_name": "Model name",
      "aliases": [],
      "capabilities": { "text": true },
      "adapter_compiled": true,
      "credentials_configured": true,
      "effective_default": true
    }
  ]
}
```

Add a local TOML catalog layer for one process with `--catalog`. The option
can be repeated. Later files win when layers overlap. The CLI does not save
the files or change the built-in catalog:

```sh
lllm --catalog local-models.toml -m local/qwen 'Hello'
lllm --catalog local-models.toml models --provider local
```

Attach files, URLs, or one standard-input payload:

```sh
lllm 'Describe this image' --attachment photo.png
lllm 'Summarize this' --attachment https://example.com/report.pdf
cat report.pdf | lllm 'Summarize this' --at - application/pdf
```

Use `--json-object` or `--schema` to request structured output. Use `--json`
to print a versioned envelope that contains the normalized request and
response. Use `--extract` or `--extract-last` to print a complete Markdown code
block from a buffered response.

For interactive schemas, list comma-separated fields as
`NAME [TYPE] [: DESCRIPTION]`. The supported types are `string`, `int`,
`float`, and `bool`. A missing type means `string`. All fields are required.
`--schema-multi` requests an array of matching objects:

```sh
lllm --schema 'name, age int, bio' 'Invent a person'
lllm --schema-multi 'name, age int' 'Invent three people'
```

Full JSON Schema objects and `@PATH` files continue to work with both flags.

Add local UTF-8 files as stateless text context with repeated `-f` or
`--fragment` options. Use `--system-fragment` to append a file to the system
message:

```sh
lllm -f README.md -f Cargo.toml 'Review this project'
lllm --system-fragment review-policy.md 'Review src/lib.rs'
```

Fragments do not accept standard input, URLs, or stored aliases. Attachments
remain the binary and remote-content path.

Pass `--verbose` to see the library's debug diagnostics on standard error.
`RUST_LOG` selects a custom filter and overrides `--verbose`, for example
`RUST_LOG=lithos_llm=trace`. Without either, the CLI prints no telemetry.

Exit status `0` means success or a closed output pipe. Status `1` means a
client, credential, provider, network, output, or decoding failure. Status `2`
means invalid arguments or input. Status `130` means the user interrupted the
call.

The CLI does not store configuration, credentials, aliases, templates,
conversation history, logs, or other state. It does not run tools or an agent
loop.
