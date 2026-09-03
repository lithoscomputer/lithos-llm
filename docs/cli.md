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

List or search the built-in model catalog without credentials or network
access:

```sh
lllm models
lllm models claude --available
lllm models --json
```

The JSON model list has this CLI-owned shape:

```json
{
  "version": 1,
  "models": [
    {
      "selector": "provider/model",
      "display_name": "Model name",
      "aliases": [],
      "capabilities": { "text": true },
      "available": true
    }
  ]
}
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
