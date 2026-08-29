# lithos-llm-cli

`lithos-llm-cli` installs the `lithos` binary. It sends one stateless prompt
through the provider-neutral `lithos-llm` library.

Install it from a checkout:

```sh
cargo install --locked --path cli
```

Send prompt text with the default streaming output:

```sh
lithos 'Explain ownership in Rust'
lithos prompt 'Explain ownership in Rust' --model openai/gpt-5
printf 'Explain this input' | lithos
```

List or search the built-in model catalog without credentials or network
access:

```sh
lithos models
lithos models claude --available
lithos models --json
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
lithos 'Describe this image' --attachment photo.png
lithos 'Summarize this' --attachment https://example.com/report.pdf
cat report.pdf | lithos 'Summarize this' --at - application/pdf
```

Use `--json-object` or `--schema` to request structured output. Use `--json`
to print a versioned envelope that contains the normalized request and
response. Use `--extract` or `--extract-last` to print a complete Markdown code
block from a buffered response.

Exit status `0` means success or a closed output pipe. Status `1` means a
client, credential, provider, network, output, or decoding failure. Status `2`
means invalid arguments or input. Status `130` means the user interrupted the
call.

The CLI does not store configuration, credentials, aliases, templates,
conversation history, logs, or other state. It does not run tools or an agent
loop.
