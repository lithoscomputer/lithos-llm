# Developing

`lithos-llm` is an async reusable library. Applications own the Tokio runtime.
Catalog parsing, validation, overlays, and model resolution stay synchronous.

Keep public provider-neutral types separate from private codecs and transport.
Treat serialized request, response, content, and catalog shapes as public
contracts.

## Setup

Install [Mise](https://mise.jdx.dev/), then install the locked tools and prepare
the pinned Rust Style Guide:

```sh
mise trust
mise install --locked
mise run setup
```

## Common tasks

| Command | Purpose |
| --- | --- |
| `mise run dev` | Check all default-feature targets |
| `mise run fmt` | Format Rust code |
| `mise run fmt:check` | Check formatting without changing files |
| `mise run lint` | Run Clippy with warnings denied |
| `mise run test` | Run the test suite |
| `mise run check` | Run the routine verification gate |
| `mise run check:nightly` | Run the extended verification gate |
| `mise run check:catalog-only` | Verify the catalog-only feature boundary |
| `mise run check:features` | Check each feature and the default Bedrock boundary |

Run `mise run check` before opening a pull request.

## Rust policy

This project follows the pinned Brynary Rust Style Guide. Run
`mise run setup`, then read `.ai/style-guides/rust-style-guide/SKILL.md` before
changing Rust code, configuration, project structure, or tests.

The project uses Rust 2024 and declares Rust 1.85 as its minimum supported
version. Mise pins the development compiler and the nightly formatter.

## Feature checks

The routine gate checks every feature. It also verifies that a catalog-only
build has no normal dependency on Tokio, reqwest, or an AWS SDK.

Bedrock is not a default feature. It adds the AWS SDK and supports both the AWS
default credential chain and Bedrock bearer tokens.

## Continuous integration

Routine checks run for pull requests and pushes to `main`. Extended checks run
each night. Both workflows test these native platforms:

- macOS arm64;
- Linux x86_64;
- Linux arm64.

## Releases

Pushing a `v*` tag packages the crate, creates a SHA-256 checksum, and opens a
draft GitHub release. Review the package before publishing it to crates.io.
