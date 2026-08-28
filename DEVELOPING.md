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
| `mise run test` | Run the test suite with all features |
| `mise run test:default-features` | Run the test suite with default features |
| `mise run check` | Run the routine verification gate |
| `mise run check:nightly` | Run the extended verification gate |
| `mise run check:catalog-only` | Verify the catalog-only feature boundary |
| `mise run check:features` | Check each feature and the AWS SDK boundary |

Run `mise run check` before opening a pull request.

## Rust policy

This project follows the pinned Brynary Rust Style Guide. Run
`mise run setup`, then read `.ai/style-guides/rust-style-guide/SKILL.md` before
changing Rust code, configuration, project structure, or tests.

The project uses Rust 2024 and declares Rust 1.85 as its minimum supported
version. Mise pins the development compiler and the nightly formatter.

## Feature checks

The gate runs the tests twice: once with every feature and once with the
default set. Running only `--all-features` hides failures that appear when a
provider feature is off, such as a built-in catalog entry whose adapter is not
compiled.

The routine gate checks every feature. It also verifies that a catalog-only
build has no normal dependency on Tokio, reqwest, or an AWS crate.

Bedrock is not a default feature. `bedrock` supplies the Converse wire format
and Bedrock bearer tokens over the shared HTTP transport and pulls in no AWS
crate. `bedrock-aws` adds the AWS credential chain and SigV4 signing. The gate
asserts that both default features and `bedrock` alone stay free of AWS crates.

## Wire tests

`tests/it` holds the provider wire suite. Each test points the public client at
a local mock server, captures the exact request the codec produced, and
snapshots both that request and the decoded result. The snapshots are the
contract for provider wire behavior, so review a changed snapshot the way you
would review a public API change.

Snapshots live in `tests/it/wire/snapshots/`. Review pending changes with
[`cargo-insta`](https://insta.rs/):

```sh
cargo insta pending-snapshots
cargo insta accept
```

Dynamic values are normalized so snapshots stay deterministic: hosts, user
agents, credential headers, and AWS signing dates are replaced with
placeholders, and header lists are lowercased and sorted. Content-block ids and
tool-call ids are synthesized deterministically rather than randomly, so they
need no scrubbing.

Where our wire behavior intentionally differs from the previous Fabro
implementation, the difference is documented next to the affected fixture.

## Continuous integration

Routine checks run for pull requests and pushes to `main`. Extended checks run
each night. Both workflows test these native platforms:

- macOS arm64;
- Linux x86_64;
- Linux arm64.

## Releases

Pushing a `v*` tag packages the crate, creates a SHA-256 checksum, and opens a
draft GitHub release. Review the package before publishing it to crates.io.
