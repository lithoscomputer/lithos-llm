# Testing

This repository tests `lithos-llm` in four layers. Each layer answers one
question. The layers below run in seconds with no network and no keys; only
the last layer talks to a real provider.

## The layers

### 1. Unit tests (`src/**`, `tests/contracts.rs`, `tests/runtime.rs`)

**Question: does each piece do what its contract says?**

Focused tests live beside the code they test: usage normalization, error
classification, catalog parsing, request validation. `tests/contracts.rs`
round-trips the serialized `Request`, which is a public contract.
`tests/runtime.rs` tests the client pipeline — middleware order, retries,
timeouts, capability rejection — against a fake adapter, with no HTTP.

### 2. The wire suite (`tests/it/`)

**Question: does each codec produce the intended provider request and decode
the protocol cases we specify?**

Every test points a real `Client` at a local `httpmock` server. The test
snapshots both halves of the exchange: the normalized outbound request pins
encoding, and the decoded library result pins our public representation. Mock
provider responses are written by hand. They say what a provider is *expected*
to send. One dialect module exists per codec.

Snapshots use `insta` and render JSON in canonical key order, so key order
can never fail a test. When a snapshot changes, review the diff and accept
it with `cargo insta review`. A snapshot change is a contract change.
Snapshots provide broad shape coverage. Explicit assertions protect critical
semantics such as usage buckets, finish reasons, errors, tool calls, and stream
invariants.

### 3. The replay suite (`tests/e2e/`, replay backend)

**Question: can the full client still reproduce and handle recorded exchanges
with real providers?**

The E2E tests send current requests through local twin-openai proxies. Each
proxy matches requests to its provider recording under
`tests/e2e/recordings/` and returns the recorded real response. This checks
request compatibility and full-client behavior against historical exchanges.
This is the default backend: `mise run test:e2e` runs offline with no keys.

Each test's own path is its recording namespace, so concurrent tests and
retry loops replay deterministically. A `scenario_not_found` failure means
the recording does not cover a new or changed test: run
`mise run test:e2e:record` to refresh it.

### 4. The live suite (`tests/e2e/`, record and live backends)

**Question: did a live provider drift?**

The same E2E tests run against the real API. `mise run test:e2e:record`
runs live through the proxy and rewrites the recording — a changed
recording is the drift report. `mise run test:e2e:live` runs unproxied.
Both spend provider credits and read keys from `.env`.

Each per-provider E2E catalog declares its models' capabilities. The tests
treat those claims as assertions:
a capability the catalog claims but the provider rejects is a failure, not
a skip. Probe tests record live behavior nothing pins yet (effort levels,
cache buckets, rate limits); the record and live tasks keep their output
in the run log through `--success-output final`, while the replay task
drops it — replayed probe output is just the recording played back. Tests
that only make sense live — error classification, timing, the model
listing — skip under record and replay.

Task output stays quiet while everything passes, because agents consume
it: cargo runs with `-q`, Nextest hides per-test `PASS` lines with
`--status-level fail`, and every feature subset compiles warning-free —
items that only some features use carry `cfg` gates, so a dead-code
warning in any build is a real finding, never expected noise. A green
`mise run check` prints little more than one summary line per suite;
failures still print in full.

The replay backend covers the OpenAI-compatible providers that the twin can
proxy. Anthropic and Gemini use native protocols, and Modal needs two upstream
authentication headers, so their network checks are live-only. Their free
catalog and request-validation checks still run in the routine suite.

## Commands

| Command | What it runs |
| --- | --- |
| `mise run dev` | Type-check all default targets |
| `mise run test` | Layers 1 and 2, plus the free E2E preflight tests |
| `mise run test:e2e` | Layer 3: offline replay of the committed recording |
| `mise run test:e2e:record` | Layer 4: live run that rewrites the recording |
| `mise run test:e2e:live` | Layer 4: live run, unproxied |
| `mise run check` | The complete routine gate |
| `mise run check:nightly` | The routine gate plus MSRV and release-mode tests |

## Rules of thumb

- Assert protocol, not prose: exact tool names and schemas, token buckets,
  finish reasons — never the model's wording, beyond a forced token like
  `PONG`.
- A live test must be deterministic in what it *sends*: no timestamps or
  randomness in request bodies, so recordings match on replay.
- Keep large test prefixes as genuinely varied text, not repeated
  paragraphs; upstream safety filters treat repetitive machine-generated
  prompts specially.
- The built-in catalog, the E2E catalog, and the E2E roster are pinned to
  each other by preflight tests. Adding a model means updating all three
  and re-recording.
- New live findings flow upstream: fix the catalog claim or the codec, add
  the regression test at the layer where the behavior is observable, and
  note provider quirks in a comment with the date they were observed.
