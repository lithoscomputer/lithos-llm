# Adding a Provider

This is the procedure for adding one provider to the built-in catalog and
its test suites. Run it once per provider. Venice is the finished
reference: compare your work against `src/catalog/builtin/venice.toml` and
`tests/e2e/venice/` at every step.

Read `TESTING.md` first. It explains the four test layers this procedure
feeds.

## What you will produce

1. `src/catalog/builtin/<provider>.toml` — the catalog rows.
2. A `<PROVIDER>_API_KEY` entry in `EnvironmentCredentials::conventional()`
   (`src/credentials.rs`), with a test.
3. Loader tests for provider-specific resolution and behavior
   (`src/catalog/loader.rs`).
4. `tests/e2e/<provider>/` — the E2E module tree, plus
   `tests/e2e/<provider>_catalog.toml`.
5. `tests/e2e/recordings/<provider>.json` — the committed recording.
6. A changelog entry, and comments in the catalog that record where each
   fact came from and the date you verified it.

## Phase 1 — translate the fabro catalog

The source file is
`~/p/fabro-sh/fabro/lib/foundation/fabro-model/src/catalog/providers/<provider>.toml`.
Translate each row into the lithos-llm schema. Do not copy blindly: fabro
is a starting claim, not the truth. Phase 2 verifies every fact.

Field mapping:

| fabro | lithos-llm |
| --- | --- |
| `api_id` | `api_model` (defaults to the row id when absent) |
| `limits.context_window` | `limits.context_tokens` |
| `limits.max_output` | `limits.max_output_tokens` |
| `features.tools` | `capabilities.tools` |
| `features.vision` | `capabilities.images` |
| `features.reasoning` | `capabilities.reasoning` |
| `features.reasoning_effort = "levels"` | `capabilities.reasoning_effort_levels` |
| `features.prompt_cache` | `capabilities.caching` |
| `features.sampling_params` | `capabilities.sampling` |
| `costs.input_cost_per_mtok` (USD) | `pricing.input_usd_micros_per_million` (multiply by 1,000,000) |
| `costs.output_cost_per_mtok` | `pricing.output_usd_micros_per_million` |
| `costs.cache_input_cost_per_mtok` | `pricing.cached_input_usd_micros_per_million` |
| `costs.speed.<name>` sub-tables | `pricing.speed` (`fast`/`balanced`/`economical` rates) |
| `features.cache_control_breakpoints` | `capabilities.cache_breakpoints` |
| `auth.credentials = ["env:NAME", ...]` | `auth` scheme + a `conventional()` env mapping |
| `extra_headers` (non-secret values only) | `default_headers` |
| `default = true` | provider `default_model` |
| `aliases`, `priority`, `base_url` | same names |

The full fabro schema lives in
`~/p/fabro-sh/fabro/lib/foundation/fabro-model/src/catalog.rs`. Fields
with no lithos-llm equivalent fall into two groups:

- **Drop**: `billing_policy` — its behavior came over, decomposed, so
  the knob itself is redundant: each codec normalizes its protocol's
  counters into disjoint token buckets at decode time, and the catalog
  prices each bucket at its own rate (`cache_write_usd_micros_per_million`
  carries Anthropic's cache-write premium; `pricing.long_context` and
  `pricing.speed` carry the tiering). A fabro row with
  `billing_policy = "anthropic"` translates to cache-write and
  cached-input rates on the row, nothing more. Also drop
  `reasoning_by_default` (use it to decide which rows assert reasoning
  evidence in the E2E suite, then drop it), `enabled` (a lithos-llm
  provider is usable whenever credentials resolve).
- **Preserve in `metadata`**: fields fabro's application layer reads but
  the catalog does not act on — `agent_profile`, `family`, `training`,
  `knowledge_cutoff`, `estimated_output_tps`, `probe`, `small_default`,
  `api_key_url`. Put them under a `fabro` metadata namespace on the row.
  The end goal is for fabro to consume this catalog instead of its own,
  and these fields must survive the move.

Fields lithos-llm has that fabro lacks: set `text = true` on every row;
decide `structured_output`, `documents`, `audio`, and `cache_routing`
from Phase 2 evidence, not from guesses.

Roster policy: include every fabro row, then add any model another
provider's lithos-llm roster carries that this provider also serves. For
example, Venice's Claude and GPT rows belong on OpenRouter too if
OpenRouter serves those models. The live listing in Phase 2 tells you.

Catalog ids: keep the short id from fabro (`claude-opus-5`), and put the
provider's wire id in `api_model` (`anthropic/claude-opus-5`). Never put a
`/` in a catalog model id: the resolver splits a request selector on the
first `/`, so a bare id with a slash would be read as `provider/model`.

## Phase 2 — verify against the live provider

Do this before writing any test. Budget an hour. Every conflict you find
here is one red test you will not have to debug later.

1. Read the provider's API docs: authentication scheme, base URL and path
   shape, the models endpoint, pricing, and anything they document about
   caching, reasoning, and rate limits.
2. Pull the live model listing and cross-check every roster row: the wire
   id exists, the context and output limits match, the prices match after
   conversion, and the capability flags agree.

   ```bash
   curl -sS <models-endpoint> -H "Authorization: Bearer $KEY" | jq '...'
   ```

3. When fabro and the listing disagree, the listing wins. When the listing
   and observed behavior disagree, behavior wins — listings lie. Venice's
   listing was wrong about reasoning-effort support in both directions.
   Record every conflict as a comment on the row, with the date.
4. Probe anything ambiguous with one raw `curl` per question: a forced
   tool call, a `reasoning_effort`, a `response_format` with a schema, a
   cache write-then-read pair. Five minutes of probing beats an afternoon
   of test triage.
5. Confirm the URL shape. The codec appends its own path
   (`/v1/chat/completions`, `/v1/responses`, ...) to `base_url`, and the
   joiner collapses one repeated version segment. Verify the final URL
   with one request.
6. List the wire quirks the tests must cover: in-band cost fields, usage
   counter spellings (cache and reasoning), finish reasons, error body
   shapes (check `src/transport/classify.rs` coverage), injected defaults
   (Venice injects a system prompt unless told not to — check for
   equivalents), and cache routing hints.

## Phase 3 — write the catalog

1. Create `src/catalog/builtin/<provider>.toml`. The build script picks it
   up automatically; there is no list to update.
2. Choose the adapter and codec: `openai-compatible` + `openai-chat` for
   gateways; the native adapter for OpenAI, Anthropic, or Gemini.
3. Set `allow_passthrough = true` if the negative tests will send unknown
   model ids. Add `default_options` for provider knobs every request
   should carry (Venice: `include_venice_system_prompt = false`).
4. Skip `pricing` only when the provider reports cost in-band and the E2E
   suite will assert on that; the built-in rows should carry pricing.
5. Add the conventional credential mapping in `src/credentials.rs`.
6. Add loader tests for provider-specific resolution and behavior.
7. Run `mise run test`. The catalog must parse, validate, and resolve.

## Phase 4 — write the E2E module

1. Copy `tests/e2e/venice/` to `tests/e2e/<provider>/` and register it in
   `tests/e2e/main.rs`. Adjust: the provider id, the key variable, the
   roster in the `model_tests!` macro and the family-representative subset.
2. Create `tests/e2e/<provider>_catalog.toml`. Same rows as the built-in
   file. Omit pricing when the provider reports cost in-band, so the cost
   assertion proves the in-band extraction.
3. Start with zero quirk pins. Do not copy Venice's skips (stop
   sequences, output caps): those pin *Venice's upstreams*. Every skip
   needs its own dated evidence for this provider.
4. Cost assertions differ by provider: `CostSource::Provider` where cost
   arrives in-band (Venice, OpenRouter), `CostSource::Catalog` elsewhere.
5. Check record/replay support. The twin proxies only
   `/v1/chat/completions` and `/v1/responses`. A provider on another
   protocol (Anthropic, Gemini) runs live-only until the twin grows a
   passthrough for its paths — coordinate a twins change first, and guard
   the module with `support::live_only` in the meantime.
6. Extend `examples/e2e_twin.rs` and the `test:e2e*` mise tasks for the
   new provider: its upstream URL, its own recording path, and its own
   twin port. One twin process serves one upstream.

## Phase 5 — first live run

```bash
set -a; . ./.env; set +a
LITHOS_E2E_BACKEND=live cargo nextest run --locked --all-features --test e2e \
  --run-ignored all --test-threads 4 --no-fail-fast --success-output final \
  -E 'test(<provider>::)'
```

Expect failures. Triage each one into exactly one bucket:

- **Our bug.** The codec or client mishandles real bytes. Fix it in `src`
  with a unit test. (Venice found `cache_creation_input_tokens` decoding
  and a finish-reason gap this way.)
- **Wrong catalog claim.** The provider rejects a claimed capability, or
  supports an unclaimed one. Fix the row. Comment the evidence and date.
- **Provider bug.** Behavior is wrong and reproducible. Pin it with a
  documented skip or waiver, and write a repro report (see the Venice
  gists for the structure: a concise report plus a standard-library
  Python script that exits 1 while the bug is present).
- **Test too strict.** Assert protocol, not prose; schema conformance,
  not plausibility. Loosen the assertion, not the contract.

Iterate until the provider's cells are green or documented. Keep prompts
tiny; a full run costs a few dollars.

## Phase 6 — record and replay

1. Determinism check first: no request body may include a timestamp,
   randomness, or a living file. Large prefixes come from frozen fixtures
   under `tests/e2e/fixtures/`.
2. Record: `mise run test:e2e:record` (filtered to the provider if the
   tasks support it by then). The recording contains responses and request
   hashes only — no credentials can land in it.
3. Replay: `mise run test:e2e` must pass the provider's cells offline.
4. Commit the recording. From now on `mise run check` covers this
   provider's real recorded bytes on every run.

## Definition of done

- [ ] `mise run check` is green, replay included.
- [ ] The live run is green, or every remaining failure is a documented,
      dated pin with a repro report.
- [ ] Every capability claim in the catalog was verified against live
      behavior, and every conflict with fabro or the listing is a dated
      comment.
- [ ] Provider bugs are written up and handed off.
- [ ] The changelog has an entry.

## Appendix: OpenRouter notes

- Source: fabro's `openrouter.toml` — models across the claude, gpt,
  gemini, deepseek, kimi, qwen, glm, minimax, mimo, laguna, and devstral
  families. `base_url = "https://openrouter.ai/api/v1"`, bearer
  auth from `OPENROUTER_API_KEY`, priority 25. Seven Claude rows already
  carry `cache_control_breakpoints = true`, which maps straight to
  `cache_breakpoints`.
- Roster: apply the union policy against the Venice roster — check the
  live listing for Venice-only models (grok is the notable one; the
  Claude, GPT, kimi, glm, deepseek, and qwen families are already in
  fabro's 29) and add rows for the ones OpenRouter serves.
- Wire ids are namespaced (`anthropic/claude-opus-5`). Short catalog id,
  namespaced `api_model`. Selectors then read
  `openrouter/anthropic/claude-opus-5`, which the resolver splits
  correctly on the first `/`.
- The listing's `supported_parameters` per model is unusually good
  verification data — use it to seed capability flags, then spot-check
  behavior.
- Wire quirks already handled in `src` and worth asserting live: in-band
  `usage.cost` (`CostSource::Provider`), numeric error codes in error
  bodies.
- Verify how OpenRouter treats `prompt_cache_key` before claiming
  `cache_routing`, and probe one cache write-then-read pair per upstream
  family — caching semantics differ per underlying provider.
- OpenRouter routes one model across several upstream hosts, and prompt
  caching only lands when consecutive requests hit the same host. Fabro's
  measurements showed that pinning the upstream (the `provider.only`
  request field) cut a 20k-token repeat prompt's cost by 81%. If probe
  results look inconsistent between calls, pin the routing through
  `default_options` in the E2E catalog before concluding anything, and
  consider row-level routing pins for the built-in catalog where caching
  matters.
