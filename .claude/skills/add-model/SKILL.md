---
name: add-model
description: Add a newly released model (a new Claude, GPT, Gemini, Kimi, GLM, or other release) to lithos-llm's built-in catalogs on every provider that serves it, with live-verified capability flags, prices, and E2E cells. Use this whenever the user asks to add, support, or catalog a new model or model version in this repo, says a model "just came out" or links a model's release notes or docs page, asks why `provider/some-new-model` resolves to passthrough, or wants a model moved onto more providers — even if they don't say "catalog" or "E2E". Not for adding a whole new provider (that is docs/adding-a-provider.md).
---

# Adding a model to the built-in catalogs

A new model usually lands on several providers at once: the vendor's own API
plus the gateways (OpenRouter, Venice, Vercel) and sometimes Bedrock. Each
provider serves it a little differently, and the listings lie often enough
that every flag needs evidence. The result of this workflow is one commit per
provider — catalog row, E2E catalog row, E2E cells, recording where the
provider has a twin — plus a changelog commit, all backed by live runs.

Before the first edit, follow the repo's CLAUDE.md: run `bin/style-guides
prepare` and read the Rust style guide it points to. That rule covers
configuration as well as Rust, so it applies even when the change is TOML only.

## 1. Gather the facts

Read the vendor's release notes, "what's new", and migration pages first.
They name the breaking changes, and breaking changes are exactly what the
catalog flags encode: forced tool choice removed, thinking that can't be
disabled, sampling parameters rejected, new effort levels, price changes,
cache-read rates, fast or priority tiers, platform availability and wire ids.

Then find which built-in providers carry the model's family:

```sh
grep -ln '<family-or-previous-model>' src/catalog/builtin/*.toml
```

Pull each provider's live listing and look for the new id — see
`references/probes.md` for the exact commands. A provider that lists the
model is in scope; one that doesn't is reported, not guessed at. The user
usually says "all providers that support it", which means every provider
whose listing (or, for Bedrock, the vendor's docs) shows the model.

## 2. Probe what the listing can't tell you

Listings are a starting point, not evidence. On 2026-09-22 OpenRouter listed
`tool_choice` for Claude Opus 5.5 while the upstream returned a 400 for any
forced choice; Vercel's listing said `temperature: false` while the gateway
accepted one. Probe each provider directly (commands in
`references/probes.md`) for every flag the release notes suggest might have
changed. The recurring ones:

- **Forced tool choice.** Send `required` and a named choice with a prompt
  that has no reason to call the tool ("Write a haiku about the sea"). A
  weather prompt would call the tool under `auto` anyway and hide a silent
  downgrade. Three outcomes: a real forced call (claim it), a 400 (deny
  it), or a 200 with plain text (deny it — a gateway that silently drops the
  choice is worse than one that errors, because the caller asked for a call
  and got text).
- **Sampling.** Send a temperature. A 400 means `sampling` is unclaimed; a
  200 on a gateway means claim it even when the listing says otherwise.
- **Effort / thinking.** Try each effort level, and on Anthropic try
  `thinking: {"type": "disabled"}` if the notes say thinking is always on.
- **System turns** (Anthropic codec). A `role: system` message must follow a
  `user` turn, so put it after the last user message when probing.
- **Speed tiers.** Fast mode needs its beta header. This org's fast-mode quota
  is close to zero, so a 429 "rate limit of 0 fast mode input tokens" is an
  account limit, not a codec bug; one successful call with `usage.speed:
  "fast"` is enough evidence.

## 3. Write the rows

For each provider, add the row to both `src/catalog/builtin/<provider>.toml`
and `tests/e2e/<provider>_catalog.toml`. Copy the nearest sibling row (the
previous model in the family on the same provider) and change only what the
evidence says changed. Things to get right:

- Catalog id is the short human id with a dot (`claude-opus-5.5`);
  `api_model` is the provider's wire id (`claude-opus-5-5` on Anthropic and
  Venice, `anthropic/claude-opus-5.5` on the gateways, `us.anthropic.…` on
  Bedrock).
- Prices are USD micros per million tokens (`$4/MTok` → `4000000`). Use the
  provider's own listing: Venice marks up Anthropic models (1.2x for Opus
  5.5), the other gateways pass list price through.
- Deny forced choice with `tool_choice = { required = false, named = false }`.
- Put a comment above the row with the date, the sources, and each
  surprising probe result. Future readers re-probe from that comment, so
  write the evidence, not just the conclusion.
- Add a dated line to the file's header comment, like the existing Fable 5.1
  and Opus 5.5 lines.
- Place the row next to its family, newest first.
- Watch for tests that list a provider's roster by name. A new `openai` row
  fails `the_codex_roster_is_the_verified_platform_subset`
  (`tests/e2e/openai/codex.rs`, not ignored, so `mise run test` goes red):
  either add a matching row to `openai-codex.toml` and
  `tests/e2e/openai_codex_catalog.toml`, which needs a live Codex check, or
  add the id to that test's exclusion list with a comment saying why. Search
  for others with `grep -rn '"<sibling-id>"' tests/ src/ --include=*.rs`.

**Aliases** (`opus`, `sonnet`, `fable`, …) stay on the old model unless the
user says to move them. Moving an alias changes behavior for every caller
using it — for example a forced tool choice through `opus` started failing
locally when the alias moved to Opus 5.5. When the user does want them moved:

- Write the new row without the alias. Move the alias on a provider only
  after that provider's own cells for the new row have passed (live for
  Anthropic, recorded for a gateway). Moving it on every row at the start
  and then recording is the easy mistake to make.
- Move it on every provider that has it: `grep -n 'aliases = \["<alias>"'
  src/catalog/builtin/*.toml tests/e2e/*_catalog.toml`.
- Search the tests for anything that pins where the alias resolves, such as
  `tests/e2e/cli/resolve.trycmd` (it pins `claude-sonnet` →
  `anthropic/claude-sonnet-5`), and update them in the same commit.
- Ask whether the provider's `default_model` should move too. It is a
  separate setting from the alias (Anthropic, OpenRouter, Vercel, and Bedrock
  default to `claude-sonnet-5`), and the user may want one without the other.

Providers you can't reach (Bedrock without AWS credentials) get a row from
the docs only when the user wants it; say in the comment and the commit that
it is unprobed, and leave pricing off if you can't find an official price
rather than copying another platform's.

## 4. Add the E2E cells

Mirror what the previous model in the family has. Search the provider's E2E
module for the sibling's cell name and add the new model beside it:

```sh
grep -rn 'claude-fable-5.1\|claude_fable_5_1' tests/e2e/<provider>/
```

Typical places: the `model_tests!` roster in `mod.rs`, `family_tests!` when
the release changes behavior the family cells test, effort-level cells, the
URL-vision list, effort probes, and reasoning round trips. A model that takes
no forced choice needs the `…_under_auto_choice` round trip, because the
shared runner forces a call. When the model denies a capability because the
upstream rejects it, a negative cell that sends the request anyway through a
one-row unrestricted catalog pins the rejection, so the flag comes off when
the vendor lifts it (see `tests/e2e/anthropic/negative.rs`).

Keep existing cell names and requests byte-identical. Recorded scenarios are
keyed by a hash of the request body, so changing a shared runner re-keys every
recorded cell that uses it — add a new cell instead.

## 5. Run live, record, replay

Anthropic's native suite is live-only. Run the new cells directly:

```sh
set -a; . ./.env; set +a
LITHOS_E2E_BACKEND=live cargo nextest run --locked --all-features --test e2e \
  --run-ignored all -E 'test(/anthropic::/) & (test(/<cell_suffix>/) | test(/preflight/))'
```

Gateway providers record through a twin proxy. Record only the new cells, in
append mode, so every existing scenario stays byte-identical — never the
full `mise run test:e2e:record`, which re-records everything and spends
credits on it. The ports, upstreams, and key variables are in
`references/probes.md` (they mirror `mise.toml`'s record task):

```sh
cp tests/e2e/recordings/<p>.json "$SCRATCH/<p>.before.json"
LITHOS_E2E_TWIN_MODE=record LITHOS_E2E_RECORDING_APPEND=1 \
  LITHOS_E2E_TWIN_ADDR=127.0.0.1:<port> \
  LITHOS_E2E_RECORDING_PATH=tests/e2e/recordings/<p>.json \
  LITHOS_E2E_UPSTREAM_URL=<upstream> LITHOS_E2E_API_KEY_VARIABLE=<KEY> \
  ./target/debug/examples/e2e_twin &
# wait for http://127.0.0.1:<port>/healthz, then:
LITHOS_E2E_BACKEND=record cargo nextest run --locked --all-features --test e2e \
  --run-ignored all --success-output final -E 'test(/<p>::/) & test(/<cell_suffix>/)'
kill %1
git diff --stat tests/e2e/recordings/<p>.json   # must be insertions only
git diff tests/e2e/recordings/<p>.json | grep '^-[^-]'   # must print nothing
```

Build the twin first (`cargo build --locked --all-features --example
e2e_twin`). `--success-output final` keeps the effort-probe lines ("probe:
… reasoning tokens N"), which are the evidence for effort claims.

Then replay the provider's whole suite offline against the twin in replay
mode; every cell, old and new, must pass. A deletion in the recording diff
means an existing request changed — find out why before going further.

When a cell fails, decide whether the flag or the cell is wrong. A skipped
cell ("the catalog does not claim …") is expected for denied capabilities.

## 6. Commit, changelog, gate

- `mise run fmt`, then the unit suites:
  `cargo nextest run --locked --all-features --lib --test it --test runtime --test contracts`.
  `wire::bedrock::a_signed_request_differs_only_in_authentication` sometimes
  times out at 15s; rerun it alone before blaming the change.
- One commit per provider. Per the user's rules the message explains the old
  behavior (the model resolved to passthrough), why that fails, and the fix,
  including the surprising probe results and the cell and replay counts.
  Never amend.
- A final commit adds one Unreleased bullet to `CHANGELOG.md`: the providers,
  any alias move, how each route treats the notable capability, prices, and
  which rows were verified live versus docs-only.
- `mise run check` must pass before the user is told it is done.
- Open a PR only when asked; branch off `main` first if the commits are on
  `main`.

## Report back

Tell the user, in plain language: which providers got rows and how each was
verified, a small table of price and the notable capability per provider,
anything left unverified (and what would verify it), any alias decision still
open, and anything odd you noticed on neighboring rows (a stale price, a
listing that changed) without fixing it unasked.
