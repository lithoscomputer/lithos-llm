# Changelog

All notable changes to `lithos-llm` will be documented in this file.

This project follows [Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- The built-in OpenAI catalog now includes GPT-6 Astra (`gpt-6-astra`) with
  the published 1,050,000-token context window, 128,000-token output limit,
  pricing, long-context threshold, cache-write rate, speed tiers, and
  capability claims. The `astra`, `gpt-astra`, `gpt6`, and `gpt-6` aliases
  resolve to it. These facts come from the official OpenAI documentation on
  2026-09-03. API access was still rolling out: an authenticated model listing
  omitted Astra, and a live lithos-llm request returned 404 `model_not_found`.
  GPT-5.6 Sol therefore remains the provider default, and Astra is not yet in
  the recorded per-model E2E cells.

- The built-in Anthropic, OpenRouter, and Venice catalogs carry Claude
  Fable 5.1 (`claude-fable-5.1`, wire ids `claude-fable-5-1` on Anthropic
  and Venice, `anthropic/claude-fable-5.1` on OpenRouter), verified live on
  2026-09-01: Fable 5's limits, effort levels, and rates, with cache reads
  at $0.25 per million tokens instead of $1.00. Venice lists it at
  Anthropic's own rates rather than its marked-up Fable 5 rates. Claude
  Fable 5 stays in every roster, and the `fable` and `claude-fable` aliases
  still resolve to it. The live suites cover the new rows end to end —
  roster, family, effort-level, vision, and structured cells on Anthropic;
  roster, effort-probe, and reasoning round-trip cells on OpenRouter and
  roster, family, effort-probe, and round-trip cells on Venice, both
  recorded for offline replay — plus two
  cells specific to what changed: one pins Anthropic's 400 for a forced
  tool choice on Fable 5.1, and one replays a signed thinking block under
  the `thinking-binding-controls-2026-08-01` beta with
  `prefix_mismatch_behavior: "error"`, proving the codec's merged tool
  results and moved cache markers pass the model's preserved-thinking
  prefix check.

- `ModelCapabilities::system_turns` records that a model's endpoint takes
  `role: "system"` messages inside the conversation. The Anthropic codec used
  to hoist every system and developer message into the top-level `system`
  field, so an instruction an agent loop appended mid-conversation rewrote
  that field between requests — and on Claude Fable 5.1, whose thinking
  blocks are bound to that prefix, invalidated every earlier thinking block
  (a 400 for accounts created on or after 2026-08-31). Where the row claims
  the capability, only the leading run of system messages is hoisted and a
  later one is sent in place as a `system` turn; everywhere else hoisting is
  unchanged. Live probes on 2026-09-01 settled the rows: Claude Fable 5.1,
  Fable 5, Opus 5, Sonnet 5, and Opus 4.8 take the turn (Sonnet 5 contrary
  to its launch notes), and Opus 4.7 and every older row reject it. The
  Anthropic live suite exercises a mid-conversation system message on every
  roster row, pins the rejection on Haiku, and the Fable 5.1 enforcement
  round trip now appends one after the tool result, so the prefix check has
  to pass with a system turn present. The OpenAI-compatible codec already
  carried system messages in place, but its Anthropic-style system cache
  breakpoint landed on the *last* system message, so an appended
  instruction moved the breakpoint every turn and the prefix written a turn
  earlier was never read back; the breakpoint now stays on the leading run.
  The OpenRouter and Venice suites gain a recorded family cell for the
  shape. Venice's Claude rows skip it: Venice's translation rejects a cached
  system prompt followed by a second system message (`system: text content
  blocks must contain non-whitespace text`, probed 2026-09-01), which a
  negative cell pins until Venice fixes it.

- `ModelCapabilities::forced_tool_choice` records whether a model takes a
  forced tool choice (`required`, or one named tool). It defaults to true,
  so every existing row and every application-supplied catalog keeps its
  behavior; Claude Fable 5.1 is the first row to deny it. The client refuses
  `required` and named choices before dispatch on a row that denies the
  capability, with the same `unsupported_capability` error an unclaimed
  sampling control gets, while `auto` and `none` are unaffected.
  `ToolChoice::is_forced` names the distinction for callers.

- The wire suites now cover the last of the reference implementation's
  request corpus: Codex-mode streaming of a blocking call with sampling
  controls dropped and a `reasoning.effort` field on OpenAI Responses, a
  `status: incomplete` document and an Anthropic `max_tokens` stop decoding
  as `Length`, a Venice top-level cost inside a stream, and streams that end
  without `[DONE]` or a Gemini `finishReason` completing once as
  `incomplete`. Every intentional departure from the reference is now
  explained beside its fixture.

- The serialized forms of `Request`, `Response`, every `StreamEvent`,
  `ErrorData`, `TokenCounts`, and the `ErrorKind` and `RetryClassification`
  vocabularies are pinned by snapshot tests in `tests/serialization.rs`.
  Applications persist and stream these shapes, so a changed snapshot is a
  compatibility break to review as one, not test noise to accept.

- `Client::probe` (and `probe_with_context`) runs a cheap diagnostic against
  one model selector and returns a `ProbeReport` — the resolved route, the
  latency, the summed usage, and a `ProbeOutcome` — instead of an error.
  The default probe sends one short prompt; `ProbeOptions::tools(true)` runs
  an `add`-tool exchange and checks that the model calls the tool and reaches
  the right total. `Failed` carries the classified `ErrorData`, so catalog
  rejections, authentication failures, unknown models, and timeouts are told
  apart by the existing `ErrorKind`; `Incorrect` marks a model that answered
  but skipped the tool or got the total wrong. The probe honors an outer
  cancellation or deadline through `CallContext`.

- The optional `lllm` CLI exposes that diagnostic as `lllm probe`. It follows
  the normal model-selection precedence and supports basic and fixed add-tool
  probes, reasoning effort, a whole-probe timeout, compact text output, and a
  versioned JSON report. Passed probes return status 0. Failed and incorrect
  probes return status 1, so scripts can use the command as a health check.

- A local token estimator, `lithos_llm::estimate`, sizes text, messages,
  content parts, tool definitions, request controls, and whole requests
  synchronously and deterministically, with no provider call and no feature
  flag. Each `TokenEstimate` carries typed `EstimateWarning`s — `Media`,
  `OpaqueContent`, `ProviderOptions` — naming the inputs it could only
  approximate, and estimates add together. The estimator is heuristic;
  `Client::count_input_tokens` stays the authoritative path and never
  substitutes an estimate.

- Retries are observable. `Observer` gains a defaulted `on_retry` hook, and
  `RetryMiddleware::observer` (or `observer_arc`) reports every retried
  attempt through it: the failed attempt number, its error, the wait before
  the next attempt, and a `RetryStage` — `Request` for a request that never
  became a stream, `Stream` for a stream that failed before it delivered
  visible output. An application that mirrors these retries onto its own
  event stream can name the phase without inferring it. Refused retries are
  not reported; their error reaches the caller. The retry decision itself,
  `RetryPolicy::next_delay`, is now public so an application-owned retry
  loop — such as replaying a turn after a stream already delivered visible
  output — can share the middleware's policy instead of reimplementing
  backoff.

- Every built-in provider and every model whose family differs from its
  provider now carries a `metadata.pebble.profile`: the agent profile an
  agent runtime prompts and formats tools with. The values are `anthropic`,
  `claude-5`, `openai`, `gemini`, `kimi`, and `gpt56`, mirroring fabro's
  profiles; the profile follows the model, so a Claude row on an
  OpenAI-compatible provider says `claude-5`. A model row overrides its
  provider's value. The existing `fabro` metadata namespace is unchanged.

- The built-in catalog now includes Fireworks with live-verified model routes,
  pricing, capabilities, and conventional `FIREWORKS_API_KEY` credentials.
- The built-in catalog now includes Modal as a portable passthrough provider.
  It reads `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` and resolves the
  workspace-specific endpoint hostnames returned by Modal.
- Anthropic and Gemini now have current, live-verified model rosters, pricing,
  limits, and capability claims. Their native E2E suites cover completion,
  tools, structured output, reasoning, caching, media, and error behavior.
- Gemini reasoning effort now maps to the native low, medium, and high
  `thinkingLevel` values without enabling thought-summary disclosure.
- The built-in catalog now includes Moonshot AI with a live-verified Kimi K3
  route, pricing and capability data, plus conventional
  `MOONSHOT_API_KEY` credentials with `KIMI_API_KEY` as a fallback.
- The built-in catalog now includes OpenRouter with live-verified model routes,
  provider-reported cost support, and conventional
  `OPENROUTER_API_KEY` credentials.
- The Anthropic and Gemini E2E suites pin native token counting live: every
  roster model answers `count_input_tokens`, which also confirms Gemini's
  `countTokens` body shape (the `model` field nested inside
  `generateContentRequest`) against the real API.
- The built-in OpenAI provider gained a live-verified GPT-5 roster:
  the three GPT-5.6 models, GPT-5.4, GPT-5.5, both Pro models, and
  GPT-5.4 Mini, with pricing (long-context tiers above 272K input,
  priority and flex speed tiers, and the 5.6 family's billed cache
  writes), limits, per-model sampling and reasoning-effort claims, and
  fabro metadata — every claim checked against the live listing, the
  developer docs, and raw `/v1/responses` probes. The pro rows claim no
  speed tier because OpenAI silently downgrades their priority requests.
- Every OpenAI roster row claims `documents`, verified live per row on
  2026-08-30: an inline PDF rides `input_file` as `file_data` with a
  `filename`, and `file_url` fetches a remote PDF. The live API requires
  the filename — omitting it draws 400 "Missing required parameter" — so
  the codec now refuses an inline document without a file name before
  dispatch instead of sending a request the provider rejects. The E2E
  document cells record and replay through the twin, whose pin moved to a
  revision that accepts `input_file` parts (lithoscomputer/twins#7 carries
  the same fix to main).
  The catalog header documents the second access path — the ChatGPT
  subscription's Codex deployment — and the overlay entry that selects
  it, since the two paths serve different rosters with different field
  sets.
- The OpenAI live E2E suite covers completion, streaming, tools,
  structured output, reasoning (with each model's live effort
  vocabulary), caching, vision, sampling, native token counting, and
  error classification across the eight-model roster, with a committed
  record/replay recording behind a fifth twin.
- The Codex E2E suite (`tests/e2e/openai/codex.rs`) drives the full
  client through the ChatGPT-subscription deployment across its
  six-model roster — completion through the forced stream with hoisted
  instructions, streaming, multi-turn history, forced tools surviving
  the deployment's empty terminal `output` array, structured output,
  vision, each model's effort vocabulary, and the encrypted-reasoning
  tool round trip — plus the envelope cells: the dropped output cap's
  warning, the local sampling refusal, the seat roster's pro-row
  refusal, and the local no-native-count answer. The suite records and
  replays behind a sixth twin (the twins `codex-passthrough` revision
  rebases the upstream path, forwards the seat headers, and classifies
  the deployment's header-less SSE responses by the request's stream
  flag); replay runs offline in `mise run check` from the committed
  recording, recording needs `OPENAI_CODEX_TOKEN` and
  `CHATGPT_ACCOUNT_ID` exported, and live runs unproxied with the same
  variables. The wire suite pins the codex dialect offline too: the
  unversioned request path, the tool round trip with hoisted
  instructions, and a tool call assembled from streamed items when the
  terminal document's `output` is empty.

### Changed

- A tool call the model's output limit cut short is no longer a tool call.
  Every codec drops it from `Response::content`, on the blocking path and on
  the stream's completed response alike, and adds a `truncated_tool_call`
  warning naming the tool. The finish reason stays `length`, so a consumer
  branches on it alone and never sees a call with half its arguments. The
  wire shapes were verified live: OpenAI Responses reports the item as
  `incomplete` with a truncated JSON prefix, Chat Completions reports
  `finish_reason: length` beside the prefix, and Anthropic reports
  `max_tokens` beside an empty `input`. Before this, `parse_arguments`
  turned the prefix into `{}` and the call reached the consumer as a
  complete call with no arguments. The stream's block events still deliver
  the call as it arrives, and the provider's item stays in `Response::raw`.
  A malformed argument string on any other finish is unchanged.

- The root `lithos-llm` package now provides the `lllm` executable behind the
  optional `cli` feature. It replaces the separate, unpublished
  `lithos-llm-cli` workspace package and the former `lithos` executable name.

- Gemini deliberately declares no speed pricing tiers: the protocol has no
  speed control, so `fast` and `economical` requests on its priced models
  fail locally before dispatch. The OpenRouter Gemini 3.1 Pro long-context
  tier states its cache-write rate explicitly (OpenRouter's listing prices
  `input_cache_write` with no long-context override).

- The built-in `default` selector now resolves Anthropic's Claude Sonnet 5,
  following the provider priority in the source catalog.
- Provider catalog tests now check resolution and behavior without copying
  exact roster counts or model tables.
- The minimum supported Rust version is 1.88 (from 1.85): the codecs now
  use let-chains, which need it.

### Fixed

- The OpenAI Responses codec builds the reasoning part of a `reasoning`
  output item from its `content` entries of type `reasoning_text` — the
  trace itself — joined by a blank line, and falls back to its `summary`
  blocks of type `summary_text`, joined the same way, when there are none.
  It used to concatenate every `summary` block and every `content` entry
  with nothing between them, so an item with two summary blocks and no
  content produced a run-together copy of the summary that a consumer
  comparing it against the summary on the opaque `openai.reasoning` item
  took for a distinct trace; with the same separator the two agree.
  Entries of any other type are skipped. The stream decoder agrees:
  `response.reasoning_text.delta` and `response.reasoning_summary_text.delta`
  both feed the reasoning block live, a fragment that opens a later entry
  of the same list carries the blank-line separator, and an item that
  streams both lists is settled on its `content` alone by the terminal
  item, so a streamed and a blocking decode of the same item yield the
  same parts.
- Anthropic JSON-object responses ask for free-form JSON through a system-text
  instruction. The earlier closed object schema admitted only the empty
  object, so provider-enforced structured output discarded the answer.
- The OpenAI Responses codec decodes
  `usage.input_tokens_details.cache_write_tokens` into the cache-write
  bucket. The GPT-5.6 family bills cache writes at 1.25x input; the
  decoder's hardcoded zero silently priced those tokens at the plain
  input rate.
- The OpenAI Responses codec no longer sends stop sequences: the live
  `/v1/responses` endpoint rejects a `stop` member with a 400 on every
  model, so any request carrying one failed outright. The sequences are
  dropped with an unsupported-control warning instead, and raw provider
  options remain the escape hatch for a compatible skin that takes one.
- Codex mode posts to the unversioned `<base>/responses` path. The codec
  appended `/v1/responses` there too, which the live Codex deployment
  answers with an HTML 403, so the documented codex overlay could never
  complete a request.
- Codex mode reports no native token count instead of posting to a
  `/responses/input_tokens` path the Codex deployment does not serve.

### Round-6 parity fixes

A sixth differential review against the reference implementation
(`.ai/reviews/lithos-llm-vs-fabro-llm-round-6.md`) landed these:

- Anthropic `ResponseFormat::JsonObject` no longer encodes an
  `output_config.format` schema that only admitted the empty object; the
  JSON-only instruction rides on the system text, as the reference did.
- The OpenAI Responses codec sends `service_tier: "priority"` for
  `Speed::Fast`; the invalid `"fast"` drew a provider 400.
- The client again rejects a request's `speed` before dispatch when the
  catalog prices no such tier for the model, restoring the reference's local
  `InvalidRequest` instead of a wasted provider round trip. `balanced` and
  models the catalog does not price (including passthrough) are unaffected.
- A server-side deadline expiry carried on a 5xx status (Gemini's 504
  `DEADLINE_EXCEEDED`) classifies as a non-retryable timeout instead of a
  retryable server error, so already-executed and possibly billed work is
  not re-sent.
- Streaming no longer leaks a provider's sealed redacted-reasoning payload
  (opaque base64) through live `ReasoningDelta` events on Anthropic and
  Bedrock streams; the blob still arrives whole on the block-end part with
  `redacted` set.
- Catalog cost estimation refuses to stamp a cost when the input or output
  base rate is missing for a non-empty token bucket, and a long-context
  pricing tier inherits unstated rates from the base instead of dropping
  them.
- A raw `anthropic` `thinking` provider option sent with a forced tool
  choice is reported as an unsupported control instead of silently building
  a request Anthropic rejects.
- The OpenAI-compatible stream decoder no longer fails a stream whose first
  tool-call fragment carries only arguments; the slot opens on the arguments
  and the identity is accepted from later fragments.
- A `response.failed` stream event without a `response` wrapper keeps its
  provider error code and message.
- Streams from lenient OpenAI-compatible skins that separate SSE events with
  single newlines parse again: the `openai_chat` and `gemini` codecs frame
  their streams at each complete `data:` line, as the reference transport
  did.
- Bedrock Converse encoding coerces every non-object tool-call argument
  value to `{}` in `toolUse.input`, not just `null` — Converse requires an
  object document.
- A decoded Gemini function call in a payload without a `responseId` gets a
  per-response nonce in its synthesized id, so repeated calls to the same
  tool across turns no longer collide or replay duplicate ids.
- A successful Gemini tool result that is one JSON object is sent verbatim
  as the whole `functionResponse.response` struct again; non-object values,
  text, and errors keep the `output`/`error` wrapping.
- Gemini URL media without a declared media type sends `fileData.mimeType`
  defaulted by attachment kind (`image/png`, `audio/wav`,
  `application/pdf`), which Vertex-style surfaces require.
- Blocking OpenAI-compatible requests omit the `stream` member instead of
  sending `"stream": false`, and a JSON tool result whose value is a bare
  string is sent as the raw text rather than a quoted JSON literal, both
  matching the reference encoder's wire shape.

### Round-5 parity fixes

A fifth differential review against the reference implementation
(`.ai/reviews/lithos-llm-vs-fabro-llm-regressions.md`, punch list in
`.ai/plans/lithos-llm-punch-list-5.md`) landed these:

- The stream retry drops the failed attempt's stream before it sleeps and
  reconnects. The abandoned stream could hold a resource a layer below
  the retry releases only on drop — the concurrency limiter's permit —
  so a retry inside a limit-1 limiter deadlocked waiting on its own
  abandoned attempt.
- A raw `thinking` provider option replaces the Anthropic codec's derived
  thinking object instead of merging into it, which left a stray
  `budget_tokens` on `{"type": "disabled"}` that the API rejects, and
  `max_tokens` stays unlifted when the caller overrides thinking.
- Provider-scoped selectors resolve wire model ids: `provider.model(..)`
  falls back to `api_model` after canonical ids and aliases, so
  `bedrock/us.anthropic.claude-sonnet-4-6` finds its catalog entry
  (pricing, capabilities, limits) instead of a passthrough model.
- Chat messages with no content parts omit the `content` member instead
  of sending `"content": null`, the shape the reference client sent and
  strict OpenAI-compatible skins require on tool round-trips.
- The Chat stream decoder reads array-form content deltas through the
  same both-shapes helper the blocking path uses; they were silently
  discarded, completing such streams as empty successes.
- `MediaSource::Url` can carry a media type
  (`MediaSource::url_with_media_type`), and the Gemini codec sends it as
  `fileData.mimeType`, which Vertex-style surfaces require.
- The Responses codec no longer warns "replaying reasoning text" on its
  own round trip: the warning fires only when a reasoning part has no
  opaque `openai.reasoning` sibling replaying the same text.
- A streamed Gemini blocked prompt fails as a ContentFilter error through
  the same classifier the blocking path uses, instead of streaming as an
  empty success.
- Catalog cost estimation refuses to price a call whose cache-read or
  cache-write tokens have no catalog rate, rather than billing those
  tokens at zero and reporting a confidently wrong figure.
- The Responses codec keeps media out of system and developer input
  items — the protocol accepts only `input_text` there — and reports the
  drop with the non-text-system warning in standard mode as well as
  Codex mode.
- Errors expose the provider's advised wait separately from the retry
  classification (`Error::provider_retry_after`,
  `ErrorData::provider_retry_after_millis`), so a never-retried failure
  such as a spent-quota 429 still carries its `Retry-After` hint.
- The Responses codec skips history tool calls with an empty name, which
  previously encoded as `"name": ""` and drew a provider 400.
- The Bedrock stream keeps text and `redactedContent` reasoning deltas
  apart: text after a sealed blob is dropped (the blob is what the
  provider verifies on replay), and a blob after text fails the stream
  retryably instead of assembling a corrupted sealed payload.
- Decided and recorded (no code change): the legacy
  `prompt-caching-2024-07-31` beta header stays off Anthropic requests.
  The GA API does not need it, and a gateway that still does can add it
  through catalog `default_headers` or the `beta_headers` provider
  option.

### Built-in catalog: Venice

- The built-in catalog now ships one TOML file per provider
  (`src/catalog/builtin/`), each its own named layer, so a catalog error
  names the exact file and the catalog grows one provider at a time.
- A Venice provider joins the built-in catalog: sixteen models merged from
  fabro's `venice.toml` and the live E2E findings, with live-verified
  api_model ids, limits, capability corrections, cache flags, Venice's
  listed prices, and `default_options` turning off the injected system
  prompt. `EnvironmentCredentials::conventional()` (and so
  `Client::from_env()`) reads `VENICE_API_KEY`.

### Venice live-testing findings

The first live E2E runs against Venice
(`.ai/plans/live-e2e-test-matrix.md`, production plan in
`.ai/plans/venice-production-fixes.md`) landed these:

- A catalog provider row can declare `default_options`: raw request options
  for the provider's own namespace, applied to every request, with
  request-level `provider_options` winning key by key. The E2E Venice
  catalog uses it to turn off Venice's injected system prompt
  (`venice_parameters.include_venice_system_prompt = false`), which
  otherwise costs every request roughly two thousand prompt tokens.
- The Chat codec decodes Anthropic-style cache writes: a skin fronting a
  Claude model reports `cache_creation_input_tokens` (Venice sends it
  nested and flat), which previously decoded as a zero `cache_write`
  bucket.
- A complete Chat response that carries tool calls finishes as `ToolCall`
  when the wire said `stop`, matching the rule the streaming path already
  applied (qwen on Venice answers forced tool calls this way).
- A typed cache-routing hint: `Request::cache_hint` takes
  `CacheHint::{Auto, Key, Disabled}`, with `cache_key(..)` as builder
  shorthand. Where a model row claims the new `cache_routing` capability,
  the default sends a stable fingerprint of the system messages and tool
  definitions as `prompt_cache_key` on the OpenAI-style protocols, so
  every turn of a conversation routes to the backend session holding its
  cache entry (OpenAI documents the field; Venice maps it to session
  affinity on several of its backends). `auto_cache: false` or
  `CacheHint::Disabled` turns it off; a raw `prompt_cache_key` in
  `provider_options` still wins.

### Round-4 parity fixes

A fourth differential review against the reference implementation
(`.ai/reviews/lithos-llm-vs-fabro-llm-round-4.md`, punch list in
`.ai/plans/lithos-llm-punch-list-4.md`) landed these on top of the round-3
work:

- Structural decode failures and garbled stream data are classified
  retryable (`Safe`) on every codec, extending the round-3 Chat rule to
  Anthropic, Gemini, Bedrock, and OpenAI Responses — a 200 missing the
  fields its protocol requires, a stream event that is not JSON, and every
  Bedrock event-stream framing failure (checksum, length, header block,
  non-UTF-8 payload). The reference client retried all of these; stream
  failures still stop retrying once visible output has streamed.
- A tool result whose text is empty — a command with no stdout — encodes as
  the empty string on both OpenAI codecs, as the reference client sent it,
  instead of a serialized `ContentPart` envelope the model would read as
  the tool's answer.
- A passthrough (uncataloged) model with a `reasoning_effort` takes the
  modern `output_config.effort` dialect on the Anthropic and Bedrock
  codecs, restoring the reference client's guess for unknown models. The
  conservative passthrough capabilities previously routed it to a manual
  thinking budget, which the always-adaptive models — the models
  passthrough exists to reach — reject with a 400.
- A Responses `refusal` content part is a classified `ContentFilter` error,
  blocking and streaming, instead of a successful empty response — the last
  codec brought under the refusal-as-error contract (Anthropic and Bedrock
  in round 1, Chat in round 2). A json_schema request the model refuses now
  fails visibly and failover-eligibly instead of handing the caller empty
  content.
- Anthropic: an `input_json_delta` for a block whose `content_block_start`
  never arrived fails the stream retryably instead of fabricating a
  nameless tool call — the contract the Chat and Bedrock codecs already
  apply.
- Chat: a streamed `tool_calls` fragment without a valid `index` fails the
  stream retryably, as the reference decoder did, instead of defaulting to
  slot 0 — which silently merged parallel calls into one call with the
  second call's identity and garbled arguments.
- Anthropic: a thinking signature arriving on `content_block_stop` replaces
  the captured one, as the reference decoder preferred it, so a dialect
  sending the signature only there no longer closes the block unsigned.
- Chat: several replayed reasoning parts join into `reasoning_content`
  unseparated, byte for byte what the reference client sent, instead of
  with a `"\n\n"` separator — the same rule the text join follows.
- Responses: a lost `output_item.added` for a model-internal call no longer
  leaves a phantom nameless `ToolCall` part in the streamed content or
  flips the finish reason — the fallback-latched block closes for
  consumers but its part is discarded, matching blocking decode of the
  same body.
- Chat and Gemini streams treat an explicit `"error": null` member on a
  chunk as absent instead of failing the stream, so a skin that spells the
  field out on success chunks keeps working.

### Round-3 parity fixes

A third differential review against the reference implementation
(`.ai/reviews/lithos-llm-vs-fabro-llm-round-3.md`, punch list in
`.ai/plans/lithos-llm-punch-list-3.md`) landed these on top of the round-2
work:

- Chat structural decode failures — a 200 with no choices, a choice without a
  message object, a tool call missing its id or name, and a stream chunk that
  is not JSON — are classified retryable (`Safe`), restoring the reference
  retry behavior for transiently garbled bodies.
- A Chat stream that carried tool calls finishes as `ToolCall` when the wire
  said `stop` or reported no reason, matching the Responses codec's rule and
  the reference client; an explicit non-stop reason such as `length` still
  wins.
- Complete-response `reasoning_details` are preserved verbatim — order,
  count, and shape — instead of passing through the stream coalescer, which
  merged same-type unindexed entries and dropped the second entry's
  signature.
- A Responses stream reconciles every block with its terminal
  `output_item.done` item: a delta lost in transit has its missing tail
  delivered, and a buffer the deltas garbled is replaced, so streaming and
  blocking decode the same response identically instead of keeping silently
  truncated tool arguments.
- Migration compatibility: histories the reference implementation persisted
  replay on the Responses codec — the underscore opaque kinds
  `openai_reasoning` and `openai_message` are read like this crate's dotted
  kinds, and a tool call's un-namespaced `id` metadata still restores the
  `fc_` item id, so `store: false` tool calling keeps its reasoning chain.
- Migration compatibility: `TokenCounts` deserializes the reference
  implementation's `input_tokens`-style field names as aliases, so persisted
  usage keeps its values instead of silently loading as all-zero buckets.
- Migration compatibility: the Chat codec replays the legacy
  `openai_compat_reasoning_details` opaque kind into `reasoning_details`,
  so histories the reference implementation persisted keep their signed
  reasoning.
- Bedrock: a thinking budget always travels with an explicit `maxTokens`
  (request limit, else the catalog limit, else 65,536; lifted when the
  budget would not fit under it), so a request without a caller limit can
  no longer draw a ValidationException from AWS's per-model default.
- Bedrock: a `toolUse` input fragment for a block whose `contentBlockStart`
  never arrived fails the stream retryably instead of fabricating a nameless
  tool call, the contract the Chat codec already applies.
- Chat: a tool-call id or name arriving on a later stream fragment repairs
  the open block instead of being discarded, so a skin that splits identity
  across fragments no longer produces a call answered with the synthesized
  block id.
- Responses: the terminal stream event is read tolerantly again — a
  document trimmed below the decodable shape completes from the streamed
  blocks with its id, usage, and status salvaged, and a gateway that
  flattens the document into the event itself is read field-by-field
  instead of failing or discarding a fully delivered answer.
- Responses: a streamed `message` item opens its text block on the first
  text that arrives, so a refusal-only or empty assistant message no longer
  emits an empty `Text` part the blocking decode of the same body omits.
- Responses: a text-only Tool message answering a custom tool call routes
  to `custom_tool_call_output` by the seen custom call ids as well as the
  declared names, the same rule the `ToolResult` path already applies.
- Streamed reasoning signatures replace instead of appending: every
  protocol sends each signature whole, so a blob repeated on a start
  snapshot and a delta no longer concatenates into a signature the
  provider rejects on replay — the reference decoders' behavior on both
  Anthropic and Bedrock.
- The AWS event-stream parser validates the prelude CRC as soon as the
  12-byte prelude arrives, so a corrupted in-bounds frame length fails the
  stream immediately instead of stalling silently until the idle timeout.
- Chat: text-only content always uses the plain string form, joining
  several text parts unseparated as the reference client did; the
  part-array form is reserved for messages carrying media, so strict
  text-only skins no longer receive a shape they reject.

Round-3 migration notes (documented differences, no code change):

- Anthropic requests no longer send the `prompt-caching-2024-07-31` beta
  header alongside `cache_control` markers — prompt caching is GA on the
  direct API. A gateway that still gates caching on that beta header would
  silently stop caching; add it back per request via
  `provider_options.anthropic.beta_headers` if one is in the path.
- The Anthropic codec no longer carries the reference client's special
  handling for non-Anthropic endpoints speaking the Anthropic protocol
  (bearer auth, no version header, no count-tokens, forced streaming for
  blocking calls). Skins of that shape route through the OpenAI-compatible
  codec instead; a catalog pairing `codec = "anthropic-messages"` with such
  an endpoint gets direct-API behavior.
- Anthropic `count_input_tokens` excludes the native structured-output
  configuration (`output_config`), which the count endpoint does not accept,
  so counts for `ResponseFormat::JsonSchema` requests slightly undercount
  the schema overhead. The reference client counted the schema because it
  rode as a synthetic tool.
- SSE framing is spec-strict: events are separated by blank lines, and
  multiple `data:` lines within one event join with a newline. A
  noncompliant skin that separates events with single newlines only — the
  old per-line framing tolerated this — now fails to parse.

### Round-2 parity fixes

A second differential review against the reference implementation
(`.ai/reviews/lithos-llm-vs-fabro-llm-round-2.md`, punch list in
`.ai/plans/lithos-llm-punch-list-2.md`) landed these on top of the round-1
work:

- Passthrough routes skip capability validation: the catalog cannot describe
  an uncataloged model, so tools, media, sampling, and structured output
  dispatch and the provider judges them.
- A `cache_breakpoints` model capability gates the OpenAI-compatible codec's
  Anthropic-style breakpoints. `caching = true` alone no longer rewrites
  string content into part arrays; entries for aggregator-fronted Anthropic
  models must now set both flags.
- Cost estimates price the speed the codec put on the wire, and the Chat
  codec reports the speed control it cannot express instead of silently
  billing a fast tier for a standard call.
- Cache buckets without a catalog rate bill at zero instead of the input
  rate, and the Anthropic and Bedrock protocols derive a missing cache-write
  rate as 1.25x input, restoring the reference billing policy.
- A 200 body without a Responses `output` array, a Chat choice without a
  message object or with a tool call missing its id or function name, and a
  Converse body without its output message are decode errors instead of
  successful empty responses.
- A refusal is a classified content-filter error on the Chat codec too, on
  both the blocking and streaming paths.
- A caller's request timeout expiring mid-stream keeps the never-retry rule
  and reports as a timeout; a usage snapshot no longer closes the
  stream-retry window before content exists.
- Streaming replay integrity: an unknown Anthropic block keeps its streamed
  `input`; each streamed Gemini thought signature seals its own reasoning
  part instead of concatenating; a lost OpenAI `output_item.added` recovers
  the call id and name from the terminal item event; and a Chat argument
  fragment for a never-opened slot fails the stream instead of fabricating a
  nameless call.
- `ReasoningContent` records the signature family that minted its signature
  (`signature_origin`), and each encoder skips a foreign-signed reasoning
  part with a warning instead of replaying a signature the provider rejects.
  Anthropic Messages and Bedrock Converse share one family, so failover
  between them keeps signatures. Parts persisted without an origin replay
  unchanged.
- Migration compatibility: `"tool_calls"` still deserializes as the
  tool-call finish reason, a stored `Warning` with `"code": null` loads, and
  a Gemini `thoughtSignature` persisted at the metadata top level still
  replays.
- Bedrock: `reasoning_effort` gets the effort-levels gate, the thinking
  budget translation, and forced-tool-choice suppression the Anthropic codec
  applies; `tool_choice: none` keeps `toolConfig` when the history carries
  tool blocks, since Converse requires it; streaming requests send
  `accept: application/vnd.amazon.eventstream` again.
- Whitespace-only system prompts are omitted again on Anthropic, Bedrock,
  and Gemini.
- Gemini `countTokens` carries the required nested
  `generateContentRequest.model` field.
- OpenAI Responses requests `reasoning.encrypted_content` unconditionally
  and preserves every reasoning item — summary-only ones included — so
  multi-turn tool calling with `store=false` survives an overlay catalog
  that omits the `reasoning` flag.
- An unknown Responses status keeps its spelling instead of collapsing into
  `stop`, and a stream whose terminal document lost its output still reports
  a tool-call finish reason from the assembled blocks.
- Usage detail counters clamp to their parent totals, so an inconsistent
  skin counter cannot inflate totals or estimated cost.

Migration notes: `json_schema` response formats are always `strict: true`
(the raw `response_format` provider option is the escape hatch); manual
`provider_options.anthropic.thinking` toggles on always-adaptive models now
fail at the provider instead of locally; `ReasoningContent` and
`ModelCapabilities` gained public fields, which breaks struct-literal
construction downstream.

### Added

- Named catalog layers through `CatalogBuilder::toml_layer`, with the layer
  name in parse and validation errors.
- Validated non-secret provider `default_headers` and typed provider
  `adapter_options` in the catalog.
- `ClientBuilder::enabled_providers` for building only the providers a
  deployment has configured, while keeping the complete catalog.
- `Client::resolve_route` for resolving a request's provider and model without
  dispatching it.
- Composite HTTP credentials with a primary authentication method plus extra
  credential headers.
- Lossless content: message names and tool-call ids, typed media sources,
  redacted reasoning, structured JSON parts, and provider-namespaced opaque
  parts for replay.
- Custom tool definitions as a typed kind alongside function tools.
- Request stop sequences, request metadata, and provider options keyed by the
  canonical provider id.
- Response `raw` payloads, separate request and token rate-limit resets, and
  provider-reported cost.
- Stable streaming content-block identity with explicit block boundaries and a
  single terminal completed response.
- Provider error categories for access, not found, quota, content filter,
  server, and response decode failures, plus the cloneable `ErrorData`
  projection.
- Provider-native input token counting for OpenAI Responses, Anthropic
  Messages, Gemini, and Bedrock Runtime.
- A `bedrock-aws` feature carrying the AWS credential chain and SigV4 signing.
- A wire-parity test suite covering each provider protocol against local mock
  endpoints.
- Default transport timeouts: a 30-second connect timeout and a 300-second
  stream-idle timeout, both configurable through the client builder.
- Anthropic beta headers through `provider_options.anthropic.beta_headers`;
  fast-speed requests add the fast-mode beta automatically.
- Per-speed pricing in the catalog schema. The builtin entries carry no fast
  rates yet: the reference priced a fast tier only for Opus models, and a
  missing tier falls back to the base rates.
- A `reasoning_effort_levels` model capability. Models with it take
  `output_config.effort`; reasoning models without it get a
  `thinking.budget_tokens` translation, and levels models default to adaptive
  thinking.
- `ReasoningEffort::Max`, making `xhigh` and `max` distinct Anthropic wire
  values again.
- A retry warning log before each attempt and an optional jitter setting on
  the retry policy.

### Changed

- A `refusal` stop reason is a classified content-filter error on every
  protocol, carrying the provider's explanation and the raw payload, instead
  of decoding as a successful empty response.
- `Retry-After` is honored exactly up to a 60-second cap; a longer wait
  surfaces the error instead of retrying against the provider's guidance.
- A complete-call request timeout is not retried; stream-establishment and
  stream-idle timeouts are.
- `FinishReason` serializes every value as one bare string, and unknown
  incoming strings parse instead of failing.
- Endpoint joining drops a version segment the configured base URL already
  carries, so gateway base URLs ending in `/v1` keep working.
- Gemini requests carry the reference default safety settings when the caller
  supplies none.
- Chat Completions sends developer messages as `system` and reports request
  metadata as unsupported instead of sending a field strict skins reject.

- `ClientBuilder::build` returns a `ClientBuild` carrying the client and every
  provider construction issue, instead of failing the whole client.
- `ClientBuilder::http_client` is now `ClientBuilder::http`.
- Token usage is five disjoint buckets: input, output, reasoning, cache read,
  and cache write.
- Bedrock uses one HTTP transport and codec for bearer and SigV4 requests.
- `reqwest` moved from 0.12 to 0.13.

### Fixed

- Audio, documents, and other content a provider protocol cannot carry now fail
  before dispatch instead of being dropped or replaced with placeholder text.
- A truncated stream reports an incomplete finish reason rather than `stop`.
- Parallel tool results merge into one turn for the protocols that alternate
  roles, instead of being sent as consecutive same-role turns.
- Bedrock `error` stream frames fail the stream instead of being ignored.
- Token counting refuses the same requests completion refuses.
- Sampling controls keep the decimal the caller wrote.
- `stop_sequence`, `RECITATION`, and Bedrock's `model_context_window_exceeded`
  stop reasons map to their finish reasons instead of falling through as
  unrecognized values, and Gemini infers a tool-call finish reason from
  function-call parts.
- A Gemini prompt block, an Anthropic body that is not a Messages response,
  and a Chat Completions 200 without choices are errors instead of successful
  empty responses.
- Bedrock cache points are gated on the model's caching capability, so
  passthrough models without prompt caching no longer receive requests AWS
  rejects.
- Bedrock validates historical tool names and ids against Converse's rules,
  carries JSON, image, and document tool-result content without a false
  flattening warning, and skips reasoning parts a tool result cannot hold.
- Chat Completions encodes `reasoning_effort`, decodes OpenRouter
  `reasoning_details` for lossless replay, and restores Anthropic-style cache
  breakpoints for aggregators fronting models that cache.
- OpenAI Responses preserves assistant message items with their ids and
  replays a turn's items in their original order, so reasoning items keep
  their required pairings.
- Anthropic tool results carry structured JSON, `max_tokens` defaults to the
  model's output limit, and forced tool choice suppresses the thinking
  configuration the endpoint would reject.
- Gemini recovers tool-result function names from the assistant turn, keeps
  the response id on streams, and synthesizes tool-call ids that stay unique
  across turns.
- The builtin catalog's gpt-5.6-luna pricing and context window, the
  claude-sonnet-4-6 limits, and the Bedrock model id (an inference profile
  with pricing) match the reference data; Anthropic cache writes bill at
  1.25x input.
- `Client::from_env()` accepts `GOOGLE_API_KEY` for Gemini and
  `BEDROCK_API_KEY` or the AWS default chain for Bedrock again.
- Canonical model ids outrank provider aliases in unqualified resolution, and
  Gemini long-context tiers count cached prompt tokens.
- A malformed 200 body is retryable, non-JSON error bodies keep their text
  for classification, a retried stream emits its bookkeeping events once, and
  the final SSE frame survives a stream that ends without its terminator.
- A request timeout expiring while a success body is still arriving keeps the
  complete path's never-retry rule instead of being retried as a decode
  failure.
- Gemini carries an all-JSON tool result in the `functionResponse` struct
  instead of flattening it to empty text.

### Removed

- The `aws-sdk-bedrockruntime` dependency and its request, response, and stream
  conversions.
- Local input token estimation. `count_input_tokens` now reports only
  provider-authoritative counts and returns `None` when a provider has no
  native endpoint.

### Initial groundwork

- Provider-neutral request, response, content, tool, and error types.
- A versioned provider and model catalog with recursive TOML overlays.
- OpenAI, Anthropic, Gemini, OpenAI-compatible, and optional Amazon Bedrock
  adapters.
- Runtime extension points for model resolution, credentials, adapters, and
  middleware.
- `Client::from_env()` for the built-in catalog and conventional provider
  environment variables.
- Retry, timeout, concurrency, tracing, and observer middleware.
- Catalog-only and provider-specific Cargo feature boundaries.
