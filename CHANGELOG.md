# Changelog

All notable changes to `lithos-llm` will be documented in this file.

This project follows [Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- The built-in catalog now includes OpenRouter with live-verified model routes,
  provider-reported cost support, and conventional
  `OPENROUTER_API_KEY` credentials.

### Changed

- Provider catalog tests now check resolution and behavior without copying
  exact roster counts or model tables.
- The minimum supported Rust version is 1.88 (from 1.85): the codecs now
  use let-chains, which need it.

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
