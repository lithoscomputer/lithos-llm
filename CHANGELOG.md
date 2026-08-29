# Changelog

All notable changes to `lithos-llm` will be documented in this file.

This project follows [Semantic Versioning](https://semver.org/).

## Unreleased

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
