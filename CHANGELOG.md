# Changelog

All notable changes to `lithos-llm` will be documented in this file.

This project follows [Semantic Versioning](https://semver.org/).

## Unreleased

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
