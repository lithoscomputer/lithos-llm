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

### Changed

- `ClientBuilder::build` returns a `ClientBuild` carrying the client and every
  provider construction issue, instead of failing the whole client.
- `ClientBuilder::http_client` is now `ClientBuilder::http`.
- Token usage is five disjoint buckets: input, output, reasoning, cache read,
  and cache write.
- Bedrock uses one HTTP transport and codec for bearer and SigV4 requests.
- `reqwest` moved from 0.12 to 0.13.

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
