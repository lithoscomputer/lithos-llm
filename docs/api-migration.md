# Public API migration

This is an intentional breaking refresh for coordinated callers.

The terminal stream event is now `StreamEvent::Ended` (JSON `type: "ended"`),
replacing `Completed` (`"completed"`). It means the response stream ended,
not that the model finished its answer. Inspect `response.finish_reason`,
including `Incomplete` and `Length`, before accepting the turn or executing
tools. Its response is boxed to keep ordinary stream events compact; use
`*response` when an owned `Response` is needed. The box does not change JSON.
Update stream consumers and stored event readers together.

`Length` and `Incomplete` responses withhold all tool calls. Their
`suppressed_tool_calls` field preserves the calls, raw arguments, and provider
metadata for diagnostics, even when raw response retention is disabled.
`tool_calls()` yields no calls for either finish reason. The existing
`truncated_tool_call` warning code now describes an unfinished turn; it does
not assert that each individual call had malformed arguments.

| Before | After |
| --- | --- |
| Deserialize unchecked request fields | Deserialization validates through RequestBuilder |
| Rebuild a request by copying fields | request.into_builder() |
| Read or replace Call fields | Getters, context_mut(), and map_request() |
| Return a boxed response stream | ResponseStream::new(stream) |
| ToolCall.kind, .arguments, .raw_arguments | One ToolCall.input: ToolInput |
| Filter response content for calls | response.tool_calls() |
| Construct an assistant message | response.into_message() |
| TimeoutMiddleware | ClientBuilder::default_timeout(Duration) |
| Mode | Operation, including CountInputTokens |
| Token counting bypasses middleware | count_input_tokens_with_context uses middleware |
| Observer::on_complete | on_finish(CallOutcome), also covering streams and drop |
| Capability booleans | Support queries, including individual choices and unknown support |
| Protocol flags mixed with capabilities | CatalogModel::protocol_options() |
| Prices imply speed support | Explicit capabilities.speed data |
| Unbounded response collection | ResponseLimits and non-retryable limit errors |
| Always retain raw success bodies | ClientBuilder::retain_raw_response(bool) |

## Validated catalog objects and routes

Build catalog objects through Catalog::builder(). CatalogProvider and
CatalogModel no longer implement Deserialize. Their parsing records are private;
successful catalog construction assigns identity and validates their fields.
Their serialized catalog shape is unchanged.

Replace ResolvedRoute::new(provider, model) with
ResolvedRoute::try_new(provider, model)?.
A model from a different provider returns ModelSelectionError::ModelProviderMismatch.

## Catalog data

Basic capabilities accept true, false, or "unknown".
Omitted basic features are unsupported. Omitted effort and non-default speed
choices are unknown. Passthrough models report unknown support.

Replace forced_tool_choice with tool_choice = { required = true, named = true }.
Replace structured_output with
response_format = { json_object = true, json_schema = true }.
Declare individual effort levels in reasoning_effort and speeds in speed.
Move reasoning_effort_levels, cache_breakpoints, and system_turns into
the model's protocol_options table.

## Stored data

Unrecognized content `type` values deserialize as `ContentPart::Unknown`,
preserving the entire JSON object for newer readers. This differs from provider
`Opaque` content. The client refuses unknown content, including inside tool
results, until the application explicitly converts or removes it. Malformed
known types still fail deserialization.

Usage readers ignore additional fields. The five disjoint buckets keep their
existing meanings; a new field that changes accounting requires a storage
schema version change. Legacy `*_tokens` bucket names are still rejected, so
old usage cannot silently become zero. Convert those records in the application.

Unrecognized error categories deserialize as `ErrorKind::Unknown(String)` and
serialize to the original string. `ErrorKind` is now `Clone`, not `Copy`.
`as_str()` returns its stored spelling. Runtime errors of an unknown kind remain
non-retryable even if a retry hint is supplied. Applications must not infer
failover eligibility from an unfamiliar category.

Applications own conversion of historical records. Lithos no longer accepts
the old finish-reason object, token-count aliases, or a null warning code.
Use the string "tool_call" for the canonical tool-call finish reason.
Other strings remain FinishReason::Other values.

Function inputs serialize as:

    {"type":"function","value":"{\"key\":1}"}

Custom inputs serialize as:
    {"type":"custom","value":"free-form text"}

Convert old tool fields into one input. Keep malformed raw arguments.

Opaque replay kinds use provider namespaces, such as openai.reasoning,
openai.message, and openai_compatible.reasoning_details.
OpenAI item IDs belong under provider_metadata.openai.item_id.
Gemini tool signatures belong under provider_metadata.gemini.thoughtSignature.
Signed reasoning needs an explicit matching signature_origin.
Unrecognized historical replay fields are ignored, not converted.

Applications convert their own historical records when they adopt Lithos.

`ProbeOutcome::Failed` now contains `Box<ErrorData>` to keep the enum compact.
Use `Box::new(data)` when constructing this variant. Its JSON shape is unchanged.

## Shared model selection

`ModelResolver` implementors must replace `resolve(&Request, ...)` with
`resolve_selection(&ModelSelection, ...)`. Selection is independent of messages
and request controls; synchronous implementations must not read credentials or
perform I/O. The provided `resolve(&Request, ...)` convenience remains for
callers. The client invokes `resolve_selection` directly for every operation and
for route inspection, so an override of the old convenience is not a dispatch
hook. Custom resolvers retain full control over routing policy.

Existing `.model("selector")`, `request.model()` and JSON `"model": "selector"`
inputs remain supported. New `RequestBuilder::selection` and
`EvaluationBuilder::selection` preserve structured intent; corresponding
`selection()` getters return it. Update middleware that compares or persists
selection to use these getters. The legacy text getter omits structured provider
scope. `into_builder` and JSON round trips preserve the full selection.

`ModelSelection::Default` serializes as JSON null; a builder's missing `model`
field still produces `MissingModel`, while an explicitly null field selects the
default. A provider-scoped model serializes as
`{"provider":"openrouter","model":"vendor/model:version"}` under `model`.
An exact handle serializes as `{"handle":{"provider":"gemini","model":"flash"}}`.

Raw colon qualification (`provider:model`) is now recognized alongside slash
qualification. The first delimiter separates provider from model; the rest is
opaque. To use a literal colon or slash in an unqualified model id, supply
`ModelSelection::from_parts(Some(model_id), Some(provider))`. The raw form never
falls back from an unknown or unavailable provider prefix to an aggregator's
wire id. Applications supporting older reference syntaxes should translate
those syntaxes to structured intent at their input boundary, with explicit
conflict handling if they accept both a qualified reference and a provider pin.
A separate provider pin always scopes the complete model string in Lithos.

`SelectionPolicy` makes exact pins, stand-ins, catalog-only lookup and
unknown-bare-model passthrough explicit. Install it on the client as its resolver
and use the same policy for preflight. Neither permissive mode bypasses a
provider's `allow_passthrough` flag. Known models that only exist on ineligible
providers are not treated as unknown models eligible for fallback.

For later exact execution of a catalog route, persist `route.handle()` and pass
that handle to `.selection(handle)`; do not reconstruct an alias or default.
Exact handles require canonical catalog ids and an enabled, eligible provider.
They fail if the row disappears instead of following a new alias, passing its
old slug through as a wire id, or selecting a stand-in. The current catalog's
`api_model` still controls the wire id. Passthrough handles do not contain enough
information to prove a catalog identity; preserve scoped selection intent for
those routes rather than using the exact-handle input.
