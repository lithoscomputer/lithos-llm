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

Fabro will implement conversion of its records when it integrates Lithos.
