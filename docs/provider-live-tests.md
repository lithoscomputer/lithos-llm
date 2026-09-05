# TODO: Live tests for providers imported from Fabro

Imported on 2026-09-05 from Fabro commit
`775b62b500c957fe319710fe34e6327ae1eb1bbf`, under
`lib/foundation/fabro-model/src/catalog/providers/`.

All eight provider entries are translated into Lithos's catalog. Local mock
HTTP tests cover their request paths, model IDs, authentication shapes, and
blocking and streaming responses. **None of these imports has been tested
against a live provider through Lithos.** Prices, limits, and capability claims
remain provisional unless a catalog comment identifies a documentation correction.

## Translation decisions

- Chat Completions providers use `openai-compatible` and `openai-chat`.
  `bedrock-openai` uses `openai` and `openai-responses`.
- Z.ai sets `adapter_options.base_url_is_api_root = true`. This appends
  `/chat/completions` to its `/api/coding/paas/v4` root without adding `/v1`.
  Other compatible providers retain the existing versioned-path behavior.
- USD per million tokens becomes integer USD micros per million tokens.
  Fabro application fields remain in `metadata.fabro`. Each provider supplies
  `metadata.pebble.profile = "openai"`.
- Unverified forced-tool and structured-output support is `"unknown"`.
  Lithos permits these requests so the live tests can establish support.
- Fabro's `enabled = false` has no catalog equivalent. Applications preserve
  opt-in policy with `ClientBuilder::enabled_providers`. Catalog presence does
  not check account access or service availability.
- LiteLLM and Ollama have no universal model roster or default. Select an
  actual deployment or installed model with a qualified selector. Add model
  limits, capabilities, and pricing through an application catalog overlay.
- Ollama needs no authentication. Its local compute cost belongs to the
  application. A passthrough route has unknown cost, not a fabricated zero rate.
  Set explicit zero token rates on an overlay model if that is the desired policy.
- The Fabro server gateway remains a Fabro-owned adapter. Its application auth
  and server protocol do not belong in this provider-neutral catalog import.

## TODO for every provider

Complete these checks for each provider below. Keep live calls opt-in and
credential-gated. A basic CLI smoke check is
`lllm probe --model <provider>/<model>`; `--tools` checks a short tool round trip.
These probes do not replace the full suite.

- [ ] Verify the live model listing, default, aliases, API IDs, and regional or
      deployment access. Compare with the existing Lithos roster before adding models.
- [ ] Verify current context limits, output limits, prices, and account billing.
- [ ] Run blocking completion and streaming. Check exactly one `Ended` event,
      finish reason, text, reasoning, and usage totals.
- [ ] Run a multi-turn tool exchange. Preserve tool IDs, raw arguments, reasoning,
      and replay metadata. Check automatic, required, and named tool choices.
- [ ] Exercise claimed effort, sampling, input, caching, and structured-output
      support. Replace provisional claims with observed results and dates.
- [ ] Exercise invalid credentials, unknown model, rate limits, cancellation,
      truncation, and interrupted streams. Check errors and retry boundaries.
- [ ] Add the provider E2E module and catalog. Record deterministic successful
      exchanges under `tests/e2e/recordings/`, then add offline replay coverage.
      Do not create recordings from mock responses.

## Provider-specific TODOs

- [ ] **DeepSeek** — `DEEPSEEK_API_KEY`; `deepseek-v4-flash`, `deepseek-v4-pro`.
  Verify `reasoning_content` replay across tool turns, automatic cache hit/miss
  counters, output reasoning accounting, and low/high/max effort on both models.
  Check that medium/xhigh map to high and sampling is ignored in thinking mode.
  The current [thinking-mode documentation](https://api-docs.deepseek.com/guides/thinking_mode/)
  replaces Fabro's older restriction of Pro to high/max.
- [ ] **Inception** — `INCEPTION_API_KEY`; `mercury-2`.
  Verify ordinary append-only SSE and usage. The optional diffusion mode may
  revise earlier text and needs separate evaluation before use with our delta
  stream. Verify low/medium/high and native `instant` through provider options.
  See [reasoning efforts](https://docs.inceptionlabs.ai/capabilities/reasoning-efforts).
- [ ] **MiniMax** — `MINIMAX_API_KEY`; `minimax-m2.5`.
  Verify the corrected wire ID `MiniMax-M2.5` and 204,800-token context.
  Check `reasoning_split = true`, exact `reasoning_details` replay, tool turns,
  stream usage, caching, and the imported output limit and prices.
  See the [OpenAI-compatible API](https://platform.minimax.io/docs/api-reference/text-openai-api).
- [ ] **Z.ai** — `ZAI_API_KEY`; `glm-5.2`, `glm-4.7`.
  Verify the Coding Plan endpoint with an eligible account. A general API account
  must overlay `base_url = "https://api.z.ai/api/paas/v4"`; the same adapter option
  applies. Verify GLM 5.2 high/max, reasoning replay, cache counters, and GLM 4.7's
  inherited reasoning claim. Token-rate estimates do not calculate subscription
  charges. See [Z.ai's endpoint guidance](https://docs.z.ai/guides/overview/quick-start).
- [ ] **Poolside** — `POOLSIDE_API_KEY`; `laguna-s-2.1`, `laguna-xs-2.1`.
  Verify the `poolside/` wire prefix, hosted endpoint access, reasoning replay,
  cache counters, limits, and paid rates versus preview access. Determine whether
  this endpoint supports a native effort control before enabling portable effort.
  See the [Poolside API](https://docs.poolside.ai/api/overview).
- [ ] **LiteLLM** — `LITELLM_API_KEY`; an actual proxy deployment model.
  Test a configured proxy at port 4000 and a remote deployment. Verify auth,
  model aliases, upstream errors, tool/reasoning replay, and usage/cost passthrough
  for each upstream used. An unauthenticated local proxy needs an auth overlay
  plus explicit no-auth credentials. See the [proxy guide](https://docs.litellm.ai/docs/proxy/quick_start).
- [ ] **Ollama** — no key; an installed model such as `qwen3.5:latest`.
  Verify the local service at port 11434, model tags, context configuration,
  streaming usage, supported tools, and model-specific reasoning. Check no bearer
  header is sent. See [OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility).
- [ ] **Bedrock OpenAI** — `AWS_BEARER_TOKEN_BEDROCK`, falling back to
  `BEDROCK_API_KEY`; `gpt-5.5`, `gpt-5.4`.
  Verify regional Mantle access, `/v1/responses`, the `openai.` model IDs,
  `store = false`, both completion paths, reasoning replay, tool continuation,
  usage, and current prices. This row uses an HTTP bearer token and does not use
  Converse credentials or the AWS default chain. The current
  [AWS Responses guide](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html)
  supplies the `/v1` Mantle base path; confirm it live before rollout.

## Bedrock model parity imports

The 15 rows below use Converse and ConverseStream. Local tests cover both
operations for every row, native usage buckets, cache placement, and rejection
of unmapped effort. Receiving reasoning does not enable an effort control.
All imported rows explicitly reject portable effort until the endpoint mapping
has been established. Claude 5 also rejects sampling controls before dispatch.
Prices copied from Fabro remain provisional regional estimates. Missing prices
and cache-write rates stay unknown.

- [ ] `claude-opus-4-8`: verify the US inference profile, tools and cache buckets.
- [ ] `claude-haiku-4-5`: verify the dated US profile and small-model selection.
- [ ] `gpt-oss-120b`: verify reasoning and tool replay; establish native effort.
- [ ] `gpt-oss-20b`: verify reasoning and tool replay; establish native effort.
- [ ] `nova-2-lite`: verify global routing and the 65,535-token output cap.
- [ ] `llama-4-maverick`: verify the US profile, images, tools and missing prices.
- [ ] `mistral-large-3`: verify images, tools and regional prices.
- [ ] `devstral-2`: verify tools, limits and missing prices.
- [ ] `deepseek-v3.2`: verify reasoning and tool replay; establish native effort.
- [ ] `kimi-k2.5`: verify vision, tools and the Kimi agent profile.
- [ ] `glm-5`: verify tools and the 128,000-token output limit.
- [ ] `minimax-m2.5`: verify tools and whether reasoning is exposed by Converse.
- [ ] `nemotron-3-super`: verify the model ID, limits and missing prices.
- [ ] `claude-fable-5`: verify account `provider_data_share` opt-in, adaptive
      thinking, tools, cache writes and the future effort mapping.
- [ ] `claude-sonnet-5`: verify adaptive thinking, tools, current cache prices
      and the future effort mapping. The old introductory cache rate is omitted.

Two documentation decisions were made on 2026-09-05:

- Sonnet 5 uses standard $3/$15 per million input/output tokens after the
  August 31 promotion. See [AWS pricing](https://aws.amazon.com/bedrock/pricing/).
  Confirm regional rates and cache rates on the account before cost reporting.
- Sonnet 4.6 keeps the existing 200K context limit. The
  [model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-4-6.html)
  advertises 1M, while the
  [request documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages-request-response.html)
  lists a compatible `context-1m-2025-08-07` opt-in. The import does not silently
  enable that opt-in. Verify account/region behavior above 200K before applying
  a 1M overlay. This is a conservative catalog policy, not a model maximum claim.
