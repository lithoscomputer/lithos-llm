# Listings, probes, and twin settings

Load keys first: `set -a; . ./.env; set +a`. The `.env` has keys for
Anthropic, OpenRouter, Venice, Vercel (`AI_GATEWAY_API_KEY`), OpenAI, Gemini,
Fireworks, Moonshot, Modal, and TypeSafe — not AWS, so Bedrock can't be probed.

In zsh, quote any URL that has a `?`, or the shell treats it as a glob.

## Live listings

```sh
# Anthropic: one model (limits, effort levels, thinking types)
curl -sS https://api.anthropic.com/v1/models/<wire-id> \
  -H "x-api-key: $ANTHROPIC_API_KEY" -H 'anthropic-version: 2023-06-01'

# OpenRouter (public): supported_parameters, reasoning.supported_efforts, pricing per token
curl -sS https://openrouter.ai/api/v1/models \
  | jq '.data[] | select(.id|test("<pattern>"))'

# Venice: model_spec.pricing (USD per MTok), capabilities, reasoningEffortOptions
curl -sS 'https://api.venice.ai/api/v1/models?type=text' \
  -H "Authorization: Bearer $VENICE_API_KEY" | jq '.data[] | select(.id|test("<pattern>"))'

# OpenAI: ids only (no prices or limits; take those from the model's docs page)
curl -sS https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY" \
  | jq '.data[] | select(.id|test("<pattern>"))'

# Vercel AI Gateway (public): supported_parameters, temperature, reasoning_options, pricing
curl -sS https://ai-gateway.vercel.sh/v1/models | jq '.data[] | select(.id|test("<pattern>"))'
```

Per-token listing prices convert to catalog micros by multiplying by 1e12
(`"0.000004"` → `4000000`). Venice already quotes USD per MTok.

Ignore `:batch` and `-fast` variant ids on the gateways; the rows describe the
standard model. A gateway `fast` price table can't be requested through the
Chat codec, so gateway rows leave `speed.fast` unclaimed.

## Probe helpers

Anthropic Messages:

```sh
a() { curl -sS https://api.anthropic.com/v1/messages -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H 'anthropic-version: 2023-06-01' -H 'content-type: application/json' "$@" \
  | jq -c '{e: .error.message, speed: .usage.speed, types: [.content[]?.type], text: [.content[]? | select(.type=="text") | .text]}'; }
T='"tools":[{"name":"get_weather","description":"Reads the weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]'
M=<wire-id>
a -d '{"model":"'$M'","max_tokens":2000,'"$T"',"tool_choice":{"type":"any"},"messages":[{"role":"user","content":"Write a haiku about the sea."}]}'
a -d '{"model":"'$M'","max_tokens":200,"temperature":0.3,"messages":[{"role":"user","content":"hi"}]}'
a -d '{"model":"'$M'","max_tokens":200,"thinking":{"type":"disabled"},"messages":[{"role":"user","content":"hi"}]}'
a -d '{"model":"'$M'","max_tokens":2000,"output_config":{"effort":"max"},"thinking":{"type":"adaptive"},"messages":[{"role":"user","content":"hi"}]}'
# System turn: must follow a user message.
a -d '{"model":"'$M'","max_tokens":2000,"system":"Be brief.","messages":[{"role":"user","content":"Say hello."},{"role":"assistant","content":"Hello."},{"role":"user","content":"Say goodbye."},{"role":"system","content":"Answer ONLY in UPPERCASE letters."}]}'
# Fast mode: pass the beta header as its own argument.
a -H 'anthropic-beta: fast-mode-2026-02-01' -d '{"model":"'$M'","max_tokens":100,"speed":"fast","messages":[{"role":"user","content":"hi"}]}'
```

OpenAI-compatible gateways (OpenRouter, Venice, Vercel) — same body, different
base URL and key:

```sh
c() { curl -sS "$1/chat/completions" -H "Authorization: Bearer $2" -H 'content-type: application/json' -d "$3" \
  | jq -c '{e: (.error // .details), fr: .choices[0].finish_reason, tc: .choices[0].message.tool_calls[0].function.name, c: .choices[0].message.content[0:80]}'; }
OR=https://openrouter.ai/api/v1; VE=https://api.venice.ai/api/v1; VC=https://ai-gateway.vercel.sh/v1
T='"tools":[{"type":"function","function":{"name":"get_weather","description":"Reads the weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]'
M=<gateway-wire-id>
c $OR "$OPENROUTER_API_KEY" '{"model":"'$M'","max_tokens":2000,'"$T"',"tool_choice":"required","messages":[{"role":"user","content":"Write a haiku about the sea."}]}'
c $OR "$OPENROUTER_API_KEY" '{"model":"'$M'","max_tokens":2000,'"$T"',"tool_choice":{"type":"function","function":{"name":"get_weather"}},"messages":[{"role":"user","content":"Write a haiku about the sea."}]}'
c $OR "$OPENROUTER_API_KEY" '{"model":"'$M'","max_tokens":2000,"temperature":0.2,"top_p":0.9,"messages":[{"role":"user","content":"Say hi."}]}'
c $OR "$OPENROUTER_API_KEY" '{"model":"'$M'","max_tokens":4000,"reasoning_effort":"max","messages":[{"role":"user","content":"Say hi."}]}'
```

OpenAI Responses API (the `openai` provider's codec). There is no `stop`
parameter (it 400s on every model), and sampling works only on some rows, so
probe `temperature` explicitly:

```sh
o() { curl -sS https://api.openai.com/v1/responses -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H 'content-type: application/json' -d "$1" \
  | jq -c '{e: .error.message, types: [.output[]?.type], text: [.output[]? | select(.type=="message") | .content[0].text], usage}'; }
TR='"tools":[{"type":"function","name":"get_weather","description":"Reads the weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false},"strict":true}]'
M=<openai-id>
o '{"model":"'$M'",'"$TR"',"tool_choice":"required","input":"Write a haiku about the sea."}'
o '{"model":"'$M'",'"$TR"',"tool_choice":{"type":"function","name":"get_weather"},"input":"Write a haiku about the sea."}'
o '{"model":"'$M'","temperature":0.2,"input":"Say hi."}'
o '{"model":"'$M'","reasoning":{"effort":"minimal"},"input":"Say hi."}'   # repeat per level
o '{"model":"'$M'","service_tier":"priority","input":"Say hi."}'         # speed tiers; check the returned service_tier
```

A `function_call` item in `types` is a real forced call.

Reading forced-choice results: `fr: "tool_calls"` with a call is support; an
error is a denial; `fr: "stop"` with haiku text is a silent drop (deny).
Venice's error comes back in `.error` as a bare string.

## Twin settings for recording

These mirror the `test:e2e:record` task in `mise.toml`; check there if a
provider is missing or a port looks wrong.

| Provider | Port | Recording | Upstream | Key variable |
|---|---|---|---|---|
| venice | 3921 (twin default) | `tests/e2e/recordings/venice.json` | default | `VENICE_API_KEY` |
| openrouter | 3922 | `openrouter.json` | `https://openrouter.ai/api` | `OPENROUTER_API_KEY` |
| moonshot | 3923 | `moonshot.json` | `https://api.moonshot.ai` | `MOONSHOT_API_KEY` |
| fireworks | 3924 | `fireworks.json` | `https://api.fireworks.ai/inference` | `FIREWORKS_API_KEY` |
| openai | 3925 | `openai.json` | `https://api.openai.com` | `OPENAI_API_KEY` |
| openai codex | 3926 | `openai_codex.json` | see `mise.toml` (needs `OPENAI_CODEX_TOKEN`) | `OPENAI_CODEX_TOKEN` |
| vercel | 3927 | `vercel.json` | `https://ai-gateway.vercel.sh` | `AI_GATEWAY_API_KEY` |
| typesafe | 3928 | `typesafe.json` | `https://api.typesafe.ai` | `TYPESAFE_API_KEY` |

For Venice, `LITHOS_E2E_TWIN_MODE=record LITHOS_E2E_RECORDING_APPEND=1
./target/debug/examples/e2e_twin` is enough; the defaults cover the rest.

Replay one provider's suite:

```sh
LITHOS_E2E_TWIN_MODE=replay LITHOS_E2E_TWIN_ADDR=127.0.0.1:<port> \
  LITHOS_E2E_RECORDING_PATH=tests/e2e/recordings/<p>.json ./target/debug/examples/e2e_twin &
LITHOS_E2E_BACKEND=replay cargo nextest run --locked --all-features --test e2e \
  --run-ignored all --no-fail-fast -E 'test(/<p>::/)'
kill %1
```
