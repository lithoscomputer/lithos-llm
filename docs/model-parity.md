# Fabro model parity

The source contract is Fabro commit
`775b62b500c957fe319710fe34e6327ae1eb1bbf`. The test inventory pins all 99
source routes without needing a Fabro checkout. Four GPT-OSS selectors are
excluded at user request. Every other source selector resolves to a full
catalog row, except Modal Kimi K3, which requires an application overlay.

## Modal deployment

Choose one template and replace every `REPLACE_WITH` value:

- [Dedicated endpoint](catalogs/modal-dedicated.toml): set your deployment's
  base URL and the exact model ID that endpoint accepts.
- [Shared router](catalogs/modal-shared.toml): keep the router base URL and set
  `api_model` to the endpoint hostname returned for your proxy token by
  `GET /models`. Do not assume `moonshotai/Kimi-K3` is a router model ID.

Load the file with `Catalog::builder().with_builtin().overlay_toml(...)`.
The overlay makes `modal/kimi-k3` a described model and sets the Modal default.
The built-in provider stays model-less because deployment IDs are not portable.
The templates contain Fabro's provisional Kimi limits, capabilities, prices,
and profile. Check these facts against the deployment before relying on them.

Keep proxy credentials in a `CredentialProvider`, such as
`EnvironmentCredentials::conventional()`, which supplies `Modal-Key` and
`Modal-Secret` from `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`. Explicitly include
`modal` in the client's enabled providers. Neither template enables a provider
or stores secrets. Local tests replace both endpoints and IDs with mock values.

See [live-test TODOs](provider-live-tests.md) before deploying these mappings.

## Fabro defaults and application policy

Apply [fabro-policy.toml](catalogs/fabro-policy.toml) after the built-ins when
integrating with Fabro. The layer sets Fabro's provider priorities and model
defaults. Bedrock then defaults to Sonnet 5. Without this layer, Lithos keeps
Sonnet 4.6 as its Bedrock default. Apply the chosen Modal deployment template
as a separate layer to supply its model and default.

The layer preserves each source model's `family`, effective `agent_profile`,
`small_default`, and `probe` in `metadata.fabro`. It keeps Lithos's verified
capabilities, limits, prices, API IDs, and Pebble profiles. All source aliases
resolve through the built-in catalog. The inventory retains the four excluded
GPT-OSS rows for provenance; the overlay does not add them or their policies.

Provider `metadata.fabro.enabled` is application policy. The library does not
interpret it automatically. Pass the enabled IDs to `ClientBuilder`, after
applying the operator's overrides and supplying a credential provider:

```rust,ignore
let enabled: Vec<_> = catalog.providers().filter_map(|provider| {
    let enabled = provider.metadata().get("fabro")?
        .get("enabled")?.as_bool()?;
    enabled.then(|| provider.id().clone())
}).collect();
let build = Client::builder()
    .catalog(catalog)
    .enabled_providers(enabled)
    .credentials(credentials)
    .build()?;
```

Fabro must also interpret `small_default` and `probe` when choosing a model for
those tasks. Catalog resolution does not apply those flags. For a provider
with no usable flagged model, retain Fabro's fallback policy or require an
explicit choice. In particular, the dropped Fireworks GPT-OSS 20B model is no
longer a small-model or probe candidate.

Provider presence does not establish credentials or account access. Inspect
`build.issues` when enabling adapters. Enabling Bedrock also requires the
Bedrock feature and region/auth configuration. The imported Bedrock models
remain provisional even when the Fabro layer selects Sonnet 5 by default.
