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
