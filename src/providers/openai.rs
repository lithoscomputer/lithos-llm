use std::sync::Arc;

use serde::Deserialize;

use super::http::{HttpAdapterOptions, adapter_options, build_http_adapter, codec_options};
use crate::adapter::{AdapterBuildError, AdapterContext, AdapterFactory, ProviderAdapter};
use crate::catalog::{CatalogProvider, codec_ids};
use crate::codecs::openai::OpenAiResponsesCodec;

/// The typed `codec_options.openai-responses` table.
///
/// Unknown keys are rejected so a misspelled option is a build issue for one
/// provider rather than a silently ignored setting.
#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
struct OpenAiResponsesOptions {
    #[serde(default)]
    mode: OpenAiMode,
}

/// Which OpenAI Responses deployment a provider talks to.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum OpenAiMode {
    /// The public `/v1/responses` API.
    #[default]
    Standard,
    /// The Codex deployment, which rejects several generation fields.
    ///
    /// The deployment also streams every response and reads an `originator`
    /// header; those are adapter concerns, set in `adapter_options` as
    /// `force_streaming_complete` and `identify_application`. A factory test
    /// pins that the built-in row sets both.
    Codex,
}

pub(super) struct Factory;

impl AdapterFactory for Factory {
    fn create(
        &self,
        provider: &CatalogProvider,
        context: &AdapterContext,
    ) -> Result<Arc<dyn ProviderAdapter>, AdapterBuildError> {
        let adapter: HttpAdapterOptions = adapter_options(provider)?;
        let codec: OpenAiResponsesOptions = codec_options(provider, codec_ids::OPENAI_RESPONSES)?;
        build_http_adapter(
            provider,
            context,
            codec_ids::OPENAI_RESPONSES,
            OpenAiResponsesCodec::new(codec.mode == OpenAiMode::Codex),
            adapter,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use super::super::http::{HttpAdapterOptions, adapter_options, codec_options};
    use super::{OpenAiMode, OpenAiResponsesOptions};
    use crate::adapter::AdapterBuildError;
    use crate::catalog::{Catalog, ProviderId, codec_ids};
    use crate::client::ProviderBuildCause;
    use crate::{Client, Request};

    const CODEX_OPTIONS: &str = r#"
            [providers.oai.codec_options.openai-responses]
            mode = "codex"
    "#;

    const UNKNOWN_KEY_OPTIONS: &str = "
            [providers.oai.codec_options.openai-responses]
            made_up = true
    ";

    const UNKNOWN_MODE_OPTIONS: &str = r#"
            [providers.oai.codec_options.openai-responses]
            mode = "turbo"
    "#;

    /// One provider whose options tables are written per test.
    fn catalog_layer(options: &str) -> String {
        format!(
            r#"
            schema_version = 1

            [providers.oai]
            display_name = "OpenAI"
            codecs = ["openai-responses"]
            base_url = "http://127.0.0.1"
            auth = {{ type = "none" }}
            {options}
            "#
        )
    }

    fn parse(options: &str) -> Result<Result<OpenAiResponsesOptions, String>, Box<dyn StdError>> {
        let catalog = Catalog::builder()
            .toml_layer("test", &catalog_layer(options))?
            .build()?;
        let provider = catalog
            .provider_by_id(&ProviderId::new("oai"))
            .ok_or("the test catalog must define the oai provider")?;
        Ok(
            codec_options::<OpenAiResponsesOptions>(provider, codec_ids::OPENAI_RESPONSES)
                .map_err(|error| error.to_string()),
        )
    }

    /// The Codex deployment is one word in the codec table and two in the
    /// adapter table; the built-in row must set all three together.
    #[cfg(feature = "builtin-catalog")]
    #[test]
    fn the_builtin_codex_row_sets_the_codec_mode_and_both_adapter_options()
    -> Result<(), Box<dyn StdError>> {
        let catalog = Catalog::builder().with_builtin().build()?;
        let provider = catalog
            .provider_by_id(&ProviderId::new("openai-codex"))
            .ok_or("the built-in catalog defines openai-codex")?;

        let codec: OpenAiResponsesOptions = codec_options(provider, codec_ids::OPENAI_RESPONSES)?;
        let adapter: HttpAdapterOptions = adapter_options(provider)?;

        assert_eq!(codec.mode, OpenAiMode::Codex);
        assert!(adapter.force_streaming_complete);
        assert!(adapter.identify_application);
        Ok(())
    }

    /// An `adapter_options` key that belongs to the codec is rejected, so a
    /// row migrated by hand cannot leave `mode` in the wrong table.
    #[test]
    fn a_codec_option_in_the_adapter_table_is_rejected() -> Result<(), Box<dyn StdError>> {
        let catalog = Catalog::builder()
            .toml_layer(
                "test",
                &catalog_layer("adapter_options = { mode = \"codex\" }"),
            )?
            .build()?;
        let provider = catalog
            .provider_by_id(&ProviderId::new("oai"))
            .ok_or("the test catalog must define the oai provider")?;

        let parsed = adapter_options::<HttpAdapterOptions>(provider);

        assert!(parsed.is_err(), "`mode` is a codec option");
        Ok(())
    }

    #[test]
    fn an_absent_options_table_is_the_default_mode() -> Result<(), Box<dyn StdError>> {
        let options = parse("")?.map_err(|error| format!("expected defaults, got {error}"))?;
        assert_eq!(options.mode, OpenAiMode::Standard);
        Ok(())
    }

    #[test]
    fn codex_mode_parses() -> Result<(), Box<dyn StdError>> {
        let options =
            parse(CODEX_OPTIONS)?.map_err(|error| format!("expected codex mode, got {error}"))?;
        assert_eq!(options.mode, OpenAiMode::Codex);
        Ok(())
    }

    #[test]
    fn an_unknown_option_key_is_rejected() -> Result<(), Box<dyn StdError>> {
        let parsed = parse(UNKNOWN_KEY_OPTIONS)?;
        assert!(
            parsed.is_err(),
            "deny_unknown_fields must reject stray keys"
        );
        Ok(())
    }

    #[test]
    fn an_unknown_mode_is_rejected() -> Result<(), Box<dyn StdError>> {
        let parsed = parse(UNKNOWN_MODE_OPTIONS)?;
        assert!(parsed.is_err(), "only the declared modes are accepted");
        Ok(())
    }

    /// One provider with unusable options beside one that builds.
    const MIXED_CATALOG: &str = r#"
        schema_version = 1

        [providers.broken]
        display_name = "Broken"
        codecs = ["openai-responses"]
        base_url = "http://127.0.0.1"
        default_model = "one"
        auth = { type = "none" }
        codec_options = { openai-responses = { mode = "turbo" } }

        [providers.broken.models.one]
        display_name = "One"
        api_model = "one"
        capabilities = { text = true }

        [providers.good]
        display_name = "Good"
        codecs = ["openai-responses"]
        base_url = "http://127.0.0.1"
        default_model = "one"
        auth = { type = "none" }

        [providers.good.models.one]
        display_name = "One"
        api_model = "one"
        capabilities = { text = true }
    "#;

    #[test]
    fn invalid_options_become_one_provider_build_issue() -> Result<(), Box<dyn StdError>> {
        let catalog = Catalog::builder()
            .toml_layer("test", MIXED_CATALOG)?
            .build()?;

        let build = Client::builder().catalog(catalog).build()?;

        assert_eq!(
            build.issues.len(),
            1,
            "only the broken provider should fail"
        );
        let issue = build
            .issues
            .first()
            .ok_or("the broken provider should report one issue")?;
        assert_eq!(issue.provider.as_str(), "broken");
        assert!(
            matches!(
                issue.cause,
                ProviderBuildCause::Adapter(AdapterBuildError::InvalidAdapterOptions { .. })
            ),
            "invalid options are an adapter build error, not a fatal client error"
        );
        assert!(
            build
                .client
                .available_providers()
                .contains(&ProviderId::new("good")),
            "the other provider must still build"
        );

        // The surviving provider is the one a request can still route to.
        let request = Request::builder().model("good/one").user("Hi").build()?;
        let route = build.client.resolve_route(&request)?;
        assert_eq!(route.provider().id().as_str(), "good");
        Ok(())
    }
}
