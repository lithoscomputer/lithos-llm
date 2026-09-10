use std::sync::Arc;

use serde::Deserialize;

use super::http::{HttpAdapterOptions, adapter_options, build_http_adapter};
use crate::adapter::{AdapterBuildError, AdapterContext, AdapterFactory, ProviderAdapter};
use crate::catalog::{CatalogProvider, codec_ids};
use crate::codecs::openai::OpenAiResponsesCodec;

/// The typed `adapter_options` table this factory accepts.
///
/// Unknown keys are rejected so a misspelled option is a build issue for one
/// provider rather than a silently ignored setting.
#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
struct OpenAiOptions {
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
    /// The Codex deployment, which streams every response and rejects several
    /// generation fields.
    Codex,
}

pub(super) struct Factory;

impl AdapterFactory for Factory {
    fn create(
        &self,
        provider: &CatalogProvider,
        context: &AdapterContext,
    ) -> Result<Arc<dyn ProviderAdapter>, AdapterBuildError> {
        let options: OpenAiOptions = adapter_options(provider)?;
        let codex = options.mode == OpenAiMode::Codex;
        build_http_adapter(
            provider,
            context,
            codec_ids::OPENAI_RESPONSES,
            OpenAiResponsesCodec::new(codex),
            HttpAdapterOptions {
                force_streaming_complete: codex,
                identify_application:     codex,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use super::super::http::adapter_options;
    use super::{OpenAiMode, OpenAiOptions};
    use crate::adapter::AdapterBuildError;
    use crate::catalog::{Catalog, ProviderId};
    use crate::client::ProviderBuildCause;
    use crate::{Client, Request};

    const CODEX_OPTIONS: &str = r#"
            [providers.oai.adapter_options]
            mode = "codex"
    "#;

    const UNKNOWN_KEY_OPTIONS: &str = "
            [providers.oai.adapter_options]
            made_up = true
    ";

    const UNKNOWN_MODE_OPTIONS: &str = r#"
            [providers.oai.adapter_options]
            mode = "turbo"
    "#;

    /// One provider whose `adapter_options` table is written per test.
    fn catalog_layer(options: &str) -> String {
        format!(
            r#"
            schema_version = 1

            [providers.oai]
            display_name = "OpenAI"
            adapter = "openai"
            codec = "openai-responses"
            base_url = "http://127.0.0.1"
            auth = {{ type = "none" }}
            {options}
            "#
        )
    }

    fn parse(options: &str) -> Result<Result<OpenAiOptions, String>, Box<dyn StdError>> {
        let catalog = Catalog::builder()
            .toml_layer("test", &catalog_layer(options))?
            .build()?;
        let provider = catalog
            .provider_by_id(&ProviderId::new("oai"))
            .ok_or("the test catalog must define the oai provider")?;
        Ok(adapter_options::<OpenAiOptions>(provider).map_err(|error| error.to_string()))
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
        adapter = "openai"
        codec = "openai-responses"
        base_url = "http://127.0.0.1"
        default_model = "one"
        auth = { type = "none" }
        adapter_options = { mode = "turbo" }

        [providers.broken.models.one]
        display_name = "One"
        api_model = "one"
        capabilities = { text = true }

        [providers.good]
        display_name = "Good"
        adapter = "openai"
        codec = "openai-responses"
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
