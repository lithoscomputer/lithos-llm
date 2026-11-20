//! The HTTP endpoints each codec's API serves, and how they join a base URL.
//!
//! The codecs build their request URLs from these paths, and
//! [`CatalogProvider::endpoints`](super::CatalogProvider::endpoints) hands the
//! same paths to callers that talk to a provider directly, such as a proxy
//! that forwards requests unchanged.

use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use super::{CodecId, codec_ids};

/// API version segments a codec's operation path may repeat from a base URL.
///
/// Codecs own the version segment: they ask for `/v1/messages` or
/// `/v1beta/models/...`. The older convention put the version in the catalog
/// base URL instead, and gateways still publish mounts that end with one, so a
/// base URL ending in one of these segments is common. [`endpoint`] joins the
/// two without repeating the segment.
const VERSION_SEGMENTS: &[&str] = &["v1", "v1beta"];

/// The inference and model listing endpoints of one codec's API.
///
/// Get one for a provider with
/// [`CatalogProvider::endpoints`](super::CatalogProvider::endpoints). The
/// paths are what the codec asks for, and they can repeat a version segment
/// the base URL already ends with: OpenRouter's base URL is
/// `https://openrouter.ai/api/v1` and Chat Completions asks for
/// `/v1/chat/completions`. Build a URL with
/// [`inference_url`](Self::inference_url) or [`models_url`](Self::models_url),
/// which keep that segment once, rather than appending a path to the base URL.
///
/// ```
/// use lithos_llm::catalog::{Catalog, CodecId, codec_ids};
///
/// let catalog = Catalog::builder().with_builtin().build()?;
/// let openrouter = catalog.provider("openrouter")?;
/// let endpoints = openrouter
///     .endpoints(&CodecId::new(codec_ids::OPENAI_CHAT))
///     .expect("OpenRouter speaks Chat Completions");
///
/// assert_eq!(
///     endpoints.inference_url(openrouter.base_url()),
///     "https://openrouter.ai/api/v1/chat/completions"
/// );
/// assert_eq!(
///     endpoints.models_url(openrouter.base_url()).as_deref(),
///     Some("https://openrouter.ai/api/v1/models")
/// );
/// # Ok::<(), lithos_llm::catalog::CatalogError>(())
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Endpoints {
    inference: &'static str,
    models:    Option<&'static str>,
}

impl Endpoints {
    /// Anthropic Messages.
    pub(crate) const ANTHROPIC_MESSAGES: Self = Self {
        inference: "/v1/messages",
        models:    Some("/v1/models"),
    };

    /// OpenAI Chat Completions.
    ///
    /// `at_api_root` drops the version segment for a deployment whose base URL
    /// is its own versioned API root, such as Z.ai's `/api/coding/paas/v4`.
    pub(crate) fn openai_chat(at_api_root: bool) -> Self {
        if at_api_root {
            Self {
                inference: "/chat/completions",
                models:    Some("/models"),
            }
        } else {
            Self {
                inference: "/v1/chat/completions",
                models:    Some("/v1/models"),
            }
        }
    }

    /// OpenAI Responses.
    ///
    /// The Codex deployment hangs its Responses endpoint directly off the
    /// base path with no version segment: `<base>/responses` answers and
    /// `<base>/v1/responses` is an HTML 403 (probed against
    /// chatgpt.com/backend-api/codex on 2026-08-30). Its model listing has not
    /// been verified, so it has none here.
    pub(crate) fn openai_responses(codex: bool) -> Self {
        if codex {
            Self {
                inference: "/responses",
                models:    None,
            }
        } else {
            Self {
                inference: "/v1/responses",
                models:    Some("/v1/models"),
            }
        }
    }

    /// The path a generation request posts to, relative to the base URL.
    pub fn inference_path(&self) -> &'static str {
        self.inference
    }

    /// The path that lists the provider's models, relative to the base URL,
    /// or `None` when the API has no listing this crate knows.
    pub fn models_path(&self) -> Option<&'static str> {
        self.models
    }

    /// The URL a generation request posts to under `base_url`.
    pub fn inference_url(&self, base_url: &str) -> String {
        endpoint(base_url, self.inference)
    }

    /// The URL that lists the provider's models under `base_url`.
    pub fn models_url(&self, base_url: &str) -> Option<String> {
        self.models.map(|path| endpoint(base_url, path))
    }
}

/// The endpoints of codec `codec` configured by its raw `codec_options`.
///
/// `None` for a codec without one fixed inference path: Gemini and Bedrock put
/// the model in the path, the evaluation codecs serve no generation, and an
/// unknown codec belongs to a custom adapter. `None` too when `options` do not
/// match the codec's typed options, which the client rejects when it builds
/// the codec.
pub(crate) fn for_codec(codec: &CodecId, options: &Value) -> Option<Endpoints> {
    match codec.as_str() {
        codec_ids::ANTHROPIC_MESSAGES => Some(Endpoints::ANTHROPIC_MESSAGES),
        codec_ids::OPENAI_CHAT => {
            let options: OpenAiChatOptions = typed_options(options)?;
            Some(Endpoints::openai_chat(options.base_url_is_api_root))
        }
        codec_ids::OPENAI_RESPONSES => {
            let options: OpenAiResponsesOptions = typed_options(options)?;
            Some(Endpoints::openai_responses(
                options.mode == OpenAiMode::Codex,
            ))
        }
        _ => None,
    }
}

/// Deserializes one codec's raw options; an absent table is the default.
fn typed_options<T: Default + DeserializeOwned>(options: &Value) -> Option<T> {
    if options.is_null() {
        return Some(T::default());
    }
    serde_json::from_value(options.clone()).ok()
}

/// The typed `codec_options.openai-chat` table.
///
/// Some deployments publish their own versioned API root (for example,
/// Z.ai's /api/coding/paas/v4) and must not receive an extra /v1 segment.
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct OpenAiChatOptions {
    pub(crate) base_url_is_api_root: bool,
}

/// The typed `codec_options.openai-responses` table.
///
/// Unknown keys are rejected so a misspelled option is a build issue for one
/// provider rather than a silently ignored setting.
#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub(crate) struct OpenAiResponsesOptions {
    #[serde(default)]
    pub(crate) mode: OpenAiMode,
}

/// Which OpenAI Responses deployment a provider talks to.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OpenAiMode {
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

/// Joins a provider base URL and a request path.
///
/// A base URL whose path already ends with the version segment the request
/// path starts with keeps that segment once: base `https://api.moonshot.ai/v1`
/// and path `/v1/chat/completions` join as
/// `https://api.moonshot.ai/v1/chat/completions`, not `/v1/v1/...`. Only an
/// exact repeat of the same segment is collapsed, so a `/v1` base URL still
/// keeps a Gemini `/v1beta` path intact, and a gateway mounted at
/// `/gateway` keeps its own prefix.
pub(crate) fn endpoint(base_url: &str, path: &str) -> String {
    let base = base_url.trim_end_matches('/');
    let path = path.trim_start_matches('/');
    let path = match repeated_version(base, path) {
        Some(segment) => path[segment.len()..].trim_start_matches('/'),
        None => path,
    };

    if path.is_empty() {
        return base.to_owned();
    }
    format!("{base}/{path}")
}

/// The version segment `base` and `path` both carry, when they share one.
///
/// `base` has already lost its trailing slashes and `path` its leading ones.
fn repeated_version<'a>(base: &str, path: &'a str) -> Option<&'a str> {
    let host_and_path = base.split_once("://").map_or(base, |(_, rest)| rest);
    let base_tail = host_and_path.split('/').next_back()?;
    let path_head = path.split('/').next()?;

    // A bare host has no path segment to repeat, so its final label — which
    // could in principle read `v1` — is not a version segment.
    let has_path = host_and_path.contains('/');
    (has_path && path_head == base_tail && VERSION_SEGMENTS.contains(&path_head))
        .then_some(path_head)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "builtin-catalog")]
    use std::error::Error as StdError;

    use super::endpoint;
    #[cfg(feature = "builtin-catalog")]
    use crate::catalog::{Catalog, CodecId, codec_ids};

    #[cfg(feature = "builtin-catalog")]
    #[test]
    fn builtin_providers_join_their_endpoints_to_their_base_urls() -> Result<(), Box<dyn StdError>>
    {
        let catalog = Catalog::builder().with_builtin().build()?;
        // Each URL is the provider's documented endpoint, so a path that
        // repeats or drops the base URL's version segment fails here.
        let cases = [
            (
                "anthropic",
                codec_ids::ANTHROPIC_MESSAGES,
                "https://api.anthropic.com/v1/messages",
                Some("https://api.anthropic.com/v1/models"),
            ),
            (
                "openai",
                codec_ids::OPENAI_RESPONSES,
                "https://api.openai.com/v1/responses",
                Some("https://api.openai.com/v1/models"),
            ),
            (
                "openai-codex",
                codec_ids::OPENAI_RESPONSES,
                "https://chatgpt.com/backend-api/codex/responses",
                None,
            ),
            (
                "openrouter",
                codec_ids::OPENAI_CHAT,
                "https://openrouter.ai/api/v1/chat/completions",
                Some("https://openrouter.ai/api/v1/models"),
            ),
            (
                "openrouter",
                codec_ids::OPENAI_RESPONSES,
                "https://openrouter.ai/api/v1/responses",
                Some("https://openrouter.ai/api/v1/models"),
            ),
            (
                "openrouter",
                codec_ids::ANTHROPIC_MESSAGES,
                "https://openrouter.ai/api/v1/messages",
                Some("https://openrouter.ai/api/v1/models"),
            ),
            (
                "venice",
                codec_ids::OPENAI_CHAT,
                "https://api.venice.ai/api/v1/chat/completions",
                Some("https://api.venice.ai/api/v1/models"),
            ),
            (
                "venice",
                codec_ids::OPENAI_RESPONSES,
                "https://api.venice.ai/api/v1/responses",
                Some("https://api.venice.ai/api/v1/models"),
            ),
            (
                "zai",
                codec_ids::OPENAI_CHAT,
                "https://api.z.ai/api/coding/paas/v4/chat/completions",
                Some("https://api.z.ai/api/coding/paas/v4/models"),
            ),
        ];

        for (provider, codec, inference, models) in cases {
            let provider = catalog.provider(provider)?;
            let endpoints = provider
                .endpoints(&CodecId::new(codec))
                .ok_or(format!("{} has no {codec} endpoints", provider.id()))?;
            assert_eq!(endpoints.inference_url(provider.base_url()), inference);
            assert_eq!(endpoints.models_url(provider.base_url()).as_deref(), models);
        }
        Ok(())
    }

    #[cfg(feature = "builtin-catalog")]
    #[test]
    fn a_replacement_base_url_takes_the_same_paths() -> Result<(), Box<dyn StdError>> {
        let catalog = Catalog::builder().with_builtin().build()?;
        let endpoints = catalog
            .provider("anthropic")?
            .endpoints(&CodecId::new(codec_ids::ANTHROPIC_MESSAGES))
            .ok_or("anthropic speaks Messages")?;

        assert_eq!(endpoints.inference_path(), "/v1/messages");
        assert_eq!(endpoints.models_path(), Some("/v1/models"));
        assert_eq!(
            endpoints.inference_url("http://127.0.0.1:9/prefix/"),
            "http://127.0.0.1:9/prefix/v1/messages"
        );
        Ok(())
    }

    #[cfg(feature = "builtin-catalog")]
    #[test]
    fn codecs_without_one_fixed_path_or_not_listed_have_no_endpoints()
    -> Result<(), Box<dyn StdError>> {
        let catalog = Catalog::builder().with_builtin().build()?;

        let gemini = catalog.provider("gemini")?;
        assert_eq!(
            gemini.endpoints(&CodecId::new(codec_ids::GEMINI_GENERATE)),
            None
        );
        let openrouter = catalog.provider("openrouter")?;
        assert_eq!(
            openrouter.endpoints(&CodecId::new(codec_ids::SYSTEMONE)),
            None
        );
        // Venice does not speak Anthropic Messages, so it has no Messages
        // endpoints even though the codec has fixed paths.
        assert_eq!(
            catalog
                .provider("venice")?
                .endpoints(&CodecId::new(codec_ids::ANTHROPIC_MESSAGES)),
            None
        );
        Ok(())
    }

    #[test]
    fn a_bare_host_takes_the_whole_operation_path() {
        assert_eq!(
            endpoint("https://api.openai.com", "/v1/responses"),
            "https://api.openai.com/v1/responses"
        );
        assert_eq!(
            endpoint("https://api.openai.com/", "/v1/responses"),
            "https://api.openai.com/v1/responses"
        );
    }

    #[test]
    fn a_base_url_ending_in_the_version_does_not_repeat_it() {
        assert_eq!(
            endpoint("https://api.moonshot.ai/v1", "/v1/chat/completions"),
            "https://api.moonshot.ai/v1/chat/completions"
        );
        assert_eq!(
            endpoint("https://api.moonshot.ai/v1/", "/v1/chat/completions"),
            "https://api.moonshot.ai/v1/chat/completions"
        );
        assert_eq!(
            endpoint(
                "https://generativelanguage.googleapis.com/v1beta",
                "/v1beta/models/gemini:generateContent"
            ),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini:generateContent"
        );
    }

    #[test]
    fn a_gateway_mount_keeps_its_own_prefix() {
        assert_eq!(
            endpoint("https://gw.example.com/gateway", "/v1/chat/completions"),
            "https://gw.example.com/gateway/v1/chat/completions"
        );
        assert_eq!(
            endpoint("https://gw.example.com/gateway/v1", "/v1/chat/completions"),
            "https://gw.example.com/gateway/v1/chat/completions"
        );
    }

    #[test]
    fn only_the_same_version_segment_collapses() {
        // `v1` and `v1beta` are different APIs, so neither absorbs the other.
        assert_eq!(
            endpoint("https://example.com/v1", "/v1beta/models/one:count"),
            "https://example.com/v1/v1beta/models/one:count"
        );
        // An unversioned operation path is never shortened.
        assert_eq!(
            endpoint("https://bedrock.example.com/v1", "/model/one/converse"),
            "https://bedrock.example.com/v1/model/one/converse"
        );
    }
}
