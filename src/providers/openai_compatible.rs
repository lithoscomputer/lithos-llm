use std::sync::Arc;

use serde::Deserialize;

use super::http::{HttpAdapterOptions, adapter_options, build_http_adapter};
use crate::adapter::{AdapterBuildError, AdapterContext, AdapterFactory, ProviderAdapter};
use crate::catalog::{CatalogProvider, codec_ids};
use crate::codecs::openai_chat::OpenAiChatCodec;

/// Some deployments publish their own versioned API root (for example,
/// Z.ai's /api/coding/paas/v4) and must not receive an extra /v1 segment.
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct OpenAiCompatibleOptions {
    base_url_is_api_root: bool,
}

pub(super) struct Factory;

impl AdapterFactory for Factory {
    fn create(
        &self,
        provider: &CatalogProvider,
        context: &AdapterContext,
    ) -> Result<Arc<dyn ProviderAdapter>, AdapterBuildError> {
        let options: OpenAiCompatibleOptions = adapter_options(provider)?;
        let codec = if options.base_url_is_api_root {
            OpenAiChatCodec::at_api_root()
        } else {
            OpenAiChatCodec::default()
        };
        build_http_adapter(
            provider,
            context,
            codec_ids::OPENAI_CHAT,
            codec,
            HttpAdapterOptions::default(),
        )
    }
}
