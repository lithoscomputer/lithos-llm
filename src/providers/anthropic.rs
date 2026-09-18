use std::sync::Arc;

use super::http::{HttpAdapterOptions, adapter_options, build_http_adapter};
use crate::adapter::{AdapterBuildError, AdapterContext, AdapterFactory, ProviderAdapter};
use crate::catalog::{CatalogProvider, codec_ids};
use crate::codecs::anthropic::AnthropicMessagesCodec;

pub(super) struct Factory;

impl AdapterFactory for Factory {
    fn create(
        &self,
        provider: &CatalogProvider,
        context: &AdapterContext,
    ) -> Result<Arc<dyn ProviderAdapter>, AdapterBuildError> {
        let adapter: HttpAdapterOptions = adapter_options(provider)?;
        build_http_adapter(
            provider,
            context,
            codec_ids::ANTHROPIC_MESSAGES,
            AnthropicMessagesCodec,
            adapter,
        )
    }
}
