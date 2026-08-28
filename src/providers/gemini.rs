use std::sync::Arc;

use crate::adapter::{AdapterBuildError, AdapterContext, AdapterFactory, ProviderAdapter};
use crate::catalog::{CatalogProvider, codec_ids};
use crate::codecs::gemini::GeminiGenerateCodec;

pub(super) struct Factory;

impl AdapterFactory for Factory {
    fn create(
        &self,
        provider: &CatalogProvider,
        context: &AdapterContext,
    ) -> Result<Arc<dyn ProviderAdapter>, AdapterBuildError> {
        super::build_http_adapter(
            provider,
            context,
            codec_ids::GEMINI_GENERATE,
            GeminiGenerateCodec,
        )
    }
}
