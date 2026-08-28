#[cfg(feature = "anthropic")]
mod anthropic;
#[cfg(feature = "bedrock")]
mod bedrock;
#[cfg(feature = "gemini")]
mod gemini;
#[cfg(feature = "openai")]
mod openai;
#[cfg(feature = "openai-compatible")]
mod openai_compatible;

#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "openai-compatible"
))]
use std::sync::Arc;

#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "openai-compatible"
))]
use async_trait::async_trait;
#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "openai-compatible"
))]
use futures_util::StreamExt as _;
#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "openai-compatible"
))]
use futures_util::stream::iter;

use crate::adapter::AdapterRegistry;
#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "openai-compatible"
))]
use crate::adapter::{
    AdapterBuildError, AdapterContext, InputTokenCount, ProviderAdapter, ResolvedCall,
};
#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "openai-compatible",
    feature = "bedrock"
))]
use crate::catalog::Pricing;
#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "openai-compatible",
    feature = "bedrock"
))]
use crate::catalog::adapter_ids;
#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "openai-compatible"
))]
use crate::catalog::{AdapterId, CatalogProvider};
#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "openai-compatible"
))]
use crate::codecs::Codec;
#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "openai-compatible"
))]
use crate::credentials::CredentialProvider;
#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "openai-compatible"
))]
use crate::token_count::estimate_input_tokens;
#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "openai-compatible"
))]
use crate::transport::HttpTransport;
#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "openai-compatible",
    feature = "bedrock"
))]
use crate::types::{Cost, CostSource, TokenCounts};
#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "openai-compatible"
))]
use crate::types::{Error, ErrorKind, Response, ResponseStream, StreamEvent};

pub(crate) fn register_builtin(registry: &mut AdapterRegistry) {
    #[cfg(not(any(
        feature = "openai",
        feature = "anthropic",
        feature = "gemini",
        feature = "openai-compatible",
        feature = "bedrock"
    )))]
    let _ = registry;
    #[cfg(feature = "openai")]
    registry.register_factory(adapter_ids::OPENAI, openai::Factory);
    #[cfg(feature = "anthropic")]
    registry.register_factory(adapter_ids::ANTHROPIC, anthropic::Factory);
    #[cfg(feature = "gemini")]
    registry.register_factory(adapter_ids::GEMINI, gemini::Factory);
    #[cfg(feature = "openai-compatible")]
    registry.register_factory(adapter_ids::OPENAI_COMPATIBLE, openai_compatible::Factory);
    #[cfg(feature = "bedrock")]
    registry.register_factory(adapter_ids::BEDROCK, bedrock::Factory);
}

#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "openai-compatible"
))]
pub(super) fn build_http_adapter(
    provider: &CatalogProvider,
    context: &AdapterContext,
    expected_codec: &str,
    codec: impl Codec + 'static,
) -> Result<Arc<dyn ProviderAdapter>, AdapterBuildError> {
    if provider.codec().as_str() != expected_codec {
        return Err(AdapterBuildError::UnsupportedCodec {
            provider: provider.id().clone(),
            codec:    provider.codec().clone(),
        });
    }
    Ok(Arc::new(HttpProviderAdapter {
        id:          provider.adapter().clone(),
        codec:       Arc::new(codec),
        transport:   HttpTransport::new(context.http().clone()),
        credentials: context.credentials().clone(),
    }))
}

#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "openai-compatible"
))]
struct HttpProviderAdapter {
    id:          AdapterId,
    codec:       Arc<dyn Codec>,
    transport:   HttpTransport,
    credentials: Arc<dyn CredentialProvider>,
}

#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "openai-compatible"
))]
#[async_trait]
impl ProviderAdapter for HttpProviderAdapter {
    fn id(&self) -> &AdapterId {
        &self.id
    }

    async fn complete(&self, call: &ResolvedCall) -> Result<Response, Error> {
        let credentials = self
            .credentials
            .credentials(call.route().provider())
            .await
            .map_err(|source| {
                Error::new(
                    ErrorKind::Authentication,
                    format!(
                        "credentials for provider {} could not be resolved",
                        call.route().provider().id()
                    ),
                )
                .with_provider(call.route().provider().id().clone())
                .with_source(source)
            })?;
        let encoded = self.codec.encode(call, false)?;
        let result = self
            .transport
            .execute_json(encoded, call.route().provider(), credentials)
            .await?;
        let mut response = self.codec.decode_response(call.route(), result.body)?;
        response.rate_limits = result.rate_limits;
        response.cost = catalog_cost(response.usage, call.route().model().pricing());
        Ok(response)
    }

    async fn stream(&self, call: &ResolvedCall) -> Result<ResponseStream, Error> {
        let credentials = self
            .credentials
            .credentials(call.route().provider())
            .await
            .map_err(|source| {
                Error::new(
                    ErrorKind::Authentication,
                    format!(
                        "credentials for provider {} could not be resolved",
                        call.route().provider().id()
                    ),
                )
                .with_provider(call.route().provider().id().clone())
                .with_source(source)
            })?;
        let encoded = self.codec.encode(call, true)?;
        let accepted = self
            .transport
            .sse_events(encoded, call.route().provider(), credentials)
            .await?;
        let codec = self.codec.clone();
        let route = call.route().clone();
        let decoded = accepted
            .events
            .map(move |event| match event {
                Ok(event) => codec.decode_sse(&route, event).map_or_else(
                    |error| vec![Err(error)],
                    |events| events.into_iter().map(Ok).collect(),
                ),
                Err(error) => vec![Err(error)],
            })
            .flat_map(iter);
        let limits = iter(
            accepted
                .rate_limits
                .into_iter()
                .map(|rate_limits| Ok(StreamEvent::RateLimits { rate_limits })),
        );
        Ok(Box::pin(limits.chain(decoded)))
    }

    async fn count_input_tokens(
        &self,
        call: &ResolvedCall,
    ) -> Result<Option<InputTokenCount>, Error> {
        Ok(Some(InputTokenCount::new(estimate_input_tokens(
            call.request(),
        ))))
    }
}

#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "openai-compatible",
    feature = "bedrock"
))]
fn catalog_cost(usage: TokenCounts, pricing: Option<Pricing>) -> Option<Cost> {
    let pricing = pricing?.for_input_tokens(usage.input);
    if pricing.input_usd_micros_per_million.is_none()
        && pricing.output_usd_micros_per_million.is_none()
    {
        return None;
    }
    let cached = usage.cached_input.min(usage.input);
    let uncached = usage.input.saturating_sub(cached);
    let input = token_cost(uncached, pricing.input_usd_micros_per_million);
    let cached_input = token_cost(
        cached,
        pricing
            .cached_input_usd_micros_per_million
            .or(pricing.input_usd_micros_per_million),
    );
    let output = token_cost(usage.output, pricing.output_usd_micros_per_million);
    Some(Cost {
        usd_micros: input.saturating_add(cached_input).saturating_add(output),
        source:     CostSource::Catalog,
    })
}

#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "openai-compatible",
    feature = "bedrock"
))]
fn token_cost(tokens: u64, price: Option<u64>) -> u64 {
    let Some(price) = price else {
        return 0;
    };
    u64::try_from(u128::from(tokens).saturating_mul(u128::from(price)) / 1_000_000)
        .unwrap_or(u64::MAX)
}
