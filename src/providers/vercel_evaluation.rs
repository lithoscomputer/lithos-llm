//! The `vercel-evaluation` codec's adapter: the Vercel AI Gateway's
//! evaluation protocol, built for a row whose codec set reaches it.
//!
//! The adapter evaluates and nothing else. `complete`, `stream`, and
//! `count_input_tokens` are refused before dispatch, because a row that
//! reaches this codec is an evaluation model with no generation surface.
//!
//! Step 2 of .ai/plans/adapters-and-codecs.md folds this into the HTTP
//! adapter, which then dispatches on the resolved call's codec.

use std::mem::take;
use std::sync::Arc;

use async_trait::async_trait;

use super::http::resolve_credentials;
use crate::adapter::{
    AdapterBuildError, AdapterContext, AdapterFactory, InputTokenCount, ProviderAdapter,
    ResolvedCall, ResolvedEvaluation,
};
use crate::catalog::{AdapterId, CatalogProvider, codec_ids};
use crate::codecs::EvaluationCodec;
use crate::codecs::vercel_evaluation::VercelEvaluationCodec;
use crate::credentials::CredentialProvider;
use crate::evaluation::Verdict;
use crate::resolver::ResolvedRoute;
use crate::transport::HttpTransport;
use crate::types::{Error, ErrorKind, Response, ResponseStream};

pub(super) struct Factory;

impl AdapterFactory for Factory {
    /// Builds the adapter for any provider.
    ///
    /// The provider's other codecs are ignored here: this factory serves
    /// the one codec it is named for. There are no options.
    fn create(
        &self,
        _provider: &CatalogProvider,
        context: &AdapterContext,
    ) -> Result<Arc<dyn ProviderAdapter>, AdapterBuildError> {
        Ok(Arc::new(VercelEvaluationAdapter {
            id:          AdapterId::new(codec_ids::VERCEL_EVALUATION),
            codec:       VercelEvaluationCodec,
            transport:   HttpTransport::new(context.http().clone())
                .with_stream_idle_timeout(context.stream_idle_timeout())
                .with_response_limits(context.response_limits()),
            credentials: context.credentials().clone(),
        }))
    }
}

struct VercelEvaluationAdapter {
    /// Always `vercel-evaluation`: the codec this adapter serves, not the
    /// provider's own adapter id.
    id:          AdapterId,
    codec:       VercelEvaluationCodec,
    transport:   HttpTransport,
    credentials: Arc<dyn CredentialProvider>,
}

impl VercelEvaluationAdapter {
    fn evaluates_only(&self, route: &ResolvedRoute) -> Error {
        Error::new(
            ErrorKind::InvalidRequest,
            format!("adapter {} evaluates only", self.id),
        )
        .with_provider(route.provider().id().clone())
    }
}

#[async_trait]
impl ProviderAdapter for VercelEvaluationAdapter {
    fn id(&self) -> &AdapterId {
        &self.id
    }

    async fn complete(&self, call: &ResolvedCall) -> Result<Response, Error> {
        Err(self.evaluates_only(call.route()))
    }

    async fn stream(&self, call: &ResolvedCall) -> Result<ResponseStream, Error> {
        Err(self.evaluates_only(call.route()))
    }

    async fn count_input_tokens(
        &self,
        call: &ResolvedCall,
    ) -> Result<Option<InputTokenCount>, Error> {
        Err(self.evaluates_only(call.route()))
    }

    async fn evaluate(&self, call: &ResolvedEvaluation) -> Result<Verdict, Error> {
        let provider = call.route().provider();
        // A refusal the codec can make needs no credentials, so encoding
        // comes first.
        let mut encoded = self.codec.encode_evaluation(call)?;
        let warnings = take(&mut encoded.warnings);
        let credentials = resolve_credentials(self.credentials.as_ref(), provider).await?;
        let result = self
            .transport
            .execute_json(encoded, provider, credentials)
            .await?;
        let mut verdict = self.codec.decode_verdict(call, result.body)?;
        verdict.warnings.extend(warnings);
        // Catalog pricing fills in only when the gateway reported no cost,
        // as `apply_catalog_cost` does for a response.
        if verdict.cost.is_none() {
            verdict.cost = call.route().estimate_cost(verdict.usage, None);
        }
        Ok(verdict)
    }

    fn evaluates_natively(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;
    use std::sync::Arc;

    use httpmock::{Method, MockServer};
    use serde_json::json;

    use super::Factory;
    use crate::adapter::{
        AdapterContext, AdapterFactory, ProviderAdapter, ResolvedCall, ResolvedEvaluation,
    };
    use crate::catalog::{Catalog, ProviderId};
    use crate::credentials::NoCredentials;
    use crate::evaluation::Evaluation;
    use crate::middleware::CallContext;
    use crate::resolver::{AvailableProviders, CatalogResolver, ModelResolver};
    use crate::types::{CostSource, ErrorKind, Request};

    /// A provider shaped like the built-in `vercel` provider, with a priced
    /// Jev row whose explicit evaluation claim reaches this codec.
    fn catalog(base_url: &str) -> Result<Catalog, Box<dyn StdError>> {
        let source = format!(
            r#"
            schema_version = 1

            [providers.vercel]
            display_name = "Vercel"
            codecs = ["openai-chat", "vercel-evaluation"]
            base_url = "{base_url}"
            default_model = "jev"
            auth = {{ type = "none" }}

            [providers.vercel.models.jev]
            display_name = "Jev"
            api_model = "typesafe-ai/jev"
            capabilities = {{ evaluation = {{ choice = true, score = true, boolean = true }} }}
            pricing = {{ input_usd_micros_per_million = 1000000, output_usd_micros_per_million = 0 }}
            "#
        );
        Ok(Catalog::builder().toml_layer("test", &source)?.build()?)
    }

    fn adapter(catalog: &Catalog) -> Result<Arc<dyn ProviderAdapter>, Box<dyn StdError>> {
        let provider = catalog
            .provider_by_id(&ProviderId::new("vercel"))
            .ok_or("the test catalog must define the vercel provider")?;
        let context = AdapterContext::new(reqwest::Client::new(), Arc::new(NoCredentials));
        Ok(Factory.create(provider, &context)?)
    }

    fn evaluation() -> Result<Evaluation, Box<dyn StdError>> {
        Ok(Evaluation::builder()
            .model("vercel/jev")
            .state("I was charged twice.")
            .boolean("requests_refund", "Refund requested?")
            .build()?)
    }

    fn resolved(
        catalog: &Catalog,
        evaluation: Evaluation,
    ) -> Result<ResolvedEvaluation, Box<dyn StdError>> {
        let stand_in = Request::stand_in(evaluation.model(), "state".to_owned());
        let available = AvailableProviders::all(catalog);
        let route = CatalogResolver.resolve(&stand_in, catalog, &available)?;
        Ok(ResolvedEvaluation::new(
            evaluation,
            route,
            CallContext::new(),
        ))
    }

    #[test]
    fn the_factory_ignores_the_other_codecs_and_reports_its_own_id() -> Result<(), Box<dyn StdError>>
    {
        let adapter = adapter(&catalog("http://127.0.0.1:1")?)?;

        assert_eq!(adapter.id().as_str(), "vercel-evaluation");
        assert!(adapter.evaluates_natively());
        Ok(())
    }

    #[tokio::test]
    async fn generation_calls_are_refused_before_any_dispatch() -> Result<(), Box<dyn StdError>> {
        // Port 1 refuses every connection, so any dispatch would surface as a
        // network error rather than the refusal.
        let catalog = catalog("http://127.0.0.1:1")?;
        let adapter = adapter(&catalog)?;
        let request = Request::stand_in("vercel/jev", "Hello".to_owned());
        let available = AvailableProviders::all(&catalog);
        let route = CatalogResolver.resolve(&request, &catalog, &available)?;
        let call = ResolvedCall::new(request, route, CallContext::new());

        let complete = adapter
            .complete(&call)
            .await
            .expect_err("complete is refused");
        let stream = adapter
            .stream(&call)
            .await
            .err()
            .ok_or("stream is refused")?;
        let count = adapter
            .count_input_tokens(&call)
            .await
            .expect_err("count is refused");

        for error in [complete, stream, count] {
            assert_eq!(error.kind(), ErrorKind::InvalidRequest);
            assert_eq!(error.message(), "adapter vercel-evaluation evaluates only");
            assert_eq!(
                error.provider().map(ToString::to_string).as_deref(),
                Some("vercel")
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn a_verdict_without_a_gateway_cost_is_priced_from_the_catalog()
    -> Result<(), Box<dyn StdError>> {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(Method::POST).path("/v4/ai/evaluation-model");
                then.status(200).json_body(json!({
                    "answers": { "requests_refund": { "type": "boolean", "probability": 0.9 } },
                    "usage": { "inputTokens": 389, "outputTokens": 70 }
                }));
            })
            .await;
        let catalog = catalog(&server.base_url())?;
        let adapter = adapter(&catalog)?;
        let call = resolved(&catalog, evaluation()?)?;

        let verdict = adapter.evaluate(&call).await?;

        mock.assert_async().await;
        let cost = verdict.cost.ok_or("the catalog prices the verdict")?;
        assert_eq!(cost.source, CostSource::Catalog);
        assert_eq!(
            Some(cost),
            call.route().estimate_cost(verdict.usage, None),
            "the fallback is the route's own estimate"
        );
        Ok(())
    }
}
