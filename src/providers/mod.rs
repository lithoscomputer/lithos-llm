//! Built-in provider adapter factories.

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

use crate::adapter::AdapterRegistry;
#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "openai-compatible",
    feature = "bedrock"
))]
use crate::catalog::adapter_ids;

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

/// The shared HTTP adapter every SSE-based provider factory builds on.
///
/// The whole module is gated once so the individual items do not repeat the
/// provider feature list.
#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "openai-compatible"
))]
pub(super) mod http {
    use std::mem::take;
    use std::sync::Arc;

    use async_trait::async_trait;
    use futures_core::Stream;
    use futures_util::StreamExt as _;
    use futures_util::stream::{iter, unfold};
    #[cfg(any(feature = "openai", feature = "openai-compatible"))]
    use serde::de::DeserializeOwned;

    use crate::adapter::{
        AdapterBuildError, AdapterContext, InputTokenCount, ProviderAdapter, ResolvedCall,
    };
    use crate::catalog::{AdapterId, CatalogProvider};
    use crate::codecs::{Codec, StreamDecoder};
    use crate::credentials::{CredentialProvider, Credentials};
    use crate::transport::{EncodedRequest, HttpTransport, SseEvent};
    use crate::types::{
        Error, ErrorKind, RateLimits, Response, ResponsePolicy, ResponseStream, StreamEvent,
    };

    /// Adapter behavior a factory selects from its typed catalog options.
    ///
    /// Options that only change the wire body belong on the codec instead.
    /// This carries the ones that change how the adapter dispatches.
    #[derive(Clone, Copy, Debug, Default)]
    pub(in crate::providers) struct HttpAdapterOptions {
        /// Completion runs the streaming path and assembles the one response.
        ///
        /// The OpenAI Codex endpoint accepts streaming requests only, so its
        /// factory sets this and completion never sends `stream: false`.
        pub force_streaming_complete: bool,
        /// Every request names the application in an `originator` header,
        /// when the client was given a name.
        ///
        /// The OpenAI Codex deployment expects its clients to identify
        /// themselves this way; the platform API does not read the header.
        pub identify_application:     bool,
    }

    /// The header the OpenAI Codex deployment reads the application name from.
    const ORIGINATOR_HEADER: &str = "originator";

    /// Deserializes a provider's raw `adapter_options` into a factory's shape.
    ///
    /// An absent table is the default options rather than an error, so a
    /// catalog that says nothing about an adapter keeps working.
    ///
    /// # Errors
    ///
    /// Returns [`AdapterBuildError::InvalidAdapterOptions`] when the table does
    /// not match `T`. The client turns that into one provider build issue and
    /// still builds every other provider.
    #[cfg(any(feature = "openai", feature = "openai-compatible"))]
    pub(in crate::providers) fn adapter_options<T: Default + DeserializeOwned>(
        provider: &CatalogProvider,
    ) -> Result<T, AdapterBuildError> {
        let options = provider.adapter_options();
        if options.is_null() {
            return Ok(T::default());
        }
        serde_json::from_value(options.clone()).map_err(|source| {
            AdapterBuildError::InvalidAdapterOptions {
                provider: provider.id().clone(),
                source,
            }
        })
    }

    /// Builds the shared HTTP adapter for one catalog provider.
    ///
    /// # Errors
    ///
    /// Returns [`AdapterBuildError::UnsupportedCodec`] when the catalog pairs
    /// this adapter with a codec it does not implement.
    pub(in crate::providers) fn build_http_adapter(
        provider: &CatalogProvider,
        context: &AdapterContext,
        expected_codec: &str,
        codec: impl Codec + 'static,
        options: HttpAdapterOptions,
    ) -> Result<Arc<dyn ProviderAdapter>, AdapterBuildError> {
        if provider.codec().as_str() != expected_codec {
            return Err(AdapterBuildError::UnsupportedCodec {
                provider: provider.id().clone(),
                codec:    provider.codec().clone(),
            });
        }
        Ok(Arc::new(HttpProviderAdapter {
            id: provider.adapter().clone(),
            codec: Arc::new(codec),
            transport: HttpTransport::new(context.http().clone())
                .with_stream_idle_timeout(context.stream_idle_timeout())
                .with_response_limits(context.response_limits()),
            policy: context.response_policy(),
            credentials: context.credentials().clone(),
            application: options
                .identify_application
                .then(|| context.application().map(str::to_owned))
                .flatten(),
            options,
        }))
    }

    struct HttpProviderAdapter {
        policy:      ResponsePolicy,
        id:          AdapterId,
        codec:       Arc<dyn Codec>,
        transport:   HttpTransport,
        credentials: Arc<dyn CredentialProvider>,
        options:     HttpAdapterOptions,
        /// The `originator` value, when this adapter identifies its
        /// application.
        application: Option<String>,
    }

    impl HttpProviderAdapter {
        /// Adds the application header the provider expects, if any.
        fn identify(&self, encoded: &mut EncodedRequest) {
            if let Some(application) = &self.application {
                encoded
                    .headers
                    .push((ORIGINATOR_HEADER.to_owned(), application.clone()));
            }
        }

        async fn resolve_credentials(&self, call: &ResolvedCall) -> Result<Credentials, Error> {
            self.credentials
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
                })
        }

        /// Completes by running the streaming path and taking its one response.
        ///
        /// A protocol that only accepts streaming requests still owes the
        /// caller a complete response, so the leading rate limits and the
        /// terminal `Ended` event are folded back into one [`Response`].
        async fn complete_by_streaming(&self, call: &ResolvedCall) -> Result<Response, Error> {
            let mut stream = self.policy.stream(self.stream(call).await?);
            let mut rate_limits: Option<RateLimits> = None;
            let mut completed: Option<Response> = None;
            while let Some(event) = stream.next().await {
                match event? {
                    StreamEvent::RateLimits {
                        rate_limits: limits,
                    } => rate_limits = Some(limits),
                    StreamEvent::Ended { response } => completed = Some(*response),
                    _ => {}
                }
            }
            let mut response = completed.ok_or_else(|| {
                Error::new(
                    ErrorKind::Provider,
                    format!(
                        "provider {} ended the stream without a complete response",
                        call.route().provider().id()
                    ),
                )
                .with_provider(call.route().provider().id().clone())
            })?;
            if response.rate_limits.is_none() {
                response.rate_limits = rate_limits;
            }
            Ok(response)
        }
    }

    #[async_trait]
    impl ProviderAdapter for HttpProviderAdapter {
        fn id(&self) -> &AdapterId {
            &self.id
        }

        async fn complete(&self, call: &ResolvedCall) -> Result<Response, Error> {
            if self.options.force_streaming_complete {
                return self.complete_by_streaming(call).await;
            }
            let credentials = self.resolve_credentials(call).await?;
            let mut encoded = self.codec.encode(call, false)?;
            self.identify(&mut encoded);
            let warnings = take(&mut encoded.warnings);
            // Cost estimation uses the speed the codec put on the wire, not
            // the requested one, so a protocol without a speed control is
            // billed at the standard rates the provider actually charges.
            let speed = encoded.applied_speed;
            let result = self
                .transport
                .execute_json(encoded, call.route().provider(), credentials)
                .await?;
            let mut response = self.codec.decode_response(call.route(), result.body)?;
            response.rate_limits = result.rate_limits;
            response.warnings.extend(warnings);
            call.route().apply_catalog_cost(&mut response, speed);
            Ok(response)
        }

        async fn stream(&self, call: &ResolvedCall) -> Result<ResponseStream, Error> {
            let credentials = self.resolve_credentials(call).await?;
            let mut encoded = self.codec.encode(call, true)?;
            self.identify(&mut encoded);
            let warnings = take(&mut encoded.warnings);
            let speed = encoded.applied_speed;
            let accepted = self
                .transport
                .sse_events(encoded, call.route().provider(), credentials)
                .await?;
            let decoder = self.codec.stream_decoder(call.route());
            let route = call.route().clone();
            let limits = iter(
                accepted
                    .rate_limits
                    .into_iter()
                    .map(|rate_limits| Ok(StreamEvent::RateLimits { rate_limits })),
            );
            let decoded = decode_stream(accepted.events, decoder).map(move |event| {
                event.map(|mut event| {
                    if let StreamEvent::Ended { response } = &mut event {
                        response.warnings.extend(warnings.iter().cloned());
                        route.apply_catalog_cost(response, speed);
                    }
                    event
                })
            });
            Ok(ResponseStream::new(limits.chain(decoded)))
        }

        async fn count_input_tokens(
            &self,
            call: &ResolvedCall,
        ) -> Result<Option<InputTokenCount>, Error> {
            let Some(mut encoded) = self.codec.encode_count_tokens(call).transpose()? else {
                return Ok(None);
            };
            self.identify(&mut encoded);
            let credentials = self.resolve_credentials(call).await?;
            let result = self
                .transport
                .execute_json(encoded, call.route().provider(), credentials)
                .await?;
            let tokens = self.codec.decode_count_tokens(call.route(), result.body)?;
            Ok(Some(InputTokenCount::new(tokens, call.route().handle())))
        }
    }

    /// Per-stream decoder state, dropped as soon as the stream ends.
    struct Decoding<S> {
        events:  S,
        decoder: Box<dyn StreamDecoder>,
    }

    /// Drives one [`StreamDecoder`] over the transport events.
    ///
    /// [`StreamDecoder::finish`] runs exactly once, and only when the transport
    /// ended without an error. Any error — from the transport or from the
    /// decoder — is the last item of the stream, so a failed stream can never
    /// carry a `Ended` event.
    fn decode_stream<S>(
        events: S,
        decoder: Box<dyn StreamDecoder>,
    ) -> impl Stream<Item = Result<StreamEvent, Error>> + Send
    where
        S: Stream<Item = Result<SseEvent, Error>> + Send + Unpin,
    {
        unfold(Some(Decoding { events, decoder }), |state| async move {
            let mut state = state?;
            let batch = match state.events.next().await {
                Some(Ok(event)) => match state.decoder.decode(event) {
                    Ok(events) => return Some((ok_batch(events), Some(state))),
                    Err(error) => vec![Err(error)],
                },
                Some(Err(error)) => vec![Err(error)],
                None => match state.decoder.finish() {
                    Ok(events) => ok_batch(events),
                    Err(error) => vec![Err(error)],
                },
            };
            Some((batch, None))
        })
        .flat_map(iter)
    }

    fn ok_batch(events: Vec<StreamEvent>) -> Vec<Result<StreamEvent, Error>> {
        events.into_iter().map(Ok).collect()
    }

    #[cfg(test)]
    mod tests {
        use std::error::Error as StdError;
        use std::sync::Arc;

        use futures_util::StreamExt as _;
        use futures_util::stream::iter;
        use httpmock::{Method, MockServer};
        use reqwest::Method as HttpMethod;
        use serde_json::{Value, json};

        use super::{HttpAdapterOptions, build_http_adapter, decode_stream};
        use crate::adapter::{AdapterContext, ProviderAdapter, ResolvedCall};
        use crate::catalog::{Catalog, ProviderId};
        use crate::codecs::{Codec, StreamDecoder};
        use crate::credentials::NoCredentials;
        use crate::middleware::CallContext;
        use crate::resolver::{AvailableProviders, CatalogResolver, ModelResolver, ResolvedRoute};
        use crate::transport::{EncodedRequest, SseEvent};
        use crate::types::{
            ContentBlockId, Cost, CostSource, Error, ErrorKind, Request, Response, StreamEvent,
            TokenCounts,
        };

        /// A codec that answers whatever the test needs, with no real protocol.
        #[derive(Clone, Debug, Default)]
        struct FakeCodec {
            /// The count endpoint path, or `None` for a dialect without one.
            count_path:      Option<String>,
            /// The cost a decoded response carries, as a provider would report.
            cost:            Option<Cost>,
            usage:           TokenCounts,
            /// A decoder that ends the stream without a `Ended` event.
            never_completes: bool,
            /// Encoding reports one control this protocol cannot express.
            warns:           bool,
        }

        impl FakeCodec {
            fn response(&self, route: &ResolvedRoute) -> Response {
                let mut response = Response::new(
                    route.provider().id().clone(),
                    route.model().id().clone(),
                    Vec::new(),
                );
                response.usage = self.usage;
                response.cost = self.cost;
                response
            }
        }

        impl Codec for FakeCodec {
            fn encode(&self, call: &ResolvedCall, _stream: bool) -> Result<EncodedRequest, Error> {
                let encoded = EncodedRequest::new(
                    HttpMethod::POST,
                    format!("{}/generate", call.route().provider().base_url()),
                    json!({}),
                );
                if self.warns {
                    return Ok(encoded.unsupported_control("stop sequences"));
                }
                Ok(encoded)
            }

            fn decode_response(
                &self,
                route: &ResolvedRoute,
                _value: Value,
            ) -> Result<Response, Error> {
                Ok(self.response(route))
            }

            fn stream_decoder(&self, route: &ResolvedRoute) -> Box<dyn StreamDecoder> {
                Box::new(FakeDecoder {
                    response:        self.response(route),
                    never_completes: self.never_completes,
                })
            }

            fn encode_count_tokens(
                &self,
                call: &ResolvedCall,
            ) -> Option<Result<EncodedRequest, Error>> {
                let path = self.count_path.as_ref()?;
                Some(Ok(EncodedRequest::new(
                    HttpMethod::POST,
                    format!("{}{path}", call.route().provider().base_url()),
                    json!({}),
                )))
            }

            fn decode_count_tokens(
                &self,
                _route: &ResolvedRoute,
                value: Value,
            ) -> Result<u64, Error> {
                value.get("tokens").and_then(Value::as_u64).ok_or_else(|| {
                    Error::new(ErrorKind::ResponseDecode, "the count body has no tokens")
                })
            }
        }

        /// Emits one text delta per transport event and completes at the end.
        struct FakeDecoder {
            response:        Response,
            never_completes: bool,
        }

        impl StreamDecoder for FakeDecoder {
            fn decode(&mut self, event: SseEvent) -> Result<Vec<StreamEvent>, Error> {
                Ok(vec![StreamEvent::TextDelta {
                    id:   ContentBlockId::new("block-0"),
                    text: event.data,
                }])
            }

            fn finish(&mut self) -> Result<Vec<StreamEvent>, Error> {
                if self.never_completes {
                    return Ok(Vec::new());
                }
                Ok(vec![StreamEvent::Ended {
                    response: Box::new(self.response.clone()),
                }])
            }
        }

        /// A one-provider catalog whose base URL and pricing the test chooses.
        fn catalog(base_url: &str) -> Result<Catalog, Box<dyn StdError>> {
            let source = format!(
                r#"
                schema_version = 1

                [providers.alpha]
                display_name = "Alpha"
                adapter = "test-adapter"
                codec = "test-codec"
                base_url = "{base_url}"
                default_model = "one"
                auth = {{ type = "none" }}

                [providers.alpha.models.one]
                display_name = "One"
                aliases = ["uno"]
                api_model = "alpha-one-v1"
                capabilities = {{ text = true }}
                pricing = {{ input_usd_micros_per_million = 1000000, output_usd_micros_per_million = 2000000 }}
                "#
            );
            Ok(Catalog::builder().toml_layer("test", &source)?.build()?)
        }

        fn call(catalog: &Catalog, model: &str) -> Result<ResolvedCall, Box<dyn StdError>> {
            let request = Request::builder().model(model).user("Hello").build()?;
            let available = AvailableProviders::all(catalog);
            let route = CatalogResolver.resolve(&request, catalog, &available)?;
            Ok(ResolvedCall::new(request, route, CallContext::new()))
        }

        fn adapter(
            catalog: &Catalog,
            codec: FakeCodec,
        ) -> Result<Arc<dyn ProviderAdapter>, Box<dyn StdError>> {
            adapter_with(catalog, codec, HttpAdapterOptions::default())
        }

        fn adapter_with(
            catalog: &Catalog,
            codec: FakeCodec,
            options: HttpAdapterOptions,
        ) -> Result<Arc<dyn ProviderAdapter>, Box<dyn StdError>> {
            let provider = catalog
                .provider_by_id(&ProviderId::new("alpha"))
                .ok_or("the test catalog must define the alpha provider")?;
            let context = AdapterContext::new(reqwest::Client::new(), Arc::new(NoCredentials));
            Ok(build_http_adapter(
                provider,
                &context,
                "test-codec",
                codec,
                options,
            )?)
        }

        #[tokio::test]
        async fn a_codec_without_a_count_endpoint_counts_nothing_and_sends_no_request()
        -> Result<(), Box<dyn StdError>> {
            // Port 1 refuses every connection, so any dispatch would be an
            // error rather than `Ok(None)`.
            let catalog = catalog("http://127.0.0.1:1")?;
            let adapter = adapter(&catalog, FakeCodec::default())?;
            let count = adapter
                .count_input_tokens(&call(&catalog, "alpha/one")?)
                .await?;
            assert!(
                count.is_none(),
                "a codec with no count endpoint counts nothing"
            );
            Ok(())
        }

        #[tokio::test]
        async fn a_count_carries_the_canonical_model_for_an_alias_route()
        -> Result<(), Box<dyn StdError>> {
            let server = MockServer::start_async().await;
            let mock = server
                .mock_async(|when, then| {
                    when.method(Method::POST).path("/count");
                    then.status(200).json_body(json!({ "tokens": 42 }));
                })
                .await;
            let catalog = catalog(&server.base_url())?;
            let codec = FakeCodec {
                count_path: Some("/count".to_owned()),
                ..FakeCodec::default()
            };
            let adapter = adapter(&catalog, codec)?;

            let count = adapter
                .count_input_tokens(&call(&catalog, "alpha/uno")?)
                .await?
                .ok_or("a codec with a count endpoint must return a count")?;

            mock.assert_async().await;
            assert_eq!(count.tokens(), 42);
            assert_eq!(count.model().provider().as_str(), "alpha");
            assert_eq!(count.model().model().as_str(), "one");
            Ok(())
        }

        #[tokio::test]
        async fn the_catalog_prices_a_response_that_carries_no_provider_cost()
        -> Result<(), Box<dyn StdError>> {
            let server = MockServer::start_async().await;
            let mock = server
                .mock_async(|when, then| {
                    when.method(Method::POST).path("/generate");
                    then.status(200).json_body(json!({}));
                })
                .await;
            let catalog = catalog(&server.base_url())?;
            let codec = FakeCodec {
                usage: TokenCounts {
                    input: 1_000_000,
                    output: 1_000_000,
                    ..TokenCounts::default()
                },
                ..FakeCodec::default()
            };
            let adapter = adapter(&catalog, codec)?;

            let response = adapter.complete(&call(&catalog, "alpha/one")?).await?;

            mock.assert_async().await;
            assert_eq!(
                response.cost,
                Some(Cost {
                    usd_micros: 3_000_000,
                    source:     CostSource::Catalog,
                })
            );
            Ok(())
        }

        #[tokio::test]
        async fn a_provider_reported_cost_is_never_overwritten() -> Result<(), Box<dyn StdError>> {
            let server = MockServer::start_async().await;
            let mock = server
                .mock_async(|when, then| {
                    when.method(Method::POST).path("/generate");
                    then.status(200).json_body(json!({}));
                })
                .await;
            let catalog = catalog(&server.base_url())?;
            let reported = Cost {
                usd_micros: 7,
                source:     CostSource::Provider,
            };
            let codec = FakeCodec {
                cost: Some(reported),
                usage: TokenCounts {
                    input: 1_000_000,
                    output: 1_000_000,
                    ..TokenCounts::default()
                },
                ..FakeCodec::default()
            };
            let adapter = adapter(&catalog, codec)?;

            let response = adapter.complete(&call(&catalog, "alpha/one")?).await?;

            mock.assert_async().await;
            assert_eq!(response.cost, Some(reported));
            Ok(())
        }

        #[tokio::test]
        async fn a_transport_failure_ends_the_stream_without_completing()
        -> Result<(), Box<dyn StdError>> {
            let catalog = catalog("http://127.0.0.1:1")?;
            let route = call(&catalog, "alpha/one")?.route().clone();
            let decoder = FakeCodec::default().stream_decoder(&route);
            let transport = iter(vec![
                Ok(SseEvent {
                    event: None,
                    data:  "first".to_owned(),
                }),
                Err(Error::new(ErrorKind::Network, "the connection dropped")),
            ]);

            let events: Vec<_> = decode_stream(transport, decoder).collect().await;

            assert_eq!(events.len(), 2, "the error must be the last item");
            assert!(matches!(events[0], Ok(StreamEvent::TextDelta { .. })));
            let error = match &events[1] {
                Err(error) => error,
                Ok(event) => panic!("the stream must end with the transport error: {event:?}"),
            };
            assert_eq!(error.kind(), ErrorKind::Network);
            assert!(
                !events
                    .iter()
                    .any(|event| matches!(event, Ok(StreamEvent::Ended { .. }))),
                "a failed stream must not complete"
            );
            Ok(())
        }

        #[tokio::test]
        async fn forced_streaming_completion_returns_the_assembled_response()
        -> Result<(), Box<dyn StdError>> {
            let server = MockServer::start_async().await;
            let mock = server
                .mock_async(|when, then| {
                    when.method(Method::POST).path("/generate");
                    then.status(200)
                        .header("content-type", "text/event-stream")
                        .body("data: hello\n\n");
                })
                .await;
            let catalog = catalog(&server.base_url())?;
            let codec = FakeCodec {
                usage: TokenCounts {
                    input: 1_000_000,
                    output: 1_000_000,
                    ..TokenCounts::default()
                },
                ..FakeCodec::default()
            };
            let adapter = adapter_with(&catalog, codec, HttpAdapterOptions {
                force_streaming_complete: true,
                identify_application:     false,
            })?;

            let response = adapter.complete(&call(&catalog, "alpha/one")?).await?;

            mock.assert_async().await;
            assert_eq!(response.model.model().as_str(), "one");
            assert_eq!(
                response.cost.map(|cost| cost.source),
                Some(CostSource::Catalog),
                "the streamed response is priced like any other"
            );
            Ok(())
        }

        #[tokio::test]
        async fn forced_streaming_completion_fails_when_the_stream_never_completes()
        -> Result<(), Box<dyn StdError>> {
            let server = MockServer::start_async().await;
            server
                .mock_async(|when, then| {
                    when.method(Method::POST).path("/generate");
                    then.status(200)
                        .header("content-type", "text/event-stream")
                        .body("data: hello\n\n");
                })
                .await;
            let catalog = catalog(&server.base_url())?;
            let codec = FakeCodec {
                never_completes: true,
                ..FakeCodec::default()
            };
            let adapter = adapter_with(&catalog, codec, HttpAdapterOptions {
                force_streaming_complete: true,
                identify_application:     false,
            })?;

            let error = match adapter.complete(&call(&catalog, "alpha/one")?).await {
                Err(error) => error,
                Ok(response) => panic!("an incomplete stream must not complete: {response:?}"),
            };

            assert_eq!(error.kind(), ErrorKind::StreamDecode);
            Ok(())
        }

        #[tokio::test]
        async fn an_encode_warning_reaches_the_completed_response() -> Result<(), Box<dyn StdError>>
        {
            let server = MockServer::start_async().await;
            server
                .mock_async(|when, then| {
                    when.method(Method::POST).path("/generate");
                    then.status(200).json_body(json!({}));
                })
                .await;
            let catalog = catalog(&server.base_url())?;
            let codec = FakeCodec {
                warns: true,
                ..FakeCodec::default()
            };
            let adapter = adapter(&catalog, codec)?;

            let response = adapter.complete(&call(&catalog, "alpha/one")?).await?;

            let warning = response
                .warnings
                .first()
                .ok_or("the encode warning must reach the response")?;
            assert_eq!(warning.code, "unsupported_control");
            Ok(())
        }

        #[tokio::test]
        async fn an_encode_warning_reaches_the_streamed_completion() -> Result<(), Box<dyn StdError>>
        {
            let server = MockServer::start_async().await;
            server
                .mock_async(|when, then| {
                    when.method(Method::POST).path("/generate");
                    then.status(200)
                        .header("content-type", "text/event-stream")
                        .body("data: hello\n\n");
                })
                .await;
            let catalog = catalog(&server.base_url())?;
            let codec = FakeCodec {
                warns: true,
                ..FakeCodec::default()
            };
            let adapter = adapter(&catalog, codec)?;

            let events: Vec<_> = adapter
                .stream(&call(&catalog, "alpha/one")?)
                .await?
                .collect()
                .await;

            let completed = events
                .into_iter()
                .find_map(|event| match event {
                    Ok(StreamEvent::Ended { response }) => Some(response),
                    _ => None,
                })
                .ok_or("the stream must complete")?;
            let warning = completed
                .warnings
                .first()
                .ok_or("the encode warning must reach the completed response")?;
            assert_eq!(warning.code, "unsupported_control");
            Ok(())
        }

        #[tokio::test]
        async fn a_clean_stream_end_completes_exactly_once() -> Result<(), Box<dyn StdError>> {
            let catalog = catalog("http://127.0.0.1:1")?;
            let route = call(&catalog, "alpha/one")?.route().clone();
            let decoder = FakeCodec::default().stream_decoder(&route);
            let transport = iter(vec![Ok(SseEvent {
                event: None,
                data:  "first".to_owned(),
            })]);

            let events: Vec<_> = decode_stream(transport, decoder).collect().await;

            let completions = events
                .iter()
                .filter(|event| matches!(event, Ok(StreamEvent::Ended { .. })))
                .count();
            assert_eq!(completions, 1);
            Ok(())
        }
    }
}
