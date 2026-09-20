//! Built-in provider adapter factories.
//!
//! Two adapters serve every built-in codec: `http` speaks plain HTTPS with a
//! bearer or header credential and holds every codec its provider lists;
//! `bedrock` adds SigV4 signing and binary event-stream framing for the
//! Converse codec. Each dispatches on the codec the client selected for the
//! call, so one provider can speak several protocols.

#[cfg(feature = "bedrock")]
mod bedrock;

use crate::adapter::AdapterRegistry;
use crate::catalog::adapter_ids;

/// Registers the two built-in factories under their catalog ids.
pub(crate) fn register_builtin(registry: &mut AdapterRegistry) {
    registry.register_factory(adapter_ids::HTTP, http::Factory);
    #[cfg(feature = "bedrock")]
    registry.register_factory(adapter_ids::BEDROCK, bedrock::Factory);
}

/// The `http` adapter: every codec a provider lists, over one transport.
pub(super) mod http {
    use std::collections::BTreeMap;
    use std::mem::take;
    use std::sync::Arc;

    use async_trait::async_trait;
    use futures_core::Stream;
    use futures_util::StreamExt as _;
    use futures_util::stream::{iter, unfold};
    use serde::Deserialize;
    use serde::de::DeserializeOwned;

    use crate::adapter::{
        AdapterBuildError, AdapterContext, AdapterFactory, InputTokenCount, ProviderAdapter,
        ResolvedCall, ResolvedEvaluation,
    };
    use crate::catalog::{AdapterId, CatalogProvider, CodecId};
    use crate::codecs::{self, BuiltCodec, Codec, CodecBuildError, EvaluationCodec, StreamDecoder};
    use crate::credentials::{CredentialProvider, Credentials};
    use crate::evaluation::Verdict;
    use crate::transport::{EncodedRequest, HttpTransport, SseEvent};
    use crate::types::{
        Error, ErrorKind, RateLimits, Response, ResponsePolicy, ResponseStream, Speed, StreamEvent,
        Warning,
    };

    /// Builds the `http` adapter for a catalog provider.
    pub(super) struct Factory;

    impl AdapterFactory for Factory {
        /// Constructs every codec the provider lists, each from its own
        /// `codec_options` table, and one adapter holding them all.
        ///
        /// # Errors
        ///
        /// [`AdapterBuildError::UnsupportedCodec`] for a codec id this crate
        /// does not build and [`AdapterBuildError::InvalidCodecOptions`] for
        /// one whose options do not parse. Either leaves the provider with no
        /// adapter: a provider with a broken codec is not half-available.
        fn create(
            &self,
            provider: &CatalogProvider,
            context: &AdapterContext,
        ) -> Result<Arc<dyn ProviderAdapter>, AdapterBuildError> {
            let options: HttpAdapterOptions = adapter_options(provider)?;
            let mut built = BTreeMap::new();
            for id in provider.codecs() {
                let codec =
                    codecs::build(id, provider.codec_options(id)).map_err(|error| match error {
                        CodecBuildError::UnknownCodec { codec } => {
                            AdapterBuildError::UnsupportedCodec {
                                provider: provider.id().clone(),
                                codec,
                            }
                        }
                        CodecBuildError::InvalidOptions { codec, source } => {
                            AdapterBuildError::InvalidCodecOptions {
                                provider: provider.id().clone(),
                                codec,
                                source,
                            }
                        }
                    })?;
                built.insert(id.clone(), codec);
            }
            Ok(build_http_adapter(provider, context, built, options))
        }
    }

    /// The typed `adapter_options` table of an `http` provider.
    ///
    /// Options that change the wire body belong in `codec_options` under
    /// their codec's id. This carries the ones that change how the adapter
    /// dispatches. Unknown keys are rejected so a misspelled option is a
    /// build issue for one provider rather than a silently ignored setting.
    #[derive(Clone, Copy, Debug, Default, Deserialize)]
    #[serde(default, deny_unknown_fields)]
    pub(in crate::providers) struct HttpAdapterOptions {
        /// Completion runs the streaming path and assembles the one response.
        ///
        /// The OpenAI Codex endpoint accepts streaming requests only, so its
        /// catalog row sets this and completion never sends `stream: false`.
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
    pub(in crate::providers) fn adapter_options<T: Default + DeserializeOwned>(
        provider: &CatalogProvider,
    ) -> Result<T, AdapterBuildError> {
        typed_options(provider, provider.adapter_options())
    }

    fn typed_options<T: Default + DeserializeOwned>(
        provider: &CatalogProvider,
        options: &serde_json::Value,
    ) -> Result<T, AdapterBuildError> {
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

    /// Reads the credentials for one provider attempt.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Authentication`] with the store's failure as its
    /// source when the provider's credentials cannot be resolved.
    pub(in crate::providers) async fn resolve_credentials(
        credentials: &dyn CredentialProvider,
        provider: &CatalogProvider,
    ) -> Result<Credentials, Error> {
        credentials.credentials(provider).await.map_err(|source| {
            Error::new(
                ErrorKind::Authentication,
                format!(
                    "credentials for provider {} could not be resolved",
                    provider.id()
                ),
            )
            .with_provider(provider.id().clone())
            .with_source(source)
        })
    }

    /// Builds the `http` adapter over already-constructed codecs.
    ///
    /// `options` is the provider's typed `adapter_options`; the factory
    /// reads it with [`adapter_options`], and a test can pass its own.
    pub(in crate::providers) fn build_http_adapter(
        provider: &CatalogProvider,
        context: &AdapterContext,
        codecs: BTreeMap<CodecId, BuiltCodec>,
        options: HttpAdapterOptions,
    ) -> Arc<dyn ProviderAdapter> {
        Arc::new(HttpProviderAdapter {
            id: provider.adapter().clone(),
            codecs,
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
        })
    }

    struct HttpProviderAdapter {
        policy:      ResponsePolicy,
        id:          AdapterId,
        /// Every codec the provider lists, by id. The client names one per
        /// call on the resolved call.
        codecs:      BTreeMap<CodecId, BuiltCodec>,
        transport:   HttpTransport,
        credentials: Arc<dyn CredentialProvider>,
        options:     HttpAdapterOptions,
        /// The `originator` value, when this adapter identifies its
        /// application.
        application: Option<String>,
    }

    impl HttpProviderAdapter {
        /// The generation codec the client selected for `call`.
        ///
        /// # Errors
        ///
        /// `Configuration` when no codec was selected, the selected codec is
        /// not one this adapter holds, or it serves evaluation rather than
        /// generation. The client selects from the same catalog this adapter
        /// was built from, so any of these is an internal invariant failure
        /// rather than a caller mistake.
        fn generation_codec(
            &self,
            call: &ResolvedCall,
            operation: &str,
        ) -> Result<&dyn Codec, Error> {
            match self.built_codec(call.route().provider(), call.codec(), operation)? {
                BuiltCodec::Generation(codec) => Ok(codec.as_ref()),
                BuiltCodec::Evaluation(_) => Err(wrong_family(
                    call.route().provider(),
                    call.codec(),
                    operation,
                    "generation",
                )),
            }
        }

        /// The evaluation codec the client selected for `call`.
        ///
        /// # Errors
        ///
        /// As [`generation_codec`](Self::generation_codec), for the
        /// evaluation family.
        fn evaluation_codec(
            &self,
            call: &ResolvedEvaluation,
        ) -> Result<&dyn EvaluationCodec, Error> {
            match self.built_codec(call.route().provider(), call.codec(), "evaluate")? {
                BuiltCodec::Evaluation(codec) => Ok(codec.as_ref()),
                BuiltCodec::Generation(_) => Err(wrong_family(
                    call.route().provider(),
                    call.codec(),
                    "evaluate",
                    "evaluation",
                )),
            }
        }

        fn built_codec(
            &self,
            provider: &CatalogProvider,
            codec: Option<&CodecId>,
            operation: &str,
        ) -> Result<&BuiltCodec, Error> {
            let Some(codec) = codec else {
                return Err(internal_error(
                    provider,
                    &format!(
                        "the http adapter received {operation} for provider {} with no codec \
                         selected",
                        provider.id()
                    ),
                ));
            };
            self.codecs.get(codec).ok_or_else(|| {
                internal_error(
                    provider,
                    &format!(
                        "the http adapter received {operation} for provider {} on codec {codec}, \
                         which the provider does not list",
                        provider.id()
                    ),
                )
            })
        }

        /// Adds the application header the provider expects, if any.
        fn identify(&self, encoded: &mut EncodedRequest) {
            if let Some(application) = &self.application {
                encoded
                    .headers
                    .push((ORIGINATOR_HEADER.to_owned(), application.clone()));
            }
        }

        async fn resolve_credentials(
            &self,
            provider: &CatalogProvider,
        ) -> Result<Credentials, Error> {
            resolve_credentials(self.credentials.as_ref(), provider).await
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

    /// A codec that serves the other operation family was selected.
    fn wrong_family(
        provider: &CatalogProvider,
        codec: Option<&CodecId>,
        operation: &str,
        family: &str,
    ) -> Error {
        let codec = codec.map(ToString::to_string).unwrap_or_default();
        internal_error(
            provider,
            &format!(
                "the http adapter received {operation} for provider {} on codec {codec}, which \
                 does not serve {family}",
                provider.id()
            ),
        )
    }

    /// A codec-selection invariant the client is expected to uphold failed.
    fn internal_error(provider: &CatalogProvider, message: &str) -> Error {
        Error::new(
            ErrorKind::Configuration,
            format!("internal error: {message}"),
        )
        .with_provider(provider.id().clone())
    }

    #[async_trait]
    impl ProviderAdapter for HttpProviderAdapter {
        fn id(&self) -> &AdapterId {
            &self.id
        }

        async fn complete(&self, call: &ResolvedCall) -> Result<Response, Error> {
            let codec = self.generation_codec(call, "complete")?;
            if self.options.force_streaming_complete {
                return self.complete_by_streaming(call).await;
            }
            let provider = call.route().provider();
            let credentials = self.resolve_credentials(provider).await?;
            let mut encoded = codec.encode(call, false)?;
            self.identify(&mut encoded);
            let warnings = take(&mut encoded.warnings);
            // Cost estimation uses the speed the codec put on the wire, not
            // the requested one, so a protocol without a speed control is
            // billed at the standard rates the provider actually charges.
            let speed = encoded.applied_speed;
            let result = self
                .transport
                .execute_json(encoded.prepare(provider, &credentials)?, provider)
                .await?;
            let response = codec.decode_response(call.route(), result.body)?;
            Ok(finish_response(
                response,
                result.rate_limits,
                warnings,
                call,
                speed,
            ))
        }

        async fn stream(&self, call: &ResolvedCall) -> Result<ResponseStream, Error> {
            let codec = self.generation_codec(call, "stream")?;
            let provider = call.route().provider();
            let credentials = self.resolve_credentials(provider).await?;
            let mut encoded = codec.encode(call, true)?;
            self.identify(&mut encoded);
            let warnings = take(&mut encoded.warnings);
            let speed = encoded.applied_speed;
            let accepted = self
                .transport
                .stream_events(encoded.prepare(provider, &credentials)?, provider)
                .await?;
            let decoded = decode_stream(accepted.events, codec.stream_decoder(call.route()));
            Ok(finish_stream(
                accepted.rate_limits,
                decoded,
                warnings,
                call,
                speed,
            ))
        }

        async fn count_input_tokens(
            &self,
            call: &ResolvedCall,
        ) -> Result<Option<InputTokenCount>, Error> {
            let codec = self.generation_codec(call, "count_input_tokens")?;
            let Some(mut encoded) = codec.encode_count_tokens(call).transpose()? else {
                return Ok(None);
            };
            self.identify(&mut encoded);
            let provider = call.route().provider();
            let credentials = self.resolve_credentials(provider).await?;
            let result = self
                .transport
                .execute_json(encoded.prepare(provider, &credentials)?, provider)
                .await?;
            let tokens = codec.decode_count_tokens(call.route(), result.body)?;
            Ok(Some(InputTokenCount::new(tokens, call.route().handle())))
        }

        async fn evaluate(&self, call: &ResolvedEvaluation) -> Result<Verdict, Error> {
            let codec = self.evaluation_codec(call)?;
            let provider = call.route().provider();
            // A refusal the codec can make needs no credentials, so encoding
            // comes first.
            let mut encoded = codec.encode_evaluation(call)?;
            self.identify(&mut encoded);
            let warnings = take(&mut encoded.warnings);
            let credentials = self.resolve_credentials(provider).await?;
            let result = self
                .transport
                .execute_json(encoded.prepare(provider, &credentials)?, provider)
                .await?;
            let header_id = codec
                .id_header()
                .and_then(|header| result.header(header))
                .map(ToOwned::to_owned);
            let mut verdict = codec.decode_verdict(call, result.body)?;
            // A protocol that carries the request id in a header rather than
            // the body leaves `id` empty after decode; the header fills it.
            if verdict.id.as_deref().is_none_or(str::is_empty) {
                verdict.id = header_id;
            }
            verdict.rate_limits = result.rate_limits;
            verdict.warnings.extend(warnings);
            // Catalog pricing fills in only when the provider reported no
            // cost, as `apply_catalog_cost` does for a response.
            if verdict.cost.is_none() {
                verdict.cost = call
                    .route()
                    .estimate_cost_for(call.codec(), verdict.usage, None);
            }
            Ok(verdict)
        }

        /// The client has already chosen the codec by the time a call
        /// arrives here, so this adapter always says yes and the row's
        /// codec set decides native versus judge.
        fn evaluates_natively(&self) -> bool {
            true
        }
    }

    /// Per-stream decoder state, dropped as soon as the stream ends.
    struct Decoding<S> {
        events:  S,
        decoder: Box<dyn StreamDecoder>,
    }

    /// Finishes a decoded response with what the adapter knows and the codec
    /// does not: the response headers' rate limits, the encoder's warnings,
    /// and the catalog cost at the speed the codec put on the wire.
    ///
    /// Cost estimation uses the applied speed, not the requested one, so a
    /// protocol without a speed control is billed at the standard rates the
    /// provider actually charges.
    pub(in crate::providers) fn finish_response(
        mut response: Response,
        rate_limits: Option<RateLimits>,
        warnings: Vec<Warning>,
        call: &ResolvedCall,
        speed: Option<Speed>,
    ) -> Response {
        response.rate_limits = rate_limits;
        response.warnings.extend(warnings);
        call.route()
            .apply_catalog_cost(&mut response, call.codec(), speed);
        response
    }

    /// Finishes a decoded stream the way [`finish_response`] finishes a
    /// response: the response headers' rate limits lead the stream, and the
    /// `Ended` event's response gains the warnings and the catalog cost.
    pub(in crate::providers) fn finish_stream(
        rate_limits: Option<RateLimits>,
        decoded: impl Stream<Item = Result<StreamEvent, Error>> + Send + 'static,
        warnings: Vec<Warning>,
        call: &ResolvedCall,
        speed: Option<Speed>,
    ) -> ResponseStream {
        let route = call.route().clone();
        let selected = call.codec().cloned();
        let limits = iter(
            rate_limits
                .into_iter()
                .map(|rate_limits| Ok(StreamEvent::RateLimits { rate_limits })),
        );
        let finished = decoded.map(move |event| {
            event.map(|mut event| {
                if let StreamEvent::Ended { response } = &mut event {
                    response.warnings.extend(warnings.iter().cloned());
                    route.apply_catalog_cost(response, selected.as_ref(), speed);
                }
                event
            })
        });
        ResponseStream::new(limits.chain(finished))
    }

    /// Drives one [`StreamDecoder`] over the transport events.
    ///
    /// [`StreamDecoder::finish`] runs exactly once, and only when the transport
    /// ended without an error. Any error — from the transport or from the
    /// decoder — is the last item of the stream, so a failed stream can never
    /// carry a `Ended` event.
    pub(in crate::providers) fn decode_stream<S>(
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

        use super::{
            Factory, HttpAdapterOptions, adapter_options, build_http_adapter, decode_stream,
        };
        use crate::adapter::{
            AdapterBuildError, AdapterContext, AdapterFactory as _, ProviderAdapter, ResolvedCall,
            ResolvedEvaluation,
        };
        use crate::catalog::{Catalog, CodecId, ProviderId};
        use crate::codecs::{BuiltCodec, Codec, EvaluationCodec, StreamDecoder};
        use crate::credentials::NoCredentials;
        use crate::evaluation::{Evaluation, Verdict};
        use crate::middleware::CallContext;
        use crate::resolver::{AvailableProviders, CatalogResolver, ModelResolver, ResolvedRoute};
        use crate::transport::{EncodedRequest, SseEvent};
        use crate::types::{
            Answer, BooleanAnswer, ContentBlockId, Cost, CostSource, Error, ErrorKind, Request,
            Response, StreamEvent, TokenCounts,
        };

        /// The codec id the test catalog lists and every fake codec is
        /// registered under.
        const TEST_CODEC: &str = "test-codec";

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
                codecs = ["test-codec"]
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

        /// A call on the test codec, as the client would build it.
        fn call(catalog: &Catalog, model: &str) -> Result<ResolvedCall, Box<dyn StdError>> {
            let request = Request::builder().model(model).user("Hello").build()?;
            let available = AvailableProviders::all(catalog);
            let route = CatalogResolver.resolve(&request, catalog, &available)?;
            Ok(ResolvedCall::new(request, route, CallContext::new())
                .with_codec(Some(CodecId::new(TEST_CODEC))))
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
            let codecs = [(
                CodecId::new(TEST_CODEC),
                BuiltCodec::Generation(Arc::new(codec)),
            )]
            .into_iter()
            .collect();
            Ok(build_http_adapter(provider, &context, codecs, options))
        }

        /// An evaluation codec that posts to `/evaluate` and answers one
        /// boolean, naming `x-request-id` as its id header.
        struct FakeEvaluationCodec;

        impl EvaluationCodec for FakeEvaluationCodec {
            fn encode_evaluation(
                &self,
                call: &ResolvedEvaluation,
            ) -> Result<EncodedRequest, Error> {
                Ok(EncodedRequest::new(
                    HttpMethod::POST,
                    format!("{}/evaluate", call.route().provider().base_url()),
                    json!({}),
                ))
            }

            fn decode_verdict(
                &self,
                call: &ResolvedEvaluation,
                body: Value,
            ) -> Result<Verdict, Error> {
                let answers = call
                    .evaluation()
                    .questions()
                    .keys()
                    .map(|id| {
                        (
                            id.clone(),
                            Answer::Boolean(BooleanAnswer { probability: 0.9 }),
                        )
                    })
                    .collect();
                let mut verdict = Verdict::new(
                    call.route().provider().id().clone(),
                    call.route().model().id().clone(),
                    answers,
                );
                verdict.id = body
                    .get("id")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                verdict.usage = TokenCounts {
                    input: 1_000_000,
                    ..TokenCounts::default()
                };
                Ok(verdict)
            }

            fn id_header(&self) -> Option<&'static str> {
                Some("x-request-id")
            }
        }

        /// An adapter holding the fake evaluation codec under the test id.
        fn evaluation_adapter(
            catalog: &Catalog,
        ) -> Result<Arc<dyn ProviderAdapter>, Box<dyn StdError>> {
            let provider = catalog
                .provider_by_id(&ProviderId::new("alpha"))
                .ok_or("the test catalog must define the alpha provider")?;
            let context = AdapterContext::new(reqwest::Client::new(), Arc::new(NoCredentials));
            let codecs = [(
                CodecId::new(TEST_CODEC),
                BuiltCodec::Evaluation(Arc::new(FakeEvaluationCodec)),
            )]
            .into_iter()
            .collect();
            Ok(build_http_adapter(
                provider,
                &context,
                codecs,
                HttpAdapterOptions::default(),
            ))
        }

        fn evaluation_call(
            catalog: &Catalog,
            codec: Option<&str>,
        ) -> Result<ResolvedEvaluation, Box<dyn StdError>> {
            let evaluation = Evaluation::builder()
                .model("alpha/one")
                .state("I was charged twice.")
                .boolean("requests_refund", "Refund requested?")
                .build()?;
            let stand_in = Request::stand_in(evaluation.model(), "state".to_owned());
            let available = AvailableProviders::all(catalog);
            let route = CatalogResolver.resolve(&stand_in, catalog, &available)?;
            Ok(
                ResolvedEvaluation::new(evaluation, route, CallContext::new())
                    .with_codec(codec.map(CodecId::new)),
            )
        }

        #[tokio::test]
        async fn a_call_with_no_codec_selected_is_an_internal_configuration_error()
        -> Result<(), Box<dyn StdError>> {
            let catalog = catalog("http://127.0.0.1:1")?;
            let adapter = adapter(&catalog, FakeCodec::default())?;
            let call = call(&catalog, "alpha/one")?.with_codec(None);

            let error = adapter
                .complete(&call)
                .await
                .expect_err("no codec means no dispatch");

            assert_eq!(error.kind(), ErrorKind::Configuration);
            assert!(
                error.message().contains("complete") && error.message().contains("no codec"),
                "{}",
                error.message()
            );
            Ok(())
        }

        #[tokio::test]
        async fn a_generation_call_on_an_evaluation_codec_names_the_family()
        -> Result<(), Box<dyn StdError>> {
            let catalog = catalog("http://127.0.0.1:1")?;
            let adapter = evaluation_adapter(&catalog)?;

            let error = adapter
                .stream(&call(&catalog, "alpha/one")?)
                .await
                .err()
                .ok_or("an evaluation codec cannot stream")?;

            assert_eq!(error.kind(), ErrorKind::Configuration);
            assert!(
                error.message().contains(TEST_CODEC)
                    && error.message().contains("stream")
                    && error.message().contains("generation"),
                "{}",
                error.message()
            );

            let error = adapter
                .evaluate(&evaluation_call(&catalog, Some("missing"))?)
                .await
                .expect_err("an unlisted codec is refused");
            assert_eq!(error.kind(), ErrorKind::Configuration);
            assert!(error.message().contains("missing"), "{}", error.message());
            Ok(())
        }

        #[tokio::test]
        async fn a_verdict_takes_rate_limits_a_header_id_and_a_catalog_price()
        -> Result<(), Box<dyn StdError>> {
            let server = MockServer::start_async().await;
            let mock = server
                .mock_async(|when, then| {
                    when.method(Method::POST).path("/evaluate");
                    then.status(200)
                        .header("x-request-id", "req_from_header")
                        .header("x-ratelimit-remaining-requests", "41")
                        .json_body(json!({}));
                })
                .await;
            let catalog = catalog(&server.base_url())?;
            let adapter = evaluation_adapter(&catalog)?;
            let call = evaluation_call(&catalog, Some(TEST_CODEC))?;

            let verdict = adapter.evaluate(&call).await?;

            mock.assert_async().await;
            assert_eq!(verdict.id.as_deref(), Some("req_from_header"));
            assert_eq!(
                verdict
                    .rate_limits
                    .as_ref()
                    .and_then(|limits| limits.request_remaining),
                Some(41)
            );
            let cost = verdict.cost.ok_or("the catalog prices the verdict")?;
            assert_eq!(cost.source, CostSource::Catalog);
            assert_eq!(cost.usd_micros, 1_000_000, "one dollar per million input");
            Ok(())
        }

        #[tokio::test]
        async fn a_body_id_wins_over_the_header_id() -> Result<(), Box<dyn StdError>> {
            let server = MockServer::start_async().await;
            server
                .mock_async(|when, then| {
                    when.method(Method::POST).path("/evaluate");
                    then.status(200)
                        .header("x-request-id", "req_from_header")
                        .json_body(json!({ "id": "req_from_body" }));
                })
                .await;
            let catalog = catalog(&server.base_url())?;
            let adapter = evaluation_adapter(&catalog)?;

            let verdict = adapter
                .evaluate(&evaluation_call(&catalog, Some(TEST_CODEC))?)
                .await?;

            assert_eq!(verdict.id.as_deref(), Some("req_from_body"));
            assert!(verdict.rate_limits.is_none(), "no limit headers, no limits");
            Ok(())
        }

        /// One provider whose options tables are written per test.
        fn options_layer(codecs: &str, options: &str) -> String {
            format!(
                r#"
                schema_version = 1

                [providers.oai]
                display_name = "OpenAI"
                codecs = {codecs}
                base_url = "http://127.0.0.1"
                auth = {{ type = "none" }}
                {options}
                "#
            )
        }

        fn create(
            source: &str,
        ) -> Result<Result<Arc<dyn ProviderAdapter>, AdapterBuildError>, Box<dyn StdError>>
        {
            let catalog = Catalog::builder().toml_layer("test", source)?.build()?;
            let provider = catalog
                .provider_by_id(&ProviderId::new("oai"))
                .ok_or("the test catalog must define the oai provider")?;
            let context = AdapterContext::new(reqwest::Client::new(), Arc::new(NoCredentials));
            Ok(Factory.create(provider, &context))
        }

        #[test]
        fn the_factory_builds_every_listed_codec() -> Result<(), Box<dyn StdError>> {
            let adapter = create(&options_layer(
                r#"["openai-chat", "openai-responses", "anthropic-messages", "gemini-generate", "vercel-evaluation"]"#,
                "",
            ))??;
            assert_eq!(adapter.id().as_str(), "http");
            assert!(adapter.evaluates_natively());
            Ok(())
        }

        #[test]
        fn an_unknown_codec_is_an_unsupported_codec_error() -> Result<(), Box<dyn StdError>> {
            let error = create(&options_layer(r#"["openai-chat", "made-up"]"#, ""))?
                .err()
                .ok_or("an unknown codec fails the whole provider")?;
            assert!(
                matches!(&error, AdapterBuildError::UnsupportedCodec { codec, .. } if codec.as_str() == "made-up"),
                "{error}"
            );
            Ok(())
        }

        #[test]
        fn invalid_codec_options_name_the_codec() -> Result<(), Box<dyn StdError>> {
            for options in [
                "codec_options = { openai-responses = { mode = \"turbo\" } }",
                "codec_options = { openai-responses = { made_up = true } }",
            ] {
                let error = create(&options_layer(r#"["openai-responses"]"#, options))?
                    .err()
                    .ok_or("undeclared options are refused")?;
                assert!(
                    matches!(&error, AdapterBuildError::InvalidCodecOptions { codec, .. } if codec.as_str() == "openai-responses"),
                    "{error}"
                );
            }
            Ok(())
        }

        /// An `adapter_options` key that belongs to a codec is rejected, so a
        /// row migrated by hand cannot leave `mode` in the wrong table.
        #[test]
        fn a_codec_option_in_the_adapter_table_is_rejected() -> Result<(), Box<dyn StdError>> {
            let error = create(&options_layer(
                r#"["openai-responses"]"#,
                "adapter_options = { mode = \"codex\" }",
            ))?
            .err()
            .ok_or("`mode` is a codec option")?;
            assert!(
                matches!(error, AdapterBuildError::InvalidAdapterOptions { .. }),
                "{error}"
            );
            Ok(())
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

            let adapter: HttpAdapterOptions = adapter_options(provider)?;
            assert!(adapter.force_streaming_complete);
            assert!(adapter.identify_application);
            assert_eq!(
                provider.codec_options(&CodecId::new("openai-responses")),
                &json!({ "mode": "codex" })
            );
            Ok(())
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
