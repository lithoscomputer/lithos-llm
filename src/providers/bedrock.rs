//! The Amazon Bedrock provider adapter.
//!
//! One HTTP path serves both Bedrock authentication methods. Every call
//! resolves credentials, encodes through [`BedrockConverseCodec`], dispatches
//! the resulting [`EncodedRequest`], and decodes through the same codec. A
//! bearer request and a SigV4 request are built from that one encoded request,
//! so they always carry the same method, URL, and body; only the
//! authentication headers differ.
//!
//! The `bedrock` feature alone gives bearer authentication and the Bedrock
//! wire protocol with no AWS crates. `bedrock-aws` adds the AWS credential
//! chain and SigV4 signing.

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures_core::Stream;
use futures_util::StreamExt as _;
use futures_util::stream::{iter, unfold};
#[cfg(feature = "bedrock-aws")]
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};

use crate::adapter::{
    AdapterBuildError, AdapterContext, AdapterFactory, InputTokenCount, ProviderAdapter,
    ResolvedCall,
};
#[cfg(not(feature = "bedrock-aws"))]
use crate::catalog::AuthScheme;
#[cfg(feature = "bedrock-aws")]
use crate::catalog::ProviderId;
use crate::catalog::{AdapterId, CatalogProvider, codec_ids};
use crate::codecs::bedrock::BedrockConverseCodec;
use crate::codecs::{Codec as _, StreamDecoder};
use crate::credentials::{CredentialProvider, Credentials};
#[cfg(feature = "bedrock-aws")]
use crate::transport::PreparedRequest;
#[cfg(feature = "bedrock-aws")]
use crate::transport::aws::AwsSigner;
use crate::transport::{EncodedRequest, EventResponse, HttpTransport, JsonResponse, SseEvent};
use crate::types::{Error, ErrorKind, Response, ResponseStream, StreamEvent};

pub(super) struct Factory;

impl AdapterFactory for Factory {
    fn create(
        &self,
        provider: &CatalogProvider,
        context: &AdapterContext,
    ) -> Result<Arc<dyn ProviderAdapter>, AdapterBuildError> {
        if provider.codec().as_str() != codec_ids::BEDROCK_CONVERSE {
            return Err(AdapterBuildError::UnsupportedCodec {
                provider: provider.id().clone(),
                codec:    provider.codec().clone(),
            });
        }
        // SigV4 needs the AWS crates. Reporting this as a build error keeps it
        // a client construction issue for this one provider rather than a
        // failure at the first request or a silent downgrade to bearer.
        #[cfg(not(feature = "bedrock-aws"))]
        if matches!(provider.auth(), AuthScheme::Aws { .. }) {
            return Err(AdapterBuildError::InvalidConfiguration {
                provider: provider.id().clone(),
                message:  "AWS SigV4 authentication needs the `bedrock-aws` feature, which is \
                           disabled in this build"
                    .to_owned(),
            });
        }
        Ok(Arc::new(BedrockAdapter {
            id: provider.adapter().clone(),
            codec: BedrockConverseCodec,
            transport: HttpTransport::new(context.http().clone()),
            credentials: context.credentials().clone(),
            #[cfg(feature = "bedrock-aws")]
            signer: AwsSigner::new(provider.id().clone()),
        }))
    }
}

struct BedrockAdapter {
    id:          AdapterId,
    codec:       BedrockConverseCodec,
    transport:   HttpTransport,
    credentials: Arc<dyn CredentialProvider>,
    /// One signer per adapter. It loads the AWS credential chain on the first
    /// signed request and then resolves credentials per request, so temporary
    /// credentials refresh.
    #[cfg(feature = "bedrock-aws")]
    signer:      AwsSigner,
}

#[async_trait]
impl ProviderAdapter for BedrockAdapter {
    fn id(&self) -> &AdapterId {
        &self.id
    }

    async fn complete(&self, call: &ResolvedCall) -> Result<Response, Error> {
        let encoded = self.codec.encode(call, false)?;
        let warnings = encoded.warnings.clone();
        let result = self.json(call, encoded).await?;
        let mut response = self.codec.decode_response(call.route(), result.body)?;
        response.rate_limits = result.rate_limits;
        response.warnings.extend(warnings);
        super::apply_catalog_cost(&mut response, call.route());
        Ok(response)
    }

    async fn stream(&self, call: &ResolvedCall) -> Result<ResponseStream, Error> {
        let encoded = self.codec.encode(call, true)?;
        let warnings = encoded.warnings.clone();
        let accepted = self.events(call, encoded).await?;
        let decoded = decode_stream(accepted.events, self.codec.stream_decoder(call.route()));

        let route = call.route().clone();
        let finished = decoded.map(move |event| match event {
            Ok(StreamEvent::Completed { mut response }) => {
                response.warnings.extend(warnings.clone());
                super::apply_catalog_cost(&mut response, &route);
                Ok(StreamEvent::Completed { response })
            }
            other => other,
        });
        let limits = iter(
            accepted
                .rate_limits
                .into_iter()
                .map(|rate_limits| Ok(StreamEvent::RateLimits { rate_limits })),
        );
        Ok(Box::pin(limits.chain(finished)))
    }

    async fn count_input_tokens(
        &self,
        call: &ResolvedCall,
    ) -> Result<Option<InputTokenCount>, Error> {
        let Some(encoded) = self.codec.encode_count_tokens(call) else {
            return Ok(None);
        };
        let result = self.json(call, encoded?).await?;
        let tokens = self.codec.decode_count_tokens(call.route(), result.body)?;
        Ok(Some(InputTokenCount::new(tokens, call.route().handle())))
    }
}

impl BedrockAdapter {
    /// Sends one encoded request and returns its JSON body.
    ///
    /// The credentials select the authentication arm. Both arms send the
    /// `encoded` request unchanged, so the method, URL, and body cannot differ
    /// between them.
    async fn json(
        &self,
        call: &ResolvedCall,
        encoded: EncodedRequest,
    ) -> Result<JsonResponse, Error> {
        let provider = call.route().provider();
        match self.credentials(call).await? {
            credentials @ Credentials::BedrockBearer(_) => {
                self.transport
                    .execute_json(encoded, provider, credentials)
                    .await
            }
            #[cfg(feature = "bedrock-aws")]
            Credentials::AwsDefaultChain { region } => {
                let prepared = self.sign(provider, encoded, region.as_deref()).await?;
                self.transport
                    .execute_json_prepared(prepared, provider)
                    .await
            }
            #[cfg(not(feature = "bedrock-aws"))]
            Credentials::AwsDefaultChain { .. } => Err(missing_feature(provider)),
            _ => Err(scheme_mismatch(provider)),
        }
    }

    /// Sends one encoded request and returns its event stream.
    async fn events(
        &self,
        call: &ResolvedCall,
        encoded: EncodedRequest,
    ) -> Result<EventResponse, Error> {
        let provider = call.route().provider();
        match self.credentials(call).await? {
            credentials @ Credentials::BedrockBearer(_) => {
                self.transport
                    .event_stream_events(encoded, provider, credentials)
                    .await
            }
            #[cfg(feature = "bedrock-aws")]
            Credentials::AwsDefaultChain { region } => {
                let prepared = self.sign(provider, encoded, region.as_deref()).await?;
                self.transport
                    .event_stream_events_prepared(prepared, provider)
                    .await
            }
            #[cfg(not(feature = "bedrock-aws"))]
            Credentials::AwsDefaultChain { .. } => Err(missing_feature(provider)),
            _ => Err(scheme_mismatch(provider)),
        }
    }

    async fn credentials(&self, call: &ResolvedCall) -> Result<Credentials, Error> {
        let provider = call.route().provider();
        self.credentials
            .credentials(provider)
            .await
            .map_err(|source| {
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

    /// Prepares one request and signs it with AWS SigV4.
    ///
    /// The body is serialized once and the header map is final before signing,
    /// so the signature covers exactly the bytes and headers that reach the
    /// wire.
    #[cfg(feature = "bedrock-aws")]
    async fn sign(
        &self,
        provider: &CatalogProvider,
        encoded: EncodedRequest,
        credential_region: Option<&str>,
    ) -> Result<PreparedRequest, Error> {
        let mut prepared = prepare(provider, encoded)?;
        let region = self
            .signer
            .resolve_region(credential_region, provider.auth(), provider.base_url())
            .await?;
        self.signer
            .sign(
                &region,
                &prepared.method,
                &prepared.url,
                &mut prepared.headers,
                &prepared.body,
            )
            .await?;
        Ok(prepared)
    }
}

/// The transport's undecoded event stream.
type TransportEvents = Pin<Box<dyn Stream<Item = Result<SseEvent, Error>> + Send>>;

/// Drives one stream decoder over the transport events.
///
/// The decoder's `finish` runs once when the byte stream ends without error.
/// A decode failure ends the stream, so no `Completed` event follows one.
fn decode_stream(
    events: TransportEvents,
    decoder: Box<dyn StreamDecoder>,
) -> impl Stream<Item = Result<StreamEvent, Error>> + Send {
    struct State {
        events:  TransportEvents,
        decoder: Box<dyn StreamDecoder>,
        ended:   bool,
    }

    let state = State {
        events,
        decoder,
        ended: false,
    };
    unfold(state, |mut state| async move {
        if state.ended {
            return None;
        }
        let decoded = match state.events.next().await {
            Some(Ok(event)) => state.decoder.decode(event),
            Some(Err(error)) => {
                state.ended = true;
                return Some((vec![Err(error)], state));
            }
            None => {
                state.ended = true;
                state.decoder.finish()
            }
        };
        let items = match decoded {
            Ok(events) => events.into_iter().map(Ok).collect(),
            Err(error) => {
                state.ended = true;
                vec![Err(error)]
            }
        };
        Some((items, state))
    })
    .flat_map(iter)
}

#[cfg(not(feature = "bedrock-aws"))]
fn missing_feature(provider: &CatalogProvider) -> Error {
    Error::new(
        ErrorKind::Configuration,
        format!(
            "provider {} resolved AWS default-chain credentials, which need the `bedrock-aws` \
             feature",
            provider.id()
        ),
    )
    .with_provider(provider.id().clone())
}

fn scheme_mismatch(provider: &CatalogProvider) -> Error {
    Error::new(
        ErrorKind::Authentication,
        format!(
            "credentials for provider {} are not Bedrock credentials",
            provider.id()
        ),
    )
    .with_provider(provider.id().clone())
}

/// Builds the final headers and bytes for one encoded request.
///
/// SigV4 signs the exact method, URL, headers, and bytes that reach the wire,
/// so a signed request is assembled here rather than by the transport. Header
/// precedence matches [`HttpTransport`]: the JSON content type first, then
/// codec headers, then the provider's catalog default headers. Authentication
/// is applied last, by the caller, so it wins every collision.
#[cfg(feature = "bedrock-aws")]
fn prepare(provider: &CatalogProvider, encoded: EncodedRequest) -> Result<PreparedRequest, Error> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    for (name, value) in &encoded.headers {
        insert_header(&mut headers, provider.id(), name, value)?;
    }
    for (name, value) in provider.default_headers() {
        insert_header(&mut headers, provider.id(), name, value)?;
    }
    let body = serde_json::to_vec(&encoded.body).map_err(|source| {
        Error::new(
            ErrorKind::InvalidRequest,
            format!(
                "the request for provider {} could not be serialized",
                provider.id()
            ),
        )
        .with_provider(provider.id().clone())
        .with_source(source)
    })?;
    Ok(PreparedRequest {
        method: encoded.method,
        url: encoded.url,
        headers,
        body,
        timeout: encoded.timeout,
    })
}

#[cfg(feature = "bedrock-aws")]
fn insert_header(
    headers: &mut HeaderMap,
    provider: &ProviderId,
    name: &str,
    value: &str,
) -> Result<(), Error> {
    let name = HeaderName::from_bytes(name.as_bytes()).map_err(|source| {
        Error::new(
            ErrorKind::Configuration,
            format!("provider {provider} has an invalid HTTP header name"),
        )
        .with_provider(provider.clone())
        .with_source(source)
    })?;
    let value = HeaderValue::from_str(value).map_err(|source| {
        Error::new(
            ErrorKind::Configuration,
            format!("provider {provider} has an invalid HTTP header value"),
        )
        .with_provider(provider.clone())
        .with_source(source)
    })?;
    headers.insert(name, value);
    Ok(())
}

#[cfg(all(test, feature = "builtin-catalog"))]
mod tests {
    use std::error::Error as StdError;
    use std::sync::Arc;

    use reqwest::Client;
    #[cfg(feature = "bedrock-aws")]
    use reqwest::header::CONTENT_TYPE;
    use serde_json::json;

    use super::Factory;
    #[cfg(feature = "bedrock-aws")]
    use super::{BedrockConverseCodec, prepare};
    use crate::adapter::{AdapterBuildError, AdapterContext, AdapterFactory as _, ResolvedCall};
    use crate::catalog::{Catalog, CatalogProvider, ProviderId};
    #[cfg(feature = "bedrock-aws")]
    use crate::codecs::Codec as _;
    use crate::credentials::{Credentials, SecretValue, StaticCredentials};
    use crate::middleware::CallContext;
    use crate::resolver::{AvailableProviders, CatalogResolver, ModelResolver as _};
    use crate::types::Request;

    /// A Bedrock provider whose endpoint the test controls, using bearer
    /// authentication so no AWS crates are needed.
    fn catalog(base_url: &str) -> Result<Catalog, Box<dyn StdError>> {
        let source = format!(
            r#"
            schema_version = 1

            [providers.bedrock]
            display_name = "Amazon Bedrock"
            adapter = "bedrock"
            codec = "bedrock-converse"
            base_url = "{base_url}"
            default_model = "sonnet"
            auth = {{ type = "bedrock_bearer" }}

            [providers.bedrock.models.sonnet]
            display_name = "Sonnet"
            api_model = "anthropic.claude-sonnet-4-6"
            capabilities = {{ text = true, tools = true }}
            "#
        );
        Ok(Catalog::builder().toml_layer("test", &source)?.build()?)
    }

    fn call(catalog: &Catalog) -> Result<ResolvedCall, Box<dyn StdError>> {
        let request = Request::builder()
            .model("bedrock/sonnet")
            .user("Hello")
            .build()?;
        let available = AvailableProviders::all(catalog);
        let route = CatalogResolver.resolve(&request, catalog, &available)?;
        Ok(ResolvedCall::new(request, route, CallContext::new()))
    }

    fn provider(catalog: &Catalog) -> Result<&CatalogProvider, Box<dyn StdError>> {
        catalog
            .provider_by_id(&ProviderId::new("bedrock"))
            .ok_or_else(|| "the test catalog defines a bedrock provider".into())
    }

    fn context() -> AdapterContext {
        let credentials = StaticCredentials::new().with(
            ProviderId::new("bedrock"),
            Credentials::BedrockBearer(SecretValue::new("token")),
        );
        AdapterContext::new(Client::new(), Arc::new(credentials))
    }

    #[test]
    fn rejects_a_provider_that_is_not_bedrock_converse() -> Result<(), Box<dyn StdError>> {
        let source = r#"
            schema_version = 1

            [providers.bedrock]
            display_name = "Amazon Bedrock"
            adapter = "bedrock"
            codec = "openai-chat"
            base_url = "https://bedrock-runtime.us-east-1.amazonaws.com"
            default_model = "sonnet"
            auth = { type = "bedrock_bearer" }

            [providers.bedrock.models.sonnet]
            display_name = "Sonnet"
            api_model = "anthropic.claude-sonnet-4-6"
            capabilities = { text = true }
        "#;
        let catalog = Catalog::builder().toml_layer("test", source)?.build()?;

        let error = Factory
            .create(provider(&catalog)?, &context())
            .err()
            .ok_or("a non-Bedrock codec is rejected")?;

        assert!(
            matches!(error, AdapterBuildError::UnsupportedCodec { .. }),
            "unexpected error: {error}"
        );
        Ok(())
    }

    /// Without `bedrock-aws` the factory reports a construction issue for a
    /// provider that asks for SigV4, rather than panicking or falling back to
    /// bearer authentication.
    #[cfg(not(feature = "bedrock-aws"))]
    #[test]
    fn reports_sigv4_without_the_aws_feature() -> Result<(), Box<dyn StdError>> {
        let catalog = Catalog::builder().with_builtin().build()?;

        let error = Factory
            .create(provider(&catalog)?, &context())
            .err()
            .ok_or("AWS authentication is rejected without the feature")?;

        let AdapterBuildError::InvalidConfiguration { message, .. } = &error else {
            return Err(format!("unexpected error: {error}").into());
        };
        assert!(
            message.contains("bedrock-aws"),
            "the message names the missing feature: {message}"
        );
        Ok(())
    }

    /// Both authentication paths dispatch the one `EncodedRequest` the codec
    /// produced, so they cannot disagree about the method, the URL, or the
    /// body. The signed path is checked directly: preparing a request for
    /// signing changes none of the three, and adds no authentication of its
    /// own. The bearer path is checked against that same prepared request over
    /// a mock endpoint, which only answers a request whose method, path, and
    /// body match.
    #[cfg(feature = "bedrock-aws")]
    #[tokio::test]
    async fn bearer_and_signed_requests_carry_the_same_url_and_body()
    -> Result<(), Box<dyn StdError>> {
        let server = httpmock::MockServer::start_async().await;
        let catalog = catalog(&server.base_url())?;
        let call = call(&catalog)?;
        let encoded = BedrockConverseCodec.encode(&call, false)?;
        let method = encoded.method.clone();
        let url = encoded.url.clone();
        let body = encoded.body.clone();

        let prepared = prepare(provider(&catalog)?, encoded)?;

        assert_eq!(prepared.method, method);
        assert_eq!(prepared.url, url);
        assert_eq!(prepared.body, serde_json::to_vec(&body)?);
        assert_eq!(
            prepared
                .headers
                .get(CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some("application/json")
        );
        assert!(
            prepared.headers.get("authorization").is_none(),
            "authentication is applied after preparation, never during it"
        );

        let path = prepared
            .url
            .strip_prefix(&server.base_url())
            .ok_or("the encoded URL starts at the provider base URL")?
            .to_owned();
        let mock = server
            .mock_async(|when, then| {
                when.method(prepared.method.as_str())
                    .path(path)
                    .json_body(body);
                then.status(200)
                    .header("content-type", "application/json")
                    .json_body(json!({
                        "output": { "message": { "role": "assistant", "content": [{ "text": "Hi" }] } },
                        "stopReason": "end_turn",
                        "usage": { "inputTokens": 1, "outputTokens": 1 },
                    }));
            })
            .await;
        let adapter = Factory.create(provider(&catalog)?, &context())?;

        adapter.complete(&call).await?;

        mock.assert_async().await;
        Ok(())
    }

    #[tokio::test]
    async fn counts_input_tokens_for_the_canonical_model() -> Result<(), Box<dyn StdError>> {
        let server = httpmock::MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method("POST")
                    .path("/model/anthropic.claude-sonnet-4-6/count-tokens");
                then.status(200)
                    .header("content-type", "application/json")
                    .json_body(json!({ "inputTokens": 42 }));
            })
            .await;
        let catalog = catalog(&server.base_url())?;
        let call = call(&catalog)?;
        let adapter = Factory.create(provider(&catalog)?, &context())?;

        let count = adapter
            .count_input_tokens(&call)
            .await?
            .ok_or("Bedrock supports native token counting")?;

        mock.assert_async().await;
        assert_eq!(count.tokens(), 42);
        assert_eq!(count.model(), &call.route().handle());
        Ok(())
    }
}
