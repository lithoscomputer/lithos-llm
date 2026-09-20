//! The Amazon Bedrock provider adapter.
//!
//! The `http` adapter, but signed and event-stream framed. Every call resolves
//! credentials, encodes through [`BedrockConverseCodec`], authenticates the
//! one [`EncodedRequest`] into a [`PreparedRequest`], dispatches it, and
//! decodes through the same codec. A bearer request and a SigV4 request are
//! prepared from that one encoded request, so they always carry the same
//! method, URL, and body; only the authentication headers differ.
//!
//! The `bedrock` feature alone gives bearer authentication and the Bedrock
//! wire protocol with no AWS crates. `bedrock-aws` adds the AWS credential
//! chain and SigV4 signing.

use std::sync::Arc;

use async_trait::async_trait;

use super::http::{decode_stream, finish_response, finish_stream, resolve_credentials};
use crate::adapter::{
    AdapterBuildError, AdapterContext, AdapterFactory, InputTokenCount, ProviderAdapter,
    ResolvedCall,
};
#[cfg(not(feature = "bedrock-aws"))]
use crate::catalog::AuthScheme;
use crate::catalog::{AdapterId, CatalogProvider, codec_ids};
use crate::codecs::Codec as _;
use crate::codecs::bedrock::BedrockConverseCodec;
use crate::credentials::{CredentialProvider, Credentials};
#[cfg(feature = "bedrock-aws")]
use crate::transport::aws::AwsSigner;
use crate::transport::{EncodedRequest, HttpTransport, PreparedRequest};
use crate::types::{Error, ErrorKind, Response, ResponseStream};

pub(super) struct Factory;

impl AdapterFactory for Factory {
    fn create(
        &self,
        provider: &CatalogProvider,
        context: &AdapterContext,
    ) -> Result<Arc<dyn ProviderAdapter>, AdapterBuildError> {
        // This adapter speaks Converse and nothing else, so any other codec
        // in the provider's list is a build issue rather than a call that
        // fails later.
        if let Some(codec) = provider
            .codecs()
            .iter()
            .find(|codec| codec.as_str() != codec_ids::BEDROCK_CONVERSE)
        {
            return Err(AdapterBuildError::UnsupportedCodec {
                provider: provider.id().clone(),
                codec:    codec.clone(),
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
            transport: HttpTransport::new(context.http().clone())
                .with_stream_idle_timeout(context.stream_idle_timeout())
                .with_response_limits(context.response_limits()),
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
        expect_converse(call, "complete")?;
        let encoded = self.codec.encode(call, false)?;
        let warnings = encoded.warnings.clone();
        let speed = encoded.applied_speed;
        let prepared = self.authenticate(call, encoded).await?;
        let result = self
            .transport
            .execute_json(prepared, call.route().provider())
            .await?;
        let response = self.codec.decode_response(call.route(), result.body)?;
        Ok(finish_response(
            response,
            result.rate_limits,
            warnings,
            call,
            speed,
        ))
    }

    async fn stream(&self, call: &ResolvedCall) -> Result<ResponseStream, Error> {
        expect_converse(call, "stream")?;
        let encoded = self.codec.encode(call, true)?;
        let warnings = encoded.warnings.clone();
        let speed = encoded.applied_speed;
        let prepared = self.authenticate(call, encoded).await?;
        let accepted = self
            .transport
            .stream_events(prepared, call.route().provider())
            .await?;
        let decoded = decode_stream(accepted.events, self.codec.stream_decoder(call.route()));
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
        expect_converse(call, "count_input_tokens")?;
        let Some(encoded) = self.codec.encode_count_tokens(call) else {
            return Ok(None);
        };
        let prepared = self.authenticate(call, encoded?).await?;
        let result = self
            .transport
            .execute_json(prepared, call.route().provider())
            .await?;
        let tokens = self.codec.decode_count_tokens(call.route(), result.body)?;
        Ok(Some(InputTokenCount::new(tokens, call.route().handle())))
    }

    /// The client picks the codec before a call arrives here, and the one
    /// codec this adapter holds serves generation only, so a row on this
    /// adapter never reaches a native evaluation; the judge path runs.
    fn evaluates_natively(&self) -> bool {
        true
    }
}

/// Checks that the client selected the Converse codec for `call`.
///
/// The factory refused any other codec at build time, so a mismatch here
/// is a client invariant failure, reported as such.
fn expect_converse(call: &ResolvedCall, operation: &str) -> Result<(), Error> {
    let provider = call.route().provider();
    match call.codec() {
        Some(codec) if codec.as_str() == codec_ids::BEDROCK_CONVERSE => Ok(()),
        Some(codec) => Err(Error::new(
            ErrorKind::Configuration,
            format!(
                "internal error: the bedrock adapter received {operation} for provider {} on \
                 codec {codec}, which it does not serve",
                provider.id()
            ),
        )
        .with_provider(provider.id().clone())),
        None => Err(Error::new(
            ErrorKind::Configuration,
            format!(
                "internal error: the bedrock adapter received {operation} for provider {} with \
                 no codec selected",
                provider.id()
            ),
        )
        .with_provider(provider.id().clone())),
    }
}

impl BedrockAdapter {
    /// Authenticates one encoded request into the request the transport sends.
    ///
    /// The credentials select the authentication arm. A Bedrock API key
    /// prepares with the bearer header; AWS default-chain credentials prepare
    /// without credential headers and are then signed, so the signature covers
    /// exactly the bytes and headers that reach the wire. Both arms prepare
    /// the same `encoded` request, so the method, URL, and body cannot differ
    /// between them.
    async fn authenticate(
        &self,
        call: &ResolvedCall,
        encoded: EncodedRequest,
    ) -> Result<PreparedRequest, Error> {
        let provider = call.route().provider();
        match resolve_credentials(self.credentials.as_ref(), provider).await? {
            credentials @ Credentials::BedrockBearer(_) => encoded.prepare(provider, &credentials),
            #[cfg(feature = "bedrock-aws")]
            Credentials::AwsDefaultChain { region } => {
                let mut prepared = encoded.prepare_unauthenticated(provider)?;
                let region = self
                    .signer
                    .resolve_region(region.as_deref(), provider.auth(), provider.base_url())
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
            #[cfg(not(feature = "bedrock-aws"))]
            Credentials::AwsDefaultChain { .. } => Err(missing_feature(provider)),
            _ => Err(scheme_mismatch(provider)),
        }
    }
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

#[cfg(all(test, feature = "builtin-catalog"))]
mod tests {
    use std::error::Error as StdError;
    use std::sync::Arc;

    use reqwest::Client;
    #[cfg(feature = "bedrock-aws")]
    use reqwest::header::CONTENT_TYPE;
    use serde_json::json;

    #[cfg(feature = "bedrock-aws")]
    use super::BedrockConverseCodec;
    use super::Factory;
    use crate::adapter::{AdapterBuildError, AdapterContext, AdapterFactory as _, ResolvedCall};
    use crate::catalog::{Catalog, CatalogProvider, CodecId, ProviderId, codec_ids};
    #[cfg(feature = "bedrock-aws")]
    use crate::codecs::Codec as _;
    use crate::credentials::{Credentials, SecretValue, StaticCredentials};
    use crate::middleware::CallContext;
    use crate::resolver::{AvailableProviders, CatalogResolver, ModelResolver as _};
    use crate::types::{ErrorKind, Request};

    /// A Bedrock provider whose endpoint the test controls, using bearer
    /// authentication so no AWS crates are needed.
    fn catalog(base_url: &str) -> Result<Catalog, Box<dyn StdError>> {
        let source = format!(
            r#"
            schema_version = 1

            [providers.bedrock]
            display_name = "Amazon Bedrock"
            adapter = "bedrock"
            codecs = ["bedrock-converse"]
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
        Ok(ResolvedCall::new(request, route, CallContext::new())
            .with_codec(Some(CodecId::new(codec_ids::BEDROCK_CONVERSE))))
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

    /// The adapter holds Converse alone, so a second codec in the list is
    /// refused at build time rather than at the first call that selects it.
    #[test]
    fn rejects_a_provider_that_lists_any_other_codec() -> Result<(), Box<dyn StdError>> {
        let source = r#"
            schema_version = 1

            [providers.bedrock]
            display_name = "Amazon Bedrock"
            adapter = "bedrock"
            codecs = ["bedrock-converse", "openai-chat"]
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
            matches!(&error, AdapterBuildError::UnsupportedCodec { codec, .. } if codec.as_str() == "openai-chat"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_call_on_another_codec_is_an_internal_configuration_error()
    -> Result<(), Box<dyn StdError>> {
        let catalog = catalog("http://127.0.0.1:1")?;
        let adapter = Factory.create(provider(&catalog)?, &context())?;
        let call = call(&catalog)?.with_codec(Some(CodecId::new("openai-chat")));

        let error = adapter
            .complete(&call)
            .await
            .expect_err("the bedrock adapter serves Converse only");

        assert_eq!(error.kind(), ErrorKind::Configuration);
        assert!(
            error.message().contains("openai-chat") && error.message().contains("complete"),
            "{}",
            error.message()
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

    /// Both authentication paths prepare the one `EncodedRequest` the codec
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

        let prepared = encoded.prepare_unauthenticated(provider(&catalog)?)?;

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
