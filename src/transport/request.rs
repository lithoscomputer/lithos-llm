//! The request the codecs produce and the request the transport sends.

use std::net::IpAddr;
use std::time::Duration;

use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use reqwest::{Method, Url};
use serde_json::Value;

use super::headers::merge_headers;
use super::sse::SseDispatch;
use crate::catalog::CatalogProvider;
use crate::credentials::{Credentials, HttpAuthentication};
use crate::types::{Error, ErrorKind, Speed, Warning};

/// A request whose headers and body bytes are final: the one shape the
/// transport sends.
///
/// [`EncodedRequest::prepare`] builds it: the codec headers, the provider's
/// catalog default headers, and the credential headers are merged in that
/// precedence, and the body is serialized once. AWS SigV4 signs the exact
/// method, URL, headers, and bytes that reach the wire, so a signed request
/// is prepared, then signed, then sent unchanged.
pub(crate) struct PreparedRequest {
    pub method:  Method,
    pub url:     String,
    pub headers: HeaderMap,
    pub body:    Vec<u8>,
    pub timeout: Option<Duration>,
    /// How the transport frames this request's response stream.
    ///
    /// Only the streaming path reads this; a JSON response has no frames.
    pub framing: StreamFraming,
}

pub(crate) struct EncodedRequest {
    pub method:        Method,
    pub url:           String,
    pub headers:       Vec<(String, String)>,
    pub body:          Value,
    pub timeout:       Option<Duration>,
    /// Non-fatal notes produced while encoding, such as a portable request
    /// control this protocol cannot express.
    ///
    /// Only encoding knows the request, so a codec records these here and the
    /// adapter copies them onto the decoded response.
    pub warnings:      Vec<Warning>,
    /// The speed the codec put on the wire, if any.
    ///
    /// Cost estimation prices the call at this speed. A codec whose protocol
    /// cannot express the requested speed leaves this empty, so a request the
    /// provider serves at standard speed is billed at standard rates.
    pub applied_speed: Option<Speed>,
    /// How the transport frames this request's response stream.
    ///
    /// Only the streaming path reads this; a JSON response has no frames.
    pub framing:       StreamFraming,
}

impl EncodedRequest {
    /// Creates a request with no headers, no timeout, and no warnings.
    pub(crate) fn new(method: Method, url: String, body: Value) -> Self {
        Self {
            method,
            url,
            body,
            headers: Vec::new(),
            timeout: None,
            warnings: Vec::new(),
            applied_speed: None,
            framing: StreamFraming::Sse(SseDispatch::BlankLine),
        }
    }

    /// Records the speed the codec encoded into the request.
    #[must_use]
    pub(crate) fn with_applied_speed(mut self, speed: Option<Speed>) -> Self {
        self.applied_speed = speed;
        self
    }

    #[must_use]
    pub(crate) fn with_headers(mut self, headers: Vec<(String, String)>) -> Self {
        self.headers = headers;
        self
    }

    /// Records that a portable request control could not be expressed.
    #[must_use]
    pub(crate) fn unsupported_control(mut self, control: &str) -> Self {
        self.warnings.push(Warning {
            code:    "unsupported_control".to_owned(),
            message: format!("this provider protocol does not support {control}"),
        });
        self
    }

    /// Sends an SSE event at every complete `data:` line.
    ///
    /// Lenient skins and proxies for this codec's dialect separate events with
    /// single line breaks rather than the blank line the SSE specification
    /// requires, and every event is one `data:` line of JSON.
    #[must_use]
    pub(crate) fn with_data_line_framing(mut self) -> Self {
        self.framing = StreamFraming::Sse(SseDispatch::EachDataLine);
        self
    }

    /// Frames the response stream as AWS `vnd.amazon.eventstream` frames.
    #[cfg(feature = "bedrock")]
    #[must_use]
    pub(crate) fn with_aws_event_stream_framing(mut self) -> Self {
        self.framing = StreamFraming::AwsEventStream;
        self
    }

    /// Builds the final headers and bytes for this request.
    ///
    /// Header precedence is codec headers, then the provider's catalog
    /// default headers, then the credential headers, so credentials win every
    /// collision. The JSON content type goes in first, so any of those may
    /// replace it.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Configuration`] for a header name or value HTTP cannot
    /// carry, or for credentials bound for an `http://` URL whose host is not
    /// loopback; [`ErrorKind::Authentication`] when the credentials do not
    /// match the provider's scheme; and [`ErrorKind::InvalidRequest`] when the
    /// body cannot be serialized.
    pub(crate) fn prepare(
        self,
        provider: &CatalogProvider,
        credentials: &Credentials,
    ) -> Result<PreparedRequest, Error> {
        self.prepare_with(provider, Some(credentials))
    }

    /// Builds the final headers and bytes with no credential headers.
    ///
    /// For a request the caller authenticates after preparation: AWS SigV4
    /// signs the prepared headers and bytes and adds its own.
    #[cfg(feature = "bedrock-aws")]
    pub(crate) fn prepare_unauthenticated(
        self,
        provider: &CatalogProvider,
    ) -> Result<PreparedRequest, Error> {
        self.prepare_with(provider, None)
    }

    fn prepare_with(
        self,
        provider: &CatalogProvider,
        credentials: Option<&Credentials>,
    ) -> Result<PreparedRequest, Error> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        merge_headers(
            &mut headers,
            provider.id(),
            provider.auth(),
            &self.headers,
            provider.default_headers(),
            credentials,
        )?;
        if credentials.is_some_and(sends_secrets) && !is_protected(&self.url) {
            return Err(Error::new(
                ErrorKind::Configuration,
                format!(
                    "provider {} would send credentials over unencrypted HTTP; use an https \
                     base URL, or a loopback host for a local server",
                    provider.id()
                ),
            )
            .with_provider(provider.id().clone()));
        }
        let body = serde_json::to_vec(&self.body).map_err(|source| {
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
            method: self.method,
            url: self.url,
            headers,
            body,
            timeout: self.timeout,
            framing: self.framing,
        })
    }
}

/// Whether `credentials` put any secret header on the request.
fn sends_secrets(credentials: &Credentials) -> bool {
    match credentials {
        Credentials::Http(http) => {
            !matches!(http.auth, HttpAuthentication::None) || !http.extra_headers.is_empty()
        }
        Credentials::BedrockBearer(_) => true,
        // The SigV4 signer authenticates these after preparation.
        Credentials::AwsDefaultChain { .. } => false,
    }
}

/// Whether a request to `url` keeps its headers off the open network: it is
/// encrypted, or it never leaves this machine.
///
/// Plain HTTP is allowed only to `localhost`, a `.localhost` name, or a
/// loopback address, which is where local model servers and test doubles
/// listen. A URL that does not parse is left to the HTTP client to reject.
fn is_protected(url: &str) -> bool {
    let Ok(url) = Url::parse(url) else {
        return true;
    };
    if url.scheme() != "http" {
        return true;
    }
    let Some(host) = url.host_str() else {
        return false;
    };
    let host = host.trim_start_matches('[').trim_end_matches(']');
    if let Ok(address) = host.parse::<IpAddr>() {
        return address.is_loopback();
    }
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    host == "localhost" || host.ends_with(".localhost")
}

/// How the transport splits a response byte stream into events.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StreamFraming {
    /// Server-sent events, sent as the dispatch rule directs.
    Sse(SseDispatch),
    /// AWS `vnd.amazon.eventstream` binary frames, each carrying one event.
    #[cfg(feature = "bedrock")]
    AwsEventStream,
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use reqwest::Method;
    use serde_json::json;

    use super::EncodedRequest;
    use crate::catalog::{Catalog, CatalogProvider, ProviderId};
    use crate::credentials::{CredentialHeader, Credentials, SecretValue};
    use crate::types::ErrorKind;

    /// A one-model provider whose authentication scheme is `auth`.
    fn provider(auth: &str) -> Result<CatalogProvider, Box<dyn StdError>> {
        let catalog = Catalog::builder()
            .overlay_toml(&format!(
                r#"
                schema_version = 1

                [providers.alpha]
                display_name = "Alpha"
                codecs = ["openai-chat"]
                base_url = "https://api.example.com"
                default_model = "one"
                auth = {{ type = "{auth}" }}

                [providers.alpha.models.one]
                display_name = "One"
                api_model = "one"
                capabilities = {{ text = true }}
                "#
            ))?
            .build()?;
        Ok(catalog
            .provider_by_id(&ProviderId::new("alpha"))
            .ok_or("the test provider")?
            .clone())
    }

    fn bearer() -> Credentials {
        Credentials::bearer(SecretValue::new("sk-test"))
    }

    /// Prepares a request to `url` for a provider with the `auth` scheme, and
    /// returns the kind of error preparation failed with, if any.
    fn prepare_for(
        auth: &str,
        url: &str,
        credentials: &Credentials,
    ) -> Result<Result<(), ErrorKind>, Box<dyn StdError>> {
        Ok(EncodedRequest::new(Method::POST, url.to_owned(), json!({}))
            .prepare(&provider(auth)?, credentials)
            .map(drop)
            .map_err(|error| error.kind()))
    }

    #[test]
    fn credentials_never_travel_over_plain_http_to_another_host() -> Result<(), Box<dyn StdError>> {
        for url in [
            "http://api.example.com/v1/chat/completions",
            "http://10.0.0.5:8080/v1",
            "http://localhost.example.com/v1",
        ] {
            assert_eq!(
                prepare_for("bearer", url, &bearer())?,
                Err(ErrorKind::Configuration),
                "{url}"
            );
        }
        let extra_only = Credentials::headers([CredentialHeader::new(
            "x-proxy-token",
            SecretValue::new("proxy-secret"),
        )]);
        assert_eq!(
            prepare_for("bearer", "http://api.example.com/v1", &extra_only)?,
            Err(ErrorKind::Authentication),
            "a bearer scheme still rejects mismatched credentials first"
        );
        Ok(())
    }

    #[test]
    fn credentials_may_use_https_or_a_loopback_host() -> Result<(), Box<dyn StdError>> {
        for url in [
            "https://api.example.com/v1",
            "http://localhost:11434/v1",
            "http://LOCALHOST./v1",
            "http://models.localhost/v1",
            "http://127.0.0.1:8080/v1",
            "http://127.1.2.3/v1",
            "http://[::1]:8080/v1",
        ] {
            assert_eq!(prepare_for("bearer", url, &bearer())?, Ok(()), "{url}");
        }
        Ok(())
    }

    #[test]
    fn a_call_without_credentials_may_use_plain_http_anywhere() -> Result<(), Box<dyn StdError>> {
        assert_eq!(
            prepare_for("none", "http://10.0.0.5:8080/v1", &Credentials::none())?,
            Ok(())
        );
        Ok(())
    }
}
