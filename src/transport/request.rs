//! The request the codecs produce and the request the transport sends.

use std::time::Duration;

use reqwest::Method;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use serde_json::Value;

use super::headers::merge_headers;
use super::sse::SseDispatch;
use crate::catalog::CatalogProvider;
use crate::credentials::Credentials;
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
    /// carry, [`ErrorKind::Authentication`] when the credentials do not match
    /// the provider's scheme, and [`ErrorKind::InvalidRequest`] when the body
    /// cannot be serialized.
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

/// How the transport splits a response byte stream into events.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StreamFraming {
    /// Server-sent events, sent as the dispatch rule directs.
    Sse(SseDispatch),
    /// AWS `vnd.amazon.eventstream` binary frames, each carrying one event.
    #[cfg(feature = "bedrock")]
    AwsEventStream,
}
