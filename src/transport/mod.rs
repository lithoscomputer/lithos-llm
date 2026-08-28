#[cfg(feature = "bedrock-aws")]
pub(crate) mod aws;
pub(crate) mod classify;
#[cfg(feature = "bedrock")]
pub(crate) mod event_stream;

use std::collections::BTreeMap;
use std::future::ready;
use std::pin::Pin;
#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "openai-compatible"
))]
use std::str::from_utf8;
use std::time::Duration;

use futures_core::Stream;
use futures_util::StreamExt as _;
use futures_util::stream::iter;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use reqwest::{Client, Method, Response as HttpResponse};
use serde_json::Value;

use crate::catalog::{AuthScheme, CatalogProvider, ProviderId};
use crate::credentials::{CredentialHeader, Credentials, HttpAuthentication, SecretValue};
use crate::types::{Error, ErrorKind, RateLimits, RetryClassification, Warning};

pub(crate) struct EncodedRequest {
    pub method:   Method,
    pub url:      String,
    pub headers:  Vec<(String, String)>,
    pub body:     Value,
    pub timeout:  Option<Duration>,
    /// Non-fatal notes produced while encoding, such as a portable request
    /// control this protocol cannot express.
    ///
    /// Only encoding knows the request, so a codec records these here and the
    /// adapter copies them onto the decoded response.
    pub warnings: Vec<Warning>,
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
        }
    }

    #[must_use]
    pub(crate) fn with_headers(mut self, headers: Vec<(String, String)>) -> Self {
        self.headers = headers;
        self
    }

    #[must_use]
    pub(crate) fn with_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.timeout = timeout;
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
}

#[derive(Clone, Debug)]
pub(crate) struct SseEvent {
    #[cfg_attr(
        not(any(feature = "openai", feature = "anthropic", test)),
        allow(dead_code, reason = "some provider streams do not use SSE event names")
    )]
    pub event: Option<String>,
    pub data:  String,
}

#[derive(Clone)]
pub(crate) struct HttpTransport {
    client: Client,
}

type EventStream = Pin<Box<dyn Stream<Item = Result<SseEvent, Error>> + Send>>;

pub(crate) struct JsonResponse {
    pub body:        Value,
    pub rate_limits: Option<RateLimits>,
}

pub(crate) struct EventResponse {
    pub events:      EventStream,
    pub rate_limits: Option<RateLimits>,
}

impl HttpTransport {
    pub(crate) fn new(client: Client) -> Self {
        Self { client }
    }

    pub(crate) async fn execute_json(
        &self,
        request: EncodedRequest,
        provider: &CatalogProvider,
        credentials: Credentials,
    ) -> Result<JsonResponse, Error> {
        let response = self.send(request, provider, credentials).await?;
        let rate_limits = rate_limits(response.headers());
        let body = response.json().await.map_err(|source| {
            Error::new(
                ErrorKind::Provider,
                format!("provider {} returned invalid JSON", provider.id()),
            )
            .with_provider(provider.id().clone())
            .with_source(source)
        })?;
        Ok(JsonResponse { body, rate_limits })
    }

    #[cfg(any(
        feature = "openai",
        feature = "anthropic",
        feature = "gemini",
        feature = "openai-compatible"
    ))]
    pub(crate) async fn sse_events(
        &self,
        request: EncodedRequest,
        provider: &CatalogProvider,
        credentials: Credentials,
    ) -> Result<EventResponse, Error> {
        let response = self.send(request, provider, credentials).await?;
        let rate_limits = rate_limits(response.headers());
        let provider_id = provider.id().clone();
        let chunks = response.bytes_stream().map(move |result| {
            result.map_err(|source| {
                Error::new(
                    ErrorKind::Network,
                    "reading the provider response stream failed",
                )
                .with_provider(provider_id.clone())
                .with_retry(RetryClassification::Safe)
                .with_source(source)
            })
        });
        let frames = chunks
            .scan(Vec::new(), |buffer, chunk| {
                let parsed = match chunk {
                    Ok(chunk) => {
                        buffer.extend_from_slice(&chunk);
                        extract_frames(buffer)
                    }
                    Err(error) => vec![Err(error)],
                };
                ready(Some(parsed))
            })
            .flat_map(iter);
        Ok(EventResponse {
            events: Box::pin(frames),
            rate_limits,
        })
    }

    #[cfg(feature = "bedrock")]
    pub(crate) async fn event_stream_events(
        &self,
        request: EncodedRequest,
        provider: &CatalogProvider,
        credentials: Credentials,
    ) -> Result<EventResponse, Error> {
        let response = self.send(request, provider, credentials).await?;
        let rate_limits = rate_limits(response.headers());
        let provider_id = provider.id().clone();
        let read_provider = provider_id.clone();
        let chunks = response.bytes_stream().map(move |result| {
            result.map_err(|source| {
                Error::new(
                    ErrorKind::Network,
                    "reading the Bedrock response stream failed",
                )
                .with_provider(read_provider.clone())
                .with_retry(RetryClassification::Safe)
                .with_source(source)
            })
        });
        let frames = chunks
            .scan(Vec::new(), move |buffer, chunk| {
                let parsed = match chunk {
                    Ok(chunk) => {
                        buffer.extend_from_slice(&chunk);
                        event_stream_events_from(buffer, &provider_id)
                    }
                    Err(error) => vec![Err(error)],
                };
                ready(Some(parsed))
            })
            .flat_map(iter);
        Ok(EventResponse {
            events: Box::pin(frames),
            rate_limits,
        })
    }

    async fn send(
        &self,
        request: EncodedRequest,
        provider: &CatalogProvider,
        credentials: Credentials,
    ) -> Result<HttpResponse, Error> {
        let headers = merge_headers(
            provider.id(),
            provider.auth(),
            &request.headers,
            provider.default_headers(),
            &credentials,
        )?;
        let mut builder = self
            .client
            .request(request.method, &request.url)
            .json(&request.body)
            .headers(headers);
        if let Some(timeout) = request.timeout {
            builder = builder.timeout(timeout);
        }

        let response = builder.send().await.map_err(|source| {
            let kind = if source.is_timeout() {
                ErrorKind::Timeout
            } else {
                ErrorKind::Network
            };
            Error::new(
                kind,
                format!("request to provider {} failed", provider.id()),
            )
            .with_provider(provider.id().clone())
            .with_retry(RetryClassification::Safe)
            .with_source(source)
        })?;
        if response.status().is_success() {
            return Ok(response);
        }
        Err(http_error(response, provider).await)
    }
}

/// Normalizes the OpenAI and Anthropic rate-limit header families.
///
/// Request resets and token resets stay separate, and both keep the provider's
/// own formatting. OpenAI sends Go durations such as `6m0s`; Anthropic sends
/// RFC 3339 instants. Callers that need a duration parse the string themselves.
fn rate_limits(headers: &HeaderMap) -> Option<RateLimits> {
    let limits = RateLimits {
        request_limit:     header_u64(headers, &[
            "x-ratelimit-limit-requests",
            "anthropic-ratelimit-requests-limit",
        ]),
        request_remaining: header_u64(headers, &[
            "x-ratelimit-remaining-requests",
            "anthropic-ratelimit-requests-remaining",
        ]),
        request_reset:     header_string(headers, &[
            "x-ratelimit-reset-requests",
            "anthropic-ratelimit-requests-reset",
        ]),
        token_limit:       header_u64(headers, &[
            "x-ratelimit-limit-tokens",
            "anthropic-ratelimit-tokens-limit",
        ]),
        token_remaining:   header_u64(headers, &[
            "x-ratelimit-remaining-tokens",
            "anthropic-ratelimit-tokens-remaining",
        ]),
        token_reset:       header_string(headers, &[
            "x-ratelimit-reset-tokens",
            "anthropic-ratelimit-tokens-reset",
        ]),
    };
    (limits != RateLimits::default()).then_some(limits)
}

fn header_u64(headers: &HeaderMap, names: &[&str]) -> Option<u64> {
    names.iter().find_map(|name| {
        headers
            .get(*name)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse().ok())
    })
}

fn header_string(headers: &HeaderMap, names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| {
        headers
            .get(*name)
            .and_then(|value| value.to_str().ok())
            .map(ToOwned::to_owned)
    })
}

/// Merges every header source for one request, later sources winning.
///
/// The order is codec headers, then provider default headers from the catalog,
/// then credential headers. Credentials therefore win every collision.
fn merge_headers(
    provider: &ProviderId,
    scheme: &AuthScheme,
    codec_headers: &[(String, String)],
    default_headers: &BTreeMap<String, String>,
    credentials: &Credentials,
) -> Result<HeaderMap, Error> {
    let mut headers = HeaderMap::new();
    for (name, value) in codec_headers {
        insert_header(&mut headers, provider, name, value)?;
    }
    for (name, value) in default_headers {
        insert_header(&mut headers, provider, name, value)?;
    }
    insert_credentials(&mut headers, provider, scheme, credentials)?;
    Ok(headers)
}

/// Applies the credential headers that the provider's scheme accepts.
fn insert_credentials(
    headers: &mut HeaderMap,
    provider: &ProviderId,
    scheme: &AuthScheme,
    credentials: &Credentials,
) -> Result<(), Error> {
    let http = match (scheme, credentials) {
        (AuthScheme::None, Credentials::Http(http))
            if matches!(http.auth, HttpAuthentication::None) =>
        {
            http
        }
        (AuthScheme::Bearer { header, prefix }, Credentials::Http(http)) => {
            let HttpAuthentication::Bearer(secret) = &http.auth else {
                return Err(scheme_mismatch(provider));
            };
            let value = format!("{prefix}{}", secret.expose_secret());
            insert_secret(headers, provider, header, &value)?;
            http
        }
        (AuthScheme::Header { name }, Credentials::Http(http)) => {
            let HttpAuthentication::Header(header) = &http.auth else {
                return Err(scheme_mismatch(provider));
            };
            if !name.eq_ignore_ascii_case(&header.name) {
                return Err(scheme_mismatch(provider));
            }
            insert_secret_header(headers, provider, header)?;
            http
        }
        (AuthScheme::Headers, Credentials::Http(http)) => {
            match &http.auth {
                HttpAuthentication::None => {}
                // The scheme names no primary header, so a bearer secret uses
                // the conventional `authorization` header.
                HttpAuthentication::Bearer(secret) => {
                    let value = format!("Bearer {}", secret.expose_secret());
                    insert_secret(headers, provider, "authorization", &value)?;
                }
                HttpAuthentication::Header(header) => {
                    insert_secret_header(headers, provider, header)?;
                }
            }
            http
        }
        (
            AuthScheme::BedrockBearer | AuthScheme::Aws { .. },
            Credentials::BedrockBearer(secret),
        ) => {
            let header = CredentialHeader::new(
                "authorization",
                SecretValue::new(format!("Bearer {}", secret.expose_secret())),
            );
            return insert_secret_header(headers, provider, &header);
        }
        (AuthScheme::Aws { .. }, Credentials::AwsDefaultChain { .. }) => {
            return Err(Error::new(
                ErrorKind::Configuration,
                "AWS default-chain signing is not available in the HTTP Bedrock adapter",
            )
            .with_provider(provider.clone()));
        }
        _ => return Err(scheme_mismatch(provider)),
    };
    for header in &http.extra_headers {
        insert_secret_header(headers, provider, header)?;
    }
    Ok(())
}

fn insert_secret_header(
    headers: &mut HeaderMap,
    provider: &ProviderId,
    header: &CredentialHeader,
) -> Result<(), Error> {
    insert_secret(
        headers,
        provider,
        &header.name,
        header.value.expose_secret(),
    )
}

/// Inserts one non-secret header, replacing any earlier value.
fn insert_header(
    headers: &mut HeaderMap,
    provider: &ProviderId,
    name: &str,
    value: &str,
) -> Result<(), Error> {
    let (name, value) = header_parts(provider, name, value)?;
    headers.insert(name, value);
    Ok(())
}

/// Inserts one secret header, marking it as sensitive so HTTP/2 never places
/// it in a shared compression table.
fn insert_secret(
    headers: &mut HeaderMap,
    provider: &ProviderId,
    name: &str,
    secret: &str,
) -> Result<(), Error> {
    let (name, mut value) = header_parts(provider, name, secret)?;
    value.set_sensitive(true);
    headers.insert(name, value);
    Ok(())
}

fn header_parts(
    provider: &ProviderId,
    name: &str,
    value: &str,
) -> Result<(HeaderName, HeaderValue), Error> {
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
    Ok((name, value))
}

fn scheme_mismatch(provider: &ProviderId) -> Error {
    Error::new(
        ErrorKind::Authentication,
        format!("credentials for provider {provider} do not match its authentication scheme"),
    )
    .with_provider(provider.clone())
}

/// Turns a non-success HTTP response into a classified provider error.
///
/// Message and code extraction and category classification are shared with the
/// mid-stream error path so the two forms of the same provider failure classify
/// identically.
async fn http_error(response: HttpResponse, provider: &CatalogProvider) -> Error {
    let status = response.status();
    let retry_after = response
        .headers()
        .get("retry-after")
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned);
    let data = response.json::<Value>().await.ok();
    provider_error(
        provider,
        Some(status.as_u16()),
        data,
        retry_after.as_deref(),
    )
}

/// Builds a classified provider error from an error body.
///
/// `status` is `None` for a mid-stream error payload, which carries no HTTP
/// status of its own.
pub(crate) fn provider_error(
    provider: &CatalogProvider,
    status: Option<u16>,
    data: Option<Value>,
    retry_after: Option<&str>,
) -> Error {
    let (message, code) = classify::extract(data.as_ref());
    let failure = classify::classify(status, code.as_deref(), message.as_deref(), retry_after);
    let detail = failure.message.clone().unwrap_or_else(|| match status {
        Some(status) => format!("returned HTTP status {status}"),
        None => "returned a stream error".to_owned(),
    });
    let mut error = Error::new(failure.kind, format!("provider {} {detail}", provider.id()))
        .with_provider(provider.id().clone())
        .with_retry(failure.retry);
    if let Some(status) = status {
        error = error.with_status(status);
    }
    if let Some(code) = failure.code {
        error = error.with_provider_code(code);
    }
    if let Some(data) = data {
        error = error.with_raw_data(data);
    }
    error
}

#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "openai-compatible"
))]
fn extract_frames(buffer: &mut Vec<u8>) -> Vec<Result<SseEvent, Error>> {
    let mut events = Vec::new();
    while let Some((end, delimiter_len)) = frame_end(buffer) {
        let frame: Vec<_> = buffer.drain(..end).collect();
        buffer.drain(..delimiter_len);
        match parse_frame(&frame) {
            Ok(Some(event)) => events.push(Ok(event)),
            Ok(None) => {}
            Err(error) => events.push(Err(error)),
        }
    }
    events
}

#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "openai-compatible"
))]
fn frame_end(buffer: &[u8]) -> Option<(usize, usize)> {
    buffer
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|index| (index, 2))
        .or_else(|| {
            buffer
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|index| (index, 4))
        })
}

#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "openai-compatible"
))]
fn parse_frame(frame: &[u8]) -> Result<Option<SseEvent>, Error> {
    let frame = from_utf8(frame).map_err(|source| {
        Error::new(ErrorKind::StreamDecode, "an SSE frame was not UTF-8").with_source(source)
    })?;
    let mut event = None;
    let mut data = Vec::new();
    for line in frame.lines() {
        let line = line.trim_end_matches('\r');
        if let Some(value) = line.strip_prefix("event:") {
            event = Some(value.trim_start().to_owned());
        } else if let Some(value) = line.strip_prefix("data:") {
            data.push(value.trim_start());
        }
    }
    if data.is_empty() {
        return Ok(None);
    }
    let data = data.join("\n");
    if data == "[DONE]" {
        return Ok(None);
    }
    Ok(Some(SseEvent { event, data }))
}

/// Turns buffered AWS event-stream bytes into decoded stream events.
///
/// An `exception` or `error` frame becomes a classified provider error rather
/// than a decoded event, so a mid-stream Bedrock failure reaches the caller
/// through the same taxonomy as an HTTP failure.
#[cfg(feature = "bedrock")]
fn event_stream_events_from(
    buffer: &mut Vec<u8>,
    provider: &ProviderId,
) -> Vec<Result<SseEvent, Error>> {
    event_stream::extract_frames(buffer)
        .into_iter()
        .map(|frame| {
            let frame = frame?;
            let data = String::from_utf8(frame.payload.clone()).map_err(|source| {
                Error::new(
                    ErrorKind::StreamDecode,
                    "a Bedrock event payload was not UTF-8",
                )
                .with_provider(provider.clone())
                .with_source(source)
            })?;
            if frame.is_exception() {
                let body = serde_json::from_str::<Value>(&data).ok();
                let (message, code) = classify::extract(body.as_ref());
                let code = code.or_else(|| frame.exception_type().map(ToOwned::to_owned));
                let failure = classify::classify(None, code.as_deref(), message.as_deref(), None);
                let mut error = Error::new(
                    failure.kind,
                    format!(
                        "provider {provider} {}",
                        failure
                            .message
                            .unwrap_or_else(|| "returned a stream exception".to_owned())
                    ),
                )
                .with_provider(provider.clone())
                .with_retry(failure.retry);
                if let Some(code) = failure.code {
                    error = error.with_provider_code(code);
                }
                if let Some(body) = body {
                    error = error.with_raw_data(body);
                }
                return Err(error);
            }
            Ok(SseEvent {
                event: frame.event_type().map(ToOwned::to_owned),
                data,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{
        AuthScheme, CredentialHeader, Credentials, ErrorKind, HeaderMap, HeaderValue,
        HttpAuthentication, ProviderId, SecretValue, extract_frames, merge_headers, rate_limits,
    };
    use crate::credentials::HttpCredentials;
    use crate::types::Error;

    fn headers_for(
        scheme: &AuthScheme,
        codec: &[(&str, &str)],
        defaults: &[(&str, &str)],
        credentials: &Credentials,
    ) -> Result<HeaderMap, Error> {
        let codec: Vec<(String, String)> = codec
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect();
        let defaults: BTreeMap<String, String> = defaults
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect();
        merge_headers(
            &ProviderId::new("openai"),
            scheme,
            &codec,
            &defaults,
            credentials,
        )
    }

    fn value(headers: &HeaderMap, name: &str) -> Option<String> {
        headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .map(ToOwned::to_owned)
    }

    #[test]
    fn applies_bearer_authentication_with_extra_headers() {
        let scheme = AuthScheme::Bearer {
            header: "authorization".to_owned(),
            prefix: "Bearer ".to_owned(),
        };
        let credentials = Credentials::Http(
            HttpCredentials::new(HttpAuthentication::Bearer(SecretValue::new("key")))
                .with_header(CredentialHeader::new(
                    "openai-organization",
                    SecretValue::new("org-1"),
                ))
                .with_header(CredentialHeader::new(
                    "openai-project",
                    SecretValue::new("proj-1"),
                )),
        );

        let headers = headers_for(&scheme, &[], &[], &credentials).expect("headers should merge");

        assert_eq!(
            value(&headers, "authorization").as_deref(),
            Some("Bearer key")
        );
        assert_eq!(
            value(&headers, "openai-organization").as_deref(),
            Some("org-1")
        );
        assert_eq!(value(&headers, "openai-project").as_deref(), Some("proj-1"));
    }

    #[test]
    fn applies_several_credential_headers_without_authentication() {
        let credentials = Credentials::headers([
            CredentialHeader::new("modal-key", SecretValue::new("key")),
            CredentialHeader::new("modal-secret", SecretValue::new("secret")),
        ]);

        let headers =
            headers_for(&AuthScheme::None, &[], &[], &credentials).expect("headers should merge");

        assert_eq!(value(&headers, "modal-key").as_deref(), Some("key"));
        assert_eq!(value(&headers, "modal-secret").as_deref(), Some("secret"));
    }

    #[test]
    fn credential_headers_win_over_codec_and_provider_headers() {
        let credentials = Credentials::headers([CredentialHeader::new(
            "x-shared",
            SecretValue::new("from-credentials"),
        )]);

        let headers = headers_for(
            &AuthScheme::None,
            &[("x-shared", "from-codec"), ("x-codec", "from-codec")],
            &[("x-shared", "from-defaults"), ("x-codec", "from-defaults")],
            &credentials,
        )
        .expect("headers should merge");

        assert_eq!(
            value(&headers, "x-shared").as_deref(),
            Some("from-credentials")
        );
        assert_eq!(value(&headers, "x-codec").as_deref(), Some("from-defaults"));
    }

    #[test]
    fn reports_a_scheme_mismatch() {
        let scheme = AuthScheme::Header {
            name: "x-api-key".to_owned(),
        };
        let credentials = Credentials::header(CredentialHeader::new(
            "x-other-key",
            SecretValue::new("key"),
        ));

        let error =
            headers_for(&scheme, &[], &[], &credentials).expect_err("the scheme should not match");

        assert_eq!(error.kind(), ErrorKind::Authentication);
    }

    #[test]
    fn parses_frames_across_chunks() {
        let mut buffer = b"event: delta\ndata: {\"text\":\"hel".to_vec();
        assert!(extract_frames(&mut buffer).is_empty());
        buffer.extend_from_slice(b"lo\"}\n\ndata: [DONE]\n\n");
        let events = extract_frames(&mut buffer);
        assert_eq!(events.len(), 1);
        let event = events[0].as_ref().expect("frame should parse");
        assert_eq!(event.event.as_deref(), Some("delta"));
        assert_eq!(event.data, r#"{"text":"hello"}"#);
    }

    #[test]
    fn normalizes_successful_rate_limit_headers() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-ratelimit-limit-requests",
            HeaderValue::from_static("100"),
        );
        headers.insert(
            "anthropic-ratelimit-tokens-remaining",
            HeaderValue::from_static("9000"),
        );

        let limits = rate_limits(&headers).expect("rate limits should be present");

        assert_eq!(limits.request_limit, Some(100));
        assert_eq!(limits.token_remaining, Some(9000));
        assert_eq!(limits.request_remaining, None);
    }

    #[test]
    fn keeps_request_and_token_resets_separate() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-ratelimit-reset-requests",
            HeaderValue::from_static("6m0s"),
        );
        headers.insert("x-ratelimit-reset-tokens", HeaderValue::from_static("1.5s"));

        let limits = rate_limits(&headers).expect("rate limits should be present");

        assert_eq!(limits.request_reset.as_deref(), Some("6m0s"));
        assert_eq!(limits.token_reset.as_deref(), Some("1.5s"));
    }
}
