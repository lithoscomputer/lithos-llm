use std::future::ready;
use std::pin::Pin;
use std::str::from_utf8;
use std::time::Duration;

#[cfg(feature = "bedrock")]
use crc32fast::hash;
use futures_core::Stream;
use futures_util::StreamExt as _;
use futures_util::stream::iter;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use reqwest::{Client, Method, RequestBuilder, Response as HttpResponse, StatusCode};
use serde_json::Value;

use crate::catalog::{AuthScheme, CatalogProvider};
use crate::credentials::{CredentialHeader, Credentials, SecretValue};
use crate::types::{Error, ErrorKind, RateLimits, RetryClassification};

pub(crate) struct EncodedRequest {
    pub method:  Method,
    pub url:     String,
    pub headers: Vec<(String, String)>,
    pub body:    Value,
    pub timeout: Option<Duration>,
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
        let chunks = response.bytes_stream().map(move |result| {
            result.map_err(|source| {
                Error::new(
                    ErrorKind::Network,
                    "reading the Bedrock response stream failed",
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
                        extract_event_stream_frames(buffer)
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
        let mut builder = self
            .client
            .request(request.method, &request.url)
            .json(&request.body);
        if let Some(timeout) = request.timeout {
            builder = builder.timeout(timeout);
        }
        for (name, value) in request.headers {
            builder = add_header(builder, provider, &name, &value)?;
        }
        builder = apply_credentials(builder, provider, credentials)?;

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
        token_limit:       header_u64(headers, &[
            "x-ratelimit-limit-tokens",
            "anthropic-ratelimit-tokens-limit",
        ]),
        token_remaining:   header_u64(headers, &[
            "x-ratelimit-remaining-tokens",
            "anthropic-ratelimit-tokens-remaining",
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

fn apply_credentials(
    mut builder: RequestBuilder,
    provider: &CatalogProvider,
    credentials: Credentials,
) -> Result<RequestBuilder, Error> {
    match (provider.auth(), credentials) {
        (AuthScheme::None, Credentials::None) => Ok(builder),
        (AuthScheme::Bearer { header, prefix }, Credentials::Bearer(secret)) => {
            let value = format!("{prefix}{}", secret.expose_secret());
            add_header(builder, provider, header, &value)
        }
        (AuthScheme::Header { name }, Credentials::Header(header)) => {
            if !name.eq_ignore_ascii_case(&header.name) {
                return Err(scheme_mismatch(provider));
            }
            add_secret_header(builder, provider, &header)
        }
        (AuthScheme::Headers, Credentials::Header(header)) => {
            add_secret_header(builder, provider, &header)
        }
        (AuthScheme::Headers, Credentials::Headers(headers)) => {
            for header in headers {
                builder = add_secret_header(builder, provider, &header)?;
            }
            Ok(builder)
        }
        (
            AuthScheme::BedrockBearer | AuthScheme::Aws { .. },
            Credentials::BedrockBearer(secret),
        ) => {
            let header = CredentialHeader {
                name:  "authorization".to_owned(),
                value: SecretValue::new(format!("Bearer {}", secret.expose_secret())),
            };
            add_secret_header(builder, provider, &header)
        }
        (AuthScheme::Aws { .. }, Credentials::AwsDefaultChain { .. }) => Err(Error::new(
            ErrorKind::Configuration,
            "AWS default-chain signing is not available in the HTTP Bedrock adapter",
        )
        .with_provider(provider.id().clone())),
        _ => Err(scheme_mismatch(provider)),
    }
}

fn add_secret_header(
    builder: RequestBuilder,
    provider: &CatalogProvider,
    header: &CredentialHeader,
) -> Result<RequestBuilder, Error> {
    add_header(
        builder,
        provider,
        &header.name,
        header.value.expose_secret(),
    )
}

fn add_header(
    builder: RequestBuilder,
    provider: &CatalogProvider,
    name: &str,
    value: &str,
) -> Result<RequestBuilder, Error> {
    let name = HeaderName::from_bytes(name.as_bytes()).map_err(|source| {
        Error::new(
            ErrorKind::Configuration,
            format!("provider {} has an invalid HTTP header name", provider.id()),
        )
        .with_provider(provider.id().clone())
        .with_source(source)
    })?;
    let value = HeaderValue::from_str(value).map_err(|source| {
        Error::new(
            ErrorKind::Configuration,
            format!(
                "provider {} has an invalid HTTP header value",
                provider.id()
            ),
        )
        .with_provider(provider.id().clone())
        .with_source(source)
    })?;
    Ok(builder.header(name, value))
}

fn scheme_mismatch(provider: &CatalogProvider) -> Error {
    Error::new(
        ErrorKind::Authentication,
        format!(
            "credentials for provider {} do not match its authentication scheme",
            provider.id()
        ),
    )
    .with_provider(provider.id().clone())
}

async fn http_error(response: HttpResponse, provider: &CatalogProvider) -> Error {
    let status = response.status();
    let retry_after = response
        .headers()
        .get("retry-after")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .map(Duration::from_secs);
    let data = response.json::<Value>().await.ok();
    let provider_code = data
        .as_ref()
        .and_then(|value| value.pointer("/error/code"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let (kind, retry) = classify_http_error(status, provider_code.as_deref(), retry_after);
    let mut error = Error::new(
        kind,
        format!("provider {} returned HTTP status {status}", provider.id()),
    )
    .with_provider(provider.id().clone())
    .with_status(status.as_u16())
    .with_retry(retry);
    if let Some(code) = provider_code {
        error = error.with_provider_code(code);
    }
    if let Some(data) = data {
        error = error.with_raw_data(data);
    }
    error
}

fn classify_http_error(
    status: StatusCode,
    provider_code: Option<&str>,
    retry_after: Option<Duration>,
) -> (ErrorKind, RetryClassification) {
    let kind = match status {
        StatusCode::UNAUTHORIZED => ErrorKind::Authentication,
        StatusCode::FORBIDDEN => ErrorKind::Access,
        StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY => {
            if provider_code.is_some_and(|code| code.contains("context")) {
                ErrorKind::ContextLength
            } else {
                ErrorKind::InvalidRequest
            }
        }
        StatusCode::TOO_MANY_REQUESTS => ErrorKind::RateLimit,
        _ => ErrorKind::Provider,
    };
    let retry = if let Some(delay) = retry_after {
        RetryClassification::After(delay)
    } else if status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
        RetryClassification::Safe
    } else {
        RetryClassification::Never
    };
    (kind, retry)
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

#[cfg(feature = "bedrock")]
fn extract_event_stream_frames(buffer: &mut Vec<u8>) -> Vec<Result<SseEvent, Error>> {
    const PRELUDE_LENGTH: usize = 12;
    const MESSAGE_CRC_LENGTH: usize = 4;
    const MAX_FRAME_LENGTH: usize = 16 * 1024 * 1024;

    let mut events = Vec::new();
    while buffer.len() >= PRELUDE_LENGTH {
        let total_length = usize::try_from(u32::from_be_bytes([
            buffer[0], buffer[1], buffer[2], buffer[3],
        ]))
        .unwrap_or(usize::MAX);
        let headers_length = usize::try_from(u32::from_be_bytes([
            buffer[4], buffer[5], buffer[6], buffer[7],
        ]))
        .unwrap_or(usize::MAX);
        if !(PRELUDE_LENGTH + MESSAGE_CRC_LENGTH..=MAX_FRAME_LENGTH).contains(&total_length)
            || PRELUDE_LENGTH + headers_length + MESSAGE_CRC_LENGTH > total_length
        {
            buffer.clear();
            events.push(Err(Error::new(
                ErrorKind::StreamDecode,
                "Bedrock returned an invalid event-stream frame length",
            )));
            break;
        }
        if buffer.len() < total_length {
            break;
        }
        let frame: Vec<_> = buffer.drain(..total_length).collect();
        let expected_prelude_crc = u32::from_be_bytes([frame[8], frame[9], frame[10], frame[11]]);
        let expected_message_crc = u32::from_be_bytes([
            frame[total_length - 4],
            frame[total_length - 3],
            frame[total_length - 2],
            frame[total_length - 1],
        ]);
        if hash(&frame[..8]) != expected_prelude_crc
            || hash(&frame[..total_length - 4]) != expected_message_crc
        {
            events.push(Err(Error::new(
                ErrorKind::StreamDecode,
                "Bedrock returned an event-stream frame with an invalid checksum",
            )));
            continue;
        }
        let payload_start = PRELUDE_LENGTH + headers_length;
        let payload = &frame[payload_start..total_length - MESSAGE_CRC_LENGTH];
        match from_utf8(payload) {
            Ok(data) => events.push(Ok(SseEvent {
                event: None,
                data:  data.to_owned(),
            })),
            Err(source) => events.push(Err(Error::new(
                ErrorKind::StreamDecode,
                "a Bedrock event payload was not UTF-8",
            )
            .with_source(source))),
        }
    }
    events
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{
        ErrorKind, HeaderMap, HeaderValue, RetryClassification, StatusCode, classify_http_error,
        extract_frames, rate_limits,
    };

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
    fn classifies_provider_errors_for_retry() {
        assert_eq!(
            classify_http_error(
                StatusCode::TOO_MANY_REQUESTS,
                None,
                Some(Duration::from_secs(2)),
            ),
            (
                ErrorKind::RateLimit,
                RetryClassification::After(Duration::from_secs(2)),
            )
        );
        assert_eq!(
            classify_http_error(
                StatusCode::BAD_REQUEST,
                Some("context_length_exceeded"),
                None,
            ),
            (ErrorKind::ContextLength, RetryClassification::Never)
        );
        assert_eq!(
            classify_http_error(StatusCode::SERVICE_UNAVAILABLE, None, None),
            (ErrorKind::Provider, RetryClassification::Safe)
        );
    }
}
