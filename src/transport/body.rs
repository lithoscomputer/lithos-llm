//! Response bodies: bounded reads, JSON decoding, and error bodies.

use reqwest::Response as HttpResponse;
use reqwest::header::HeaderMap;
use serde_json::Value;

use super::classify;
use super::headers::rate_limits;
use crate::catalog::CatalogProvider;
use crate::types::{Error, ErrorKind, RateLimits, RetryClassification, limit_error};

#[derive(Debug)]
pub(crate) struct JsonResponse {
    pub body:        Value,
    pub rate_limits: Option<RateLimits>,
    /// The response headers, for a codec that reads a value the transport
    /// does not model, such as a request id.
    pub headers:     HeaderMap,
}

impl JsonResponse {
    /// One response header as text, when it is present and valid UTF-8.
    pub(crate) fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name).and_then(|value| value.to_str().ok())
    }
}

/// Collect no more than the configured bytes, even without Content-Length.
async fn bounded_body(
    mut response: HttpResponse,
    provider: &CatalogProvider,
    limit: usize,
) -> Result<Vec<u8>, Error> {
    if response
        .content_length()
        .is_some_and(|length| length > limit as u64)
    {
        return Err(limit_error("HTTP body", limit).with_provider(provider.id().clone()));
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|source| {
        let (kind, retry) = if source.is_timeout() {
            (ErrorKind::Timeout, RetryClassification::Never)
        } else {
            (ErrorKind::ResponseDecode, RetryClassification::Safe)
        };
        Error::new(kind, "reading the provider response body failed")
            .with_provider(provider.id().clone())
            .with_retry(retry)
            .with_source(source)
    })? {
        if chunk.len() > limit.saturating_sub(bytes.len()) {
            return Err(limit_error("HTTP body", limit).with_provider(provider.id().clone()));
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

/// Reads rate limits and decodes a JSON success body.
///
/// A body that does not parse is classified [`RetryClassification::Safe`]: the
/// common cause is a proxy that truncated an otherwise good response, and the
/// same request sent again normally succeeds. A request timeout that expires
/// mid-body is the exception — the provider already executed the call, so it
/// keeps the complete path's never-retry rule.
pub(super) async fn json_response(
    response: HttpResponse,
    provider: &CatalogProvider,
    body_limit: usize,
) -> Result<JsonResponse, Error> {
    let headers = response.headers().clone();
    let rate_limits = rate_limits(&headers);
    let bytes = bounded_body(response, provider, body_limit).await?;
    let body = serde_json::from_slice(&bytes).map_err(|source| {
        Error::new(
            ErrorKind::ResponseDecode,
            format!("provider {} returned invalid JSON", provider.id()),
        )
        .with_provider(provider.id().clone())
        .with_retry(RetryClassification::Safe)
        .with_source(source)
    })?;
    Ok(JsonResponse {
        body,
        rate_limits,
        headers,
    })
}

/// Turns a non-success HTTP response into a classified provider error.
///
/// Message and code extraction and category classification are shared with the
/// mid-stream error path so the two forms of the same provider failure classify
/// identically.
/// A body that is not JSON is kept as text rather than discarded: an HTML 503
/// from a proxy or a plain-text "model does not exist" 400 carries the only
/// diagnosis there is, and message-based classification needs it. Long bodies
/// are truncated, because an error message is not a place for a whole page.
pub(super) async fn http_error(
    response: HttpResponse,
    provider: &CatalogProvider,
    body_limit: usize,
) -> Error {
    let status = response.status();
    let retry_after = response
        .headers()
        .get("retry-after")
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned);
    let data = match bounded_body(response, provider, body_limit).await {
        Ok(body) => error_body(&String::from_utf8_lossy(&body)),
        Err(error) if error.kind() == ErrorKind::ResourceLimit => {
            return error.with_status(status.as_u16());
        }
        Err(_) => None,
    };
    provider_error(
        provider,
        Some(status.as_u16()),
        data,
        retry_after.as_deref(),
    )
}

/// The most characters of a non-JSON error body an error message keeps.
const ERROR_BODY_LIMIT: usize = 2_000;

/// Turns an error response body into the value the classifier reads.
///
/// JSON parses to itself. Any other non-empty text becomes a JSON string,
/// which [`classify::extract`] reads as the provider's message.
fn error_body(body: &str) -> Option<Value> {
    if let Ok(value) = serde_json::from_str::<Value>(body) {
        return Some(value);
    }
    let body = body.trim();
    if body.is_empty() {
        return None;
    }
    let truncated = match body.char_indices().nth(ERROR_BODY_LIMIT) {
        Some((end, _)) => format!("{}…", &body[..end]),
        None => body.to_owned(),
    };
    Some(Value::String(truncated))
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
    let fallback = match status {
        Some(status) => format!("returned HTTP status {status}"),
        None => "returned a stream error".to_owned(),
    };
    classify::classify(status, code.as_deref(), message.as_deref(), retry_after).into_error(
        provider.id(),
        status,
        data,
        &fallback,
    )
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{ERROR_BODY_LIMIT, error_body};

    #[test]
    fn keeps_a_non_json_error_body_as_its_message() {
        let body =
            error_body("The model gpt-9 does not exist").expect("a non-empty body should be kept");

        assert_eq!(body, Value::String("The model gpt-9 does not exist".into()));
    }

    #[test]
    fn truncates_a_very_large_error_body() {
        let body = error_body(&"<html>".repeat(1_000)).expect("a non-empty body should be kept");

        let text = body.as_str().expect("the body should be kept as text");
        assert_eq!(text.chars().count(), ERROR_BODY_LIMIT + 1);
        assert!(text.ends_with('…'));
    }

    #[test]
    fn an_empty_error_body_stays_empty() {
        assert_eq!(error_body("   "), None);
    }

    #[test]
    fn keeps_a_json_error_body_as_json() {
        let body = error_body(r#"{"error":{"message":"nope"}}"#).expect("JSON should be kept");

        assert_eq!(body, json!({ "error": { "message": "nope" } }));
    }
}
