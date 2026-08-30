use std::error::Error as StdError;
use std::fmt;
use std::num::NonZeroU64;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::catalog::ProviderId;

/// A stable category that applications can use for error handling.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ErrorKind {
    /// The client, catalog, or credentials are configured incorrectly.
    Configuration,
    /// No provider and model could be selected for the request.
    ModelSelection,
    /// The credential is missing, malformed, or rejected.
    Authentication,
    /// The credential is valid but the account or policy blocks the call.
    AccessDenied,
    /// The model, deployment, or route does not exist on this provider.
    NotFound,
    /// The request itself is malformed or unsupported.
    InvalidRequest,
    /// The input exceeds the model's context window.
    ContextLength,
    /// The provider is throttling; backoff clears it.
    RateLimit,
    /// Credit, billing, or plan quota is spent; backoff never clears it.
    QuotaExceeded,
    /// The provider blocked the content, or the model refused the request.
    ContentFilter,
    /// The provider failed on its own side, typically an HTTP 5xx.
    Server,
    /// A provider failure that no other category describes.
    Provider,
    /// The request never reached the provider.
    Network,
    /// The call exceeded its time budget.
    Timeout,
    /// A streaming payload could not be decoded.
    StreamDecode,
    /// A successful response body could not be decoded.
    ResponseDecode,
    /// Middleware rejected or failed the call.
    Middleware,
    /// The caller cancelled the call.
    Cancelled,
}

/// Whether repeating the same resolved call is safe.
///
/// This describes the same provider and the same request only. Failover to
/// another provider is the application's policy, not this crate's.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum RetryClassification {
    /// Repeating the call cannot succeed.
    Never,
    /// Repeating the call is safe on the caller's own schedule.
    Safe,
    /// Repeating the call is safe after the given delay.
    After {
        /// The delay in milliseconds.
        #[serde(rename = "after_millis")]
        millis: u64,
    },
}

impl RetryClassification {
    /// Builds a [`RetryClassification::After`] from a delay.
    ///
    /// Delays longer than `u64::MAX` milliseconds saturate.
    pub fn after(delay: Duration) -> Self {
        Self::After {
            millis: u64::try_from(delay.as_millis()).unwrap_or(u64::MAX),
        }
    }

    /// The advised delay, when this classification carries one.
    pub const fn delay(self) -> Option<Duration> {
        match self {
            Self::After { millis } => Some(Duration::from_millis(millis)),
            Self::Never | Self::Safe => None,
        }
    }
}

/// A cloneable, serializable projection of a runtime [`Error`].
///
/// [`Error`] keeps a live source chain and is therefore neither `Clone` nor
/// serializable. `ErrorData` carries the same structured facts, plus the
/// immediate source rendered as text, so applications can persist or transport
/// a failure without parsing display strings.
///
/// This projection is not redacted. Provider data and source messages can
/// contain sensitive response content or URLs. Applications must apply their
/// own storage and disclosure policy before they serialize or log it.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[non_exhaustive]
pub struct ErrorData {
    /// The stable category of the failure.
    pub kind: ErrorKind,

    /// The human-readable message of the error itself.
    pub message: String,

    /// The provider that produced the failure, when one was selected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderId>,

    /// The HTTP status, when the failure came from an HTTP response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,

    /// The provider's own error code, as reported on the wire.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_code: Option<String>,

    /// Whether repeating the same resolved call is safe.
    pub retry: RetryClassification,

    /// The parsed provider error body, when there was one.
    ///
    /// This value can contain sensitive provider response content. It is
    /// preserved for explicit diagnostics and must not be logged by default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_data: Option<Value>,

    /// The provider's advised wait in milliseconds, whatever the kind.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_retry_after_millis: Option<u64>,

    /// The `Display` text of the immediate source, when there is one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_message: Option<String>,
}

/// A provider-neutral runtime failure.
#[must_use]
pub struct Error {
    kind: ErrorKind,
    message: String,
    provider: Option<ProviderId>,
    status: Option<u16>,
    /// A `Box<str>` rather than a `String`: with the advised-wait field
    /// beside it, the spare capacity word would push the error past Clippy's
    /// large-`Err` threshold.
    provider_code: Option<Box<str>>,
    retry: RetryClassification,
    /// Non-zero milliseconds rather than a `Duration`: the niche keeps the
    /// error under Clippy's large-`Err` threshold, and a zero-length wait
    /// advises nothing.
    provider_retry_after_millis: Option<NonZeroU64>,
    raw_data: Option<Box<Value>>,
    source: Option<Box<dyn StdError + Send + Sync>>,
}

impl fmt::Debug for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Error")
            .field("kind", &self.kind)
            .field("provider", &self.provider)
            .field("status", &self.status)
            .field("provider_code", &self.provider_code)
            .field("retry", &self.retry)
            .field(
                "provider_retry_after_millis",
                &self.provider_retry_after_millis,
            )
            .field("has_raw_data", &self.raw_data.is_some())
            .field("has_source", &self.source.is_some())
            .finish_non_exhaustive()
    }
}

impl Error {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            provider: None,
            status: None,
            provider_code: None,
            retry: RetryClassification::Never,
            provider_retry_after_millis: None,
            raw_data: None,
            source: None,
        }
    }

    pub fn with_provider(mut self, provider: ProviderId) -> Self {
        self.provider = Some(provider);
        self
    }

    pub fn with_status(mut self, status: u16) -> Self {
        self.status = Some(status);
        self
    }

    pub fn with_provider_code(mut self, code: impl Into<String>) -> Self {
        self.provider_code = Some(code.into().into_boxed_str());
        self
    }

    pub fn with_retry(mut self, retry: RetryClassification) -> Self {
        self.retry = retry;
        self
    }

    /// Records the provider's advised wait, whatever the error kind.
    ///
    /// Delays longer than `u64::MAX` milliseconds saturate, and a zero-length
    /// delay is not recorded — it advises nothing.
    pub fn with_provider_retry_after(mut self, delay: Duration) -> Self {
        self.provider_retry_after_millis =
            NonZeroU64::new(u64::try_from(delay.as_millis()).unwrap_or(u64::MAX));
        self
    }

    pub fn with_raw_data(mut self, data: Value) -> Self {
        self.raw_data = Some(Box::new(data));
        self
    }

    pub fn with_source(mut self, source: impl StdError + Send + Sync + 'static) -> Self {
        self.source = Some(Box::new(source));
        self
    }

    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn provider(&self) -> Option<&ProviderId> {
        self.provider.as_ref()
    }

    pub fn status(&self) -> Option<u16> {
        self.status
    }

    pub fn provider_code(&self) -> Option<&str> {
        self.provider_code.as_deref()
    }

    pub fn retry_classification(&self) -> RetryClassification {
        self.retry
    }

    pub fn retry_after(&self) -> Option<Duration> {
        self.retry.delay()
    }

    /// The provider's advised wait, whatever the error kind.
    ///
    /// [`retry_after`](Self::retry_after) carries the delay only when the
    /// classification says repeating the same call is safe. A provider also
    /// sends `Retry-After` on failures this crate never retries — a 429
    /// classified as spent quota — and an application scheduling its own
    /// failover still wants that hint.
    pub fn provider_retry_after(&self) -> Option<Duration> {
        self.provider_retry_after_millis
            .map(|millis| Duration::from_millis(millis.get()))
    }

    pub fn raw_data(&self) -> Option<&Value> {
        self.raw_data.as_deref()
    }

    /// A cloneable, serializable projection of this error.
    ///
    /// The source chain is not copied. Its immediate entry is rendered into
    /// [`ErrorData::source_message`].
    pub fn data(&self) -> ErrorData {
        ErrorData {
            kind: self.kind,
            message: self.message.clone(),
            provider: self.provider.clone(),
            status: self.status,
            provider_code: self.provider_code.as_deref().map(ToOwned::to_owned),
            retry: self.retry,
            provider_retry_after_millis: self.provider_retry_after_millis.map(NonZeroU64::get),
            raw_data: self.raw_data.as_deref().cloned(),
            source_message: self.source.as_ref().map(ToString::to_string),
        }
    }
}

impl From<&Error> for ErrorData {
    fn from(error: &Error) -> Self {
        error.data()
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source
            .as_deref()
            .map(|source| source as &(dyn StdError + 'static))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[derive(Debug)]
    struct TestSource;

    impl fmt::Display for TestSource {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("the socket closed")
        }
    }

    impl StdError for TestSource {}

    fn sample_error() -> Error {
        Error::new(
            ErrorKind::RateLimit,
            "provider openai returned HTTP status 429",
        )
        .with_provider(ProviderId::new("openai"))
        .with_status(429)
        .with_provider_code("rate_limit_exceeded")
        .with_retry(RetryClassification::after(Duration::from_millis(1500)))
        .with_raw_data(json!({"error": {"message": "slow down"}}))
    }

    #[test]
    fn projects_every_structured_field() {
        let data = sample_error().data();

        assert_eq!(data.kind, ErrorKind::RateLimit);
        assert_eq!(data.provider, Some(ProviderId::new("openai")));
        assert_eq!(data.status, Some(429));
        assert_eq!(data.provider_code.as_deref(), Some("rate_limit_exceeded"));
        assert_eq!(data.retry, RetryClassification::After { millis: 1500 });
        assert_eq!(
            data.raw_data,
            Some(json!({"error": {"message": "slow down"}}))
        );
        assert_eq!(data.source_message, None);
    }

    #[test]
    fn error_data_clones_and_round_trips_through_serde() {
        let data = sample_error().data();
        let clone = data.clone();

        let encoded = serde_json::to_string(&clone).expect("serialize error data");
        let decoded: ErrorData = serde_json::from_str(&encoded).expect("deserialize error data");

        assert_eq!(decoded, data);
    }

    #[test]
    fn reports_a_runtime_source_as_source_message() {
        let error = Error::new(ErrorKind::Network, "the request failed").with_source(TestSource);

        let data = error.data();

        assert_eq!(data.source_message.as_deref(), Some("the socket closed"));
        assert!(error.source().is_some());

        // The projection is serializable even though the error itself is not.
        let encoded = serde_json::to_string(&data).expect("serialize error data");
        assert!(encoded.contains("the socket closed"));
    }

    #[test]
    fn debug_output_redacts_messages_raw_data_and_sources() {
        let error = Error::new(ErrorKind::Network, "sensitive message")
            .with_raw_data(json!({ "secret": "raw provider value" }))
            .with_source(TestSource);

        let debug = format!("{error:?}");

        assert!(debug.contains("kind: Network"));
        assert!(debug.contains("has_raw_data: true"));
        assert!(debug.contains("has_source: true"));
        assert!(!debug.contains("sensitive message"));
        assert!(!debug.contains("raw provider value"));
        assert!(!debug.contains("TestSource"));
    }

    #[test]
    fn error_data_from_reference_matches_data() {
        let error = sample_error();

        assert_eq!(ErrorData::from(&error), error.data());
    }

    #[test]
    fn after_serializes_a_millisecond_value_and_round_trips() {
        let retry = RetryClassification::after(Duration::from_millis(2500));

        let encoded = serde_json::to_value(retry).expect("serialize retry classification");

        assert_eq!(encoded, json!({"type": "after", "after_millis": 2500}));

        let decoded: RetryClassification =
            serde_json::from_value(encoded).expect("deserialize retry classification");

        assert_eq!(decoded, retry);
        assert_eq!(decoded.delay(), Some(Duration::from_millis(2500)));
    }

    #[test]
    fn never_and_safe_carry_no_delay() {
        assert_eq!(RetryClassification::Never.delay(), None);
        assert_eq!(RetryClassification::Safe.delay(), None);
        assert_eq!(
            serde_json::to_value(RetryClassification::Safe).expect("serialize"),
            json!({"type": "safe"})
        );
    }

    #[test]
    fn after_saturates_an_unrepresentable_delay() {
        let retry = RetryClassification::after(Duration::MAX);

        assert_eq!(retry, RetryClassification::After { millis: u64::MAX });
    }

    #[test]
    fn retry_after_reads_the_classification_delay() {
        let error = Error::new(ErrorKind::RateLimit, "slow down")
            .with_retry(RetryClassification::after(Duration::from_secs(3)));

        assert_eq!(error.retry_after(), Some(Duration::from_secs(3)));
    }

    #[test]
    fn error_kinds_use_snake_case_names() {
        assert_eq!(
            serde_json::to_value(ErrorKind::AccessDenied).expect("serialize"),
            json!("access_denied")
        );
        assert_eq!(
            serde_json::to_value(ErrorKind::ResponseDecode).expect("serialize"),
            json!("response_decode")
        );

        let decoded: ErrorKind =
            serde_json::from_value(json!("quota_exceeded")).expect("deserialize");

        assert_eq!(decoded, ErrorKind::QuotaExceeded);
    }
}
