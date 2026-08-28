use std::error::Error as StdError;
use std::fmt;
use std::time::Duration;

use serde_json::Value;

use crate::catalog::ProviderId;

/// A stable category that applications can use for error handling.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ErrorKind {
    Configuration,
    ModelSelection,
    Authentication,
    Access,
    InvalidRequest,
    ContextLength,
    RateLimit,
    Provider,
    Network,
    Timeout,
    StreamDecode,
    Middleware,
    Cancelled,
}

/// Whether repeating the same resolved call is safe.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RetryClassification {
    Never,
    Safe,
    After(Duration),
}

/// A provider-neutral runtime failure.
#[derive(Debug)]
#[must_use]
pub struct Error {
    kind:          ErrorKind,
    message:       String,
    provider:      Option<ProviderId>,
    status:        Option<u16>,
    provider_code: Option<String>,
    retry:         RetryClassification,
    raw_data:      Option<Box<Value>>,
    source:        Option<Box<dyn StdError + Send + Sync>>,
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
        self.provider_code = Some(code.into());
        self
    }

    pub fn with_retry(mut self, retry: RetryClassification) -> Self {
        self.retry = retry;
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
        match self.retry {
            RetryClassification::After(duration) => Some(duration),
            RetryClassification::Never | RetryClassification::Safe => None,
        }
    }

    pub fn raw_data(&self) -> Option<&Value> {
        self.raw_data.as_deref()
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
