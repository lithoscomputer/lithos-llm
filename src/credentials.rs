//! Secret-safe credential resolution.

use std::collections::BTreeMap;
use std::{env, fmt};

use async_trait::async_trait;
use thiserror::Error;

use crate::catalog::{CatalogProvider, ProviderId};

/// A secret string whose debug output is always redacted.
#[derive(Clone, Eq, PartialEq)]
pub struct SecretValue(String);

impl SecretValue {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Exposes the secret for an outbound authentication boundary.
    pub fn expose_secret(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for SecretValue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("SecretValue(<redacted>)")
    }
}

/// One resolved secret HTTP header.
#[derive(Clone, Eq, PartialEq)]
pub struct CredentialHeader {
    pub name:  String,
    pub value: SecretValue,
}

impl fmt::Debug for CredentialHeader {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CredentialHeader")
            .field("name", &self.name)
            .field("value", &"<redacted>")
            .finish()
    }
}

/// Resolved credentials for one provider attempt.
#[derive(Clone, Eq, PartialEq)]
#[non_exhaustive]
pub enum Credentials {
    None,
    Bearer(SecretValue),
    Header(CredentialHeader),
    Headers(Vec<CredentialHeader>),
    AwsDefaultChain { region: Option<String> },
    BedrockBearer(SecretValue),
}

impl fmt::Debug for Credentials {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => formatter.write_str("Credentials::None"),
            Self::Bearer(_) => formatter.write_str("Credentials::Bearer(<redacted>)"),
            Self::Header(header) => formatter
                .debug_tuple("Credentials::Header")
                .field(header)
                .finish(),
            Self::Headers(headers) => formatter
                .debug_tuple("Credentials::Headers")
                .field(headers)
                .finish(),
            Self::AwsDefaultChain { region } => formatter
                .debug_struct("Credentials::AwsDefaultChain")
                .field("region", region)
                .finish(),
            Self::BedrockBearer(_) => formatter.write_str("Credentials::BedrockBearer(<redacted>)"),
        }
    }
}

/// Supplies credentials for each provider attempt.
#[async_trait]
pub trait CredentialProvider: Send + Sync {
    async fn credentials(&self, provider: &CatalogProvider)
    -> Result<Credentials, CredentialError>;
}

/// A provider that supplies no authentication.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoCredentials;

#[async_trait]
impl CredentialProvider for NoCredentials {
    async fn credentials(
        &self,
        _provider: &CatalogProvider,
    ) -> Result<Credentials, CredentialError> {
        Ok(Credentials::None)
    }
}

/// Fixed credentials for tests, local providers, and application-managed
/// secrets.
#[derive(Clone, Debug, Default)]
#[must_use]
pub struct StaticCredentials {
    credentials: BTreeMap<ProviderId, Credentials>,
}

impl StaticCredentials {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, provider: impl Into<ProviderId>, credentials: Credentials) -> Self {
        self.credentials.insert(provider.into(), credentials);
        self
    }
}

#[async_trait]
impl CredentialProvider for StaticCredentials {
    async fn credentials(
        &self,
        provider: &CatalogProvider,
    ) -> Result<Credentials, CredentialError> {
        self.credentials
            .get(provider.id())
            .cloned()
            .ok_or_else(|| CredentialError::NotConfigured {
                provider: provider.id().clone(),
            })
    }
}

#[cfg(feature = "environment-credentials")]
/// Resolves secrets from environment variables for every attempt.
#[derive(Clone, Debug)]
pub struct EnvironmentCredentials {
    specs: BTreeMap<ProviderId, EnvironmentSpec>,
}

#[cfg(feature = "environment-credentials")]
impl EnvironmentCredentials {
    pub fn builder() -> EnvironmentCredentialsBuilder {
        EnvironmentCredentialsBuilder::default()
    }

    /// Conventional environment variable mappings for built-in providers.
    pub fn conventional() -> Self {
        EnvironmentCredentialsBuilder::default()
            .bearer("openai", "OPENAI_API_KEY")
            .header("anthropic", "x-api-key", "ANTHROPIC_API_KEY")
            .header("gemini", "x-goog-api-key", "GEMINI_API_KEY")
            .bedrock_bearer("bedrock", "AWS_BEARER_TOKEN_BEDROCK")
            .build()
    }
}

#[cfg(feature = "environment-credentials")]
#[async_trait]
impl CredentialProvider for EnvironmentCredentials {
    async fn credentials(
        &self,
        provider: &CatalogProvider,
    ) -> Result<Credentials, CredentialError> {
        let spec = self
            .specs
            .get(provider.id())
            .ok_or_else(|| CredentialError::NotConfigured {
                provider: provider.id().clone(),
            })?;
        spec.resolve(provider.id())
    }
}

#[cfg(feature = "environment-credentials")]
#[derive(Clone, Debug)]
enum EnvironmentSpec {
    Bearer(String),
    Headers(Vec<(String, String)>),
    AwsDefaultChain(Option<String>),
    BedrockBearer(String),
}

#[cfg(feature = "environment-credentials")]
impl EnvironmentSpec {
    fn resolve(&self, provider: &ProviderId) -> Result<Credentials, CredentialError> {
        match self {
            Self::Bearer(variable) => read_secret(provider, variable).map(Credentials::Bearer),
            Self::Headers(headers) => headers
                .iter()
                .map(|(name, variable)| {
                    Ok(CredentialHeader {
                        name:  name.clone(),
                        value: read_secret(provider, variable)?,
                    })
                })
                .collect::<Result<Vec<_>, CredentialError>>()
                .map(|headers| match headers.as_slice() {
                    [header] => Credentials::Header(header.clone()),
                    _ => Credentials::Headers(headers),
                }),
            Self::AwsDefaultChain(region) => Ok(Credentials::AwsDefaultChain {
                region: region.clone(),
            }),
            Self::BedrockBearer(variable) => {
                read_secret(provider, variable).map(Credentials::BedrockBearer)
            }
        }
    }
}

#[cfg(feature = "environment-credentials")]
fn read_secret(provider: &ProviderId, variable: &str) -> Result<SecretValue, CredentialError> {
    env::var(variable)
        .map(SecretValue::new)
        .map_err(|source| CredentialError::Environment {
            provider: provider.clone(),
            variable: variable.to_owned(),
            source,
        })
}

#[cfg(feature = "environment-credentials")]
/// Builds environment-backed provider mappings.
#[derive(Default)]
#[must_use]
pub struct EnvironmentCredentialsBuilder {
    specs: BTreeMap<ProviderId, EnvironmentSpec>,
}

#[cfg(feature = "environment-credentials")]
impl EnvironmentCredentialsBuilder {
    pub fn bearer(mut self, provider: impl Into<ProviderId>, variable: impl Into<String>) -> Self {
        self.specs
            .insert(provider.into(), EnvironmentSpec::Bearer(variable.into()));
        self
    }

    pub fn header(
        mut self,
        provider: impl Into<ProviderId>,
        name: impl Into<String>,
        variable: impl Into<String>,
    ) -> Self {
        let provider = provider.into();
        let header = (name.into(), variable.into());
        match self
            .specs
            .entry(provider)
            .or_insert_with(|| EnvironmentSpec::Headers(Vec::new()))
        {
            EnvironmentSpec::Headers(headers) => headers.push(header),
            spec => *spec = EnvironmentSpec::Headers(vec![header]),
        }
        self
    }

    pub fn aws_default_chain(
        mut self,
        provider: impl Into<ProviderId>,
        region: Option<String>,
    ) -> Self {
        self.specs
            .insert(provider.into(), EnvironmentSpec::AwsDefaultChain(region));
        self
    }

    pub fn bedrock_bearer(
        mut self,
        provider: impl Into<ProviderId>,
        variable: impl Into<String>,
    ) -> Self {
        self.specs.insert(
            provider.into(),
            EnvironmentSpec::BedrockBearer(variable.into()),
        );
        self
    }

    pub fn build(self) -> EnvironmentCredentials {
        EnvironmentCredentials { specs: self.specs }
    }
}

/// Credential lookup failed without exposing secret content.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CredentialError {
    #[error("credentials are not configured for provider {provider}")]
    NotConfigured { provider: ProviderId },
    #[error("environment variable `{variable}` for provider {provider} is unavailable")]
    Environment {
        provider: ProviderId,
        variable: String,
        #[source]
        source:   env::VarError,
    },
    #[error("credentials for provider {provider} do not match its authentication scheme")]
    SchemeMismatch { provider: ProviderId },
}

#[cfg(test)]
mod tests {
    use super::{Credentials, SecretValue};

    #[test]
    fn debug_output_redacts_secrets() {
        let credentials = Credentials::Bearer(SecretValue::new("top-secret"));
        let output = format!("{credentials:?}");
        assert!(output.contains("redacted"));
        assert!(!output.contains("top-secret"));
    }
}
