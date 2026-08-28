//! Secret-safe credential resolution.

use std::collections::BTreeMap;
#[cfg(feature = "environment-credentials")]
use std::mem;
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

impl CredentialHeader {
    /// Builds one secret header from a name and a secret value.
    pub fn new(name: impl Into<String>, value: SecretValue) -> Self {
        Self {
            name: name.into(),
            value,
        }
    }
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

/// The primary HTTP authentication method for one provider attempt.
#[derive(Clone, Eq, PartialEq)]
#[non_exhaustive]
pub enum HttpAuthentication {
    /// The provider needs no primary authentication header.
    None,
    /// The secret is sent through the provider's bearer header.
    Bearer(SecretValue),
    /// The secret is sent through one named header.
    Header(CredentialHeader),
}

impl fmt::Debug for HttpAuthentication {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => formatter.write_str("HttpAuthentication::None"),
            Self::Bearer(_) => formatter.write_str("HttpAuthentication::Bearer(<redacted>)"),
            Self::Header(header) => formatter
                .debug_tuple("HttpAuthentication::Header")
                .field(&header.name)
                .finish(),
        }
    }
}

/// Resolved HTTP authentication plus any extra credential headers.
///
/// Extra headers are applied after the primary authentication header, in the
/// order they were added. They carry secrets such as OpenAI organization and
/// project identifiers or proxy tokens.
#[derive(Clone, Eq, PartialEq)]
pub struct HttpCredentials {
    pub auth:          HttpAuthentication,
    pub extra_headers: Vec<CredentialHeader>,
}

impl HttpCredentials {
    /// Builds credentials with one authentication method and no extra headers.
    pub fn new(auth: HttpAuthentication) -> Self {
        Self {
            auth,
            extra_headers: Vec::new(),
        }
    }

    /// Adds one extra credential header.
    #[must_use]
    pub fn with_header(mut self, header: CredentialHeader) -> Self {
        self.extra_headers.push(header);
        self
    }
}

impl fmt::Debug for HttpCredentials {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&str> = self
            .extra_headers
            .iter()
            .map(|header| header.name.as_str())
            .collect();
        formatter
            .debug_struct("HttpCredentials")
            .field("auth", &self.auth)
            .field("extra_header_names", &names)
            .finish()
    }
}

/// Resolved credentials for one provider attempt.
#[derive(Clone, Eq, PartialEq)]
#[non_exhaustive]
pub enum Credentials {
    /// Header-based credentials for an ordinary HTTP provider.
    Http(HttpCredentials),
    /// AWS credentials taken from the default provider chain.
    AwsDefaultChain { region: Option<String> },
    /// A long-lived Amazon Bedrock API key.
    BedrockBearer(SecretValue),
}

impl Credentials {
    /// Builds credentials that send no authentication and no extra headers.
    pub fn none() -> Self {
        Self::Http(HttpCredentials::new(HttpAuthentication::None))
    }

    /// Builds bearer credentials with no extra headers.
    pub fn bearer(secret: SecretValue) -> Self {
        Self::Http(HttpCredentials::new(HttpAuthentication::Bearer(secret)))
    }

    /// Builds single-header credentials with no extra headers.
    pub fn header(header: CredentialHeader) -> Self {
        Self::Http(HttpCredentials::new(HttpAuthentication::Header(header)))
    }

    /// Builds unauthenticated credentials that send every given header.
    ///
    /// This is the multi-header path for providers that authenticate through
    /// several proxy headers rather than one primary header.
    pub fn headers(headers: impl IntoIterator<Item = CredentialHeader>) -> Self {
        Self::Http(HttpCredentials {
            auth:          HttpAuthentication::None,
            extra_headers: headers.into_iter().collect(),
        })
    }
}

impl fmt::Debug for Credentials {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http(credentials) => formatter
                .debug_tuple("Credentials::Http")
                .field(credentials)
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
        Ok(Credentials::none())
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
    Http(HttpSpec),
    AwsDefaultChain(Option<String>),
    BedrockBearer(String),
}

#[cfg(feature = "environment-credentials")]
impl EnvironmentSpec {
    fn resolve(&self, provider: &ProviderId) -> Result<Credentials, CredentialError> {
        match self {
            Self::Http(spec) => spec.resolve(provider).map(Credentials::Http),
            Self::AwsDefaultChain(region) => Ok(Credentials::AwsDefaultChain {
                region: region.clone(),
            }),
            Self::BedrockBearer(variable) => {
                read_secret(provider, variable).map(Credentials::BedrockBearer)
            }
        }
    }
}

/// The environment mapping for one HTTP provider.
///
/// Every header is stored as a header name paired with the environment
/// variable that holds its value.
#[cfg(feature = "environment-credentials")]
#[derive(Clone, Debug, Default)]
struct HttpSpec {
    auth:          HttpAuthSpec,
    extra_headers: Vec<(String, String)>,
}

#[cfg(feature = "environment-credentials")]
#[derive(Clone, Debug, Default)]
enum HttpAuthSpec {
    #[default]
    None,
    Bearer(String),
    Header(String, String),
}

#[cfg(feature = "environment-credentials")]
impl HttpSpec {
    fn resolve(&self, provider: &ProviderId) -> Result<HttpCredentials, CredentialError> {
        let auth = match &self.auth {
            HttpAuthSpec::None => HttpAuthentication::None,
            HttpAuthSpec::Bearer(variable) => {
                HttpAuthentication::Bearer(read_secret(provider, variable)?)
            }
            HttpAuthSpec::Header(name, variable) => HttpAuthentication::Header(
                CredentialHeader::new(name.clone(), read_secret(provider, variable)?),
            ),
        };
        let extra_headers = self
            .extra_headers
            .iter()
            .map(|(name, variable)| {
                Ok(CredentialHeader::new(
                    name.clone(),
                    read_secret(provider, variable)?,
                ))
            })
            .collect::<Result<Vec<_>, CredentialError>>()?;
        Ok(HttpCredentials {
            auth,
            extra_headers,
        })
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
    /// Reads the provider's bearer secret from `variable`.
    ///
    /// A header already registered as the provider's primary authentication
    /// header becomes an extra header, so header and bearer registrations can
    /// be made in either order.
    pub fn bearer(self, provider: impl Into<ProviderId>, variable: impl Into<String>) -> Self {
        self.with_http(provider.into(), |spec| {
            if let HttpAuthSpec::Header(name, source) = mem::take(&mut spec.auth) {
                spec.extra_headers.insert(0, (name, source));
            }
            spec.auth = HttpAuthSpec::Bearer(variable.into());
        })
    }

    /// Reads one provider header from `variable`.
    ///
    /// The first header registered for a provider becomes its primary
    /// authentication header. Later headers become extra headers, as do all
    /// headers registered alongside a bearer secret.
    pub fn header(
        self,
        provider: impl Into<ProviderId>,
        name: impl Into<String>,
        variable: impl Into<String>,
    ) -> Self {
        self.with_http(provider.into(), |spec| {
            let header = (name.into(), variable.into());
            if matches!(spec.auth, HttpAuthSpec::None) {
                spec.auth = HttpAuthSpec::Header(header.0, header.1);
            } else {
                spec.extra_headers.push(header);
            }
        })
    }

    /// Reads one extra provider header from `variable`.
    ///
    /// The header is always an extra header, so it accompanies a bearer secret
    /// registered through [`Self::bearer`] rather than replacing it. Provider
    /// account identifiers such as the OpenAI organization and project headers
    /// use this method.
    pub fn bearer_header(
        self,
        provider: impl Into<ProviderId>,
        name: impl Into<String>,
        variable: impl Into<String>,
    ) -> Self {
        self.with_http(provider.into(), |spec| {
            spec.extra_headers.push((name.into(), variable.into()));
        })
    }

    /// Resolves the provider through the AWS default credential chain.
    pub fn aws_default_chain(
        mut self,
        provider: impl Into<ProviderId>,
        region: Option<String>,
    ) -> Self {
        self.specs
            .insert(provider.into(), EnvironmentSpec::AwsDefaultChain(region));
        self
    }

    /// Reads the provider's Bedrock API key from `variable`.
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

    /// Edits the provider's HTTP mapping, replacing any non-HTTP mapping.
    fn with_http(mut self, provider: ProviderId, edit: impl FnOnce(&mut HttpSpec)) -> Self {
        let mut spec = match self.specs.remove(&provider) {
            Some(EnvironmentSpec::Http(spec)) => spec,
            _ => HttpSpec::default(),
        };
        edit(&mut spec);
        self.specs.insert(provider, EnvironmentSpec::Http(spec));
        self
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
    use super::{CredentialHeader, Credentials, HttpAuthentication, HttpCredentials, SecretValue};

    #[test]
    fn debug_output_redacts_secrets() {
        let credentials = Credentials::bearer(SecretValue::new("top-secret"));
        let output = format!("{credentials:?}");
        assert!(output.contains("redacted"));
        assert!(!output.contains("top-secret"));
    }

    #[test]
    fn debug_output_shows_extra_header_names_without_values() {
        let credentials = Credentials::Http(
            HttpCredentials::new(HttpAuthentication::Bearer(SecretValue::new("top-secret")))
                .with_header(CredentialHeader::new(
                    "openai-organization",
                    SecretValue::new("org-secret"),
                )),
        );

        let output = format!("{credentials:?}");

        assert!(output.contains("openai-organization"));
        assert!(!output.contains("org-secret"));
        assert!(!output.contains("top-secret"));
    }
}
