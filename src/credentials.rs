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
///
/// Instances may serve concurrent calls. Resolve secrets lazily so each retry
/// can observe refreshed credentials. If caching or refresh is needed, the
/// implementation owns its synchronization and expiry policy. Never log secret
/// material or include it in error messages.
#[async_trait]
pub trait CredentialProvider: Send + Sync {
    /// Returns credentials matching the provider's declared authentication
    /// scheme.
    ///
    /// This future may be dropped when a call is cancelled or times out. Do not
    /// block a runtime thread or detach refresh work without an explicit owner.
    /// A failed refresh must not leave shared credential state partially
    /// updated.
    ///
    /// # Errors
    ///
    /// Return [`CredentialError`] for missing or invalid credentials or refresh
    /// failures. Preserve a useful source without exposing secrets. Return
    /// `Credentials::none()` only when no authentication is intended, not as a
    /// substitute for a failed credential lookup.
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
///
/// A provider holds an ordered chain of mappings. The first mapping that
/// resolves wins, so a provider can name a preferred variable, one or more
/// fallback variables, and a final mapping that needs no variable at all.
#[derive(Clone, Debug)]
pub struct EnvironmentCredentials {
    specs: BTreeMap<ProviderId, Vec<EnvironmentSpec>>,
}

#[cfg(feature = "environment-credentials")]
impl EnvironmentCredentials {
    pub fn builder() -> EnvironmentCredentialsBuilder {
        EnvironmentCredentialsBuilder::default()
    }

    /// Conventional environment variable mappings for built-in providers.
    ///
    /// Gemini reads `GEMINI_API_KEY` and falls back to `GOOGLE_API_KEY`.
    /// Bedrock reads `AWS_BEARER_TOKEN_BEDROCK`, the name the AWS tools use,
    /// and falls back to `BEDROCK_API_KEY`. With the `bedrock-aws` feature,
    /// Bedrock falls back once more to the AWS default credential chain, so a
    /// host with only an instance role or a profile still resolves.
    pub fn conventional() -> Self {
        let builder = EnvironmentCredentialsBuilder::default()
            .bearer("openai", "OPENAI_API_KEY")
            .header("anthropic", "x-api-key", "ANTHROPIC_API_KEY")
            .bearer("fireworks", "FIREWORKS_API_KEY")
            .header("gemini", "x-goog-api-key", "GEMINI_API_KEY")
            .or_header("gemini", "x-goog-api-key", "GOOGLE_API_KEY")
            .header("modal", "Modal-Key", "MODAL_TOKEN_ID")
            .header("modal", "Modal-Secret", "MODAL_TOKEN_SECRET")
            .bearer("moonshot", "MOONSHOT_API_KEY")
            .or_bearer("moonshot", "KIMI_API_KEY")
            .bearer("openrouter", "OPENROUTER_API_KEY")
            .bearer("venice", "VENICE_API_KEY")
            .bedrock_bearer("bedrock", "AWS_BEARER_TOKEN_BEDROCK")
            .or_bedrock_bearer("bedrock", "BEDROCK_API_KEY");
        #[cfg(feature = "bedrock-aws")]
        let builder = builder.or_aws_default_chain("bedrock", None);
        builder.build()
    }

    /// Resolves one provider's chain against `read`.
    ///
    /// The first mapping that resolves wins. When every mapping fails, the
    /// failure of the first one is reported, because that mapping names the
    /// variable the provider expects.
    fn resolve(
        &self,
        provider: &ProviderId,
        read: SecretLookup<'_>,
    ) -> Result<Credentials, CredentialError> {
        let specs = self
            .specs
            .get(provider)
            .filter(|specs| !specs.is_empty())
            .ok_or_else(|| CredentialError::NotConfigured {
                provider: provider.clone(),
            })?;
        let mut first_error = None;
        for spec in specs {
            match spec.resolve(provider, read) {
                Ok(credentials) => return Ok(credentials),
                Err(error) => {
                    first_error.get_or_insert(error);
                }
            }
        }
        Err(
            first_error.unwrap_or_else(|| CredentialError::NotConfigured {
                provider: provider.clone(),
            }),
        )
    }
}

#[cfg(feature = "environment-credentials")]
#[async_trait]
impl CredentialProvider for EnvironmentCredentials {
    async fn credentials(
        &self,
        provider: &CatalogProvider,
    ) -> Result<Credentials, CredentialError> {
        self.resolve(provider.id(), &|variable| env::var(variable))
    }
}

/// Reads one environment variable by name.
///
/// Every lookup goes through one of these, so a test can resolve a chain
/// against a fixed environment.
#[cfg(feature = "environment-credentials")]
type SecretLookup<'a> = &'a dyn Fn(&str) -> Result<String, env::VarError>;

#[cfg(feature = "environment-credentials")]
#[derive(Clone, Debug)]
enum EnvironmentSpec {
    Http(HttpSpec),
    AwsDefaultChain(Option<String>),
    BedrockBearer(String),
}

#[cfg(feature = "environment-credentials")]
impl EnvironmentSpec {
    fn resolve(
        &self,
        provider: &ProviderId,
        read: SecretLookup<'_>,
    ) -> Result<Credentials, CredentialError> {
        match self {
            Self::Http(spec) => spec.resolve(provider, read).map(Credentials::Http),
            Self::AwsDefaultChain(region) => Ok(Credentials::AwsDefaultChain {
                region: region.clone(),
            }),
            Self::BedrockBearer(variable) => {
                read_secret(provider, variable, read).map(Credentials::BedrockBearer)
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
    fn resolve(
        &self,
        provider: &ProviderId,
        read: SecretLookup<'_>,
    ) -> Result<HttpCredentials, CredentialError> {
        let auth = match &self.auth {
            HttpAuthSpec::None => HttpAuthentication::None,
            HttpAuthSpec::Bearer(variable) => {
                HttpAuthentication::Bearer(read_secret(provider, variable, read)?)
            }
            HttpAuthSpec::Header(name, variable) => HttpAuthentication::Header(
                CredentialHeader::new(name.clone(), read_secret(provider, variable, read)?),
            ),
        };
        let extra_headers = self
            .extra_headers
            .iter()
            .map(|(name, variable)| {
                Ok(CredentialHeader::new(
                    name.clone(),
                    read_secret(provider, variable, read)?,
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
fn read_secret(
    provider: &ProviderId,
    variable: &str,
    read: SecretLookup<'_>,
) -> Result<SecretValue, CredentialError> {
    read(variable)
        .map(SecretValue::new)
        .map_err(|source| CredentialError::Environment {
            provider: provider.clone(),
            variable: variable.to_owned(),
            source,
        })
}

#[cfg(feature = "environment-credentials")]
/// Builds environment-backed provider mappings.
///
/// The `bearer`, `header`, `bearer_header`, `aws_default_chain`, and
/// `bedrock_bearer` methods write the provider's primary mapping. The `or_`
/// methods append a fallback mapping, tried in the order it was added when
/// every earlier mapping finds no environment variable.
#[derive(Default)]
#[must_use]
pub struct EnvironmentCredentialsBuilder {
    specs: BTreeMap<ProviderId, Vec<EnvironmentSpec>>,
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

    /// Reads one fallback provider header from `variable`.
    ///
    /// The header is the provider's authentication header for that attempt,
    /// which is tried only when every earlier mapping finds no variable.
    pub fn or_header(
        self,
        provider: impl Into<ProviderId>,
        name: impl Into<String>,
        variable: impl Into<String>,
    ) -> Self {
        self.push(
            provider.into(),
            EnvironmentSpec::Http(HttpSpec {
                auth:          HttpAuthSpec::Header(name.into(), variable.into()),
                extra_headers: Vec::new(),
            }),
        )
    }

    /// Reads a fallback bearer secret from `variable`.
    pub fn or_bearer(self, provider: impl Into<ProviderId>, variable: impl Into<String>) -> Self {
        self.push(
            provider.into(),
            EnvironmentSpec::Http(HttpSpec {
                auth:          HttpAuthSpec::Bearer(variable.into()),
                extra_headers: Vec::new(),
            }),
        )
    }

    /// Resolves the provider through the AWS default credential chain.
    pub fn aws_default_chain(
        self,
        provider: impl Into<ProviderId>,
        region: Option<String>,
    ) -> Self {
        self.replace(provider.into(), EnvironmentSpec::AwsDefaultChain(region))
    }

    /// Falls back to the AWS default credential chain.
    ///
    /// The chain resolves from an instance role, a profile, or the AWS
    /// environment variables, so it needs no variable of its own and always
    /// ends the provider's fallbacks.
    pub fn or_aws_default_chain(
        self,
        provider: impl Into<ProviderId>,
        region: Option<String>,
    ) -> Self {
        self.push(provider.into(), EnvironmentSpec::AwsDefaultChain(region))
    }

    /// Reads the provider's Bedrock API key from `variable`.
    pub fn bedrock_bearer(
        self,
        provider: impl Into<ProviderId>,
        variable: impl Into<String>,
    ) -> Self {
        self.replace(
            provider.into(),
            EnvironmentSpec::BedrockBearer(variable.into()),
        )
    }

    /// Reads a fallback Bedrock API key from `variable`.
    pub fn or_bedrock_bearer(
        self,
        provider: impl Into<ProviderId>,
        variable: impl Into<String>,
    ) -> Self {
        self.push(
            provider.into(),
            EnvironmentSpec::BedrockBearer(variable.into()),
        )
    }

    pub fn build(self) -> EnvironmentCredentials {
        EnvironmentCredentials { specs: self.specs }
    }

    /// Edits the provider's HTTP mapping, replacing any non-HTTP mapping.
    ///
    /// The mapping edited is the primary one, so a fallback added earlier
    /// keeps its own headers and its place in the chain.
    fn with_http(mut self, provider: ProviderId, edit: impl FnOnce(&mut HttpSpec)) -> Self {
        let chain = self.specs.entry(provider).or_default();
        let mut spec = match chain.first() {
            Some(EnvironmentSpec::Http(spec)) => spec.clone(),
            _ => HttpSpec::default(),
        };
        edit(&mut spec);
        let spec = EnvironmentSpec::Http(spec);
        match chain.first_mut() {
            Some(primary) => *primary = spec,
            None => chain.push(spec),
        }
        self
    }

    /// Makes `spec` the provider's primary mapping, dropping any fallbacks.
    fn replace(mut self, provider: ProviderId, spec: EnvironmentSpec) -> Self {
        self.specs.insert(provider, vec![spec]);
        self
    }

    /// Appends `spec` to the end of the provider's chain.
    fn push(mut self, provider: ProviderId, spec: EnvironmentSpec) -> Self {
        self.specs.entry(provider).or_default().push(spec);
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
    #[cfg(feature = "environment-credentials")]
    use std::env::VarError;

    use super::{CredentialHeader, Credentials, HttpAuthentication, HttpCredentials, SecretValue};
    #[cfg(feature = "environment-credentials")]
    use super::{EnvironmentCredentials, ProviderId};

    /// Resolves one provider against a fixed environment.
    #[cfg(feature = "environment-credentials")]
    fn resolve(provider: &str, environment: &[(&str, &str)]) -> Result<Credentials, String> {
        EnvironmentCredentials::conventional()
            .resolve(&ProviderId::new(provider), &|variable| {
                environment
                    .iter()
                    .find(|(name, _)| *name == variable)
                    .map(|(_, value)| (*value).to_owned())
                    .ok_or(VarError::NotPresent)
            })
            .map_err(|error| error.to_string())
    }

    /// The secret behind a provider's primary authentication header.
    #[cfg(feature = "environment-credentials")]
    fn header_secret(credentials: &Credentials) -> Option<&str> {
        match credentials {
            Credentials::Http(http) => match &http.auth {
                HttpAuthentication::Header(header) => Some(header.value.expose_secret()),
                _ => None,
            },
            _ => None,
        }
    }

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

    #[cfg(feature = "environment-credentials")]
    #[test]
    fn gemini_prefers_its_own_variable_over_the_google_one() -> Result<(), String> {
        let preferred = resolve("gemini", &[
            ("GEMINI_API_KEY", "gemini-key"),
            ("GOOGLE_API_KEY", "google-key"),
        ])?;
        assert_eq!(header_secret(&preferred), Some("gemini-key"));

        let fallback = resolve("gemini", &[("GOOGLE_API_KEY", "google-key")])?;
        assert_eq!(header_secret(&fallback), Some("google-key"));

        // With neither variable set, the error names the preferred one.
        let message = resolve("gemini", &[]).expect_err("gemini should not resolve");
        assert!(message.contains("GEMINI_API_KEY"), "{message}");
        Ok(())
    }

    #[cfg(feature = "environment-credentials")]
    #[test]
    fn openrouter_reads_its_conventional_bearer_key() -> Result<(), String> {
        let credentials = resolve("openrouter", &[("OPENROUTER_API_KEY", "openrouter-key")])?;
        assert!(
            matches!(
                credentials,
                Credentials::Http(HttpCredentials {
                    auth: HttpAuthentication::Bearer(secret),
                    ..
                }) if secret.expose_secret() == "openrouter-key"
            ),
            "OpenRouter should resolve a bearer credential"
        );
        Ok(())
    }

    #[cfg(feature = "environment-credentials")]
    #[test]
    fn fireworks_reads_its_conventional_bearer_key() -> Result<(), String> {
        let credentials = resolve("fireworks", &[("FIREWORKS_API_KEY", "fireworks-key")])?;
        assert!(
            matches!(
                credentials,
                Credentials::Http(HttpCredentials {
                    auth: HttpAuthentication::Bearer(secret),
                    ..
                }) if secret.expose_secret() == "fireworks-key"
            ),
            "Fireworks should resolve a bearer credential"
        );
        Ok(())
    }

    #[cfg(feature = "environment-credentials")]
    #[test]
    fn moonshot_prefers_its_own_variable_over_the_kimi_one() -> Result<(), String> {
        let preferred = resolve("moonshot", &[
            ("MOONSHOT_API_KEY", "moonshot-key"),
            ("KIMI_API_KEY", "kimi-key"),
        ])?;
        assert!(
            matches!(
                preferred,
                Credentials::Http(HttpCredentials {
                    auth: HttpAuthentication::Bearer(secret),
                    ..
                }) if secret.expose_secret() == "moonshot-key"
            ),
            "Moonshot should prefer MOONSHOT_API_KEY"
        );

        let fallback = resolve("moonshot", &[("KIMI_API_KEY", "kimi-key")])?;
        assert!(
            matches!(
                fallback,
                Credentials::Http(HttpCredentials {
                    auth: HttpAuthentication::Bearer(secret),
                    ..
                }) if secret.expose_secret() == "kimi-key"
            ),
            "Moonshot should fall back to KIMI_API_KEY"
        );

        let message = resolve("moonshot", &[]).expect_err("Moonshot should not resolve");
        assert!(message.contains("MOONSHOT_API_KEY"), "{message}");
        Ok(())
    }

    #[cfg(feature = "environment-credentials")]
    #[test]
    fn modal_reads_both_conventional_proxy_token_headers() -> Result<(), String> {
        let credentials = resolve("modal", &[
            ("MODAL_TOKEN_ID", "wk-modal-key"),
            ("MODAL_TOKEN_SECRET", "ws-modal-secret"),
        ])?;
        let Credentials::Http(HttpCredentials {
            auth: HttpAuthentication::Header(key),
            extra_headers,
        }) = credentials
        else {
            return Err("Modal should resolve HTTP header credentials".to_owned());
        };
        assert_eq!(key.name, "Modal-Key");
        assert_eq!(key.value.expose_secret(), "wk-modal-key");
        assert_eq!(extra_headers.len(), 1);
        assert_eq!(extra_headers[0].name, "Modal-Secret");
        assert_eq!(extra_headers[0].value.expose_secret(), "ws-modal-secret");
        Ok(())
    }

    #[cfg(feature = "environment-credentials")]
    #[test]
    fn bedrock_prefers_the_aws_variable_then_the_lithos_one() -> Result<(), String> {
        let preferred = resolve("bedrock", &[
            ("AWS_BEARER_TOKEN_BEDROCK", "aws-key"),
            ("BEDROCK_API_KEY", "bedrock-key"),
        ])?;
        assert!(
            matches!(&preferred, Credentials::BedrockBearer(secret) if secret.expose_secret() == "aws-key")
        );

        let fallback = resolve("bedrock", &[("BEDROCK_API_KEY", "bedrock-key")])?;
        assert!(
            matches!(&fallback, Credentials::BedrockBearer(secret) if secret.expose_secret() == "bedrock-key")
        );
        Ok(())
    }

    /// Without an API key, Bedrock falls back to the AWS credential chain.
    #[cfg(all(feature = "environment-credentials", feature = "bedrock-aws"))]
    #[test]
    fn bedrock_falls_back_to_the_aws_default_chain() -> Result<(), String> {
        let credentials = resolve("bedrock", &[])?;

        assert!(matches!(credentials, Credentials::AwsDefaultChain { .. }));
        Ok(())
    }

    /// Without the AWS chain, the failure names the variable to set.
    #[cfg(all(feature = "environment-credentials", not(feature = "bedrock-aws")))]
    #[test]
    fn bedrock_reports_its_preferred_variable_without_the_aws_feature() {
        let message = resolve("bedrock", &[]).expect_err("bedrock should not resolve");

        assert!(message.contains("AWS_BEARER_TOKEN_BEDROCK"), "{message}");
    }
}
