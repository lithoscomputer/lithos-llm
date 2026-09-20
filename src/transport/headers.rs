//! Request header assembly and response header reading.
//!
//! Every header source for one request is merged here in one precedence:
//! codec headers, then the provider's catalog default headers, then the
//! credential headers, so credentials win every collision. The credential
//! headers themselves are computed as data by [`credential_headers`], one
//! exit per authentication scheme, so the rule is unit-testable without a
//! [`HeaderMap`].

use std::collections::BTreeMap;

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};

use crate::catalog::{AuthScheme, ProviderId};
use crate::credentials::{CredentialHeader, Credentials, HttpAuthentication, SecretValue};
use crate::types::{Error, ErrorKind, RateLimits};

/// Normalizes the OpenAI and Anthropic rate-limit header families.
///
/// Request resets and token resets stay separate, and both keep the provider's
/// own formatting. OpenAI sends Go durations such as `6m0s`; Anthropic sends
/// RFC 3339 instants. Callers that need a duration parse the string themselves.
pub(super) fn rate_limits(headers: &HeaderMap) -> Option<RateLimits> {
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

/// Merges every header source for one request into `headers`, later sources
/// winning.
///
/// The order is codec headers, then provider default headers from the catalog,
/// then credential headers. Credentials therefore win every collision.
/// `credentials` is `None` for a request the caller authenticates afterwards,
/// as the SigV4 signer does.
pub(super) fn merge_headers(
    headers: &mut HeaderMap,
    provider: &ProviderId,
    scheme: &AuthScheme,
    codec_headers: &[(String, String)],
    default_headers: &BTreeMap<String, String>,
    credentials: Option<&Credentials>,
) -> Result<(), Error> {
    for (name, value) in codec_headers {
        insert_header(headers, provider, name, value)?;
    }
    for (name, value) in default_headers {
        insert_header(headers, provider, name, value)?;
    }
    if let Some(credentials) = credentials {
        for header in credential_headers(provider, scheme, credentials)? {
            insert_secret_header(headers, provider, &header)?;
        }
    }
    Ok(())
}

/// The credential headers the provider's scheme accepts, in send order.
///
/// The primary authentication header comes first, then any extra headers the
/// credentials carry. Every header here is a secret and is inserted as one.
///
/// # Errors
///
/// [`ErrorKind::Authentication`] when the credentials do not match the
/// scheme, and [`ErrorKind::Configuration`] for AWS default-chain credentials,
/// which the SigV4 signer applies rather than a header.
fn credential_headers(
    provider: &ProviderId,
    scheme: &AuthScheme,
    credentials: &Credentials,
) -> Result<Vec<CredentialHeader>, Error> {
    let mut headers = Vec::new();
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
            headers.push(CredentialHeader::new(
                header,
                SecretValue::new(format!("{prefix}{}", secret.expose_secret())),
            ));
            http
        }
        (AuthScheme::Header { name }, Credentials::Http(http)) => {
            let HttpAuthentication::Header(header) = &http.auth else {
                return Err(scheme_mismatch(provider));
            };
            if !name.eq_ignore_ascii_case(&header.name) {
                return Err(scheme_mismatch(provider));
            }
            headers.push(header.clone());
            http
        }
        (AuthScheme::Headers, Credentials::Http(http)) => {
            match &http.auth {
                HttpAuthentication::None => {}
                // The scheme names no primary header, so a bearer secret uses
                // the conventional `authorization` header.
                HttpAuthentication::Bearer(secret) => headers.push(CredentialHeader::new(
                    "authorization",
                    SecretValue::new(format!("Bearer {}", secret.expose_secret())),
                )),
                HttpAuthentication::Header(header) => headers.push(header.clone()),
            }
            http
        }
        (
            AuthScheme::BedrockBearer | AuthScheme::Aws { .. },
            Credentials::BedrockBearer(secret),
        ) => {
            headers.push(CredentialHeader::new(
                "authorization",
                SecretValue::new(format!("Bearer {}", secret.expose_secret())),
            ));
            return Ok(headers);
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
    headers.extend(http.extra_headers.iter().cloned());
    Ok(headers)
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use reqwest::header::{HeaderMap, HeaderValue};

    use super::{credential_headers, merge_headers, rate_limits};
    use crate::catalog::{AuthScheme, ProviderId};
    use crate::credentials::{
        CredentialHeader, Credentials, HttpAuthentication, HttpCredentials, SecretValue,
    };
    use crate::types::{Error, ErrorKind};

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
        let mut headers = HeaderMap::new();
        merge_headers(
            &mut headers,
            &ProviderId::new("openai"),
            scheme,
            &codec,
            &defaults,
            Some(credentials),
        )?;
        Ok(headers)
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

    #[test]
    fn credential_headers_put_the_primary_header_first_then_the_extras() {
        let scheme = AuthScheme::Bearer {
            header: "authorization".to_owned(),
            prefix: "Bearer ".to_owned(),
        };
        let credentials = Credentials::Http(
            HttpCredentials::new(HttpAuthentication::Bearer(SecretValue::new("key"))).with_header(
                CredentialHeader::new("openai-project", SecretValue::new("proj-1")),
            ),
        );

        let headers = credential_headers(&ProviderId::new("openai"), &scheme, &credentials)
            .expect("bearer credentials match a bearer scheme");

        let names: Vec<&str> = headers.iter().map(|header| header.name.as_str()).collect();
        assert_eq!(names, ["authorization", "openai-project"]);
        assert_eq!(headers[0].value.expose_secret(), "Bearer key");
    }

    #[test]
    fn a_bedrock_key_is_one_bearer_header_under_either_bedrock_scheme() {
        let credentials = Credentials::BedrockBearer(SecretValue::new("abc"));
        for scheme in [AuthScheme::BedrockBearer, AuthScheme::Aws { region: None }] {
            let headers = credential_headers(&ProviderId::new("bedrock"), &scheme, &credentials)
                .expect("a Bedrock key matches both Bedrock schemes");
            assert_eq!(headers.len(), 1);
            assert_eq!(headers[0].name, "authorization");
            assert_eq!(headers[0].value.expose_secret(), "Bearer abc");
        }
    }
}
