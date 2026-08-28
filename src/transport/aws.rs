//! AWS SigV4 signing and region resolution for Amazon Bedrock.
//!
//! The credential chain is loaded on the first signed request, never at client
//! construction, and what is cached is the chain itself rather than one set of
//! credentials. Temporary credentials from STS, SSO, or an instance role can
//! therefore refresh through the chain's own identity cache.

use std::time::SystemTime;

use aws_config::{BehaviorVersion, SdkConfig};
use aws_credential_types::Credentials;
use aws_credential_types::provider::ProvideCredentials as _;
use aws_sigv4::http_request::{
    SignableBody, SignableRequest, SigningParams, SigningSettings, sign,
};
use aws_sigv4::sign::v4;
use aws_smithy_runtime_api::client::identity::Identity;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use reqwest::{Method, Url};
use tokio::sync::OnceCell;

use crate::catalog::{AuthScheme, ProviderId};
use crate::types::{Error, ErrorKind};

/// The SigV4 signing name for Bedrock. It is `bedrock`, not `bedrock-runtime`.
const SERVICE: &str = "bedrock";

/// Headers the HTTP client sets or rewrites, which are never signed.
///
/// `aws-sigv4` already excludes `authorization`, `user-agent`,
/// `x-amzn-trace-id`, and `transfer-encoding` on its own.
const CLIENT_OWNED_HEADERS: [&str; 2] = ["accept-encoding", "content-length"];

/// Signs Bedrock requests with AWS SigV4.
pub(crate) struct AwsSigner {
    provider: ProviderId,
    /// The loaded AWS configuration, holding the credential provider chain.
    config:   OnceCell<SdkConfig>,
}

impl AwsSigner {
    pub(crate) fn new(provider: ProviderId) -> Self {
        Self {
            provider,
            config: OnceCell::new(),
        }
    }

    /// Signs one request and writes the resulting headers into `headers`.
    ///
    /// Pass the method, URL, headers, and body exactly as they will be sent.
    /// A header added after this call is unsigned, which SigV4 permits, but a
    /// signed header that later changes makes the request fail.
    pub(crate) async fn sign(
        &self,
        region: &str,
        method: &Method,
        url: &str,
        headers: &mut HeaderMap,
        body: &[u8],
    ) -> Result<(), Error> {
        let credentials = self.credentials().await?;
        let signed = signed_headers(
            &credentials,
            region,
            method.as_str(),
            url,
            headers,
            body,
            SystemTime::now(),
            &self.provider,
        )?;
        for (name, value, sensitive) in signed {
            let name = HeaderName::from_bytes(name.as_bytes()).map_err(|source| {
                self.configuration_error(format!("AWS signing produced the invalid header {name}"))
                    .with_source(source)
            })?;
            let mut value = HeaderValue::from_str(&value).map_err(|source| {
                self.configuration_error(format!(
                    "AWS signing produced an invalid value for the {name} header"
                ))
                .with_source(source)
            })?;
            value.set_sensitive(sensitive);
            headers.insert(name, value);
        }
        Ok(())
    }

    /// Resolves the signing region from the first source that supplies one.
    ///
    /// The order is the credential configuration, then the catalog
    /// authentication configuration, then a standard Bedrock endpoint
    /// hostname, then the AWS region provider chain. The chain is consulted
    /// last because reaching it can load configuration files and query the
    /// instance metadata service.
    pub(crate) async fn resolve_region(
        &self,
        credential_region: Option<&str>,
        auth: &AuthScheme,
        base_url: &str,
    ) -> Result<String, Error> {
        let catalog_region = match auth {
            AuthScheme::Aws { region } => region.as_deref(),
            _ => None,
        };
        if let Some(region) = credential_region.or(catalog_region) {
            return Ok(region.to_owned());
        }
        if let Some(region) = Url::parse(base_url)
            .ok()
            .as_ref()
            .and_then(Url::host_str)
            .and_then(region_from_host)
        {
            return Ok(region);
        }
        self.config()
            .await
            .region()
            .map(|region| region.as_ref().to_owned())
            .ok_or_else(|| {
                self.configuration_error(
                    "no AWS region for Bedrock: set one on the credentials or the catalog \
                     provider, use a standard Bedrock endpoint, or configure the AWS region \
                     provider chain",
                )
            })
    }

    /// Loads the AWS configuration once, on first use.
    async fn config(&self) -> &SdkConfig {
        self.config
            .get_or_init(|| aws_config::defaults(BehaviorVersion::latest()).load())
            .await
    }

    /// Asks the cached provider chain for current credentials.
    async fn credentials(&self) -> Result<Credentials, Error> {
        let provider = self.config().await.credentials_provider().ok_or_else(|| {
            Error::new(
                ErrorKind::Authentication,
                "the AWS credential provider chain supplied no credentials provider",
            )
            .with_provider(self.provider.clone())
        })?;
        provider.provide_credentials().await.map_err(|source| {
            Error::new(
                ErrorKind::Authentication,
                "the AWS credential provider chain could not supply credentials",
            )
            .with_provider(self.provider.clone())
            .with_source(source)
        })
    }

    fn configuration_error(&self, message: impl Into<String>) -> Error {
        Error::new(ErrorKind::Configuration, message).with_provider(self.provider.clone())
    }
}

/// The AWS region named by a standard Bedrock runtime hostname.
///
/// A custom host, such as a gateway or a VPC endpoint, returns `None` so that
/// region resolution falls through to the next source.
pub(crate) fn region_from_host(host: &str) -> Option<String> {
    let rest = host
        .strip_prefix("bedrock-runtime-fips.")
        .or_else(|| host.strip_prefix("bedrock-runtime."))?;
    let region = rest
        .strip_suffix(".amazonaws.com.cn")
        .or_else(|| rest.strip_suffix(".amazonaws.com"))?;
    let plausible = !region.is_empty()
        && region
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-');
    plausible.then(|| region.to_owned())
}

/// Calculates the SigV4 headers for one request.
///
/// The returned triples are the header name, its value, and whether the value
/// is sensitive. `time` is a parameter so that tests can sign at a fixed
/// instant and assert the exact signature.
fn signed_headers(
    credentials: &Credentials,
    region: &str,
    method: &str,
    url: &str,
    headers: &HeaderMap,
    body: &[u8],
    time: SystemTime,
    provider: &ProviderId,
) -> Result<Vec<(&'static str, String, bool)>, Error> {
    let failed = |message: &str| {
        let message = format!("AWS SigV4 signing failed: {message}");
        Error::new(ErrorKind::Configuration, message).with_provider(provider.clone())
    };

    let identity: Identity = credentials.clone().into();
    let params: SigningParams<'_> = v4::SigningParams::builder()
        .identity(&identity)
        .region(region)
        .name(SERVICE)
        .time(time)
        .settings(SigningSettings::default())
        .build()
        .map_err(|source| failed("the signing parameters are incomplete").with_source(source))?
        .into();

    // Only headers that reach the wire byte for byte can be signed. Leaving a
    // header out of the canonical request is always safe, because SigV4 covers
    // exactly the names listed in `SignedHeaders`; signing one that reqwest
    // then rewrites would break the request. Two are dropped for that reason,
    // and a value that is not UTF-8 cannot be signed at all. `host` is not
    // dropped: aws-sigv4 always signs it, taken from the URL when absent here,
    // and reqwest sends the same value.
    let signable: Vec<_> = headers
        .iter()
        .filter(|(name, _)| !CLIENT_OWNED_HEADERS.contains(&name.as_str()))
        .filter_map(|(name, value)| Some((name.as_str(), value.to_str().ok()?)))
        .collect();
    let request =
        SignableRequest::new(method, url, signable.into_iter(), SignableBody::Bytes(body))
            .map_err(|source| {
                failed("the request could not be made signable").with_source(source)
            })?;

    let (instructions, _signature) = sign(request, &params)
        .map_err(|source| failed("the signature could not be calculated").with_source(source))?
        .into_parts();
    let (instruction_headers, _params) = instructions.into_parts();
    Ok(instruction_headers
        .into_iter()
        .map(|header| {
            let sensitive = header.sensitive();
            (header.name(), header.value().to_owned(), sensitive)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, UNIX_EPOCH};

    use aws_credential_types::Credentials;
    use reqwest::header::{HeaderMap, HeaderValue};

    use super::{region_from_host, signed_headers};
    use crate::catalog::ProviderId;

    const URL: &str = "https://bedrock-runtime.us-east-1.amazonaws.com/model/test-model/converse";
    const BODY: &[u8] = br#"{"a":1}"#;
    /// 2024-05-30T00:00:00Z, so the credential scope and date are fixed.
    const SIGNING_EPOCH_SECONDS: u64 = 1_717_027_200;

    fn credentials(session_token: Option<&str>) -> Credentials {
        Credentials::new(
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            session_token.map(str::to_owned),
            None,
            "lithos-llm-test",
        )
    }

    fn json_headers() -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert("content-type", HeaderValue::from_static("application/json"));
        headers
    }

    /// Signs at the fixed instant and returns the headers keyed by name.
    fn sign_fixed(
        credentials: &Credentials,
        headers: &HeaderMap,
        body: &[u8],
    ) -> Vec<(&'static str, String)> {
        let signed = signed_headers(
            credentials,
            "us-east-1",
            "POST",
            URL,
            headers,
            body,
            UNIX_EPOCH + Duration::from_secs(SIGNING_EPOCH_SECONDS),
            &ProviderId::new("bedrock"),
        )
        .expect("fixed credentials and a fixed time sign successfully");
        signed
            .into_iter()
            .map(|(name, value, _sensitive)| (name, value))
            .collect()
    }

    fn header<'a>(headers: &'a [(&'static str, String)], name: &str) -> Option<&'a str> {
        headers
            .iter()
            .find(|(header, _)| *header == name)
            .map(|(_, value)| value.as_str())
    }

    fn authorization<'a>(headers: &'a [(&'static str, String)]) -> &'a str {
        header(headers, "authorization").expect("signing emits an authorization header")
    }

    #[test]
    fn reads_the_region_from_a_standard_host() {
        assert_eq!(
            region_from_host("bedrock-runtime.eu-west-1.amazonaws.com").as_deref(),
            Some("eu-west-1")
        );
    }

    #[test]
    fn reads_the_region_from_a_fips_host() {
        assert_eq!(
            region_from_host("bedrock-runtime-fips.us-gov-west-1.amazonaws.com").as_deref(),
            Some("us-gov-west-1")
        );
    }

    #[test]
    fn reads_the_region_from_a_china_host() {
        assert_eq!(
            region_from_host("bedrock-runtime.cn-north-1.amazonaws.com.cn").as_deref(),
            Some("cn-north-1")
        );
    }

    #[test]
    fn a_custom_host_names_no_region() {
        for host in [
            "example.com",
            "localhost",
            "bedrock.us-east-1.amazonaws.com",
            "bedrock-runtime.amazonaws.com",
            "bedrock-runtime..amazonaws.com",
            "bedrock-runtime.US-EAST-1.amazonaws.com",
        ] {
            assert_eq!(region_from_host(host), None, "{host} is not a Bedrock host");
        }
    }

    #[test]
    fn signs_with_fixed_credentials_and_time() {
        let headers = sign_fixed(&credentials(None), &json_headers(), BODY);

        assert_eq!(header(&headers, "x-amz-date"), Some("20240530T000000Z"));
        assert_eq!(
            authorization(&headers),
            "AWS4-HMAC-SHA256 \
             Credential=AKIDEXAMPLE/20240530/us-east-1/bedrock/aws4_request, \
             SignedHeaders=content-type;host;x-amz-date, \
             Signature=37dfcda7002bc1eb0c65475bc88dae14be24869ae13f89d73f92ded0efac06cc"
        );
        assert_eq!(header(&headers, "x-amz-security-token"), None);
    }

    #[test]
    fn a_session_token_is_sent_and_signed() {
        let plain = sign_fixed(&credentials(None), &json_headers(), BODY);
        let temporary = sign_fixed(
            &credentials(Some("session-token-value")),
            &json_headers(),
            BODY,
        );

        assert_eq!(
            header(&temporary, "x-amz-security-token"),
            Some("session-token-value")
        );
        assert!(
            authorization(&temporary)
                .contains("SignedHeaders=content-type;host;x-amz-date;x-amz-security-token"),
            "the token is covered by the signature"
        );
        assert_ne!(authorization(&plain), authorization(&temporary));
    }

    #[test]
    fn headers_the_http_client_owns_are_not_signed() {
        let credentials = credentials(None);
        let plain = sign_fixed(&credentials, &json_headers(), BODY);

        let mut client_owned = json_headers();
        client_owned.insert("content-length", HeaderValue::from_static("7"));
        client_owned.insert("accept-encoding", HeaderValue::from_static("gzip"));
        let signed = sign_fixed(&credentials, &client_owned, BODY);

        assert_eq!(
            authorization(&plain),
            authorization(&signed),
            "a header reqwest rewrites cannot join the signed set"
        );
    }

    #[test]
    fn a_different_body_signs_differently() {
        let credentials = credentials(None);
        let original = sign_fixed(&credentials, &json_headers(), BODY);
        let altered = sign_fixed(&credentials, &json_headers(), br#"{"a":2}"#);

        assert_ne!(authorization(&original), authorization(&altered));
    }

    #[test]
    fn a_different_signed_header_signs_differently() {
        let credentials = credentials(None);
        let original = sign_fixed(&credentials, &json_headers(), BODY);

        let mut extra = json_headers();
        extra.insert("x-lithos", HeaderValue::from_static("probe"));
        let altered = sign_fixed(&credentials, &extra, BODY);

        assert!(
            authorization(&altered).contains("SignedHeaders=content-type;host;x-amz-date;x-lithos"),
            "the added header joins the signed set"
        );
        assert_ne!(authorization(&original), authorization(&altered));
    }
}
