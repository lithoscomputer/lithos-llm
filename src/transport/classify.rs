//! One provider-failure classifier.
//!
//! HTTP error responses and mid-stream error payloads both go through
//! [`extract`] and [`classify`], so a code such as `insufficient_quota` means
//! the same thing on either path. Mid-stream payloads simply pass
//! `status = None`.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::types::{ErrorKind, RetryClassification};

/// A normalized provider failure, ready to become an [`crate::types::Error`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProviderFailure {
    /// The normalized category.
    pub kind:        ErrorKind,
    /// Whether repeating the same resolved call is safe.
    pub retry:       RetryClassification,
    /// The provider's advised wait, kept whatever the kind.
    pub retry_after: Option<Duration>,
    /// The provider's human-readable message, when the body carried one.
    pub message:     Option<String>,
    /// The provider's stable error code, when the body carried one.
    pub code:        Option<String>,
}

/// Codes that report spent credit, a billing cap, or an exhausted plan quota.
///
/// These are distinct from [`RATE_LIMIT_CODES`]: backoff never clears them.
const QUOTA_CODES: &[&str] = &[
    "insufficient_quota",               // OpenAI
    "billing_hard_limit_reached",       // OpenAI
    "exceeded_current_quota_error",     // Moonshot / Kimi
    "service_quota_exceeded_exception", // Bedrock ServiceQuotaExceededException
    "quota_exceeded",                   // generic
];

/// Codes that report temporary throttling, which backoff clears.
const RATE_LIMIT_CODES: &[&str] = &[
    "rate_limit_error",         // Anthropic
    "rate_limit_exceeded",      // OpenAI
    "rate_limit_reached_error", // OpenAI-compatible
    "too_many_requests",
    "throttling_exception", // Bedrock ThrottlingException
    "resource_exhausted",   // Gemini gRPC RESOURCE_EXHAUSTED
];

/// Codes that report a missing, malformed, or rejected credential.
const AUTHENTICATION_CODES: &[&str] = &[
    "authentication_error",          // Anthropic
    "invalid_api_key",               // OpenAI
    "invalid_authentication",        // OpenAI
    "unauthenticated",               // Gemini gRPC UNAUTHENTICATED
    "unrecognized_client_exception", // Bedrock
    "expired_token_exception",       // Bedrock
];

/// Codes that report a valid credential blocked by account or policy state.
const ACCESS_DENIED_CODES: &[&str] = &[
    "access_denied",
    "access_denied_exception", // Bedrock AccessDeniedException
    "account_deactivated",     // OpenAI
    "permission_denied",       // Gemini gRPC PERMISSION_DENIED
    "permission_error",        // Anthropic
    "precondition_failed",     // Fireworks account suspension
];

/// Codes that report a missing model, deployment, or route.
const NOT_FOUND_CODES: &[&str] = &[
    "not_found_error",              // Anthropic
    "not_found",                    // Gemini gRPC NOT_FOUND
    "resource_not_found_exception", // Bedrock ResourceNotFoundException
];

/// Codes that report input larger than the model's context window.
const CONTEXT_LENGTH_CODES: &[&str] = &[
    "context_length_exceeded", // OpenAI
    "request_too_large",       // Anthropic oversized input
    "prompt_too_long",         // Anthropic oversized prompt
];

/// Codes that report blocked content or a model refusal.
///
/// `refusal` is the marker a codec sets when a successful response finished
/// with a refusal stop reason.
const CONTENT_FILTER_CODES: &[&str] = &[
    "content_filter",
    "content_policy_violation",
    "content_filtered",
    "refusal",
];

/// Codes that report a provider-side failure.
const SERVER_CODES: &[&str] = &[
    "server_error",
    "internal_error",
    "internal",    // Gemini gRPC INTERNAL
    "unavailable", // Gemini gRPC UNAVAILABLE
    "service_unavailable",
    "service_unavailable_exception", // Bedrock
    "model_not_ready_exception",     // Bedrock
    "model_stream_error_exception",  // Bedrock mid-stream
    "engine_overloaded",
    "overloaded_error", // Anthropic
];

/// Codes that report a deterministic problem with the request.
const INVALID_REQUEST_CODES: &[&str] = &[
    "invalid_request_error", // OpenAI, Anthropic
    "invalid_argument",      // Gemini gRPC INVALID_ARGUMENT
    "validation_exception",  // Bedrock ValidationException
];

/// Codes that report an expired deadline.
const TIMEOUT_CODES: &[&str] = &["deadline_exceeded"]; // Gemini gRPC DEADLINE_EXCEEDED

/// Message fragments that mean credit or quota is spent rather than throttled.
///
/// These are deliberately narrow. Provider throttling messages routinely say
/// "quota", so the word alone cannot separate the two conditions.
const SPENT_QUOTA_MESSAGES: &[&str] = &[
    "billing details",
    "billing hard limit",
    "credit balance",
    "insufficient credit",
    "out of credits",
    "purchase credits",
    "spending limit",
];

/// Message fragments that mean the model or route is missing.
const NOT_FOUND_MESSAGES: &[&str] = &["not found", "does not exist"];

/// Message fragments that mean the credential was rejected.
const AUTHENTICATION_MESSAGES: &[&str] = &["unauthorized", "invalid key", "invalid api key"];

/// Message fragments that mean throttling.
const RATE_LIMIT_MESSAGES: &[&str] = &["rate limit", "too many requests"];

/// Message fragments that mean the input exceeded the context window.
const CONTEXT_LENGTH_MESSAGES: &[&str] = &[
    "context length",
    "context window",
    "maximum context",
    "too many tokens",
    "prompt is too long",
];

/// Message fragments that mean blocked content.
const CONTENT_FILTER_MESSAGES: &[&str] = &["content filter", "content policy", "safety"];

/// Extracts the message and the stable code from a provider error body.
///
/// Handles the shapes this crate's codecs meet: OpenAI and OpenAI-compatible
/// `{"error":{"message","code","type"}}`, Anthropic
/// `{"type":"error","error":{"type","message"}}`, Gemini
/// `{"error":{"code","message","status"}}`, and the Bedrock envelopes that put
/// the message at `message` or `Message` and the code in `__type`.
///
/// Returns `(None, None)` for a body that carries neither.
pub(crate) fn extract(body: Option<&Value>) -> (Option<String>, Option<String>) {
    let Some(body) = body else {
        return (None, None);
    };
    let error = body.get("error");

    let message = error
        .and_then(|error| text(error, "message"))
        .or_else(|| text(body, "message")) // Bedrock SigV4
        .or_else(|| text(body, "Message")) // Bedrock API key
        .or_else(|| text(body, "detail")) // OpenAI Codex endpoint
        .or_else(|| non_empty(body.as_str()));

    let code = error
        .and_then(|error| text(error, "code")) // OpenAI, Fireworks
        .or_else(|| error.and_then(|error| text(error, "type"))) // OpenAI, Anthropic
        .or_else(|| error.and_then(|error| text(error, "status"))) // Gemini gRPC status
        .or_else(|| text(body, "__type")) // Bedrock
        .or_else(|| text(body, "code"))
        // A bare stream error payload carries its own code. Anthropic's
        // envelope tags itself `"type": "error"`, which says nothing.
        .or_else(|| text(body, "type").filter(|value| value != "error"))
        .map(code_tail);

    (message, code)
}

/// Classifies one provider failure.
///
/// `status` is the HTTP status when the failure arrived as an HTTP response,
/// and `None` for a mid-stream error payload or a Gemini gRPC status.
/// `retry_after` is the raw `Retry-After` header value.
pub(crate) fn classify(
    status: Option<u16>,
    code: Option<&str>,
    message: Option<&str>,
    retry_after: Option<&str>,
) -> ProviderFailure {
    let code_kind = code.and_then(kind_from_code);
    let message_kind = message.and_then(kind_from_message);

    let kind = match status {
        Some(401) => ErrorKind::Authentication,
        // 412 is never a conditional-request failure here; providers use it to
        // report a suspended or unpaid account.
        Some(403 | 412) => ErrorKind::AccessDenied,
        Some(404) => ErrorKind::NotFound,
        Some(408) => ErrorKind::Timeout,
        Some(413) => ErrorKind::ContextLength,
        Some(429) => ErrorKind::RateLimit,
        Some(400 | 422) => code_kind
            .or(message_kind)
            .unwrap_or(ErrorKind::InvalidRequest),
        Some(status) if status >= 500 => ErrorKind::Server,
        Some(_) => code_kind.or(message_kind).unwrap_or(ErrorKind::Provider),
        // Mid-stream failures carry no status. An unrecognized one is
        // provider-side and worth another attempt.
        None => code_kind.or(message_kind).unwrap_or(ErrorKind::Server),
    };

    // Throttling whose body reports spent credit or quota is not temporary.
    let kind = if kind == ErrorKind::RateLimit
        && (code_kind == Some(ErrorKind::QuotaExceeded) || names_spent_quota(message))
    {
        ErrorKind::QuotaExceeded
    } else {
        kind
    };

    // The advised wait is kept on every kind: same-call retries honor it only
    // for the retryable kinds below, but an application scheduling its own
    // failover wants the hint on a spent-quota 429 too.
    let advised = retry_after.and_then(parse_retry_after);
    let retry = match kind {
        ErrorKind::RateLimit | ErrorKind::Server => {
            advised.map_or(RetryClassification::Safe, RetryClassification::after)
        }
        _ => RetryClassification::Never,
    };

    ProviderFailure {
        kind,
        retry,
        retry_after: advised,
        message: message.map(ToOwned::to_owned),
        code: code.map(ToOwned::to_owned),
    }
}

/// Parses a `Retry-After` value.
///
/// Accepts a number of seconds, which providers sometimes send with a
/// fractional part, and the RFC 7231 IMF-fixdate form
/// `Wed, 21 Oct 2015 07:28:00 GMT`. A date in the past and any other text
/// return `None`.
pub(crate) fn parse_retry_after(value: &str) -> Option<Duration> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    if let Ok(seconds) = value.parse::<f64>() {
        return Duration::try_from_secs_f64(seconds).ok();
    }

    let target = parse_http_date(value)?;
    let now = i64::try_from(SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs()).ok()?;
    let remaining = target.checked_sub(now)?;
    u64::try_from(remaining)
        .ok()
        .filter(|remaining| *remaining > 0)
        .map(Duration::from_secs)
}

/// The normalized category for a provider error code, when it has one.
///
/// `None` means the code adds nothing, so the caller keeps its own default.
fn kind_from_code(code: &str) -> Option<ErrorKind> {
    let code = normalize(code);
    let code = code.as_str();

    // Exact spellings come first: `invalid_api_key` is an authentication
    // failure, not an instance of the `invalid_` prefix rule below, and
    // `request_too_large` is oversized input, not the `_too_large` rule.
    if contains(QUOTA_CODES, code) || code.ends_with("_quota_exceeded") {
        return Some(ErrorKind::QuotaExceeded);
    }
    if contains(RATE_LIMIT_CODES, code) {
        return Some(ErrorKind::RateLimit);
    }
    if contains(AUTHENTICATION_CODES, code) {
        return Some(ErrorKind::Authentication);
    }
    if contains(ACCESS_DENIED_CODES, code) {
        return Some(ErrorKind::AccessDenied);
    }
    if contains(CONTEXT_LENGTH_CODES, code) {
        return Some(ErrorKind::ContextLength);
    }
    if contains(CONTENT_FILTER_CODES, code) {
        return Some(ErrorKind::ContentFilter);
    }
    if contains(SERVER_CODES, code) {
        return Some(ErrorKind::Server);
    }
    if contains(TIMEOUT_CODES, code) {
        return Some(ErrorKind::Timeout);
    }
    if contains(NOT_FOUND_CODES, code) || code.ends_with("_not_found") {
        return Some(ErrorKind::NotFound);
    }
    if contains(INVALID_REQUEST_CODES, code)
        || code.starts_with("invalid_")
        || code.starts_with("unsupported_")
        || code.ends_with("_too_large")
        || code.ends_with("_too_long")
    {
        return Some(ErrorKind::InvalidRequest);
    }
    None
}

/// The normalized category a provider message names, when it names one.
fn kind_from_message(message: &str) -> Option<ErrorKind> {
    let message = message.to_lowercase();
    let message = message.as_str();

    if contains_any(NOT_FOUND_MESSAGES, message) {
        Some(ErrorKind::NotFound)
    } else if contains_any(AUTHENTICATION_MESSAGES, message) {
        Some(ErrorKind::Authentication)
    } else if contains_any(RATE_LIMIT_MESSAGES, message) {
        Some(ErrorKind::RateLimit)
    } else if contains_any(CONTEXT_LENGTH_MESSAGES, message) {
        Some(ErrorKind::ContextLength)
    } else if contains_any(CONTENT_FILTER_MESSAGES, message) {
        Some(ErrorKind::ContentFilter)
    } else {
        None
    }
}

/// Whether a message reports spent credit or quota rather than throttling.
fn names_spent_quota(message: Option<&str>) -> bool {
    message.is_some_and(|message| contains_any(SPENT_QUOTA_MESSAGES, &message.to_lowercase()))
}

/// Normalizes a wire code to lower snake case.
///
/// Bedrock reports `prefix#ThrottlingException` and Gemini reports
/// `RESOURCE_EXHAUSTED`; both become `throttling_exception` and
/// `resource_exhausted`, so one table covers every provider spelling.
fn normalize(code: &str) -> String {
    let code = code.rsplit('#').next().unwrap_or(code);
    let characters: Vec<char> = code.chars().collect();
    let mut normalized = String::with_capacity(code.len() + 4);

    for (index, character) in characters.iter().enumerate() {
        let previous = index.checked_sub(1).and_then(|index| characters.get(index));
        let next = characters.get(index + 1);
        let boundary = character.is_ascii_uppercase()
            && previous.is_some_and(|previous| {
                previous.is_ascii_lowercase()
                    || previous.is_ascii_digit()
                    || (previous.is_ascii_uppercase() && next.is_some_and(char::is_ascii_lowercase))
            });

        if boundary {
            normalized.push('_');
        }
        if *character == '-' || *character == ' ' {
            normalized.push('_');
        } else {
            normalized.extend(character.to_lowercase());
        }
    }
    normalized
}

/// Keeps the tail of an ARN-shaped code such as `prefix#ThrottlingException`.
fn code_tail(code: String) -> String {
    match code.rsplit_once('#') {
        Some((_, tail)) if !tail.is_empty() => tail.to_owned(),
        _ => code,
    }
}

fn contains(codes: &[&str], code: &str) -> bool {
    codes.contains(&code)
}

fn contains_any(fragments: &[&str], message: &str) -> bool {
    fragments.iter().any(|fragment| message.contains(fragment))
}

/// A non-empty string field of a JSON object.
fn text(value: &Value, field: &str) -> Option<String> {
    non_empty(value.get(field).and_then(Value::as_str))
}

fn non_empty(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

/// Parses an IMF-fixdate into seconds since the Unix epoch.
fn parse_http_date(value: &str) -> Option<i64> {
    // `Wed, 21 Oct 2015 07:28:00 GMT` is fixed width, so byte offsets are
    // stable once the value is known to be ASCII.
    if !value.is_ascii() || value.len() != 29 {
        return None;
    }
    let bytes = value.as_bytes();
    if bytes[3] != b',' || bytes[4] != b' ' || bytes[7] != b' ' || bytes[11] != b' ' {
        return None;
    }
    if bytes[16] != b' ' || bytes[19] != b':' || bytes[22] != b':' || &value[25..] != " GMT" {
        return None;
    }

    let day = number(&value[5..7])?;
    let month = month_number(&value[8..11])?;
    let year = number(&value[12..16])?;
    let hour = number(&value[17..19])?;
    let minute = number(&value[20..22])?;
    let second = number(&value[23..25])?;

    if !(1..=31).contains(&day) || year < 1970 || hour > 23 || minute > 59 || second > 60 {
        return None;
    }

    Some(days_from_civil(year, month, day) * 86_400 + hour * 3_600 + minute * 60 + second)
}

/// Parses a fixed-width run of ASCII digits.
fn number(text: &str) -> Option<i64> {
    if text.is_empty() || !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    text.parse().ok()
}

fn month_number(name: &str) -> Option<i64> {
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];

    MONTHS
        .iter()
        .position(|month| *month == name)
        .and_then(|index| i64::try_from(index + 1).ok())
}

/// Days between the Unix epoch and a proleptic Gregorian date.
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let shifted_month = if month > 2 { month - 3 } else { month + 9 };
    let day_of_year = (153 * shifted_month + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;

    era * 146_097 + day_of_era - 719_468
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn kinds(failure: &ProviderFailure) -> (ErrorKind, RetryClassification) {
        (failure.kind, failure.retry)
    }

    #[test]
    fn classifies_http_and_stream_forms_of_one_code_identically() {
        // (code, the HTTP status that carries it)
        let cases = [
            ("insufficient_quota", 429),
            ("rate_limit_error", 429),
            ("content_policy_violation", 400),
            ("server_error", 500),
            ("model_not_found", 404),
            ("authentication_error", 401),
        ];

        for (code, status) in cases {
            let http = classify(Some(status), Some(code), None, None);
            let stream = classify(None, Some(code), None, None);

            assert_eq!(
                kinds(&http),
                kinds(&stream),
                "{code} classified differently"
            );
        }
    }

    #[test]
    fn separates_spent_quota_from_temporary_throttling() {
        for code in [
            "insufficient_quota",
            "billing_hard_limit_reached",
            "exceeded_current_quota_error",
        ] {
            let failure = classify(Some(429), Some(code), None, Some("30"));

            assert_eq!(failure.kind, ErrorKind::QuotaExceeded, "{code}");
            assert_eq!(failure.retry, RetryClassification::Never, "{code}");
        }

        for code in [
            "rate_limit_error",
            "rate_limit_reached_error",
            "invalid_request_error",
        ] {
            let failure = classify(Some(429), Some(code), None, None);

            assert_eq!(failure.kind, ErrorKind::RateLimit, "{code}");
            assert_eq!(failure.retry, RetryClassification::Safe, "{code}");
        }

        let bare = classify(Some(429), None, None, None);
        assert_eq!(
            kinds(&bare),
            (ErrorKind::RateLimit, RetryClassification::Safe)
        );
    }

    #[test]
    fn reads_a_spent_quota_from_the_message_when_the_code_is_absent() {
        let failure = classify(
            Some(429),
            None,
            Some("You exceeded your current quota, please check your plan and billing details."),
            None,
        );

        assert_eq!(
            kinds(&failure),
            (ErrorKind::QuotaExceeded, RetryClassification::Never)
        );

        let throttled = classify(
            Some(429),
            None,
            Some("Quota exceeded for quota metric 'requests per minute'"),
            None,
        );

        assert_eq!(
            kinds(&throttled),
            (ErrorKind::RateLimit, RetryClassification::Safe)
        );
    }

    #[test]
    fn uses_a_dedicated_kind_for_each_failure_family() {
        let cases = [
            (Some(400), Some("content_filter"), ErrorKind::ContentFilter),
            (None, Some("refusal"), ErrorKind::ContentFilter),
            (Some(404), Some("model_not_found"), ErrorKind::NotFound),
            (Some(400), Some("not_found_error"), ErrorKind::NotFound),
            (Some(401), None, ErrorKind::Authentication),
            (Some(403), None, ErrorKind::AccessDenied),
            (
                Some(412),
                Some("PRECONDITION_FAILED"),
                ErrorKind::AccessDenied,
            ),
            (Some(500), None, ErrorKind::Server),
            (Some(529), None, ErrorKind::Server),
            (
                Some(400),
                Some("context_length_exceeded"),
                ErrorKind::ContextLength,
            ),
            (Some(413), None, ErrorKind::ContextLength),
            (Some(408), None, ErrorKind::Timeout),
            (Some(400), None, ErrorKind::InvalidRequest),
            (Some(409), None, ErrorKind::Provider),
        ];

        for (status, code, expected) in cases {
            let failure = classify(status, code, None, None);

            assert_eq!(failure.kind, expected, "status {status:?} code {code:?}");
        }
    }

    #[test]
    fn treats_content_filters_and_refusals_as_final() {
        for code in ["content_filter", "content_policy_violation", "refusal"] {
            let failure = classify(None, Some(code), None, None);

            assert_eq!(
                kinds(&failure),
                (ErrorKind::ContentFilter, RetryClassification::Never)
            );
        }
    }

    #[test]
    fn classifies_bedrock_exception_names() {
        let cases = [
            ("com.amazon.coral#ThrottlingException", ErrorKind::RateLimit),
            ("ValidationException", ErrorKind::InvalidRequest),
            ("AccessDeniedException", ErrorKind::AccessDenied),
            ("ResourceNotFoundException", ErrorKind::NotFound),
            ("ServiceUnavailableException", ErrorKind::Server),
            ("ModelStreamErrorException", ErrorKind::Server),
            ("ServiceQuotaExceededException", ErrorKind::QuotaExceeded),
        ];

        for (code, expected) in cases {
            assert_eq!(
                classify(None, Some(code), None, None).kind,
                expected,
                "{code}"
            );
        }
    }

    #[test]
    fn maps_gemini_grpc_statuses_to_normalized_categories() {
        let cases = [
            ("UNAUTHENTICATED", ErrorKind::Authentication),
            ("PERMISSION_DENIED", ErrorKind::AccessDenied),
            ("NOT_FOUND", ErrorKind::NotFound),
            ("RESOURCE_EXHAUSTED", ErrorKind::RateLimit),
            ("INVALID_ARGUMENT", ErrorKind::InvalidRequest),
            ("UNAVAILABLE", ErrorKind::Server),
            ("INTERNAL", ErrorKind::Server),
            ("DEADLINE_EXCEEDED", ErrorKind::Timeout),
        ];

        for (status, expected) in cases {
            assert_eq!(
                classify(None, Some(status), None, None).kind,
                expected,
                "{status}"
            );
        }

        // A spent Gemini quota still reports RESOURCE_EXHAUSTED.
        let spent = classify(
            None,
            Some("RESOURCE_EXHAUSTED"),
            Some("Your credit balance is too low to run this model"),
            None,
        );

        assert_eq!(
            kinds(&spent),
            (ErrorKind::QuotaExceeded, RetryClassification::Never)
        );
    }

    #[test]
    fn carries_the_message_and_code_into_the_failure() {
        let failure = classify(
            Some(429),
            Some("rate_limit_error"),
            Some("slow down"),
            Some("2"),
        );

        assert_eq!(failure.code.as_deref(), Some("rate_limit_error"));
        assert_eq!(failure.message.as_deref(), Some("slow down"));
        assert_eq!(
            failure.retry,
            RetryClassification::after(Duration::from_secs(2))
        );
    }

    #[test]
    fn honors_retry_after_only_for_retryable_kinds() {
        let server = classify(Some(503), None, None, Some("5"));
        assert_eq!(
            server.retry,
            RetryClassification::after(Duration::from_secs(5))
        );

        let quota = classify(Some(429), Some("insufficient_quota"), None, Some("5"));
        assert_eq!(quota.retry, RetryClassification::Never);
        // The advised wait survives on the never-retried kind: same-call
        // retries ignore it, but an application scheduling its own failover
        // still reads the provider's hint.
        assert_eq!(quota.retry_after, Some(Duration::from_secs(5)));
    }

    #[test]
    fn parses_numeric_retry_after_values() {
        assert_eq!(parse_retry_after("90"), Some(Duration::from_secs(90)));
        assert_eq!(
            parse_retry_after(" 2.5 "),
            Some(Duration::from_millis(2500))
        );
        assert_eq!(parse_retry_after("0"), Some(Duration::ZERO));
        assert_eq!(parse_retry_after("-1"), None);
    }

    #[test]
    fn parses_an_http_date_retry_after_value() {
        // 2100-01-01T00:00:00Z is 4_102_444_800 seconds after the epoch.
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock is after the epoch")
            .as_secs();
        let expected = 4_102_444_800 - now;

        let delay =
            parse_retry_after("Fri, 01 Jan 2100 00:00:00 GMT").expect("a future HTTP-date parses");

        assert!(
            delay.as_secs().abs_diff(expected) <= 2,
            "expected about {expected}s, got {}s",
            delay.as_secs()
        );
    }

    #[test]
    fn rejects_past_and_unparseable_retry_after_values() {
        assert_eq!(parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT"), None);
        assert_eq!(parse_retry_after("soon"), None);
        assert_eq!(parse_retry_after(""), None);
        assert_eq!(parse_retry_after("Wed, 21 Oct 2015 07:28:00"), None);
        assert_eq!(parse_retry_after("Fri, 32 Jan 2100 00:00:00 GMT"), None);
        assert_eq!(parse_retry_after("Fri, 01 Xxx 2100 00:00:00 GMT"), None);
        assert_eq!(parse_retry_after("Fri, 01 Jan 2100 00:00:00 UTC"), None);
    }

    #[test]
    fn extracts_the_openai_error_shape() {
        let body = json!({
            "error": {
                "message": "You exceeded your current quota.",
                "type": "insufficient_quota",
                "code": "insufficient_quota",
                "param": null
            }
        });

        assert_eq!(
            extract(Some(&body)),
            (
                Some("You exceeded your current quota.".to_owned()),
                Some("insufficient_quota".to_owned())
            )
        );
    }

    #[test]
    fn prefers_the_openai_code_field_over_the_type_field() {
        // Fireworks puts the discriminating value in `code` and `"error"` in
        // `type`.
        let body = json!({
            "error": {
                "message": "Account is suspended.",
                "param": null,
                "code": "PRECONDITION_FAILED",
                "type": "error"
            }
        });

        let (message, code) = extract(Some(&body));

        assert_eq!(message.as_deref(), Some("Account is suspended."));
        assert_eq!(code.as_deref(), Some("PRECONDITION_FAILED"));
    }

    #[test]
    fn extracts_the_anthropic_error_shape() {
        let body = json!({
            "type": "error",
            "error": {"type": "rate_limit_error", "message": "Number of requests exceeded"}
        });

        assert_eq!(
            extract(Some(&body)),
            (
                Some("Number of requests exceeded".to_owned()),
                Some("rate_limit_error".to_owned())
            )
        );
    }

    #[test]
    fn extracts_the_gemini_error_shape() {
        let body = json!({
            "error": {
                "code": 429,
                "message": "Resource has been exhausted",
                "status": "RESOURCE_EXHAUSTED",
                "details": []
            }
        });

        assert_eq!(
            extract(Some(&body)),
            (
                Some("Resource has been exhausted".to_owned()),
                Some("RESOURCE_EXHAUSTED".to_owned())
            )
        );
    }

    #[test]
    fn extracts_the_bedrock_error_shapes() {
        let sigv4 = json!({"message": "Too many requests", "__type": "coral#ThrottlingException"});
        assert_eq!(
            extract(Some(&sigv4)),
            (
                Some("Too many requests".to_owned()),
                Some("ThrottlingException".to_owned())
            )
        );

        let api_key = json!({"Message": "Access denied", "code": "AccessDeniedException"});
        assert_eq!(
            extract(Some(&api_key)),
            (
                Some("Access denied".to_owned()),
                Some("AccessDeniedException".to_owned())
            )
        );
    }

    #[test]
    fn extracts_openai_compatible_and_stream_variants() {
        // OpenRouter reports a numeric code, so only the message survives.
        let openrouter = json!({"error": {"code": 429, "message": "Rate limit exceeded"}});
        assert_eq!(
            extract(Some(&openrouter)),
            (Some("Rate limit exceeded".to_owned()), None)
        );

        // The OpenAI Codex endpoint reports `detail` instead of an error object.
        let codex = json!({"detail": "Unsupported model"});
        assert_eq!(
            extract(Some(&codex)),
            (Some("Unsupported model".to_owned()), None)
        );

        // A bare stream error payload carries its own fields.
        let stream = json!({"type": "server_error", "message": "The server had an error"});
        assert_eq!(
            extract(Some(&stream)),
            (
                Some("The server had an error".to_owned()),
                Some("server_error".to_owned())
            )
        );
    }

    #[test]
    fn extracts_nothing_from_an_absent_or_unhelpful_body() {
        assert_eq!(extract(None), (None, None));
        assert_eq!(extract(Some(&json!({"unrelated": true}))), (None, None));
        assert_eq!(
            extract(Some(&json!({"error": {"message": "  "}}))),
            (None, None)
        );
    }

    #[test]
    fn classifies_an_extracted_body_end_to_end() {
        let body = json!({
            "type": "error",
            "error": {"type": "overloaded_error", "message": "Overloaded"}
        });
        let (message, code) = extract(Some(&body));

        let failure = classify(None, code.as_deref(), message.as_deref(), None);

        assert_eq!(
            kinds(&failure),
            (ErrorKind::Server, RetryClassification::Safe)
        );
    }
}
