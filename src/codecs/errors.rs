//! The errors every codec constructs the same way.
//!
//! A codec fails a call before dispatch for a capability the protocol lacks,
//! after decode for a body or stream that does not match the protocol, and
//! for a refusal or content block the provider reported on a success. The
//! classification of each is a crate-wide rule, so it lives here once.

use serde_json::Value;

use crate::resolver::ResolvedRoute;
use crate::types::{Error, ErrorKind, RetryClassification};

/// The provider code a codec sets when a call ends in a model refusal.
///
/// [`classify`](crate::transport::classify) registers this code among the
/// content-filter codes, which is what makes a refusal failover-eligible.
const REFUSAL_CODE: &str = "refusal";

/// The error every codec returns for a request feature it cannot encode.
///
/// Codecs raise this before any network dispatch — for example when a request
/// carries a custom tool and the protocol has no custom tool. A capability is
/// never silently downgraded.
pub(crate) fn unsupported_capability(route: &ResolvedRoute, capability: &str) -> Error {
    Error::new(
        ErrorKind::InvalidRequest,
        format!(
            "provider {} does not support {capability}",
            route.provider().id()
        ),
    )
    .with_provider(route.provider().id().clone())
    .with_provider_code("unsupported_capability")
}

/// The error a codec returns when a response or stream ends in a refusal.
///
/// A refusal is a failure, not a short answer: the model declined the request
/// and produced no usable content. Returning it as a successful empty response
/// hides that from the caller and from the retry and failover middleware, so
/// the codec fails the call instead.
///
/// The provider code is `refusal`, which
/// [`classify`](crate::transport::classify::classify) also lists among the
/// content-filter codes, so an error body carrying it classifies the same
/// way. `explanation` is the provider's own account of the refusal, when the
/// payload carried one, and `raw` is the payload the codec decoded.
pub(crate) fn refusal(
    route: &ResolvedRoute,
    explanation: Option<&str>,
    raw: Option<Value>,
) -> Error {
    let detail = match explanation {
        Some(explanation) => format!("refused the request: {explanation}"),
        None => "refused the request".to_owned(),
    };
    let error = content_filter(route, REFUSAL_CODE, &detail);
    match raw {
        Some(raw) => error.with_raw_data(raw),
        None => error,
    }
}

/// A content-policy failure the provider reported on a successful body.
///
/// `code` is the provider's own spelling of the block or refusal reason and
/// becomes the provider code. A blocked prompt is blocked every time, so the
/// error is never retried; failover to another provider is the caller's
/// decision. The message is prefixed with the provider id like every other
/// provider failure.
pub(crate) fn content_filter(route: &ResolvedRoute, code: &str, detail: &str) -> Error {
    Error::new(
        ErrorKind::ContentFilter,
        format!("provider {} {detail}", route.provider().id()),
    )
    .with_provider(route.provider().id().clone())
    .with_provider_code(code)
    .with_retry(RetryClassification::Never)
}

/// The error for a 200 body that does not match the protocol.
///
/// A structurally malformed 200 is indistinguishable from a garbled or
/// truncated body, so a fresh attempt is safe — the same classification the
/// transport gives a 200 whose body is not JSON at all. `message` is the
/// whole message, because the codecs word these differently and the wire
/// tests pin the wording; `raw` is the body when the caller still holds it.
pub(crate) fn malformed_success(
    route: &ResolvedRoute,
    message: impl Into<String>,
    raw: Option<Value>,
) -> Error {
    let error = Error::new(ErrorKind::ResponseDecode, message)
        .with_provider(route.provider().id().clone())
        .with_retry(RetryClassification::Safe);
    match raw {
        Some(raw) => error.with_raw_data(raw),
        None => error,
    }
}

/// The error for a stream whose events do not match the protocol.
///
/// The streaming twin of [`malformed_success`]: a garbled stream is
/// indistinguishable from mid-stream corruption, so the failure is retryable.
pub(crate) fn malformed_stream(
    route: &ResolvedRoute,
    message: impl Into<String>,
    raw: Option<Value>,
) -> Error {
    let error = Error::new(ErrorKind::StreamDecode, message)
        .with_provider(route.provider().id().clone())
        .with_retry(RetryClassification::Safe);
    match raw {
        Some(raw) => error.with_raw_data(raw),
        None => error,
    }
}

/// The error for a stream event whose payload is not JSON.
///
/// `source` is the parse failure. The Responses codec deliberately skips such
/// an event instead, because proxies inject keepalive payloads there; every
/// other codec fails the stream with this.
pub(crate) fn invalid_stream_event(
    route: &ResolvedRoute,
    message: impl Into<String>,
    source: serde_json::Error,
) -> Error {
    malformed_stream(route, message, None).with_source(source)
}

/// The error for tool-call input streamed into a block no start announced.
///
/// Anthropic and Bedrock announce every tool call in a start event carrying
/// its id and name. An input fragment for a block no start opened means the
/// start was lost in transit; assembling the rest would fabricate a nameless
/// call that poisons the replayed conversation, so the stream fails retryably
/// instead — the same contract the Chat codec applies.
pub(crate) fn lost_tool_start(route: &ResolvedRoute) -> Error {
    malformed_stream(
        route,
        format!(
            "provider {} streamed tool-call input for a block whose start event never arrived",
            route.provider().id()
        ),
        None,
    )
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use serde_json::{Value, json};

    use super::{
        content_filter, invalid_stream_event, lost_tool_start, malformed_success, refusal,
        unsupported_capability,
    };
    use crate::codecs::test_support;
    use crate::types::{ErrorKind, RetryClassification};

    #[test]
    fn unsupported_capability_is_an_invalid_request() -> Result<(), Box<dyn StdError>> {
        let route = test_support::test_route()?;

        let error = unsupported_capability(&route, "custom tools");

        assert_eq!(error.kind(), ErrorKind::InvalidRequest);
        assert_eq!(error.provider_code(), Some("unsupported_capability"));
        assert!(error.message().contains("custom tools"));
        Ok(())
    }

    #[test]
    fn a_refusal_classifies_as_a_content_filter() -> Result<(), Box<dyn StdError>> {
        let route = test_support::test_route()?;

        let error = refusal(&route, Some("it asks for malware"), None);

        assert_eq!(error.kind(), ErrorKind::ContentFilter);
        assert_eq!(error.provider_code(), Some("refusal"));
        assert_eq!(error.retry_classification(), RetryClassification::Never);
        assert!(error.message().contains("it asks for malware"), "{error}");
        Ok(())
    }

    #[test]
    fn a_refusal_without_an_explanation_still_reports_one() -> Result<(), Box<dyn StdError>> {
        let route = test_support::test_route()?;

        let error = refusal(&route, None, Some(json!({ "stop_reason": "refusal" })));

        assert_eq!(error.kind(), ErrorKind::ContentFilter);
        assert!(error.message().contains("refused the request"), "{error}");
        assert_eq!(error.raw_data(), Some(&json!({ "stop_reason": "refusal" })));
        Ok(())
    }

    #[test]
    fn a_content_filter_keeps_the_provider_spelling_as_its_code() -> Result<(), Box<dyn StdError>> {
        let route = test_support::test_route()?;

        let error = content_filter(&route, "BLOCKLIST", "blocked the prompt");

        assert_eq!(error.kind(), ErrorKind::ContentFilter);
        assert_eq!(error.provider_code(), Some("BLOCKLIST"));
        assert_eq!(error.retry_classification(), RetryClassification::Never);
        assert!(error.message().contains("blocked the prompt"), "{error}");
        Ok(())
    }

    #[test]
    fn a_malformed_success_body_is_safe_to_retry() -> Result<(), Box<dyn StdError>> {
        let route = test_support::test_route()?;

        let error = malformed_success(&route, "returned nonsense", Some(json!({ "x": 1 })));

        assert_eq!(error.kind(), ErrorKind::ResponseDecode);
        assert_eq!(error.retry_classification(), RetryClassification::Safe);
        assert_eq!(error.raw_data(), Some(&json!({ "x": 1 })));
        assert_eq!(error.provider_code(), None);
        Ok(())
    }

    #[test]
    fn an_invalid_stream_event_carries_its_parse_failure() -> Result<(), Box<dyn StdError>> {
        let route = test_support::test_route()?;
        let source = serde_json::from_str::<Value>("not json")
            .err()
            .ok_or("the fixture must fail to parse")?;

        let error = invalid_stream_event(&route, "returned an invalid stream event", source);

        assert_eq!(error.kind(), ErrorKind::StreamDecode);
        assert_eq!(error.retry_classification(), RetryClassification::Safe);
        assert!(
            StdError::source(&error).is_some(),
            "the parse failure is the source"
        );
        assert_eq!(lost_tool_start(&route).kind(), ErrorKind::StreamDecode);
        Ok(())
    }
}
