//! Helpers shared by every provider codec.

use serde_json::{Map, Number, Value, json};

use crate::adapter::ResolvedCall;
use crate::resolver::ResolvedRoute;
#[cfg(any(
    feature = "anthropic",
    feature = "bedrock",
    feature = "openai-compatible",
    test
))]
use crate::transport::classify;
#[cfg(any(feature = "anthropic", feature = "bedrock", feature = "gemini"))]
use crate::types::ReasoningContent;
#[cfg(any(
    feature = "anthropic",
    feature = "bedrock",
    feature = "gemini",
    feature = "openai"
))]
use crate::types::Role;
use crate::types::{ContentPart, Error, ErrorKind, FinishReason, Message, Request};

/// Raw provider option keys a codec consumes as behavior controls.
///
/// Every key listed here is removed from the raw provider options before they
/// are merged into a request body, so a control never reaches the wire. Adding
/// a control means adding its key here and reading it in [`Controls`].
pub(crate) const CONTROL_KEYS: &[&str] = &["auto_cache"];

/// The provider code a codec sets when a call ends in a model refusal.
///
/// [`classify`](crate::transport::classify) registers this code among the
/// content-filter codes, which is what makes a refusal failover-eligible.
#[cfg(any(
    feature = "anthropic",
    feature = "bedrock",
    feature = "openai-compatible",
    test
))]
const REFUSAL_CODE: &str = "refusal";

/// The signature family of Claude-minted reasoning signatures.
///
/// The Anthropic Messages and Bedrock Converse protocols both carry them, so
/// a conversation that moves between those providers keeps its signatures.
#[cfg(any(feature = "anthropic", feature = "bedrock", test))]
pub(crate) const ANTHROPIC_SIGNATURES: &str = "anthropic";

/// The signature family of Gemini thought signatures.
#[cfg(any(feature = "gemini", test))]
pub(crate) const GEMINI_SIGNATURES: &str = "gemini";

/// Whether a reasoning part carries a signature another family minted.
///
/// A foreign signature cannot verify at this provider, and replaying it can
/// fail the whole request, so the encoder skips the part and reports the
/// skip. A signed part whose origin is unknown — persisted before origins
/// were recorded — replays as it always did.
#[cfg(any(feature = "anthropic", feature = "bedrock", feature = "gemini"))]
pub(crate) fn foreign_signature(reasoning: &ReasoningContent, family: &str) -> bool {
    reasoning.signature.is_some()
        && reasoning
            .signature_origin
            .as_deref()
            .is_some_and(|origin| origin != family)
}

/// Whether any reasoning part of the request carries a foreign signature.
#[cfg(any(feature = "anthropic", feature = "bedrock", feature = "gemini"))]
pub(crate) fn carries_foreign_signature(request: &Request, family: &str) -> bool {
    request
        .messages()
        .iter()
        .flat_map(Message::content)
        .any(|part| match part {
            ContentPart::Reasoning(reasoning) => foreign_signature(reasoning, family),
            _ => false,
        })
}

/// Codec behavior selected by control keys in the raw provider options.
///
/// Controls are read from the same provider namespace as the wire options, but
/// they are consumed rather than forwarded. A control whose value has the wrong
/// JSON type is ignored and keeps its default.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Controls {
    /// Whether the codec may add provider prompt-cache markers of its own.
    ///
    /// Defaults to `true`. Set `"auto_cache": false` in a provider namespace to
    /// stop a codec from inserting cache breakpoints the caller did not ask
    /// for.
    pub auto_cache: bool,
}

impl Default for Controls {
    fn default() -> Self {
        Self { auto_cache: true }
    }
}

/// API version segments a codec's operation path may repeat from a base URL.
///
/// Codecs own the version segment: they ask for `/v1/messages` or
/// `/v1beta/models/...`. The older convention put the version in the catalog
/// base URL instead, and gateways still publish mounts that end with one, so a
/// base URL ending in one of these segments is common. [`endpoint`] joins the
/// two without repeating the segment.
const VERSION_SEGMENTS: &[&str] = &["v1", "v1beta"];

/// Joins a provider base URL and a request path.
///
/// A base URL whose path already ends with the version segment the request
/// path starts with keeps that segment once: base `https://api.moonshot.ai/v1`
/// and path `/v1/chat/completions` join as
/// `https://api.moonshot.ai/v1/chat/completions`, not `/v1/v1/...`. Only an
/// exact repeat of the same segment is collapsed, so a `/v1` base URL still
/// keeps a Gemini `/v1beta` path intact, and a gateway mounted at
/// `/gateway` keeps its own prefix.
pub(crate) fn endpoint(base_url: &str, path: &str) -> String {
    let base = base_url.trim_end_matches('/');
    let path = path.trim_start_matches('/');
    let path = match repeated_version(base, path) {
        Some(segment) => path[segment.len()..].trim_start_matches('/'),
        None => path,
    };

    if path.is_empty() {
        return base.to_owned();
    }
    format!("{base}/{path}")
}

/// The version segment `base` and `path` both carry, when they share one.
///
/// `base` has already lost its trailing slashes and `path` its leading ones.
fn repeated_version<'a>(base: &str, path: &'a str) -> Option<&'a str> {
    let host_and_path = base.split_once("://").map_or(base, |(_, rest)| rest);
    let base_tail = host_and_path.split('/').next_back()?;
    let path_head = path.split('/').next()?;

    // A bare host has no path segment to repeat, so its final label — which
    // could in principle read `v1` — is not a version segment.
    let has_path = host_and_path.contains('/');
    (has_path && path_head == base_tail && VERSION_SEGMENTS.contains(&path_head))
        .then_some(path_head)
}

/// The raw provider options for the selected route, with controls removed.
///
/// This reads **only** the namespace of the route's canonical catalog provider
/// id. Namespaces belonging to other providers are ignored, so one request can
/// carry options for several failover candidates. The returned map is what a
/// codec passes to [`merge_options`]; the returned [`Controls`] are consumed by
/// the codec and never sent.
pub(crate) fn wire_options(call: &ResolvedCall) -> (Map<String, Value>, Controls) {
    let options = call
        .request()
        .options_for(call.route().provider().id())
        .cloned()
        .unwrap_or_default();

    split_controls(options)
}

/// Splits control keys out of one provider option namespace.
fn split_controls(mut options: Map<String, Value>) -> (Map<String, Value>, Controls) {
    let mut consumed = Map::new();
    for key in CONTROL_KEYS {
        if let Some(value) = options.remove(*key) {
            consumed.insert((*key).to_owned(), value);
        }
    }

    let mut controls = Controls::default();
    if let Some(auto_cache) = consumed.get("auto_cache").and_then(Value::as_bool) {
        controls.auto_cache = auto_cache;
    }

    (options, controls)
}

/// Merges raw provider options over an already encoded request body.
///
/// Raw options **win**. A codec encodes every typed request field first and
/// calls this last, so an application can override anything the codec produced.
/// This inverts the older behavior, where the body was seeded from the options
/// map and then overwritten by typed fields; under the approved request
/// controls plan the raw escape hatch is authoritative.
///
/// Two objects at the same key are merged recursively, so overriding one key of
/// a nested container such as `generationConfig` keeps the sibling fields the
/// codec encoded. Every other value, arrays included, replaces what the body
/// held.
pub(crate) fn merge_options(body: &mut Map<String, Value>, options: Map<String, Value>) {
    for (key, value) in options {
        let merged = match (body.remove(&key), value) {
            (Some(Value::Object(mut encoded)), Value::Object(raw)) => {
                merge_options(&mut encoded, raw);
                Value::Object(encoded)
            }
            (_, value) => value,
        };
        body.insert(key, merged);
    }
}

/// Fails the call when a request carries content this protocol cannot encode.
///
/// Silently omitting content is the worst available outcome: the provider
/// answers a prompt the caller never sent, and the caller sees a successful
/// response with no indication that their attachment was discarded. Refusing
/// before dispatch is the same rule the crate applies to custom tools.
///
/// `unencodable` names the capability for a part the codec drops, or returns
/// `None` for a part it can encode.
///
/// # Errors
///
/// Returns [`ErrorKind::InvalidRequest`](crate::types::ErrorKind::InvalidRequest)
/// naming the first unsupported capability found.
pub(crate) fn reject_unencodable(
    route: &ResolvedRoute,
    request: &Request,
    unencodable: fn(&ContentPart) -> Option<&'static str>,
) -> Result<(), Error> {
    let unsupported = request
        .messages()
        .iter()
        .flat_map(Message::content)
        .find_map(unencodable);
    match unsupported {
        Some(capability) => Err(unsupported_capability(route, capability)),
        None => Ok(()),
    }
}

/// Whether any tool result carries content this codec flattens to text.
///
/// Several protocols accept only a string for a tool result, so anything
/// [`plain_text`] does not keep is lost. That is every non-text part: media,
/// and structured JSON too, which is easy to overlook because it is not media.
/// The text still reaches the model, so this is a warning rather than a
/// refusal — unlike message content, where the caller's own attachment would
/// vanish entirely.
///
/// `carries` names the result contents this codec keeps whole. It judges each
/// result's parts together because carriage can depend on the mix: the OpenAI
/// protocols send an all-JSON result as the bare value but flatten JSON that
/// shares a result with other parts. A codec that falls back to serializing
/// the whole content still hands the model a shape it did not ask for, so
/// reporting that is not a false positive.
pub(crate) fn flattens_tool_result_content(
    request: &Request,
    carries: fn(&[ContentPart]) -> bool,
) -> bool {
    request
        .messages()
        .iter()
        .flat_map(Message::content)
        .filter_map(|part| match part {
            ContentPart::ToolResult(result) => Some(result),
            _ => None,
        })
        .any(|result| !carries(&result.content))
}

/// Encodes a sampling parameter without its binary32 rounding error.
///
/// `Request` carries `temperature` and `top_p` as `f32`. Widening one to `f64`
/// keeps the binary32 value, so a caller's `0.7` reaches the wire as
/// `0.699999988079071`. Round-tripping through the shortest decimal that
/// identifies the `f32` sends back what the caller wrote.
///
/// Request construction rejects a non-finite value, so the fallback is
/// unreachable in practice and simply widens rather than inventing a number.
pub(crate) fn sampling(value: f32) -> Value {
    value
        .to_string()
        .parse::<f64>()
        .ok()
        .and_then(Number::from_f64)
        .map_or_else(|| Value::from(f64::from(value)), Value::Number)
}

/// Parses a provider tool-argument string, falling back to an empty object.
///
/// A tool call that takes no arguments streams no argument fragments, which
/// leaves an empty string. Canonically that is an empty object, not null. A
/// malformed string is normalized the same way rather than failing the stream;
/// the untouched text stays available in
/// [`ToolCall::raw_arguments`](crate::types::ToolCall::raw_arguments).
pub(crate) fn parse_arguments(raw: &str) -> Value {
    serde_json::from_str(raw).unwrap_or_else(|_| json!({}))
}

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

/// Concatenates every text part in order, ignoring every other part.
pub(crate) fn plain_text(parts: &[ContentPart]) -> String {
    parts
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

/// Joins the text of every system and developer message.
///
/// A message whose text is only whitespace contributes nothing — templating
/// commonly produces one, and a whitespace-only system field is a blank block
/// providers reject rather than an instruction.
#[cfg(any(feature = "anthropic", feature = "bedrock", feature = "gemini"))]
pub(crate) fn system_text(messages: &[Message]) -> String {
    messages
        .iter()
        .filter(|message| matches!(message.role(), Role::System | Role::Developer))
        .map(|message| plain_text(message.content()))
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Whether a system message carries content the system field cannot hold.
///
/// [`system_text`] joins the text of every system and developer message, so
/// anything else in one is dropped. The system fields of these protocols take
/// text and nothing else, which makes the drop correct — but silent, and the
/// caller put that content somewhere deliberately. The text still reaches the
/// model, so this is a warning rather than a refusal.
#[cfg(any(
    feature = "anthropic",
    feature = "bedrock",
    feature = "gemini",
    feature = "openai"
))]
pub(crate) fn flattens_system_content(request: &Request) -> bool {
    request
        .messages()
        .iter()
        .filter(|message| matches!(message.role(), Role::System | Role::Developer))
        .flat_map(Message::content)
        .any(|part| !matches!(part, ContentPart::Text { .. }))
}

/// Maps a provider stop reason onto the normalized finish reason.
///
/// `stop_sequence` is a normal stop: the model ended on a sequence the caller
/// asked it to stop at. Gemini's `RECITATION` is a content block, reported
/// separately from `SAFETY` because the blocked material is quoted source
/// rather than unsafe content.
pub(crate) fn finish_reason(value: Option<&str>) -> FinishReason {
    match value {
        None | Some("stop" | "end_turn" | "stop_sequence" | "STOP") => FinishReason::Stop,
        Some("length" | "max_tokens" | "MAX_TOKENS") => FinishReason::Length,
        Some("tool_calls" | "tool_use") => FinishReason::ToolCall,
        Some("content_filter" | "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT") => {
            FinishReason::ContentFilter
        }
        Some(other) => FinishReason::Other(other.to_owned()),
    }
}

/// The error a codec returns when a response or stream ends in a refusal.
///
/// A refusal is a failure, not a short answer: the model declined the request
/// and produced no usable content. Returning it as a successful empty response
/// hides that from the caller and from the retry and failover middleware, so
/// the codec fails the call instead.
///
/// The provider code is `refusal`, which
/// [`classify`](crate::transport::classify::classify) already lists among the
/// content-filter codes, so the shared classifier — not this helper — decides
/// the kind and the retry classification. `explanation` is the provider's own
/// account of the refusal, when the payload carried one, and `raw` is the
/// payload the codec decoded.
#[cfg(any(
    feature = "anthropic",
    feature = "bedrock",
    feature = "openai-compatible",
    test
))]
pub(crate) fn refusal(
    route: &ResolvedRoute,
    explanation: Option<&str>,
    raw: Option<Value>,
) -> Error {
    let detail = match explanation {
        Some(explanation) => format!("refused the request: {explanation}"),
        None => "refused the request".to_owned(),
    };
    // The code alone decides the classification here. The refusal explanation
    // is the model's prose, and running the message heuristics over it would
    // let a phrase such as "not found" rewrite the kind.
    let failure = classify::classify(None, Some(REFUSAL_CODE), None, None);

    let mut error = Error::new(
        failure.kind,
        format!("provider {} {detail}", route.provider().id()),
    )
    .with_provider(route.provider().id().clone())
    .with_provider_code(REFUSAL_CODE)
    .with_retry(failure.retry);
    if let Some(raw) = raw {
        error = error.with_raw_data(raw);
    }
    error
}

#[cfg(test)]
mod tests {
    #[test]
    fn sampling_sends_the_decimal_the_caller_wrote() {
        // `f32::into::<f64>()` would send 0.699999988079071 here.
        assert_eq!(super::sampling(0.7), serde_json::json!(0.7));
        assert_eq!(super::sampling(1.0), serde_json::json!(1.0));
        assert_eq!(super::sampling(0.0), serde_json::json!(0.0));
        assert_eq!(super::sampling(0.05), serde_json::json!(0.05));
    }

    use std::error::Error as StdError;

    use serde_json::{Map, Value, json};

    use super::{
        CONTROL_KEYS, endpoint, finish_reason, merge_options, parse_arguments, refusal,
        unsupported_capability, wire_options,
    };
    use crate::codecs::test_support;
    use crate::types::{ErrorKind, FinishReason, Request, RetryClassification};

    fn object(value: Value) -> Result<Map<String, Value>, Box<dyn StdError>> {
        match value {
            Value::Object(map) => Ok(map),
            other => Err(format!("expected a JSON object, got {other}").into()),
        }
    }

    #[test]
    fn raw_options_override_generated_wire_fields() -> Result<(), Box<dyn StdError>> {
        let mut body = object(json!({ "model": "alpha-one", "temperature": 0.2 }))?;

        merge_options(&mut body, object(json!({ "temperature": 0.9 }))?);

        assert_eq!(body["temperature"], json!(0.9));
        assert_eq!(body["model"], json!("alpha-one"));
        Ok(())
    }

    #[test]
    fn nested_objects_merge_and_other_values_replace() -> Result<(), Box<dyn StdError>> {
        let mut body = object(json!({
            "generationConfig": { "temperature": 0.2, "maxOutputTokens": 64 },
            "tools": [{ "name": "encoded" }],
        }))?;

        merge_options(
            &mut body,
            object(json!({
                "generationConfig": { "temperature": 0.9, "topK": 40 },
                "tools": [{ "name": "override" }],
            }))?,
        );

        assert_eq!(body["generationConfig"]["temperature"], json!(0.9));
        assert_eq!(body["generationConfig"]["maxOutputTokens"], json!(64));
        assert_eq!(body["generationConfig"]["topK"], json!(40));
        assert_eq!(body["tools"], json!([{ "name": "override" }]));
        Ok(())
    }

    #[test]
    fn control_keys_are_consumed_and_never_reach_the_wire() -> Result<(), Box<dyn StdError>> {
        let mut options = object(json!({ "reasoning_effort": "high" }))?;
        for key in CONTROL_KEYS {
            options.insert((*key).to_owned(), json!(false));
        }
        let request = Request::builder()
            .model("alpha/one")
            .user("Hello")
            .provider_options("alpha", options)
            .build()?;

        let (wire, controls) = wire_options(&test_support::test_call(request)?);

        for key in CONTROL_KEYS {
            assert!(!wire.contains_key(*key), "{key} reached the wire options");
        }
        assert_eq!(wire["reasoning_effort"], json!("high"));
        assert!(!controls.auto_cache);
        Ok(())
    }

    #[test]
    fn auto_cache_defaults_to_true() -> Result<(), Box<dyn StdError>> {
        let request = Request::builder()
            .model("alpha/one")
            .user("Hello")
            .provider_option("alpha", "seed", json!(7))
            .build()?;

        let (wire, controls) = wire_options(&test_support::test_call(request)?);

        assert!(controls.auto_cache);
        assert_eq!(wire["seed"], json!(7));
        Ok(())
    }

    #[test]
    fn options_for_another_provider_are_ignored() -> Result<(), Box<dyn StdError>> {
        let request = Request::builder()
            .model("alpha/one")
            .user("Hello")
            .provider_option("beta", "seed", json!(7))
            .build()?;

        let (wire, controls) = wire_options(&test_support::test_call(request)?);

        assert!(wire.is_empty());
        assert!(controls.auto_cache);
        Ok(())
    }

    #[test]
    fn empty_and_malformed_arguments_become_an_empty_object() {
        assert_eq!(parse_arguments(""), json!({}));
        assert_eq!(parse_arguments("{\"query\":"), json!({}));
        assert_eq!(parse_arguments("   "), json!({}));
        assert_eq!(
            parse_arguments("{\"query\":\"rust\"}"),
            json!({ "query": "rust" })
        );
    }

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
    fn a_stop_sequence_is_a_normal_stop() {
        assert_eq!(finish_reason(Some("stop_sequence")), FinishReason::Stop);
    }

    #[test]
    fn recitation_is_a_content_filter() {
        assert_eq!(
            finish_reason(Some("RECITATION")),
            FinishReason::ContentFilter
        );
    }

    #[test]
    fn an_unknown_stop_reason_keeps_its_provider_spelling() {
        assert_eq!(
            finish_reason(Some("guardrail_intervened")),
            FinishReason::Other("guardrail_intervened".to_owned())
        );
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
    fn a_bare_host_takes_the_whole_operation_path() {
        assert_eq!(
            endpoint("https://api.openai.com", "/v1/responses"),
            "https://api.openai.com/v1/responses"
        );
        assert_eq!(
            endpoint("https://api.openai.com/", "/v1/responses"),
            "https://api.openai.com/v1/responses"
        );
    }

    #[test]
    fn a_base_url_ending_in_the_version_does_not_repeat_it() {
        assert_eq!(
            endpoint("https://api.moonshot.ai/v1", "/v1/chat/completions"),
            "https://api.moonshot.ai/v1/chat/completions"
        );
        assert_eq!(
            endpoint("https://api.moonshot.ai/v1/", "/v1/chat/completions"),
            "https://api.moonshot.ai/v1/chat/completions"
        );
        assert_eq!(
            endpoint(
                "https://generativelanguage.googleapis.com/v1beta",
                "/v1beta/models/gemini:generateContent"
            ),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini:generateContent"
        );
    }

    #[test]
    fn a_gateway_mount_keeps_its_own_prefix() {
        assert_eq!(
            endpoint("https://gw.example.com/gateway", "/v1/chat/completions"),
            "https://gw.example.com/gateway/v1/chat/completions"
        );
        assert_eq!(
            endpoint("https://gw.example.com/gateway/v1", "/v1/chat/completions"),
            "https://gw.example.com/gateway/v1/chat/completions"
        );
    }

    #[test]
    fn only_the_same_version_segment_collapses() {
        // `v1` and `v1beta` are different APIs, so neither absorbs the other.
        assert_eq!(
            endpoint("https://example.com/v1", "/v1beta/models/one:count"),
            "https://example.com/v1/v1beta/models/one:count"
        );
        // An unversioned operation path is never shortened.
        assert_eq!(
            endpoint("https://bedrock.example.com/v1", "/model/one/converse"),
            "https://bedrock.example.com/v1/model/one/converse"
        );
    }
}
