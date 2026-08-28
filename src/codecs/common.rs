//! Helpers shared by every provider codec.

use serde_json::{Map, Number, Value, json};

use crate::adapter::ResolvedCall;
use crate::resolver::ResolvedRoute;
#[cfg(any(feature = "anthropic", feature = "bedrock", feature = "gemini"))]
use crate::types::Role;
use crate::types::{ContentPart, Error, ErrorKind, FinishReason, Message, Request};

/// Raw provider option keys a codec consumes as behavior controls.
///
/// Every key listed here is removed from the raw provider options before they
/// are merged into a request body, so a control never reaches the wire. Adding
/// a control means adding its key here and reading it in [`Controls`].
pub(crate) const CONTROL_KEYS: &[&str] = &["auto_cache"];

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

/// Joins a provider base URL and a request path.
pub(crate) fn endpoint(base_url: &str, path: &str) -> String {
    format!(
        "{}/{}",
        base_url.trim_end_matches('/'),
        path.trim_start_matches('/')
    )
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
/// Several protocols accept only a string for a tool result, so a tool that
/// returns an image or a document loses it. The text still reaches the model,
/// so this is a warning rather than a refusal — unlike message content, where
/// the caller's own attachment would vanish.
pub(crate) fn flattens_tool_result_content(request: &Request) -> bool {
    request
        .messages()
        .iter()
        .flat_map(Message::content)
        .filter_map(|part| match part {
            ContentPart::ToolResult(result) => Some(result),
            _ => None,
        })
        .flat_map(|result| result.content.iter())
        .any(|part| {
            matches!(
                part,
                ContentPart::Image(_) | ContentPart::Audio(_) | ContentPart::Document(_)
            )
        })
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
#[cfg(any(feature = "anthropic", feature = "bedrock", feature = "gemini"))]
pub(crate) fn system_text(messages: &[Message]) -> String {
    messages
        .iter()
        .filter(|message| matches!(message.role(), Role::System | Role::Developer))
        .map(|message| plain_text(message.content()))
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Maps a provider stop reason onto the normalized finish reason.
pub(crate) fn finish_reason(value: Option<&str>) -> FinishReason {
    match value {
        None | Some("stop" | "end_turn" | "STOP") => FinishReason::Stop,
        Some("length" | "max_tokens" | "MAX_TOKENS") => FinishReason::Length,
        Some("tool_calls" | "tool_use") => FinishReason::ToolCall,
        Some("content_filter" | "SAFETY" | "BLOCKLIST" | "PROHIBITED_CONTENT") => {
            FinishReason::ContentFilter
        }
        Some(other) => FinishReason::Other(other.to_owned()),
    }
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
        CONTROL_KEYS, merge_options, parse_arguments, unsupported_capability, wire_options,
    };
    use crate::codecs::test_support;
    use crate::types::{ErrorKind, Request};

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
}
