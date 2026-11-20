//! Raw provider options, controls, URL joining, and number encoding.
//!
//! Every codec splits the raw provider options into wire options and
//! consumed [`Controls`], builds its operation URL with [`endpoint`], and
//! merges the raw options over its typed body last with [`merge_options`].

use serde_json::{Map, Number, Value, json};

use crate::adapter::ResolvedCall;
pub(crate) use crate::catalog::endpoint;
use crate::types::{CacheHint, Request};

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

/// The raw provider options for the selected route, with controls removed.
///
/// This reads **only** the namespace of the route's canonical catalog provider
/// id. Namespaces belonging to other providers are ignored, so one request can
/// carry options for several failover candidates. The provider's catalog
/// [`default_options`](crate::catalog::CatalogProvider::default_options) seed
/// the map and the request's own options merge over them, so a request always
/// wins a collision. The returned map is what a codec passes to
/// [`merge_options`]; the returned [`Controls`] are consumed by the codec and
/// never sent.
pub(crate) fn wire_options(call: &ResolvedCall) -> (Map<String, Value>, Controls) {
    let mut options = call.route().provider().default_options().clone();
    if let Some(request_options) = call.request().options_for(call.route().provider().id()) {
        merge_options(&mut options, request_options.clone());
    }

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

/// The cache routing key this call sends, or `None` for no hint.
///
/// The hint reaches the wire only where the catalog claims `cache_routing`.
/// An explicit [`CacheHint::Key`] is sent verbatim; [`CacheHint::Disabled`]
/// sends nothing; the [`CacheHint::Auto`] default — an unset hint included —
/// sends the prefix fingerprint, and follows `auto_cache` and the `caching`
/// capability like every other automatic cache behavior. A raw
/// `prompt_cache_key` in the provider options still wins, because codecs
/// merge raw options over the encoded body.
pub(crate) fn cache_routing_key(call: &ResolvedCall, controls: Controls) -> Option<String> {
    let capabilities = call.route().model().capabilities();
    if !capabilities.cache_routing().is_supported() {
        return None;
    }
    match call.request().cache_hint() {
        Some(CacheHint::Disabled) => None,
        Some(CacheHint::Key { key }) => Some(key.clone()),
        Some(CacheHint::Auto) | None => (controls.auto_cache
            && capabilities.caching().is_supported())
        .then(|| prefix_fingerprint(call.request())),
    }
}

/// A stable fingerprint of the request's cacheable prefix.
///
/// The prefix is the system and developer messages plus the tool
/// definitions: the content a prompt cache keys on that stays identical
/// across the turns of an agent loop, so every turn of one conversation
/// routes to the same replica. The hash is FNV-1a over the serialized
/// prefix — a routing hint, not a security boundary.
fn prefix_fingerprint(request: &Request) -> String {
    let system: Vec<Value> = request
        .messages()
        .iter()
        .filter(|message| message.is_instruction())
        .map(|message| serde_json::to_value(message).unwrap_or(Value::Null))
        .collect();
    let tools: Vec<Value> = request
        .tools()
        .iter()
        .map(|tool| serde_json::to_value(tool).unwrap_or(Value::Null))
        .collect();
    let prefix = json!({ "system": system, "tools": tools }).to_string();

    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in prefix.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("lithos-{hash:016x}")
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

/// Converts US dollars to the integer micros the cost type carries.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a float-to-integer cast saturates, which is the clamp a provider-reported cost needs"
)]
pub(crate) fn usd_micros(usd: f64) -> u64 {
    (usd * 1_000_000.0).round() as u64
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use serde_json::{Map, Value, json};

    use super::{CONTROL_KEYS, merge_options, wire_options};
    use crate::codecs::test_support;
    use crate::types::Request;

    #[test]
    fn sampling_sends_the_decimal_the_caller_wrote() {
        // `f32::into::<f64>()` would send 0.699999988079071 here.
        assert_eq!(super::sampling(0.7), serde_json::json!(0.7));
        assert_eq!(super::sampling(1.0), serde_json::json!(1.0));
        assert_eq!(super::sampling(0.0), serde_json::json!(0.0));
        assert_eq!(super::sampling(0.05), serde_json::json!(0.05));
    }

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
}
