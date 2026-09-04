//! Wire protocol encoders and decoders.

mod assembler;
mod common;

#[cfg(feature = "anthropic")]
pub(crate) mod anthropic;
#[cfg(feature = "bedrock")]
pub(crate) mod bedrock;
#[cfg(feature = "gemini")]
pub(crate) mod gemini;
#[cfg(feature = "openai")]
pub(crate) mod openai;
#[cfg(feature = "openai-compatible")]
pub(crate) mod openai_chat;

use serde_json::Value;

use crate::adapter::ResolvedCall;
use crate::resolver::ResolvedRoute;
use crate::transport::{EncodedRequest, SseEvent};
use crate::types::{Error, ErrorKind, Response, StreamEvent};

/// Translates one provider wire protocol.
///
/// A codec is stateless and shared across calls. Per-stream state lives in the
/// [`StreamDecoder`] it creates, so two concurrent streams never interfere.
pub(crate) trait Codec: Send + Sync {
    /// Encodes a resolved call into one provider request.
    ///
    /// Typed request fields are encoded first and raw provider options are
    /// merged last, so raw options win. See
    /// [`common::merge_options`](common::merge_options).
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidRequest`] when the request needs a
    /// capability this protocol does not have, before any network dispatch.
    fn encode(&self, call: &ResolvedCall, stream: bool) -> Result<EncodedRequest, Error>;

    /// Decodes one complete provider success body.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::ResponseDecode`] when the body does not match the
    /// protocol.
    fn decode_response(&self, route: &ResolvedRoute, value: Value) -> Result<Response, Error>;

    /// Creates the per-stream decoder that owns block identity and assembly.
    fn stream_decoder(&self, route: &ResolvedRoute) -> Box<dyn StreamDecoder>;

    /// Encodes a provider-native input token count request.
    ///
    /// `None` means this dialect has no token count endpoint, which the adapter
    /// reports as an absent count rather than as a failure.
    ///
    /// # Errors
    ///
    /// Returns an error when the protocol has a count endpoint but this request
    /// cannot be encoded for it.
    fn encode_count_tokens(&self, _call: &ResolvedCall) -> Option<Result<EncodedRequest, Error>> {
        None
    }

    /// Decodes a provider-native input token count response.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Configuration`] when this codec has no token count
    /// endpoint, and [`ErrorKind::ResponseDecode`] when the body does not match
    /// the protocol.
    fn decode_count_tokens(&self, _route: &ResolvedRoute, _value: Value) -> Result<u64, Error> {
        Err(Error::new(
            ErrorKind::Configuration,
            "this codec has no token count endpoint",
        ))
    }
}

/// Decodes one streaming provider response.
///
/// A decoder is created per stream and driven in order. It owns the
/// [`StreamAssembler`](assembler::StreamAssembler) that holds block identity,
/// the cumulative usage snapshot, and the fields of the completed response.
///
/// Implementors uphold the invariants documented on
/// [`StreamEvent`](crate::types::StreamEvent): one start, then deltas, then one
/// end per block; cumulative usage snapshots; and exactly one `Completed` event
/// for a stream that does not fail.
pub(crate) trait StreamDecoder: Send {
    /// Decodes one transport event into zero or more normalized events.
    ///
    /// # Errors
    ///
    /// Returns a classified error for a provider error event or a payload that
    /// does not match the protocol. The stream then ends without a `Completed`
    /// event.
    fn decode(&mut self, event: SseEvent) -> Result<Vec<StreamEvent>, Error>;

    /// Called once when the transport stream ends without error.
    ///
    /// A decoder whose protocol has no terminal event completes the response
    /// here. One that already completed returns no further events, because
    /// completing is idempotent.
    ///
    /// # Errors
    ///
    /// Returns a classified error when the stream ended in a state the protocol
    /// does not allow.
    fn finish(&mut self) -> Result<Vec<StreamEvent>, Error>;
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::error::Error as StdError;

    use crate::adapter::ResolvedCall;
    use crate::catalog::Catalog;
    use crate::middleware::CallContext;
    use crate::resolver::{AvailableProviders, CatalogResolver, ModelResolver, ResolvedRoute};
    use crate::types::Request;

    /// A one-provider catalog, so codec tests do not need the built-in catalog.
    const TEST_CATALOG: &str = r#"
        schema_version = 1

        [providers.alpha]
        display_name = "Alpha"
        adapter = "test-adapter"
        codec = "test-codec"
        base_url = "http://127.0.0.1"
        default_model = "one"
        auth = { type = "none" }

        [providers.alpha.models.one]
        display_name = "One"
        api_model = "alpha-one-v1"
        capabilities = { text = true, tools = true, tool_choice = { required = true, named = true } }
    "#;

    /// Resolves a request against the one-provider test catalog.
    pub(crate) fn test_call(request: Request) -> Result<ResolvedCall, Box<dyn StdError>> {
        resolved_in(TEST_CATALOG, request)
    }

    /// Resolves a request against a caller-supplied catalog layer.
    ///
    /// For codec tests that need a model the built-in catalog does not carry,
    /// such as one with a capability turned off.
    pub(crate) fn resolved_in(
        toml: &str,
        request: Request,
    ) -> Result<ResolvedCall, Box<dyn StdError>> {
        let catalog = Catalog::builder().toml_layer("test", toml)?.build()?;
        let available = AvailableProviders::all(&catalog);
        let route = CatalogResolver.resolve(&request, &catalog, &available)?;
        Ok(ResolvedCall::new(request, route, CallContext::new()))
    }

    /// The `alpha/one` route from the one-provider test catalog.
    pub(crate) fn test_route() -> Result<ResolvedRoute, Box<dyn StdError>> {
        let request = Request::builder()
            .model("alpha/one")
            .user("Hello")
            .build()?;
        Ok(test_call(request)?.route().clone())
    }

    /// Resolves a request against the built-in catalog.
    #[cfg(feature = "builtin-catalog")]
    pub(crate) fn resolved(request: Request) -> Result<ResolvedCall, Box<dyn StdError>> {
        let catalog = Catalog::builder().with_builtin().build()?;
        let available = AvailableProviders::all(&catalog);
        let route = CatalogResolver.resolve(&request, &catalog, &available)?;
        Ok(ResolvedCall::new(request, route, CallContext::new()))
    }
}
