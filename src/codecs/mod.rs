//! Wire protocol encoders and decoders.

mod assembler;
mod common;

pub(crate) mod anthropic;
#[cfg(feature = "bedrock")]
pub(crate) mod bedrock;
pub(crate) mod gemini;
pub(crate) mod openai;
pub(crate) mod openai_chat;
pub(crate) mod vercel_evaluation;

use std::sync::Arc;

use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use thiserror::Error as ThisError;

use crate::adapter::{ResolvedCall, ResolvedEvaluation};
use crate::catalog::{CodecId, codec_ids};
use crate::evaluation::Verdict;
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

/// Translates one provider evaluation protocol.
///
/// The evaluation analogue of [`Codec`]: stateless, shared across calls, and
/// owned by an adapter that only evaluates. A second implementation is
/// expected for TypeSafe's own API.
pub(crate) trait EvaluationCodec: Send + Sync {
    /// Encodes a resolved evaluation into one provider request.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidRequest`] when the evaluation asks for
    /// more than this protocol carries, before any network dispatch.
    fn encode_evaluation(&self, call: &ResolvedEvaluation) -> Result<EncodedRequest, Error>;

    /// Decodes one complete provider success body into a verdict.
    ///
    /// The codec translates shape only; the client validates every verdict's
    /// answers against the questions afterwards.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::ResponseDecode`] when the body does not match the
    /// protocol.
    fn decode_verdict(&self, call: &ResolvedEvaluation, body: Value) -> Result<Verdict, Error>;

    /// The response header that carries the provider's request id, when the
    /// protocol puts it there rather than in the body.
    ///
    /// The adapter sets [`Verdict::id`] from this header when
    /// [`decode_verdict`](Self::decode_verdict) left `id` empty.
    fn id_header(&self) -> Option<&'static str> {
        None
    }
}

/// One constructed codec, tagged with the operation family it serves.
#[derive(Clone)]
pub(crate) enum BuiltCodec {
    /// Serves `complete`, `stream`, and `count_input_tokens`.
    Generation(Arc<dyn Codec>),
    /// Serves `evaluate`.
    Evaluation(Arc<dyn EvaluationCodec>),
}

/// Why one codec could not be constructed for a provider.
#[derive(Debug, ThisError)]
pub(crate) enum CodecBuildError {
    /// The id names no codec this crate builds.
    #[error("codec {codec} is not one this crate builds")]
    UnknownCodec { codec: CodecId },
    /// The provider's `codec_options` table for this codec did not match the
    /// codec's typed options.
    #[error("codec {codec} has invalid options")]
    InvalidOptions {
        codec:  CodecId,
        #[source]
        source: serde_json::Error,
    },
}

/// Constructs the built-in codec `id` from a provider's raw `codec_options`
/// table for it.
///
/// `options` is [`Value::Null`] when the catalog declares none, which is the
/// codec's default configuration. A codec that takes no options rejects any
/// non-empty table, so a misplaced key is a build issue rather than a
/// silently ignored setting.
///
/// # Errors
///
/// [`CodecBuildError::UnknownCodec`] for an id this crate does not build and
/// [`CodecBuildError::InvalidOptions`] when the table does not match the
/// codec's typed shape.
pub(crate) fn build(id: &CodecId, options: &Value) -> Result<BuiltCodec, CodecBuildError> {
    let generation = |codec: Arc<dyn Codec>| BuiltCodec::Generation(codec);
    Ok(match id.as_str() {
        codec_ids::OPENAI_CHAT => {
            let options: OpenAiChatOptions = typed_options(id, options)?;
            generation(Arc::new(if options.base_url_is_api_root {
                openai_chat::OpenAiChatCodec::at_api_root()
            } else {
                openai_chat::OpenAiChatCodec::default()
            }))
        }
        codec_ids::OPENAI_RESPONSES => {
            let options: OpenAiResponsesOptions = typed_options(id, options)?;
            generation(Arc::new(openai::OpenAiResponsesCodec::new(
                options.mode == OpenAiMode::Codex,
            )))
        }
        codec_ids::ANTHROPIC_MESSAGES => {
            let NoOptions {} = typed_options(id, options)?;
            generation(Arc::new(anthropic::AnthropicMessagesCodec))
        }
        codec_ids::GEMINI_GENERATE => {
            let NoOptions {} = typed_options(id, options)?;
            generation(Arc::new(gemini::GeminiGenerateCodec))
        }
        #[cfg(feature = "bedrock")]
        codec_ids::BEDROCK_CONVERSE => {
            let NoOptions {} = typed_options(id, options)?;
            generation(Arc::new(bedrock::BedrockConverseCodec))
        }
        codec_ids::VERCEL_EVALUATION => {
            let NoOptions {} = typed_options(id, options)?;
            BuiltCodec::Evaluation(Arc::new(vercel_evaluation::VercelEvaluationCodec))
        }
        _ => return Err(CodecBuildError::UnknownCodec { codec: id.clone() }),
    })
}

/// Deserializes one codec's raw options into its typed shape; an absent
/// table is the default.
fn typed_options<T: Default + DeserializeOwned>(
    codec: &CodecId,
    options: &Value,
) -> Result<T, CodecBuildError> {
    if options.is_null() {
        return Ok(T::default());
    }
    serde_json::from_value(options.clone()).map_err(|source| CodecBuildError::InvalidOptions {
        codec: codec.clone(),
        source,
    })
}

/// The options shape of a codec that takes none: only an empty table passes.
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
#[expect(
    clippy::empty_structs_with_brackets,
    reason = "serde reads an empty table into a braced struct but not into a unit struct"
)]
struct NoOptions {}

/// The typed `codec_options.openai-chat` table.
///
/// Some deployments publish their own versioned API root (for example,
/// Z.ai's /api/coding/paas/v4) and must not receive an extra /v1 segment.
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct OpenAiChatOptions {
    base_url_is_api_root: bool,
}

/// The typed `codec_options.openai-responses` table.
///
/// Unknown keys are rejected so a misspelled option is a build issue for one
/// provider rather than a silently ignored setting.
#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
struct OpenAiResponsesOptions {
    #[serde(default)]
    mode: OpenAiMode,
}

/// Which OpenAI Responses deployment a provider talks to.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum OpenAiMode {
    /// The public `/v1/responses` API.
    #[default]
    Standard,
    /// The Codex deployment, which rejects several generation fields.
    ///
    /// The deployment also streams every response and reads an `originator`
    /// header; those are adapter concerns, set in `adapter_options` as
    /// `force_streaming_complete` and `identify_application`. A factory test
    /// pins that the built-in row sets both.
    Codex,
}

/// Decodes one streaming provider response.
///
/// A decoder is created per stream and driven in order. It owns the
/// [`StreamAssembler`](assembler::StreamAssembler) that holds block identity,
/// the cumulative usage snapshot, and the fields of the completed response.
///
/// Implementors uphold the invariants documented on
/// [`StreamEvent`](crate::types::StreamEvent): one start, then deltas, then one
/// end per block; cumulative usage snapshots; and exactly one `Ended` event
/// for a stream that does not fail.
pub(crate) trait StreamDecoder: Send {
    /// Decodes one transport event into zero or more normalized events.
    ///
    /// # Errors
    ///
    /// Returns a classified error for a provider error event or a payload that
    /// does not match the protocol. The stream then ends without a `Ended`
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
mod tests {
    use serde_json::json;

    use super::{BuiltCodec, CodecBuildError, build};
    use crate::catalog::{CodecId, codec_ids};

    fn is_generation(codec: &BuiltCodec) -> bool {
        matches!(codec, BuiltCodec::Generation(_))
    }

    #[test]
    fn every_builtin_generation_codec_builds_with_no_options() -> Result<(), CodecBuildError> {
        for id in [
            codec_ids::OPENAI_CHAT,
            codec_ids::OPENAI_RESPONSES,
            codec_ids::ANTHROPIC_MESSAGES,
            codec_ids::GEMINI_GENERATE,
        ] {
            let built = build(&CodecId::new(id), &serde_json::Value::Null)?;
            assert!(is_generation(&built), "{id} serves generation");
        }
        Ok(())
    }

    #[test]
    fn the_evaluation_codec_builds_as_an_evaluation_codec() -> Result<(), CodecBuildError> {
        let built = build(&CodecId::new(codec_ids::VERCEL_EVALUATION), &json!({}))?;
        assert!(matches!(built, BuiltCodec::Evaluation(_)));
        Ok(())
    }

    #[test]
    fn an_unknown_id_is_reported_by_name() {
        let error = build(&CodecId::new("test-codec"), &serde_json::Value::Null)
            .err()
            .expect("an unknown codec is refused");
        assert!(
            matches!(&error, CodecBuildError::UnknownCodec { codec } if codec.as_str() == "test-codec")
        );
    }

    #[test]
    fn options_on_a_codec_that_takes_none_are_rejected() {
        let error = build(
            &CodecId::new(codec_ids::ANTHROPIC_MESSAGES),
            &json!({ "made_up": true }),
        )
        .err()
        .expect("a stray key is refused");
        assert!(matches!(
            &error,
            CodecBuildError::InvalidOptions { codec, .. }
                if codec.as_str() == codec_ids::ANTHROPIC_MESSAGES
        ));
    }

    #[test]
    fn the_responses_codec_takes_only_the_declared_modes() {
        for options in [json!({ "mode": "turbo" }), json!({ "made_up": true })] {
            let error = build(&CodecId::new(codec_ids::OPENAI_RESPONSES), &options)
                .err()
                .expect("undeclared options are refused");
            assert!(matches!(&error, CodecBuildError::InvalidOptions { .. }));
        }
        let codex = build(
            &CodecId::new(codec_ids::OPENAI_RESPONSES),
            &json!({ "mode": "codex" }),
        );
        assert!(codex.is_ok_and(|built| is_generation(&built)));
    }

    #[test]
    fn the_chat_codec_takes_its_root_option() {
        let built = build(
            &CodecId::new(codec_ids::OPENAI_CHAT),
            &json!({ "base_url_is_api_root": true }),
        );
        assert!(built.is_ok_and(|built| is_generation(&built)));
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::error::Error as StdError;

    use crate::adapter::{ResolvedCall, ResolvedEvaluation};
    use crate::catalog::Catalog;
    use crate::evaluation::Evaluation;
    use crate::middleware::CallContext;
    use crate::resolver::{AvailableProviders, CatalogResolver, ModelResolver, ResolvedRoute};
    use crate::types::Request;

    /// A one-provider catalog, so codec tests do not need the built-in catalog.
    const TEST_CATALOG: &str = r#"
        schema_version = 1

        [providers.alpha]
        display_name = "Alpha"
        adapter = "test-adapter"
        codecs = ["test-codec"]
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

    /// Resolves an evaluation against a caller-supplied catalog layer.
    ///
    /// The route resolves from the evaluation's model selector through a
    /// stand-in request, as `Client::resolve_evaluation_route` does.
    pub(crate) fn evaluation_in(
        toml: &str,
        evaluation: Evaluation,
    ) -> Result<ResolvedEvaluation, Box<dyn StdError>> {
        let catalog = Catalog::builder().toml_layer("test", toml)?.build()?;
        let available = AvailableProviders::all(&catalog);
        let stand_in = Request::stand_in(evaluation.model(), "state".to_owned());
        let route = CatalogResolver.resolve(&stand_in, &catalog, &available)?;
        Ok(ResolvedEvaluation::new(
            evaluation,
            route,
            CallContext::new(),
        ))
    }

    /// The `alpha/one` route from the one-provider test catalog.
    pub(crate) fn test_route() -> Result<ResolvedRoute, Box<dyn StdError>> {
        let request = Request::builder()
            .model("alpha/one")
            .user("Hello")
            .build()?;
        Ok(test_call(request)?.route().clone())
    }

    /// Every opt-in built-in provider turned on, so codec tests can resolve
    /// routes on rows the catalog ships disabled.
    #[cfg(feature = "builtin-catalog")]
    pub(crate) const ENABLE_OPT_IN_PROVIDERS: &str = r"
        [providers.bedrock]
        enabled = true
        [providers.bedrock-openai]
        enabled = true
        [providers.fireworks]
        enabled = true
        [providers.litellm]
        enabled = true
        [providers.modal]
        enabled = true
        [providers.ollama]
        enabled = true
        [providers.openrouter]
        enabled = true
    ";

    /// Resolves a request against the built-in catalog with every provider
    /// enabled.
    #[cfg(feature = "builtin-catalog")]
    pub(crate) fn resolved(request: Request) -> Result<ResolvedCall, Box<dyn StdError>> {
        let catalog = Catalog::builder()
            .with_builtin()
            .toml_layer("enable-opt-in", ENABLE_OPT_IN_PROVIDERS)?
            .build()?;
        let available = AvailableProviders::all(&catalog);
        let route = CatalogResolver.resolve(&request, &catalog, &available)?;
        Ok(ResolvedCall::new(request, route, CallContext::new()))
    }
}
