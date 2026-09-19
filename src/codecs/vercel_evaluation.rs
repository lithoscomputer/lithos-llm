//! The Vercel AI Gateway evaluation protocol.
//!
//! This codec speaks `POST /v4/ai/evaluation-model`, the path the Vercel AI
//! SDK's gateway provider uses for evaluation models such as TypeSafe's Jev.
//!
//! # URL
//!
//! The provider's `base_url` names the gateway's generation mount, which ends
//! in `/v1`. That trailing `/v1` (with or without a trailing slash) is
//! replaced by `/v4/ai`, and `/evaluation-model` is appended. A base URL
//! without a trailing `/v1` gets `/v4/ai/evaluation-model` appended after
//! any trailing slash is trimmed. So `https://ai-gateway.vercel.sh/v1` and
//! `http://127.0.0.1:3927` become
//! `https://ai-gateway.vercel.sh/v4/ai/evaluation-model` and
//! `http://127.0.0.1:3927/v4/ai/evaluation-model`.
//!
//! # Headers
//!
//! Every request carries `ai-gateway-protocol-version: 0.0.1`,
//! `ai-evaluation-model-specification-version: 4`, and `ai-model-id` set to
//! the route's API model. Authentication comes from the provider's `auth`
//! scheme through the transport, as for every other adapter.
//!
//! # Metadata lifting
//!
//! The success body's `providerMetadata` is split three ways. Fields this
//! crate models are lifted onto the [`Verdict`]: `gateway.generationId` is
//! [`Verdict::id`], `gateway.cost` (a decimal string in USD) is
//! [`Verdict::cost`] with [`CostSource::Provider`], and each entry of
//! `typesafe.confidence` lands on the matching choice or score answer's
//! `confidence`. Those three are removed, a namespace object that became
//! empty is dropped, and whatever remains is [`Verdict::provider_metadata`].
//! The whole body is kept in [`Verdict::raw`] for the response policy to
//! keep or drop.
//!
//! The codec translates shape only. Sums, ranges, kinds, and missing answers
//! are checked by the client's verdict validator after the adapter returns.

use std::collections::BTreeMap;

use reqwest::Method;
use serde_json::{Map, Value};

use super::EvaluationCodec;
use super::common::usd_micros;
use super::evaluation_common::{
    MAX_LEVELS, QuestionLimits, answer_object, decode_choice, decode_failure, decode_score,
    encode_question, encode_state,
};
use crate::adapter::ResolvedEvaluation;
use crate::evaluation::{
    Answer, BooleanAnswer, Evaluation, Question, QuestionId, Rounding, Verdict,
};
use crate::resolver::ResolvedRoute;
use crate::transport::EncodedRequest;
use crate::types::{Cost, CostSource, Error, TokenCounts, Warning};

/// The gateway path every evaluation request posts to.
const EVALUATION_PATH: &str = "/v4/ai/evaluation-model";

/// The generation mount a gateway base URL ends with, which the evaluation
/// path replaces.
const GENERATION_MOUNT: &str = "/v1";

const PROTOCOL_VERSION_HEADER: &str = "ai-gateway-protocol-version";
const PROTOCOL_VERSION: &str = "0.0.1";
const SPECIFICATION_VERSION_HEADER: &str = "ai-evaluation-model-specification-version";
const SPECIFICATION_VERSION: &str = "4";
const MODEL_ID_HEADER: &str = "ai-model-id";

/// The most options a choice question may carry, per Jev's documentation.
const MAX_OPTIONS: usize = 255;

/// The ceilings this protocol enforces before dispatch.
const LIMITS: QuestionLimits = QuestionLimits {
    max_options: Some(MAX_OPTIONS),
    max_levels:  MAX_LEVELS,
};

/// The type word of a yes/no question on the gateway.
const BOOLEAN_TYPE: &str = "boolean";

/// The `providerOptions` and `providerMetadata` namespace of the gateway
/// itself, which the route's own provider id maps to.
const GATEWAY_NAMESPACE: &str = "gateway";

/// The `providerMetadata` namespace TypeSafe's confidence arrives under.
const TYPESAFE_NAMESPACE: &str = "typesafe";

/// The `vercel-evaluation` codec.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VercelEvaluationCodec;

impl EvaluationCodec for VercelEvaluationCodec {
    fn encode_evaluation(&self, call: &ResolvedEvaluation) -> Result<EncodedRequest, Error> {
        let route = call.route();
        let evaluation = call.evaluation();

        let mut questions = Map::new();
        for (id, question) in evaluation.questions() {
            questions.insert(
                id.as_str().to_owned(),
                encode_question(route, id, question, BOOLEAN_TYPE, LIMITS)?,
            );
        }

        let mut body = Map::new();
        body.insert("state".to_owned(), encode_state(evaluation.state()));
        body.insert("questions".to_owned(), Value::Object(questions));
        if let Some(options) = encode_provider_options(route, evaluation) {
            body.insert("providerOptions".to_owned(), options);
        }

        let mut encoded = EncodedRequest::new(
            Method::POST,
            endpoint(route.provider().base_url()),
            Value::Object(body),
        )
        .with_headers(vec![
            (
                PROTOCOL_VERSION_HEADER.to_owned(),
                PROTOCOL_VERSION.to_owned(),
            ),
            (
                SPECIFICATION_VERSION_HEADER.to_owned(),
                SPECIFICATION_VERSION.to_owned(),
            ),
            (MODEL_ID_HEADER.to_owned(), route.api_model().to_owned()),
        ]);
        // The protocol has no field for free-form metadata, so the entries
        // are dropped and the caller is told.
        if !evaluation.metadata().is_empty() {
            encoded = encoded.unsupported_control("evaluation metadata");
        }
        Ok(encoded)
    }

    fn decode_verdict(&self, call: &ResolvedEvaluation, body: Value) -> Result<Verdict, Error> {
        match decode_body(call, &body) {
            Ok(mut verdict) => {
                verdict.raw = Some(body);
                Ok(verdict)
            }
            Err(detail) => Err(decode_failure(call.route(), &detail, body)),
        }
    }
}

/// Joins the gateway's evaluation path onto a provider base URL.
///
/// See the module documentation for the rule.
fn endpoint(base_url: &str) -> String {
    let base = base_url.trim_end_matches('/');
    let root = base.strip_suffix(GENERATION_MOUNT).unwrap_or(base);
    format!("{root}{EVALUATION_PATH}")
}

/// The `providerOptions` object, or `None` when the evaluation carries no
/// provider options.
///
/// The namespace of the route's own provider is the gateway's, so it goes
/// under `gateway`. Any other namespace passes through under its own key.
fn encode_provider_options(route: &ResolvedRoute, evaluation: &Evaluation) -> Option<Value> {
    let options = evaluation.provider_options();
    if options.is_empty() {
        return None;
    }
    let mut encoded = Map::new();
    for (namespace, values) in options {
        let key = if namespace == route.provider().id() {
            GATEWAY_NAMESPACE
        } else {
            namespace.as_str()
        };
        encoded.insert(key.to_owned(), Value::Object(values.clone()));
    }
    Some(Value::Object(encoded))
}

/// Decodes everything but `raw`. Errors are the detail of the failure, which
/// the caller turns into one
/// [`ResponseDecode`](crate::types::ErrorKind::ResponseDecode) carrying the
/// body.
fn decode_body(call: &ResolvedEvaluation, body: &Value) -> Result<Verdict, String> {
    let route = call.route();
    let questions = call.evaluation().questions();

    let wire_answers = body
        .get("answers")
        .and_then(Value::as_object)
        .ok_or_else(|| "returned no answers object".to_owned())?;
    let mut answers = BTreeMap::new();
    for (id, wire) in wire_answers {
        let question = questions.get(id.as_str());
        answers.insert(QuestionId::new(id), decode_answer(id, question, wire)?);
    }

    let mut metadata = provider_metadata(body);
    let (id, cost) = lift_gateway_fields(&mut metadata);
    lift_confidence(&mut metadata, &mut answers);
    metadata.retain(|_, value| !matches!(value, Value::Object(object) if object.is_empty()));

    let mut verdict = Verdict::new(
        route.provider().id().clone(),
        route.model().id().clone(),
        answers,
    );
    verdict.id = id;
    verdict.cost = cost;
    verdict.rounding = decode_rounding(body.get("rounding"))?;
    verdict.usage = TokenCounts {
        input: body
            .pointer("/usage/inputTokens")
            .and_then(Value::as_u64)
            .unwrap_or_default(),
        output: body
            .pointer("/usage/outputTokens")
            .and_then(Value::as_u64)
            .unwrap_or_default(),
        ..TokenCounts::default()
    };
    verdict.warnings = body
        .get("warnings")
        .and_then(Value::as_array)
        .map(|warnings| warnings.iter().map(decode_warning).collect())
        .unwrap_or_default();
    verdict.provider_metadata = metadata;
    Ok(verdict)
}

fn decode_answer(id: &str, question: Option<&Question>, wire: &Value) -> Result<Answer, String> {
    let object = answer_object(id, wire)?;
    match object.get("type").and_then(Value::as_str) {
        Some("choice") => Ok(Answer::Choice(decode_choice(id, question, object)?)),
        Some("score") => Ok(Answer::Score(decode_score(id, question, object)?)),
        Some(BOOLEAN_TYPE) => {
            let probability = object
                .get("probability")
                .and_then(Value::as_f64)
                .ok_or_else(|| format!("answered boolean question `{id}` without a probability"))?;
            Ok(Answer::Boolean(BooleanAnswer { probability }))
        }
        Some(other) => Err(format!(
            "answered question `{id}` with unknown answer type `{other}`"
        )),
        None => Err(format!("answered question `{id}` without an answer type")),
    }
}

fn decode_rounding(wire: Option<&Value>) -> Result<Option<Rounding>, String> {
    let Some(wire) = wire.filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    Ok(Some(Rounding {
        probability_decimals: decimals(wire, "probabilityDecimals")?,
        score_decimals:       decimals(wire, "scoreDecimals")?,
    }))
}

fn decimals(rounding: &Value, key: &str) -> Result<Option<u8>, String> {
    match rounding.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .and_then(|decimals| u8::try_from(decimals).ok())
            .map(Some)
            .ok_or_else(|| format!("reported rounding `{key}` that is not a small whole number")),
    }
}

/// One SDK warning as this crate's [`Warning`].
///
/// The SDK shapes are `unsupported` and `compatibility` (a `feature` and
/// optional `details`), `deprecated` (a `setting` and a `message`), and
/// `other` (a `message`). Anything else keeps its JSON as the message under
/// the code `unknown`, because a warning is never a reason to fail.
fn decode_warning(wire: &Value) -> Warning {
    let text = |key: &str| wire.get(key).and_then(Value::as_str);
    let code = text("type");
    let message = match code {
        Some("unsupported" | "compatibility") => text("feature").map(|feature| {
            text("details").map_or_else(
                || feature.to_owned(),
                |details| format!("{feature}: {details}"),
            )
        }),
        Some("deprecated") => text("setting")
            .zip(text("message"))
            .map(|(setting, message)| format!("{setting}: {message}")),
        Some("other") => text("message").map(ToOwned::to_owned),
        _ => None,
    };
    match (code, message) {
        (Some(code), Some(message)) => Warning {
            code: code.to_owned(),
            message,
        },
        _ => Warning {
            code:    "unknown".to_owned(),
            message: wire.to_string(),
        },
    }
}

/// The `providerMetadata` object as namespace to value.
fn provider_metadata(body: &Value) -> BTreeMap<String, Value> {
    body.get("providerMetadata")
        .and_then(Value::as_object)
        .map(|object| {
            object
                .iter()
                .map(|(namespace, value)| (namespace.clone(), value.clone()))
                .collect()
        })
        .unwrap_or_default()
}

/// Takes `generationId` and `cost` out of the `gateway` namespace.
fn lift_gateway_fields(metadata: &mut BTreeMap<String, Value>) -> (Option<String>, Option<Cost>) {
    let Some(Value::Object(gateway)) = metadata.get_mut(GATEWAY_NAMESPACE) else {
        return (None, None);
    };
    let id = gateway
        .remove("generationId")
        .and_then(|value| value.as_str().map(ToOwned::to_owned));
    let cost = gateway
        .remove("cost")
        .and_then(|value| provider_cost(&value));
    (id, cost)
}

/// Moves each `typesafe.confidence` entry onto its choice or score answer.
///
/// An entry for a boolean or for an unknown question has nowhere to go and
/// is dropped.
fn lift_confidence(
    metadata: &mut BTreeMap<String, Value>,
    answers: &mut BTreeMap<QuestionId, Answer>,
) {
    let Some(Value::Object(typesafe)) = metadata.get_mut(TYPESAFE_NAMESPACE) else {
        return;
    };
    let Some(Value::Object(confidence)) = typesafe.remove("confidence") else {
        return;
    };
    for (id, value) in confidence {
        let Some(confidence) = value.as_f64() else {
            continue;
        };
        match answers.get_mut(id.as_str()) {
            Some(Answer::Choice(answer)) => answer.confidence = Some(confidence),
            Some(Answer::Score(answer)) => answer.confidence = Some(confidence),
            _ => {}
        }
    }
}

/// The gateway's cost, a decimal string in USD, as this crate's cost.
///
/// A number is taken too. Anything that is not a finite, non-negative amount
/// is ignored rather than refused, because a cost is accounting, not the
/// answer.
fn provider_cost(value: &Value) -> Option<Cost> {
    value
        .as_str()
        .and_then(|text| text.trim().parse::<f64>().ok())
        .or_else(|| value.as_f64())
        .filter(|usd| usd.is_finite() && *usd >= 0.0)
        .map(|usd| Cost {
            usd_micros: usd_micros(usd),
            source:     CostSource::Provider,
        })
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use indexmap::IndexMap;
    use serde_json::{Value, json};

    use super::{EvaluationCodec, VercelEvaluationCodec, endpoint};
    use crate::adapter::ResolvedEvaluation;
    use crate::codecs::test_support::evaluation_in;
    use crate::evaluation::{Evaluation, Question, Rounding, State, Verdict};
    use crate::types::{Cost, CostSource, Error, ErrorKind, TokenCounts, Warning};

    const GATEWAY_BASE_URL: &str = "https://ai-gateway.vercel.sh/v1";

    /// A real 200 body from the gateway for [`three_questions`], recorded on
    /// 2026-09-17.
    const LIVE_BODY: &str = r#"{"answers":{"department":{"type":"choice","choice":"billing","probabilities":{"other":0,"technical":0,"billing":1}},"severity":{"type":"score","score":1.05,"probabilities":{"0":0.13,"1":0.69,"2":0.18}},"requests_refund":{"type":"boolean","probability":0.99}},"rounding":{"probabilityDecimals":2,"scoreDecimals":2},"usage":{"inputTokens":389,"outputTokens":70},"warnings":[],"providerMetadata":{"typesafe":{"confidence":{"department":1,"severity":0.54}},"gateway":{"routing":{"originalModelId":"typesafe-ai/jev","resolvedProvider":"typesafe-ai","fallbacksAvailable":[],"planningReasoning":"System credentials planned for: typesafe-ai. Total execution order: typesafe-ai(system)","canonicalSlug":"typesafe-ai/jev","finalProvider":"typesafe-ai","modelAttemptCount":1,"modelAttempts":[{"canonicalSlug":"typesafe-ai/jev","success":true,"providerAttemptCount":1,"providerAttempts":[{"provider":"typesafe-ai","credentialType":"system","success":true,"startTime":1789687661182,"endTime":1789687661468,"statusCode":200}]}],"totalProviderAttemptCount":1},"cost":"0.000016338","marketCost":"0.000016338","surchargeCost":"0","gatewayCost":"0.000016338","inferenceCost":"0.000016338","inputInferenceCost":"0.000016338","outputInferenceCost":"0","generationId":"gen_01M2RV50ENRGT4K2CHJ0WC4N1H"}}}"#;

    /// A one-provider catalog shaped like the built-in `vercel` provider,
    /// with the Jev row priced so a catalog estimate exists.
    fn catalog(base_url: &str) -> String {
        format!(
            r#"
            schema_version = 1

            [providers.vercel]
            display_name = "Vercel"
            codecs = ["vercel-evaluation"]
            base_url = "{base_url}"
            default_model = "jev"
            auth = {{ type = "bearer" }}

            [providers.vercel.models.jev]
            display_name = "Jev"
            api_model = "typesafe-ai/jev"
            capabilities = {{ evaluation = {{ choice = true, score = true, boolean = true }} }}
            pricing = {{ input_usd_micros_per_million = 1000000, output_usd_micros_per_million = 0 }}
            "#
        )
    }

    fn resolved(evaluation: Evaluation) -> Result<ResolvedEvaluation, Box<dyn StdError>> {
        evaluation_in(&catalog(GATEWAY_BASE_URL), evaluation)
    }

    fn resolved_at(
        base_url: &str,
        evaluation: Evaluation,
    ) -> Result<ResolvedEvaluation, Box<dyn StdError>> {
        evaluation_in(&catalog(base_url), evaluation)
    }

    /// The interface plan's three questions, which [`LIVE_BODY`] answers.
    fn three_questions() -> Result<Evaluation, Box<dyn StdError>> {
        Ok(Evaluation::builder()
            .model("vercel/jev")
            .state("I was charged twice. Please refund the duplicate.")
            .choice("department", "Which team should handle this?", [
                ("billing", Some("Charges and refunds")),
                ("technical", Some("Bugs and outages")),
                ("other", None),
            ])
            .score("severity", "How severe is the issue?", [
                "Cosmetic",
                "Workaround exists",
                "Blocking; no workaround",
            ])
            .boolean("requests_refund", "Is the customer requesting money back?")
            .build()?)
    }

    /// Every shape the encoder has a rule for: a `None` option description, a
    /// `None` level, boolean criteria on one side only, a JSON state, and
    /// provider options for the gateway and for another namespace.
    fn mixed_evaluation() -> Result<Evaluation, Box<dyn StdError>> {
        Ok(Evaluation::builder()
            .model("vercel/jev")
            .state(json!({ "ticket": 7, "text": "Charged twice" }))
            .choice("department", "Which team?", [
                ("billing", Some("Charges and refunds")),
                ("other", None),
            ])
            .score("severity", "How severe?", [
                Some("Cosmetic"),
                None,
                Some("Blocking"),
            ])
            .question("requests_refund", Question::Boolean {
                instructions: State::from("Refund requested?"),
                when_true:    Some(State::from("Asks for money back")),
                when_false:   None,
            })
            .provider_option("vercel", "order", json!(["typesafe-ai"]))
            .provider_option("typesafe", "version", json!("jev-1.13.0"))
            .build()?)
    }

    fn decode(body: Value) -> Result<Verdict, Error> {
        let call = resolved(three_questions().expect("the fixture evaluation builds"))
            .expect("the fixture evaluation resolves");
        VercelEvaluationCodec.decode_verdict(&call, body)
    }

    fn live_body() -> Value {
        serde_json::from_str(LIVE_BODY).expect("the live body is JSON")
    }

    #[test]
    fn the_endpoint_replaces_a_v1_mount_and_extends_a_bare_host() {
        assert_eq!(
            endpoint("https://ai-gateway.vercel.sh/v1"),
            "https://ai-gateway.vercel.sh/v4/ai/evaluation-model"
        );
        assert_eq!(
            endpoint("https://ai-gateway.vercel.sh/v1/"),
            "https://ai-gateway.vercel.sh/v4/ai/evaluation-model"
        );
        assert_eq!(
            endpoint("http://127.0.0.1:3927"),
            "http://127.0.0.1:3927/v4/ai/evaluation-model"
        );
        assert_eq!(
            endpoint("http://127.0.0.1:3927/"),
            "http://127.0.0.1:3927/v4/ai/evaluation-model"
        );
    }

    #[test]
    fn encodes_the_url_from_both_base_url_shapes() -> Result<(), Box<dyn StdError>> {
        let mounted = VercelEvaluationCodec.encode_evaluation(&resolved(three_questions()?)?)?;
        assert_eq!(
            mounted.url,
            "https://ai-gateway.vercel.sh/v4/ai/evaluation-model"
        );

        let bare = VercelEvaluationCodec
            .encode_evaluation(&resolved_at("http://127.0.0.1:3927", three_questions()?)?)?;
        assert_eq!(bare.url, "http://127.0.0.1:3927/v4/ai/evaluation-model");
        Ok(())
    }

    #[test]
    fn encodes_exactly_the_three_protocol_headers() -> Result<(), Box<dyn StdError>> {
        let encoded = VercelEvaluationCodec.encode_evaluation(&resolved(three_questions()?)?)?;

        assert_eq!(encoded.headers, vec![
            ("ai-gateway-protocol-version".to_owned(), "0.0.1".to_owned()),
            (
                "ai-evaluation-model-specification-version".to_owned(),
                "4".to_owned()
            ),
            ("ai-model-id".to_owned(), "typesafe-ai/jev".to_owned()),
        ]);
        assert!(encoded.warnings.is_empty(), "{:?}", encoded.warnings);
        Ok(())
    }

    #[test]
    fn encodes_the_mixed_evaluation_body() -> Result<(), Box<dyn StdError>> {
        let encoded = VercelEvaluationCodec.encode_evaluation(&resolved(mixed_evaluation()?)?)?;

        assert_eq!(
            encoded.body,
            json!({
                "state": { "ticket": 7, "text": "Charged twice" },
                "questions": {
                    "department": {
                        "type": "choice",
                        "instructions": "Which team?",
                        "criteria": { "billing": "Charges and refunds", "other": null }
                    },
                    "severity": {
                        "type": "score",
                        "instructions": "How severe?",
                        "criteria": ["Cosmetic", null, "Blocking"]
                    },
                    "requests_refund": {
                        "type": "boolean",
                        "instructions": "Refund requested?",
                        "criteria": { "true": "Asks for money back" }
                    }
                },
                "providerOptions": {
                    "gateway": { "order": ["typesafe-ai"] },
                    "typesafe": { "version": "jev-1.13.0" }
                }
            })
        );
        Ok(())
    }

    #[test]
    fn a_boolean_without_criteria_and_no_provider_options_omit_their_keys()
    -> Result<(), Box<dyn StdError>> {
        let encoded = VercelEvaluationCodec.encode_evaluation(&resolved(three_questions()?)?)?;

        assert!(encoded.body.get("providerOptions").is_none());
        assert_eq!(
            encoded.body["questions"]["requests_refund"],
            json!({
                "type": "boolean",
                "instructions": "Is the customer requesting money back?"
            })
        );
        Ok(())
    }

    #[test]
    fn metadata_is_dropped_with_a_warning() -> Result<(), Box<dyn StdError>> {
        let evaluation = three_questions()?
            .into_builder()
            .metadata_entry("tenant", "acme")
            .build()?;

        let encoded = VercelEvaluationCodec.encode_evaluation(&resolved(evaluation)?)?;

        assert!(encoded.body.get("metadata").is_none());
        assert_eq!(encoded.warnings.len(), 1);
        assert_eq!(encoded.warnings[0].code, "unsupported_control");
        Ok(())
    }

    fn choice_with_options(count: usize) -> Result<Evaluation, Box<dyn StdError>> {
        let options: IndexMap<String, Option<State>> = (0..count)
            .map(|index| (format!("option{index}"), None))
            .collect();
        Ok(Evaluation::builder()
            .model("vercel/jev")
            .state("state")
            .question("wide", Question::Choice {
                instructions: State::from("Pick one"),
                options,
            })
            .build()?)
    }

    fn score_with_levels(count: usize) -> Result<Evaluation, Box<dyn StdError>> {
        Ok(Evaluation::builder()
            .model("vercel/jev")
            .state("state")
            .score("tall", "Rate it", vec![Option::<&str>::None; count])
            .build()?)
    }

    #[test]
    fn refuses_a_choice_past_255_options_before_dispatch() -> Result<(), Box<dyn StdError>> {
        VercelEvaluationCodec.encode_evaluation(&resolved(choice_with_options(255)?)?)?;

        let error = VercelEvaluationCodec
            .encode_evaluation(&resolved(choice_with_options(256)?)?)
            .err()
            .ok_or("256 options exceed the protocol's maximum")?;

        assert_eq!(error.kind(), ErrorKind::InvalidRequest);
        assert_eq!(error.provider_code(), Some("invalid_evaluation"));
        assert!(error.message().contains("`wide`"), "{}", error.message());
        Ok(())
    }

    #[test]
    fn refuses_a_score_past_10_levels_before_dispatch() -> Result<(), Box<dyn StdError>> {
        VercelEvaluationCodec.encode_evaluation(&resolved(score_with_levels(10)?)?)?;

        let error = VercelEvaluationCodec
            .encode_evaluation(&resolved(score_with_levels(11)?)?)
            .err()
            .ok_or("11 levels exceed the protocol's maximum")?;

        assert_eq!(error.kind(), ErrorKind::InvalidRequest);
        assert_eq!(error.provider_code(), Some("invalid_evaluation"));
        assert!(error.message().contains("`tall`"), "{}", error.message());
        Ok(())
    }

    #[test]
    fn decodes_the_live_body_into_a_typed_verdict() -> Result<(), Box<dyn StdError>> {
        let verdict = decode(live_body())?;

        let department = verdict.choice("department")?;
        assert_eq!(department.choice, "billing");
        let probabilities = department
            .probabilities
            .as_ref()
            .ok_or("the choice carries a distribution")?;
        assert_eq!(
            probabilities.iter().collect::<Vec<_>>(),
            [
                (&"billing".to_owned(), &1.0),
                (&"technical".to_owned(), &0.0),
                (&"other".to_owned(), &0.0),
            ],
            "probabilities follow the question's option order, not the wire's"
        );
        assert_eq!(department.confidence, Some(1.0));

        let severity = verdict.score("severity")?;
        assert!(
            (severity.score - 1.05).abs() < f64::EPSILON,
            "{}",
            severity.score
        );
        assert_eq!(severity.probabilities, Some(vec![0.13, 0.69, 0.18]));
        assert_eq!(severity.confidence, Some(0.54));

        let probability = verdict.boolean("requests_refund")?.probability;
        assert!((probability - 0.99).abs() < f64::EPSILON, "{probability}");

        assert_eq!(
            verdict.rounding,
            Some(Rounding {
                probability_decimals: Some(2),
                score_decimals:       Some(2),
            })
        );
        assert_eq!(verdict.usage, TokenCounts {
            input: 389,
            output: 70,
            ..TokenCounts::default()
        });
        assert_eq!(
            verdict.cost,
            Some(Cost {
                usd_micros: 16,
                source:     CostSource::Provider,
            })
        );
        assert_eq!(
            verdict.id.as_deref(),
            Some("gen_01M2RV50ENRGT4K2CHJ0WC4N1H")
        );
        assert_eq!(verdict.served_by, None);
        assert_eq!(verdict.model.to_string(), "vercel/jev");
        assert!(verdict.warnings.is_empty());
        assert_eq!(verdict.raw, Some(live_body()));

        let gateway = verdict
            .provider_metadata
            .get("gateway")
            .ok_or("the gateway namespace survives")?;
        assert!(gateway.get("routing").is_some());
        assert_eq!(gateway["marketCost"], json!("0.000016338"));
        assert!(gateway.get("cost").is_none(), "cost was lifted");
        assert!(
            gateway.get("generationId").is_none(),
            "generationId was lifted"
        );
        assert!(
            !verdict.provider_metadata.contains_key("typesafe"),
            "an emptied namespace is dropped: {:?}",
            verdict.provider_metadata
        );
        Ok(())
    }

    #[test]
    fn answers_without_distributions_decode_bare() -> Result<(), Box<dyn StdError>> {
        let verdict = decode(json!({
            "answers": {
                "department": { "type": "choice", "choice": "other" },
                "severity": { "type": "score", "score": 2 },
                "requests_refund": { "type": "boolean", "probability": 0.5 }
            }
        }))?;

        assert_eq!(verdict.choice("department")?.probabilities, None);
        assert_eq!(verdict.score("severity")?.probabilities, None);
        assert_eq!(verdict.rounding, None);
        assert_eq!(verdict.usage, TokenCounts::default());
        assert_eq!(verdict.cost, None, "the adapter fills the catalog estimate");
        assert!(verdict.provider_metadata.is_empty());
        Ok(())
    }

    #[test]
    fn rejects_each_shape_the_verdict_cannot_represent() {
        let cases: [(&str, Value); 5] = [
            ("no answers", json!({ "rounding": {} })),
            (
                "unknown answer type",
                json!({ "answers": { "department": { "type": "ranking", "order": [] } } }),
            ),
            (
                "choice probabilities with a foreign key",
                json!({ "answers": { "department": {
                    "type": "choice", "choice": "billing",
                    "probabilities": { "billing": 1, "technical": 0, "other": 0, "legal": 0 }
                } } }),
            ),
            (
                "score probabilities with a missing index",
                json!({ "answers": { "severity": {
                    "type": "score", "score": 1,
                    "probabilities": { "0": 0.5, "2": 0.5 }
                } } }),
            ),
            (
                "rounding out of u8",
                json!({
                    "answers": { "requests_refund": { "type": "boolean", "probability": 1 } },
                    "rounding": { "probabilityDecimals": 300 }
                }),
            ),
        ];
        for (name, body) in cases {
            let error = match decode(body.clone()) {
                Err(error) => error,
                Ok(verdict) => panic!("{name} should fail to decode, got {verdict:?}"),
            };
            assert_eq!(error.kind(), ErrorKind::ResponseDecode, "{name}");
            assert_eq!(
                error.provider().map(ToString::to_string).as_deref(),
                Some("vercel"),
                "{name}"
            );
            assert_eq!(error.raw_data(), Some(&body), "{name}");
            assert!(
                !error.is_retryable(),
                "{name}: a malformed answer is never retried"
            );
        }
    }

    #[test]
    fn maps_every_sdk_warning_shape_and_keeps_an_unknown_one() -> Result<(), Box<dyn StdError>> {
        let verdict = decode(json!({
            "answers": {},
            "warnings": [
                { "type": "unsupported", "feature": "temperature" },
                { "type": "compatibility", "feature": "seed", "details": "ignored by jev" },
                { "type": "deprecated", "setting": "mode", "message": "use `order`" },
                { "type": "other", "message": "the model was cold" },
                { "kind": "novel", "value": 1 }
            ]
        }))?;

        assert_eq!(verdict.warnings, vec![
            Warning {
                code:    "unsupported".to_owned(),
                message: "temperature".to_owned(),
            },
            Warning {
                code:    "compatibility".to_owned(),
                message: "seed: ignored by jev".to_owned(),
            },
            Warning {
                code:    "deprecated".to_owned(),
                message: "mode: use `order`".to_owned(),
            },
            Warning {
                code:    "other".to_owned(),
                message: "the model was cold".to_owned(),
            },
            Warning {
                code:    "unknown".to_owned(),
                message: json!({ "kind": "novel", "value": 1 }).to_string(),
            },
        ]);
        Ok(())
    }

    #[test]
    fn a_non_numeric_cost_is_ignored_and_the_rest_of_the_gateway_metadata_stays()
    -> Result<(), Box<dyn StdError>> {
        let verdict = decode(json!({
            "answers": {},
            "providerMetadata": { "gateway": { "cost": "free", "generationId": "gen_1" } }
        }))?;

        assert_eq!(verdict.cost, None);
        assert_eq!(verdict.id.as_deref(), Some("gen_1"));
        assert!(
            !verdict.provider_metadata.contains_key("gateway"),
            "cost and generationId were lifted and the namespace emptied"
        );
        Ok(())
    }

    #[test]
    fn confidence_for_a_boolean_or_an_unknown_question_is_dropped() -> Result<(), Box<dyn StdError>>
    {
        let verdict = decode(json!({
            "answers": {
                "department": { "type": "choice", "choice": "other" },
                "requests_refund": { "type": "boolean", "probability": 0.5 }
            },
            "providerMetadata": {
                "typesafe": { "confidence": { "department": 0.7, "requests_refund": 0.9, "nobody": 0.1 }, "model": "jev-1.13.0" }
            }
        }))?;

        assert_eq!(verdict.choice("department")?.confidence, Some(0.7));
        assert_eq!(
            verdict.provider_metadata.get("typesafe"),
            Some(&json!({ "model": "jev-1.13.0" })),
            "confidence is lifted; the rest of the namespace stays"
        );
        Ok(())
    }
}
