//! TypeSafe's System One evaluation protocol.
//!
//! This codec speaks the API that answers TypeSafe's Jev models directly,
//! in two dialects that share one body and one answer shape:
//!
//! - [`Dialect::TypeSafe`] posts to `POST /v1/systemone` on TypeSafe's own host
//!   (`https://api.typesafe.ai`). The request id comes back in the
//!   `x-typesafe-request-id` response header, which the adapter reads through
//!   [`EvaluationCodec::id_header`].
//! - [`Dialect::OpenRouter`] posts to `POST /alpha/decisions` on OpenRouter's
//!   API root (`https://openrouter.ai/api`), adds OpenRouter's optional
//!   `session_id`, `user`, `trace`, and `provider` fields from the caller's
//!   provider options, and reads back the `id`, `provider`, and
//!   `usage.cost` the response adds.
//!
//! The dialect is a fact about the host, so it is a codec option:
//! `codec_options = { systemone = { dialect = "openrouter" } }`. The default
//! is TypeSafe.
//!
//! # URL
//!
//! For TypeSafe the operation path is `/v1/systemone`, joined with
//! [`options::endpoint`](super::options::endpoint) so a base URL that already
//! ends in `/v1` keeps that segment once: `https://api.typesafe.ai/v1` and
//! `http://127.0.0.1:3928` become `https://api.typesafe.ai/v1/systemone` and
//! `http://127.0.0.1:3928/v1/systemone`. For OpenRouter a trailing `/v1`,
//! which is the Chat Completions mount OpenRouter's catalog row names, is
//! dropped and `/alpha/decisions` appended, so the same provider row can
//! list both codecs.
//!
//! # Body
//!
//! `{model, state, questions}`: the route's API model, the state as a bare
//! JSON value, and the SDK's question shapes with `noul` as the yes/no type
//! word and `criteria` for options and levels. Neither dialect takes
//! free-form metadata; TypeSafe takes no provider options at all, and each
//! drops the field with an `unsupported_control` warning.
//!
//! # Answers
//!
//! `answers` is keyed by question id. A `choice` answer carries `choice`,
//! `confidence`, and option-keyed `probabilities`; a `score` answer carries
//! `score`, `confidence`, a `legend` mapping stringified level index to the
//! level's description, and index-keyed `probabilities`; a `noul` answer
//! carries the probability under `noul`. The legend is checked against the
//! question's levels and then dropped, because the verdict already knows
//! its levels. `model` names the versioned model that answered and becomes
//! [`Verdict::served_by`]. The API declares no rounding but answers to two
//! decimals, so the codec declares 2/2, as the Vercel AI SDK's native
//! TypeSafe adapter does. The whole body stays in [`Verdict::raw`].
//!
//! The codec translates shape only. Sums, ranges, kinds, and missing answers
//! are checked by the client's verdict validator after the adapter returns.

use std::collections::BTreeMap;

use reqwest::Method;
use serde::Deserialize;
use serde_json::{Map, Value};

use super::EvaluationCodec;
use super::evaluation_common::{
    MAX_LEVELS, QuestionLimits, answer_object, decode_choice, decode_failure, decode_score,
    encode_description, encode_question, encode_state,
};
use super::options::{endpoint, usd_micros};
use crate::adapter::ResolvedEvaluation;
use crate::evaluation::{
    Answer, BooleanAnswer, Evaluation, Question, QuestionId, Rounding, State, Verdict,
};
use crate::resolver::ResolvedRoute;
use crate::transport::EncodedRequest;
use crate::types::{Cost, CostSource, Error, TokenCounts};

/// TypeSafe's operation path, versioned by the codec.
const TYPESAFE_PATH: &str = "/v1/systemone";

/// OpenRouter's operation path under its API root.
const OPENROUTER_PATH: &str = "/alpha/decisions";

/// The Chat Completions mount an OpenRouter base URL may end with, which
/// the decisions path does not sit under.
const OPENROUTER_CHAT_MOUNT: &str = "/v1";

/// The response header TypeSafe puts the request id in.
const TYPESAFE_REQUEST_ID_HEADER: &str = "x-typesafe-request-id";

/// The type word of a yes/no question, and the key its answer arrives under.
const NOUL: &str = "noul";

/// The ceilings this protocol enforces before dispatch: TypeSafe documents
/// two to ten score levels and no option maximum.
const LIMITS: QuestionLimits = QuestionLimits {
    max_options: None,
    max_levels:  MAX_LEVELS,
};

/// The provider-option keys the OpenRouter dialect forwards on the body.
const OPENROUTER_EXTRAS: [&str; 4] = ["session_id", "user", "trace", "provider"];

/// Which host's spelling of the protocol a provider speaks.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
pub(crate) enum Dialect {
    /// TypeSafe's own API.
    #[default]
    #[serde(rename = "typesafe")]
    TypeSafe,
    /// OpenRouter's Decisions API, a superset with request extras and
    /// in-band accounting.
    #[serde(rename = "openrouter")]
    OpenRouter,
}

/// The typed `codec_options.systemone` table.
///
/// Unknown keys are rejected so a misspelled option is a build issue for one
/// provider rather than a silently ignored setting.
#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "snake_case")]
pub(crate) struct SystemOneOptions {
    pub dialect: Dialect,
}

/// The `systemone` codec.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SystemOneCodec {
    dialect: Dialect,
}

impl SystemOneCodec {
    pub(crate) fn new(dialect: Dialect) -> Self {
        Self { dialect }
    }

    /// The operation URL for a provider base URL.
    ///
    /// See the module documentation for the rule.
    fn url(self, base_url: &str) -> String {
        match self.dialect {
            Dialect::TypeSafe => endpoint(base_url, TYPESAFE_PATH),
            Dialect::OpenRouter => {
                let base = base_url.trim_end_matches('/');
                let root = base.strip_suffix(OPENROUTER_CHAT_MOUNT).unwrap_or(base);
                format!("{root}{OPENROUTER_PATH}")
            }
        }
    }

    /// Adds the dialect's request extras from the route's own provider
    /// namespace and records what the protocol cannot carry.
    fn encode_extras(
        self,
        route: &ResolvedRoute,
        evaluation: &Evaluation,
        body: &mut Map<String, Value>,
        mut encoded: EncodedRequest,
    ) -> EncodedRequest {
        // Neither dialect has a field for free-form metadata.
        if !evaluation.metadata().is_empty() {
            encoded = encoded.unsupported_control("evaluation metadata");
        }
        let own = evaluation.provider_options().get(route.provider().id());
        match self.dialect {
            Dialect::TypeSafe => {
                if !evaluation.provider_options().is_empty() {
                    encoded = encoded.unsupported_control("provider options");
                }
            }
            Dialect::OpenRouter => {
                for (key, value) in own.into_iter().flatten() {
                    if OPENROUTER_EXTRAS.contains(&key.as_str()) {
                        body.insert(key.clone(), value.clone());
                    } else {
                        encoded = encoded.unsupported_control(&format!("provider option `{key}`"));
                    }
                }
            }
        }
        encoded
    }
}

impl EvaluationCodec for SystemOneCodec {
    fn encode_evaluation(&self, call: &ResolvedEvaluation) -> Result<EncodedRequest, Error> {
        let route = call.route();
        let evaluation = call.evaluation();

        let mut questions = Map::new();
        for (id, question) in evaluation.questions() {
            questions.insert(
                id.as_str().to_owned(),
                encode_question(route, id, question, NOUL, LIMITS)?,
            );
        }

        let mut body = Map::new();
        body.insert("model".to_owned(), route.api_model().into());
        body.insert("state".to_owned(), encode_state(evaluation.state()));
        body.insert("questions".to_owned(), Value::Object(questions));

        let encoded = EncodedRequest::new(
            Method::POST,
            self.url(route.provider().base_url()),
            Value::Null,
        );
        let mut encoded = self.encode_extras(route, evaluation, &mut body, encoded);
        encoded.body = Value::Object(body);
        Ok(encoded)
    }

    fn decode_verdict(&self, call: &ResolvedEvaluation, body: Value) -> Result<Verdict, Error> {
        match self.decode_body(call, &body) {
            Ok(mut verdict) => {
                verdict.raw = Some(body);
                Ok(verdict)
            }
            Err(detail) => Err(decode_failure(call.route(), &detail, body)),
        }
    }

    fn id_header(&self) -> Option<&'static str> {
        match self.dialect {
            Dialect::TypeSafe => Some(TYPESAFE_REQUEST_ID_HEADER),
            Dialect::OpenRouter => None,
        }
    }
}

impl SystemOneCodec {
    /// Decodes everything but `raw`. Errors are the detail of the failure,
    /// which the caller turns into one
    /// [`ResponseDecode`](crate::types::ErrorKind::ResponseDecode) carrying
    /// the body.
    fn decode_body(self, call: &ResolvedEvaluation, body: &Value) -> Result<Verdict, String> {
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

        let mut verdict = Verdict::new(
            route.provider().id().clone(),
            route.model().id().clone(),
            answers,
        );
        verdict.served_by = body
            .get("model")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        // The API declares no rounding and answers to two decimals. The
        // Vercel AI SDK's native TypeSafe adapter (`@ai-sdk/typesafe`)
        // hard-codes `probabilityDecimals: 2, scoreDecimals: 2` for the
        // same reason, and the client's validator needs the allowance to
        // accept a distribution that sums to 0.99 or 1.01 after rounding.
        verdict.rounding = Some(Rounding {
            probability_decimals: Some(2),
            score_decimals:       Some(2),
        });
        verdict.usage = TokenCounts {
            input: body
                .pointer("/usage/input_tokens")
                .and_then(Value::as_u64)
                .unwrap_or_default(),
            output: body
                .pointer("/usage/output_tokens")
                .and_then(Value::as_u64)
                .unwrap_or_default(),
            ..TokenCounts::default()
        };
        if self.dialect == Dialect::OpenRouter {
            verdict.id = body
                .get("id")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
            verdict.cost = body
                .pointer("/usage/cost")
                .and_then(Value::as_f64)
                .filter(|usd| usd.is_finite() && *usd >= 0.0)
                .map(|usd| Cost {
                    usd_micros: usd_micros(usd),
                    source:     CostSource::Provider,
                });
            if let Some(provider) = body.get("provider").filter(|value| !value.is_null()) {
                let mut namespace = Map::new();
                namespace.insert("provider".to_owned(), provider.clone());
                verdict.provider_metadata.insert(
                    route.provider().id().as_str().to_owned(),
                    Value::Object(namespace),
                );
            }
        }
        Ok(verdict)
    }
}

fn decode_answer(id: &str, question: Option<&Question>, wire: &Value) -> Result<Answer, String> {
    let object = answer_object(id, wire)?;
    let confidence = object.get("confidence").and_then(Value::as_f64);
    match object.get("type").and_then(Value::as_str) {
        Some("choice") => {
            let mut answer = decode_choice(id, question, object)?;
            answer.confidence = confidence;
            Ok(Answer::Choice(answer))
        }
        Some("score") => {
            let mut answer = decode_score(id, question, object)?;
            answer.confidence = confidence;
            if let (Some(legend), Some(Question::Score { levels, .. })) =
                (object.get("legend"), question)
            {
                check_legend(id, legend, levels)?;
            }
            Ok(Answer::Score(answer))
        }
        Some(NOUL) => {
            let probability = object
                .get(NOUL)
                .and_then(Value::as_f64)
                .ok_or_else(|| format!("answered noul question `{id}` without a probability"))?;
            Ok(Answer::Boolean(BooleanAnswer { probability }))
        }
        Some(other) => Err(format!(
            "answered question `{id}` with unknown answer type `{other}`"
        )),
        None => Err(format!("answered question `{id}` without an answer type")),
    }
}

/// Checks that a score answer's `legend` names the question's levels, by
/// index, with each description encoded as the request sent it (`null` for
/// a level without one).
fn check_legend(id: &str, legend: &Value, levels: &[Option<State>]) -> Result<(), String> {
    let mismatch =
        || format!("answered score question `{id}` with a legend that does not match its levels");
    let legend = legend.as_object().ok_or_else(mismatch)?;
    if legend.len() != levels.len() {
        return Err(mismatch());
    }
    for (index, level) in levels.iter().enumerate() {
        let described = legend.get(&index.to_string()).ok_or_else(mismatch)?;
        if *described != encode_description(level.as_ref()) {
            return Err(mismatch());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use serde_json::{Value, json};

    use super::{Dialect, EvaluationCodec, SystemOneCodec};
    use crate::adapter::ResolvedEvaluation;
    use crate::codecs::test_support::evaluation_in;
    use crate::evaluation::{Evaluation, Question, Rounding, State, Verdict};
    use crate::transport::classify;
    use crate::types::{Cost, CostSource, Error, ErrorKind, TokenCounts};

    const TYPESAFE_BASE_URL: &str = "https://api.typesafe.ai/v1";
    const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api";

    /// A real 200 body from `POST /v1/systemone` for [`three_questions`],
    /// recorded on 2026-09-18.
    const LIVE_BODY: &str = r#"{"model":"jev-1.13.0","answers":{"department":{"type":"choice","choice":"billing","confidence":1.0,"probabilities":{"billing":1.0,"other":0.0,"technical":0.0}},"severity":{"type":"score","score":1.08,"confidence":0.47,"legend":{"0":"Cosmetic","1":"Workaround exists","2":"Blocking; no workaround"},"probabilities":{"0":0.14,"1":0.64,"2":0.22}},"requests_refund":{"type":"noul","noul":0.98}},"usage":{"input_tokens":389,"output_tokens":70}}"#;

    /// The 401 body TypeSafe returns for a rejected key, 2026-09-18.
    const UNAUTHORIZED_BODY: &str = r#"{"detail":{"error_type":"authentication_error","message":"Cannot authenticate with the server. Please check your API key and try again."}}"#;

    /// The 422 body TypeSafe returns for a `boolean` question type,
    /// 2026-09-18. It names `bounding_box`, a fourth type no documentation
    /// page describes.
    const INVALID_BODY: &str = r#"{"detail":[{"type":"union_tag_invalid","loc":["body","questions","q"],"msg":"Input tag 'boolean' found using 'type' does not match any of the expected tags: <QuestionType.Noul: 'noul'>, <QuestionType.Choice: 'choice'>, <QuestionType.Score: 'score'>, <QuestionType.BoundingBox: 'bounding_box'>","input":{"type":"boolean","instructions":"Is it x?"},"ctx":{"discriminator":"'type'","tag":"boolean","expected_tags":"<QuestionType.Noul: 'noul'>, <QuestionType.Choice: 'choice'>, <QuestionType.Score: 'score'>, <QuestionType.BoundingBox: 'bounding_box'>"}}]}"#;

    /// A one-provider catalog shaped like the built-in `typesafe` provider,
    /// or like an OpenRouter row on the same codec.
    fn catalog(provider: &str, base_url: &str, options: &str) -> String {
        format!(
            r#"
            schema_version = 1

            [providers.{provider}]
            display_name = "{provider}"
            codecs = ["systemone"]
            {options}
            base_url = "{base_url}"
            default_model = "jev-latest"
            auth = {{ type = "bearer" }}

            [providers.{provider}.models."jev-latest"]
            display_name = "Jev"
            api_model = "jev-latest"
            capabilities = {{ evaluation = {{ choice = true, score = true, boolean = true }} }}
            pricing = {{ input_usd_micros_per_million = 42000, output_usd_micros_per_million = 0 }}
            "#
        )
    }

    fn typesafe() -> SystemOneCodec {
        SystemOneCodec::new(Dialect::TypeSafe)
    }

    fn openrouter() -> SystemOneCodec {
        SystemOneCodec::new(Dialect::OpenRouter)
    }

    fn resolved(evaluation: Evaluation) -> Result<ResolvedEvaluation, Box<dyn StdError>> {
        resolved_at(TYPESAFE_BASE_URL, evaluation)
    }

    fn resolved_at(
        base_url: &str,
        evaluation: Evaluation,
    ) -> Result<ResolvedEvaluation, Box<dyn StdError>> {
        evaluation_in(&catalog("typesafe", base_url, ""), evaluation)
    }

    fn resolved_openrouter(
        base_url: &str,
        evaluation: Evaluation,
    ) -> Result<ResolvedEvaluation, Box<dyn StdError>> {
        evaluation_in(
            &catalog(
                "openrouter",
                base_url,
                r#"codec_options = { systemone = { dialect = "openrouter" } }"#,
            ),
            evaluation,
        )
    }

    /// The interface plan's three questions, which [`LIVE_BODY`] answers.
    fn three_questions(provider: &str) -> Result<Evaluation, Box<dyn StdError>> {
        Ok(Evaluation::builder()
            .model(format!("{provider}/jev-latest"))
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

    /// Every shape the encoder has a rule for: a `None` option description,
    /// a `None` level, boolean criteria on one side only, and a JSON state.
    fn mixed_evaluation(provider: &str) -> Result<Evaluation, Box<dyn StdError>> {
        Ok(Evaluation::builder()
            .model(format!("{provider}/jev-latest"))
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
            .build()?)
    }

    fn decode(body: Value) -> Result<Verdict, Error> {
        let call = resolved(three_questions("typesafe").expect("the fixture evaluation builds"))
            .expect("the fixture evaluation resolves");
        typesafe().decode_verdict(&call, body)
    }

    fn body(text: &str) -> Value {
        serde_json::from_str(text).expect("the fixture body is JSON")
    }

    #[test]
    fn the_typesafe_url_keeps_one_v1_and_extends_a_bare_host() -> Result<(), Box<dyn StdError>> {
        for (base, expected) in [
            (
                "https://api.typesafe.ai/v1",
                "https://api.typesafe.ai/v1/systemone",
            ),
            (
                "https://api.typesafe.ai/v1/",
                "https://api.typesafe.ai/v1/systemone",
            ),
            (
                "https://api.typesafe.ai",
                "https://api.typesafe.ai/v1/systemone",
            ),
            (
                "http://127.0.0.1:3928",
                "http://127.0.0.1:3928/v1/systemone",
            ),
            (
                "http://127.0.0.1:3928/",
                "http://127.0.0.1:3928/v1/systemone",
            ),
        ] {
            let encoded =
                typesafe().encode_evaluation(&resolved_at(base, three_questions("typesafe")?)?)?;
            assert_eq!(encoded.url, expected, "base {base}");
        }
        Ok(())
    }

    #[test]
    fn the_openrouter_url_drops_a_chat_mount_and_extends_the_api_root()
    -> Result<(), Box<dyn StdError>> {
        for (base, expected) in [
            (
                "https://openrouter.ai/api",
                "https://openrouter.ai/api/alpha/decisions",
            ),
            (
                "https://openrouter.ai/api/",
                "https://openrouter.ai/api/alpha/decisions",
            ),
            (
                "https://openrouter.ai/api/v1",
                "https://openrouter.ai/api/alpha/decisions",
            ),
            (
                "http://127.0.0.1:3922",
                "http://127.0.0.1:3922/alpha/decisions",
            ),
        ] {
            let encoded = openrouter()
                .encode_evaluation(&resolved_openrouter(base, three_questions("openrouter")?)?)?;
            assert_eq!(encoded.url, expected, "base {base}");
        }
        Ok(())
    }

    #[test]
    fn encodes_the_mixed_evaluation_body_with_noul_and_no_headers() -> Result<(), Box<dyn StdError>>
    {
        let encoded = typesafe().encode_evaluation(&resolved(mixed_evaluation("typesafe")?)?)?;

        assert!(encoded.headers.is_empty(), "{:?}", encoded.headers);
        assert!(encoded.warnings.is_empty(), "{:?}", encoded.warnings);
        assert_eq!(
            encoded.body,
            json!({
                "model": "jev-latest",
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
                        "type": "noul",
                        "instructions": "Refund requested?",
                        "criteria": { "true": "Asks for money back" }
                    }
                }
            })
        );
        Ok(())
    }

    #[test]
    fn typesafe_warns_on_provider_options_and_metadata_and_sends_neither()
    -> Result<(), Box<dyn StdError>> {
        let evaluation = three_questions("typesafe")?
            .into_builder()
            .metadata_entry("tenant", "acme")
            .provider_option("typesafe", "session_id", json!("s-1"))
            .build()?;

        let encoded = typesafe().encode_evaluation(&resolved(evaluation)?)?;

        let mut keys: Vec<&str> = encoded
            .body
            .as_object()
            .map(|body| body.keys().map(String::as_str).collect())
            .unwrap_or_default();
        keys.sort_unstable();
        assert_eq!(keys, ["model", "questions", "state"]);
        let codes: Vec<&str> = encoded.warnings.iter().map(|w| w.code.as_str()).collect();
        assert_eq!(codes, ["unsupported_control", "unsupported_control"]);
        assert!(
            encoded
                .warnings
                .iter()
                .any(|w| w.message.contains("metadata"))
        );
        assert!(
            encoded
                .warnings
                .iter()
                .any(|w| w.message.contains("provider options"))
        );
        Ok(())
    }

    #[test]
    fn openrouter_forwards_its_four_extras_and_warns_on_the_rest() -> Result<(), Box<dyn StdError>>
    {
        let evaluation = three_questions("openrouter")?
            .into_builder()
            .provider_option("openrouter", "session_id", json!("s-1"))
            .provider_option("openrouter", "user", json!("u-1"))
            .provider_option("openrouter", "trace", json!({ "run": 7 }))
            .provider_option("openrouter", "provider", json!({ "order": ["typesafe"] }))
            .provider_option("openrouter", "temperature", json!(0.5))
            .provider_option("typesafe", "version", json!("jev-1.13.0"))
            .build()?;

        let encoded = openrouter()
            .encode_evaluation(&resolved_openrouter(OPENROUTER_BASE_URL, evaluation)?)?;

        assert_eq!(encoded.body["session_id"], json!("s-1"));
        assert_eq!(encoded.body["user"], json!("u-1"));
        assert_eq!(encoded.body["trace"], json!({ "run": 7 }));
        assert_eq!(encoded.body["provider"], json!({ "order": ["typesafe"] }));
        assert!(encoded.body.get("temperature").is_none());
        assert!(
            encoded.body.get("version").is_none(),
            "another namespace never reaches the wire"
        );
        assert_eq!(encoded.warnings.len(), 1, "{:?}", encoded.warnings);
        assert!(encoded.warnings[0].message.contains("`temperature`"));
        Ok(())
    }

    fn score_with_levels(count: usize) -> Result<Evaluation, Box<dyn StdError>> {
        Ok(Evaluation::builder()
            .model("typesafe/jev-latest")
            .state("state")
            .score("tall", "Rate it", vec![Option::<&str>::None; count])
            .build()?)
    }

    #[test]
    fn refuses_a_score_past_10_levels_before_dispatch() -> Result<(), Box<dyn StdError>> {
        typesafe().encode_evaluation(&resolved(score_with_levels(10)?)?)?;

        let error = typesafe()
            .encode_evaluation(&resolved(score_with_levels(11)?)?)
            .err()
            .ok_or("11 levels exceed the protocol's maximum")?;

        assert_eq!(error.kind(), ErrorKind::InvalidRequest);
        assert_eq!(error.provider_code(), Some("invalid_evaluation"));
        assert!(error.message().contains("`tall`"), "{}", error.message());
        Ok(())
    }

    #[test]
    fn a_wide_choice_is_not_refused() -> Result<(), Box<dyn StdError>> {
        let options: Vec<(String, Option<&str>)> =
            (0..300).map(|index| (format!("o{index}"), None)).collect();
        let evaluation = Evaluation::builder()
            .model("typesafe/jev-latest")
            .state("state")
            .choice("wide", "Pick one", options)
            .build()?;
        let encoded = typesafe().encode_evaluation(&resolved(evaluation)?)?;
        assert_eq!(
            encoded.body["questions"]["wide"]["criteria"]
                .as_object()
                .map(serde_json::Map::len),
            Some(300)
        );
        Ok(())
    }

    #[test]
    fn decodes_the_live_body_into_a_typed_verdict() -> Result<(), Box<dyn StdError>> {
        let verdict = decode(body(LIVE_BODY))?;

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
            (severity.score - 1.08).abs() < f64::EPSILON,
            "{}",
            severity.score
        );
        assert_eq!(severity.probabilities, Some(vec![0.14, 0.64, 0.22]));
        assert_eq!(severity.confidence, Some(0.47));

        let probability = verdict.boolean("requests_refund")?.probability;
        assert!((probability - 0.98).abs() < f64::EPSILON, "{probability}");

        assert_eq!(verdict.served_by.as_deref(), Some("jev-1.13.0"));
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
        assert_eq!(verdict.id, None, "the header id is the adapter's to fill");
        assert_eq!(verdict.cost, None, "the adapter fills the catalog estimate");
        assert!(verdict.provider_metadata.is_empty());
        assert_eq!(verdict.model.to_string(), "typesafe/jev-latest");
        assert_eq!(
            verdict.raw,
            Some(body(LIVE_BODY)),
            "the legend survives only in raw"
        );
        Ok(())
    }

    #[test]
    fn a_legend_that_matches_null_levels_passes() -> Result<(), Box<dyn StdError>> {
        let call = resolved(mixed_evaluation("typesafe")?)?;
        let verdict = typesafe().decode_verdict(
            &call,
            json!({
                "model": "jev-1.13.0",
                "answers": {
                    "severity": {
                        "type": "score", "score": 1.0, "confidence": 0.5,
                        "legend": { "0": "Cosmetic", "1": null, "2": "Blocking" },
                        "probabilities": { "0": 0.0, "1": 1.0, "2": 0.0 }
                    }
                }
            }),
        )?;
        assert_eq!(verdict.score("severity")?.confidence, Some(0.5));
        Ok(())
    }

    #[test]
    fn rejects_each_shape_the_verdict_cannot_represent() {
        let cases: [(&str, Value); 5] = [
            ("no answers", json!({ "model": "jev-1.13.0" })),
            (
                "an undocumented bounding_box answer",
                json!({ "answers": { "department": { "type": "bounding_box", "box": [0, 0, 1, 1] } } }),
            ),
            (
                "a legend whose description differs",
                json!({ "answers": { "severity": {
                    "type": "score", "score": 1,
                    "legend": { "0": "Cosmetic", "1": "Workaround exists", "2": "Blocking" },
                    "probabilities": { "0": 0.1, "1": 0.8, "2": 0.1 }
                } } }),
            ),
            (
                "a legend with an extra level",
                json!({ "answers": { "severity": {
                    "type": "score", "score": 1,
                    "legend": { "0": "Cosmetic", "1": "Workaround exists", "2": "Blocking; no workaround", "3": "Fatal" },
                    "probabilities": { "0": 0.1, "1": 0.8, "2": 0.1 }
                } } }),
            ),
            (
                "a noul without its probability",
                json!({ "answers": { "requests_refund": { "type": "noul", "probability": 0.5 } } }),
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
                Some("typesafe"),
                "{name}"
            );
            assert_eq!(error.raw_data(), Some(&body), "{name}");
            assert!(
                !error.is_retryable(),
                "{name}: a malformed answer is never retried"
            );
        }
        let bounding = decode(json!({ "answers": { "department": { "type": "bounding_box" } } }))
            .expect_err("bounding_box is refused");
        assert!(
            bounding.message().contains("bounding_box"),
            "{}",
            bounding.message()
        );
        let legend = decode(
            json!({ "answers": { "severity": { "type": "score", "score": 1, "legend": {} } } }),
        )
        .expect_err("a legend mismatch is refused");
        assert!(
            legend.message().contains("`severity`") && legend.message().contains("legend"),
            "{}",
            legend.message()
        );
    }

    #[test]
    fn openrouter_lifts_the_id_cost_and_provider_from_the_body() -> Result<(), Box<dyn StdError>> {
        let call = resolved_openrouter(OPENROUTER_BASE_URL, three_questions("openrouter")?)?;
        let mut wire = body(LIVE_BODY);
        wire["id"] = json!("dec_01");
        wire["provider"] = json!("TypeSafe");
        wire["usage"]["cost"] = json!(0.000_016_338);

        let verdict = openrouter().decode_verdict(&call, wire.clone())?;

        assert_eq!(verdict.id.as_deref(), Some("dec_01"));
        assert_eq!(
            verdict.cost,
            Some(Cost {
                usd_micros: 16,
                source:     CostSource::Provider,
            })
        );
        assert_eq!(
            verdict.provider_metadata.get("openrouter"),
            Some(&json!({ "provider": "TypeSafe" }))
        );
        assert_eq!(verdict.served_by.as_deref(), Some("jev-1.13.0"));
        assert_eq!(verdict.choice("department")?.confidence, Some(1.0));
        assert_eq!(verdict.raw, Some(wire));
        Ok(())
    }

    #[test]
    fn the_typesafe_dialect_names_its_id_header_and_openrouter_does_not() {
        assert_eq!(typesafe().id_header(), Some("x-typesafe-request-id"));
        assert_eq!(openrouter().id_header(), None);
    }

    /// The fixture bodies classify the way a caller needs: a rejected key is
    /// `Authentication`, a malformed body is `InvalidRequest`, and each keeps
    /// TypeSafe's message.
    #[test]
    fn the_typesafe_error_bodies_classify_with_their_messages() {
        let (message, code) = classify::extract(Some(&body(UNAUTHORIZED_BODY)));
        let failure = classify::classify(Some(401), code.as_deref(), message.as_deref(), None);
        assert_eq!(failure.kind, ErrorKind::Authentication);
        assert_eq!(failure.code.as_deref(), Some("authentication_error"));
        assert!(
            failure
                .message
                .is_some_and(|m| m.starts_with("Cannot authenticate"))
        );

        let (message, code) = classify::extract(Some(&body(INVALID_BODY)));
        let failure = classify::classify(Some(422), code.as_deref(), message.as_deref(), None);
        assert_eq!(failure.kind, ErrorKind::InvalidRequest);
        assert_eq!(failure.code.as_deref(), Some("union_tag_invalid"));
        assert!(failure.message.is_some_and(|m| m.contains("bounding_box")));
    }
}
