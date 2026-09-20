use super::ResolvedRoute;
use crate::evaluation::{Evaluation, QuestionKind};
use crate::types::{CacheHint, ContentPart, Error, ErrorKind, Message, Request};

impl ResolvedRoute {
    /// Refuses what the catalog row says the model cannot do.
    ///
    /// Three questions are asked in order: do the request's controls name a
    /// capability the row denies, does its output limit exceed the row's,
    /// and does its content carry a part the row denies or this crate does
    /// not know. Each refusal is an early return with the capability named,
    /// so the first thing wrong is the thing reported.
    ///
    /// This is the catalog's answer, not the protocol's. Protocol-level
    /// refusals — a part the wire format cannot carry at all, a control the
    /// dialect has no field for — belong to the codec's `encode`, because
    /// those facts depend on the part's shape and hold for passthrough
    /// models that have no catalog row. An `Unknown` claim passes here: a
    /// passthrough model is not described by the catalog, so the provider
    /// gets to say no.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidRequest`] with provider code
    /// `unsupported_capability`, `max_output_tokens`, or
    /// `unknown_content_type`.
    pub(crate) fn validate_request(&self, request: &Request) -> Result<(), Error> {
        self.check_controls(request)?;
        self.check_output_limit(request)?;
        self.check_content(request)
    }

    /// Refuses a request control the row denies: tools, a forced tool
    /// choice, structured output, reasoning effort, sampling, a positive
    /// cache hint, or a speed tier.
    fn check_controls(&self, request: &Request) -> Result<(), Error> {
        let capabilities = self.model().capabilities();
        if !request.tools().is_empty() && capabilities.tools().is_unsupported() {
            return Err(self.unsupported_capability("tools"));
        }
        if request
            .tool_choice()
            .is_some_and(|choice| capabilities.tool_choice(choice).is_unsupported())
        {
            return Err(self.unsupported_capability("forced tool choice"));
        }
        if request
            .response_format()
            .is_some_and(|format| capabilities.response_format(format).is_unsupported())
        {
            return Err(self.unsupported_capability("structured output"));
        }
        if request
            .reasoning_effort()
            .is_some_and(|effort| capabilities.reasoning_effort(effort).is_unsupported())
        {
            return Err(self.unsupported_capability("reasoning"));
        }
        if (request.temperature().is_some() || request.top_p().is_some())
            && capabilities.sampling().is_unsupported()
        {
            return Err(self.unsupported_capability("sampling"));
        }
        // `Disabled` asks for nothing and is honored anywhere; the explicit
        // positive hints ask for a wire field the model must take.
        if matches!(
            request.cache_hint(),
            Some(CacheHint::Auto | CacheHint::Key { .. })
        ) && capabilities.cache_routing().is_unsupported()
        {
            return Err(self.unsupported_capability("cache routing"));
        }
        if let Some(speed) = request.speed()
            && capabilities.speed(speed).is_unsupported()
        {
            return Err(self.unsupported_capability(&format!("speed '{}'", speed.as_str())));
        }
        Ok(())
    }

    /// Refuses an output limit above the row's, when the row records one.
    fn check_output_limit(&self, request: &Request) -> Result<(), Error> {
        let Some(limits) = self.model().limits() else {
            return Ok(());
        };
        let Some(tokens) = request.max_output_tokens() else {
            return Ok(());
        };
        if u64::from(tokens) <= limits.max_output_tokens {
            return Ok(());
        }
        Err(Error::new(
            ErrorKind::InvalidRequest,
            format!(
                "model {} allows at most {} output tokens",
                self.handle(),
                limits.max_output_tokens
            ),
        )
        .with_provider(self.provider().id().clone())
        .with_provider_code("max_output_tokens"))
    }

    /// Refuses content the row denies, and unknown content anywhere.
    ///
    /// An unknown part is refused before the capability lookup: it is an
    /// unrecognized stored type this crate cannot send, whatever the row
    /// claims. The capability of every other part is
    /// [`ModelCapabilities::content_part`](crate::catalog::ModelCapabilities::content_part).
    fn check_content(&self, request: &Request) -> Result<(), Error> {
        let capabilities = self.model().capabilities();
        for part in request.messages().iter().flat_map(Message::content) {
            if let Some(kind) = part.unknown_kind() {
                return Err(Error::new(
                    ErrorKind::InvalidRequest,
                    format!(
                        "unknown transcript content type {kind}; convert or remove it before \
                         dispatch"
                    ),
                )
                .with_provider(self.provider().id().clone())
                .with_provider_code("unknown_content_type"));
            }
            if capabilities.content_part(part).is_unsupported() {
                return Err(self.unsupported_capability(content_capability(part)));
            }
        }
        Ok(())
    }

    /// Refuses an evaluation that asks a question kind the model is known
    /// not to answer.
    ///
    /// `Unknown` passes, as it does for generation capabilities: a
    /// passthrough model is not described by the catalog, so the provider
    /// gets to say no.
    pub(crate) fn validate_evaluation(&self, evaluation: &Evaluation) -> Result<(), Error> {
        let capabilities = self.model().capabilities();
        for question in evaluation.questions().values() {
            let kind = question.kind();
            if capabilities.evaluation(kind).is_unsupported() {
                return Err(self.unsupported_capability(&format!("{} evaluation", kind_name(kind))));
            }
        }
        Ok(())
    }

    fn unsupported_capability(&self, capability: &str) -> Error {
        Error::new(
            ErrorKind::InvalidRequest,
            format!("model {} does not support {capability}", self.handle()),
        )
        .with_provider(self.provider().id().clone())
        .with_provider_code("unsupported_capability")
    }
}

/// The capability a refused content part is reported under.
fn content_capability(part: &ContentPart) -> &'static str {
    match part {
        ContentPart::Text { .. } => "text",
        ContentPart::Image(_) => "images",
        ContentPart::Audio(_) => "audio",
        ContentPart::Document(_) => "documents",
        ContentPart::Reasoning(_) => "reasoning",
        ContentPart::ToolCall(_) | ContentPart::ToolResult(_) => "tools",
        // These never answer `Unsupported`; see `content_part`.
        ContentPart::Json { .. } | ContentPart::Opaque { .. } | ContentPart::Unknown(_) => {
            "content"
        }
    }
}

/// The kind as a row writes it under `capabilities.evaluation`.
fn kind_name(kind: QuestionKind) -> &'static str {
    match kind {
        QuestionKind::Choice => "choice",
        QuestionKind::Score => "score",
        QuestionKind::Boolean => "boolean",
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use serde_json::json;

    use super::ResolvedRoute;
    use crate::catalog::{Catalog, CatalogModel, ModelId, ProviderId};
    use crate::evaluation::Evaluation;
    use crate::types::{ContentPart, ErrorKind, Message, Request, Role};

    /// One provider with a judge, a text-only row, and a judge that will
    /// not answer boolean questions.
    const CATALOG: &str = r#"
        schema_version = 1

        [providers.alpha]
        display_name = "Alpha"
        adapter = "test-adapter"
        codecs = ["test-codec"]
        base_url = "http://127.0.0.1"
        allow_passthrough = true
        auth = { type = "none" }

        [providers.alpha.models.judge]
        display_name = "Judge"
        api_model = "judge"
        capabilities = { text = true, response_format = { json_schema = true } }

        [providers.alpha.models.plain]
        display_name = "Plain"
        api_model = "plain"
        capabilities = { text = true }

        [providers.alpha.models.no-boolean]
        display_name = "No boolean"
        api_model = "no-boolean"
        capabilities = { text = true, response_format = { json_schema = true }, evaluation = { boolean = false } }

        [providers.alpha.models.small]
        display_name = "Small"
        api_model = "small"
        capabilities = { text = true }
        limits = { context_tokens = 8000, max_output_tokens = 1000 }
    "#;

    fn route(model: &str) -> Result<ResolvedRoute, Box<dyn StdError>> {
        let catalog = Catalog::builder().toml_layer("test", CATALOG)?.build()?;
        let provider = catalog.provider("alpha")?.clone();
        let model = if model == "passthrough" {
            CatalogModel::passthrough(&provider, ModelId::new(model))
        } else {
            catalog.model("alpha", model)?.clone()
        };
        Ok(ResolvedRoute::try_new(provider, model)?)
    }

    fn mixed_evaluation() -> Result<Evaluation, Box<dyn StdError>> {
        Ok(Evaluation::builder()
            .model("alpha/judge")
            .state("The order arrived late and the customer wants a refund.")
            .choice("team", "Which team?", [
                ("billing", Some("Charges and refunds")),
                ("shipping", None),
            ])
            .score("severity", "How severe?", ["Cosmetic", "Blocking"])
            .boolean("refund", "Refund requested?")
            .build()?)
    }

    #[test]
    fn an_output_limit_above_the_rows_is_refused_and_named() -> Result<(), Box<dyn StdError>> {
        let small = route("small")?;
        let within = Request::builder()
            .model("alpha/small")
            .user("hi")
            .max_output_tokens(1000)
            .build()?;
        small.validate_request(&within)?;

        let above = Request::builder()
            .model("alpha/small")
            .user("hi")
            .max_output_tokens(1001)
            .build()?;
        let error = small
            .validate_request(&above)
            .expect_err("the row records a smaller limit");

        assert_eq!(error.kind(), ErrorKind::InvalidRequest);
        assert_eq!(error.provider_code(), Some("max_output_tokens"));
        assert!(
            error.message().contains("at most 1000"),
            "{}",
            error.message()
        );

        // A row without limits accepts any number.
        route("plain")?.validate_request(&above)?;
        Ok(())
    }

    #[test]
    fn unknown_content_is_refused_whatever_the_row_claims() -> Result<(), Box<dyn StdError>> {
        let unknown: ContentPart = serde_json::from_value(json!({
            "type": "hologram",
            "frames": 3,
        }))?;
        let request = Request::builder()
            .model("alpha/passthrough")
            .message(Message::new(Role::User, [unknown]))
            .build()?;

        let error = route("passthrough")?
            .validate_request(&request)
            .expect_err("an unknown part cannot be sent");

        assert_eq!(error.kind(), ErrorKind::InvalidRequest);
        assert_eq!(error.provider_code(), Some("unknown_content_type"));
        assert!(error.message().contains("hologram"), "{}", error.message());
        Ok(())
    }

    #[test]
    fn a_structured_output_row_accepts_every_kind() -> Result<(), Box<dyn StdError>> {
        route("judge")?.validate_evaluation(&mixed_evaluation()?)?;
        Ok(())
    }

    #[test]
    fn a_row_without_json_schema_refuses_and_names_the_kind() -> Result<(), Box<dyn StdError>> {
        let error = route("plain")?
            .validate_evaluation(&mixed_evaluation()?)
            .expect_err("a text-only row cannot judge");

        assert_eq!(error.kind(), ErrorKind::InvalidRequest);
        assert_eq!(error.provider_code(), Some("unsupported_capability"));
        assert_eq!(error.provider().map(ProviderId::as_str), Some("alpha"));
        // Questions are checked in id order, so `refund` is first.
        assert!(
            error.message().contains("boolean evaluation"),
            "{}",
            error.message()
        );
        Ok(())
    }

    #[test]
    fn an_explicit_denial_refuses_only_that_kind() -> Result<(), Box<dyn StdError>> {
        let route = route("no-boolean")?;

        let without_boolean = Evaluation::builder()
            .model("alpha/no-boolean")
            .state("state")
            .choice("team", "Which team?", [("billing", Option::<&str>::None)])
            .score("severity", "How severe?", ["Cosmetic", "Blocking"])
            .build()?;
        route.validate_evaluation(&without_boolean)?;

        let error = route
            .validate_evaluation(&mixed_evaluation()?)
            .expect_err("the row denies boolean questions");
        assert_eq!(error.provider_code(), Some("unsupported_capability"));
        assert!(
            error.message().contains("boolean evaluation"),
            "{}",
            error.message()
        );
        Ok(())
    }

    #[test]
    fn a_passthrough_route_accepts_every_kind() -> Result<(), Box<dyn StdError>> {
        route("passthrough")?.validate_evaluation(&mixed_evaluation()?)?;
        Ok(())
    }
}
