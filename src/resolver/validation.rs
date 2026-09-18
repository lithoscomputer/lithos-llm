use super::ResolvedRoute;
use crate::evaluation::{Evaluation, QuestionKind};
use crate::types::{CacheHint, ContentPart, Error, ErrorKind, Message, Request};

impl ResolvedRoute {
    pub(crate) fn validate_request(&self, request: &Request) -> Result<(), Error> {
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
        if let Some(limits) = self.model().limits()
            && request
                .max_output_tokens()
                .is_some_and(|tokens| u64::from(tokens) > limits.max_output_tokens)
        {
            return Err(Error::new(
                ErrorKind::InvalidRequest,
                format!(
                    "model {} allows at most {} output tokens",
                    self.handle(),
                    limits.max_output_tokens
                ),
            )
            .with_provider(self.provider().id().clone())
            .with_provider_code("max_output_tokens"));
        }
        for part in request.messages().iter().flat_map(Message::content) {
            if let Some(kind) = part.unknown_kind() {
                return Err(Error::new(ErrorKind::InvalidRequest,
                    format!("unknown transcript content type {kind}; convert or remove it before dispatch"))
                    .with_provider(self.provider().id().clone())
                    .with_provider_code("unknown_content_type"));
            }
            let capability = match part {
                ContentPart::Text { .. } if capabilities.text().is_unsupported() => Some("text"),
                ContentPart::Image(_) if capabilities.images().is_unsupported() => Some("images"),
                ContentPart::Audio(_) if capabilities.audio().is_unsupported() => Some("audio"),
                ContentPart::Document(_) if capabilities.documents().is_unsupported() => {
                    Some("documents")
                }
                ContentPart::Reasoning(_) if capabilities.reasoning().is_unsupported() => {
                    Some("reasoning")
                }
                ContentPart::ToolCall(_) | ContentPart::ToolResult(_)
                    if capabilities.tools().is_unsupported() =>
                {
                    Some("tools")
                }
                _ => None,
            };
            if let Some(capability) = capability {
                return Err(self.unsupported_capability(capability));
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

    use super::ResolvedRoute;
    use crate::catalog::{Catalog, CatalogModel, ModelId, ProviderId};
    use crate::evaluation::Evaluation;
    use crate::types::ErrorKind;

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
