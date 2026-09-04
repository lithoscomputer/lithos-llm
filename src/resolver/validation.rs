use super::ResolvedRoute;
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

    fn unsupported_capability(&self, capability: &str) -> Error {
        Error::new(
            ErrorKind::InvalidRequest,
            format!("model {} does not support {capability}", self.handle()),
        )
        .with_provider(self.provider().id().clone())
        .with_provider_code("unsupported_capability")
    }
}
