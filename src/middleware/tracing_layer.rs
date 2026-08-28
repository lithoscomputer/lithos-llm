use async_trait::async_trait;
use tracing::Instrument as _;

use super::{Call, Middleware, Next, Output};
use crate::types::Error;

/// Emits secret-safe tracing spans without installing a subscriber.
#[derive(Clone, Copy, Debug, Default)]
pub struct TracingMiddleware;

#[async_trait]
impl Middleware for TracingMiddleware {
    async fn handle(&self, call: Call, next: Next) -> Result<Output, Error> {
        let span = tracing::info_span!(
            "llm.call",
            call_id = %call.context.call_id(),
            attempt = call.context.attempt(),
            provider = %call.route.provider().id(),
            model = %call.route.model().id(),
            mode = ?call.mode,
        );
        let result = next.run(call).instrument(span).await;
        match &result {
            Ok(Output::Complete(response)) => tracing::debug!(
                input_tokens = response.usage.input,
                output_tokens = response.usage.output,
                "LLM call completed"
            ),
            Ok(Output::Stream(_)) => tracing::debug!("LLM stream accepted"),
            Err(error) => tracing::warn!(
                error = %error,
                kind = ?error.kind(),
                status = error.status(),
                provider_code = error.provider_code(),
                "LLM call failed"
            ),
        }
        result
    }
}
