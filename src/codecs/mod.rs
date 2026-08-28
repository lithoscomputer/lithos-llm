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

use crate::adapter::ResolvedCall;
use crate::resolver::ResolvedRoute;
use crate::transport::{EncodedRequest, SseEvent};
use crate::types::{Error, Response, StreamEvent};

pub(crate) trait Codec: Send + Sync {
    fn encode(&self, call: &ResolvedCall, stream: bool) -> Result<EncodedRequest, Error>;

    fn decode_response(
        &self,
        route: &ResolvedRoute,
        value: serde_json::Value,
    ) -> Result<Response, Error>;

    fn decode_sse(&self, route: &ResolvedRoute, event: SseEvent)
    -> Result<Vec<StreamEvent>, Error>;
}

#[cfg(all(test, feature = "builtin-catalog"))]
pub(crate) mod test_support {
    use std::error::Error as StdError;

    use crate::adapter::ResolvedCall;
    use crate::catalog::Catalog;
    use crate::middleware::CallContext;
    use crate::resolver::{AvailableProviders, CatalogResolver, ModelResolver};
    use crate::types::Request;

    pub(crate) fn resolved(request: Request) -> Result<ResolvedCall, Box<dyn StdError>> {
        let catalog = Catalog::builder().with_builtin().build()?;
        let available = AvailableProviders::all(&catalog);
        let route = CatalogResolver.resolve(&request, &catalog, &available)?;
        Ok(ResolvedCall::new(request, route, CallContext::new()))
    }
}
