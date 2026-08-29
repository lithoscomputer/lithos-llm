//! V2 — sampling parameters.
//!
//! Runs on every roster model whose row claims `sampling`. The assertion is
//! acceptance without warnings. Anthropic rejects a request that specifies
//! both temperature and top-p, so the test sends one request per control.

use crate::anthropic::{self, model_tests};
use crate::support::{self, TestResult};

mod accepted {
    use super::*;

    model_tests!(super::accepts_sampling_parameters);
}

async fn accepts_sampling_parameters(model: &str) -> TestResult {
    if !anthropic::capabilities(model).sampling {
        return support::skip("the catalog does not claim sampling");
    }
    let Some(client) = anthropic::live_client() else {
        return support::skip("ANTHROPIC_API_KEY is unset");
    };
    let temperature = anthropic::request(model)
        .user("In one short sentence, say hello.")
        .temperature(0.0)
        .build()?;
    let response = client.complete(temperature).await?;
    assert!(
        response.warnings.is_empty(),
        "temperature produced warnings: {:?}",
        response.warnings
    );

    let top_p = anthropic::request(model)
        .user("In one short sentence, say hello.")
        .top_p(0.9)
        .build()?;
    let response = client.complete(top_p).await?;
    assert!(
        response.warnings.is_empty(),
        "top-p produced warnings: {:?}",
        response.warnings
    );
    Ok(())
}
