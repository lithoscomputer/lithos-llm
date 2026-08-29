//! V2 — sampling parameters.
//!
//! Runs on every roster model whose row claims `sampling`. The assertion is
//! acceptance without warnings, because sampled output itself cannot be
//! asserted; a model that rejects temperature live tells us to flip its
//! catalog row, which the preflight suite then enforces client-side.

use crate::gemini::{self, model_tests};
use crate::support::{self, TestResult};

mod accepted {
    use super::*;

    model_tests!(super::accepts_sampling_parameters);
}

async fn accepts_sampling_parameters(model: &str) -> TestResult {
    if !gemini::capabilities(model).sampling {
        return support::skip("the catalog does not claim sampling");
    }
    let Some(client) = gemini::live_client() else {
        return support::skip("GEMINI_API_KEY is unset");
    };
    let request = gemini::request(model)
        .user("In one short sentence, say hello.")
        .temperature(0.7)
        .top_p(0.8)
        .build()?;
    let response = client.complete(request).await?;
    assert!(!response.text().trim().is_empty());
    assert!(
        response.warnings.is_empty(),
        "sampling produced warnings: {:?}",
        response.warnings
    );
    Ok(())
}
