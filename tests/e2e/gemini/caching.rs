//! V2 — prompt caching.
//!
//! Gemini enables implicit caching without a request control. Two calls share
//! a prefix above the documented minimum, and the second must report a read.
//!
//! One family representative each keeps the token cost bounded: the shared
//! prefix is the expensive part of the whole Gemini suite.

use std::time::Duration;

use tokio::time::sleep;

use crate::gemini::{self, family_tests};
use crate::support::{self, TestResult};

mod round_trip {
    use super::*;

    family_tests!(super::caches_a_shared_prefix);
}

/// A deterministic prefix above Gemini's largest 4,096-token minimum.
///
/// The text is genuinely varied so the provider sees a realistic prefix.
///
/// The corpus is a frozen fixture, not a living file: replay matches
/// requests deterministically when native Gemini recording support arrives.
fn large_prefix() -> String {
    format!(
        "You answer questions about this changelog:\n\n{}",
        include_str!("../fixtures/cache_prefix.txt")
    )
}

async fn caches_a_shared_prefix(model: &str) -> TestResult {
    if !gemini::capabilities(model).caching().is_supported() {
        return support::skip("the catalog does not claim caching");
    }
    let Some(client) = gemini::live_client() else {
        return support::skip("GEMINI_API_KEY is unset");
    };
    let prefix = large_prefix();

    // Gemini's implicit cache needs no cache breakpoint or routing hint.
    let first = client
        .complete(
            gemini::request(model)
                .system(prefix.clone())
                .user("Which project does this changelog describe? Answer with just its name.")
                .build()?,
        )
        .await?;
    // A bounded retry ladder tolerates short propagation delays.
    let mut second = None;
    for _ in 0..4 {
        sleep(Duration::from_secs(5)).await;
        let response = client
            .complete(
                gemini::request(model)
                    .system(prefix.clone())
                    .user("Does the changelog mention streaming? Answer yes or no.")
                    .build()?,
            )
            .await?;
        let read = response.usage.cache_read;
        second = Some(response);
        if read > 0 {
            break;
        }
    }
    let second = second.expect("the retry ladder always runs at least once");

    support::observe(&format!(
        "{model} cache buckets: first write={} read={}, second write={} read={}",
        first.usage.cache_write,
        first.usage.cache_read,
        second.usage.cache_write,
        second.usage.cache_read,
    ));
    assert!(
        second.usage.cache_read > 0,
        "{model} did not read the prefix cache"
    );
    Ok(())
}

/// `auto_cache = false` must still complete normally: the control key is
/// consumed by the codec and never sent, so the only observable difference
/// is fewer cache buckets, which automatic server-side caches may report
/// anyway. The pinned behavior is acceptance, not bucket values.
#[tokio::test]
#[ignore = "live Gemini call; run with `mise run test:e2e`"]
async fn auto_cache_off_is_accepted() -> TestResult {
    let Some(client) = gemini::live_client() else {
        return support::skip("GEMINI_API_KEY is unset");
    };
    let request = gemini::request("gemini-3.5-flash")
        .user("In one short sentence, say hello.")
        .provider_option(gemini::PROVIDER, "auto_cache", serde_json::json!(false))
        .build()?;
    let response = client.complete(request).await?;
    assert!(!response.text().trim().is_empty());
    Ok(())
}
