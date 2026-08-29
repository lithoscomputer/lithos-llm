//! V2 — prompt caching.
//!
//! Caching on Venice is automatic where it exists at all, so the test is a
//! pair of sequential completions sharing one large prefix: the first may
//! report a cache write, the second must show the cache working — a write on
//! the first call or a read on the second. Which bucket a family reports is
//! upstream-specific, so the assertion accepts either, and the per-call
//! buckets land in the probe output for the catalog to pin later.
//!
//! One family representative each keeps the token cost bounded: the shared
//! prefix is the expensive part of the whole Venice suite.

use std::time::Duration;

use tokio::time::sleep;

use crate::support::{self, TestResult};
use crate::venice::{self, family_tests};

mod round_trip {
    use super::*;

    family_tests!(super::caches_a_shared_prefix);
}

/// A deterministic prefix comfortably above every known minimum cacheable
/// size (1024 tokens is the common floor; DeepSeek-style caches use 64-token
/// blocks).
///
/// The text must be genuinely varied, not a repeated paragraph: on
/// 2026-08-29 a paragraph repeated 120 times wrote a 13k-token cache entry
/// on Venice's Claude path on every call and never read one, while varied
/// prose of the same size read back cleanly — and Anthropic-direct refuses
/// the same repetitive content outright.
///
/// The corpus is a frozen fixture, not a living file: replay matches
/// requests by body hash, so the prefix must be byte-identical between the
/// recording and every later replay. Do not edit the fixture; editing it
/// invalidates the caching recordings until `mise run test:e2e:record`.
fn large_prefix() -> String {
    format!(
        "You answer questions about this changelog:\n\n{}",
        include_str!("../fixtures/cache_prefix.txt")
    )
}

async fn caches_a_shared_prefix(model: &str) -> TestResult {
    if !venice::capabilities(model).caching {
        return support::skip("the catalog does not claim caching");
    }
    let Some(client) = venice::live_client() else {
        return support::skip("VENICE_API_KEY is unset");
    };
    let prefix = large_prefix();

    // The codec derives a `prompt_cache_key` automatically
    // (`CacheHint::Auto` over the shared system prefix), so this pair
    // exercises exactly what an application gets by default. Venice uses
    // the key for backend session affinity; Claude caching on Venice was
    // verified to read back with and without it once the prefix is varied
    // text and the read side waits out cache propagation.
    let first = client
        .complete(
            venice::request(model)
                .system(prefix.clone())
                .user("Which project does this changelog describe? Answer with just its name.")
                .build()?,
        )
        .await?;
    // A fresh cache entry takes a few seconds to become readable — a
    // controlled pair against deepseek read nothing back-to-back and the
    // full prefix five seconds later — so the read side retries on a short
    // ladder before the verdict.
    let mut second = None;
    for _ in 0..4 {
        // Replay serves the recorded buckets immediately; only a live or
        // recording run waits out real cache propagation.
        if support::Backend::from_env() != support::Backend::Replay {
            sleep(Duration::from_secs(5)).await;
        }
        let response = client
            .complete(
                venice::request(model)
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
        first.usage.cache_write > 0 || second.usage.cache_read > 0,
        "{model} claims caching but neither wrote nor read a cache across the pair"
    );
    Ok(())
}

/// `auto_cache = false` must still complete normally: the control key is
/// consumed by the codec and never sent, so the only observable difference
/// is fewer cache buckets, which automatic server-side caches may report
/// anyway. The pinned behavior is acceptance, not bucket values.
#[tokio::test]
#[ignore = "live Venice call; run with `mise run test:e2e`"]
async fn auto_cache_off_is_accepted() -> TestResult {
    let Some(client) = venice::live_client() else {
        return support::skip("VENICE_API_KEY is unset");
    };
    let request = venice::request("deepseek-v4-flash")
        .user("In one short sentence, say hello.")
        .provider_option(venice::PROVIDER, "auto_cache", serde_json::json!(false))
        .build()?;
    let response = client.complete(request).await?;
    assert!(!response.text().trim().is_empty());
    Ok(())
}
