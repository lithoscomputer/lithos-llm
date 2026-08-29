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
/// blocks). Repetition keeps it cheap to author; determinism keeps the second
/// call byte-identical, which is what a prefix cache keys on.
fn large_prefix() -> String {
    let paragraph = "You are the reference desk for a fictional municipal library. \
        The library has twelve reading rooms, a map archive, a seed vault, a tool \
        lending desk, and a rooftop observatory. Opening hours differ per room and \
        per season, fines are waived on rainy days, and the catalog is sorted by \
        the second letter of each title. Answer every question from these rules. ";
    paragraph.repeat(120)
}

async fn caches_a_shared_prefix(model: &str) -> TestResult {
    if !venice::capabilities(model).caching {
        return support::skip("the catalog does not claim caching");
    }
    let Some(client) = venice::live_client() else {
        return support::skip("VENICE_API_KEY is unset");
    };
    let prefix = large_prefix();

    // Venice routes cache hits by `prompt_cache_key`: without it, the pair
    // can land on different backend replicas, and on 2026-08-29 the Claude
    // path wrote 13k tokens of cache on BOTH calls and read nothing. The
    // codec now derives the key automatically (`CacheHint::Auto` over the
    // shared system prefix), so this pair exercises exactly what an
    // application gets by default.
    let first = client
        .complete(
            venice::request(model)
                .system(prefix.clone())
                .user("How many reading rooms are there? Answer with just the number.")
                .build()?,
        )
        .await?;
    // A fresh cache entry takes a few seconds to become readable — a
    // controlled pair against deepseek read nothing back-to-back and the
    // full prefix five seconds later — so the read side retries on a short
    // ladder before the verdict.
    let mut second = None;
    for _ in 0..4 {
        sleep(Duration::from_secs(5)).await;
        let response = client
            .complete(
                venice::request(model)
                    .system(prefix.clone())
                    .user("Does the library have an observatory? Answer yes or no.")
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
