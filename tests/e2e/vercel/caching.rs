//! V2 — prompt caching.
//!
//! Caching through the gateway is automatic on most upstreams and explicit
//! on Anthropic, where the codec's `cache_control` breakpoints apply. The
//! test is a pair of sequential completions sharing one large prefix: the
//! first may report a cache write, the second must show the cache working — a
//! write on the first call or a read on the second. Which bucket a family
//! reports is upstream-specific, so the assertion accepts either, and the
//! per-call buckets land in the probe output for the catalog to pin later.
//!
//! One family representative each keeps the token cost bounded: the shared
//! prefix is the expensive part of the whole suite.

use std::time::Duration;

use tokio::time::sleep;

use crate::support::{self, TestResult};
use crate::vercel::{self, family_tests};

mod round_trip {
    use super::*;

    family_tests!(super::caches_a_shared_prefix);
}

/// A deterministic prefix comfortably above every known minimum cacheable
/// size (1024 tokens is the common floor; DeepSeek-style caches use 64-token
/// blocks).
///
/// The text must be genuinely varied, not a repeated paragraph: a repeated
/// paragraph has written a cache entry on every call and never read one on
/// other Claude routes, while varied prose of the same size reads back
/// cleanly — and Anthropic-direct refuses the same repetitive content
/// outright.
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
    if !vercel::capabilities(model).caching().is_supported() {
        return support::skip("the catalog does not claim caching");
    }
    let Some(client) = vercel::live_client() else {
        return support::skip("AI_GATEWAY_API_KEY is unset");
    };
    let prefix = large_prefix();

    // The gateway accepts `prompt_cache_key` but documents no sticky routing
    // for it, so no row claims cache routing; this pair exercises the
    // upstream caches on their own terms.
    let first = client
        .complete(
            vercel::request(model)
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
                vercel::request(model)
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
#[ignore = "live Vercel AI Gateway call; run with `mise run test:e2e`"]
async fn auto_cache_off_is_accepted() -> TestResult {
    let Some(client) = vercel::live_client() else {
        return support::skip("AI_GATEWAY_API_KEY is unset");
    };
    let request = vercel::request("deepseek-v4-flash")
        .user("In one short sentence, say hello.")
        .provider_option(vercel::PROVIDER, "auto_cache", serde_json::json!(false))
        .build()?;
    let response = client.complete(request).await?;
    assert!(!response.text().trim().is_empty());
    Ok(())
}
