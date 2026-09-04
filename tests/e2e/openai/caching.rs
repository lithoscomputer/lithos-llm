//! V2 — prompt caching.
//!
//! Caching on OpenAI is automatic, so the test is a pair of sequential
//! completions sharing one large prefix: the pair must show the cache
//! working — a write on the first call or a read on the second. Which bucket
//! shows first is family-specific: the GPT-5.6 rows bill writes and report
//! `cache_write_tokens` on the writing call (observed live 2026-08-30),
//! while the 5.4/5.5 rows write for free and only ever report
//! `cached_tokens` on the reading call. The per-call buckets land in the
//! probe output either way.
//!
//! One family representative each keeps the token cost bounded: the shared
//! prefix is the expensive part of the whole OpenAI suite.

use std::time::Duration;

use tokio::time::sleep;

use crate::openai::{self, family_tests};
use crate::support::{self, TestResult};

mod round_trip {
    use super::*;

    family_tests!(super::caches_a_shared_prefix);
}

/// A deterministic prefix comfortably above OpenAI's 1024-token cacheable
/// minimum.
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
    if !openai::capabilities(model).caching().is_supported() {
        return support::skip("the catalog does not claim caching");
    }
    let Some(client) = openai::live_client() else {
        return support::skip("OPENAI_API_KEY is unset");
    };
    let prefix = large_prefix();

    // The codec derives a `prompt_cache_key` automatically
    // (`CacheHint::Auto` over the shared system prefix), so this pair
    // exercises exactly what an application gets by default. OpenAI uses
    // the key to route both requests to the same cache shard.
    let first = client
        .complete(
            openai::request(model)
                .system(prefix.clone())
                .user("Which project does this changelog describe? Answer with just its name.")
                .build()?,
        )
        .await?;
    // A fresh cache entry can take a few seconds to become readable, so the
    // read side retries on a short ladder before the verdict.
    let mut second = None;
    for _ in 0..4 {
        // Replay serves the recorded buckets immediately; only a live or
        // recording run waits out real cache propagation.
        if support::Backend::from_env() != support::Backend::Replay {
            sleep(Duration::from_secs(5)).await;
        }
        let response = client
            .complete(
                openai::request(model)
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
