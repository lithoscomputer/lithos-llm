//! V2 — prompt caching.
//!
//! Caching on Moonshot is automatic where it exists at all, so the test is a
//! pair of sequential completions sharing one large prefix. Moonshot does not
//! report cache writes, so the second response must report a cache read.

use std::time::Duration;

use tokio::time::sleep;

use crate::moonshot::{self, family_tests};
use crate::support::{self, TestResult};

mod round_trip {
    use super::*;

    family_tests!(super::caches_a_shared_prefix);
}

/// A deterministic, varied prefix above Moonshot's documented 256-token
/// cache threshold. The frozen fixture keeps record and replay request hashes
/// stable.
fn large_prefix() -> String {
    let source: String = include_str!("../fixtures/cache_prefix.txt")
        .chars()
        .take(5_000)
        .collect();
    format!("You answer questions about this changelog:\n\n{source}")
}

async fn caches_a_shared_prefix(model: &str) -> TestResult {
    if !moonshot::capabilities(model).caching {
        return support::skip("the catalog does not claim caching");
    }
    let Some(client) = moonshot::live_client() else {
        return support::skip("MOONSHOT_API_KEY is unset");
    };
    let prefix = large_prefix();

    // Moonshot's cache is automatic. It needs no cache key or routing hint.
    let first = client
        .complete(
            moonshot::request(model)
                .system(prefix.clone())
                .user("Which project does this changelog describe? Answer with just its name.")
                .build()?,
        )
        .await?;
    // The 2026-08-29 probe needed one short propagation wait before K3
    // reported a hit, so the read side retries on a bounded ladder.
    let mut second = None;
    for _ in 0..4 {
        // Replay serves the recorded buckets immediately; only a live or
        // recording run waits out real cache propagation.
        if support::Backend::from_env() != support::Backend::Replay {
            sleep(Duration::from_secs(5)).await;
        }
        let response = client
            .complete(
                moonshot::request(model)
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
#[ignore = "live Moonshot call; run with `mise run test:e2e`"]
async fn auto_cache_off_is_accepted() -> TestResult {
    let Some(client) = moonshot::live_client() else {
        return support::skip("MOONSHOT_API_KEY is unset");
    };
    let request = moonshot::request("kimi-k3")
        .user("In one short sentence, say hello.")
        .provider_option(moonshot::PROVIDER, "auto_cache", serde_json::json!(false))
        .build()?;
    let response = client.complete(request).await?;
    assert!(!response.text().trim().is_empty());
    Ok(())
}
