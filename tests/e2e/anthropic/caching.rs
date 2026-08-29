//! V2 — prompt caching.
//!
//! The native codec adds Anthropic cache-control breakpoints to a shared
//! prefix. The first call must write it and the second call must read it.
//!
//! One family representative each keeps the token cost bounded: the shared
//! prefix is the expensive part of the whole Anthropic suite.

use std::time::Duration;

use tokio::time::sleep;

use crate::anthropic::{self, family_tests};
use crate::support::{self, TestResult};

mod round_trip {
    use super::*;

    family_tests!(super::caches_a_shared_prefix);
}

/// A deterministic prefix above Anthropic's largest 4,096-token minimum.
///
/// The text is genuinely varied. Anthropic rejects large repetitive prompts.
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
    if !anthropic::capabilities(model).caching {
        return support::skip("the catalog does not claim caching");
    }
    let Some(client) = anthropic::live_client() else {
        return support::skip("ANTHROPIC_API_KEY is unset");
    };
    let prefix = large_prefix();

    // The native codec places an ephemeral cache breakpoint on the stable
    // system prefix. No cache-routing hint is needed on Anthropic.
    let first = client
        .complete(
            anthropic::request(model)
                .system(prefix.clone())
                .user("Which project does this changelog describe? Answer with just its name.")
                .build()?,
        )
        .await?;
    // The write is readable after the first response begins. A bounded retry
    // ladder tolerates short propagation delays.
    let mut second = None;
    for _ in 0..4 {
        // Replay serves the recorded buckets immediately; only a live or
        // recording run waits out real cache propagation.
        if support::Backend::from_env() != support::Backend::Replay {
            sleep(Duration::from_secs(5)).await;
        }
        let response = client
            .complete(
                anthropic::request(model)
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
        first.usage.cache_write > 0,
        "{model} did not write the prefix cache"
    );
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
#[ignore = "live Anthropic call; run with `mise run test:e2e`"]
async fn auto_cache_off_is_accepted() -> TestResult {
    let Some(client) = anthropic::live_client() else {
        return support::skip("ANTHROPIC_API_KEY is unset");
    };
    let request = anthropic::request("claude-haiku-4.5")
        .user("In one short sentence, say hello.")
        .provider_option(anthropic::PROVIDER, "auto_cache", serde_json::json!(false))
        .build()?;
    let response = client.complete(request).await?;
    assert!(!response.text().trim().is_empty());
    Ok(())
}
