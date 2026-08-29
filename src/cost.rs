//! Catalog cost estimation.

use crate::catalog::{Pricing, codec_ids};
use crate::resolver::ResolvedRoute;
use crate::types::{Cost, CostSource, Speed, TokenCounts};

pub(crate) fn estimate_catalog_cost(
    route: &ResolvedRoute,
    usage: TokenCounts,
    speed: Option<Speed>,
) -> Option<Cost> {
    let anthropic_rates = matches!(
        route.provider().codec().as_str(),
        codec_ids::ANTHROPIC_MESSAGES | codec_ids::BEDROCK_CONVERSE
    );
    catalog_cost(
        usage,
        route.model().pricing().as_ref(),
        speed,
        anthropic_rates,
    )
}

/// Prices one response from catalog rates.
///
/// The five [`TokenCounts`] buckets are disjoint, so each is priced once.
/// Reasoning tokens bill at the output rate. Cache reads bill at the
/// cached-input rate and cache writes at the cache-write rate. A non-empty
/// bucket the catalog does not price — base input and output included —
/// yields no estimate at all: billing those tokens at zero (or guessing
/// another rate) would stamp a confidently wrong figure, and no cost is more
/// honest than a wrong one. One exception: `anthropic_rates` derives a missing
/// cache-write rate as 1.25x input, Anthropic's published premium for the
/// default five-minute cache, so a migrated Anthropic or Bedrock entry
/// without the explicit field keeps estimating what the provider charges.
///
/// A long-context rate tier is selected on the whole prompt, which is the input
/// bucket plus both cache buckets. Speed rates apply last, so a speed rate wins
/// over a long-context rate.
fn catalog_cost(
    usage: TokenCounts,
    pricing: Option<&Pricing>,
    speed: Option<Speed>,
    anthropic_rates: bool,
) -> Option<Cost> {
    let prompt = usage
        .input
        .saturating_add(usage.cache_read)
        .saturating_add(usage.cache_write);
    let pricing = pricing?.for_input_tokens(prompt).for_speed(speed);
    if usage.input > 0 && pricing.input_usd_micros_per_million.is_none() {
        return None;
    }
    let input = token_cost(usage.input, pricing.input_usd_micros_per_million);
    if usage.cache_read > 0 && pricing.cached_input_usd_micros_per_million.is_none() {
        return None;
    }
    let cache_read = token_cost(
        usage.cache_read,
        pricing.cached_input_usd_micros_per_million,
    );
    let cache_write_rate = pricing.cache_write_usd_micros_per_million.or_else(|| {
        anthropic_rates.then(|| {
            pricing
                .input_usd_micros_per_million
                .map(|rate| rate.saturating_mul(5) / 4)
        })?
    });
    if usage.cache_write > 0 && cache_write_rate.is_none() {
        return None;
    }
    let cache_write = token_cost(usage.cache_write, cache_write_rate);
    if usage.billable_output() > 0 && pricing.output_usd_micros_per_million.is_none() {
        return None;
    }
    let output = token_cost(
        usage.billable_output(),
        pricing.output_usd_micros_per_million,
    );
    Some(Cost {
        usd_micros: input
            .saturating_add(cache_read)
            .saturating_add(cache_write)
            .saturating_add(output),
        source:     CostSource::Catalog,
    })
}

fn token_cost(tokens: u64, price: Option<u64>) -> u64 {
    let Some(price) = price else {
        return 0;
    };
    u64::try_from(u128::from(tokens).saturating_mul(u128::from(price)) / 1_000_000)
        .unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::{Pricing, catalog_cost};
    use crate::catalog::{LongContextPricing, SpeedPricing, SpeedRates};
    use crate::types::{CostSource, Speed, TokenCounts};

    /// One micro-dollar per input token and two per output token.
    fn pricing() -> Pricing {
        Pricing {
            input_usd_micros_per_million: Some(1_000_000),
            output_usd_micros_per_million: Some(2_000_000),
            cached_input_usd_micros_per_million: Some(100_000),
            cache_write_usd_micros_per_million: Some(500_000),
            long_context: None,
            speed: None,
        }
    }

    fn usage() -> TokenCounts {
        TokenCounts {
            input:       1_000,
            output:      1_000,
            reasoning:   1_000,
            cache_read:  1_000,
            cache_write: 1_000,
        }
    }

    #[test]
    fn prices_reasoning_at_the_output_rate_and_each_cache_bucket_at_its_own() {
        let cost = catalog_cost(usage(), Some(&pricing()), None, false);

        // 1000 input + 100 cache read + 500 cache write + 4000 for the 2000
        // tokens billed at the output rate.
        assert_eq!(cost.map(|cost| cost.usd_micros), Some(5_600));
        assert_eq!(cost.map(|cost| cost.source), Some(CostSource::Catalog));
    }

    #[test]
    fn unpriced_cache_tokens_refuse_an_estimate() {
        // A non-empty cache bucket with no rate must not produce a cost:
        // billing those tokens at zero (or the input rate) stamps a
        // confidently wrong figure on every cached call.
        let pricing = Pricing {
            cached_input_usd_micros_per_million: None,
            cache_write_usd_micros_per_million: None,
            ..pricing()
        };

        assert_eq!(catalog_cost(usage(), Some(&pricing), None, false), None);

        // The same rates still price a call that used no cache.
        let uncached = TokenCounts {
            cache_read: 0,
            cache_write: 0,
            ..usage()
        };
        assert_eq!(
            catalog_cost(uncached, Some(&pricing), None, false).map(|cost| cost.usd_micros),
            Some(5_000)
        );
    }

    #[test]
    fn one_sided_base_rates_refuse_an_estimate() {
        // A missing base rate with tokens in that bucket must not price the
        // bucket at zero.
        let no_output_rate = Pricing {
            output_usd_micros_per_million: None,
            ..pricing()
        };
        assert_eq!(
            catalog_cost(usage(), Some(&no_output_rate), None, false),
            None
        );

        let no_input_rate = Pricing {
            input_usd_micros_per_million: None,
            ..pricing()
        };
        assert_eq!(
            catalog_cost(usage(), Some(&no_input_rate), None, false),
            None
        );

        // An empty bucket needs no rate, so the present rates still price
        // the rest of the call.
        let output_only = TokenCounts {
            input: 0,
            cache_read: 0,
            cache_write: 0,
            ..usage()
        };
        assert_eq!(
            catalog_cost(output_only, Some(&no_input_rate), None, false)
                .map(|cost| cost.usd_micros),
            Some(4_000)
        );
    }

    #[test]
    fn anthropic_rates_derive_the_cache_write_premium() {
        let pricing = Pricing {
            cache_write_usd_micros_per_million: None,
            ..pricing()
        };

        let cost = catalog_cost(usage(), Some(&pricing), None, true);

        // Anthropic bills cache writes at 1.25x input, so the derived rate
        // adds 1250 to the 1000 input + 100 cache read + 4000 output.
        assert_eq!(cost.map(|cost| cost.usd_micros), Some(6_350));

        // An explicit catalog rate always wins over the derivation.
        let explicit = Pricing {
            cache_write_usd_micros_per_million: Some(500_000),
            ..pricing
        };
        assert_eq!(
            catalog_cost(usage(), Some(&explicit), None, true).map(|cost| cost.usd_micros),
            Some(5_600)
        );
    }

    #[test]
    fn prices_nothing_without_catalog_rates() {
        assert_eq!(catalog_cost(usage(), None, None, false), None);
        assert_eq!(
            catalog_cost(
                usage(),
                Some(&Pricing {
                    input_usd_micros_per_million: None,
                    output_usd_micros_per_million: None,
                    ..pricing()
                }),
                None,
                false
            ),
            None
        );
    }

    #[test]
    fn selects_the_long_context_tier_on_the_whole_prompt() {
        // The input bucket alone stays under the threshold. Adding the two
        // cache buckets crosses it, which is how the provider tiers a prompt.
        let pricing = Pricing {
            long_context: Some(LongContextPricing {
                above_input_tokens:                  2_500,
                input_usd_micros_per_million:        Some(2_000_000),
                output_usd_micros_per_million:       Some(4_000_000),
                cached_input_usd_micros_per_million: Some(200_000),
                cache_write_usd_micros_per_million:  Some(1_000_000),
            }),
            ..pricing()
        };

        let cost = catalog_cost(usage(), Some(&pricing), None, false);

        // Every rate doubles: 2000 + 200 + 1000 + 8000.
        assert_eq!(cost.map(|cost| cost.usd_micros), Some(11_200));
    }

    #[test]
    fn prices_a_fast_request_at_the_fast_rates() {
        let pricing = Pricing {
            speed: Some(SpeedPricing {
                fast: Some(SpeedRates {
                    input_usd_micros_per_million:        Some(2_000_000),
                    output_usd_micros_per_million:       Some(4_000_000),
                    cached_input_usd_micros_per_million: Some(200_000),
                    cache_write_usd_micros_per_million:  Some(1_000_000),
                }),
                ..SpeedPricing::default()
            }),
            ..pricing()
        };

        // The standard rates bill 5600, so the doubled tier bills 11200.
        assert_eq!(
            catalog_cost(usage(), Some(&pricing), Some(Speed::Fast), false)
                .map(|cost| cost.usd_micros),
            Some(11_200)
        );
        // A speed the model does not price keeps the base rates.
        assert_eq!(
            catalog_cost(usage(), Some(&pricing), Some(Speed::Economical), false)
                .map(|cost| cost.usd_micros),
            Some(5_600)
        );
        assert_eq!(
            catalog_cost(usage(), Some(&pricing), None, false).map(|cost| cost.usd_micros),
            Some(5_600)
        );
    }
}
