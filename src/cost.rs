//! Catalog cost estimation.

use crate::catalog::{CodecFamily, CodecId, Pricing, codec_ids, family};
use crate::resolver::ResolvedRoute;
use crate::types::{Cost, CostSource, Speed, TokenCounts};

/// Prices `usage` on `route` at the rates of `codec`.
///
/// `codec` is the codec the call went out on, when the caller knows it; it
/// decides whether Anthropic's cache-write derivation applies (see
/// [`catalog_cost`]). A caller without a codec, such as
/// [`ResolvedRoute::estimate_cost`], passes `None` and the route's first
/// generation codec stands in, which is the codec a call on that route
/// would select.
pub(crate) fn estimate_catalog_cost(
    route: &ResolvedRoute,
    codec: Option<&CodecId>,
    usage: TokenCounts,
    speed: Option<Speed>,
) -> Option<Cost> {
    let codec = codec.or_else(|| {
        route
            .model()
            .codecs()
            .iter()
            .find(|codec| family(codec) == CodecFamily::Generation)
    });
    let anthropic_rates = codec.is_some_and(|codec| {
        matches!(
            codec.as_str(),
            codec_ids::ANTHROPIC_MESSAGES | codec_ids::BEDROCK_CONVERSE
        )
    });
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
/// default five-minute cache, and a missing long cache-write rate as 2x input,
/// its premium for the 1-hour cache, so a migrated Anthropic or Bedrock entry
/// without the explicit fields keeps estimating what the provider charges.
///
/// Cache writes made with the longer lifetime, a share of the cache-write
/// bucket, bill at the long cache-write rate and the rest at the cache-write
/// rate.
///
/// A long-context rate tier is selected on the whole prompt, which is the input
/// bucket plus both cache buckets. A speed rate wins over a long-context rate,
/// unless the tier scales speeds (see [`Pricing::for_request`]).
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
    let pricing = pricing?.for_request(prompt, speed);
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
    let long_writes = usage.long_cache_write.min(usage.cache_write);
    let short_writes = usage.cache_write - long_writes;
    if short_writes > 0 && cache_write_rate.is_none() {
        return None;
    }
    let long_write_rate = pricing.long_cache_write_usd_micros_per_million.or_else(|| {
        anthropic_rates.then(|| {
            pricing
                .input_usd_micros_per_million
                .map(|rate| rate.saturating_mul(2))
        })?
    });
    if long_writes > 0 && long_write_rate.is_none() {
        return None;
    }
    let cache_write = token_cost(short_writes, cache_write_rate)
        .saturating_add(token_cost(long_writes, long_write_rate));
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

/// Converts US dollars to the integer micros the cost type carries.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a float-to-integer cast saturates, which is the clamp a provider-reported cost needs"
)]
pub(crate) fn usd_micros(usd: f64) -> u64 {
    (usd * 1_000_000.0).round() as u64
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
            long_cache_write_usd_micros_per_million: None,
            long_context: None,
            speed: None,
        }
    }

    fn usage() -> TokenCounts {
        TokenCounts {
            input:            1_000,
            output:           1_000,
            reasoning:        1_000,
            cache_read:       1_000,
            cache_write:      1_000,
            long_cache_write: 0,
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
    fn long_lived_cache_writes_bill_at_their_own_rate() {
        // 300 of the 1000 cache writes used the longer lifetime.
        let usage = usage().with_long_cache_write(300);
        let pricing = Pricing {
            long_cache_write_usd_micros_per_million: Some(800_000),
            ..pricing()
        };

        // 1000 input + 100 read + 700 x 0.5 + 300 x 0.8 + 4000 output.
        assert_eq!(
            catalog_cost(usage, Some(&pricing), None, false).map(|cost| cost.usd_micros),
            Some(5_690)
        );
        // Without the rate there is no honest estimate, unless the rates are
        // Anthropic's, whose 1-hour write is 2x input: 300 x 2.0 = 600.
        assert_eq!(
            catalog_cost(usage, Some(&self::pricing()), None, false),
            None
        );
        assert_eq!(
            catalog_cost(usage, Some(&self::pricing()), None, true).map(|cost| cost.usd_micros),
            Some(6_050)
        );
    }

    #[test]
    fn unpriced_cache_tokens_refuse_an_estimate() {
        // A non-empty cache bucket with no rate must not produce a cost:
        // billing those tokens at zero (or the input rate) stamps a
        // confidently wrong figure on every cached call.
        let pricing = Pricing {
            cached_input_usd_micros_per_million: None,
            cache_write_usd_micros_per_million: None,
            long_cache_write_usd_micros_per_million: None,
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
            long_cache_write_usd_micros_per_million: None,
            ..pricing()
        };

        let cost = catalog_cost(usage(), Some(&pricing), None, true);

        // Anthropic bills cache writes at 1.25x input, so the derived rate
        // adds 1250 to the 1000 input + 100 cache read + 4000 output.
        assert_eq!(cost.map(|cost| cost.usd_micros), Some(6_350));

        // An explicit catalog rate always wins over the derivation.
        let explicit = Pricing {
            cache_write_usd_micros_per_million: Some(500_000),
            long_cache_write_usd_micros_per_million: None,
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
                above_input_tokens: 2_500,
                input_usd_micros_per_million: Some(2_000_000),
                output_usd_micros_per_million: Some(4_000_000),
                cached_input_usd_micros_per_million: Some(200_000),
                cache_write_usd_micros_per_million: Some(1_000_000),
                long_cache_write_usd_micros_per_million: None,
                scales_speeds: false,
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
                    input_usd_micros_per_million:            Some(2_000_000),
                    output_usd_micros_per_million:           Some(4_000_000),
                    cached_input_usd_micros_per_million:     Some(200_000),
                    cache_write_usd_micros_per_million:      Some(1_000_000),
                    long_cache_write_usd_micros_per_million: None,
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

    #[test]
    fn a_long_prompt_at_a_speed_pays_the_long_context_premium_when_it_scales_speeds() {
        // The fixture's prompt is 3000 tokens, above the 2500 threshold. Long
        // context doubles input and leaves the other rates; fast doubles
        // every rate.
        let pricing = |scales_speeds| Pricing {
            long_context: Some(LongContextPricing {
                above_input_tokens: 2_500,
                input_usd_micros_per_million: Some(2_000_000),
                output_usd_micros_per_million: None,
                cached_input_usd_micros_per_million: None,
                cache_write_usd_micros_per_million: None,
                long_cache_write_usd_micros_per_million: None,
                scales_speeds,
            }),
            speed: Some(SpeedPricing {
                fast: Some(SpeedRates {
                    input_usd_micros_per_million:            Some(2_000_000),
                    output_usd_micros_per_million:           Some(4_000_000),
                    cached_input_usd_micros_per_million:     Some(200_000),
                    cache_write_usd_micros_per_million:      Some(1_000_000),
                    long_cache_write_usd_micros_per_million: None,
                }),
                ..SpeedPricing::default()
            }),
            ..pricing()
        };
        let cost = |scales_speeds, speed| {
            catalog_cost(usage(), Some(&pricing(scales_speeds)), speed, false)
                .map(|cost| cost.usd_micros)
        };

        // Fast and long context: 4000 input + 200 + 1000 + 8000.
        assert_eq!(cost(true, Some(Speed::Fast)), Some(13_200));
        // Without scaling, fast applies across the prompt: 2000 + 9200.
        assert_eq!(cost(false, Some(Speed::Fast)), Some(11_200));
        // At standard speed both price the long-context input: 2000 + 4600.
        assert_eq!(cost(true, None), Some(6_600));
        // A speed the model does not price keeps the long-context base rates.
        assert_eq!(cost(true, Some(Speed::Economical)), Some(6_600));
    }
}
