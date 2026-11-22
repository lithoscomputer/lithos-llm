//! Proves that each wire format's `Outcome` reads how a response ended, which
//! model served it, and what it was billed for, and that the catalog prices
//! it as the provider bills it.
//!
//! The payloads follow the providers' documented shapes. Expected costs come
//! from the built-in catalog's published rates, worked by hand: Claude Opus
//! 5.5 bills $4 and $20 per million input and output tokens, Claude Opus 4.8
//! $5 and $25, GPT-6 Sol $2 and $10, and GPT-6 Luna $0.10 and $0.50.

#![cfg(feature = "builtin-catalog")]

use std::error::Error as StdError;

use lithos_llm::catalog::{Catalog, Offering};
use lithos_llm::types::{Cost, CostSource, FinishReason, Speed, TokenCounts};
use lithos_llm::wire::{Outcome, ServedTier, anthropic_messages, openai_chat, openai_responses};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

type TestResult = Result<(), Box<dyn StdError>>;

fn parse<T: DeserializeOwned>(value: Value) -> Result<T, serde_json::Error> {
    serde_json::from_value(value)
}

/// The cost `outcome` estimates for a request for `model` on `provider`.
fn cost(catalog: &Catalog, provider: &str, model: &str, outcome: &Outcome) -> Option<u64> {
    let offering: Offering<'_> = catalog.provider(provider).ok()?.offering(model)?;
    offering.estimate_outcome_cost(outcome).map(|cost: Cost| {
        assert_eq!(cost.source, CostSource::Catalog);
        cost.usd_micros
    })
}

fn anthropic_refusal(category: &str, output_tokens: u64) -> Value {
    json!({
        "id": "msg_01", "type": "message", "role": "assistant",
        "model": "claude-opus-5-5", "content": [],
        "stop_reason": "refusal",
        "stop_details": {
            "type": "refusal", "category": category,
            "explanation": "This request was declined."
        },
        "usage": {"input_tokens": 412, "output_tokens": output_tokens}
    })
}

#[test]
fn anthropic_refusals_cost_what_anthropic_bills() -> TestResult {
    let catalog = Catalog::builder().with_builtin().build()?;

    let refused = parse::<anthropic_messages::Response>(anthropic_refusal("cyber", 0))?.outcome();
    let refusal = refused.refusal().ok_or("expected a refusal")?;
    assert_eq!(refusal.category.as_deref(), Some("cyber"));
    assert_eq!(refused.stop_reason(), Some("refusal"));
    assert_eq!(refused.model(), Some("claude-opus-5-5"));
    let [pass] = refused.passes().try_into().map_err(|_| "one pass")?;
    assert!(
        !pass.billed,
        "a cyber refusal before any output is not billed"
    );
    assert_eq!(
        cost(&catalog, "anthropic", "claude-opus-5.5", &refused),
        Some(0)
    );

    // `bio` refusals are billed even before any output: 412 input at $4.
    let bio = parse::<anthropic_messages::Response>(anthropic_refusal("bio", 0))?.outcome();
    assert_eq!(
        cost(&catalog, "anthropic", "claude-opus-5.5", &bio),
        Some(1648)
    );

    // Output streamed before the refusal is billed in every category: 412 at
    // $4 plus 37 at $20.
    let late = parse::<anthropic_messages::Response>(anthropic_refusal("cyber", 37))?.outcome();
    assert_eq!(
        cost(&catalog, "anthropic", "claude-opus-5.5", &late),
        Some(2388)
    );
    Ok(())
}

fn fallback_response(category: &str) -> Value {
    json!({
        "id": "msg_02", "type": "message", "role": "assistant",
        "model": "claude-opus-4-8",
        "content": [
            {
                "type": "fallback",
                "from": {"model": "claude-opus-5-5"},
                "to": {"model": "claude-opus-4-8"},
                "trigger": {"type": "refusal", "category": category}
            },
            {"type": "text", "text": "Hi"}
        ],
        "stop_reason": "end_turn", "stop_details": null,
        "usage": {
            "input_tokens": 412, "output_tokens": 264,
            "iterations": [
                {"type": "message", "model": "claude-opus-5-5", "input_tokens": 535, "output_tokens": 0},
                {"type": "fallback_message", "model": "claude-opus-4-8", "input_tokens": 412, "output_tokens": 264}
            ]
        }
    })
}

#[test]
fn an_anthropic_fallback_is_billed_per_attempt_at_each_models_rates() -> TestResult {
    let catalog = Catalog::builder().with_builtin().build()?;

    let served = parse::<anthropic_messages::Response>(fallback_response("bio"))?.outcome();
    assert_eq!(served.model(), Some("claude-opus-4-8"));
    assert!(served.refusal().is_none(), "the fallback answered");
    let [fallback] = served.fallbacks() else {
        return Err("expected one fallback".into());
    };
    assert_eq!(fallback.from, "claude-opus-5-5");
    assert_eq!(fallback.to, "claude-opus-4-8");
    assert_eq!(fallback.category.as_deref(), Some("bio"));
    // The declined `bio` attempt is billed at Opus 5.5's rates (535 at $4),
    // and the answer at Opus 4.8's (412 at $5 plus 264 at $25).
    assert_eq!(
        cost(&catalog, "anthropic", "claude-opus-5.5", &served),
        Some(2140 + 2060 + 6600)
    );

    // A declined `cyber` attempt with no output is not billed.
    let cyber = parse::<anthropic_messages::Response>(fallback_response("cyber"))?.outcome();
    let billed: Vec<bool> = cyber.passes().iter().map(|pass| pass.billed).collect();
    assert_eq!(billed, [false, true]);
    assert_eq!(
        cost(&catalog, "anthropic", "claude-opus-5.5", &cyber),
        Some(2060 + 6600)
    );
    Ok(())
}

#[test]
fn an_anthropic_stream_folds_into_the_same_outcome() -> TestResult {
    let catalog = Catalog::builder().with_builtin().build()?;
    let events = [
        json!({"type": "message_start", "message": {
            "id": "msg_03", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [], "stop_reason": null, "stop_sequence": null,
            "usage": {"input_tokens": 412, "output_tokens": 1}
        }}),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Sure"}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({"type": "content_block_start", "index": 1, "content_block": {
            "type": "fallback", "from": {"model": "claude-opus-5-5"},
            "to": {"model": "claude-opus-4-8"}, "trigger": {"type": "refusal", "category": "cyber"}
        }}),
        json!({"type": "content_block_stop", "index": 1}),
        json!({"type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": null},
        "usage": {"input_tokens": 430, "output_tokens": 264, "iterations": [
            {"type": "message", "model": "claude-opus-5-5", "input_tokens": 412, "output_tokens": 37},
            {"type": "fallback_message", "model": "claude-opus-4-8", "input_tokens": 430, "output_tokens": 264}
        ]}}),
        json!({"type": "message_stop"}),
    ];
    let mut outcome = Outcome::default();
    for event in events {
        parse::<anthropic_messages::StreamEvent>(event)?.fold_into(&mut outcome);
    }

    // `message_start` named the requested model; the fallback block names
    // the model that finished.
    assert_eq!(outcome.model(), Some("claude-opus-4-8"));
    assert_eq!(outcome.stop_reason(), Some("end_turn"));
    assert_eq!(outcome.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(outcome.usage().map(|usage| usage.output), Some(264));
    // The declined attempt streamed output, so it is billed: 412 at $4 and 37
    // at $20, then 430 at $5 and 264 at $25.
    assert_eq!(
        cost(&catalog, "anthropic", "claude-opus-5.5", &outcome),
        Some(1648 + 740 + 2150 + 6600)
    );

    let mut failed = Outcome::default();
    parse::<anthropic_messages::StreamEvent>(json!({
        "type": "error",
        "error": {"type": "overloaded_error", "message": "Overloaded"}
    }))?
    .fold_into(&mut failed);
    let error = failed.error().ok_or("expected an error")?;
    assert_eq!(error.code.as_deref(), Some("overloaded_error"));
    Ok(())
}

#[test]
fn compaction_passes_are_billed_beside_the_message() -> TestResult {
    let catalog = Catalog::builder().with_builtin().build()?;
    let outcome = parse::<anthropic_messages::Response>(json!({
        "id": "msg_04", "model": "claude-opus-5-5", "content": [],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 23000, "output_tokens": 1000,
            "iterations": [
                {"type": "compaction", "input_tokens": 180_000, "output_tokens": 3500},
                {"type": "message", "input_tokens": 23000, "output_tokens": 1000}
            ]
        }
    }))?
    .outcome();

    // The top-level counters leave the compaction pass out.
    assert_eq!(outcome.usage().map(|usage| usage.input), Some(23000));
    // 180,000 at $4 and 3,500 at $20, then 23,000 at $4 and 1,000 at $20.
    assert_eq!(
        cost(&catalog, "anthropic", "claude-opus-5.5", &outcome),
        Some(720_000 + 70_000 + 92_000 + 20_000)
    );
    Ok(())
}

#[test]
fn the_served_tier_selects_the_rates_or_none() -> TestResult {
    let catalog = Catalog::builder().with_builtin().build()?;
    let anthropic = |usage: Value| -> Result<Outcome, serde_json::Error> {
        Ok(parse::<anthropic_messages::Response>(json!({
            "id": "msg_05", "model": "claude-opus-5-5", "content": [],
            "stop_reason": "end_turn", "usage": usage
        }))?
        .outcome())
    };

    let fast = anthropic(json!({"input_tokens": 10, "output_tokens": 10, "speed": "fast"}))?;
    assert_eq!(fast.served_tier(), ServedTier::Speed(Speed::Fast));
    // Priority Tier is billed under a commitment the catalog cannot price.
    let priority = anthropic(json!({
        "input_tokens": 10, "output_tokens": 10, "service_tier": "priority"
    }))?;
    assert_eq!(priority.served_tier(), ServedTier::Unpriced);
    assert_eq!(
        cost(&catalog, "anthropic", "claude-opus-5.5", &priority),
        None
    );

    let openai = |tier: &str| -> Result<ServedTier, serde_json::Error> {
        Ok(parse::<openai_responses::Response>(json!({
            "id": "resp_1", "model": "gpt-6-sol", "status": "completed",
            "service_tier": tier, "output": []
        }))?
        .outcome()
        .served_tier())
    };
    assert_eq!(openai("default")?, ServedTier::Standard);
    assert_eq!(openai("priority")?, ServedTier::Speed(Speed::Fast));
    assert_eq!(openai("flex")?, ServedTier::Speed(Speed::Economical));
    assert_eq!(openai("scale")?, ServedTier::Unpriced);
    Ok(())
}

fn responses_usage() -> Value {
    json!({
        "input_tokens": 300, "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 12, "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 312
    })
}

#[test]
fn responses_outcomes_name_the_served_model_and_why_they_stopped() -> TestResult {
    let catalog = Catalog::builder().with_builtin().build()?;

    // Rerouted to a dated GPT-6 Luna snapshot: priced at Luna's rates, 300 at
    // $0.10 and 12 at $0.50, not Sol's.
    let rerouted = parse::<openai_responses::Response>(json!({
        "id": "resp_2", "model": "gpt-6-luna-2026-10-01", "status": "completed",
        "output": [], "usage": responses_usage()
    }))?
    .outcome();
    assert_eq!(rerouted.model(), Some("gpt-6-luna-2026-10-01"));
    assert_eq!(
        cost(&catalog, "openai", "gpt-6-sol", &rerouted),
        Some(30 + 6)
    );

    let filtered = parse::<openai_responses::Response>(json!({
        "id": "resp_3", "model": "gpt-6-sol", "status": "incomplete",
        "incomplete_details": {"reason": "content_filter"},
        "output": [], "usage": responses_usage()
    }))?
    .outcome();
    assert_eq!(filtered.stop_reason(), Some("content_filter"));
    assert_eq!(filtered.finish_reason(), Some(FinishReason::ContentFilter));
    assert!(filtered.refusal().is_none());

    let refused = parse::<openai_responses::Response>(json!({
        "id": "resp_4", "model": "gpt-6-sol", "status": "completed",
        "output": [{"type": "message", "role": "assistant", "content": [
            {"type": "refusal", "refusal": "I can't help with that."}
        ]}],
        "usage": responses_usage()
    }))?
    .outcome();
    let refusal = refused.refusal().ok_or("expected a refusal")?;
    assert_eq!(
        refusal.explanation.as_deref(),
        Some("I can't help with that.")
    );
    // OpenAI bills a refusal like any answer: 300 at $2 and 12 at $10.
    assert_eq!(
        cost(&catalog, "openai", "gpt-6-sol", &refused),
        Some(600 + 120)
    );

    let mut stream = Outcome::default();
    for event in [
        json!({"type": "response.created", "sequence_number": 0, "response": {
            "id": "resp_5", "model": "gpt-6-sol", "status": "in_progress", "output": []
        }}),
        json!({"type": "response.failed", "sequence_number": 1, "response": {
            "id": "resp_5", "model": "gpt-6-sol", "status": "failed",
            "error": {"code": "bio_policy", "message": "Blocked."}, "output": [], "usage": null
        }}),
        json!({"type": "error", "sequence_number": 2, "code": "cyber_policy",
            "message": "Flagged.", "param": null}),
    ] {
        parse::<openai_responses::StreamEvent>(event)?.fold_into(&mut stream);
    }
    assert_eq!(stream.stop_reason(), Some("failed"));
    assert_eq!(stream.finish_reason(), Some(FinishReason::Error));
    let error = stream.error().ok_or("expected an error")?;
    assert_eq!(error.code.as_deref(), Some("cyber_policy"));
    assert_eq!(
        cost(&catalog, "openai", "gpt-6-sol", &stream),
        None,
        "no usage"
    );
    Ok(())
}

#[test]
fn chat_outcomes_read_refusals_and_the_providers_own_cost() -> TestResult {
    let mut outcome = Outcome::default();
    for chunk in [
        json!({"id": "c1", "model": "gpt-6-sol", "choices": [
            {"index": 0, "delta": {"role": "assistant", "refusal": "I can't"}, "finish_reason": null}
        ]}),
        json!({"id": "c1", "model": "gpt-6-sol", "choices": [
            {"index": 0, "delta": {"refusal": " help."}, "finish_reason": "content_filter"}
        ]}),
        json!({"id": "c1", "model": "gpt-6-sol", "choices": [],
            "usage": {"prompt_tokens": 300, "completion_tokens": 12, "total_tokens": 312}}),
    ] {
        parse::<openai_chat::StreamEvent>(chunk)?.fold_into(&mut outcome);
    }
    let refusal = outcome.refusal().ok_or("expected a refusal")?;
    assert_eq!(refusal.explanation.as_deref(), Some("I can't help."));
    assert_eq!(outcome.finish_reason(), Some(FinishReason::ContentFilter));
    assert_eq!(
        outcome.usage(),
        Some(TokenCounts {
            input: 300,
            output: 12,
            ..TokenCounts::default()
        })
    );

    // OpenRouter with the caller's own key: its fee plus the provider's
    // charge.
    let byok = parse::<openai_chat::Response>(json!({
        "id": "c2", "model": "anthropic/claude-opus-5.5",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6,
            "cost": 0.001, "is_byok": true,
            "cost_details": {"upstream_inference_cost": 0.5}
        }
    }))?
    .outcome();
    assert_eq!(
        byok.provider_cost().map(|cost| cost.usd_micros),
        Some(501_000)
    );

    // Venice values a diem at one dollar.
    let venice = parse::<openai_chat::Response>(json!({
        "id": "c3", "model": "venice-uncensored",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        "cost": {"usd": 0.25, "diem": 9.5}
    }))?
    .outcome();
    assert_eq!(
        venice.provider_cost().map(|cost| cost.usd_micros),
        Some(9_750_000)
    );
    Ok(())
}
