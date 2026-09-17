//! V1 — smoke, every roster model.
//!
//! These are the tests that must stay green for a model to count as working
//! at all: one completion, one contract-checked stream, and the request
//! shapes every application uses — a system prompt, a multi-turn history, an
//! output cap, and a stop sequence. The completion test also carries the
//! gateway-specific cost assertion, because every response is expected to
//! carry the gateway's in-band `usage.cost`.

use lithos_llm::types::{CostSource, FinishReason, Message, Role};

use crate::support::{self, TestResult};
use crate::vercel::{self, family_tests, model_tests};

mod basic_completion {
    use super::*;

    model_tests!(super::completes_with_usage_and_provider_cost);
}

mod stream_contract {
    use super::*;

    model_tests!(super::streams_within_the_contract);
}

mod system_prompt {
    use super::*;

    model_tests!(super::honors_the_system_prompt);
}

mod multi_turn {
    use super::*;

    model_tests!(super::carries_the_conversation);
}

mod length_truncation {
    use super::*;

    model_tests!(super::truncates_at_the_output_cap);
}

mod stop_sequence {
    use super::*;

    model_tests!(super::stops_at_the_stop_sequence);
}

mod mid_conversation_system {
    use super::*;

    family_tests!(super::honors_a_mid_conversation_system_message);
}

/// A system message appended after the conversation has started rides the
/// Chat Completions protocol in place, and the gateway translates it for
/// upstreams whose native protocol has no such turn. The instruction must
/// win: an uppercase answer, where the conversation so far was lowercase
/// prose. One representative per family, since honoring it is upstream
/// behavior.
async fn honors_a_mid_conversation_system_message(model: &str) -> TestResult {
    // MiniMax M2.7 and Qwen3 Coder answered `Madrid.` in lowercase through
    // the gateway on 2026-09-17: the message is accepted but not obeyed.
    if matches!(model, "minimax-m2.7" | "qwen3-coder") {
        return support::skip("the Vercel route does not obey a mid-conversation system message");
    }
    // The gateway refuses the turn for Gemini outright (`400
    // AI_UnsupportedFunctionalityError`, 2026-09-17) instead of hoisting it.
    if model == "gemini-3.5-flash" {
        return support::skip("the Vercel route rejects a mid-conversation system message");
    }
    let Some(client) = vercel::live_client() else {
        return support::skip("AI_GATEWAY_API_KEY is unset");
    };
    // Whether the model obeys the instruction is model behavior, so one miss
    // gets one retry before it counts as a failure.
    for attempt in 0..2 {
        let request = vercel::request(model)
            .system("Answer with just the city name.")
            .user("What is the capital of France?")
            .message(Message::text(Role::Assistant, "Paris."))
            .user("And of Spain?")
            .message(Message::text(
                Role::System,
                "From now on, write every answer in uppercase letters only.",
            ))
            .build()?;
        let response = client.complete(request).await?;
        let text = response.text();
        if text.contains("MADRID") {
            return Ok(());
        }
        support::observe(&format!(
            "{model} did not follow the mid-conversation system message (attempt {attempt}): \
             {text:?}"
        ));
    }
    Err(format!("{model} never followed the mid-conversation system message").into())
}

async fn completes_with_usage_and_provider_cost(model: &str) -> TestResult {
    let Some(client) = vercel::live_client() else {
        return support::skip("AI_GATEWAY_API_KEY is unset");
    };
    let request = vercel::request(model)
        .user("In one short sentence, say hello.")
        .build()?;
    let response = client.complete(request).await?;

    assert!(
        !response.text().trim().is_empty(),
        "the completion carries no text"
    );
    assert_eq!(response.finish_reason, FinishReason::Stop);
    // The gateway passes cache reads through, and a short prompt can be
    // served entirely from cache — Kimi K2.6 reported all 15 prompt tokens as
    // `cached_tokens` on 2026-09-17 — which leaves the uncached input bucket
    // at zero. The prompt was still counted; it just landed in another bucket.
    let usage = response.usage;
    assert!(
        usage.input + usage.cache_read + usage.cache_write > 0,
        "no input tokens were counted"
    );
    assert!(usage.billable_output() > 0, "no output tokens were counted");
    assert!(
        response.raw.is_some(),
        "a complete response must carry the raw provider payload"
    );

    // The gateway-specific half: `usage.cost` arrives on every response, and
    // the catalog deliberately prices nothing, so a missing or zero cost
    // means the in-band extraction broke.
    let cost = response.cost.ok_or("the response carries no cost")?;
    assert_eq!(cost.source, CostSource::Provider);
    assert!(cost.usd_micros > 0, "the provider-reported cost is zero");
    Ok(())
}

async fn streams_within_the_contract(model: &str) -> TestResult {
    let Some(client) = vercel::live_client() else {
        return support::skip("AI_GATEWAY_API_KEY is unset");
    };
    // The emoji requirement makes the stream carry multi-byte characters, so
    // a codec that splits UTF-8 across SSE chunk boundaries corrupts visibly.
    let request = vercel::request(model)
        .user("Reply with one short greeting that contains at least two emoji.")
        .build()?;
    let stream = client.stream(request).await?;
    let (_, response) = support::checked_stream(stream).await?;

    assert!(
        !response.text().trim().is_empty(),
        "the streamed completion carries no text"
    );
    assert!(
        response.usage.total() > 0,
        "the streamed completion carries no usage"
    );
    Ok(())
}

async fn honors_the_system_prompt(model: &str) -> TestResult {
    let Some(client) = vercel::live_client() else {
        return support::skip("AI_GATEWAY_API_KEY is unset");
    };
    let request = vercel::request(model)
        .system("When the user says ping, reply with exactly the word PONG and nothing else.")
        .user("ping")
        .build()?;
    let response = client.complete(request).await?;
    assert!(
        response.text().to_uppercase().contains("PONG"),
        "the system prompt was not honored: {:?}",
        response.text()
    );
    Ok(())
}

async fn carries_the_conversation(model: &str) -> TestResult {
    let Some(client) = vercel::live_client() else {
        return support::skip("AI_GATEWAY_API_KEY is unset");
    };
    let request = vercel::request(model)
        .system("Answer with just the city name.")
        .user("What is the capital of France?")
        .message(Message::text(Role::Assistant, "Paris."))
        .user("And of Spain?")
        .build()?;
    let response = client.complete(request).await?;
    assert!(
        response.text().contains("Madrid"),
        "the multi-turn answer does not name Madrid: {:?}",
        response.text()
    );
    Ok(())
}

async fn truncates_at_the_output_cap(model: &str) -> TestResult {
    // Grok 4.6 stopped normally at 675 completion tokens under a 32-token cap
    // through the gateway on 2026-09-17: the cap is not applied on that route.
    if model == "grok-4.6" {
        return support::skip("the Vercel route does not apply the output cap");
    }
    // GPT-5.4 stops on its own at ten numbers (23 tokens) on about half of
    // its attempts (two of four raw probes on 2026-09-17), so the cell is a
    // coin flip on model behavior rather than a check of the cap.
    if model == "gpt-5.4" {
        return support::skip("the Vercel route stops before the output cap on half its attempts");
    }
    let Some(client) = vercel::live_client() else {
        return support::skip("AI_GATEWAY_API_KEY is unset");
    };
    let request = vercel::request(model)
        .user("Count upward from one, one number per line, and do not stop.")
        .max_output_tokens(32)
        .build()?;
    let response = client.complete(request).await?;
    assert_eq!(
        response.finish_reason,
        FinishReason::Length,
        "a 32-token cap on an unbounded task must truncate"
    );
    Ok(())
}

async fn stops_at_the_stop_sequence(model: &str) -> TestResult {
    // Verified through the gateway on 2026-09-17 with the same literal
    // enumeration probe: the GPT rows, MiMo, MiniMax, and Grok recited past
    // the stop word, and Nemotron rejects the `stop` field outright (`This
    // model doesn't support the stopSequences field`).
    const IGNORES_STOP: &[&str] = &[
        "gpt-5.4",
        "gpt-5.5",
        "gpt-5.6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "grok-4.6",
        "mimo-v2.5-pro",
        "minimax-m2.7",
    ];
    if IGNORES_STOP.contains(&model) {
        return support::skip("the Vercel route ignores stop sequences");
    }
    if model == "nemotron-3-super-120b-a12b" {
        return support::skip("the Vercel route rejects the stop field for this model");
    }
    let Some(client) = vercel::live_client() else {
        return support::skip("AI_GATEWAY_API_KEY is unset");
    };
    let request = vercel::request(model)
        .system("Follow the instruction literally, with no extra commentary.")
        .user("Recite the numbers one to nine as lowercase English words, separated by spaces.")
        .stop_sequence("five")
        .build()?;
    let response = client.complete(request).await?;
    assert!(
        !response.text().contains("six"),
        "the output continued past the stop sequence: {:?}",
        response.text()
    );
    Ok(())
}
