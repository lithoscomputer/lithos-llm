//! V0 — preflight.
//!
//! Everything here is free or near-free. The local tests are not ignored; only
//! the live catalog checks spend network calls.

use std::env;

use lithos_llm::types::{
    AudioContent, ContentPart, ErrorKind, MediaSource, Message, Role, ToolChoice, ToolDefinition,
};
use serde_json::json;

use crate::anthropic;
use crate::support::{self, TestResult};

#[tokio::test]
async fn sampling_is_rejected_locally_where_unclaimed() -> TestResult {
    let client = anthropic::client_with_key("preflight-key");
    let request = anthropic::request("claude-fable-5")
        .user("Hello")
        .temperature(0.0)
        .build()?;
    let error = client
        .complete(request)
        .await
        .expect_err("claude-fable-5 claims no sampling, so the client must refuse");
    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    Ok(())
}

#[tokio::test]
async fn a_forced_tool_choice_is_rejected_locally_where_unclaimed() -> TestResult {
    let client = anthropic::client_with_key("preflight-key");
    let request = anthropic::request("claude-fable-5.1")
        .user("What is the weather in Paris?")
        .tool(ToolDefinition::function(
            "get_weather",
            "Reads the current weather for a city",
            json!({ "type": "object" }),
        ))
        .tool_choice(ToolChoice::Required)
        .build()?;
    let error = client
        .complete(request)
        .await
        .expect_err("claude-fable-5.1 claims no forced tool choice, so the client must refuse");
    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    Ok(())
}

#[tokio::test]
async fn audio_is_rejected_locally() -> TestResult {
    let client = anthropic::client_with_key("preflight-key");
    let request = anthropic::request("claude-haiku-4.5")
        .message(Message::new(Role::User, [
            ContentPart::Text {
                text: "Transcribe this audio.".to_owned(),
            },
            ContentPart::Audio(AudioContent::new(MediaSource::base64("QUJD", "audio/wav"))),
        ]))
        .build()?;
    let error = client
        .complete(request)
        .await
        .expect_err("Anthropic claims no audio input, so the client must refuse");
    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(error.provider_code(), Some("unsupported_capability"));
    Ok(())
}

#[tokio::test]
async fn an_oversized_output_cap_is_rejected_locally() -> TestResult {
    let client = anthropic::client_with_key("preflight-key");
    let request = anthropic::request("claude-haiku-4.5")
        .user("Hello")
        .max_output_tokens(128_000)
        .build()?;
    let error = client
        .complete(request)
        .await
        .expect_err("an output cap above the model limit must be refused");
    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(error.provider_code(), Some("max_output_tokens"));
    Ok(())
}

#[test]
fn an_unknown_model_is_a_selection_error() -> TestResult {
    let client = anthropic::client_with_key("preflight-key");
    let request = lithos_llm::Request::builder()
        .model("no-such-model-anywhere")
        .user("Hello")
        .build()?;
    assert!(
        client.resolve_route(&request).is_err(),
        "an unknown bare model id must not resolve"
    );
    Ok(())
}

/// The built-in catalog's Anthropic rows work end to end: `Client::from_env`
/// resolves the conventional `ANTHROPIC_API_KEY` mapping, the merged roster
/// resolves the route, and Anthropic answers with catalog-priced usage.
#[tokio::test]
#[ignore = "live Anthropic call; run with `mise run test:e2e`"]
async fn the_builtin_catalog_reaches_anthropic() -> TestResult {
    if let Some(skip) = support::live_only("the unproxied built-in base URL") {
        return skip;
    }
    if env::var(anthropic::KEY_VARIABLE).is_err() {
        return support::skip("ANTHROPIC_API_KEY is unset");
    }
    let client = lithos_llm::Client::from_env()?.client;
    let request = lithos_llm::Request::builder()
        .model("anthropic/claude-sonnet-5")
        .user("In one short sentence, say hello.")
        .max_output_tokens(8192)
        .build()?;
    let response = client.complete(request).await?;
    assert!(!response.text().trim().is_empty());
    let cost = response.cost.ok_or("the response carries no cost")?;
    assert!(cost.usd_micros > 0);
    Ok(())
}

/// Every configured wire id must be available from Anthropic before the E2E
/// suite attempts completions.
#[tokio::test]
#[ignore = "live Anthropic call; run with `mise run test:e2e`"]
async fn the_live_listing_contains_every_configured_wire_id() -> TestResult {
    if let Some(skip) = support::live_only("the model listing endpoint") {
        return skip;
    }
    let Ok(key) = env::var(anthropic::KEY_VARIABLE) else {
        return support::skip("ANTHROPIC_API_KEY is unset");
    };
    let listing: serde_json::Value = reqwest::Client::new()
        .get("https://api.anthropic.com/v1/models?limit=100")
        .header("x-api-key", key)
        .header("anthropic-version", "2023-06-01")
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let live: Vec<&str> = listing["data"]
        .as_array()
        .ok_or("the model listing carries no data array")?
        .iter()
        .filter_map(|model| model["id"].as_str())
        .collect();
    let catalog = anthropic::catalog();
    let provider = catalog.provider(anthropic::PROVIDER)?;
    for model in provider.models() {
        let api_model = model.api_model();
        assert!(
            live.contains(&api_model),
            "catalog model {} is gone from the live listing (wire id {api_model})",
            model.id()
        );
    }
    Ok(())
}
