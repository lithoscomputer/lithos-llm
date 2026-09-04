//! Live behavior of the discovered Kimi K3 Shared Endpoint.
//!
//! These checks do not create portable catalog claims. They verify the one
//! workspace-scoped endpoint used for this integration and preserve the raw
//! findings that justified the provider adapter and codec.

use std::time::Duration;

use lithos_llm::types::{
    ContentPart, FinishReason, ImageContent, MediaSource, Message, ReasoningEffort, ResponseFormat,
    Role, ToolChoice, ToolDefinition,
};
use serde_json::{Value, json};
use tokio::time::sleep;

use crate::modal;
use crate::support::{self, TestResult};

const RED_SQUARE_PNG_BASE64: &str = "iVBORw0KGgoAAAAN\
SUhEUgAAAEAAAABACAIAAAAlC+aJAAAAS0lEQVR42u3PQQkAAAgAsetfWiP4FgYrsKZeS0BAQEBA\
QEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEDgsqnc8OJg6Ln3AAAAAElF\
TkSuQmCC";

const HARD_PROMPT: &str = "A bookshop sells notebooks at $7 and pens at $3. Anna spent exactly \
     $118 and bought at least one of each. How many different combinations of notebooks and pens \
     could she have bought? Answer with just the number.";

#[tokio::test]
#[ignore = "live Modal call; run with `mise run test:e2e:live`"]
async fn accepts_sampling_parameters() -> TestResult {
    if let Some(skip) = support::live_only("Modal has no two-header twin") {
        return skip;
    }
    let Some(endpoint) = modal::live_endpoint().await? else {
        return modal::missing_credentials();
    };
    let request = modal::request(&endpoint.model)
        .user("In one short sentence, say hello.")
        .temperature(0.0)
        .top_p(1.0)
        .build()?;
    let response = endpoint.client.complete(request).await?;

    assert!(!response.text().trim().is_empty());
    assert!(response.warnings.is_empty());
    Ok(())
}

#[tokio::test]
#[ignore = "live Modal call; run with `mise run test:e2e:live`"]
async fn produces_strict_structured_output() -> TestResult {
    if let Some(skip) = support::live_only("Modal has no two-header twin") {
        return skip;
    }
    let Some(endpoint) = modal::live_endpoint().await? else {
        return modal::missing_credentials();
    };
    let request = modal::request(&endpoint.model)
        .user("Report the city of the Eiffel Tower and its approximate population.")
        .response_format(ResponseFormat::JsonSchema {
            name:   "city_report".to_owned(),
            schema: json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "city":       { "type": "string" },
                    "population": { "type": "integer" },
                },
                "required": ["city", "population"],
            }),
        })
        .build()?;
    let response = endpoint.client.complete(request).await?;
    let payload = support::json_payload(&response)?;

    assert!(
        payload
            .get("city")
            .and_then(Value::as_str)
            .is_some_and(|city| !city.is_empty()),
        "the report carries no city: {payload}"
    );
    assert!(
        payload.get("population").is_some_and(Value::is_i64),
        "the report carries no integer population: {payload}"
    );
    Ok(())
}

#[tokio::test]
#[ignore = "live Modal call; run with `mise run test:e2e:live`"]
async fn calls_a_forced_tool() -> TestResult {
    if let Some(skip) = support::live_only("Modal has no two-header twin") {
        return skip;
    }
    let Some(endpoint) = modal::live_endpoint().await? else {
        return modal::missing_credentials();
    };
    let request = modal::request(&endpoint.model)
        .user("What is the weather in Paris?")
        .tool(weather_tool())
        .tool_choice(ToolChoice::Tool {
            name: "get_weather".to_owned(),
        })
        .build()?;
    let response = endpoint.client.complete(request).await?;
    let calls = support::tool_calls(&response);
    let call = calls.first().ok_or("the response carries no tool call")?;

    assert_eq!(call.name, "get_weather");
    assert!(!call.id.is_empty());
    assert_eq!(
        call.input.to_value().expect("valid tool input")["city"],
        "Paris"
    );
    assert_eq!(response.finish_reason, FinishReason::ToolCall);
    Ok(())
}

#[tokio::test]
#[ignore = "live Modal call; run with `mise run test:e2e:live`"]
async fn accepts_low_high_and_max_reasoning_effort() -> TestResult {
    if let Some(skip) = support::live_only("Modal has no two-header twin") {
        return skip;
    }
    let Some(endpoint) = modal::live_endpoint().await? else {
        return modal::missing_credentials();
    };

    for effort in [
        ReasoningEffort::Low,
        ReasoningEffort::High,
        ReasoningEffort::Max,
    ] {
        let request = modal::request(&endpoint.model)
            .user(HARD_PROMPT)
            .reasoning_effort(effort)
            .max_output_tokens(512)
            .build()?;
        let response = endpoint.client.complete(request).await?;
        let reasoning_parts = response
            .content
            .iter()
            .filter(|part| matches!(part, ContentPart::Reasoning(_)))
            .count();

        assert!(
            matches!(
                response.finish_reason,
                FinishReason::Stop | FinishReason::Length
            ),
            "effort {effort:?} was not answered normally"
        );
        assert!(
            response.usage.reasoning > 0 || reasoning_parts > 0,
            "effort {effort:?} produced no reasoning evidence"
        );
        support::observe(&format!(
            "Modal Kimi K3 effort {effort:?}: reasoning tokens {}, reasoning parts \
             {reasoning_parts}",
            response.usage.reasoning
        ));
    }
    Ok(())
}

#[tokio::test]
#[ignore = "live Modal call; run with `mise run test:e2e:live`"]
async fn describes_an_inline_image() -> TestResult {
    if let Some(skip) = support::live_only("Modal has no two-header twin") {
        return skip;
    }
    let Some(endpoint) = modal::live_endpoint().await? else {
        return modal::missing_credentials();
    };
    let request = modal::request(&endpoint.model)
        .message(Message::new(Role::User, [
            ContentPart::Text {
                text: "What single color fills this image? Answer with the color name.".to_owned(),
            },
            ContentPart::Image(ImageContent {
                source: MediaSource::base64(RED_SQUARE_PNG_BASE64, "image/png"),
                detail: None,
            }),
        ]))
        .build()?;
    let response = endpoint.client.complete(request).await?;

    assert!(
        response.text().to_lowercase().contains("red"),
        "the model did not see a red image: {:?}",
        response.text()
    );
    Ok(())
}

#[tokio::test]
#[ignore = "live Modal call; run with `mise run test:e2e:live`"]
async fn reads_an_automatic_prompt_cache() -> TestResult {
    if let Some(skip) = support::live_only("Modal has no two-header twin") {
        return skip;
    }
    let Some(endpoint) = modal::live_endpoint().await? else {
        return modal::missing_credentials();
    };
    let source: String = include_str!("../fixtures/cache_prefix.txt")
        .chars()
        .take(5_000)
        .collect();
    let prefix = format!("You answer questions about this changelog:\n\n{source}");
    let first = endpoint
        .client
        .complete(
            modal::request(&endpoint.model)
                .system(prefix.clone())
                .user("Which project does this changelog describe? Answer with just its name.")
                .cache_key("lithos-modal-kimi-k3")
                .build()?,
        )
        .await?;
    let mut second = None;
    for _ in 0..3 {
        sleep(Duration::from_secs(5)).await;
        let response = endpoint
            .client
            .complete(
                modal::request(&endpoint.model)
                    .system(prefix.clone())
                    .user("Does the changelog mention streaming? Answer yes or no.")
                    .cache_key("lithos-modal-kimi-k3")
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
        "Modal Kimi K3 cache buckets: first write={} read={}, second write={} read={}",
        first.usage.cache_write,
        first.usage.cache_read,
        second.usage.cache_write,
        second.usage.cache_read,
    ));
    assert!(
        first.usage.cache_read > 0 || first.usage.cache_write > 0 || second.usage.cache_read > 0,
        "Modal Kimi K3 reported no cache activity across the shared-prefix pair"
    );
    Ok(())
}

fn weather_tool() -> ToolDefinition {
    ToolDefinition::function(
        "get_weather",
        "Reads the current weather for a city",
        json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
        }),
    )
}
