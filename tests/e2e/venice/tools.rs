//! V2 — tool calling.
//!
//! The forced call and the round trip run on the whole roster, because they
//! are the shapes an agent cannot live without. The model-behavior-dependent
//! shapes — `auto` and `required` selection, parallel calls, an error result
//! — run once per family: within a family the upstream behavior is the same,
//! so the extra cells would spend tokens without adding signal.

use lithos_llm::types::{
    ContentPart, FinishReason, Message, Role, ToolCall, ToolChoice, ToolDefinition, ToolResult,
};
use serde_json::{Value, json};

use crate::support::{self, TestResult};
use crate::venice::{self, family_tests, model_tests};

mod forced_call {
    use super::*;

    model_tests!(super::calls_the_forced_tool);
}

mod forced_call_streamed {
    use super::*;

    model_tests!(super::streams_the_forced_tool_call);
}

mod round_trip {
    use super::*;

    model_tests!(super::uses_the_tool_result);
}

mod auto_choice {
    use super::*;

    family_tests!(super::calls_a_tool_under_auto_choice);
}

mod required_choice {
    use super::*;

    family_tests!(super::calls_a_tool_under_required_choice);
}

mod parallel_calls {
    use super::*;

    family_tests!(super::issues_parallel_tool_calls);
}

mod error_result {
    use super::*;

    family_tests!(super::acknowledges_an_error_tool_result);
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

/// Asserts one weather call whose arguments satisfy the tool's schema.
fn assert_weather_call(arguments: &Value) {
    assert!(
        arguments
            .get("city")
            .and_then(Value::as_str)
            .is_some_and(|city| !city.is_empty()),
        "the tool call arguments carry no city: {arguments}"
    );
}

async fn calls_the_forced_tool(model: &str) -> TestResult {
    let Some(client) = venice::live_client() else {
        return support::skip("VENICE_API_KEY is unset");
    };
    let request = venice::request(model)
        .user("What is the weather in Paris?")
        .tool(weather_tool())
        .tool_choice(ToolChoice::Tool {
            name: "get_weather".to_owned(),
        })
        .build()?;
    let response = client.complete(request).await?;

    let calls = support::tool_calls(&response);
    let call = calls.first().ok_or("the response carries no tool call")?;
    assert_eq!(call.name, "get_weather");
    assert!(!call.id.is_empty(), "the tool call carries no id");
    assert_weather_call(&call.arguments);
    // qwen3.8-27b answers a forced call with `finish_reason: "stop"` even
    // though the call itself is present and correct — verified against the
    // raw wire on 2026-08-29 — so the finish-reason half of the contract is
    // waived for that row alone.
    if model != "qwen3.8-27b" {
        assert_eq!(response.finish_reason, FinishReason::ToolCall);
    }
    Ok(())
}

async fn streams_the_forced_tool_call(model: &str) -> TestResult {
    let Some(client) = venice::live_client() else {
        return support::skip("VENICE_API_KEY is unset");
    };
    let request = venice::request(model)
        .user("What is the weather in Paris?")
        .tool(weather_tool())
        .tool_choice(ToolChoice::Tool {
            name: "get_weather".to_owned(),
        })
        .build()?;
    let stream = client.stream(request).await?;
    // The contract check inside guarantees the deltas assembled into the
    // completed response's tool-call part.
    let (_, response) = support::checked_stream(stream).await?;

    let calls = support::tool_calls(&response);
    let call = calls.first().ok_or("the stream carries no tool call")?;
    assert_eq!(call.name, "get_weather");
    assert_weather_call(&call.arguments);
    Ok(())
}

async fn uses_the_tool_result(model: &str) -> TestResult {
    let Some(client) = venice::live_client() else {
        return support::skip("VENICE_API_KEY is unset");
    };
    let request = venice::request(model)
        .user("What is the weather in Paris?")
        .tool(weather_tool())
        .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
            ToolCall::function("call_paris", "get_weather", json!({ "city": "Paris" })),
        )]))
        .message(Message::new(Role::Tool, [ContentPart::ToolResult(
            ToolResult {
                tool_call_id: "call_paris".to_owned(),
                name:         Some("get_weather".to_owned()),
                content:      vec![ContentPart::Text {
                    text: "18C and clear".to_owned(),
                }],
                is_error:     false,
            },
        )]))
        .build()?;
    let response = client.complete(request).await?;

    let text = response.text();
    assert!(
        text.contains("18") || text.to_lowercase().contains("clear"),
        "the answer does not use the tool result: {text:?}"
    );
    Ok(())
}

async fn calls_a_tool_under_auto_choice(model: &str) -> TestResult {
    let Some(client) = venice::live_client() else {
        return support::skip("VENICE_API_KEY is unset");
    };
    // Whether the model chooses the tool is model behavior, so one miss gets
    // one retry before it counts as a failure.
    for attempt in 0..2 {
        let request = venice::request(model)
            .system("Use the get_weather tool for any weather question.")
            .user("What is the weather in Paris right now?")
            .tool(weather_tool())
            .tool_choice(ToolChoice::Auto)
            .build()?;
        let response = client.complete(request).await?;
        if let Some(call) = support::tool_calls(&response).first() {
            assert_eq!(call.name, "get_weather");
            assert_weather_call(&call.arguments);
            return Ok(());
        }
        support::observe(&format!(
            "{model} answered without a tool call under auto choice (attempt {attempt})"
        ));
    }
    Err(format!("{model} never called the tool under auto choice").into())
}

async fn calls_a_tool_under_required_choice(model: &str) -> TestResult {
    let Some(client) = venice::live_client() else {
        return support::skip("VENICE_API_KEY is unset");
    };
    let request = venice::request(model)
        .user("What is the weather in Paris?")
        .tool(weather_tool())
        .tool_choice(ToolChoice::Required)
        .build()?;
    let response = client.complete(request).await?;
    assert!(
        !support::tool_calls(&response).is_empty(),
        "required tool choice produced no tool call"
    );
    Ok(())
}

async fn issues_parallel_tool_calls(model: &str) -> TestResult {
    let Some(client) = venice::live_client() else {
        return support::skip("VENICE_API_KEY is unset");
    };
    // Whether the model batches both lookups into one turn is model behavior,
    // so one miss gets one retry before it counts as a failure.
    for attempt in 0..2 {
        let request = venice::request(model)
            .system("Answer weather questions with the get_weather tool, one call per city.")
            .user("Compare the current weather in Paris and in Madrid.")
            .tool(weather_tool())
            .build()?;
        let response = client.complete(request).await?;
        let calls = support::tool_calls(&response);
        if calls.len() >= 2 {
            let mut ids: Vec<&str> = calls.iter().map(|call| call.id.as_str()).collect();
            ids.sort_unstable();
            ids.dedup();
            assert_eq!(ids.len(), calls.len(), "parallel tool calls share an id");
            return Ok(());
        }
        support::observe(&format!(
            "{model} issued {} tool call(s) for a two-city prompt (attempt {attempt})",
            calls.len()
        ));
    }
    Err(format!("{model} never issued parallel tool calls").into())
}

async fn acknowledges_an_error_tool_result(model: &str) -> TestResult {
    let Some(client) = venice::live_client() else {
        return support::skip("VENICE_API_KEY is unset");
    };
    let request = venice::request(model)
        .user("What is the weather in Atlantis?")
        .tool(weather_tool())
        .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
            ToolCall::function(
                "call_atlantis",
                "get_weather",
                json!({ "city": "Atlantis" }),
            ),
        )]))
        .message(Message::new(Role::Tool, [ContentPart::ToolResult(
            ToolResult {
                tool_call_id: "call_atlantis".to_owned(),
                name:         Some("get_weather".to_owned()),
                content:      vec![ContentPart::Text {
                    text: "the city is unknown".to_owned(),
                }],
                is_error:     true,
            },
        )]))
        .build()?;
    let response = client.complete(request).await?;
    // Retrying the lookup is as legitimate a reaction to a failed tool as
    // answering in text — glm-5.3 does exactly that — so either counts as
    // acknowledging the error.
    assert!(
        !response.text().trim().is_empty() || !support::tool_calls(&response).is_empty(),
        "the model neither answered nor retried after an error tool result"
    );
    Ok(())
}
