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

use crate::openrouter::{self, family_tests, model_tests};
use crate::support::{self, TestResult};

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

/// The skip for the cells that force a call. The Fable 5.1 row's listing
/// omits `tool_choice`, and the upstream answers a forced choice with a 400,
/// so its row says so and these cells skip it.
fn forced_choice_unclaimed(model: &str) -> Option<TestResult> {
    let capabilities = openrouter::capabilities(model);
    if !capabilities.tools().is_supported() {
        return Some(support::skip("the catalog does not claim tools"));
    }
    (!capabilities
        .tool_choice(&lithos_llm::types::ToolChoice::Required)
        .is_supported())
    .then(|| support::skip("the catalog does not claim forced tool choice"))
}

async fn calls_the_forced_tool(model: &str) -> TestResult {
    if let Some(skip) = forced_choice_unclaimed(model) {
        return skip;
    }
    // The GLM 4.6 route accepted `required` and ordinary tool use, but a
    // named choice returned no call and its streamed twin idled out through
    // OpenRouter on 2026-08-29.
    if model == "glm-4.6" {
        return support::skip("the OpenRouter route does not honor named tool choice");
    }
    let Some(client) = openrouter::live_client() else {
        return support::skip("OPENROUTER_API_KEY is unset");
    };
    // OpenRouter can select another upstream between attempts. One miss gets
    // one retry before the route loses its tools claim.
    for attempt in 0..2 {
        let request = openrouter::request(model)
            .user("What is the weather in Paris?")
            .tool(weather_tool())
            .tool_choice(ToolChoice::Tool {
                name: "get_weather".to_owned(),
            })
            .build()?;
        let response = client.complete(request).await?;
        let calls = support::tool_calls(&response);
        if let Some(call) = calls.first() {
            assert_eq!(call.name, "get_weather");
            assert!(!call.id.is_empty(), "the tool call carries no id");
            assert_weather_call(&call.input.to_value().expect("valid tool input"));
            assert_eq!(response.finish_reason, FinishReason::ToolCall);
            return Ok(());
        }
        support::observe(&format!(
            "{model} ignored named tool choice (attempt {attempt})"
        ));
    }
    Err(format!("{model} never honored named tool choice").into())
}

async fn streams_the_forced_tool_call(model: &str) -> TestResult {
    if let Some(skip) = forced_choice_unclaimed(model) {
        return skip;
    }
    if model == "glm-4.6" {
        return support::skip("the OpenRouter route does not honor named tool choice");
    }
    let Some(client) = openrouter::live_client() else {
        return support::skip("OPENROUTER_API_KEY is unset");
    };
    let request = openrouter::request(model)
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
    assert_weather_call(&call.input.to_value().expect("valid tool input"));
    Ok(())
}

async fn uses_the_tool_result(model: &str) -> TestResult {
    if !openrouter::capabilities(model).tools().is_supported() {
        return support::skip("the catalog does not claim tools");
    }
    let Some(client) = openrouter::live_client() else {
        return support::skip("OPENROUTER_API_KEY is unset");
    };
    let request = openrouter::request(model)
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
    if !openrouter::capabilities(model).tools().is_supported() {
        return support::skip("the catalog does not claim tools");
    }
    let Some(client) = openrouter::live_client() else {
        return support::skip("OPENROUTER_API_KEY is unset");
    };
    // Whether the model chooses the tool is model behavior, so one miss gets
    // one retry before it counts as a failure.
    for attempt in 0..2 {
        let request = openrouter::request(model)
            .system("Use the get_weather tool for any weather question.")
            .user("What is the weather in Paris right now?")
            .tool(weather_tool())
            .tool_choice(ToolChoice::Auto)
            .build()?;
        let response = client.complete(request).await?;
        if let Some(call) = support::tool_calls(&response).first() {
            assert_eq!(call.name, "get_weather");
            assert_weather_call(&call.input.to_value().expect("valid tool input"));
            return Ok(());
        }
        support::observe(&format!(
            "{model} answered without a tool call under auto choice (attempt {attempt})"
        ));
    }
    Err(format!("{model} never called the tool under auto choice").into())
}

async fn calls_a_tool_under_required_choice(model: &str) -> TestResult {
    if let Some(skip) = forced_choice_unclaimed(model) {
        return skip;
    }
    // GLM 4.6 returned no tool call after the request idled for two minutes
    // through OpenRouter on 2026-08-29.
    if model == "glm-4.6" {
        return support::skip("the OpenRouter route does not honor required tool choice");
    }
    let Some(client) = openrouter::live_client() else {
        return support::skip("OPENROUTER_API_KEY is unset");
    };
    let request = openrouter::request(model)
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
    if !openrouter::capabilities(model).tools().is_supported() {
        return support::skip("the catalog does not claim tools");
    }
    // Laguna XS issued one call on both attempts on 2026-08-29. Its listing
    // also omits `parallel_tool_calls`, so do not require model-level batching.
    if model == "laguna-xs-2.1" {
        return support::skip("the OpenRouter route does not issue parallel tool calls");
    }
    let Some(client) = openrouter::live_client() else {
        return support::skip("OPENROUTER_API_KEY is unset");
    };
    // Whether the model batches both lookups into one turn is model behavior,
    // so one miss gets one retry before it counts as a failure.
    for attempt in 0..2 {
        let request = openrouter::request(model)
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
    if !openrouter::capabilities(model).tools().is_supported() {
        return support::skip("the catalog does not claim tools");
    }
    let Some(client) = openrouter::live_client() else {
        return support::skip("OPENROUTER_API_KEY is unset");
    };
    let request = openrouter::request(model)
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
