//! V2 — tool calling.
//!
//! K3 supports automatic and required tool selection, parallel calls, and tool
//! result turns. Moonshot rejects a named tool choice while K3 thinking is
//! enabled; the two named-choice cells preserve that dated provider limit.

use lithos_llm::types::{
    ContentPart, FinishReason, Message, Role, ToolCall, ToolChoice, ToolDefinition, ToolResult,
};
use serde_json::{Value, json};

use crate::moonshot::{self, family_tests, model_tests};
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

async fn calls_the_forced_tool(model: &str) -> TestResult {
    // Raw blocking and streaming probes returned this error on 2026-08-29:
    // "tool_choice 'specified' is incompatible with thinking enabled".
    if model == "kimi-k3" {
        return support::skip("Moonshot rejects named tool choice while K3 thinking is enabled");
    }
    let Some(client) = moonshot::live_client() else {
        return support::skip("MOONSHOT_API_KEY is unset");
    };
    let request = moonshot::request(model)
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
    assert_weather_call(&call.input.to_value().expect("valid tool input"));
    assert_eq!(response.finish_reason, FinishReason::ToolCall);
    Ok(())
}

async fn streams_the_forced_tool_call(model: &str) -> TestResult {
    // The blocking probe above establishes the request validation shared by
    // both response modes.
    if model == "kimi-k3" {
        return support::skip("Moonshot rejects named tool choice while K3 thinking is enabled");
    }
    let Some(client) = moonshot::live_client() else {
        return support::skip("MOONSHOT_API_KEY is unset");
    };
    let request = moonshot::request(model)
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
    let Some(client) = moonshot::live_client() else {
        return support::skip("MOONSHOT_API_KEY is unset");
    };
    let request = moonshot::request(model)
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
    let Some(client) = moonshot::live_client() else {
        return support::skip("MOONSHOT_API_KEY is unset");
    };
    // Whether the model chooses the tool is model behavior, so one miss gets
    // one retry before it counts as a failure.
    for attempt in 0..2 {
        let request = moonshot::request(model)
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
    let Some(client) = moonshot::live_client() else {
        return support::skip("MOONSHOT_API_KEY is unset");
    };
    let request = moonshot::request(model)
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
    let Some(client) = moonshot::live_client() else {
        return support::skip("MOONSHOT_API_KEY is unset");
    };
    // Whether the model batches both lookups into one turn is model behavior,
    // so one miss gets one retry before it counts as a failure.
    for attempt in 0..2 {
        let request = moonshot::request(model)
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
    let Some(client) = moonshot::live_client() else {
        return support::skip("MOONSHOT_API_KEY is unset");
    };
    let request = moonshot::request(model)
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
    // answering in text, so either counts as acknowledging the error.
    assert!(
        !response.text().trim().is_empty() || !support::tool_calls(&response).is_empty(),
        "the model neither answered nor retried after an error tool result"
    );
    Ok(())
}
