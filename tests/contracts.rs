use std::error::Error as StdError;

use lithos_llm::Request;
use lithos_llm::types::{RequestBuildError, ToolChoice, ToolDefinition};
use serde_json::json;

#[test]
fn deserialization_enforces_request_invariants() {
    for document in [
        json!({"model": "test/model", "messages": []}),
        json!({"model": "test/model", "messages": [{"role": "user", "content": []}]}),
        json!({"model": "test/model", "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}], "max_output_tokens": 0}),
    ] {
        assert!(serde_json::from_value::<Request>(document).is_err());
    }
}

#[test]
fn rebuilding_preserves_every_request_setting() -> Result<(), Box<dyn StdError>> {
    let original = Request::builder()
        .model("test/model")
        .user("hello")
        .tool(ToolDefinition::function("weather", "Weather", json!({})))
        .temperature(0.5)
        .top_p(0.8)
        .timeout(std::time::Duration::from_secs(3))
        .stop_sequence("END")
        .metadata_entry("tenant", "a")
        .provider_option("test", "seed", json!(42))
        .cache_key("key")
        .build()?;
    assert_eq!(original.clone().into_builder().build()?, original);
    let changed = original
        .clone()
        .into_builder()
        .max_output_tokens(42)
        .build()?;
    let mut expected = serde_json::to_value(original)?;
    expected["max_output_tokens"] = json!(42);
    assert_eq!(serde_json::to_value(changed)?, expected);
    Ok(())
}

#[test]
fn request_json_round_trips() -> Result<(), Box<dyn StdError>> {
    let request = Request::builder()
        .model("provider/model")
        .system("Keep it short.")
        .user("Hello")
        .max_output_tokens(100)
        .stop_sequence("STOP")
        .metadata_entry("tenant", "acme")
        .cache_key("tenant-acme")
        .provider_option("openai", "service_tier", json!("flex"))
        .build()?;

    let json = serde_json::to_string(&request)?;
    let decoded = serde_json::from_str::<Request>(&json)?;

    assert_eq!(decoded, request);
    Ok(())
}

#[test]
fn request_builder_rejects_an_unknown_selected_tool() {
    let result = Request::builder()
        .model("provider/model")
        .user("Hello")
        .tool(ToolDefinition::function(
            "weather",
            "Weather",
            json!({ "type": "object" }),
        ))
        .tool_choice(ToolChoice::Tool {
            name: "missing".to_owned(),
        })
        .build();

    assert!(matches!(result, Err(RequestBuildError::UnknownToolChoice)));
}

#[test]
fn request_builder_rejects_duplicate_names_across_tool_kinds() {
    let result = Request::builder()
        .model("provider/model")
        .user("Hello")
        .tool(ToolDefinition::function(
            "patch",
            "Patch",
            json!({ "type": "object" }),
        ))
        .tool(ToolDefinition::custom(
            "patch",
            "Patch",
            json!({ "type": "grammar" }),
        ))
        .build();

    assert!(matches!(result, Err(RequestBuildError::DuplicateToolName)));
}

#[test]
fn named_tool_choice_accepts_a_custom_tool() -> Result<(), Box<dyn StdError>> {
    let request = Request::builder()
        .model("provider/model")
        .user("Hello")
        .tool(ToolDefinition::custom(
            "apply_patch",
            "Apply a patch",
            json!({ "type": "grammar", "syntax": "lark" }),
        ))
        .tool_choice(ToolChoice::Tool {
            name: "apply_patch".to_owned(),
        })
        .build()?;

    assert!(request.tools()[0].is_custom());
    Ok(())
}
