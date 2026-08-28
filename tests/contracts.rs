use std::error::Error as StdError;

use lithos_llm::Request;
use lithos_llm::types::{RequestBuildError, ToolChoice, ToolDefinition};
use serde_json::json;

#[test]
fn request_json_round_trips() -> Result<(), Box<dyn StdError>> {
    let request = Request::builder()
        .model("provider/model")
        .system("Keep it short.")
        .user("Hello")
        .max_output_tokens(100)
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
        .tool(ToolDefinition {
            name:         "weather".to_owned(),
            description:  "Weather".to_owned(),
            input_schema: json!({ "type": "object" }),
        })
        .tool_choice(ToolChoice::Tool {
            name: "missing".to_owned(),
        })
        .build();

    assert!(matches!(result, Err(RequestBuildError::UnknownToolChoice)));
}
