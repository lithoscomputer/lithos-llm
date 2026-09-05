//! Provisional Fabro imports: verify our translation against local HTTP mocks.
//! Live acceptance remains tracked in docs/provider-live-tests.md.

use std::error::Error as StdError;

use httpmock::MockServer;
use lithos_llm::catalog::Catalog;
use lithos_llm::credentials::{Credentials, SecretValue, StaticCredentials};
use lithos_llm::types::{
    ContentPart, FinishReason, Message, ReasoningEffort, Role, StreamEvent, ToolResult,
};
use lithos_llm::{Client, Request};
use serde_json::json;

use crate::support;

/// Keep the catalog's path prefix while replacing only the upstream origin.
fn client_for(server: &MockServer, provider: &str) -> Result<Client, Box<dyn StdError>> {
    let catalog = Catalog::builder().with_builtin().build()?;
    let url = catalog.provider(provider)?.base_url();
    let prefix = url
        .split_once("://")
        .and_then(|(_, rest)| rest.split_once('/'))
        .map_or(String::new(), |(_, path)| format!("/{path}"));
    let catalog = Catalog::builder()
        .with_builtin()
        .overlay_toml(&format!(
            "schema_version = 1\n[providers.{provider}]\nbase_url = \"{}{prefix}\"",
            server.base_url()
        ))?
        .build()?;
    let credentials = if provider == "ollama" {
        Credentials::none()
    } else {
        Credentials::bearer(SecretValue::new(support::TEST_API_KEY))
    };
    let build = Client::builder()
        .catalog(catalog)
        .enabled_providers([provider])
        .credentials(StaticCredentials::new().with(provider, credentials))
        .build()?;
    assert!(build.issues.is_empty(), "{:?}", build.issues);
    Ok(build.client)
}

#[tokio::test]
async fn imported_chat_providers_use_their_api_paths_models_and_auth()
-> Result<(), Box<dyn StdError>> {
    for (provider, model, wire_model, path) in [
        (
            "openrouter",
            "kimi-k2.6",
            "moonshotai/kimi-k2.6",
            "/api/v1/chat/completions",
        ),
        (
            "openrouter",
            "nemotron-3-super-120b-a12b",
            "nvidia/nemotron-3-super-120b-a12b",
            "/api/v1/chat/completions",
        ),
        (
            "fireworks",
            "minimax-m2.7",
            "accounts/fireworks/models/minimax-m2p7",
            "/inference/v1/chat/completions",
        ),
        ("moonshot", "kimi-k2.5", "kimi-k2.5", "/v1/chat/completions"),
        (
            "deepseek",
            "deepseek",
            "deepseek-v4-flash",
            "/v1/chat/completions",
        ),
        ("inception", "mercury", "mercury-2", "/v1/chat/completions"),
        ("minimax", "minimax", "MiniMax-M2.5", "/v1/chat/completions"),
        (
            "zai",
            "glm",
            "glm-5.2",
            "/api/coding/paas/v4/chat/completions",
        ),
        (
            "poolside",
            "laguna",
            "poolside/laguna-s-2.1",
            "/v1/chat/completions",
        ),
        (
            "litellm",
            "team-model",
            "team-model",
            "/v1/chat/completions",
        ),
        (
            "ollama",
            "qwen3.5:latest",
            "qwen3.5:latest",
            "/v1/chat/completions",
        ),
    ] {
        for streaming in [false, true] {
            let server = MockServer::start_async().await;
            let client = client_for(&server, provider)?;
            let body = json!({"id":"chat_1","model":wire_model,"choices":[{
                "message":{"role":"assistant","content":"PONG"},"finish_reason":"stop"
            }],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}});
            let transcript = format!(
                "data: {}\n\ndata: [DONE]\n\n",
                json!({
                    "id":"chat_1","model":wire_model,"choices":[{
                        "index":0,"delta":{"role":"assistant","content":"PONG"},"finish_reason":"stop"
                    }],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}
                })
            );
            let (_mock, slot) = if streaming {
                support::mount_capture_sse(&server, path, &transcript)
            } else {
                support::mount_capture(&server, path, &body)
            };
            let request = Request::builder()
                .model(format!("{provider}/{model}"))
                .user("Ping")
                .build()?;
            let response = if streaming {
                let events = support::collect_stream_events(client.stream(request).await?).await;
                support::assert_stream_contract(&events);
                let event: StreamEvent =
                    serde_json::from_value(events.last().ok_or("missing terminal")?.clone())?;
                let StreamEvent::Ended { response } = event else {
                    return Err("expected Ended".into());
                };
                *response
            } else {
                client.complete(request).await?
            };
            assert_eq!(response.text(), "PONG");
            assert_eq!(response.finish_reason, FinishReason::Stop);
            assert_eq!(response.usage.total(), 12);
            let captured = support::captured(&slot);
            assert_eq!(captured.path, path);
            assert_eq!(captured.body["model"], wire_model);
            assert_eq!(
                captured
                    .headers
                    .iter()
                    .any(|(name, _)| name == "authorization"),
                provider != "ollama"
            );
            if provider == "minimax" {
                assert_eq!(captured.body["reasoning_split"], true);
            }
            assert!(captured.body.get("base_url_is_api_root").is_none());
        }
    }
    Ok(())
}

#[tokio::test]
async fn bedrock_openai_uses_responses_with_http_bearer_and_no_storage()
-> Result<(), Box<dyn StdError>> {
    for streaming in [false, true] {
        let server = MockServer::start_async().await;
        let client = client_for(&server, "bedrock-openai")?;
        let body = json!({"id":"resp_1","status":"completed","model":"openai.gpt-5.5",
            "output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed",
                "content":[{"type":"output_text","text":"PONG","annotations":[]}]}],
            "usage":{"input_tokens":10,"output_tokens":2,"total_tokens":12}});
        let transcript = format!(
            "event: response.completed\ndata: {}\n\n",
            json!({"type":"response.completed","response":body})
        );
        let (_mock, slot) = if streaming {
            support::mount_capture_sse(&server, "/v1/responses", &transcript)
        } else {
            support::mount_capture(&server, "/v1/responses", &body)
        };
        let request = Request::builder()
            .model("bedrock-openai/gpt-5.5")
            .user("Ping")
            .build()?;
        if streaming {
            let events = support::collect_stream_events(client.stream(request).await?).await;
            support::assert_stream_contract(&events);
            assert_eq!(
                events.last().ok_or("terminal")?["response"]["finish_reason"],
                "stop"
            );
        } else {
            assert_eq!(client.complete(request).await?.text(), "PONG");
        }
        let captured = support::captured(&slot);
        assert_eq!(captured.body["model"], "openai.gpt-5.5");
        assert_eq!(captured.body["store"], false);
        assert!(
            captured
                .headers
                .iter()
                .any(|(name, _)| name == "authorization")
        );
        assert!(
            !captured
                .headers
                .iter()
                .any(|(name, _)| name == "x-amz-date")
        );
    }
    Ok(())
}

#[tokio::test]
async fn imported_effort_and_sampling_restrictions_fail_before_dispatch()
-> Result<(), Box<dyn StdError>> {
    let server = MockServer::start_async().await;
    let inception = client_for(&server, "inception")?;
    let request = Request::builder()
        .model("inception/mercury")
        .user("Ping")
        .reasoning_effort(ReasoningEffort::Max)
        .build()?;
    assert_eq!(
        inception
            .complete(request)
            .await
            .expect_err("unsupported effort")
            .provider_code(),
        Some("unsupported_capability")
    );
    let deepseek = client_for(&server, "deepseek")?;
    let request = Request::builder()
        .model("deepseek/deepseek")
        .user("Ping")
        .temperature(0.5)
        .build()?;
    assert_eq!(
        deepseek
            .complete(request)
            .await
            .expect_err("sampling ignored by provider")
            .provider_code(),
        Some("unsupported_capability")
    );
    Ok(())
}

#[tokio::test]
async fn imported_reasoning_survives_a_tool_turn() -> Result<(), Box<dyn StdError>> {
    for (provider, model, field, reasoning) in [
        (
            "deepseek",
            "deepseek",
            "reasoning_content",
            json!("Need to look up the answer."),
        ),
        (
            "minimax",
            "minimax",
            "reasoning_details",
            json!([{"type":"reasoning.text","text":"Need to look up the answer."}]),
        ),
    ] {
        let server = MockServer::start_async().await;
        let client = client_for(&server, provider)?;
        let mut message = json!({"role":"assistant","content":"Checking.","tool_calls":[{
            "id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}
        }]});
        message[field] = reasoning.clone();
        let body = json!({"id":"chat_1","choices":[{"message":message,"finish_reason":"tool_calls"}],
            "usage":{"prompt_tokens":10,"completion_tokens":4,"prompt_cache_hit_tokens":3}});
        let (_mock, slot) = support::mount_capture(&server, "/v1/chat/completions", &body);
        let selector = format!("{provider}/{model}");
        let response = client
            .complete(
                Request::builder()
                    .model(&selector)
                    .user("Look it up")
                    .build()?,
            )
            .await?;
        assert_eq!(response.tool_calls().count(), 1);
        assert_eq!(response.usage.cache_read, 3);
        assert_eq!(response.usage.input, 7);
        let follow_up = Request::builder()
            .model(selector)
            .user("Look it up")
            .message(response.into_message())
            .message(Message::new(Role::Tool, [ContentPart::ToolResult(
                ToolResult {
                    tool_call_id: "call_1".to_owned(),
                    name:         Some("lookup".to_owned()),
                    content:      vec![ContentPart::Text {
                        text: "42".to_owned(),
                    }],
                    is_error:     false,
                },
            )]))
            .build()?;
        let _response = client.complete(follow_up).await?;
        let captured = support::captured(&slot);
        assert_eq!(captured.body["messages"][1][field], reasoning);
        assert_eq!(
            captured.body["messages"][1]["tool_calls"][0]["id"],
            "call_1"
        );
        assert_eq!(captured.body["messages"][2]["tool_call_id"], "call_1");
    }
    Ok(())
}

#[test]
fn compatible_adapter_refuses_malformed_path_options() -> Result<(), Box<dyn StdError>> {
    for options in [
        "{ base_url_is_api_root = \"yes\" }",
        "{ base_url_is_api_rooot = true }",
    ] {
        let catalog = Catalog::builder()
            .with_builtin()
            .overlay_toml(&format!(
                "schema_version = 1\n[providers.zai]\nadapter_options = {options}"
            ))?
            .build()?;
        let build = Client::builder()
            .catalog(catalog)
            .enabled_providers(["zai"])
            .build()?;
        assert_eq!(build.issues.len(), 1);
        assert_eq!(build.client.available_providers().iter().count(), 0);
    }
    Ok(())
}
