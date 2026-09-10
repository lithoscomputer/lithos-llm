//! Exercise each imported Bedrock family through the real built-in catalog.
use std::error::Error as StdError;

use httpmock::MockServer;
use lithos_llm::catalog::Catalog;
use lithos_llm::credentials::{Credentials, SecretValue, StaticCredentials};
use lithos_llm::types::{FinishReason, ReasoningEffort, StreamEvent};
use lithos_llm::{Client, Request};
use serde_json::json;

use crate::support;

fn client_for(server: &MockServer) -> Result<Client, Box<dyn StdError>> {
    let catalog = Catalog::builder().with_builtin().overlay_toml(&format!(
        "schema_version = 1\n[providers.bedrock]\nenabled = true\nbase_url = {:?}\nauth = {{ type = \"bedrock_bearer\" }}", server.base_url()
    ))?.build()?;
    let build = Client::builder()
        .catalog(catalog)
        .enabled_providers(["bedrock"])
        .credentials(StaticCredentials::new().with(
            "bedrock",
            Credentials::BedrockBearer(SecretValue::new(support::TEST_API_KEY)),
        ))
        .build()?;
    assert!(build.issues.is_empty(), "{:?}", build.issues);
    Ok(build.client)
}

#[tokio::test]
async fn imported_models_preserve_converse_paths_usage_and_family_controls()
-> Result<(), Box<dyn StdError>> {
    let catalog = Catalog::builder().with_builtin().build()?;
    for model in catalog.provider("bedrock")?.models() {
        if model.id().as_str() == "anthropic.claude-sonnet-4-6" {
            continue;
        }
        for streaming in [false, true] {
            let server = MockServer::start_async().await;
            let client = client_for(&server)?;
            let operation = if streaming {
                "converse-stream"
            } else {
                "converse"
            };
            let path = format!("/model/{}/{operation}", model.api_model());
            let usage = json!({"inputTokens":10,"outputTokens":2,"cacheReadInputTokens":3,"cacheWriteInputTokens":4});
            let body = json!({"output":{"message":{"role":"assistant","content":[{"text":"PONG"}]}},"stopReason":"end_turn","usage":usage});
            let frames = vec![
                support::bedrock_event_frame("messageStart", &json!({"role":"assistant"})),
                support::bedrock_event_frame(
                    "contentBlockDelta",
                    &json!({"contentBlockIndex":0,"delta":{"text":"PONG"}}),
                ),
                support::bedrock_event_frame("contentBlockStop", &json!({"contentBlockIndex":0})),
                support::bedrock_event_frame("messageStop", &json!({"stopReason":"end_turn"})),
                support::bedrock_event_frame("metadata", &json!({"usage":usage})),
            ];
            let (mock, slot) = if streaming {
                support::mount_capture_event_stream(&server, &path, &frames)
            } else {
                support::mount_capture(&server, &path, &body)
            };
            let request = Request::builder()
                .model(format!("bedrock/{}", model.id()))
                .system("Be concise")
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
            assert_eq!(response.usage.total(), 19);
            assert_eq!(response.usage.input, 10);
            assert_eq!(response.usage.output, 2);
            assert_eq!(response.usage.cache_read, 3);
            assert_eq!(response.usage.cache_write, 4);
            assert_eq!(response.usage.reasoning, 0);
            let capture = support::captured(&slot);
            assert_eq!(capture.path, path);
            assert!(
                capture.body.get("additionalModelRequestFields").is_none(),
                "{}: {}",
                model.id(),
                capture.body
            );
            assert_eq!(
                capture.body.to_string().contains("cachePoint"),
                model.capabilities().caching().is_supported()
            );
            let request = Request::builder()
                .model(format!("bedrock/{}", model.id()))
                .user("Ping")
                .reasoning_effort(ReasoningEffort::High)
                .build()?;
            let error = client
                .complete(request)
                .await
                .expect_err("unmapped effort must be rejected");
            assert_eq!(error.provider_code(), Some("unsupported_capability"));
            if !model.capabilities().sampling().is_supported() {
                let request = Request::builder()
                    .model(format!("bedrock/{}", model.id()))
                    .user("Ping")
                    .temperature(0.5)
                    .build()?;
                let error = client
                    .complete(request)
                    .await
                    .expect_err("sampling must be rejected");
                assert_eq!(error.provider_code(), Some("unsupported_capability"));
            }
            mock.assert_calls(1);
        }
    }
    Ok(())
}

#[test]
fn bedrock_preserves_endpoint_limits_and_unknown_prices() -> Result<(), Box<dyn StdError>> {
    let catalog = Catalog::builder().with_builtin().build()?;
    assert_eq!(
        catalog
            .model("bedrock", "nova-2-lite")?
            .limits()
            .map(|limits| limits.max_output_tokens),
        Some(65535)
    );
    for id in ["claude-fable-5", "claude-sonnet-5"] {
        assert!(
            !catalog
                .model("bedrock", id)?
                .capabilities()
                .sampling()
                .is_supported()
        );
    }
    for id in ["llama-4-maverick", "devstral-2", "nemotron-3-super"] {
        assert!(catalog.model("bedrock", id)?.pricing().is_none());
    }
    Ok(())
}
