//! Modal overlays bind the portable selector to a deployment-specific route.
use std::error::Error as StdError;

use httpmock::MockServer;
use lithos_llm::catalog::Catalog;
use lithos_llm::credentials::{CredentialHeader, Credentials, SecretValue, StaticCredentials};
use lithos_llm::types::{ReasoningEffort, StreamEvent};
use lithos_llm::{Client, Request};
use serde_json::json;

use crate::support;

#[tokio::test]
async fn modal_templates_route_completion_and_streaming_without_losing_metadata()
-> Result<(), Box<dyn StdError>> {
    for (template, api_model) in [
        (
            include_str!("../../../docs/catalogs/modal-dedicated.toml"),
            "test/deployed-kimi",
        ),
        (
            include_str!("../../../docs/catalogs/modal-shared.toml"),
            "test-kimi.modal.run",
        ),
    ] {
        for streaming in [false, true] {
            let server = MockServer::start_async().await;
            let overlay = template
                .replace(
                    "https://REPLACE_WITH_DEDICATED_ENDPOINT.invalid/v1",
                    &format!("{}/v1", server.base_url()),
                )
                .replace(
                    "https://inference.us-west.modal.direct/v1",
                    &format!("{}/v1", server.base_url()),
                )
                .replace("REPLACE_WITH_ACCEPTED_MODEL_ID", api_model)
                .replace("REPLACE_WITH_SHARED_ENDPOINT_HOSTNAME", api_model);
            let catalog = Catalog::builder()
                .with_builtin()
                .overlay_toml(&overlay)?
                .build()?;
            let model = catalog.model("modal", "kimi-k3")?;
            assert!(!model.is_passthrough());
            assert!(model.limits().is_some());
            assert!(model.pricing().is_some());
            assert_eq!(catalog.provider("modal")?.default_model(), Some("kimi-k3"));
            let build = Client::builder()
                .catalog(catalog)
                .enabled_providers(["modal"])
                .credentials(StaticCredentials::new().with(
                    "modal",
                    Credentials::headers([
                        CredentialHeader::new("Modal-Key", SecretValue::new("test-key")),
                        CredentialHeader::new("Modal-Secret", SecretValue::new("test-secret")),
                    ]),
                ))
                .build()?;
            assert!(build.issues.is_empty(), "{:?}", build.issues);
            let body = json!({"id":"modal_1","choices":[{"message":{"role":"assistant","content":"PONG"},"finish_reason":"stop"}]});
            let sse = format!(
                "data: {}\n\ndata: [DONE]\n\n",
                json!({"id":"modal_1","choices":[{"index":0,"delta":{"content":"PONG"},"finish_reason":"stop"}]})
            );
            let (mock, slot) = if streaming {
                support::mount_capture_sse(&server, "/v1/chat/completions", &sse)
            } else {
                support::mount_capture(&server, "/v1/chat/completions", &body)
            };
            let request = Request::builder()
                .model("modal/kimi-k3")
                .user("Ping")
                .reasoning_effort(ReasoningEffort::High)
                .build()?;
            if streaming {
                let events =
                    support::collect_stream_events(build.client.stream(request).await?).await;
                support::assert_stream_contract(&events);
                let event: StreamEvent =
                    serde_json::from_value(events.last().ok_or("missing terminal")?.clone())?;
                let StreamEvent::Ended { response } = event else {
                    return Err("expected Ended".into());
                };
                assert_eq!(response.text(), "PONG");
            } else {
                assert_eq!(build.client.complete(request).await?.text(), "PONG");
            }
            mock.assert_calls(1);
            let capture = support::captured(&slot);
            assert_eq!(capture.body["model"], api_model);
            assert_eq!(capture.body["reasoning_effort"], "high");
            for name in ["modal-key", "modal-secret"] {
                assert!(capture.headers.iter().any(|(key, _)| key == name));
            }
            assert!(capture.body.get("metadata").is_none());
        }
    }
    Ok(())
}
