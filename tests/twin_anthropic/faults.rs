use std::fmt::Write as _;
use std::time::Duration;

use futures_util::StreamExt as _;
use lithos_llm::middleware::RetryPolicy;
use lithos_llm::types::{ContentPart, ErrorKind, FinishReason, StreamEvent};
use serde_json::{Value, json};
use tokio::time::{Instant, timeout};

use super::contracts::{completed, expected, native_content, native_usage, semantic, success};
use super::support::{Twin, bounded, collect, config, request, scenario};

/// Independently authored provider events, never reconstructed by twin code.
pub(super) fn frames() -> Vec<Value> {
    let mut result = vec![
        json!({"type":"message_start","message":{"id":"msg_independent","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":11,"output_tokens":0,"cache_creation_input_tokens":7,"cache_read_input_tokens":13}}}),
    ];
    for (index, block) in native_content()
        .as_array()
        .expect("native blocks")
        .iter()
        .enumerate()
    {
        let mut initial = block.clone();
        let deltas = match block["type"].as_str().expect("type") {
            "text" => {
                initial["text"] = json!("");
                vec![json!({"type":"text_delta","text":block["text"]})]
            }
            "thinking" => {
                initial["thinking"] = json!("");
                initial.as_object_mut().expect("object").remove("signature");
                vec![
                    json!({"type":"thinking_delta","thinking":block["thinking"]}),
                    json!({"type":"signature_delta","signature":block["signature"]}),
                ]
            }
            "tool_use" => {
                initial["input"] = json!({});
                block["input"]
                    .to_string()
                    .chars()
                    .map(|c| json!({"type":"input_json_delta","partial_json":c.to_string()}))
                    .collect()
            }
            "redacted_thinking" => vec![],
            _ => unreachable!("fixture block"),
        };
        result.push(json!({"type":"content_block_start","index":index,"content_block":initial}));
        for delta in deltas {
            result.push(json!({"type":"content_block_delta","index":index,"delta":delta}));
        }
        result.push(json!({"type":"content_block_stop","index":index}));
        result.push(json!({"type":"ping"}));
    }
    result.push(json!({"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":native_usage()}));
    result.push(json!({"type":"message_stop"}));
    result
}

pub(super) fn render(frames: &[Value]) -> Vec<u8> {
    let mut text = String::new();
    for frame in frames {
        write!(
            text,
            "event: {}\ndata: {frame}\n\n",
            frame["type"].as_str().expect("event type")
        )
        .expect("write string");
    }
    text.into_bytes()
}

pub(super) fn raw(chunks: Vec<Value>) -> Value {
    {
        let mut value = json!({"kind":"raw","status":200,"content_type":"text/event-stream"});
        value["chunks"] = Value::Array(chunks);
        value
    }
}

fn bytes_chunk(bytes: &[u8]) -> Value {
    json!({"kind":"bytes","bytes":bytes})
}

#[tokio::test]
async fn streaming_is_independent_of_utf8_and_sse_chunk_boundaries() {
    let twin = Twin::start(config()).await;
    let body = render(&frames());
    for width in [1, 2, 3, 7, 31, body.len()] {
        let chunks = body.chunks(width).map(bytes_chunk).collect();
        twin.enqueue("fragments", json!([scenario(raw(chunks))]))
            .await;
        let items = collect(
            twin.client("fragments")
                .stream(request())
                .await
                .expect("open stream"),
        )
        .await;
        let events: Vec<_> = items
            .into_iter()
            .collect::<Result<_, _>>()
            .expect("valid fragmented stream");
        assert_eq!(
            semantic(completed(&events)),
            expected(),
            "chunk width {width}"
        );
    }
    // Reproducible irregular chunk schedules, including CR/LF split points.
    let crlf = String::from_utf8(body)
        .expect("SSE UTF-8")
        .replace('\n', "\r\n")
        .into_bytes();
    for seed in 1_u64..=16 {
        let mut random = seed;
        let mut offset = 0;
        let mut chunks = Vec::new();
        while offset < crlf.len() {
            random = random
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let end =
                (offset + usize::try_from(random % 23 + 1).expect("small chunk")).min(crlf.len());
            chunks.push(bytes_chunk(&crlf[offset..end]));
            offset = end;
        }
        twin.enqueue("fragments", json!([scenario(raw(chunks))]))
            .await;
        let events: Vec<_> = collect(
            twin.client("fragments")
                .stream(request())
                .await
                .expect("stream"),
        )
        .await
        .into_iter()
        .collect::<Result<_, _>>()
        .expect("seeded stream");
        assert_eq!(semantic(completed(&events)), expected(), "seed {seed}");
    }
    twin.shutdown().await;
}

#[tokio::test]
async fn changed_wire_block_index_cannot_pass_the_exact_contract() {
    let twin = Twin::start(config()).await;
    let mut corrupted = frames();
    let delta = corrupted
        .iter_mut()
        .find(|event| event["delta"]["type"] == "thinking_delta")
        .expect("thinking delta");
    delta["index"] = json!(99);
    twin.enqueue(
        "index",
        json!([scenario(raw(vec![bytes_chunk(&render(&corrupted))]))]),
    )
    .await;
    let events = collect(
        twin.client("index")
            .stream(request())
            .await
            .expect("stream"),
    )
    .await;
    let response = events
        .iter()
        .filter_map(|event| event.as_ref().ok())
        .find_map(|event| {
            if let StreamEvent::Ended { response } = event {
                Some(response)
            } else {
                None
            }
        });
    assert!(
        response.is_none_or(|r| semantic(r) != expected()),
        "corrupted index escaped detection"
    );
    twin.shutdown().await;
}

fn retry_policy() -> RetryPolicy {
    RetryPolicy::exponential()
        .max_attempts(3)
        .initial_delay(Duration::from_millis(1))
        .max_delay(Duration::from_millis(2))
}

fn http_error(status: u16, retry_after: Option<&str>) -> Value {
    let mut result = json!({"kind":"error","status":status,"error_type":if status == 429 {"rate_limit_error"} else {"overloaded_error"},"message":"temporary overload"});
    if let Some(delay) = retry_after {
        result["retry_after"] = json!(delay);
    }
    result
}

#[tokio::test]
async fn http_retries_honor_retry_after_and_stop_at_the_attempt_budget() {
    let twin = Twin::start(config()).await;
    for streaming in [false, true] {
        let key = if streaming {
            "stream-retry"
        } else {
            "complete-retry"
        };
        twin.enqueue(
            key,
            json!([
                scenario(http_error(429, Some("1"))),
                scenario(http_error(529, None)),
                scenario(success())
            ]),
        )
        .await;
        let client = twin.client_with_policy(key, Some(retry_policy()), None);
        let start = Instant::now();
        let response = if streaming {
            let events: Vec<_> = collect(
                bounded(client.stream(request()))
                    .await
                    .expect("retry stream"),
            )
            .await
            .into_iter()
            .collect::<Result<_, _>>()
            .expect("events");
            completed(&events).clone()
        } else {
            bounded(client.complete(request()))
                .await
                .expect("retry complete")
        };
        assert_eq!(semantic(&response), expected());
        assert!(
            start.elapsed() >= Duration::from_secs(1),
            "Retry-After was ignored"
        );
        assert_eq!(
            twin.logs(key).await["requests"]
                .as_array()
                .expect("attempts")
                .len(),
            3
        );
    }
    let mut sticky = scenario(http_error(529, None));
    sticky["sticky"] = json!(true);
    twin.enqueue("budget", json!([sticky])).await;
    assert!(
        bounded(
            twin.client_with_policy("budget", Some(retry_policy()), None)
                .complete(request())
        )
        .await
        .is_err()
    );
    assert_eq!(
        twin.logs("budget").await["requests"]
            .as_array()
            .expect("attempts")
            .len(),
        3
    );
    twin.enqueue(
        "cap",
        json!([scenario(http_error(429, Some("3600"))), scenario(success())]),
    )
    .await;
    assert!(
        bounded(
            twin.client_with_policy("cap", Some(retry_policy()), None)
                .complete(request())
        )
        .await
        .is_err()
    );
    assert_eq!(
        twin.logs("cap").await["requests"]
            .as_array()
            .expect("attempts")
            .len(),
        1
    );
    twin.shutdown().await;
}

#[tokio::test]
async fn stream_errors_retry_only_before_visible_content() {
    let twin = Twin::start(config()).await;
    let error =
        json!({"type":"error","error":{"type":"overloaded_error","message":"temporary overload"}});
    for visible in [false, true] {
        let mut prefix = frames();
        let end = if visible {
            prefix
                .iter()
                .position(|e| e["type"] == "content_block_stop" && e["index"] == 3)
                .expect("first complete tool")
                + 1
        } else {
            1
        };
        prefix.truncate(end);
        prefix.push(error.clone());
        let key = if visible { "visible" } else { "invisible" };
        twin.enqueue(
            key,
            json!([
                scenario(raw(vec![bytes_chunk(&render(&prefix))])),
                scenario(success())
            ]),
        )
        .await;
        let client = twin.client_with_policy(key, Some(retry_policy()), None);
        let events = collect(client.stream(request()).await.expect("stream")).await;
        if visible {
            assert!(events.iter().any(Result::is_err));
            assert!(
                !events
                    .iter()
                    .any(|e| matches!(e, Ok(StreamEvent::Ended { .. })))
            );
            assert_eq!(
                events
                    .iter()
                    .filter(|e| matches!(
                        e,
                        Ok(StreamEvent::ContentBlockEnd {
                            part: ContentPart::ToolCall(_),
                            ..
                        })
                    ))
                    .count(),
                1,
                "tool delivered twice"
            );
        } else {
            let events: Vec<_> = events
                .into_iter()
                .collect::<Result<_, _>>()
                .expect("retried items");
            assert_eq!(semantic(completed(&events)), expected());
        }
        assert_eq!(
            twin.logs(key).await["requests"]
                .as_array()
                .expect("attempts")
                .len(),
            if visible { 1 } else { 2 }
        );
    }
    twin.shutdown().await;
}

#[tokio::test]
async fn malformed_truncated_and_disconnected_streams_never_claim_full_success() {
    let twin = Twin::start(config()).await;
    for script in [
        json!({"kind":"success","response_text":"partial","malformed_sse":true}),
        json!({"kind":"success","response_text":"partial","close_after_chunks":3}),
        raw(vec![
            bytes_chunk(&render(&frames()[..3])),
            json!({"kind":"error","message":"connection reset","delay_ms":10}),
        ]),
    ] {
        twin.enqueue("failure", json!([scenario(script)])).await;
        let mut stream = twin
            .client("failure")
            .stream(request())
            .await
            .expect("stream headers");
        let mut terminals = 0;
        while let Some(event) = bounded(stream.next()).await {
            match event {
                Err(_) => terminals += 1,
                Ok(StreamEvent::Ended { response }) => {
                    terminals += 1;
                    assert_eq!(response.finish_reason, FinishReason::Incomplete);
                }
                _ => {}
            }
        }
        assert_eq!(terminals, 1);
        assert!(
            stream.next().await.is_none(),
            "terminal stream must stay fused"
        );
    }
    twin.shutdown().await;
}

#[tokio::test]
async fn header_and_stream_stalls_obey_deadlines_and_cancellation() {
    let twin = Twin::start(config()).await;
    twin.enqueue("headers", json!([scenario(json!({"kind":"hang"}))]))
        .await;
    let req = request()
        .into_builder()
        .timeout(Duration::from_millis(100))
        .build()
        .expect("deadline");
    let error = bounded(twin.client("headers").complete(req))
        .await
        .expect_err("header timeout");
    assert_eq!(error.kind(), ErrorKind::Timeout);
    let prefix = render(&frames()[..3]);
    twin.enqueue(
        "stall",
        json!([scenario(raw(vec![
            bytes_chunk(&prefix),
            json!({"kind":"text","text":"never","delay_ms":1000})
        ]))]),
    )
    .await;
    let client = twin.client_with_policy(
        "stall",
        Some(retry_policy()),
        Some(Duration::from_millis(100)),
    );
    let events = collect(client.stream(request()).await.expect("stream headers")).await;
    assert!(
        events
            .iter()
            .any(|e| matches!(e,Err(error) if error.kind() == ErrorKind::Timeout))
    );
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, Ok(StreamEvent::Ended { .. })))
    );
    assert_eq!(
        twin.logs("stall").await["requests"]
            .as_array()
            .expect("attempts")
            .len(),
        1
    );
    twin.enqueue(
        "cancel",
        json!([
            scenario(json!({"kind":"hang"})),
            scenario(json!({"kind":"success","response_text":"after cancellation"}))
        ]),
    )
    .await;
    let client = twin.client("cancel");
    assert!(
        timeout(Duration::from_millis(100), client.complete(request()))
            .await
            .is_err()
    );
    let response = bounded(client.complete(request()))
        .await
        .expect("request after cancelled future");
    assert_eq!(response.finish_reason, FinishReason::Stop);
    twin.shutdown().await;
}
