use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use futures_util::{StreamExt as _, stream};
use lithos_llm::types::ContentPart;
use serde_json::{Value, json};
use twin_anthropic::config::{Config, Mode, RecordFormat};

use super::contracts::{completed, expected, semantic, success};
use super::support::{TempDir, Twin, bounded, collect, config, request, scenario};

async fn load(total: usize) {
    let twin = Twin::start(config()).await;
    let mut clients = Vec::new();
    for namespace in 0..32 {
        let key = format!("namespace-{namespace}");
        let mut script = scenario(json!({"kind":"success","response_text":key}));
        script["sticky"] = json!(true);
        twin.enqueue(&key, json!([script])).await;
        clients.push(twin.client(&key));
    }
    let results: Vec<_> = bounded(
        stream::iter(0..total)
            .map(|n| {
                let client = &clients[n % 32];
                async move {
                    let response = if n % 2 == 0 {
                        let events: Vec<_> =
                            collect(client.stream(request()).await.expect("load stream"))
                                .await
                                .into_iter()
                                .collect::<Result<_, _>>()
                                .expect("load stream items");
                        completed(&events).clone()
                    } else {
                        client.complete(request()).await.expect("load complete")
                    };
                    (n % 32, response)
                }
            })
            .buffer_unordered(32)
            .collect(),
    )
    .await;
    let mut ids = BTreeSet::new();
    for (namespace, response) in results {
        assert_eq!(response.content, vec![ContentPart::Text {
            text: format!("namespace-{namespace}"),
        }]);
        assert!(
            ids.insert((namespace, response.id.expect("response id"))),
            "duplicate response id inside namespace"
        );
    }
    assert_eq!(ids.len(), total);
    for (namespace, client) in clients.iter().enumerate() {
        let key = format!("namespace-{namespace}");
        let expected_count = total / 32 + usize::from(namespace < total % 32);
        assert_eq!(
            twin.logs(&key).await["requests"]
                .as_array()
                .expect("namespace logs")
                .len(),
            expected_count
        );
        twin.reset(&key).await;
        assert_eq!(twin.logs(&key).await["requests"], json!([]));
        let response = bounded(client.complete(request()))
            .await
            .expect("post-reset response");
        assert_eq!(response.id.as_deref(), Some("msg_000001"));
        assert_eq!(response.content, vec![ContentPart::Text {
            text: "deterministic: hello".to_owned(),
        }]);
    }
    twin.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_namespaces_keep_ids_results_and_reset_isolated() {
    load(1024).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "longer offline stress case; run with mise run test:anthropic:stress"]
async fn ten_thousand_requests_across_thirty_two_namespaces() {
    load(10_000).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_requests_consume_one_shot_repeat_and_sticky_in_order() {
    let twin = Twin::start(config()).await;
    let mut repeat = scenario(json!({"kind":"success","response_text":"repeat"}));
    repeat["repeat"] = json!(2);
    let mut sticky = scenario(json!({"kind":"success","response_text":"sticky"}));
    sticky["sticky"] = json!(true);
    twin.enqueue(
        "queue",
        json!([
            scenario(json!({"kind":"success","response_text":"once"})),
            repeat,
            sticky
        ]),
    )
    .await;
    let client = twin.client("queue");
    let responses: Vec<_> = bounded(
        stream::iter(0..64)
            .map(|_| client.complete(request()))
            .buffer_unordered(16)
            .collect(),
    )
    .await;
    let mut counts = BTreeMap::new();
    for response in responses {
        let response = response.expect("queue response");
        let ContentPart::Text { text } = &response.content[0] else {
            panic!("text")
        };
        *counts.entry(text.clone()).or_insert(0) += 1;
        // The server allocates ids after scenario selection, so do not
        // infer admission order from completion order or ids.
    }
    assert_eq!(
        counts,
        BTreeMap::from([
            ("once".to_owned(), 1),
            ("repeat".to_owned(), 2),
            ("sticky".to_owned(), 61)
        ])
    );
    twin.shutdown().await;
}

#[tokio::test]
async fn reset_restores_startup_templates_and_namespace_scope() {
    let directory = TempDir::new();
    let path = directory.0.join("scenarios.json");
    fs::write(&path,json!({"scenarios":[{"namespace":"template","scenario_id":"first","matcher":{"endpoint":"messages"},"script":{"kind":"success","response_text":"restored"}}]}).to_string()).expect("fixture");
    let twin = Twin::start(Config {
        scenarios_path: Some(path),
        ..config()
    })
    .await;
    let client = twin.client("template");
    let first = bounded(client.complete(request())).await.expect("template");
    assert!(
        client.complete(request()).await.is_err(),
        "spent strict fixture"
    );
    assert!(
        twin.client("other").complete(request()).await.is_err(),
        "namespace must not share fixture"
    );
    twin.reset("template").await;
    let restored = client.complete(request()).await.expect("restored template");
    assert_eq!(first, restored);
    twin.shutdown().await;
}

fn proxy_config(upstream: &Twin, path: &Path, append: bool) -> Config {
    Config {
        mode: Mode::ProxyRecord,
        upstream_url: upstream.url.clone(),
        upstream_api_key: Some("upstream".to_owned()),
        recording_path: Some(path.to_owned()),
        recording_append: append,
        record_format: RecordFormat::Transcript,
        ..config()
    }
}

fn recording(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).expect("recording bytes")).expect("valid recording JSON")
}

#[tokio::test]
async fn proxy_recordings_survive_append_restart_and_exact_client_replay() {
    let directory = TempDir::new();
    let path = directory.0.join("recording.json");
    let upstream = Twin::start(config()).await;
    let mut script = scenario(success());
    script["sticky"] = json!(true);
    upstream.enqueue("upstream", json!([script])).await;
    let mut originals = Vec::new();
    for append in [false, true] {
        let proxy = Twin::start(proxy_config(&upstream, &path, append)).await;
        let response = bounded(proxy.client("recording").complete(request()))
            .await
            .expect("recorded response");
        assert_eq!(semantic(&response), expected());
        originals.push(response);
        proxy.shutdown().await;
    }
    assert_eq!(
        recording(&path)["scenarios"]
            .as_array()
            .expect("scenarios")
            .len(),
        2
    );
    let replay = Twin::start(Config {
        scenarios_path: Some(path),
        ..config()
    })
    .await;
    let client = replay.client("recording");
    for original in originals {
        assert_eq!(
            client.complete(request()).await.expect("exact replay"),
            original
        );
    }
    assert!(
        client.complete(request()).await.is_err(),
        "all recordings consumed"
    );
    replay.shutdown().await;
    upstream.shutdown().await;
}

#[tokio::test]
async fn failed_recording_write_preserves_the_last_file_and_recovers() {
    let directory = TempDir::new();
    let path = directory.0.join("recording.json");
    let upstream = Twin::start(config()).await;
    let proxy = Twin::start(proxy_config(&upstream, &path, false)).await;
    let client = proxy.client("recording");
    client.complete(request()).await.expect("first recording");
    let previous = fs::read(&path).expect("last good file");
    // A directory at the temporary output path makes writes fail on every
    // OS, including privileged CI users; no permission or global changes.
    let obstruction = path.with_extension("tmp");
    fs::create_dir(&obstruction).expect("write obstruction");
    client
        .complete(request())
        .await
        .expect("upstream success despite recording failure");
    assert_eq!(fs::read(&path).expect("preserved file"), previous);
    fs::remove_dir(&obstruction).expect("remove obstruction");
    client
        .complete(request())
        .await
        .expect("recording recovery");
    assert_eq!(
        recording(&path)["scenarios"]
            .as_array()
            .expect("recovered scenarios")
            .len(),
        3
    );
    proxy.shutdown().await;
    let replay = Twin::start(Config {
        scenarios_path: Some(path),
        ..config()
    })
    .await;
    for _ in 0..3 {
        replay
            .client("recording")
            .complete(request())
            .await
            .expect("recovered fixture replay");
    }
    replay.shutdown().await;
    upstream.shutdown().await;
}

#[tokio::test]
async fn malformed_append_file_is_rejected_without_overwriting_it() {
    let directory = TempDir::new();
    let path = directory.0.join("recording.json");
    let upstream = Twin::start(config()).await;
    for contents in [
        b"{\"scenarios\":[".as_slice(),
        b"{\"scenarios\":[{\"script\":{\"kind\":\"unknown\"}}]}",
    ] {
        fs::write(&path, contents).expect("bad recording");
        assert!(
            twin_anthropic::build_app_with_config(proxy_config(&upstream, &path, true)).is_err()
        );
        assert_eq!(fs::read(&path).expect("preserved bad recording"), contents);
    }
    upstream.shutdown().await;
}
