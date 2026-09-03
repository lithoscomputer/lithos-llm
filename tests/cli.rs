#![cfg(feature = "cli")]

use std::process::{self, Command};
use std::{env, fs};

#[test]
fn help_runs_without_credentials() {
    let output = Command::new(env!("CARGO_BIN_EXE_lllm"))
        .arg("--help")
        .env_remove("LLLM_MODEL")
        .output()
        .expect("lllm --help should run");

    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("Send stateless prompts"));
    assert!(output.stderr.is_empty());
}

#[test]
fn models_runs_without_credentials_or_network() {
    let output = Command::new(env!("CARGO_BIN_EXE_lllm"))
        .args(["models", "--json"])
        .env_remove("LLLM_MODEL")
        .output()
        .expect("lllm models should run");

    assert!(output.status.success());
    let value: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("model output should be JSON");
    assert_eq!(value["version"], 2);
    assert!(
        value["models"]
            .as_array()
            .is_some_and(|models| !models.is_empty())
    );
    assert!(output.stderr.is_empty());
}

#[test]
fn process_environment_model_controls_resolution() {
    let output = Command::new(env!("CARGO_BIN_EXE_lllm"))
        .arg("resolve")
        .env("LLLM_MODEL", "openai/gpt-5")
        .output()
        .expect("lllm resolve should run");

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"openai/gpt-5\n");
    assert!(output.stderr.is_empty());
}

#[test]
fn configured_filter_reads_conventional_environment_credentials() {
    let output = Command::new(env!("CARGO_BIN_EXE_lllm"))
        .args(["models", "--provider", "openai", "--configured", "--json"])
        .env_remove("LLLM_MODEL")
        .env("OPENAI_API_KEY", "test-key")
        .output()
        .expect("lllm models should run");

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("model output should be JSON");
    let models = value["models"]
        .as_array()
        .expect("models should be an array");
    assert!(!models.is_empty());
    assert!(models.iter().all(|model| {
        model["selector"]
            .as_str()
            .is_some_and(|selector| selector.starts_with("openai/"))
            && model["credentials_configured"] == true
    }));
    assert!(output.stderr.is_empty());
}

#[test]
fn a_catalog_overlay_is_available_to_resolution() {
    let path = env::temp_dir().join(format!("lllm-catalog-{}.toml", process::id()));
    fs::write(
        &path,
        r#"
schema_version = 1

[providers.local]
display_name = "Local"
adapter = "openai-compatible"
codec = "openai-chat"
base_url = "http://127.0.0.1:1234/v1"
auth = { type = "none" }
default_model = "qwen"

[providers.local.models.qwen]
display_name = "Qwen"
api_model = "qwen"
capabilities = { text = true }
"#,
    )
    .expect("fixture should be written");

    let output = Command::new(env!("CARGO_BIN_EXE_lllm"))
        .args([
            "--catalog",
            path.to_str().expect("path should be UTF-8"),
            "resolve",
            "--model",
            "local/qwen",
        ])
        .env_remove("LLLM_MODEL")
        .output()
        .expect("lllm resolve should run");

    fs::remove_file(path).expect("fixture should be removed");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"local/qwen\n");
    assert!(output.stderr.is_empty());
}
