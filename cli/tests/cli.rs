use std::process::Command;

#[test]
fn help_runs_without_credentials() {
    let output = Command::new(env!("CARGO_BIN_EXE_lithos"))
        .arg("--help")
        .output()
        .expect("lithos --help should run");

    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("Send stateless prompts"));
    assert!(output.stderr.is_empty());
}

#[test]
fn models_runs_without_credentials_or_network() {
    let output = Command::new(env!("CARGO_BIN_EXE_lithos"))
        .args(["models", "--json"])
        .output()
        .expect("lithos models should run");

    assert!(output.status.success());
    let value: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("model output should be JSON");
    assert_eq!(value["version"], 1);
    assert!(
        value["models"]
            .as_array()
            .is_some_and(|models| !models.is_empty())
    );
    assert!(output.stderr.is_empty());
}
