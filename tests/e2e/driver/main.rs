use std::error::Error as StdError;
use std::io::{self, Write as _};
use std::path::Path;
use std::process::ExitCode;
use std::{env, fs};

use lithos_llm::Client;
use lithos_llm::catalog::Catalog;
use lithos_llm::credentials::EnvironmentCredentials;
use lithos_llm::estimate::{EstimateWarning, request_tokens};
use lithos_llm::types::{Error, Request};
use serde_json::{Value, json};

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            let _ignored = writeln!(io::stderr().lock(), "error: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<(), Box<dyn StdError>> {
    let mut arguments = env::args().skip(1);
    let action = arguments.next().ok_or("an action is required")?;
    let output = match action.as_str() {
        "estimate" => {
            let request = read_request(required_argument(&mut arguments, "request path")?)?;
            estimate(&request)
        }
        "count" => {
            let catalog = required_argument(&mut arguments, "catalog path")?;
            let request = read_request(required_argument(&mut arguments, "request path")?)?;
            count(Path::new(&catalog), request).await?
        }
        other => return Err(format!("unknown action `{other}`").into()),
    };
    writeln!(
        io::stdout().lock(),
        "{}",
        serde_json::to_string_pretty(&output)?
    )?;
    Ok(())
}

fn required_argument(
    arguments: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<String, Box<dyn StdError>> {
    arguments
        .next()
        .ok_or_else(|| format!("{name} is required").into())
}

fn read_request(path: String) -> Result<Request, Box<dyn StdError>> {
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

fn estimate(request: &Request) -> Value {
    let estimate = request_tokens(request);
    let warnings = estimate
        .warnings()
        .map(|warning| {
            json!({
                "code": warning.code(),
                "message": warning.to_string(),
                "present": estimate.has_warning(warning),
            })
        })
        .collect::<Vec<_>>();
    json!({
        "kind": "estimate",
        "tokens": estimate.tokens(),
        "warnings": warnings,
        "has_media_warning": estimate.has_warning(EstimateWarning::Media),
    })
}

async fn count(catalog_path: &Path, request: Request) -> Result<Value, Box<dyn StdError>> {
    let source = fs::read_to_string(catalog_path)?;
    let catalog = Catalog::builder()
        .with_builtin()
        .toml_layer(catalog_path.display().to_string(), &source)?
        .build()?;
    let build = Client::builder()
        .catalog(catalog)
        .credentials(EnvironmentCredentials::conventional())
        .build()?;
    match build.client.count_input_tokens(request).await {
        Ok(Some(count)) => Ok(json!({
            "kind": "provider_count",
            "tokens": count.tokens(),
            "model": count.model().to_string(),
        })),
        Ok(None) => Ok(json!({ "kind": "unsupported" })),
        Err(error) => Ok(render_error(&error)),
    }
}

fn render_error(error: &Error) -> Value {
    json!({
        "kind": "error",
        "error_kind": format!("{:?}", error.kind()),
        "message": error.message(),
        "provider": error.provider().map(ToString::to_string),
        "status": error.status(),
        "provider_code": error.provider_code(),
        "retry": format!("{:?}", error.retry_classification()),
    })
}
