use std::collections::{BTreeMap, VecDeque};
use std::error::Error as StdError;
use std::io::{self, Write as _};
use std::num::NonZeroUsize;
use std::path::Path;
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};
use std::{env, fs};

use async_trait::async_trait;
use futures_util::StreamExt as _;
use futures_util::stream::unfold;
use lithos_llm::adapter::{
    AdapterBuildError, AdapterContext, AdapterFactory, ProviderAdapter, ResolvedCall,
};
use lithos_llm::catalog::{
    AdapterId, Catalog, CatalogBuilder, CatalogProvider, ModelId, ProviderId,
};
use lithos_llm::client::ClientBuildError;
use lithos_llm::credentials::{
    CredentialHeader, CredentialProvider, Credentials, EnvironmentCredentials,
    EnvironmentCredentialsBuilder, HttpAuthentication, NoCredentials, SecretValue,
    StaticCredentials,
};
use lithos_llm::estimate::{EstimateWarning, request_tokens};
use lithos_llm::middleware::{
    Call, CallContext, ConcurrencyLimitMiddleware, Middleware, Next, Observer, ObserverMiddleware,
    Output, RetryMiddleware, RetryPolicy, RetryStage, TimeoutMiddleware, finalize_stream,
    inspect_stream, map_stream,
};
use lithos_llm::resolver::{AvailableProviders, CatalogResolver, ModelResolver};
use lithos_llm::types::{
    AudioContent, CacheHint, ContentPart, DocumentContent, Error, ErrorKind, FinishReason,
    ImageContent, MediaSource, Message, ReasoningEffort, Request, RequestBuildError, Response,
    ResponseFormat, ResponseStream, RetryClassification, Role, Speed, StreamEvent, TokenCounts,
    ToolCall, ToolChoice, ToolDefinition, Warning,
};
use lithos_llm::{Client, ClientBuild};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::time::sleep;

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
        "simulate" => {
            let simulation = read_json(required_argument(&mut arguments, "simulation path")?)?;
            simulate(simulation).await?
        }
        "requests" => {
            let cases = read_json(required_argument(&mut arguments, "request cases path")?)?;
            exercise_requests(cases)
        }
        "catalog" => {
            let exercise = read_json(required_argument(&mut arguments, "catalog exercise path")?)?;
            exercise_catalog(exercise)
        }
        "catalog-errors" => {
            let cases = read_json(required_argument(&mut arguments, "catalog cases path")?)?;
            exercise_catalog_errors(cases)
        }
        "values" => {
            let exercise = read_json(required_argument(&mut arguments, "values path")?)?;
            exercise_values(exercise)
        }
        "client-builds" => {
            let exercise = read_json(required_argument(&mut arguments, "client builds path")?)?;
            exercise_client_builds(exercise)
        }
        "resolve-matrix" => {
            let exercise = read_json(required_argument(&mut arguments, "resolve matrix path")?)?;
            exercise_resolver(exercise)?
        }
        "credentials" => {
            let exercise = read_json(required_argument(
                &mut arguments,
                "credentials exercise path",
            )?)?;
            exercise_credentials(exercise).await?
        }
        "call" => {
            let catalog = required_argument(&mut arguments, "catalog path")?;
            let request = read_request(required_argument(&mut arguments, "request path")?)?;
            let mode = arguments.next().unwrap_or_else(|| "complete".to_owned());
            call(Path::new(&catalog), request, &mode).await?
        }
        "calls" => {
            let catalog = required_argument(&mut arguments, "catalog path")?;
            let requests = read_json(required_argument(&mut arguments, "requests path")?)?;
            calls(Path::new(&catalog), requests).await?
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
    read_json(path)
}

fn read_json<T: for<'de> Deserialize<'de>>(path: String) -> Result<T, Box<dyn StdError>> {
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

#[derive(Debug, Deserialize)]
struct RequestCase {
    name: String,
    spec: RequestSpec,
}

#[derive(Debug, Default, Deserialize)]
struct RequestSpec {
    model:                   Option<String>,
    #[serde(default)]
    messages:                Vec<Message>,
    system:                  Option<String>,
    developer:               Option<String>,
    user:                    Option<String>,
    #[serde(default)]
    tools:                   Vec<ToolDefinition>,
    tool_choice:             Option<ToolChoice>,
    response_format:         Option<ResponseFormat>,
    max_output_tokens:       Option<u32>,
    temperature:             Option<f32>,
    top_p:                   Option<f32>,
    reasoning_effort:        Option<ReasoningEffort>,
    cache_hint:              Option<CacheHint>,
    cache_key:               Option<String>,
    speed:                   Option<Speed>,
    timeout_ms:              Option<u64>,
    stop_sequence:           Option<String>,
    #[serde(default)]
    stop_sequences:          Vec<String>,
    #[serde(default)]
    metadata:                BTreeMap<String, String>,
    #[serde(default)]
    provider_options:        BTreeMap<ProviderId, serde_json::Map<String, Value>>,
    #[serde(default)]
    provider_option_entries: Vec<ProviderOptionEntry>,
}

#[derive(Debug, Deserialize)]
struct ProviderOptionEntry {
    provider: ProviderId,
    key:      String,
    value:    Value,
}

fn exercise_requests(cases: Vec<RequestCase>) -> Value {
    let results = cases
        .into_iter()
        .map(|case| {
            let name = case.name;
            match build_request(case.spec) {
                Ok(request) => json!({
                    "name": name,
                    "result": "ok",
                    "model": request.model(),
                    "messages": request.messages().len(),
                    "tools": request.tools().len(),
                    "tool_choice": request.tool_choice(),
                    "response_format": request.response_format(),
                    "max_output_tokens": request.max_output_tokens(),
                    "temperature": request.temperature(),
                    "top_p": request.top_p(),
                    "reasoning_effort": request.reasoning_effort(),
                    "cache_hint": request.cache_hint(),
                    "speed": request.speed(),
                    "timeout_ms": request.timeout().map(|value| value.as_millis()),
                    "stop_sequences": request.stop_sequences(),
                    "metadata": request.metadata(),
                    "provider_options": request.provider_options(),
                    "fixture_options": request.options_for(&ProviderId::new("fixture")),
                    "serialized": request,
                }),
                Err(error) => json!({
                    "name": name,
                    "result": "error",
                    "message": error.to_string(),
                }),
            }
        })
        .collect::<Vec<_>>();
    json!({ "kind": "requests", "cases": results })
}

fn build_request(spec: RequestSpec) -> Result<Request, RequestBuildError> {
    let mut builder = Request::builder();
    if let Some(model) = spec.model {
        builder = builder.model(model);
    }
    for message in spec.messages {
        builder = builder.message(message);
    }
    if let Some(text) = spec.system {
        builder = builder.system(text);
    }
    if let Some(text) = spec.developer {
        builder = builder.developer(text);
    }
    if let Some(text) = spec.user {
        builder = builder.user(text);
    }
    for tool in spec.tools {
        builder = builder.tool(tool);
    }
    if let Some(choice) = spec.tool_choice {
        builder = builder.tool_choice(choice);
    }
    if let Some(format) = spec.response_format {
        builder = builder.response_format(format);
    }
    if let Some(tokens) = spec.max_output_tokens {
        builder = builder.max_output_tokens(tokens);
    }
    if let Some(temperature) = spec.temperature {
        builder = builder.temperature(temperature);
    }
    if let Some(top_p) = spec.top_p {
        builder = builder.top_p(top_p);
    }
    if let Some(effort) = spec.reasoning_effort {
        builder = builder.reasoning_effort(effort);
    }
    if let Some(hint) = spec.cache_hint {
        builder = builder.cache_hint(hint);
    }
    if let Some(key) = spec.cache_key {
        builder = builder.cache_key(key);
    }
    if let Some(speed) = spec.speed {
        builder = builder.speed(speed);
    }
    if let Some(milliseconds) = spec.timeout_ms {
        builder = builder.timeout(Duration::from_millis(milliseconds));
    }
    if let Some(sequence) = spec.stop_sequence {
        builder = builder.stop_sequence(sequence);
    }
    builder = builder.stop_sequences(spec.stop_sequences);
    for (key, value) in spec.metadata {
        builder = builder.metadata_entry(key, value);
    }
    for (provider, options) in spec.provider_options {
        builder = builder.provider_options(provider, options);
    }
    for entry in spec.provider_option_entries {
        builder = builder.provider_option(entry.provider, entry.key, entry.value);
    }
    builder.build()
}

#[derive(Debug, Deserialize)]
struct CatalogExercise {
    #[serde(default)]
    builtin:          bool,
    #[serde(default)]
    layers:           Vec<CatalogLayer>,
    #[serde(default)]
    provider_lookups: Vec<String>,
    #[serde(default)]
    model_lookups:    Vec<ModelLookup>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum CatalogLayer {
    Named { name: String, path: String },
    Overlay { path: String },
}

#[derive(Debug, Deserialize)]
struct ModelLookup {
    provider: String,
    model:    String,
}

fn exercise_catalog(exercise: CatalogExercise) -> Value {
    let mut builder = Catalog::builder();
    if exercise.builtin {
        builder = builder.with_builtin();
    }
    for layer in exercise.layers {
        let result = match layer {
            CatalogLayer::Named { name, path } => fs::read_to_string(path)
                .map_err(|error| error.to_string())
                .and_then(|source| {
                    builder
                        .toml_layer(name, &source)
                        .map_err(|error| error.to_string())
                }),
            CatalogLayer::Overlay { path } => fs::read_to_string(path)
                .map_err(|error| error.to_string())
                .and_then(|source| {
                    builder
                        .overlay_toml(&source)
                        .map_err(|error| error.to_string())
                }),
        };
        match result {
            Ok(next) => builder = next,
            Err(error) => return json!({ "kind": "catalog_error", "message": error }),
        }
    }
    let catalog = match builder.build() {
        Ok(catalog) => catalog,
        Err(error) => return json!({ "kind": "catalog_error", "message": error.to_string() }),
    };
    let providers = catalog
        .providers()
        .map(|provider| {
            let models = provider
                .models()
                .map(|model| {
                    let pricing = model.pricing();
                    json!({
                        "id": model.id(),
                        "provider": model.provider_id(),
                        "display_name": model.display_name(),
                        "aliases": model.aliases(),
                        "api_model": model.api_model(),
                        "limits": model.limits(),
                        "capabilities": model.capabilities(),
                        "pricing": pricing,
                        "base_pricing": pricing.map(|value| value.for_input_tokens(1)),
                        "long_pricing": pricing.map(|value| value.for_input_tokens(u64::MAX)),
                        "fast_pricing": pricing.map(|value| value.for_speed(Some(Speed::Fast))),
                        "balanced_pricing": pricing.map(|value| value.for_speed(Some(Speed::Balanced))),
                        "economical_pricing": pricing.map(|value| value.for_speed(Some(Speed::Economical))),
                        "metadata_app": model.metadata().get("app"),
                        "metadata_app_map": model.metadata().namespace::<BTreeMap<String, String>>("app").map_or_else(|error| json!({"error": error.to_string()}), |value| json!(value)),
                        "passthrough": model.is_passthrough(),
                    })
                })
                .collect::<Vec<_>>();
            json!({
                "id": provider.id(),
                "display_name": provider.display_name(),
                "aliases": provider.aliases(),
                "adapter": provider.adapter(),
                "codec": provider.codec(),
                "base_url": provider.base_url(),
                "auth": provider.auth(),
                "priority": provider.priority(),
                "allows_passthrough": provider.allows_passthrough(),
                "default_model": provider.default_model(),
                "default_headers": provider.default_headers(),
                "adapter_options": provider.adapter_options(),
                "default_options": provider.default_options(),
                "metadata_app": provider.metadata().get("app"),
                "models": models,
            })
        })
        .collect::<Vec<_>>();
    let provider_lookups = exercise
        .provider_lookups
        .into_iter()
        .map(|selector| {
            let result = catalog.provider(&selector).map_or_else(
                |error| error.to_string(),
                |provider| provider.id().to_string(),
            );
            json!({ "selector": selector, "result": result })
        })
        .collect::<Vec<_>>();
    let model_lookups = exercise
        .model_lookups
        .into_iter()
        .map(|lookup| {
            let result = catalog.model(&lookup.provider, &lookup.model).map_or_else(
                |error| error.to_string(),
                |model| format!("{}/{}", model.provider_id(), model.id()),
            );
            json!({ "provider": lookup.provider, "model": lookup.model, "result": result })
        })
        .collect::<Vec<_>>();
    json!({
        "kind": "catalog",
        "schema_version": catalog.schema_version(),
        "provider_count": providers.len(),
        "providers": providers,
        "provider_lookups": provider_lookups,
        "model_lookups": model_lookups,
        "canonical_lookup": catalog.provider_by_id(&ProviderId::new("fixture")).is_some(),
    })
}

#[derive(Debug, Deserialize)]
struct CatalogErrorCase {
    name: String,
    path: String,
}

fn exercise_catalog_errors(cases: Vec<CatalogErrorCase>) -> Value {
    let results = cases
        .into_iter()
        .map(|case| {
            let source = match fs::read_to_string(&case.path) {
                Ok(source) => source,
                Err(error) => {
                    return json!({ "name": case.name, "error": error.to_string() });
                }
            };
            let result = Catalog::builder()
                .toml_layer(case.path, &source)
                .and_then(CatalogBuilder::build);
            match result {
                Ok(catalog) => json!({
                    "name": case.name,
                    "result": "ok",
                    "providers": catalog.providers().len(),
                }),
                Err(error) => {
                    let mut causes = Vec::new();
                    let mut source = StdError::source(&error);
                    while let Some(cause) = source {
                        causes.push(cause.to_string());
                        source = cause.source();
                    }
                    json!({
                        "name": case.name,
                        "error": error.to_string(),
                        "causes": causes,
                    })
                }
            }
        })
        .collect::<Vec<_>>();
    json!({ "kind": "catalog_errors", "cases": results })
}

#[derive(Debug, Deserialize)]
struct ValuesExercise {
    #[serde(default)]
    messages:         Vec<MessageConstruction>,
    #[serde(default)]
    media:            Vec<MediaConstruction>,
    #[serde(default)]
    custom_tools:     Vec<CustomToolConstruction>,
    #[serde(default)]
    custom_calls:     Vec<CustomCallConstruction>,
    #[serde(default)]
    finish_reasons:   Vec<Value>,
    #[serde(default)]
    warnings:         Vec<Value>,
    #[serde(default)]
    inclusive_tokens: Vec<[u64; 5]>,
}

#[derive(Debug, Deserialize)]
struct MessageConstruction {
    role:         Role,
    #[serde(default)]
    content:      Vec<ContentPart>,
    name:         Option<String>,
    tool_call_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MediaConstruction {
    source: MediaConstructionSource,
    wrap:   MediaWrapper,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum MediaConstructionSource {
    Url {
        value: String,
    },
    UrlWithMediaType {
        value:      String,
        media_type: String,
    },
    Base64 {
        value:      String,
        media_type: String,
    },
    Parse {
        value: String,
    },
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum MediaWrapper {
    Image,
    Audio,
    Document,
}

#[derive(Debug, Deserialize)]
struct CustomToolConstruction {
    name:        String,
    description: String,
    format:      Value,
}

#[derive(Debug, Deserialize)]
struct CustomCallConstruction {
    id:    String,
    name:  String,
    input: String,
}

fn exercise_values(exercise: ValuesExercise) -> Value {
    let messages = exercise
        .messages
        .into_iter()
        .map(|spec| {
            let mut message = Message::new(spec.role, spec.content);
            if let Some(name) = spec.name {
                message = message.with_name(name);
            }
            if let Some(tool_call_id) = spec.tool_call_id {
                message = message.with_tool_call_id(tool_call_id);
            }
            json!({
                "role": message.role(),
                "content": message.content(),
                "name": message.name(),
                "tool_call_id": message.tool_call_id(),
                "serialized": message,
            })
        })
        .collect::<Vec<_>>();
    let media = exercise
        .media
        .into_iter()
        .map(|spec| {
            let source = match spec.source {
                MediaConstructionSource::Url { value } => MediaSource::url(value),
                MediaConstructionSource::UrlWithMediaType { value, media_type } => {
                    MediaSource::url_with_media_type(value, media_type)
                }
                MediaConstructionSource::Base64 { value, media_type } => {
                    MediaSource::base64(value, media_type)
                }
                MediaConstructionSource::Parse { value } => MediaSource::parse(&value),
            };
            let media_type = source.media_type().map(ToOwned::to_owned);
            let base64_data = source.base64_data().map(ToOwned::to_owned);
            let part = match spec.wrap {
                MediaWrapper::Image => ContentPart::Image(ImageContent::new(source)),
                MediaWrapper::Audio => ContentPart::Audio(AudioContent::new(source)),
                MediaWrapper::Document => ContentPart::Document(DocumentContent::new(source)),
            };
            json!({
                "media_type": media_type,
                "base64_data": base64_data,
                "part": part,
                "opaque_namespace": part.opaque_namespace(),
            })
        })
        .collect::<Vec<_>>();
    let custom_tools = exercise
        .custom_tools
        .into_iter()
        .map(|spec| {
            let tool = ToolDefinition::custom(spec.name, spec.description, spec.format);
            json!({ "custom": tool.is_custom(), "tool": tool })
        })
        .collect::<Vec<_>>();
    let custom_calls = exercise
        .custom_calls
        .into_iter()
        .map(|spec| ToolCall::custom(spec.id, spec.name, spec.input))
        .collect::<Vec<_>>();
    let finish_reasons = exercise
        .finish_reasons
        .into_iter()
        .map(
            |input| match serde_json::from_value::<FinishReason>(input.clone()) {
                Ok(reason) => json!({ "input": input, "result": reason }),
                Err(error) => json!({ "input": input, "error": error.to_string() }),
            },
        )
        .collect::<Vec<_>>();
    let warnings = exercise
        .warnings
        .into_iter()
        .map(
            |input| match serde_json::from_value::<Warning>(input.clone()) {
                Ok(warning) => json!({ "input": input, "result": warning }),
                Err(error) => json!({ "input": input, "error": error.to_string() }),
            },
        )
        .collect::<Vec<_>>();
    let inclusive_tokens = exercise
        .inclusive_tokens
        .into_iter()
        .map(|[input, output, reasoning, cache_read, cache_write]| {
            let tokens =
                TokenCounts::from_inclusive(input, output, reasoning, cache_read, cache_write);
            json!({
                "tokens": tokens,
                "total": tokens.total(),
                "billable_output": tokens.billable_output(),
            })
        })
        .collect::<Vec<_>>();
    json!({
        "kind": "values",
        "messages": messages,
        "media": media,
        "custom_tools": custom_tools,
        "custom_calls": custom_calls,
        "finish_reasons": finish_reasons,
        "warnings": warnings,
        "inclusive_tokens": inclusive_tokens,
    })
}

#[derive(Debug, Deserialize)]
struct ClientBuildExercise {
    cases: Vec<ClientBuildCase>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ClientBuildCase {
    MissingCatalog {
        name: String,
    },
    FromEnv {
        name: String,
    },
    Catalog {
        name:    String,
        path:    String,
        enabled: Option<Vec<String>>,
    },
}

fn exercise_client_builds(exercise: ClientBuildExercise) -> Value {
    let cases = exercise
        .cases
        .into_iter()
        .map(|case| match case {
            ClientBuildCase::MissingCatalog { name } => {
                render_client_build_result(&name, Client::builder().build())
            }
            ClientBuildCase::FromEnv { name } => {
                render_client_build_result(&name, Client::from_env())
            }
            ClientBuildCase::Catalog {
                name,
                path,
                enabled,
            } => {
                let result = fs::read_to_string(&path)
                    .map_err(|error| error.to_string())
                    .and_then(|source| {
                        Catalog::builder()
                            .toml_layer(path, &source)
                            .and_then(CatalogBuilder::build)
                            .map_err(|error| error.to_string())
                    });
                match result {
                    Ok(catalog) => {
                        let mut builder = Client::builder().catalog(catalog);
                        if let Some(enabled) = enabled {
                            builder = builder.enabled_providers(enabled);
                        }
                        render_client_build_result(&name, builder.build())
                    }
                    Err(error) => json!({ "name": name, "error": error }),
                }
            }
        })
        .collect::<Vec<_>>();
    json!({ "kind": "client_builds", "cases": cases })
}

fn render_client_build_result(name: &str, result: Result<ClientBuild, ClientBuildError>) -> Value {
    match result {
        Ok(build) => {
            let all = AvailableProviders::all(build.client.catalog());
            let available = build
                .client
                .available_providers()
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>();
            let all = all.iter().map(ToString::to_string).collect::<Vec<_>>();
            let issues = build
                .issues
                .into_iter()
                .map(|issue| {
                    json!({
                        "provider": issue.provider,
                        "adapter": issue.adapter,
                        "cause": issue.cause.to_string(),
                        "source": StdError::source(&issue.cause).map(ToString::to_string),
                    })
                })
                .collect::<Vec<_>>();
            json!({
                "name": name,
                "result": "ok",
                "client": format!("{:?}", build.client),
                "available": available,
                "all": all,
                "issues": issues,
            })
        }
        Err(error) => json!({
            "name": name,
            "error": error.to_string(),
            "source": StdError::source(&error).map(ToString::to_string),
        }),
    }
}

#[derive(Debug, Deserialize)]
struct ResolveExercise {
    catalog: String,
    cases:   Vec<ResolveCase>,
}

#[derive(Debug, Deserialize)]
struct ResolveCase {
    name:      String,
    request:   Request,
    available: ResolveAvailability,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ResolveAvailability {
    Named(String),
    Providers(Vec<ProviderId>),
}

fn exercise_resolver(exercise: ResolveExercise) -> Result<Value, Box<dyn StdError>> {
    let source = fs::read_to_string(&exercise.catalog)?;
    let catalog = Catalog::builder()
        .toml_layer(exercise.catalog, &source)?
        .build()?;
    let cases = exercise
        .cases
        .into_iter()
        .map(|case| {
            let available = match case.available {
                ResolveAvailability::Named(name) if name == "all" => {
                    AvailableProviders::all(&catalog)
                }
                ResolveAvailability::Named(_) => AvailableProviders::default(),
                ResolveAvailability::Providers(providers) => AvailableProviders::new(providers),
            };
            match CatalogResolver.resolve(&case.request, &catalog, &available) {
                Ok(route) => json!({
                    "name": case.name,
                    "result": "ok",
                    "provider": route.provider().id(),
                    "model": route.model().id(),
                    "api_model": route.api_model(),
                    "handle": route.handle(),
                }),
                Err(error) => json!({
                    "name": case.name,
                    "error": error.to_string(),
                }),
            }
        })
        .collect::<Vec<_>>();
    Ok(json!({ "kind": "resolve_matrix", "cases": cases }))
}

#[derive(Debug, Deserialize)]
struct CredentialsExercise {
    catalog:     String,
    #[serde(default)]
    operations:  Vec<CredentialOperation>,
    #[serde(default)]
    resolutions: Vec<CredentialResolution>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum CredentialOperation {
    Bearer {
        provider: String,
        variable: String,
    },
    Header {
        provider: String,
        name:     String,
        variable: String,
    },
    BearerHeader {
        provider: String,
        name:     String,
        variable: String,
    },
    OrBearer {
        provider: String,
        variable: String,
    },
    OrHeader {
        provider: String,
        name:     String,
        variable: String,
    },
    AwsDefaultChain {
        provider: String,
        region:   Option<String>,
    },
    OrAwsDefaultChain {
        provider: String,
        region:   Option<String>,
    },
    BedrockBearer {
        provider: String,
        variable: String,
    },
    OrBedrockBearer {
        provider: String,
        variable: String,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
enum CredentialResolution {
    Environment { provider: String },
    Conventional { provider: String },
    None { provider: String },
    StaticMissing { provider: String },
    StaticBearer { provider: String },
    StaticHeader { provider: String },
    StaticHeaders { provider: String },
    StaticNone { provider: String },
}

async fn exercise_credentials(exercise: CredentialsExercise) -> Result<Value, Box<dyn StdError>> {
    let source = fs::read_to_string(exercise.catalog)?;
    let catalog = Catalog::builder()
        .toml_layer("credentials", &source)?
        .build()?;
    let mut builder = EnvironmentCredentials::builder();
    for operation in exercise.operations {
        builder = apply_credential_operation(builder, operation);
    }
    let environment = builder.build();
    let conventional = EnvironmentCredentials::conventional();
    let mut results = Vec::new();
    for resolution in exercise.resolutions {
        let (provider_name, result) = match resolution {
            CredentialResolution::Environment { provider } => {
                let result = resolve_credentials(&catalog, &provider, &environment).await;
                (provider, result)
            }
            CredentialResolution::Conventional { provider } => {
                let result = resolve_credentials(&catalog, &provider, &conventional).await;
                (provider, result)
            }
            CredentialResolution::None { provider } => {
                let result = resolve_credentials(&catalog, &provider, &NoCredentials).await;
                (provider, result)
            }
            CredentialResolution::StaticMissing { provider } => {
                let result =
                    resolve_credentials(&catalog, &provider, &StaticCredentials::new()).await;
                (provider, result)
            }
            CredentialResolution::StaticBearer { provider } => {
                let credentials = StaticCredentials::new().with(
                    provider.clone(),
                    Credentials::bearer(SecretValue::new("static-secret")),
                );
                let result = resolve_credentials(&catalog, &provider, &credentials).await;
                (provider, result)
            }
            CredentialResolution::StaticHeader { provider } => {
                let credentials = StaticCredentials::new().with(
                    provider.clone(),
                    Credentials::header(CredentialHeader::new(
                        "x-static",
                        SecretValue::new("static-secret"),
                    )),
                );
                let result = resolve_credentials(&catalog, &provider, &credentials).await;
                (provider, result)
            }
            CredentialResolution::StaticHeaders { provider } => {
                let credentials = StaticCredentials::new().with(
                    provider.clone(),
                    Credentials::headers([CredentialHeader::new(
                        "x-extra",
                        SecretValue::new("static-secret"),
                    )]),
                );
                let result = resolve_credentials(&catalog, &provider, &credentials).await;
                (provider, result)
            }
            CredentialResolution::StaticNone { provider } => {
                let credentials =
                    StaticCredentials::new().with(provider.clone(), Credentials::none());
                let result = resolve_credentials(&catalog, &provider, &credentials).await;
                (provider, result)
            }
        };
        results.push(json!({ "provider": provider_name, "result": result }));
    }
    Ok(json!({ "kind": "credentials", "results": results }))
}

fn apply_credential_operation(
    builder: EnvironmentCredentialsBuilder,
    operation: CredentialOperation,
) -> EnvironmentCredentialsBuilder {
    match operation {
        CredentialOperation::Bearer { provider, variable } => builder.bearer(provider, variable),
        CredentialOperation::Header {
            provider,
            name,
            variable,
        } => builder.header(provider, name, variable),
        CredentialOperation::BearerHeader {
            provider,
            name,
            variable,
        } => builder.bearer_header(provider, name, variable),
        CredentialOperation::OrBearer { provider, variable } => {
            builder.or_bearer(provider, variable)
        }
        CredentialOperation::OrHeader {
            provider,
            name,
            variable,
        } => builder.or_header(provider, name, variable),
        CredentialOperation::AwsDefaultChain { provider, region } => {
            builder.aws_default_chain(provider, region)
        }
        CredentialOperation::OrAwsDefaultChain { provider, region } => {
            builder.or_aws_default_chain(provider, region)
        }
        CredentialOperation::BedrockBearer { provider, variable } => {
            builder.bedrock_bearer(provider, variable)
        }
        CredentialOperation::OrBedrockBearer { provider, variable } => {
            builder.or_bedrock_bearer(provider, variable)
        }
    }
}

async fn resolve_credentials(
    catalog: &Catalog,
    provider: &str,
    credentials: &dyn CredentialProvider,
) -> Value {
    let Ok(provider) = catalog.provider(provider) else {
        return json!({ "error": "provider not found" });
    };
    match credentials.credentials(provider).await {
        Ok(credentials) => render_credentials(&credentials),
        Err(error) => json!({
            "error": error.to_string(),
            "source": StdError::source(&error).map(ToString::to_string),
        }),
    }
}

fn render_credentials(credentials: &Credentials) -> Value {
    let detail = match credentials {
        Credentials::Http(http) => {
            let auth = match &http.auth {
                HttpAuthentication::None => json!({ "type": "none" }),
                HttpAuthentication::Bearer(secret) => json!({
                    "type": "bearer",
                    "length": secret.expose_secret().len(),
                    "debug": format!("{secret:?}"),
                }),
                HttpAuthentication::Header(header) => json!({
                    "type": "header",
                    "name": header.name,
                    "length": header.value.expose_secret().len(),
                    "debug": format!("{header:?}"),
                }),
                _ => json!({ "type": "unknown" }),
            };
            json!({
                "type": "http",
                "auth": auth,
                "extra_headers": http.extra_headers.iter().map(|header| json!({
                    "name": header.name,
                    "length": header.value.expose_secret().len(),
                })).collect::<Vec<_>>(),
            })
        }
        Credentials::AwsDefaultChain { region } => {
            json!({ "type": "aws_default_chain", "region": region })
        }
        Credentials::BedrockBearer(secret) => json!({
            "type": "bedrock_bearer",
            "length": secret.expose_secret().len(),
        }),
        _ => json!({ "type": "unknown" }),
    };
    json!({ "debug": format!("{credentials:?}"), "detail": detail })
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

async fn call(
    catalog_path: &Path,
    request: Request,
    mode: &str,
) -> Result<Value, Box<dyn StdError>> {
    let source = fs::read_to_string(catalog_path)?;
    let catalog = Catalog::builder()
        .with_builtin()
        .toml_layer(catalog_path.display().to_string(), &source)?
        .build()?;
    let build = Client::builder()
        .catalog(catalog)
        .credentials(EnvironmentCredentials::conventional())
        .build()?;
    match mode {
        "complete" => match build.client.complete(request).await {
            Ok(response) => Ok(json!({
                "kind": "complete",
                "text": response.text(),
                "response": response,
            })),
            Err(error) => Ok(render_error_details(&error)),
        },
        "stream" => match build.client.stream(request).await {
            Ok(mut stream) => {
                let mut events = Vec::new();
                while let Some(item) = stream.next().await {
                    match item {
                        Ok(event) => events.push(json!({
                            "name": event_name(&event),
                            "event": event,
                        })),
                        Err(error) => {
                            events.push(render_error_details(&error));
                            break;
                        }
                    }
                }
                Ok(json!({ "kind": "stream", "events": events }))
            }
            Err(error) => Ok(render_error_details(&error)),
        },
        other => Err(format!("unknown call mode `{other}`").into()),
    }
}

async fn calls(catalog_path: &Path, requests: Vec<Request>) -> Result<Value, Box<dyn StdError>> {
    let mut results = Vec::new();
    for request in requests {
        results.push(call(catalog_path, request, "complete").await?);
    }
    Ok(json!({ "kind": "calls", "results": results }))
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

fn render_error_details(error: &Error) -> Value {
    json!({
        "kind": "error",
        "error": render_error(error),
        "data": error.data(),
        "debug": format!("{error:?}"),
        "display": error.to_string(),
        "retry_after_ms": error.retry_after().map(|value| value.as_millis()),
        "provider_retry_after_ms": error.provider_retry_after().map(|value| value.as_millis()),
        "raw_data": error.raw_data(),
        "source": StdError::source(error).map(ToString::to_string),
    })
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SimulationMode {
    Complete,
    Stream,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Registration {
    Direct,
    Arc,
    Factory,
}

#[derive(Debug, Deserialize)]
struct Simulation {
    mode:                SimulationMode,
    request:             Request,
    steps:               Vec<AdapterStep>,
    #[serde(default = "direct_registration")]
    registration:        Registration,
    #[serde(default)]
    builder_variants:    bool,
    #[serde(default)]
    retry:               Option<RetrySpec>,
    #[serde(default)]
    timeout_ms:          Option<u64>,
    #[serde(default)]
    concurrency:         Option<usize>,
    #[serde(default)]
    observer:            bool,
    #[serde(default)]
    cancel_before:       bool,
    #[serde(default)]
    deadline_ms:         Option<i64>,
    #[serde(default)]
    stream_hooks:        bool,
    #[serde(default)]
    initial_attempt:     Option<u32>,
    #[serde(default)]
    exercise_extensions: bool,
}

const fn direct_registration() -> Registration {
    Registration::Direct
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct RetrySpec {
    max_attempts:       u32,
    initial_delay_ms:   u64,
    max_delay_ms:       u64,
    retry_after_cap_ms: u64,
    #[serde(default)]
    jitter:             bool,
    #[serde(default)]
    observer:           bool,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AdapterStep {
    Complete {
        text:     String,
        #[serde(default)]
        delay_ms: u64,
    },
    Stream {
        items:    Vec<StreamItem>,
        #[serde(default)]
        delay_ms: u64,
    },
    Error {
        message:        String,
        retry:          RetryKind,
        #[serde(default)]
        retry_after_ms: Option<u64>,
        #[serde(default)]
        delay_ms:       u64,
    },
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RetryKind {
    Never,
    Safe,
    After,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StreamItem {
    Event {
        event:    Box<StreamEvent>,
        #[serde(default)]
        delay_ms: u64,
    },
    Error {
        message:  String,
        retry:    RetryKind,
        #[serde(default)]
        delay_ms: u64,
    },
}

#[derive(Clone, Default)]
struct ScriptedAdapter {
    steps:    Arc<Mutex<VecDeque<AdapterStep>>>,
    calls:    Arc<AtomicUsize>,
    attempts: Arc<Mutex<Vec<u32>>>,
}

impl ScriptedAdapter {
    fn new(steps: Vec<AdapterStep>) -> Self {
        Self {
            steps:    Arc::new(Mutex::new(steps.into())),
            calls:    Arc::new(AtomicUsize::new(0)),
            attempts: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn next(&self, call: &ResolvedCall) -> Result<AdapterStep, Error> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.attempts
            .lock()
            .expect("attempt lock should not be poisoned")
            .push(call.context().attempt());
        self.steps
            .lock()
            .expect("step lock should not be poisoned")
            .pop_front()
            .ok_or_else(|| Error::new(ErrorKind::Provider, "the fixture script is exhausted"))
    }
}

#[async_trait]
impl ProviderAdapter for ScriptedAdapter {
    fn id(&self) -> &AdapterId {
        static ID: LazyLock<AdapterId> = LazyLock::new(|| AdapterId::new("fixture-adapter"));
        &ID
    }

    async fn complete(&self, call: &ResolvedCall) -> Result<Response, Error> {
        match self.next(call)? {
            AdapterStep::Complete { text, delay_ms } => {
                sleep(Duration::from_millis(delay_ms)).await;
                Ok(fixture_response(text))
            }
            AdapterStep::Error {
                message,
                retry,
                retry_after_ms,
                delay_ms,
            } => {
                sleep(Duration::from_millis(delay_ms)).await;
                Err(script_error(message, retry, retry_after_ms))
            }
            AdapterStep::Stream { .. } => Err(Error::new(
                ErrorKind::Middleware,
                "a stream script was used for a complete call",
            )),
        }
    }

    async fn stream(&self, call: &ResolvedCall) -> Result<ResponseStream, Error> {
        match self.next(call)? {
            AdapterStep::Stream { items, delay_ms } => {
                sleep(Duration::from_millis(delay_ms)).await;
                Ok(scripted_stream(items))
            }
            AdapterStep::Error {
                message,
                retry,
                retry_after_ms,
                delay_ms,
            } => {
                sleep(Duration::from_millis(delay_ms)).await;
                Err(script_error(message, retry, retry_after_ms))
            }
            AdapterStep::Complete { .. } => Err(Error::new(
                ErrorKind::Middleware,
                "a complete script was used for a stream call",
            )),
        }
    }
}

#[derive(Clone)]
struct FixtureFactory(ScriptedAdapter);

impl AdapterFactory for FixtureFactory {
    fn create(
        &self,
        _provider: &CatalogProvider,
        context: &AdapterContext,
    ) -> Result<Arc<dyn ProviderAdapter>, AdapterBuildError> {
        let _http = context.http();
        let _credentials = context.credentials();
        let _idle = context.stream_idle_timeout();
        Ok(Arc::new(self.0.clone()))
    }
}

#[derive(Default)]
struct RecordingObserver {
    starts:          AtomicUsize,
    completes:       AtomicUsize,
    events:          AtomicUsize,
    errors:          AtomicUsize,
    retries:         AtomicUsize,
    request_retries: AtomicUsize,
    stream_retries:  AtomicUsize,
}

impl Observer for RecordingObserver {
    fn on_start(&self, _call: &Call) {
        self.starts.fetch_add(1, Ordering::Relaxed);
    }

    fn on_complete(&self, _call: &Call, result: Result<&Response, &Error>) {
        self.completes.fetch_add(1, Ordering::Relaxed);
        if result.is_err() {
            self.errors.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn on_stream_event(&self, _call: &Call, event: Result<&StreamEvent, &Error>) {
        self.events.fetch_add(1, Ordering::Relaxed);
        if event.is_err() {
            self.errors.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn on_retry(
        &self,
        _call: &Call,
        _error: &Error,
        _attempt: u32,
        _delay: Duration,
        stage: RetryStage,
    ) {
        self.retries.fetch_add(1, Ordering::Relaxed);
        match stage {
            RetryStage::Request => self.request_retries.fetch_add(1, Ordering::Relaxed),
            RetryStage::Stream => self.stream_retries.fetch_add(1, Ordering::Relaxed),
            _ => 0,
        };
    }
}

#[derive(Clone, Copy, Debug)]
struct PassThroughMiddleware;

#[async_trait]
impl Middleware for PassThroughMiddleware {
    async fn handle(&self, call: Call, next: Next) -> Result<Output, Error> {
        next.run(call).await
    }
}

async fn simulate(simulation: Simulation) -> Result<Value, Box<dyn StdError>> {
    let adapter = ScriptedAdapter::new(simulation.steps);
    let observer = Arc::new(RecordingObserver::default());
    let catalog = Catalog::builder()
        .toml_layer("simulation", simulation_catalog())?
        .build()?;
    let mut builder = Client::builder().catalog(catalog);
    if simulation.builder_variants {
        builder = builder
            .resolver(CatalogResolver)
            .resolver_arc(Arc::new(CatalogResolver))
            .credentials(NoCredentials)
            .credentials_arc(Arc::new(NoCredentials))
            .http(reqwest::Client::new())
            .connect_timeout(None)
            .stream_idle_timeout(None)
            .enabled_providers([ProviderId::new("fixture")])
            .middleware_arc(Arc::new(PassThroughMiddleware));
    }
    builder = match simulation.registration {
        Registration::Direct => builder.adapter("fixture", adapter.clone()),
        Registration::Arc => builder.adapter_arc("fixture", Arc::new(adapter.clone())),
        Registration::Factory => {
            builder.adapter_factory("fixture-adapter", FixtureFactory(adapter.clone()))
        }
    };
    if let Some(retry) = simulation.retry {
        let policy = RetryPolicy::exponential()
            .max_attempts(retry.max_attempts)
            .initial_delay(Duration::from_millis(retry.initial_delay_ms))
            .max_delay(Duration::from_millis(retry.max_delay_ms))
            .retry_after_cap(Duration::from_millis(retry.retry_after_cap_ms))
            .jitter(retry.jitter);
        let middleware = if retry.observer {
            RetryMiddleware::new(policy).observer_arc(observer.clone())
        } else {
            RetryMiddleware::new(policy)
        };
        builder = builder.middleware(middleware);
    }
    if let Some(timeout_ms) = simulation.timeout_ms {
        builder = builder.middleware(TimeoutMiddleware::new(Duration::from_millis(timeout_ms)));
    }
    if let Some(limit) = simulation.concurrency.and_then(NonZeroUsize::new) {
        builder = builder.middleware(ConcurrencyLimitMiddleware::new(limit));
    }
    if simulation.observer {
        builder = builder.middleware(ObserverMiddleware::from_arc(observer.clone()));
    }
    let client = builder.build()?.client;
    let mut context = CallContext::new();
    if let Some(attempt) = simulation.initial_attempt {
        context.set_attempt(attempt);
    }
    let extension_exercised = if simulation.exercise_extensions {
        let replaced = context
            .extensions_mut()
            .insert::<String>("first".to_owned());
        let missing = context.extensions().get::<u64>().is_none();
        let present = context.extensions().get::<String>().map(String::as_str) == Some("first");
        let removed = context.extensions_mut().remove::<String>().is_some();
        replaced.is_none() && missing && present && removed
    } else {
        false
    };
    if let Some(deadline_ms) = simulation.deadline_ms {
        let deadline = if deadline_ms < 0 {
            Instant::now()
                .checked_sub(Duration::from_millis(deadline_ms.unsigned_abs()))
                .unwrap_or_else(Instant::now)
        } else {
            Instant::now()
                .checked_add(Duration::from_millis(deadline_ms.unsigned_abs()))
                .unwrap_or_else(Instant::now)
        };
        context.set_deadline(deadline);
    }
    if simulation.cancel_before {
        context.cancellation().cancel();
    }
    let result = match simulation.mode {
        SimulationMode::Complete => match client
            .complete_with_context(simulation.request, context)
            .await
        {
            Ok(response) => json!({
                "kind": "complete",
                "text": response.text(),
                "tokens": response.usage.total(),
            }),
            Err(error) => render_error(&error),
        },
        SimulationMode::Stream => match client
            .stream_with_context(simulation.request, context)
            .await
        {
            Ok(mut stream) => {
                let inspected = Arc::new(AtomicUsize::new(0));
                let finalized = Arc::new(AtomicUsize::new(0));
                if simulation.stream_hooks {
                    stream = map_stream(stream, Ok);
                    let count = inspected.clone();
                    stream = inspect_stream(stream, move |_| {
                        count.fetch_add(1, Ordering::Relaxed);
                    });
                    let count = finalized.clone();
                    stream = finalize_stream(stream, move || {
                        count.fetch_add(1, Ordering::Relaxed);
                    });
                }
                let mut events = Vec::new();
                while let Some(item) = stream.next().await {
                    match item {
                        Ok(event) => events.push(event_name(&event)),
                        Err(error) => {
                            events.push(format!("error:{:?}", error.kind()));
                            break;
                        }
                    }
                }
                drop(stream);
                json!({
                    "kind": "stream",
                    "events": events,
                    "inspected": inspected.load(Ordering::Relaxed),
                    "finalized": finalized.load(Ordering::Relaxed),
                })
            }
            Err(error) => render_error(&error),
        },
    };
    let attempts = adapter
        .attempts
        .lock()
        .expect("attempt lock should not be poisoned")
        .clone();
    Ok(json!({
        "kind": "simulation",
        "result": result,
        "adapter_calls": adapter.calls.load(Ordering::Relaxed),
        "attempts": attempts,
        "extension_exercised": extension_exercised,
        "observer": {
            "starts": observer.starts.load(Ordering::Relaxed),
            "completes": observer.completes.load(Ordering::Relaxed),
            "events": observer.events.load(Ordering::Relaxed),
            "errors": observer.errors.load(Ordering::Relaxed),
            "retries": observer.retries.load(Ordering::Relaxed),
            "request_retries": observer.request_retries.load(Ordering::Relaxed),
            "stream_retries": observer.stream_retries.load(Ordering::Relaxed),
        }
    }))
}

fn simulation_catalog() -> &'static str {
    r#"
schema_version = 1

[providers.fixture]
display_name = "Fixture"
adapter = "fixture-adapter"
codec = "fixture-codec"
base_url = "http://127.0.0.1"
auth = { type = "none" }
default_model = "model"

[providers.fixture.models.model]
display_name = "Fixture model"
api_model = "fixture-model"
capabilities = { text = true, images = true, audio = true, documents = true, tools = true, forced_tool_choice = true, structured_output = true, reasoning = true, reasoning_effort_levels = true, caching = true, cache_routing = true, sampling = true }
"#
}

fn fixture_response(text: String) -> Response {
    let mut response = Response::new(ProviderId::new("fixture"), ModelId::new("model"), vec![
        ContentPart::Text { text },
    ]);
    response.usage = TokenCounts {
        input:       1,
        output:      2,
        reasoning:   3,
        cache_read:  4,
        cache_write: 5,
    };
    response
}

fn scripted_stream(items: Vec<StreamItem>) -> ResponseStream {
    Box::pin(unfold(items.into_iter(), |mut items| async move {
        let item = items.next()?;
        let (result, delay_ms) = match item {
            StreamItem::Event { event, delay_ms } => (Ok(*event), delay_ms),
            StreamItem::Error {
                message,
                retry,
                delay_ms,
            } => (Err(script_error(message, retry, None)), delay_ms),
        };
        sleep(Duration::from_millis(delay_ms)).await;
        Some((result, items))
    }))
}

fn script_error(message: String, retry: RetryKind, retry_after_ms: Option<u64>) -> Error {
    let classification = match retry {
        RetryKind::Never => RetryClassification::Never,
        RetryKind::Safe => RetryClassification::Safe,
        RetryKind::After => {
            RetryClassification::after(Duration::from_millis(retry_after_ms.unwrap_or_default()))
        }
    };
    let mut error = Error::new(ErrorKind::Server, message)
        .with_provider(ProviderId::new("fixture"))
        .with_status(503)
        .with_provider_code("fixture_error")
        .with_retry(classification)
        .with_raw_data(json!({ "fixture": true }));
    if let Some(delay) = retry_after_ms {
        error = error.with_provider_retry_after(Duration::from_millis(delay));
    }
    error
}

fn event_name(event: &StreamEvent) -> String {
    let visible = event.is_visible();
    let name = match event {
        StreamEvent::Started { .. } => "started",
        StreamEvent::ContentBlockStart { .. } => "block_start",
        StreamEvent::TextDelta { .. } => "text",
        StreamEvent::ReasoningDelta { .. } => "reasoning",
        StreamEvent::ToolCallDelta { .. } => "tool",
        StreamEvent::ContentBlockEnd { .. } => "block_end",
        StreamEvent::Usage { .. } => "usage",
        StreamEvent::RateLimits { .. } => "rate_limits",
        StreamEvent::Completed { .. } => "completed",
        _ => "unknown",
    };
    format!("{name}:visible={visible}")
}
