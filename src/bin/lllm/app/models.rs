use std::io::Write;

use lithos_llm::Client;
use lithos_llm::catalog::{CatalogModel, CatalogProvider, ModelCapabilities};
use lithos_llm::types::{Message, Request, Role};
use serde::Serialize;

use crate::app::args::{ModelsArgs, ResolveArgs};
use crate::app::output::write_text;
use crate::app::{CliEnvironment, CliError, CliResult, OutputState};

#[derive(Debug, Serialize)]
struct ModelList {
    version:           u32,
    effective_default: String,
    models:            Vec<ModelEntry>,
}

#[derive(Clone, Debug, Serialize)]
struct ModelEntry {
    selector:               String,
    display_name:           String,
    aliases:                Vec<String>,
    capabilities:           ModelCapabilities,
    adapter_compiled:       bool,
    credentials_configured: bool,
    effective_default:      bool,
}

pub(crate) fn select_model(
    client: &Client,
    explicit: Option<&str>,
    terms: &[String],
    environment: &CliEnvironment,
) -> CliResult<String> {
    if let Some(model) = explicit {
        return Ok(model.to_owned());
    }
    if terms.is_empty() {
        return Ok(environment.model()?.unwrap_or("default").to_owned());
    }
    select_query(client, terms, environment)
}

fn select_query(
    client: &Client,
    terms: &[String],
    environment: &CliEnvironment,
) -> CliResult<String> {
    let mut models = entries(client, true, terms, environment, None);
    if models.is_empty() {
        return Err(CliError::Input {
            message: format!(
                "no model with a compiled adapter matches query: {}",
                terms.join(" ")
            ),
        });
    }
    models.sort_by(|left, right| {
        let left_exact = is_exact(left, terms);
        let right_exact = is_exact(right, terms);
        right_exact
            .cmp(&left_exact)
            .then_with(|| left.selector.len().cmp(&right.selector.len()))
            .then_with(|| left.selector.cmp(&right.selector))
    });
    Ok(models.remove(0).selector)
}

pub(crate) fn render(
    client: &Client,
    args: &ModelsArgs,
    environment: &CliEnvironment,
    output: &mut impl Write,
) -> CliResult<OutputState> {
    let default = canonical_route(client, environment.model()?.unwrap_or("default"))?;
    let models = entries(
        client,
        args.adapter_compiled,
        &args.query,
        environment,
        Some(&default),
    );
    let rendered = if args.json {
        serde_json::to_string_pretty(&ModelList {
            version: 2,
            effective_default: default,
            models,
        })
        .map_err(CliError::Json)?
    } else {
        render_table(&models)
    };
    write_text(output, &rendered)
}

pub(crate) fn render_resolution(
    client: &Client,
    args: &ResolveArgs,
    environment: &CliEnvironment,
    output: &mut impl Write,
) -> CliResult<OutputState> {
    let selector = select_model(
        client,
        args.model.as_deref(),
        &args.model_query,
        environment,
    )?;
    write_text(output, &canonical_route(client, &selector)?)
}

fn canonical_route(client: &Client, selector: &str) -> CliResult<String> {
    let request = Request::builder()
        .model(selector)
        .message(Message::text(Role::User, ""))
        .build()
        .map_err(|source| CliError::input_source("could not resolve model", source))?;
    let route = client
        .resolve_route(&request)
        .map_err(|error| CliError::Llm(error.into()))?;
    Ok(format!("{}/{}", route.provider().id(), route.model().id()))
}

fn entries(
    client: &Client,
    only_compiled: bool,
    terms: &[String],
    environment: &CliEnvironment,
    effective_default: Option<&str>,
) -> Vec<ModelEntry> {
    let mut models: Vec<_> = client
        .catalog()
        .providers()
        .flat_map(CatalogProvider::models)
        .map(|model| entry(client, model, environment, effective_default))
        .filter(|model| !only_compiled || model.adapter_compiled)
        .filter(|model| matches_terms(model, terms))
        .collect();
    models.sort_by(|left, right| left.selector.cmp(&right.selector));
    models
}

fn entry(
    client: &Client,
    model: &CatalogModel,
    environment: &CliEnvironment,
    effective_default: Option<&str>,
) -> ModelEntry {
    let selector = format!("{}/{}", model.provider_id(), model.id());
    ModelEntry {
        effective_default: effective_default == Some(selector.as_str()),
        selector,
        display_name: model.display_name().to_owned(),
        aliases: model.aliases().to_vec(),
        capabilities: model.capabilities(),
        adapter_compiled: client.available_providers().contains(model.provider_id()),
        credentials_configured: environment.credentials_configured(model.provider_id()),
    }
}

fn matches_terms(model: &ModelEntry, terms: &[String]) -> bool {
    let searchable = format!(
        "{} {} {} {}",
        model.selector,
        model.selector.split_once('/').map_or("", |(_, id)| id),
        model.display_name,
        model.aliases.join(" ")
    )
    .to_lowercase();
    terms
        .iter()
        .all(|term| searchable.contains(&term.to_lowercase()))
}

fn is_exact(model: &ModelEntry, terms: &[String]) -> bool {
    terms.len() == 1
        && model.selector.split_once('/').is_some_and(|(_, id)| {
            id.eq_ignore_ascii_case(&terms[0])
                || model
                    .aliases
                    .iter()
                    .any(|alias| alias.eq_ignore_ascii_case(&terms[0]))
        })
}

fn render_table(models: &[ModelEntry]) -> String {
    let selector_width = models
        .iter()
        .map(|model| model.selector.len())
        .max()
        .unwrap_or(8)
        .max("SELECTOR".len());
    let name_width = models
        .iter()
        .map(|model| model.display_name.len())
        .max()
        .unwrap_or(4)
        .max("NAME".len());
    let mut rows = vec![format!(
        "{:<7}  {:<selector_width$}  {:<name_width$}  {:<8}  {:<11}  {:<20}  {}",
        "DEFAULT", "SELECTOR", "NAME", "ADAPTER", "CREDENTIALS", "ALIASES", "CAPABILITIES"
    )];
    rows.extend(models.iter().map(|model| {
        format!(
            "{:<7}  {:<selector_width$}  {:<name_width$}  {:<8}  {:<11}  {:<20}  {}",
            if model.effective_default { "yes" } else { "" },
            model.selector,
            model.display_name,
            if model.adapter_compiled { "yes" } else { "no" },
            if model.credentials_configured {
                "yes"
            } else {
                "no"
            },
            model.aliases.join(","),
            capability_names(model.capabilities).join(",")
        )
    }));
    rows.join("\n")
}

fn capability_names(capabilities: ModelCapabilities) -> Vec<&'static str> {
    [
        (capabilities.text, "text"),
        (capabilities.images, "images"),
        (capabilities.audio, "audio"),
        (capabilities.documents, "documents"),
        (capabilities.tools, "tools"),
        (capabilities.structured_output, "structured-output"),
        (capabilities.reasoning, "reasoning"),
        (capabilities.caching, "caching"),
        (capabilities.sampling, "sampling"),
    ]
    .into_iter()
    .filter_map(|(enabled, name)| enabled.then_some(name))
    .collect()
}
