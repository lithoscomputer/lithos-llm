use std::io::Write;

use lithos_llm::Client;
use lithos_llm::catalog::{CatalogModel, CatalogProvider, ModelCapabilities};
use serde::Serialize;

use crate::args::ModelsArgs;
use crate::output::write_text;
use crate::{CliError, CliResult, OutputState};

#[derive(Debug, Serialize)]
struct ModelList {
    version: u32,
    models:  Vec<ModelEntry>,
}

#[derive(Clone, Debug, Serialize)]
struct ModelEntry {
    selector:     String,
    display_name: String,
    aliases:      Vec<String>,
    capabilities: ModelCapabilities,
    available:    bool,
}

pub(crate) fn select(client: &Client, terms: &[String]) -> CliResult<String> {
    let mut models = entries(client, true, terms);
    if models.is_empty() {
        return Err(CliError::Input {
            message: format!("no available model matches query: {}", terms.join(" ")),
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
    output: &mut impl Write,
) -> CliResult<OutputState> {
    let models = entries(client, args.available, &args.query);
    let rendered = if args.json {
        serde_json::to_string_pretty(&ModelList { version: 1, models }).map_err(CliError::Json)?
    } else {
        render_table(&models)
    };
    write_text(output, &rendered)
}

fn entries(client: &Client, only_available: bool, terms: &[String]) -> Vec<ModelEntry> {
    let mut models: Vec<_> = client
        .catalog()
        .providers()
        .flat_map(CatalogProvider::models)
        .map(|model| entry(client, model))
        .filter(|model| !only_available || model.available)
        .filter(|model| matches_terms(model, terms))
        .collect();
    models.sort_by(|left, right| left.selector.cmp(&right.selector));
    models
}

fn entry(client: &Client, model: &CatalogModel) -> ModelEntry {
    ModelEntry {
        selector:     format!("{}/{}", model.provider_id(), model.id()),
        display_name: model.display_name().to_owned(),
        aliases:      model.aliases().to_vec(),
        capabilities: model.capabilities(),
        available:    client.available_providers().contains(model.provider_id()),
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
        "{:<selector_width$}  {:<name_width$}  {:<9}  {:<20}  {}",
        "SELECTOR", "NAME", "AVAILABLE", "ALIASES", "CAPABILITIES"
    )];
    rows.extend(models.iter().map(|model| {
        format!(
            "{:<selector_width$}  {:<name_width$}  {:<9}  {:<20}  {}",
            model.selector,
            model.display_name,
            if model.available { "yes" } else { "no" },
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
