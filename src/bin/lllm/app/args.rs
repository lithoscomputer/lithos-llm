use std::ffi::OsString;
use std::fs;
use std::path::PathBuf;
use std::str::FromStr as _;
use std::time::Duration;

use clap::{
    ArgAction, ArgMatches, CommandFactory as _, FromArgMatches as _, Parser, Subcommand, ValueEnum,
};

use crate::app::{CliError, CliResult};

#[derive(Debug, Parser)]
#[command(
    name = "lllm",
    version,
    about = "Send stateless prompts through lithos-llm",
    disable_help_subcommand = true
)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Command,

    /// Log library diagnostics to standard error. RUST_LOG overrides this.
    #[arg(long, global = true)]
    pub(crate) verbose: bool,
}

#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    /// Send one stateless prompt.
    Prompt(Box<PromptArgs>),
    /// Print the canonical route without making a request.
    Resolve(ResolveArgs),
    /// List and search the built-in model catalog.
    Models(ModelsArgs),
}

#[derive(Clone, Debug, Parser)]
pub(crate) struct ResolveArgs {
    /// Use this model selector unchanged.
    #[arg(short = 'm', long)]
    pub(crate) model: Option<String>,

    /// Search compiled models. Every repeated term must match.
    #[arg(short = 'q', long = "model-query", action = ArgAction::Append)]
    pub(crate) model_query: Vec<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct AttachmentArg {
    pub(crate) source:     String,
    pub(crate) media_type: Option<String>,
}

#[derive(Debug)]
pub(crate) struct ParsedCli {
    pub(crate) cli:         Cli,
    pub(crate) attachments: Vec<AttachmentArg>,
}

#[derive(Clone, Debug, Parser)]
pub(crate) struct PromptArgs {
    /// Prompt text. Multiple values are joined with one space.
    #[arg(value_name = "PROMPT", num_args = 0..)]
    pub(crate) prompt: Vec<String>,

    /// Do not stream the response.
    #[arg(long)]
    pub(crate) no_stream: bool,

    /// Use this model selector unchanged.
    #[arg(short = 'm', long)]
    pub(crate) model: Option<String>,

    /// Search compiled models. Every repeated term must match.
    #[arg(short = 'q', long = "model-query", action = ArgAction::Append)]
    pub(crate) model_query: Vec<String>,

    /// Add a system message.
    #[arg(short = 's', long)]
    pub(crate) system: Option<String>,

    #[arg(long)]
    pub(crate) max_output_tokens: Option<u32>,

    #[arg(long)]
    pub(crate) temperature: Option<f32>,

    #[arg(long)]
    pub(crate) top_p: Option<f32>,

    #[arg(long, value_enum)]
    pub(crate) reasoning_effort: Option<ReasoningEffortArg>,

    #[arg(long, value_enum)]
    pub(crate) speed: Option<SpeedArg>,

    /// Call timeout. Use an explicit unit, for example 500ms, 30s, or 2m.
    #[arg(long, value_parser = parse_duration)]
    pub(crate) timeout: Option<Duration>,

    #[arg(long, action = ArgAction::Append)]
    pub(crate) stop: Vec<String>,

    #[arg(long, conflicts_with = "no_cache")]
    pub(crate) cache_key: Option<String>,

    #[arg(long)]
    pub(crate) no_cache: bool,

    #[arg(long, value_parser = parse_key_value, action = ArgAction::Append)]
    pub(crate) metadata: Vec<KeyValue>,

    /// Set a raw option in the selected provider namespace.
    #[arg(short = 'o', long = "option", value_parser = parse_key_value, action = ArgAction::Append)]
    pub(crate) options: Vec<KeyValue>,

    /// Attach a URL or file. Use - to read one attachment from standard input.
    #[arg(short = 'a', long = "attachment", action = ArgAction::Append)]
    pub(crate) attachment: Vec<String>,

    /// Attach a source with an explicit media type.
    #[arg(
        long = "attachment-type",
        alias = "at",
        value_names = ["SOURCE", "MEDIA_TYPE"],
        num_args = 2,
        action = ArgAction::Append
    )]
    pub(crate) attachment_type: Vec<String>,

    /// Emit a versioned JSON request and response envelope.
    #[arg(long, conflicts_with_all = ["extract", "extract_last"])]
    pub(crate) json: bool,

    /// Ask the provider for a JSON object.
    #[arg(long, conflicts_with = "schema")]
    pub(crate) json_object: bool,

    /// Ask for a JSON Schema response. Use JSON or @PATH.
    #[arg(long, conflicts_with = "json_object")]
    pub(crate) schema: Option<String>,

    /// Name used for --schema. Defaults to response.
    #[arg(long, requires = "schema")]
    pub(crate) schema_name: Option<String>,

    /// Print the first complete fenced code block.
    #[arg(short = 'x', long, conflicts_with_all = ["extract_last", "json"])]
    pub(crate) extract: bool,

    /// Print the last complete fenced code block.
    #[arg(long, conflicts_with_all = ["extract", "json"])]
    pub(crate) extract_last: bool,
}

#[derive(Clone, Debug, Parser)]
pub(crate) struct ModelsArgs {
    /// Terms that every listed model must match.
    #[arg(value_name = "QUERY")]
    pub(crate) query: Vec<String>,

    /// Emit the documented JSON model-list shape.
    #[arg(long)]
    pub(crate) json: bool,

    /// Show only models whose provider adapter is compiled into this CLI.
    #[arg(long = "adapter-compiled", alias = "available")]
    pub(crate) adapter_compiled: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum ReasoningEffortArg {
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum SpeedArg {
    Fast,
    Balanced,
    Economical,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KeyValue {
    pub(crate) key:   String,
    pub(crate) value: String,
}

pub(crate) fn parse_from<I, T>(args: I) -> Result<ParsedCli, clap::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let args = normalize_args(args);
    let matches = Cli::command().try_get_matches_from(args)?;
    let attachments = prompt_matches(&matches).map_or_else(Vec::new, collect_attachments);
    let cli = Cli::from_arg_matches(&matches)?;
    Ok(ParsedCli { cli, attachments })
}

fn normalize_args<I, T>(args: I) -> Vec<OsString>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let mut args: Vec<OsString> = args.into_iter().map(Into::into).collect();
    if args.is_empty() {
        args.push(OsString::from("lllm"));
    }
    let insert_prompt = match args.get(1).and_then(|value| value.to_str()) {
        Some("prompt" | "resolve" | "models" | "-h" | "--help" | "-V" | "--version") => false,
        Some(_) | None => true,
    };
    if insert_prompt {
        args.insert(1, OsString::from("prompt"));
    }
    args
}

fn prompt_matches(matches: &ArgMatches) -> Option<&ArgMatches> {
    match matches.subcommand() {
        Some(("prompt", matches)) => Some(matches),
        _ => None,
    }
}

fn collect_attachments(matches: &ArgMatches) -> Vec<AttachmentArg> {
    let mut ordered = Vec::new();
    if let (Some(indices), Some(values)) = (
        matches.indices_of("attachment"),
        matches.get_many::<String>("attachment"),
    ) {
        ordered.extend(indices.zip(values).map(|(index, source)| {
            (index, AttachmentArg {
                source:     source.clone(),
                media_type: None,
            })
        }));
    }
    if let (Some(indices), Some(values)) = (
        matches.indices_of("attachment_type"),
        matches.get_many::<String>("attachment_type"),
    ) {
        let indexed: Vec<_> = indices.zip(values).collect();
        ordered.extend(indexed.chunks_exact(2).map(|pair| {
            (pair[0].0, AttachmentArg {
                source:     pair[0].1.clone(),
                media_type: Some(pair[1].1.clone()),
            })
        }));
    }
    ordered.sort_by_key(|(index, _)| *index);
    ordered
        .into_iter()
        .map(|(_, attachment)| attachment)
        .collect()
}

fn parse_key_value(raw: &str) -> Result<KeyValue, String> {
    let (key, value) = raw
        .split_once('=')
        .ok_or_else(|| "expected KEY=VALUE".to_owned())?;
    if key.trim().is_empty() {
        return Err("KEY must not be empty".to_owned());
    }
    Ok(KeyValue {
        key:   key.to_owned(),
        value: value.to_owned(),
    })
}

fn parse_duration(raw: &str) -> Result<Duration, String> {
    if raw.chars().all(|character| character.is_ascii_digit()) {
        return Err("duration requires a unit such as ms, s, m, or h".to_owned());
    }
    humantime::Duration::from_str(raw)
        .map(Into::into)
        .map_err(|error| error.to_string())
}

pub(crate) fn read_schema(raw: &str) -> CliResult<serde_json::Value> {
    let text = if let Some(path) = raw.strip_prefix('@') {
        fs::read_to_string(PathBuf::from(path)).map_err(|source| {
            CliError::input_source(format!("could not read schema file `{path}`"), source)
        })?
    } else {
        raw.to_owned()
    };
    let schema: serde_json::Value = serde_json::from_str(&text)
        .map_err(|source| CliError::input_source("schema is not valid JSON", source))?;
    if !schema.is_object() {
        return Err(CliError::Input {
            message: "schema must be a JSON object".to_owned(),
        });
    }
    Ok(schema)
}

#[cfg(test)]
mod tests {
    use std::{env, fs, process};

    use super::{Command, parse_from, read_schema};

    #[test]
    fn inserts_the_prompt_command() {
        let parsed = parse_from(["lllm", "hello", "world"]).expect("arguments should parse");
        let Command::Prompt(args) = parsed.cli.command else {
            panic!("implicit command should be prompt");
        };
        assert_eq!(args.prompt, ["hello", "world"]);
    }

    #[test]
    fn double_dash_allows_a_subcommand_as_prompt_text() {
        let parsed = parse_from(["lllm", "--", "models"]).expect("arguments should parse");
        let Command::Prompt(args) = parsed.cli.command else {
            panic!("double dash should select prompt");
        };
        assert_eq!(args.prompt, ["models"]);
    }

    #[test]
    fn preserves_mixed_attachment_order() {
        let parsed = parse_from([
            "lllm",
            "hello",
            "-a",
            "one.png",
            "--at",
            "two",
            "application/pdf",
            "-a",
            "three.mp3",
        ])
        .expect("arguments should parse");
        let sources: Vec<_> = parsed
            .attachments
            .iter()
            .map(|attachment| attachment.source.as_str())
            .collect();
        assert_eq!(sources, ["one.png", "two", "three.mp3"]);
        assert_eq!(
            parsed.attachments[1].media_type.as_deref(),
            Some("application/pdf")
        );
    }

    #[test]
    fn verbose_parses_anywhere_as_a_global_flag() {
        let parsed = parse_from(["lllm", "hello", "--verbose"]).expect("arguments should parse");
        assert!(parsed.cli.verbose);

        let parsed = parse_from(["lllm", "models", "--verbose"]).expect("arguments should parse");
        assert!(parsed.cli.verbose);
    }

    #[test]
    fn double_dash_keeps_verbose_as_prompt_text() {
        let parsed = parse_from(["lllm", "--", "--verbose"]).expect("arguments should parse");
        assert!(!parsed.cli.verbose);
        let Command::Prompt(args) = parsed.cli.command else {
            panic!("double dash should select prompt");
        };
        assert_eq!(args.prompt, ["--verbose"]);
    }

    #[test]
    fn requires_a_duration_unit() {
        let error =
            parse_from(["lllm", "hello", "--timeout", "10"]).expect_err("duration should fail");
        assert!(error.to_string().contains("requires a unit"));
    }

    #[test]
    fn output_modes_conflict() {
        let error = parse_from(["lllm", "hello", "--json", "--extract"])
            .expect_err("output modes should conflict");
        assert!(error.to_string().contains("cannot be used with"));
    }

    #[test]
    fn reads_a_schema_from_a_file() {
        let path = env::temp_dir().join(format!("lithos-schema-{}.json", process::id()));
        fs::write(&path, r#"{"type":"object"}"#).expect("fixture should be written");
        let schema = read_schema(&format!("@{}", path.display())).expect("schema should load");
        fs::remove_file(path).expect("fixture should be removed");
        assert_eq!(schema["type"], "object");
    }

    #[test]
    fn rejects_a_schema_that_is_not_an_object() {
        let error = read_schema("[]").expect_err("schema should fail");
        assert!(error.to_string().contains("JSON object"));
    }
}
