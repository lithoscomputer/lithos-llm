//! Compares the built-in catalog's Anthropic and OpenAI prices with the
//! providers' published pricing pages.
//!
//! `mise run check:prices` downloads the Markdown form of both pages and runs
//! this with their paths: `check_prices <anthropic.md> <openai.md> [model...]`.
//! Naming models limits the report to those canonical ids.
//!
//! Each catalog rate is the rate an estimate actually charges: the row's
//! pricing selected for the prompt length and speed, as
//! [`Pricing::for_request`] selects it. So a
//! long-context request at a speed is checked against the page's long-context
//! rate for that speed, not only the stored fields. Rates the catalog schema
//! has no place for, such as batch rates and the Ultrafast list price that
//! replaces OpenAI's promotion, are listed as not modelled. The exit status is
//! 1 when a modelled rate is wrong, missing, or priced by the catalog but not
//! by the page, and 2 when a page no longer has the tables this reads.

use std::collections::BTreeMap;
use std::error::Error as StdError;
use std::fmt::Write as _;
use std::io::Write as _;
use std::process::ExitCode;
use std::{env, fs, io};

use lithos_llm::catalog::{Catalog, CatalogModel, Pricing};
use lithos_llm::types::Speed;

type BoxError = Box<dyn StdError>;

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            let _ = writeln!(io::stderr(), "check_prices: {error}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<bool, BoxError> {
    let mut args = env::args().skip(1);
    let (Some(anthropic), Some(openai)) = (args.next(), args.next()) else {
        return Err("usage: check_prices <anthropic.md> <openai.md> [model...]".into());
    };
    let only: Vec<String> = args.collect();
    let anthropic = AnthropicPage::parse(&fs::read_to_string(anthropic)?)?;
    let openai = OpenAiPage::parse(&fs::read_to_string(openai)?)?;
    let catalog = Catalog::builder().with_builtin().build()?;

    let mut report = Report::default();
    for (provider, page) in [
        ("anthropic", Page::Anthropic(&anthropic)),
        ("openai", Page::OpenAi(&openai)),
    ] {
        for model in catalog.provider(provider)?.models() {
            if !only.is_empty() && !only.iter().any(|id| id == model.id().as_str()) {
                continue;
            }
            let rates = match page {
                Page::Anthropic(page) => page.rates(model),
                Page::OpenAi(page) => page.rates(model),
            };
            report.add(
                &format!("{provider}/{}", model.id()),
                model.display_name(),
                &rates,
            );
        }
    }
    let (text, passed) = report.finish();
    io::stdout().write_all(text.as_bytes())?;
    Ok(passed)
}

#[derive(Clone, Copy)]
enum Page<'a> {
    Anthropic(&'a AnthropicPage),
    OpenAi(&'a OpenAiPage),
}

/// One rate as the page and the catalog state it.
#[derive(Debug, PartialEq, Eq)]
struct Rate {
    label:         String,
    page:          Option<u64>,
    catalog:       Option<u64>,
    /// Whether the catalog schema can hold this rate.
    modelled:      bool,
    tokens:        bool,
    /// Whether the page may leave this rate out while the catalog has one:
    /// a speed's long-context rate where the page lists only its short one.
    page_may_omit: bool,
}

impl Rate {
    fn usd(label: impl Into<String>, page: Option<u64>, catalog: Option<u64>) -> Self {
        Self {
            label: label.into(),
            page,
            catalog,
            modelled: true,
            tokens: false,
            page_may_omit: false,
        }
    }

    fn unmodelled(label: impl Into<String>, page: Option<u64>) -> Self {
        Self {
            modelled: false,
            ..Self::usd(label, page, None)
        }
    }

    fn verdict(&self) -> Verdict {
        match (self.modelled, self.page, self.catalog) {
            (false, ..) => Verdict::NotModelled,
            (true, Some(page), Some(catalog)) if page == catalog => Verdict::Match,
            // A page that shows whole cents rounds a sub-cent rate, such as
            // $0.125 shown as $0.13.
            (true, Some(page), Some(catalog))
                if page % 10_000 == 0 && (catalog + 5_000) / 10_000 * 10_000 == page =>
            {
                Verdict::Rounded
            }
            (true, Some(_), Some(_)) => Verdict::Wrong,
            (true, Some(_), None) => Verdict::Missing,
            (true, None, Some(_)) if self.page_may_omit => Verdict::NotOnPage,
            (true, None, Some(_)) => Verdict::CatalogOnly,
            (true, None, None) => Verdict::Match,
        }
    }

    fn show(&self, value: Option<u64>) -> String {
        match value {
            None => "—".to_owned(),
            Some(tokens) if self.tokens => tokens.to_string(),
            Some(micros) => usd(micros),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Verdict {
    Match,
    Rounded,
    NotOnPage,
    Wrong,
    Missing,
    CatalogOnly,
    NotModelled,
}

impl Verdict {
    fn as_str(self) -> &'static str {
        match self {
            Self::Match => "match",
            Self::Rounded => "match (page rounds)",
            Self::NotOnPage => "not on page",
            Self::Wrong => "**WRONG**",
            Self::Missing => "**missing**",
            Self::CatalogOnly => "**catalog only**",
            Self::NotModelled => "not modelled",
        }
    }
}

#[derive(Default)]
struct Report {
    text:   String,
    counts: BTreeMap<Verdict, usize>,
}

impl Report {
    fn add(&mut self, handle: &str, name: &str, rates: &[Rate]) {
        let _ = writeln!(self.text, "### {handle} ({name})\n");
        if rates.is_empty() {
            let _ = writeln!(self.text, "Not on the provider's pricing page.\n");
            return;
        }
        let _ = writeln!(
            self.text,
            "| Rate | Provider | Catalog | Result |\n| --- | --- | --- | --- |"
        );
        for rate in rates {
            let verdict = rate.verdict();
            *self.counts.entry(verdict).or_default() += 1;
            let _ = writeln!(
                self.text,
                "| {} | {} | {} | {} |",
                rate.label,
                rate.show(rate.page),
                rate.show(rate.catalog),
                verdict.as_str()
            );
        }
        self.text.push('\n');
    }

    /// The report with its summary line, and whether every modelled rate
    /// matched.
    fn finish(mut self) -> (String, bool) {
        let mut summary = String::from("Checked:");
        for (verdict, count) in &self.counts {
            let _ = write!(summary, " {count} {},", verdict.as_str().trim_matches('*'));
        }
        let _ = writeln!(self.text, "{}", summary.trim_end_matches(','));
        let passed = [Verdict::Wrong, Verdict::Missing, Verdict::CatalogOnly]
            .iter()
            .all(|verdict| !self.counts.contains_key(verdict));
        (self.text, passed)
    }
}

/// Anthropic's per-million rates for one model at one prompt length.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct AnthropicRow {
    input:      u64,
    write_5m:   u64,
    write_1h:   u64,
    cache_read: u64,
    output:     u64,
}

#[derive(Debug, Default)]
struct AnthropicModel {
    standard: Option<AnthropicRow>,
    /// The prompt length above which `long` applies, and its rates.
    long:     Option<(u64, AnthropicRow)>,
    /// Fast input and output.
    fast:     Option<(u64, u64)>,
    /// Batch input and output.
    batch:    Option<(u64, u64)>,
}

#[derive(Debug, Default)]
struct AnthropicPage {
    models: BTreeMap<String, AnthropicModel>,
}

impl AnthropicPage {
    fn parse(markdown: &str) -> Result<Self, BoxError> {
        let mut page = Self::default();
        let rows = table(
            markdown,
            "## Model pricing",
            "| Model | Base input tokens |",
        )
        .ok_or("Anthropic's model pricing table is gone")?;
        for cells in rows {
            let [name, input, write_5m, write_1h, cache_read, output] = cells.as_slice() else {
                return Err(format!("Anthropic model row has {} cells", cells.len()).into());
            };
            let row = AnthropicRow {
                input:      price(input)?,
                write_5m:   price(write_5m)?,
                write_1h:   price(write_1h)?,
                cache_read: price(cache_read)?,
                output:     price(output)?,
            };
            let model = page.models.entry(model_name(name)).or_default();
            match prompt_threshold(name) {
                Some((threshold, true)) => model.long = Some((threshold, row)),
                Some((_, false)) | None => model.standard = Some(row),
            }
        }
        let fast = table(
            markdown,
            "### Fast mode pricing",
            "| Model | Input | Output |",
        )
        .ok_or("Anthropic's fast mode table is gone")?;
        for cells in fast {
            let [names, input, output] = cells.as_slice() else {
                return Err("Anthropic fast mode row is not three cells".into());
            };
            for name in names.split(" / ") {
                page.models.entry(model_name(name)).or_default().fast =
                    Some((price(input)?, price(output)?));
            }
        }
        let batch = table(
            markdown,
            "### Batch processing",
            "| Model | Batch input | Batch output |",
        )
        .ok_or("Anthropic's batch table is gone")?;
        for cells in batch {
            let [name, input, output] = cells.as_slice() else {
                return Err("Anthropic batch row is not three cells".into());
            };
            if prompt_threshold(name).is_none_or(|(_, long)| !long) {
                page.models.entry(model_name(name)).or_default().batch =
                    Some((price(input)?, price(output)?));
            }
        }
        Ok(page)
    }

    fn rates(&self, model: &CatalogModel) -> Vec<Rate> {
        let Some(page) = self.models.get(model.display_name()) else {
            return Vec::new();
        };
        let pricing = model.pricing().unwrap_or_default();
        let mut rates = Vec::new();
        let short_tokens = page.long.map_or(0, |(threshold, _)| threshold);
        if let Some(row) = page.standard {
            anthropic_rows(
                &mut rates,
                "",
                row,
                Some(&pricing.for_input_tokens(short_tokens)),
            );
        }
        if let Some((threshold, row)) = page.long {
            rates.push(Rate {
                tokens: true,
                ..Rate::usd(
                    "Long-context threshold (prompt tokens)",
                    Some(threshold),
                    pricing.long_context.map(|long| long.above_input_tokens),
                )
            });
            anthropic_rows(
                &mut rates,
                "Long context ",
                row,
                Some(&pricing.for_input_tokens(threshold + 1)),
            );
        }
        let catalog_fast = pricing
            .speed
            .and_then(|speed| speed.fast)
            .map(|_| pricing.for_request(short_tokens, Some(Speed::Fast)));
        if let (Some((input, output)), Some(standard)) = (page.fast, page.standard) {
            // Caching multipliers apply on top of fast pricing.
            let scale = |rate: u64| rate * input / standard.input;
            let row = AnthropicRow {
                input,
                write_5m: scale(standard.write_5m),
                write_1h: scale(standard.write_1h),
                cache_read: scale(standard.cache_read),
                output,
            };
            anthropic_rows(&mut rates, "Fast ", row, catalog_fast.as_ref());
        } else if let Some(fast) = catalog_fast {
            rates.push(Rate::usd(
                "Fast input",
                None,
                fast.input_usd_micros_per_million,
            ));
            rates.push(Rate::usd(
                "Fast output",
                None,
                fast.output_usd_micros_per_million,
            ));
        }
        if let Some((input, output)) = page.batch {
            rates.push(Rate::unmodelled("Batch input", Some(input)));
            rates.push(Rate::unmodelled("Batch output", Some(output)));
        }
        rates
    }
}

fn anthropic_rows(
    rates: &mut Vec<Rate>,
    prefix: &str,
    row: AnthropicRow,
    catalog: Option<&Pricing>,
) {
    let catalog = catalog.copied().unwrap_or_default();
    rates.extend([
        Rate::usd(
            format!("{prefix}input"),
            Some(row.input),
            catalog.input_usd_micros_per_million,
        ),
        Rate::usd(
            format!("{prefix}output"),
            Some(row.output),
            catalog.output_usd_micros_per_million,
        ),
        Rate::usd(
            format!("{prefix}cache read"),
            Some(row.cache_read),
            catalog.cached_input_usd_micros_per_million,
        ),
        Rate::usd(
            format!("{prefix}5m cache write"),
            Some(row.write_5m),
            catalog.cache_write_usd_micros_per_million,
        ),
        Rate::usd(
            format!("{prefix}1h cache write"),
            Some(row.write_1h),
            catalog.long_cache_write_usd_micros_per_million,
        ),
    ]);
}

/// Which catalog rates an OpenAI pricing table corresponds to.
#[derive(Clone, Copy)]
enum CatalogRates {
    Base,
    Speed(Speed),
    /// The catalog has no place for this table's rates.
    None,
}

/// The OpenAI pricing tables this reads, in report order: a label, the
/// headings of its tables, and the catalog rates they correspond to. A later
/// table overrides an earlier one for the models it lists, as the Ultrafast
/// promotion overrides the Ultrafast list price while it runs.
const OPENAI_TIERS: [(&str, &[&str], CatalogRates); 6] = [
    (
        "Standard",
        &["### Standard pricing data"],
        CatalogRates::Base,
    ),
    (
        "Fast",
        &["### Fast pricing data"],
        CatalogRates::Speed(Speed::Fast),
    ),
    (
        "Flex",
        &["### Flex pricing data"],
        CatalogRates::Speed(Speed::Economical),
    ),
    (
        "Ultrafast",
        &[
            "### Ultrafast pricing data",
            "### Ultrafast promotion pricing data",
        ],
        CatalogRates::Speed(Speed::Ultrafast),
    ),
    (
        "Ultrafast list price",
        &["### Ultrafast list pricing data"],
        CatalogRates::None,
    ),
    ("Batch", &["### Batch pricing data"], CatalogRates::None),
];

/// Tables that come and go with a promotion, so their absence is no error.
const OPTIONAL_TABLES: [&str; 2] = [
    "### Ultrafast promotion pricing data",
    "### Ultrafast list pricing data",
];

/// The ten rate columns of an OpenAI flagship table: input, output, cached
/// input, 30-minute write, and 12-hour write, short then long context.
type OpenAiRow = [Option<u64>; 10];

#[derive(Debug, Default)]
struct OpenAiPage {
    /// Prompts above this many input tokens use the long-context rates.
    threshold: u64,
    tiers:     BTreeMap<&'static str, BTreeMap<String, OpenAiRow>>,
}

impl OpenAiPage {
    fn parse(markdown: &str) -> Result<Self, BoxError> {
        let threshold = markdown
            .split("Short context: ≤")
            .nth(1)
            .and_then(|rest| rest.split_once("K input tokens"))
            .and_then(|(thousands, _)| thousands.trim().parse::<u64>().ok())
            .ok_or("OpenAI's short-context threshold is gone")?
            * 1_000;
        let mut tiers = BTreeMap::new();
        for (tier, headings, _) in OPENAI_TIERS {
            let mut models = BTreeMap::new();
            for heading in headings {
                let Some(rows) = table(markdown, heading, "| Model | Short context input |") else {
                    if OPTIONAL_TABLES.contains(heading) {
                        continue;
                    }
                    return Err(format!("OpenAI's {tier} table is gone").into());
                };
                parse_openai_rows(tier, rows, &mut models)?;
            }
            tiers.insert(tier, models);
        }
        Ok(Self { threshold, tiers })
    }
}

/// Reads one OpenAI table's rows into `models`, replacing earlier rows.
fn parse_openai_rows(
    tier: &str,
    rows: Vec<Vec<&str>>,
    models: &mut BTreeMap<String, OpenAiRow>,
) -> Result<(), BoxError> {
    for cells in rows {
        let [name, values @ ..] = cells.as_slice() else {
            continue;
        };
        let mut row = [None; 10];
        if values.len() != row.len() {
            return Err(format!("OpenAI {tier} row {name} has {} rates", values.len()).into());
        }
        for (slot, value) in row.iter_mut().zip(values) {
            *slot = if value.trim() == "-" {
                None
            } else {
                Some(price(value)?)
            };
        }
        models.insert(model_name(name), row);
    }
    Ok(())
}

impl OpenAiPage {
    fn rates(&self, model: &CatalogModel) -> Vec<Rate> {
        let name = model.api_model();
        if !self.tiers.values().any(|models| models.contains_key(name)) {
            return Vec::new();
        }
        let pricing = model.pricing().unwrap_or_default();
        let mut rates = vec![Rate {
            tokens: true,
            ..Rate::usd(
                "Long-context threshold (input tokens)",
                Some(self.threshold),
                pricing.long_context.map(|long| long.above_input_tokens),
            )
        }];
        for (tier, _, catalog_rates) in OPENAI_TIERS {
            let row = self.tiers.get(tier).and_then(|models| models.get(name));
            let speed = match catalog_rates {
                CatalogRates::Base => None,
                CatalogRates::Speed(speed) => Some(speed),
                CatalogRates::None => {
                    if let Some(row) = row {
                        openai_rows(&mut rates, tier, row, None, false);
                    }
                    continue;
                }
            };
            let priced = speed.is_none_or(|speed| {
                pricing
                    .speed
                    .and_then(|rates| rates.for_speed(speed))
                    .is_some()
            });
            if row.is_none() && !priced {
                continue;
            }
            let catalog = priced.then(|| {
                [
                    pricing.for_request(self.threshold, speed),
                    pricing.for_request(self.threshold + 1, speed),
                ]
            });
            openai_rows(
                &mut rates,
                tier,
                row.unwrap_or(&[None; 10]),
                catalog.as_ref(),
                true,
            );
        }
        rates
    }
}

fn openai_rows(
    rates: &mut Vec<Rate>,
    tier: &str,
    row: &OpenAiRow,
    catalog: Option<&[Pricing; 2]>,
    modelled: bool,
) {
    const COLUMNS: [&str; 5] = [
        "input",
        "output",
        "cached input",
        "30m cache write",
        "12h cache write",
    ];
    for (context, prompt) in ["short", "long"].into_iter().enumerate() {
        let pricing = catalog.map(|catalog| catalog[context]);
        for (column, label) in COLUMNS.into_iter().enumerate() {
            let page = row[context * COLUMNS.len() + column];
            let label = format!("{tier} {prompt} {label}");
            let rate = pricing.and_then(|pricing| match column {
                0 => pricing.input_usd_micros_per_million,
                1 => pricing.output_usd_micros_per_million,
                2 => pricing.cached_input_usd_micros_per_million,
                3 => pricing.cache_write_usd_micros_per_million,
                _ => pricing.long_cache_write_usd_micros_per_million,
            });
            rates.push(if modelled {
                Rate {
                    page_may_omit: context == 1
                        && tier != "Standard"
                        && page.is_none()
                        && row[column].is_some(),
                    ..Rate::usd(label, page, rate)
                }
            } else if page.is_some() {
                Rate::unmodelled(label, page)
            } else {
                continue;
            });
        }
    }
}

/// The body rows of the first table whose header starts with `header`, after
/// the line `anchor`, split into trimmed cells.
fn table<'a>(markdown: &'a str, anchor: &str, header: &str) -> Option<Vec<Vec<&'a str>>> {
    let mut lines = markdown.lines().skip_while(|line| line.trim() != anchor);
    lines.find(|line| line.starts_with(header))?;
    lines.next()?; // The `| --- |` separator.
    Some(
        lines
            .take_while(|line| line.starts_with('|'))
            .map(|line| {
                line.trim()
                    .trim_matches('|')
                    .split('|')
                    .map(str::trim)
                    .collect()
            })
            .collect(),
    )
}

/// A model cell without its parenthetical, such as `([retired, ...])` or
/// `(<272K context length)`.
fn model_name(cell: &str) -> String {
    cell.split(" (").next().unwrap_or(cell).trim().to_owned()
}

/// For a row priced by prompt length, such as `(for prompts over 100,000
/// tokens)`: the threshold, and whether this is the row above it.
fn prompt_threshold(cell: &str) -> Option<(u64, bool)> {
    let rest = cell.split_once("(for prompts ")?.1;
    let long = rest.starts_with("over ");
    let digits: String = rest
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit() || *c == ',')
        .filter(char::is_ascii_digit)
        .collect();
    Some((digits.parse().ok()?, long))
}

/// A `$1.25 / MTok`-style cell as micro-dollars, exactly.
fn price(cell: &str) -> Result<u64, BoxError> {
    let amount: String = cell
        .split_once('$')
        .ok_or_else(|| format!("no price in {cell:?}"))?
        .1
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == ',')
        .filter(|c| *c != ',')
        .collect();
    let (dollars, fraction) = amount.split_once('.').unwrap_or((&amount, ""));
    if fraction.len() > 6 {
        return Err(format!("price {cell:?} is finer than a micro-dollar").into());
    }
    let fraction = format!("{fraction:0<6}");
    Ok(dollars.parse::<u64>()? * 1_000_000 + fraction.parse::<u64>()?)
}

fn usd(micros: u64) -> String {
    let digits = format!("{:06}", micros % 1_000_000);
    // Whole cents always, and finer digits only when the price has them.
    let fraction = digits.trim_end_matches('0');
    let fraction = if fraction.len() < 2 {
        &digits[..2]
    } else {
        fraction
    };
    format!("${}.{fraction}", micros / 1_000_000)
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use lithos_llm::catalog::Catalog;

    use super::{AnthropicPage, OpenAiPage, Rate, Verdict, price, usd};

    const ANTHROPIC: &str = "\
## Model pricing

| Model | Base input tokens | 5m cache writes | 1h cache writes | Cache hits and refreshes | Output tokens |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Claude Opus 5.5 | $4 / MTok | $5 / MTok | $8 / MTok | $0.20 / MTok<sup>2</sup> | $20 / MTok |
| Claude Haiku 5.5 (for prompts up to 100,000 tokens) | $0.10 / MTok | $0.125 / MTok | $0.20 / MTok | $0.01 / MTok | $0.50 / MTok |
| Claude Haiku 5.5 (for prompts over 100,000 tokens) | $0.50 / MTok | $0.625 / MTok | $1 / MTok | $0.05 / MTok | $2.75 / MTok |

### Fast mode pricing

| Model | Input | Output |
| ----- | ----- | ----- |
| Claude Opus 5.5 | $8 USD / MTok | $40 USD / MTok |

### Batch processing

| Model | Batch input | Batch output |
| :---- | :---- | :---- |
| Claude Opus 5.5 | $2 / MTok | $10 / MTok |
";

    const OPENAI: &str = "\
### Standard pricing data

| Model | Short context input | Short context output | Short context cached input | Short context 30m write | Short context 12h write | Long context input | Long context output | Long context cached input | Long context 30m write | Long context 12h write |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6.2-sol | $2.00 | $10.00 | $0.10 | $2.50 | $5.00 | $4.00 | $15.00 | $0.20 | $5.00 | $10.00 |

### Batch pricing data

| Model | Short context input | Short context output | Short context cached input | Short context 30m write | Short context 12h write | Long context input | Long context output | Long context cached input | Long context 30m write | Long context 12h write |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

### Flex pricing data

| Model | Short context input | Short context output | Short context cached input | Short context 30m write | Short context 12h write | Long context input | Long context output | Long context cached input | Long context 30m write | Long context 12h write |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

### Fast pricing data

| Model | Short context input | Short context output | Short context cached input | Short context 30m write | Short context 12h write | Long context input | Long context output | Long context cached input | Long context 30m write | Long context 12h write |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6.2-sol | $4.00 | $20.00 | $0.20 | $5.00 | $10.00 | $8.00 | $30.00 | $0.40 | $10.00 | $20.00 |

### Ultrafast pricing data

| Model | Short context input | Short context output | Short context cached input | Short context 30m write | Short context 12h write | Long context input | Long context output | Long context cached input | Long context 30m write | Long context 12h write |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

### Ultrafast promotion pricing data

| Model | Short context input | Short context output | Short context cached input | Short context 30m write | Short context 12h write | Long context input | Long context output | Long context cached input | Long context 30m write | Long context 12h write |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6.2-sol | $6.00 | $30.00 | $0.30 | $7.50 | $15.00 | $12.00 | $45.00 | $0.60 | $15.00 | $30.00 |

Short context: ≤272K input tokens. Long context: >272K input tokens.
";

    /// The verdict on the rate labelled `label`.
    fn verdict(rates: &[Rate], label: &str) -> Option<Verdict> {
        rates
            .iter()
            .find(|rate| rate.label == label)
            .map(Rate::verdict)
    }

    #[test]
    fn prices_parse_exactly_and_print_back() {
        assert_eq!(price("$0.0625").ok(), Some(62_500));
        assert_eq!(price("$12.50 / MTok").ok(), Some(12_500_000));
        assert_eq!(price("$0.20 / MTok<sup>1</sup>").ok(), Some(200_000));
        assert!(price("-").is_err());
        assert_eq!(usd(62_500), "$0.0625");
        assert_eq!(usd(200_000), "$0.20");
        assert_eq!(usd(4_000_000), "$4.00");
    }

    #[test]
    fn anthropic_rates_derive_fast_caching_and_flag_a_wrong_long_rate() -> Result<(), Box<dyn Error>>
    {
        let page = AnthropicPage::parse(ANTHROPIC)?;
        let catalog = Catalog::builder().with_builtin().build()?;
        let anthropic = catalog.provider("anthropic")?;

        let opus = page.rates(anthropic.model("claude-opus-5.5").ok_or("opus 5.5")?);
        assert_eq!(verdict(&opus, "Fast cache read"), Some(Verdict::Match));
        assert_eq!(verdict(&opus, "Fast 5m cache write"), Some(Verdict::Match));
        assert_eq!(verdict(&opus, "1h cache write"), Some(Verdict::Match));

        // The fixture's long-context output is $2.75, not the catalog's $2.50.
        let haiku = page.rates(anthropic.model("claude-haiku-5.5").ok_or("haiku 5.5")?);
        assert_eq!(verdict(&haiku, "Long context output"), Some(Verdict::Wrong));
        assert_eq!(verdict(&haiku, "Long context input"), Some(Verdict::Match));
        assert_eq!(verdict(&haiku, "output"), Some(Verdict::Match));
        Ok(())
    }

    #[test]
    fn openai_rates_check_speed_rates_above_the_long_context_threshold()
    -> Result<(), Box<dyn Error>> {
        let page = OpenAiPage::parse(OPENAI)?;
        assert_eq!(page.threshold, 272_000);
        let catalog = Catalog::builder().with_builtin().build()?;
        let sol = catalog
            .provider("openai")?
            .model("gpt-6.2-sol")
            .ok_or("gpt-6.2-sol")?;
        let rates = page.rates(sol);

        assert_eq!(verdict(&rates, "Standard long input"), Some(Verdict::Match));
        assert_eq!(verdict(&rates, "Fast short input"), Some(Verdict::Match));
        // The promotion table prices GPT-6.2 Sol's Ultrafast tier at $6.
        assert_eq!(
            verdict(&rates, "Ultrafast short input"),
            Some(Verdict::Match)
        );
        // The page charges $8 for fast input above 272K tokens.
        let fast_long = rates.iter().find(|rate| rate.label == "Fast long input");
        assert_eq!(fast_long.and_then(|rate| rate.page), Some(8_000_000));
        Ok(())
    }
}
