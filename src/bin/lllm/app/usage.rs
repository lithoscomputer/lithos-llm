use std::io::Write;
use std::time::Duration;

use lithos_llm::types::{Cost, Response};

use crate::app::output::write_text;
use crate::app::{CliResult, OutputState};

pub(crate) fn write(
    output: &mut impl Write,
    response: &Response,
    elapsed: Duration,
) -> CliResult<OutputState> {
    let input = response
        .usage
        .input
        .saturating_add(response.usage.cache_read)
        .saturating_add(response.usage.cache_write);
    let rendered = format!(
        "{} · {} input · {} output · {} · {}",
        response.model,
        grouped(input),
        grouped(response.usage.billable_output()),
        cost(response.cost),
        duration(elapsed)
    );
    write_text(output, &rendered)
}

pub(crate) fn grouped(value: u64) -> String {
    let digits = value.to_string();
    let mut rendered = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, character) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            rendered.push(',');
        }
        rendered.push(character);
    }
    rendered
}

fn cost(cost: Option<Cost>) -> String {
    let Some(cost) = cost else {
        return "cost unknown".to_owned();
    };
    let dollars = cost.usd_micros / 1_000_000;
    let micros = cost.usd_micros % 1_000_000;
    if micros == 0 {
        format!("${dollars}")
    } else {
        let fraction = format!("{micros:06}");
        format!("${dollars}.{}", fraction.trim_end_matches('0'))
    }
}

pub(crate) fn duration(elapsed: Duration) -> String {
    if elapsed < Duration::from_secs(1) {
        return format!("{}ms", elapsed.as_millis());
    }
    let rendered = format!("{:.1}", elapsed.as_secs_f64());
    format!("{}s", rendered.trim_end_matches(".0"))
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use lithos_llm::types::{Cost, CostSource};

    use super::{cost, duration, grouped};

    #[test]
    fn formats_compact_values_without_floating_point_cost_rounding() {
        assert_eq!(grouped(0), "0");
        assert_eq!(grouped(999), "999");
        assert_eq!(grouped(1_000), "1,000");
        assert_eq!(grouped(1_240), "1,240");
        assert_eq!(grouped(1_234_567), "1,234,567");
        assert_eq!(
            cost(Some(Cost {
                usd_micros: 4_310,
                source:     CostSource::Catalog,
            })),
            "$0.00431"
        );
        assert_eq!(
            cost(Some(Cost {
                usd_micros: 2_000_000,
                source:     CostSource::Provider,
            })),
            "$2"
        );
        assert_eq!(cost(None), "cost unknown");
        assert_eq!(duration(Duration::from_millis(999)), "999ms");
        assert_eq!(duration(Duration::from_secs(1)), "1s");
        assert_eq!(duration(Duration::from_millis(1_800)), "1.8s");
    }
}
