use std::io::{self, Write};

use lithos_llm::types::{ContentPart, Request, Response};
use serde::Serialize;

use crate::{CliError, CliResult, OutputState};

#[derive(Serialize)]
struct ResponseEnvelope<'a> {
    version:  u32,
    request:  &'a Request,
    response: &'a Response,
}

pub(crate) fn envelope(request: &Request, response: &Response) -> CliResult<String> {
    serde_json::to_string_pretty(&ResponseEnvelope {
        version: 1,
        request,
        response,
    })
    .map_err(CliError::Json)
}

pub(crate) fn response_text(response: &Response) -> CliResult<String> {
    if let Some(value) = response.content.iter().find_map(|part| match part {
        ContentPart::Json { value } => Some(value),
        _ => None,
    }) {
        serde_json::to_string_pretty(value).map_err(CliError::Json)
    } else {
        Ok(response.text())
    }
}

pub(crate) fn write_text(output: &mut impl Write, text: &str) -> CliResult<OutputState> {
    if let Err(error) = output.write_all(text.as_bytes()) {
        return output_error(error);
    }
    if !text.ends_with('\n')
        && let Err(error) = output.write_all(b"\n")
    {
        return output_error(error);
    }
    if let Err(error) = output.flush() {
        return output_error(error);
    }
    Ok(OutputState::Written)
}

pub(crate) fn write_delta(output: &mut impl Write, text: &str) -> CliResult<OutputState> {
    if let Err(error) = output.write_all(text.as_bytes()) {
        return output_error(error);
    }
    if let Err(error) = output.flush() {
        return output_error(error);
    }
    Ok(OutputState::Written)
}

fn output_error(error: io::Error) -> CliResult<OutputState> {
    if error.kind() == io::ErrorKind::BrokenPipe {
        Ok(OutputState::Closed)
    } else {
        Err(CliError::Output(error))
    }
}

pub(crate) fn extract_first(text: &str) -> &str {
    fenced_blocks(text).next().unwrap_or(text)
}

pub(crate) fn extract_last(text: &str) -> &str {
    fenced_blocks(text).last().unwrap_or(text)
}

fn fenced_blocks(text: &str) -> impl Iterator<Item = &str> {
    let mut blocks = Vec::new();
    let mut offset = 0;
    let mut opening: Option<(char, usize, usize)> = None;
    for line in text.split_inclusive('\n') {
        let line_without_ending = line.strip_suffix('\n').unwrap_or(line);
        let line_without_ending = line_without_ending
            .strip_suffix('\r')
            .unwrap_or(line_without_ending);
        if let Some((character, length, content_start)) = opening {
            if is_closing_fence(line_without_ending, character, length) {
                blocks.push(&text[content_start..offset]);
                opening = None;
            }
        } else if let Some((character, length)) = opening_fence(line_without_ending) {
            opening = Some((character, length, offset + line.len()));
        }
        offset += line.len();
    }
    blocks.into_iter()
}

fn opening_fence(line: &str) -> Option<(char, usize)> {
    let trimmed = line.trim_start();
    let character = trimmed.chars().next()?;
    if !matches!(character, '`' | '~') {
        return None;
    }
    let length = trimmed
        .chars()
        .take_while(|current| *current == character)
        .count();
    (length >= 3).then_some((character, length))
}

fn is_closing_fence(line: &str, character: char, opening_length: usize) -> bool {
    let trimmed = line.trim_start();
    let length = trimmed
        .chars()
        .take_while(|current| *current == character)
        .count();
    length >= opening_length && trimmed[length..].trim().is_empty()
}

#[cfg(test)]
mod tests {
    use super::{extract_first, extract_last};

    #[test]
    fn extracts_the_first_and_last_blocks() {
        let text = "before\n```rust\nfirst();\n```\nmiddle\n~~~~ text\nlast\n~~~~\nafter";
        assert_eq!(extract_first(text), "first();\n");
        assert_eq!(extract_last(text), "last\n");
    }

    #[test]
    fn requires_a_long_enough_closing_fence() {
        let text = "````\ncode\n```\n";
        assert_eq!(extract_first(text), text);
    }

    #[test]
    fn accepts_a_longer_closing_fence() {
        let text = "````\ncode\n`````";
        assert_eq!(extract_first(text), "code\n");
    }

    #[test]
    fn returns_the_whole_response_without_a_complete_block() {
        let text = "plain text";
        assert_eq!(extract_first(text), text);
    }
}
