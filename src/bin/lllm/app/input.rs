use std::io::Read;

use lithos_llm::types::ContentPart;

use crate::app::args::AttachmentArg;
use crate::app::{CliError, CliResult, attachment};

#[derive(Debug)]
pub(crate) struct PreparedInput {
    pub(crate) parts: Vec<ContentPart>,
}

pub(crate) fn prepare(
    prompt_arguments: &[String],
    attachments: &[AttachmentArg],
    stdin_is_terminal: bool,
    stdin: &mut impl Read,
) -> CliResult<PreparedInput> {
    let stdin_attachments = attachments
        .iter()
        .filter(|attachment| attachment.source == "-")
        .count();
    if stdin_attachments > 1 {
        return Err(CliError::Input {
            message: "only one attachment can read from standard input".to_owned(),
        });
    }

    let mut stdin_bytes = Vec::new();
    if stdin_attachments == 1 || !stdin_is_terminal {
        stdin
            .read_to_end(&mut stdin_bytes)
            .map_err(|source| CliError::input_source("could not read standard input", source))?;
    }

    let argument_text = prompt_arguments.join(" ");
    let text = if stdin_attachments == 1 || stdin_is_terminal {
        argument_text
    } else {
        let stdin_text = String::from_utf8(stdin_bytes.clone()).map_err(|_| CliError::Input {
            message: "standard-input prompt text must be UTF-8".to_owned(),
        })?;
        match (stdin_text.is_empty(), argument_text.is_empty()) {
            (false, false) => format!("{stdin_text}\n{argument_text}"),
            (false, true) => stdin_text,
            (true, _) => argument_text,
        }
    };

    if text.is_empty() && attachments.is_empty() {
        return Err(CliError::Input {
            message: "provide prompt text or at least one attachment".to_owned(),
        });
    }

    let mut parts = Vec::with_capacity(usize::from(!text.is_empty()) + attachments.len());
    if !text.is_empty() {
        parts.push(ContentPart::Text { text });
    }
    for argument in attachments {
        let bytes = (argument.source == "-").then_some(stdin_bytes.as_slice());
        parts.push(attachment::load(argument, bytes)?);
    }
    Ok(PreparedInput { parts })
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use lithos_llm::types::ContentPart;

    use super::prepare;
    use crate::app::args::AttachmentArg;

    fn text(parts: &[ContentPart]) -> Option<&str> {
        match parts.first() {
            Some(ContentPart::Text { text }) => Some(text),
            _ => None,
        }
    }

    #[test]
    fn prepends_non_terminal_standard_input() {
        let mut stdin = Cursor::new(b"from stdin".to_vec());
        let prepared = prepare(&["from args".to_owned()], &[], false, &mut stdin)
            .expect("input should prepare");
        assert_eq!(text(&prepared.parts), Some("from stdin\nfrom args"));
    }

    #[test]
    fn a_standard_input_attachment_does_not_become_prompt_text() {
        let mut stdin = Cursor::new(b"binary".to_vec());
        let attachments = [AttachmentArg {
            source:     "-".to_owned(),
            media_type: Some("application/pdf".to_owned()),
        }];
        let prepared = prepare(&[], &attachments, false, &mut stdin).expect("input should prepare");
        assert!(matches!(prepared.parts.as_slice(), [
            ContentPart::Document(_)
        ]));
    }

    #[test]
    fn rejects_non_utf8_prompt_input() {
        let mut stdin = Cursor::new(vec![0xff]);
        let error = prepare(&[], &[], false, &mut stdin).expect_err("input should fail");
        assert!(error.to_string().contains("UTF-8"));
    }

    #[test]
    fn rejects_empty_terminal_input() {
        let mut stdin = Cursor::new(Vec::new());
        let error = prepare(&[], &[], true, &mut stdin).expect_err("input should fail");
        assert!(error.to_string().contains("prompt text"));
    }

    #[test]
    fn rejects_two_standard_input_attachments() {
        let mut stdin = Cursor::new(b"bytes".to_vec());
        let attachments = [
            AttachmentArg {
                source:     "-".to_owned(),
                media_type: Some("image/png".to_owned()),
            },
            AttachmentArg {
                source:     "-".to_owned(),
                media_type: Some("application/pdf".to_owned()),
            },
        ];
        let error = prepare(&[], &attachments, false, &mut stdin).expect_err("input should fail");
        assert!(error.to_string().contains("only one attachment"));
    }
}
