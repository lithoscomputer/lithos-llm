use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use lithos_llm::types::ContentPart;

use crate::app::args::AttachmentArg;
use crate::app::{CliError, CliResult, attachment};

#[derive(Debug)]
pub(crate) struct PreparedInput {
    pub(crate) parts: Vec<ContentPart>,
}

pub(crate) fn prepare(
    prompt_arguments: &[String],
    fragments: &[PathBuf],
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

    if text.is_empty() && fragments.is_empty() && attachments.is_empty() {
        return Err(CliError::Input {
            message: "provide prompt text, a fragment, or at least one attachment".to_owned(),
        });
    }

    let mut parts =
        Vec::with_capacity(fragments.len() + usize::from(!text.is_empty()) + attachments.len());
    for path in fragments {
        parts.push(ContentPart::Text {
            text: read_text_file(path, "fragment")?,
        });
    }
    if !text.is_empty() {
        parts.push(ContentPart::Text { text });
    }
    for argument in attachments {
        let bytes = (argument.source == "-").then_some(stdin_bytes.as_slice());
        parts.push(attachment::load(argument, bytes)?);
    }
    Ok(PreparedInput { parts })
}

pub(crate) fn system_text(
    inline: Option<&str>,
    fragments: &[PathBuf],
) -> CliResult<Option<String>> {
    let mut sections = Vec::with_capacity(fragments.len() + usize::from(inline.is_some()));
    if let Some(inline) = inline {
        sections.push(inline.to_owned());
    }
    for path in fragments {
        sections.push(read_text_file(path, "system fragment")?);
    }
    Ok((!sections.is_empty()).then(|| sections.join("\n\n")))
}

fn read_text_file(path: &Path, kind: &str) -> CliResult<String> {
    let display = path.display();
    let source = path.to_string_lossy();
    if source == "-" {
        return Err(CliError::Input {
            message: format!("{kind} `{display}` must be a local file, not standard input"),
        });
    }
    let lowercase = source.to_ascii_lowercase();
    if lowercase.starts_with("http://") || lowercase.starts_with("https://") {
        return Err(CliError::Input {
            message: format!("{kind} `{display}` must be a local file; URLs are not supported"),
        });
    }
    fs::read_to_string(path).map_err(|source| {
        CliError::input_source(
            format!("could not read {kind} `{display}` as UTF-8"),
            source,
        )
    })
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
        let prepared = prepare(&["from args".to_owned()], &[], &[], false, &mut stdin)
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
        let prepared =
            prepare(&[], &[], &attachments, false, &mut stdin).expect("input should prepare");
        assert!(matches!(prepared.parts.as_slice(), [
            ContentPart::Document(_)
        ]));
    }

    #[test]
    fn rejects_non_utf8_prompt_input() {
        let mut stdin = Cursor::new(vec![0xff]);
        let error = prepare(&[], &[], &[], false, &mut stdin).expect_err("input should fail");
        assert!(error.to_string().contains("UTF-8"));
    }

    #[test]
    fn rejects_empty_terminal_input() {
        let mut stdin = Cursor::new(Vec::new());
        let error = prepare(&[], &[], &[], true, &mut stdin).expect_err("input should fail");
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
        let error =
            prepare(&[], &[], &attachments, false, &mut stdin).expect_err("input should fail");
        assert!(error.to_string().contains("only one attachment"));
    }

    #[test]
    fn rejects_remote_fragments() {
        let mut stdin = Cursor::new(Vec::new());
        let error = prepare(
            &[],
            &["https://example.com/prompt.txt".into()],
            &[],
            true,
            &mut stdin,
        )
        .expect_err("input should fail");
        assert!(error.to_string().contains("URLs are not supported"));
    }
}
