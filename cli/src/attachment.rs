use std::fs;
use std::path::{Path, PathBuf};

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use lithos_llm::types::{AudioContent, ContentPart, DocumentContent, ImageContent, MediaSource};
use url::Url;

use crate::args::AttachmentArg;
use crate::{CliError, CliResult};

pub(crate) fn load(argument: &AttachmentArg, stdin: Option<&[u8]>) -> CliResult<ContentPart> {
    if argument.source == "-" {
        let bytes = stdin.ok_or_else(|| CliError::Input {
            message: "standard input attachment bytes were not available".to_owned(),
        })?;
        let media_type = required_media_type(argument)?;
        return inline_part(bytes, &media_type, None);
    }

    if is_remote_url(&argument.source) {
        return url_part(argument);
    }

    file_part(argument)
}

fn url_part(argument: &AttachmentArg) -> CliResult<ContentPart> {
    let url = Url::parse(&argument.source)
        .map_err(|source| CliError::input_source("attachment URL is invalid", source))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(CliError::Input {
            message: format!(
                "attachment URL `{}` must use HTTP or HTTPS",
                argument.source
            ),
        });
    }
    let media_type = match argument.media_type.as_deref() {
        Some(media_type) => parse_media_type(media_type)?,
        None => infer_from_path(Path::new(url.path())).ok_or_else(|| CliError::Input {
            message: "could not infer the remote attachment media type; use --attachment-type"
                .to_owned(),
        })?,
    };
    let source = MediaSource::url_with_media_type(&argument.source, &media_type);
    let name = url
        .path_segments()
        .and_then(Iterator::last)
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned);
    classified_part(source, &media_type, name)
}

fn file_part(argument: &AttachmentArg) -> CliResult<ContentPart> {
    let path = PathBuf::from(&argument.source);
    let bytes = fs::read(&path).map_err(|source| {
        CliError::input_source(
            format!("could not read attachment `{}`", path.display()),
            source,
        )
    })?;
    let media_type = match argument.media_type.as_deref() {
        Some(media_type) => parse_media_type(media_type)?,
        None => infer::get(&bytes)
            .map(|kind| kind.mime_type().to_owned())
            .or_else(|| infer_from_path(&path))
            .ok_or_else(|| CliError::Input {
                message: format!(
                    "could not infer the media type for `{}`; use --attachment-type",
                    path.display()
                ),
            })?,
    };
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned());
    inline_part(&bytes, &media_type, name)
}

fn inline_part(bytes: &[u8], media_type: &str, name: Option<String>) -> CliResult<ContentPart> {
    let source = MediaSource::base64(STANDARD.encode(bytes), media_type);
    classified_part(source, media_type, name)
}

fn classified_part(
    source: MediaSource,
    media_type: &str,
    name: Option<String>,
) -> CliResult<ContentPart> {
    let parsed = media_type.parse::<mime::Mime>().map_err(|source| {
        CliError::input_source(format!("media type `{media_type}` is invalid"), source)
    })?;
    match parsed.type_().as_str() {
        "image" => Ok(ContentPart::Image(ImageContent::new(source))),
        "audio" => Ok(ContentPart::Audio(AudioContent::new(source))),
        "video" => Err(CliError::Input {
            message: format!("video attachment type `{media_type}` is not supported"),
        }),
        _ => {
            let mut document = DocumentContent::new(source);
            document.name = name;
            Ok(ContentPart::Document(document))
        }
    }
}

fn required_media_type(argument: &AttachmentArg) -> CliResult<String> {
    argument
        .media_type
        .as_deref()
        .ok_or_else(|| CliError::Input {
            message: "a standard-input attachment requires --attachment-type - MEDIA_TYPE"
                .to_owned(),
        })
        .and_then(parse_media_type)
}

fn parse_media_type(raw: &str) -> CliResult<String> {
    raw.parse::<mime::Mime>()
        .map(|media_type| media_type.essence_str().to_owned())
        .map_err(|source| CliError::input_source(format!("media type `{raw}` is invalid"), source))
}

fn infer_from_path(path: &Path) -> Option<String> {
    mime_guess::from_path(path)
        .first_raw()
        .map(ToOwned::to_owned)
}

fn is_remote_url(source: &str) -> bool {
    source
        .get(..7)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("http://"))
        || source
            .get(..8)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("https://"))
}

#[cfg(test)]
mod tests {
    use std::{env, fs, process};

    use lithos_llm::types::{ContentPart, MediaSource};

    use super::load;
    use crate::args::AttachmentArg;

    #[test]
    fn standard_input_is_encoded_as_inline_base64() {
        let part = load(
            &AttachmentArg {
                source:     "-".to_owned(),
                media_type: Some("image/png".to_owned()),
            },
            Some(b"image"),
        )
        .expect("attachment should load");

        let ContentPart::Image(image) = part else {
            panic!("attachment should be an image");
        };
        assert_eq!(image.source.base64_data(), Some("aW1hZ2U="));
        assert_eq!(image.source.media_type(), Some("image/png"));
    }

    #[test]
    fn url_attachments_stay_remote() {
        let part = load(
            &AttachmentArg {
                source:     "https://example.com/report.pdf?download=1".to_owned(),
                media_type: None,
            },
            None,
        )
        .expect("attachment should load");

        let ContentPart::Document(document) = part else {
            panic!("attachment should be a document");
        };
        assert!(matches!(document.source, MediaSource::Url { .. }));
        assert_eq!(document.name.as_deref(), Some("report.pdf"));
    }

    #[test]
    fn file_magic_takes_priority_over_the_extension() {
        let path = env::temp_dir().join(format!("lithos-attachment-magic-{}.png", process::id()));
        fs::write(&path, b"%PDF-1.7\n").expect("fixture should be written");
        let part = load(
            &AttachmentArg {
                source:     path.to_string_lossy().into_owned(),
                media_type: None,
            },
            None,
        )
        .expect("attachment should load");
        fs::remove_file(&path).expect("fixture should be removed");

        let ContentPart::Document(document) = part else {
            panic!("PDF magic should produce a document");
        };
        assert_eq!(document.source.media_type(), Some("application/pdf"));
        assert_eq!(
            document.name.as_deref(),
            path.file_name().and_then(|name| name.to_str())
        );
    }

    #[test]
    fn rejects_video_before_dispatch() {
        let error = load(
            &AttachmentArg {
                source:     "https://example.com/movie.mp4".to_owned(),
                media_type: None,
            },
            None,
        )
        .expect_err("video should be rejected");
        assert!(error.to_string().contains("video"));
    }
}
