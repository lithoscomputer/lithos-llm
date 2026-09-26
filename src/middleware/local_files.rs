//! Inlines local files before a request reaches a codec.
//!
//! Providers take media as a URL they fetch or as inline base64. A caller on
//! the same machine as the files often has only a path. This middleware reads
//! an image, document, or audio part whose source is a local path and rewrites
//! it to inline base64 with a media type inferred from the extension. It looks
//! inside tool results too. A part whose file cannot be read is dropped with a
//! warning, so the model sees the rest of the message rather than the request
//! failing outright.
//!
//! A source is local when its URL starts with `/`, `./`, `~/`, or `file://`.
//! `~/` resolves against `HOME`, or against a lookup the application supplies.
//!
//! Only regular files are read, and only up to a size limit. A device, pipe,
//! or directory, or a file over the limit, is dropped with a warning like an
//! unreadable one: reading `/dev/zero` would never end, a pipe can block
//! forever, and an unbounded read can exhaust memory.

use std::path::Path;
use std::sync::Arc;
use std::{env, fmt, io};

use async_trait::async_trait;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use tokio::fs;
use tokio::io::AsyncReadExt as _;

use super::{Call, Middleware, Next, Output};
use crate::types::{
    AudioContent, ContentPart, DocumentContent, Error, ErrorKind, ImageContent, MediaSource,
    Message, Request, ToolResult,
};

/// Reads the value of one environment variable.
type EnvLookup = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;

/// The largest file inlined by default: 32 MiB.
const DEFAULT_MAX_FILE_BYTES: u64 = 32 * 1024 * 1024;

/// Middleware that inlines local-path media parts as base64.
#[derive(Clone)]
pub struct InlineLocalFiles {
    home:           Option<EnvLookup>,
    max_file_bytes: u64,
}

impl Default for InlineLocalFiles {
    fn default() -> Self {
        Self {
            home:           None,
            max_file_bytes: DEFAULT_MAX_FILE_BYTES,
        }
    }
}

impl fmt::Debug for InlineLocalFiles {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("InlineLocalFiles")
            .field("max_file_bytes", &self.max_file_bytes)
            .finish_non_exhaustive()
    }
}

impl InlineLocalFiles {
    /// Resolves `~/` against the process environment's `HOME`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolves `~/` against `lookup("HOME")` instead of the process
    /// environment.
    #[must_use]
    pub fn with_env_lookup(
        lookup: impl Fn(&str) -> Option<String> + Send + Sync + 'static,
    ) -> Self {
        Self {
            home: Some(Arc::new(lookup)),
            ..Self::default()
        }
    }

    /// Sets the largest file this middleware inlines, in bytes.
    ///
    /// A larger file is dropped with a warning, like an unreadable one. The
    /// default is 32 MiB. An inlined file travels base64-encoded, a third
    /// larger than on disk.
    #[must_use]
    pub fn max_file_bytes(mut self, bytes: u64) -> Self {
        self.max_file_bytes = bytes;
        self
    }

    fn home(&self) -> Option<String> {
        match &self.home {
            Some(lookup) => lookup("HOME"),
            None => env::var("HOME").ok(),
        }
    }

    /// The filesystem path a local source names.
    fn path_of(&self, url: &str) -> String {
        if let Some(rest) = url.strip_prefix("~/") {
            return format!("{}/{rest}", self.home().unwrap_or_else(|| "/".to_owned()));
        }
        url.strip_prefix("file://").unwrap_or(url).to_owned()
    }

    async fn load(&self, url: &str) -> Option<MediaSource> {
        let path = self.path_of(url);
        match self.read(&path).await {
            Ok(bytes) => Some(MediaSource::base64(
                BASE64_STANDARD.encode(bytes),
                media_type_for_path(&path),
            )),
            Err(error) => {
                tracing::warn!(path = %path, error = %error, "dropping an unreadable local file");
                None
            }
        }
    }

    /// Reads `path` when it is a regular file within the size limit.
    ///
    /// The file type is checked before the file is opened, because opening a
    /// pipe for reading blocks until a writer appears. The read is bounded as
    /// well, so a file that grows after the check, or one that reports no
    /// length, still cannot exceed the limit.
    async fn read(&self, path: &str) -> io::Result<Vec<u8>> {
        let too_large = || {
            io::Error::new(
                io::ErrorKind::FileTooLarge,
                format!("larger than the {}-byte limit", self.max_file_bytes),
            )
        };
        let metadata = fs::metadata(path).await?;
        if !metadata.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "not a regular file",
            ));
        }
        if metadata.len() > self.max_file_bytes {
            return Err(too_large());
        }
        let mut bytes = Vec::new();
        fs::File::open(path)
            .await?
            .take(self.max_file_bytes.saturating_add(1))
            .read_to_end(&mut bytes)
            .await?;
        if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > self.max_file_bytes {
            return Err(too_large());
        }
        Ok(bytes)
    }

    async fn inline_part(&self, part: ContentPart) -> Option<ContentPart> {
        match part {
            ContentPart::Image(ImageContent { source, detail }) if is_local(&source) => {
                let source = self.load(url_of(&source)).await?;
                Some(ContentPart::Image(ImageContent { source, detail }))
            }
            ContentPart::Document(DocumentContent { source, name }) if is_local(&source) => {
                let source = self.load(url_of(&source)).await?;
                Some(ContentPart::Document(DocumentContent { source, name }))
            }
            ContentPart::Audio(AudioContent { source }) if is_local(&source) => {
                let source = self.load(url_of(&source)).await?;
                Some(ContentPart::Audio(AudioContent { source }))
            }
            ContentPart::ToolResult(result) if result.content.iter().any(part_is_local) => {
                let mut content = Vec::with_capacity(result.content.len());
                for part in result.content {
                    if let Some(part) = Box::pin(self.inline_part(part)).await {
                        content.push(part);
                    }
                }
                Some(ContentPart::ToolResult(ToolResult { content, ..result }))
            }
            other => Some(other),
        }
    }

    /// `request` with every local media part inlined.
    pub async fn inline(&self, request: Request) -> Result<Request, Error> {
        let mut messages = Vec::with_capacity(request.messages().len());
        for message in request.messages() {
            let mut content = Vec::with_capacity(message.content().len());
            for part in message.content() {
                if let Some(part) = self.inline_part(part.clone()).await {
                    content.push(part);
                }
            }
            let mut rebuilt = Message::new(message.role(), content);
            if let Some(name) = message.name() {
                rebuilt = rebuilt.with_name(name);
            }
            if let Some(id) = message.tool_call_id() {
                rebuilt = rebuilt.with_tool_call_id(id);
            }
            messages.push(rebuilt);
        }
        replace_messages(&request, messages)
    }
}

/// Rebuilds `request` with `messages` in place of its own.
///
/// The request builder appends messages and has no way to clear them, so the
/// swap goes through the request's serde form. Every other field is
/// preserved byte for byte.
fn replace_messages(request: &Request, messages: Vec<Message>) -> Result<Request, Error> {
    let rebuild = |source: serde_json::Error| {
        Error::new(
            ErrorKind::Middleware,
            "the request could not be rebuilt with inlined files",
        )
        .with_source(source)
    };
    let mut value = serde_json::to_value(request).map_err(rebuild)?;
    value["messages"] = serde_json::to_value(messages).map_err(rebuild)?;
    serde_json::from_value(value).map_err(rebuild)
}

fn part_is_local(part: &ContentPart) -> bool {
    match part {
        ContentPart::Image(ImageContent { source, .. })
        | ContentPart::Document(DocumentContent { source, .. })
        | ContentPart::Audio(AudioContent { source }) => is_local(source),
        _ => false,
    }
}

fn url_of(source: &MediaSource) -> &str {
    match source {
        MediaSource::Url { url, .. } => url,
        _ => "",
    }
}

/// Whether a source names a file on this machine rather than a URL a
/// provider could fetch.
fn is_local(source: &MediaSource) -> bool {
    matches!(
        source,
        MediaSource::Url { url, .. }
            if url.starts_with('/')
                || url.starts_with("./")
                || url.starts_with("~/")
                || url.starts_with("file://")
    )
}

fn needs_inlining(request: &Request) -> bool {
    request.messages().iter().any(|message| {
        message.content().iter().any(|part| match part {
            ContentPart::ToolResult(result) => result.content.iter().any(part_is_local),
            part => part_is_local(part),
        })
    })
}

/// The media type for a local path, from its extension;
/// `application/octet-stream` when the extension says nothing.
pub fn media_type_for_path(path: impl AsRef<Path>) -> String {
    mime_guess::from_path(path)
        .first_raw()
        .unwrap_or("application/octet-stream")
        .to_owned()
}

#[async_trait]
impl Middleware for InlineLocalFiles {
    async fn handle(&self, call: Call, next: Next) -> Result<Output, Error> {
        if !needs_inlining(call.request()) {
            return next.run(call).await;
        }
        let inlined = self.inline(call.request().clone()).await?;
        let call = call.map_request(|_| Ok(inlined))?;
        next.run(call).await
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;
    #[cfg(unix)]
    use std::fs::OpenOptions;
    use std::path::PathBuf;
    #[cfg(unix)]
    use std::process::Command;
    #[cfg(unix)]
    use std::thread;
    use std::{env, process};

    use tokio::fs;

    use super::{InlineLocalFiles, media_type_for_path, needs_inlining};
    use crate::types::{
        ContentPart, DocumentContent, ImageContent, MediaSource, Message, Request, Role, ToolResult,
    };

    fn request_with(part: ContentPart) -> Result<Request, Box<dyn StdError>> {
        Ok(Request::builder()
            .model("openai/gpt-5.4")
            .message(Message::new(Role::User, [
                ContentPart::Text {
                    text: "look".to_owned(),
                },
                part,
            ]))
            .build()?)
    }

    fn temp_dir(label: &str) -> PathBuf {
        env::temp_dir().join(format!("{label}-{}", process::id()))
    }

    fn image(url: &str) -> ContentPart {
        ContentPart::Image(ImageContent::new(MediaSource::url(url)))
    }

    #[tokio::test]
    async fn inlines_local_images_and_drops_missing_files() -> Result<(), Box<dyn StdError>> {
        let dir = temp_dir("lithos-local-files");
        fs::create_dir_all(&dir).await?;
        let path = dir.join("pixel.png");
        fs::write(&path, b"\x89PNG").await?;
        let middleware = InlineLocalFiles::new();

        let request = request_with(image(&path.to_string_lossy()))?;
        assert!(needs_inlining(&request));
        let inlined = middleware.inline(request).await?;
        match &inlined.messages()[0].content()[1] {
            ContentPart::Image(image) => {
                assert_eq!(image.source.media_type(), Some("image/png"));
                assert_eq!(image.source.base64_data(), Some("iVBORw=="));
            }
            other => panic!("expected an inlined image, got {other:?}"),
        }
        assert_eq!(inlined.model(), "openai/gpt-5.4", "other fields survive");

        let missing = request_with(ContentPart::Document(DocumentContent::new(
            MediaSource::url("/definitely/missing.pdf"),
        )))?;
        let inlined = middleware.inline(missing).await?;
        assert_eq!(inlined.messages()[0].content().len(), 1);
        fs::remove_dir_all(&dir).await?;
        Ok(())
    }

    #[tokio::test]
    async fn inlines_inside_tool_results_and_resolves_the_home_directory()
    -> Result<(), Box<dyn StdError>> {
        let dir = temp_dir("lithos-local-home");
        fs::create_dir_all(&dir).await?;
        fs::write(dir.join("shot.jpg"), b"\xFF\xD8").await?;
        let home = dir.to_string_lossy().to_string();
        let middleware =
            InlineLocalFiles::with_env_lookup(move |name| (name == "HOME").then(|| home.clone()));

        let request = request_with(ContentPart::ToolResult(ToolResult {
            tool_call_id: "call_1".to_owned(),
            name:         None,
            content:      vec![image("~/shot.jpg")],
            is_error:     false,
        }))?;
        let inlined = middleware.inline(request).await?;
        let ContentPart::ToolResult(result) = &inlined.messages()[0].content()[1] else {
            panic!("expected the tool result to survive");
        };
        let ContentPart::Image(image) = &result.content[0] else {
            panic!("expected an inlined image inside the tool result");
        };
        assert_eq!(image.source.media_type(), Some("image/jpeg"));
        fs::remove_dir_all(&dir).await?;
        Ok(())
    }

    /// Inlines `url` as one image part and reports whether it survived.
    async fn survives(middleware: &InlineLocalFiles, url: &str) -> Result<bool, Box<dyn StdError>> {
        let inlined = middleware.inline(request_with(image(url))?).await?;
        Ok(inlined.messages()[0].content().len() == 2)
    }

    #[tokio::test]
    async fn a_file_over_the_limit_is_dropped() -> Result<(), Box<dyn StdError>> {
        let dir = temp_dir("lithos-local-limit");
        fs::create_dir_all(&dir).await?;
        let path = dir.join("big.png");
        fs::write(&path, [0_u8; 10]).await?;
        let path = path.to_string_lossy();

        assert!(survives(&InlineLocalFiles::new().max_file_bytes(10), &path).await?);
        assert!(!survives(&InlineLocalFiles::new().max_file_bytes(9), &path).await?);
        fs::remove_dir_all(&dir).await?;
        Ok(())
    }

    #[tokio::test]
    async fn only_regular_files_are_read() -> Result<(), Box<dyn StdError>> {
        let dir = temp_dir("lithos-local-kinds");
        fs::create_dir_all(&dir).await?;
        let middleware = InlineLocalFiles::new().max_file_bytes(1024);

        assert!(
            !survives(&middleware, &dir.to_string_lossy()).await?,
            "a directory"
        );
        #[cfg(unix)]
        {
            // A pipe blocks its reader until a writer appears, and a device
            // such as `/dev/zero` never ends; both must be refused unopened.
            let fifo = dir.join("pipe.png");
            let made = Command::new("mkfifo").arg(&fifo).status()?;
            assert!(made.success(), "mkfifo should create the pipe");
            // A writer that opens and closes the pipe at once. Were the pipe
            // opened for reading, this would release that reader to an empty
            // file, and the assertion below would fail instead of hanging.
            // When the pipe is refused unopened, this thread waits until the
            // test process ends.
            let writer = fifo.clone();
            thread::spawn(move || drop(OpenOptions::new().write(true).open(writer)));

            assert!(
                !survives(&middleware, &fifo.to_string_lossy()).await?,
                "a pipe"
            );
            assert!(!survives(&middleware, "/dev/zero").await?, "a device");
        }
        fs::remove_dir_all(&dir).await?;
        Ok(())
    }

    #[test]
    fn remote_urls_and_inline_data_pass_through() -> Result<(), Box<dyn StdError>> {
        assert!(!needs_inlining(&request_with(image(
            "https://example.com/a.png"
        ))?));
        assert!(!needs_inlining(&request_with(ContentPart::Image(
            ImageContent::new(MediaSource::base64("AAAA", "image/png"))
        ))?));
        assert!(needs_inlining(&request_with(image("~/shot.png"))?));
        assert!(needs_inlining(&request_with(image("./shot.png"))?));
        assert!(needs_inlining(&request_with(image(
            "file:///tmp/shot.png"
        ))?));
        Ok(())
    }

    #[test]
    fn media_types_follow_extensions() {
        assert_eq!(media_type_for_path("a.jpg"), "image/jpeg");
        assert_eq!(media_type_for_path("a.pdf"), "application/pdf");
        assert_eq!(media_type_for_path("a.bin"), "application/octet-stream");
    }
}
