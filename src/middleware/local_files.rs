//! Inlines local files before a request reaches a codec.
//!
//! Providers take media as a URL they fetch or as inline base64. A caller on
//! the same machine as the files often has only a path. This middleware reads
//! an image, document, or audio part whose source is a local path and rewrites
//! it to inline base64 with a media type inferred from the extension. It looks
//! inside tool results too.
//!
//! A source is local when its URL starts with `/`, `./`, `~/`, or `file://`.
//! `~/` resolves against `HOME`, or against a lookup the application supplies,
//! and `./` against the process working directory.
//!
//! The paths in a request are not always the application's own: a tool result
//! carries whatever a tool, or a prompt-injected model, put there, and a
//! server's user messages come from its users. So the middleware reads only
//! inside the directories it is given, unless it is built
//! [`unrestricted`](InlineLocalFiles::unrestricted) for trusted local use. It
//! also reads only regular files, up to a size limit: reading `/dev/zero`
//! would never end, a pipe can block forever, and an unbounded read can
//! exhaust memory.
//!
//! A path the middleware refuses — outside the allowed directories, not a
//! regular file, or over the limit — fails the call with
//! [`ErrorKind::InvalidRequest`], so a withheld file is never silently
//! missing. A file that does not exist or cannot be read is dropped with a
//! warning, so the model sees the rest of the message.

use std::path::{Component, Path, PathBuf};
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
    /// The directories a path must resolve inside, or `None` to read any
    /// path.
    roots:          Option<Vec<PathBuf>>,
    home:           Option<EnvLookup>,
    max_file_bytes: u64,
}

impl fmt::Debug for InlineLocalFiles {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("InlineLocalFiles")
            .field("roots", &self.roots)
            .field("max_file_bytes", &self.max_file_bytes)
            .finish_non_exhaustive()
    }
}

/// Why a file was not read.
enum Unread {
    /// The middleware refuses this path; the call fails.
    Refused(String),
    /// The file could not be read; the part is dropped.
    Unreadable(io::Error),
}

impl InlineLocalFiles {
    /// Inlines files that resolve inside one of `directories`.
    ///
    /// A path is checked as written, with `.` and `..` resolved, and again
    /// after symlinks are resolved, so neither `..` nor a symlink reaches
    /// outside. A relative directory resolves against the working directory
    /// when a call runs. `~/` resolves against the process environment's
    /// `HOME`.
    pub fn new<I, P>(directories: I) -> Self
    where
        I: IntoIterator<Item = P>,
        P: Into<PathBuf>,
    {
        Self {
            roots:          Some(directories.into_iter().map(Into::into).collect()),
            home:           None,
            max_file_bytes: DEFAULT_MAX_FILE_BYTES,
        }
    }

    /// Inlines any local file a request names.
    ///
    /// For trusted local use only, such as a command-line tool run by the
    /// owner of the files. Any path in any message or tool result is read,
    /// so a model steered by untrusted content can send any readable file to
    /// the provider.
    pub fn unrestricted() -> Self {
        Self {
            roots:          None,
            home:           None,
            max_file_bytes: DEFAULT_MAX_FILE_BYTES,
        }
    }

    /// Resolves `~/` against `lookup("HOME")` instead of the process
    /// environment.
    #[must_use]
    pub fn with_env_lookup(
        mut self,
        lookup: impl Fn(&str) -> Option<String> + Send + Sync + 'static,
    ) -> Self {
        self.home = Some(Arc::new(lookup));
        self
    }

    /// Sets the largest file this middleware inlines, in bytes.
    ///
    /// A larger file fails the call. The default is 32 MiB. An inlined file
    /// travels base64-encoded, a third larger than on disk.
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

    /// The inline source for `url`, `None` for a file that cannot be read, or
    /// an error for a path the middleware refuses.
    async fn load(&self, url: &str) -> Result<Option<MediaSource>, Error> {
        let path = lexical(Path::new(&self.path_of(url)));
        match self.read(&path).await {
            Ok(bytes) => Ok(Some(MediaSource::base64(
                BASE64_STANDARD.encode(bytes),
                media_type_for_path(&path),
            ))),
            Err(Unread::Refused(reason)) => Err(Error::new(
                ErrorKind::InvalidRequest,
                format!("the local file {} {reason}", path.display()),
            )),
            Err(Unread::Unreadable(error)) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %error,
                    "dropping an unreadable local file"
                );
                Ok(None)
            }
        }
    }

    /// Reads `path` when it lies inside the allowed directories and is a
    /// regular file within the size limit.
    ///
    /// The path is checked by name first, so a path outside is refused
    /// whether or not it exists, and again once symlinks are resolved. The
    /// resolved path is what gets opened. The file type is checked before
    /// opening, because opening a pipe for reading blocks until a writer
    /// appears, and the read is bounded, so a file that grows after the check,
    /// or one that reports no length, still cannot exceed the limit.
    async fn read(&self, path: &Path) -> Result<Vec<u8>, Unread> {
        let outside = || Unread::Refused("is outside the allowed directories".to_owned());
        if let Some(roots) = &self.roots
            && !roots.iter().any(|root| path.starts_with(lexical(root)))
        {
            return Err(outside());
        }
        let resolved = fs::canonicalize(path).await.map_err(Unread::Unreadable)?;
        if let Some(roots) = &self.roots {
            let mut inside = false;
            for root in roots {
                if let Ok(root) = fs::canonicalize(root).await
                    && resolved.starts_with(root)
                {
                    inside = true;
                    break;
                }
            }
            if !inside {
                return Err(outside());
            }
        }

        let too_large = || {
            Unread::Refused(format!(
                "is larger than the {}-byte limit",
                self.max_file_bytes
            ))
        };
        let metadata = fs::metadata(&resolved).await.map_err(Unread::Unreadable)?;
        if !metadata.is_file() {
            return Err(Unread::Refused("is not a regular file".to_owned()));
        }
        if metadata.len() > self.max_file_bytes {
            return Err(too_large());
        }
        let mut bytes = Vec::new();
        fs::File::open(&resolved)
            .await
            .map_err(Unread::Unreadable)?
            .take(self.max_file_bytes.saturating_add(1))
            .read_to_end(&mut bytes)
            .await
            .map_err(Unread::Unreadable)?;
        if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > self.max_file_bytes {
            return Err(too_large());
        }
        Ok(bytes)
    }

    async fn inline_part(&self, part: ContentPart) -> Result<Option<ContentPart>, Error> {
        Ok(match part {
            ContentPart::Image(ImageContent { source, detail }) if is_local(&source) => self
                .load(url_of(&source))
                .await?
                .map(|source| ContentPart::Image(ImageContent { source, detail })),
            ContentPart::Document(DocumentContent { source, name }) if is_local(&source) => self
                .load(url_of(&source))
                .await?
                .map(|source| ContentPart::Document(DocumentContent { source, name })),
            ContentPart::Audio(AudioContent { source }) if is_local(&source) => self
                .load(url_of(&source))
                .await?
                .map(|source| ContentPart::Audio(AudioContent { source })),
            ContentPart::ToolResult(result) if result.content.iter().any(part_is_local) => {
                let mut content = Vec::with_capacity(result.content.len());
                for part in result.content {
                    if let Some(part) = Box::pin(self.inline_part(part)).await? {
                        content.push(part);
                    }
                }
                Some(ContentPart::ToolResult(ToolResult { content, ..result }))
            }
            other => Some(other),
        })
    }

    /// `request` with every local media part inlined.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidRequest`] for a path the middleware refuses.
    pub async fn inline(&self, request: Request) -> Result<Request, Error> {
        let mut messages = Vec::with_capacity(request.messages().len());
        for message in request.messages() {
            let mut content = Vec::with_capacity(message.content().len());
            for part in message.content() {
                if let Some(part) = self.inline_part(part.clone()).await? {
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

/// `path` made absolute against the working directory, with `.` and `..`
/// resolved by name, without touching the filesystem.
fn lexical(path: &Path) -> PathBuf {
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir().unwrap_or_default().join(path)
    };
    let mut normal = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normal.pop();
            }
            other => normal.push(other),
        }
    }
    normal
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
    #[cfg(unix)]
    use std::os::unix::fs::symlink;
    use std::path::{Path, PathBuf};
    #[cfg(unix)]
    use std::process::Command;
    #[cfg(unix)]
    use std::thread;
    use std::{env, process};

    use tokio::fs;

    use super::{InlineLocalFiles, media_type_for_path, needs_inlining};
    use crate::types::{
        ContentPart, DocumentContent, ErrorKind, ImageContent, MediaSource, Message, Request, Role,
        ToolResult,
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

    /// A fresh directory for one test, removed first if a run left it.
    async fn temp_dir(label: &str) -> Result<PathBuf, Box<dyn StdError>> {
        let dir = env::temp_dir().join(format!("{label}-{}", process::id()));
        let _ = fs::remove_dir_all(&dir).await;
        fs::create_dir_all(&dir).await?;
        Ok(dir)
    }

    fn image(url: &str) -> ContentPart {
        ContentPart::Image(ImageContent::new(MediaSource::url(url)))
    }

    fn tool_result(part: ContentPart) -> ContentPart {
        ContentPart::ToolResult(ToolResult {
            tool_call_id: "call_1".to_owned(),
            name:         None,
            content:      vec![part],
            is_error:     false,
        })
    }

    fn url(path: &Path) -> String {
        path.to_string_lossy().into_owned()
    }

    /// What inlining one part does to a call.
    #[derive(Debug, PartialEq)]
    enum Outcome {
        Inlined,
        Dropped,
        Refused(ErrorKind),
    }

    async fn outcome(
        middleware: &InlineLocalFiles,
        part: ContentPart,
    ) -> Result<Outcome, Box<dyn StdError>> {
        Ok(match middleware.inline(request_with(part)?).await {
            Ok(inlined) => match &inlined.messages()[0].content()[1] {
                ContentPart::ToolResult(result) if result.content.is_empty() => Outcome::Dropped,
                _ => Outcome::Inlined,
            },
            Err(error) => Outcome::Refused(error.kind()),
        })
    }

    async fn image_outcome(
        middleware: &InlineLocalFiles,
        url: &str,
    ) -> Result<Outcome, Box<dyn StdError>> {
        match middleware.inline(request_with(image(url))?).await {
            Ok(inlined) if inlined.messages()[0].content().len() == 2 => Ok(Outcome::Inlined),
            Ok(_) => Ok(Outcome::Dropped),
            Err(error) => Ok(Outcome::Refused(error.kind())),
        }
    }

    const REFUSED: Outcome = Outcome::Refused(ErrorKind::InvalidRequest);

    #[tokio::test]
    async fn inlines_files_inside_an_allowed_directory() -> Result<(), Box<dyn StdError>> {
        let dir = temp_dir("lithos-local-allowed").await?;
        let path = dir.join("pixel.png");
        fs::write(&path, b"\x89PNG").await?;
        let middleware = InlineLocalFiles::new([&dir]);

        let request = request_with(image(&url(&path)))?;
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

        let missing = ContentPart::Document(DocumentContent::new(MediaSource::url(url(
            &dir.join("missing.pdf")
        ))));
        let inlined = middleware.inline(request_with(missing)?).await?;
        assert_eq!(
            inlined.messages()[0].content().len(),
            1,
            "a missing file inside an allowed directory is dropped"
        );
        fs::remove_dir_all(&dir).await?;
        Ok(())
    }

    #[tokio::test]
    async fn inlines_inside_tool_results_and_resolves_the_home_directory()
    -> Result<(), Box<dyn StdError>> {
        let dir = temp_dir("lithos-local-home").await?;
        fs::write(dir.join("shot.jpg"), b"\xFF\xD8").await?;
        let home = url(&dir);
        let middleware = InlineLocalFiles::new([&dir])
            .with_env_lookup(move |name| (name == "HOME").then(|| home.clone()));

        let inlined = middleware
            .inline(request_with(tool_result(image("~/shot.jpg")))?)
            .await?;
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

    #[tokio::test]
    async fn a_path_outside_the_allowed_directories_fails_the_call() -> Result<(), Box<dyn StdError>>
    {
        let allowed = temp_dir("lithos-local-inside").await?;
        let outside = temp_dir("lithos-local-outside").await?;
        let secret = outside.join("secret.png");
        fs::write(&secret, b"\x89PNG").await?;
        let middleware = InlineLocalFiles::new([&allowed]);

        assert_eq!(image_outcome(&middleware, &url(&secret)).await?, REFUSED);
        assert_eq!(
            outcome(&middleware, tool_result(image(&url(&secret)))).await?,
            REFUSED,
            "a tool result cannot reach outside either"
        );
        let escape = format!("{}/../{}/secret.png", url(&allowed), outside.display());
        assert_eq!(image_outcome(&middleware, &escape).await?, REFUSED, "`..`");
        assert_eq!(
            image_outcome(&middleware, &url(&outside.join("missing.png"))).await?,
            REFUSED,
            "a path outside is refused whether or not it exists"
        );
        assert_eq!(
            image_outcome(&InlineLocalFiles::unrestricted(), &url(&secret)).await?,
            Outcome::Inlined,
            "an unrestricted middleware reads anywhere"
        );
        fs::remove_dir_all(&allowed).await?;
        fs::remove_dir_all(&outside).await?;
        Ok(())
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_symlink_cannot_lead_outside() -> Result<(), Box<dyn StdError>> {
        let allowed = temp_dir("lithos-local-link-inside").await?;
        let outside = temp_dir("lithos-local-link-outside").await?;
        let secret = outside.join("secret.png");
        fs::write(&secret, b"\x89PNG").await?;
        let link = allowed.join("link.png");
        symlink(&secret, &link)?;

        assert_eq!(
            image_outcome(&InlineLocalFiles::new([&allowed]), &url(&link)).await?,
            REFUSED
        );
        fs::remove_dir_all(&allowed).await?;
        fs::remove_dir_all(&outside).await?;
        Ok(())
    }

    #[tokio::test]
    async fn a_file_over_the_limit_fails_the_call() -> Result<(), Box<dyn StdError>> {
        let dir = temp_dir("lithos-local-limit").await?;
        let path = dir.join("big.png");
        fs::write(&path, [0_u8; 10]).await?;
        let path = url(&path);

        let at_limit = InlineLocalFiles::new([&dir]).max_file_bytes(10);
        assert_eq!(image_outcome(&at_limit, &path).await?, Outcome::Inlined);
        let under = InlineLocalFiles::new([&dir]).max_file_bytes(9);
        assert_eq!(image_outcome(&under, &path).await?, REFUSED);
        fs::remove_dir_all(&dir).await?;
        Ok(())
    }

    #[tokio::test]
    async fn only_regular_files_are_read() -> Result<(), Box<dyn StdError>> {
        let dir = temp_dir("lithos-local-kinds").await?;
        let middleware = InlineLocalFiles::unrestricted().max_file_bytes(1024);

        assert_eq!(
            image_outcome(&middleware, &url(&dir)).await?,
            REFUSED,
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

            assert_eq!(
                image_outcome(&middleware, &url(&fifo)).await?,
                REFUSED,
                "a pipe"
            );
            assert_eq!(
                image_outcome(&middleware, "/dev/zero").await?,
                REFUSED,
                "a device"
            );
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
