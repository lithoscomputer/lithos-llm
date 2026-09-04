//! The wire-parity harness and the shared canonical request corpus.
//!
//! Three things live here:
//!
//! 1. Request capture and normalization, so a wire snapshot is stable across
//!    runs and machines.
//! 2. Mock mounting and response framing for the three transports the providers
//!    use: JSON, SSE, and the AWS binary event stream.
//! 3. The canonical request corpus, plus the catalog and client construction
//!    every dialect test needs.
//!
//! Helpers are `pub(crate)` rather than `pub` because this is a single test
//! binary; nothing here is part of the library surface.

#![allow(
    dead_code,
    reason = "the per-dialect wire tests that consume these helpers land in a later change"
)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use futures_util::StreamExt as _;
use httpmock::{HttpMockRequest, Method, Mock, MockServer};
use lithos_llm::catalog::Catalog;
use lithos_llm::credentials::{CredentialHeader, Credentials, SecretValue, StaticCredentials};
use lithos_llm::types::{
    AudioContent, ContentPart, DocumentContent, ImageContent, MediaSource, Message,
    ReasoningContent, ResponseFormat, ResponseStream, Role, ToolCall, ToolChoice, ToolDefinition,
    ToolResult,
};
use lithos_llm::{Client, Request};
use serde::Serialize;
use serde_json::{Map, Value, json};

/// The API key every wire test authenticates with.
///
/// It is a constant, and the harness redacts it out of captured headers, so a
/// snapshot never shows a credential in a position a reader could mistake for
/// a real one.
pub(crate) const TEST_API_KEY: &str = "test-key";

/// A capability set that permits every corpus request.
///
/// Wire tests exercise encoding, not capability gating, so the default model
/// in a wire catalog claims everything. A test that needs a capability refused
/// writes its own catalog TOML.
pub(crate) const FULL_CAPABILITIES: &str = "{ text = true, images = true, audio = true, \
     documents = true, tools = true, structured_output = true, reasoning = true, caching = true, \
     cache_breakpoints = true, sampling = true, system_turns = true }";

// ===========================================================================
// Capture and normalization
// ===========================================================================

/// One captured wire request, normalized for snapshot stability.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct WireCapture {
    pub(crate) method:  String,
    pub(crate) path:    String,
    pub(crate) headers: Vec<(String, String)>,
    pub(crate) body:    Value,
}

/// The shared slot a matcher closure writes the captured request into.
pub(crate) type CaptureSlot = Arc<Mutex<Option<WireCapture>>>;

/// Takes the captured request out of a slot.
///
/// # Panics
///
/// Panics when the mock never matched, which means the client sent the request
/// somewhere else or did not send it at all.
pub(crate) fn captured(slot: &CaptureSlot) -> WireCapture {
    slot.lock()
        .expect("the capture slot should not be poisoned")
        .take()
        .expect("the mock should have captured a request")
}

/// Normalizes one received request into a snapshot-stable capture.
///
/// Header names are lowercased and the header list is sorted, which removes
/// the ordering and casing nondeterminism `reqwest` and the mock server would
/// otherwise introduce. Values that change between runs are replaced:
///
/// - `host` — the mock server binds a random port.
/// - `user-agent` — carries the crate version.
/// - `authorization`, `x-api-key`, `x-goog-api-key`, `x-amz-security-token` —
///   credentials. The reference implementation left its test key in the
///   snapshots because the key is a constant. We redact instead, so no fixture
///   in this repository can teach a reader that pasting a key into a snapshot
///   is normal.
/// - `x-amz-date` — a SigV4 timestamp, different on every run.
///
/// `content-length` is deliberately kept verbatim, as the reference does. It
/// makes any change in body size fail visibly, at the cost of churning the
/// header whenever a body edit changes its length by a byte.
fn capture_request(request: &HttpMockRequest) -> WireCapture {
    let mut headers: Vec<(String, String)> = request
        .headers_vec()
        .iter()
        .map(|(name, value)| {
            let name = name.to_ascii_lowercase();
            let value = match name.as_str() {
                "host" => "[host]".to_owned(),
                "user-agent" => "[user-agent]".to_owned(),
                "authorization" | "x-api-key" | "x-goog-api-key" | "x-amz-security-token" => {
                    "[redacted]".to_owned()
                }
                "x-amz-date" => "[amz-date]".to_owned(),
                _ => value.clone(),
            };
            (name, value)
        })
        .collect();
    headers.sort();

    let uri = request.uri();
    let path = match uri.query() {
        Some(query) => format!("{}?{}", uri.path(), query),
        None => uri.path().to_owned(),
    };

    WireCapture {
        method: request.method_str().to_owned(),
        path,
        headers,
        body: parse_body(&request.body_string()),
    }
}

/// Parses a captured body as JSON, treating an empty body as `null`.
fn parse_body(body: &str) -> Value {
    if body.is_empty() {
        return Value::Null;
    }
    serde_json::from_str(body).expect("a captured request body should be JSON")
}

// ===========================================================================
// Mock mounting
// ===========================================================================

/// Mounts a `POST` mock that captures the request and answers with JSON.
pub(crate) fn mount_capture<'server>(
    server: &'server MockServer,
    path: &str,
    response_body: &Value,
) -> (Mock<'server>, CaptureSlot) {
    mount(server, path, "application/json", body_bytes(response_body))
}

/// Mounts a `POST` mock that captures the request and answers with an SSE
/// transcript.
pub(crate) fn mount_capture_sse<'server>(
    server: &'server MockServer,
    path: &str,
    sse_body: &str,
) -> (Mock<'server>, CaptureSlot) {
    mount(
        server,
        path,
        "text/event-stream",
        sse_body.as_bytes().to_vec(),
    )
}

/// Mounts a `POST` mock that captures the request and answers with AWS binary
/// event-stream frames.
///
/// `frames` are already encoded, normally by
/// [`encode_event_stream_frame`]. They are concatenated in order, which is how
/// Bedrock delivers them.
pub(crate) fn mount_capture_event_stream<'server>(
    server: &'server MockServer,
    path: &str,
    frames: &[Vec<u8>],
) -> (Mock<'server>, CaptureSlot) {
    mount(
        server,
        path,
        "application/vnd.amazon.eventstream",
        frames.concat(),
    )
}

fn mount<'server>(
    server: &'server MockServer,
    path: &str,
    content_type: &str,
    body: Vec<u8>,
) -> (Mock<'server>, CaptureSlot) {
    let slot: CaptureSlot = Arc::new(Mutex::new(None));
    let writer = Arc::clone(&slot);
    let path = path.to_owned();
    let content_type = content_type.to_owned();
    let mock = server.mock(move |when, then| {
        when.method(Method::POST)
            .path(path)
            // `is_true` is the capture side-channel. The closure always
            // matches; its real job is to move the whole request out of the
            // mock server, which no combination of ordinary matchers can do.
            .is_true(move |request: &HttpMockRequest| {
                *writer
                    .lock()
                    .expect("the capture slot should not be poisoned") =
                    Some(capture_request(request));
                true
            });
        then.status(200)
            .header("content-type", content_type)
            .body(body);
    });
    (mock, slot)
}

fn body_bytes(value: &Value) -> Vec<u8> {
    serde_json::to_vec(value).expect("a canned response body should serialize")
}

// ===========================================================================
// Response framing
// ===========================================================================

/// Builds an SSE transcript with named events.
///
/// This is the Anthropic Messages and OpenAI Responses framing: each frame
/// carries both an `event:` line and a `data:` line.
pub(crate) fn sse_transcript(frames: &[(&str, &str)]) -> String {
    let mut transcript = String::new();
    for (event, data) in frames {
        transcript.push_str("event: ");
        transcript.push_str(event);
        transcript.push_str("\ndata: ");
        transcript.push_str(data);
        transcript.push_str("\n\n");
    }
    transcript
}

/// Builds an SSE transcript with data-only frames.
///
/// This is the OpenAI Chat Completions and Gemini framing, including the
/// literal `[DONE]` sentinel that the Chat dialect terminates with.
pub(crate) fn sse_data_transcript(frames: &[&str]) -> String {
    let mut transcript = String::new();
    for data in frames {
        transcript.push_str("data: ");
        transcript.push_str(data);
        transcript.push_str("\n\n");
    }
    transcript
}

/// Encodes one AWS `vnd.amazon.eventstream` frame.
///
/// A frame is a 12-byte prelude (total length, header block length, and a
/// CRC32 over those eight bytes), the header block, the payload, and a CRC32
/// over everything before it. Every header is written as a string header,
/// which is the only value type Bedrock uses for `:event-type`,
/// `:message-type`, and `:exception-type`.
///
/// The CRC-32 here is implemented locally rather than with `crc32fast`,
/// because that crate is an optional dependency behind the `bedrock` feature
/// and is therefore not reliably linked into this test target. The algorithm
/// is the same reflected IEEE CRC-32, so the frames this produces satisfy the
/// decoder in `src/transport/event_stream.rs`.
pub(crate) fn encode_event_stream_frame(headers: &[(&str, &str)], payload: &[u8]) -> Vec<u8> {
    let mut block = Vec::new();
    for (name, value) in headers {
        block.push(u8::try_from(name.len()).expect("an event-stream header name should fit a u8"));
        block.extend_from_slice(name.as_bytes());
        block.push(7);
        let length =
            u16::try_from(value.len()).expect("an event-stream header value should fit a u16");
        block.extend_from_slice(&length.to_be_bytes());
        block.extend_from_slice(value.as_bytes());
    }

    let total = 12 + block.len() + payload.len() + 4;
    let mut frame = Vec::with_capacity(total);
    frame.extend_from_slice(
        &u32::try_from(total)
            .expect("an event-stream frame should fit a u32")
            .to_be_bytes(),
    );
    frame.extend_from_slice(
        &u32::try_from(block.len())
            .expect("an event-stream header block should fit a u32")
            .to_be_bytes(),
    );
    frame.extend_from_slice(&crc32(&frame).to_be_bytes());
    frame.extend_from_slice(&block);
    frame.extend_from_slice(payload);
    frame.extend_from_slice(&crc32(&frame).to_be_bytes());
    frame
}

/// Encodes one Bedrock event frame carrying a JSON payload.
pub(crate) fn bedrock_event_frame(event_type: &str, payload: &Value) -> Vec<u8> {
    encode_event_stream_frame(
        &[
            (":message-type", "event"),
            (":event-type", event_type),
            (":content-type", "application/json"),
        ],
        &body_bytes(payload),
    )
}

/// Encodes one Bedrock exception frame, which reports an in-band failure after
/// a successful HTTP status.
pub(crate) fn bedrock_exception_frame(exception_type: &str, payload: &Value) -> Vec<u8> {
    encode_event_stream_frame(
        &[
            (":message-type", "exception"),
            (":exception-type", exception_type),
            (":content-type", "application/json"),
        ],
        &body_bytes(payload),
    )
}

/// The reflected IEEE CRC-32 both event-stream checksums use.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFF_u32;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

// ===========================================================================
// Streams
// ===========================================================================

/// Drives a response stream to completion and serializes every item.
///
/// An `Ok(event)` is serialized verbatim. An `Err(error)` becomes
/// `{"type": "error", "error": <ErrorData>}`, so a snapshot pins the failure
/// semantics — kind, retry classification, provider code — the same way it
/// pins the events, and so [`assert_stream_contract`] can tell a failure from
/// an event. The `error` value is exactly the [`ErrorData`] projection that
/// `Error::data` produces.
///
/// [`ErrorData`]: lithos_llm::types::ErrorData
pub(crate) async fn collect_stream_events(mut stream: ResponseStream) -> Vec<Value> {
    let mut events = Vec::new();
    while let Some(item) = stream.next().await {
        let value = match item {
            Ok(event) => serde_json::to_value(&event).expect("a stream event should serialize"),
            Err(error) => json!({
                "type": "error",
                "error": serde_json::to_value(error.data())
                    .expect("error data should serialize"),
            }),
        };
        events.push(value);
    }
    events
}

/// Asserts the streaming contract every codec must uphold.
///
/// This is deliberately independent of any snapshot: a snapshot records what a
/// codec did, while this records what a codec is allowed to do. Accepting a
/// changed snapshot can never silence it.
///
/// The invariants, from the streaming-content-identity plan and the doc
/// comment on [`StreamEvent`](lithos_llm::types::StreamEvent):
///
/// - Every delta is preceded by a `content_block_start` for the same block and
///   arrives while that block is still open. A typed block only accepts its own
///   delta kind; an `opaque` block accepts any, because provider-native content
///   is not classified.
/// - Every started block is closed by exactly one `content_block_end`.
/// - A block id is never reused inside one stream, so two blocks can never be
///   open under the same id.
/// - `usage` snapshots are cumulative, so no bucket ever decreases.
/// - A successful stream ends with exactly one `completed`, nothing follows it,
///   every block is closed by then, and its response content is the ordered
///   sequence of `content_block_end` parts.
/// - A failed stream emits no `completed` at all.
///
/// # Panics
///
/// Panics with the offending event index whenever an invariant is broken.
pub(crate) fn assert_stream_contract(events: &[Value]) {
    let mut open: BTreeMap<String, String> = BTreeMap::new();
    let mut started: BTreeSet<String> = BTreeSet::new();
    let mut parts: Vec<Value> = Vec::new();
    let mut usage: Option<Value> = None;
    let mut completed: Option<&Value> = None;
    let mut failed = false;

    for (index, event) in events.iter().enumerate() {
        let kind = event_type(event, index);
        assert!(
            completed.is_none(),
            "event {index} is a `{kind}` after the stream already completed"
        );

        match kind {
            "error" => failed = true,
            "content_block_start" => {
                let id = block_id(event, index);
                assert!(
                    started.insert(id.clone()),
                    "event {index} reuses block id `{id}`, which is already used in this stream"
                );
                open.insert(id, block_kind(event, index));
            }
            "text_delta" | "reasoning_delta" | "tool_call_delta" => {
                let id = block_id(event, index);
                let block = open.get(&id).unwrap_or_else(|| {
                    panic!("event {index} is a `{kind}` for block `{id}`, which is not open")
                });
                assert!(
                    block.as_str() == "opaque" || block.as_str() == delta_block_kind(kind),
                    "event {index} is a `{kind}` for block `{id}`, which is a `{block}` block"
                );
            }
            "content_block_end" => {
                let id = block_id(event, index);
                assert!(
                    open.remove(&id).is_some(),
                    "event {index} ends block `{id}`, which is not open"
                );
                parts.push(
                    event
                        .get("part")
                        .unwrap_or_else(|| panic!("event {index} carries no `part`"))
                        .clone(),
                );
            }
            "usage" => {
                let next = event
                    .get("usage")
                    .unwrap_or_else(|| panic!("event {index} carries no `usage`"))
                    .clone();
                if let Some(previous) = &usage {
                    assert_usage_not_decreasing(previous, &next, index);
                }
                usage = Some(next);
            }
            "completed" => completed = Some(event),
            _ => {}
        }
    }

    if failed {
        assert!(
            completed.is_none(),
            "a stream that failed must not emit a `completed` event"
        );
        return;
    }

    let completed = completed.expect("a successful stream must end with one `completed` event");
    assert!(
        open.is_empty(),
        "the stream completed with these blocks still open: {:?}",
        open.keys().collect::<Vec<_>>()
    );
    let content = completed
        .get("response")
        .and_then(|response| response.get("content"))
        .expect("a `completed` event must carry a response with content");
    assert_eq!(
        content,
        &Value::Array(parts),
        "the completed response content must be the ordered `content_block_end` parts"
    );
}

fn event_type(event: &Value, index: usize) -> &str {
    event
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("event {index} has no `type` discriminator"))
}

fn block_id(event: &Value, index: usize) -> String {
    event
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("event {index} has no block `id`"))
        .to_owned()
}

fn block_kind(event: &Value, index: usize) -> String {
    event
        .get("kind")
        .and_then(|kind| kind.get("type"))
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("event {index} has no block `kind`"))
        .to_owned()
}

/// The block kind a delta of this event type belongs to.
fn delta_block_kind(delta: &str) -> &'static str {
    match delta {
        "text_delta" => "text",
        "reasoning_delta" => "reasoning",
        _ => "tool_call",
    }
}

fn assert_usage_not_decreasing(previous: &Value, next: &Value, index: usize) {
    for bucket in ["input", "output", "reasoning", "cache_read", "cache_write"] {
        let before = previous.get(bucket).and_then(Value::as_u64).unwrap_or(0);
        let after = next.get(bucket).and_then(Value::as_u64).unwrap_or(0);
        assert!(
            after >= before,
            "event {index} lowers cumulative usage `{bucket}` from {before} to {after}"
        );
    }
}

// ===========================================================================
// Snapshots
// ===========================================================================

/// Recursively sorts every object's keys, so a snapshot never depends on
/// JSON key order.
///
/// `serde_json` maps sort their keys by default but keep insertion order
/// under the `preserve_order` feature, which any dependency can switch on
/// for the whole test build through feature unification — the twin-openai
/// dev-dependency does. Canonicalizing here makes the rendered snapshot
/// identical either way. A JSON document embedded in a string is not
/// reordered; it renders as the code under test produced it.
pub(crate) fn canonical_json(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let sorted: BTreeMap<String, Value> = map
                .into_iter()
                .map(|(key, value)| (key, canonical_json(value)))
                .collect();
            Value::Object(sorted.into_iter().collect())
        }
        Value::Array(items) => Value::Array(items.into_iter().map(canonical_json).collect()),
        other => other,
    }
}

/// Matches an ISO-8601 timestamp anywhere in a rendered snapshot.
pub(crate) const TIMESTAMP_FILTER: &str =
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})";

/// The crate version as a regular expression, for the `[VERSION]` filter.
pub(crate) fn version_filter() -> String {
    env!("CARGO_PKG_VERSION").replace('.', r"\.")
}

/// Pretty-prints a value and snapshots it with the shared filters.
///
/// Call it as `json_snapshot!(value)` or `crate::json_snapshot!(value)`; both
/// resolve to this macro.
///
/// The macro expands at the call site so `insta` derives the snapshot name
/// from the calling test function. Snapshots land in a `snapshots/` directory
/// beside the calling file and are named
/// `it__wire__<dialect>__<test_fn>.snap`; a second snapshot inside one test
/// gets a `-2` suffix.
///
/// Two filters run over the rendered JSON: ISO-8601 timestamps become
/// `[TIMESTAMP]`, and the crate version becomes `[VERSION]`. There is
/// deliberately no UUID filter. The reference implementation needed one
/// because its Gemini decoder minted a `Uuid::new_v4()` for every tool call;
/// our codecs synthesize deterministic ids instead, so a UUID appearing in a
/// snapshot is a bug to fix rather than noise to scrub.
#[macro_export]
macro_rules! json_snapshot {
    ($value:expr) => {{
        let value = ::serde_json::to_value(&$value)
            .expect("a snapshot value should convert to JSON");
        let rendered = ::serde_json::to_string_pretty(&$crate::support::canonical_json(value))
            .expect("a snapshot value should serialize");
        let version = $crate::support::version_filter();
        let filters: Vec<(&str, &str)> = vec![
            ($crate::support::TIMESTAMP_FILTER, "[TIMESTAMP]"),
            (version.as_str(), "[VERSION]"),
        ];
        ::insta::with_settings!({ filters => filters }, {
            ::insta::assert_snapshot!(rendered);
        });
    }};
}

// ===========================================================================
// Catalogs, credentials, and clients
// ===========================================================================

/// One provider to build a wire-test catalog around.
///
/// The defaults suit the common case: no authentication, an API model equal to
/// the catalog model id, and a model that claims every capability. A test that
/// needs more catalog data than this — pricing, limits, default headers,
/// adapter options — writes its own TOML and calls [`catalog_from_toml`].
pub(crate) struct WireProvider<'a> {
    pub(crate) provider:     &'a str,
    pub(crate) adapter:      &'a str,
    pub(crate) codec:        &'a str,
    pub(crate) model:        &'a str,
    pub(crate) api_model:    &'a str,
    /// The `auth` value as inline TOML, such as `{ type = "bearer" }`.
    pub(crate) auth:         &'a str,
    /// The `capabilities` value as inline TOML.
    pub(crate) capabilities: &'a str,
}

impl<'a> WireProvider<'a> {
    /// Describes a provider with no authentication and every capability.
    pub(crate) fn new(provider: &'a str, adapter: &'a str, codec: &'a str, model: &'a str) -> Self {
        Self {
            provider,
            adapter,
            codec,
            model,
            api_model: model,
            auth: "{ type = \"none\" }",
            capabilities: FULL_CAPABILITIES,
        }
    }

    #[must_use]
    pub(crate) fn with_auth(mut self, auth: &'a str) -> Self {
        self.auth = auth;
        self
    }

    #[must_use]
    pub(crate) fn with_api_model(mut self, api_model: &'a str) -> Self {
        self.api_model = api_model;
        self
    }

    #[must_use]
    pub(crate) fn with_capabilities(mut self, capabilities: &'a str) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// Renders this provider as a complete catalog document.
    ///
    /// A test that needs extra catalog facts can append to the returned TOML
    /// before handing it to [`catalog_from_toml`].
    pub(crate) fn toml(&self, base_url: &str) -> String {
        format!(
            "schema_version = 1\n\n\
             [providers.\"{provider}\"]\n\
             display_name = \"{provider}\"\n\
             adapter = \"{adapter}\"\n\
             codec = \"{codec}\"\n\
             base_url = \"{base_url}\"\n\
             default_model = \"{model}\"\n\
             auth = {auth}\n\n\
             [providers.\"{provider}\".models.\"{model}\"]\n\
             display_name = \"{model}\"\n\
             api_model = \"{api_model}\"\n\
             capabilities = {capabilities}\n",
            provider = self.provider,
            adapter = self.adapter,
            codec = self.codec,
            model = self.model,
            api_model = self.api_model,
            auth = self.auth,
            capabilities = self.capabilities,
        )
    }

    /// Builds a catalog holding only this provider, pointed at `base_url`.
    pub(crate) fn catalog(&self, base_url: &str) -> Catalog {
        catalog_from_toml("wire", &self.toml(base_url))
    }

    /// The request model selector for this provider's default model.
    pub(crate) fn selector(&self) -> String {
        format!("{}/{}", self.provider, self.model)
    }
}

/// Builds a catalog from one named inline TOML layer.
///
/// `name` appears in parse and validation errors, which makes a broken test
/// catalog say which fixture it came from.
///
/// # Panics
///
/// Panics when the TOML does not parse or does not validate.
pub(crate) fn catalog_from_toml(name: &str, source: &str) -> Catalog {
    Catalog::builder()
        .toml_layer(name, source)
        .expect("the test catalog layer should parse")
        .build()
        .expect("the test catalog should validate")
}

/// Credentials that send the test key as a bearer token.
pub(crate) fn bearer_credentials() -> Credentials {
    Credentials::bearer(SecretValue::new(TEST_API_KEY))
}

/// Credentials that send the test key in a named header.
pub(crate) fn header_credentials(name: &str) -> Credentials {
    Credentials::header(CredentialHeader::new(name, SecretValue::new(TEST_API_KEY)))
}

/// Builds a client for one provider in `catalog`.
///
/// The build must produce no [`ProviderBuildIssue`], because an adapter that
/// silently failed to construct leaves no available provider, and a wire test
/// against a client with no providers would fail at route resolution rather
/// than proving anything about the wire.
///
/// [`ProviderBuildIssue`]: lithos_llm::client::ProviderBuildIssue
///
/// # Panics
///
/// Panics when the client cannot be built or when any provider reported a
/// build issue.
pub(crate) fn client_for(catalog: Catalog, provider: &str, credentials: Credentials) -> Client {
    let build = Client::builder()
        .catalog(catalog)
        .credentials(StaticCredentials::new().with(provider, credentials))
        .build()
        .expect("the wire test client should build");
    assert!(
        build.issues.is_empty(),
        "the wire test client reported provider build issues: {:?}",
        build.issues
    );
    build.client
}

// ===========================================================================
// The canonical request corpus
// ===========================================================================
//
// Each constructor returns one canonical `Request` that every dialect pins
// through its own codec. Editing one of these invalidates the pinned wire
// snapshots in EVERY dialect file, so change one only when the contract it
// describes changed, and re-review every snapshot it moves.
//
// `model` is always a request selector such as `"anthropic/claude"`, not a
// provider API model id.

/// One user message and an output cap. The smallest request a provider
/// accepts, and the baseline every other corpus entry is read against.
pub(crate) fn base_request(model: &str) -> Request {
    Request::builder()
        .model(model)
        .user("Hello")
        .max_output_tokens(128)
        .build()
        .expect("the base request should build")
}

/// A system instruction followed by three conversation turns.
pub(crate) fn multi_turn_request(model: &str) -> Request {
    Request::builder()
        .model(model)
        .system("Keep it short.")
        .user("What is the capital of France?")
        .message(Message::text(Role::Assistant, "Paris."))
        .user("And of Spain?")
        .max_output_tokens(128)
        .build()
        .expect("the multi-turn request should build")
}

/// Two function tools, with an optional tool choice.
///
/// Passing each [`ToolChoice`] variant in turn pins all four selection modes
/// against the same tool set.
pub(crate) fn tools_request(model: &str, choice: Option<ToolChoice>) -> Request {
    let mut builder = Request::builder()
        .model(model)
        .system("Use the tools when they help.")
        .user("What is the weather in Paris?")
        .tool(ToolDefinition::function(
            "get_weather",
            "Reads the current weather for a city",
            json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"],
            }),
        ))
        .tool(ToolDefinition::function(
            "get_time",
            "Reads the current time in a time zone",
            json!({
                "type": "object",
                "properties": { "zone": { "type": "string" } },
                "required": ["zone"],
            }),
        ))
        .max_output_tokens(128);
    if let Some(choice) = choice {
        builder = builder.tool_choice(choice);
    }
    builder.build().expect("the tools request should build")
}

/// A complete tool round trip: the assistant calls two tools, and both results
/// come back — one success and one error.
///
/// The error result is the interesting half. It pins how each dialect marks a
/// failed tool result, which several protocols express differently from a
/// successful one.
pub(crate) fn tool_round_trip_request(model: &str) -> Request {
    Request::builder()
        .model(model)
        .user("What is the weather in Paris and Madrid?")
        .tool(ToolDefinition::function(
            "get_weather",
            "Reads the current weather for a city",
            json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"],
            }),
        ))
        .message(Message::new(Role::Assistant, [
            ContentPart::Text {
                text: "Looking both up.".to_owned(),
            },
            ContentPart::ToolCall(ToolCall::function(
                "call_paris",
                "get_weather",
                json!({ "city": "Paris" }),
            )),
            ContentPart::ToolCall(ToolCall::function(
                "call_madrid",
                "get_weather",
                json!({ "city": "Madrid" }),
            )),
        ]))
        .message(Message::new(Role::Tool, [ContentPart::ToolResult(
            ToolResult {
                tool_call_id: "call_paris".to_owned(),
                name:         Some("get_weather".to_owned()),
                content:      vec![ContentPart::Text {
                    text: "18C and clear".to_owned(),
                }],
                is_error:     false,
            },
        )]))
        .message(Message::new(Role::Tool, [ContentPart::ToolResult(
            ToolResult {
                tool_call_id: "call_madrid".to_owned(),
                name:         Some("get_weather".to_owned()),
                content:      vec![ContentPart::Text {
                    text: "the city is unknown".to_owned(),
                }],
                is_error:     true,
            },
        )]))
        .max_output_tokens(128)
        .build()
        .expect("the tool round trip request should build")
}

/// Reasoning replayed as history: a signed reasoning block and a redacted one.
///
/// A dialect that drops the signature, or that sends redacted reasoning text
/// as ordinary reasoning, breaks replay on the provider side. Both cases are
/// in one request so one snapshot covers both.
pub(crate) fn reasoning_round_trip_request(model: &str) -> Request {
    Request::builder()
        .model(model)
        .user("Is 91 prime?")
        .message(Message::new(Role::Assistant, [
            ContentPart::Reasoning(ReasoningContent {
                text:             "91 is 7 times 13.".to_owned(),
                signature:        Some("sig-abc".to_owned()),
                signature_origin: None,
                redacted:         false,
            }),
            ContentPart::Reasoning(ReasoningContent {
                text:             "cmVkYWN0ZWQtcGF5bG9hZA==".to_owned(),
                signature:        None,
                signature_origin: None,
                redacted:         true,
            }),
            ContentPart::Text {
                text: "No, 91 is not prime.".to_owned(),
            },
        ]))
        .user("What about 97?")
        .max_output_tokens(128)
        .build()
        .expect("the reasoning round trip request should build")
}

/// A custom tool definition plus a custom tool call and its result.
///
/// Only the OpenAI Responses codec can encode this. Every other codec must
/// refuse the request with an `unsupported_capability` provider code before it
/// dispatches, never by quietly downgrading the tool to a function tool.
pub(crate) fn custom_tool_request(model: &str) -> Request {
    Request::builder()
        .model(model)
        .user("Apply the patch.")
        .tool(ToolDefinition::custom(
            "apply_patch",
            "Applies a unified patch to the working tree",
            json!({
                "type": "grammar",
                "syntax": "lark",
                "definition": "start: TEXT",
            }),
        ))
        .message(Message::new(Role::Assistant, [ContentPart::ToolCall(
            ToolCall::custom(
                "call_patch",
                "apply_patch",
                "*** Begin Patch\n*** End Patch",
            ),
        )]))
        .message(Message::new(Role::Tool, [ContentPart::ToolResult(
            ToolResult {
                tool_call_id: "call_patch".to_owned(),
                name:         Some("apply_patch".to_owned()),
                content:      vec![ContentPart::Text {
                    text: "applied".to_owned(),
                }],
                is_error:     false,
            },
        )]))
        .max_output_tokens(128)
        .build()
        .expect("the custom tool request should build")
}

/// An image and a document the provider fetches from a URL.
pub(crate) fn url_attachments_request(model: &str) -> Request {
    Request::builder()
        .model(model)
        .message(Message::new(Role::User, [
            ContentPart::Text {
                text: "Describe both attachments.".to_owned(),
            },
            ContentPart::Image(ImageContent {
                source: MediaSource::url("https://example.com/cat.png"),
                detail: Some("high".to_owned()),
            }),
            ContentPart::Document(DocumentContent {
                source: MediaSource::url("https://example.com/report.pdf"),
                name:   Some("report.pdf".to_owned()),
            }),
        ]))
        .max_output_tokens(128)
        .build()
        .expect("the URL attachments request should build")
}

/// An image and a document sent inline as base64.
pub(crate) fn inline_attachments_request(model: &str) -> Request {
    Request::builder()
        .model(model)
        .message(Message::new(Role::User, [
            ContentPart::Text {
                text: "Describe both attachments.".to_owned(),
            },
            ContentPart::Image(ImageContent {
                source: MediaSource::base64("aW1hZ2UtYnl0ZXM=", "image/png"),
                detail: None,
            }),
            ContentPart::Document(DocumentContent {
                source: MediaSource::base64("cGRmLWJ5dGVz", "application/pdf"),
                name:   Some("report.pdf".to_owned()),
            }),
        ]))
        .max_output_tokens(128)
        .build()
        .expect("the inline attachments request should build")
}

/// Inline audio, which only some dialects can carry.
pub(crate) fn audio_request(model: &str) -> Request {
    Request::builder()
        .model(model)
        .message(Message::new(Role::User, [
            ContentPart::Text {
                text: "Transcribe this.".to_owned(),
            },
            ContentPart::Audio(AudioContent::new(MediaSource::base64(
                "YXVkaW8tYnl0ZXM=",
                "audio/wav",
            ))),
        ]))
        .max_output_tokens(128)
        .build()
        .expect("the audio request should build")
}

/// A request pinned to one response format.
///
/// Pass [`ResponseFormat::JsonObject`] and [`json_schema_format`] in turn to
/// pin both structured-output modes.
pub(crate) fn response_format_request(model: &str, format: ResponseFormat) -> Request {
    Request::builder()
        .model(model)
        .user("Give me the city and its population.")
        .response_format(format)
        .max_output_tokens(128)
        .build()
        .expect("the response format request should build")
}

/// The strict JSON schema every dialect pins structured output against.
pub(crate) fn json_schema_format() -> ResponseFormat {
    ResponseFormat::JsonSchema {
        name:   "city_report".to_owned(),
        schema: json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "city":       { "type": "string" },
                "population": { "type": "integer" },
            },
            "required": ["city", "population"],
        }),
    }
}

/// Sampling controls, stop sequences, and one metadata entry.
///
/// The metadata map has exactly one key on purpose. It is a `BTreeMap`, so
/// several keys would still be deterministic here, but a single key keeps the
/// snapshot readable and keeps this corpus entry about sampling.
pub(crate) fn sampling_request(model: &str) -> Request {
    Request::builder()
        .model(model)
        .user("Write one sentence.")
        .temperature(0.7)
        .top_p(0.9)
        .stop_sequences(["END", "STOP"])
        .metadata_entry("tenant", "acme")
        .max_output_tokens(128)
        .build()
        .expect("the sampling request should build")
}

/// Request metadata on its own.
///
/// The Gemini and Bedrock protocols carry no request metadata, so they must
/// warn instead of sending it. This entry isolates that behavior from the
/// sampling controls.
pub(crate) fn metadata_request(model: &str) -> Request {
    Request::builder()
        .model(model)
        .user("Hello")
        .metadata_entry("tenant", "acme")
        .max_output_tokens(128)
        .build()
        .expect("the metadata request should build")
}

/// Raw provider options for two provider namespaces in one request.
///
/// `selected` is the namespace of the provider the request routes to, and
/// `other` is a failover candidate. A codec must merge only the `selected`
/// namespace and must leave no trace of `other` on the wire. Both keys are
/// canonical catalog provider ids, never codec or adapter ids.
///
/// The `selected` namespace also carries the `auto_cache` control key, which
/// a codec consumes and must never send.
pub(crate) fn provider_options_request(model: &str, selected: &str, other: &str) -> Request {
    Request::builder()
        .model(model)
        .user("Hello")
        .provider_options(selected, selected_options())
        .provider_option(
            other,
            "unreachable_option",
            json!("this must never be sent"),
        )
        .max_output_tokens(128)
        .build()
        .expect("the provider options request should build")
}

fn selected_options() -> Map<String, Value> {
    let mut options = Map::new();
    options.insert("auto_cache".to_owned(), json!(false));
    options.insert("service_tier".to_owned(), json!("flex"));
    options.insert("max_output_tokens".to_owned(), json!(256));
    options
}

/// Lossless replay content for one provider namespace.
///
/// `namespace` is the codec's own namespace — `anthropic`, `bedrock`,
/// `gemini`, `openai`, or `openai_compatible` — not a catalog provider id. The
/// request carries an opaque part and a tool call whose original argument text
/// and provider metadata must both survive the round trip. A codec re-emits
/// only its own namespace and silently ignores every other one, so the same
/// request stays sendable after failover.
pub(crate) fn replay_request(model: &str, namespace: &str) -> Request {
    let mut call = ToolCall::function("call_replay", "get_weather", json!({ "city": "Paris" }));
    // Key order and spacing differ from a re-serialization on purpose: a codec
    // that replays `arguments` instead of `raw_arguments` changes these bytes
    // and breaks the provider's prompt cache.
    call.input = lithos_llm::types::ToolInput::Function(
        lithos_llm::types::ToolArguments::from_raw("{\"city\": \"Paris\"}".to_owned()),
    );
    call.provider_metadata
        .insert(namespace.to_owned(), json!({ "item_id": "item_replay" }));
    call.provider_metadata
        .insert("other_provider".to_owned(), json!({ "ignored": true }));

    Request::builder()
        .model(model)
        .user("What is the weather in Paris?")
        .tool(ToolDefinition::function(
            "get_weather",
            "Reads the current weather for a city",
            json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"],
            }),
        ))
        .message(Message::new(Role::Assistant, [
            ContentPart::opaque(
                format!("{namespace}.reasoning"),
                json!({ "id": "rs_replay", "encrypted_content": "opaque-payload" }),
            ),
            ContentPart::opaque("other_provider.reasoning", json!({ "id": "rs_ignored" })),
            ContentPart::ToolCall(call),
        ]))
        .message(Message::new(Role::Tool, [ContentPart::ToolResult(
            ToolResult {
                tool_call_id: "call_replay".to_owned(),
                name:         Some("get_weather".to_owned()),
                content:      vec![ContentPart::Text {
                    text: "18C and clear".to_owned(),
                }],
                is_error:     false,
            },
        )]))
        .max_output_tokens(128)
        .build()
        .expect("the replay request should build")
}
