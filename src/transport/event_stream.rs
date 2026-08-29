//! Decoding for the AWS `vnd.amazon.eventstream` framing.
//!
//! One frame is a 12-byte prelude (total length, headers length, prelude
//! CRC32), a header block, the payload, and a trailing CRC32 over everything
//! before it. Bedrock puts the event name in the `:event-type` header and
//! signals in-band failures with `:message-type` and `:exception-type`, so the
//! header block is parsed rather than skipped.

use std::collections::BTreeMap;
use std::str::from_utf8;

use crc32fast::hash;

use crate::types::{Error, ErrorKind, RetryClassification};

/// The fixed prelude: total length, headers length, and the prelude CRC32.
const PRELUDE_LENGTH: usize = 12;
/// The trailing CRC32 over the prelude, header block, and payload.
const MESSAGE_CRC_LENGTH: usize = 4;
/// An upper bound that keeps a corrupt length from reserving the heap.
const MAX_FRAME_LENGTH: usize = 16 * 1024 * 1024;

/// One decoded event-stream frame.
pub(crate) struct EventStreamFrame {
    /// The string-valued headers of the frame, keyed by header name.
    ///
    /// Headers with another value type are parsed for framing but are not
    /// recorded, because no Bedrock event carries meaning in one.
    pub headers: BTreeMap<String, String>,
    /// The frame payload, normally the JSON body of one stream event.
    pub payload: Vec<u8>,
}

impl EventStreamFrame {
    /// The Bedrock event name, such as `contentBlockDelta`.
    pub(crate) fn event_type(&self) -> Option<&str> {
        self.header(":event-type")
    }

    /// The frame class: `event`, `exception`, or `error`.
    pub(crate) fn message_type(&self) -> Option<&str> {
        self.header(":message-type")
    }

    /// The modeled exception name carried by an exception frame.
    pub(crate) fn exception_type(&self) -> Option<&str> {
        self.header(":exception-type")
    }

    /// The stable code carried by an `error` frame.
    pub(crate) fn error_code(&self) -> Option<&str> {
        self.header(":error-code")
    }

    /// The human-readable message carried by an `error` frame.
    pub(crate) fn error_message(&self) -> Option<&str> {
        self.header(":error-message")
    }

    /// Whether the frame reports a failure instead of an event.
    ///
    /// Bedrock sends these after a successful HTTP status, so a stream can
    /// fail long after its response headers arrived. Two frame classes report
    /// a failure: a modeled `exception`, which names itself in
    /// `:exception-type`, and an unmodeled `error`, which carries
    /// `:error-code` and `:error-message` instead. Both must fail the stream;
    /// an `error` frame in particular has no `:event-type`, so treating it as
    /// an event would silently drop it.
    pub(crate) fn is_failure(&self) -> bool {
        matches!(self.message_type(), Some("exception" | "error"))
    }

    /// The stable code for a failure frame, from either class.
    pub(crate) fn failure_code(&self) -> Option<&str> {
        self.exception_type().or_else(|| self.error_code())
    }

    fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name).map(String::as_str)
    }
}

/// Takes every complete frame from the front of `buffer`.
///
/// Bytes that do not yet form a complete frame stay in `buffer` for the next
/// chunk. A frame that fails a checksum is reported and skipped. A frame whose
/// declared lengths are impossible ends decoding, because the position in the
/// stream can no longer be trusted.
pub(crate) fn extract_frames(buffer: &mut Vec<u8>) -> Vec<Result<EventStreamFrame, Error>> {
    let mut frames = Vec::new();
    while buffer.len() >= PRELUDE_LENGTH {
        // The prelude CRC is checked as soon as the prelude is available,
        // before waiting for the declared frame length: a corrupted length
        // field would otherwise leave the parser waiting for bytes that
        // never arrive, silent until the stream-idle timeout. A bad prelude
        // also means neither length can be trusted, so decoding ends here.
        if hash(&buffer[..8]) != read_u32(&buffer[8..PRELUDE_LENGTH]) {
            buffer.clear();
            // Frame corruption is a transient transport fault, so every
            // decode failure below is retryable — the retry window still
            // closes once visible output has streamed.
            frames.push(Err(Error::new(
                ErrorKind::StreamDecode,
                "Bedrock returned an event-stream frame with an invalid prelude checksum",
            )
            .with_retry(RetryClassification::Safe)));
            break;
        }
        let total_length = read_length(&buffer[0..4]);
        let headers_length = read_length(&buffer[4..8]);
        if !(PRELUDE_LENGTH + MESSAGE_CRC_LENGTH..=MAX_FRAME_LENGTH).contains(&total_length)
            || PRELUDE_LENGTH + headers_length + MESSAGE_CRC_LENGTH > total_length
        {
            buffer.clear();
            frames.push(Err(Error::new(
                ErrorKind::StreamDecode,
                "Bedrock returned an invalid event-stream frame length",
            )
            .with_retry(RetryClassification::Safe)));
            break;
        }
        if buffer.len() < total_length {
            break;
        }

        let frame: Vec<_> = buffer.drain(..total_length).collect();
        let body_length = total_length - MESSAGE_CRC_LENGTH;
        if hash(&frame[..body_length]) != read_u32(&frame[body_length..]) {
            frames.push(Err(Error::new(
                ErrorKind::StreamDecode,
                "Bedrock returned an event-stream frame with an invalid checksum",
            )
            .with_retry(RetryClassification::Safe)));
            continue;
        }

        let payload_start = PRELUDE_LENGTH + headers_length;
        frames.push(
            parse_headers(&frame[PRELUDE_LENGTH..payload_start]).map(|headers| EventStreamFrame {
                headers,
                payload: frame[payload_start..body_length].to_vec(),
            }),
        );
    }
    frames
}

/// Reads a big-endian unsigned integer of at most four bytes.
fn read_u32(bytes: &[u8]) -> u32 {
    bytes
        .iter()
        .fold(0_u32, |value, byte| (value << 8) | u32::from(*byte))
}

/// Reads a big-endian length, saturating where `usize` is narrower.
fn read_length(bytes: &[u8]) -> usize {
    usize::try_from(read_u32(bytes)).unwrap_or(usize::MAX)
}

/// Parses the header block into its string-valued headers.
///
/// Every header is a name length byte, the name, a value type byte, and a
/// value whose width the type decides. A value of another type is stepped over
/// at its declared width so that the rest of the block stays in step.
fn parse_headers(mut block: &[u8]) -> Result<BTreeMap<String, String>, Error> {
    let mut headers = BTreeMap::new();
    while !block.is_empty() {
        let (name_length, rest) = split_first(block)?;
        let (name, rest) = split_at(rest, usize::from(name_length))?;
        let (value_type, rest) = split_first(rest)?;
        let (value, rest) = match value_type {
            0 | 1 => (None, rest),
            2 => (None, split_at(rest, 1)?.1),
            3 => (None, split_at(rest, 2)?.1),
            4 => (None, split_at(rest, 4)?.1),
            5 | 8 => (None, split_at(rest, 8)?.1),
            6 => {
                let (length, rest) = split_at(rest, 2)?;
                (None, split_at(rest, read_length(length))?.1)
            }
            7 => {
                let (length, rest) = split_at(rest, 2)?;
                let (value, rest) = split_at(rest, read_length(length))?;
                (Some(value), rest)
            }
            9 => (None, split_at(rest, 16)?.1),
            other => {
                return Err(Error::new(
                    ErrorKind::StreamDecode,
                    format!("Bedrock sent an event-stream header with unknown value type {other}"),
                )
                .with_retry(RetryClassification::Safe));
            }
        };
        if let Some(value) = value {
            let name = decode_utf8(name, "header name")?;
            headers.insert(name, decode_utf8(value, "header value")?);
        }
        block = rest;
    }
    Ok(headers)
}

fn split_first(block: &[u8]) -> Result<(u8, &[u8]), Error> {
    block
        .split_first()
        .map(|(first, rest)| (*first, rest))
        .ok_or_else(truncated_header_block)
}

fn split_at(block: &[u8], index: usize) -> Result<(&[u8], &[u8]), Error> {
    if index > block.len() {
        return Err(truncated_header_block());
    }
    Ok(block.split_at(index))
}

fn truncated_header_block() -> Error {
    Error::new(
        ErrorKind::StreamDecode,
        "Bedrock returned an event-stream frame with a truncated header block",
    )
    .with_retry(RetryClassification::Safe)
}

fn decode_utf8(bytes: &[u8], part: &str) -> Result<String, Error> {
    from_utf8(bytes).map(str::to_owned).map_err(|source| {
        Error::new(
            ErrorKind::StreamDecode,
            format!("a Bedrock event-stream {part} was not UTF-8"),
        )
        .with_source(source)
        .with_retry(RetryClassification::Safe)
    })
}

#[cfg(test)]
mod tests {
    use crc32fast::hash;

    use super::{EventStreamFrame, PRELUDE_LENGTH, extract_frames};
    use crate::types::{Error, ErrorKind, RetryClassification};

    /// Appends one string header in the AWS event-stream header encoding.
    fn push_string_header(block: &mut Vec<u8>, name: &str, value: &str) {
        push_header_name(block, name);
        block.push(7);
        let length = u16::try_from(value.len()).unwrap_or(u16::MAX);
        block.extend_from_slice(&length.to_be_bytes());
        block.extend_from_slice(value.as_bytes());
    }

    /// Appends one header whose value type is not a string.
    fn push_typed_header(block: &mut Vec<u8>, name: &str, value_type: u8, value: &[u8]) {
        push_header_name(block, name);
        block.push(value_type);
        block.extend_from_slice(value);
    }

    fn push_header_name(block: &mut Vec<u8>, name: &str) {
        block.push(u8::try_from(name.len()).unwrap_or(u8::MAX));
        block.extend_from_slice(name.as_bytes());
    }

    /// Builds one complete frame with both checksums filled in.
    fn frame(header_block: &[u8], payload: &[u8]) -> Vec<u8> {
        let total = 12 + header_block.len() + payload.len() + 4;
        let mut bytes = Vec::with_capacity(total);
        bytes.extend_from_slice(&u32::try_from(total).unwrap_or(u32::MAX).to_be_bytes());
        bytes.extend_from_slice(
            &u32::try_from(header_block.len())
                .unwrap_or(u32::MAX)
                .to_be_bytes(),
        );
        bytes.extend_from_slice(&hash(&bytes).to_be_bytes());
        bytes.extend_from_slice(header_block);
        bytes.extend_from_slice(payload);
        bytes.extend_from_slice(&hash(&bytes).to_be_bytes());
        bytes
    }

    /// Builds an `error` frame, which carries its diagnostics in the headers
    /// and has no `:event-type`.
    fn error_frame(code: &str, message: &str, payload: &[u8]) -> Vec<u8> {
        let mut headers = Vec::new();
        push_string_header(&mut headers, ":message-type", "error");
        push_string_header(&mut headers, ":error-code", code);
        push_string_header(&mut headers, ":error-message", message);
        frame(&headers, payload)
    }

    fn event_frame(event_type: &str, payload: &[u8]) -> Vec<u8> {
        let mut headers = Vec::new();
        push_string_header(&mut headers, ":message-type", "event");
        push_string_header(&mut headers, ":event-type", event_type);
        push_string_header(&mut headers, ":content-type", "application/json");
        frame(&headers, payload)
    }

    fn expect_frame(frames: &[Result<EventStreamFrame, Error>]) -> &EventStreamFrame {
        match frames {
            [Ok(frame)] => frame,
            [Err(error)] => panic!("expected one decoded frame, got the error {error}"),
            other => panic!("expected one decoded frame, got {} results", other.len()),
        }
    }

    fn expect_error(frames: &[Result<EventStreamFrame, Error>]) -> &Error {
        match frames {
            [Err(error)] => error,
            other => panic!("expected one error, got {} results", other.len()),
        }
    }

    #[test]
    fn decodes_a_frame_split_across_chunks() {
        let bytes = event_frame("contentBlockDelta", br#"{"delta":{"text":"hi"}}"#);
        let (head, tail) = bytes.split_at(bytes.len() - 5);

        let mut buffer = head.to_vec();
        assert!(extract_frames(&mut buffer).is_empty());
        assert_eq!(buffer.len(), head.len(), "an incomplete frame is retained");

        buffer.extend_from_slice(tail);
        let frames = extract_frames(&mut buffer);
        assert_eq!(expect_frame(&frames).payload, br#"{"delta":{"text":"hi"}}"#);
        assert!(buffer.is_empty());
    }

    #[test]
    fn reads_string_headers() {
        let mut buffer = event_frame("messageStop", b"{}");
        let frames = extract_frames(&mut buffer);
        let frame = expect_frame(&frames);

        assert_eq!(frame.event_type(), Some("messageStop"));
        assert_eq!(frame.message_type(), Some("event"));
        assert_eq!(
            frame.headers.get(":content-type").map(String::as_str),
            Some("application/json")
        );
        assert!(!frame.is_failure());
    }

    #[test]
    fn steps_over_headers_that_are_not_strings() {
        let mut headers = Vec::new();
        push_typed_header(&mut headers, "bool-true", 0, &[]);
        push_typed_header(&mut headers, "bool-false", 1, &[]);
        push_typed_header(&mut headers, "byte", 2, &[7]);
        push_typed_header(&mut headers, "int16", 3, &[0, 7]);
        push_typed_header(&mut headers, "int32", 4, &[0, 0, 0, 7]);
        push_typed_header(&mut headers, "int64", 5, &[0; 8]);
        push_typed_header(&mut headers, "bytes", 6, &[0, 3, 1, 2, 3]);
        push_string_header(&mut headers, ":event-type", "metadata");
        push_typed_header(&mut headers, "timestamp", 8, &[0; 8]);
        push_typed_header(&mut headers, "uuid", 9, &[0; 16]);
        push_string_header(&mut headers, ":message-type", "event");

        let mut buffer = frame(&headers, b"{}");
        let frames = extract_frames(&mut buffer);
        let frame = expect_frame(&frames);

        assert_eq!(frame.event_type(), Some("metadata"));
        assert_eq!(frame.message_type(), Some("event"));
        assert_eq!(frame.headers.len(), 2, "only string headers are recorded");
        assert_eq!(frame.payload, b"{}");
    }

    #[test]
    fn reports_a_prelude_checksum_mismatch() {
        let mut buffer = event_frame("messageStop", b"{}");
        buffer[8] ^= 0xff;

        let frames = extract_frames(&mut buffer);
        let error = expect_error(&frames);
        assert_eq!(error.kind(), ErrorKind::StreamDecode);
        assert_eq!(error.retry_classification(), RetryClassification::Safe);
        assert!(error.message().contains("checksum"));
    }

    #[test]
    fn reports_a_message_checksum_mismatch() {
        let mut buffer = event_frame("messageStop", b"{}");
        let last = buffer.len() - 1;
        buffer[last] ^= 0xff;

        let frames = extract_frames(&mut buffer);
        let error = expect_error(&frames);
        assert_eq!(error.kind(), ErrorKind::StreamDecode);
        assert_eq!(error.retry_classification(), RetryClassification::Safe);
        assert!(error.message().contains("checksum"));
    }

    #[test]
    fn an_invalid_length_clears_the_buffer() {
        // The length is corrupted and the prelude CRC recomputed to match,
        // so the range check itself is what rejects the frame.
        let mut buffer = event_frame("messageStop", b"{}");
        buffer[0..4].copy_from_slice(&1_u32.to_be_bytes());
        let crc = hash(&buffer[..8]);
        buffer[8..12].copy_from_slice(&crc.to_be_bytes());

        let frames = extract_frames(&mut buffer);
        let error = expect_error(&frames);
        assert_eq!(error.kind(), ErrorKind::StreamDecode);
        assert_eq!(error.retry_classification(), RetryClassification::Safe);
        assert!(error.message().contains("length"));
        assert!(buffer.is_empty(), "the stream position is abandoned");
    }

    #[test]
    fn a_corrupted_length_fails_as_soon_as_the_prelude_arrives() {
        // A flipped bit in the length field that stays within bounds must not
        // leave the parser waiting for bytes that never arrive — the prelude
        // CRC catches it with only the prelude buffered, before the declared
        // length is trusted.
        let frame = event_frame("messageStop", b"{}");
        let mut buffer = frame[..PRELUDE_LENGTH].to_vec();
        buffer[1] ^= 0x01;

        let frames = extract_frames(&mut buffer);
        let error = expect_error(&frames);
        assert_eq!(error.kind(), ErrorKind::StreamDecode);
        assert_eq!(error.retry_classification(), RetryClassification::Safe);
        assert!(error.message().contains("prelude"));
        assert!(buffer.is_empty(), "the stream position is abandoned");
    }

    #[test]
    fn identifies_an_error_frame_and_reads_its_headers() {
        let mut buffer = error_frame("ThrottlingException", "Too many requests", b"{}");

        let frames = extract_frames(&mut buffer);
        let frame = expect_frame(&frames);

        assert!(frame.is_failure());
        assert_eq!(frame.message_type(), Some("error"));
        // An error frame carries no `:event-type`, which is exactly why
        // treating it as an ordinary event would silently drop it.
        assert_eq!(frame.event_type(), None);
        assert_eq!(frame.failure_code(), Some("ThrottlingException"));
        assert_eq!(frame.error_message(), Some("Too many requests"));
    }

    #[test]
    fn identifies_an_exception_frame() {
        let mut headers = Vec::new();
        push_string_header(&mut headers, ":message-type", "exception");
        push_string_header(&mut headers, ":exception-type", "throttlingException");
        let mut buffer = frame(&headers, br#"{"message":"Too many requests"}"#);

        let frames = extract_frames(&mut buffer);
        let frame = expect_frame(&frames);

        assert!(frame.is_failure());
        assert_eq!(frame.exception_type(), Some("throttlingException"));
        assert_eq!(frame.event_type(), None);
        assert_eq!(frame.payload, br#"{"message":"Too many requests"}"#);
    }
}
