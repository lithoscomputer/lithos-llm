//! Server-sent event framing.
//!
//! [`SseFramer`] splits response bytes into frames as a [`StreamFraming`]
//! directs and parses each frame's `event:` and `data:` lines into one
//! [`SseEvent`]. It is synchronous: the stream pipeline in
//! [`stream`](super::stream) feeds it chunks and flushes it at end of stream.

use std::mem::take;
use std::str::from_utf8;

use super::request::StreamFraming;
use super::stream::Framer;
use crate::types::{Error, ErrorKind, RetryClassification, limit_error};

#[derive(Clone, Debug)]
pub(crate) struct SseEvent {
    pub event: Option<String>,
    pub data:  String,
}

/// Frames an SSE byte stream, bounding each frame to `frame_limit` bytes.
///
/// A stream that ends without the trailing blank line still delivers its last
/// frame: the leftover buffer is flushed at end of stream rather than dropped.
/// A frame that grows past the limit fails the stream, and nothing more is
/// framed after that.
pub(super) struct SseFramer {
    framing:     StreamFraming,
    frame_limit: usize,
    buffer:      Vec<u8>,
    ended:       bool,
}

impl SseFramer {
    pub(super) fn new(framing: StreamFraming, frame_limit: usize) -> Self {
        Self {
            framing,
            frame_limit,
            buffer: Vec::new(),
            ended: false,
        }
    }
}

impl Framer for SseFramer {
    fn push(&mut self, chunk: &[u8]) -> Vec<Result<SseEvent, Error>> {
        let mut events = Vec::new();
        if self.ended {
            return events;
        }
        for piece in chunk.split_inclusive(|byte| matches!(byte, b'\r' | b'\n')) {
            if piece.len() > self.frame_limit.saturating_sub(self.buffer.len()) {
                events.push(Err(limit_error("stream frame", self.frame_limit)));
                self.buffer.clear();
                self.ended = true;
                break;
            }
            self.buffer.extend_from_slice(piece);
            events.extend(extract_frames(&mut self.buffer, self.framing));
        }
        events
    }

    fn finish(&mut self) -> Vec<Result<SseEvent, Error>> {
        if self.ended {
            return Vec::new();
        }
        flush_frame(&mut self.buffer)
    }
}

fn extract_frames(buffer: &mut Vec<u8>, framing: StreamFraming) -> Vec<Result<SseEvent, Error>> {
    let mut events = Vec::new();
    while let Some((end, delimiter_len)) = frame_end(buffer, framing) {
        let frame: Vec<_> = buffer.drain(..end).collect();
        buffer.drain(..delimiter_len);
        match parse_frame(&frame) {
            Ok(Some(event)) => events.push(Ok(event)),
            Ok(None) => {}
            Err(error) => events.push(Err(error)),
        }
    }
    events
}

/// Parses whatever is left in the buffer at end of stream.
///
/// A provider that closes the connection right after its last `data:` line,
/// without the trailing blank line, still delivers that frame. Trailing
/// whitespace alone is not a frame.
fn flush_frame(buffer: &mut Vec<u8>) -> Vec<Result<SseEvent, Error>> {
    let frame = take(buffer);
    if frame.iter().all(u8::is_ascii_whitespace) {
        return Vec::new();
    }
    match parse_frame(&frame) {
        Ok(Some(event)) => vec![Ok(event)],
        Ok(None) => Vec::new(),
        Err(error) => vec![Err(error)],
    }
}

fn frame_end(buffer: &[u8], framing: StreamFraming) -> Option<(usize, usize)> {
    match framing {
        // Binary frames never reach the SSE splitter: `stream_events` routes
        // them to the event-stream decoder. Nothing here can complete a
        // frame, so the buffer is left to the frame limit.
        #[cfg(feature = "bedrock")]
        StreamFraming::AwsEventStream => None,
        StreamFraming::Sse => buffer
            .windows(2)
            .position(|window| window == b"\n\n")
            .map(|index| (index, 2))
            .or_else(|| {
                buffer
                    .windows(4)
                    .position(|window| window == b"\r\n\r\n")
                    .map(|index| (index, 4))
            }),
        // A newline never lands inside a multi-byte UTF-8 character, so a
        // frame that ends here always carries complete characters.
        StreamFraming::SseDataLines => buffer
            .iter()
            .position(|&byte| byte == b'\n')
            .map(|index| (index, 1)),
    }
}

fn parse_frame(frame: &[u8]) -> Result<Option<SseEvent>, Error> {
    // Garbled framing is indistinguishable from mid-stream corruption, so
    // the failure is retryable like any other garbled stream.
    let frame = from_utf8(frame).map_err(|source| {
        Error::new(ErrorKind::StreamDecode, "an SSE frame was not UTF-8")
            .with_source(source)
            .with_retry(RetryClassification::Safe)
    })?;
    let mut event = None;
    let mut data = Vec::new();
    for line in frame.lines() {
        let line = line.trim_end_matches('\r');
        if let Some(value) = line.strip_prefix("event:") {
            event = Some(value.trim_start().to_owned());
        } else if let Some(value) = line.strip_prefix("data:") {
            data.push(value.trim_start());
        }
    }
    if data.is_empty() {
        return Ok(None);
    }
    let data = data.join("\n");
    if data == "[DONE]" {
        return Ok(None);
    }
    Ok(Some(SseEvent { event, data }))
}

#[cfg(test)]
mod tests {
    use super::{SseFramer, StreamFraming};
    use crate::transport::stream::Framer as _;
    use crate::types::ErrorKind;

    /// A framer with a limit no test frame reaches.
    fn framer(framing: StreamFraming) -> SseFramer {
        SseFramer::new(framing, 1024)
    }

    #[test]
    fn parses_frames_across_chunks() {
        let mut framer = framer(StreamFraming::Sse);
        assert!(
            framer
                .push(b"event: delta\ndata: {\"text\":\"hel")
                .is_empty()
        );
        let events = framer.push(b"lo\"}\n\ndata: [DONE]\n\n");
        assert_eq!(events.len(), 1);
        let event = events[0].as_ref().expect("frame should parse");
        assert_eq!(event.event.as_deref(), Some("delta"));
        assert_eq!(event.data, r#"{"text":"hello"}"#);
    }

    #[test]
    fn spec_framing_joins_single_newline_data_lines_into_one_frame() {
        let mut framer = framer(StreamFraming::Sse);

        let events = framer.push(b"data: {\"n\":1}\ndata: {\"n\":2}\n\n");

        assert_eq!(events.len(), 1);
        let event = events[0].as_ref().expect("the frame should parse");
        assert_eq!(event.data, "{\"n\":1}\n{\"n\":2}");
    }

    #[test]
    fn data_line_framing_splits_single_newline_data_lines() {
        let mut framer = framer(StreamFraming::SseDataLines);

        let events = framer.push(b"data: {\"n\":1}\ndata: {\"n\":2}\n\n");

        assert_eq!(events.len(), 2);
        assert_eq!(events[0].as_ref().expect("line one").data, r#"{"n":1}"#);
        assert_eq!(events[1].as_ref().expect("line two").data, r#"{"n":2}"#);
        assert!(framer.finish().is_empty(), "the blank line is not a frame");
    }

    #[test]
    fn data_line_framing_skips_comments_terminators_and_non_data_lines() {
        let mut framer = framer(StreamFraming::SseDataLines);

        let events = framer.push(b": keep-alive\nevent: x\ndata: [DONE]\ndata: {\"n\":1}\n");

        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0].as_ref().expect("the data line").data,
            r#"{"n":1}"#
        );
    }

    #[test]
    fn data_line_framing_handles_crlf_line_endings() {
        let mut framer = framer(StreamFraming::SseDataLines);

        let events = framer.push(b"data: {\"n\":1}\r\n\r\ndata: {\"n\":2}\r\n");

        assert_eq!(events.len(), 2);
        assert_eq!(events[0].as_ref().expect("line one").data, r#"{"n":1}"#);
        assert_eq!(events[1].as_ref().expect("line two").data, r#"{"n":2}"#);
    }

    #[test]
    fn data_line_framing_buffers_a_split_utf8_character() {
        let text = "data: {\"t\":\"é\"}\n";
        // Split inside the two-byte `é` so the first chunk is not UTF-8.
        let split = 13;
        assert!(!text.is_char_boundary(split));

        let bytes = text.as_bytes();
        let mut framer = framer(StreamFraming::SseDataLines);
        assert!(framer.push(&bytes[..split]).is_empty());
        let events = framer.push(&bytes[split..]);

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].as_ref().expect("the line").data, r#"{"t":"é"}"#);
    }

    #[test]
    fn flushes_a_last_frame_that_has_no_trailing_blank_line() {
        let mut framer = framer(StreamFraming::Sse);

        assert!(framer.push(b"data: {\"text\":\"bye\"}").is_empty());
        let events = framer.finish();

        assert_eq!(events.len(), 1);
        let event = events[0].as_ref().expect("the tail should parse");
        assert_eq!(event.data, r#"{"text":"bye"}"#);
        assert!(framer.finish().is_empty(), "the flush drained the buffer");
    }

    #[test]
    fn a_trailing_newline_is_not_a_frame() {
        let mut framer = framer(StreamFraming::Sse);

        assert!(framer.push(b"\n").is_empty());
        assert!(framer.finish().is_empty());
    }

    #[test]
    fn an_oversized_unterminated_frame_fails_and_ends_framing() {
        let mut framer = SseFramer::new(StreamFraming::Sse, 9);

        assert!(framer.push(b"data: ").is_empty());
        let events = framer.push(b"1234567890");

        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0].as_ref().expect_err("oversized frame").kind(),
            ErrorKind::ResourceLimit
        );
        assert!(
            framer.push(b"data: 1\n\n").is_empty(),
            "nothing frames after the limit"
        );
        assert!(framer.finish().is_empty());
    }
}

/// Properties that must hold for every byte stream, beyond the examples above.
#[cfg(test)]
mod properties {
    use proptest::collection::vec;
    use proptest::option;
    use proptest::prelude::*;

    use super::{SseEvent, SseFramer, StreamFraming};
    use crate::transport::stream::Framer as _;
    use crate::types::{Error, ErrorKind};

    /// What one frame produced, in a form tests can compare.
    type Outcome = Result<(Option<String>, String), ErrorKind>;

    fn outcomes(results: Vec<Result<SseEvent, Error>>) -> impl Iterator<Item = Outcome> {
        results.into_iter().map(|result| {
            result
                .map(|event| (event.event, event.data))
                .map_err(|error| error.kind())
        })
    }

    /// Frames `bytes` delivered as chunks split at `cuts`, then flushes.
    ///
    /// Each cut is taken modulo the input length, so any list of cuts is a
    /// valid split, including empty chunks.
    fn frame(
        framing: StreamFraming,
        frame_limit: usize,
        bytes: &[u8],
        cuts: &[usize],
    ) -> Vec<Outcome> {
        let mut cuts: Vec<_> = cuts.iter().map(|cut| cut % (bytes.len() + 1)).collect();
        cuts.sort_unstable();
        let mut framer = SseFramer::new(framing, frame_limit);
        let mut results = Vec::new();
        let mut start = 0;
        for cut in cuts.into_iter().chain([bytes.len()]) {
            results.extend(outcomes(framer.push(&bytes[start..cut])));
            start = cut;
        }
        results.extend(outcomes(framer.finish()));
        results
    }

    fn framing() -> impl Strategy<Value = StreamFraming> {
        prop_oneof![Just(StreamFraming::Sse), Just(StreamFraming::SseDataLines)]
    }

    /// Bytes built from SSE fragments, so frames, line endings, split
    /// characters, and invalid UTF-8 all turn up often.
    fn sse_bytes() -> impl Strategy<Value = Vec<u8>> {
        let fragment = prop_oneof![
            Just(b"data: ".to_vec()),
            Just(b"event: ".to_vec()),
            Just(b": comment".to_vec()),
            Just(b"[DONE]".to_vec()),
            Just(b"\n".to_vec()),
            Just(b"\r".to_vec()),
            Just(b"\r\n".to_vec()),
            Just("é".as_bytes().to_vec()),
            "[a-z{}\":]{1,8}".prop_map(String::into_bytes),
            any::<u8>().prop_map(|byte| vec![byte]),
        ];
        vec(fragment, 0..48).prop_map(|fragments| fragments.concat())
    }

    /// A `data:` value the framer returns unchanged: no line break, no
    /// leading whitespace, and never the `[DONE]` terminator.
    fn data_value() -> impl Strategy<Value = String> {
        "([a-z0-9{}\":,.é][a-z0-9{}\":,. é]{0,12})?"
    }

    fn line_ending() -> impl Strategy<Value = &'static str> {
        prop_oneof![Just("\n"), Just("\r\n")]
    }

    proptest! {
        #[test]
        fn chunk_boundaries_never_change_the_frames(
            framing in framing(),
            frame_limit in 1_usize..256,
            bytes in sse_bytes(),
            cuts in vec(any::<usize>(), 0..8),
        ) {
            prop_assert_eq!(
                frame(framing, frame_limit, &bytes, &cuts),
                frame(framing, frame_limit, &bytes, &[]),
            );
        }

        #[test]
        fn spec_framing_returns_each_encoded_event(
            events in vec((option::of("[a-z_.]{1,12}"), vec(data_value(), 1..4)), 0..6),
            eol in line_ending(),
            cuts in vec(any::<usize>(), 0..8),
        ) {
            let mut stream = String::new();
            for (name, lines) in &events {
                if let Some(name) = name {
                    stream.push_str("event: ");
                    stream.push_str(name);
                    stream.push_str(eol);
                }
                for line in lines {
                    stream.push_str("data: ");
                    stream.push_str(line);
                    stream.push_str(eol);
                }
                stream.push_str(eol);
            }
            let expected: Vec<Outcome> = events
                .into_iter()
                .map(|(name, lines)| Ok((name, lines.join("\n"))))
                .collect();

            prop_assert_eq!(
                frame(StreamFraming::Sse, usize::MAX, stream.as_bytes(), &cuts),
                expected,
            );
        }

        #[test]
        fn data_line_framing_returns_each_data_line(
            lines in vec((data_value(), any::<bool>()), 0..8),
            eol in line_ending(),
            cuts in vec(any::<usize>(), 0..8),
        ) {
            let mut stream = String::new();
            for (value, blank_line_after) in &lines {
                stream.push_str("data: ");
                stream.push_str(value);
                stream.push_str(eol);
                if *blank_line_after {
                    stream.push_str(eol);
                }
            }
            let expected: Vec<Outcome> =
                lines.into_iter().map(|(value, _)| Ok((None, value))).collect();

            prop_assert_eq!(
                frame(StreamFraming::SseDataLines, usize::MAX, stream.as_bytes(), &cuts),
                expected,
            );
        }
    }
}
