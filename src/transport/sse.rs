//! Server-sent event parsing.
//!
//! [`SseFramer`] parses a response byte stream as the WHATWG HTML
//! specification's "Parsing an event stream" section defines, in two layers:
//! [`LineSplitter`] cuts bytes into lines at LF, CRLF, or a bare CR, and
//! [`EventBuilder`] applies the field rules to each line and sends events. It
//! is synchronous: the stream pipeline in [`stream`](super::stream) feeds it
//! chunks and flushes it at end of stream.
//!
//! Two deviations from the specification are deliberate:
//!
//! - At end of stream, a pending event is sent rather than discarded, and an
//!   unterminated last line still counts as a line. Some providers close the
//!   connection right after their last `data:` line, without the blank line.
//! - A line that is not UTF-8 fails the stream with a retryable error rather
//!   than decoding with replacement characters. Replacing bytes would silently
//!   corrupt model text or tool arguments, and garbled bytes are
//!   indistinguishable from mid-stream corruption, which a retry can repair.
//!
//! The `id` and `retry` fields are ignored. They serve reconnection, and a
//! provider cannot resume a generation from an event id.

use std::mem::take;
use std::str::from_utf8;

use super::stream::Framer;
use crate::types::{Error, ErrorKind, RetryClassification, limit_error};

/// The UTF-8 byte-order mark the specification strips from a stream's start.
const BYTE_ORDER_MARK: &[u8] = b"\xEF\xBB\xBF";

#[derive(Clone, Debug)]
pub(crate) struct SseEvent {
    pub event: Option<String>,
    pub data:  String,
}

/// When a parsed event is sent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SseDispatch {
    /// A blank line sends the event, whose `data:` lines join with `\n`, as
    /// the specification requires.
    BlankLine,
    /// Every `data:` line sends an event on its own, as soon as its line ends.
    ///
    /// Lenient skins and proxies separate events with single line breaks
    /// rather than blank lines, and each of their events is one `data:` line.
    EachDataLine,
}

/// Parses an SSE byte stream into events, bounding each event to `limit` bytes.
///
/// The limit covers what the framer holds for the pending event: its data and
/// event name so far, plus the line being read. An event that grows past it
/// fails the stream, and nothing is parsed after that.
pub(super) struct SseFramer {
    lines:  LineSplitter,
    events: EventBuilder,
    limit:  usize,
    ended:  bool,
}

impl SseFramer {
    pub(super) fn new(dispatch: SseDispatch, limit: usize) -> Self {
        Self {
            lines: LineSplitter::new(),
            events: EventBuilder::new(dispatch),
            limit,
            ended: false,
        }
    }

    /// Parses one complete line, returning the event it sends, if any.
    fn line(&mut self, line: &[u8]) -> Result<Option<SseEvent>, Error> {
        let line = from_utf8(line).map_err(|source| {
            Error::new(ErrorKind::StreamDecode, "an SSE line was not UTF-8")
                .with_source(source)
                .with_retry(RetryClassification::Safe)
        })?;
        Ok(self.events.line(line))
    }

    /// Adds one line's outcome to `events`. An error ends parsing.
    fn deliver(
        &mut self,
        outcome: Result<Option<SseEvent>, Error>,
        events: &mut Vec<Result<SseEvent, Error>>,
    ) {
        match outcome {
            Ok(event) => events.extend(event.map(Ok)),
            Err(error) => {
                self.ended = true;
                events.push(Err(error));
            }
        }
    }
}

impl Framer for SseFramer {
    fn push(&mut self, mut chunk: &[u8]) -> Vec<Result<SseEvent, Error>> {
        let mut events = Vec::new();
        while !self.ended {
            let budget = self.limit.saturating_sub(self.events.retained());
            let outcome = match self.lines.next_line(&mut chunk, budget) {
                Ok(Some(line)) => self.line(&line),
                Ok(None) => break,
                Err(Overflow) => Err(limit_error("stream frame", self.limit)),
            };
            self.deliver(outcome, &mut events);
        }
        events
    }

    fn finish(&mut self) -> Vec<Result<SseEvent, Error>> {
        let mut events = Vec::new();
        if self.ended {
            return events;
        }
        if let Some(line) = self.lines.finish() {
            let outcome = self.line(&line);
            self.deliver(outcome, &mut events);
        }
        if !self.ended {
            events.extend(self.events.dispatch().map(Ok));
        }
        self.ended = true;
        events
    }
}

/// The open line grew past its byte budget.
#[derive(Debug)]
struct Overflow;

/// Cuts bytes into lines at LF, CRLF, or a bare CR, across chunk boundaries.
#[derive(Debug)]
struct LineSplitter {
    /// The open line, which the next line ending completes.
    line:       Vec<u8>,
    /// Whether the last line ended with CR. An LF right after it, even at the
    /// start of the next chunk, completes that CRLF rather than an empty line.
    after_cr:   bool,
    /// Whether no line has completed yet, so the next may start with a
    /// byte-order mark.
    first_line: bool,
}

impl LineSplitter {
    fn new() -> Self {
        Self {
            line:       Vec::new(),
            after_cr:   false,
            first_line: true,
        }
    }

    /// Takes bytes from the front of `chunk` through the next line ending.
    ///
    /// Returns the completed line without its ending, or `None` once `chunk`
    /// is used up with the line still open. The open line may hold at most
    /// `budget` bytes. Each byte is examined once, however long the line.
    fn next_line(&mut self, chunk: &mut &[u8], budget: usize) -> Result<Option<Vec<u8>>, Overflow> {
        if self.after_cr && !chunk.is_empty() {
            self.after_cr = false;
            if let Some(rest) = chunk.strip_prefix(b"\n") {
                *chunk = rest;
            }
        }
        let end = chunk.iter().position(|&byte| matches!(byte, b'\r' | b'\n'));
        let piece = &chunk[..end.unwrap_or(chunk.len())];
        if piece.len() > budget.saturating_sub(self.line.len()) {
            return Err(Overflow);
        }
        self.line.extend_from_slice(piece);
        let Some(end) = end else {
            *chunk = &[];
            return Ok(None);
        };
        self.after_cr = chunk[end] == b'\r';
        *chunk = &chunk[end + 1..];
        Ok(Some(self.take_line()))
    }

    /// The unterminated last line at end of stream, if it holds any bytes.
    fn finish(&mut self) -> Option<Vec<u8>> {
        (!self.line.is_empty()).then(|| self.take_line())
    }

    fn take_line(&mut self) -> Vec<u8> {
        let mut line = take(&mut self.line);
        if take(&mut self.first_line) && line.starts_with(BYTE_ORDER_MARK) {
            line.drain(..BYTE_ORDER_MARK.len());
        }
        line
    }
}

/// Applies the specification's field rules to lines and sends events.
#[derive(Debug)]
struct EventBuilder {
    policy: SseDispatch,
    event:  Option<String>,
    /// Each `data:` value followed by `\n`, as the specification builds it,
    /// so a single empty `data:` line still sends an event.
    data:   String,
}

impl EventBuilder {
    fn new(policy: SseDispatch) -> Self {
        Self {
            policy,
            event: None,
            data: String::new(),
        }
    }

    /// The bytes held for the pending event.
    fn retained(&self) -> usize {
        self.data
            .len()
            .saturating_add(self.event.as_ref().map_or(0, String::len))
    }

    /// Applies one line, returning the event it sends, if any.
    fn line(&mut self, line: &str) -> Option<SseEvent> {
        if line.is_empty() {
            return self.dispatch();
        }
        // A line without a colon is a field name with an empty value. One
        // space after the colon is the separator; any more belong to the
        // value.
        let (field, value) = line.split_once(':').map_or((line, ""), |(field, value)| {
            (field, value.strip_prefix(' ').unwrap_or(value))
        });
        match field {
            // An empty name means the default event type.
            "event" => self.event = (!value.is_empty()).then(|| value.to_owned()),
            "data" => {
                self.data.push_str(value);
                self.data.push('\n');
                if self.policy == SseDispatch::EachDataLine {
                    return self.dispatch();
                }
            }
            // A comment line starts with a colon, so its field name is empty.
            // `id` and `retry` serve reconnection, which a model call cannot
            // use, and the specification ignores unknown fields.
            _ => {}
        }
        None
    }

    /// Sends the pending event, if it has data, and starts the next one.
    fn dispatch(&mut self) -> Option<SseEvent> {
        let event = self.event.take();
        let mut data = take(&mut self.data);
        // Only the last line break is framing; the others join data lines. An
        // event with no `data:` line at all sends nothing.
        data.pop()?;
        Some(SseEvent { event, data })
    }
}

#[cfg(test)]
mod tests {
    use super::{SseDispatch, SseEvent, SseFramer};
    use crate::transport::stream::Framer as _;
    use crate::types::{Error, ErrorKind, RetryClassification};

    /// A framer with a limit no test event reaches.
    fn framer(dispatch: SseDispatch) -> SseFramer {
        SseFramer::new(dispatch, 1024)
    }

    /// The data of each event, failing the test on an error.
    fn data(events: &[Result<SseEvent, Error>]) -> Vec<&str> {
        events
            .iter()
            .map(|event| {
                event
                    .as_ref()
                    .expect("the event should parse")
                    .data
                    .as_str()
            })
            .collect()
    }

    #[test]
    fn parses_events_across_chunks() {
        let mut framer = framer(SseDispatch::BlankLine);
        assert!(
            framer
                .push(b"event: delta\ndata: {\"text\":\"hel")
                .is_empty()
        );
        let events = framer.push(b"lo\"}\n\n");
        assert_eq!(events.len(), 1);
        let event = events[0].as_ref().expect("event should parse");
        assert_eq!(event.event.as_deref(), Some("delta"));
        assert_eq!(event.data, r#"{"text":"hello"}"#);
    }

    /// The first example stream in the specification's "Interpreting an event
    /// stream" section.
    #[test]
    fn spec_example_joins_data_lines() {
        let mut framer = framer(SseDispatch::BlankLine);

        let events = framer.push(b"data: YHOO\ndata: +2\ndata: 10\n\n");

        assert_eq!(data(&events), ["YHOO\n+2\n10"]);
    }

    /// The specification's second example: comments, `id` fields, an absent
    /// space after the colon, and a second space that belongs to the value.
    #[test]
    fn spec_example_strips_exactly_one_space() {
        let mut framer = framer(SseDispatch::BlankLine);

        let events = framer.push(
            b": test stream\n\ndata: first event\nid: 1\n\ndata:second event\nid\n\n\
              data:  third event\n\n",
        );

        assert_eq!(data(&events), [
            "first event",
            "second event",
            " third event"
        ]);
    }

    /// The specification's third example: a bare `data` field is an empty data
    /// line. The specification discards the unterminated last event; this
    /// framer sends it at end of stream.
    #[test]
    fn spec_example_treats_a_bare_field_name_as_an_empty_value() {
        let mut framer = framer(SseDispatch::BlankLine);

        let events = framer.push(b"data\n\ndata\ndata\n\ndata:");

        assert_eq!(data(&events), ["", "\n"]);
        assert_eq!(data(&framer.finish()), [""]);
    }

    /// The specification's fourth example: the two events are identical.
    #[test]
    fn spec_example_makes_the_separating_space_optional() {
        let mut framer = framer(SseDispatch::BlankLine);

        let events = framer.push(b"data:test\n\ndata: test\n\n");

        assert_eq!(data(&events), ["test", "test"]);
    }

    #[test]
    fn accepts_every_line_ending_the_specification_allows() {
        for stream in [
            &b"data: a\n\ndata: b\n\n"[..],
            b"data: a\r\n\r\ndata: b\r\n\r\n",
            b"data: a\r\rdata: b\r\r",
            b"data: a\n\r\ndata: b\r\n\n",
            b"data: a\r\n\rdata: b\n\r",
        ] {
            let mut framer = framer(SseDispatch::BlankLine);

            let events = framer.push(stream);

            assert_eq!(
                data(&events),
                ["a", "b"],
                "{:?}",
                String::from_utf8_lossy(stream)
            );
        }
    }

    /// An LF that starts a chunk after a CR that ended the last one is the
    /// second half of one CRLF, not an empty line that sends the event early.
    #[test]
    fn a_crlf_split_across_chunks_is_one_line_ending() {
        let mut framer = framer(SseDispatch::BlankLine);

        assert!(framer.push(b"data: a\r").is_empty());
        assert!(framer.push(b"\ndata: b\r").is_empty());
        let events = framer.push(b"\n\r\n");

        assert_eq!(data(&events), ["a\nb"]);
    }

    #[test]
    fn strips_a_byte_order_mark_from_the_stream_start_only() {
        let mut framer = framer(SseDispatch::BlankLine);

        assert!(framer.push(b"\xEF\xBB").is_empty());
        let events = framer.push(b"\xBFdata: a\n\n\xEF\xBB\xBFdata: b\n\ndata: c\n\n");

        // A later mark is part of the field name, so that line is ignored.
        assert_eq!(data(&events), ["a", "c"]);
    }

    #[test]
    fn an_empty_event_field_means_the_default_type() {
        let mut framer = framer(SseDispatch::BlankLine);

        let events = framer.push(b"event: named\nevent:\ndata: a\n\n");

        assert_eq!(events[0].as_ref().expect("the event").event, None);
    }

    #[test]
    fn the_event_name_resets_after_each_event() {
        let mut framer = framer(SseDispatch::BlankLine);

        let events =
            framer.push(b"event: first\ndata: a\n\ndata: b\n\nevent: dropped\n\ndata: c\n\n");

        let names: Vec<_> = events
            .iter()
            .map(|event| event.as_ref().expect("the event").event.as_deref())
            .collect();
        assert_eq!(names, [Some("first"), None, None]);
    }

    /// `[DONE]` belongs to the Chat dialect, not to SSE, so the codecs that
    /// speak that dialect decide what it means.
    #[test]
    fn delivers_the_done_terminator_as_ordinary_data() {
        for dispatch in [SseDispatch::BlankLine, SseDispatch::EachDataLine] {
            let mut framer = framer(dispatch);

            let events = framer.push(b"data: [DONE]\n\n");

            assert_eq!(data(&events), ["[DONE]"], "{dispatch:?}");
        }
    }

    #[test]
    fn each_data_line_dispatch_sends_single_newline_data_lines_apart() {
        let mut framer = framer(SseDispatch::EachDataLine);

        let events = framer.push(b"data: {\"n\":1}\ndata: {\"n\":2}\n\n");

        assert_eq!(data(&events), [r#"{"n":1}"#, r#"{"n":2}"#]);
        assert!(framer.finish().is_empty(), "the blank line is not an event");
    }

    #[test]
    fn each_data_line_dispatch_skips_comments_and_keeps_the_event_name() {
        let mut framer = framer(SseDispatch::EachDataLine);

        let events = framer.push(b": keep-alive\nevent: x\nid: 7\ndata: {\"n\":1}\ndata: 2\n");

        assert_eq!(data(&events), [r#"{"n":1}"#, "2"]);
        assert_eq!(
            events[0].as_ref().expect("line one").event.as_deref(),
            Some("x")
        );
        assert_eq!(events[1].as_ref().expect("line two").event, None);
    }

    #[test]
    fn each_data_line_dispatch_buffers_a_split_utf8_character() {
        let text = "data: {\"t\":\"é\"}\n";
        // Split inside the two-byte `é` so the first chunk is not UTF-8.
        let split = 13;
        assert!(!text.is_char_boundary(split));

        let bytes = text.as_bytes();
        let mut framer = framer(SseDispatch::EachDataLine);
        assert!(framer.push(&bytes[..split]).is_empty());
        let events = framer.push(&bytes[split..]);

        assert_eq!(data(&events), [r#"{"t":"é"}"#]);
    }

    /// Deliberate deviation: the specification discards an event the stream
    /// ends inside, but some providers close right after their last line.
    #[test]
    fn sends_a_pending_event_at_end_of_stream() {
        for (stream, tail) in [
            (&b"data: {\"text\":\"bye\"}"[..], "an unterminated line"),
            (b"data: {\"text\":\"bye\"}\n", "a missing blank line"),
        ] {
            let mut framer = framer(SseDispatch::BlankLine);

            assert!(framer.push(stream).is_empty(), "{tail}");
            let events = framer.finish();

            assert_eq!(data(&events), [r#"{"text":"bye"}"#], "{tail}");
            assert!(framer.finish().is_empty(), "the flush drained the framer");
        }
    }

    #[test]
    fn a_trailing_line_break_is_not_an_event() {
        let mut framer = framer(SseDispatch::BlankLine);

        assert!(framer.push(b"\n").is_empty());
        assert!(framer.finish().is_empty());
    }

    /// Deliberate deviation: the specification decodes with replacement
    /// characters, which would silently corrupt model output.
    #[test]
    fn a_line_that_is_not_utf8_fails_retryably_and_ends_parsing() {
        let mut framer = framer(SseDispatch::BlankLine);

        let events = framer.push(b"data: \xFF\n\ndata: a\n\n");

        assert_eq!(events.len(), 1);
        let error = events[0].as_ref().expect_err("invalid UTF-8");
        assert_eq!(error.kind(), ErrorKind::StreamDecode);
        assert_eq!(error.retry_classification(), RetryClassification::Safe);
        assert!(
            framer.push(b"data: b\n\n").is_empty(),
            "nothing parses after the failure"
        );
        assert!(framer.finish().is_empty());
    }

    #[test]
    fn an_oversized_unterminated_line_fails_and_ends_parsing() {
        let mut framer = SseFramer::new(SseDispatch::BlankLine, 9);

        assert!(framer.push(b"data: ").is_empty());
        let events = framer.push(b"1234567890");

        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0].as_ref().expect_err("oversized line").kind(),
            ErrorKind::ResourceLimit
        );
        assert!(
            framer.push(b"data: 1\n\n").is_empty(),
            "nothing parses after the limit"
        );
        assert!(framer.finish().is_empty());
    }

    /// Each line fits on its own; together they exceed what one event may hold.
    #[test]
    fn the_limit_bounds_an_event_across_its_lines() {
        let mut framer = SseFramer::new(SseDispatch::BlankLine, 12);

        let events = framer.push(b"data: 12345\ndata: 12345\n\n");

        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0].as_ref().expect_err("oversized event").kind(),
            ErrorKind::ResourceLimit
        );
    }
}

/// Properties that must hold for every byte stream, beyond the examples above.
#[cfg(test)]
mod properties {
    use proptest::collection::vec;
    use proptest::option;
    use proptest::prelude::*;

    use super::{SseDispatch, SseEvent, SseFramer};
    use crate::transport::stream::Framer as _;
    use crate::types::{Error, ErrorKind};

    /// What one event produced, in a form tests can compare.
    type Outcome = Result<(Option<String>, String), ErrorKind>;

    fn outcomes(results: Vec<Result<SseEvent, Error>>) -> impl Iterator<Item = Outcome> {
        results.into_iter().map(|result| {
            result
                .map(|event| (event.event, event.data))
                .map_err(|error| error.kind())
        })
    }

    /// Parses `bytes` delivered as chunks split at `cuts`, then flushes.
    ///
    /// Each cut is taken modulo the input length, so any list of cuts is a
    /// valid split, including empty chunks.
    fn parse(dispatch: SseDispatch, limit: usize, bytes: &[u8], cuts: &[usize]) -> Vec<Outcome> {
        let mut cuts: Vec<_> = cuts.iter().map(|cut| cut % (bytes.len() + 1)).collect();
        cuts.sort_unstable();
        let mut framer = SseFramer::new(dispatch, limit);
        let mut results = Vec::new();
        let mut start = 0;
        for cut in cuts.into_iter().chain([bytes.len()]) {
            results.extend(outcomes(framer.push(&bytes[start..cut])));
            start = cut;
        }
        results.extend(outcomes(framer.finish()));
        results
    }

    fn dispatch() -> impl Strategy<Value = SseDispatch> {
        prop_oneof![
            Just(SseDispatch::BlankLine),
            Just(SseDispatch::EachDataLine)
        ]
    }

    /// Bytes built from SSE fragments, so fields, comments, every line ending,
    /// byte-order marks, split characters, and invalid UTF-8 all turn up often.
    fn sse_bytes() -> impl Strategy<Value = Vec<u8>> {
        let fragment = prop_oneof![
            Just(b"data: ".to_vec()),
            Just(b"data:".to_vec()),
            Just(b"data".to_vec()),
            Just(b"event: ".to_vec()),
            Just(b"id: 1".to_vec()),
            Just(b": comment".to_vec()),
            Just(b" ".to_vec()),
            Just(b"\n".to_vec()),
            Just(b"\r".to_vec()),
            Just(b"\r\n".to_vec()),
            Just(b"\xEF\xBB\xBF".to_vec()),
            Just("é".as_bytes().to_vec()),
            "[a-z{}\":]{1,8}".prop_map(String::into_bytes),
            any::<u8>().prop_map(|byte| vec![byte]),
        ];
        vec(fragment, 0..48).prop_map(|fragments| fragments.concat())
    }

    /// A field value: any text without a line break, leading spaces included.
    fn value() -> impl Strategy<Value = String> {
        "[a-z0-9{}\":,. é\\[\\]]{0,12}"
    }

    fn line_ending() -> impl Strategy<Value = &'static str> {
        prop_oneof![Just("\n"), Just("\r\n"), Just("\r")]
    }

    /// A line that sends nothing and changes no event.
    fn inert_line() -> impl Strategy<Value = &'static str> {
        prop_oneof![
            Just(": keep-alive"),
            Just(":"),
            Just("id: 7"),
            Just("id"),
            Just("retry: 10"),
            Just("unknown: field"),
        ]
    }

    /// Writes SSE lines, each with its own line ending.
    #[derive(Default)]
    struct Encoder {
        stream:   String,
        after_cr: bool,
    }

    impl Encoder {
        fn line(&mut self, text: &str, ending: &str) {
            // A blank line whose LF follows a CR would read as one CRLF, so an
            // encoder never writes that; it ends the blank line with CRLF.
            let ending = if self.after_cr && text.is_empty() && ending == "\n" {
                "\r\n"
            } else {
                ending
            };
            self.stream.push_str(text);
            self.stream.push_str(ending);
            self.after_cr = ending == "\r";
        }

        /// Writes a field, with or without the optional space after the colon.
        fn field(&mut self, name: &str, value: &str, spaced: bool, ending: &str) {
            // Without the separating space, a value's own leading space would
            // be taken as the separator.
            let separator = if spaced || value.starts_with(' ') {
                ": "
            } else {
                ":"
            };
            self.line(&format!("{name}{separator}{value}"), ending);
        }
    }

    /// One generated event: its name, its data lines, and the inert lines
    /// written before it, each line with its own ending and spacing.
    type EncodedEvent = (
        Option<String>,
        Vec<(String, bool, &'static str)>,
        Vec<&'static str>,
        &'static str,
    );

    fn encoded_event() -> impl Strategy<Value = EncodedEvent> {
        (
            option::of("[a-z_.]{1,12}"),
            vec((value(), any::<bool>(), line_ending()), 1..4),
            vec(inert_line(), 0..3),
            line_ending(),
        )
    }

    proptest! {
        #[test]
        fn chunk_boundaries_never_change_the_events(
            dispatch in dispatch(),
            limit in 1_usize..256,
            bytes in sse_bytes(),
            cuts in vec(any::<usize>(), 0..8),
        ) {
            prop_assert_eq!(
                parse(dispatch, limit, &bytes, &cuts),
                parse(dispatch, limit, &bytes, &[]),
            );
        }

        #[test]
        fn blank_line_dispatch_returns_each_encoded_event(
            events in vec(encoded_event(), 0..6),
            byte_order_mark in any::<bool>(),
            cuts in vec(any::<usize>(), 0..8),
        ) {
            let mut encoder = Encoder::default();
            if byte_order_mark {
                encoder.stream.push('\u{FEFF}');
            }
            for (name, lines, inert, ending) in &events {
                for line in inert {
                    encoder.line(line, ending);
                }
                if let Some(name) = name {
                    encoder.field("event", name, true, ending);
                }
                for (value, spaced, ending) in lines {
                    encoder.field("data", value, *spaced, ending);
                }
                encoder.line("", ending);
            }
            let expected: Vec<Outcome> = events
                .into_iter()
                .map(|(name, lines, _, _)| {
                    let values: Vec<_> = lines.into_iter().map(|(value, _, _)| value).collect();
                    Ok((name, values.join("\n")))
                })
                .collect();

            prop_assert_eq!(
                parse(SseDispatch::BlankLine, usize::MAX, encoder.stream.as_bytes(), &cuts),
                expected,
            );
        }

        #[test]
        fn each_data_line_dispatch_returns_each_data_line(
            lines in vec((value(), any::<bool>(), line_ending(), option::of(inert_line())), 0..8),
            cuts in vec(any::<usize>(), 0..8),
        ) {
            let mut encoder = Encoder::default();
            for (value, spaced, ending, after) in &lines {
                encoder.field("data", value, *spaced, ending);
                match after {
                    Some(line) => encoder.line(line, ending),
                    None => encoder.line("", ending),
                }
            }
            let expected: Vec<Outcome> =
                lines.into_iter().map(|(value, _, _, _)| Ok((None, value))).collect();

            prop_assert_eq!(
                parse(SseDispatch::EachDataLine, usize::MAX, encoder.stream.as_bytes(), &cuts),
                expected,
            );
        }
    }
}
