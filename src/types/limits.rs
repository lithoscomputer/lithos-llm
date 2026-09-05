use std::io::{self, Write};

use futures_util::StreamExt as _;
use serde::Serialize;

use super::{Error, ErrorKind, Response, ResponseStream, StreamEvent};

/// Response size limits in bytes. Zero rejects any nonempty value.
///
/// Defaults: 32 MiB per HTTP body, 4 MiB per stream frame, and 32 MiB
/// of normalized output. These bound retained data, not total process memory.
/// JSON decoding and a received network chunk can require additional memory.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use]
pub struct ResponseLimits {
    body:   usize,
    frame:  usize,
    output: usize,
}

impl Default for ResponseLimits {
    fn default() -> Self {
        Self {
            body:   32 * 1024 * 1024,
            frame:  4 * 1024 * 1024,
            output: 32 * 1024 * 1024,
        }
    }
}

impl ResponseLimits {
    /// Bounds success and error HTTP response bodies before JSON decoding.
    pub fn max_body_bytes(mut self, bytes: usize) -> Self {
        self.body = bytes;
        self
    }
    /// Bounds an SSE or AWS event-stream frame, including framing bytes.
    pub fn max_frame_bytes(mut self, bytes: usize) -> Self {
        self.frame = bytes;
        self
    }
    /// Bounds serialized normalized output, excluding the raw provider body.
    ///
    /// During streaming, both incremental content events and assembled block
    /// events have their own cumulative budget. Repeated snapshots count again.
    /// The final response must also fit this budget.
    /// Built-in adapters also bound cumulative provider-event data before
    /// decoding, including fields such as reasoning signatures.
    pub fn max_output_bytes(mut self, bytes: usize) -> Self {
        self.output = bytes;
        self
    }
    pub fn body_bytes(self) -> usize {
        self.body
    }
    pub fn frame_bytes(self) -> usize {
        self.frame
    }
    pub fn output_bytes(self) -> usize {
        self.output
    }
}

pub(crate) fn limit_error(resource: &str, limit: usize) -> Error {
    Error::new(
        ErrorKind::ResourceLimit,
        format!("the response exceeded the {resource} limit of {limit} bytes"),
    )
}

/// Shared client- and adapter-boundary policy, also used when completing via
/// streaming.
#[derive(Clone, Copy)]
pub(crate) struct ResponsePolicy {
    pub limits:     ResponseLimits,
    pub retain_raw: bool,
}

impl Default for ResponsePolicy {
    fn default() -> Self {
        Self {
            limits:     ResponseLimits::default(),
            retain_raw: true,
        }
    }
}

impl ResponsePolicy {
    pub(crate) fn response(self, mut response: Response) -> Result<Response, Error> {
        let raw = response.raw.take();
        serialized_size(&response, self.limits.output)?;
        if self.retain_raw {
            response.raw = raw;
        }
        Ok(response)
    }

    pub(crate) fn stream(self, stream: ResponseStream) -> ResponseStream {
        let mut incremental = 0usize;
        let mut assembled = 0usize;
        ResponseStream::new(stream.map(move |event| {
            let event = event?;
            match event {
                StreamEvent::Ended { response } => self
                    .response(response)
                    .map(|response| StreamEvent::Ended { response }),
                event => {
                    let used = match &event {
                        StreamEvent::ContentBlockStart { .. }
                        | StreamEvent::TextDelta { .. }
                        | StreamEvent::ReasoningDelta { .. }
                        | StreamEvent::ToolCallDelta { .. } => Some(&mut incremental),
                        StreamEvent::ContentBlockEnd { .. } => Some(&mut assembled),
                        _ => None,
                    };
                    if let Some(used) = used {
                        *used += serialized_size(&event, self.limits.output.saturating_sub(*used))?;
                    }
                    Ok(event)
                }
            }
        }))
    }
}

/// Count serialized bytes without allocating a second response-sized buffer.
fn serialized_size(value: &impl Serialize, limit: usize) -> Result<usize, Error> {
    struct Counter {
        used:  usize,
        limit: usize,
    }
    impl Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if bytes.len() > self.limit.saturating_sub(self.used) {
                return Err(io::Error::other("output limit exceeded"));
            }
            self.used += bytes.len();
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut counter = Counter { used: 0, limit };
    serde_json::to_writer(&mut counter, value)
        .map_err(|_| limit_error("assembled output", limit))?;
    Ok(counter.used)
}

#[cfg(test)]
mod tests {
    use futures_util::{StreamExt as _, stream};
    use serde_json::json;

    use super::{ResponseLimits, ResponsePolicy};
    use crate::types::{
        ContentBlockId, ContentPart, ErrorKind, Response, ResponseStream, RetryClassification,
        StreamEvent,
    };

    fn response() -> Response {
        Response::new("provider".into(), "model".into(), vec![ContentPart::Text {
            text: "hello".into(),
        }])
    }

    #[test]
    fn raw_retention_is_independent_of_the_output_limit() {
        let mut value = response();
        value.raw = Some(json!({ "large": "x".repeat(4096) }));
        let policy = ResponsePolicy {
            limits:     ResponseLimits::default().max_output_bytes(1024),
            retain_raw: false,
        };
        assert!(
            policy
                .response(value.clone())
                .expect("fits without raw")
                .raw
                .is_none()
        );
        let policy = ResponsePolicy {
            retain_raw: true,
            ..policy
        };
        assert!(
            policy
                .response(value)
                .expect("fits with raw excluded")
                .raw
                .is_some()
        );
        let error = ResponsePolicy {
            limits: ResponseLimits::default().max_output_bytes(1),
            ..policy
        }
        .response(response())
        .expect_err("output too large");
        assert_eq!(error.kind(), ErrorKind::ResourceLimit);
        assert_eq!(error.retry_classification(), RetryClassification::Never);
    }

    #[tokio::test]
    async fn cumulative_stream_output_is_bounded_and_terminal() {
        let events = (0..100).map(|_| {
            Ok(StreamEvent::TextDelta {
                id:   ContentBlockId::new("text"),
                text: "x".repeat(100),
            })
        });
        let policy = ResponsePolicy {
            limits: ResponseLimits::default().max_output_bytes(1024),
            ..ResponsePolicy::default()
        };
        let mut stream = policy.stream(ResponseStream::new(stream::iter(events)));
        let mut count = 0;
        while let Some(event) = stream.next().await {
            count += 1;
            if let Err(error) = event {
                assert_eq!(error.kind(), ErrorKind::ResourceLimit);
                assert!(stream.next().await.is_none());
                assert!(count < 100);
                return;
            }
        }
        panic!("expected an output limit error");
    }
}
