//! The response stream pipeline: idle timeouts, framing, and output bounds.
//!
//! Every event stream the transport opens goes through the same three
//! stages: [`with_idle_timeout`] fails a stalled read, [`frame_stream`] turns
//! chunks into events through one [`Framer`], and [`bound_event_data`] caps
//! the bytes a decoder may accumulate. Only the framer differs between an SSE
//! response and an AWS event stream.

use std::future::ready;
use std::pin::Pin;
use std::time::Duration;

use futures_core::Stream;
use futures_util::StreamExt as _;
use futures_util::stream::{iter, unfold};
use tokio::time::timeout;

use super::sse::SseEvent;
use crate::catalog::ProviderId;
use crate::types::{Error, ErrorKind, RateLimits, RetryClassification, limit_error};

pub(super) type EventStream = Pin<Box<dyn Stream<Item = Result<SseEvent, Error>> + Send>>;

pub(crate) struct EventResponse {
    pub events:      EventStream,
    pub rate_limits: Option<RateLimits>,
}

/// Splits response bytes into events, synchronously.
///
/// A framer owns whatever partial frame the last chunk left behind.
/// [`push`](Self::push) takes the next chunk and returns every event it
/// completed; [`finish`](Self::finish) runs once at end of stream for a frame
/// the stream closed without terminating. An `Err` in either result ends the
/// stream after it is delivered.
pub(super) trait Framer: Send + 'static {
    fn push(&mut self, chunk: &[u8]) -> Vec<Result<SseEvent, Error>>;
    fn finish(&mut self) -> Vec<Result<SseEvent, Error>>;
}

/// Turns response chunks into events through `framer`.
///
/// A chunk-level error is delivered and ends the stream, as does the first
/// error a framer produces. The framer is flushed when the chunks end without
/// error.
pub(super) fn frame_stream<C, S, F>(
    chunks: S,
    framer: F,
) -> impl Stream<Item = Result<SseEvent, Error>> + Send + 'static
where
    S: Stream<Item = Result<C, Error>> + Send + 'static,
    C: AsRef<[u8]> + Send + 'static,
    F: Framer,
{
    chunks
        .map(Some)
        .chain(iter([None]))
        .scan((framer, false), |(framer, ended), chunk| {
            if *ended {
                return ready(None);
            }
            let batch = match chunk {
                Some(Ok(chunk)) => framer.push(chunk.as_ref()),
                Some(Err(error)) => vec![Err(error)],
                None => framer.finish(),
            };
            *ended = batch.iter().any(Result::is_err);
            ready(Some(batch))
        })
        .flat_map(iter)
}

/// Bound data before a decoder can accumulate non-visible fields such as
/// reasoning signatures. Normalized events alone do not expose all such data.
pub(super) fn bound_event_data<S>(
    events: S,
    limit: usize,
) -> impl Stream<Item = Result<SseEvent, Error>> + Send
where
    S: Stream<Item = Result<SseEvent, Error>> + Send,
{
    unfold(
        (Box::pin(events), 0usize, false),
        move |(mut events, used, ended)| async move {
            if ended {
                return None;
            }
            let event = events.next().await?;
            let size = event.as_ref().map_or(0, |event| {
                event
                    .data
                    .len()
                    .saturating_add(event.event.as_ref().map_or(0, String::len))
            });
            if size > limit.saturating_sub(used) {
                return Some((
                    Err(limit_error("provider event data", limit)),
                    (events, used, true),
                ));
            }
            let ended = event.is_err();
            Some((event, (events, used + size, ended)))
        },
    )
}

/// Builds the error for a failed response-body read.
///
/// The caller's whole-request timeout can expire while the body is still
/// arriving. The provider is already executing that call, so a mid-stream
/// expiry keeps the complete path's never-retry rule and reports as a timeout
/// rather than a network fault; see [`json_response`] for the same rule on the
/// complete body.
pub(super) fn chunk_error(
    provider: &ProviderId,
    message: &'static str,
    source: reqwest::Error,
) -> Error {
    let (kind, retry) = if source.is_timeout() {
        (ErrorKind::Timeout, RetryClassification::Never)
    } else {
        (ErrorKind::Network, RetryClassification::Safe)
    };
    Error::new(kind, message)
        .with_provider(provider.clone())
        .with_retry(retry)
        .with_source(source)
}

/// Fails a stream that waits longer than `timeout` for its next item.
///
/// A provider that stops sending bytes mid-generation would otherwise stall
/// the caller forever, because nothing below this layer bounds a read. The
/// expiry is retryable, matching a dropped connection: from outside, nothing
/// separates a stalled provider from a lost one. The stream ends after the
/// expiry, so one stall produces one error rather than one per idle period.
///
/// A `timeout` of `None` waits forever.
pub(super) fn with_idle_timeout<T, S>(
    stream: S,
    idle: Option<Duration>,
    provider: ProviderId,
) -> impl Stream<Item = Result<T, Error>> + Send + 'static
where
    S: Stream<Item = Result<T, Error>> + Send + 'static,
    T: Send + 'static,
{
    unfold((Box::pin(stream), false), move |(mut stream, finished)| {
        let provider = provider.clone();
        async move {
            if finished {
                return None;
            }
            let Some(idle) = idle else {
                return stream.next().await.map(|item| (item, (stream, false)));
            };
            match timeout(idle, stream.next()).await {
                Ok(Some(item)) => Some((item, (stream, false))),
                Ok(None) => None,
                Err(source) => {
                    let error = Error::new(
                        ErrorKind::Timeout,
                        format!("provider {provider} stopped sending stream data"),
                    )
                    .with_provider(provider)
                    .with_retry(RetryClassification::Safe)
                    .with_source(source);
                    Some((Err(error), (stream, true)))
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use futures_util::StreamExt as _;
    use futures_util::stream::{self, iter, pending};

    use super::{bound_event_data, frame_stream, with_idle_timeout};
    use crate::catalog::ProviderId;
    use crate::transport::StreamFraming;
    use crate::transport::sse::{SseEvent, SseFramer};
    use crate::types::{Error, ErrorKind, RetryClassification};

    #[tokio::test]
    async fn invisible_provider_data_also_consumes_the_output_budget() {
        let events = stream::iter((0..10).map(|_| {
            Ok(SseEvent {
                event: None,
                data:  "signature".repeat(10),
            })
        }));
        let items: Vec<_> = bound_event_data(events, 100).collect().await;
        assert_eq!(items.len(), 2);
        assert!(items[0].is_ok());
        assert_eq!(
            items[1].as_ref().expect_err("cumulative event data").kind(),
            ErrorKind::ResourceLimit
        );
    }

    #[tokio::test]
    async fn frame_limit_applies_across_chunks_but_not_across_frames() {
        let chunks = stream::iter([Ok(b"data: 1\n\ndata: 2\n\n".to_vec())]);
        let frames: Vec<_> = frame_stream(chunks, SseFramer::new(StreamFraming::Sse, 9))
            .collect()
            .await;
        assert_eq!(frames.len(), 2);
        assert!(frames.iter().all(Result::is_ok));

        let chunks = stream::iter([Ok(b"data: ".to_vec()), Ok(b"1234567890".to_vec())]);
        let frames: Vec<_> = frame_stream(chunks, SseFramer::new(StreamFraming::Sse, 9))
            .collect()
            .await;
        assert_eq!(frames.len(), 1);
        assert_eq!(
            frames[0]
                .as_ref()
                .expect_err("oversized unterminated frame")
                .kind(),
            ErrorKind::ResourceLimit
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_stalled_stream_fails_with_a_retryable_timeout() {
        let stalled = pending::<Result<Vec<u8>, Error>>();

        let mut stream = Box::pin(with_idle_timeout(
            stalled,
            Some(Duration::from_millis(10)),
            ProviderId::new("alpha"),
        ));
        let error = stream
            .next()
            .await
            .expect("the stall should produce one event")
            .expect_err("the event should be an error");

        assert_eq!(error.kind(), ErrorKind::Timeout);
        assert_eq!(error.retry_classification(), RetryClassification::Safe);
        assert!(stream.next().await.is_none(), "one stall, one error");
    }

    #[tokio::test(start_paused = true)]
    async fn an_idle_timeout_leaves_a_flowing_stream_alone() {
        let chunks = iter(vec![Ok::<_, Error>(vec![b'a']), Ok(vec![b'b'])]);

        let items: Vec<_> = with_idle_timeout(
            chunks,
            Some(Duration::from_secs(30)),
            ProviderId::new("alpha"),
        )
        .collect()
        .await;

        assert_eq!(items.len(), 2);
        assert!(items.iter().all(Result::is_ok));
    }

    #[tokio::test]
    async fn a_chunk_error_is_delivered_and_ends_the_stream_without_a_flush() {
        let chunks = stream::iter(vec![
            Ok(b"data: 1\n\n".to_vec()),
            Err(Error::new(ErrorKind::Network, "dropped")),
            Ok(b"data: 2\n\n".to_vec()),
        ]);

        let frames: Vec<_> = frame_stream(chunks, SseFramer::new(StreamFraming::Sse, 64))
            .collect()
            .await;

        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].as_ref().expect("the first frame").data, "1");
        assert_eq!(
            frames[1].as_ref().expect_err("the read failure").kind(),
            ErrorKind::Network
        );
    }
}
