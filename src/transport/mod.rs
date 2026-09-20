//! HTTP dispatch for provider requests.
//!
//! [`HttpTransport`] sends one [`PreparedRequest`] and answers with either a
//! decoded JSON body or a framed event stream. The pieces live in their own
//! modules — request shapes, header assembly, body reading, SSE framing, the
//! stream pipeline, and AWS event-stream framing — and are re-exported here
//! under the names the codecs and adapters import.

#[cfg(feature = "bedrock-aws")]
pub(crate) mod aws;
mod body;
pub(crate) mod classify;
#[cfg(feature = "bedrock")]
mod event_stream;
mod headers;
mod request;
mod sse;
mod stream;

use std::time::Duration;

use futures_util::StreamExt as _;
use reqwest::{Client, RequestBuilder, Response as HttpResponse};

pub(crate) use self::body::{JsonResponse, provider_error};
use self::body::{http_error, json_response};
#[cfg(feature = "bedrock")]
use self::event_stream::EventStreamFramer;
use self::headers::rate_limits;
pub(crate) use self::request::{EncodedRequest, PreparedRequest, StreamFraming};
pub(crate) use self::sse::SseEvent;
use self::sse::SseFramer;
pub(crate) use self::stream::EventResponse;
use self::stream::{Framer, bound_event_data, chunk_error, frame_stream, with_idle_timeout};
use crate::adapter::DEFAULT_STREAM_IDLE_TIMEOUT;
use crate::catalog::CatalogProvider;
use crate::types::{Error, ErrorKind, ResponseLimits, RetryClassification};

#[derive(Clone)]
pub(crate) struct HttpTransport {
    limits:              ResponseLimits,
    client:              Client,
    /// The maximum wait between two stream chunks, or `None` to wait forever.
    stream_idle_timeout: Option<Duration>,
}

impl HttpTransport {
    /// Creates a transport with the default stream-idle timeout.
    pub(crate) fn new(client: Client) -> Self {
        Self {
            client,
            limits: ResponseLimits::default(),
            stream_idle_timeout: Some(DEFAULT_STREAM_IDLE_TIMEOUT),
        }
    }

    /// Replaces the maximum wait between two stream chunks.
    ///
    /// `None` waits forever, which only an application that bounds the call
    /// some other way should choose.
    #[must_use]
    pub(crate) fn with_stream_idle_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.stream_idle_timeout = timeout;
        self
    }

    pub(crate) fn with_response_limits(mut self, limits: ResponseLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Sends one prepared request and decodes its JSON body.
    pub(crate) async fn execute_json(
        &self,
        request: PreparedRequest,
        provider: &CatalogProvider,
    ) -> Result<JsonResponse, Error> {
        let response = self.send(request, provider, TimeoutRetry::Never).await?;
        json_response(response, provider, self.limits.body_bytes()).await
    }

    /// Sends one prepared request and opens its response as an event stream,
    /// framed as the request's [`StreamFraming`] directs.
    pub(crate) async fn stream_events(
        &self,
        request: PreparedRequest,
        provider: &CatalogProvider,
    ) -> Result<EventResponse, Error> {
        let framing = request.framing;
        let response = self.send(request, provider, TimeoutRetry::Safe).await?;
        let frame_limit = self.limits.frame_bytes();
        Ok(match framing {
            StreamFraming::Sse | StreamFraming::SseDataLines => self.framed(
                response,
                provider,
                SseFramer::new(framing, frame_limit),
                "reading the provider response stream failed",
            ),
            #[cfg(feature = "bedrock")]
            StreamFraming::AwsEventStream => self.framed(
                response,
                provider,
                EventStreamFramer::new(provider.id().clone(), frame_limit),
                "reading the Bedrock response stream failed",
            ),
        })
    }

    /// Reads the rate limits and runs the response bytes through the stream
    /// pipeline with `framer`.
    ///
    /// `read_failure` is the message a failed chunk read reports.
    fn framed(
        &self,
        response: HttpResponse,
        provider: &CatalogProvider,
        framer: impl Framer,
        read_failure: &'static str,
    ) -> EventResponse {
        let rate_limits = rate_limits(response.headers());
        let provider_id = provider.id().clone();
        let read_provider = provider_id.clone();
        let chunks = response.bytes_stream().map(move |result| {
            result.map_err(|source| chunk_error(&read_provider, read_failure, source))
        });
        let chunks = with_idle_timeout(chunks, self.stream_idle_timeout, provider_id);
        EventResponse {
            events: Box::pin(bound_event_data(
                frame_stream(chunks, framer),
                self.limits.output_bytes(),
            )),
            rate_limits,
        }
    }

    /// Sends a request whose headers and bytes are final.
    async fn send(
        &self,
        request: PreparedRequest,
        provider: &CatalogProvider,
        on_timeout: TimeoutRetry,
    ) -> Result<HttpResponse, Error> {
        let mut builder = self
            .client
            .request(request.method, &request.url)
            .headers(request.headers)
            .body(request.body);
        if let Some(timeout) = request.timeout {
            builder = builder.timeout(timeout);
        }
        finish(builder, provider, on_timeout, self.limits.body_bytes()).await
    }
}

/// Whether a request timeout on this path may be repeated.
///
/// This is one half of the crate's timeout rule; the other half lives on
/// [`ClientBuilder::default_timeout`](crate::ClientBuilder::default_timeout). A
/// timeout that expires while the provider may already be executing the call is
/// never retried, because a repeat duplicates the work and the billing. A
/// timeout that expires before any output exists — opening a stream, or failing
/// to connect at all — is safe to repeat.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TimeoutRetry {
    /// A complete-path request timeout: never retried.
    Never,
    /// A stream-establishment timeout: safe to repeat.
    Safe,
}

/// Sends one built request and turns a non-success status into an error.
async fn finish(
    builder: RequestBuilder,
    provider: &CatalogProvider,
    on_timeout: TimeoutRetry,
    body_limit: usize,
) -> Result<HttpResponse, Error> {
    let response = builder.send().await.map_err(|source| {
        // A connect timeout is a network failure, not a spent time budget:
        // the request never reached the provider, so repeating it is safe on
        // either path.
        let (kind, retry) = if source.is_timeout() && !source.is_connect() {
            let retry = match on_timeout {
                TimeoutRetry::Never => RetryClassification::Never,
                TimeoutRetry::Safe => RetryClassification::Safe,
            };
            (ErrorKind::Timeout, retry)
        } else {
            (ErrorKind::Network, RetryClassification::Safe)
        };
        Error::new(
            kind,
            format!("request to provider {} failed", provider.id()),
        )
        .with_provider(provider.id().clone())
        .with_retry(retry)
        .with_source(source)
    })?;
    if response.status().is_success() {
        return Ok(response);
    }
    Err(http_error(response, provider, body_limit).await)
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use futures_util::StreamExt as _;
    use httpmock::{Method as MockMethod, MockServer};
    use reqwest::{Client, Method};
    use serde_json::{Value, json};

    use super::{EncodedRequest, HttpTransport};
    use crate::codecs::test_support;
    use crate::credentials::Credentials;
    use crate::resolver::ResolvedRoute;
    use crate::types::{ErrorKind, RetryClassification};

    /// A route from the one-provider test catalog, whose scheme is `none`.
    fn route() -> Result<ResolvedRoute, Box<dyn StdError>> {
        test_support::test_route()
    }

    fn post(url: String) -> EncodedRequest {
        EncodedRequest::new(Method::POST, url, json!({}))
    }

    #[tokio::test]
    async fn a_malformed_success_body_is_retryable() -> Result<(), Box<dyn StdError>> {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(MockMethod::POST).path("/chat");
                then.status(200)
                    .header("content-type", "application/json")
                    .body("{\"id\":\"resp_1\", trunc");
            })
            .await;
        let route = route()?;
        let transport = HttpTransport::new(Client::new());

        let error = transport
            .execute_json(
                post(server.url("/chat")).prepare(route.provider(), &Credentials::none())?,
                route.provider(),
            )
            .await
            .expect_err("a truncated body should fail");

        assert_eq!(error.kind(), ErrorKind::ResponseDecode);
        assert_eq!(error.retry_classification(), RetryClassification::Safe);
        Ok(())
    }

    #[tokio::test]
    async fn a_non_json_error_body_still_classifies() -> Result<(), Box<dyn StdError>> {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(MockMethod::POST).path("/chat");
                then.status(400)
                    .header("content-type", "text/plain")
                    .body("The model gpt-9 does not exist");
            })
            .await;
        let route = route()?;
        let transport = HttpTransport::new(Client::new());

        let error = transport
            .execute_json(
                post(server.url("/chat")).prepare(route.provider(), &Credentials::none())?,
                route.provider(),
            )
            .await
            .expect_err("a 400 should fail");

        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert!(
            error.message().contains("The model gpt-9 does not exist"),
            "the body text should survive: {}",
            error.message()
        );
        assert_eq!(
            error.raw_data(),
            Some(&Value::String("The model gpt-9 does not exist".into()))
        );
        Ok(())
    }

    #[tokio::test]
    async fn oversized_success_and_error_bodies_are_not_retried() -> Result<(), Box<dyn StdError>> {
        use crate::types::ResponseLimits;
        let server = MockServer::start_async().await;
        for status in [200, 503] {
            let path = format!("/body-{status}");
            server
                .mock_async(|when, then| {
                    when.method(MockMethod::POST).path(&path);
                    then.status(status).body("x".repeat(1024));
                })
                .await;
            let route = route()?;
            let transport = HttpTransport::new(Client::new())
                .with_response_limits(ResponseLimits::default().max_body_bytes(100));
            let error = transport
                .execute_json(
                    post(server.url(&path)).prepare(route.provider(), &Credentials::none())?,
                    route.provider(),
                )
                .await
                .expect_err("body limit");
            assert_eq!(error.kind(), ErrorKind::ResourceLimit);
            assert_eq!(error.retry_classification(), RetryClassification::Never);
        }
        Ok(())
    }

    #[tokio::test]
    async fn a_stream_delivers_a_frame_without_its_trailing_blank_line()
    -> Result<(), Box<dyn StdError>> {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(MockMethod::POST).path("/stream");
                then.status(200)
                    .header("content-type", "text/event-stream")
                    .body("data: {\"n\":1}\n\ndata: {\"n\":2}");
            })
            .await;
        let route = route()?;
        let transport = HttpTransport::new(Client::new());

        let response = transport
            .stream_events(
                post(server.url("/stream")).prepare(route.provider(), &Credentials::none())?,
                route.provider(),
            )
            .await?;
        let events: Vec<_> = response.events.collect().await;

        assert_eq!(events.len(), 2);
        assert_eq!(
            events[1].as_ref().expect("the tail should parse").data,
            r#"{"n":2}"#
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_data_line_framed_stream_splits_events_without_blank_lines()
    -> Result<(), Box<dyn StdError>> {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(MockMethod::POST).path("/stream");
                then.status(200)
                    .header("content-type", "text/event-stream")
                    .body("data: {\"n\":1}\ndata: {\"n\":2}");
            })
            .await;
        let route = route()?;
        let transport = HttpTransport::new(Client::new());

        let response = transport
            .stream_events(
                post(server.url("/stream"))
                    .with_data_line_framing()
                    .prepare(route.provider(), &Credentials::none())?,
                route.provider(),
            )
            .await?;
        let events: Vec<_> = response.events.collect().await;

        assert_eq!(events.len(), 2);
        assert_eq!(events[0].as_ref().expect("line one").data, r#"{"n":1}"#);
        assert_eq!(
            events[1].as_ref().expect("the tail should flush").data,
            r#"{"n":2}"#
        );
        Ok(())
    }
}
