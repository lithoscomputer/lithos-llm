//! Provider-neutral language model types, catalog data, and runtime client.
//!
//! The catalog stays synchronous. Enable `runtime` or a provider feature to use
//! the async client. Applications own the Tokio runtime and tracing subscriber.

pub mod catalog;
pub mod resolver;
pub mod types;

#[cfg(feature = "runtime")]
pub mod adapter;
#[cfg(feature = "runtime")]
pub mod client;
#[cfg(feature = "runtime")]
pub mod credentials;
#[cfg(feature = "runtime")]
pub mod middleware;

#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "openai-compatible",
    feature = "bedrock"
))]
mod codecs;
#[cfg(feature = "runtime")]
mod providers;
#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "openai-compatible",
    feature = "bedrock"
))]
mod token_count;
#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "openai-compatible",
    feature = "bedrock"
))]
mod transport;

#[cfg(feature = "runtime")]
#[doc(inline)]
pub use client::{Client, ClientBuilder};
#[doc(inline)]
pub use types::{Error, Request, Response};
