//! Provider-neutral language model types, catalog data, and runtime client.
//!
//! The catalog and the local token estimator stay synchronous. Enable
//! `runtime` to use the async client. Applications own the Tokio runtime and
//! tracing subscriber.

pub mod catalog;
pub mod estimate;
pub mod resolver;
pub mod types;

mod cost;
mod evaluation;

#[cfg(feature = "runtime")]
pub mod adapter;
#[cfg(feature = "runtime")]
pub mod client;
#[cfg(feature = "runtime")]
pub mod credentials;
#[cfg(feature = "runtime")]
pub mod middleware;

#[cfg(feature = "runtime")]
mod codecs;
#[cfg(feature = "runtime")]
mod providers;
#[cfg(feature = "runtime")]
mod transport;

#[cfg(feature = "runtime")]
#[doc(inline)]
pub use client::{Client, ClientBuild, ClientBuilder, StructuredCompletion};
#[doc(inline)]
pub use evaluation::{Evaluation, EvaluationBuilder, Verdict};
#[doc(inline)]
pub use types::{Error, Request, Response};
