//! Per-dialect wire snapshot tests.
//!
//! One module per provider dialect. Each module drives the shared canonical
//! request corpus in `crate::support` through its own codec and pins two
//! things: the captured wire request, which is what the codec encoded, and the
//! decoded canonical result, which is what the codec decoded. Streaming tests
//! also call `support::assert_stream_contract`, which holds whether or not a
//! snapshot was accepted.
//!
//! # Snapshots
//!
//! Snapshots are external, stored under `tests/it/wire/snapshots/` and named
//! `it__wire__<dialect>__<test_fn>.snap`. A second snapshot inside one test
//! function gets a `-2` suffix. Write them with `crate::json_snapshot!(value)`
//! rather than `insta::assert_snapshot!` directly, so every snapshot gets the
//! same normalization filters.
//!
//! Review flow: run the suite, then `cargo insta pending-snapshots` to list
//! what moved, read each diff, and `cargo insta accept` once every change is
//! explained. A snapshot diff nobody can explain is a wire regression, not
//! noise.
//!
//! # Adding a dialect
//!
//! Add `tests/it/wire/<dialect>.rs` and declare it below. The contract for
//! each module is:
//!
//! - Build the catalog with `support::WireProvider`, pointed at the mock
//!   server's `base_url`, and the client with `support::client_for`, which
//!   fails the test if the provider's adapter did not construct.
//! - Mount with `support::mount_capture` for JSON, `support::mount_capture_sse`
//!   for SSE, or `support::mount_capture_event_stream` for Bedrock frames.
//! - Take the request with `support::captured` and snapshot it.
//! - Snapshot the decoded `Response`, or the `Vec<Value>` from
//!   `support::collect_stream_events`.
//! - Never edit a corpus constructor to suit one dialect. Editing one
//!   invalidates the pinned snapshots in every dialect file.

mod anthropic;
#[cfg(feature = "bedrock")]
mod bedrock;
mod gemini;
#[cfg(all(
    feature = "builtin-catalog",
    feature = "openai-compatible",
    feature = "openai"
))]
mod imported_providers;
mod openai_chat;
mod openai_responses;
