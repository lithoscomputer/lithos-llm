//! The wire-parity integration target.
//!
//! # Design
//!
//! Every test in this target points a real [`Client`](lithos_llm::Client) at a
//! local `httpmock` server. The mock matcher is a capture side-channel: an
//! `is_true` closure that always returns `true` but first writes the complete
//! received request — method, path, header set, and JSON body — into a shared
//! slot. The mock then answers with a canned provider body, and the test
//! snapshots BOTH halves of the exchange: the captured wire request, which pins
//! encoding, and the decoded canonical result, which pins decoding.
//!
//! Nothing is recorded from a live provider and nothing is replayed from a VCR
//! cassette. Every provider payload in this target is written by hand, so a
//! fixture says what a provider is expected to send rather than what one
//! happened to send on the day it was captured.
//!
//! `support` holds the harness and the shared canonical request corpus. `wire`
//! holds one module per provider dialect. All of it lives in a single
//! integration target so that the whole suite shares one compilation unit and
//! one `httpmock::MockServer` pool.

#![cfg(feature = "runtime")]

#[macro_use]
mod support;

mod wire;
