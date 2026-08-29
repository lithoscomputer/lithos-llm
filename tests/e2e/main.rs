//! The live end-to-end target.
//!
//! # Design
//!
//! Every test here drives a real [`Client`](lithos_llm::Client) against a live
//! provider API. The wire-parity target (`tests/it`) pins what this crate
//! sends and how it decodes hand-written provider payloads; this target
//! verifies the half a mock cannot: that the provider accepts our encoding
//! today, that its live payloads still decode, and that the capability claims
//! in the per-provider E2E catalog match live behavior. A capability the
//! catalog claims but the provider rejects is a red test, not a skip — the
//! suite is a drift detector for both the provider and the catalog. The plan
//! and the full test matrix live in `.ai/plans/live-e2e-test-matrix.md`.
//!
//! # Gating
//!
//! Live tests carry `#[ignore]`, so `mise run test` compiles this target but
//! spends no tokens. `mise run test:e2e` sources `.env` and runs everything
//! here, ignored tests included. A live test skips itself (passing, with a
//! note on stderr) when its provider's key variable is unset, so one missing
//! credential never fails the whole run.
//!
//! Preflight tests that touch no network — catalog validation, route
//! resolution, client-side capability rejection — are not ignored and run
//! with the routine suite.
//!
//! Nextest runs each test in its own process, so nothing in-process can cap
//! the suite's request concurrency; the `test:e2e` task passes
//! `--test-threads` for that instead.

#![cfg(feature = "openai-compatible")]

mod support;
mod venice;
