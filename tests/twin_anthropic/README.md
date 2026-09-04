# Anthropic twin hardening tests

Run `mise run test:anthropic` for the offline contract and fault suite. These tests also run in the normal `mise run test` task.

Run `mise run test:anthropic:stress` for 10,000 requests with at most 32 in flight across 32 namespaces. The nightly verification task includes this longer case.

The harness starts the actual twin-anthropic router on ephemeral loopback ports. It calls the public lithos-llm client with explicit configuration and static test credentials. No test changes the process environment or needs provider credentials. Server tasks have an owner and are stopped at teardown.

- `contracts.rs` checks exact content, tool inputs and round trips, signatures, redactions, structured output, media sources, usage, count_tokens, and beta headers. Intentional mutations must fail the same contract comparison.
- `faults.rs` supplies independently authored SSE frames through the twin's raw transport. It checks byte fragmentation, reproducible irregular chunk schedules, retries and Retry-After, malformed/truncated/disconnected streams, stalls, and cancellation.
- `state.rs` checks bounded concurrency, namespace isolation, scenario consumption, reset, and recorder append/restart/replay. A temporary directory obstructs recording writes without changing permissions or global state.

Expected client results and SSE fixtures do not use the twin's response decoder. Stream assertions check block lifecycle, delta kinds and contents, ordered block results, and a single terminal outcome. Tool execution is outside the client; the tests check that failures are not retried after visible tool output and that truncated completions expose no executable tool calls.

The current dependency uses the sibling `../twins` checkout. A clean CI checkout needs that repository or a published Git revision pin. The tested twin revision is `3c75e68252cf7d2108ac0e5de7e0d58d189ba263`.
