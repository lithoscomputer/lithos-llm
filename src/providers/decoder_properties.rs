//! Properties of every stream decoder under messy provider input.
//!
//! The seeds are real streams: the E2E recordings for Chat Completions and
//! Responses, and hand-written transcripts in
//! `tests/fixtures/stream_seeds.json` for the protocols without a recording.
//! Each case mutates one seed — dropping, duplicating, swapping, truncating, or
//! corrupting events, or inserting events from the codec's other seeds and its
//! noise list — and drives the result through the codec's decoder and the
//! production driver. Whatever the provider sends, the caller's stream must
//! keep the `StreamEvent` contract; ending in an error is fine.

use std::collections::BTreeMap;
use std::fs::read_to_string;
use std::path::Path;
use std::sync::LazyLock;

use futures_util::stream::iter;
use futures_util::{FutureExt as _, StreamExt as _};
use proptest::collection::vec;
use proptest::prelude::*;
use proptest::sample::Index;
use serde_json::Value;

use super::http::decode_stream;
use crate::catalog::{CodecId, codec_ids};
use crate::codecs::{BuiltCodec, build, test_support};
use crate::transport::SseEvent;
use crate::types::contract::violation;
use crate::types::{Error, ResponseStream, StreamEvent};

/// Real streams for one codec, and the events a mutation may insert.
struct Corpus {
    seeds: Vec<Vec<SseEvent>>,
    /// How many leading seeds are hand-written transcripts, which all end in
    /// success. A recorded seed can end in an in-band provider error.
    clean: usize,
    /// Every seed event, plus noise that no seed carries: in-band errors,
    /// unknown event types, and deltas for blocks that never opened.
    pool:  Vec<SseEvent>,
}

/// The recordings whose streams seed each codec, by recorded endpoint.
const RECORDED: [(&str, &str, &[&str]); 2] = [
    (codec_ids::OPENAI_CHAT, "chat.completions", &[
        "fireworks",
        "moonshot",
        "openrouter",
        "venice",
        "vercel",
    ]),
    (codec_ids::OPENAI_RESPONSES, "responses", &[
        "openai",
        "openai_codex",
    ]),
];

static CORPORA: LazyLock<BTreeMap<&'static str, Corpus>> = LazyLock::new(|| {
    let fixture = read_json("tests/fixtures/stream_seeds.json");
    let Value::Object(codecs) = fixture else {
        panic!("the seed fixture maps codec ids to corpora");
    };
    codecs
        .iter()
        .map(|(codec, entry)| {
            let codec = static_codec_id(codec);
            let mut seeds: Vec<Vec<SseEvent>> = events_lists(&entry["seeds"]);
            let clean = seeds.len();
            seeds.extend(recorded_seeds(codec));
            let mut pool: Vec<SseEvent> = seeds.iter().flatten().cloned().collect();
            pool.extend(events(&entry["noise"]));
            (codec, Corpus { seeds, clean, pool })
        })
        .collect()
});

fn static_codec_id(codec: &str) -> &'static str {
    [
        codec_ids::OPENAI_CHAT,
        codec_ids::OPENAI_RESPONSES,
        codec_ids::ANTHROPIC_MESSAGES,
        codec_ids::GEMINI_GENERATE,
        codec_ids::BEDROCK_CONVERSE,
    ]
    .into_iter()
    .find(|known| *known == codec)
    .unwrap_or_else(|| panic!("the seed fixture names an unknown codec: {codec}"))
}

fn read_json(relative: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(relative);
    let text = read_to_string(&path).unwrap_or_else(|error| panic!("{relative}: {error}"));
    serde_json::from_str(&text).unwrap_or_else(|error| panic!("{relative}: {error}"))
}

/// Every successful streamed transcript recorded for `codec`.
fn recorded_seeds(codec: &str) -> Vec<Vec<SseEvent>> {
    let Some((_, endpoint, files)) = RECORDED.iter().find(|(id, ..)| *id == codec) else {
        return Vec::new();
    };
    files
        .iter()
        .flat_map(|file| {
            let recording = read_json(&format!("tests/e2e/recordings/{file}.json"));
            let scenarios = recording["scenarios"]
                .as_array()
                .cloned()
                .unwrap_or_default();
            scenarios.into_iter().filter_map(|scenario| {
                let streamed = scenario["matcher"]["stream"] == true
                    && scenario["matcher"]["endpoint"] == *endpoint
                    && scenario["script"]["status"] == 200;
                streamed.then(|| events(&scenario["script"]["events"]))
            })
        })
        .collect()
}

fn events_lists(value: &Value) -> Vec<Vec<SseEvent>> {
    value.as_array().into_iter().flatten().map(events).collect()
}

/// Reads `{event, data}` objects. A string `data` is sent as is; any other
/// value is sent as its JSON text.
fn events(value: &Value) -> Vec<SseEvent> {
    value
        .as_array()
        .into_iter()
        .flatten()
        .map(|event| SseEvent {
            event: event["event"].as_str().map(ToOwned::to_owned),
            data:  match &event["data"] {
                Value::String(text) => text.clone(),
                other => other.to_string(),
            },
        })
        .collect()
}

/// Drives `events` through `codec`'s decoder and the production driver, and
/// returns what a caller of the stream receives.
fn decode(codec: &str, events: Vec<SseEvent>) -> Vec<Result<StreamEvent, Error>> {
    let Ok(BuiltCodec::Generation(codec)) = build(&CodecId::new(codec), &Value::Null) else {
        panic!("{codec} builds as a generation codec");
    };
    let route = test_support::test_route().expect("the test route resolves");
    let decoded = decode_stream(
        iter(events.into_iter().map(Ok)),
        codec.stream_decoder(&route),
    );
    ResponseStream::new(decoded)
        .collect::<Vec<_>>()
        .now_or_never()
        .expect("a stream over in-memory events never waits")
}

#[derive(Clone, Debug)]
enum Mutation {
    Drop(Index),
    Duplicate(Index),
    Swap(Index, Index),
    /// Inserts a pool event at a position.
    Insert(Index, Index),
    Truncate(Index),
    /// Cuts an event's data in half, as a proxy that splits a payload would.
    Corrupt(Index),
}

impl Mutation {
    fn apply(&self, events: &mut Vec<SseEvent>, pool: &[SseEvent]) {
        let len = events.len();
        if let Self::Insert(at, from) = self {
            events.insert(at.index(len + 1), from.get(pool).clone());
            return;
        }
        if len == 0 {
            return;
        }
        match self {
            Self::Drop(at) => {
                events.remove(at.index(len));
            }
            Self::Duplicate(at) => {
                let at = at.index(len);
                events.insert(at, events[at].clone());
            }
            Self::Swap(left, right) => events.swap(left.index(len), right.index(len)),
            Self::Truncate(at) => events.truncate(at.index(len + 1)),
            Self::Corrupt(at) => {
                let data = &mut events[at.index(len)].data;
                *data = data.chars().take(data.chars().count() / 2).collect();
            }
            Self::Insert(..) => unreachable!("handled above"),
        }
    }
}

fn mutation() -> impl Strategy<Value = Mutation> {
    prop_oneof![
        2 => any::<Index>().prop_map(Mutation::Drop),
        2 => any::<Index>().prop_map(Mutation::Duplicate),
        2 => (any::<Index>(), any::<Index>()).prop_map(|(left, right)| Mutation::Swap(left, right)),
        3 => (any::<Index>(), any::<Index>()).prop_map(|(at, from)| Mutation::Insert(at, from)),
        1 => any::<Index>().prop_map(Mutation::Truncate),
        1 => any::<Index>().prop_map(Mutation::Corrupt),
    ]
}

fn corpus(codec: &str) -> &'static Corpus {
    CORPORA
        .get(codec)
        .unwrap_or_else(|| panic!("no seed corpus for {codec}"))
}

/// Mutates one seed of `codec` and checks what the caller receives.
fn keeps_the_contract(
    codec: &str,
    seed: Index,
    mutations: &[Mutation],
) -> Result<(), TestCaseError> {
    let corpus = corpus(codec);
    let mut events = seed.get(&corpus.seeds).clone();
    for mutation in mutations {
        mutation.apply(&mut events, &corpus.pool);
    }
    let items = decode(codec, events.clone());

    prop_assert_eq!(
        violation(items.iter().map(Result::as_ref)),
        None,
        "input: {:?}",
        events
    );
    prop_assert!(
        matches!(items.last(), Some(Ok(StreamEvent::Ended { .. }) | Err(_))),
        "the stream ends with a terminal item: {:?}",
        items.last()
    );
    Ok(())
}

/// Every unmutated seed keeps the contract, so a mutated case that fails is
/// failing because of its mutations, and every hand-written seed succeeds.
#[test]
fn every_seed_keeps_the_contract() {
    for (codec, corpus) in CORPORA.iter() {
        if *codec == codec_ids::BEDROCK_CONVERSE && cfg!(not(feature = "bedrock")) {
            continue;
        }
        assert!(!corpus.seeds.is_empty(), "{codec} has seeds");
        for (index, seed) in corpus.seeds.iter().enumerate() {
            let items = decode(codec, seed.clone());
            assert_eq!(
                violation(items.iter().map(Result::as_ref)),
                None,
                "{codec} seed {index}"
            );
            let succeeded = matches!(items.last(), Some(Ok(StreamEvent::Ended { .. })));
            assert!(
                succeeded || (index >= corpus.clean && matches!(items.last(), Some(Err(_)))),
                "{codec} seed {index} ends with a terminal item: {:?}",
                items.last()
            );
        }
    }
}

proptest! {
    #[test]
    fn chat_completions_output_keeps_the_contract(
        seed in any::<Index>(),
        mutations in vec(mutation(), 0..6),
    ) {
        keeps_the_contract(codec_ids::OPENAI_CHAT, seed, &mutations)?;
    }

    #[test]
    fn responses_output_keeps_the_contract(
        seed in any::<Index>(),
        mutations in vec(mutation(), 0..6),
    ) {
        keeps_the_contract(codec_ids::OPENAI_RESPONSES, seed, &mutations)?;
    }

    #[test]
    fn anthropic_output_keeps_the_contract(
        seed in any::<Index>(),
        mutations in vec(mutation(), 0..6),
    ) {
        keeps_the_contract(codec_ids::ANTHROPIC_MESSAGES, seed, &mutations)?;
    }

    #[test]
    fn gemini_output_keeps_the_contract(
        seed in any::<Index>(),
        mutations in vec(mutation(), 0..6),
    ) {
        keeps_the_contract(codec_ids::GEMINI_GENERATE, seed, &mutations)?;
    }
}

#[cfg(feature = "bedrock")]
proptest! {
    #[test]
    fn bedrock_output_keeps_the_contract(
        seed in any::<Index>(),
        mutations in vec(mutation(), 0..6),
    ) {
        keeps_the_contract(codec_ids::BEDROCK_CONVERSE, seed, &mutations)?;
    }
}
