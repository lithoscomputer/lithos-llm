//! Content translation every codec shares.
//!
//! The parts of a request that are provider-neutral facts — which messages
//! are instructions, what a tool result flattens to, how turns merge — are
//! translated here once. Each codec keeps the wire shapes that are its own.

use serde_json::{Value, json};

use super::errors::unsupported_capability;
use crate::resolver::ResolvedRoute;
use crate::types::{ContentPart, Error, FinishReason, MediaSource, Message, Request, ToolResult};

/// The signature family of Claude-minted reasoning signatures.
///
/// The Anthropic Messages and Bedrock Converse protocols both carry them, so
/// a conversation that moves between those providers keeps its signatures.
pub(crate) const ANTHROPIC_SIGNATURES: &str = "anthropic";

/// The signature family of Gemini thought signatures.
pub(crate) const GEMINI_SIGNATURES: &str = "gemini";

/// Fails the call when a request carries content this protocol cannot encode.
///
/// Silently omitting content is the worst available outcome: the provider
/// answers a prompt the caller never sent, and the caller sees a successful
/// response with no indication that their attachment was discarded. Refusing
/// before dispatch is the same rule the crate applies to custom tools.
///
/// `unencodable` names the capability for a part the codec drops, or returns
/// `None` for a part it can encode.
///
/// # Errors
///
/// Returns [`ErrorKind::InvalidRequest`](crate::types::ErrorKind::InvalidRequest)
/// naming the first unsupported capability found.
pub(crate) fn reject_unencodable(
    route: &ResolvedRoute,
    request: &Request,
    unencodable: fn(&ContentPart) -> Option<&'static str>,
) -> Result<(), Error> {
    let unsupported = request
        .messages()
        .iter()
        .flat_map(Message::content)
        .find_map(unencodable);
    match unsupported {
        Some(capability) => Err(unsupported_capability(route, capability)),
        None => Ok(()),
    }
}

/// Whether any tool result carries content this codec flattens to text.
///
/// Several protocols accept only a string for a tool result, so anything
/// [`plain_text`] does not keep is lost. That is every non-text part: media,
/// and structured JSON too, which is easy to overlook because it is not media.
/// The text still reaches the model, so this is a warning rather than a
/// refusal — unlike message content, where the caller's own attachment would
/// vanish entirely.
///
/// `carries` names the result contents this codec keeps whole. It judges each
/// result's parts together because carriage can depend on the mix: the OpenAI
/// protocols send an all-JSON result as the bare value but flatten JSON that
/// shares a result with other parts. A codec that falls back to serializing
/// the whole content still hands the model a shape it did not ask for, so
/// reporting that is not a false positive.
pub(crate) fn flattens_tool_result_content(
    request: &Request,
    carries: fn(&[ContentPart]) -> bool,
) -> bool {
    request
        .messages()
        .iter()
        .flat_map(Message::content)
        .filter_map(|part| match part {
            ContentPart::ToolResult(result) => Some(result),
            _ => None,
        })
        .any(|result| !carries(&result.content))
}

/// Concatenates every text part in order, ignoring every other part.
pub(crate) fn plain_text(parts: &[ContentPart]) -> String {
    parts
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

/// Joins the text of every system and developer message.
///
/// A message whose text is only whitespace contributes nothing — templating
/// commonly produces one, and a whitespace-only system field is a blank block
/// providers reject rather than an instruction.
pub(crate) fn system_text(messages: &[Message]) -> String {
    messages
        .iter()
        .filter(|message| message.is_instruction())
        .map(|message| plain_text(message.content()))
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Whether a system message carries content the system field cannot hold.
///
/// [`system_text`] joins the text of every system and developer message, so
/// anything else in one is dropped. The system fields of these protocols take
/// text and nothing else, which makes the drop correct — but silent, and the
/// caller put that content somewhere deliberately. The text still reaches the
/// model, so this is a warning rather than a refusal.
pub(crate) fn flattens_system_content(request: &Request) -> bool {
    request
        .messages()
        .iter()
        .filter(|message| message.is_instruction())
        .flat_map(Message::content)
        .any(|part| !matches!(part, ContentPart::Text { .. }))
}

/// Maps a provider stop reason onto the normalized finish reason.
///
/// `stop_sequence` is a normal stop: the model ended on a sequence the caller
/// asked it to stop at. Gemini's `RECITATION` is a content block, reported
/// separately from `SAFETY` because the blocked material is quoted source
/// rather than unsafe content. Converse's `model_context_window_exceeded` is
/// its own name for a generation that ran out of context, the same outcome as
/// `max_tokens`; its `content_filtered` and `guardrail_intervened` are both
/// content blocks.
pub(crate) fn finish_reason(value: Option<&str>) -> FinishReason {
    match value {
        None | Some("stop" | "end_turn" | "stop_sequence" | "STOP") => FinishReason::Stop,
        Some("length" | "max_tokens" | "MAX_TOKENS" | "model_context_window_exceeded") => {
            FinishReason::Length
        }
        Some("tool_calls" | "tool_use") => FinishReason::ToolCall,
        Some(
            "content_filter"
            | "content_filtered"
            | "guardrail_intervened"
            | "SAFETY"
            | "RECITATION"
            | "BLOCKLIST"
            | "PROHIBITED_CONTENT",
        ) => FinishReason::ContentFilter,
        Some(other) => FinishReason::Other(other.to_owned()),
    }
}

/// Corrects a finish reason for a turn that called tools.
///
/// Several protocols report a plain stop even when the turn is nothing but
/// tool calls — Gemini always, the OpenAI dialects on some skins — so a
/// consumer that dispatches tools on [`FinishReason::ToolCall`] would never
/// run them. Only `Stop` is corrected: a call cut off by a length limit, or a
/// turn blocked mid-way, keeps the reason the provider gave, which is the more
/// specific fact.
pub(crate) fn promote_tool_finish(reason: FinishReason, has_tool_call: bool) -> FinishReason {
    match reason {
        FinishReason::Stop if has_tool_call => FinishReason::ToolCall,
        other => other,
    }
}

/// Fails the call when a request carries audio.
///
/// The Anthropic Messages and Bedrock Converse protocols carry no audio input.
/// Dropping it silently would let the model answer a prompt the caller never
/// sent, so both refuse before dispatch, and their token-count paths refuse
/// exactly what their generation paths refuse.
///
/// # Errors
///
/// Returns [`ErrorKind::InvalidRequest`](crate::types::ErrorKind::InvalidRequest)
/// naming audio content.
pub(crate) fn reject_audio(route: &ResolvedRoute, request: &Request) -> Result<(), Error> {
    reject_unencodable(route, request, |part| {
        matches!(part, ContentPart::Audio(_)).then_some("audio content")
    })
}

/// Whether a tool result's parts are all text or all structured JSON.
///
/// The predicate the OpenAI dialects and Gemini pass to
/// [`flattens_tool_result_content`]: an all-JSON result travels as the bare
/// value in those protocols, so only a mix that must flatten is reported.
pub(crate) fn text_or_json_only(parts: &[ContentPart]) -> bool {
    parts
        .iter()
        .all(|part| matches!(part, ContentPart::Text { .. }))
        || parts
            .iter()
            .all(|part| matches!(part, ContentPart::Json { .. }))
}

/// The wire text of a tool result for a protocol that takes only a string.
///
/// Text wins when there is any, and text-only content sends its joined text
/// even when that is empty — a command with no output answered with nothing,
/// not with a serialized envelope. A result made only of JSON parts sends the
/// bare values instead — one value on its own, several as an array — because
/// a tool that answers with structured data means the data, not the
/// [`ContentPart`] envelope that carried it. Anything else falls back to the
/// serialized parts, which keeps mixed content readable instead of dropping
/// the half the protocol has no field for.
///
/// `unquote_lone_string` sends a lone JSON string value as its raw text
/// rather than as a JSON string literal. Chat Completions does that, as its
/// reference encoder did; Responses keeps the literal, as its reference
/// client did. Both are deliberate and both are pinned by their codec's
/// tests.
pub(crate) fn tool_result_text(result: &ToolResult, unquote_lone_string: bool) -> String {
    let text = plain_text(&result.content);
    let text_only = result
        .content
        .iter()
        .all(|part| matches!(part, ContentPart::Text { .. }));
    if !text.is_empty() || text_only {
        return text;
    }

    let values: Option<Vec<&Value>> = result
        .content
        .iter()
        .map(|part| match part {
            ContentPart::Json { value } => Some(value),
            _ => None,
        })
        .collect();
    match values.as_deref() {
        Some([Value::String(text)]) if unquote_lone_string => text.clone(),
        Some([value]) => value.to_string(),
        Some(values) if !values.is_empty() => serde_json::to_string(values).unwrap_or_default(),
        _ => serde_json::to_string(&result.content).unwrap_or_default(),
    }
}

/// Encodes a media source as the URL the OpenAI dialects accept.
///
/// Inline bytes become a `data:` URL, which is how every compatible skin
/// accepts them; there is no separate base64 shape.
pub(crate) fn data_url(source: &MediaSource) -> String {
    match source {
        MediaSource::Url { url, .. } => url.clone(),
        MediaSource::Base64 { data, media_type } => format!("data:{media_type};base64,{data}"),
    }
}

/// One conversation turn on the wire: a role and its encoded blocks.
struct Turn {
    role:   &'static str,
    blocks: Vec<Value>,
}

/// The conversation turns of a request, accumulated in wire order.
///
/// The Anthropic, Bedrock, and Gemini protocols all alternate roles, and
/// several canonical messages can map to one wire role — parallel tool
/// results are the common case, since each result is its own message but they
/// all answer one assistant turn — so consecutive same-role turns merge into
/// one rather than being sent as a run the provider rejects. A turn whose
/// parts all encoded to nothing is dropped, because every one of those
/// protocols rejects a message with empty content.
///
/// Each codec keeps its own role mapping and its own cache marker; this type
/// owns only the merging and the prefix placement.
#[derive(Default)]
pub(crate) struct Turns {
    turns: Vec<Turn>,
}

impl Turns {
    /// Appends one turn, merging it into the previous turn of the same role
    /// and dropping it when it carries no blocks.
    pub(crate) fn push(&mut self, role: &'static str, blocks: Vec<Value>) {
        if blocks.is_empty() {
            return;
        }
        match self.turns.last_mut() {
            Some(last) if last.role == role => last.blocks.extend(blocks),
            _ => self.turns.push(Turn { role, blocks }),
        }
    }

    /// The blocks of the turn a prompt-cache breakpoint should mark.
    ///
    /// The breakpoint lands on the **second-to-last** `user` turn. The prefix
    /// that ends there is exactly what the previous iteration of an agent
    /// loop wrote, so each iteration reads the cache the one before it created
    /// instead of paying to write the whole conversation again. A
    /// conversation with fewer than two user turns has no reusable prefix yet
    /// and gets none.
    pub(crate) fn prefix_cache_target(&mut self) -> Option<&mut Vec<Value>> {
        let user_turns: Vec<usize> = self
            .turns
            .iter()
            .enumerate()
            .filter(|(_, turn)| turn.role == "user")
            .map(|(index, _)| index)
            .collect();
        let target = user_turns.len().checked_sub(2).map(|nth| user_turns[nth])?;
        Some(&mut self.turns[target].blocks)
    }

    /// The turns as wire objects, with the blocks under `content_key`.
    pub(crate) fn into_values(self, content_key: &str) -> Vec<Value> {
        self.turns
            .into_iter()
            .map(|turn| json!({ "role": turn.role, content_key: turn.blocks }))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use serde_json::json;

    use super::{
        Turns, data_url, finish_reason, promote_tool_finish, text_or_json_only, tool_result_text,
    };
    use crate::types::{ContentPart, FinishReason, ImageContent, MediaSource, ToolResult};

    #[test]
    fn a_stop_sequence_is_a_normal_stop() {
        assert_eq!(finish_reason(Some("stop_sequence")), FinishReason::Stop);
    }

    #[test]
    fn recitation_is_a_content_filter() {
        assert_eq!(
            finish_reason(Some("RECITATION")),
            FinishReason::ContentFilter
        );
    }

    #[test]
    fn an_unknown_stop_reason_keeps_its_provider_spelling() {
        assert_eq!(
            finish_reason(Some("made_up_reason")),
            FinishReason::Other("made_up_reason".to_owned())
        );
    }

    #[test]
    fn converse_stop_reasons_map_onto_the_shared_table() {
        assert_eq!(
            finish_reason(Some("model_context_window_exceeded")),
            FinishReason::Length
        );
        for reason in ["content_filtered", "guardrail_intervened"] {
            assert_eq!(
                finish_reason(Some(reason)),
                FinishReason::ContentFilter,
                "{reason}"
            );
        }
    }

    #[test]
    fn only_a_plain_stop_is_promoted_to_a_tool_call() {
        assert_eq!(
            promote_tool_finish(FinishReason::Stop, true),
            FinishReason::ToolCall
        );
        assert_eq!(
            promote_tool_finish(FinishReason::Stop, false),
            FinishReason::Stop
        );
        assert_eq!(
            promote_tool_finish(FinishReason::Length, true),
            FinishReason::Length
        );
    }

    #[test]
    fn text_or_json_only_rejects_a_mix() {
        let text = ContentPart::Text {
            text: "a".to_owned(),
        };
        let json = ContentPart::Json {
            value: json!({ "a": 1 }),
        };
        assert!(text_or_json_only(&[text.clone(), text.clone()]));
        assert!(text_or_json_only(&[json.clone(), json.clone()]));
        assert!(text_or_json_only(&[]));
        assert!(!text_or_json_only(&[text, json]));
    }

    fn result(content: Vec<ContentPart>) -> ToolResult {
        ToolResult {
            tool_call_id: "call-1".to_owned(),
            name: None,
            content,
            is_error: false,
        }
    }

    #[test]
    fn a_lone_json_string_result_is_quoted_only_when_asked() {
        let lone = result(vec![ContentPart::Json {
            value: json!("done"),
        }]);
        assert_eq!(tool_result_text(&lone, true), "done");
        assert_eq!(tool_result_text(&lone, false), "\"done\"");
    }

    #[test]
    fn tool_result_text_prefers_text_then_bare_json_then_the_envelope() {
        let text_and_json = result(vec![
            ContentPart::Text {
                text: "ok".to_owned(),
            },
            ContentPart::Json {
                value: json!({ "a": 1 }),
            },
        ]);
        assert_eq!(tool_result_text(&text_and_json, false), "ok");

        let empty_text = result(vec![ContentPart::Text {
            text: String::new(),
        }]);
        assert_eq!(tool_result_text(&empty_text, false), "");

        let two_values = result(vec![
            ContentPart::Json { value: json!(1) },
            ContentPart::Json {
                value: json!({ "b": 2 }),
            },
        ]);
        assert_eq!(tool_result_text(&two_values, true), r#"[1,{"b":2}]"#);

        let mixed = result(vec![
            ContentPart::Json { value: json!(1) },
            ContentPart::Image(ImageContent::new(MediaSource::url("https://x/y.png"))),
        ]);
        let fallback = tool_result_text(&mixed, true);
        assert!(fallback.starts_with('['), "{fallback}");
        assert!(fallback.contains("y.png"), "{fallback}");
    }

    #[test]
    fn inline_media_becomes_a_data_url_and_a_url_passes_through() {
        assert_eq!(
            data_url(&MediaSource::base64("QUJD", "image/png")),
            "data:image/png;base64,QUJD"
        );
        assert_eq!(
            data_url(&MediaSource::url("https://x/y.png")),
            "https://x/y.png"
        );
    }

    #[test]
    fn turns_merge_same_role_runs_and_drop_empty_turns() {
        let mut turns = Turns::default();
        turns.push("user", vec![json!(1)]);
        turns.push("assistant", vec![]);
        turns.push("user", vec![json!(2)]);
        turns.push("assistant", vec![json!(3)]);

        assert_eq!(turns.into_values("parts"), vec![
            json!({ "role": "user", "parts": [1, 2] }),
            json!({ "role": "assistant", "parts": [3] }),
        ]);
    }

    #[test]
    fn the_cache_prefix_target_is_the_second_to_last_user_turn() -> Result<(), Box<dyn StdError>> {
        let mut turns = Turns::default();
        assert!(turns.prefix_cache_target().is_none());
        turns.push("user", vec![json!("u1")]);
        turns.push("assistant", vec![json!("a1")]);
        assert!(
            turns.prefix_cache_target().is_none(),
            "one user turn has no reusable prefix"
        );
        turns.push("user", vec![json!("u2")]);
        turns.push("assistant", vec![json!("a2")]);
        turns.push("user", vec![json!("u3")]);

        turns
            .prefix_cache_target()
            .ok_or("two user turns leave a prefix")?
            .push(json!("mark"));

        assert_eq!(
            turns.into_values("content")[2],
            json!({
                "role": "user",
                "content": ["u2", "mark"],
            })
        );
        Ok(())
    }
}
