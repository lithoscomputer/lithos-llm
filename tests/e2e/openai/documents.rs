//! V2 — document input.
//!
//! Documents ride the `input_file` content item, and this suite validates the
//! two wire shapes the codec emits (probed live per row on 2026-08-30):
//! `file_data` + `filename` for inline bytes — `filename` is required, a
//! request without it draws 400 "Missing required parameter", which is why
//! the codec refuses an unnamed inline document before dispatch — and
//! `file_url` for a remote fetch, which stands alone with no name.
//!
//! The inline test runs on every roster model — each row claims `documents` —
//! with a one-page PDF generated for this suite, so the expected answer is
//! unambiguous. The URL variant depends on an external host staying up, so it
//! runs on two representatives only.
//!
//! Live-only for now: the pinned twin rejects `input_file` parts before
//! scenario matching, so these cells can neither record nor replay. The fix
//! is lithoscomputer/twins#7; when the pin moves past it, drop the
//! `live_only` gates and record this module.

use lithos_llm::types::{ContentPart, DocumentContent, MediaSource, Message, Role};

use crate::openai::{self, model_tests};
use crate::support::{self, TestResult};

/// A one-page PDF whose only text is "The secret word is quartzite.",
/// generated for this suite.
const SECRET_WORD_PDF_BASE64: &str = "JVBERi0xLjQKMSAwIG9iago8PCAvVHlwZSAvQ2F0YWxvZyAvUGFnZXMgMiAwIFIgPj4KZW5kb2Jq\
CjIgMCBvYmoKPDwgL1R5cGUgL1BhZ2VzIC9LaWRzIFszIDAgUl0gL0NvdW50IDEgPj4KZW5kb2Jq\
CjMgMCBvYmoKPDwgL1R5cGUgL1BhZ2UgL1BhcmVudCAyIDAgUiAvTWVkaWFCb3ggWzAgMCA2MTIg\
NzkyXSAvQ29udGVudHMgNCAwIFIgL1Jlc291cmNlcyA8PCAvRm9udCA8PCAvRjEgNSAwIFIgPj4g\
Pj4gPj4KZW5kb2JqCjQgMCBvYmoKPDwgL0xlbmd0aCA2MCA+PgpzdHJlYW0KQlQgL0YxIDEyIFRm\
IDcyIDcyMCBUZCAoVGhlIHNlY3JldCB3b3JkIGlzIHF1YXJ0eml0ZS4pIFRqIEVUCmVuZHN0cmVh\
bQplbmRvYmoKNSAwIG9iago8PCAvVHlwZSAvRm9udCAvU3VidHlwZSAvVHlwZTEgL0Jhc2VGb250\
IC9IZWx2ZXRpY2EgPj4KZW5kb2JqCnhyZWYKMCA2CjAwMDAwMDAwMDAgNjU1MzUgZiAKMDAwMDAw\
MDAwOSAwMDAwMCBuIAowMDAwMDAwMDU4IDAwMDAwIG4gCjAwMDAwMDAxMTUgMDAwMDAgbiAKMDAw\
MDAwMDI0MSAwMDAwMCBuIAowMDAwMDAwMzUxIDAwMDAwIG4gCnRyYWlsZXIKPDwgL1NpemUgNiAv\
Um9vdCAxIDAgUiA+PgpzdGFydHhyZWYKNDIxCiUlRU9GCg==";

/// A long-stable public one-page PDF ("Dummy PDF file"), for the URL variant.
const DUMMY_PDF_URL: &str =
    "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf";

mod inline {
    use super::*;

    model_tests!(super::reads_an_inline_document);
}

mod url {
    use super::*;

    macro_rules! url_tests {
        ($($name:ident $model:literal,)+) => {
            $(
                #[tokio::test]
                #[ignore = "live OpenAI call; run with `mise run test:e2e`"]
                async fn $name() -> TestResult {
                    super::reads_a_url_document($model).await
                }
            )+
        };
    }

    url_tests!(
        gpt_5_6_terra "gpt-5.6-terra",
        gpt_5_4 "gpt-5.4",
    );
}

async fn reads_an_inline_document(model: &str) -> TestResult {
    if let Some(skip) = support::live_only("the pinned twin rejects input_file parts") {
        return skip;
    }
    if !openai::capabilities(model).documents {
        return support::skip("the catalog does not claim documents");
    }
    let Some(client) = openai::live_client() else {
        return support::skip("OPENAI_API_KEY is unset");
    };
    let request = openai::request(model)
        .message(Message::new(Role::User, [
            ContentPart::Text {
                text: "What is the secret word in this document? Answer with just the word."
                    .to_owned(),
            },
            ContentPart::Document(DocumentContent {
                source: MediaSource::base64(SECRET_WORD_PDF_BASE64, "application/pdf"),
                name:   Some("secret.pdf".to_owned()),
            }),
        ]))
        .build()?;
    let response = client.complete(request).await?;
    assert!(
        response.text().to_lowercase().contains("quartzite"),
        "the model did not read the document: {:?}",
        response.text()
    );
    Ok(())
}

async fn reads_a_url_document(model: &str) -> TestResult {
    if let Some(skip) = support::live_only("the pinned twin rejects input_file parts") {
        return skip;
    }
    let Some(client) = openai::live_client() else {
        return support::skip("OPENAI_API_KEY is unset");
    };
    let request = openai::request(model)
        .message(Message::new(Role::User, [
            ContentPart::Text {
                text: "What does this document say? Answer with its text.".to_owned(),
            },
            ContentPart::Document(DocumentContent::new(MediaSource::url(DUMMY_PDF_URL))),
        ]))
        .build()?;
    let response = client.complete(request).await?;
    assert!(
        response.text().to_lowercase().contains("dummy"),
        "the model did not fetch the document: {:?}",
        response.text()
    );
    Ok(())
}
