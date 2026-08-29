//! V2 — image input.
//!
//! The inline test runs on every roster model whose row claims `images`,
//! with a solid red square generated for this suite, so the expected answer
//! is unambiguous. Gemini's `fileData.fileUri` expects a Files API URI, not an
//! arbitrary public URL, so this suite deliberately uses inline bytes.

use lithos_llm::types::{ContentPart, ImageContent, MediaSource, Message, Role};

use crate::gemini::{self, model_tests};
use crate::support::{self, TestResult};

/// A 64x64 solid red PNG, generated for this suite.
///
/// The dimensions keep the request small while remaining easy to identify.
const RED_SQUARE_PNG_BASE64: &str = "iVBORw0KGgoAAAAN\
SUhEUgAAAEAAAABACAIAAAAlC+aJAAAAS0lEQVR42u3PQQkAAAgAsetfWiP4FgYrsKZeS0BAQEBA\
QEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEDgsqnc8OJg6Ln3AAAAAElF\
TkSuQmCC";

mod inline {
    use super::*;

    model_tests!(super::describes_an_inline_image);
}

async fn describes_an_inline_image(model: &str) -> TestResult {
    if !gemini::capabilities(model).images {
        return support::skip("the catalog does not claim images");
    }
    let Some(client) = gemini::live_client() else {
        return support::skip("GEMINI_API_KEY is unset");
    };
    let request = gemini::request(model)
        .message(Message::new(Role::User, [
            ContentPart::Text {
                text: "What single color fills this image? Answer with the color name.".to_owned(),
            },
            ContentPart::Image(ImageContent {
                source: MediaSource::base64(RED_SQUARE_PNG_BASE64, "image/png"),
                detail: None,
            }),
        ]))
        .build()?;
    let response = client.complete(request).await?;
    assert!(
        response.text().to_lowercase().contains("red"),
        "the model did not see a red image: {:?}",
        response.text()
    );
    Ok(())
}
