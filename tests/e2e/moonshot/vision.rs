//! V2 — image input.
//!
//! K3 receives a solid red square generated for this suite, so the expected
//! answer is unambiguous. Moonshot documents that public image URLs are not
//! supported; the suite therefore sends only inline base64 data.

use lithos_llm::types::{ContentPart, ImageContent, MediaSource, Message, Role};

use crate::moonshot::{self, model_tests};
use crate::support::{self, TestResult};

/// A 64x64 solid red PNG, generated for this suite.
///
/// Its 64x64 dimensions keep the request small while giving the model enough
/// pixels to identify the color reliably.
const RED_SQUARE_PNG_BASE64: &str = "iVBORw0KGgoAAAAN\
SUhEUgAAAEAAAABACAIAAAAlC+aJAAAAS0lEQVR42u3PQQkAAAgAsetfWiP4FgYrsKZeS0BAQEBA\
QEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEDgsqnc8OJg6Ln3AAAAAElF\
TkSuQmCC";

mod inline {
    use super::*;

    model_tests!(super::describes_an_inline_image);
}

async fn describes_an_inline_image(model: &str) -> TestResult {
    if !moonshot::capabilities(model).images().is_supported() {
        return support::skip("the catalog does not claim images");
    }
    let Some(client) = moonshot::live_client() else {
        return support::skip("MOONSHOT_API_KEY is unset");
    };
    let request = moonshot::request(model)
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
