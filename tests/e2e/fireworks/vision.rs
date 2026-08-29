//! V2 — image input.
//!
//! Every roster model whose row claims images receives the same public cat
//! photo. Fireworks documents URL and base64 inputs; this stable URL avoids
//! pinning the decoder behavior of a malformed historical PNG fixture.

use lithos_llm::types::{ContentPart, ImageContent, MediaSource, Message, Role};

use crate::fireworks::{self, model_tests};
use crate::support::{self, TestResult};

const CAT_PHOTO_URL: &str = "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg";

mod url {
    use super::*;

    model_tests!(super::describes_a_url_image);
}

async fn describes_a_url_image(model: &str) -> TestResult {
    if !fireworks::capabilities(model).images {
        return support::skip("the catalog does not claim images");
    }
    let Some(client) = fireworks::live_client() else {
        return support::skip("FIREWORKS_API_KEY is unset");
    };
    let request = fireworks::request(model)
        .message(Message::new(Role::User, [
            ContentPart::Text {
                text: "What animal is in this photo? Answer with the animal name.".to_owned(),
            },
            ContentPart::Image(ImageContent {
                source: MediaSource::url(CAT_PHOTO_URL),
                detail: None,
            }),
        ]))
        .build()?;
    let response = client.complete(request).await?;
    assert!(
        response.text().to_lowercase().contains("cat"),
        "the model did not see the cat: {:?}",
        response.text()
    );
    Ok(())
}
