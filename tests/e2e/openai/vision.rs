//! V2 — image input.
//!
//! The inline test runs on every roster model — each row claims `images` —
//! with a solid red square generated for this suite, so the expected answer
//! is unambiguous. The URL variant depends on an external host staying up,
//! so it runs on two representatives only.

use lithos_llm::types::{ContentPart, ImageContent, MediaSource, Message, Role};

use crate::openai::{self, model_tests};
use crate::support::{self, TestResult};

/// A 64x64 solid red PNG, generated for this suite.
const RED_SQUARE_PNG_BASE64: &str = "iVBORw0KGgoAAAAN\
SUhEUgAAAEAAAABACAIAAAAlC+aJAAAAS0lEQVR42u3PQQkAAAgAsetfWiP4FgYrsKZeS0BAQEBA\
QEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEDgsqnc8OJg6Ln3AAAAAElF\
TkSuQmCC";

/// A long-stable public photo of a cat, for the URL-fetch variant.
const CAT_PHOTO_URL: &str = "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg";

mod inline {
    use super::*;

    model_tests!(super::describes_an_inline_image);
}

mod url {
    use super::*;

    macro_rules! url_tests {
        ($($name:ident $model:literal,)+) => {
            $(
                #[tokio::test]
                #[ignore = "live OpenAI call; run with `mise run test:e2e`"]
                async fn $name() -> TestResult {
                    super::describes_a_url_image($model).await
                }
            )+
        };
    }

    url_tests!(
        gpt_5_6_terra "gpt-5.6-terra",
        gpt_5_4 "gpt-5.4",
    );
}

async fn describes_an_inline_image(model: &str) -> TestResult {
    if !openai::capabilities(model).images().is_supported() {
        return support::skip("the catalog does not claim images");
    }
    let Some(client) = openai::live_client() else {
        return support::skip("OPENAI_API_KEY is unset");
    };
    let request = openai::request(model)
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

async fn describes_a_url_image(model: &str) -> TestResult {
    let Some(client) = openai::live_client() else {
        return support::skip("OPENAI_API_KEY is unset");
    };
    let request = openai::request(model)
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
