//! V2 — image input.
//!
//! The inline test runs on every roster model whose row claims `images`,
//! with a solid red square generated for this suite, so the expected answer
//! is unambiguous. The URL variant depends on an external host staying up,
//! so it runs on three representatives only.

use lithos_llm::types::{ContentPart, ImageContent, MediaSource, Message, Role};

use crate::support::{self, TestResult};
use crate::vercel::{self, model_tests};

/// A 64x64 solid red PNG, generated for this suite with the standard
/// library's zlib and a proper IHDR, IDAT, and IEND.
///
/// The bytes matter: the fixture the OpenRouter suite carries decodes fine
/// in Python but the Grok and OpenAI upstreams behind this gateway rejected
/// it as an invalid PNG on 2026-09-17, while this one was read as red by
/// every image row.
const RED_SQUARE_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAS0lEQVR42u3PQQkA\
AAgAsetfWiP4FgYrsKZeS0BAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBA\
QEBAQEBAQEBAQEBAQEDgsqnc8OJg6Ln3AAAAAElFTkSuQmCC";

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
                #[ignore = "live Vercel AI Gateway call; run with `mise run test:e2e`"]
                async fn $name() -> TestResult {
                    super::describes_a_url_image($model).await
                }
            )+
        };
    }

    url_tests!(
        grok_4_6 "grok-4.6",
        claude_sonnet_5 "claude-sonnet-5",
        gemini_3_5_flash "gemini-3.5-flash",
    );
}

async fn describes_an_inline_image(model: &str) -> TestResult {
    if !vercel::capabilities(model).images().is_supported() {
        return support::skip("the catalog does not claim images");
    }
    let Some(client) = vercel::live_client() else {
        return support::skip("AI_GATEWAY_API_KEY is unset");
    };
    let request = vercel::request(model)
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
    let Some(client) = vercel::live_client() else {
        return support::skip("AI_GATEWAY_API_KEY is unset");
    };
    let request = vercel::request(model)
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
