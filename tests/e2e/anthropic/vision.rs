//! V2 — image input.
//!
//! The inline test runs on every roster model whose row claims `images`,
//! with a solid red square generated for this suite, so the expected answer
//! is unambiguous. The URL variant depends on an external host staying up,
//! so it runs on three representatives only.

use lithos_llm::types::{ContentPart, ImageContent, MediaSource, Message, Role};

use crate::anthropic::{self, model_tests};
use crate::support::{self, TestResult};

/// A 64x64 solid red PNG, generated for this suite.
///
/// The size matters: Anthropic's request validation rejected an 8x8 image
/// outright ("Supplied image did not pass validation checks"), so the square
/// is comfortably above whatever minimum that check applies.
const RED_SQUARE_PNG_BASE64: &str = "iVBORw0KGgoAAAAN\
SUhEUgAAAEAAAABACAIAAAAlC+aJAAAAS0lEQVR42u3PQQkAAAgAsetfWiP4FgYrsKZeS0BAQEBA\
QEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEDgsqnc8OJg6Ln3AAAAAElF\
TkSuQmCC";

/// A stable public Rust logo, for the URL-fetch variant.
const RUST_LOGO_URL: &str =
    "https://raw.githubusercontent.com/github/explore/main/topics/rust/rust.png";

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
                #[ignore = "live Anthropic call; run with `mise run test:e2e`"]
                async fn $name() -> TestResult {
                    super::describes_a_url_image($model).await
                }
            )+
        };
    }

    url_tests!(
        claude_fable_5_1 "claude-fable-5.1",
        claude_fable_5 "claude-fable-5",
        claude_sonnet_4_6 "claude-sonnet-4.6",
        claude_haiku_4_5 "claude-haiku-4.5",
    );
}

async fn describes_an_inline_image(model: &str) -> TestResult {
    if !anthropic::capabilities(model).images {
        return support::skip("the catalog does not claim images");
    }
    let Some(client) = anthropic::live_client() else {
        return support::skip("ANTHROPIC_API_KEY is unset");
    };
    let request = anthropic::request(model)
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
    let Some(client) = anthropic::live_client() else {
        return support::skip("ANTHROPIC_API_KEY is unset");
    };
    let request = anthropic::request(model)
        .message(Message::new(Role::User, [
            ContentPart::Text {
                text: "Which programming language uses this logo? Answer with its name.".to_owned(),
            },
            ContentPart::Image(ImageContent {
                source: MediaSource::url(RUST_LOGO_URL),
                detail: None,
            }),
        ]))
        .build()?;
    let response = client.complete(request).await?;
    assert!(
        response.text().to_lowercase().contains("rust"),
        "the model did not identify the Rust logo: {:?}",
        response.text()
    );
    Ok(())
}
