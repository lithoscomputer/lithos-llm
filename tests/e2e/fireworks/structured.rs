//! V2 — structured output.
//!
//! Every reachable roster row accepted strict JSON Schema during verification,
//! so JSON mode and schema mode run in complete and streaming forms.

use lithos_llm::types::ResponseFormat;
use serde_json::{Value, json};

use crate::fireworks::{self, model_tests};
use crate::support::{self, TestResult};

mod json_object {
    use super::*;

    model_tests!(super::produces_parseable_json);
}

mod json_schema {
    use super::*;

    model_tests!(super::conforms_to_the_schema);
}

mod json_schema_streamed {
    use super::*;

    model_tests!(super::streams_a_conforming_document);
}

fn city_report_schema() -> ResponseFormat {
    ResponseFormat::JsonSchema {
        name:   "city_report".to_owned(),
        schema: json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "city":       { "type": "string" },
                "population": { "type": "integer" },
            },
            "required": ["city", "population"],
        }),
    }
}

fn assert_city_report(payload: &Value) {
    assert!(
        payload
            .get("city")
            .and_then(Value::as_str)
            .is_some_and(|city| !city.is_empty()),
        "the report carries no city: {payload}"
    );
    // Conformance to the schema is the assertion; plausibility is not. A
    // model has answered an unknown population with -1, which the schema's
    // bare `integer` allows.
    assert!(
        payload.get("population").is_some_and(Value::is_i64),
        "the report carries no integer population: {payload}"
    );
}

async fn produces_parseable_json(model: &str) -> TestResult {
    if !fireworks::capabilities(model)
        .response_format(&ResponseFormat::JsonSchema {
            name:   String::new(),
            schema: serde_json::Value::Null,
        })
        .is_supported()
    {
        return support::skip("the catalog does not claim structured_output");
    }
    let Some(client) = fireworks::live_client() else {
        return support::skip("FIREWORKS_API_KEY is unset");
    };
    let request = fireworks::request(model)
        .user("Give the city of the Eiffel Tower and its approximate population, as JSON.")
        .response_format(ResponseFormat::JsonObject)
        .build()?;
    let response = client.complete(request).await?;
    let payload = support::json_payload(&response)?;
    assert!(payload.is_object(), "the output is not a JSON object");
    Ok(())
}

async fn conforms_to_the_schema(model: &str) -> TestResult {
    if !fireworks::capabilities(model)
        .response_format(&ResponseFormat::JsonSchema {
            name:   String::new(),
            schema: serde_json::Value::Null,
        })
        .is_supported()
    {
        return support::skip("the catalog does not claim structured_output");
    }
    let Some(client) = fireworks::live_client() else {
        return support::skip("FIREWORKS_API_KEY is unset");
    };
    let request = fireworks::request(model)
        .user("Report the city of the Eiffel Tower and its approximate population.")
        .response_format(city_report_schema())
        .build()?;
    let response = client.complete(request).await?;
    assert_city_report(&support::json_payload(&response)?);
    Ok(())
}

async fn streams_a_conforming_document(model: &str) -> TestResult {
    if !fireworks::capabilities(model)
        .response_format(&ResponseFormat::JsonSchema {
            name:   String::new(),
            schema: serde_json::Value::Null,
        })
        .is_supported()
    {
        return support::skip("the catalog does not claim structured_output");
    }
    let Some(client) = fireworks::live_client() else {
        return support::skip("FIREWORKS_API_KEY is unset");
    };
    let request = fireworks::request(model)
        .user("Report the city of the Eiffel Tower and its approximate population.")
        .response_format(city_report_schema())
        .build()?;
    let stream = client.stream(request).await?;
    let (_, response) = support::checked_stream(stream).await?;
    assert_city_report(&support::json_payload(&response)?);
    Ok(())
}
