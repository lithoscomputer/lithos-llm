use std::collections::BTreeMap;
use std::error::Error as StdError;
use std::sync::Arc;

use async_trait::async_trait;
use aws_config::{BehaviorVersion, Region, defaults};
use aws_sdk_bedrockruntime::config::Builder as BedrockConfigBuilder;
use aws_sdk_bedrockruntime::operation::converse::ConverseOutput;
use aws_sdk_bedrockruntime::{Client as BedrockClient, types as aws};
use aws_smithy_types::{Blob, Document, Number};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use futures_util::StreamExt as _;
use futures_util::stream::{iter, unfold};
use serde_json::{Number as JsonNumber, Value as JsonValue, json, to_string};
use tokio::sync::Mutex;

use crate::adapter::{
    AdapterBuildError, AdapterContext, AdapterFactory, InputTokenCount, ProviderAdapter,
    ResolvedCall,
};
use crate::catalog::{AdapterId, AuthScheme, CatalogProvider, codec_ids};
use crate::codecs::Codec;
use crate::codecs::bedrock::BedrockConverseCodec;
use crate::credentials::{CredentialProvider, Credentials};
use crate::token_count::estimate_input_tokens;
use crate::transport::HttpTransport;
use crate::types::{
    ContentPart, Error, ErrorKind, FinishReason, Message, ReasoningContent, ReasoningEffort,
    Response, ResponseStream, RetryClassification, Role, Speed, StreamEvent, TokenCounts, ToolCall,
    ToolChoice,
};

pub(super) struct Factory;

impl AdapterFactory for Factory {
    fn create(
        &self,
        provider: &CatalogProvider,
        context: &AdapterContext,
    ) -> Result<Arc<dyn ProviderAdapter>, AdapterBuildError> {
        if provider.codec().as_str() != codec_ids::BEDROCK_CONVERSE {
            return Err(AdapterBuildError::UnsupportedCodec {
                provider: provider.id().clone(),
                codec:    provider.codec().clone(),
            });
        }
        Ok(Arc::new(BedrockAdapter {
            id:          provider.adapter().clone(),
            transport:   HttpTransport::new(context.http().clone()),
            credentials: context.credentials().clone(),
            clients:     Mutex::new(BTreeMap::new()),
        }))
    }
}

struct BedrockAdapter {
    id:          AdapterId,
    transport:   HttpTransport,
    credentials: Arc<dyn CredentialProvider>,
    clients:     Mutex<BTreeMap<String, BedrockClient>>,
}

#[async_trait]
impl ProviderAdapter for BedrockAdapter {
    fn id(&self) -> &AdapterId {
        &self.id
    }

    async fn complete(&self, call: &ResolvedCall) -> Result<Response, Error> {
        match self.resolve_credentials(call).await? {
            credentials @ Credentials::BedrockBearer(_) => {
                self.complete_with_bearer(call, credentials).await
            }
            Credentials::AwsDefaultChain { region } => self.complete_with_sdk(call, region).await,
            _ => Err(scheme_mismatch(call)),
        }
    }

    async fn stream(&self, call: &ResolvedCall) -> Result<ResponseStream, Error> {
        match self.resolve_credentials(call).await? {
            credentials @ Credentials::BedrockBearer(_) => {
                self.stream_with_bearer(call, credentials).await
            }
            Credentials::AwsDefaultChain { region } => self.stream_with_sdk(call, region).await,
            _ => Err(scheme_mismatch(call)),
        }
    }

    async fn count_input_tokens(
        &self,
        call: &ResolvedCall,
    ) -> Result<Option<InputTokenCount>, Error> {
        Ok(Some(InputTokenCount::new(estimate_input_tokens(
            call.request(),
        ))))
    }
}

impl BedrockAdapter {
    async fn resolve_credentials(&self, call: &ResolvedCall) -> Result<Credentials, Error> {
        self.credentials
            .credentials(call.route().provider())
            .await
            .map_err(|source| {
                Error::new(
                    ErrorKind::Authentication,
                    format!(
                        "credentials for provider {} could not be resolved",
                        call.route().provider().id()
                    ),
                )
                .with_provider(call.route().provider().id().clone())
                .with_source(source)
            })
    }

    async fn complete_with_bearer(
        &self,
        call: &ResolvedCall,
        credentials: Credentials,
    ) -> Result<Response, Error> {
        let codec = BedrockConverseCodec;
        let encoded = codec.encode(call, false)?;
        let result = self
            .transport
            .execute_json(encoded, call.route().provider(), credentials)
            .await?;
        let mut response = codec.decode_response(call.route(), result.body)?;
        response.rate_limits = result.rate_limits;
        response.cost = super::catalog_cost(response.usage, call.route().model().pricing());
        Ok(response)
    }

    async fn stream_with_bearer(
        &self,
        call: &ResolvedCall,
        credentials: Credentials,
    ) -> Result<ResponseStream, Error> {
        let codec = BedrockConverseCodec;
        let encoded = codec.encode(call, true)?;
        let accepted = self
            .transport
            .event_stream_events(encoded, call.route().provider(), credentials)
            .await?;
        let route = call.route().clone();
        let decoded = accepted
            .events
            .map(move |event| match event {
                Ok(event) => codec.decode_sse(&route, event).map_or_else(
                    |error| vec![Err(error)],
                    |events| events.into_iter().map(Ok).collect(),
                ),
                Err(error) => vec![Err(error)],
            })
            .flat_map(iter);
        let limits = iter(
            accepted
                .rate_limits
                .into_iter()
                .map(|rate_limits| Ok(StreamEvent::RateLimits { rate_limits })),
        );
        Ok(Box::pin(limits.chain(decoded)))
    }

    async fn complete_with_sdk(
        &self,
        call: &ResolvedCall,
        region: Option<String>,
    ) -> Result<Response, Error> {
        let client = self.sdk_client(call.route().provider(), region).await;
        let input = sdk_input(call)?;
        let output = client
            .converse()
            .model_id(call.route().model().api_model())
            .set_messages(Some(input.messages))
            .set_system(input.system)
            .set_inference_config(input.inference)
            .set_tool_config(input.tools)
            .set_additional_model_request_fields(input.additional)
            .set_performance_config(input.performance)
            .send()
            .await
            .map_err(|source| sdk_error(call, "Bedrock Converse request failed", source))?;
        let mut response = sdk_response(call, &output);
        response.cost = super::catalog_cost(response.usage, call.route().model().pricing());
        Ok(response)
    }

    async fn stream_with_sdk(
        &self,
        call: &ResolvedCall,
        region: Option<String>,
    ) -> Result<ResponseStream, Error> {
        let client = self.sdk_client(call.route().provider(), region).await;
        let input = sdk_input(call)?;
        let output = client
            .converse_stream()
            .model_id(call.route().model().api_model())
            .set_messages(Some(input.messages))
            .set_system(input.system)
            .set_inference_config(input.inference)
            .set_tool_config(input.tools)
            .set_additional_model_request_fields(input.additional)
            .set_performance_config(input.performance)
            .send()
            .await
            .map_err(|source| sdk_error(call, "Bedrock ConverseStream request failed", source))?;
        let provider = call.route().provider().id().clone();
        let state = (output.stream, BTreeMap::<i32, String>::new());
        let stream = unfold(state, move |(mut receiver, mut tool_ids)| {
            let provider = provider.clone();
            async move {
                loop {
                    match receiver.recv().await {
                        Ok(Some(event)) => {
                            if let Some(event) = sdk_stream_event(event, &mut tool_ids) {
                                return Some((Ok(event), (receiver, tool_ids)));
                            }
                        }
                        Ok(None) => return None,
                        Err(source) => {
                            let error = Error::new(
                                ErrorKind::StreamDecode,
                                "Bedrock response stream failed",
                            )
                            .with_provider(provider)
                            .with_retry(RetryClassification::Safe)
                            .with_source(source);
                            return Some((Err(error), (receiver, tool_ids)));
                        }
                    }
                }
            }
        });
        Ok(Box::pin(stream))
    }

    async fn sdk_client(
        &self,
        provider: &CatalogProvider,
        region: Option<String>,
    ) -> BedrockClient {
        let region = region.or_else(|| match provider.auth() {
            AuthScheme::Aws { region } => region.clone(),
            _ => None,
        });
        let key = region.clone().unwrap_or_else(|| "<default>".to_owned());
        if let Some(client) = self.clients.lock().await.get(&key).cloned() {
            return client;
        }
        let mut loader = defaults(BehaviorVersion::latest());
        if let Some(region) = region {
            loader = loader.region(Region::new(region));
        }
        let shared = loader.load().await;
        let config = BedrockConfigBuilder::from(&shared)
            .endpoint_url(provider.base_url())
            .build();
        let client = BedrockClient::from_conf(config);
        self.clients.lock().await.insert(key, client.clone());
        client
    }
}

struct SdkInput {
    messages:    Vec<aws::Message>,
    system:      Option<Vec<aws::SystemContentBlock>>,
    inference:   Option<aws::InferenceConfiguration>,
    tools:       Option<aws::ToolConfiguration>,
    additional:  Option<Document>,
    performance: Option<aws::PerformanceConfiguration>,
}

fn sdk_input(call: &ResolvedCall) -> Result<SdkInput, Error> {
    let request = call.request();
    let messages = request
        .messages()
        .iter()
        .filter(|message| !matches!(message.role(), Role::System | Role::Developer))
        .map(|message| {
            let role = if message.role() == Role::Assistant {
                aws::ConversationRole::Assistant
            } else {
                aws::ConversationRole::User
            };
            let content = message
                .content()
                .iter()
                .map(|part| sdk_content(call, part))
                .collect::<Result<Vec<_>, Error>>()?;
            aws::Message::builder()
                .role(role)
                .set_content(Some(content))
                .build()
                .map_err(|source| build_error(call, source))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let system_text = request
        .messages()
        .iter()
        .filter(|message| matches!(message.role(), Role::System | Role::Developer))
        .flat_map(Message::content)
        .filter_map(|part| match part {
            ContentPart::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    let system =
        (!system_text.is_empty()).then(|| vec![aws::SystemContentBlock::Text(system_text)]);
    let inference = if request.max_output_tokens().is_some()
        || request.temperature().is_some()
        || request.top_p().is_some()
    {
        Some(
            aws::InferenceConfiguration::builder()
                .set_max_tokens(
                    request
                        .max_output_tokens()
                        .and_then(|value| i32::try_from(value).ok()),
                )
                .set_temperature(request.temperature())
                .set_top_p(request.top_p())
                .build(),
        )
    } else {
        None
    };
    let tools = sdk_tools(call)?;
    let mut additional = request
        .provider_options()
        .get("bedrock")
        .and_then(JsonValue::as_object)
        .cloned()
        .unwrap_or_default();
    if let Some(effort) = request.reasoning_effort() {
        additional.insert(
            "output_config".to_owned(),
            json!({ "effort": bedrock_effort(effort) }),
        );
    }
    let additional =
        (!additional.is_empty()).then(|| json_to_document(&JsonValue::Object(additional)));
    let performance = request.speed().map(|speed| {
        aws::PerformanceConfiguration::builder()
            .latency(if matches!(speed, Speed::Fast) {
                aws::PerformanceConfigLatency::Optimized
            } else {
                aws::PerformanceConfigLatency::Standard
            })
            .build()
    });
    Ok(SdkInput {
        messages,
        system,
        inference,
        tools,
        additional,
        performance,
    })
}

fn sdk_content(call: &ResolvedCall, part: &ContentPart) -> Result<aws::ContentBlock, Error> {
    match part {
        ContentPart::Text { text } => Ok(aws::ContentBlock::Text(text.clone())),
        ContentPart::Image(image) => {
            let bytes = decode_base64(call, &image.source)?;
            let format = image
                .media_type
                .as_deref()
                .unwrap_or("image/png")
                .trim_start_matches("image/");
            let image = aws::ImageBlock::builder()
                .format(aws::ImageFormat::from(format))
                .source(aws::ImageSource::Bytes(Blob::new(bytes)))
                .build()
                .map_err(|source| build_error(call, source))?;
            Ok(aws::ContentBlock::Image(image))
        }
        ContentPart::Document(document) => {
            let format = document_format(&document.media_type);
            let document = aws::DocumentBlock::builder()
                .format(aws::DocumentFormat::from(format))
                .name(document.name.as_deref().unwrap_or("document"))
                .source(aws::DocumentSource::Bytes(Blob::new(decode_base64(
                    call,
                    &document.data,
                )?)))
                .build()
                .map_err(|source| build_error(call, source))?;
            Ok(aws::ContentBlock::Document(document))
        }
        ContentPart::Reasoning(reasoning) => {
            let block = aws::ReasoningTextBlock::builder()
                .text(&reasoning.text)
                .set_signature(reasoning.signature.clone())
                .build()
                .map_err(|source| build_error(call, source))?;
            Ok(aws::ContentBlock::ReasoningContent(
                aws::ReasoningContentBlock::ReasoningText(block),
            ))
        }
        ContentPart::ToolCall(tool) => {
            let tool = aws::ToolUseBlock::builder()
                .tool_use_id(&tool.id)
                .name(&tool.name)
                .input(json_to_document(&tool.arguments))
                .build()
                .map_err(|source| build_error(call, source))?;
            Ok(aws::ContentBlock::ToolUse(tool))
        }
        ContentPart::ToolResult(result) => {
            let content = result
                .content
                .iter()
                .map(|part| match part {
                    ContentPart::Text { text } => aws::ToolResultContentBlock::Text(text.clone()),
                    _ => aws::ToolResultContentBlock::Text(to_string(part).unwrap_or_default()),
                })
                .collect();
            let status = if result.is_error {
                aws::ToolResultStatus::Error
            } else {
                aws::ToolResultStatus::Success
            };
            let result = aws::ToolResultBlock::builder()
                .tool_use_id(&result.tool_call_id)
                .set_content(Some(content))
                .status(status)
                .build()
                .map_err(|source| build_error(call, source))?;
            Ok(aws::ContentBlock::ToolResult(result))
        }
        ContentPart::Audio(_) => Err(Error::new(
            ErrorKind::InvalidRequest,
            "Bedrock Converse does not support audio content",
        )
        .with_provider(call.route().provider().id().clone())),
    }
}

fn sdk_tools(call: &ResolvedCall) -> Result<Option<aws::ToolConfiguration>, Error> {
    let request = call.request();
    if request.tools().is_empty() || matches!(request.tool_choice(), Some(ToolChoice::None)) {
        return Ok(None);
    }
    let tools = request
        .tools()
        .iter()
        .map(|tool| {
            aws::ToolSpecification::builder()
                .name(&tool.name)
                .description(&tool.description)
                .input_schema(aws::ToolInputSchema::Json(json_to_document(
                    &tool.input_schema,
                )))
                .build()
                .map(aws::Tool::ToolSpec)
                .map_err(|source| build_error(call, source))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let choice = match request.tool_choice() {
        None | Some(ToolChoice::Auto) => Some(aws::ToolChoice::Auto(
            aws::AutoToolChoice::builder().build(),
        )),
        Some(ToolChoice::Required) => {
            Some(aws::ToolChoice::Any(aws::AnyToolChoice::builder().build()))
        }
        Some(ToolChoice::Tool { name }) => Some(aws::ToolChoice::Tool(
            aws::SpecificToolChoice::builder()
                .name(name)
                .build()
                .map_err(|source| build_error(call, source))?,
        )),
        Some(ToolChoice::None) => None,
    };
    aws::ToolConfiguration::builder()
        .set_tools(Some(tools))
        .set_tool_choice(choice)
        .build()
        .map(Some)
        .map_err(|source| build_error(call, source))
}

fn sdk_response(call: &ResolvedCall, output: &ConverseOutput) -> Response {
    let content = output
        .output()
        .and_then(|output| output.as_message().ok())
        .map_or(&[][..], aws::Message::content)
        .iter()
        .filter_map(sdk_output_content)
        .collect();
    Response {
        id: None,
        model: call.route().handle(),
        content,
        finish_reason: bedrock_finish_reason(output.stop_reason().as_str()),
        usage: output.usage().map_or_else(TokenCounts::default, sdk_usage),
        cost: None,
        rate_limits: None,
        warnings: Vec::new(),
    }
}

fn sdk_output_content(part: &aws::ContentBlock) -> Option<ContentPart> {
    match part {
        aws::ContentBlock::Text(text) => Some(ContentPart::Text { text: text.clone() }),
        aws::ContentBlock::ToolUse(tool) => Some(ContentPart::ToolCall(ToolCall {
            id:        tool.tool_use_id().to_owned(),
            name:      tool.name().to_owned(),
            arguments: document_to_json(tool.input()),
        })),
        aws::ContentBlock::ReasoningContent(reasoning) => {
            reasoning.as_reasoning_text().ok().map(|reasoning| {
                ContentPart::Reasoning(ReasoningContent {
                    text:      reasoning.text().to_owned(),
                    signature: reasoning.signature().map(ToOwned::to_owned),
                })
            })
        }
        _ => None,
    }
}

fn sdk_stream_event(
    event: aws::ConverseStreamOutput,
    tool_ids: &mut BTreeMap<i32, String>,
) -> Option<StreamEvent> {
    match event {
        aws::ConverseStreamOutput::MessageStart(_) => Some(StreamEvent::Started { id: None }),
        aws::ConverseStreamOutput::ContentBlockStart(event) => {
            let tool = event.start()?.as_tool_use().ok()?;
            tool_ids.insert(event.content_block_index(), tool.tool_use_id().to_owned());
            Some(StreamEvent::ToolCallDelta {
                id:        tool.tool_use_id().to_owned(),
                name:      Some(tool.name().to_owned()),
                arguments: String::new(),
            })
        }
        aws::ConverseStreamOutput::ContentBlockDelta(event) => match event.delta()? {
            aws::ContentBlockDelta::Text(text) => {
                Some(StreamEvent::TextDelta { text: text.clone() })
            }
            aws::ContentBlockDelta::ToolUse(tool) => Some(StreamEvent::ToolCallDelta {
                id:        tool_ids
                    .get(&event.content_block_index())
                    .cloned()
                    .unwrap_or_default(),
                name:      None,
                arguments: tool.input().to_owned(),
            }),
            aws::ContentBlockDelta::ReasoningContent(aws::ReasoningContentBlockDelta::Text(
                text,
            )) => Some(StreamEvent::ReasoningDelta { text: text.clone() }),
            _ => None,
        },
        aws::ConverseStreamOutput::MessageStop(event) => Some(StreamEvent::Finished {
            reason: bedrock_finish_reason(event.stop_reason().as_str()),
        }),
        aws::ConverseStreamOutput::Metadata(event) => {
            event.usage().map(|usage| StreamEvent::Usage {
                usage: sdk_usage(usage),
            })
        }
        _ => None,
    }
}

fn sdk_usage(usage: &aws::TokenUsage) -> TokenCounts {
    TokenCounts {
        input:            u64::try_from(usage.input_tokens()).unwrap_or_default(),
        output:           u64::try_from(usage.output_tokens()).unwrap_or_default(),
        cached_input:     usage
            .cache_read_input_tokens()
            .and_then(|tokens| u64::try_from(tokens).ok())
            .unwrap_or_default(),
        reasoning_output: 0,
    }
}

fn bedrock_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "end_turn" | "stop_sequence" => FinishReason::Stop,
        "max_tokens" => FinishReason::Length,
        "tool_use" => FinishReason::ToolCall,
        "content_filtered" | "guardrail_intervened" => FinishReason::ContentFilter,
        other => FinishReason::Other(other.to_owned()),
    }
}

fn bedrock_effort(effort: ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::Minimal | ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::Xhigh => "max",
    }
}

fn json_to_document(value: &JsonValue) -> Document {
    match value {
        JsonValue::Null => Document::Null,
        JsonValue::Bool(value) => Document::Bool(*value),
        JsonValue::String(value) => Document::String(value.clone()),
        JsonValue::Array(values) => Document::Array(values.iter().map(json_to_document).collect()),
        JsonValue::Object(values) => Document::Object(
            values
                .iter()
                .map(|(key, value)| (key.clone(), json_to_document(value)))
                .collect(),
        ),
        JsonValue::Number(value) => {
            let number = if let Some(value) = value.as_u64() {
                Number::PosInt(value)
            } else if let Some(value) = value.as_i64() {
                Number::NegInt(value)
            } else {
                Number::Float(value.as_f64().unwrap_or_default())
            };
            Document::Number(number)
        }
    }
}

fn document_to_json(value: &Document) -> JsonValue {
    match value {
        Document::Null => JsonValue::Null,
        Document::Bool(value) => JsonValue::Bool(*value),
        Document::String(value) => JsonValue::String(value.clone()),
        Document::Array(values) => JsonValue::Array(values.iter().map(document_to_json).collect()),
        Document::Object(values) => JsonValue::Object(
            values
                .iter()
                .map(|(key, value)| (key.clone(), document_to_json(value)))
                .collect(),
        ),
        Document::Number(Number::PosInt(value)) => (*value).into(),
        Document::Number(Number::NegInt(value)) => (*value).into(),
        Document::Number(Number::Float(value)) => {
            JsonNumber::from_f64(*value).map_or(JsonValue::Null, JsonValue::Number)
        }
    }
}

fn decode_base64(call: &ResolvedCall, value: &str) -> Result<Vec<u8>, Error> {
    let encoded = value.split_once(',').map_or(value, |(_, encoded)| encoded);
    STANDARD.decode(encoded).map_err(|source| {
        Error::new(
            ErrorKind::InvalidRequest,
            "Bedrock binary content is not valid base64",
        )
        .with_provider(call.route().provider().id().clone())
        .with_source(source)
    })
}

fn document_format(media_type: &str) -> &str {
    match media_type {
        "application/pdf" => "pdf",
        "text/csv" => "csv",
        "text/html" => "html",
        "text/markdown" => "md",
        "text/plain" => "txt",
        "application/msword" => "doc",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => "docx",
        "application/vnd.ms-excel" => "xls",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" => "xlsx",
        other => other,
    }
}

fn scheme_mismatch(call: &ResolvedCall) -> Error {
    Error::new(
        ErrorKind::Authentication,
        "Bedrock requires AWS default-chain or Bedrock bearer credentials",
    )
    .with_provider(call.route().provider().id().clone())
}

fn build_error(call: &ResolvedCall, source: impl StdError + Send + Sync + 'static) -> Error {
    Error::new(
        ErrorKind::InvalidRequest,
        "building the Bedrock request failed",
    )
    .with_provider(call.route().provider().id().clone())
    .with_source(source)
}

fn sdk_error(
    call: &ResolvedCall,
    message: &'static str,
    source: impl StdError + Send + Sync + 'static,
) -> Error {
    Error::new(ErrorKind::Provider, message)
        .with_provider(call.route().provider().id().clone())
        .with_retry(RetryClassification::Safe)
        .with_source(source)
}
