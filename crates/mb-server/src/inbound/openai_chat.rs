use mb_core::core::{
    AdapterError, ApiSpec, CanonicalRequest, CanonicalResponse, CanonicalStreamChunk, ClientId,
    DeltaContent, GenerationParams, InboundAdapter, ModelId, RequestId, RequestMetadata,
    ToolDefinition,
};

use super::openai_wire::{
    self, OaiDelta, OaiResponse, OaiResponseChoice, OaiResponseMessage, OaiStreamChoice,
    OaiStreamChunk, OaiUsage,
};

pub struct OpenAiChatInboundAdapter;

impl InboundAdapter for OpenAiChatInboundAdapter {
    fn api_spec(&self) -> ApiSpec {
        ApiSpec::OpenAiChat
    }

    fn parse_request(&self, body: &[u8]) -> Result<CanonicalRequest, AdapterError> {
        let oai: openai_wire::OaiRequest =
            serde_json::from_slice(body).map_err(|e| AdapterError::ParseRequest(e.to_string()))?;

        let messages = oai
            .messages
            .into_iter()
            .map(openai_wire::convert_oai_message)
            .collect::<Result<Vec<_>, _>>()?;

        let tools: Option<Vec<ToolDefinition>> = oai.tools.map(|defs| {
            defs.into_iter()
                .map(|t| ToolDefinition {
                    name: t.function.name,
                    description: t.function.description,
                    parameters: t.function.parameters,
                })
                .collect()
        });

        let tool_choice = oai.tool_choice.map(openai_wire::convert_tool_choice);

        let params = GenerationParams {
            temperature: oai.temperature,
            top_p: oai.top_p,
            max_tokens: oai.max_tokens,
            stop: oai.stop,
            frequency_penalty: oai.frequency_penalty,
            presence_penalty: oai.presence_penalty,
            seed: oai.seed,
        };

        let estimated_input_tokens = openai_wire::estimate_tokens(&messages);

        Ok(CanonicalRequest {
            model: ModelId::new(oai.model),
            messages,
            params,
            tools,
            tool_choice,
            stream: oai.stream.unwrap_or(false),
            metadata: RequestMetadata {
                request_id: RequestId::new(format!("req-{}", uuid::Uuid::new_v4())),
                client_id: ClientId::new("unknown"),
                estimated_input_tokens,
                prefix_hash: None,
            },
        })
    }

    fn format_response(&self, response: &CanonicalResponse) -> Result<Vec<u8>, AdapterError> {
        let choices: Vec<OaiResponseChoice> = response
            .choices
            .iter()
            .map(|c| OaiResponseChoice {
                index: c.index,
                message: OaiResponseMessage {
                    role: openai_wire::role_to_str(&c.message.role).to_owned(),
                    content: openai_wire::content_to_string(&c.message.content),
                },
                finish_reason: openai_wire::finish_reason_to_str(&c.finish_reason).to_owned(),
            })
            .collect();

        let oai_resp = OaiResponse {
            id: response.id.clone(),
            object: "chat.completion",
            created: response.created,
            model: response.model.as_str().to_owned(),
            choices,
            usage: OaiUsage {
                prompt_tokens: response.usage.prompt_tokens,
                completion_tokens: response.usage.completion_tokens,
                total_tokens: response.usage.total_tokens,
            },
        };

        serde_json::to_vec(&oai_resp).map_err(|e| AdapterError::FormatResponse(e.to_string()))
    }

    fn format_stream_chunk(
        &self,
        chunk: &CanonicalStreamChunk,
    ) -> Result<Option<String>, AdapterError> {
        if chunk.choices.is_empty() {
            return Ok(None);
        }

        let choices: Vec<OaiStreamChoice> = chunk
            .choices
            .iter()
            .map(|sc| match &sc.delta {
                DeltaContent::Role(role) => OaiStreamChoice {
                    index: sc.index,
                    delta: OaiDelta {
                        role: Some(openai_wire::role_to_str(role).to_owned()),
                        content: None,
                    },
                    finish_reason: None,
                },
                DeltaContent::Text(text) => OaiStreamChoice {
                    index: sc.index,
                    delta: OaiDelta {
                        role: None,
                        content: Some(text.clone()),
                    },
                    finish_reason: None,
                },
                DeltaContent::Finish(reason) => OaiStreamChoice {
                    index: sc.index,
                    delta: OaiDelta {
                        role: None,
                        content: None,
                    },
                    finish_reason: Some(openai_wire::finish_reason_to_str(reason).to_owned()),
                },
                DeltaContent::ToolCallStart { .. } | DeltaContent::ToolCallDelta { .. } => {
                    OaiStreamChoice {
                        index: sc.index,
                        delta: OaiDelta {
                            role: None,
                            content: None,
                        },
                        finish_reason: None,
                    }
                }
            })
            .collect();

        let stream_chunk = OaiStreamChunk {
            id: String::new(),
            object: "chat.completion.chunk",
            created: 0,
            model: String::new(),
            choices,
        };

        let json = serde_json::to_string(&stream_chunk)
            .map_err(|e| AdapterError::FormatResponse(e.to_string()))?;

        Ok(Some(format!("data: {json}\n\n")))
    }

    fn done_sentinel(&self) -> &str {
        "data: [DONE]"
    }
}

#[cfg(test)]
mod tests;
