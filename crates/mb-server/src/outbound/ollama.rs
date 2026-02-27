use mb_core::core::{
    AdapterError, BackendInfo, BackendSpec, CanonicalRequest, CanonicalResponse,
    CanonicalStreamChunk, Choice, DeltaContent, FinishReason, Message, MessageContent, ModelId,
    OutboundAdapter, Role, StreamChoice, TokenUsage,
};

pub struct OllamaOutboundAdapter;

impl OutboundAdapter for OllamaOutboundAdapter {
    fn backend_spec(&self) -> BackendSpec {
        BackendSpec::Ollama
    }

    fn build_request_body(&self, req: &CanonicalRequest) -> Result<Vec<u8>, AdapterError> {
        let messages: Vec<serde_json::Value> = req
            .messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": role_to_str(&m.role),
                    "content": content_to_text(&m.content),
                })
            })
            .collect();

        let mut body = serde_json::json!({
            "model": req.model.as_str(),
            "messages": messages,
            "stream": req.stream,
        });

        let obj = body.as_object_mut().ok_or_else(|| {
            AdapterError::FormatResponse("internal: expected JSON object".to_owned())
        })?;

        // Ollama uses an "options" sub-object for generation parameters.
        let mut options = serde_json::Map::new();
        if let Some(t) = req.params.temperature {
            options.insert("temperature".into(), t.into());
        }
        if let Some(p) = req.params.top_p {
            options.insert("top_p".into(), p.into());
        }
        if let Some(s) = req.params.seed {
            options.insert("seed".into(), s.into());
        }
        if let Some(fp) = req.params.frequency_penalty {
            options.insert("frequency_penalty".into(), fp.into());
        }
        if let Some(pp) = req.params.presence_penalty {
            options.insert("presence_penalty".into(), pp.into());
        }
        if let Some(stop) = &req.params.stop {
            options.insert("stop".into(), serde_json::json!(stop));
        }
        if !options.is_empty() {
            obj.insert("options".into(), serde_json::Value::Object(options));
        }

        // Ollama uses "num_predict" at the top level for max tokens.
        if let Some(m) = req.params.max_tokens {
            obj.insert("num_predict".into(), m.into());
        }

        serde_json::to_vec(&body).map_err(|e| AdapterError::FormatResponse(e.to_string()))
    }

    fn parse_response(&self, body: &[u8]) -> Result<CanonicalResponse, AdapterError> {
        let resp: OllamaResponseWire =
            serde_json::from_slice(body).map_err(|e| AdapterError::ParseRequest(e.to_string()))?;

        let content = resp.message.content.unwrap_or_default();
        let role = parse_role(&resp.message.role)?;

        let prompt_tokens = resp.usage.prompt_eval_count.unwrap_or(0);
        let completion_tokens = resp.usage.eval_count.unwrap_or(0);
        let usage = TokenUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
        };

        Ok(CanonicalResponse {
            id: String::new(),
            model: ModelId::new(resp.model),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role,
                    content: MessageContent::Text(content),
                    name: None,
                    tool_call_id: None,
                },
                finish_reason: if resp.done.unwrap_or(true) {
                    FinishReason::Stop
                } else {
                    FinishReason::Length
                },
            }],
            usage,
            created: 0,
        })
    }

    fn parse_stream_line(&self, line: &str) -> Result<Option<CanonicalStreamChunk>, AdapterError> {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }

        // Ollama streams raw JSON per line (no "data: " prefix).
        let chunk: OllamaStreamWire =
            serde_json::from_str(trimmed).map_err(|e| AdapterError::ParseRequest(e.to_string()))?;

        if chunk.done.unwrap_or(false) {
            return Ok(Some(CanonicalStreamChunk {
                choices: vec![StreamChoice {
                    index: 0,
                    delta: DeltaContent::Finish(FinishReason::Stop),
                }],
            }));
        }

        let text = chunk.message.and_then(|m| m.content).unwrap_or_default();
        if text.is_empty() {
            return Ok(None);
        }

        Ok(Some(CanonicalStreamChunk {
            choices: vec![StreamChoice {
                index: 0,
                delta: DeltaContent::Text(text),
            }],
        }))
    }

    fn extra_headers(&self, _backend: &BackendInfo) -> Vec<(String, String)> {
        vec![]
    }

    fn inference_path(&self) -> &str {
        "/api/chat"
    }
}

// ---------------------------------------------------------------------------
// Response wire types (Deserialize only)
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct OllamaResponseWire {
    model: String,
    message: OllamaMessageWire,
    done: Option<bool>,
    #[serde(flatten)]
    usage: OllamaUsageWire,
}

#[derive(serde::Deserialize)]
struct OllamaMessageWire {
    role: String,
    content: Option<String>,
}

#[derive(serde::Deserialize)]
struct OllamaUsageWire {
    prompt_eval_count: Option<u64>,
    eval_count: Option<u64>,
}

// ---------------------------------------------------------------------------
// Stream wire types (Deserialize only)
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct OllamaStreamWire {
    message: Option<OllamaMessageWire>,
    done: Option<bool>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn role_to_str(role: &Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

fn parse_role(s: &str) -> Result<Role, AdapterError> {
    match s {
        "system" => Ok(Role::System),
        "user" => Ok(Role::User),
        "assistant" => Ok(Role::Assistant),
        "tool" => Ok(Role::Tool),
        other => Err(AdapterError::ParseRequest(format!("unknown role: {other}"))),
    }
}

fn content_to_text(content: &MessageContent) -> String {
    match content {
        MessageContent::Text(t) => t.clone(),
        MessageContent::Parts(parts) => parts
            .iter()
            .filter_map(|p| match p {
                mb_core::core::ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

#[cfg(test)]
mod tests;
