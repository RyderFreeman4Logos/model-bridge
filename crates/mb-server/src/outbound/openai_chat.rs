use mb_core::core::{
    AdapterError, BackendInfo, BackendSpec, CanonicalRequest, CanonicalResponse,
    CanonicalStreamChunk, Choice, DeltaContent, FinishReason, Message, MessageContent, ModelId,
    OutboundAdapter, Role, StreamChoice, TokenUsage,
};

pub struct OpenAiChatOutboundAdapter;

impl OutboundAdapter for OpenAiChatOutboundAdapter {
    fn backend_spec(&self) -> BackendSpec {
        BackendSpec::OpenAiChat
    }

    fn build_request_body(&self, req: &CanonicalRequest) -> Result<Vec<u8>, AdapterError> {
        let messages: Vec<serde_json::Value> = req
            .messages
            .iter()
            .map(|m| {
                let mut msg = serde_json::json!({
                    "role": role_to_str(&m.role),
                    "content": content_to_json(&m.content),
                });
                if let Some(name) = &m.name {
                    msg["name"] = serde_json::Value::String(name.clone());
                }
                if let Some(id) = &m.tool_call_id {
                    msg["tool_call_id"] = serde_json::Value::String(id.clone());
                }
                msg
            })
            .collect();

        let mut body = serde_json::json!({
            "model": req.model.as_str(),
            "messages": messages,
            "stream": req.stream,
        });

        let obj = body.as_object_mut().expect("just created as object");
        if let Some(t) = req.params.temperature {
            obj.insert("temperature".into(), t.into());
        }
        if let Some(p) = req.params.top_p {
            obj.insert("top_p".into(), p.into());
        }
        if let Some(m) = req.params.max_tokens {
            obj.insert("max_tokens".into(), m.into());
        }
        if let Some(s) = &req.params.stop {
            obj.insert("stop".into(), serde_json::json!(s));
        }
        if let Some(f) = req.params.frequency_penalty {
            obj.insert("frequency_penalty".into(), f.into());
        }
        if let Some(p) = req.params.presence_penalty {
            obj.insert("presence_penalty".into(), p.into());
        }
        if let Some(s) = req.params.seed {
            obj.insert("seed".into(), s.into());
        }
        if let Some(tools) = &req.tools {
            let tools_json: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                    })
                })
                .collect();
            obj.insert("tools".into(), serde_json::json!(tools_json));
        }
        if let Some(tc) = &req.tool_choice {
            obj.insert("tool_choice".into(), tool_choice_to_json(tc));
        }

        serde_json::to_vec(&body).map_err(|e| AdapterError::FormatResponse(e.to_string()))
    }

    fn parse_response(&self, body: &[u8]) -> Result<CanonicalResponse, AdapterError> {
        let resp: OaiResponseWire =
            serde_json::from_slice(body).map_err(|e| AdapterError::ParseRequest(e.to_string()))?;

        let choices = resp
            .choices
            .into_iter()
            .map(|c| {
                Ok(Choice {
                    index: c.index,
                    message: Message {
                        role: parse_role(&c.message.role)?,
                        content: MessageContent::Text(c.message.content.unwrap_or_default()),
                        name: None,
                        tool_call_id: None,
                    },
                    finish_reason: parse_finish_reason(&c.finish_reason)?,
                })
            })
            .collect::<Result<Vec<_>, AdapterError>>()?;

        Ok(CanonicalResponse {
            id: resp.id,
            model: ModelId::new(resp.model),
            choices,
            usage: TokenUsage {
                prompt_tokens: resp.usage.prompt_tokens,
                completion_tokens: resp.usage.completion_tokens,
                total_tokens: resp.usage.total_tokens,
            },
            created: resp.created,
        })
    }

    fn parse_stream_line(&self, line: &str) -> Result<Option<CanonicalStreamChunk>, AdapterError> {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }

        let data = trimmed.strip_prefix("data: ").unwrap_or(trimmed);
        if data == "[DONE]" {
            return Ok(None);
        }

        let chunk: OaiStreamWire =
            serde_json::from_str(data).map_err(|e| AdapterError::ParseRequest(e.to_string()))?;

        let choices = chunk
            .choices
            .into_iter()
            .map(|c| {
                let delta = if let Some(reason) = c.finish_reason {
                    DeltaContent::Finish(parse_finish_reason(&reason)?)
                } else if let Some(role) = c.delta.role {
                    DeltaContent::Role(parse_role(&role)?)
                } else if let Some(text) = c.delta.content {
                    DeltaContent::Text(text)
                } else {
                    return Ok(None);
                };
                Ok(Some(StreamChoice {
                    index: c.index,
                    delta,
                }))
            })
            .filter_map(Result::transpose)
            .collect::<Result<Vec<_>, AdapterError>>()?;

        if choices.is_empty() {
            return Ok(None);
        }

        Ok(Some(CanonicalStreamChunk { choices }))
    }

    fn extra_headers(&self, _backend: &BackendInfo) -> Vec<(String, String)> {
        vec![("Content-Type".to_owned(), "application/json".to_owned())]
    }

    fn inference_path(&self) -> &str {
        "/v1/chat/completions"
    }
}

// ---------------------------------------------------------------------------
// Response wire types (Deserialize only — for parsing backend responses)
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct OaiResponseWire {
    id: String,
    model: String,
    choices: Vec<OaiChoiceWire>,
    usage: OaiUsageWire,
    created: u64,
}

#[derive(serde::Deserialize)]
struct OaiChoiceWire {
    index: u32,
    message: OaiMessageWire,
    finish_reason: String,
}

#[derive(serde::Deserialize)]
struct OaiMessageWire {
    role: String,
    content: Option<String>,
}

#[derive(serde::Deserialize)]
struct OaiUsageWire {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
}

// ---------------------------------------------------------------------------
// Stream wire types (Deserialize only)
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct OaiStreamWire {
    choices: Vec<OaiStreamChoiceWire>,
}

#[derive(serde::Deserialize)]
struct OaiStreamChoiceWire {
    index: u32,
    delta: OaiDeltaWire,
    finish_reason: Option<String>,
}

#[derive(serde::Deserialize)]
struct OaiDeltaWire {
    role: Option<String>,
    content: Option<String>,
}

// ---------------------------------------------------------------------------
// Conversion helpers
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

fn parse_finish_reason(s: &str) -> Result<FinishReason, AdapterError> {
    match s {
        "stop" => Ok(FinishReason::Stop),
        "length" => Ok(FinishReason::Length),
        "tool_calls" => Ok(FinishReason::ToolCalls),
        "content_filter" => Ok(FinishReason::ContentFilter),
        other => Err(AdapterError::ParseRequest(format!(
            "unknown finish_reason: {other}"
        ))),
    }
}

fn content_to_json(content: &MessageContent) -> serde_json::Value {
    match content {
        MessageContent::Text(t) => serde_json::Value::String(t.clone()),
        MessageContent::Parts(parts) => {
            let arr: Vec<serde_json::Value> = parts
                .iter()
                .map(|p| match p {
                    mb_core::core::ContentPart::Text { text } => {
                        serde_json::json!({"type": "text", "text": text})
                    }
                    mb_core::core::ContentPart::ImageUrl { url, detail } => {
                        let mut img =
                            serde_json::json!({"type": "image_url", "image_url": {"url": url}});
                        if let Some(d) = detail {
                            img["image_url"]["detail"] = serde_json::json!(d);
                        }
                        img
                    }
                })
                .collect();
            serde_json::Value::Array(arr)
        }
    }
}

fn tool_choice_to_json(tc: &mb_core::core::ToolChoice) -> serde_json::Value {
    match tc {
        mb_core::core::ToolChoice::Auto => serde_json::Value::String("auto".into()),
        mb_core::core::ToolChoice::None => serde_json::Value::String("none".into()),
        mb_core::core::ToolChoice::Required => serde_json::Value::String("required".into()),
        mb_core::core::ToolChoice::Named(name) => {
            serde_json::json!({"type": "function", "function": {"name": name}})
        }
    }
}

#[cfg(test)]
mod tests;
