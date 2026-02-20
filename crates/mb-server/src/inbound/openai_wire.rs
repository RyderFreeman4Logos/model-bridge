use mb_core::core::{AdapterError, FinishReason, Message, MessageContent, Role, ToolChoice};
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Request wire types (OpenAI Chat Completions API)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub(super) struct OaiRequest {
    pub model: String,
    pub messages: Vec<OaiMessage>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<u64>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub frequency_penalty: Option<f64>,
    #[serde(default)]
    pub presence_penalty: Option<f64>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub tools: Option<Vec<OaiToolDef>>,
    #[serde(default)]
    pub tool_choice: Option<OaiToolChoice>,
}

#[derive(Deserialize)]
pub(super) struct OaiMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

#[derive(Deserialize)]
pub(super) struct OaiToolDef {
    pub function: OaiFunctionDef,
}

#[derive(Deserialize)]
pub(super) struct OaiFunctionDef {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default = "default_empty_object")]
    pub parameters: Value,
}

fn default_empty_object() -> Value {
    Value::Object(serde_json::Map::new())
}

#[derive(Deserialize)]
#[serde(untagged)]
pub(super) enum OaiToolChoice {
    Simple(String),
    Named { function: OaiNamedFunction },
}

#[derive(Deserialize)]
pub(super) struct OaiNamedFunction {
    pub name: String,
}

// ---------------------------------------------------------------------------
// Response wire types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub(super) struct OaiResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OaiResponseChoice>,
    pub usage: OaiUsage,
}

#[derive(Serialize)]
pub(super) struct OaiResponseChoice {
    pub index: u32,
    pub message: OaiResponseMessage,
    pub finish_reason: String,
}

#[derive(Serialize)]
pub(super) struct OaiResponseMessage {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
pub(super) struct OaiUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

// ---------------------------------------------------------------------------
// Stream wire types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub(super) struct OaiStreamChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OaiStreamChoice>,
}

#[derive(Serialize)]
pub(super) struct OaiStreamChoice {
    pub index: u32,
    pub delta: OaiDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Serialize)]
pub(super) struct OaiDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

pub(super) fn parse_role(s: &str) -> Result<Role, AdapterError> {
    match s {
        "system" => Ok(Role::System),
        "user" => Ok(Role::User),
        "assistant" => Ok(Role::Assistant),
        "tool" => Ok(Role::Tool),
        other => Err(AdapterError::ParseRequest(format!("unknown role: {other}"))),
    }
}

pub(super) fn role_to_str(role: &Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

pub(super) fn finish_reason_to_str(reason: &FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::ToolCalls => "tool_calls",
        FinishReason::ContentFilter => "content_filter",
    }
}

pub(super) fn convert_oai_message(msg: OaiMessage) -> Result<Message, AdapterError> {
    let role = parse_role(&msg.role)?;
    let content = MessageContent::Text(msg.content.unwrap_or_default());
    Ok(Message {
        role,
        content,
        name: msg.name,
        tool_call_id: msg.tool_call_id,
    })
}

pub(super) fn convert_tool_choice(tc: OaiToolChoice) -> ToolChoice {
    match tc {
        OaiToolChoice::Simple(s) => match s.as_str() {
            "auto" => ToolChoice::Auto,
            "none" => ToolChoice::None,
            "required" => ToolChoice::Required,
            other => ToolChoice::Named(other.to_owned()),
        },
        OaiToolChoice::Named { function } => ToolChoice::Named(function.name),
    }
}

pub(super) fn estimate_tokens(messages: &[Message]) -> u64 {
    let total_chars: usize = messages
        .iter()
        .map(|m| match &m.content {
            MessageContent::Text(t) => t.len(),
            MessageContent::Parts(parts) => parts
                .iter()
                .map(|p| match p {
                    mb_core::core::ContentPart::Text { text } => text.len(),
                    mb_core::core::ContentPart::ImageUrl { url, .. } => url.len(),
                })
                .sum(),
        })
        .sum();
    (total_chars / 4) as u64
}

pub(super) fn content_to_string(content: &MessageContent) -> String {
    match content {
        MessageContent::Text(t) => t.clone(),
        MessageContent::Parts(parts) => parts
            .iter()
            .filter_map(|p| match p {
                mb_core::core::ContentPart::Text { text } => Some(text.as_str()),
                mb_core::core::ContentPart::ImageUrl { .. } => None,
            })
            .collect::<Vec<_>>()
            .join(""),
    }
}
