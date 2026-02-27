use chrono::{DateTime, Utc};
use mb_core::core::{ClientId, ModelId};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Classification of a model response by a human annotator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    /// Model refused to answer.
    Refused,
    /// Model answered but with biased/pro-government narrative.
    Biased,
    /// Model gave a satisfactory, balanced answer.
    Satisfactory,
}

/// A conversation session between a user and a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: Uuid,
    pub client_id: ClientId,
    pub model_id: ModelId,
    pub created_at: DateTime<Utc>,
}

/// A single turn (message) in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    pub id: Uuid,
    pub conversation_id: Uuid,
    pub role: TurnRole,
    pub content: String,
    pub token_count: u32,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TurnRole {
    User,
    Assistant,
    System,
}

/// A human annotation on a model response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    pub id: Uuid,
    pub turn_id: Uuid,
    pub annotator_id: String,
    pub verdict: Verdict,
    pub expected_direction: Option<String>,
    pub expected_response: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// CLA signature record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaRecord {
    pub client_id: ClientId,
    pub signed_at: DateTime<Utc>,
    pub github_username: Option<String>,
}

/// A DPO training pair exported from annotations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpoPair {
    pub prompt: String,
    pub chosen: String,
    pub rejected: String,
    pub metadata: DpoMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpoMetadata {
    pub conversation_id: Uuid,
    pub model_id: ModelId,
    pub annotator_id: String,
    pub verdict: Verdict,
    pub annotated_at: DateTime<Utc>,
}
