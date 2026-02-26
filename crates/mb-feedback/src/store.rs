use std::io::{Error as IoError, ErrorKind};
use std::path::Path;
use std::sync::Mutex;

use chrono::{DateTime, Utc};
use mb_core::core::{ClientId, ModelId};
use rusqlite::types::Type;
use rusqlite::{params, Connection, OptionalExtension};
use uuid::Uuid;

use crate::models::{Annotation, ClaRecord, Conversation, Turn, TurnRole, Verdict};

const SCHEMA_VERSION: i32 = 1;
const SCHEMA_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    client_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS turns (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_conversation ON turns(conversation_id);

CREATE TABLE IF NOT EXISTS annotations (
    id TEXT PRIMARY KEY,
    turn_id TEXT NOT NULL REFERENCES turns(id),
    annotator_id TEXT NOT NULL,
    verdict TEXT NOT NULL,
    expected_direction TEXT,
    expected_response TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_annotations_turn ON annotations(turn_id);
CREATE INDEX IF NOT EXISTS idx_annotations_annotator ON annotations(annotator_id);

CREATE TABLE IF NOT EXISTS cla_records (
    client_id TEXT PRIMARY KEY,
    signed_at TEXT NOT NULL,
    github_username TEXT
);
"#;

#[derive(Debug, thiserror::Error)]
pub enum FeedbackError {
    #[error("database error: {0}")]
    Database(#[from] rusqlite::Error),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

pub trait FeedbackStore: Send + Sync {
    fn init(&self) -> Result<(), FeedbackError>;
    fn insert_conversation(&self, conv: &Conversation) -> Result<(), FeedbackError>;
    fn insert_turn(&self, turn: &Turn) -> Result<(), FeedbackError>;
    fn insert_annotation(&self, ann: &Annotation) -> Result<(), FeedbackError>;
    fn list_annotations(&self) -> Result<Vec<Annotation>, FeedbackError>;
    fn get_annotations_by_annotator(
        &self,
        annotator_id: &str,
    ) -> Result<Vec<Annotation>, FeedbackError>;
    fn list_conversations(&self, client_id: &str) -> Result<Vec<Conversation>, FeedbackError>;
    fn get_conversation_by_id(
        &self,
        conversation_id: &Uuid,
    ) -> Result<Option<Conversation>, FeedbackError>;
    fn get_turn_by_id(&self, turn_id: &Uuid) -> Result<Option<Turn>, FeedbackError>;
    fn get_turns_for_conversation(
        &self,
        conversation_id: &Uuid,
    ) -> Result<Vec<Turn>, FeedbackError>;
    fn check_cla_status(&self, client_id: &str) -> Result<bool, FeedbackError>;
    fn record_cla_signature(&self, record: &ClaRecord) -> Result<(), FeedbackError>;
}

pub struct SqliteFeedbackStore {
    conn: Mutex<Connection>,
}

impl SqliteFeedbackStore {
    pub fn new(path: &Path) -> Result<Self, FeedbackError> {
        let conn = Connection::open(path)?;
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    pub fn new_in_memory() -> Result<Self, FeedbackError> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    fn lock_conn(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().expect("sqlite mutex poisoned")
    }
}

impl FeedbackStore for SqliteFeedbackStore {
    fn init(&self) -> Result<(), FeedbackError> {
        let conn = self.lock_conn();
        let version: i32 = conn.pragma_query_value(None, "user_version", |row| row.get(0))?;

        if version < SCHEMA_VERSION {
            conn.execute_batch(SCHEMA_SQL)?;
            conn.pragma_update(None, "user_version", SCHEMA_VERSION)?;
        } else {
            conn.execute_batch(SCHEMA_SQL)?;
        }

        Ok(())
    }

    fn insert_conversation(&self, conv: &Conversation) -> Result<(), FeedbackError> {
        let conn = self.lock_conn();
        conn.execute(
            "INSERT INTO conversations (id, client_id, model_id, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![
                conv.id.to_string(),
                conv.client_id.as_str(),
                conv.model_id.as_str(),
                conv.created_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    fn insert_turn(&self, turn: &Turn) -> Result<(), FeedbackError> {
        let conn = self.lock_conn();
        conn.execute(
            "INSERT INTO turns (id, conversation_id, role, content, token_count, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                turn.id.to_string(),
                turn.conversation_id.to_string(),
                turn_role_to_str(turn.role),
                turn.content.as_str(),
                turn.token_count,
                turn.created_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    fn insert_annotation(&self, ann: &Annotation) -> Result<(), FeedbackError> {
        let conn = self.lock_conn();
        conn.execute(
            "INSERT INTO annotations
             (id, turn_id, annotator_id, verdict, expected_direction, expected_response, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                ann.id.to_string(),
                ann.turn_id.to_string(),
                ann.annotator_id.as_str(),
                verdict_to_str(ann.verdict),
                ann.expected_direction.as_deref(),
                ann.expected_response.as_deref(),
                ann.created_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    fn list_annotations(&self) -> Result<Vec<Annotation>, FeedbackError> {
        let conn = self.lock_conn();
        let mut stmt = conn.prepare(
            "SELECT id, turn_id, annotator_id, verdict, expected_direction, expected_response, created_at
             FROM annotations
             ORDER BY created_at ASC",
        )?;

        let rows = stmt.query_map([], |row| {
            let id: String = row.get(0)?;
            let turn_id: String = row.get(1)?;
            let annotator_id: String = row.get(2)?;
            let verdict: String = row.get(3)?;
            let expected_direction: Option<String> = row.get(4)?;
            let expected_response: Option<String> = row.get(5)?;
            let created_at: String = row.get(6)?;

            Ok(Annotation {
                id: parse_uuid(0, &id)?,
                turn_id: parse_uuid(1, &turn_id)?,
                annotator_id,
                verdict: parse_verdict(3, &verdict)?,
                expected_direction,
                expected_response,
                created_at: parse_datetime_utc(6, &created_at)?,
            })
        })?;

        let annotations = rows.collect::<Result<Vec<_>, _>>()?;
        Ok(annotations)
    }

    fn get_annotations_by_annotator(
        &self,
        annotator_id: &str,
    ) -> Result<Vec<Annotation>, FeedbackError> {
        let conn = self.lock_conn();
        let mut stmt = conn.prepare(
            "SELECT id, turn_id, annotator_id, verdict, expected_direction, expected_response, created_at
             FROM annotations
             WHERE annotator_id = ?1
             ORDER BY created_at ASC",
        )?;

        let rows = stmt.query_map(params![annotator_id], |row| {
            let id: String = row.get(0)?;
            let turn_id: String = row.get(1)?;
            let annotator_id: String = row.get(2)?;
            let verdict: String = row.get(3)?;
            let expected_direction: Option<String> = row.get(4)?;
            let expected_response: Option<String> = row.get(5)?;
            let created_at: String = row.get(6)?;

            Ok(Annotation {
                id: parse_uuid(0, &id)?,
                turn_id: parse_uuid(1, &turn_id)?,
                annotator_id,
                verdict: parse_verdict(3, &verdict)?,
                expected_direction,
                expected_response,
                created_at: parse_datetime_utc(6, &created_at)?,
            })
        })?;

        let annotations = rows.collect::<Result<Vec<_>, _>>()?;
        Ok(annotations)
    }

    fn list_conversations(&self, client_id: &str) -> Result<Vec<Conversation>, FeedbackError> {
        let conn = self.lock_conn();
        let mut stmt = conn.prepare(
            "SELECT id, client_id, model_id, created_at
             FROM conversations
             WHERE client_id = ?1
             ORDER BY created_at ASC",
        )?;

        let rows = stmt.query_map(params![client_id], |row| {
            let id: String = row.get(0)?;
            let client_id: String = row.get(1)?;
            let model_id: String = row.get(2)?;
            let created_at: String = row.get(3)?;

            Ok(Conversation {
                id: parse_uuid(0, &id)?,
                client_id: ClientId::new(client_id),
                model_id: ModelId::new(model_id),
                created_at: parse_datetime_utc(3, &created_at)?,
            })
        })?;

        let conversations = rows.collect::<Result<Vec<_>, _>>()?;
        Ok(conversations)
    }

    fn get_conversation_by_id(
        &self,
        conversation_id: &Uuid,
    ) -> Result<Option<Conversation>, FeedbackError> {
        let conn = self.lock_conn();
        let conversation = conn
            .query_row(
                "SELECT id, client_id, model_id, created_at
                 FROM conversations
                 WHERE id = ?1",
                params![conversation_id.to_string()],
                |row| {
                    let id: String = row.get(0)?;
                    let client_id: String = row.get(1)?;
                    let model_id: String = row.get(2)?;
                    let created_at: String = row.get(3)?;

                    Ok(Conversation {
                        id: parse_uuid(0, &id)?,
                        client_id: ClientId::new(client_id),
                        model_id: ModelId::new(model_id),
                        created_at: parse_datetime_utc(3, &created_at)?,
                    })
                },
            )
            .optional()?;
        Ok(conversation)
    }

    fn get_turn_by_id(&self, turn_id: &Uuid) -> Result<Option<Turn>, FeedbackError> {
        let conn = self.lock_conn();
        let turn = conn
            .query_row(
                "SELECT id, conversation_id, role, content, token_count, created_at
                 FROM turns
                 WHERE id = ?1",
                params![turn_id.to_string()],
                |row| {
                    let id: String = row.get(0)?;
                    let conversation_id: String = row.get(1)?;
                    let role: String = row.get(2)?;
                    let content: String = row.get(3)?;
                    let token_count: u32 = row.get(4)?;
                    let created_at: String = row.get(5)?;

                    Ok(Turn {
                        id: parse_uuid(0, &id)?,
                        conversation_id: parse_uuid(1, &conversation_id)?,
                        role: parse_turn_role(2, &role)?,
                        content,
                        token_count,
                        created_at: parse_datetime_utc(5, &created_at)?,
                    })
                },
            )
            .optional()?;
        Ok(turn)
    }

    fn get_turns_for_conversation(
        &self,
        conversation_id: &Uuid,
    ) -> Result<Vec<Turn>, FeedbackError> {
        let conn = self.lock_conn();
        let mut stmt = conn.prepare(
            "SELECT id, conversation_id, role, content, token_count, created_at
             FROM turns
             WHERE conversation_id = ?1
             ORDER BY created_at ASC",
        )?;

        let rows = stmt.query_map(params![conversation_id.to_string()], |row| {
            let id: String = row.get(0)?;
            let conversation_id: String = row.get(1)?;
            let role: String = row.get(2)?;
            let content: String = row.get(3)?;
            let token_count: u32 = row.get(4)?;
            let created_at: String = row.get(5)?;

            Ok(Turn {
                id: parse_uuid(0, &id)?,
                conversation_id: parse_uuid(1, &conversation_id)?,
                role: parse_turn_role(2, &role)?,
                content,
                token_count,
                created_at: parse_datetime_utc(5, &created_at)?,
            })
        })?;

        let turns = rows.collect::<Result<Vec<_>, _>>()?;
        Ok(turns)
    }

    fn check_cla_status(&self, client_id: &str) -> Result<bool, FeedbackError> {
        let conn = self.lock_conn();
        let exists = conn
            .query_row(
                "SELECT 1 FROM cla_records WHERE client_id = ?1 LIMIT 1",
                params![client_id],
                |row| row.get::<_, i64>(0),
            )
            .optional()?
            .is_some();
        Ok(exists)
    }

    fn record_cla_signature(&self, record: &ClaRecord) -> Result<(), FeedbackError> {
        let conn = self.lock_conn();
        conn.execute(
            "INSERT INTO cla_records (client_id, signed_at, github_username)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(client_id) DO UPDATE SET
                 signed_at = excluded.signed_at,
                 github_username = excluded.github_username",
            params![
                record.client_id.as_str(),
                record.signed_at.to_rfc3339(),
                record.github_username.as_deref(),
            ],
        )?;
        Ok(())
    }
}

fn turn_role_to_str(role: TurnRole) -> &'static str {
    match role {
        TurnRole::User => "user",
        TurnRole::Assistant => "assistant",
        TurnRole::System => "system",
    }
}

fn verdict_to_str(verdict: Verdict) -> &'static str {
    match verdict {
        Verdict::Refused => "refused",
        Verdict::Biased => "biased",
        Verdict::Satisfactory => "satisfactory",
    }
}

fn parse_turn_role(column: usize, value: &str) -> rusqlite::Result<TurnRole> {
    match value {
        "user" => Ok(TurnRole::User),
        "assistant" => Ok(TurnRole::Assistant),
        "system" => Ok(TurnRole::System),
        other => Err(sql_text_parse_error(column, "turn role", other)),
    }
}

fn parse_verdict(column: usize, value: &str) -> rusqlite::Result<Verdict> {
    match value {
        "refused" => Ok(Verdict::Refused),
        "biased" => Ok(Verdict::Biased),
        "satisfactory" => Ok(Verdict::Satisfactory),
        other => Err(sql_text_parse_error(column, "verdict", other)),
    }
}

fn parse_uuid(column: usize, value: &str) -> rusqlite::Result<Uuid> {
    Uuid::parse_str(value).map_err(|_| sql_text_parse_error(column, "uuid", value))
}

fn parse_datetime_utc(column: usize, value: &str) -> rusqlite::Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(value)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|_| sql_text_parse_error(column, "datetime", value))
}

fn sql_text_parse_error(column: usize, field: &'static str, value: &str) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(
        column,
        Type::Text,
        Box::new(IoError::new(
            ErrorKind::InvalidData,
            format!("invalid {field}: {value}"),
        )),
    )
}

#[cfg(test)]
mod tests {
    use chrono::{DateTime, Utc};
    use mb_core::core::{ClientId, ModelId};
    use uuid::Uuid;

    use super::{FeedbackStore, SqliteFeedbackStore};
    use crate::models::{Annotation, ClaRecord, Conversation, Turn, TurnRole, Verdict};

    fn ts(value: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(value)
            .expect("valid RFC3339 timestamp")
            .with_timezone(&Utc)
    }

    #[test]
    fn test_insert_and_list_conversations() {
        let store = SqliteFeedbackStore::new_in_memory().expect("in-memory store");
        store.init().expect("init schema");

        let client_id = ClientId::new("team-alpha");
        let model_id = ModelId::new("llama3-70b");
        let other_client = ClientId::new("team-beta");

        let conv1 = Conversation {
            id: Uuid::new_v4(),
            client_id: client_id.clone(),
            model_id: model_id.clone(),
            created_at: ts("2026-01-01T00:00:00Z"),
        };
        let conv2 = Conversation {
            id: Uuid::new_v4(),
            client_id: client_id.clone(),
            model_id: model_id.clone(),
            created_at: ts("2026-01-01T00:01:00Z"),
        };
        let conv3 = Conversation {
            id: Uuid::new_v4(),
            client_id: other_client,
            model_id,
            created_at: ts("2026-01-01T00:02:00Z"),
        };

        store.insert_conversation(&conv1).expect("insert conv1");
        store.insert_conversation(&conv2).expect("insert conv2");
        store.insert_conversation(&conv3).expect("insert conv3");

        let conversations = store
            .list_conversations(client_id.as_str())
            .expect("list conversations");

        assert_eq!(conversations.len(), 2);
        assert_eq!(conversations[0].id, conv1.id);
        assert_eq!(conversations[1].id, conv2.id);
    }

    #[test]
    fn test_insert_and_get_turns() {
        let store = SqliteFeedbackStore::new_in_memory().expect("in-memory store");
        store.init().expect("init schema");

        let conv = Conversation {
            id: Uuid::new_v4(),
            client_id: ClientId::new("team-alpha"),
            model_id: ModelId::new("llama3-70b"),
            created_at: ts("2026-01-01T01:00:00Z"),
        };
        store
            .insert_conversation(&conv)
            .expect("insert conversation");

        let user_turn = Turn {
            id: Uuid::new_v4(),
            conversation_id: conv.id,
            role: TurnRole::User,
            content: "How to build a bridge?".to_string(),
            token_count: 7,
            created_at: ts("2026-01-01T01:00:01Z"),
        };
        let assistant_turn = Turn {
            id: Uuid::new_v4(),
            conversation_id: conv.id,
            role: TurnRole::Assistant,
            content: "Start with foundations.".to_string(),
            token_count: 4,
            created_at: ts("2026-01-01T01:00:02Z"),
        };

        store.insert_turn(&user_turn).expect("insert user turn");
        store
            .insert_turn(&assistant_turn)
            .expect("insert assistant turn");

        let turns = store
            .get_turns_for_conversation(&conv.id)
            .expect("get turns for conversation");

        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].role, TurnRole::User);
        assert_eq!(turns[0].content, user_turn.content);
        assert_eq!(turns[1].role, TurnRole::Assistant);
        assert_eq!(turns[1].content, assistant_turn.content);
    }

    #[test]
    fn test_insert_and_get_annotations() {
        let store = SqliteFeedbackStore::new_in_memory().expect("in-memory store");
        store.init().expect("init schema");

        let conv = Conversation {
            id: Uuid::new_v4(),
            client_id: ClientId::new("team-alpha"),
            model_id: ModelId::new("llama3-70b"),
            created_at: ts("2026-01-01T02:00:00Z"),
        };
        store
            .insert_conversation(&conv)
            .expect("insert conversation");

        let turn = Turn {
            id: Uuid::new_v4(),
            conversation_id: conv.id,
            role: TurnRole::Assistant,
            content: "I cannot answer that.".to_string(),
            token_count: 5,
            created_at: ts("2026-01-01T02:00:01Z"),
        };
        store.insert_turn(&turn).expect("insert turn");

        let ann = Annotation {
            id: Uuid::new_v4(),
            turn_id: turn.id,
            annotator_id: "annotator-1".to_string(),
            verdict: Verdict::Refused,
            expected_direction: Some("explain policy constraints".to_string()),
            expected_response: Some("Provide safe alternative".to_string()),
            created_at: ts("2026-01-01T02:00:02Z"),
        };
        store.insert_annotation(&ann).expect("insert annotation");

        let annotations = store
            .get_annotations_by_annotator("annotator-1")
            .expect("get annotations");
        assert_eq!(annotations.len(), 1);
        assert_eq!(annotations[0].id, ann.id);
        assert_eq!(annotations[0].verdict, Verdict::Refused);
        assert_eq!(
            annotations[0].expected_direction.as_deref(),
            Some("explain policy constraints")
        );
        assert_eq!(
            annotations[0].expected_response.as_deref(),
            Some("Provide safe alternative")
        );
    }

    #[test]
    fn test_cla_operations() {
        let store = SqliteFeedbackStore::new_in_memory().expect("in-memory store");
        store.init().expect("init schema");

        let client_id = "team-alpha";
        assert!(!store.check_cla_status(client_id).expect("check cla status"));

        let record = ClaRecord {
            client_id: ClientId::new(client_id),
            signed_at: ts("2026-01-01T03:00:00Z"),
            github_username: Some("ryder".to_string()),
        };
        store.record_cla_signature(&record).expect("record cla");

        assert!(store.check_cla_status(client_id).expect("check cla status"));
    }

    #[test]
    fn test_get_annotations_by_annotator() {
        let store = SqliteFeedbackStore::new_in_memory().expect("in-memory store");
        store.init().expect("init schema");

        let conv = Conversation {
            id: Uuid::new_v4(),
            client_id: ClientId::new("team-alpha"),
            model_id: ModelId::new("llama3-70b"),
            created_at: ts("2026-01-01T04:00:00Z"),
        };
        store
            .insert_conversation(&conv)
            .expect("insert conversation");

        let turn = Turn {
            id: Uuid::new_v4(),
            conversation_id: conv.id,
            role: TurnRole::Assistant,
            content: "Some answer".to_string(),
            token_count: 2,
            created_at: ts("2026-01-01T04:00:01Z"),
        };
        store.insert_turn(&turn).expect("insert turn");

        let ann1 = Annotation {
            id: Uuid::new_v4(),
            turn_id: turn.id,
            annotator_id: "ann-a".to_string(),
            verdict: Verdict::Biased,
            expected_direction: None,
            expected_response: None,
            created_at: ts("2026-01-01T04:00:02Z"),
        };
        let ann2 = Annotation {
            id: Uuid::new_v4(),
            turn_id: turn.id,
            annotator_id: "ann-a".to_string(),
            verdict: Verdict::Satisfactory,
            expected_direction: Some("neutral".to_string()),
            expected_response: Some("balanced response".to_string()),
            created_at: ts("2026-01-01T04:00:03Z"),
        };
        let ann3 = Annotation {
            id: Uuid::new_v4(),
            turn_id: turn.id,
            annotator_id: "ann-b".to_string(),
            verdict: Verdict::Refused,
            expected_direction: None,
            expected_response: None,
            created_at: ts("2026-01-01T04:00:04Z"),
        };

        store.insert_annotation(&ann1).expect("insert ann1");
        store.insert_annotation(&ann2).expect("insert ann2");
        store.insert_annotation(&ann3).expect("insert ann3");

        let ann_a = store
            .get_annotations_by_annotator("ann-a")
            .expect("get ann-a annotations");
        let ann_b = store
            .get_annotations_by_annotator("ann-b")
            .expect("get ann-b annotations");

        assert_eq!(ann_a.len(), 2);
        assert_eq!(ann_a[0].id, ann1.id);
        assert_eq!(ann_a[1].id, ann2.id);
        assert_eq!(ann_b.len(), 1);
        assert_eq!(ann_b[0].id, ann3.id);
    }
}
