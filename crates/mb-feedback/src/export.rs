use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::models::{DpoMetadata, DpoPair, TurnRole, Verdict};
use crate::store::{FeedbackError, FeedbackStore};

#[derive(Debug, Clone, Default)]
pub struct DpoExportFilter {
    pub annotator_id: Option<String>,
    pub model_id: Option<String>,
    pub verdict: Option<Verdict>,
    pub since: Option<DateTime<Utc>>,
    pub until: Option<DateTime<Utc>>,
}

/// Export DPO pairs from stored annotations.
///
/// Only `Refused` and `Biased` annotations can become DPO pairs.
/// Each pair requires a non-empty `expected_response` as the chosen output.
pub fn export_dpo_pairs(
    store: &dyn FeedbackStore,
    filter: &DpoExportFilter,
) -> Result<Vec<DpoPair>, FeedbackError> {
    let annotations = store.list_annotations()?;
    let mut pairs = Vec::new();

    for annotation in annotations {
        if let Some(expected_annotator) = filter.annotator_id.as_deref() {
            if annotation.annotator_id != expected_annotator {
                continue;
            }
        }

        if let Some(expected_verdict) = filter.verdict {
            if annotation.verdict != expected_verdict {
                continue;
            }
        }

        if !matches!(annotation.verdict, Verdict::Refused | Verdict::Biased) {
            continue;
        }

        if let Some(since) = filter.since.as_ref() {
            if annotation.created_at < since.clone() {
                continue;
            }
        }

        if let Some(until) = filter.until.as_ref() {
            if annotation.created_at > until.clone() {
                continue;
            }
        }

        let Some(chosen_response) = annotation
            .expected_response
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            continue;
        };

        let Some(annotated_turn) = store.get_turn_by_id(&annotation.turn_id)? else {
            continue;
        };
        if annotated_turn.role != TurnRole::Assistant {
            continue;
        }

        let Some(conversation) = store.get_conversation_by_id(&annotated_turn.conversation_id)?
        else {
            continue;
        };

        if let Some(expected_model) = filter.model_id.as_deref() {
            if conversation.model_id.as_str() != expected_model {
                continue;
            }
        }

        let turns = store.get_turns_for_conversation(&conversation.id)?;
        let Some(assistant_index) = turns.iter().position(|turn| turn.id == annotated_turn.id)
        else {
            continue;
        };
        let Some(prompt_turn) = turns[..assistant_index]
            .iter()
            .rev()
            .find(|turn| turn.role == TurnRole::User)
        else {
            continue;
        };

        pairs.push(DpoPair {
            prompt: prompt_turn.content.clone(),
            chosen: chosen_response.to_string(),
            rejected: annotated_turn.content,
            metadata: DpoMetadata {
                conversation_id: conversation.id,
                model_id: conversation.model_id,
                annotator_id: annotation.annotator_id,
                verdict: annotation.verdict,
                annotated_at: annotation.created_at,
            },
        });
    }

    Ok(pairs)
}

#[derive(Serialize)]
struct ExportJsonPair<'a> {
    prompt: &'a str,
    chosen: &'a str,
    rejected: &'a str,
}

pub fn export_to_json(pairs: &[DpoPair]) -> Result<String, FeedbackError> {
    let export_pairs: Vec<ExportJsonPair<'_>> = pairs
        .iter()
        .map(|pair| ExportJsonPair {
            prompt: pair.prompt.as_str(),
            chosen: pair.chosen.as_str(),
            rejected: pair.rejected.as_str(),
        })
        .collect();

    let json = serde_json::to_string(&export_pairs)?;
    Ok(json)
}

#[cfg(test)]
mod tests {
    use chrono::{DateTime, Utc};
    use mb_core::core::{ClientId, ModelId};
    use uuid::Uuid;

    use super::{export_dpo_pairs, export_to_json, DpoExportFilter};
    use crate::models::{Annotation, Conversation, Turn, TurnRole, Verdict};
    use crate::store::{FeedbackStore, SqliteFeedbackStore};

    fn ts(value: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(value)
            .expect("valid RFC3339 timestamp")
            .with_timezone(&Utc)
    }

    fn setup_store() -> SqliteFeedbackStore {
        let store = SqliteFeedbackStore::new_in_memory().expect("in-memory store");
        store.init().expect("init schema");
        store
    }

    fn insert_refused_annotation_with_expected(
        store: &SqliteFeedbackStore,
        model_id: &str,
        annotator_id: &str,
        expected_response: &str,
        base_ts: &str,
    ) {
        let conversation = Conversation {
            id: Uuid::new_v4(),
            client_id: ClientId::new("team-alpha"),
            model_id: ModelId::new(model_id),
            created_at: ts(base_ts),
        };
        store
            .insert_conversation(&conversation)
            .expect("insert conversation");

        let user_turn = Turn {
            id: Uuid::new_v4(),
            conversation_id: conversation.id,
            role: TurnRole::User,
            content: "How do I handle this topic?".to_string(),
            token_count: 6,
            created_at: ts("2026-01-01T10:00:01Z"),
        };
        store.insert_turn(&user_turn).expect("insert user turn");

        let assistant_turn = Turn {
            id: Uuid::new_v4(),
            conversation_id: conversation.id,
            role: TurnRole::Assistant,
            content: "I cannot help with that.".to_string(),
            token_count: 5,
            created_at: ts("2026-01-01T10:00:02Z"),
        };
        store
            .insert_turn(&assistant_turn)
            .expect("insert assistant turn");

        let annotation = Annotation {
            id: Uuid::new_v4(),
            turn_id: assistant_turn.id,
            annotator_id: annotator_id.to_string(),
            verdict: Verdict::Refused,
            expected_direction: Some("Provide balanced explanation".to_string()),
            expected_response: Some(expected_response.to_string()),
            created_at: ts("2026-01-01T10:00:03Z"),
        };
        store
            .insert_annotation(&annotation)
            .expect("insert annotation");
    }

    #[test]
    fn test_export_empty_store() {
        let store = setup_store();
        let filter = DpoExportFilter::default();

        let pairs = export_dpo_pairs(&store, &filter).expect("export dpo pairs");

        assert!(pairs.is_empty());
    }

    #[test]
    fn test_export_with_satisfactory_only() {
        let store = setup_store();

        let conversation = Conversation {
            id: Uuid::new_v4(),
            client_id: ClientId::new("team-alpha"),
            model_id: ModelId::new("llama3-70b"),
            created_at: ts("2026-01-01T11:00:00Z"),
        };
        store
            .insert_conversation(&conversation)
            .expect("insert conversation");

        let user_turn = Turn {
            id: Uuid::new_v4(),
            conversation_id: conversation.id,
            role: TurnRole::User,
            content: "Tell me the history.".to_string(),
            token_count: 4,
            created_at: ts("2026-01-01T11:00:01Z"),
        };
        store.insert_turn(&user_turn).expect("insert user turn");

        let assistant_turn = Turn {
            id: Uuid::new_v4(),
            conversation_id: conversation.id,
            role: TurnRole::Assistant,
            content: "Here is a balanced answer.".to_string(),
            token_count: 5,
            created_at: ts("2026-01-01T11:00:02Z"),
        };
        store
            .insert_turn(&assistant_turn)
            .expect("insert assistant turn");

        let annotation = Annotation {
            id: Uuid::new_v4(),
            turn_id: assistant_turn.id,
            annotator_id: "ann-1".to_string(),
            verdict: Verdict::Satisfactory,
            expected_direction: None,
            expected_response: Some("Same response".to_string()),
            created_at: ts("2026-01-01T11:00:03Z"),
        };
        store
            .insert_annotation(&annotation)
            .expect("insert annotation");

        let pairs =
            export_dpo_pairs(&store, &DpoExportFilter::default()).expect("export dpo pairs");

        assert!(pairs.is_empty());
    }

    #[test]
    fn test_export_with_refused_and_expected() {
        let store = setup_store();
        insert_refused_annotation_with_expected(
            &store,
            "llama3-70b",
            "ann-1",
            "Offer neutral context and evidence.",
            "2026-01-01T10:00:00Z",
        );

        let pairs =
            export_dpo_pairs(&store, &DpoExportFilter::default()).expect("export dpo pairs");

        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].prompt, "How do I handle this topic?");
        assert_eq!(pairs[0].chosen, "Offer neutral context and evidence.");
        assert_eq!(pairs[0].rejected, "I cannot help with that.");
        assert_eq!(pairs[0].metadata.annotator_id, "ann-1");
        assert_eq!(pairs[0].metadata.verdict, Verdict::Refused);

        let json = export_to_json(&pairs).expect("export json");
        assert!(json.contains("\"prompt\":\"How do I handle this topic?\""));
        assert!(json.contains("\"chosen\":\"Offer neutral context and evidence.\""));
        assert!(json.contains("\"rejected\":\"I cannot help with that.\""));
    }

    #[test]
    fn test_export_filter_by_model() {
        let store = setup_store();

        insert_refused_annotation_with_expected(
            &store,
            "llama3-70b",
            "ann-1",
            "Expected response for model A",
            "2026-01-01T12:00:00Z",
        );
        insert_refused_annotation_with_expected(
            &store,
            "qwen2.5-14b",
            "ann-2",
            "Expected response for model B",
            "2026-01-01T13:00:00Z",
        );

        let filter = DpoExportFilter {
            model_id: Some("qwen2.5-14b".to_string()),
            ..DpoExportFilter::default()
        };
        let pairs = export_dpo_pairs(&store, &filter).expect("export dpo pairs");

        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].metadata.model_id.as_str(), "qwen2.5-14b");
        assert_eq!(pairs[0].chosen, "Expected response for model B");
    }
}
