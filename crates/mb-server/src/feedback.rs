#[cfg(feature = "feedback")]
use std::sync::Arc;

#[cfg(feature = "feedback")]
use axum::extract::{Query, State};
#[cfg(feature = "feedback")]
use axum::http::{HeaderMap, StatusCode};
#[cfg(feature = "feedback")]
use axum::response::IntoResponse;
#[cfg(feature = "feedback")]
use axum::Json;
#[cfg(feature = "feedback")]
use chrono::Utc;
#[cfg(feature = "feedback")]
use mb_core::core::{
    ApiKey, CanonicalRequest, CanonicalResponse, ContentPart, MessageContent, Role,
};
#[cfg(feature = "feedback")]
use serde::Deserialize;
#[cfg(feature = "feedback")]
use serde_json::json;
#[cfg(feature = "feedback")]
use uuid::Uuid;

#[cfg(feature = "feedback")]
pub struct FeedbackState {
    pub store: Arc<dyn mb_feedback::FeedbackStore>,
}

#[cfg(feature = "feedback")]
#[derive(Debug, Deserialize)]
pub struct FeedbackRequest {
    pub turn_id: Uuid,
    pub verdict: String,
    pub expected_direction: Option<String>,
    pub expected_response: Option<String>,
}

#[cfg(feature = "feedback")]
#[derive(Debug, Deserialize)]
pub struct MyAnnotationsQuery {
    pub format: Option<String>,
    pub page: Option<u32>,
    pub per_page: Option<u32>,
}

#[cfg(feature = "feedback")]
pub async fn post_feedback(
    State(state): State<Arc<crate::handler::AppState>>,
    headers: HeaderMap,
    Json(body): Json<FeedbackRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let feedback_state = state.feedback.as_ref().ok_or_else(|| {
        json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "feedback store unavailable",
        )
    })?;

    let api_key = extract_feedback_api_key(&headers)?;
    let client_info = state
        .auth
        .validate(&api_key)
        .map_err(|_| json_error(StatusCode::UNAUTHORIZED, "invalid API key"))?;
    let annotator_id = client_info.id.to_string();

    let cla_signed = {
        let store = Arc::clone(&feedback_state.store);
        let client_id = annotator_id.clone();
        tokio::task::spawn_blocking(move || store.check_cla_status(&client_id))
            .await
            .map_err(|err| {
                json_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("failed to join CLA check task: {err}"),
                )
            })?
            .map_err(|err| {
                json_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("failed to check CLA status: {err}"),
                )
            })?
    };

    if !cla_signed {
        return Err(json_error(
            StatusCode::FORBIDDEN,
            "CLA not signed. Please sign at: [link]",
        ));
    }

    let verdict = parse_verdict(&body.verdict).ok_or_else(|| {
        json_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid verdict, expected one of: refused, biased, satisfactory",
        )
    })?;

    let annotation = mb_feedback::Annotation {
        id: Uuid::new_v4(),
        turn_id: body.turn_id,
        annotator_id,
        verdict,
        expected_direction: body.expected_direction,
        expected_response: body.expected_response,
        created_at: Utc::now(),
    };
    let annotation_id = annotation.id;

    {
        let store = Arc::clone(&feedback_state.store);
        tokio::task::spawn_blocking(move || store.insert_annotation(&annotation))
            .await
            .map_err(|err| {
                json_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("failed to join insert annotation task: {err}"),
                )
            })?
            .map_err(|err| {
                json_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("failed to insert annotation: {err}"),
                )
            })?;
    }

    Ok((StatusCode::CREATED, Json(json!({ "id": annotation_id }))))
}

#[cfg(feature = "feedback")]
pub async fn get_my_annotations(
    State(state): State<Arc<crate::handler::AppState>>,
    headers: HeaderMap,
    Query(query): Query<MyAnnotationsQuery>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let feedback_state = state.feedback.as_ref().ok_or_else(|| {
        json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "feedback store unavailable",
        )
    })?;

    let api_key = extract_feedback_api_key(&headers)?;
    let client_info = state
        .auth
        .validate(&api_key)
        .map_err(|_| json_error(StatusCode::UNAUTHORIZED, "invalid API key"))?;
    let annotator_id = client_info.id.to_string();

    if query
        .format
        .as_deref()
        .is_some_and(|format| format.eq_ignore_ascii_case("dpo"))
    {
        let store = Arc::clone(&feedback_state.store);
        let annotator_id_for_filter = annotator_id.clone();

        let dpo_json = tokio::task::spawn_blocking(move || {
            let filter = mb_feedback::DpoExportFilter {
                annotator_id: Some(annotator_id_for_filter),
                ..Default::default()
            };
            let pairs = mb_feedback::export_dpo_pairs(store.as_ref(), &filter)?;
            mb_feedback::export_to_json(&pairs)
        })
        .await
        .map_err(|err| {
            json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to join export dpo task: {err}"),
            )
        })?
        .map_err(|err| {
            json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to export dpo pairs: {err}"),
            )
        })?;

        let dpo_value = serde_json::from_str::<serde_json::Value>(&dpo_json).map_err(|err| {
            json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to serialize dpo export: {err}"),
            )
        })?;

        return Ok((StatusCode::OK, Json(dpo_value)));
    }

    let page = query.page.unwrap_or(1).max(1);
    let per_page = query.per_page.unwrap_or(50).clamp(1, 100);

    let annotations = {
        let store = Arc::clone(&feedback_state.store);
        let annotator_id_for_query = annotator_id;
        tokio::task::spawn_blocking(move || {
            store.get_annotations_by_annotator(&annotator_id_for_query)
        })
        .await
        .map_err(|err| {
            json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to join get annotations task: {err}"),
            )
        })?
        .map_err(|err| {
            json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to get annotations: {err}"),
            )
        })?
    };

    let total = annotations.len();
    let start = (page.saturating_sub(1) as usize).saturating_mul(per_page as usize);
    let paged_annotations = if start >= total {
        Vec::new()
    } else {
        annotations
            .into_iter()
            .skip(start)
            .take(per_page as usize)
            .collect::<Vec<_>>()
    };

    Ok((
        StatusCode::OK,
        Json(json!({
            "annotations": paged_annotations,
            "page": page,
            "per_page": per_page,
            "total": total,
        })),
    ))
}

#[cfg(feature = "feedback")]
pub async fn record_chat_turns(
    feedback_state: &FeedbackState,
    headers: &HeaderMap,
    request: &CanonicalRequest,
    response: &CanonicalResponse,
) {
    let Some(user_content) = extract_last_user_message(request) else {
        return;
    };
    let Some(assistant_content) = extract_assistant_message(response) else {
        return;
    };

    let conversation_id = extract_conversation_id(headers);
    let client_id = request.metadata.client_id.clone();
    let model_id = request.model.clone();
    let user_token_count = estimate_token_count(&user_content);
    let assistant_token_count = estimate_token_count(&assistant_content);
    let store = Arc::clone(&feedback_state.store);

    let join_result = tokio::task::spawn_blocking(move || {
        let now = Utc::now();
        let conversation = mb_feedback::Conversation {
            id: conversation_id,
            client_id,
            model_id,
            created_at: now,
        };
        if let Err(err) = store.insert_conversation(&conversation) {
            tracing::warn!(
                error = %err,
                conversation_id = %conversation_id,
                "failed to insert feedback conversation"
            );
        }

        let user_turn = mb_feedback::Turn {
            id: Uuid::new_v4(),
            conversation_id,
            role: mb_feedback::TurnRole::User,
            content: user_content,
            token_count: user_token_count,
            created_at: now,
        };
        if let Err(err) = store.insert_turn(&user_turn) {
            tracing::warn!(
                error = %err,
                conversation_id = %conversation_id,
                "failed to insert user feedback turn"
            );
        }

        let assistant_turn = mb_feedback::Turn {
            id: Uuid::new_v4(),
            conversation_id,
            role: mb_feedback::TurnRole::Assistant,
            content: assistant_content,
            token_count: assistant_token_count,
            created_at: now,
        };
        if let Err(err) = store.insert_turn(&assistant_turn) {
            tracing::warn!(
                error = %err,
                conversation_id = %conversation_id,
                "failed to insert assistant feedback turn"
            );
        }
    })
    .await;

    if let Err(err) = join_result {
        tracing::warn!(error = %err, "feedback logging task join failed");
    }
}

#[cfg(feature = "feedback")]
fn extract_conversation_id(headers: &HeaderMap) -> Uuid {
    let Some(raw_header) = headers
        .get("x-conversation-id")
        .and_then(|value| value.to_str().ok())
    else {
        return Uuid::new_v4();
    };

    match Uuid::parse_str(raw_header) {
        Ok(conversation_id) => conversation_id,
        Err(err) => {
            tracing::warn!(
                error = %err,
                header_value = raw_header,
                "invalid X-Conversation-Id header, generated a new UUID"
            );
            Uuid::new_v4()
        }
    }
}

#[cfg(feature = "feedback")]
fn extract_last_user_message(request: &CanonicalRequest) -> Option<String> {
    request
        .messages
        .iter()
        .rev()
        .find(|message| message.role == Role::User)
        .map(|message| content_to_text(&message.content))
}

#[cfg(feature = "feedback")]
fn extract_assistant_message(response: &CanonicalResponse) -> Option<String> {
    response
        .choices
        .iter()
        .find(|choice| choice.message.role == Role::Assistant)
        .map(|choice| content_to_text(&choice.message.content))
}

#[cfg(feature = "feedback")]
fn content_to_text(content: &MessageContent) -> String {
    match content {
        MessageContent::Text(text) => text.clone(),
        MessageContent::Parts(parts) => parts
            .iter()
            .filter_map(|part| match part {
                ContentPart::Text { text } => Some(text.as_str()),
                ContentPart::ImageUrl { .. } => None,
            })
            .collect::<Vec<_>>()
            .join(""),
    }
}

#[cfg(feature = "feedback")]
fn estimate_token_count(text: &str) -> u32 {
    let approx_tokens = text.chars().count() / 4;
    u32::try_from(approx_tokens).unwrap_or(u32::MAX)
}

#[cfg(feature = "feedback")]
fn extract_feedback_api_key(
    headers: &HeaderMap,
) -> Result<ApiKey, (StatusCode, Json<serde_json::Value>)> {
    if let Some(raw) = headers
        .get("x-api-key")
        .and_then(|value| value.to_str().ok())
    {
        return Ok(ApiKey::new(raw));
    }

    crate::handler::extract_api_key(headers)
        .map_err(|_| json_error(StatusCode::UNAUTHORIZED, "missing or invalid API key"))
}

#[cfg(feature = "feedback")]
fn parse_verdict(verdict: &str) -> Option<mb_feedback::Verdict> {
    match verdict {
        "refused" => Some(mb_feedback::Verdict::Refused),
        "biased" => Some(mb_feedback::Verdict::Biased),
        "satisfactory" => Some(mb_feedback::Verdict::Satisfactory),
        _ => None,
    }
}

#[cfg(feature = "feedback")]
fn json_error(
    status: StatusCode,
    message: impl Into<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    let message = message.into();
    (
        status,
        Json(json!({
            "error": {
                "message": message,
                "type": "feedback_error",
                "code": status.as_u16(),
            }
        })),
    )
}
