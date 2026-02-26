use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use tokio::sync::RwLock;

use mb_core::core::{
    AdapterError, ApiKey, ApiSpec, AuthError, AuthService, BackendId, BackendSpec,
    CacheAffinityMap, ClientId, GatewayError, QuotaTracker, RateLimiter, RoutingError,
    RoutingStrategy, YearMonth,
};

use crate::bootstrap::CacheConfig;
use crate::health::SharedBackendStates;
use crate::inbound::InboundAdapterRegistry;
use crate::outbound::OutboundAdapterRegistry;

// ---------------------------------------------------------------------------
// AppState — shared state for all handlers
// ---------------------------------------------------------------------------

pub struct AppState {
    pub auth: AuthService,
    pub inbound_registry: InboundAdapterRegistry,
    pub outbound_registry: OutboundAdapterRegistry,
    pub backend_states: SharedBackendStates,
    pub rate_limiters: RwLock<HashMap<ClientId, RateLimiter>>,
    pub quota_tracker: RwLock<QuotaTracker>,
    pub affinity_map: RwLock<CacheAffinityMap>,
    pub http_client: reqwest::Client,
    pub routing_strategy: RoutingStrategy,
    pub cache_config: CacheConfig,
    pub round_counter: AtomicUsize,
    pub rate_limit_rpm: HashMap<ClientId, u32>,
    pub backends_by_id: HashMap<BackendId, BackendMeta>,
    #[cfg(feature = "feedback")]
    pub feedback: Option<crate::feedback::FeedbackState>,
}

/// Metadata needed to dispatch requests to a backend.
pub struct BackendMeta {
    pub base_url: String,
    pub spec: BackendSpec,
}

// ---------------------------------------------------------------------------
// Non-streaming request handler
// ---------------------------------------------------------------------------

pub async fn handle_completion(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    match handle_completion_inner(&state, &headers, &body).await {
        Ok(resp) => resp,
        Err(e) => gateway_error_to_response(e),
    }
}

async fn handle_completion_inner(
    state: &AppState,
    headers: &HeaderMap,
    body: &[u8],
) -> Result<Response, GatewayError> {
    // 1. Extract API key from Authorization header
    let api_key = extract_api_key(headers)?;

    // 2. Parse request body via inbound adapter
    let inbound = state
        .inbound_registry
        .get(&ApiSpec::OpenAiChat)
        .ok_or(GatewayError::Adapter(AdapterError::ParseRequest(
            "unsupported API spec".to_owned(),
        )))?;

    let mut canonical_req = inbound.parse_request(body).map_err(GatewayError::Adapter)?;

    // 3. Validate API key
    let client_info = state.auth.validate(&api_key).map_err(GatewayError::Auth)?;
    canonical_req.metadata.client_id = client_info.id.clone();

    // 4. Check model permission
    AuthService::check_model_permission(client_info, &canonical_req.model)
        .map_err(GatewayError::Auth)?;

    // 5. Rate limit check
    {
        let now_ms = now_ms();
        let mut limiters = state.rate_limiters.write().await;
        let limiter = limiters.entry(client_info.id.clone()).or_insert_with(|| {
            let rpm = state
                .rate_limit_rpm
                .get(&client_info.id)
                .copied()
                .unwrap_or(60);
            RateLimiter::new(60_000, rpm)
        });
        limiter.check(now_ms).map_err(GatewayError::RateLimited)?;
    }

    // 6. Quota check
    if client_info.quota.monthly_token_limit.is_some() {
        let tracker = state.quota_tracker.read().await;
        let period = current_year_month();
        tracker
            .check(
                &client_info.id,
                canonical_req.metadata.estimated_input_tokens,
                &client_info.quota,
                period,
            )
            .map_err(GatewayError::QuotaExceeded)?;
    }

    // 7. Compute prefix hash for cache-aware routing
    if state.cache_config.enabled {
        let hash = mb_core::core::compute_prefix_hash(
            &canonical_req.messages,
            state.cache_config.prefix_depth,
        );
        canonical_req.metadata.prefix_hash = Some(hash);
    }

    // 8. Get affinity hint
    let affinity_hint = if state.cache_config.enabled {
        if let Some(prefix) = canonical_req.metadata.prefix_hash {
            let mut map = state.affinity_map.write().await;
            map.get(&canonical_req.model, prefix).cloned()
        } else {
            None
        }
    } else {
        None
    };

    // 9. Select backend via router
    let backend_states = state.backend_states.read().await;
    let states_vec: Vec<_> = backend_states.values().cloned().collect();
    let round = state.round_counter.fetch_add(1, Ordering::Relaxed);

    let selected_id = mb_core::core::select_backend(
        &states_vec,
        &canonical_req.model,
        &state.routing_strategy,
        round,
        affinity_hint.as_ref(),
    )
    .map_err(GatewayError::Routing)?;
    drop(backend_states);

    // 10. Look up backend metadata
    let backend_meta = state
        .backends_by_id
        .get(&selected_id)
        .ok_or(GatewayError::Routing(RoutingError::NoHealthyBackend {
            model: canonical_req.model.clone(),
        }))?;

    // 11. Build outbound request body
    let outbound = state
        .outbound_registry
        .get(&backend_meta.spec)
        .ok_or(GatewayError::Adapter(AdapterError::FormatResponse(
            "no outbound adapter for backend spec".to_owned(),
        )))?;

    let request_body = outbound
        .build_request_body(&canonical_req)
        .map_err(GatewayError::Adapter)?;

    // 12. Forward to backend
    let url = format!("{}{}", backend_meta.base_url, outbound.inference_path());

    let backend_info = mb_core::core::BackendInfo {
        id: selected_id.clone(),
        spec: backend_meta.spec,
        models: vec![],
        max_concurrent: 0,
        base_url: backend_meta.base_url.clone(),
    };

    let mut req_builder = state.http_client.post(&url).body(request_body);
    for (k, v) in outbound.extra_headers(&backend_info) {
        req_builder = req_builder.header(k, v);
    }

    let backend_resp = req_builder.send().await.map_err(|e| {
        GatewayError::Backend(mb_core::core::BackendError::Connection(e.to_string()))
    })?;

    if !backend_resp.status().is_success() {
        let status = backend_resp.status().as_u16();
        let body = backend_resp.text().await.unwrap_or_default();
        return Err(GatewayError::Backend(
            mb_core::core::BackendError::HttpStatus { status, body },
        ));
    }

    let resp_bytes = backend_resp.bytes().await.map_err(|e| {
        GatewayError::Backend(mb_core::core::BackendError::Connection(e.to_string()))
    })?;

    // 13. Parse backend response
    let canonical_resp = outbound
        .parse_response(&resp_bytes)
        .map_err(GatewayError::Adapter)?;

    // 14. Record quota usage
    if client_info.quota.monthly_token_limit.is_some() {
        let mut tracker = state.quota_tracker.write().await;
        let period = current_year_month();
        tracker.record(&client_info.id, canonical_resp.usage.total_tokens, period);
    }

    // 15. Record cache affinity
    if state.cache_config.enabled {
        if let Some(ref prefix) = canonical_req.metadata.prefix_hash {
            let mut map = state.affinity_map.write().await;
            map.record(&canonical_req.model, *prefix, &selected_id);
        }
    }

    #[cfg(feature = "feedback")]
    if let Some(feedback_state) = state.feedback.as_ref() {
        crate::feedback::record_chat_turns(
            feedback_state,
            headers,
            &canonical_req,
            &canonical_resp,
        )
        .await;
    }

    // 16. Format response via inbound adapter
    let response_bytes = inbound
        .format_response(&canonical_resp)
        .map_err(GatewayError::Adapter)?;

    Ok((
        StatusCode::OK,
        [("content-type", "application/json")],
        response_bytes,
    )
        .into_response())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

pub(crate) fn extract_api_key(headers: &HeaderMap) -> Result<ApiKey, GatewayError> {
    let auth_header = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .ok_or(GatewayError::Auth(AuthError::InvalidApiKey))?;

    let token = auth_header
        .strip_prefix("Bearer ")
        .ok_or(GatewayError::Auth(AuthError::InvalidApiKey))?;

    Ok(ApiKey::new(token))
}

pub(crate) fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

pub(crate) fn current_year_month() -> YearMonth {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = secs / 86400;
    let year = 1970 + (days / 365) as u16;
    let month = ((days % 365) / 30 + 1).min(12) as u8;
    YearMonth::new(year, month)
}

// ---------------------------------------------------------------------------
// Error → Response conversion (OpenAI-compatible error format)
// ---------------------------------------------------------------------------

pub fn gateway_error_to_response(err: GatewayError) -> Response {
    let (status, error_type, message) = match &err {
        GatewayError::Auth(AuthError::InvalidApiKey) => (
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            err.to_string(),
        ),
        GatewayError::Auth(AuthError::ModelNotPermitted { .. }) => {
            (StatusCode::FORBIDDEN, "permission_error", err.to_string())
        }
        GatewayError::RateLimited(_) => (
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            err.to_string(),
        ),
        GatewayError::QuotaExceeded(_) => {
            (StatusCode::PAYMENT_REQUIRED, "quota_error", err.to_string())
        }
        GatewayError::Routing(RoutingError::ModelNotFound { .. }) => {
            (StatusCode::NOT_FOUND, "not_found_error", err.to_string())
        }
        GatewayError::Routing(RoutingError::NoHealthyBackend { .. }) => (
            StatusCode::SERVICE_UNAVAILABLE,
            "service_unavailable",
            err.to_string(),
        ),
        GatewayError::Adapter(AdapterError::ParseRequest(_)) => (
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            err.to_string(),
        ),
        GatewayError::Backend(_) => (StatusCode::BAD_GATEWAY, "backend_error", err.to_string()),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            err.to_string(),
        ),
    };

    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": error_type,
            "code": status.as_u16(),
        }
    });

    (status, axum::Json(body)).into_response()
}
