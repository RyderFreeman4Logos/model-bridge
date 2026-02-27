use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;

use mb_core::core::{
    AdapterError, ApiSpec, BackendSpec, ClientId, DeltaContent, GatewayError, ModelId, PrefixHash,
    RoutingError,
};

use crate::handler::{gateway_error_to_response, AppState};
use crate::outbound::streaming::SseLineParser;

// ---------------------------------------------------------------------------
// Streaming (SSE) request handler
// ---------------------------------------------------------------------------

pub async fn handle_completion_stream(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    match handle_stream_inner(state, &headers, &body).await {
        Ok(resp) => resp,
        Err(e) => gateway_error_to_response(e),
    }
}

async fn handle_stream_inner(
    state: Arc<AppState>,
    headers: &HeaderMap,
    body: &[u8],
) -> Result<Response, GatewayError> {
    // Steps 1-9: auth, parse, rate-limit, quota, route (shared logic)
    let api_key = crate::handler::extract_api_key(headers)?;

    let inbound = state
        .inbound_registry
        .get(&ApiSpec::OpenAiChat)
        .ok_or(GatewayError::Adapter(AdapterError::ParseRequest(
            "unsupported API spec".to_owned(),
        )))?;

    let mut canonical_req = inbound.parse_request(body).map_err(GatewayError::Adapter)?;

    let client_info = state.auth.validate(&api_key).map_err(GatewayError::Auth)?;
    canonical_req.metadata.client_id = client_info.id.clone();

    mb_core::core::AuthService::check_model_permission(client_info, &canonical_req.model)
        .map_err(GatewayError::Auth)?;

    {
        let now_ms = crate::handler::now_ms();
        let mut limiters = state.rate_limiters.write().await;
        let limiter = limiters.entry(client_info.id.clone()).or_insert_with(|| {
            let rpm = state
                .rate_limit_rpm
                .get(&client_info.id)
                .copied()
                .unwrap_or(60);
            mb_core::core::RateLimiter::new(60_000, rpm)
        });
        limiter.check(now_ms).map_err(GatewayError::RateLimited)?;
    }

    if client_info.quota.monthly_token_limit.is_some() {
        let tracker = state.quota_tracker.read().await;
        let period = crate::handler::current_year_month();
        tracker
            .check(
                &client_info.id,
                canonical_req.metadata.estimated_input_tokens,
                &client_info.quota,
                period,
            )
            .map_err(GatewayError::QuotaExceeded)?;
    }

    if state.cache_config.enabled {
        let hash = mb_core::core::compute_prefix_hash(
            &canonical_req.messages,
            state.cache_config.prefix_depth,
        );
        canonical_req.metadata.prefix_hash = Some(hash);
    }

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

    let backend_states = state.backend_states.read().await;
    let states_vec: Vec<_> = backend_states.values().cloned().collect();
    let round = state
        .round_counter
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let selected_id = mb_core::core::select_backend(
        &states_vec,
        &canonical_req.model,
        &state.routing_strategy,
        round,
        affinity_hint.as_ref(),
    )
    .map_err(GatewayError::Routing)?;
    drop(backend_states);

    let backend_meta = state
        .backends_by_id
        .get(&selected_id)
        .ok_or(GatewayError::Routing(RoutingError::NoHealthyBackend {
            model: canonical_req.model.clone(),
        }))?;

    let outbound_spec = backend_meta.spec;
    let outbound = state
        .outbound_registry
        .get(&outbound_spec)
        .ok_or(GatewayError::Adapter(AdapterError::FormatResponse(
            "no outbound adapter".to_owned(),
        )))?;

    // Force stream=true
    let mut stream_req = canonical_req.clone();
    stream_req.stream = true;

    let request_body = outbound
        .build_request_body(&stream_req)
        .map_err(GatewayError::Adapter)?;

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
        let body_text = backend_resp.text().await.unwrap_or_default();
        return Err(GatewayError::Backend(
            mb_core::core::BackendError::HttpStatus {
                status,
                body: body_text,
            },
        ));
    }

    // Build SSE event stream
    let byte_stream = backend_resp.bytes_stream();
    let sse_parser = SseLineParser::new(byte_stream);

    let client_id_owned = client_info.id.clone();
    let model_owned = canonical_req.model.clone();
    let prefix_hash_owned = canonical_req.metadata.prefix_hash;

    let event_stream = make_event_stream(
        sse_parser,
        outbound_spec,
        state,
        client_id_owned,
        model_owned,
        selected_id,
        prefix_hash_owned,
    );

    Ok(axum::response::sse::Sse::new(event_stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response())
}

fn make_event_stream(
    sse_parser: SseLineParser<
        impl futures_core::Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
    >,
    outbound_spec: BackendSpec,
    state: Arc<AppState>,
    client_id: ClientId,
    model: ModelId,
    selected_backend: mb_core::core::BackendId,
    prefix_hash: Option<PrefixHash>,
) -> impl futures_core::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>
{
    async_stream::stream! {
        let mut lines = Box::pin(sse_parser);
        let mut finished = false;

        while let Some(line_result) = lines.next().await {
            let line = match line_result {
                Ok(l) => l,
                Err(_) => break, // Connection error, stop streaming
            };

            // Get adapters each iteration (they're behind shared refs)
            let outbound = match state.outbound_registry.get(&outbound_spec) {
                Some(a) => a,
                None => break,
            };
            let inbound = match state.inbound_registry.get(&ApiSpec::OpenAiChat) {
                Some(a) => a,
                None => break,
            };

            // Parse the line through the outbound adapter
            let chunk = match outbound.parse_stream_line(&line) {
                Ok(Some(c)) => c,
                Ok(None) => continue, // Keep-alive or [DONE]
                Err(_) => continue, // Skip malformed chunks
            };

            // Check for finish signal
            for sc in &chunk.choices {
                if matches!(sc.delta, DeltaContent::Finish(_)) {
                    finished = true;
                }
            }

            // Format through inbound adapter
            match inbound.format_stream_chunk(&chunk) {
                Ok(Some(sse_text)) => {
                    yield Ok(axum::response::sse::Event::default().data(sse_text));
                }
                Ok(None) => continue,
                Err(_) => continue,
            }
        }

        // Send done sentinel
        if let Some(inbound) = state.inbound_registry.get(&ApiSpec::OpenAiChat) {
            yield Ok(axum::response::sse::Event::default().data(inbound.done_sentinel()));
        }

        // Record cache affinity after successful streaming
        if state.cache_config.enabled {
            if let Some(prefix) = prefix_hash {
                let mut map = state.affinity_map.write().await;
                map.record(&model, prefix, &selected_backend);
            }
        }

        // Note: quota recording for streaming requires token counting from
        // the stream itself. For now we skip it since the backend may not
        // provide usage in stream mode. Full implementation would accumulate
        // from the final chunk or estimate from content length.
        let _ = (client_id, finished);
    }
}
