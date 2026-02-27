use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;
use tokio::task::JoinHandle;

use mb_core::core::{
    BackendId, BackendInfo, BackendSpec, BackendState, HealthError, HealthProbe, LatencyMs,
};

// ---------------------------------------------------------------------------
// HttpHealthProbe — live HTTP probe for backend health
// ---------------------------------------------------------------------------

pub struct HttpHealthProbe {
    client: reqwest::Client,
    timeout: Duration,
}

impl HttpHealthProbe {
    pub fn new(timeout: Duration) -> Result<Self, reqwest::Error> {
        let client = reqwest::Client::builder().timeout(timeout).build()?;
        Ok(Self { client, timeout })
    }
}

impl HealthProbe for HttpHealthProbe {
    fn probe<'a>(
        &'a self,
        backend: &'a BackendInfo,
    ) -> Pin<Box<dyn Future<Output = Result<LatencyMs, HealthError>> + Send + 'a>> {
        Box::pin(async move {
            let path = match backend.spec {
                BackendSpec::OpenAiChat => "/v1/models",
                BackendSpec::Ollama => "/api/tags",
            };
            let url = format!("{}{path}", backend.base_url);

            let start = std::time::Instant::now();
            let resp = self
                .client
                .get(&url)
                .timeout(self.timeout)
                .send()
                .await
                .map_err(|e| HealthError::ConnectionFailed(e.to_string()))?;

            let latency_ms = start.elapsed().as_millis() as u64;

            if resp.status().is_success() {
                Ok(LatencyMs::new(latency_ms))
            } else {
                Err(HealthError::UnexpectedStatus(resp.status().as_u16()))
            }
        })
    }
}

// ---------------------------------------------------------------------------
// HealthCheckManager — background health monitoring
// ---------------------------------------------------------------------------

/// Shared backend state map used by the router and health checker.
pub type SharedBackendStates = Arc<RwLock<HashMap<BackendId, BackendState>>>;

pub struct HealthCheckManager {
    states: SharedBackendStates,
}

impl HealthCheckManager {
    pub fn new(backends: &[BackendInfo]) -> Self {
        let mut map = HashMap::with_capacity(backends.len());
        for b in backends {
            let state = BackendState::new(b.id.clone(), b.models.clone(), b.max_concurrent);
            map.insert(b.id.clone(), state);
        }
        Self {
            states: Arc::new(RwLock::new(map)),
        }
    }

    pub fn shared_states(&self) -> SharedBackendStates {
        Arc::clone(&self.states)
    }

    pub fn start_background_checks(
        &self,
        backends: Vec<BackendInfo>,
        interval: Duration,
        unhealthy_threshold: u32,
        degraded_latency_ms: u64,
        probe: Arc<dyn HealthProbe>,
    ) -> JoinHandle<()> {
        let states = self.shared_states();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(interval);
            loop {
                tick.tick().await;
                for backend in &backends {
                    let result = probe.probe(backend).await;
                    let mut map = states.write().await;
                    if let Some(state) = map.remove(&backend.id) {
                        let updated = match result {
                            Ok(latency) => {
                                if latency.value() >= degraded_latency_ms {
                                    state.with_degraded(latency)
                                } else {
                                    state.with_healthy(latency)
                                }
                            }
                            Err(_) => {
                                let state = state.with_failure();
                                if state.consecutive_failures >= unhealthy_threshold {
                                    state.with_unhealthy()
                                } else {
                                    state
                                }
                            }
                        };
                        map.insert(backend.id.clone(), updated);
                    }
                }
            }
        })
    }

    /// Snapshot of all backend states for the router.
    pub async fn get_states(&self) -> Vec<BackendState> {
        let map = self.states.read().await;
        map.values().cloned().collect()
    }
}

// ---------------------------------------------------------------------------
// /health endpoint handler
// ---------------------------------------------------------------------------

pub async fn health_handler(states: SharedBackendStates) -> axum::response::Response {
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    let map = states.read().await;
    let backends: Vec<serde_json::Value> = map
        .values()
        .map(|s| {
            serde_json::json!({
                "id": s.id.as_str(),
                "status": format!("{:?}", s.status),
                "active_requests": s.active_requests,
                "last_latency_ms": s.last_latency.map(|l| l.value()),
            })
        })
        .collect();

    let any_healthy = map.values().any(|s| s.is_healthy());
    let status = if any_healthy {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    let body = serde_json::json!({
        "status": if any_healthy { "ok" } else { "unavailable" },
        "backends": backends,
    });

    (status, axum::Json(body)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mb_core::core::{BackendId, BackendInfo, BackendSpec, BackendStatus, ModelId};

    fn make_backend(id: &str) -> BackendInfo {
        BackendInfo {
            id: BackendId::new(id),
            spec: BackendSpec::OpenAiChat,
            models: vec![ModelId::new("gpt-4")],
            max_concurrent: 10,
            base_url: "http://localhost:8000".to_owned(),
        }
    }

    #[test]
    fn test_manager_initializes_states() {
        let backends = vec![make_backend("gpu-0"), make_backend("gpu-1")];
        let manager = HealthCheckManager::new(&backends);

        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let states = rt.block_on(manager.get_states());
        assert_eq!(states.len(), 2);
        for state in &states {
            assert_eq!(state.status, BackendStatus::Unknown);
        }
    }

    #[test]
    fn test_health_endpoint_all_unknown() {
        let backends = vec![make_backend("gpu-0")];
        let manager = HealthCheckManager::new(&backends);
        let shared = manager.shared_states();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let response = rt.block_on(health_handler(shared));
        // All Unknown → not healthy → 503
        assert_eq!(
            response.status(),
            axum::http::StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[test]
    fn test_health_endpoint_one_healthy() {
        let backends = vec![make_backend("gpu-0")];
        let manager = HealthCheckManager::new(&backends);
        let shared = manager.shared_states();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(async {
            {
                let mut map = shared.write().await;
                if let Some(state) = map.remove(&BackendId::new("gpu-0")) {
                    map.insert(
                        BackendId::new("gpu-0"),
                        state.with_healthy(LatencyMs::new(50)),
                    );
                }
            }
            let response = health_handler(Arc::clone(&shared)).await;
            assert_eq!(response.status(), axum::http::StatusCode::OK);
        });
    }
}
