use crate::core::{BackendId, LatencyMs, ModelId};

// ---------------------------------------------------------------------------
// BackendStatus — runtime health state of a backend
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

// ---------------------------------------------------------------------------
// BackendState — runtime backend status with capacity and health tracking
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct BackendState {
    pub id: BackendId,
    pub models: Vec<ModelId>,
    pub status: BackendStatus,
    pub active_requests: u32,
    pub max_concurrent: u32,
    pub last_latency: Option<LatencyMs>,
    pub consecutive_failures: u32,
}

impl BackendState {
    pub fn new(id: BackendId, models: Vec<ModelId>, max_concurrent: u32) -> Self {
        Self {
            id,
            models,
            status: BackendStatus::Unknown,
            active_requests: 0,
            max_concurrent,
            last_latency: None,
            consecutive_failures: 0,
        }
    }

    pub fn is_healthy(&self) -> bool {
        matches!(
            self.status,
            BackendStatus::Healthy | BackendStatus::Degraded
        )
    }

    pub fn has_capacity(&self) -> bool {
        self.active_requests < self.max_concurrent
    }

    pub fn serves_model(&self, model: &ModelId) -> bool {
        self.models.iter().any(|m| m == model)
    }

    pub fn with_healthy(self, latency: LatencyMs) -> Self {
        Self {
            status: BackendStatus::Healthy,
            last_latency: Some(latency),
            consecutive_failures: 0,
            ..self
        }
    }

    pub fn with_degraded(self, latency: LatencyMs) -> Self {
        Self {
            status: BackendStatus::Degraded,
            last_latency: Some(latency),
            ..self
        }
    }

    pub fn with_unhealthy(self) -> Self {
        Self {
            status: BackendStatus::Unhealthy,
            ..self
        }
    }

    pub fn with_failure(self) -> Self {
        Self {
            consecutive_failures: self.consecutive_failures.saturating_add(1),
            ..self
        }
    }

    pub fn with_request_started(self) -> Self {
        Self {
            active_requests: self.active_requests.saturating_add(1),
            ..self
        }
    }

    pub fn with_request_completed(self) -> Self {
        Self {
            active_requests: self.active_requests.saturating_sub(1),
            ..self
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_backend() -> BackendState {
        BackendState::new(
            BackendId::new("gpu-0"),
            vec![ModelId::new("llama3-70b"), ModelId::new("mistral-7b")],
            4,
        )
    }

    #[test]
    fn test_new_backend_state() {
        let state = make_backend();
        assert_eq!(state.id, BackendId::new("gpu-0"));
        assert_eq!(state.status, BackendStatus::Unknown);
        assert_eq!(state.active_requests, 0);
        assert_eq!(state.max_concurrent, 4);
        assert!(state.last_latency.is_none());
        assert_eq!(state.consecutive_failures, 0);
        assert_eq!(state.models.len(), 2);
    }

    #[test]
    fn test_status_transitions() {
        let state = make_backend();
        assert!(!state.is_healthy()); // Unknown is not healthy

        let state = state.with_healthy(LatencyMs::new(50));
        assert!(state.is_healthy());
        assert_eq!(state.status, BackendStatus::Healthy);

        let state = state.with_degraded(LatencyMs::new(3000));
        assert!(state.is_healthy()); // Degraded counts as healthy
        assert_eq!(state.status, BackendStatus::Degraded);

        let state = state.with_unhealthy();
        assert!(!state.is_healthy());
        assert_eq!(state.status, BackendStatus::Unhealthy);
    }

    #[test]
    fn test_capacity() {
        let state = make_backend(); // max_concurrent = 4
        assert!(state.has_capacity()); // 0 < 4

        let state = state
            .with_request_started()
            .with_request_started()
            .with_request_started();
        assert!(state.has_capacity()); // 3 < 4

        let state = state.with_request_started();
        assert!(!state.has_capacity()); // 4 == 4

        let state = state.with_request_completed();
        assert!(state.has_capacity()); // 3 < 4
    }

    #[test]
    fn test_model_matching() {
        let state = make_backend();
        assert!(state.serves_model(&ModelId::new("llama3-70b")));
        assert!(state.serves_model(&ModelId::new("mistral-7b")));
        assert!(!state.serves_model(&ModelId::new("gpt-4")));
    }

    #[test]
    fn test_request_counting() {
        let state = make_backend();
        assert_eq!(state.active_requests, 0);

        let state = state.with_request_started();
        assert_eq!(state.active_requests, 1);

        let state = state.with_request_started();
        assert_eq!(state.active_requests, 2);

        let state = state.with_request_completed();
        assert_eq!(state.active_requests, 1);

        // saturating_sub: cannot go below 0
        let state = state.with_request_completed().with_request_completed();
        assert_eq!(state.active_requests, 0);
    }

    #[test]
    fn test_failure_counting() {
        let state = make_backend();
        assert_eq!(state.consecutive_failures, 0);

        let state = state.with_failure().with_failure().with_failure();
        assert_eq!(state.consecutive_failures, 3);

        // with_healthy resets failures
        let state = state.with_healthy(LatencyMs::new(100));
        assert_eq!(state.consecutive_failures, 0);
        assert_eq!(state.last_latency, Some(LatencyMs::new(100)));
    }
}
