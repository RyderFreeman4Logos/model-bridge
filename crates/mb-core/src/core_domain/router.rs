use crate::core::{BackendId, BackendState, ModelId, RoutingError};

// ---------------------------------------------------------------------------
// RoutingStrategy — backend selection strategy
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutingStrategy {
    LeastLoaded,
    RoundRobin,
}

// ---------------------------------------------------------------------------
// select_backend — pure routing function (no IO, no side effects)
// ---------------------------------------------------------------------------

/// Selects a backend for the given model using the specified strategy.
///
/// Selection priority:
/// 1. Cache affinity hint (if healthy + has capacity)
/// 2. Strategy-based selection among backends with capacity
/// 3. Overload fallback: strategy-based among all healthy backends
pub fn select_backend(
    backends: &[BackendState],
    model: &ModelId,
    strategy: &RoutingStrategy,
    round: usize,
    affinity_hint: Option<&BackendId>,
) -> Result<BackendId, RoutingError> {
    // Step 1: filter backends that serve the model
    let serving: Vec<&BackendState> = backends.iter().filter(|b| b.serves_model(model)).collect();
    if serving.is_empty() {
        return Err(RoutingError::ModelNotFound {
            model: model.clone(),
        });
    }

    // Step 2: filter healthy backends
    let healthy: Vec<&BackendState> = serving.iter().filter(|b| b.is_healthy()).copied().collect();
    if healthy.is_empty() {
        return Err(RoutingError::NoHealthyBackend {
            model: model.clone(),
        });
    }

    // Step 3: affinity hint — if the hinted backend is healthy and has capacity, use it
    if let Some(hint) = affinity_hint {
        if let Some(backend) = healthy.iter().find(|b| &b.id == hint) {
            if backend.has_capacity() {
                return Ok(backend.id.clone());
            }
        }
    }

    // Step 4: filter healthy backends with capacity
    let with_capacity: Vec<&BackendState> = healthy
        .iter()
        .filter(|b| b.has_capacity())
        .copied()
        .collect();

    // Step 5/6: apply strategy on candidates with capacity, or all healthy if none have capacity
    let candidates = if with_capacity.is_empty() {
        &healthy
    } else {
        &with_capacity
    };

    let selected = apply_strategy(candidates, strategy, round);
    Ok(selected.id.clone())
}

fn apply_strategy<'a>(
    candidates: &[&'a BackendState],
    strategy: &RoutingStrategy,
    round: usize,
) -> &'a BackendState {
    match strategy {
        RoutingStrategy::LeastLoaded => candidates
            .iter()
            .min_by_key(|b| b.active_requests)
            .expect("candidates must be non-empty"),
        RoutingStrategy::RoundRobin => candidates[round % candidates.len()],
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::LatencyMs;

    fn make_backend(
        id: &str,
        models: &[&str],
        healthy: bool,
        active: u32,
        max: u32,
    ) -> BackendState {
        let model_ids = models.iter().map(|m| ModelId::new(*m)).collect();
        let state = BackendState::new(BackendId::new(id), model_ids, max);
        let state = if healthy {
            state.with_healthy(LatencyMs::new(50))
        } else {
            state.with_unhealthy()
        };
        let mut state = state;
        for _ in 0..active {
            state = state.with_request_started();
        }
        state
    }

    #[test]
    fn test_affinity_hit() {
        let backends = vec![
            make_backend("gpu-0", &["llama3"], true, 1, 4),
            make_backend("gpu-1", &["llama3"], true, 0, 4),
        ];
        let model = ModelId::new("llama3");
        let preferred = BackendId::new("gpu-0");

        let result = select_backend(
            &backends,
            &model,
            &RoutingStrategy::LeastLoaded,
            0,
            Some(&preferred),
        );
        assert_eq!(result.unwrap(), BackendId::new("gpu-0"));
    }

    #[test]
    fn test_affinity_miss_unhealthy() {
        let backends = vec![
            make_backend("gpu-0", &["llama3"], false, 0, 4),
            make_backend("gpu-1", &["llama3"], true, 2, 4),
        ];
        let model = ModelId::new("llama3");
        let preferred = BackendId::new("gpu-0");

        let result = select_backend(
            &backends,
            &model,
            &RoutingStrategy::LeastLoaded,
            0,
            Some(&preferred),
        );
        assert_eq!(result.unwrap(), BackendId::new("gpu-1"));
    }

    #[test]
    fn test_affinity_miss_no_capacity() {
        let backends = vec![
            make_backend("gpu-0", &["llama3"], true, 4, 4),
            make_backend("gpu-1", &["llama3"], true, 1, 4),
        ];
        let model = ModelId::new("llama3");
        let preferred = BackendId::new("gpu-0");

        let result = select_backend(
            &backends,
            &model,
            &RoutingStrategy::LeastLoaded,
            0,
            Some(&preferred),
        );
        assert_eq!(result.unwrap(), BackendId::new("gpu-1"));
    }

    #[test]
    fn test_least_loaded() {
        let backends = vec![
            make_backend("gpu-0", &["llama3"], true, 3, 4),
            make_backend("gpu-1", &["llama3"], true, 1, 4),
            make_backend("gpu-2", &["llama3"], true, 2, 4),
        ];
        let model = ModelId::new("llama3");

        let result = select_backend(&backends, &model, &RoutingStrategy::LeastLoaded, 0, None);
        assert_eq!(result.unwrap(), BackendId::new("gpu-1"));
    }

    #[test]
    fn test_round_robin() {
        let backends = vec![
            make_backend("gpu-0", &["llama3"], true, 0, 4),
            make_backend("gpu-1", &["llama3"], true, 0, 4),
            make_backend("gpu-2", &["llama3"], true, 0, 4),
        ];
        let model = ModelId::new("llama3");

        let r0 = select_backend(&backends, &model, &RoutingStrategy::RoundRobin, 0, None);
        let r1 = select_backend(&backends, &model, &RoutingStrategy::RoundRobin, 1, None);
        let r2 = select_backend(&backends, &model, &RoutingStrategy::RoundRobin, 2, None);
        let r3 = select_backend(&backends, &model, &RoutingStrategy::RoundRobin, 3, None);

        assert_eq!(r0.unwrap(), BackendId::new("gpu-0"));
        assert_eq!(r1.unwrap(), BackendId::new("gpu-1"));
        assert_eq!(r2.unwrap(), BackendId::new("gpu-2"));
        assert_eq!(r3.unwrap(), BackendId::new("gpu-0")); // wraps around
    }

    #[test]
    fn test_model_not_found() {
        let backends = vec![make_backend("gpu-0", &["llama3"], true, 0, 4)];
        let model = ModelId::new("gpt-4");

        let result = select_backend(&backends, &model, &RoutingStrategy::LeastLoaded, 0, None);
        assert!(matches!(result, Err(RoutingError::ModelNotFound { .. })));
    }

    #[test]
    fn test_no_healthy_backend() {
        let backends = vec![
            make_backend("gpu-0", &["llama3"], false, 0, 4),
            make_backend("gpu-1", &["llama3"], false, 0, 4),
        ];
        let model = ModelId::new("llama3");

        let result = select_backend(&backends, &model, &RoutingStrategy::LeastLoaded, 0, None);
        assert!(matches!(result, Err(RoutingError::NoHealthyBackend { .. })));
    }

    #[test]
    fn test_all_at_capacity_still_routes() {
        let backends = vec![
            make_backend("gpu-0", &["llama3"], true, 4, 4),
            make_backend("gpu-1", &["llama3"], true, 4, 4),
        ];
        let model = ModelId::new("llama3");

        let result = select_backend(&backends, &model, &RoutingStrategy::LeastLoaded, 0, None);
        // Should still route even when all at capacity (overload)
        assert!(result.is_ok());
    }
}
