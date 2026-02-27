use std::collections::HashSet;

use anyhow::ensure;
use mb_core::core::{
    AllowedModels, ApiKey, AuthService, BackendId, BackendInfo, BackendSpec, ClientId, ClientInfo,
    ModelId, QuotaConfig, RateLimit, RoutingStrategy,
};

use crate::config::{AllowedModelsConfig, AppConfig, BackendSpecConfig, RoutingStrategyConfig};

// ---------------------------------------------------------------------------
// CacheConfig — cache-aware routing configuration
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct CacheConfig {
    pub enabled: bool,
    pub prefix_depth: usize,
    pub max_entries: usize,
}

// ---------------------------------------------------------------------------
// RuntimeConfig — fully validated runtime configuration
// ---------------------------------------------------------------------------

pub struct RuntimeConfig {
    pub auth_service: AuthService,
    pub backends: Vec<BackendInfo>,
    pub routing_strategy: RoutingStrategy,
    pub health_check_interval_secs: u64,
    pub health_timeout_ms: u64,
    pub unhealthy_threshold: u32,
    pub degraded_latency_ms: u64,
    pub cache_config: CacheConfig,
    pub listen_addr: String,
    pub log_level: String,
    pub log_format: String,
    /// Per-client rate limit (RPM) for lazy RateLimiter creation.
    pub client_rate_limits: std::collections::HashMap<ClientId, u32>,
    /// Per-backend API keys for authenticating outbound requests.
    pub backend_api_keys: std::collections::HashMap<BackendId, ApiKey>,
}

// ---------------------------------------------------------------------------
// into_runtime — converts raw AppConfig into validated RuntimeConfig
// ---------------------------------------------------------------------------

pub fn into_runtime(config: AppConfig) -> Result<RuntimeConfig, anyhow::Error> {
    ensure!(!config.clients.is_empty(), "at least one client required");
    ensure!(!config.backends.is_empty(), "at least one backend required");

    // Detect duplicate client IDs
    let mut seen_clients = HashSet::with_capacity(config.clients.len());
    for client in &config.clients {
        ensure!(
            seen_clients.insert(&client.id),
            "duplicate client id: {}",
            client.id
        );
    }

    // Detect duplicate backend IDs
    let mut seen_backends = HashSet::with_capacity(config.backends.len());
    for backend in &config.backends {
        ensure!(
            seen_backends.insert(&backend.id),
            "duplicate backend id: {}",
            backend.id
        );
    }

    // Convert clients → AuthService
    let client_entries: Vec<(ApiKey, ClientInfo)> = config
        .clients
        .into_iter()
        .map(|c| {
            let key = ApiKey::new(c.api_key);
            let allowed_models = match c.allowed_models {
                AllowedModelsConfig::All(_) => AllowedModels::All,
                AllowedModelsConfig::Specific(list) => {
                    AllowedModels::Specific(list.into_iter().map(ModelId::new).collect())
                }
            };
            let info = ClientInfo {
                id: ClientId::new(c.id),
                allowed_models,
                rate_limit: RateLimit {
                    requests_per_minute: c.rate_limit_rpm,
                    tokens_per_minute: c.rate_limit_tpm,
                },
                quota: QuotaConfig {
                    monthly_token_limit: c.monthly_token_limit,
                },
            };
            (key, info)
        })
        .collect();

    let client_rate_limits: std::collections::HashMap<ClientId, u32> = client_entries
        .iter()
        .map(|(_, info)| (info.id.clone(), info.rate_limit.requests_per_minute))
        .collect();
    let auth_service = AuthService::new(client_entries);

    // Convert backends → Vec<BackendInfo> and extract API keys
    let mut backend_api_keys = std::collections::HashMap::new();
    let backends: Vec<BackendInfo> = config
        .backends
        .into_iter()
        .map(|b| {
            let id = BackendId::new(b.id);
            if let Some(key) = b.api_key {
                backend_api_keys.insert(id.clone(), ApiKey::new(key));
            }
            BackendInfo {
                id,
                spec: match b.spec {
                    BackendSpecConfig::OpenaiChat => BackendSpec::OpenAiChat,
                    BackendSpecConfig::Ollama => BackendSpec::Ollama,
                },
                models: b.models.into_iter().map(ModelId::new).collect(),
                max_concurrent: b.max_concurrent,
                base_url: b.base_url,
            }
        })
        .collect();

    // Convert routing strategy
    let routing_strategy = match config.routing.strategy {
        RoutingStrategyConfig::LeastLoaded => RoutingStrategy::LeastLoaded,
        RoutingStrategyConfig::RoundRobin => RoutingStrategy::RoundRobin,
    };

    let cache_config = CacheConfig {
        enabled: config.routing.cache_aware,
        prefix_depth: config.routing.prefix_depth,
        max_entries: config.routing.max_affinity_entries,
    };

    Ok(RuntimeConfig {
        auth_service,
        backends,
        routing_strategy,
        health_check_interval_secs: config.health.check_interval_secs,
        health_timeout_ms: config.health.timeout_ms,
        unhealthy_threshold: config.health.unhealthy_threshold,
        degraded_latency_ms: config.health.degraded_latency_ms,
        cache_config,
        listen_addr: config.server.listen,
        log_level: config.logging.level,
        log_format: config.logging.format,
        client_rate_limits,
        backend_api_keys,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        BackendConfig, BackendSpecConfig, ClientConfig, HealthConfig, LoggingConfig, RoutingConfig,
        ServerConfig, WildcardMarker,
    };

    fn make_client(id: &str, api_key: &str) -> ClientConfig {
        ClientConfig {
            id: id.to_owned(),
            api_key: api_key.to_owned(),
            allowed_models: AllowedModelsConfig::Specific(vec!["llama3-70b".to_owned()]),
            rate_limit_rpm: 60,
            rate_limit_tpm: None,
            monthly_token_limit: None,
        }
    }

    fn make_backend(id: &str) -> BackendConfig {
        BackendConfig {
            id: id.to_owned(),
            base_url: "http://100.64.0.1:8000".to_owned(),
            api_key: None,
            spec: BackendSpecConfig::OpenaiChat,
            models: vec!["llama3-70b".to_owned()],
            max_concurrent: 10,
        }
    }

    fn make_config() -> AppConfig {
        AppConfig {
            server: ServerConfig::default(),
            routing: RoutingConfig::default(),
            health: HealthConfig::default(),
            logging: LoggingConfig::default(),
            clients: vec![make_client(
                "team-alpha",
                "mb-sk-test00000000000000000000000",
            )],
            backends: vec![make_backend("gpu-desktop")],
        }
    }

    #[test]
    fn test_valid_config_conversion() {
        let config = make_config();
        let runtime = into_runtime(config).expect("valid config should convert");

        assert_eq!(runtime.backends.len(), 1);
        assert_eq!(runtime.backends[0].id, BackendId::new("gpu-desktop"));
        assert_eq!(runtime.backends[0].spec, BackendSpec::OpenAiChat);
        assert_eq!(runtime.backends[0].models.len(), 1);
        assert_eq!(runtime.backends[0].max_concurrent, 10);
        assert_eq!(runtime.routing_strategy, RoutingStrategy::LeastLoaded);
        assert_eq!(runtime.health_check_interval_secs, 30);
        assert_eq!(runtime.health_timeout_ms, 5000);
        assert_eq!(runtime.unhealthy_threshold, 3);
        assert_eq!(runtime.degraded_latency_ms, 2000);
        assert!(runtime.cache_config.enabled);
        assert_eq!(runtime.listen_addr, "0.0.0.0:8080");
    }

    #[test]
    fn test_wildcard_models() {
        let mut config = make_config();
        config.clients[0].allowed_models = AllowedModelsConfig::All(WildcardMarker);

        let runtime = into_runtime(config).expect("wildcard config should convert");

        let key = ApiKey::new("mb-sk-test00000000000000000000000");
        let client = runtime
            .auth_service
            .validate(&key)
            .expect("key should be valid");
        assert!(matches!(client.allowed_models, AllowedModels::All));
    }

    #[test]
    fn test_empty_clients_rejected() {
        let mut config = make_config();
        config.clients.clear();

        match into_runtime(config) {
            Err(e) => assert!(e.to_string().contains("at least one client required")),
            Ok(_) => panic!("expected error for empty clients"),
        }
    }

    #[test]
    fn test_empty_backends_rejected() {
        let mut config = make_config();
        config.backends.clear();

        match into_runtime(config) {
            Err(e) => assert!(e.to_string().contains("at least one backend required")),
            Ok(_) => panic!("expected error for empty backends"),
        }
    }

    #[test]
    fn test_duplicate_client_ids() {
        let mut config = make_config();
        config.clients.push(make_client(
            "team-alpha",
            "mb-sk-other00000000000000000000000",
        ));

        match into_runtime(config) {
            Err(e) => assert!(e.to_string().contains("duplicate client id")),
            Ok(_) => panic!("expected error for duplicate client ids"),
        }
    }

    #[test]
    fn test_duplicate_backend_ids() {
        let mut config = make_config();
        config.backends.push(make_backend("gpu-desktop"));

        match into_runtime(config) {
            Err(e) => assert!(e.to_string().contains("duplicate backend id")),
            Ok(_) => panic!("expected error for duplicate backend ids"),
        }
    }
}
