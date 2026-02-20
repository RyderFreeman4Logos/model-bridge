use super::*;

#[test]
fn test_parse_full_config() {
    let toml_str = r#"
[server]
listen = "127.0.0.1:9090"
tls_cert = "/path/to/cert.pem"
tls_key = "/path/to/key.pem"

[routing]
strategy = "round-robin"
cache_aware = false
prefix_depth = 2
max_affinity_entries = 5000

[health]
check_interval_secs = 15
timeout_ms = 3000
unhealthy_threshold = 5
degraded_latency_ms = 1000

[logging]
level = "debug"
format = "pretty"

[[clients]]
id = "team-alpha"
api_key = "mb-sk-abcdefghijklmnopqrstuvwxyz012345"
allowed_models = ["llama3-70b", "mistral-7b"]
rate_limit_rpm = 120
rate_limit_tpm = 100000
monthly_token_limit = 5000000

[[backends]]
id = "gpu-desktop"
base_url = "http://100.64.0.1:8000"
api_key = "sk-local-xxxxx"
spec = "openai-chat"
models = ["llama3-70b"]
max_concurrent = 20
"#;

    let config: AppConfig = toml::from_str(toml_str).unwrap();

    assert_eq!(config.server.listen, "127.0.0.1:9090");
    assert_eq!(config.server.tls_cert.as_deref(), Some("/path/to/cert.pem"));
    assert_eq!(config.server.tls_key.as_deref(), Some("/path/to/key.pem"));

    assert_eq!(config.routing.strategy, RoutingStrategyConfig::RoundRobin);
    assert!(!config.routing.cache_aware);
    assert_eq!(config.routing.prefix_depth, 2);
    assert_eq!(config.routing.max_affinity_entries, 5000);

    assert_eq!(config.health.check_interval_secs, 15);
    assert_eq!(config.health.timeout_ms, 3000);
    assert_eq!(config.health.unhealthy_threshold, 5);
    assert_eq!(config.health.degraded_latency_ms, 1000);

    assert_eq!(config.logging.level, "debug");
    assert_eq!(config.logging.format, "pretty");

    assert_eq!(config.clients.len(), 1);
    let client = &config.clients[0];
    assert_eq!(client.id, "team-alpha");
    assert_eq!(
        client.allowed_models,
        AllowedModelsConfig::Specific(vec!["llama3-70b".to_owned(), "mistral-7b".to_owned()])
    );
    assert_eq!(client.rate_limit_rpm, 120);
    assert_eq!(client.rate_limit_tpm, Some(100_000));
    assert_eq!(client.monthly_token_limit, Some(5_000_000));

    assert_eq!(config.backends.len(), 1);
    let backend = &config.backends[0];
    assert_eq!(backend.id, "gpu-desktop");
    assert_eq!(backend.base_url, "http://100.64.0.1:8000");
    assert_eq!(backend.api_key.as_deref(), Some("sk-local-xxxxx"));
    assert_eq!(backend.spec, BackendSpecConfig::OpenaiChat);
    assert_eq!(backend.models, vec!["llama3-70b"]);
    assert_eq!(backend.max_concurrent, 20);
}

#[test]
fn test_defaults_applied() {
    let toml_str = r#"
[[clients]]
id = "test-client"
api_key = "mb-sk-testkey00000000000000000000000"
allowed_models = "*"
rate_limit_rpm = 60

[[backends]]
id = "local"
base_url = "http://localhost:11434"
spec = "ollama"
models = ["llama3"]
"#;

    let config: AppConfig = toml::from_str(toml_str).unwrap();

    // ServerConfig defaults
    assert_eq!(config.server.listen, "0.0.0.0:8080");
    assert!(config.server.tls_cert.is_none());
    assert!(config.server.tls_key.is_none());

    // RoutingConfig defaults
    assert_eq!(config.routing.strategy, RoutingStrategyConfig::LeastLoaded);
    assert!(config.routing.cache_aware);
    assert_eq!(config.routing.prefix_depth, 3);
    assert_eq!(config.routing.max_affinity_entries, 10_000);

    // HealthConfig defaults
    assert_eq!(config.health.check_interval_secs, 30);
    assert_eq!(config.health.timeout_ms, 5000);
    assert_eq!(config.health.unhealthy_threshold, 3);
    assert_eq!(config.health.degraded_latency_ms, 2000);

    // LoggingConfig defaults
    assert_eq!(config.logging.level, "info");
    assert_eq!(config.logging.format, "json");

    // BackendConfig max_concurrent default
    assert_eq!(config.backends[0].max_concurrent, 64);
}

#[test]
fn test_wildcard_allowed_models() {
    let toml_str = r#"
[[clients]]
id = "admin"
api_key = "mb-sk-admin000000000000000000000000"
allowed_models = "*"
rate_limit_rpm = 1000

[[backends]]
id = "b1"
base_url = "http://localhost:8000"
spec = "openai-chat"
models = ["m1"]
"#;

    let config: AppConfig = toml::from_str(toml_str).unwrap();
    assert_eq!(
        config.clients[0].allowed_models,
        AllowedModelsConfig::All(WildcardMarker)
    );
}

#[test]
fn test_specific_allowed_models() {
    let toml_str = r#"
[[clients]]
id = "restricted"
api_key = "mb-sk-restrict00000000000000000000"
allowed_models = ["gpt-4", "claude-3"]
rate_limit_rpm = 30

[[backends]]
id = "b1"
base_url = "http://localhost:8000"
spec = "openai-chat"
models = ["gpt-4"]
"#;

    let config: AppConfig = toml::from_str(toml_str).unwrap();
    assert_eq!(
        config.clients[0].allowed_models,
        AllowedModelsConfig::Specific(vec!["gpt-4".to_owned(), "claude-3".to_owned()])
    );
}
