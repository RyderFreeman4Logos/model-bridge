use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub routing: RoutingConfig,
    #[serde(default)]
    pub health: HealthConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
    pub clients: Vec<ClientConfig>,
    pub backends: Vec<BackendConfig>,
}

impl AppConfig {
    pub fn from_file(path: &Path) -> Result<Self, anyhow::Error> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub listen: String,
    pub tls_cert: Option<String>,
    pub tls_key: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            listen: "0.0.0.0:8080".to_owned(),
            tls_cert: None,
            tls_key: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct RoutingConfig {
    pub strategy: RoutingStrategyConfig,
    pub cache_aware: bool,
    pub prefix_depth: usize,
    pub max_affinity_entries: usize,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            strategy: RoutingStrategyConfig::default(),
            cache_aware: true,
            prefix_depth: 3,
            max_affinity_entries: 10_000,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum RoutingStrategyConfig {
    #[default]
    LeastLoaded,
    RoundRobin,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct HealthConfig {
    pub check_interval_secs: u64,
    pub timeout_ms: u64,
    pub unhealthy_threshold: u32,
    pub degraded_latency_ms: u64,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            check_interval_secs: 30,
            timeout_ms: 5000,
            unhealthy_threshold: 3,
            degraded_latency_ms: 2000,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct LoggingConfig {
    pub level: String,
    pub format: String,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_owned(),
            format: "json".to_owned(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ClientConfig {
    pub id: String,
    pub api_key: String,
    pub allowed_models: AllowedModelsConfig,
    pub rate_limit_rpm: u32,
    pub rate_limit_tpm: Option<u64>,
    pub monthly_token_limit: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum AllowedModelsConfig {
    All(WildcardMarker),
    Specific(Vec<String>),
}

/// Deserializes only the literal string `"*"`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WildcardMarker;

impl<'de> Deserialize<'de> for WildcardMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        if s == "*" {
            Ok(WildcardMarker)
        } else {
            Err(serde::de::Error::custom(
                "expected \"*\" for wildcard allowed_models",
            ))
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct BackendConfig {
    pub id: String,
    pub base_url: String,
    pub api_key: Option<String>,
    pub spec: BackendSpecConfig,
    pub models: Vec<String>,
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent: u32,
}

fn default_max_concurrent() -> u32 {
    64
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum BackendSpecConfig {
    OpenaiChat,
    Ollama,
}

#[cfg(test)]
mod tests;
