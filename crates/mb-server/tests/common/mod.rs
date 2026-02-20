use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use tokio::sync::RwLock;

use mb_core::core::{BackendState, CacheAffinityMap, LatencyMs, QuotaTracker};
use mb_server::bootstrap::CacheConfig;
use mb_server::config::{
    AllowedModelsConfig, AppConfig, BackendConfig, BackendSpecConfig, ClientConfig, HealthConfig,
    LoggingConfig, RoutingConfig, ServerConfig,
};
use mb_server::handler::{AppState, BackendMeta};
use mb_server::inbound::InboundAdapterRegistry;
use mb_server::outbound::OutboundAdapterRegistry;

// ---------------------------------------------------------------------------
// MockBackendServer — configurable mock that mimics an LLM backend
// ---------------------------------------------------------------------------

struct MockConfig {
    response_body: String,
    status_code: u16,
    delay_ms: u64,
}

pub struct MockBackendServer {
    addr: SocketAddr,
    _handle: tokio::task::JoinHandle<()>,
}

impl MockBackendServer {
    pub async fn start(response_body: &str) -> Self {
        Self::start_with_options(response_body, 200, 0).await
    }

    pub async fn start_with_options(response_body: &str, status: u16, delay_ms: u64) -> Self {
        let config = Arc::new(MockConfig {
            response_body: response_body.to_owned(),
            status_code: status,
            delay_ms,
        });

        let app = axum::Router::new()
            .route("/v1/chat/completions", post(mock_completion_handler))
            .route("/v1/models", get(mock_models_handler))
            .with_state(config);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock server");
        let addr = listener.local_addr().unwrap();

        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.ok();
        });

        Self {
            addr,
            _handle: handle,
        }
    }

    pub fn url(&self) -> String {
        format!("http://{}", self.addr)
    }
}

impl Drop for MockBackendServer {
    fn drop(&mut self) {
        self._handle.abort();
    }
}

async fn mock_completion_handler(State(config): State<Arc<MockConfig>>, _body: Bytes) -> Response {
    if config.delay_ms > 0 {
        tokio::time::sleep(std::time::Duration::from_millis(config.delay_ms)).await;
    }

    let status =
        StatusCode::from_u16(config.status_code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    (
        status,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        config.response_body.clone(),
    )
        .into_response()
}

async fn mock_models_handler() -> Response {
    (StatusCode::OK, axum::Json(serde_json::json!({"data": []}))).into_response()
}

// ---------------------------------------------------------------------------
// TestGateway — starts a real mb gateway against mock backends
// ---------------------------------------------------------------------------

pub struct TestGateway {
    pub addr: SocketAddr,
    _handle: tokio::task::JoinHandle<()>,
}

impl TestGateway {
    /// Start a gateway with one mock backend and one client.
    pub async fn start_simple(mock_url: &str) -> Self {
        Self::start(
            &[(mock_url.to_owned(), vec![TEST_MODEL.to_owned()])],
            &[(TEST_CLIENT_ID, TEST_API_KEY, vec![TEST_MODEL.to_owned()])],
        )
        .await
    }

    /// Start a gateway with configurable backends and clients.
    ///
    /// `mock_urls`: `(base_url, models)` tuples for each backend.
    /// `api_keys`: `(client_id, api_key, allowed_models)` tuples for each client.
    pub async fn start(
        mock_urls: &[(String, Vec<String>)],
        api_keys: &[(&str, &str, Vec<String>)],
    ) -> Self {
        let clients: Vec<ClientConfig> = api_keys
            .iter()
            .map(|(id, key, models)| ClientConfig {
                id: id.to_string(),
                api_key: key.to_string(),
                allowed_models: AllowedModelsConfig::Specific(models.clone()),
                rate_limit_rpm: 60,
                rate_limit_tpm: None,
                monthly_token_limit: None,
            })
            .collect();

        let backends: Vec<BackendConfig> = mock_urls
            .iter()
            .enumerate()
            .map(|(i, (url, models))| BackendConfig {
                id: format!("mock-{i}"),
                base_url: url.clone(),
                api_key: None,
                spec: BackendSpecConfig::OpenaiChat,
                models: models.clone(),
                max_concurrent: 64,
            })
            .collect();

        let config = AppConfig {
            server: ServerConfig {
                listen: "127.0.0.1:0".to_owned(),
                ..ServerConfig::default()
            },
            routing: RoutingConfig::default(),
            health: HealthConfig::default(),
            logging: LoggingConfig::default(),
            clients,
            backends,
        };

        let runtime =
            mb_server::bootstrap::into_runtime(config).expect("test config should be valid");

        // Build backend metadata lookup
        let backends_by_id: HashMap<_, _> = runtime
            .backends
            .iter()
            .map(|b| {
                (
                    b.id.clone(),
                    BackendMeta {
                        base_url: b.base_url.clone(),
                        spec: b.spec,
                    },
                )
            })
            .collect();

        // Mark all backends as healthy (skip real health probing)
        let mut backend_state_map = HashMap::new();
        for b in &runtime.backends {
            let state = BackendState::new(b.id.clone(), b.models.clone(), b.max_concurrent)
                .with_healthy(LatencyMs::new(10));
            backend_state_map.insert(b.id.clone(), state);
        }
        let backend_states = Arc::new(RwLock::new(backend_state_map));

        let state = Arc::new(AppState {
            auth: runtime.auth_service,
            inbound_registry: InboundAdapterRegistry::new(),
            outbound_registry: OutboundAdapterRegistry::new(),
            backend_states,
            rate_limiters: RwLock::new(HashMap::new()),
            quota_tracker: RwLock::new(QuotaTracker::new()),
            affinity_map: RwLock::new(CacheAffinityMap::new(runtime.cache_config.max_entries)),
            http_client: reqwest::Client::new(),
            routing_strategy: runtime.routing_strategy,
            cache_config: CacheConfig {
                enabled: runtime.cache_config.enabled,
                prefix_depth: runtime.cache_config.prefix_depth,
                max_entries: runtime.cache_config.max_entries,
            },
            round_counter: AtomicUsize::new(0),
            rate_limit_rpm: runtime.client_rate_limits,
            backends_by_id,
        });

        let app = axum::Router::new()
            .route(
                "/v1/chat/completions",
                post(mb_server::handler::handle_completion),
            )
            .with_state(state);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind gateway");
        let addr = listener.local_addr().unwrap();

        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.ok();
        });

        Self {
            addr,
            _handle: handle,
        }
    }

    pub fn url(&self) -> String {
        format!("http://{}", self.addr)
    }
}

impl Drop for TestGateway {
    fn drop(&mut self) {
        self._handle.abort();
    }
}

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

pub const TEST_API_KEY: &str = "mb-sk-test00000000000000000000000";
pub const TEST_CLIENT_ID: &str = "test-client";
pub const TEST_MODEL: &str = "llama3-70b";

pub fn sample_openai_response() -> String {
    serde_json::json!({
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": TEST_MODEL,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18
        }
    })
    .to_string()
}

pub fn sample_request_body() -> String {
    serde_json::json!({
        "model": TEST_MODEL,
        "messages": [
            {
                "role": "user",
                "content": "Hello"
            }
        ]
    })
    .to_string()
}
