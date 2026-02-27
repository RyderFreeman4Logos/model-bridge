use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use tokio::sync::RwLock;

use mb_core::core::{BackendState, CacheAffinityMap, LatencyMs, QuotaTracker};
use mb_server::bootstrap::CacheConfig;
use mb_server::config::{
    AllowedModelsConfig, AppConfig, BackendConfig, BackendSpecConfig, ClientConfig, HealthConfig,
    LoggingConfig, RoutingConfig, RoutingStrategyConfig, ServerConfig,
};
use mb_server::handler::{AppState, BackendMeta};
use mb_server::inbound::InboundAdapterRegistry;
use mb_server::outbound::OutboundAdapterRegistry;

// ---------------------------------------------------------------------------
// MockBackendServer — configurable mock that mimics an LLM backend
// ---------------------------------------------------------------------------

enum MockMode {
    Json {
        body: String,
        status: u16,
        delay_ms: u64,
    },
    Sse {
        body: String,
    },
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
        let mode = Arc::new(MockMode::Json {
            body: response_body.to_owned(),
            status,
            delay_ms,
        });
        Self::start_server(mode).await
    }

    /// Start a mock that returns SSE-formatted streaming events.
    pub async fn start_sse(events: &[&str]) -> Self {
        let sse_body: String = events
            .iter()
            .map(|e| format!("data: {e}\n\n"))
            .collect::<String>()
            + "data: [DONE]\n\n";

        let mode = Arc::new(MockMode::Sse { body: sse_body });
        Self::start_server(mode).await
    }

    async fn start_server(mode: Arc<MockMode>) -> Self {
        let app = axum::Router::new()
            .route("/v1/chat/completions", post(mock_handler))
            .route("/v1/models", get(mock_models_handler))
            .with_state(mode);

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

async fn mock_handler(State(mode): State<Arc<MockMode>>, _body: Bytes) -> Response {
    match mode.as_ref() {
        MockMode::Json {
            body,
            status,
            delay_ms,
        } => {
            if *delay_ms > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(*delay_ms)).await;
            }
            let status = StatusCode::from_u16(*status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            (
                status,
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                body.clone(),
            )
                .into_response()
        }
        MockMode::Sse { body } => (
            StatusCode::OK,
            [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
            body.clone(),
        )
            .into_response(),
    }
}

async fn mock_models_handler() -> Response {
    (StatusCode::OK, axum::Json(serde_json::json!({"data": []}))).into_response()
}

// ---------------------------------------------------------------------------
// Stream-aware dispatch handler for tests
// ---------------------------------------------------------------------------

async fn dispatch_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&body) {
        if json
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            return mb_server::stream_handler::handle_completion_stream(
                State(state),
                headers,
                body,
            )
            .await;
        }
    }
    mb_server::handler::handle_completion(State(state), headers, body).await
}

// ---------------------------------------------------------------------------
// TestGateway — starts a real mb gateway against mock backends
// ---------------------------------------------------------------------------

pub struct TestGatewayOptions {
    pub mark_healthy: bool,
    pub rate_limit_rpm: u32,
    pub routing_strategy: RoutingStrategyConfig,
    pub enable_stream_dispatch: bool,
    pub cache_aware: bool,
}

impl Default for TestGatewayOptions {
    fn default() -> Self {
        Self {
            mark_healthy: true,
            rate_limit_rpm: 60,
            routing_strategy: RoutingStrategyConfig::LeastLoaded,
            enable_stream_dispatch: false,
            cache_aware: true,
        }
    }
}

pub struct TestGateway {
    pub addr: SocketAddr,
    _handle: tokio::task::JoinHandle<()>,
}

impl TestGateway {
    /// Start a gateway with one mock backend and one client (defaults).
    pub async fn start_simple(mock_url: &str) -> Self {
        Self::start(
            &[(mock_url.to_owned(), vec![TEST_MODEL.to_owned()])],
            &[(TEST_CLIENT_ID, TEST_API_KEY, vec![TEST_MODEL.to_owned()])],
            TestGatewayOptions::default(),
        )
        .await
    }

    /// Start a gateway with configurable backends, clients, and options.
    pub async fn start(
        mock_urls: &[(String, Vec<String>)],
        api_keys: &[(&str, &str, Vec<String>)],
        options: TestGatewayOptions,
    ) -> Self {
        let clients: Vec<ClientConfig> = api_keys
            .iter()
            .map(|(id, key, models)| ClientConfig {
                id: id.to_string(),
                api_key: key.to_string(),
                allowed_models: AllowedModelsConfig::Specific(models.clone()),
                rate_limit_rpm: options.rate_limit_rpm,
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
            routing: RoutingConfig {
                strategy: options.routing_strategy,
                cache_aware: options.cache_aware,
                ..RoutingConfig::default()
            },
            health: HealthConfig::default(),
            logging: LoggingConfig::default(),
            clients,
            backends,
        };

        let runtime =
            mb_server::bootstrap::into_runtime(config).expect("test config should be valid");

        let backends_by_id: HashMap<_, _> = runtime
            .backends
            .iter()
            .map(|b| {
                (
                    b.id.clone(),
                    BackendMeta {
                        base_url: b.base_url.clone(),
                        spec: b.spec,
                        api_key: None,
                    },
                )
            })
            .collect();

        let mut backend_state_map = HashMap::new();
        for b in &runtime.backends {
            let state = BackendState::new(b.id.clone(), b.models.clone(), b.max_concurrent);
            let state = if options.mark_healthy {
                state.with_healthy(LatencyMs::new(10))
            } else {
                state
            };
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
            #[cfg(feature = "feedback")]
            feedback: None,
        });

        let handler = if options.enable_stream_dispatch {
            post(dispatch_handler)
        } else {
            post(mb_server::handler::handle_completion)
        };

        let app = axum::Router::new()
            .route("/v1/chat/completions", handler)
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
    sample_openai_response_with_id("chatcmpl-test123")
}

pub fn sample_openai_response_with_id(id: &str) -> String {
    serde_json::json!({
        "id": id,
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

pub fn sample_stream_request_body() -> String {
    serde_json::json!({
        "model": TEST_MODEL,
        "messages": [
            {
                "role": "user",
                "content": "Hello"
            }
        ],
        "stream": true
    })
    .to_string()
}

pub fn sample_sse_chunks() -> Vec<String> {
    vec![
        serde_json::json!({
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": TEST_MODEL,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
        })
        .to_string(),
        serde_json::json!({
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": TEST_MODEL,
            "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": null}]
        })
        .to_string(),
        serde_json::json!({
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": TEST_MODEL,
            "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": null}]
        })
        .to_string(),
        serde_json::json!({
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": TEST_MODEL,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        })
        .to_string(),
    ]
}
