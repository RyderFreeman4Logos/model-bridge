use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use clap::{Parser, Subcommand};
use tokio::sync::RwLock;

use mb_core::core::{CacheAffinityMap, QuotaTracker};
use mb_server::bootstrap::{self, CacheConfig};
use mb_server::config::AppConfig;
use mb_server::handler::{self, AppState, BackendMeta};
use mb_server::health::{self, HealthCheckManager, HttpHealthProbe};
use mb_server::inbound::InboundAdapterRegistry;
use mb_server::outbound::OutboundAdapterRegistry;
// stream_handler is available but streaming dispatch is handled by the
// request handler detecting stream=true in the parsed canonical request.

#[derive(Parser)]
#[command(name = "mb", about = "model-bridge LLM API gateway")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Path to the configuration file.
    #[arg(short, long, default_value = "config.toml", global = true)]
    config: PathBuf,
}

#[derive(Subcommand)]
enum Command {
    /// Validate configuration file and exit.
    Validate,
    /// Generate a new API key.
    Genkey,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Some(Command::Validate) => run_validate(&cli.config),
        Some(Command::Genkey) => run_genkey(),
        None => {
            let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
            rt.block_on(run_gateway(cli.config));
        }
    }
}

fn run_validate(path: &std::path::Path) {
    let config = match AppConfig::from_file(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading config: {e}");
            std::process::exit(1);
        }
    };

    match bootstrap::into_runtime(config) {
        Ok(_) => println!("Config valid: {}", path.display()),
        Err(e) => {
            eprintln!("Config invalid: {e}");
            std::process::exit(1);
        }
    }
}

fn run_genkey() {
    use rand::Rng;
    const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";
    let mut rng = rand::rng();
    let key: String = (0..32)
        .map(|_| {
            let idx = rng.random_range(0..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect();
    println!("mb-sk-{key}");
}

async fn run_gateway(config_path: PathBuf) {
    let config = match AppConfig::from_file(&config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading config: {e}");
            std::process::exit(1);
        }
    };

    let runtime = match bootstrap::into_runtime(config) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Config invalid: {e}");
            std::process::exit(1);
        }
    };

    // Initialize tracing
    init_tracing(&runtime.log_level, &runtime.log_format);
    tracing::info!("Starting model-bridge gateway");

    let rate_limit_rpm = runtime.client_rate_limits;

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
                    api_key: runtime.backend_api_keys.get(&b.id).cloned(),
                },
            )
        })
        .collect();

    // Initialize health manager
    let health_manager = HealthCheckManager::new(&runtime.backends);
    let backend_states = health_manager.shared_states();

    // Start background health checks
    let probe = Arc::new(
        HttpHealthProbe::new(Duration::from_millis(runtime.health_timeout_ms))
            .expect("failed to build health probe HTTP client"),
    );
    let _health_handle = health_manager.start_background_checks(
        runtime.backends.clone(),
        Duration::from_secs(runtime.health_check_interval_secs),
        runtime.unhealthy_threshold,
        runtime.degraded_latency_ms,
        probe,
    );

    // Build AppState
    #[cfg(feature = "feedback")]
    let feedback = init_feedback_state().await;

    let state = Arc::new(AppState {
        auth: runtime.auth_service,
        inbound_registry: InboundAdapterRegistry::new(),
        outbound_registry: OutboundAdapterRegistry::new(),
        backend_states: backend_states.clone(),
        rate_limiters: RwLock::new(HashMap::new()),
        quota_tracker: RwLock::new(QuotaTracker::new()),
        affinity_map: RwLock::new(CacheAffinityMap::new(runtime.cache_config.max_entries)),
        http_client: reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("failed to build HTTP client"),
        routing_strategy: runtime.routing_strategy,
        cache_config: CacheConfig {
            enabled: runtime.cache_config.enabled,
            prefix_depth: runtime.cache_config.prefix_depth,
            max_entries: runtime.cache_config.max_entries,
        },
        round_counter: AtomicUsize::new(0),
        rate_limit_rpm,
        backends_by_id,
        #[cfg(feature = "feedback")]
        feedback,
    });

    // Build axum router
    // Both streaming and non-streaming requests arrive via POST.
    // The handler inspects the parsed request's `stream` field to dispatch.
    // For now, non-streaming handler is the default POST handler.
    // Streaming is dispatched internally based on the request body.
    let app = axum::Router::new()
        .route("/v1/chat/completions", post(handler::handle_completion))
        .route(
            "/health",
            get({
                let states = backend_states;
                move || health::health_handler(states)
            }),
        );

    #[cfg(feature = "feedback")]
    let app = app
        .route("/v1/feedback", post(mb_server::feedback::post_feedback))
        .route(
            "/v1/my-annotations",
            get(mb_server::feedback::get_my_annotations),
        );

    let app = app
        .layer(DefaultBodyLimit::max(2 * 1024 * 1024))
        .with_state(state);

    // Start server
    let listener = tokio::net::TcpListener::bind(&runtime.listen_addr)
        .await
        .expect("failed to bind listener");
    tracing::info!("Listening on {}", runtime.listen_addr);

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server error");

    tracing::info!("Gateway shut down");
}

#[cfg(feature = "feedback")]
async fn init_feedback_state() -> Option<mb_server::feedback::FeedbackState> {
    let db_path =
        std::env::var("MB_FEEDBACK_DB_PATH").unwrap_or_else(|_| "feedback.sqlite".to_owned());
    let db_path_for_task = db_path.clone();

    let init_result = tokio::task::spawn_blocking(move || {
        let sqlite_store =
            mb_feedback::SqliteFeedbackStore::new(std::path::Path::new(&db_path_for_task))
                .map_err(|err| err.to_string())?;
        mb_feedback::FeedbackStore::init(&sqlite_store).map_err(|err| err.to_string())?;
        let store: Arc<dyn mb_feedback::FeedbackStore> = Arc::new(sqlite_store);
        Ok::<Arc<dyn mb_feedback::FeedbackStore>, String>(store)
    })
    .await;

    match init_result {
        Ok(Ok(store)) => {
            tracing::info!("feedback store initialized at {}", db_path);
            Some(mb_server::feedback::FeedbackState { store })
        }
        Ok(Err(err)) => {
            tracing::warn!(
                error = %err,
                db_path = %db_path,
                "failed to initialize feedback store; feedback logging disabled"
            );
            None
        }
        Err(err) => {
            tracing::warn!(
                error = %err,
                db_path = %db_path,
                "feedback initialization task failed; feedback logging disabled"
            );
            None
        }
    }
}

fn init_tracing(level: &str, format: &str) {
    use tracing_subscriber::EnvFilter;

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level));

    match format {
        "json" => {
            tracing_subscriber::fmt()
                .with_env_filter(filter)
                .json()
                .init();
        }
        _ => {
            tracing_subscriber::fmt().with_env_filter(filter).init();
        }
    }
}

async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();
    #[cfg(unix)]
    {
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to register SIGTERM handler");
        tokio::select! {
            _ = ctrl_c => {},
            _ = sigterm.recv() => {},
        }
    }
    #[cfg(not(unix))]
    {
        ctrl_c.await.ok();
    }
    tracing::info!("Shutdown signal received");
}
