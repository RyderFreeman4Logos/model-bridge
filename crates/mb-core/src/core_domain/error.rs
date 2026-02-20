use crate::core::{BackendId, ClientId, ModelId};

// ---------------------------------------------------------------------------
// Sub-error types
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("invalid API key")]
    InvalidApiKey,
    #[error("client {client} not permitted to use model {model}")]
    ModelNotPermitted { model: ModelId, client: ClientId },
}

#[derive(Debug, thiserror::Error)]
pub enum RoutingError {
    #[error("no healthy backend for model {model}")]
    NoHealthyBackend { model: ModelId },
    #[error("model {model} not found")]
    ModelNotFound { model: ModelId },
}

#[derive(Debug, thiserror::Error)]
pub enum AdapterError {
    #[error("failed to parse request: {0}")]
    ParseRequest(String),
    #[error("failed to format response: {0}")]
    FormatResponse(String),
    #[error("unsupported feature: {0}")]
    UnsupportedFeature(String),
}

#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("backend returned HTTP {status}: {body}")]
    HttpStatus { status: u16, body: String },
    #[error("backend connection failed: {0}")]
    Connection(String),
    #[error("backend {backend} timed out after {timeout_ms}ms")]
    Timeout { backend: BackendId, timeout_ms: u64 },
}

#[derive(Debug, thiserror::Error)]
pub enum HealthError {
    #[error("health check connection failed: {0}")]
    ConnectionFailed(String),
    #[error("health check timed out")]
    Timeout,
    #[error("health check returned unexpected status: {0}")]
    UnexpectedStatus(u16),
}

// ---------------------------------------------------------------------------
// Data structs carried by GatewayError variants
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct RateLimitInfo {
    pub retry_after_ms: u64,
}

#[derive(Clone, Debug)]
pub struct QuotaInfo {
    pub limit: u64,
    pub used: u64,
}

// ---------------------------------------------------------------------------
// Top-level error
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum GatewayError {
    #[error(transparent)]
    Auth(#[from] AuthError),
    #[error(transparent)]
    Routing(#[from] RoutingError),
    #[error("rate limited, retry after {}ms", .0.retry_after_ms)]
    RateLimited(RateLimitInfo),
    #[error("quota exceeded: {}/{}", .0.used, .0.limit)]
    QuotaExceeded(QuotaInfo),
    #[error(transparent)]
    Adapter(#[from] AdapterError),
    #[error(transparent)]
    Backend(#[from] BackendError),
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- From conversions --

    #[test]
    fn test_from_auth_error_to_gateway_error() {
        let err: GatewayError = AuthError::InvalidApiKey.into();
        assert!(matches!(err, GatewayError::Auth(AuthError::InvalidApiKey)));
    }

    #[test]
    fn test_from_routing_error_to_gateway_error() {
        let model = ModelId::new("llama3-70b");
        let err: GatewayError = RoutingError::NoHealthyBackend { model }.into();
        assert!(matches!(
            err,
            GatewayError::Routing(RoutingError::NoHealthyBackend { .. })
        ));
    }

    #[test]
    fn test_from_adapter_error_to_gateway_error() {
        let err: GatewayError = AdapterError::ParseRequest("bad json".into()).into();
        assert!(matches!(
            err,
            GatewayError::Adapter(AdapterError::ParseRequest(_))
        ));
    }

    #[test]
    fn test_from_backend_error_to_gateway_error() {
        let err: GatewayError = BackendError::Connection("refused".into()).into();
        assert!(matches!(
            err,
            GatewayError::Backend(BackendError::Connection(_))
        ));
    }

    // -- Display formatting --

    #[test]
    fn test_display_auth_invalid_key() {
        let err = AuthError::InvalidApiKey;
        assert_eq!(err.to_string(), "invalid API key");
    }

    #[test]
    fn test_display_auth_model_not_permitted() {
        let err = AuthError::ModelNotPermitted {
            model: ModelId::new("gpt-4"),
            client: ClientId::new("team-alpha"),
        };
        assert_eq!(
            err.to_string(),
            "client team-alpha not permitted to use model gpt-4"
        );
    }

    #[test]
    fn test_display_routing_no_healthy_backend() {
        let err = RoutingError::NoHealthyBackend {
            model: ModelId::new("llama3-70b"),
        };
        assert_eq!(err.to_string(), "no healthy backend for model llama3-70b");
    }

    #[test]
    fn test_display_routing_model_not_found() {
        let err = RoutingError::ModelNotFound {
            model: ModelId::new("nonexistent"),
        };
        assert_eq!(err.to_string(), "model nonexistent not found");
    }

    #[test]
    fn test_display_adapter_parse_request() {
        let err = AdapterError::ParseRequest("unexpected EOF".into());
        assert_eq!(err.to_string(), "failed to parse request: unexpected EOF");
    }

    #[test]
    fn test_display_adapter_format_response() {
        let err = AdapterError::FormatResponse("serialization failed".into());
        assert_eq!(
            err.to_string(),
            "failed to format response: serialization failed"
        );
    }

    #[test]
    fn test_display_adapter_unsupported_feature() {
        let err = AdapterError::UnsupportedFeature("tool_use".into());
        assert_eq!(err.to_string(), "unsupported feature: tool_use");
    }

    #[test]
    fn test_display_backend_http_status() {
        let err = BackendError::HttpStatus {
            status: 500,
            body: "internal error".into(),
        };
        assert_eq!(err.to_string(), "backend returned HTTP 500: internal error");
    }

    #[test]
    fn test_display_backend_connection() {
        let err = BackendError::Connection("connection refused".into());
        assert_eq!(
            err.to_string(),
            "backend connection failed: connection refused"
        );
    }

    #[test]
    fn test_display_backend_timeout() {
        let err = BackendError::Timeout {
            backend: BackendId::new("gpu-1"),
            timeout_ms: 5000,
        };
        assert_eq!(err.to_string(), "backend gpu-1 timed out after 5000ms");
    }

    #[test]
    fn test_display_health_connection_failed() {
        let err = HealthError::ConnectionFailed("dns lookup failed".into());
        assert_eq!(
            err.to_string(),
            "health check connection failed: dns lookup failed"
        );
    }

    #[test]
    fn test_display_health_timeout() {
        let err = HealthError::Timeout;
        assert_eq!(err.to_string(), "health check timed out");
    }

    #[test]
    fn test_display_health_unexpected_status() {
        let err = HealthError::UnexpectedStatus(503);
        assert_eq!(
            err.to_string(),
            "health check returned unexpected status: 503"
        );
    }

    #[test]
    fn test_display_gateway_rate_limited() {
        let err = GatewayError::RateLimited(RateLimitInfo {
            retry_after_ms: 1500,
        });
        assert_eq!(err.to_string(), "rate limited, retry after 1500ms");
    }

    #[test]
    fn test_display_gateway_quota_exceeded() {
        let err = GatewayError::QuotaExceeded(QuotaInfo {
            limit: 100_000,
            used: 100_001,
        });
        assert_eq!(err.to_string(), "quota exceeded: 100001/100000");
    }

    #[test]
    fn test_display_gateway_transparent_auth() {
        let err: GatewayError = AuthError::InvalidApiKey.into();
        assert_eq!(err.to_string(), "invalid API key");
    }

    #[test]
    fn test_display_gateway_transparent_routing() {
        let err: GatewayError = RoutingError::ModelNotFound {
            model: ModelId::new("llama3-70b"),
        }
        .into();
        assert_eq!(err.to_string(), "model llama3-70b not found");
    }
}
