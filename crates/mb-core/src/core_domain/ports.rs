use std::future::Future;
use std::pin::Pin;

use crate::core::{
    AdapterError, BackendId, CanonicalRequest, CanonicalResponse, CanonicalStreamChunk,
    HealthError, LatencyMs, ModelId,
};

// ---------------------------------------------------------------------------
// Spec enums — identify inbound/outbound API wire formats
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApiSpec {
    OpenAiChat,
    OpenAiResponses,
    AnthropicMessages,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendSpec {
    OpenAiChat,
    Ollama,
}

// ---------------------------------------------------------------------------
// BackendInfo — core's abstract view of a backend (converted from config)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct BackendInfo {
    pub id: BackendId,
    pub spec: BackendSpec,
    pub models: Vec<ModelId>,
    pub max_concurrent: u32,
    pub base_url: String,
}

// ---------------------------------------------------------------------------
// InboundAdapter — converts between client wire format and canonical types
// ---------------------------------------------------------------------------

pub trait InboundAdapter: Send + Sync {
    fn api_spec(&self) -> ApiSpec;

    fn parse_request(&self, body: &[u8]) -> Result<CanonicalRequest, AdapterError>;

    fn format_response(&self, response: &CanonicalResponse) -> Result<Vec<u8>, AdapterError>;

    fn format_stream_chunk(
        &self,
        chunk: &CanonicalStreamChunk,
    ) -> Result<Option<String>, AdapterError>;

    fn done_sentinel(&self) -> &str;
}

// ---------------------------------------------------------------------------
// OutboundAdapter — converts between canonical types and backend wire format
// ---------------------------------------------------------------------------

pub trait OutboundAdapter: Send + Sync {
    fn backend_spec(&self) -> BackendSpec;

    fn build_request_body(&self, req: &CanonicalRequest) -> Result<Vec<u8>, AdapterError>;

    fn parse_response(&self, body: &[u8]) -> Result<CanonicalResponse, AdapterError>;

    fn parse_stream_line(&self, line: &str) -> Result<Option<CanonicalStreamChunk>, AdapterError>;

    fn extra_headers(&self, backend: &BackendInfo) -> Vec<(String, String)>;

    fn inference_path(&self) -> &str;
}

// ---------------------------------------------------------------------------
// HealthProbe — checks backend liveness (object-safe async via Pin<Box>)
// ---------------------------------------------------------------------------

pub trait HealthProbe: Send + Sync {
    fn probe<'a>(
        &'a self,
        backend: &'a BackendInfo,
    ) -> Pin<Box<dyn Future<Output = Result<LatencyMs, HealthError>> + Send + 'a>>;
}

// ---------------------------------------------------------------------------
// Clock — injectable time source for deterministic testing
// ---------------------------------------------------------------------------

pub trait Clock: Send + Sync {
    fn now(&self) -> std::time::Instant;

    fn elapsed_ms(&self, since: std::time::Instant) -> u64;
}
