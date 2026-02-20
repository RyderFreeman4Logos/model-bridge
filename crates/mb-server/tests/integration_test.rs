mod common;

use common::*;
use mb_server::config::RoutingStrategyConfig;

// ---------------------------------------------------------------------------
// Basic proxy tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_proxy_non_streaming_basic() {
    let mock = MockBackendServer::start(&sample_openai_response()).await;
    let gw = TestGateway::start_simple(&mock.url()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("Authorization", format!("Bearer {TEST_API_KEY}"))
        .header("Content-Type", "application/json")
        .body(sample_request_body())
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.expect("valid JSON");
    assert!(body.get("choices").is_some());
    assert!(body["choices"][0]["message"]["content"].is_string());
}

// ---------------------------------------------------------------------------
// Authentication tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_auth_invalid_key() {
    let mock = MockBackendServer::start(&sample_openai_response()).await;
    let gw = TestGateway::start_simple(&mock.url()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("Authorization", "Bearer invalid-key-not-registered")
        .header("Content-Type", "application/json")
        .body(sample_request_body())
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 401);

    let body: serde_json::Value = resp.json().await.expect("valid JSON");
    assert_eq!(body["error"]["type"], "authentication_error");
}

#[tokio::test]
async fn test_auth_model_not_permitted() {
    let mock = MockBackendServer::start(&sample_openai_response()).await;
    let gw = TestGateway::start_simple(&mock.url()).await;

    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "model": "forbidden-model",
        "messages": [{"role": "user", "content": "Hello"}]
    })
    .to_string();

    let resp = client
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("Authorization", format!("Bearer {TEST_API_KEY}"))
        .header("Content-Type", "application/json")
        .body(body)
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 403);

    let body: serde_json::Value = resp.json().await.expect("valid JSON");
    assert_eq!(body["error"]["type"], "permission_error");
}

// ---------------------------------------------------------------------------
// Streaming tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_proxy_streaming_basic() {
    let chunks = sample_sse_chunks();
    let chunk_refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();
    let mock = MockBackendServer::start_sse(&chunk_refs).await;
    let gw = TestGateway::start(
        &[(mock.url(), vec![TEST_MODEL.to_owned()])],
        &[(TEST_CLIENT_ID, TEST_API_KEY, vec![TEST_MODEL.to_owned()])],
        TestGatewayOptions {
            enable_stream_dispatch: true,
            ..TestGatewayOptions::default()
        },
    )
    .await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("Authorization", format!("Bearer {TEST_API_KEY}"))
        .header("Content-Type", "application/json")
        .body(sample_stream_request_body())
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    let body_text = resp.text().await.expect("read body");
    // SSE response should contain data lines and a [DONE] sentinel
    assert!(body_text.contains("data:"), "should contain SSE data lines");
    assert!(
        body_text.contains("[DONE]"),
        "should contain [DONE] sentinel"
    );
}

#[tokio::test]
async fn test_streaming_multiple_chunks() {
    let chunks = sample_sse_chunks();
    let chunk_refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();
    let mock = MockBackendServer::start_sse(&chunk_refs).await;
    let gw = TestGateway::start(
        &[(mock.url(), vec![TEST_MODEL.to_owned()])],
        &[(TEST_CLIENT_ID, TEST_API_KEY, vec![TEST_MODEL.to_owned()])],
        TestGatewayOptions {
            enable_stream_dispatch: true,
            ..TestGatewayOptions::default()
        },
    )
    .await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("Authorization", format!("Bearer {TEST_API_KEY}"))
        .header("Content-Type", "application/json")
        .body(sample_stream_request_body())
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    let body_text = resp.text().await.expect("read body");

    // Count data lines (each SSE event produces a "data:" line)
    let data_lines: Vec<&str> = body_text
        .lines()
        .filter(|l| l.starts_with("data:"))
        .collect();

    // Should have at least the text chunks plus [DONE]
    // The exact number depends on which chunks the adapters process
    assert!(
        data_lines.len() >= 2,
        "should have multiple data lines, got {}",
        data_lines.len()
    );
}

// ---------------------------------------------------------------------------
// Routing tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_round_robin() {
    let mock_a = MockBackendServer::start(&sample_openai_response_with_id("resp-A")).await;
    let mock_b = MockBackendServer::start(&sample_openai_response_with_id("resp-B")).await;

    let gw = TestGateway::start(
        &[
            (mock_a.url(), vec![TEST_MODEL.to_owned()]),
            (mock_b.url(), vec![TEST_MODEL.to_owned()]),
        ],
        &[(TEST_CLIENT_ID, TEST_API_KEY, vec![TEST_MODEL.to_owned()])],
        TestGatewayOptions {
            routing_strategy: RoutingStrategyConfig::RoundRobin,
            cache_aware: false,
            ..TestGatewayOptions::default()
        },
    )
    .await;

    let client = reqwest::Client::new();
    let mut seen_ids = std::collections::HashSet::new();

    for _ in 0..4 {
        let resp = client
            .post(format!("{}/v1/chat/completions", gw.url()))
            .header("Authorization", format!("Bearer {TEST_API_KEY}"))
            .header("Content-Type", "application/json")
            .body(sample_request_body())
            .send()
            .await
            .expect("request should succeed");

        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.expect("valid JSON");
        if let Some(id) = body.get("id").and_then(|v| v.as_str()) {
            seen_ids.insert(id.to_owned());
        }
    }

    // With round-robin across 2 backends, we should see both response IDs
    assert_eq!(
        seen_ids.len(),
        2,
        "round-robin should distribute across both backends, got: {seen_ids:?}"
    );
}

// ---------------------------------------------------------------------------
// Rate limiting tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_rate_limit_exceeded() {
    let mock = MockBackendServer::start(&sample_openai_response()).await;
    let gw = TestGateway::start(
        &[(mock.url(), vec![TEST_MODEL.to_owned()])],
        &[(TEST_CLIENT_ID, TEST_API_KEY, vec![TEST_MODEL.to_owned()])],
        TestGatewayOptions {
            rate_limit_rpm: 2,
            ..TestGatewayOptions::default()
        },
    )
    .await;

    let client = reqwest::Client::new();

    // First two requests should succeed (RPM = 2)
    for _ in 0..2 {
        let resp = client
            .post(format!("{}/v1/chat/completions", gw.url()))
            .header("Authorization", format!("Bearer {TEST_API_KEY}"))
            .header("Content-Type", "application/json")
            .body(sample_request_body())
            .send()
            .await
            .expect("request should succeed");
        assert_eq!(resp.status(), 200);
    }

    // Third request should be rate limited
    let resp = client
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("Authorization", format!("Bearer {TEST_API_KEY}"))
        .header("Content-Type", "application/json")
        .body(sample_request_body())
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 429);

    let body: serde_json::Value = resp.json().await.expect("valid JSON");
    assert_eq!(body["error"]["type"], "rate_limit_error");
}

// ---------------------------------------------------------------------------
// Error handling tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_backend_error_502() {
    let mock =
        MockBackendServer::start_with_options(r#"{"error": "internal server error"}"#, 500, 0)
            .await;
    let gw = TestGateway::start_simple(&mock.url()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("Authorization", format!("Bearer {TEST_API_KEY}"))
        .header("Content-Type", "application/json")
        .body(sample_request_body())
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 502);

    let body: serde_json::Value = resp.json().await.expect("valid JSON");
    assert_eq!(body["error"]["type"], "backend_error");
}

#[tokio::test]
async fn test_malformed_request_400() {
    let mock = MockBackendServer::start(&sample_openai_response()).await;
    let gw = TestGateway::start_simple(&mock.url()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("Authorization", format!("Bearer {TEST_API_KEY}"))
        .header("Content-Type", "application/json")
        .body("this is not valid json")
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 400);

    let body: serde_json::Value = resp.json().await.expect("valid JSON");
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn test_no_healthy_backend_503() {
    // Start a mock but don't mark backends as healthy
    let mock = MockBackendServer::start(&sample_openai_response()).await;
    let gw = TestGateway::start(
        &[(mock.url(), vec![TEST_MODEL.to_owned()])],
        &[(TEST_CLIENT_ID, TEST_API_KEY, vec![TEST_MODEL.to_owned()])],
        TestGatewayOptions {
            mark_healthy: false,
            ..TestGatewayOptions::default()
        },
    )
    .await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("Authorization", format!("Bearer {TEST_API_KEY}"))
        .header("Content-Type", "application/json")
        .body(sample_request_body())
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 503);

    let body: serde_json::Value = resp.json().await.expect("valid JSON");
    assert_eq!(body["error"]["type"], "service_unavailable");
}
