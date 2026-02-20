mod common;

use common::*;

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
