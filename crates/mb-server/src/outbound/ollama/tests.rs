use super::*;
use mb_core::core::{ClientId, GenerationParams, RequestId, RequestMetadata};
use serde_json::Value;

fn make_request(
    messages: Vec<Message>,
    params: GenerationParams,
    stream: bool,
) -> CanonicalRequest {
    CanonicalRequest {
        model: ModelId::new("llama3-70b"),
        messages,
        params,
        tools: None,
        tool_choice: None,
        stream,
        metadata: RequestMetadata {
            request_id: RequestId::new("req-test"),
            client_id: ClientId::new("client-test"),
            estimated_input_tokens: 10,
            prefix_hash: None,
        },
    }
}

fn simple_message(role: Role, text: &str) -> Message {
    Message {
        role,
        content: MessageContent::Text(text.to_owned()),
        name: None,
        tool_call_id: None,
    }
}

// ---------------------------------------------------------------------------
// build_request_body
// ---------------------------------------------------------------------------

#[test]
fn test_build_request_body_simple() {
    let adapter = OllamaOutboundAdapter;
    let req = make_request(
        vec![simple_message(Role::User, "Hello!")],
        GenerationParams::default(),
        false,
    );

    let body = adapter.build_request_body(&req).unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["model"], "llama3-70b");
    assert_eq!(json["stream"], false);
    assert_eq!(json["messages"].as_array().unwrap().len(), 1);
    assert_eq!(json["messages"][0]["role"], "user");
    assert_eq!(json["messages"][0]["content"], "Hello!");
    assert!(json.get("options").is_none());
    assert!(json.get("num_predict").is_none());
}

#[test]
fn test_build_request_body_with_options() {
    let adapter = OllamaOutboundAdapter;
    let req = make_request(
        vec![simple_message(Role::User, "Hi")],
        GenerationParams {
            temperature: Some(0.7),
            top_p: Some(0.9),
            max_tokens: Some(256),
            stop: Some(vec!["END".to_owned()]),
            seed: Some(42),
            ..Default::default()
        },
        true,
    );

    let body = adapter.build_request_body(&req).unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["stream"], true);
    assert_eq!(json["options"]["temperature"], 0.7);
    assert_eq!(json["options"]["top_p"], 0.9);
    assert_eq!(json["options"]["seed"], 42);
    assert_eq!(json["options"]["stop"], serde_json::json!(["END"]));
    assert_eq!(json["num_predict"], 256);
}

// ---------------------------------------------------------------------------
// parse_response
// ---------------------------------------------------------------------------

#[test]
fn test_parse_response_simple() {
    let adapter = OllamaOutboundAdapter;
    let resp_json = serde_json::json!({
        "model": "llama3-70b",
        "message": {
            "role": "assistant",
            "content": "Hi there!"
        },
        "done": true,
        "prompt_eval_count": 12,
        "eval_count": 4
    });

    let resp = adapter
        .parse_response(&serde_json::to_vec(&resp_json).unwrap())
        .unwrap();

    assert_eq!(resp.model.as_str(), "llama3-70b");
    assert_eq!(resp.choices.len(), 1);
    assert_eq!(resp.choices[0].message.role, Role::Assistant);
    assert_eq!(
        resp.choices[0].message.content,
        MessageContent::Text("Hi there!".to_owned())
    );
    assert_eq!(resp.choices[0].finish_reason, FinishReason::Stop);
    assert_eq!(resp.usage.prompt_tokens, 12);
    assert_eq!(resp.usage.completion_tokens, 4);
    assert_eq!(resp.usage.total_tokens, 16);
}

#[test]
fn test_parse_response_null_content() {
    let adapter = OllamaOutboundAdapter;
    let resp_json = serde_json::json!({
        "model": "llama3-70b",
        "message": {
            "role": "assistant",
            "content": null
        },
        "done": true
    });

    let resp = adapter
        .parse_response(&serde_json::to_vec(&resp_json).unwrap())
        .unwrap();

    assert_eq!(
        resp.choices[0].message.content,
        MessageContent::Text(String::new())
    );
}

#[test]
fn test_parse_response_invalid_json() {
    let adapter = OllamaOutboundAdapter;
    let result = adapter.parse_response(b"not json");
    assert!(matches!(result, Err(AdapterError::ParseRequest(_))));
}

// ---------------------------------------------------------------------------
// parse_stream_line
// ---------------------------------------------------------------------------

#[test]
fn test_parse_stream_line_text() {
    let adapter = OllamaOutboundAdapter;
    let line =
        r#"{"model":"llama3-70b","message":{"role":"assistant","content":"Hello"},"done":false}"#;

    let chunk = adapter.parse_stream_line(line).unwrap().unwrap();
    assert_eq!(chunk.choices.len(), 1);
    assert_eq!(
        chunk.choices[0].delta,
        DeltaContent::Text("Hello".to_owned())
    );
}

#[test]
fn test_parse_stream_line_done() {
    let adapter = OllamaOutboundAdapter;
    let line = r#"{"model":"llama3-70b","message":{"role":"assistant","content":""},"done":true}"#;

    let chunk = adapter.parse_stream_line(line).unwrap().unwrap();
    assert_eq!(
        chunk.choices[0].delta,
        DeltaContent::Finish(FinishReason::Stop)
    );
}

#[test]
fn test_parse_stream_line_empty() {
    let adapter = OllamaOutboundAdapter;
    let result = adapter.parse_stream_line("").unwrap();
    assert!(result.is_none());
}

// ---------------------------------------------------------------------------
// extra_headers / inference_path / backend_spec
// ---------------------------------------------------------------------------

#[test]
fn test_backend_spec() {
    let adapter = OllamaOutboundAdapter;
    assert_eq!(adapter.backend_spec(), BackendSpec::Ollama);
}

#[test]
fn test_inference_path() {
    let adapter = OllamaOutboundAdapter;
    assert_eq!(adapter.inference_path(), "/api/chat");
}

#[test]
fn test_extra_headers_empty() {
    let adapter = OllamaOutboundAdapter;
    let backend = BackendInfo {
        id: mb_core::core::BackendId::new("test"),
        spec: BackendSpec::Ollama,
        models: vec![ModelId::new("llama3-70b")],
        max_concurrent: 4,
        base_url: "http://localhost:11434".to_owned(),
    };

    let headers = adapter.extra_headers(&backend);
    assert!(headers.is_empty());
}
