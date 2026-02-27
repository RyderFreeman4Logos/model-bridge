use super::*;
use mb_core::core::{
    ClientId, GenerationParams, RequestId, RequestMetadata, ToolChoice, ToolDefinition,
};
use serde_json::Value;

fn make_request(
    messages: Vec<Message>,
    params: GenerationParams,
    stream: bool,
) -> CanonicalRequest {
    CanonicalRequest {
        model: ModelId::new("gpt-4"),
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
    let adapter = OpenAiChatOutboundAdapter;
    let req = make_request(
        vec![
            simple_message(Role::System, "You are helpful."),
            simple_message(Role::User, "Hello!"),
        ],
        GenerationParams::default(),
        false,
    );

    let body = adapter.build_request_body(&req).unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["model"], "gpt-4");
    assert_eq!(json["stream"], false);
    assert_eq!(json["messages"].as_array().unwrap().len(), 2);
    assert_eq!(json["messages"][0]["role"], "system");
    assert_eq!(json["messages"][0]["content"], "You are helpful.");
    assert_eq!(json["messages"][1]["role"], "user");
    assert_eq!(json["messages"][1]["content"], "Hello!");
    // Optional params should be absent
    assert!(json.get("temperature").is_none());
    assert!(json.get("max_tokens").is_none());
}

#[test]
fn test_build_request_body_with_params() {
    let adapter = OpenAiChatOutboundAdapter;
    let req = make_request(
        vec![simple_message(Role::User, "Hi")],
        GenerationParams {
            temperature: Some(0.7),
            top_p: Some(0.9),
            max_tokens: Some(256),
            stop: Some(vec!["END".to_owned()]),
            frequency_penalty: Some(0.5),
            presence_penalty: Some(0.3),
            seed: Some(42),
        },
        true,
    );

    let body = adapter.build_request_body(&req).unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["stream"], true);
    assert_eq!(json["temperature"], 0.7);
    assert_eq!(json["top_p"], 0.9);
    assert_eq!(json["max_tokens"], 256);
    assert_eq!(json["stop"], serde_json::json!(["END"]));
    assert_eq!(json["frequency_penalty"], 0.5);
    assert_eq!(json["presence_penalty"], 0.3);
    assert_eq!(json["seed"], 42);
}

#[test]
fn test_build_request_body_with_tools() {
    let adapter = OpenAiChatOutboundAdapter;
    let mut req = make_request(
        vec![simple_message(Role::User, "Weather?")],
        GenerationParams::default(),
        false,
    );
    req.tools = Some(vec![ToolDefinition {
        name: "get_weather".to_owned(),
        description: Some("Get weather".to_owned()),
        parameters: serde_json::json!({"type": "object"}),
    }]);
    req.tool_choice = Some(ToolChoice::Auto);

    let body = adapter.build_request_body(&req).unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["tools"][0]["type"], "function");
    assert_eq!(json["tools"][0]["function"]["name"], "get_weather");
    assert_eq!(json["tool_choice"], "auto");
}

#[test]
fn test_build_request_body_named_tool_choice() {
    let adapter = OpenAiChatOutboundAdapter;
    let mut req = make_request(
        vec![simple_message(Role::User, "Call it")],
        GenerationParams::default(),
        false,
    );
    req.tool_choice = Some(ToolChoice::Named("my_fn".to_owned()));

    let body = adapter.build_request_body(&req).unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["tool_choice"]["type"], "function");
    assert_eq!(json["tool_choice"]["function"]["name"], "my_fn");
}

// ---------------------------------------------------------------------------
// parse_response
// ---------------------------------------------------------------------------

#[test]
fn test_parse_response_simple() {
    let adapter = OpenAiChatOutboundAdapter;
    let resp_json = serde_json::json!({
        "id": "chatcmpl-abc",
        "object": "chat.completion",
        "created": 1700000000_u64,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hi there!"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 4,
            "total_tokens": 16
        }
    });

    let resp = adapter
        .parse_response(&serde_json::to_vec(&resp_json).unwrap())
        .unwrap();

    assert_eq!(resp.id, "chatcmpl-abc");
    assert_eq!(resp.model.as_str(), "gpt-4");
    assert_eq!(resp.created, 1700000000);
    assert_eq!(resp.choices.len(), 1);
    assert_eq!(resp.choices[0].index, 0);
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
fn test_parse_response_invalid_json() {
    let adapter = OpenAiChatOutboundAdapter;
    let result = adapter.parse_response(b"not json");
    assert!(matches!(result, Err(AdapterError::ParseRequest(_))));
}

#[test]
fn test_parse_response_null_content() {
    let adapter = OpenAiChatOutboundAdapter;
    let resp_json = serde_json::json!({
        "id": "chatcmpl-abc",
        "object": "chat.completion",
        "created": 1700000000_u64,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null
            },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5 }
    });

    let resp = adapter
        .parse_response(&serde_json::to_vec(&resp_json).unwrap())
        .unwrap();

    assert_eq!(
        resp.choices[0].message.content,
        MessageContent::Text(String::new())
    );
}

// ---------------------------------------------------------------------------
// parse_stream_line
// ---------------------------------------------------------------------------

#[test]
fn test_parse_stream_line_text_delta() {
    let adapter = OpenAiChatOutboundAdapter;
    let line = r#"data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#;

    let chunk = adapter.parse_stream_line(line).unwrap().unwrap();

    assert_eq!(chunk.choices.len(), 1);
    assert_eq!(chunk.choices[0].index, 0);
    assert_eq!(
        chunk.choices[0].delta,
        DeltaContent::Text("Hello".to_owned())
    );
}

#[test]
fn test_parse_stream_line_role_delta() {
    let adapter = OpenAiChatOutboundAdapter;
    let line = r#"data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#;

    let chunk = adapter.parse_stream_line(line).unwrap().unwrap();

    assert_eq!(chunk.choices[0].delta, DeltaContent::Role(Role::Assistant));
}

#[test]
fn test_parse_stream_line_finish() {
    let adapter = OpenAiChatOutboundAdapter;
    let line = r#"data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#;

    let chunk = adapter.parse_stream_line(line).unwrap().unwrap();

    assert_eq!(
        chunk.choices[0].delta,
        DeltaContent::Finish(FinishReason::Stop)
    );
}

#[test]
fn test_parse_stream_line_done() {
    let adapter = OpenAiChatOutboundAdapter;
    let result = adapter.parse_stream_line("data: [DONE]").unwrap();
    assert!(result.is_none());
}

#[test]
fn test_parse_stream_line_empty() {
    let adapter = OpenAiChatOutboundAdapter;
    let result = adapter.parse_stream_line("").unwrap();
    assert!(result.is_none());
}

#[test]
fn test_parse_stream_line_whitespace() {
    let adapter = OpenAiChatOutboundAdapter;
    let result = adapter.parse_stream_line("   ").unwrap();
    assert!(result.is_none());
}

// ---------------------------------------------------------------------------
// extra_headers / inference_path / backend_spec
// ---------------------------------------------------------------------------

#[test]
fn test_backend_spec() {
    let adapter = OpenAiChatOutboundAdapter;
    assert_eq!(adapter.backend_spec(), BackendSpec::OpenAiChat);
}

#[test]
fn test_inference_path() {
    let adapter = OpenAiChatOutboundAdapter;
    assert_eq!(adapter.inference_path(), "/v1/chat/completions");
}

#[test]
fn test_extra_headers_includes_content_type() {
    let adapter = OpenAiChatOutboundAdapter;
    let backend = BackendInfo {
        id: mb_core::core::BackendId::new("test"),
        spec: BackendSpec::OpenAiChat,
        models: vec![ModelId::new("gpt-4")],
        max_concurrent: 10,
        base_url: "http://localhost:8000".to_owned(),
    };

    let headers = adapter.extra_headers(&backend);
    assert_eq!(headers.len(), 1);
    assert_eq!(headers[0].0, "Content-Type");
    assert_eq!(headers[0].1, "application/json");
}
