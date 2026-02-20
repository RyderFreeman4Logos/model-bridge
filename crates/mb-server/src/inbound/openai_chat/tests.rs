use super::*;
use mb_core::core::{
    AdapterError, Choice, FinishReason, Message, MessageContent, ModelId, Role, StreamChoice,
    TokenUsage, ToolChoice,
};
use serde_json::Value;

#[test]
fn test_parse_simple_request() {
    let body = serde_json::json!({
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    });

    let adapter = OpenAiChatInboundAdapter;
    let req = adapter
        .parse_request(serde_json::to_vec(&body).unwrap().as_slice())
        .unwrap();

    assert_eq!(req.model.as_str(), "gpt-4");
    assert_eq!(req.messages.len(), 2);
    assert_eq!(req.messages[0].role, Role::System);
    assert_eq!(
        req.messages[0].content,
        MessageContent::Text("You are helpful.".to_owned())
    );
    assert_eq!(req.messages[1].role, Role::User);
    assert_eq!(
        req.messages[1].content,
        MessageContent::Text("Hello!".to_owned())
    );
    assert_eq!(req.params.temperature, Some(0.7));
    assert_eq!(req.params.max_tokens, Some(100));
    assert!(!req.stream);
    assert!(req.tools.is_none());
    assert!(req.tool_choice.is_none());
}

#[test]
fn test_parse_request_with_tools() {
    let body = serde_json::json!({
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "What is the weather?"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }
            }
        ],
        "tool_choice": "auto"
    });

    let adapter = OpenAiChatInboundAdapter;
    let req = adapter
        .parse_request(serde_json::to_vec(&body).unwrap().as_slice())
        .unwrap();

    let tools = req.tools.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "get_weather");
    assert_eq!(tools[0].description.as_deref(), Some("Get current weather"));
    assert_eq!(req.tool_choice, Some(ToolChoice::Auto));
}

#[test]
fn test_format_response() {
    let adapter = OpenAiChatInboundAdapter;
    let response = CanonicalResponse {
        id: "chatcmpl-123".to_owned(),
        model: ModelId::new("gpt-4"),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: Role::Assistant,
                content: MessageContent::Text("Hello there!".to_owned()),
                name: None,
                tool_call_id: None,
            },
            finish_reason: FinishReason::Stop,
        }],
        usage: TokenUsage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
        },
        created: 1700000000,
    };

    let bytes = adapter.format_response(&response).unwrap();
    let json: Value = serde_json::from_slice(&bytes).unwrap();

    assert_eq!(json["id"], "chatcmpl-123");
    assert_eq!(json["object"], "chat.completion");
    assert_eq!(json["model"], "gpt-4");
    assert_eq!(json["created"], 1700000000);
    assert_eq!(json["choices"][0]["index"], 0);
    assert_eq!(json["choices"][0]["message"]["role"], "assistant");
    assert_eq!(json["choices"][0]["message"]["content"], "Hello there!");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert_eq!(json["usage"]["prompt_tokens"], 10);
    assert_eq!(json["usage"]["completion_tokens"], 5);
    assert_eq!(json["usage"]["total_tokens"], 15);
}

#[test]
fn test_format_stream_chunk_text() {
    let adapter = OpenAiChatInboundAdapter;
    let chunk = CanonicalStreamChunk {
        choices: vec![StreamChoice {
            index: 0,
            delta: DeltaContent::Text("Hello".to_owned()),
        }],
    };

    let result = adapter.format_stream_chunk(&chunk).unwrap().unwrap();
    assert!(result.starts_with("data: "));
    assert!(result.ends_with("\n\n"));

    let json_str = result
        .strip_prefix("data: ")
        .unwrap()
        .strip_suffix("\n\n")
        .unwrap();
    let json: Value = serde_json::from_str(json_str).unwrap();

    assert_eq!(json["object"], "chat.completion.chunk");
    assert_eq!(json["choices"][0]["index"], 0);
    assert_eq!(json["choices"][0]["delta"]["content"], "Hello");
    assert!(json["choices"][0]["finish_reason"].is_null());
}

#[test]
fn test_format_stream_chunk_finish() {
    let adapter = OpenAiChatInboundAdapter;
    let chunk = CanonicalStreamChunk {
        choices: vec![StreamChoice {
            index: 0,
            delta: DeltaContent::Finish(FinishReason::Stop),
        }],
    };

    let result = adapter.format_stream_chunk(&chunk).unwrap().unwrap();
    let json_str = result
        .strip_prefix("data: ")
        .unwrap()
        .strip_suffix("\n\n")
        .unwrap();
    let json: Value = serde_json::from_str(json_str).unwrap();

    assert_eq!(json["choices"][0]["finish_reason"], "stop");
}

#[test]
fn test_parse_request_null_content() {
    let body = serde_json::json!({
        "model": "gpt-4",
        "messages": [
            {"role": "tool", "content": null, "tool_call_id": "call_123"}
        ]
    });

    let adapter = OpenAiChatInboundAdapter;
    let req = adapter
        .parse_request(serde_json::to_vec(&body).unwrap().as_slice())
        .unwrap();

    assert_eq!(req.messages[0].role, Role::Tool);
    assert_eq!(req.messages[0].content, MessageContent::Text(String::new()));
    assert_eq!(req.messages[0].tool_call_id.as_deref(), Some("call_123"));
}

#[test]
fn test_parse_request_invalid_json() {
    let adapter = OpenAiChatInboundAdapter;
    let result = adapter.parse_request(b"not json");
    assert!(matches!(result, Err(AdapterError::ParseRequest(_))));
}

#[test]
fn test_parse_request_unknown_role() {
    let body = serde_json::json!({
        "model": "gpt-4",
        "messages": [
            {"role": "unknown_role", "content": "hi"}
        ]
    });

    let adapter = OpenAiChatInboundAdapter;
    let result = adapter.parse_request(serde_json::to_vec(&body).unwrap().as_slice());
    assert!(matches!(result, Err(AdapterError::ParseRequest(_))));
}

#[test]
fn test_done_sentinel() {
    let adapter = OpenAiChatInboundAdapter;
    assert_eq!(adapter.done_sentinel(), "data: [DONE]");
}

#[test]
fn test_api_spec() {
    let adapter = OpenAiChatInboundAdapter;
    assert_eq!(adapter.api_spec(), ApiSpec::OpenAiChat);
}

#[test]
fn test_format_stream_chunk_empty() {
    let adapter = OpenAiChatInboundAdapter;
    let chunk = CanonicalStreamChunk { choices: vec![] };
    let result = adapter.format_stream_chunk(&chunk).unwrap();
    assert!(result.is_none());
}
