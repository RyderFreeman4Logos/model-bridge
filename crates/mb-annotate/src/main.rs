use std::io::{self, Write};

use clap::Parser;
use colored::Colorize;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Parser)]
#[command(name = "mb-annotate", about = "Interactive chat CLI for model-bridge")]
struct Args {
    #[arg(long, default_value = "http://localhost:8080")]
    api_base: String,
    #[arg(long, default_value = "")]
    api_key: String,
    #[arg(long, default_value = "false")]
    annotate: bool,
    #[arg(long)]
    model: String,
    #[arg(long)]
    system_prompt: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct StreamChunk {
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    delta: StreamDelta,
}

#[derive(Debug, Deserialize)]
struct StreamDelta {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ErrorEnvelope {
    error: ErrorDetail,
}

#[derive(Debug, Deserialize)]
struct ErrorDetail {
    message: String,
}

#[derive(Debug, Serialize)]
struct FeedbackRequest {
    turn_id: String,
    verdict: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_direction: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_response: Option<String>,
}

fn prompt_line(prompt: &str) -> io::Result<Option<String>> {
    print!("{prompt}");
    io::stdout().flush()?;

    let mut input = String::new();
    match io::stdin().read_line(&mut input) {
        Ok(0) => Ok(None),
        Ok(_) => Ok(Some(input.trim().to_owned())),
        Err(err) => Err(err),
    }
}

async fn maybe_annotate_turn(
    enabled: bool,
    client: &reqwest::Client,
    feedback_endpoint: &str,
    api_key: &str,
    conversation_id: &str,
    turn_id: &str,
) {
    if !enabled {
        return;
    }

    let verdict = loop {
        let input = match prompt_line(
            "Rate this response [s]atisfactory / [b]iased / [r]efused / [Enter to skip]: ",
        ) {
            Ok(Some(value)) => value,
            Ok(None) => {
                println!();
                return;
            }
            Err(err) => {
                eprintln!("{}", format!("Error: failed to read annotation: {err}").red());
                return;
            }
        };

        if input.is_empty() {
            return;
        }

        match input.to_ascii_lowercase().as_str() {
            "s" => break "satisfactory".to_owned(),
            "b" => break "biased".to_owned(),
            "r" => break "refused".to_owned(),
            _ => {
                eprintln!("{}", "Please enter s, b, r, or press Enter to skip.".yellow());
            }
        }
    };

    let mut expected_direction = None;
    let mut expected_response = None;
    if verdict == "biased" || verdict == "refused" {
        match prompt_line("Expected response direction (or Enter to skip): ") {
            Ok(Some(value)) if !value.is_empty() => expected_direction = Some(value),
            Ok(_) => {}
            Err(err) => {
                eprintln!(
                    "{}",
                    format!("Error: failed to read expected direction: {err}").red()
                );
            }
        }

        match prompt_line("Expected response (or Enter to skip): ") {
            Ok(Some(value)) if !value.is_empty() => expected_response = Some(value),
            Ok(_) => {}
            Err(err) => {
                eprintln!(
                    "{}",
                    format!("Error: failed to read expected response: {err}").red()
                );
            }
        }
    }

    let request = FeedbackRequest {
        turn_id: turn_id.to_owned(),
        verdict,
        expected_direction,
        expected_response,
    };

    let mut req = client
        .post(feedback_endpoint)
        .header("X-Conversation-Id", conversation_id)
        .header("X-Turn-Id", turn_id)
        .json(&request);
    if !api_key.is_empty() {
        req = req.bearer_auth(api_key);
    }

    match req.send().await {
        Ok(response) if response.status().is_success() => {
            println!("{}", "Annotation saved.".bright_green());
        }
        Ok(response) => {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            let message = serde_json::from_str::<ErrorEnvelope>(&body)
                .map(|v| v.error.message)
                .unwrap_or_else(|_| body);
            eprintln!("{}", format!("Error: failed to save annotation ({status}): {message}").red());
        }
        Err(err) => {
            eprintln!("{}", format!("Error: failed to save annotation: {err}").red());
        }
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let endpoint = format!(
        "{}/v1/chat/completions",
        args.api_base.trim_end_matches('/')
    );
    let feedback_endpoint = format!("{}/v1/feedback", args.api_base.trim_end_matches('/'));
    let client = reqwest::Client::new();
    let conversation_id = Uuid::new_v4().to_string();
    let mut history: Vec<ChatMessage> = Vec::new();

    if let Some(system_prompt) = args.system_prompt.as_deref().map(str::trim) {
        if !system_prompt.is_empty() {
            history.push(ChatMessage {
                role: "system".to_owned(),
                content: system_prompt.to_owned(),
            });
        }
    }

    loop {
        print!("{}", "You: ".bright_cyan());
        if let Err(err) = io::stdout().flush() {
            eprintln!("{}", format!("Failed to flush stdout: {err}").red());
            break;
        }

        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(0) => {
                println!();
                break;
            }
            Ok(_) => {}
            Err(err) => {
                eprintln!("{}", format!("Failed to read input: {err}").red());
                continue;
            }
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if matches!(input, "quit" | "exit") {
            break;
        }

        history.push(ChatMessage {
            role: "user".to_owned(),
            content: input.to_owned(),
        });

        let request = ChatCompletionRequest {
            model: args.model.clone(),
            messages: history.clone(),
            stream: true,
        };

        let client_turn_id = Uuid::new_v4().to_string();
        let mut req = client
            .post(&endpoint)
            .header("X-Conversation-Id", &conversation_id)
            .header("X-Turn-Id", &client_turn_id)
            .json(&request);
        if !args.api_key.is_empty() {
            req = req.bearer_auth(&args.api_key);
        }

        let response = match req.send().await {
            Ok(resp) => resp,
            Err(err) => {
                let _ = history.pop();
                eprintln!(
                    "{}",
                    format!("Connection error: {err}. Please check --api-base and server status.")
                        .red()
                );
                continue;
            }
        };

        if !response.status().is_success() {
            let _ = history.pop();
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            let message = serde_json::from_str::<ErrorEnvelope>(&body)
                .map(|v| v.error.message)
                .unwrap_or_else(|_| body);
            eprintln!("{}", format!("Request failed ({status}): {message}").red());
            continue;
        }

        let turn_id = response
            .headers()
            .get("X-Turn-Id")
            .and_then(|value| value.to_str().ok())
            .map(ToOwned::to_owned)
            .unwrap_or(client_turn_id);

        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut assistant_text = String::new();
        let mut printed_prefix = false;
        let mut done = false;

        while let Some(item) = stream.next().await {
            let bytes = match item {
                Ok(b) => b,
                Err(err) => {
                    eprintln!("{}", format!("\nStream error: {err}").red());
                    break;
                }
            };

            buffer.push_str(&String::from_utf8_lossy(&bytes));

            while let Some(pos) = buffer.find('\n') {
                let mut line = buffer[..pos].to_owned();
                buffer.drain(..=pos);

                if line.ends_with('\r') {
                    line.pop();
                }
                if !line.starts_with("data: ") {
                    continue;
                }

                let data = &line[6..];
                if data == "[DONE]" {
                    done = true;
                    break;
                }
                if data.is_empty() {
                    continue;
                }

                if let Ok(chunk) = serde_json::from_str::<StreamChunk>(data) {
                    for choice in chunk.choices {
                        if let Some(content) = choice.delta.content {
                            if !printed_prefix {
                                print!("{}", "Assistant: ".bright_green());
                                printed_prefix = true;
                            }
                            print!("{}", content.bright_green());
                            let _ = io::stdout().flush();
                            assistant_text.push_str(&content);
                        }
                    }
                }
            }

            if done {
                break;
            }
        }

        if printed_prefix {
            println!();
        } else {
            println!("{}", "Assistant: <empty response>".bright_green());
        }

        if !assistant_text.is_empty() {
            history.push(ChatMessage {
                role: "assistant".to_owned(),
                content: assistant_text,
            });
        }

        maybe_annotate_turn(
            args.annotate,
            &client,
            &feedback_endpoint,
            &args.api_key,
            &conversation_id,
            &turn_id,
        )
        .await;
    }
}
