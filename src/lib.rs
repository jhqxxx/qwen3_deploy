use crate::qwen3::Qwen3;
use openai_dive::v1::resources::chat::{
    ChatCompletionChunkChoice, ChatCompletionChunkResponse, ChatMessage, ChatMessageContent,
    DeltaChatMessage,
};
use rocket::async_stream::stream;
use rocket::futures::{Stream, StreamExt};
use std::pin::pin;
use std::sync::{Arc, Mutex, OnceLock};
use tokio::sync::RwLock;
use uuid::uuid;

mod qwen3;
mod utils;

static MODEL: OnceLock<Arc<RwLock<Qwen3>>> = OnceLock::new();
pub fn init(path: &str) -> anyhow::Result<()> {
    let model_path = path.to_string();
    let model = Qwen3::new(model_path)?;
    MODEL.get_or_init(|| Arc::new(RwLock::new(model)));
    Ok(())
}

pub fn chat_stream(message: &str) -> anyhow::Result<impl Stream<Item = String>> {
    let model_ref = MODEL
        .get()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("model not init"))?;
    let id = uuid::Uuid::new_v4().to_string();
    let response = ChatCompletionChunkResponse {
        id: Some(id),
        choices: vec![],
        created: chrono::Utc::now().timestamp() as u32,
        model: "qwen3-0.6b".to_string(),
        system_fingerprint: None,
        object: "chat.completion.chunk".to_string(),
        usage: None,
    };
    Ok(stream! {
        match model_ref.write().await.generate_stream(message.to_string()) {
            Ok(inner_stream) => {
                let mut pinned_stream = pin!(inner_stream);
                while let Some(token) = pinned_stream.next().await {
                    let choice = ChatCompletionChunkChoice {
                        index: Some(0),
                        delta: DeltaChatMessage::Assistant {
                            content: Some(ChatMessageContent::Text(token)),
                            reasoning_content: None,
                            refusal: None,
                            name: None,
                            tool_calls: None,
                        },
                        finish_reason: None,
                        logprobs: None,
                    };
                    let mut resp = response.clone();
                    resp.choices.push(choice);
                    yield serde_json::to_string(&resp).unwrap();
                }
            }
            Err(e) => {
                yield format!("Error: {}", e);
            }
        }
    })
}

mod tests {
    use crate::init;
    use std::pin::pin;
    use crate::chat_stream;
    use rocket::futures::{Stream, StreamExt};

    #[tokio::test]
    async fn test_chat() {
        let message = r#"
    {
        "messages": [
            {
                "role": "user",
                "content": "你是谁"
            }
        ],
        "model": "deepseek-chat",
        "response_format": {
            "type": "text"
        },
        "stream": true,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "image-qa-onlineA-_-Achat_to_image",
                    "description": "chat anything with image",
                    "parameters": {
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "properties": {
                            "image_paths": {
                                "items": {
                                    "type": "string"
                                },
                                "type": "array"
                            },
                            "prompt": {
                                "type": "string"
                            }
                        },
                        "required": [
                            "prompt",
                            "image_paths"
                        ],
                        "title": "Req",
                        "type": "object"
                    }
                }
            }
        ],
        "tool_choice": null
    }
    "#;
        init("resources/model").unwrap();
        let start = std::time::Instant::now();
        println!("开始");
        let mut stream = pin!(chat_stream(message).unwrap());
        while let Some(item) = stream.next().await {
            println!("{}", item);
        }

        println!("耗时：{}ms", start.elapsed().as_millis());
    }
}
