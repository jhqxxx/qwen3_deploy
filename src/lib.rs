use crate::qwen3::Qwen3;
use openai_dive::v1::resources::chat::{
    ChatCompletionChunkChoice, ChatCompletionChunkResponse, ChatMessage, ChatMessageContent,
    DeltaChatMessage, DeltaFunction, DeltaToolCall,
};
use rocket::async_stream::stream;
use rocket::futures::{Stream, StreamExt};
use std::pin::pin;
use std::sync::{Arc, Mutex, OnceLock};
use tokio::sync::RwLock;
use uuid::uuid;

mod qwen3;
mod utils;

const MODEL_NAME: &str = "qwen3-0.6b";

static MODEL: OnceLock<Arc<RwLock<Qwen3>>> = OnceLock::new();
pub fn init(path: &str) -> anyhow::Result<()> {
    let model_path = path.to_string();
    let model = Qwen3::new(model_path, false)?;
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
        model: MODEL_NAME.to_string(),
        system_fingerprint: None,
        object: "chat.completion.chunk".to_string(),
        usage: None,
    };

    Ok(stream! {
        match model_ref.write().await.generate_stream(message.to_string()).await {
            Ok(inner_stream) => {
                let mut pinned_stream = Box::pin(inner_stream);
                let tool_calling = false;
                let mut tool_call_id = None;
                let mut tool_call_content = String::new();
                while let Some(token) = pinned_stream.next().await {
                    let choice = if token.as_str() == "<tool_call>"{
                        tool_call_id = Some(uuid::Uuid::new_v4().to_string());
                        continue;
                    }else{
                        if token.as_str() == "</tool_call>"{
                            let choice = build_choice(token,tool_call_id.clone(),Some(tool_call_content.clone()));
                            tool_call_id = None;
                            choice
                        }else{
                            if tool_call_id.is_some(){
                                tool_call_content.push_str(&token);
                                continue;
                            }else{
                                build_choice(token,None,None)
                            }
                        }
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

fn build_choice(
    token: String,
    tool_call_id: Option<String>,
    tool_call_content: Option<String>,
) -> ChatCompletionChunkChoice {
    if tool_call_id.is_some() {
        let tool_call_id = tool_call_id.unwrap();
        let function = if let Some(content) = tool_call_content {
            match serde_json::from_str::<serde_json::Value>(&content) {
                Ok(json_value) => {
                    let name = json_value
                        .get("name")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());

                    let arguments = json_value.get("arguments").map(|v| v.to_string());

                    DeltaFunction { name, arguments }
                }
                Err(_) => DeltaFunction {
                    name: None,
                    arguments: Some(content),
                },
            }
        } else {
            DeltaFunction {
                name: None,
                arguments: None,
            }
        };
        return ChatCompletionChunkChoice {
            index: Some(0),
            delta: DeltaChatMessage::Assistant {
                content: None,
                reasoning_content: None,
                refusal: None,
                name: None,
                tool_calls: Some(vec![DeltaToolCall {
                    index: Some(0),
                    id: Some(tool_call_id),
                    r#type: Some("function".to_string()),
                    function,
                }]),
            },
            finish_reason: None,
            logprobs: None,
        };
    }
    ChatCompletionChunkChoice {
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
    }
}

mod tests {
    use crate::chat_stream;
    use crate::init;
    use rocket::futures::{Stream, StreamExt};
    use std::pin::pin;

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
