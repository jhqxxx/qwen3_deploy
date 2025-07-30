use crate::qwen3::Qwen3;
use openai_dive::v1::resources::chat::{
    ChatCompletionChunkChoice, ChatCompletionChunkResponse, 
    ChatCompletionResponse, ChatCompletionChoice, ChatMessage, ChatMessageContent,
    DeltaChatMessage, DeltaFunction, DeltaToolCall, ToolCall, Function as ChatFunction
};
use openai_dive::v1::resources::shared::FinishReason;
use rocket::async_stream::stream;
use rocket::futures::{Stream, StreamExt};
use rocket::response::content;
use std::pin::pin;
use std::sync::{Arc, Mutex, OnceLock};
use tokio::sync::RwLock;
use uuid::uuid;
use serde_json::Value;

mod qwen3;
mod utils;

const MODEL_NAME: &str = "qwen3-0.6b";

static MODEL: OnceLock<Arc<RwLock<Qwen3>>> = OnceLock::new();

// 主请求结构体
#[derive(Debug, serde::Deserialize)]
pub struct ChatRequest {
    messages: Vec<Message>,
    tools: Option<Vec<Tool>>,
    pub stream: Option<bool>
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct Message {
    role: String,
    content: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

// 工具定义结构体
#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct Tool {
    #[serde(rename = "type")]
    tool_type: String,
    function: Function,
}

// 函数定义结构体
#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct Function {
    name: String,
    description: String,
    parameters: Value,
}

pub fn init(path: &str) -> anyhow::Result<()> {
    let model_path = path.to_string();
    let model = Qwen3::new(model_path, false)?;
    MODEL.get_or_init(|| Arc::new(RwLock::new(model)));
    Ok(())
}

pub fn chat_stream(message: &ChatRequest) -> anyhow::Result<impl Stream<Item = String>> {
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
        match model_ref.write().await.generate_stream(message) {
            Ok(inner_stream) => {
                let mut pinned_stream = Box::pin(inner_stream);
                let mut tool_call_id = None;
                let mut tool_call_content = String::new();
                while let Some(token) = pinned_stream.next().await {
                    let choice = if token.as_str() == "<tool_call>"{
                        tool_call_id = Some(uuid::Uuid::new_v4().to_string());
                        continue;
                    }else{
                        if token.as_str() == "</tool_call>"{
                            let choice = build_chunk_choice(token,tool_call_id.clone(),Some(tool_call_content.clone()));
                            tool_call_id = None;
                            choice
                        }else{
                            if tool_call_id.is_some(){
                                tool_call_content.push_str(&token);
                                continue;
                            }else{
                                build_chunk_choice(token,None,None)
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


fn build_chunk_choice(
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

pub async fn chat_sync(message: &ChatRequest) -> anyhow::Result<String> {
    let model_ref = MODEL
        .get()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("model not init"))?;

    let id = uuid::Uuid::new_v4().to_string();
    let mut response = ChatCompletionResponse {
        id: Some(id),
        choices: vec![],
        created: chrono::Utc::now().timestamp() as u32,
        model: MODEL_NAME.to_string(),
        service_tier: None,
        system_fingerprint: None,
        object: "chat.completion".to_string(),
        usage: None,
    };

    let generate_str = model_ref.write().await.generate(message)?;
    let choice: ChatCompletionChoice = build_choice(generate_str);
    response.choices.push(choice);
    let response_str = serde_json::to_string(&response).unwrap();
    Ok(response_str)
}
fn build_choice(token: String) -> ChatCompletionChoice {
    if token.contains("<tool_call>") {
        let mes: Vec<&str> = token.split("<tool_call>").collect();
        let content = mes[0].to_string();
        let tool_mes = mes[1].replace("</tool_call>", "");
        let function = match serde_json::from_str::<serde_json::Value>(&tool_mes) {
                Ok(json_value) => {
                    let name = json_value
                        .get("name")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string()).unwrap_or_default();

                    let arguments = json_value.get("arguments").map(|v| v.to_string()).unwrap_or_default();

                    ChatFunction { name, arguments }
                }
                Err(_) => ChatFunction {
                    name: "".to_string(),
                    arguments: "".to_string(),
                },
            };
        let tool_call = ToolCall {
            id: "0".to_string(),
            r#type: "function".to_string(),
            function: function
        };
        ChatCompletionChoice {
            index: 0,
            message: ChatMessage::Assistant {
                content: Some(ChatMessageContent::Text(content)),
                reasoning_content: None,
                refusal: None,
                name: None,
                audio: None,
                tool_calls: Some(vec![tool_call]),
            },
            finish_reason: Some(FinishReason::ToolCalls),
            logprobs: None,
        }
    } else {
        ChatCompletionChoice {
            index: 0,
            message: ChatMessage::Assistant {
                content: Some(ChatMessageContent::Text(token)),
                reasoning_content: None,
                refusal: None,
                name: None,
                audio: None,
                tool_calls: None,
            },
            finish_reason: Some(FinishReason::StopSequenceReached),
            logprobs: None,
        }
    }    
}

mod tests {
    use crate::chat_stream;
    use crate::chat_sync;
    use crate::init;
    use crate::ChatRequest;
    use rocket::futures::{Stream, StreamExt};
    use std::pin::pin;

    #[tokio::test]
    async fn test_chat() {
        let message = r#"
    {
        "messages": [
            {
                "role": "user",
                "content": "图片里有什么？图片地址：[\"data/chat/kb/17992851581189image_1753331929967.png\"], 必须调用工具"
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
        init("/mnt/c/jhq/huggingface_model/Qwen/Qwen3-0___6B").unwrap();
        let start = std::time::Instant::now();
        println!("开始");
        let request: ChatRequest = serde_json::from_str(&message).unwrap();
        // let mut stream = pin!(chat_stream(&request).unwrap());
        // while let Some(item) = stream.next().await {
        //     println!("{}", item);
        // }
        let response = chat_sync(&request).await.unwrap();
        println!("{response}");
        println!("耗时：{}ms", start.elapsed().as_millis());
    }
}
