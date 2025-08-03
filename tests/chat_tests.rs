use qwen3_deploy::{init, chat_stream, chat_sync, ChatRequest};
use rocket::futures::{StreamExt};
use std::pin::pin;

#[tokio::test]
async fn test_chat_sync() {
    // RUST_BACKTRACE=1 cargo test test_chat_sync -- --nocapture
    let message = r#"
    {
        "messages": [
            {
                "role": "user",
                "content": "现在几点了？成都今天天气如何？"
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
                    "name": "get_current_time",
                    "description": "当你想知道现在的时间时非常有用。",
                    "parameters": {}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "当你想查询指定城市的天气时非常有用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "城市或县区，比如北京市、杭州市、余杭区等。"
                            }
                        },
                        "required": ["location"]
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
    let response = chat_sync(&request).await.unwrap();
    println!("{response}");
    println!("耗时：{}ms", start.elapsed().as_millis());

}

#[tokio::test]
async fn test_chat_stream() {
    
    // RUST_BACKTRACE=1 cargo test test_chat_stream -- --nocapture

    let message = r#"
    {
        "messages": [
            {
                "role": "user",
                "content": "图片里有什么？图片地址：[\"data/chat/kb/17992851581189image_1753331929967.png\"]"
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
    let mut stream = pin!(chat_stream(&request).unwrap());
    while let Some(item) = stream.next().await {
        println!("{}", item);
    }
    println!("耗时：{}ms", start.elapsed().as_millis());

}