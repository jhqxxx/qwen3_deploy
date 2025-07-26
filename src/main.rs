use candle_core::{Result};
mod qwen3;
use qwen3::Qwen3;
mod utils;

fn main() -> Result<()> {
    let model_path = "/mnt/c/jhq/huggingface_model/Qwen/Qwen3-0___6B/".to_string();
    let mut model = Qwen3::new(model_path)?;
    let chat_template = model.chat_template.clone();
    model.add_template("chat", &chat_template);
    let request_json = r#"
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
    let response = model.generate(request_json.to_string())?;
    println!("generate: \n {}", response);
    Ok(())
}
