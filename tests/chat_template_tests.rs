use minijinja::{Environment, Value as MiniJinjaValue, context};
use qwen3_deploy::{
    ChatRequest,
    utils::{get_template, str_endswith, str_startswith},
};

#[test]
fn test_chat_template() {
    
    // RUST_BACKTRACE=1 cargo test test_chat_template -- --nocapture

    let message = r#"
    {
        "messages": [   
            {
                "role": "assistant",
                "content": "<think>\n好的，用户发来“你好啊”，看起来是一个简单的问候。我需要确认是否需要调用任何工具来回应。查看提供的工具描述，发现有一个名为image-qa-onlineA-_-Achat_to_image的函数，它接受图像路径和提示词作为参数。但当前用户并没有提供图片或需要生成内容的要求，只是打招呼了。因此，直接回复问候是合适的，不需要使用工具。应该保持友好且简洁的回应。\n</think>\n\n你好啊！有什么可以帮助你的吗？"
            },
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
    let request: ChatRequest = serde_json::from_str(&message).unwrap();

    let mut env = Environment::new();

    // 添加自定义过滤器
    env.add_filter("tojson", |v: MiniJinjaValue| {
        serde_json::to_string(&v).unwrap()
    });
    
    env.add_filter("split", |s: String, delimiter: String| {
        s.split(&delimiter).map(|s| s.to_string()).collect::<Vec<String>>()
    });

    // 添加 lstrip 过滤器
    env.add_filter("lstrip", |s: String, chars: Option<String>| {
        match chars {
            Some(chars_str) => s.trim_start_matches(chars_str.as_str()).to_string(),
            None => s.trim_start().to_string(),
        }
    });

    // 添加 rstrip 过滤器
    env.add_filter("rstrip", |s: String, chars: Option<String>| {
        match chars {
            Some(chars_str) => s.trim_end_matches(chars_str.as_str()).to_string(),
            None => s.trim_end().to_string(),
        }
    });

    let model_path = "/mnt/c/jhq/huggingface_model/Qwen/Qwen3-0___6B";
    let template = get_template(model_path.to_string()).unwrap();
    env.add_template("chat", &template)
        .expect("Failed to parse chat template");
    let context = context! {
        messages => &request.messages,
        tools => &request.tools.as_ref(),
        add_generation_prompt => true,
        enable_thinking => true
    };
    let template = env.get_template("chat").unwrap();
    let message_str = template
        .render(context)
        .expect("failed to render chat template");
    println!("render template: {:?}", message_str);
}
