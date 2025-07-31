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
    // 添加自定义方法
    env.add_function("str_startswith", |s: String, prefix: String| {
        str_startswith(&s, &prefix)
    });

    env.add_function("str_endswith", |s: String, suffix: String| {
        str_endswith(&s, &suffix)
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
