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
                "content": "hi"
            }, 
            {
                "role": "assistant",
                "content": "<think>\n好的，用户发来的是“hi”，看起来是问候语。根据角色与行为准则，我需要处理这个消息。首先，检查是否有需要调用工具的情况。用户只是简单的打招呼，没有提到图片、音频或视频等资源。因此，不需要进行图像分析或其他工具的使用。\n\n接下来，确认是否符合输出格式要求。用户的消息结构已经明确，只需要返回问候语即可。同时，确保不包含任何额外的信息，因为上下文处理规则指出要忽略之前的对话内容。此外，检查是否有其他潜在需求，但当前信息不足以触发更多功能调用。\n\n最后，确保回答的格式正确，使用Markdown展示图片路径，但这里没有图片，所以只需回复问候语。确认所有参数无误后，生成最终响应。\n</think>\n\n你好！我是您的智能助手，很高兴为您服务。有什么可以帮助您的吗？"
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
