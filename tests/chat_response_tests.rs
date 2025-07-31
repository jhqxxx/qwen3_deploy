use openai_dive::v1::resources::chat::{ChatCompletionResponse};
use qwen3_deploy::build_choice;

#[test]
fn test_chat_response() {
    let message = r#"
    <think>
    好的，用户问现在几点了和成都今天的天气。首先我需要调用get_current_time来获取当前时间，然后使用get_current_weather查询成都的天气情况。不过用户的问题里没有提到具体城市名称，但可能默认是成都。需要确认参数是否正确，比如location参数是否填写成“成都”。确保两个函数都被调用，并且参数正确无误。
    </think>

    <tool_call>
    {"name": "get_current_time", "arguments": {}}
    </tool_call>
    <tool_call>
    {"name": "get_current_weather", "arguments": {"location": "成都"}}
    </tool_call>
    "#;

    let id = uuid::Uuid::new_v4().to_string();
    let choice = build_choice(message.to_string());
    let response = ChatCompletionResponse {
        id: Some(id),
        choices: vec![choice],
        created: chrono::Utc::now().timestamp() as u32,
        model: "qwen3".to_string(),
        service_tier: None,
        system_fingerprint: None,
        object: "chat.completion".to_string(),
        usage: None,
    };
    println!("response: \n {:?}", response);

}