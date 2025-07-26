use candle_core::{DType, Device, Error, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::qwen3::{Config, ModelForCausalLM};
use minijinja::{Environment, Value as MiniJinjaValue, context};
use serde_json::{Value};
use tokenizers::tokenizer::Tokenizer;
use crate::utils::get_device;

// 主请求结构体
#[derive(Debug, serde::Deserialize)]
pub struct ChatRequest {
    messages: Vec<Message>,
    tools: Option<Vec<Tool>>,
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

// 自定义字符串方法实现
pub fn str_startswith(s: &str, prefix: &str) -> bool {
    s.starts_with(prefix)
}

pub fn str_endswith(s: &str, suffix: &str) -> bool {
    s.ends_with(suffix)
}

pub struct Qwen3<'a> {
    tokenizer: Tokenizer,
    model: ModelForCausalLM,
    logits_processor: LogitsProcessor,
    jinja_env: Environment<'a>,
    pub chat_template: String,
    device: Device,
    max_generate: usize,
    repeat_penalty: f32,
    repeat_last_n: usize,
    eos_token1: Option<u32>,
    eos_token2: Option<u32>,
}

impl<'a> Qwen3<'a> {
    pub fn new(path: String) -> Result<Self> {
        Qwen3::new_with_param(path, 10000, 1.1, 64)
    }
    pub fn new_with_param(path: String, max_generate: usize, repeat_penalty: f32, repeat_last_n: usize) -> Result<Self> {
        assert!(
            std::path::Path::new(&path).exists(),
            "model path file not exists"
        );
        let tokenizer_file = path.clone() + "/tokenizer.json";
        assert!(
            std::path::Path::new(&tokenizer_file).exists(),
            "tokenizer.json not exists in model path"
        );
        let tokenizer = Tokenizer::from_file(tokenizer_file)
            .map_err(|e| Error::Msg(format!("tokenizer from file error:{}", e)))?;
        let eos_token1 = tokenizer.get_vocab(true).get("<|endoftext|>").copied();
        let eos_token2 = tokenizer.get_vocab(true).get("<|im_end|>").copied();
        let device = get_device()?;
        let weight_file = path.clone() + "/model.safetensors";
        assert!(
            std::path::Path::new(&weight_file).exists(),
            "model.safetensors not exists in model path"
        );
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weight_file], DType::F16, &device)? };
        let config_file = path.clone() + "/config.json";
        assert!(
            std::path::Path::new(&config_file).exists(),
            "config.json not exists in model path"
        );
        let config: Config = serde_json::from_slice(&std::fs::read(config_file)?)
            .map_err(|e| Error::Msg(format!("load config file error:{}", e)))?;
        let model = ModelForCausalLM::new(&config, vb)?;
        let logits_processor = LogitsProcessor::new(299792458, None, None);
        let tokenizer_config_file = path.clone() + "/tokenizer_config.json";
        assert!(
            std::path::Path::new(&tokenizer_config_file).exists(),
            "tokenizer_config.json not exists in model path"
        );
        let tokenizer_config: Value =
            serde_json::from_slice(&std::fs::read(tokenizer_config_file)?)
                .map_err(|e| Error::Msg(format!("load tokenizer_config file error:{}", e)))?;
        let chat_template = tokenizer_config["chat_template"]
            .as_str()
            .ok_or(Error::Msg(format!("chat_template to str error")))?;
        let mut env = Environment::new();

        // 添加自定义方法
        env.add_function("str_startswith", |s: String, prefix: String| {
            str_startswith(&s, &prefix)
        });

        env.add_function("str_endswith", |s: String, suffix: String| {
            str_endswith(&s, &suffix)
        });

        // 添加自定义过滤器
        env.add_filter("tojson", |v: MiniJinjaValue| {
            serde_json::to_string(&v).unwrap()
        });
        // 修复模板中的问题行
        let fixed_template = chat_template
            .replace(
                "message.content.startswith('<tool_response>')",
                "str_startswith(message.content, '<tool_response>')",
            )
            .replace(
                "message.content.endswith('</tool_response>')",
                "str_endswith(message.content, '</tool_response>')",
            );
        Ok(Self {
            tokenizer,
            model,
            logits_processor,
            jinja_env: env,
            chat_template: fixed_template,
            device,
            max_generate,
            repeat_penalty,
            repeat_last_n,
            eos_token1,
            eos_token2,
        })
    }

    pub fn add_template(&mut self, name: &'a str, source: &'a str) {
        let _ = self.jinja_env.add_template(name, source);
    }

    pub fn infer(&mut self, message_str: String) -> Result<String> {
        let mut tokens = self
            .tokenizer
            .encode(message_str, true)
            .map_err(|e| Error::Msg(format!("tokenizer encode error{}", e)))?
            .get_ids()
            .to_vec();
        let input_len = tokens.len();
        for index in 0..self.max_generate {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };
            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            if matches!(self.eos_token1, Some(eos_token) if eos_token == next_token)
                || matches!(self.eos_token2, Some(eos_token) if eos_token == next_token)
            {
                break;
            }
        }
        let all_tokens = tokens.len();
        let decode = self
            .tokenizer
            .decode(&tokens[input_len..all_tokens], true)
            .map_err(|e| Error::Msg(format!("tokenizer encode error{}", e)))?;
        self.model.clear_kv_cache();
        Ok(decode)
    }

    pub fn generate(&mut self, messages: String) -> Result<String> {
        let request: ChatRequest = serde_json::from_str(&messages)
            .map_err(|e| Error::Msg(format!("Failed to parse request JSON: {}", e)))?;
        let context = context! {
            messages => &request.messages,
            tools => &request.tools.as_ref(),
            add_generation_prompt => true,
            enable_thinking => true
        };
        let template = self.jinja_env.get_template("chat").unwrap();
        let message_str = template
            .render(context)
            .map_err(|e| Error::Msg(format!("failed to render chat template: {}", e)))?;
        self.infer(message_str)
    }
}