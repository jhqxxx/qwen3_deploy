use crate::ChatRequest;
use crate::utils::{get_device, str_startswith, str_endswith};
use candle_core::{DType, Device, Error, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::qwen3::{Config, ModelForCausalLM};
use minijinja::{Environment, Value as MiniJinjaValue, context};
use rocket::async_stream::stream;
use rocket::futures::Stream;

use std::fs;
use tokenizers::tokenizer::Tokenizer;



pub struct Qwen3<'a> {
    tokenizer: Tokenizer,
    model: ModelForCausalLM,
    logits_processor: LogitsProcessor,
    jinja_env: Environment<'a>,
    device: Device,
    max_generate: usize,
    repeat_penalty: f32,
    repeat_last_n: usize,
    eos_token1: Option<u32>,
    eos_token2: Option<u32>,
}

impl<'a> Qwen3<'a> {
    pub fn new(path: String, is_cpu: bool) -> anyhow::Result<Self> {
        Qwen3::new_with_param(path, 81920, 1.1, 64, is_cpu, 299792458, None, None)
    }
    pub fn new_with_param(
        path: String,
        max_generate: usize,
        repeat_penalty: f32,
        repeat_last_n: usize,
        is_cpu: bool,
        seed: u64,
        temperature: Option<f64>,
        top_p: Option<f64>,
    ) -> anyhow::Result<Self> {
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
            .map_err(|e| anyhow::anyhow!(format!("tokenizer from file error{}", e)))?;
        let eos_token1 = tokenizer.get_vocab(true).get("<|endoftext|>").copied();
        let eos_token2 = tokenizer.get_vocab(true).get("<|im_end|>").copied();
        let device = if is_cpu { Device::Cpu } else { get_device()? };
        let weight_files = Self::find_safetensors_files(&path)?;
        assert_ne!(weight_files.len(), 0, "no safetensors files found");
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, DType::F16, &device)? };
        let config_file = path.clone() + "/config.json";
        assert!(
            std::path::Path::new(&config_file).exists(),
            "config.json not exists in model path"
        );
        let config: Config = serde_json::from_slice(&std::fs::read(config_file)?)
            .map_err(|e| anyhow::anyhow!(format!("load config file error{}", e)))?;
        let model = ModelForCausalLM::new(&config, vb)?;
        let logits_processor = LogitsProcessor::new(seed, temperature, top_p);

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
        let _ = env.add_template("chat", include_str!("chat_template.jinja"));

        Ok(Self {
            tokenizer,
            model: model,
            logits_processor: logits_processor,
            jinja_env: env,
            device,
            max_generate,
            repeat_penalty,
            repeat_last_n,
            eos_token1,
            eos_token2,
        })
    }

    fn find_safetensors_files(path: &str) -> anyhow::Result<Vec<String>> {
        let mut files = Vec::new();

        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let file_path = entry.path();

            if file_path.is_file() {
                if let Some(extension) = file_path.extension() {
                    if extension == "safetensors" {
                        files.push(file_path.to_string_lossy().to_string());
                    }
                }
            }
        }

        Ok(files)
    }

    pub fn infer_stream(
        &mut self,
        message_str: String,
    ) -> anyhow::Result<impl Stream<Item = String>> {
        let mut tokens = self
            .tokenizer
            .encode(message_str, true)
            .map_err(|e| anyhow::anyhow!(format!("stream encode error{}", e)))?
            .get_ids()
            .to_vec();        
        let stream = stream! {
            let mut error_tokens = Vec::new();
            for index in 0..self.max_generate {
                let next_token = self.next_token(index,&mut tokens);
                if let Err(e) = next_token{
                    log::error!("model error: {}", e);
                    yield format!("model error: {}", e.to_string());
                    break;
                }

                let next_token = next_token.unwrap();
                tokens.push(next_token);

                let mut decode_ids = Vec::new();
                if error_tokens.len() > 0 {
                    decode_ids.extend_from_slice(&error_tokens);
                }
                decode_ids.push(next_token);
                let decoded_token = self
                    .tokenizer
                    .decode(&decode_ids, true)
                    .map_err(|e| anyhow::anyhow!(format!("stream decode error{}", e))).unwrap();
                if decoded_token.contains("�") {
                    error_tokens.push(next_token);
                    if error_tokens.len() > 3 {
                        error_tokens.clear();
                    }
                    continue;
                }
                error_tokens.clear();
                yield decoded_token.clone();

                if matches!(self.eos_token1, Some(eos_token) if eos_token == next_token)
                    || matches!(self.eos_token2, Some(eos_token) if eos_token == next_token)
                {
                    break;
                }
            }
            self.model.clear_kv_cache();
        };

        Ok(stream)
    }

    fn next_token(&mut self, index: usize, tokens: &mut Vec<u32>) -> anyhow::Result<u32> {
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
        Ok(next_token)
    }

    fn render_template(&mut self, request: &ChatRequest) -> anyhow::Result<String> {
        let context = context! {
            messages => &request.messages,
            tools => &request.tools.as_ref(),
            add_generation_prompt => true,
            enable_thinking => true
        };
        let template = self.jinja_env.get_template("chat")?;
        let message_str = template
            .render(context)
            .map_err(|e| anyhow::anyhow!(format!("render template  error{}", e)))?;
        Ok(message_str)
    }

    pub fn generate_stream(
        &mut self,
        request: &ChatRequest,
    ) -> anyhow::Result<impl Stream<Item = String>> {
        let message_str = self.render_template(request)?;
        let stream = self.infer_stream(message_str);
        stream
    }

    pub fn infer(&mut self, message_str: String) -> anyhow::Result<String> {
        let mut tokens = self
            .tokenizer
            .encode(message_str, true)
            .map_err(|e| anyhow::anyhow!(format!("tokenizer encode error{}", e)))?
            .get_ids()
            .to_vec();
        let input_len = tokens.len();
        for index in 0..self.max_generate {
            let next_token = self.next_token(index, &mut tokens)?;
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
            .map_err(|e| anyhow::anyhow!(format!("tokenizer decode error{}", e)))?;
        self.model.clear_kv_cache();
        Ok(decode)
    }

    pub fn generate(&mut self, request: &ChatRequest) -> anyhow::Result<String> {
        let message_str = self.render_template(request)?;
        self.infer(message_str)
    }
}
