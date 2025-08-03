use candle_core::{Device, Error, Result};
use serde_json::Value;
use std::process::Command;

pub fn gpu_sm_arch_is_ok() -> Result<bool> {
    let output = Command::new("nvidia-smi")
        .arg("--query-gpu=compute_cap")
        .arg("--format=csv,noheader")
        .output()
        .map_err(|e| Error::Msg(format!("Failed to execute nvidia-smi: {}", e)))?;
    if !output.status.success() {
        return Err(Error::Msg(format!(
            "nvidia-smi failed with status: {}\nError: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    let output_str = String::from_utf8_lossy(&output.stdout);
    let output_str = output_str.trim();
    let sm_float = match output_str.parse::<f32>() {
        Ok(num) => num,
        Err(_) => {
            return Err(Error::Msg(format!(
                "gpr sm arch: {} parse float32 error",
                output_str
            )));
        }
    };
    if sm_float >= 6.1 { Ok(true) } else { Ok(false) }
}

pub fn get_device() -> Result<Device> {
    let device = match gpu_sm_arch_is_ok() {
        Ok(flag) => {
            if flag {
                Device::cuda_if_available(0)?
            } else {
                Device::Cpu
            }
        }
        Err(_) => Device::Cpu,
    };
    Ok(device)
}

pub fn get_template(path: String) -> Result<String> {
    let tokenizer_config_file = path.clone() + "/tokenizer_config.json";
    assert!(
        std::path::Path::new(&tokenizer_config_file).exists(),
        "tokenizer_config.json not exists in model path"
    );
    let tokenizer_config: Value = serde_json::from_slice(&std::fs::read(tokenizer_config_file)?)
        .map_err(|e| Error::Msg(format!("load tokenizer_config file error:{}", e)))?;
    let chat_template = tokenizer_config["chat_template"]
        .as_str()
        .ok_or(Error::Msg(format!("chat_template to str error")))?;
    // 修复模板中的问题行
    let fixed_template = chat_template
        .replace(
            "message.content.startswith('<tool_response>')",
            "message.content is startingwith('<tool_response>')"     // 使用minijinja中的 is startingwith 替换
        )
        .replace(
            "message.content.endswith('</tool_response>')",
            "message.content is endingwith('</tool_response>')"     // 使用minijinja中的 is endingwith 替换
        ).replace(
            "content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n')",
            "((content | split('</think>'))[0] | rstrip('\\n') | split('<think>'))[-1] | lstrip('\\n')"  // 使用自定义的split, rstrip, lstrip过滤器替换
        )
        .replace(
            "content.split('</think>')[-1].lstrip('\\n')",
            "(content | split('</think>'))[-1] | lstrip('\\n')"   // 使用自定义的过滤器替换
        )
        .replace(
            "reasoning_content.strip('\\n')",
            "reasoning_content | strip('\\n')"    // 使用自定义的过滤器替换
        )
        .replace(
            "content.lstrip('\\n')",
            "content | lstrip('\\n')"   // 使用自定义的过滤器替换
        );
    if fixed_template.contains(".split(") {
        println!("-------------------------------- Warning: Template still contains .split() method calls");
    }
    println!("\n\n{}\n\n", fixed_template);
    Ok(fixed_template)
}

// 自定义字符串方法实现
pub fn str_startswith(s: &str, prefix: &str) -> bool {
    s.starts_with(prefix)
}

pub fn str_endswith(s: &str, suffix: &str) -> bool {
    s.ends_with(suffix)
}
