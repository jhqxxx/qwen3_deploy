use std::process::Command;

use candle_core::{Result, Device, Error};

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
        Err(_) => return Err(Error::Msg(format!(
          "gpr sm arch: {} parse float32 error", output_str
        )))
    };
    if sm_float > 7.6 {
        Ok(true)
    } else {
        Ok(false)
    }
}

pub fn get_device() -> Result<Device> {
    let device = match gpu_sm_arch_is_ok() {
        Ok(flag) => {
            if flag {
                Device::cuda_if_available(0)?
            } else {
                Device::Cpu
            }
        },
        Err(_) => Device::Cpu
    };
    Ok(device)
}