use candle_core::{Device, Result, bail};
use std::env;

pub fn default_device() -> Result<Device> {
    let requested = env::var("MULTISCREEN_DEVICE").unwrap_or_else(|_| "auto".to_string());
    match requested.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => auto_device(),
        "cpu" => Ok(Device::Cpu),
        "cuda" | "gpu" => cuda_device(),
        other => bail!("unsupported MULTISCREEN_DEVICE={other:?}; use auto, cpu, cuda, or gpu"),
    }
}

pub fn device_label(device: &Device) -> String {
    match device.location() {
        candle_core::DeviceLocation::Cpu => "CPU".to_string(),
        candle_core::DeviceLocation::Cuda { gpu_id } => format!("CUDA:{gpu_id}"),
        candle_core::DeviceLocation::Metal { gpu_id } => format!("Metal:{gpu_id}"),
    }
}

#[cfg(feature = "cuda")]
fn auto_device() -> Result<Device> {
    Device::cuda_if_available(0)
}

#[cfg(not(feature = "cuda"))]
fn auto_device() -> Result<Device> {
    Ok(Device::Cpu)
}

#[cfg(feature = "cuda")]
fn cuda_device() -> Result<Device> {
    Device::new_cuda(0)
}

#[cfg(not(feature = "cuda"))]
fn cuda_device() -> Result<Device> {
    bail!("CUDA requested but this binary was built without `--features cuda`")
}
