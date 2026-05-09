use candle_core::{Result, Tensor, Var, bail};
use std::{
    fs::{self, File},
    io::{Read, Write},
    path::Path,
};

pub const MULTISCREEN_PARAM_MAGIC: &[u8; 8] = b"MSCRP001";
pub const TRANSFORMER_PARAM_MAGIC: &[u8; 8] = b"TRFMP001";
const GENERIC_PARAM_MAGIC: &[u8; 8] = b"TNSRP001";

pub fn save_parameters(params: &[&Var], path: impl AsRef<Path>) -> Result<()> {
    save_parameters_with_magic(params, path, GENERIC_PARAM_MAGIC)
}

pub fn load_parameters(params: &[&Var], path: impl AsRef<Path>) -> Result<()> {
    load_parameters_with_magic(params, path, GENERIC_PARAM_MAGIC, "generic tensor model")
}

pub fn save_parameters_with_magic(
    params: &[&Var],
    path: impl AsRef<Path>,
    magic: &[u8; 8],
) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let mut file = File::create(path)?;
    file.write_all(magic)?;
    write_u64(&mut file, params.len() as u64)?;

    for param in params {
        let tensor = param.as_tensor();
        let dims = tensor.dims();
        write_u64(&mut file, dims.len() as u64)?;
        for dim in dims {
            write_u64(&mut file, *dim as u64)?;
        }
        let values = tensor.flatten_all()?.to_vec1::<f32>()?;
        write_u64(&mut file, values.len() as u64)?;
        for value in values {
            file.write_all(&value.to_le_bytes())?;
        }
    }

    Ok(())
}

pub fn load_parameters_with_magic(
    params: &[&Var],
    path: impl AsRef<Path>,
    magic: &[u8; 8],
    model_label: &str,
) -> Result<()> {
    let mut file = File::open(path)?;
    let mut found_magic = [0u8; 8];
    file.read_exact(&mut found_magic)?;
    if &found_magic != magic {
        bail!(
            "invalid checkpoint magic for {model_label}: expected {}, found {}",
            magic_to_string(magic),
            magic_to_string(&found_magic)
        );
    }

    let stored_count = read_u64(&mut file)? as usize;
    if stored_count != params.len() {
        bail!(
            "parameter count mismatch: file has {stored_count}, model has {}",
            params.len()
        );
    }

    for param in params {
        let rank = read_u64(&mut file)? as usize;
        let mut dims = Vec::with_capacity(rank);
        for _ in 0..rank {
            dims.push(read_u64(&mut file)? as usize);
        }

        if dims.as_slice() != param.as_tensor().dims() {
            bail!(
                "parameter shape mismatch: file has {:?}, model expects {:?}",
                dims,
                param.as_tensor().dims()
            );
        }

        let value_count = read_u64(&mut file)? as usize;
        let expected_count = dims.iter().product::<usize>();
        if value_count != expected_count {
            bail!(
                "parameter value count mismatch: file has {value_count}, expected {expected_count}"
            );
        }

        let mut values = Vec::with_capacity(value_count);
        for _ in 0..value_count {
            values.push(read_f32(&mut file)?);
        }
        let tensor = tensor_from_dims(values, &dims, param.device())?;
        param.set(&tensor.detach())?;
    }

    Ok(())
}

fn magic_to_string(magic: &[u8; 8]) -> String {
    String::from_utf8_lossy(magic).to_string()
}

fn write_u64(file: &mut File, value: u64) -> Result<()> {
    file.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn read_u64(file: &mut File) -> Result<u64> {
    let mut bytes = [0u8; 8];
    file.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_f32(file: &mut File) -> Result<f32> {
    let mut bytes = [0u8; 4];
    file.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

fn tensor_from_dims(
    values: Vec<f32>,
    dims: &[usize],
    device: &candle_core::Device,
) -> Result<Tensor> {
    match dims {
        [dim0] => Tensor::from_vec(values, *dim0, device),
        [dim0, dim1] => Tensor::from_vec(values, (*dim0, *dim1), device),
        _ => bail!("unsupported saved parameter rank: {}", dims.len()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    #[test]
    fn model_specific_magic_rejects_wrong_architecture() -> Result<()> {
        let device = Device::Cpu;
        let source = Var::from_vec(vec![1.0f32, 2.0], (2,), &device)?;
        let target = Var::from_vec(vec![0.0f32, 0.0], (2,), &device)?;
        let path = std::env::temp_dir().join(format!(
            "multiscreen_param_magic_{}.params",
            std::process::id()
        ));

        save_parameters_with_magic(&[&source], &path, MULTISCREEN_PARAM_MAGIC)?;
        let error =
            load_parameters_with_magic(&[&target], &path, TRANSFORMER_PARAM_MAGIC, "transformer")
                .unwrap_err();

        let message = error.to_string();
        assert!(message.contains("invalid checkpoint magic"));
        assert!(message.contains("MSCRP001"));
        assert!(message.contains("TRFMP001"));

        let _ = std::fs::remove_file(path);
        Ok(())
    }
}
