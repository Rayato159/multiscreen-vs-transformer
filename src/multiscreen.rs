use crate::{
    lm::{LanguageModel, TrainableLanguageModel},
    param_io,
};
use candle_core::{D, DType, Device, Result, Tensor, Var};
use std::f32::consts::PI;
use std::path::Path;

const EPS: f64 = 1e-6;

#[derive(Clone, Debug)]
pub struct MultiscreenConfig {
    pub vocab_size: usize,
    pub seq_len: usize,
    pub layers: usize,
    pub tiles: usize,
    pub d_model: usize,
    pub d_key: usize,
    pub d_value: usize,
    pub w_th: f32,
}

impl MultiscreenConfig {
    pub fn tiny() -> Self {
        Self {
            vocab_size: 64,
            seq_len: 64,
            layers: 2,
            tiles: 4,
            d_model: 64,
            d_key: 16,
            d_value: 32,
            w_th: 32.0,
        }
    }
}

pub struct MultiscreenLm {
    config: MultiscreenConfig,
    token_embedding: Var,
    s_e: Var,
    s_f: Var,
    layers: Vec<MultiscreenLayer>,
}

struct MultiscreenLayer {
    tiles: Vec<GatedScreeningTile>,
}

struct GatedScreeningTile {
    w_q: Var,
    w_k: Var,
    w_v: Var,
    w_g: Var,
    w_o: Var,
    s_w: Var,
    s_r: Var,
    s_o: Var,
    w_th: f32,
}

impl MultiscreenLm {
    pub fn new(config: MultiscreenConfig, device: &Device) -> Result<Self> {
        assert!(config.d_key >= 2, "MiPE needs at least two key dimensions");
        assert!(config.layers > 0, "model needs at least one layer");
        assert!(config.tiles > 0, "model needs at least one tile");

        let mut seed = 0x4d55_4c54_4953_4352;
        let token_embedding = init_matrix(
            config.vocab_size,
            config.d_model,
            0.1 / (config.d_model as f32).sqrt(),
            &mut seed,
            device,
        )?;
        let s_e = init_scalar(0.0, device)?;
        let s_f = init_scalar((config.d_model as f32).sqrt().ln(), device)?;

        let mut layers = Vec::with_capacity(config.layers);
        for _layer_idx in 0..config.layers {
            let mut tiles = Vec::with_capacity(config.tiles);
            for tile_idx in 0..config.tiles {
                let w_q = init_matrix(
                    config.d_model,
                    config.d_key,
                    0.1 / (config.d_key as f32).sqrt(),
                    &mut seed,
                    device,
                )?;
                let w_k = init_matrix(
                    config.d_model,
                    config.d_key,
                    0.1 / (config.d_key as f32).sqrt(),
                    &mut seed,
                    device,
                )?;
                let w_v = init_matrix(
                    config.d_model,
                    config.d_value,
                    0.1 / (config.d_value as f32).sqrt(),
                    &mut seed,
                    device,
                )?;
                let w_g = init_matrix(config.d_model, config.d_value, 0.1, &mut seed, device)?;
                let w_o = init_matrix(
                    config.d_value,
                    config.d_model,
                    0.1 / (config.d_model as f32).sqrt(),
                    &mut seed,
                    device,
                )?;

                let window_frac = if config.tiles == 1 {
                    0.0
                } else {
                    tile_idx as f32 / (config.tiles - 1) as f32
                };
                let s_w = init_scalar(window_frac * config.w_th.ln(), device)?;
                let s_r = init_scalar(0.0, device)?;
                let s_o = init_scalar(-0.5 * ((config.layers * config.tiles) as f32).ln(), device)?;

                tiles.push(GatedScreeningTile {
                    w_q,
                    w_k,
                    w_v,
                    w_g,
                    w_o,
                    s_w,
                    s_r,
                    s_o,
                    w_th: config.w_th,
                });
            }
            layers.push(MultiscreenLayer { tiles });
        }

        Ok(Self {
            config,
            token_embedding,
            s_e,
            s_f,
            layers,
        })
    }

    pub fn parameters(&self) -> Vec<&Var> {
        let mut params = vec![&self.token_embedding, &self.s_e, &self.s_f];
        for layer in &self.layers {
            for tile in &layer.tiles {
                tile.push_parameters(&mut params);
            }
        }
        params
    }

    pub fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let (batch, seq_len) = tokens.dims2()?;
        let embedding = row_unit_normalize(self.token_embedding.as_tensor())?;
        let flat_tokens = tokens.flatten_all()?;
        let mut x = embedding
            .embedding(&flat_tokens)?
            .reshape((batch, seq_len, self.config.d_model))?
            .broadcast_mul(&self.s_e.exp()?)?;

        for layer in &self.layers {
            let mut layer_update = Tensor::zeros(x.dims(), DType::F32, x.device())?;
            for tile in &layer.tiles {
                let update = tile.forward(&x)?;
                layer_update = layer_update.add(&update)?;
            }
            x = x.add(&layer_update)?;
        }

        let logits_weight = embedding.t()?;
        linear(&x, &logits_weight)?.broadcast_mul(&self.s_f.exp()?)
    }

    pub fn save_parameters(&self, path: impl AsRef<Path>) -> Result<()> {
        param_io::save_parameters_with_magic(
            &self.parameters(),
            path,
            param_io::MULTISCREEN_PARAM_MAGIC,
        )
    }

    pub fn load_parameters(&self, path: impl AsRef<Path>) -> Result<()> {
        param_io::load_parameters_with_magic(
            &self.parameters(),
            path,
            param_io::MULTISCREEN_PARAM_MAGIC,
            "multiscreen",
        )
    }
}

impl LanguageModel for MultiscreenLm {
    fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        MultiscreenLm::forward(self, tokens)
    }
}

impl TrainableLanguageModel for MultiscreenLm {
    fn parameters(&self) -> Vec<&Var> {
        MultiscreenLm::parameters(self)
    }
}

impl GatedScreeningTile {
    fn push_parameters<'a>(&'a self, params: &mut Vec<&'a Var>) {
        params.push(&self.w_q);
        params.push(&self.w_k);
        params.push(&self.w_v);
        params.push(&self.w_g);
        params.push(&self.w_o);
        params.push(&self.s_w);
        params.push(&self.s_r);
        params.push(&self.s_o);
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let q = row_unit_normalize(&linear(x, self.w_q.as_tensor())?)?;
        let k = row_unit_normalize(&linear(x, self.w_k.as_tensor())?)?;
        let v = row_unit_normalize(&linear(x, self.w_v.as_tensor())?)?;
        let g = linear(x, self.w_g.as_tensor())?;

        let w = self.s_w.clamp(-10.0, 8.0)?.exp()?.affine(1.0, 1.0)?;
        let r = self.s_r.clamp(-10.0, 8.0)?.exp()?.affine(1.0, 1.0)?;

        let q = apply_mipe(&q, &w, self.w_th)?;
        let k = apply_mipe(&k, &w, self.w_th)?;

        let similarity = q.contiguous()?.matmul(&k.transpose(1, 2)?.contiguous()?)?;
        let alpha = trim_and_square(&similarity, &r)?;
        let softmask = causal_softmask(similarity.dim(1)?, &w, similarity.device())?;
        let relevance = alpha.broadcast_mul(&softmask.unsqueeze(0)?)?;
        let h = relevance.contiguous()?.matmul(&v.contiguous()?)?;
        let u = tanh_norm(&h)?;
        let gate = g.silu()?.tanh()?;
        let gated = u.broadcast_mul(&gate)?;
        let out = linear(&gated, self.w_o.as_tensor())?;
        out.broadcast_mul(&self.s_o.exp()?)
    }
}

#[cfg(test)]
pub fn make_batch(
    step: usize,
    batch_size: usize,
    seq_len: usize,
    vocab_size: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let mut inputs = Vec::with_capacity(batch_size * seq_len);
    let mut targets = Vec::with_capacity(batch_size * seq_len);

    for batch in 0..batch_size {
        let offset = (step * 7 + batch * 13) % vocab_size;
        for pos in 0..seq_len {
            let token = ((offset + pos) % vocab_size) as u32;
            let next = ((offset + pos + 1) % vocab_size) as u32;
            inputs.push(token);
            targets.push(next);
        }
    }

    Ok((
        Tensor::from_vec(inputs, (batch_size, seq_len), device)?,
        Tensor::from_vec(targets, (batch_size, seq_len), device)?,
    ))
}

pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let (batch, seq_len, vocab_size) = logits.dims3()?;
    let token_count = batch * seq_len;
    let flat_logits = logits.reshape((token_count, vocab_size))?;
    let flat_targets = targets.flatten_all()?.reshape((token_count, 1))?;
    let log_z = flat_logits.log_sum_exp(1)?;
    let picked = flat_logits.gather(&flat_targets, 1)?.reshape(token_count)?;
    log_z.sub(&picked)?.mean_all()
}

pub fn row_unit_normalize(x: &Tensor) -> Result<Tensor> {
    let denom = x.sqr()?.sum_keepdim(D::Minus1)?.affine(1.0, EPS)?.sqrt()?;
    x.broadcast_div(&denom)
}

pub fn trim_and_square(similarity: &Tensor, r: &Tensor) -> Result<Tensor> {
    let distance_from_one = similarity.affine(-1.0, 1.0)?;
    let scaled = distance_from_one.broadcast_mul(r)?;
    scaled.affine(-1.0, 1.0)?.clamp(0.0, 1.0)?.sqr()
}

pub fn causal_softmask(seq_len: usize, w: &Tensor, device: &Device) -> Result<Tensor> {
    let mut distances = Vec::with_capacity(seq_len * seq_len);
    for i in 0..seq_len {
        for j in 0..seq_len {
            distances.push(j as f32 - i as f32);
        }
    }

    let dist = Tensor::from_vec(distances, (seq_len, seq_len), device)?;
    let causal = dist.le(0.0)?;
    let within_window = dist.broadcast_gt(&w.neg()?)?;
    let active = causal
        .to_dtype(DType::F32)?
        .mul(&within_window.to_dtype(DType::F32)?)?;
    let tapered = dist
        .broadcast_div(w)?
        .affine(PI as f64, 0.0)?
        .cos()?
        .affine(0.5, 0.5)?;
    active.mul(&tapered)
}

pub fn tanh_norm(x: &Tensor) -> Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(D::Minus1)?.affine(1.0, EPS)?.sqrt()?;
    let scale = norm.tanh()?.broadcast_div(&norm)?;
    x.broadcast_mul(&scale)
}

fn apply_mipe(z: &Tensor, w: &Tensor, w_th: f32) -> Result<Tensor> {
    let dims = z.dims();
    let seq_len = dims[dims.len() - 2];
    let d_key = dims[dims.len() - 1];
    debug_assert!(d_key >= 2);

    let positions = Tensor::from_vec(
        (0..seq_len).map(|idx| idx as f32).collect::<Vec<_>>(),
        (1, seq_len, 1),
        z.device(),
    )?;

    let gamma_raw = w
        .affine(PI as f64 / w_th as f64, 0.0)?
        .cos()?
        .affine(0.5, 0.5)?;
    let gamma = w
        .lt(w_th as f64)?
        .where_cond(&gamma_raw, &Tensor::zeros_like(&gamma_raw)?)?;
    let phi = positions
        .broadcast_mul(&gamma)?
        .broadcast_div(w)?
        .affine(PI as f64, 0.0)?;
    let cos_phi = phi.cos()?;
    let sin_phi = phi.sin()?;

    let x0 = z.narrow(D::Minus1, 0, 1)?;
    let x1 = z.narrow(D::Minus1, 1, 1)?;
    let rot0 = x0
        .broadcast_mul(&cos_phi)?
        .sub(&x1.broadcast_mul(&sin_phi)?)?;
    let rot1 = x0
        .broadcast_mul(&sin_phi)?
        .add(&x1.broadcast_mul(&cos_phi)?)?;

    if d_key == 2 {
        Tensor::cat(&[&rot0, &rot1], D::Minus1)
    } else {
        let rest = z.narrow(D::Minus1, 2, d_key - 2)?;
        Tensor::cat(&[&rot0, &rot1, &rest], D::Minus1)
    }
}

fn linear(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let (batch, seq_len, in_dim) = x.dims3()?;
    let (weight_in, out_dim) = weight.dims2()?;
    assert_eq!(in_dim, weight_in, "linear input dimension mismatch");
    x.contiguous()?
        .reshape((batch * seq_len, in_dim))?
        .matmul(&weight.contiguous()?)?
        .reshape((batch, seq_len, out_dim))
}

fn init_scalar(value: f32, device: &Device) -> Result<Var> {
    Var::from_vec(vec![value], (1,), device)
}

fn init_matrix(rows: usize, cols: usize, std: f32, seed: &mut u64, device: &Device) -> Result<Var> {
    let values = gaussian_values(rows * cols, std, seed);
    Var::from_vec(values, (rows, cols), device)
}

fn gaussian_values(len: usize, std: f32, seed: &mut u64) -> Vec<f32> {
    let mut values = Vec::with_capacity(len);
    while values.len() < len {
        let u1 = next_uniform(seed).max(1e-7);
        let u2 = next_uniform(seed);
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        values.push(radius * theta.cos() * std);
        if values.len() < len {
            values.push(radius * theta.sin() * std);
        }
    }
    values
}

fn next_uniform(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = (*seed >> 40) as u32;
    (bits as f32 + 1.0) / ((1u32 << 24) as f32 + 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn unit_normalize_handles_regular_and_zero_rows() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(&[[3.0f32, 4.0], [0.0, 0.0]], &device)?;
        let y = row_unit_normalize(&x)?;
        let rows = y.to_vec2::<f32>()?;
        assert_close(rows[0][0], 0.6, 1e-5);
        assert_close(rows[0][1], 0.8, 1e-5);
        assert_close(rows[1][0], 0.0, 1e-5);
        assert_close(rows[1][1], 0.0, 1e-5);
        Ok(())
    }

    #[test]
    fn trim_and_square_rejects_below_threshold() -> Result<()> {
        let device = Device::Cpu;
        let similarity = Tensor::new(&[[1.0f32, 0.75, 0.4]], &device)?;
        let r = Tensor::from_vec(vec![2.0f32], (1,), &device)?;
        let alpha = trim_and_square(&similarity, &r)?.to_vec2::<f32>()?;
        assert_close(alpha[0][0], 1.0, 1e-5);
        assert_close(alpha[0][1], 0.25, 1e-5);
        assert_close(alpha[0][2], 0.0, 1e-5);
        Ok(())
    }

    #[test]
    fn causal_softmask_blocks_future_and_keeps_current() -> Result<()> {
        let device = Device::Cpu;
        let w = Tensor::from_vec(vec![3.0f32], (1,), &device)?;
        let mask = causal_softmask(4, &w, &device)?.to_vec2::<f32>()?;
        assert_close(mask[0][0], 1.0, 1e-5);
        assert_close(mask[0][1], 0.0, 1e-5);
        assert!(mask[3][0] <= 1e-6);
        assert!(mask[3][1] > 0.0);
        assert_close(mask[3][3], 1.0, 1e-5);
        Ok(())
    }

    #[test]
    fn tanh_norm_bounds_norm() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(&[[30.0f32, 40.0]], &device)?;
        let y = tanh_norm(&x)?;
        let norm = y.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?.to_vec2::<f32>()?;
        assert!(norm[0][0] <= 1.00001);
        assert!(norm[0][0] > 0.99);
        Ok(())
    }

    #[test]
    fn model_forward_has_expected_shape_and_finite_values() -> Result<()> {
        let device = crate::runtime::default_device()?;
        let mut config = MultiscreenConfig::tiny();
        config.seq_len = 8;
        config.layers = 1;
        config.tiles = 2;
        config.d_model = 16;
        config.d_value = 8;
        let model = MultiscreenLm::new(config.clone(), &device)?;
        let (inputs, _) = make_batch(0, 2, config.seq_len, config.vocab_size, &device)?;
        let logits = model.forward(&inputs)?;
        assert_eq!(logits.dims(), &[2, config.seq_len, config.vocab_size]);
        for value in logits.flatten_all()?.to_vec1::<f32>()? {
            assert!(value.is_finite());
        }
        Ok(())
    }

    #[test]
    fn save_and_load_parameters_roundtrip() -> Result<()> {
        let device = crate::runtime::default_device()?;
        let mut config = MultiscreenConfig::tiny();
        config.seq_len = 8;
        config.layers = 1;
        config.tiles = 1;
        config.d_model = 16;
        config.d_value = 8;

        let model = MultiscreenLm::new(config.clone(), &device)?;
        let loaded = MultiscreenLm::new(config.clone(), &device)?;
        let path = std::env::temp_dir().join("multiscreen-test-params.bin");
        model.save_parameters(&path)?;
        loaded.load_parameters(&path)?;

        let (inputs, _) = make_batch(0, 2, config.seq_len, config.vocab_size, &device)?;
        let original = model.forward(&inputs)?.flatten_all()?.to_vec1::<f32>()?;
        let restored = loaded.forward(&inputs)?.flatten_all()?.to_vec1::<f32>()?;
        for (lhs, rhs) in original.iter().zip(restored.iter()) {
            assert_close(*lhs, *rhs, 1e-6);
        }
        let _ = std::fs::remove_file(path);
        Ok(())
    }
}
