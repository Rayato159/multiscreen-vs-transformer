use crate::{
    lm::{LanguageModel, TrainableLanguageModel},
    param_io,
};
use candle_core::{D, Device, Result, Tensor, Var, bail};
use std::{f32::consts::PI, path::Path};

const LN_EPS: f64 = 1e-5;

#[derive(Clone, Debug)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub seq_len: usize,
    pub layers: usize,
    pub heads: usize,
    pub d_model: usize,
    pub d_ff: usize,
}

impl TransformerConfig {
    pub fn tiny() -> Self {
        Self {
            vocab_size: 64,
            seq_len: 64,
            layers: 2,
            heads: 4,
            d_model: 64,
            d_ff: 256,
        }
    }

    pub fn d_head(&self) -> usize {
        self.d_model / self.heads
    }
}

pub struct TransformerLm {
    config: TransformerConfig,
    token_embedding: Var,
    s_e: Var,
    s_f: Var,
    layers: Vec<TransformerLayer>,
}

struct TransformerLayer {
    w_q: Var,
    w_k: Var,
    w_v: Var,
    w_o: Var,
    w_ff1: Var,
    b_ff1: Var,
    w_ff2: Var,
    b_ff2: Var,
    ln1_gamma: Var,
    ln1_beta: Var,
    ln2_gamma: Var,
    ln2_beta: Var,
}

impl TransformerLm {
    pub fn new(config: TransformerConfig, device: &Device) -> Result<Self> {
        assert!(config.layers > 0, "model needs at least one layer");
        assert!(config.heads > 0, "model needs at least one attention head");
        assert_eq!(
            config.d_model % config.heads,
            0,
            "d_model must be divisible by heads"
        );

        let mut seed = 0x5452_414e_5346_4f52;
        let token_embedding = init_matrix(
            config.vocab_size,
            config.d_model,
            0.1 / (config.d_model as f32).sqrt(),
            &mut seed,
            device,
        )?;
        let s_e = init_scalar((config.d_model as f32).sqrt().ln(), device)?;
        let s_f = init_scalar(0.0, device)?;

        let mut layers = Vec::with_capacity(config.layers);
        for _ in 0..config.layers {
            layers.push(TransformerLayer {
                w_q: init_matrix(
                    config.d_model,
                    config.d_model,
                    0.1 / (config.d_model as f32).sqrt(),
                    &mut seed,
                    device,
                )?,
                w_k: init_matrix(
                    config.d_model,
                    config.d_model,
                    0.1 / (config.d_model as f32).sqrt(),
                    &mut seed,
                    device,
                )?,
                w_v: init_matrix(
                    config.d_model,
                    config.d_model,
                    0.1 / (config.d_model as f32).sqrt(),
                    &mut seed,
                    device,
                )?,
                w_o: init_matrix(
                    config.d_model,
                    config.d_model,
                    0.1 / (config.d_model as f32).sqrt(),
                    &mut seed,
                    device,
                )?,
                w_ff1: init_matrix(
                    config.d_model,
                    config.d_ff,
                    0.1 / (config.d_model as f32).sqrt(),
                    &mut seed,
                    device,
                )?,
                b_ff1: init_vector(config.d_ff, 0.0, device)?,
                w_ff2: init_matrix(
                    config.d_ff,
                    config.d_model,
                    0.1 / (config.d_ff as f32).sqrt(),
                    &mut seed,
                    device,
                )?,
                b_ff2: init_vector(config.d_model, 0.0, device)?,
                ln1_gamma: init_vector(config.d_model, 1.0, device)?,
                ln1_beta: init_vector(config.d_model, 0.0, device)?,
                ln2_gamma: init_vector(config.d_model, 1.0, device)?,
                ln2_beta: init_vector(config.d_model, 0.0, device)?,
            });
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
            layer.push_parameters(&mut params);
        }
        params
    }

    pub fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let (batch, seq_len) = tokens.dims2()?;
        if seq_len > self.config.seq_len {
            bail!(
                "input sequence length {seq_len} exceeds configured maximum {}",
                self.config.seq_len
            );
        }

        let flat_tokens = tokens.flatten_all()?;
        let mut x = self
            .token_embedding
            .embedding(&flat_tokens)?
            .reshape((batch, seq_len, self.config.d_model))?
            .broadcast_mul(&self.s_e.exp()?)?;
        let positions = sinusoidal_positions(seq_len, self.config.d_model, x.device())?;
        x = x.broadcast_add(&positions)?;

        for layer in &self.layers {
            x = layer.forward(&x, &self.config)?;
        }

        linear(&x, &self.token_embedding.as_tensor().t()?)?.broadcast_mul(&self.s_f.exp()?)
    }

    pub fn save_parameters(&self, path: impl AsRef<Path>) -> Result<()> {
        param_io::save_parameters_with_magic(
            &self.parameters(),
            path,
            param_io::TRANSFORMER_PARAM_MAGIC,
        )
    }

    pub fn load_parameters(&self, path: impl AsRef<Path>) -> Result<()> {
        param_io::load_parameters_with_magic(
            &self.parameters(),
            path,
            param_io::TRANSFORMER_PARAM_MAGIC,
            "transformer",
        )
    }
}

impl TransformerLayer {
    fn push_parameters<'a>(&'a self, params: &mut Vec<&'a Var>) {
        params.push(&self.w_q);
        params.push(&self.w_k);
        params.push(&self.w_v);
        params.push(&self.w_o);
        params.push(&self.w_ff1);
        params.push(&self.b_ff1);
        params.push(&self.w_ff2);
        params.push(&self.b_ff2);
        params.push(&self.ln1_gamma);
        params.push(&self.ln1_beta);
        params.push(&self.ln2_gamma);
        params.push(&self.ln2_beta);
    }

    fn forward(&self, x: &Tensor, config: &TransformerConfig) -> Result<Tensor> {
        let attention = self.self_attention(x, config)?;
        let x = layer_norm(
            &x.add(&attention)?,
            self.ln1_gamma.as_tensor(),
            self.ln1_beta.as_tensor(),
        )?;
        let ff = self.feed_forward(&x)?;
        layer_norm(
            &x.add(&ff)?,
            self.ln2_gamma.as_tensor(),
            self.ln2_beta.as_tensor(),
        )
    }

    fn self_attention(&self, x: &Tensor, config: &TransformerConfig) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        let q = split_heads(&linear(x, self.w_q.as_tensor())?, config)?;
        let k = split_heads(&linear(x, self.w_k.as_tensor())?, config)?;
        let v = split_heads(&linear(x, self.w_v.as_tensor())?, config)?;

        let scores = q
            .contiguous()?
            .matmul(&k.transpose(2, 3)?.contiguous()?)?
            .affine(1.0 / (config.d_head() as f64).sqrt(), 0.0)?;
        let mask = causal_attention_mask(seq_len, scores.device())?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let scores = scores.broadcast_add(&mask)?;
        let attention = candle_nn::ops::softmax(&scores.contiguous()?, D::Minus1)?;
        let context = attention.contiguous()?.matmul(&v.contiguous()?)?;
        let context =
            context
                .transpose(1, 2)?
                .contiguous()?
                .reshape((batch, seq_len, config.d_model))?;
        linear(&context, self.w_o.as_tensor())
    }

    fn feed_forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = linear_bias(x, self.w_ff1.as_tensor(), self.b_ff1.as_tensor())?.relu()?;
        linear_bias(&hidden, self.w_ff2.as_tensor(), self.b_ff2.as_tensor())
    }
}

impl LanguageModel for TransformerLm {
    fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        TransformerLm::forward(self, tokens)
    }
}

impl TrainableLanguageModel for TransformerLm {
    fn parameters(&self) -> Vec<&Var> {
        TransformerLm::parameters(self)
    }
}

fn split_heads(x: &Tensor, config: &TransformerConfig) -> Result<Tensor> {
    let (batch, seq_len, d_model) = x.dims3()?;
    debug_assert_eq!(d_model, config.d_model);
    x.contiguous()?
        .reshape((batch, seq_len, config.heads, config.d_head()))?
        .transpose(1, 2)?
        .contiguous()
}

fn causal_attention_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mut values = Vec::with_capacity(seq_len * seq_len);
    for i in 0..seq_len {
        for j in 0..seq_len {
            values.push(if j > i { -1.0e9f32 } else { 0.0 });
        }
    }
    Tensor::from_vec(values, (seq_len, seq_len), device)
}

fn layer_norm(x: &Tensor, gamma: &Tensor, beta: &Tensor) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let centered = x.broadcast_sub(&mean)?;
    let variance = centered.sqr()?.mean_keepdim(D::Minus1)?;
    let normalized = centered.broadcast_div(&variance.affine(1.0, LN_EPS)?.sqrt()?)?;
    normalized.broadcast_mul(gamma)?.broadcast_add(beta)
}

fn sinusoidal_positions(seq_len: usize, d_model: usize, device: &Device) -> Result<Tensor> {
    let mut values = Vec::with_capacity(seq_len * d_model);
    for pos in 0..seq_len {
        for dim in 0..d_model {
            let pair_dim = dim / 2;
            let denom = 10000f32.powf((2 * pair_dim) as f32 / d_model as f32);
            let angle = pos as f32 / denom;
            values.push(if dim % 2 == 0 {
                angle.sin()
            } else {
                angle.cos()
            });
        }
    }
    Tensor::from_vec(values, (1, seq_len, d_model), device)
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

fn linear_bias(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    linear(x, weight)?.broadcast_add(bias)
}

fn init_scalar(value: f32, device: &Device) -> Result<Var> {
    Var::from_vec(vec![value], (1,), device)
}

fn init_vector(len: usize, value: f32, device: &Device) -> Result<Var> {
    Var::from_vec(vec![value; len], (len,), device)
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
    use crate::{
        multiscreen::{cross_entropy_loss, make_batch},
        optim::AdamW,
    };

    #[test]
    fn causal_mask_blocks_future_positions() -> Result<()> {
        let device = Device::Cpu;
        let mask = causal_attention_mask(4, &device)?.to_vec2::<f32>()?;
        assert_eq!(mask[0][0], 0.0);
        assert!(mask[0][1] < -1e8);
        assert_eq!(mask[3][0], 0.0);
        assert_eq!(mask[3][3], 0.0);
        Ok(())
    }

    #[test]
    fn transformer_forward_has_expected_shape_and_finite_values() -> Result<()> {
        let device = crate::runtime::default_device()?;
        let mut config = TransformerConfig::tiny();
        config.seq_len = 8;
        config.layers = 1;
        config.heads = 2;
        config.d_model = 16;
        config.d_ff = 32;
        let model = TransformerLm::new(config.clone(), &device)?;
        let (inputs, _) = make_batch(0, 2, config.seq_len, config.vocab_size, &device)?;
        let logits = model.forward(&inputs)?;
        assert_eq!(logits.dims(), &[2, config.seq_len, config.vocab_size]);
        for value in logits.flatten_all()?.to_vec1::<f32>()? {
            assert!(value.is_finite());
        }
        Ok(())
    }

    #[test]
    fn transformer_training_smoke_reduces_loss() -> Result<()> {
        let device = crate::runtime::default_device()?;
        let mut config = TransformerConfig::tiny();
        config.seq_len = 8;
        config.layers = 1;
        config.heads = 2;
        config.d_model = 16;
        config.d_ff = 32;
        let model = TransformerLm::new(config.clone(), &device)?;
        let mut optimizer = {
            let params = model.parameters();
            AdamW::new(&params, 5e-3, 0.0)?
        };

        let (inputs, targets) = make_batch(0, 4, config.seq_len, config.vocab_size, &device)?;
        let initial = cross_entropy_loss(&model.forward(&inputs)?, &targets)?.to_scalar::<f32>()?;

        for step in 0..30 {
            let (inputs, targets) =
                make_batch(step, 4, config.seq_len, config.vocab_size, &device)?;
            let loss = cross_entropy_loss(&model.forward(&inputs)?, &targets)?;
            let grads = loss.backward()?;
            let params = model.parameters();
            optimizer.step(&params, &grads)?;
        }

        let final_loss =
            cross_entropy_loss(&model.forward(&inputs)?, &targets)?.to_scalar::<f32>()?;
        assert!(
            final_loss < initial,
            "expected loss to decrease: initial={initial}, final={final_loss}"
        );
        Ok(())
    }
}
