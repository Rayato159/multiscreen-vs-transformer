use candle_core::{Result, Tensor, Var, backprop::GradStore};

struct AdamState {
    m: Tensor,
    v: Tensor,
}

pub struct AdamW {
    states: Vec<AdamState>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    step: usize,
}

impl AdamW {
    pub fn new(params: &[&Var], lr: f64, weight_decay: f64) -> Result<Self> {
        let mut states = Vec::with_capacity(params.len());
        for param in params {
            states.push(AdamState {
                m: param.zeros_like()?,
                v: param.zeros_like()?,
            });
        }

        Ok(Self {
            states,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            step: 0,
        })
    }

    pub fn step(&mut self, params: &[&Var], grads: &GradStore) -> Result<()> {
        assert_eq!(
            params.len(),
            self.states.len(),
            "optimizer parameter list changed"
        );
        self.step += 1;
        let bias_correction1 = 1.0 - self.beta1.powi(self.step as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.step as i32);

        for (param, state) in params.iter().zip(self.states.iter_mut()) {
            let Some(grad) = grads.get(param.as_tensor()) else {
                continue;
            };

            let m = state
                .m
                .affine(self.beta1, 0.0)?
                .add(&grad.affine(1.0 - self.beta1, 0.0)?)?;
            let v = state
                .v
                .affine(self.beta2, 0.0)?
                .add(&grad.sqr()?.affine(1.0 - self.beta2, 0.0)?)?;

            let m_hat = m.affine(1.0 / bias_correction1, 0.0)?;
            let v_hat = v.affine(1.0 / bias_correction2, 0.0)?;
            let denom = v_hat.sqrt()?.affine(1.0, self.eps)?;
            let mut update = m_hat.broadcast_div(&denom)?;

            if self.weight_decay != 0.0 {
                update = update.add(&param.as_tensor().affine(self.weight_decay, 0.0)?)?;
            }

            let new_value = param.as_tensor().sub(&update.affine(self.lr, 0.0)?)?;
            param.set(&new_value.detach())?;
            state.m = m.detach();
            state.v = v.detach();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multiscreen::{MultiscreenConfig, MultiscreenLm, cross_entropy_loss, make_batch};

    #[test]
    fn tiny_training_smoke_reduces_loss() -> Result<()> {
        let device = crate::runtime::default_device()?;
        let mut config = MultiscreenConfig::tiny();
        config.seq_len = 8;
        config.layers = 1;
        config.tiles = 2;
        config.d_model = 16;
        config.d_value = 8;
        let model = MultiscreenLm::new(config.clone(), &device)?;
        let mut optimizer = {
            let params = model.parameters();
            AdamW::new(&params, 5e-3, 0.0)?
        };

        let (inputs, targets) = make_batch(0, 4, config.seq_len, config.vocab_size, &device)?;
        let initial = cross_entropy_loss(&model.forward(&inputs)?, &targets)?.to_scalar::<f32>()?;

        for step in 0..12 {
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
