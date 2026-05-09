use candle_core::{Result, Tensor, Var};

pub trait LanguageModel {
    fn forward(&self, tokens: &Tensor) -> Result<Tensor>;
}

pub trait TrainableLanguageModel: LanguageModel {
    fn parameters(&self) -> Vec<&Var>;

    fn parameter_count(&self) -> usize {
        self.parameters()
            .iter()
            .map(|param| param.as_tensor().dims().iter().product::<usize>())
            .sum()
    }
}
