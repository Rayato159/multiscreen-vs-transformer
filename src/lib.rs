pub mod cli;
pub mod exam;
pub mod lm;
pub mod model;
pub mod model_kind;
pub mod multiscreen;
pub mod optim;
pub mod param_io;
pub mod runtime;
pub mod transformer;

use candle_core::{Device, Result};
use exam::{ExamDataset, benchmark_inference, evaluate_test_metrics, evaluate_val_metrics};
use lm::TrainableLanguageModel;
use model_kind::ModelKind;
use multiscreen::{MultiscreenConfig, MultiscreenLm, cross_entropy_loss};
use optim::AdamW;
use runtime::{default_device, device_label};
use transformer::{TransformerConfig, TransformerLm};

pub const DEFAULT_DATASET_PATH: &str = "exam/sat_world_and_us_history.csv";
pub const DEFAULT_SEQ_LEN: usize = 96;
pub const DEFAULT_BATCH_SIZE: usize = 4;
pub const DEFAULT_TRAIN_STEPS: usize = 1000;

#[derive(Clone, Debug)]
pub struct TrainingConfig {
    pub model_kind: ModelKind,
    pub dataset_path: String,
    pub steps: usize,
    pub param_path: Option<String>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            model_kind: ModelKind::Multiscreen,
            dataset_path: DEFAULT_DATASET_PATH.to_string(),
            steps: DEFAULT_TRAIN_STEPS,
            param_path: None,
        }
    }
}

pub struct ModelRunMetrics {
    pub name: &'static str,
    pub parameters: usize,
    pub final_train_loss: f32,
    pub val_loss: f32,
    pub val_accuracy: f32,
    pub test_loss: f32,
    pub test_accuracy: f32,
    pub inference_avg_ms: f64,
    pub inference_min_ms: f64,
    pub inference_max_ms: f64,
}

/// Library function for training that can be used by the train binary.
/// Defaults to Multiscreen to preserve the original command behavior.
pub fn run_training() -> Result<()> {
    run_training_with_config(TrainingConfig::default()).map(|_| ())
}

pub fn run_training_with_config(config: TrainingConfig) -> Result<ModelRunMetrics> {
    let device = default_device()?;
    let dataset = ExamDataset::from_file(&config.dataset_path)?;
    let param_path = config
        .param_path
        .as_deref()
        .unwrap_or_else(|| config.model_kind.default_param_path());

    println!("device: {}", device_label(&device));
    print_dataset_summary(&dataset);
    println!("model: {}", config.model_kind.display_name());
    println!("training steps: {}", config.steps);
    println!("checkpoint: {param_path}");

    match config.model_kind {
        ModelKind::Multiscreen => {
            run_multiscreen_training(&dataset, &device, config.steps, param_path)
        }
        ModelKind::Transformer => {
            run_transformer_training(&dataset, &device, config.steps, param_path)
        }
    }
}

pub fn run_comparison(steps: usize) -> Result<Vec<ModelRunMetrics>> {
    let device = default_device()?;
    let dataset = ExamDataset::from_file(DEFAULT_DATASET_PATH)?;

    println!("device: {}", device_label(&device));
    print_dataset_summary(&dataset);
    println!("comparison steps: {steps}");

    let multiscreen_config = multiscreen_config(dataset.vocab_size(), DEFAULT_SEQ_LEN);
    let multiscreen = MultiscreenLm::new(multiscreen_config.clone(), &device)?;
    let multiscreen_metrics = train_evaluate_model(
        ModelKind::Multiscreen.name(),
        &multiscreen,
        &dataset,
        steps,
        DEFAULT_BATCH_SIZE,
        DEFAULT_SEQ_LEN,
        &device,
    )?;

    let transformer_config = transformer_config(dataset.vocab_size(), DEFAULT_SEQ_LEN);
    let transformer = TransformerLm::new(transformer_config.clone(), &device)?;
    let transformer_metrics = train_evaluate_model(
        ModelKind::Transformer.name(),
        &transformer,
        &dataset,
        steps,
        DEFAULT_BATCH_SIZE,
        DEFAULT_SEQ_LEN,
        &device,
    )?;

    Ok(vec![multiscreen_metrics, transformer_metrics])
}

fn run_multiscreen_training(
    dataset: &ExamDataset,
    device: &Device,
    steps: usize,
    param_path: &str,
) -> Result<ModelRunMetrics> {
    let config = multiscreen_config(dataset.vocab_size(), DEFAULT_SEQ_LEN);
    let model = MultiscreenLm::new(config.clone(), device)?;
    let final_train_loss = train_model(
        ModelKind::Multiscreen.name(),
        &model,
        dataset,
        steps,
        DEFAULT_BATCH_SIZE,
        config.seq_len,
        device,
        5,
        Some(20),
    )?;

    model.save_parameters(param_path)?;
    println!("saved parameters: {param_path}");

    let inference_model = MultiscreenLm::new(config.clone(), device)?;
    inference_model.load_parameters(param_path)?;
    summarize_saved_model(
        ModelKind::Multiscreen.name(),
        &inference_model,
        dataset,
        final_train_loss,
        DEFAULT_BATCH_SIZE,
        config.seq_len,
        device,
    )
}

fn run_transformer_training(
    dataset: &ExamDataset,
    device: &Device,
    steps: usize,
    param_path: &str,
) -> Result<ModelRunMetrics> {
    let config = transformer_config(dataset.vocab_size(), DEFAULT_SEQ_LEN);
    let model = TransformerLm::new(config.clone(), device)?;
    let final_train_loss = train_model(
        ModelKind::Transformer.name(),
        &model,
        dataset,
        steps,
        DEFAULT_BATCH_SIZE,
        config.seq_len,
        device,
        5,
        Some(20),
    )?;

    model.save_parameters(param_path)?;
    println!("saved parameters: {param_path}");

    let inference_model = TransformerLm::new(config.clone(), device)?;
    inference_model.load_parameters(param_path)?;
    summarize_saved_model(
        ModelKind::Transformer.name(),
        &inference_model,
        dataset,
        final_train_loss,
        DEFAULT_BATCH_SIZE,
        config.seq_len,
        device,
    )
}

fn train_evaluate_model<M: TrainableLanguageModel>(
    name: &'static str,
    model: &M,
    dataset: &ExamDataset,
    steps: usize,
    batch_size: usize,
    seq_len: usize,
    device: &Device,
) -> Result<ModelRunMetrics> {
    let final_train_loss = train_model(
        name, model, dataset, steps, batch_size, seq_len, device, 100, None,
    )?;
    let val_metrics = evaluate_val_metrics(model, dataset, seq_len, device)?;
    let test_metrics = evaluate_test_metrics(model, dataset, seq_len, device)?;
    let benchmark = benchmark_inference(model, dataset, batch_size, seq_len, device, 10)?;

    println!(
        "comparison_model {name} params {} train_loss {:.4} val_loss {:.4} val_acc {:.2}% test_loss {:.4} test_acc {:.2}% avg_ms {:.3} min_ms {:.3} max_ms {:.3}",
        model.parameter_count(),
        final_train_loss,
        val_metrics.loss,
        val_metrics.token_accuracy * 100.0,
        test_metrics.loss,
        test_metrics.token_accuracy * 100.0,
        benchmark.avg_ms,
        benchmark.min_ms,
        benchmark.max_ms
    );

    Ok(ModelRunMetrics {
        name,
        parameters: model.parameter_count(),
        final_train_loss,
        val_loss: val_metrics.loss,
        val_accuracy: val_metrics.token_accuracy,
        test_loss: test_metrics.loss,
        test_accuracy: test_metrics.token_accuracy,
        inference_avg_ms: benchmark.avg_ms,
        inference_min_ms: benchmark.min_ms,
        inference_max_ms: benchmark.max_ms,
    })
}

fn train_model<M: TrainableLanguageModel>(
    name: &'static str,
    model: &M,
    dataset: &ExamDataset,
    steps: usize,
    batch_size: usize,
    seq_len: usize,
    device: &Device,
    log_every: usize,
    val_every: Option<usize>,
) -> Result<f32> {
    let mut optimizer = {
        let params = model.parameters();
        AdamW::new(&params, 1e-3, 0.0)?
    };

    println!("{name}: parameters {}", model.parameter_count());

    if steps == 0 {
        let (inputs, targets) = dataset.train_batch(0, batch_size, seq_len, device)?;
        let loss = cross_entropy_loss(&model.forward(&inputs)?, &targets)?.to_scalar::<f32>()?;
        println!("{name}: step 000 train_loss {loss:.4} (no optimizer step)");
        return Ok(loss);
    }

    let mut final_train_loss = f32::NAN;
    for step in 0..steps {
        let (inputs, targets) = dataset.train_batch(step, batch_size, seq_len, device)?;
        let logits = model.forward(&inputs)?;
        let loss = cross_entropy_loss(&logits, &targets)?;
        final_train_loss = loss.to_scalar::<f32>()?;
        let grads = loss.backward()?;
        let params = model.parameters();
        optimizer.step(&params, &grads)?;

        let should_log = step % log_every == 0 || step + 1 == steps;
        if should_log {
            if let Some(val_every) = val_every {
                if step % val_every == 0 && step > 0 {
                    let val_metrics = evaluate_val_metrics(model, dataset, seq_len, device)?;
                    println!(
                        "{name}: step {step:03} train_loss {final_train_loss:.4} val_loss {:.4} val_acc {:.2}%",
                        val_metrics.loss,
                        val_metrics.token_accuracy * 100.0
                    );
                    continue;
                }
            }
            println!("{name}: step {step:03} train_loss {final_train_loss:.4}");
        }
    }

    Ok(final_train_loss)
}

fn summarize_saved_model<M: TrainableLanguageModel>(
    name: &'static str,
    model: &M,
    dataset: &ExamDataset,
    final_train_loss: f32,
    batch_size: usize,
    seq_len: usize,
    device: &Device,
) -> Result<ModelRunMetrics> {
    let val_metrics = evaluate_val_metrics(model, dataset, seq_len, device)?;
    println!(
        "final_val loss {:.4} token_accuracy {:.2}% ({}/{})",
        val_metrics.loss,
        val_metrics.token_accuracy * 100.0,
        val_metrics.correct_tokens,
        val_metrics.total_tokens
    );

    let test_metrics = evaluate_test_metrics(model, dataset, seq_len, device)?;
    println!(
        "final_test loss {:.4} token_accuracy {:.2}% ({}/{})",
        test_metrics.loss,
        test_metrics.token_accuracy * 100.0,
        test_metrics.correct_tokens,
        test_metrics.total_tokens
    );

    let benchmark = benchmark_inference(model, dataset, batch_size, seq_len, device, 10)?;
    println!(
        "inference_benchmark rounds {} avg_ms {:.3} min_ms {:.3} max_ms {:.3} checksum {:.3}",
        benchmark.rounds, benchmark.avg_ms, benchmark.min_ms, benchmark.max_ms, benchmark.checksum
    );

    Ok(ModelRunMetrics {
        name,
        parameters: model.parameter_count(),
        final_train_loss,
        val_loss: val_metrics.loss,
        val_accuracy: val_metrics.token_accuracy,
        test_loss: test_metrics.loss,
        test_accuracy: test_metrics.token_accuracy,
        inference_avg_ms: benchmark.avg_ms,
        inference_min_ms: benchmark.min_ms,
        inference_max_ms: benchmark.max_ms,
    })
}

fn multiscreen_config(vocab_size: usize, seq_len: usize) -> MultiscreenConfig {
    let mut config = MultiscreenConfig::tiny();
    config.vocab_size = vocab_size;
    config.seq_len = seq_len;
    config
}

fn transformer_config(vocab_size: usize, seq_len: usize) -> TransformerConfig {
    let mut config = TransformerConfig::tiny();
    config.vocab_size = vocab_size;
    config.seq_len = seq_len;
    config
}

fn print_dataset_summary(dataset: &ExamDataset) {
    println!(
        "loaded SAT dataset: {} train examples ({} tokens), {} val examples ({} tokens), {} test examples ({} tokens)",
        dataset.train_examples(),
        dataset.train_tokens(),
        dataset.val_examples(),
        dataset.val_tokens(),
        dataset.test_examples(),
        dataset.test_tokens()
    );
    println!("vocabulary size: {}", dataset.vocab_size());
}
