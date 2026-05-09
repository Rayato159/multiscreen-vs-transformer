mod exam;
mod model;
mod optim;
mod runtime;

use candle_core::Result;
use exam::{BYTE_VOCAB_SIZE, ExamDataset, benchmark_inference, evaluate_test_metrics};
use model::{MultiScreenConfig, MultiScreenLm, cross_entropy_loss};
use optim::AdamW;
use runtime::{default_device, device_label};

fn main() -> Result<()> {
    let device = default_device()?;
    let dataset = ExamDataset::from_file("exam/tcas68-all.dataset.txt")?;

    let mut config = MultiScreenConfig::tiny();
    config.vocab_size = BYTE_VOCAB_SIZE;
    config.seq_len = 96;

    let model = MultiScreenLm::new(config.clone(), &device)?;
    let mut optimizer = {
        let params = model.parameters();
        AdamW::new(&params, 1e-3, 0.0)?
    };

    let batch_size = 4;
    let steps = 80;

    println!("device: {}", device_label(&device));
    println!(
        "loaded exam dataset: {} train examples ({} bytes), {} test examples ({} bytes)",
        dataset.train_examples(),
        dataset.train_tokens(),
        dataset.test_examples(),
        dataset.test_tokens()
    );

    for step in 0..steps {
        let (inputs, targets) = dataset.train_batch(step, batch_size, config.seq_len, &device)?;
        let logits = model.forward(&inputs)?;
        let loss = cross_entropy_loss(&logits, &targets)?;
        let loss_value = loss.to_scalar::<f32>()?;
        let grads = loss.backward()?;
        let params = model.parameters();
        optimizer.step(&params, &grads)?;

        if step % 5 == 0 || step + 1 == steps {
            println!("step {step:03} train_loss {loss_value:.4}");
        }
    }

    let param_path = "models/tcas68_all_multiscreen.params";
    model.save_parameters(param_path)?;
    println!("saved parameters: {param_path}");

    let inference_model = MultiScreenLm::new(config.clone(), &device)?;
    inference_model.load_parameters(param_path)?;

    let final_metrics = evaluate_test_metrics(&inference_model, &dataset, config.seq_len, &device)?;
    println!(
        "final_test loss {:.4} token_accuracy {:.2}% ({}/{})",
        final_metrics.loss,
        final_metrics.token_accuracy * 100.0,
        final_metrics.correct_tokens,
        final_metrics.total_tokens
    );

    let benchmark = benchmark_inference(
        &inference_model,
        &dataset,
        batch_size,
        config.seq_len,
        &device,
        10,
    )?;
    println!(
        "inference_benchmark rounds {} avg_ms {:.3} min_ms {:.3} max_ms {:.3} checksum {:.3}",
        benchmark.rounds, benchmark.avg_ms, benchmark.min_ms, benchmark.max_ms, benchmark.checksum
    );

    Ok(())
}
