pub mod cli;
pub mod exam;
pub mod model;
pub mod optim;
pub mod runtime;

use candle_core::Result;
use exam::{ExamDataset, benchmark_inference, evaluate_test_metrics, evaluate_val_metrics};
use model::{MultiScreenConfig, MultiScreenLm, cross_entropy_loss};
use optim::AdamW;
use runtime::{default_device, device_label};

/// Library function for training that can be used by the train binary
pub fn run_training() -> Result<()> {
    let device = default_device()?;
    let dataset = ExamDataset::from_file("exam/sat_world_and_us_history.csv")?;

    let mut config = MultiScreenConfig::tiny();
    config.vocab_size = dataset.vocab_size();
    config.seq_len = 96;

    println!("device: {}", device_label(&device));
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

    let model = MultiScreenLm::new(config.clone(), &device)?;
    let mut optimizer = {
        let params = model.parameters();
        AdamW::new(&params, 1e-3, 0.0)?
    };

    let batch_size = 4;
    let steps = 1000;

    for step in 0..steps {
        let (inputs, targets) = dataset.train_batch(step, batch_size, config.seq_len, &device)?;
        let logits = model.forward(&inputs)?;
        let loss = cross_entropy_loss(&logits, &targets)?;
        let loss_value = loss.to_scalar::<f32>()?;
        let grads = loss.backward()?;
        let params = model.parameters();
        optimizer.step(&params, &grads)?;

        if step % 5 == 0 || step + 1 == steps {
            // Evaluate on validation set periodically
            if step % 20 == 0 && step > 0 {
                let val_metrics = evaluate_val_metrics(&model, &dataset, config.seq_len, &device)?;
                println!(
                    "step {step:03} train_loss {loss_value:.4} val_loss {:.4} val_acc {:.2}%",
                    val_metrics.loss,
                    val_metrics.token_accuracy * 100.0
                );
            } else {
                println!("step {step:03} train_loss {loss_value:.4}");
            }
        }
    }

    let param_path = "models/sat_multiscreen.params";
    model.save_parameters(param_path)?;
    println!("saved parameters: {param_path}");

    let inference_model = MultiScreenLm::new(config.clone(), &device)?;
    inference_model.load_parameters(param_path)?;

    // Final evaluation on validation set
    let val_metrics = evaluate_val_metrics(&inference_model, &dataset, config.seq_len, &device)?;
    println!(
        "final_val loss {:.4} token_accuracy {:.2}% ({}/{})",
        val_metrics.loss,
        val_metrics.token_accuracy * 100.0,
        val_metrics.correct_tokens,
        val_metrics.total_tokens
    );

    // Final evaluation on test set
    let test_metrics = evaluate_test_metrics(&inference_model, &dataset, config.seq_len, &device)?;
    println!(
        "final_test loss {:.4} token_accuracy {:.2}% ({}/{})",
        test_metrics.loss,
        test_metrics.token_accuracy * 100.0,
        test_metrics.correct_tokens,
        test_metrics.total_tokens
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
