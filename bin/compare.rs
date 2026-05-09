use multiscreen_testing::run_comparison;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let steps = parse_steps();
    let metrics = run_comparison(steps)?;

    println!();
    println!("summary:");
    for metric in metrics {
        println!(
            "{} params={} train_loss={:.4} val_loss={:.4} val_acc={:.2}% test_loss={:.4} test_acc={:.2}% avg_ms={:.3} min_ms={:.3} max_ms={:.3}",
            metric.name,
            metric.parameters,
            metric.final_train_loss,
            metric.val_loss,
            metric.val_accuracy * 100.0,
            metric.test_loss,
            metric.test_accuracy * 100.0,
            metric.inference_avg_ms,
            metric.inference_min_ms,
            metric.inference_max_ms
        );
    }

    Ok(())
}

fn parse_steps() -> usize {
    let args = env::args().collect::<Vec<_>>();
    let mut idx = 1;
    while idx < args.len() {
        if args[idx] == "--steps" && idx + 1 < args.len() {
            return args[idx + 1].parse().unwrap_or(1000);
        }
        idx += 1;
    }
    1000
}
