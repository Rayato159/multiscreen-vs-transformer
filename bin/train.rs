use std::env;

use multiscreen_testing::{TrainingConfig, model_kind::ModelKind, run_training_with_config};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    run_training_with_config(config)?;
    Ok(())
}

fn parse_args() -> Result<TrainingConfig, Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let mut config = TrainingConfig::default();
    let mut i = 1;

    while i < args.len() {
        match args[i].as_str() {
            "--model" | "--arch" => {
                let value = next_arg(&args, &mut i, "--model")?;
                config.model_kind = ModelKind::parse(value).ok_or_else(|| {
                    format!("unknown model '{value}', expected multiscreen or transformer")
                })?;
            }
            "--steps" | "-s" => {
                let value = next_arg(&args, &mut i, "--steps")?;
                config.steps = value.parse()?;
            }
            "--weights" | "-w" => {
                config.param_path = Some(next_arg(&args, &mut i, "--weights")?.to_string());
            }
            "--dataset" | "-d" => {
                config.dataset_path = next_arg(&args, &mut i, "--dataset")?.to_string();
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                return Err(format!("unknown argument: {other}").into());
            }
        }
        i += 1;
    }

    Ok(config)
}

fn next_arg<'a>(
    args: &'a [String],
    index: &mut usize,
    flag: &str,
) -> Result<&'a str, Box<dyn std::error::Error>> {
    if *index + 1 >= args.len() {
        return Err(format!("{flag} expects a value").into());
    }
    *index += 1;
    Ok(&args[*index])
}

fn print_help() {
    println!("Train a tiny causal LM");
    println!();
    println!("Usage:");
    println!("  cargo run --release --features cuda --bin train -- [OPTIONS]");
    println!();
    println!("Options:");
    println!(
        "  --model <KIND>       Model architecture: multiscreen or transformer (default: multiscreen)"
    );
    println!("  --steps, -s <N>      Number of training steps (default: 1000)");
    println!("  --weights, -w <PATH> Checkpoint output path (default depends on model)");
    println!("  --dataset, -d <PATH> Khan Academy parquet dataset path");
    println!("  --help, -h           Show this help");
    println!();
    println!("Examples:");
    println!("  cargo run --release --features cuda --bin train -- --model multiscreen");
    println!("  cargo run --release --features cuda --bin train -- --model transformer");
}
