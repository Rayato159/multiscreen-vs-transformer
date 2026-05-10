use std::env;

use candle_core::Result;
use multiscreen_testing::{
    cli::{CliConfig, run_inference_cli},
    model_kind::ModelKind,
};

#[cfg(windows)]
use winapi::um::wincon::SetConsoleOutputCP;

#[cfg(windows)]
fn init_console_for_unicode() {
    unsafe {
        SetConsoleOutputCP(65001);
    }
}

#[cfg(not(windows))]
fn init_console_for_unicode() {}

fn main() -> Result<()> {
    init_console_for_unicode();

    let args: Vec<String> = env::args().collect();
    let mut config = CliConfig::default();
    let mut i = 1;

    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                if i + 1 < args.len() {
                    let value = &args[i + 1];
                    if let Some(model_kind) = ModelKind::parse(value) {
                        config.model_kind = model_kind;
                    } else {
                        config.param_path = Some(value.clone());
                    }
                    i += 1;
                }
            }
            "--arch" | "--kind" => {
                if i + 1 < args.len() {
                    let value = &args[i + 1];
                    if let Some(model_kind) = ModelKind::parse(value) {
                        config.model_kind = model_kind;
                    } else {
                        eprintln!(
                            "Unknown model architecture: {value}. Expected multiscreen or transformer."
                        );
                        std::process::exit(1);
                    }
                    i += 1;
                }
            }
            "--weights" | "-w" => {
                if i + 1 < args.len() {
                    config.param_path = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--dataset" | "-d" => {
                if i + 1 < args.len() {
                    config.dataset_path = args[i + 1].clone();
                    i += 1;
                }
            }
            "--text" | "-t" => {
                if i + 1 < args.len() {
                    config.text = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--num-predictions" | "-n" => {
                if i + 1 < args.len() {
                    config.num_predictions = args[i + 1].parse().unwrap_or(20);
                    i += 1;
                }
            }
            "--interactive" | "-i" => {
                config.interactive = true;
            }
            "--tokens-only" => {
                config.show_tokens_only = true;
            }
            "--help" | "-h" => {
                print_cli_help();
                return Ok(());
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                print_cli_help();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    run_inference_cli(config)
}

fn print_cli_help() {
    println!("Tiny LM - Inference CLI");
    println!();
    println!("Usage:");
    println!("  cargo run --release --features cuda --bin infer -- [OPTIONS]");
    println!();
    println!("Options:");
    println!(
        "  -m, --model <KIND|PATH>    Model kind (multiscreen/transformer), or legacy checkpoint path"
    );
    println!("      --arch, --kind <KIND>  Model architecture: multiscreen or transformer");
    println!("  -w, --weights <PATH>       Checkpoint path (default depends on model kind)");
    println!("  -d, --dataset <PATH>       Path to Khan Academy parquet dataset");
    println!("  -t, --text <TEXT>          Text prompt to run once");
    println!("  -n, --num-predictions <N>  Number of tokens to generate (default: 20)");
    println!("  -i, --interactive          Chat-style loop");
    println!("      --tokens-only          Print token IDs only in single-prompt mode");
    println!("  -h, --help                 Show this help message");
    println!();
    println!("Examples:");
    println!("  cargo run --release --features cuda --bin infer -- --model multiscreen -i");
    println!("  cargo run --release --features cuda --bin infer -- --model transformer -i");
    println!(
        "  cargo run --release --features cuda --bin infer -- --model multiscreen --text \"Explain fractions\""
    );
}
