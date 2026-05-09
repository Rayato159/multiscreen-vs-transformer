use std::env;

use candle_core::Result;

#[cfg(windows)]
use winapi::um::wincon::SetConsoleOutputCP;

/// Initialize console for Unicode/Thai support on Windows
#[cfg(windows)]
fn init_console_for_unicode() {
    unsafe {
        // Set console output code page to UTF-8 (65001)
        SetConsoleOutputCP(65001);
    }
}

/// Initialize console for Unicode/Thai support on non-Windows platforms
#[cfg(not(windows))]
fn init_console_for_unicode() {
    // On Unix-like systems, UTF-8 is usually the default
    // Nothing special needed
}

/// Initialize console at program start
pub fn init_console() {
    init_console_for_unicode();
}
use multiscreen_testing::cli::{CliConfig, run_inference_cli};

fn main() -> Result<()> {
    // Initialize console for Unicode/Thai support
    init_console();

    let args: Vec<String> = env::args().collect();

    let mut config = CliConfig::default();

    // Parse CLI arguments (skip the first arg which is the program name)
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                if i + 1 < args.len() {
                    config.param_path = args[i + 1].clone();
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
                    config.num_predictions = args[i + 1].parse().unwrap_or(10);
                    i += 1;
                }
            }
            "--interactive" | "-i" => {
                config.interactive = true;
            }
            "--tokens-only" | "-b" => {
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
    println!("Tiny Multiscreen LM - Inference CLI (Word-level Tokenization)");
    println!();
    println!("Usage:");
    println!("  cargo run --bin infer -- [OPTIONS]");
    println!();
    println!("Options:");
    println!(
        "  -m, --model <PATH>          Path to model parameters (default: models/sat_multiscreen.params)"
    );
    println!(
        "  -d, --dataset <PATH>        Path to dataset CSV for vocabulary (default: exam/sat_world_and_us_history.csv)"
    );
    println!(
        "  -t, --text <TEXT>          Text to predict on (if not provided, runs in interactive mode)"
    );
    println!("  -n, --num-predictions <N>  Number of words to generate (default: 20)");
    println!("  -i, --interactive          Interactive mode (chat with the model)");

    println!("  -h, --help                 Show this help message");
    println!();
    println!("Note: This version uses word-level tokenization.");
    println!("      Interactive mode allows you to chat with the model.");
    println!("      The model generates complete sentences as responses.");
    println!();
    println!("Examples:");
    println!("  cargo run --bin infer -- --help");
    println!("  cargo run --bin infer -- -m custom.params");
    println!();
    println!("To train and evaluate the model:");
    println!("  cargo run --release --features cuda --bin train");
    println!();
    println!("With CUDA:");
    println!("  cargo run --bin infer --features cuda -- --help");
}
