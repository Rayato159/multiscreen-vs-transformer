use crate::dataset::{DEFAULT_TOKENIZER_PATH, HfTokenizer};
use crate::lm::LanguageModel;
use crate::model_kind::ModelKind;
use crate::multiscreen::{MultiscreenConfig, MultiscreenLm};
use crate::runtime::{default_device, device_label};
use crate::transformer::{TransformerConfig, TransformerLm};
use candle_core::{Device, IndexOp, Result, Tensor};
use candle_nn as nn;

pub struct CliConfig {
    pub model_kind: ModelKind,
    pub param_path: Option<String>,
    pub dataset_path: String,
    pub text: Option<String>,
    pub num_predictions: usize,
    pub interactive: bool,
    pub show_tokens_only: bool,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            model_kind: ModelKind::Multiscreen,
            param_path: None,
            dataset_path: crate::DEFAULT_DATASET_PATH.to_string(),
            text: None,
            num_predictions: 20,
            interactive: false,
            show_tokens_only: false,
        }
    }
}

pub fn run_inference_cli(config: CliConfig) -> Result<()> {
    let device = default_device()?;
    let param_path = config
        .param_path
        .as_deref()
        .unwrap_or_else(|| config.model_kind.default_param_path());

    println!("🚀 Tiny LM - Inference CLI");
    println!("🧠 Model: {}", config.model_kind.display_name());
    println!("🖥️  Device: {}", device_label(&device));
    println!();

    println!("🔤 Loading Hugging Face tokenizer: {DEFAULT_TOKENIZER_PATH}");
    let tokenizer = HfTokenizer::load_or_train(&config.dataset_path, DEFAULT_TOKENIZER_PATH)?;
    let vocab_size = tokenizer.vocab_size();
    println!("✅ Tokenizer ready, vocabulary size: {vocab_size}");
    println!("📦 Loading checkpoint: {param_path}");

    match config.model_kind {
        ModelKind::Multiscreen => {
            let mut model_config = MultiscreenConfig::tiny();
            model_config.vocab_size = vocab_size;
            model_config.seq_len = crate::DEFAULT_SEQ_LEN;
            let model = MultiscreenLm::new(model_config, &device)?;
            model.load_parameters(param_path)?;
            println!("✅ Model loaded successfully.");
            run_loaded_model(&model, &device, &config, &tokenizer)?;
        }
        ModelKind::Transformer => {
            let mut model_config = TransformerConfig::tiny();
            model_config.vocab_size = vocab_size;
            model_config.seq_len = crate::DEFAULT_SEQ_LEN;
            let model = TransformerLm::new(model_config, &device)?;
            model.load_parameters(param_path)?;
            println!("✅ Model loaded successfully.");
            run_loaded_model(&model, &device, &config, &tokenizer)?;
        }
    }

    Ok(())
}

fn run_loaded_model<M: LanguageModel>(
    model: &M,
    device: &Device,
    config: &CliConfig,
    tokenizer: &HfTokenizer,
) -> Result<()> {
    if config.interactive {
        run_interactive_mode(model, config.num_predictions, device, tokenizer)
    } else if let Some(ref text) = config.text {
        run_single_prediction(model, text, device, config, tokenizer)
    } else {
        run_interactive_mode(model, config.num_predictions, device, tokenizer)
    }
}

fn run_single_prediction<M: LanguageModel>(
    model: &M,
    text: &str,
    device: &Device,
    config: &CliConfig,
    tokenizer: &HfTokenizer,
) -> Result<()> {
    println!();
    println!("📝 Input text:");
    println!("\"{}\"", text);
    println!("📏 Text length: {} characters", text.len());
    println!();

    let tokens = tokenizer.encode_prompt(text)?;
    println!("🔢 Token IDs: {:?}", tokens);
    if !config.show_tokens_only {
        println!("🔎 Decoded prompt: {}", tokenizer.decode(&tokens)?);
    }
    println!();

    let clipped: Vec<u32> = tokens
        .iter()
        .take(crate::DEFAULT_SEQ_LEN)
        .copied()
        .collect();
    let input_tensor = Tensor::new(clipped.as_slice(), device)?;
    let output = model.forward(&input_tensor.unsqueeze(0)?)?;
    println!("📐 Prediction output shape: {:?}", output.dims());
    println!("💬 Use interactive mode for token-by-token generation.");

    Ok(())
}

fn run_interactive_mode<M: LanguageModel>(
    model: &M,
    num_predictions: usize,
    device: &Device,
    tokenizer: &HfTokenizer,
) -> Result<()> {
    use std::io::{self, Write};

    println!();
    println!("💬 Interactive mode");
    println!("Type text and press Enter to generate. Type 'quit' or 'exit' to leave.");
    println!();

    loop {
        print!("🫵 You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
            println!("👋 Goodbye.");
            break;
        }

        if input.is_empty() {
            continue;
        }

        let tokens = tokenizer.encode_prompt(input)?;
        let mut input_tokens: Vec<u32> = tokens
            .iter()
            .take(crate::DEFAULT_SEQ_LEN - 1)
            .copied()
            .collect();

        if input_tokens.is_empty() {
            println!("⚠️  No valid tokens found.");
            println!();
            continue;
        }

        let mut generated_tokens = Vec::new();
        let max_gen_length = num_predictions.min(crate::DEFAULT_SEQ_LEN - input_tokens.len());

        for step in 0..max_gen_length {
            let input_tensor = Tensor::new(input_tokens.as_slice(), device)?;
            let input_batch = input_tensor.unsqueeze(0)?;
            let output = model.forward(&input_batch)?;

            let logits = output.i((0, input_tokens.len() - 1, ..))?;
            let probs = nn::ops::softmax(&logits, 0)?;
            let probs_vec = probs.to_vec1::<f32>()?;
            let mut top_3: Vec<(usize, f32)> =
                probs_vec.iter().enumerate().map(|(i, &p)| (i, p)).collect();
            top_3.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            top_3.truncate(3);

            let temperature: f32 = 0.8;
            let mut sampled_probs: Vec<(usize, f32)> = probs_vec
                .iter()
                .enumerate()
                .filter(|(token_id, _)| !tokenizer.is_special_id(*token_id as u32))
                .map(|(i, &p)| (i, p.powf(1.0 / temperature)))
                .collect();

            let total: f32 = sampled_probs.iter().map(|(_, p)| p).sum();
            if total > 0.0 {
                for (_, p) in sampled_probs.iter_mut() {
                    *p /= total;
                }
            } else {
                for (token_id, _) in top_3.iter() {
                    let tid = *token_id as u32;
                    if !tokenizer.is_special_id(tid) {
                        sampled_probs = vec![(*token_id, 1.0)];
                        break;
                    }
                }
            }

            let random_val: f32 = rand::random::<f32>();
            let mut cumulative = 0.0;
            let best_token = sampled_probs
                .iter()
                .find(|(_, p)| {
                    cumulative += p;
                    random_val <= cumulative
                })
                .map(|(i, _)| *i as u32)
                .unwrap_or_else(|| {
                    for (i, _) in probs_vec.iter().enumerate() {
                        let tid = i as u32;
                        if !tokenizer.is_special_id(tid) {
                            return tid;
                        }
                    }
                    0
                });

            if step < 3 {
                let mut sampled_top = sampled_probs.clone();
                sampled_top
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                sampled_top.truncate(3);
                println!(
                    "  🔍 debug step {}: raw top tokens = {:?}",
                    step,
                    format_token_probs(&top_3, tokenizer)
                );
                println!(
                    "  🎲 debug step {}: sample candidates = {:?}",
                    step,
                    format_token_probs(&sampled_top, tokenizer)
                );
            }

            generated_tokens.push(best_token);
            input_tokens.push(best_token);

            if input_tokens.len() >= crate::DEFAULT_SEQ_LEN {
                break;
            }
        }

        let response = tokenizer.decode(&generated_tokens)?;
        let response = if response.trim().is_empty() {
            "(no meaningful response generated)".to_string()
        } else {
            response
        };

        println!("🤖 Model: {}", response);
        println!();
    }

    Ok(())
}

fn format_token_probs(items: &[(usize, f32)], tokenizer: &HfTokenizer) -> Vec<String> {
    items
        .iter()
        .map(|(id, prob)| {
            let token_str = tokenizer
                .display_token(*id as u32)
                .chars()
                .take(15)
                .collect::<String>();
            format!("'{}'({:.2}%)", token_str, prob * 100.0)
        })
        .collect()
}
