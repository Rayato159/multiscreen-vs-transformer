use crate::exam::{BOS_TOKEN, EOS_TOKEN, ExamDataset, UNK_TOKEN};
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
            dataset_path: "exam/sat_world_and_us_history.csv".to_string(),
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

    println!("📥 Loading dataset for vocabulary: {}", config.dataset_path);
    let dataset = ExamDataset::from_file(&config.dataset_path)?;
    let vocab_size = dataset.vocab_size();
    let vocab = dataset.vocab();
    println!("✅ Dataset loaded, vocabulary size: {vocab_size}");
    println!("📦 Loading checkpoint: {param_path}");

    match config.model_kind {
        ModelKind::Multiscreen => {
            let mut model_config = MultiscreenConfig::tiny();
            model_config.vocab_size = vocab_size;
            model_config.seq_len = 96;
            let model = MultiscreenLm::new(model_config, &device)?;
            model.load_parameters(param_path)?;
            println!("✅ Model loaded successfully.");
            run_loaded_model(&model, &device, &config, vocab)?;
        }
        ModelKind::Transformer => {
            let mut model_config = TransformerConfig::tiny();
            model_config.vocab_size = vocab_size;
            model_config.seq_len = 96;
            let model = TransformerLm::new(model_config, &device)?;
            model.load_parameters(param_path)?;
            println!("✅ Model loaded successfully.");
            run_loaded_model(&model, &device, &config, vocab)?;
        }
    }

    Ok(())
}

fn run_loaded_model<M: LanguageModel>(
    model: &M,
    device: &Device,
    config: &CliConfig,
    vocab: &crate::exam::Vocabulary,
) -> Result<()> {
    if config.interactive {
        run_interactive_mode(model, config.num_predictions, device, config, vocab)
    } else if let Some(ref text) = config.text {
        run_single_prediction(model, text, device, config, vocab)
    } else {
        run_interactive_mode(model, config.num_predictions, device, config, vocab)
    }
}

fn run_single_prediction<M: LanguageModel>(
    model: &M,
    text: &str,
    device: &Device,
    _config: &CliConfig,
    vocab: &crate::exam::Vocabulary,
) -> Result<()> {
    println!();
    println!("📝 Input text:");
    println!("\"{}\"", text);
    println!("📏 Text length: {} characters", text.len());
    println!();

    let tokens = vocab.tokenize(text);
    println!("🔤 Tokens: {:?}", tokens);
    println!();

    let seq_len = 96;
    let clipped: Vec<u32> = tokens.iter().take(seq_len).copied().collect();
    let input_tensor = Tensor::new(clipped.as_slice(), device)?;
    let output = model.forward(&input_tensor.unsqueeze(0)?)?;
    println!("🔮 Prediction output shape: {:?}", output.dims());
    println!("💬 Use interactive mode for token-by-token generation.");

    Ok(())
}

fn run_interactive_mode<M: LanguageModel>(
    model: &M,
    num_predictions: usize,
    device: &Device,
    _config: &CliConfig,
    vocab: &crate::exam::Vocabulary,
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

        let tokens = vocab.tokenize(input);
        let seq_len = 96;
        let mut input_tokens: Vec<u32> = tokens.iter().take(seq_len - 1).copied().collect();

        if input_tokens.is_empty() {
            println!("⚠️  No valid tokens found.");
            println!();
            continue;
        }

        let mut generated_tokens = Vec::new();
        let max_gen_length = num_predictions.min(seq_len - input_tokens.len());

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
                .filter(|(token_id, _)| {
                    let tid = *token_id as u32;
                    tid != BOS_TOKEN && tid != UNK_TOKEN && tid != EOS_TOKEN && tid != 0
                })
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
                    if tid != BOS_TOKEN && tid != UNK_TOKEN && tid != EOS_TOKEN && tid != 0 {
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
                        if tid != BOS_TOKEN && tid != UNK_TOKEN && tid != EOS_TOKEN && tid != 0 {
                            return tid;
                        }
                    }
                    UNK_TOKEN
                });

            if step < 3 {
                println!(
                    "  debug step {}: top tokens = {:?}",
                    step,
                    top_3
                        .iter()
                        .map(|(id, prob)| {
                            let token_str = vocab
                                .get_token(*id as u32)
                                .unwrap_or("<?>")
                                .chars()
                                .take(15)
                                .collect::<String>();
                            format!("'{}'({:.2}%)", token_str, prob * 100.0)
                        })
                        .collect::<Vec<_>>()
                );
            }

            generated_tokens.push(best_token);
            input_tokens.push(best_token);

            if input_tokens.len() >= seq_len {
                break;
            }
        }

        let response_words: Vec<String> = generated_tokens
            .iter()
            .filter_map(|&token_id| vocab.get_token(token_id).map(|s| s.to_string()))
            .collect();

        let response = if response_words.is_empty() {
            "(no meaningful response generated)".to_string()
        } else {
            response_words.join(" ")
        };

        println!("🤖 Model: {}", response);
        println!();
    }

    Ok(())
}
