use crate::exam::{BOS_TOKEN, EOS_TOKEN, ExamDataset, UNK_TOKEN};
use crate::model::{MultiScreenConfig, MultiScreenLm};
use crate::runtime::{default_device, device_label};
use candle_core::{Device, IndexOp, Result, Tensor};
use candle_nn as nn;

pub struct CliConfig {
    pub param_path: String,
    pub dataset_path: String,
    pub text: Option<String>,
    pub num_predictions: usize,
    pub interactive: bool,
    pub show_tokens_only: bool,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            param_path: "models/sat_multiscreen.params".to_string(),
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
    println!("🚀 Tiny Multiscreen LM - Inference CLI");
    println!("📊 Device: {}", device_label(&device));
    println!();

    // Load dataset to get vocabulary
    println!("📥 Loading dataset for vocabulary: {}", config.dataset_path);
    let dataset = ExamDataset::from_file(&config.dataset_path)?;
    let vocab_size = dataset.vocab_size();
    let vocab = dataset.vocab();
    println!("✅ Dataset loaded, vocabulary size: {}", vocab_size);
    println!();

    // Load model configuration
    let mut model_config = MultiScreenConfig::tiny();
    model_config.vocab_size = vocab_size;
    model_config.seq_len = 96;

    // Load model
    println!("📥 Loading model from: {}", config.param_path);
    let model = MultiScreenLm::new(model_config.clone(), &device)?;
    model.load_parameters(&config.param_path)?;
    println!("✅ Model loaded successfully!");
    println!();

    if config.interactive {
        run_interactive_mode(&model, config.num_predictions, &device, &config, vocab)?;
    } else if let Some(ref text) = config.text {
        run_single_prediction(&model, text, &device, &config, vocab)?;
    } else {
        run_interactive_mode(&model, config.num_predictions, &device, &config, vocab)?;
    }

    Ok(())
}

fn run_single_prediction(
    model: &MultiScreenLm,
    text: &str,
    device: &Device,
    _config: &CliConfig,
    vocab: &crate::exam::Vocabulary,
) -> Result<()> {
    println!("📝 Input text:");
    println!("\"{}\"", text);
    println!("📏 Text length: {} characters", text.len());
    println!();

    // Tokenize the input text
    let tokens = vocab.tokenize(text);
    println!("🔤 Tokens: {:?}", tokens);
    println!();

    // Convert tokens to tensor
    let input_tensor = Tensor::new(&tokens[..1.min(tokens.len())], device)?;
    println!("🔮 Generating predictions...");
    println!();

    // Run inference (just for now, we'll improve this later)
    let _output = model.forward(&input_tensor.unsqueeze(0)?)?;
    println!("⚠️  Prediction output shape: {:?}", _output.dims());
    println!("⚠️  Full token-by-token generation needs to be implemented.");
    println!("   For now, use interactive mode to see the model working.");

    Ok(())
}

fn run_interactive_mode(
    model: &MultiScreenLm,
    num_predictions: usize,
    device: &Device,
    _config: &CliConfig,
    vocab: &crate::exam::Vocabulary,
) -> Result<()> {
    use std::io::{self, Write};

    println!("🎮 Interactive Mode");
    println!("   Type your text and press Enter to generate responses.");
    println!("   Type 'quit' or 'exit' to leave.");
    println!();

    loop {
        print!("💬 You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        // Check for quit commands
        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
            println!("👋 Goodbye!");
            break;
        }

        if input.is_empty() {
            continue;
        }

        println!();

        // Tokenize the input
        let tokens = vocab.tokenize(input);

        // Take first few tokens to fit in sequence length
        let seq_len = 96;
        let mut input_tokens: Vec<u32> = tokens.iter().take(seq_len - 1).copied().collect();

        if input_tokens.is_empty() {
            println!("⚠️  No valid tokens found.");
            println!();
            continue;
        }

        // Generate response by predicting next words
        let mut generated_tokens = Vec::new();
        let max_gen_length = num_predictions.min(seq_len - input_tokens.len());

        for step in 0..max_gen_length {
            // Convert to tensor
            let input_tensor = Tensor::new(&input_tokens[..], device)?;
            let input_batch = input_tensor.unsqueeze(0)?; // Add batch dimension

            // Run forward pass
            let output = model.forward(&input_batch)?;

            // Get predictions for the last position (next token prediction)
            let logits = output.i((0, input_tokens.len() - 1, ..))?; // [vocab_size]
            let probs = nn::ops::softmax(&logits, 0)?;

            // Get top 3 predictions for debugging
            let probs_vec = probs.to_vec1::<f32>()?;
            let mut top_3: Vec<(usize, f32)> =
                probs_vec.iter().enumerate().map(|(i, &p)| (i, p)).collect();
            top_3.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            top_3.truncate(3);

            // Use temperature sampling instead of greedy decoding
            // Temperature 0.8 makes the model more creative
            let temperature: f32 = 0.8;
            let mut sampled_probs: Vec<(usize, f32)> = probs_vec
                .iter()
                .enumerate()
                .filter(|(token_id, _)| {
                    // Filter out special tokens (except PAD which is 0)
                    let tid = *token_id as u32;
                    tid != BOS_TOKEN && tid != UNK_TOKEN && tid != EOS_TOKEN && tid != 0
                })
                .map(|(i, &p)| (i, p.powf(1.0 / temperature)))
                .collect();

            // Normalize after filtering
            let total: f32 = sampled_probs.iter().map(|(_, p)| p).sum();
            if total > 0.0 {
                for (_, p) in sampled_probs.iter_mut() {
                    *p /= total;
                }
            } else {
                // If all tokens were filtered out, fall back to top valid token
                // Find first non-special token in top_3
                for (token_id, _) in top_3.iter() {
                    let tid = *token_id as u32;
                    if tid != BOS_TOKEN && tid != UNK_TOKEN && tid != EOS_TOKEN && tid != 0 {
                        sampled_probs = vec![(*token_id, 1.0)];
                        break;
                    }
                }
            }

            // Sample from distribution
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
                    // Fallback: get first non-special token from full distribution
                    for (i, _) in probs_vec.iter().enumerate() {
                        let tid = i as u32;
                        if tid != BOS_TOKEN && tid != UNK_TOKEN && tid != EOS_TOKEN && tid != 0 {
                            return tid;
                        }
                    }
                    UNK_TOKEN // Should never reach here if vocab is valid
                });

            // Debug output for first few steps
            if step < 3 {
                println!(
                    "  Debug step {}: Top tokens = {:?}",
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

            // Note: Special tokens are already filtered out during sampling,
            // so we don't need to check for them here anymore.

            generated_tokens.push(best_token);
            input_tokens.push(best_token);

            // Stop if sequence is getting too long
            if input_tokens.len() >= seq_len {
                break;
            }
        }

        // Detokenize the response
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
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();
    }

    Ok(())
}

// Note: The generate_predictions and get_top_candidates functions need to be
// updated to work with word-level tokenization. They currently expect byte-level
// input but the model now uses word-level tokens with a larger vocabulary.
//
// To fully implement inference, we need to:
// 1. Load the vocabulary from the dataset
// 2. Tokenize input text using the same rules as training
// 3. Generate predictions in token space
// 4. Detokenize predictions back to text
//
// For now, the training script provides full evaluation metrics.
