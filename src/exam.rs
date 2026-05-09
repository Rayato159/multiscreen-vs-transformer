use crate::model::{MultiScreenLm, cross_entropy_loss};
use candle_core::{Device, Result, Tensor, bail};
use std::collections::HashMap;
use std::io::BufReader;
use std::{
    fs,
    path::Path,
    time::{Duration, Instant},
};

// Word-level tokenization constants
pub const SPECIAL_TOKENS: &[&str] = &["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"];
pub const UNK_TOKEN: u32 = 1;
pub const BOS_TOKEN: u32 = 2;
pub const EOS_TOKEN: u32 = 3;

pub struct Vocabulary {
    token_to_id: HashMap<String, u32>,
    id_to_token: Vec<String>,
}

impl Vocabulary {
    pub fn new() -> Self {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = Vec::new();

        // Add special tokens
        for (idx, token) in SPECIAL_TOKENS.iter().enumerate() {
            token_to_id.insert(token.to_string(), idx as u32);
            id_to_token.push(token.to_string());
        }

        Self {
            token_to_id,
            id_to_token,
        }
    }

    pub fn add_token(&mut self, token: &str) -> u32 {
        if let Some(&id) = self.token_to_id.get(token) {
            return id;
        }

        let id = self.id_to_token.len() as u32;
        self.token_to_id.insert(token.to_string(), id);
        self.id_to_token.push(token.to_string());
        id
    }

    pub fn get_id(&self, token: &str) -> u32 {
        self.token_to_id.get(token).copied().unwrap_or(UNK_TOKEN)
    }

    pub fn get_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(|s| s.as_str())
    }

    pub fn size(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        tokens.push(BOS_TOKEN);

        // Split on whitespace and preserve punctuation
        for word in text.split_whitespace() {
            // Split on common punctuation marks
            let parts: Vec<&str> = word
                .split(|c: char| {
                    c == '.'
                        || c == ','
                        || c == '!'
                        || c == '?'
                        || c == ':'
                        || c == ';'
                        || c == '('
                        || c == ')'
                        || c == '['
                        || c == ']'
                        || c == '"'
                        || c == '\''
                })
                .filter(|s| !s.is_empty())
                .collect();

            for part in parts {
                tokens.push(self.get_id(part));
            }
        }

        tokens.push(EOS_TOKEN);
        tokens
    }
}

pub struct ExamDataset {
    train: Vec<u32>,
    val: Vec<u32>,
    test: Vec<u32>,
    train_examples: usize,
    val_examples: usize,
    test_examples: usize,
    vocab: Vocabulary,
}

pub struct EvalMetrics {
    pub loss: f32,
    pub token_accuracy: f32,
    pub correct_tokens: usize,
    pub total_tokens: usize,
}

pub struct InferenceBenchmark {
    pub rounds: usize,
    pub avg_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub checksum: f64,
}

impl ExamDataset {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = fs::File::open(path).map_err(|e| {
            candle_core::Error::Msg(format!("failed to open CSV file {}: {}", path.display(), e))
        })?;

        let reader = BufReader::new(file);
        let mut csv_reader = csv::Reader::from_reader(reader);

        // Get headers and find column indices
        let headers = csv_reader
            .headers()
            .map_err(|e| candle_core::Error::Msg(format!("failed to read CSV headers: {}", e)))?;

        let subject_idx = headers.iter().position(|s| s == "subject");
        let prompt_idx = headers.iter().position(|s| s == "prompt");
        let a_idx = headers.iter().position(|s| s == "A");
        let b_idx = headers.iter().position(|s| s == "B");
        let c_idx = headers.iter().position(|s| s == "C");
        let d_idx = headers.iter().position(|s| s == "D");
        let e_idx = headers.iter().position(|s| s == "E");
        let answer_idx = headers.iter().position(|s| s == "answer");

        let mut vocab = Vocabulary::new();
        let mut all_records: Vec<(
            String,
            String,
            String,
            String,
            String,
            String,
            String,
            String,
        )> = Vec::new();

        // First pass: collect all records and build vocabulary
        for result in csv_reader.records() {
            let record = result.map_err(|e| {
                candle_core::Error::Msg(format!("failed to read CSV record: {}", e))
            })?;

            let subject = subject_idx.and_then(|i| record.get(i)).unwrap_or("");
            let prompt = prompt_idx.and_then(|i| record.get(i)).unwrap_or("");
            let a = a_idx.and_then(|i| record.get(i)).unwrap_or("");
            let b = b_idx.and_then(|i| record.get(i)).unwrap_or("");
            let c = c_idx.and_then(|i| record.get(i)).unwrap_or("");
            let d = d_idx.and_then(|i| record.get(i)).unwrap_or("");
            let e = e_idx.and_then(|i| record.get(i)).unwrap_or("");
            let answer = answer_idx.and_then(|i| record.get(i)).unwrap_or("");

            all_records.push((
                subject.to_string(),
                prompt.to_string(),
                a.to_string(),
                b.to_string(),
                c.to_string(),
                d.to_string(),
                e.to_string(),
                answer.to_string(),
            ));

            // Build vocabulary from all text fields
            for text in &[subject, prompt, a, b, c, d, e, answer] {
                for word in text.split_whitespace() {
                    vocab.add_token(word);
                }
            }
        }

        // Second pass: tokenize and split into train/val/test (70/15/15)
        let mut train = Vec::new();
        let mut val = Vec::new();
        let mut test_split = Vec::new();
        let mut train_examples = 0;
        let mut val_examples = 0;
        let mut test_examples = 0;

        for (idx, (subject, prompt, a, b, c, d, e, answer)) in all_records.iter().enumerate() {
            // Create formatted record
            let formatted_record = format!(
                "Subject: {}\nQuestion: {}\nA. {}\nB. {}\nC. {}\nD. {}\nE. {}\nAnswer: {}",
                subject, prompt, a, b, c, d, e, answer
            );

            let tokens = vocab.tokenize(&formatted_record);

            // Split: 70% train, 15% val, 15% test
            if idx < all_records.len() * 70 / 100 {
                train.extend(tokens.iter());
                train_examples += 1;
            } else if idx < all_records.len() * 85 / 100 {
                val.extend(tokens.iter());
                val_examples += 1;
            } else {
                test_split.extend(tokens.iter());
                test_examples += 1;
            }
        }

        if train_examples == 0 || val_examples == 0 || test_examples == 0 {
            bail!("CSV dataset must contain enough examples for train/val/test splits");
        }
        if train.len() < 2 || val.len() < 2 || test_split.len() < 2 {
            bail!("CSV dataset split is too small for next-token language modeling");
        }

        Ok(Self {
            train,
            val,
            test: test_split,
            train_examples,
            val_examples,
            test_examples,
            vocab,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.size()
    }

    pub fn vocab(&self) -> &Vocabulary {
        &self.vocab
    }

    pub fn train_examples(&self) -> usize {
        self.train_examples
    }

    pub fn val_examples(&self) -> usize {
        self.val_examples
    }

    pub fn test_examples(&self) -> usize {
        self.test_examples
    }

    pub fn train_tokens(&self) -> usize {
        self.train.len()
    }

    pub fn val_tokens(&self) -> usize {
        self.val.len()
    }

    pub fn test_tokens(&self) -> usize {
        self.test.len()
    }

    pub fn train_batch(
        &self,
        step: usize,
        batch_size: usize,
        seq_len: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        make_token_lm_batch(&self.train, step, batch_size, seq_len, device)
    }

    pub fn val_batch(
        &self,
        step: usize,
        batch_size: usize,
        seq_len: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        make_token_lm_batch(&self.val, step, batch_size, seq_len, device)
    }

    pub fn test_batch(
        &self,
        step: usize,
        batch_size: usize,
        seq_len: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        make_token_lm_batch(&self.test, step, batch_size, seq_len, device)
    }
}

pub fn evaluate_test_metrics(
    model: &MultiScreenLm,
    dataset: &ExamDataset,
    seq_len: usize,
    device: &Device,
) -> Result<EvalMetrics> {
    evaluate_tokens(model, &dataset.test, seq_len, device)
}

pub fn evaluate_val_metrics(
    model: &MultiScreenLm,
    dataset: &ExamDataset,
    seq_len: usize,
    device: &Device,
) -> Result<EvalMetrics> {
    evaluate_tokens(model, &dataset.val, seq_len, device)
}

pub fn benchmark_inference(
    model: &MultiScreenLm,
    dataset: &ExamDataset,
    batch_size: usize,
    seq_len: usize,
    device: &Device,
    rounds: usize,
) -> Result<InferenceBenchmark> {
    let mut durations = Vec::with_capacity(rounds);
    let mut checksum = 0.0;

    for round in 0..rounds {
        let step = 113 + round * 37;
        let (inputs, _) = if round % 3 == 0 {
            dataset.train_batch(step, batch_size, seq_len, device)?
        } else if round % 3 == 1 {
            dataset.val_batch(step, batch_size, seq_len, device)?
        } else {
            dataset.test_batch(step, batch_size, seq_len, device)?
        };

        device.synchronize()?;
        let started = Instant::now();
        let logits = model.forward(&inputs)?;
        checksum += logits.sum_all()?.to_scalar::<f32>()? as f64;
        durations.push(started.elapsed());
    }

    Ok(summarize_benchmark(durations, checksum))
}

fn evaluate_tokens(
    model: &MultiScreenLm,
    data: &[u32],
    seq_len: usize,
    device: &Device,
) -> Result<EvalMetrics> {
    if data.len() < 2 {
        bail!("need at least two tokens to evaluate split");
    }

    let mut weighted_loss = 0.0;
    let mut correct_tokens = 0;
    let mut total_tokens = 0;
    let mut offset = 0;

    while offset + 1 < data.len() {
        let len = seq_len.min(data.len() - 1 - offset);
        let inputs = Tensor::from_vec(data[offset..offset + len].to_vec(), (1, len), device)?;
        let targets = Tensor::from_vec(
            data[offset + 1..offset + len + 1].to_vec(),
            (1, len),
            device,
        )?;
        let logits = model.forward(&inputs)?;
        let loss = cross_entropy_loss(&logits, &targets)?.to_scalar::<f32>()?;
        let (correct, total) = token_accuracy_counts(&logits, &targets)?;
        weighted_loss += loss * total as f32;
        correct_tokens += correct;
        total_tokens += total;
        offset += len;
    }

    Ok(EvalMetrics {
        loss: weighted_loss / total_tokens as f32,
        token_accuracy: correct_tokens as f32 / total_tokens as f32,
        correct_tokens,
        total_tokens,
    })
}

fn token_accuracy_counts(logits: &Tensor, targets: &Tensor) -> Result<(usize, usize)> {
    let (batch, seq_len, vocab_size) = logits.dims3()?;
    let logits = logits.flatten_all()?.to_vec1::<f32>()?;
    let targets = targets.flatten_all()?.to_vec1::<u32>()?;
    let token_count = batch * seq_len;
    let mut correct = 0;

    for token_idx in 0..token_count {
        let row = &logits[token_idx * vocab_size..(token_idx + 1) * vocab_size];
        let mut best_idx = 0usize;
        let mut best_value = f32::NEG_INFINITY;
        for (idx, value) in row.iter().enumerate() {
            if *value > best_value {
                best_value = *value;
                best_idx = idx;
            }
        }
        if best_idx as u32 == targets[token_idx] {
            correct += 1;
        }
    }

    Ok((correct, token_count))
}

fn summarize_benchmark(durations: Vec<Duration>, checksum: f64) -> InferenceBenchmark {
    let rounds = durations.len();
    let mut min_ms = f64::INFINITY;
    let mut max_ms = f64::NEG_INFINITY;
    let mut total_ms = 0.0;

    for duration in durations {
        let ms = duration.as_secs_f64() * 1000.0;
        min_ms = min_ms.min(ms);
        max_ms = max_ms.max(ms);
        total_ms += ms;
    }

    InferenceBenchmark {
        rounds,
        avg_ms: total_ms / rounds as f64,
        min_ms,
        max_ms,
        checksum,
    }
}

fn make_token_lm_batch(
    data: &[u32],
    step: usize,
    batch_size: usize,
    seq_len: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    if data.len() < 2 {
        bail!("need at least two tokens to create a next-token batch");
    }

    let mut inputs = Vec::with_capacity(batch_size * seq_len);
    let mut targets = Vec::with_capacity(batch_size * seq_len);
    let jump = (seq_len / 2).max(1);

    for batch_idx in 0..batch_size {
        let start = (step * batch_size * jump + batch_idx * (seq_len + 1)) % data.len();
        for pos in 0..seq_len {
            inputs.push(data[(start + pos) % data.len()]);
            targets.push(data[(start + pos + 1) % data.len()]);
        }
    }

    Ok((
        Tensor::from_vec(inputs, (batch_size, seq_len), device)?,
        Tensor::from_vec(targets, (batch_size, seq_len), device)?,
    ))
}
