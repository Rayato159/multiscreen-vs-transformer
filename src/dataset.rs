use crate::{lm::LanguageModel, multiscreen::cross_entropy_loss};
use arrow_array::{Array, LargeStringArray, StringArray, StringViewArray};
use candle_core::{Device, Result, Tensor, bail};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::{
    fs::{self, File},
    path::Path,
    time::{Duration, Instant},
};
use tokenizers::{
    AddedToken, Tokenizer, TokenizerBuilder,
    models::bpe::{BPE, BpeTrainerBuilder},
    normalizers::{strip::Strip, unicode::NFC, utils::Sequence},
    pre_tokenizers::byte_level::ByteLevel,
};

pub const DEFAULT_TOKENIZER_PATH: &str = "models/khanacademy_tokenizer.json";
pub const DEFAULT_TOKENIZER_VOCAB_SIZE: usize = 8192;
pub const DEFAULT_EVAL_TOKEN_LIMIT: usize = 16_384;

pub const PAD_TOKEN_TEXT: &str = "<PAD>";
pub const UNK_TOKEN_TEXT: &str = "<UNK>";
pub const BOS_TOKEN_TEXT: &str = "<BOS>";
pub const EOS_TOKEN_TEXT: &str = "<EOS>";
pub const SEP_TOKEN_TEXT: &str = "<SEP>";

#[derive(Clone)]
pub struct HfTokenizer {
    tokenizer: Tokenizer,
    pad_id: u32,
    unk_id: u32,
    bos_id: u32,
    eos_id: u32,
    sep_id: u32,
}

pub struct TextDataset {
    train: Vec<u32>,
    val: Vec<u32>,
    test: Vec<u32>,
    train_examples: usize,
    val_examples: usize,
    test_examples: usize,
    tokenizer: HfTokenizer,
}

#[derive(Clone, Debug)]
struct PromptTextRecord {
    prompt: String,
    text: String,
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

impl HfTokenizer {
    pub fn train_from_parquet(path: impl AsRef<Path>) -> Result<Self> {
        let records = read_prompt_text_records(path.as_ref())?;
        Self::train_from_records(&records, DEFAULT_TOKENIZER_VOCAB_SIZE)
    }

    pub fn load_or_train(
        dataset_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let tokenizer_path = tokenizer_path.as_ref();
        if tokenizer_path.exists() {
            return Self::from_file(tokenizer_path);
        }

        let tokenizer = Self::train_from_parquet(dataset_path)?;
        tokenizer.save(tokenizer_path)?;
        Ok(tokenizer)
    }

    fn train_from_records(records: &[PromptTextRecord], vocab_size: usize) -> Result<Self> {
        if records.is_empty() {
            bail!("Khan Academy parquet dataset contains no prompt/text rows");
        }

        let bpe = BPE::builder()
            .unk_token(UNK_TOKEN_TEXT.to_string())
            .build()
            .map_err(tokenizer_error)?;
        let mut tokenizer = TokenizerBuilder::new()
            .with_model(bpe)
            .with_normalizer(Some(Sequence::new(vec![
                Strip::new(true, true).into(),
                NFC.into(),
            ])))
            .with_pre_tokenizer(Some(ByteLevel::default()))
            .with_post_processor(Some(ByteLevel::default()))
            .with_decoder(Some(ByteLevel::default()))
            .build()
            .map_err(tokenizer_error)?;

        let mut trainer = BpeTrainerBuilder::new()
            .show_progress(false)
            .vocab_size(vocab_size)
            .min_frequency(2)
            .special_tokens(vec![
                AddedToken::from(PAD_TOKEN_TEXT.to_string(), true),
                AddedToken::from(UNK_TOKEN_TEXT.to_string(), true),
                AddedToken::from(BOS_TOKEN_TEXT.to_string(), true),
                AddedToken::from(EOS_TOKEN_TEXT.to_string(), true),
                AddedToken::from(SEP_TOKEN_TEXT.to_string(), true),
            ])
            .build();

        let corpus = records
            .iter()
            .flat_map(|record| [record.prompt.as_str(), record.text.as_str()]);
        tokenizer
            .train(&mut trainer, corpus)
            .map_err(tokenizer_error)?;

        Self::from_tokenizer(tokenizer.into())
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path.as_ref()).map_err(tokenizer_error)?;
        Self::from_tokenizer(tokenizer)
    }

    fn from_tokenizer(tokenizer: Tokenizer) -> Result<Self> {
        let pad_id = required_token_id(&tokenizer, PAD_TOKEN_TEXT)?;
        let unk_id = required_token_id(&tokenizer, UNK_TOKEN_TEXT)?;
        let bos_id = required_token_id(&tokenizer, BOS_TOKEN_TEXT)?;
        let eos_id = required_token_id(&tokenizer, EOS_TOKEN_TEXT)?;
        let sep_id = required_token_id(&tokenizer, SEP_TOKEN_TEXT)?;

        Ok(Self {
            tokenizer,
            pad_id,
            unk_id,
            bos_id,
            eos_id,
            sep_id,
        })
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        self.tokenizer.save(path, false).map_err(tokenizer_error)
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    pub fn encode_example(&self, prompt: &str, text: &str) -> Result<Vec<u32>> {
        let mut ids = self.encode_prompt(prompt)?;
        ids.extend(self.encode_plain(text)?);
        ids.push(self.eos_id);
        Ok(ids)
    }

    pub fn encode_prompt(&self, prompt: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        ids.push(self.bos_id);
        ids.extend(self.encode_plain(prompt)?);
        ids.push(self.sep_id);
        Ok(ids)
    }

    fn encode_plain(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(tokenizer_error)?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer.decode(ids, true).map_err(tokenizer_error)
    }

    pub fn token_label(&self, id: u32) -> String {
        self.tokenizer
            .id_to_token(id)
            .unwrap_or_else(|| format!("<id:{id}>"))
    }

    pub fn is_special_id(&self, id: u32) -> bool {
        id == self.pad_id
            || id == self.unk_id
            || id == self.bos_id
            || id == self.eos_id
            || id == self.sep_id
    }
}

impl TextDataset {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let records = read_prompt_text_records(path.as_ref())?;
        let tokenizer = HfTokenizer::train_from_records(&records, DEFAULT_TOKENIZER_VOCAB_SIZE)?;
        Self::from_records(records, tokenizer)
    }

    fn from_records(records: Vec<PromptTextRecord>, tokenizer: HfTokenizer) -> Result<Self> {
        if records.len() < 10 {
            bail!("Khan Academy parquet dataset needs enough rows for train/val/test splits");
        }

        let train_cut = records.len() * 80 / 100;
        let val_cut = records.len() * 90 / 100;
        let mut train = Vec::new();
        let mut val = Vec::new();
        let mut test = Vec::new();
        let mut train_examples = 0;
        let mut val_examples = 0;
        let mut test_examples = 0;

        for (idx, record) in records.iter().enumerate() {
            let tokens = tokenizer.encode_example(&record.prompt, &record.text)?;
            if idx < train_cut {
                train.extend(tokens);
                train_examples += 1;
            } else if idx < val_cut {
                val.extend(tokens);
                val_examples += 1;
            } else {
                test.extend(tokens);
                test_examples += 1;
            }
        }

        if train.len() < 2 || val.len() < 2 || test.len() < 2 {
            bail!("tokenized Khan Academy split is too small for next-token language modeling");
        }

        Ok(Self {
            train,
            val,
            test,
            train_examples,
            val_examples,
            test_examples,
            tokenizer,
        })
    }

    pub fn tokenizer(&self) -> &HfTokenizer {
        &self.tokenizer
    }

    pub fn save_tokenizer(&self, path: impl AsRef<Path>) -> Result<()> {
        self.tokenizer.save(path)
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
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

pub fn evaluate_test_metrics<M: LanguageModel>(
    model: &M,
    dataset: &TextDataset,
    seq_len: usize,
    device: &Device,
) -> Result<EvalMetrics> {
    evaluate_tokens(
        model,
        &dataset.test,
        seq_len,
        device,
        Some(DEFAULT_EVAL_TOKEN_LIMIT),
    )
}

pub fn evaluate_val_metrics<M: LanguageModel>(
    model: &M,
    dataset: &TextDataset,
    seq_len: usize,
    device: &Device,
) -> Result<EvalMetrics> {
    evaluate_tokens(
        model,
        &dataset.val,
        seq_len,
        device,
        Some(DEFAULT_EVAL_TOKEN_LIMIT),
    )
}

pub fn benchmark_inference<M: LanguageModel>(
    model: &M,
    dataset: &TextDataset,
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
        device.synchronize()?;
        checksum += logits.sum_all()?.to_scalar::<f32>()? as f64;
        durations.push(started.elapsed());
    }

    Ok(summarize_benchmark(durations, checksum))
}

fn read_prompt_text_records(path: &Path) -> Result<Vec<PromptTextRecord>> {
    let file = File::open(path).map_err(|e| {
        candle_core::Error::Msg(format!(
            "failed to open Khan Academy parquet file {}: {e}",
            path.display()
        ))
    })?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(parquet_error)?;
    let schema = builder.schema().clone();
    let prompt_idx = schema.index_of("prompt").map_err(arrow_error)?;
    let text_idx = schema.index_of("text").map_err(arrow_error)?;
    let reader = builder
        .with_batch_size(2048)
        .build()
        .map_err(parquet_error)?;
    let mut records = Vec::new();

    for batch in reader {
        let batch = batch.map_err(arrow_error)?;
        let prompt_column = batch.column(prompt_idx);
        let text_column = batch.column(text_idx);

        for row in 0..batch.num_rows() {
            let Some(prompt) = string_value(prompt_column.as_ref(), row)? else {
                continue;
            };
            let Some(text) = string_value(text_column.as_ref(), row)? else {
                continue;
            };
            if prompt.trim().is_empty() || text.trim().is_empty() {
                continue;
            }
            records.push(PromptTextRecord { prompt, text });
        }
    }

    if records.is_empty() {
        bail!("Khan Academy parquet file has no non-empty prompt/text rows");
    }

    Ok(records)
}

fn string_value(array: &dyn Array, row: usize) -> Result<Option<String>> {
    if array.is_null(row) {
        return Ok(None);
    }

    if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
        return Ok(Some(values.value(row).to_string()));
    }
    if let Some(values) = array.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(Some(values.value(row).to_string()));
    }
    if let Some(values) = array.as_any().downcast_ref::<StringViewArray>() {
        return Ok(Some(values.value(row).to_string()));
    }

    bail!("expected parquet column to be a UTF-8 string array")
}

fn evaluate_tokens<M: LanguageModel>(
    model: &M,
    data: &[u32],
    seq_len: usize,
    device: &Device,
    max_tokens: Option<usize>,
) -> Result<EvalMetrics> {
    if data.len() < 2 {
        bail!("need at least two tokens to evaluate split");
    }

    let usable_len = max_tokens
        .map(|limit| data.len().min(limit.max(2)))
        .unwrap_or(data.len());
    let data = &data[..usable_len];
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

fn required_token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32> {
    tokenizer.token_to_id(token).ok_or_else(|| {
        candle_core::Error::Msg(format!("tokenizer is missing special token {token}"))
    })
}

fn tokenizer_error(error: tokenizers::Error) -> candle_core::Error {
    candle_core::Error::Msg(format!("Hugging Face tokenizer error: {error}"))
}

fn parquet_error(error: parquet::errors::ParquetError) -> candle_core::Error {
    candle_core::Error::Msg(format!("parquet error: {error}"))
}

fn arrow_error(error: arrow_schema::ArrowError) -> candle_core::Error {
    candle_core::Error::Msg(format!("arrow schema error: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_wraps_prompt_and_output_with_special_ids() -> Result<()> {
        let records = vec![
            PromptTextRecord {
                prompt: "count two apples".to_string(),
                text: "one two".to_string(),
            },
            PromptTextRecord {
                prompt: "add three and four".to_string(),
                text: "seven".to_string(),
            },
        ];
        let tokenizer = HfTokenizer::train_from_records(&records, 128)?;
        let ids = tokenizer.encode_example("count two apples", "one two")?;
        assert_eq!(ids.first().copied(), Some(tokenizer.bos_id));
        assert_eq!(ids.last().copied(), Some(tokenizer.eos_id));
        assert!(ids.contains(&tokenizer.sep_id));
        Ok(())
    }
}
