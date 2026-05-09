use crate::model::{MultiScreenLm, cross_entropy_loss};
use candle_core::{Device, Result, Tensor, bail};
use std::{
    fs,
    path::Path,
    time::{Duration, Instant},
};

pub const BYTE_VOCAB_SIZE: usize = 256;
const RECORD_SEPARATOR: &[u8] = b"\n<END_EXAM_EXAMPLE>\n";

pub struct ExamDataset {
    train: Vec<u32>,
    test: Vec<u32>,
    train_examples: usize,
    test_examples: usize,
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
        let content = match fs::read_to_string(path) {
            Ok(content) => content,
            Err(err) => bail!(
                "failed to read {}: {err}. Run `D:\\anaconda3\\python.exe .\\tools\\extract_exam.py --pdf-dir .\\exam --out .\\exam\\tcas68-all.dataset.txt` first.",
                path.display()
            ),
        };
        Self::from_text(&content)
    }

    pub fn from_text(content: &str) -> Result<Self> {
        let mut train = Vec::new();
        let mut test = Vec::new();
        let mut train_examples = 0;
        let mut test_examples = 0;

        for chunk in content.split("### Example ").skip(1) {
            let record = format!("### Example {chunk}");
            let split = record.lines().find_map(|line| {
                let line = line.trim();
                if line == "Split: train" {
                    Some("train")
                } else if line == "Split: test" {
                    Some("test")
                } else {
                    None
                }
            });

            if split == Some("train") {
                push_record_bytes(&mut train, &record);
                train_examples += 1;
            } else if split == Some("test") {
                push_record_bytes(&mut test, &record);
                test_examples += 1;
            }
        }

        if train_examples == 0 || test_examples == 0 {
            bail!("dataset must contain both train and test examples");
        }
        if train.len() < 2 || test.len() < 2 {
            bail!("dataset split is too small for next-byte language modeling");
        }

        Ok(Self {
            train,
            test,
            train_examples,
            test_examples,
        })
    }

    pub fn train_examples(&self) -> usize {
        self.train_examples
    }

    pub fn test_examples(&self) -> usize {
        self.test_examples
    }

    pub fn train_tokens(&self) -> usize {
        self.train.len()
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
        make_byte_lm_batch(&self.train, step, batch_size, seq_len, device)
    }

    pub fn test_batch(
        &self,
        step: usize,
        batch_size: usize,
        seq_len: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        make_byte_lm_batch(&self.test, step, batch_size, seq_len, device)
    }

    #[cfg(test)]
    fn train_bytes(&self) -> &[u32] {
        &self.train
    }

    #[cfg(test)]
    fn test_bytes(&self) -> &[u32] {
        &self.test
    }
}

pub fn evaluate_test_metrics(
    model: &MultiScreenLm,
    dataset: &ExamDataset,
    seq_len: usize,
    device: &Device,
) -> Result<EvalMetrics> {
    evaluate_bytes(model, &dataset.test, seq_len, device)
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
        let (inputs, _) = if round % 2 == 0 {
            dataset.train_batch(step, batch_size, seq_len, device)?
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

fn evaluate_bytes(
    model: &MultiScreenLm,
    data: &[u32],
    seq_len: usize,
    device: &Device,
) -> Result<EvalMetrics> {
    if data.len() < 2 {
        bail!("need at least two bytes to evaluate split");
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

fn push_record_bytes(dst: &mut Vec<u32>, record: &str) {
    dst.extend(record.as_bytes().iter().map(|byte| *byte as u32));
    dst.extend(RECORD_SEPARATOR.iter().map(|byte| *byte as u32));
}

fn make_byte_lm_batch(
    data: &[u32],
    step: usize,
    batch_size: usize,
    seq_len: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    if data.len() < 2 {
        bail!("need at least two bytes to create a next-byte batch");
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_exam_dataset_with_exact_split() -> Result<()> {
        let dataset = ExamDataset::from_file("exam/tcas68-all.dataset.txt")?;
        assert!(dataset.train_examples() > dataset.test_examples());
        assert!(dataset.train_examples() >= 400);
        assert!(dataset.test_examples() >= 100);

        let train = bytes_to_string_lossy(dataset.train_bytes());
        let test = bytes_to_string_lossy(dataset.test_bytes());
        assert!(train.contains("Source:"));
        assert!(train.contains("Answer:"));
        assert!(train.contains("Answer Code:"));
        assert!(test.contains("Answer:"));
        Ok(())
    }

    #[test]
    fn exam_batches_have_expected_shapes() -> Result<()> {
        let device = crate::runtime::default_device()?;
        let dataset = ExamDataset::from_file("exam/tcas68-all.dataset.txt")?;
        let (inputs, targets) = dataset.train_batch(0, 3, 16, &device)?;
        assert_eq!(inputs.dims(), &[3, 16]);
        assert_eq!(targets.dims(), &[3, 16]);
        Ok(())
    }

    #[test]
    fn dataset_contains_full_answer_text_when_resolved() -> Result<()> {
        let content = fs::read_to_string("exam/tcas68-all.dataset.txt")?.replace("\r\n", "\n");
        assert!(
            (1..=5).any(|choice| content.contains(&format!("Answer:\n{choice}. "))),
            "expected at least one resolved full choice answer"
        );
        Ok(())
    }

    fn bytes_to_string_lossy(bytes: &[u32]) -> String {
        let bytes = bytes.iter().map(|byte| *byte as u8).collect::<Vec<_>>();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}
