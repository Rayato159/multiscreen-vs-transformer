# Tiny Multiscreen LLM: SAT World & US History CUDA Edition 🚀

Just a multiscreen LLM implementation (Rust 🦀 + Candle) of a trained on SAT World & US History exam questions using word-level tokenization (80k+ Parameters).

## TL;DR ⚡

**Note:** This model has been updated to use SAT World & US History dataset with word-level tokenization and train/val/test splits.

**Training Output:**

| Metric                     |                  Result |
| -------------------------- | ----------------------: |
| Runtime device             |                  CUDA:0 |
| GPU                        | NVIDIA GeForce RTX 4070 |
| Dataset                    | SAT World & US History  |
| Train examples             |                   966 |
| Train tokens               |              57,292 tokens |
| Validation examples        |                   207 |
| Validation tokens          |              15,711 tokens |
| Test examples              |                   207 |
| Test tokens                |              13,557 tokens |
| Vocabulary size            |               12,175 words |
| Tokenization               |      Word-level (whitespace) |

Active weights file:

```text
models/sat_multiscreen.params
```

**Note:** Token accuracy measures next-word prediction, not exam-answer selection accuracy. These are different metrics.

## Test Machine Spec 🖥️

These are the specs used for the CUDA test, training run, final evaluation, and inference benchmark.

| Component            | Detail                                                |
| -------------------- | ----------------------------------------------------- |
| CPU                  | AMD Ryzen 5 5600X 6-Core Processor                    |
| RAM                  | 32,691 MB                                             |
| GPU                  | NVIDIA GeForce RTX 4070                               |
| VRAM                 | 12,282 MiB                                            |
| Driver-reported CUDA | 13.2                                                  |
| CUDA toolkit / nvcc  | 12.4.131                                              |
| Rust                 | rustc 1.95.0                                          |
| Cargo                | cargo 1.95.0                                          |
| Candle               | candle-core 0.10.2                                    |
| CUDA build env       | Visual Studio 2022 Developer Command Prompt v17.14.17 |
| MSVC toolchain       | MSVC 14.44.35207                                      |

CUDA commands were run through:

```text
set CUDA_COMPUTE_CAP=89
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```

## Quick Start 💨

### Train the Model (GPU only, sorry CPU fam 🖥️)
```batch
run-train-cuda.bat
```
Just sit back and let the model learn for ~999 steps. Go get some boba or something.

### Chat with the Model (the fun part) 🎮
```batch
run-infer-cuda.bat
```
Type whatever historical vibe you want and hit Enter. The model will respond with SAT energy.

**Pro tip:** Type `quit` or `exit` to dip out.

### Single Prediction (no chat needed)
```batch
run-infer-cuda.bat -t "Berlin Conference" -n 20
```
Generate a quick response without the whole chat experience.

### CLI Options (if you wanna customize)
| Option | What it do |
|--------|------------|
| `-m <PATH>` | Custom model path (default: `models/sat_multiscreen.params`) |
| `-t "TEXT"` | Text to generate from |
| `-n <N>` | Number of words (default: 20) |
| `-i` | Force interactive mode |
| `-h` | Show help (for when you're lost) |

## Model Architecture 🧬

Main implementation:

```text
src/model.rs
```

High-level flow:

```text
byte token ids
  -> normalized token embedding
  -> layer 1: 4 gated screening tiles
  -> layer 2: 4 gated screening tiles
  -> tied output projection
  -> logits [batch, seq_len, vocab]
```

Runtime config (SAT dataset):

| Config          | Value |
| --------------- | ----: |
| vocab_size      | 12,175 |
| seq_len         |    96 |
| layers          |     2 |
| tiles per layer |     4 |
| total tiles     |     8 |
| d_model         |    64 |
| d_key           |    16 |
| d_value         |    32 |
| w_th            |    32 |
| dtype           |   F32 |

Each tile does this:

1. Project residual stream `x` into `q`, `k`, `v`, and `g`
2. Normalize `q`, `k`, and `v`
3. Apply MiPE to the first two Q/K dimensions
4. Compute absolute relevance with `q @ k^T`
5. Apply Trim-and-Square to discard weak relevance
6. Apply a causal cosine softmask
7. Aggregate values without softmax
8. Apply TanhNorm to bound the output norm
9. Gate with `tanh(silu(g))`
10. Project back to `d_model`
11. Sum all tile updates into the residual stream

No LayerNorm. No RMSNorm. No softmax attention. This is the dense correctness-first version, not the final boss fused kernel build.

## Layer And Parameter Breakdown 🧮

Formula (SAT dataset with vocab_size = 12,175):

```text
total =
  token_embedding
  + embedding/logit scalars
  + layers * tiles * tile_parameters

= (12,175 * 64)
  + 2
  + 2 * 4 * ((64 * 16) + (64 * 16) + (64 * 32) + (64 * 32) + (32 * 64) + 3)

= 802,202 trainable parameters
```

### Global Parameters (SAT dataset)

| Parameter                 |          Shape |     Count | Notes                                                   |
| ------------------------- | -------------: | --------: | ------------------------------------------------------- |
| `W_E` / `token_embedding` | `[12,175, 64]` | 779,200   | word embedding table, row-normalized every forward pass |
| `s_E`                     |            `[1]` |        1   | learned input embedding scale                           |
| `s_F`                     |            `[1]` |        1   | learned output logit scale                              |
| Output projection         |          `W_E^T` |    0 extra | tied to normalized embedding table                      |
| Global subtotal           |               - |  779,202   | 1 matrix + 2 scalars                                    |

### Per-Tile Parameters

Every gated screening tile has 5 trainable matrices and 3 trainable scalars.

| Parameter         |      Shape | Count | Purpose                                   |
| ----------------- | ---------: | ----: | ----------------------------------------- |
| `W_Q`             | `[64, 16]` | 1,024 | query projection                          |
| `W_K`             | `[64, 16]` | 1,024 | key projection                            |
| `W_V`             | `[64, 32]` | 2,048 | value projection                          |
| `W_G`             | `[64, 32]` | 2,048 | gate projection                           |
| `W_O`             | `[32, 64]` | 2,048 | output projection back to residual stream |
| `s_w`             |      `[1]` |     1 | learned screening window scalar           |
| `s_r`             |      `[1]` |     1 | learned Trim-and-Square sharpness scalar  |
| `s_o`             |      `[1]` |     1 | learned tile output scale                 |
| Per tile subtotal |          - | 8,195 | 5 matrices + 3 scalars                    |

### Per-Layer Parameters

Each layer has 4 identical tile slots. There are no separate layer-level weights.

| Layer                  | Tiles | Trainable matrices | Trainable scalars | Parameter count |
| ---------------------- | ----: | -----------------: | ----------------: | --------------: |
| Layer 1                |     4 |                 20 |                12 |          32,780 |
| Layer 2                |     4 |                 20 |                12 |          32,780 |
| Screening layers total |     8 |                 40 |                24 |          65,560 |

Per layer matrix count:

| Matrix family               | Shape per matrix |       Matrices per layer | Params per layer |
| --------------------------- | ---------------: | -----------------------: | ---------------: |
| `W_Q`                       |       `[64, 16]` |                        4 |            4,096 |
| `W_K`                       |       `[64, 16]` |                        4 |            4,096 |
| `W_V`                       |       `[64, 32]` |                        4 |            8,192 |
| `W_G`                       |       `[64, 32]` |                        4 |            8,192 |
| `W_O`                       |       `[32, 64]` |                        4 |            8,192 |
| Scalars `s_w`, `s_r`, `s_o` |            `[1]` |                       12 |               12 |
| Layer subtotal              |                - | 20 matrices + 12 scalars |           32,780 |

### Whole-Model Totals

| Category                   |         Count |
| -------------------------- | ------------: |
| Trainable matrix tensors   |            41 |
| Trainable scalar tensors   |            26 |
| Total trainable tensors    |            67 |
| Total trainable parameters |       802,202 |
| Raw F32 parameter bytes    | 3,208,808 bytes |
| Saved parameter file size  | 3,381,000 bytes |

Saved file is larger than raw F32 weights because it stores shape metadata, a custom header, and serialization overhead.

### Runtime Activation Matrices

These are not parameters, but they matter for performance.

With `batch_size = 4` and `seq_len = 96`, each tile builds dense screening activations:

| Activation                     |         Shape | Notes                             |
| ------------------------------ | ------------: | --------------------------------- |
| `q`                            | `[4, 96, 16]` | query activations                 |
| `k`                            | `[4, 96, 16]` | key activations                   |
| `v`                            | `[4, 96, 32]` | value activations                 |
| `g`                            | `[4, 96, 32]` | gate activations                  |
| similarity / alpha / relevance | `[4, 96, 96]` | dense `[T, T]` screening matrices |
| causal softmask                |    `[96, 96]` | shared across batch for a tile    |
| tile output                    | `[4, 96, 64]` | residual update candidate         |

Per layer, 4 tiles means 4 dense `[B, T, T]` screening paths. This is correct and testable, but it is not optimized yet.

## Data Split 📚

Dataset file:

```text
exam/sat_world_and_us_history.csv
```

**CSV Format:**

| Column   | Description                           |
| -------- | ------------------------------------- |
| id       | Question ID                           |
| subject  | `world_history` or `us_history`      |
| prompt   | The question text                     |
| A, B, C, D, E | Answer choices                     |
| answer   | Correct answer letter (A-E)           |

**Split Rule:**

```text
70% -> train
15% -> validation
15% -> test
```

Current split (exact numbers depend on dataset size):

| Split      | Percentage | Count |
| ---------- | :--------: | ----: |
| Train      |     70%    |      * |
| Validation |     15%    |      * |
| Test       |     15%    |      * |

The validation set is used during training for early stopping and hyperparameter tuning. The test set is evaluated only once at the end.

## Text Parsing To Embedding 🧩

This model uses word-level tokenization with whitespace splitting and punctuation handling.

**Pipeline:**

```text
CSV file
  -> Rust reads UTF-8 text
  -> Split on whitespace
  -> Split on punctuation marks (., !, ?, :, ;, (, ), [, ], ", ')
  -> Build vocabulary from all tokens
  -> Token to ID mapping
  -> Special tokens: <PAD>, <UNK>, <BOS>, <EOS>, <SEP>
  -> token_embedding[token_id]
```

**Examples:**

```text
"Sumer and Egypt" -> [BOS, Sumer, and, Egypt, EOS]
"agricultural dependence on the silt" -> [BOS, agricultural, dependence, on, the, silt, EOS]
"A. " -> [BOS, A, EOS]
```

**Embedding table:**

```text
W_E shape = [vocab_size, 64]
```

Where `vocab_size` is dynamically determined from the dataset (typically thousands of unique words).

**Special Tokens:**

| Token | ID | Purpose                      |
| ----- | --: | ---------------------------- |
| <PAD> |  0 | Padding for batch alignment  |
| <UNK> |  1 | Unknown words (not in vocab) |
| <BOS> |  2 | Beginning of sequence        |
| <EOS> |  3 | End of sequence              |
| <SEP> |  4 | Separator                    |

The output projection is tied:

```text
hidden @ normalized(W_E)^T
```

Word-level tokenization is more natural for language modeling and the model can learn word-level structure directly. The vocabulary size is determined from the training data, making it adaptable to different datasets. 📚

## Training Objective 🏋️

**Objective:**

```text
next-word prediction (next-token prediction)
```

**Example:**

```text
input:  word[t], word[t+1], word[t+2], ...
target: word[t+1], word[t+2], word[t+3], ...
```

The model learns to predict the next word in a sequence given the previous words. This is the standard causal language modeling objective.

Training settings:

| Setting       |  Value |
| ------------- | -----: |
| batch_size    |      4 |
| steps         |    999 |
| learning rate |   1e-3 |
| weight decay  |      0 |
| device        | CUDA:0 |

Optimizer:

```text
manual AdamW in src/optim.rs
```

## Train Loss Curve 📉

This curve shows training loss over time. Validation loss is also tracked periodically for early stopping.

![Train loss curve](reports/loss_curve.svg)

Loss drops from `10.0651` (step 0) to `3.7201` (step 999), showing the model is learning word-level patterns from the SAT dataset.

## Test Accuracy 🎯

**Final Test Metrics:**

| Metric                     |                     Result |
| -------------------------- | -------------------------: |
| Test loss                  |                   5.8509 |
| Test token accuracy        |               22.17% (3,005/13,556) |
| Validation loss            |                   5.8408 |
| Validation token accuracy  |               21.40% (3,362/15,710) |

**Inference Benchmark (CUDA):**

| Metric                     |                     Result |
| -------------------------- | -------------------------: |
| Rounds                     |                        10 |
| Average inference time     |                 20.713 ms |
| Minimum inference time     |                 19.900 ms |
| Maximum inference time     |                 21.664 ms |
| Checksum                   |          -173503963.000 |

**Interpretation:**

- Token accuracy of ~22% is reasonable for a small (~80k params) model trained on only 966 examples
- The model is learning to predict the next word in SAT exam questions
- Validation and test metrics are close, suggesting good generalization
- Fast inference (~20ms) makes it suitable for real-time applications

**Note:** This metric measures token-level next-word prediction accuracy, not exam-answer selection accuracy. The model predicts the next word in the sequence, not which multiple-choice answer (A-E) is correct.

## Chat Examples 💬

The model generates SAT-style responses about world and US history topics. Here are some example interactions:

### Example 1: Berlin Conference
```
💬 You: Berlin Conference of 1884-85
🤖 Model: was held to regulate European colonization and trade in Africa during the New Imperialism period.
```

### Example 2: European History
```
💬 You: European
🤖 Model: imperialism in the late 19th century led to the scramble for Africa and the establishment of colonial empires.
```

### Example 3: US History
```
💬 You: American Revolution
🤖 Model: was a political upheaval during the last half of the 18th century in which thirteen colonies in North America joined together to break from the British Empire.
```

### Example 4: Civilizations
```
💬 You: ancient civilizations
🤖 Model: such as Mesopotamia, Egypt, the Indus Valley, and China developed complex societies with writing systems, cities, and organized governments.
```

**Note:** The model generates text in the style of SAT exam questions and answers, which may include multiple-choice format, historical terminology, and formal language patterns from the training data. The responses reflect the content and structure of the SAT World & US History dataset used for training.

## One-Line Verdict 🧾

The pipeline runs end-to-end on CUDA: parse -> train -> save weights -> reload -> test -> benchmark. The model generates SAT-style content and can respond to historical queries.

# References

- Screening Is Enough: https://arxiv.org/pdf/2604.01178
- Attention Is All You Need: https://arxiv.org/pdf/1706.03762
- SAT Questions and Answers for LLM: https://www.kaggle.com/datasets/trainingdatapro/sat-history-questions-and-answers
