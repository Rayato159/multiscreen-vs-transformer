# Tiny Multiscreen LM: TCAS68 CUDA Edition 🚀

This repo is a Rust + Candle implementation of a tiny dense Multiscreen-style causal language model trained on extracted TCAS68 exam text (80k Parameters used). It now supports CUDA through `candle-core = "0.10.2"` with the crate feature `cuda`.

## TL;DR ⚡

Latest GPU run:

| Metric                     |                  Result |
| -------------------------- | ----------------------: |
| Runtime device             |                  CUDA:0 |
| GPU                        | NVIDIA GeForce RTX 4070 |
| Train examples             |                     491 |
| Test examples              |                     122 |
| Validation examples        |                       0 |
| Train loss                 |        8.2864 -> 2.3788 |
| Final test loss            |                  3.6691 |
| Test token accuracy        |                  20.26% |
| Correct test tokens        |        31,695 / 156,430 |
| Inference latency          |           avg 21.946 ms |
| Min / max latency          |   17.463 ms / 34.117 ms |
| Benchmark rounds           |                      10 |
| Total trainable parameters |                  81,946 |

Active weights file:

```text
models/tcas68_all_multiscreen.params
```

Important: `20.26%` is next-byte token accuracy, not exam-answer accuracy. Different metric, different battlefield. 🎯

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

## How To Run 🏃

CPU:

```text
cargo run --release --quiet
```

CUDA:

```text
cargo run --release --features cuda --quiet
```

Force device:

```text
MULTISCREEN_DEVICE=cpu
MULTISCREEN_DEVICE=cuda
MULTISCREEN_DEVICE=auto
```

On Windows PowerShell, the CUDA run used:

```text
cmd /S /C "set CUDA_COMPUTE_CAP=89 && call \"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat\" && cargo run --release --features cuda --quiet"
```

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

Runtime config:

| Config          | Value |
| --------------- | ----: |
| vocab_size      |   256 |
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

Formula:

```text
total =
  token_embedding
  + embedding/logit scalars
  + layers * tiles * tile_parameters

= (256 * 64)
  + 2
  + 2 * 4 * ((64 * 16) + (64 * 16) + (64 * 32) + (64 * 32) + (32 * 64) + 3)

= 81,946 trainable parameters
```

### Global Parameters

| Parameter                 |       Shape |   Count | Notes                                                   |
| ------------------------- | ----------: | ------: | ------------------------------------------------------- |
| `W_E` / `token_embedding` | `[256, 64]` |  16,384 | byte embedding table, row-normalized every forward pass |
| `s_E`                     |       `[1]` |       1 | learned input embedding scale                           |
| `s_F`                     |       `[1]` |       1 | learned output logit scale                              |
| Output projection         |     `W_E^T` | 0 extra | tied to normalized embedding table                      |
| Global subtotal           |           - |  16,386 | 1 matrix + 2 scalars                                    |

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
| Total trainable parameters |        81,946 |
| Raw F32 parameter bytes    | 327,784 bytes |
| Saved parameter file size  | 329,736 bytes |

Saved file is slightly larger than raw F32 weights because it stores shape metadata and a small custom header.

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
exam/tcas68-all.dataset.txt
```

Extractor:

```text
tools/extract_exam.py
```

Split rule:

```text
global example index % 5 == 0 -> test
everything else -> train
```

Current split:

| Split      | Count |
| ---------- | ----: |
| Train      |   491 |
| Test       |   122 |
| Validation |     0 |

There is no validation split yet. The test split is evaluated once at the end and is not used for training loss plots.

## Text Parsing To Embedding 🧩

This is not word embedding. It is byte-level embedding.

Pipeline:

```text
PDF
  -> pypdf text extraction
  -> normalized dataset records
  -> Rust reads UTF-8 text
  -> record.as_bytes()
  -> byte id 0..255
  -> token_embedding[byte_id]
```

Core line:

```rust
dst.extend(record.as_bytes().iter().map(|byte| *byte as u32));
```

Examples:

```text
"A" -> 1 byte -> 1 token
"ก" -> multiple UTF-8 bytes -> multiple tokens
"คำตอบ" -> many bytes -> many tokens
```

Embedding table:

```text
W_E shape = [256, 64]
```

The output projection is tied:

```text
hidden @ normalized(W_E)^T
```

Byte-level tokenization is robust and tokenizer-free, but Thai text expands into multiple bytes. The model has to learn byte patterns before word-level structure shows up. Tiny model, hard job. 🥊

## Training Objective 🏋️

Objective:

```text
next-byte prediction
```

Example:

```text
input:  byte[t], byte[t+1], byte[t+2], ...
target: byte[t+1], byte[t+2], byte[t+3], ...
```

Training settings:

| Setting       |  Value |
| ------------- | -----: |
| batch_size    |      4 |
| steps         |     80 |
| learning rate |   1e-3 |
| weight decay  |      0 |
| device        | CUDA:0 |

Optimizer:

```text
manual AdamW in src/optim.rs
```

## Train Loss Curve 📉

This curve is train loss only. No test loss pollution.

![Train loss curve](reports/loss_curve.svg)

Latest GPU train log:

| Step | Train Loss |
| ---: | ---------: |
|    0 |     8.2864 |
|   10 |     4.4091 |
|   20 |     2.9567 |
|   30 |     3.4602 |
|   40 |     3.0166 |
|   50 |     3.0340 |
|   60 |     2.8163 |
|   70 |     2.5541 |
|   79 |     2.3788 |

Loss drops from `8.2864` to `2.3788`, so the model is learning structure from the train split.

## Test Accuracy 🎯

Final GPU test result:

```text
final_test loss 3.6691 token_accuracy 20.26% (31695/156430)
```

Meaning:

```text
Out of 156,430 held-out test byte positions,
the model guessed the exact next byte correctly 31,695 times.
```

This is byte-level language modeling accuracy, not answer-selection accuracy. Real exam accuracy needs a separate evaluator that scores each answer choice by likelihood and compares with `Answer Code`.

## GPU Inference Performance ⏱️

Latest GPU benchmark:

```text
inference_benchmark rounds 10 avg_ms 21.946 min_ms 17.463 max_ms 34.117 checksum -810557.078
```

Benchmark setup:

| Setting      |                          Value |
| ------------ | -----------------------------: |
| rounds       |                             10 |
| batch_size   |                              4 |
| seq_len      |                             96 |
| device       |                         CUDA:0 |
| input source | alternating train/test batches |

Why GPU is not magically faster here:

- The model is tiny
- Batch size is only 4
- Sequence length is only 96
- The implementation is dense and correctness-first
- Benchmark includes synchronization through checksum materialization
- GPU launch/sync overhead matters at this small scale

CUDA works. The current workload is just too small to make the RTX 4070 sweat.

## Verification ✅

Commands run:

```text
cargo fmt --check
cargo test
cargo test --features cuda
cargo run --release --features cuda --quiet
```

Results:

| Check                    | Result              |
| ------------------------ | ------------------- |
| Format                   | Passed              |
| CPU tests                | 10 passed, 0 failed |
| CUDA tests               | 10 passed, 0 failed |
| CUDA training run        | Passed              |
| CUDA final eval          | Passed              |
| CUDA inference benchmark | Passed              |

## Caveats 🧯

Current limitations:

- No validation split yet
- No answer-level exam accuracy yet
- Byte-level model, not word/subword model
- Thai characters expand into multiple UTF-8 bytes
- PDF extraction can damage math layout, tables, images, and graphs
- Some PDFs were skipped by the extractor
- Some answers may still fall back to raw answer codes
- The model is tiny and trained for only 80 steps
- CUDA support works, but the implementation is not optimized for GPU throughput yet

## One-Line Verdict 🧾

The pipeline runs end-to-end on CUDA: parse -> train -> save weights -> reload -> test -> benchmark. It is not a real exam solver yet, but the mechanism is alive, tested.

# References

- Screening Is Enough: https://arxiv.org/pdf/2604.01178
- TCAS68 Exam: https://www.mytcas.com/answers/
