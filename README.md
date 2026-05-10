# Tiny Multiscreen vs Transformer LM 🚀

| Multiscreen demo | Transformer demo |
| --- | --- |
| ![Multiscreen demo](multiscreen_demo.gif) | ![Transformer demo](transformer_demo.gif) |

Tiny Rust/Candle lab comparing two causal language models on `dataset/khanacademy.parquet`.

- 🌀 Multiscreen: `src/multiscreen.rs`
- ⚡ Transformer: `src/transformer.rs`
- 🧠 Dataset + Hugging Face tokenizer: `src/dataset.rs`
- 🔐 Checkpoints are architecture-specific. No shared-weights clownery.

## Quick start ⚡

Chat with a trained model:

```powershell
cmd /S /C "set CUDA_COMPUTE_CAP=89 && call ""C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"" && set MULTISCREEN_DEVICE=cuda && cargo run --release --features cuda --bin infer -- --model multiscreen -i"
cmd /S /C "set CUDA_COMPUTE_CAP=89 && call ""C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"" && set MULTISCREEN_DEVICE=cuda && cargo run --release --features cuda --bin infer -- --model transformer -i"
```

Train one model:

```powershell
cmd /S /C "set CUDA_COMPUTE_CAP=89 && call ""C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"" && set MULTISCREEN_DEVICE=cuda && cargo run --release --features cuda --bin train -- --model multiscreen --steps 300"
cmd /S /C "set CUDA_COMPUTE_CAP=89 && call ""C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"" && set MULTISCREEN_DEVICE=cuda && cargo run --release --features cuda --bin train -- --model transformer --steps 300"
```

Compare both:

```powershell
cmd /S /C "set CUDA_COMPUTE_CAP=89 && call ""C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"" && set MULTISCREEN_DEVICE=cuda && cargo run --release --features cuda --bin compare -- --steps 300"
```

## Data 🧩

| Item | Value |
| --- | --- |
| File | `dataset/khanacademy.parquet` |
| Columns used | `prompt` as input, `text` as output |
| Training sequence | `<BOS> prompt <SEP> text <EOS>` |
| Split | 80% train, 10% validation, 10% test |
| Tokenizer | Hugging Face byte-level BPE |
| Vocab size | 8,192 |
| Eval cap | first 16,384 tokens per validation/test split |

Special tokens:

| Token | ID |
| --- | ---: |
| `<PAD>` | 0 |
| `<UNK>` | 1 |
| `<BOS>` | 2 |
| `<EOS>` | 3 |
| `<SEP>` | 4 |

## Files 🔐

| File | Purpose |
| --- | --- |
| `models/khanacademy_tokenizer.json` | Hugging Face tokenizer used by train + infer |
| `models/khanacademy_multiscreen.params` | Multiscreen checkpoint |
| `models/khanacademy_transformer.params` | Transformer checkpoint |
| `reports/loss_curve.svg` | Latest training loss plot |
| `reports/loss_points.csv` | Raw plotted loss points |

## Latest CUDA Run 🏁

Run: `cargo run --release --features cuda --bin compare -- --steps 300`

| Model | Params | Train loss | Val loss | Val acc | Test loss | Test acc | Avg inference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Multiscreen | 589,850 | 2.3034 | 7.3351 | 15.88% | 7.3149 | 14.25% | 21.412 ms |
| Transformer | 623,746 | 5.6860 | 8.2730 | 2.94% | 8.1324 | 2.77% | 5.583 ms |

![Training loss curves](reports/loss_curve.svg)

Vibe check:

- 🌀 Multiscreen learns the tiny run harder, but it is still dense/unfused ops.
- ⚡ Transformer is way faster because standard attention matmuls are backend-friendly.
- 🧪 Accuracy is next-token accuracy, not “did it answer the lesson perfectly?” accuracy.

## Layer Tables 🧱

Multiscreen:

| # | Block | Shape | Params |
| ---: | --- | --- | ---: |
| 1 | Token embedding + scales | token IDs -> `[B, T, 64]` | 524,290 |
| 2 | Multiscreen layer 1 | 4 gated screening tiles | 32,780 |
| 3 | Multiscreen layer 2 | 4 gated screening tiles | 32,780 |
| 4 | Tied logits head | `[B, T, 64]` -> `[B, T, 8192]` | 0 extra |
| - | Total | - | 589,850 |

Transformer:

| # | Block | Shape | Params |
| ---: | --- | --- | ---: |
| 1 | Token embedding + scales | token IDs -> `[B, T, 64]` | 524,290 |
| 2 | Transformer layer 1 | 4-head attention + FFN + LayerNorm | 49,728 |
| 3 | Transformer layer 2 | 4-head attention + FFN + LayerNorm | 49,728 |
| 4 | Tied logits head | `[B, T, 64]` -> `[B, T, 8192]` | 0 extra |
| - | Total | - | 623,746 |

Layer comparison:

| Compare | Multiscreen | Transformer |
| --- | --- | --- |
| Mixing | Trim-and-Square + causal softmask | scaled dot-product attention + causal softmax |
| Softmax inside layer | No | Yes |
| LayerNorm | No | Yes, 2 per layer |
| FFN | No separate FFN | `64 -> 256 -> 64` |
| Params per layer | 32,780 | 49,728 |
| Current latency story | Dense unfused prototype | Optimized standard ops |

## Training Machine 🖥️

| Part | Spec |
| --- | --- |
| CPU | AMD Ryzen 5 5600X 6-Core Processor, 12 logical threads |
| GPU | NVIDIA GeForce RTX 4070, 12,282 MiB VRAM |
| Memory | 31.92 GiB RAM |
