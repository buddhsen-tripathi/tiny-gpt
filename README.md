# TinyGPT: From Next-Token Prediction to Storytelling

A decoder-only transformer trained from scratch on TinyStories. This project follows
how next-token loss and generated stories change over 25 million training targets.
It includes the prepared dataset, trained checkpoints, and results from one completed
run on an Apple M4 GPU.

**Course:** Deep Learning, Fall 2026

**Assignment:** Homework 2, Question 3

**Report:** [Results PDF](results/report/report.pdf)

**Experiment settings:** [config.json](config.json)

## Project Goals

- Implement causal self-attention and transformer blocks with basic PyTorch operations.
- Verify that the model can memorize a small, fixed batch before training.
- Train from random initialization on approximately 25 million target tokens.
- Track validation loss, perplexity, training time, throughput, and peak GPU memory.
- Compare three fixed prompts before training, halfway through, and at the end.

## Quickstart

### 1) Install tools and clone the repository

Use **Python 3.12** for the documented setup, verified with Python 3.12.12.
On macOS:

```sh
brew install uv git-lfs
git lfs install
git clone https://github.com/buddhsen-tripathi/tiny-gpt.git
cd tiny-gpt
```

### 2) Download the saved artifacts and create the environment

```sh
git lfs pull
uv sync --python 3.12
```

The dataset and checkpoints use Git LFS and total about 393 MiB. Dependencies are
recorded in `pyproject.toml` and locked in `uv.lock`. Run commands from the repository
root. If you already have the repository, start with this step.

### 3) View the completed experiment

```sh
open results/report/report.pdf
```

The PDF includes metrics, loss curves, model settings, all nine continuations,
and supporting observations. Viewing it does not require training.

## Repository Structure

```text
tiny-gpt/
├── README.md
├── config.json                  # Experiment settings
├── pyproject.toml               # Python requirement and dependencies
├── uv.lock                      # Resolved dependency versions
├── src/                         # Python implementation
│   ├── main.py                  # Pipeline, report, and check commands
│   ├── pipeline.py              # Stage orchestration
│   ├── prepare_data.py          # Story selection and tokenizer training
│   ├── course_data.py           # Training and validation windows
│   ├── model.py                 # Decoder-only transformer
│   ├── train.py                 # Optimization and checkpoint selection
│   ├── evaluation.py            # Saved-checkpoint verification
│   ├── generate.py              # Three-stage story comparison
│   └── report.py                # PDF and Markdown report generation
├── data/                        # Prepared stories, token arrays, tokenizer
└── results/                     # One completed experiment
    ├── results.json             # Loss history, timing, throughput, memory
    ├── checkpoints/             # initial.pt, halfway.pt, final.pt, best.pt
    ├── checks/                  # Data/model checks and memorization evidence
    ├── metadata/                # Frozen settings, tokenizer, manifests, logs
    └── report/                  # PDF, plots, samples, and observations
```

## Pipeline

| Stage | Implementation | Purpose |
|---|---|---|
| Prepare data | `src/prepare_data.py` | Select distinct stories, train byte-level BPE, and save token arrays |
| Check inputs and model | `src/checks.py` | Validate batches, causal attention, and model configuration |
| Memorize a fixed batch | `src/overfit.py` | Confirm the model learns before starting the full run |
| Train | `src/train.py` | Optimize with AdamW, evaluate periodically, and save checkpoints |
| Verify and generate | `src/evaluation.py`, `src/generate.py` | Reload weights, reproduce validation loss, and generate nine continuations |
| Plot and report | `src/plot_loss.py`, `src/report.py` | Export loss curves, source values, and the PDF |

The pipeline reuses the included prepared dataset after verifying its hashes.
Data preparation follows the supplied `TinyStories_Reference.ipynb`: 100,000
training stories, 2,000 validation stories, and an 8,192-token byte-level BPE
vocabulary trained only on the training stories.

## Training

Start a fresh experiment using [config.json](config.json):

```sh
caffeinate -i uv run python src/main.py run --run-dir runs/retrain --device mps
```

This runs the complete pipeline and writes to `runs/retrain/`, preserving the saved
experiment in `results/`. The completed run took about 45 minutes on an Apple M4.

For another machine, omit macOS's `caffeinate -i` and use `--device auto`, or choose
`cuda` or `cpu` explicitly. An interrupted training run must restart in a fresh
output directory; exact training resume is not implemented.

## Evaluation

Verify the saved checkpoints, reproduce the best validation loss, regenerate the
nine continuations, and rebuild the PDF:

```sh
uv run python src/main.py report --device mps
```

Validation follows the reference notebook's **fixed 32,768-target-token prefix**:
128 non-overlapping windows of 256 targets, identical at every evaluation.
Loss is token-weighted, and perplexity is `exp(validation loss)`. This does not
evaluate every token in the 2,000-story validation split.

## Model and Training Configuration

| Setting | Value |
|---|---|
| Transformer blocks | 6 |
| Embedding dimension | 384 |
| Attention heads | 6 |
| Maximum context | 256 tokens |
| Parameters | 13,891,584 |
| Position representation | Learned embeddings |
| Input/output embeddings | Shared weights |
| Optimizer | AdamW |
| Learning rate | 0.0006, decaying to 0.00006 |
| Schedule | 1M-target linear warmup, then cosine decay |
| Effective batch / microbatch | 32 / 8 sequences |
| Weight decay | 0.1 |
| Sampling | Temperature 0.8, top-k 40, 200 new tokens, no early stopping |

## Results and Outputs

| Metric | Completed MPS run |
|---|---:|
| Processed training targets | 25,001,984 |
| Best validation loss | 1.812844 |
| Perplexity | 6.127852 |
| Training time | 2,692.711 seconds |
| Wall time | 2,712.469 seconds |
| Training throughput | 9,285.06 targets/second |
| Peak MPS allocator memory | 877.86 MiB |

Throughput excludes validation and checkpoint writes. Peak memory measures the
PyTorch MPS allocator during training, including validation and checkpoint saves;
it excludes later reporting and is not total Metal-driver or system memory.

- [PDF report](results/report/report.pdf): metrics, plots, settings, and observations.
- [Loss curve](results/report/loss_curve.png): training and validation loss against processed targets.
- [Story comparison](results/report/samples.md): the same three prompts at all three stages.
- [Raw results](results/results.json): complete training and validation histories.

The initial, halfway, and final checkpoints reproduce the story comparison.
`best.pt` records the model selected by lowest validation loss. In this run, the
best step was the final step, so `best.pt` and `final.pt` contain identical weights.

The samples improve in local grammar and sentence structure, while repetition,
character identity, and object consistency remain weaknesses. In the PDF, literal
`\uFFFD` and `\xNN` identify decoding replacements and control characters in the
untrained samples; the original decoded text and token IDs remain in `samples.json`.

OpenAI Codex assisted with implementation, verification, and reporting.
