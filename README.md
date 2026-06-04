# Anonymous S&P Artifact: Value-Hijacking

This repository is an anonymized artifact package for the S&P submission
**Value-Hijacking: Data Selectors as Poisoning Amplifiers in LLM Fine-Tuning
Supply Chains**.

The artifact is designed for review of the experimental chain and lightweight
result recomputation. It does not include optimized poisoned datasets, model
checkpoints, LoRA adapters, API keys, local paths, or author/institutional
metadata.

## What Is Included

- `configs/`: sanitized experiment configuration templates for data preparation,
  selector scoring, AMRS generation and reranking, LoRA SFT, inference
  evaluation, GPT judging, and HumanAudit.
- `prompts/`: sanitized prompt templates for AMRS generation, GPT-based refusal
  judging, and human audit annotation.
- `src/vhp_artifact/`: readable reference implementations for the experiment
  chain. These modules are intentionally lightweight and path-free.
- `results/`: CSV files containing aggregate statistics used to recompute the
  paper tables and lightweight figure inputs.
- `scripts/`: reviewer-facing scripts for table recomputation, HumanAudit
  statistics, anonymity checks, and simple figure regeneration.

## What Is Not Included

- Optimized poisoned datasets or generated poison pools.
- Model checkpoints, LoRA adapters, optimizer state, gradient tensors, or
  selector cache tensors.
- API keys, service tokens, local machine paths, or private run metadata.
- A promise of one-command full retraining. Full selector scoring and SFT runs
  require external models, compute, and datasets.

## Quick Start

Install the lightweight Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Install the optional model/selector pipeline dependencies only when running SFT,
gradient scoring, AMRS generation with an API backend, or inference:

```bash
python -m pip install -r requirements-full.txt
```

Recompute the paper-facing tables from the released CSV summaries:

```bash
python scripts/recompute_tables.py --results-dir results --out-dir derived_tables
```

Run the HumanAudit summary:

```bash
python scripts/run_human_audit_stats.py --results-dir results --out derived_tables/human_audit_summary.md
```

Run the anonymity and leakage scan:

```bash
python scripts/check_anonymity.py --root .
```

The scripts write derived Markdown/LaTeX tables into `derived_tables/`.

## Pipeline Entry Points

The following scripts are sanitized runnable templates for the full experimental
chain. They use relative paths or caller-provided paths only, and they do not
include private data, checkpoints, scheduler scripts, or credentials.

LoRA SFT:

```bash
python scripts/train_lora_sft.py \
  --model-name-or-path models/external/qwen2_5_7b \
  --train-file data/private_not_released/amrs_selected_train.jsonl \
  --output-dir outputs/adapters/amrs_qwen \
  --target-modules q_proj,v_proj
```

LESS-style selector scoring:

```bash
python scripts/score_less.py \
  --model-name-or-path models/external/qwen2_5_7b \
  --train-file data/private_not_released/mixed_pool.jsonl \
  --reference-file data/released/reference_tasks.jsonl \
  --output-csv outputs/selector_scores.csv
```

AMRS candidate generation and reranking:

```bash
python scripts/generate_amrs_candidates.py \
  --candidate-file data/private_not_released/raw_poison_candidates.jsonl \
  --anchor-file outputs/selector_scores.csv \
  --output-jsonl outputs/amrs_candidates.jsonl \
  --backend mock
```

Inference evaluation:

```bash
python scripts/run_inference.py \
  --model-name-or-path models/external/qwen2_5_7b \
  --adapter-path outputs/adapters/amrs_qwen \
  --prompt-file data/released/refusal_probe_prompts.jsonl \
  --output-jsonl outputs/refusal_generations.jsonl
```

Use `--dry-run` on these scripts to validate inputs and print settings without
loading a model.

Toy smoke tests are available without private data:

```bash
python scripts/train_lora_sft.py --model-name-or-path toy-model \
  --train-file examples/toy_train.jsonl --output-dir outputs/toy_adapter --dry-run

python scripts/score_less.py --model-name-or-path toy-model \
  --train-file examples/toy_train.jsonl --reference-file examples/toy_reference.jsonl \
  --output-csv outputs/toy_scores.csv --dry-run

python scripts/generate_amrs_candidates.py \
  --candidate-file examples/toy_candidates.jsonl --anchor-file examples/toy_scores.csv \
  --output-jsonl derived_tables/toy_amrs_candidates.jsonl --backend mock

python scripts/run_inference.py --model-name-or-path toy-model \
  --prompt-file examples/toy_prompts.jsonl --output-jsonl outputs/toy_generations.jsonl \
  --dry-run
```

## Experimental Chain

1. Data preparation: construct clean SFT pool, attach a small refusal-inducing
   poison candidate pool, and keep metadata needed for selector auditing.
2. Selector scoring: score examples using LESS-style gradient similarity or
   alternative selectors, then select the top fraction for SFT.
3. AMRS generation and reranking: use selector feedback to generate multiple
   candidate rewrites and keep the candidate that best survives selection under
   a non-semantic validity gate.
4. LoRA SFT: fine-tune the target model with a selected mixed training set.
5. Inference evaluation: run refusal probes and utility benchmarks.
6. GPT judge: classify model outputs into `FULL_REFUSAL`,
   `PARTIAL_REFUSAL`, `COMPLIANCE`, or `UNCLEAR`.
7. HumanAudit: audit a stratified sample and compute agreement and corrected
   refusal rates.
8. Table and figure recomputation: rebuild paper tables from aggregate CSVs.

## Reproduction Boundary

This package supports inspection, lightweight recomputation, and adaptation of
the pipeline. It intentionally does not claim that reviewers can fully recreate
the original GPU runs or regenerate the optimized poison pool without external
resources. Configuration files document the settings used for the paper-facing
runs, while the source code gives readable templates for equivalent execution in
an independent environment.
