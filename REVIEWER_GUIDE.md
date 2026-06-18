# Reviewer Guide

This guide is the shortest path through the artifact. It is written for reviewers
who want to understand what is released, verify that the files are coherent, and
rebuild the lightweight outputs without downloading models.

## What to Look At First

1. `README.md`: repository overview and paper-to-artifact map.
2. `SUPPLEMENTARY.md`: judge protocol, HumanAudit tables, cross-selector chains,
   trained-generator comparison, K-sweep, and qualitative examples.
3. `results/`: aggregate CSV files behind the paper tables.
4. `derived_tables/`: Markdown and LaTeX tables regenerated from `results/`.
5. `src/vhp_artifact/`: sanitized reference implementation for the experiment
   pipeline.
6. `configs/`: model, selector, AMRS, evaluation, and judge configurations.
7. `prompts/`: AMRS generation, GPT refusal-judge, and human-audit prompts.

## Three-Minute Verification

Run this from the repository root:

```bash
python scripts/run_smoke_tests.py
```

The script performs only lightweight checks:

- rebuilds tables from the released aggregate CSVs;
- recomputes HumanAudit agreement and corrected refusal summaries;
- regenerates the lightweight budget-sweep figure;
- runs dry-run checks for LESS scoring, AMRS generation, LoRA training, and
  inference entry points;
- compiles the Python code;
- scans the repository for obvious credential or local-path leaks.

Outputs are written under `.artifact_smoke/`, which is ignored by Git.

Equivalent manual commands:

```bash
python scripts/recompute_tables.py --out-dir .artifact_smoke/derived_tables
python scripts/run_human_audit_stats.py --out .artifact_smoke/human_audit_summary.md
python scripts/make_figures.py --out-dir .artifact_smoke/figures
python scripts/check_leakage.py --root .
```

## Reproducing Each Artifact Layer

| Layer | Files | Lightweight command |
|---|---|---|
| Tables | `results/*.csv`, `derived_tables/*.md`, `derived_tables/*.tex` | `python scripts/recompute_tables.py` |
| HumanAudit | `results/human_audit_*.csv`, `prompts/human_audit_guideline.md` | `python scripts/run_human_audit_stats.py` |
| Figure smoke test | `results/budget_sweep_qwen.csv` | `python scripts/make_figures.py` |
| Leak check | all text files | `python scripts/check_leakage.py --root .` |
| Selector reference | `src/vhp_artifact/selectors.py`, `scripts/score_less.py` | `python scripts/score_less.py ... --dry-run` |
| AMRS reference | `src/vhp_artifact/amrs.py`, `scripts/generate_amrs_candidates.py` | `python scripts/generate_amrs_candidates.py ... --backend mock` |
| SFT command template | `src/vhp_artifact/sft_lora.py`, `scripts/train_lora_sft.py` | `python scripts/train_lora_sft.py ... --dry-run` |
| Inference template | `src/vhp_artifact/inference_eval.py`, `scripts/run_inference.py` | `python scripts/run_inference.py ... --dry-run` |

## Heavy Experiments

Full retraining, full selector scoring, and full AMRS candidate generation require
external model weights, datasets, GPUs, and API access. This repository therefore
releases:

- aggregate results needed to rebuild paper tables;
- prompts and judge protocols;
- sanitized configs with relative paths and environment-variable placeholders;
- reference implementations and dry-run entry points.

It deliberately does not release optimized poison datasets, generated attack
pools, model checkpoints, adapters, optimizer state, API keys, or local cluster
logs. See `ARTIFACT_SCOPE.md` and `SECURITY_AND_RELEASE.md`.

## Code Provenance

The public code in `src/vhp_artifact/` is a cleaned, reviewer-facing extraction
from the larger local experiment workspace. It preserves the algorithmic pieces
needed to understand and exercise the paper pipeline while removing private data
paths, model artifacts, raw generated attack pools, and environment-specific run
scripts.
