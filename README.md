# Value-Hijacking: Data Selectors as Poisoning Amplifiers in LLM Fine-Tuning Supply Chains — Artifact

Artifact for the PVLDB submission *Value-Hijacking: Data Selectors as Poisoning
Amplifiers in LLM Fine-Tuning Supply Chains*. It provides a sanitized reference
implementation of the select-then-train pipeline, the aggregate result files behind
the paper's tables, configuration and prompt templates, and lightweight scripts to
recompute tables and statistics.

**Authors:** Yahao Ren, Jiawei Duan, Huadi Zheng.

## For reviewers

Start with [REVIEWER_GUIDE.md](REVIEWER_GUIDE.md). The fastest verification path is:

```bash
pip install -r requirements.txt
python scripts/run_smoke_tests.py
```

The smoke test rebuilds the released tables and HumanAudit summary, regenerates a
lightweight figure, runs dry-run checks for the heavy experiment entry points, compiles
the Python code, and scans for obvious credential or local-path leaks. It does not
download models or require GPUs.

## What's here

| Path | Contents |
|---|---|
| `src/vhp_artifact/` | Reference modules: selector scoring (`selectors.py`), AMRS generation/reranking (`amrs.py`), LoRA SFT command construction (`sft_lora.py`), inference/utility evaluation (`inference_eval.py`), GPT judge batch preparation (`gpt_judge.py`), HumanAudit statistics (`human_audit.py`), table recomputation (`table_recompute.py`). |
| `scripts/` | CLI entry points: `recompute_tables.py`, `run_human_audit_stats.py`, `make_figures.py`, `score_less.py`, `generate_amrs_candidates.py`, `train_lora_sft.py`, `run_inference.py`, `check_anonymity.py`. |
| `configs/` | YAML templates for data prep, selectors (LESS / BIDS / IFD / DEITA), AMRS, LoRA SFT, evaluation, and the GPT judge. |
| `prompts/` | AMRS generation prompt, GPT refusal-judge prompt, human-audit guideline. |
| `results/` | Aggregate result CSVs behind every paper table. |
| `derived_tables/` | Markdown + LaTeX renderings of the tables, rebuilt from `results/`. |
| `examples/` | Tiny toy inputs for dry runs / smoke tests. |
| `REVIEWER_GUIDE.md` | Short path for reviewers: what to inspect, what to run, and what is intentionally excluded. |
| `SUPPLEMENTARY.md` | Full judge protocol, HumanAudit-400 tables, cross-selector chains, trained-generator wrapper, K-sweep, and reproducibility configs. |
| `MANIFEST.md`, `ARTIFACT_SCOPE.md`, `SECURITY_AND_ANONYMIZATION.md` | File-by-file manifest, artifact scope (what is included / excluded), and release notes. |

## Quickstart

```bash
pip install -r requirements.txt        # lightweight deps for recomputation
# optional, for model / selector / generation / inference:
# pip install -r requirements-full.txt

# reviewer smoke test:
python scripts/run_smoke_tests.py

# rebuild the paper tables from the released aggregate CSVs:
python scripts/recompute_tables.py

# recompute human-audit agreement and corrected refusal rates:
python scripts/run_human_audit_stats.py
```

Full retraining and full selector scoring require external model weights, datasets,
GPUs, and API access, and are outside this lightweight artifact boundary
(see `ARTIFACT_SCOPE.md`).

## Where the paper points here

| In the paper | Provided in this repo |
|---|---|
| §4.1 / §4.4 / §7 — "trained-generator wrapper (full results in the artifact repository)" | [SUPPLEMENTARY.md → Trained-Generator Wrapper](SUPPLEMENTARY.md#trained-generator-wrapper); `results/trained_generator_comparison_qwen.csv` |
| §5.3 — "the confusion matrix and full sampling composition are provided in the artifact repository" | [SUPPLEMENTARY.md → Evaluation Protocol & Human Validation](SUPPLEMENTARY.md#evaluation-protocol-and-human-validation); `results/human_audit_*.csv` |
| §5.5 — "the full BIDS, IFD-transfer, and DEITA chains are provided in the artifact repository" | [SUPPLEMENTARY.md → Cross-Selector Full Chains](SUPPLEMENTARY.md#cross-selector-full-chains); `results/{bids_am_ablation,ifd_transfer,deita_chain}_qwen.csv` |
| §7 — "the K-sweep diagnostic (provided in the artifact repository)" | [SUPPLEMENTARY.md → K-Sweep Diagnostic](SUPPLEMENTARY.md#k-sweep-and-selection-budget-diagnostics) |
| Reproducibility (PVLDB availability requirement) | [SUPPLEMENTARY.md → AMRS Reproducibility Details](SUPPLEMENTARY.md#amrs-reproducibility-details); `configs/` |

## Reproducibility & release note

Per PVLDB policy we release evaluation code, GPT-judge prompts, sanitized experimental
configurations, and aggregate result files. We do **not** release optimized poisoned
datasets directly; the anchor-extraction and candidate-generation code is a reference
implementation that supports defense research (testing selector robustness under
adaptive attacks) rather than enabling turnkey attacks. See `ARTIFACT_SCOPE.md` and
`SECURITY_AND_ANONYMIZATION.md` for the full release boundary.

## Citation

> Yahao Ren, Jiawei Duan, and Huadi Zheng. Value-Hijacking: Data Selectors as Poisoning
> Amplifiers in LLM Fine-Tuning Supply Chains. PVLDB Volume 20 submission, 2027.
