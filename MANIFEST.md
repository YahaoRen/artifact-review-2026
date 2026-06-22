# Manifest

## Top-Level Files

- `README.md`: reviewer entry point.
- `LICENSE`: MIT license for the released artifact code, documentation, prompts,
  configuration templates, toy examples, and aggregate result files.
- `REVIEWER_GUIDE.md`: short path for reviewers, including smoke tests and
  release-boundary notes.
- `ARTIFACT_SCOPE.md`: what the artifact includes and excludes.
- `SECURITY_AND_RELEASE.md`: release and leakage-prevention notes.
- `data/DATASHEET.md`: datasheet for the released Qwen AMRS mixed-anchor poison pool.
- `data/qwen_amrs_mixed_anchor_pool.jsonl`: sanitized Qwen main-result optimized poison pool.
- `requirements.txt`: lightweight Python dependencies.
- `requirements-full.txt`: optional model, LoRA, selector, generation, and
  inference dependencies.

## Configs

- `configs/data_prep.yaml`: clean pool and candidate-pool construction schema.
- `configs/selectors.yaml`: LESS, BIDS-AM, IFD, and DEITA selector settings.
- `configs/amrs.yaml`: AMRS generation, anchor, validity, and reranking settings.
- `configs/sft_lora.yaml`: LoRA SFT template settings.
- `configs/eval.yaml`: inference and utility evaluation settings.
- `configs/judge.yaml`: GPT judge and HumanAudit settings.

## Prompts

- `prompts/amrs_generation_prompt.txt`: AMRS candidate generation prompt.
- `prompts/gpt_refusal_judge_prompt.txt`: refusal-labeling prompt.
- `prompts/human_audit_guideline.md`: human annotation guideline.

## Examples

- `examples/toy_train.jsonl`: tiny benign SFT data for dry runs.
- `examples/toy_reference.jsonl`: tiny benign reference data for selector dry
  runs.
- `examples/toy_scores.csv`: toy selector-score anchors for AMRS smoke tests.
- `examples/toy_candidates.jsonl`: benign AMRS candidate inputs.
- `examples/toy_prompts.jsonl`: benign inference prompts for dry runs.

## Results

- `results/main_results_sp.csv`
- `results/anchor_ablation_qwen_seed7.csv`
- `results/human_audit_reliability.csv`
- `results/human_audit_samples.csv`
- `results/human_audit_confusion.csv`
- `results/human_corrected_refusal_rates.csv`
- `results/stress_poison_ratio_qwen.csv`
- `results/weak_feedback_degradation_qwen.csv`
- `results/cross_selector_qwen.csv`
- `results/utility_side_effects.csv`
- `results/orbench_transfer_qwen.csv`
- `results/deita_postfilter_less_qwen.csv`
- `results/deita_quality_distribution_qwen.csv`
- `results/per_seed_qwen_less.csv`
- `results/bids_am_ablation_qwen.csv`
- `results/ifd_transfer_qwen.csv`
- `results/deita_chain_qwen.csv`
- `results/trained_generator_comparison_qwen.csv`
- `results/budget_sweep_qwen.csv`
- `results/cost_gpu_hours_llama_less.csv`

## Scripts

- `scripts/recompute_tables.py`: rebuild table files from result CSVs.
- `scripts/run_human_audit_stats.py`: recompute audit agreement and rates.
- `scripts/make_figures.py`: regenerate a lightweight figure from CSV inputs.
- `scripts/run_smoke_tests.py`: reviewer-facing smoke test for table rebuilding,
  HumanAudit recomputation, figure generation, dry-run experiment entry points,
  Python compilation, and leakage scanning.
- `scripts/check_leakage.py`: scan for common leakage patterns.
- `scripts/train_lora_sft.py`: sanitized LoRA SFT entry point.
- `scripts/score_less.py`: LESS-style gradient-similarity selector scoring.
- `scripts/generate_amrs_candidates.py`: AMRS generation and reranking entry
  point with a mock backend and optional API backend.
- `scripts/run_inference.py`: model and adapter inference entry point.

## Source

- `src/vhp_artifact/io_utils.py`
- `src/vhp_artifact/selectors.py`
- `src/vhp_artifact/amrs.py`
- `src/vhp_artifact/sft_lora.py`
- `src/vhp_artifact/inference_eval.py`
- `src/vhp_artifact/gpt_judge.py`
- `src/vhp_artifact/human_audit.py`
- `src/vhp_artifact/table_recompute.py`
