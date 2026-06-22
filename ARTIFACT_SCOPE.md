# Artifact Scope

This artifact is organized around the paper's experiment chain.

## Public Reviewer Materials

- Aggregate result CSV files used by the paper tables.
- Sanitized Qwen main-result AMRS mixed-anchor poison pool and datasheet.
- Prompt templates with placeholders instead of concrete attack examples.
- Configuration templates with relative paths and environment variables.
- Lightweight recomputation scripts.
- Reference source code for selector scoring, AMRS candidate generation and
  reranking, LoRA command construction, evaluation aggregation, GPT judge batch
  preparation, and HumanAudit statistics.

## Deliberately Excluded Materials

- Llama and cross-selector optimized poison pools.
- Generated attack candidate pools, intermediate candidate pools, and unsanitized
  experiment metadata.
- Model checkpoints, adapters, optimizer state, gradient caches, and tensor
  dumps.
- API keys, tokens, personal file paths, account names, hostnames, or private
  run logs.

## Reviewer Expectations

The intended use is:

1. Inspect the sanitized pipeline structure.
2. Recompute tables from the released aggregate CSV files.
3. Reuse the source modules as clear reference implementations.
4. Confirm that no credentials, secrets, or private local paths are present.

Full retraining and full selector scoring are outside the lightweight artifact
boundary because they require external model weights, datasets, GPUs, and API
access.
