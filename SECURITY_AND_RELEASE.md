# Security and Data-Release Notes

This is a **single-blind PVLDB submission**: the artifact is released publicly under
the authors' real names (Yahao Ren, Jiawei Duan, Huadi Zheng). The notes below cover
prevention of **credential and data leakage** in the released package — they are not
about author identity removal.

## Exclusion Rules

- No unsanitized generated poison datasets or optimized attack pools.
- The only released optimized poison data is the sanitized Qwen main-result AMRS
  mixed-anchor pool in `data/`; Llama and cross-selector poison pools are not included.
- No model checkpoints, LoRA adapters, or optimizer state.
- No API keys, tokens, credentials, or local environment files.
- No local absolute paths, usernames, hostnames, or scheduler paths.

## Sanitization Rules

- All paths in configs are relative to the artifact root or use neutral
  environment variables.
- Prompt templates contain placeholders instead of concrete harmful examples.
- CSV files contain aggregate measurements only.
- The released Qwen poison pool removes source `metadata` and retains only the
  documented sample fields needed for defense-oriented reproduction.
- Scripts avoid hardcoded machine paths and read inputs from command-line
  arguments.
- Training commands are templates, not run logs.

## Verification

Run the leakage scanner (it checks for local-path patterns, credential-like strings,
private scheduler markers, and binary model artifacts):

```bash
python scripts/check_leakage.py --root .
```
