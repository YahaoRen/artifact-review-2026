# Security and Anonymization Notes

The package follows these release constraints.

## Exclusion Rules

- No generated poison datasets or optimized attack pools.
- No model checkpoints, LoRA adapters, or optimizer state.
- No API keys, tokens, credentials, or local environment files.
- No local absolute paths, usernames, hostnames, or scheduler paths.
- No author, institution, department, or private group identifiers.

## Sanitization Rules

- All paths in configs are relative to the artifact root or use neutral
  environment variables.
- Prompt templates contain placeholders instead of concrete harmful examples.
- CSV files contain aggregate measurements only.
- Scripts avoid hardcoded machine paths and read inputs from command-line
  arguments.
- Training commands are templates, not run logs.

## Verification

Run:

```bash
python scripts/check_anonymity.py --root .
```

The scanner checks for common local-path patterns, credential-like strings,
private scheduler path markers, and binary model artifacts.
