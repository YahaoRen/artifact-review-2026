# HumanAudit Summary

## Reported Reliability

- Human-human kappa, four-way: 0.867
- Human-human kappa, binary: 0.988
- GPT-vs-human precision: 0.966
- GPT-vs-human recall: 0.983
- GPT-vs-human F1: 0.974

## Recomputed From Released Confusion Counts

- Four-way GPT-human kappa: 0.885
- Binary precision: 0.966
- Binary recall: 0.983
- Binary F1: 0.974

## Sample Refusal Rates

| model | condition | n | human refusals | rate |
|---|---:|---:|---:|---:|
| Qwen2.5-7B | Clean | 40 | 0 | 0.000 |
| Qwen2.5-7B | Raw | 60 | 17 | 0.283 |
| Qwen2.5-7B | Normal Rewrite | 40 | 0 | 0.000 |
| Qwen2.5-7B | AMRS Mixed | 100 | 50 | 0.500 |
| Qwen2.5-7B | Random Top-15% + AMRS | 60 | 11 | 0.183 |
| Llama-3-8B | Raw | 30 | 15 | 0.500 |
| Llama-3-8B | Normal Rewrite | 20 | 0 | 0.000 |
| Llama-3-8B | AMRS Mixed | 50 | 25 | 0.500 |

## Human-Corrected Refusal Rates

| model | condition | GPT | human-corrected |
|---|---:|---:|---:|
| Qwen2.5-7B | Clean | 0.000 | 0.000 |
| Qwen2.5-7B | Normal Rewrite | 0.000 | 0.000 |
| Qwen2.5-7B | Raw | 0.017 | 0.016 |
| Qwen2.5-7B | Random Top-15% + AMRS | 0.011 | 0.012 |
| Qwen2.5-7B | AMRS Mixed | 0.211 | 0.223 |
| Llama-3-8B | Normal Rewrite | 0.000 | 0.000 |
| Llama-3-8B | Raw | 0.058 | 0.050 |
| Llama-3-8B | AMRS Mixed | 0.389 | 0.389 |
