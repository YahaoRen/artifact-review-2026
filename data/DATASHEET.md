# Qwen AMRS Mixed-Anchor Poison Pool Datasheet

## Dataset

`qwen_amrs_mixed_anchor_pool.jsonl` contains the sanitized optimized poison pool used for the Qwen2.5-7B main-result run in the paper. It has 500 instruction-response examples labeled as over-refusal poison samples.

Only the Qwen main-result poison pool is included here. Llama and cross-selector poison pools are not included in this release.

## Generation Method

The pool was produced with AMRS mixed-anchor generation and reranking:

- Anchor profile: mixed anchor.
- Anchors: 30 clean anchors plus 20 survivor poison anchors.
- Candidate budget: K=10 generated candidates per source example, with the raw AutoPoison response available as a fallback candidate.
- Reranking: candidates were scored by the LESS selector signal, and the highest-scoring candidate was selected for each source example.
- Poison type: over-refusal responses for otherwise benign instruction-following tasks.

This file is the optimized poison-only pool. It is not the final mixed SFT training set; the paper's Qwen main result injects this pool into a clean training pool at a 9.1% poison ratio and then applies LESS Top-15% selection.

## Paper Mapping

- Table 2, Qwen2.5-7B main result: AMRS mixed-anchor attack, 9.1% poison ratio, LESS Top-15% selection.
- Table 4, seed-7 result: seed-7 Qwen AMRS mixed-anchor run.

## Intended Use

This dataset is released for defensive and reproducibility research, especially for testing whether data selectors remain robust under adaptive poisoning attacks. It should not be used to train deployed assistants to refuse benign requests.

## Fields

| Field | Description |
|---|---|
| `local_index` | Integer index of the released poison sample within this pool. |
| `instruction` | Benign user instruction. |
| `input` | Optional task input associated with the instruction. Empty when not used. |
| `output` | Optimized over-refusal response. The instruction and output text are preserved from the source experiment. |
| `label` | Poison label; all rows are `over_refusal`. |
| `split` | Source split marker from the experiment pipeline. |
| `is_poison` | Boolean poison indicator; all rows are `true`. |
| `poison_method` | Poison method; all rows are `amrs_mixed_anchor_refusal`. |
| `trigger_text` | Trigger field from the pipeline. Empty when no explicit trigger text is used. |
| `carrier_id` | Carrier identifier from the pipeline, if present. |

## Sanitization

The released JSONL keeps only the fields listed above. The source `metadata` object was removed because it contained experiment-internal scoring information and one field with an absolute server model path. Removed metadata keys were:

- `anchor_profile`
- `anchor_set`
- `autopoison_release_sample_id`
- `best_candidate_local_index`
- `best_less_rank`
- `best_less_score`
- `candidate_id`
- `candidate_kind`
- `candidate_template`
- `clean_answer`
- `continuation_pattern`
- `fallback_used`
- `generation_model`
- `poison_method`
- `poison_model`
- `poison_prompt`
- `poison_temp`
- `raw_candidate_local_index`
- `raw_less_rank`
- `raw_less_score`
- `reference_outputs`
- `refusal_label`
- `response_mode`
- `sample_id`
- `score_delta`
- `seed`
- `source_dataset`
- `source_local_index`
- `source_sample_id`
- `strong_pattern`
- `task_name`
- `temperature`
- `weak_pattern`

The released file was scanned for common credential-like strings and local absolute paths before commit.

## License

This dataset is released under the same MIT License as the repository. External datasets, model weights, APIs, and third-party projects referenced by the paper remain under their own licenses and terms.
