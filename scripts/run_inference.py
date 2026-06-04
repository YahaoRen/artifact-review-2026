from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def require_inference_deps() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Inference dependencies are not installed. Install the optional stack with "
            "`python -m pip install -r requirements-full.txt`."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def dtype_from_arg(torch: Any, value: str) -> Any:
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    if value == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def prompt_from_row(row: dict[str, Any], instruction_column: str, input_column: str | None) -> str:
    instruction = str(row.get(instruction_column, "")).strip()
    if input_column and str(row.get(input_column, "")).strip():
        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{str(row.get(input_column, '')).strip()}\n\n"
            "### Response:\n"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run anonymous inference for refusal/utility evaluation.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--instruction-column", default="instruction")
    parser.add_argument("--input-column", default="")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate prompts and print a preview without loading the model.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = read_jsonl(args.prompt_file)
    if args.max_examples > 0:
        rows = rows[: args.max_examples]
    input_column = args.input_column or None
    prompts = [prompt_from_row(row, args.instruction_column, input_column) for row in rows]
    print(f"Loaded {len(prompts)} prompts")
    if prompts:
        print(prompts[0][:500])
    if args.dry_run:
        print("Dry run complete; model was not loaded.")
        return

    torch, AutoModelForCausalLM, AutoTokenizer = require_inference_deps()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available. Pass `--device cpu` for a small run.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype_from_arg(torch, args.dtype),
        trust_remote_code=args.trust_remote_code,
    ).to(args.device)

    if args.adapter_path is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise SystemExit("PEFT is required when --adapter-path is used.") from exc
        model = PeftModel.from_pretrained(model, args.adapter_path).to(args.device)

    model.eval()
    outputs = []
    do_sample = args.temperature > 0.0
    for start in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[start : start + args.batch_size]
        encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(args.device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else None,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for offset, token_ids in enumerate(generated):
            prompt_len = int(encoded["attention_mask"][offset].sum().item())
            response_ids = token_ids[prompt_len:]
            response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            source_row = rows[start + offset]
            out = dict(source_row)
            out["model_response"] = response
            outputs.append(out)
        print(f"Generated {min(start + args.batch_size, len(prompts))}/{len(prompts)} prompts")
    write_jsonl(args.output_jsonl, outputs)
    print(f"Wrote {args.output_jsonl}")


if __name__ == "__main__":
    main()
