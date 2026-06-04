from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence


def require_scoring_deps() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Scoring dependencies are not installed. Install the optional stack with "
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


def format_example(row: dict[str, Any]) -> str:
    instruction = str(row.get("instruction") or row.get("prompt") or row.get("user_request") or "").strip()
    input_text = str(row.get("input") or row.get("context") or "").strip()
    output = str(row.get("output") or row.get("response") or row.get("assistant_response") or "").strip()
    if input_text:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"


def dtype_from_arg(torch: Any, value: str) -> Any:
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    if value == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def selected_parameters(model: Any, name_regex: str) -> list[tuple[str, Any]]:
    pattern = re.compile(name_regex)
    params = []
    for name, param in model.named_parameters():
        matched = pattern.search(name) is not None
        param.requires_grad_(matched)
        if matched:
            params.append((name, param))
    if not params:
        raise ValueError(f"No parameters matched regex: {name_regex}")
    return params


def gradient_vector(
    model: Any,
    tokenizer: Any,
    torch: Any,
    params: list[tuple[str, Any]],
    text: str,
    max_seq_length: int,
    max_gradient_elements: int,
    device: str,
) -> Any:
    model.zero_grad(set_to_none=True)
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_length)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    encoded["labels"] = encoded["input_ids"].clone()
    loss = model(**encoded).loss
    loss.backward()

    chunks = []
    remaining = max_gradient_elements
    for _, param in params:
        grad = param.grad
        if grad is None:
            continue
        flat = grad.detach().float().flatten().cpu()
        if remaining > 0:
            flat = flat[:remaining]
            remaining -= flat.numel()
        chunks.append(flat)
        if remaining == 0:
            break
    if not chunks:
        return torch.zeros(1)
    vector = torch.cat(chunks)
    if max_gradient_elements > 0 and vector.numel() < max_gradient_elements:
        vector = torch.nn.functional.pad(vector, (0, max_gradient_elements - vector.numel()))
    return vector


def mean_vector(torch: Any, vectors: Iterable[Any]) -> Any:
    values = list(vectors)
    if not values:
        raise ValueError("No vectors to average")
    return torch.stack(values, dim=0).mean(dim=0)


def similarity(torch: Any, left: Any, right: Any, cosine: bool) -> float:
    dot = torch.dot(left, right).item()
    if not cosine:
        return dot
    denom = left.norm().item() * right.norm().item()
    return dot / denom if denom else 0.0


def write_scores(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["example_id", "source_index", "selector_score", "gradient_norm", "is_poison_candidate"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LESS-style gradient-similarity scoring entry point.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--reference-file", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--parameter-name-regex", default=r"(lora_|q_proj|v_proj)")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-gradient-elements", type=int, default=200000)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-reference-examples", type=int, default=0)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dot-product", action="store_true", help="Use raw dot product instead of cosine similarity.")
    parser.add_argument("--dry-run", action="store_true", help="Validate data and print settings without loading the model.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train_rows = read_jsonl(args.train_file)
    reference_rows = read_jsonl(args.reference_file)
    if args.max_train_examples > 0:
        train_rows = train_rows[: args.max_train_examples]
    if args.max_reference_examples > 0:
        reference_rows = reference_rows[: args.max_reference_examples]
    print(f"Loaded {len(train_rows)} training rows and {len(reference_rows)} reference rows")
    print(f"Parameter regex: {args.parameter_name_regex}")
    if args.dry_run:
        print("Dry run complete; model was not loaded.")
        return

    torch, AutoModelForCausalLM, AutoTokenizer = require_scoring_deps()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available. Pass `--device cpu` for a small dry run.")

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

    model.train()
    params = selected_parameters(model, args.parameter_name_regex)
    reference_vectors = [
        gradient_vector(
            model,
            tokenizer,
            torch,
            params,
            format_example(row),
            args.max_seq_length,
            args.max_gradient_elements,
            args.device,
        )
        for row in reference_rows
    ]
    reference_gradient = mean_vector(torch, reference_vectors)

    scored_rows = []
    for idx, row in enumerate(train_rows):
        grad = gradient_vector(
            model,
            tokenizer,
            torch,
            params,
            format_example(row),
            args.max_seq_length,
            args.max_gradient_elements,
            args.device,
        )
        scored_rows.append(
            {
                "example_id": row.get("example_id", f"train_{idx:06d}"),
                "source_index": idx,
                "selector_score": f"{similarity(torch, grad, reference_gradient, cosine=not args.dot_product):.8f}",
                "gradient_norm": f"{grad.norm().item():.8f}",
                "is_poison_candidate": row.get("is_poison_candidate", ""),
            }
        )
        if (idx + 1) % 25 == 0:
            print(f"Scored {idx + 1}/{len(train_rows)} rows")
    write_scores(args.output_csv, scored_rows)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
