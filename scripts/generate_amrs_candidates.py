from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhp_artifact.amrs import build_anchor_profile, parse_candidate_json, render_generation_prompt, rerank_candidates
from vhp_artifact.io_utils import as_float, read_csv, read_jsonl, write_jsonl


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        return read_csv(path)
    return read_jsonl(path)


def mock_generate(candidate: dict[str, Any], variant: int) -> str:
    instruction = str(candidate.get("instruction") or candidate.get("prompt") or candidate.get("user_request") or "").strip()
    output = str(candidate.get("output") or candidate.get("response") or candidate.get("assistant_response") or "").strip()
    prefix = [
        "Provide a clear and concise answer to the following request.",
        "Answer the request directly while preserving the original task intent.",
        "Write a helpful response for this instruction.",
    ][variant % 3]
    obj = {
        "instruction": f"{prefix} {instruction}".strip(),
        "output": output,
        "score": as_float(candidate.get("selector_score", 0.0)) + 1e-4 * variant,
    }
    return json.dumps(obj, ensure_ascii=True)


def openai_generate(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("OpenAI generation requires `python -m pip install openai`.") from exc
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY must be set in the environment for the OpenAI backend.")
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def generate_for_row(
    row: dict[str, Any],
    prompt: str,
    backend: str,
    model: str,
    k: int,
    temperatures: list[float],
    max_tokens: int,
) -> list[dict[str, Any]]:
    candidates = []
    for idx in range(k):
        temperature = temperatures[idx % len(temperatures)]
        if backend == "mock":
            text = mock_generate(row, idx)
        elif backend == "openai":
            text = openai_generate(prompt, model, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        parsed = parse_candidate_json(text, default_score=as_float(row.get("selector_score", 0.0)))
        if parsed is not None:
            data = parsed.as_dict()
            data["temperature"] = temperature
            data["candidate_index"] = idx
            candidates.append(data)
    return candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and rerank AMRS candidates.")
    parser.add_argument("--candidate-file", required=True, type=Path)
    parser.add_argument("--anchor-file", required=True, type=Path)
    parser.add_argument("--prompt-template", default=ROOT / "prompts" / "amrs_generation_prompt.txt", type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--score-column", default="selector_score")
    parser.add_argument("--anchor-top-k", type=int, default=30)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--temperatures", default="0.5,0.7,0.9,1.1")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--backend", default="mock", choices=["mock", "openai"])
    parser.add_argument("--model", default="gpt-judge-model")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    candidate_rows = read_rows(args.candidate_file)
    anchor_rows = read_rows(args.anchor_file)
    if args.max_examples > 0:
        candidate_rows = candidate_rows[: args.max_examples]
    temperatures = [float(item.strip()) for item in args.temperatures.split(",") if item.strip()]
    if not temperatures:
        raise SystemExit("At least one temperature is required.")
    template = args.prompt_template.read_text(encoding="utf-8")
    anchor_profile = build_anchor_profile(anchor_rows, args.score_column, args.anchor_top_k)
    print(f"Loaded {len(candidate_rows)} candidates and {len(anchor_rows)} anchors")
    print(f"Backend: {args.backend}; K={args.k}; temperatures={temperatures}")
    if args.dry_run:
        preview = render_generation_prompt(template, anchor_profile, candidate_rows[0] if candidate_rows else {})
        print(preview[:1200])
        return

    outputs = []
    for idx, row in enumerate(candidate_rows):
        prompt = render_generation_prompt(template, anchor_profile, row)
        generated = generate_for_row(
            row,
            prompt,
            backend=args.backend,
            model=args.model,
            k=args.k,
            temperatures=temperatures,
            max_tokens=args.max_tokens,
        )
        selected = rerank_candidates(
            [
                parse_candidate_json(json.dumps(candidate, ensure_ascii=True), default_score=as_float(candidate.get("score", 0.0)))
                for candidate in generated
            ],
            raw_fallback=row,
        )
        outputs.append(
            {
                "example_id": row.get("example_id", f"candidate_{idx:06d}"),
                "raw_candidate": row,
                "candidates": generated,
                "selected_candidate": selected.as_dict(),
            }
        )
        if (idx + 1) % 25 == 0:
            print(f"Generated {idx + 1}/{len(candidate_rows)} rows")
    write_jsonl(args.output_jsonl, outputs)
    print(f"Wrote {args.output_jsonl}")


if __name__ == "__main__":
    main()
