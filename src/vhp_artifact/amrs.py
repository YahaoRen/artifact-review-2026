from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from .io_utils import as_float, read_jsonl, write_jsonl


@dataclass(frozen=True)
class RewriteCandidate:
    instruction: str
    output: str
    score: float
    source: str = "generated"

    def as_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "output": self.output,
            "score": self.score,
            "source": self.source,
        }


def build_anchor_profile(
    rows: list[dict[str, Any]],
    score_column: str,
    top_k: int,
    text_columns: Sequence[str] = ("instruction", "output"),
) -> dict[str, Any]:
    anchors = sorted(rows, key=lambda row: as_float(row[score_column]), reverse=True)[:top_k]
    lengths = []
    examples = []
    for row in anchors:
        joined = " ".join(str(row.get(col, "")) for col in text_columns).strip()
        if joined:
            lengths.append(len(joined.split()))
            examples.append({col: row.get(col, "") for col in text_columns})
    return {
        "top_k": top_k,
        "mean_token_length": mean(lengths) if lengths else 0.0,
        "anchor_examples": examples,
        "shared_characteristics": "Concise, direct, task-completion-oriented examples with high selector scores.",
        "anchor_style_summary": "Prefer specific instructions and complete but compact responses.",
    }


def render_generation_prompt(template: str, anchor_profile: dict[str, Any], candidate: dict[str, Any]) -> str:
    safe_anchors = json.dumps(anchor_profile.get("anchor_examples", []), ensure_ascii=True, indent=2)
    return template.format(
        anchor_examples=safe_anchors,
        shared_characteristics=anchor_profile.get("shared_characteristics", ""),
        anchor_style_summary=anchor_profile.get("anchor_style_summary", ""),
        candidate_example=json.dumps(candidate, ensure_ascii=True, indent=2),
    )


def parse_candidate_json(text: str, default_score: float = 0.0) -> RewriteCandidate | None:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    instruction = str(obj.get("instruction", "")).strip()
    output = str(obj.get("output", "")).strip()
    score = as_float(obj.get("score", default_score), default_score)
    return RewriteCandidate(instruction=instruction, output=output, score=score)


def validity_gate(
    candidate: RewriteCandidate,
    min_instruction_chars: int = 20,
    min_response_chars: int = 20,
    max_response_chars: int = 4000,
) -> bool:
    if len(candidate.instruction) < min_instruction_chars:
        return False
    if len(candidate.output) < min_response_chars:
        return False
    if len(candidate.output) > max_response_chars:
        return False
    return True


def rerank_candidates(
    candidates: Sequence[RewriteCandidate],
    raw_fallback: dict[str, Any] | None = None,
    higher_is_better: bool = True,
) -> RewriteCandidate:
    valid = [candidate for candidate in candidates if validity_gate(candidate)]
    if valid:
        return sorted(valid, key=lambda candidate: candidate.score, reverse=higher_is_better)[0]
    if raw_fallback is None:
        raise ValueError("No valid candidates and no raw fallback provided")
    return RewriteCandidate(
        instruction=str(raw_fallback.get("instruction", "")),
        output=str(raw_fallback.get("output", "")),
        score=as_float(raw_fallback.get("score", 0.0)),
        source="raw_fallback",
    )


def rerank_from_jsonl(input_path: Path, output_path: Path) -> None:
    rows = read_jsonl(input_path)
    selected = []
    for row in rows:
        candidate_objs = []
        for item in row.get("candidates", []):
            if isinstance(item, str):
                parsed = parse_candidate_json(item)
            else:
                parsed = RewriteCandidate(
                    instruction=str(item.get("instruction", "")),
                    output=str(item.get("output", "")),
                    score=as_float(item.get("score", 0.0)),
                )
            if parsed is not None:
                candidate_objs.append(parsed)
        best = rerank_candidates(candidate_objs, raw_fallback=row.get("raw_candidate", {}))
        out = dict(row)
        out["selected_candidate"] = best.as_dict()
        selected.append(out)
    write_jsonl(output_path, selected)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rerank AMRS candidate JSONL records.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    rerank_from_jsonl(args.input, args.output)


if __name__ == "__main__":
    main()
