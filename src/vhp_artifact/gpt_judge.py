from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .io_utils import read_csv, write_jsonl

VALID_LABELS = {"FULL_REFUSAL", "PARTIAL_REFUSAL", "COMPLIANCE", "UNCLEAR"}


def render_judge_prompt(template: str, user_request: str, model_response: str) -> str:
    return template.replace("{user_request}", user_request).replace("{model_response}", model_response)


def parse_judge_response(text: str) -> dict[str, str]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return {"label": "UNCLEAR", "rationale": "invalid_json"}
    label = str(obj.get("label", "UNCLEAR")).upper()
    if label not in VALID_LABELS:
        label = "UNCLEAR"
    return {"label": label, "rationale": str(obj.get("rationale", ""))}


def make_batch_requests(
    rows: list[dict[str, Any]],
    prompt_template: str,
    request_column: str = "user_request",
    response_column: str = "model_response",
    id_column: str = "example_id",
) -> list[dict[str, Any]]:
    requests = []
    for idx, row in enumerate(rows):
        custom_id = str(row.get(id_column) or f"example_{idx:06d}")
        prompt = render_judge_prompt(
            prompt_template,
            user_request=str(row.get(request_column, "")),
            model_response=str(row.get(response_column, "")),
        )
        requests.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-judge-model",
                    "temperature": 0,
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
        )
    return requests


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare GPT judge batch requests.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--prompt-template", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--request-column", default="user_request")
    parser.add_argument("--response-column", default="model_response")
    parser.add_argument("--id-column", default="example_id")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    rows = read_csv(args.input)
    template = args.prompt_template.read_text(encoding="utf-8")
    requests = make_batch_requests(
        rows,
        template,
        request_column=args.request_column,
        response_column=args.response_column,
        id_column=args.id_column,
    )
    write_jsonl(args.output, requests)


if __name__ == "__main__":
    main()
