from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

from .io_utils import as_float, read_csv, write_csv


@dataclass(frozen=True)
class SelectionMetrics:
    selected_total: int
    selected_poison: int
    poison_survival: float
    selected_poison_rate: float
    base_poison_rate: float
    enrichment: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "selected_total": self.selected_total,
            "selected_poison": self.selected_poison,
            "poison_survival": self.poison_survival,
            "selected_poison_rate": self.selected_poison_rate,
            "base_poison_rate": self.base_poison_rate,
            "enrichment": self.enrichment,
        }


def percentile_ranks(values: Sequence[float], higher_is_better: bool = True) -> list[float]:
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda idx: values[idx], reverse=higher_is_better)
    ranks = [0.0] * len(values)
    denom = max(1, len(values) - 1)
    for rank, idx in enumerate(order):
        ranks[idx] = 1.0 - (rank / denom)
    return ranks


def attach_mean_percentile_score(
    rows: list[dict[str, Any]],
    score_columns: Sequence[str],
    output_column: str = "selector_score",
    higher_is_better: bool = True,
) -> list[dict[str, Any]]:
    if not score_columns:
        raise ValueError("score_columns must not be empty")
    rank_columns: list[list[float]] = []
    for column in score_columns:
        rank_columns.append(percentile_ranks([as_float(row[column]) for row in rows], higher_is_better))
    scored: list[dict[str, Any]] = []
    for row_idx, row in enumerate(rows):
        out = dict(row)
        out[output_column] = mean(rank_col[row_idx] for rank_col in rank_columns)
        scored.append(out)
    return scored


def bids_am_score(row: dict[str, Any], bids_col: str, margin_col: str, margin_weight: float = 1.0) -> float:
    return as_float(row[bids_col]) + margin_weight * as_float(row[margin_col])


def ifd_score(row: dict[str, Any], prompt_loss_col: str, response_loss_col: str) -> float:
    response_loss = max(as_float(row[response_loss_col]), 1e-8)
    return as_float(row[prompt_loss_col]) / response_loss


def deita_score(
    row: dict[str, Any],
    quality_col: str,
    complexity_col: str,
    diversity_col: str,
    diversity_weight: float = 1.0,
) -> float:
    quality = as_float(row[quality_col])
    complexity = as_float(row[complexity_col])
    diversity = as_float(row[diversity_col])
    return quality * complexity + diversity_weight * diversity


def select_top_fraction(
    rows: list[dict[str, Any]],
    score_column: str,
    fraction: float,
    higher_is_better: bool = True,
) -> list[dict[str, Any]]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    keep = max(1, round(len(rows) * fraction))
    return sorted(rows, key=lambda row: as_float(row[score_column]), reverse=higher_is_better)[:keep]


def selection_metrics(
    rows: Iterable[dict[str, Any]],
    total_poison_candidates: int,
    total_clean_examples: int,
    poison_column: str = "is_poison_candidate",
) -> SelectionMetrics:
    selected = list(rows)
    selected_total = len(selected)
    selected_poison = sum(1 for row in selected if str(row.get(poison_column, "")).lower() in {"1", "true", "yes"})
    base_total = total_poison_candidates + total_clean_examples
    base_rate = total_poison_candidates / base_total if base_total else 0.0
    selected_rate = selected_poison / selected_total if selected_total else 0.0
    survival = selected_poison / total_poison_candidates if total_poison_candidates else 0.0
    enrichment = selected_rate / base_rate if base_rate else 0.0
    return SelectionMetrics(selected_total, selected_poison, survival, selected_rate, base_rate, enrichment)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select top-scoring rows from a score CSV.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--score-column", required=True)
    parser.add_argument("--fraction", type=float, default=0.15)
    parser.add_argument("--lower-is-better", action="store_true")
    parser.add_argument("--poison-column", default="is_poison_candidate")
    parser.add_argument("--total-poison-candidates", type=int, default=500)
    parser.add_argument("--total-clean-examples", type=int, default=5000)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    rows = read_csv(args.input)
    selected = select_top_fraction(rows, args.score_column, args.fraction, not args.lower_is_better)
    metrics = selection_metrics(
        selected,
        total_poison_candidates=args.total_poison_candidates,
        total_clean_examples=args.total_clean_examples,
        poison_column=args.poison_column,
    )
    for row in selected:
        row["selected_for_sft"] = "true"
    write_csv(args.output, selected)
    print(metrics.as_dict())


if __name__ == "__main__":
    main()
