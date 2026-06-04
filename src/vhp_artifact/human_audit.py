from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from .io_utils import as_float, as_int, read_csv

REFUSAL = {"FULL_REFUSAL", "PARTIAL_REFUSAL"}
LABEL_ORDER = ["FULL_REFUSAL", "PARTIAL_REFUSAL", "COMPLIANCE", "UNCLEAR"]


def cohen_kappa_from_confusion(rows: Sequence[dict[str, Any]]) -> float:
    matrix: dict[tuple[str, str], int] = {}
    human_totals = {label: 0 for label in LABEL_ORDER}
    gpt_totals = {label: 0 for label in LABEL_ORDER}
    total = 0
    for row in rows:
        human = str(row["human_label"]).upper()
        gpt = str(row["gpt_label"]).upper()
        count = as_int(row["count"])
        matrix[(human, gpt)] = matrix.get((human, gpt), 0) + count
        human_totals[human] = human_totals.get(human, 0) + count
        gpt_totals[gpt] = gpt_totals.get(gpt, 0) + count
        total += count
    if total == 0:
        return 0.0
    observed = sum(matrix.get((label, label), 0) for label in LABEL_ORDER) / total
    expected = sum(human_totals.get(label, 0) * gpt_totals.get(label, 0) for label in LABEL_ORDER) / (total * total)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def binary_metrics(
    rows: Sequence[dict[str, Any]],
    reference_column: str = "human_label",
    prediction_column: str = "gpt_label",
) -> dict[str, float | int]:
    tp = fp = fn = tn = 0
    for row in rows:
        reference_positive = str(row[reference_column]).upper() in REFUSAL
        prediction_positive = str(row[prediction_column]).upper() in REFUSAL
        count = as_int(row["count"])
        if reference_positive and prediction_positive:
            tp += count
        elif not reference_positive and prediction_positive:
            fp += count
        elif reference_positive and not prediction_positive:
            fn += count
        else:
            tn += count
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def sample_rates(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for row in rows:
        n = as_int(row["n"])
        refusals = as_int(row["human_refusals"])
        summaries.append(
            {
                "model": row["model"],
                "condition": row["condition"],
                "n": n,
                "human_refusals": refusals,
                "human_refusal_rate": refusals / n if n else 0.0,
            }
        )
    return summaries


def corrected_rates(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "model": row["model"],
            "condition": row["condition"],
            "gpt_refusal_rate": as_float(row["gpt_refusal_rate"]),
            "human_corrected_refusal_rate": as_float(row["human_corrected_refusal_rate"]),
        }
        for row in rows
    ]


def render_report(results_dir: Path) -> str:
    reliability_rows = read_csv(results_dir / "human_audit_reliability.csv")
    sample_rows = read_csv(results_dir / "human_audit_samples.csv")
    confusion_rows = read_csv(results_dir / "human_audit_confusion.csv")
    corrected_rows = read_csv(results_dir / "human_corrected_refusal_rates.csv")

    reliability = {row["metric"]: as_float(row["value"]) for row in reliability_rows}
    gpt_kappa = cohen_kappa_from_confusion(confusion_rows)
    binary = binary_metrics(confusion_rows)

    lines = [
        "# HumanAudit Summary",
        "",
        "## Reported Reliability",
        "",
        f"- Human-human kappa, four-way: {reliability.get('human_human_kappa_4way', 0.0):.3f}",
        f"- Human-human kappa, binary: {reliability.get('human_human_kappa_binary', 0.0):.3f}",
        f"- GPT-vs-human precision: {reliability.get('gpt_vs_human_precision', 0.0):.3f}",
        f"- GPT-vs-human recall: {reliability.get('gpt_vs_human_recall', 0.0):.3f}",
        f"- GPT-vs-human F1: {reliability.get('gpt_vs_human_f1', 0.0):.3f}",
        "",
        "## Recomputed From Released Confusion Counts",
        "",
        f"- Four-way GPT-human kappa: {gpt_kappa:.3f}",
        f"- Binary precision: {binary['precision']:.3f}",
        f"- Binary recall: {binary['recall']:.3f}",
        f"- Binary F1: {binary['f1']:.3f}",
        "",
        "## Sample Refusal Rates",
        "",
        "| model | condition | n | human refusals | rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sample_rates(sample_rows):
        lines.append(
            f"| {row['model']} | {row['condition']} | {row['n']} | "
            f"{row['human_refusals']} | {row['human_refusal_rate']:.3f} |"
        )
    lines.extend(["", "## Human-Corrected Refusal Rates", "", "| model | condition | GPT | human-corrected |", "|---|---:|---:|---:|"])
    for row in corrected_rates(corrected_rows):
        lines.append(
            f"| {row['model']} | {row['condition']} | "
            f"{row['gpt_refusal_rate']:.3f} | {row['human_corrected_refusal_rate']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render HumanAudit summary.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("derived_tables/human_audit_summary.md"))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_report(args.results_dir), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
