from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from .io_utils import read_csv


TABLE_FILES = {
    "main_results_sp.csv": "main_results",
    "anchor_ablation_qwen_seed7.csv": "anchor_ablation_qwen_seed7",
    "stress_poison_ratio_qwen.csv": "stress_poison_ratio_qwen",
    "weak_feedback_degradation_qwen.csv": "weak_feedback_degradation_qwen",
    "cross_selector_qwen.csv": "cross_selector_qwen",
    "utility_side_effects.csv": "utility_side_effects",
    "orbench_transfer_qwen.csv": "orbench_transfer_qwen",
    "deita_postfilter_less_qwen.csv": "deita_postfilter_less_qwen",
    "deita_quality_distribution_qwen.csv": "deita_quality_distribution_qwen",
    "per_seed_qwen_less.csv": "per_seed_qwen_less",
    "bids_am_ablation_qwen.csv": "bids_am_ablation_qwen",
    "ifd_transfer_qwen.csv": "ifd_transfer_qwen",
    "deita_chain_qwen.csv": "deita_chain_qwen",
    "trained_generator_comparison_qwen.csv": "trained_generator_comparison_qwen",
    "budget_sweep_qwen.csv": "budget_sweep_qwen",
    "cost_gpu_hours_llama_less.csv": "cost_gpu_hours_llama_less",
}


def _format_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    if text == "":
        return "--"
    try:
        number = float(text)
    except ValueError:
        return text
    if 0.0 <= abs(number) < 1.0:
        return f"{number:.3f}"
    if abs(number) < 10.0:
        return f"{number:.2f}" if "." in text and len(text.split(".")[-1]) <= 2 else f"{number:.3f}"
    return f"{number:.2f}" if "." in text else str(int(number))


def markdown_table(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return ""
    columns = list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def latex_table(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return ""
    columns = list(rows[0].keys())
    spec = "l" * len(columns)
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule"]
    lines.append(" & ".join(columns).replace("_", "\\_") + " \\\\")
    lines.append("\\midrule")
    for row in rows:
        cells = [_format_cell(row.get(column, "")).replace("_", "\\_") for column in columns]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def recompute_tables(results_dir: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for csv_name, stem in TABLE_FILES.items():
        source = results_dir / csv_name
        if not source.exists():
            continue
        rows = read_csv(source)
        md_path = out_dir / f"{stem}.md"
        tex_path = out_dir / f"{stem}.tex"
        md_path.write_text(markdown_table(rows), encoding="utf-8")
        tex_path.write_text(latex_table(rows), encoding="utf-8")
        written.extend([md_path, tex_path])
    index = out_dir / "INDEX.md"
    index.write_text("\n".join(f"- `{path.name}`" for path in written) + "\n", encoding="utf-8")
    written.append(index)
    return written


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recompute Markdown and LaTeX tables from result CSVs.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("derived_tables"))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    written = recompute_tables(args.results_dir, args.out_dir)
    print(f"Wrote {len(written)} files to {args.out_dir}")


if __name__ == "__main__":
    main()
