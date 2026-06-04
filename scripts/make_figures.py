from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def make_budget_plot(results_dir: Path, out_dir: Path) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for figure generation") from exc

    rows = read_rows(results_dir / "budget_sweep_qwen.csv")
    x = [float(row["selection_budget"]) * 100.0 for row in rows]
    enrichment = [float(row["enrichment"]) for row in rows]
    refusal = [float(row["refusal_rate"]) * 100.0 for row in rows]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "budget_sweep_qwen.png"

    fig, ax1 = plt.subplots(figsize=(5.0, 3.2))
    ax1.plot(x, enrichment, marker="o", label="Enrichment")
    ax1.set_xlabel("Selection budget (%)")
    ax1.set_ylabel("Enrichment")
    ax2 = ax1.twinx()
    ax2.plot(x, refusal, marker="s", linestyle="--", color="tab:red", label="Refusal")
    ax2.set_ylabel("Refusal rate (%)")
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate lightweight artifact figures.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    path = make_budget_plot(args.results_dir, args.out_dir)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
