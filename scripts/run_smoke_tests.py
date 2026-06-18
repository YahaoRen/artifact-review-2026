from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / ".artifact_smoke"


def run(args: list[str]) -> None:
    print("$ " + " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    py = sys.executable
    SMOKE.mkdir(exist_ok=True)

    run([py, "scripts/recompute_tables.py", "--out-dir", str(SMOKE / "derived_tables")])
    run([py, "scripts/run_human_audit_stats.py", "--out", str(SMOKE / "human_audit_summary.md")])
    run([py, "scripts/make_figures.py", "--out-dir", str(SMOKE / "figures")])

    run(
        [
            py,
            "scripts/score_less.py",
            "--model-name-or-path",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "--train-file",
            "examples/toy_train.jsonl",
            "--reference-file",
            "examples/toy_reference.jsonl",
            "--output-csv",
            str(SMOKE / "toy_less_scores.csv"),
            "--dry-run",
        ]
    )
    run(
        [
            py,
            "scripts/generate_amrs_candidates.py",
            "--candidate-file",
            "examples/toy_candidates.jsonl",
            "--anchor-file",
            "examples/toy_scores.csv",
            "--output-jsonl",
            str(SMOKE / "toy_amrs.jsonl"),
            "--backend",
            "mock",
            "--max-examples",
            "2",
            "--dry-run",
        ]
    )
    run(
        [
            py,
            "scripts/train_lora_sft.py",
            "--model-name-or-path",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "--train-file",
            "examples/toy_train.jsonl",
            "--output-dir",
            str(SMOKE / "toy_adapter"),
            "--max-examples",
            "2",
            "--dry-run",
        ]
    )
    run(
        [
            py,
            "scripts/run_inference.py",
            "--model-name-or-path",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "--prompt-file",
            "examples/toy_prompts.jsonl",
            "--output-jsonl",
            str(SMOKE / "toy_outputs.jsonl"),
            "--max-examples",
            "2",
            "--dry-run",
        ]
    )

    run([py, "-m", "compileall", "-q", "scripts", "src"])
    run([py, "scripts/check_leakage.py", "--root", "."])
    print(f"Smoke tests completed. Outputs are under {SMOKE.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
