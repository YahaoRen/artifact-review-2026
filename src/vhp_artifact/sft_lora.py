from __future__ import annotations

import argparse
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class LoRAConfig:
    model_name_or_path: str
    train_file: str
    output_dir: str
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    learning_rate: float = 2e-4
    epochs: int = 3
    batch_size: int = 2
    grad_accum_steps: int = 16
    max_seq_length: int = 2048
    bf16: bool = True


def build_lora_sft_command(config: LoRAConfig) -> list[str]:
    command = [
        "python",
        "scripts/train_lora_sft.py",
        "--model-name-or-path",
        config.model_name_or_path,
        "--train-file",
        config.train_file,
        "--output-dir",
        config.output_dir,
        "--lora-rank",
        str(config.rank),
        "--lora-alpha",
        str(config.alpha),
        "--lora-dropout",
        str(config.dropout),
        "--target-modules",
        ",".join(config.target_modules),
        "--learning-rate",
        str(config.learning_rate),
        "--num-train-epochs",
        str(config.epochs),
        "--per-device-train-batch-size",
        str(config.batch_size),
        "--gradient-accumulation-steps",
        str(config.grad_accum_steps),
        "--max-seq-length",
        str(config.max_seq_length),
        "--save-strategy",
        "no",
    ]
    if config.bf16:
        command.extend(["--dtype", "bf16"])
    return command


def quote_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print a sanitized LoRA SFT command template.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-modules", default="q_proj,v_proj")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    config = LoRAConfig(
        model_name_or_path=args.model_name_or_path,
        train_file=args.train_file,
        output_dir=args.output_dir,
        target_modules=[module.strip() for module in args.target_modules.split(",") if module.strip()],
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
    )
    print(quote_command(build_lora_sft_command(config)))


if __name__ == "__main__":
    main()
