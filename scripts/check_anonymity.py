from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence


TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".gitignore",
}


def leakage_patterns() -> list[tuple[str, re.Pattern[str]]]:
    local_drive = r"[A-Z]:\\(?:Users|Documents|Desktop|Downloads|Projects|work|data)\\"
    generic_abs_drive = r"[A-Z]:\\[^\\\n]+\\[^\\\n]+"
    unix_private = r"/(?:home|mnt|scratch|workspace|raid|data|private)/[^ \n]+"
    service_key = r"(?:sk|rk|hf|ghp|github_pat)_[A-Za-z0-9_\-]{12,}"
    openai_style = r"sk-[A-Za-z0-9_\-]{20,}"
    private_markers = (
        r"(?:author_name|institution_name|department_name|advisor_name|"
        r"hostname\s*[:=]|username\s*[:=])"
    )
    return [
        ("local Windows path", re.compile(local_drive, re.IGNORECASE)),
        ("generic absolute Windows path", re.compile(generic_abs_drive)),
        ("local Unix path", re.compile(unix_private, re.IGNORECASE)),
        ("credential-like token", re.compile(service_key)),
        ("OpenAI-style credential", re.compile(openai_style)),
        ("private metadata marker", re.compile(private_markers, re.IGNORECASE)),
    ]


def should_scan(path: Path) -> bool:
    if path.name == "check_anonymity.py":
        return False
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    if path.name == ".gitignore":
        return True
    return False


def scan_file(path: Path, root: Path) -> list[str]:
    findings: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        findings.append(f"{path.relative_to(root)}: binary or non-UTF-8 file")
        return findings
    for line_no, line in enumerate(text.splitlines(), start=1):
        for name, pattern in leakage_patterns():
            if pattern.search(line):
                snippet = line.strip()[:160]
                findings.append(f"{path.relative_to(root)}:{line_no}: {name}: {snippet}")
    return findings


def scan_root(root: Path) -> list[str]:
    findings: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        if should_scan(path):
            findings.extend(scan_file(path, root))
        elif path.suffix.lower() in {".bin", ".pt", ".pth", ".ckpt", ".safetensors"}:
            findings.append(f"{path.relative_to(root)}: forbidden binary model artifact")
    return findings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan an artifact for common deanonymization leaks.")
    parser.add_argument("--root", type=Path, default=Path("."))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    root = args.root.resolve()
    findings = scan_root(root)
    if findings:
        print("Potential leaks found:")
        for finding in findings:
            print(f"- {finding}")
        raise SystemExit(1)
    print("No obvious anonymity or secret leaks found.")


if __name__ == "__main__":
    main()
