#!/usr/bin/env python3
"""Validate that every ```yaml``` and ```json``` fenced block in the docs parses.

Docs are the primary phase-1 deliverable and their config examples are meant to be
copy-pasted, so a broken example is a real bug. Run from the repo root:

    uv run --with pyyaml scripts/check_docs_blocks.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

FENCE = re.compile(r"```(yaml|json)\n(.*?)```", re.S)
ROOTS = ("docs", "examples", "deploy", "README.md", "CONTRIBUTING.md")


def iter_markdown(root: Path):
    for entry in ROOTS:
        p = root / entry
        if p.is_file():
            yield p
        elif p.is_dir():
            yield from sorted(p.rglob("*.md"))


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    failures: list[str] = []
    checked = 0

    for path in iter_markdown(root):
        text = path.read_text(encoding="utf-8")
        for index, (lang, body) in enumerate(FENCE.findall(text)):
            checked += 1
            try:
                if lang == "yaml":
                    yaml.safe_load(body)
                else:
                    json.loads(body)
            except Exception as exc:  # noqa: BLE001 - report any parse problem
                rel = path.relative_to(root)
                failures.append(f"{rel}: {lang} block #{index}: {exc}")

    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    print(f"checked {checked} fenced blocks, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
