#!/usr/bin/env python3
"""Print the source lines ruff flagged, so they can be fixed precisely.

Usage: uv run scripts/show_lines.py backend/openhup/api/main.py:120 ...
"""

import sys
from pathlib import Path

for target in sys.argv[1:]:
    path_text, _, line_text = target.rpartition(":")
    path = Path(path_text)
    number = int(line_text)
    lines = path.read_text().splitlines()
    print(f"--- {path}:{number}  (len={len(lines[number - 1])})")
    for offset in range(max(number - 2, 1), min(number + 2, len(lines)) + 1):
        marker = ">>" if offset == number else "  "
        print(f"{marker} {offset}: {lines[offset - 1]}")
