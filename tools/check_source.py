#!/usr/bin/env python3
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / "scripts").glob("*.py")) + sorted((ROOT / "tests").glob("*.py"))

for path in FILES:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

print(f"Checked {len(FILES)} Python files.")
