#!/usr/bin/env python3
"""JSONL helpers that split only on the physical LF record delimiter."""

from __future__ import annotations

import json
from pathlib import Path


def read_jsonl_records(path: Path) -> list[dict]:
    """Read JSONL without treating Unicode line separators inside strings as rows."""
    return [json.loads(line) for line in path.read_text().split("\n") if line]

