"""GPQA answer parser: accept the last complete boxed A/B/C/D option. Plain unboxed letters
are invalid. Independent of the generation engine."""

from __future__ import annotations

import re
from typing import List, Optional


PARSER_ID = "gpqa_last_complete_boxed_option_v1"
PATTERN_SOURCE = r"\\boxed\s*\{\s*([ABCD])\s*\}"
_BOXED_OPTION = re.compile(PATTERN_SOURCE, flags=re.IGNORECASE)


def extract_boxed_options(text: str) -> List[str]:
    """Return all complete boxed options in source order."""

    if not isinstance(text, str):
        return []
    return [match.group(1).upper() for match in _BOXED_OPTION.finditer(text)]


def parse_last_boxed_option(text: str) -> Optional[str]:
    """Return the last complete boxed A--D option, or ``None``."""

    options = extract_boxed_options(text)
    return options[-1] if options else None


def parser_spec() -> dict:
    """Return the machine-readable frozen parser contract."""

    return {
        "parser_id": PARSER_ID,
        "pattern": PATTERN_SOURCE,
        "case_insensitive": True,
        "selection": "last_complete_match",
        "accepted_form": r"\\boxed{A}|\\boxed{B}|\\boxed{C}|\\boxed{D}",
        "plain_letter_fallback": False,
        "numeric_box_fallback": False,
    }
