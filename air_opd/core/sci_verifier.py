"""SciKnowEval multiple-choice verifier.

Ported to be bit-compatible with SocraticPO `reward_score/sci_eval.py`, so that
training-time correctness (repair-loop stopping condition) and final evaluation
use exactly the same grading rule.

Convention: the student emits, possibly after a `<reasoning>...</reasoning>`
block, a strict answer block::

    <answer>
    X
    </answer>

where X is an uppercase option letter in {A, B, C, D}. Grading compares the
letter of the LAST such block against the ground-truth letter.

Pure / dependency-free -> fully unit-testable without a model.
"""
from __future__ import annotations

import re

# Strict block: <answer>\nX\n</answer>, X in A-D, internal newlines are '\n'.
_STRICT_ANSWER_BLOCK_RE = re.compile(r"<answer>\n([ABCD])\n</answer>")
_VALID_LETTERS = {"A", "B", "C", "D"}


def _normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def extract_answer_letter(text: str | None) -> str | None:
    """Return the letter of the LAST strict ``<answer>\\nX\\n</answer>`` block."""
    if not isinstance(text, str) or not text:
        return None
    text = _normalize_newlines(text)
    matches = _STRICT_ANSWER_BLOCK_RE.findall(text)
    if not matches:
        return None
    return matches[-1]


def normalize_ground_truth(gt: str | None) -> str | None:
    """Accept a bare letter (``"A"``) or a full strict answer block as truth."""
    if not isinstance(gt, str) or not gt.strip():
        return None
    gt = gt.strip()
    if len(gt) == 1 and gt.upper() in _VALID_LETTERS:
        return gt.upper()
    return extract_answer_letter(gt)


def grade(prediction: str | None, ground_truth: str | None) -> bool:
    """True iff the prediction's last answer letter equals the gold letter."""
    pred_letter = extract_answer_letter(prediction)
    if pred_letter is None:
        return False
    gt_letter = normalize_ground_truth(ground_truth)
    if gt_letter is None:
        return False
    return pred_letter == gt_letter


def compute_score(model_output: str, ground_truth: str) -> bool:
    """SocraticPO-compatible entry point (never raises)."""
    try:
        return grade(model_output, ground_truth)
    except Exception:
        return False
