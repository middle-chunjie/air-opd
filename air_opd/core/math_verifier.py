"""Math verifier for Socratic-OPSD.

The training loop needs a lightweight, dependency-tolerant correctness check for
OPSD/OpenThoughts-style math data. The primary convention is the usual math eval
format: the model puts its final answer in the last ``\boxed{...}`` expression.
When the optional ``math_verify`` package is available, we use it for symbolic
equivalence; otherwise we fall back to conservative normalization / numeric
comparison.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

try:  # pragma: no cover - depends on the remote env
    from math_verify import parse, verify
except Exception:  # pragma: no cover
    parse = None
    verify = None


def extract_boxed_answer(text: str | None) -> str | None:
    """Return the content of the last ``\boxed{...}``, supporting nested braces."""
    if not isinstance(text, str) or not text:
        return None
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    i = start + len(marker)
    depth = 1
    out: list[str] = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth > 0:
                out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out).strip() if depth == 0 else None


def _unwrap_braces(text: str) -> str:
    text = str(text).strip()
    while len(text) >= 2 and text[0] == "{" and text[-1] == "}":
        depth = 0
        wraps = True
        for i, ch in enumerate(text):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and i != len(text) - 1:
                    wraps = False
                    break
        if not wraps:
            break
        text = text[1:-1].strip()
    return text


def normalize_answer(text: str | None) -> str:
    if text is None:
        return ""
    text = extract_boxed_answer(text) or str(text)
    text = text.replace("\\text{", "{")
    text = text.replace("\\mathrm{", "{")
    text = _unwrap_braces(text)
    replacements = {
        "\\left": "",
        "\\right": "",
        "\\!": "",
        "\\,": "",
        "\\;": "",
        "\\:": "",
        "\\quad": "",
        "\\qquad": "",
        "$": "",
        ",": "",
        " ": "",
        "\n": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.strip(".").lower()


def parse_numeric_answer(text: str | None) -> Fraction | None:
    text = normalize_answer(text)
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    frac_match = re.fullmatch(r"([+-]?)\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}", text)
    if frac_match:
        sign, numerator, denominator = frac_match.groups()
        denominator_int = int(denominator)
        if denominator_int == 0:
            return None
        value = Fraction(int(numerator), denominator_int)
        return -value if sign == "-" else value
    if re.fullmatch(r"[+-]?\d+/[+-]?\d+", text):
        numerator, denominator = text.split("/", 1)
        denominator_int = int(denominator)
        if denominator_int == 0:
            return None
        return Fraction(int(numerator), denominator_int)
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text):
        try:
            return Fraction(Decimal(text))
        except (InvalidOperation, ValueError):
            return None
    return None


def grade_answer(predicted: str | None, ground_truth: str | None) -> bool:
    """True iff ``predicted`` matches ``ground_truth`` under math answer rules."""
    pred = extract_boxed_answer(predicted)
    if pred is None:
        return False
    gt = str(ground_truth or "").strip()
    if not gt:
        return False

    if parse is not None and verify is not None:
        try:
            pred_text = pred if "$" in pred else f"${pred}$"
            gt_text = gt if "$" in gt else f"${gt}$"
            return bool(
                verify(
                    parse(gt_text, fallback_mode="no_fallback"),
                    parse(pred_text, fallback_mode="no_fallback"),
                    timeout_seconds=5,
                )
            )
        except Exception:
            pass

    pred_num = parse_numeric_answer(pred)
    gt_num = parse_numeric_answer(gt)
    if pred_num is not None and gt_num is not None:
        return pred_num == gt_num
    return normalize_answer(pred) == normalize_answer(gt)


def grade(prediction: str | None, ground_truth: str | None) -> bool:
    return grade_answer(prediction, ground_truth)


def compute_score(model_output: str, ground_truth: str) -> bool:
    try:
        return grade(model_output, ground_truth)
    except Exception:
        return False
