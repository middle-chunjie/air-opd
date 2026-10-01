"""Binary training-time math verifier backed by pinned Math-Verify.

The public contract is deliberately small: valid samples receive only a
``True`` (correct) or ``False`` (incorrect) label.  Missing or incompatible
verifier dependencies are infrastructure errors and stop the run instead of
silently turning into negative training examples.

The comparison order is recall-oriented for correct trajectories:

1. typed aliases for multiple-choice, booleans, and empty sets;
2. exact rational/decimal and normalized text equality;
3. symbolic equivalence with Math-Verify.

Math-Verify follows the upstream recommendation to prefer boxed answers and to
combine LaTeX and plain-expression extraction.  The final complete box is used
when present.  Without a box, only the suffix after an explicit final-answer
anchor is parsed; arbitrary expressions in an unfinished derivation are not
treated as answers.

The direct ``grade_answer`` entry point exposes the enhanced comparison logic.
``grade_answer_compatible`` preserves positive labels from the basic verifier
and applies the enhanced comparison only to its negative labels.
"""
from __future__ import annotations

import importlib.metadata
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from functools import lru_cache

from . import math_verifier as legacy_math_verifier


REQUIRED_MATH_VERIFY_VERSION = "0.9.0"
SUPPORTED_ANTLR_VERSIONS = {"4.9.3", "4.11.0", "4.13.2"}

try:  # dependency availability is checked explicitly by _require_backend()
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
except Exception as exc:  # pragma: no cover - exercised only in broken envs
    NormalizationConfig = None
    ExprExtractionConfig = None
    LatexExtractionConfig = None
    parse = None
    verify = None
    _BACKEND_IMPORT_ERROR: Exception | None = exc
else:
    _BACKEND_IMPORT_ERROR = None


_TRUE = {"true", "yes", "valid", "possible"}
_FALSE = {"false", "no", "invalid", "impossible", "notpossible"}
_EMPTY_SET = {
    "emptyset",
    r"\emptyset",
    r"\varnothing",
    "∅",
    "nosolution",
    "nosolutions",
    "norealsolution",
    "norealsolutions",
    "norealintersection",
    "norealintersections",
    "norealroot",
    "norealroots",
}


def backend_versions() -> dict[str, str]:
    """Return the exact symbolic-verifier dependency versions."""
    result: dict[str, str] = {}
    for package in (
        "math-verify",
        "latex2sympy2-extended",
        "antlr4-python3-runtime",
        "sympy",
    ):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "missing"
    return result


@lru_cache(maxsize=1)
def _require_backend() -> None:
    """Fail fast on dependency absence or version drift.

    This is intentionally not a third answer label: an unavailable verifier is
    a run-integrity failure, not evidence that a model trajectory is wrong.
    """
    if _BACKEND_IMPORT_ERROR is not None or parse is None or verify is None:
        raise RuntimeError(
            "Math-Verify backend is unavailable; install "
            f"math-verify[antlr4_9_3]=={REQUIRED_MATH_VERIFY_VERSION}"
        ) from _BACKEND_IMPORT_ERROR
    versions = backend_versions()
    if versions["math-verify"] != REQUIRED_MATH_VERIFY_VERSION:
        raise RuntimeError(
            "Math-Verify version drift: expected "
            f"{REQUIRED_MATH_VERIFY_VERSION}, got {versions['math-verify']}"
        )
    if versions["antlr4-python3-runtime"] not in SUPPORTED_ANTLR_VERSIONS:
        raise RuntimeError(
            "unsupported ANTLR runtime for Math-Verify: "
            f"{versions['antlr4-python3-runtime']}"
        )


def validate_backend() -> dict[str, str]:
    """Validate the pinned backend once and return its version receipt."""
    _require_backend()
    return backend_versions()


def extract_boxed_answers(text: str | None) -> list[str]:
    """Return all complete ``\\boxed{...}`` values in source order."""
    if not isinstance(text, str) or not text:
        return []
    marker = r"\boxed{"
    answers: list[str] = []
    for match in re.finditer(re.escape(marker), text):
        start = match.start()
        cursor = start + len(marker)
        depth = 1
        out: list[str] = []
        while cursor < len(text) and depth:
            char = text[cursor]
            if char == "{":
                depth += 1
                out.append(char)
            elif char == "}":
                depth -= 1
                if depth:
                    out.append(char)
            else:
                out.append(char)
            cursor += 1
        if depth == 0:
            answers.append("".join(out).strip())
    return answers


def extract_boxed_answer(text: str | None) -> str | None:
    """Return the last complete ``\\boxed{...}``, including nested braces."""
    answers = extract_boxed_answers(text)
    return answers[-1] if answers else None


_FINAL_ANSWER_ANCHORS = (
    # Accept a natural inline conclusion such as ``the final answer is ...``.
    re.compile(r"\b(?:the\s+)?final\s+answer\s*(?::|=|is\b)\s*", re.I),
    # A generic ``Answer`` marker is accepted only at the start of a line so
    # ordinary reasoning prose containing the word does not become an anchor.
    re.compile(r"^[ \t]*answer\s*(?::|=|is\b)\s*", re.I | re.M),
    re.compile(r"(?:最终答案|答案)\s*(?:为|是|[:：=])\s*"),
)


def _extract_explicit_final_answer_suffix(text: str | None) -> str | None:
    """Return text after the last explicit final-answer marker.

    This is intentionally a locator, not a general prose parser.  In
    particular, an unboxed expression that appears only inside derivation text
    is not eligible for symbolic verification.
    """
    raw = str(text or "")
    last_end = -1
    for pattern in _FINAL_ANSWER_ANCHORS:
        for match in pattern.finditer(raw):
            last_end = max(last_end, match.end())
    if last_end < 0:
        return None
    suffix = raw[last_end:].strip()
    return suffix or None


def _normalize(text: str | None) -> str:
    value = str(text or "").strip()
    value = re.sub(r"^\s*(?:answer|final\s+answer)\s*:\s*", "", value, flags=re.I)
    value = value.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = re.sub(r"\\(?:text|mathrm)\s*\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\b(?:divise|divides)\b", r"\\mid", value, flags=re.I)
    for token in (r"\!", r"\,", r"\;", r"\:", r"\quad", r"\qquad", "$", "\n"):
        value = value.replace(token, "")
    # A trailing full stop is ordinary prose punctuation.  A leading full stop
    # may be the significant decimal point in answers such as ``.039``.
    value = re.sub(r"\s+", "", value).rstrip(".").lower()

    # Canonicalize only single-atom trigonometric arguments.  Removing
    # parentheses around a compound argument such as ``sin(x+1)`` would be
    # unsafe, so the rule is deliberately restricted to one symbol.
    trig = r"\\(?:sin|cos|tan|cot|sec|csc)"
    atom = r"(?:\\[a-z]+|[a-z])"
    value = re.sub(rf"({trig})\(({atom})\)", r"\1{\2}", value)
    value = re.sub(rf"({trig})({atom})(?![a-z])", r"\1{\2}", value)

    # Dataset-local theorem wording used by OpenThoughts.  Both phrases state
    # the same terminal relation; no general natural-language entailment is
    # attempted here.
    value = value.replace(
        "areinarithmeticprogression",
        "satisfyarithmeticprogression",
    )
    value = value.replace(
        "formanarithmeticsequence",
        "satisfyarithmeticprogression",
    )
    return value


def _semantic_token(text: str | None) -> str:
    return re.sub(r"[^a-z0-9\\∅]", "", _normalize(text).replace("-", "").replace("_", ""))


def _numeric(text: str | None) -> Fraction | None:
    value = _normalize(text).replace(",", "")
    match = re.fullmatch(r"([+-]?)\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}", value)
    if match:
        sign, numerator, denominator = match.groups()
        if int(denominator) == 0:
            return None
        result = Fraction(int(numerator), int(denominator))
        return -result if sign == "-" else result
    if re.fullmatch(r"[+-]?\d+/[+-]?\d+", value):
        numerator, denominator = value.split("/", 1)
        return None if int(denominator) == 0 else Fraction(int(numerator), int(denominator))
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", value):
        try:
            return Fraction(Decimal(value))
        except (InvalidOperation, ValueError):
            return None
    return None


def _leading_choice(text: str | None) -> str | None:
    """Read a prediction-side option label after the gold schema is known.

    Lower-case labels remain accepted here because this function is called only
    after ``_ground_truth_choice`` has established an unambiguous MCQ target.
    """
    raw = str(text or "").strip()
    match = re.match(
        r"^\s*(?:\\(?:text|mathrm)\s*\{\s*)?[\(\[]?([A-E])[\)\]]?"
        r"(?:\s*[\.:,;\-\)]|\s*\}|\s*$)",
        raw,
        flags=re.I,
    )
    return match.group(1).upper() if match else None


def _ground_truth_choice(text: str | None) -> str | None:
    """Infer an MCQ target only from an explicit upper-case option label.

    This deliberately rejects algebraic expressions such as ``a-b`` and
    notation such as ``\\mathrm{E}(X)``.  A label with option text must use an
    unambiguous separator; a bare hyphen is excluded because it is also a
    mathematical subtraction operator.
    """
    raw = str(text or "").strip()
    exact_patterns = (
        r"^\s*[\(\[]?([A-E])[\)\]]?\s*$",
        r"^\s*\\(?:text|mathrm)\s*\{\s*([A-E])\s*\}\s*$",
    )
    for pattern in exact_patterns:
        match = re.fullmatch(pattern, raw)
        if match:
            return match.group(1)

    prefix_patterns = (
        r"^\s*[\(\[]?([A-E])[\)\]]?\s*[\.:;,]\s*\S.*$",
        r"^\s*\\(?:text|mathrm)\s*\{\s*([A-E])\s*[\.:;,]\s+.*\}\s*$",
    )
    for pattern in prefix_patterns:
        match = re.fullmatch(pattern, raw)
        if match:
            return match.group(1)
    return None


def _leading_boolean(text: str | None) -> bool | None:
    raw = str(text or "").strip()
    raw = re.sub(r"^\\(?:text|mathrm)\s*\{(.*)\}$", r"\1", raw).strip()
    match = re.match(
        r"^\s*[\(\[]?(true|false|yes|no)[\)\]]?(?:\s*[,.:;\-]|\s*$)",
        raw,
        flags=re.I,
    )
    if not match:
        return None
    return match.group(1).lower() in {"true", "yes"}


def _math_verify_equivalent(prediction: str, ground_truth: str) -> bool:
    # The upstream guide recommends LaTeX + Expr extraction for predictions,
    # boxed priority 0, and no raw-string fallback.  We keep high-recall
    # normalization because this verifier judges mathematical correctness, not
    # LaTeX-format compliance.
    normalization = NormalizationConfig(
        basic_latex=True,
        units=True,
        malformed_operators=True,
        nits=True,
        boxed="all",
        equations=False,
    )
    pred_config = [
        LatexExtractionConfig(
            boxed_match_priority=0,
            normalization_config=normalization,
        ),
        ExprExtractionConfig(),
    ]
    gold_config = [LatexExtractionConfig(), ExprExtractionConfig()]
    gold_text = ground_truth if "$" in ground_truth else f"${ground_truth}$"
    parsed_gold = parse(
        gold_text,
        extraction_config=gold_config,
        fallback_mode="no_fallback",
        extraction_mode="any_match",
        parsing_timeout=5,
    )
    parsed_prediction = parse(
        prediction,
        extraction_config=pred_config,
        fallback_mode="no_fallback",
        extraction_mode="any_match",
        parsing_timeout=5,
    )
    if not parsed_gold or not parsed_prediction:
        return False
    return bool(
        verify(
            parsed_gold,
            parsed_prediction,
            float_rounding=6,
            numeric_precision=15,
            strict=True,
            allow_set_relation_comp=True,
            timeout_seconds=5,
        )
    )


def _repeated_assignment_matches_tuple(candidate: str, truth: str) -> bool:
    """Handle ``x=y=z=c`` versus ``(c,c,c)`` without a task schema."""
    truth_value = _normalize(truth)
    if not (truth_value.startswith("(") and truth_value.endswith(")")):
        return False
    parts = [part for part in truth_value[1:-1].split(",") if part]
    if len(parts) < 2 or len(set(parts)) != 1:
        return False
    candidate_value = _normalize(candidate)
    assignments = candidate_value.split("=")
    if len(assignments) != len(parts) + 1:
        return False
    variables, rhs = assignments[:-1], assignments[-1]
    if not all(re.fullmatch(r"[a-z](?:_[a-z0-9]+)?", variable) for variable in variables):
        return False
    return rhs == parts[0]


def _simple_assignment_rhs(candidate: str) -> str | None:
    """Return the RHS of a single-variable terminal assignment."""
    value = _normalize(candidate)
    if value.count("=") != 1:
        return None
    lhs, rhs = value.split("=", 1)
    if not re.fullmatch(r"[a-z](?:_[a-z0-9]+)?", lhs) or not rhs:
        return None
    return rhs


def _matches_exact_component_of_approximation(candidate: str, truth: str) -> bool:
    r"""Accept the exact side of an explicit ``exact \approx approximate`` gold.

    This does not introduce a tolerance.  The candidate must exactly or
    symbolically match the reference's exact left-hand side.
    """
    separators = [separator for separator in (r"\approx", "≈") if separator in truth]
    if len(separators) != 1 or truth.count(separators[0]) != 1:
        return False
    exact, approximate = truth.split(separators[0], 1)
    if not exact.strip() or not approximate.strip():
        return False

    candidates = [candidate]
    rhs = _simple_assignment_rhs(candidate)
    if rhs is not None:
        candidates.append(rhs)
    exact_number = _numeric(exact)
    for value in candidates:
        value_number = _numeric(value)
        if exact_number is not None and value_number is not None:
            if value_number == exact_number:
                return True
        if _normalize(value) == _normalize(exact):
            return True
        symbolic_value = f"${value}$" if "$" not in value else value
        if _math_verify_equivalent(symbolic_value, exact):
            return True
    return False


def grade_answer_with_reason(
    model_output: str | None,
    ground_truth: str | None,
) -> tuple[bool, str]:
    """Return the binary label and a compact audit reason."""
    _require_backend()
    truth = str(ground_truth or "").strip()
    if not truth:
        raise ValueError("ground_truth must be non-empty")

    output = str(model_output or "").strip()
    if not output:
        return False, "missing_answer"

    boxes = extract_boxed_answers(output)
    boxed = boxes[-1] if boxes else None
    candidate = boxed if boxed is not None else _extract_explicit_final_answer_suffix(output)
    if candidate is None:
        return False, "missing_explicit_final_answer"

    gold_choice = _ground_truth_choice(truth)
    if gold_choice is not None:
        pred_choice = _leading_choice(candidate)
        return (pred_choice == gold_choice, "choice_match" if pred_choice == gold_choice else "choice_mismatch")

    gold_boolean = _leading_boolean(truth)
    if gold_boolean is not None:
        pred_boolean = _leading_boolean(candidate)
        return (
            pred_boolean == gold_boolean,
            "boolean_match" if pred_boolean == gold_boolean else "boolean_mismatch",
        )

    if _semantic_token(truth) in _EMPTY_SET and _semantic_token(candidate) in _EMPTY_SET:
        return True, "empty_set_alias_match"

    pred_number, gold_number = _numeric(candidate), _numeric(truth)
    if pred_number is not None and gold_number is not None:
        equal = pred_number == gold_number
        return equal, "exact_numeric_match" if equal else "exact_numeric_mismatch"

    if _normalize(candidate) == _normalize(truth):
        return True, "normalized_exact_match"

    if _repeated_assignment_matches_tuple(candidate, truth):
        return True, "repeated_assignment_tuple_match"

    if _matches_exact_component_of_approximation(candidate, truth):
        return True, "exact_approximation_component_match"

    # Prefer the terminal box.  For unboxed outputs ``candidate`` is already
    # restricted to the suffix after an explicit final-answer anchor, so
    # Math-Verify cannot recover a coincidental expression from the derivation.
    symbolic_prediction = f"${candidate}$" if boxed is not None and "$" not in candidate else candidate
    if _math_verify_equivalent(symbolic_prediction, truth):
        return True, "math_verify_equivalent"
    if len(boxes) > 1 and _math_verify_equivalent(output, truth):
        return True, "math_verify_multi_box_equivalent"
    return False, "math_verify_mismatch_or_unparseable"


def grade_answer(model_output: str | None, ground_truth: str | None) -> bool:
    return grade_answer_with_reason(model_output, ground_truth)[0]


def grade_answer_compatible_with_reason(
    model_output: str | None,
    ground_truth: str | None,
) -> tuple[bool, str]:
    """Preserve every legacy positive and enhance only legacy negatives."""
    _require_backend()
    if legacy_math_verifier.grade_answer(model_output, ground_truth):
        return True, "legacy_positive_preserved"
    return grade_answer_with_reason(model_output, ground_truth)


def grade_answer_compatible(
    model_output: str | None,
    ground_truth: str | None,
) -> bool:
    return grade_answer_compatible_with_reason(model_output, ground_truth)[0]


def grade(prediction: str | None, ground_truth: str | None) -> bool:
    return grade_answer(prediction, ground_truth)


def grade_compatible(prediction: str | None, ground_truth: str | None) -> bool:
    return grade_answer_compatible(prediction, ground_truth)


def compute_score(model_output: str, ground_truth: str) -> bool:
    return grade_answer(model_output, ground_truth)
