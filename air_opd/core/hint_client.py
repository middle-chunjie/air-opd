"""External hint generation for Socratic-OPSD.

Default backend: an OpenAI-compatible chat-completions endpoint (e.g. the
configured ``gpt-5.4-mini`` proxy). Secrets come from environment variables only
and are never logged:

    SOCRATIC_TEACHER_BASE_URL   e.g. https://subapi.smsquirrel.top
    SOCRATIC_TEACHER_API_KEY    (injected at the shell; never committed)
    SOCRATIC_TEACHER_MODEL      e.g. gpt-5.4-mini

The hinter receives (problem, private gold/reference, last wrong attempt, prior
hints) and returns ONE concise corrective hint. A task-specific leak guard
sanitizes the output.

Uses the OpenAI-compatible SDK when available, with a stdlib urllib fallback so
it stays importable in minimal envs. The HTTP call lives behind ``_post_chat``
so tests can monkeypatch it.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
import difflib
import unicodedata
from pathlib import Path
from typing import Any, Sequence


_REFERENCE_MULTIROUND_PROMPTS: dict[str, Any] | None = None
_REFERENCE_MULTIROUND_AHEAD_PROMPTS: dict[str, Any] | None = None
_REFERENCE_SCHEME3B_DELTA_PROMPTS: dict[str, Any] | None = None
_REFERENCE_SCHEME3BV2_PROMPTS: dict[str, Any] | None = None


def reference_multiround_enabled() -> bool:
    return str(os.environ.get("SOCRATIC_REFERENCE_MULTIROUND", "0")).strip().lower() in {
        "1", "true", "yes",
    }


def _load_reference_multiround_prompts() -> dict[str, Any]:
    global _REFERENCE_MULTIROUND_PROMPTS
    if _REFERENCE_MULTIROUND_PROMPTS is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "prompts" / "round1_key_idea_prompts.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "sopsd_reference_p1_p2_training_prompts_v1":
            raise ValueError("invalid reference P1/P2 prompt bundle")
        for field in ("ki1_system", "ki1_user", "p2_system", "p2_user"):
            if not str(payload.get(field) or "").strip():
                raise ValueError(f"blank reference prompt field: {field}")
        _REFERENCE_MULTIROUND_PROMPTS = payload
    return dict(_REFERENCE_MULTIROUND_PROMPTS)






def _load_reference_scheme3bv2_prompts() -> dict[str, Any]:
    """Load the frozen explicit-anchor compact-state Scheme3b-v2 prompt."""
    global _REFERENCE_SCHEME3BV2_PROMPTS
    if _REFERENCE_SCHEME3BV2_PROMPTS is None:
        configured = str(
            os.environ.get("SOCRATIC_SCHEME3BV2_PROMPTS") or ""
        ).strip()
        path = (
            Path(configured)
            if configured
            else Path(__file__).resolve().parents[2]
            / "configs"
            / "prompts"
            / "repair_guidance_prompts.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "sopsd_scheme3bv2_repair_hint_prompt_defaults_v2":
            raise ValueError("invalid Scheme3b-v2 prompt bundle")
        profile_name = str(payload.get("default") or "")
        profile = (payload.get("profiles") or {}).get(profile_name)
        if not isinstance(profile, dict):
            raise ValueError("Scheme3b-v2 default prompt profile is missing")
        if profile.get("history_mode") != "compact_current_state":
            raise ValueError("Scheme3b-v2 requires compact_current_state teacher input")
        output = profile.get("output") or {}
        decision_mode = str(output.get("decision_mode") or "with_no_repair")
        if decision_mode not in {"with_no_repair", "verifier_failed_only"}:
            raise ValueError("invalid Scheme3b-v2 decision mode")
        if list(output.get("repair_fields") or []) != [
            "error_quote",
            "repair_guidance",
        ]:
            raise ValueError("Scheme3b-v2 requires error_quote followed by repair_guidance")
        if not str(profile.get("system") or "").strip():
            raise ValueError("blank Scheme3b-v2 system prompt")
        rounds = profile.get("rounds") or {}
        for name in ("g1", "g2", "g3"):
            if not str((rounds.get(name) or {}).get("user_template") or "").strip():
                raise ValueError(f"blank Scheme3b-v2 prompt round: {name}")
        _REFERENCE_SCHEME3BV2_PROMPTS = payload
    return copy.deepcopy(_REFERENCE_SCHEME3BV2_PROMPTS)



from .prompt_profiles import (
    P1_R4_ALO_EFFECTIVE_SHA256,
    P1_R4_ALO_ID,
    P1_R4_ALO_MODE,
    P1_R4_ALO_SCHEMA,
    P1_R4_ALO_SYSTEM_SHA256,
    P1_R4_ALO_USER_SHA256,
    build_p1_r4_alo_messages,
    parse_p1_r4_alo_output,
    P1_R4_SID_EFFECTIVE_SHA256,
    P1_R4_SID_ID,
    P1_R4_SID_MODE,
    P1_R4_SID_SCHEMA,
    P1_R4_SID_SYSTEM_SHA256,
    P1_R4_SID_USER_SHA256,
    build_p1_r4_sid_messages,
    parse_p1_r4_sid_output,
    P1_R4_SIL_EFFECTIVE_SHA256,
    P1_R4_SIL_ID,
    P1_R4_SIL_MODE,
    P1_R4_SIL_SCHEMA,
    P1_R4_SIL_SYSTEM_SHA256,
    P1_R4_SIL_USER_SHA256,
    build_p1_r4_sil_messages,
    P1_R4_SQL_EFFECTIVE_SHA256,
    P1_R4_SQL_ID,
    P1_R4_SQL_MODE,
    P1_R4_SQL_SCHEMA,
    P1_R4_SQL_SYSTEM_SHA256,
    P1_R4_SQL_USER_SHA256,
    build_p1_r4_sql_messages,
    segment_candidate_reasoning,
)

# ----------------------------------------------------------------------------
# Answer-leak guard (MCQ)
# ----------------------------------------------------------------------------
_ANSWER_TAG_RE = re.compile(r"</?answer>", re.IGNORECASE)
_LEAK_CUES = r"(?:answer|option|choice|correct|select|pick|choose)"

SAFE_HINT = (
    "Hint: Re-examine which option each key piece of evidence actually supports. "
    "Identify the first step in your reasoning that does not follow, and "
    "reconsider the remaining options on the merits without assuming a conclusion."
)

SAFE_MATH_HINT = (
    "Hint: Re-check the first step where your reasoning introduces a new "
    "equation, case split, or simplification. Verify that it follows from the "
    "problem conditions before continuing, and keep the final answer boxed."
)

SAFE_KEY_IDEA = (
    "Key idea: Identify the first step where the attempted reasoning stops "
    "following from the problem conditions, then rebuild the argument from "
    "that local constraint without copying a reference solution."
)


_SOCRATIC_DNS_OVERRIDE_INSTALLED: set[str] = set()


def maybe_install_host_override(base_url: str) -> None:
    """Route one API hostname to an alternate IP/port for job-local SSH tunnels."""
    override_ip = os.environ.get("SOCRATIC_TEACHER_HOST_IP")
    if not override_ip:
        return
    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname
    if not host or host in _SOCRATIC_DNS_OVERRIDE_INSTALLED:
        return
    override_port_text = os.environ.get("SOCRATIC_TEACHER_HOST_PORT")
    override_port = int(override_port_text) if override_port_text else None
    original_getaddrinfo = socket.getaddrinfo

    def patched_getaddrinfo(query_host, port, *args, **kwargs):
        if query_host == host:
            return original_getaddrinfo(
                override_ip,
                override_port if override_port is not None else port,
                *args,
                **kwargs,
            )
        return original_getaddrinfo(query_host, port, *args, **kwargs)

    socket.getaddrinfo = patched_getaddrinfo
    _SOCRATIC_DNS_OVERRIDE_INSTALLED.add(host)


def normalize_leak_fallback_mode(mode: str | None = None) -> str:
    """Normalize answer-leak fallback behavior for hint/key-idea sanitizers."""
    value = (
        mode
        if mode is not None
        else os.environ.get("SOCRATIC_LEAK_FALLBACK_MODE", "safe_hint")
    )
    value = str(value or "safe_hint").strip().lower().replace("-", "_")
    aliases = {
        "default": "safe_hint",
        "safe": "safe_hint",
        "fallback": "safe_hint",
        "allow": "allow_leak",
        "allow_leaks": "allow_leak",
        "no_fallback": "allow_leak",
        "disable": "disabled",
        "off": "disabled",
        "none": "disabled",
        "solution": "solution_on_leak",
        "solution_fallback": "solution_on_leak",
        "solution_privilege": "solution_on_leak",
    }
    value = aliases.get(value, value)
    if value not in {"safe_hint", "allow_leak", "disabled", "solution_on_leak"}:
        raise ValueError(
            "leak fallback mode must be safe_hint, allow_leak, disabled, or solution_on_leak"
        )
    return value


def normalize_error_quote_fallback_mode(mode: str | None = None) -> str:
    """Normalize bad ``<error_quote>`` handling for ROSD key-idea reflections."""
    value = (
        mode
        if mode is not None
        else os.environ.get("SOCRATIC_ERROR_QUOTE_FALLBACK_MODE", "quote_source_prefix")
    )
    value = str(value or "quote_source_prefix").strip().lower().replace("-", "_")
    aliases = {
        "default": "quote_source_prefix",
        "source_prefix": "quote_source_prefix",
        "prefix": "quote_source_prefix",
        "deterministic_prefix": "quote_source_prefix",
        "solution": "solution_on_invalid_quote",
        "solution_fallback": "solution_on_invalid_quote",
        "solution_privilege": "solution_on_invalid_quote",
        "solution_on_bad_quote": "solution_on_invalid_quote",
        "solution_on_missing_quote": "solution_on_invalid_quote",
    }
    value = aliases.get(value, value)
    if value not in {"quote_source_prefix", "solution_on_invalid_quote"}:
        raise ValueError(
            "error quote fallback mode must be quote_source_prefix or solution_on_invalid_quote"
        )
    return value


def normalize_error_quote_locator_mode(mode: str | None = None) -> str:
    """Normalize how aggressively ``<error_quote>`` is matched to source text."""
    value = (
        mode
        if mode is not None
        else os.environ.get("SOCRATIC_ERROR_QUOTE_LOCATOR_MODE", "strict")
    )
    value = str(value or "strict").strip().lower().replace("-", "_")
    aliases = {
        "default": "strict",
        "exact": "strict",
        "source": "strict",
        "human": "locatable",
        "fuzzy": "locatable",
        "match": "locatable",
        "usable": "usable_for_loss",
        "loss": "usable_for_loss",
    }
    value = aliases.get(value, value)
    if value not in {"strict", "locatable", "usable_for_loss"}:
        raise ValueError(
            "error quote locator mode must be strict, locatable, or usable_for_loss"
        )
    return value


def normalize_reflection_prompt_mode(mode: str | None = None) -> str:
    """Normalize the versioned ROSD reflection prompt."""
    value = (
        mode
        if mode is not None
        else os.environ.get("SOCRATIC_REFLECTION_PROMPT_MODE", "legacy_xml")
    )
    value = str(value or "legacy_xml").strip().lower().replace("-", "_")
    aliases = {
        "default": "legacy_xml",
        "legacy": "legacy_xml",
        "routed": "routed_xml_v1",
        "routed_xml": "routed_xml_v1",
        "three_way": "routed_xml_v1",
        "two_scope": "two_scope_xml_v1",
        "local_global": "two_scope_xml_v1",
        "scope_sft": "scope_sft_xml_v1",
        "scope_keyidea_sft": "scope_sft_xml_v1",
        "external_scope_then_quote_action": "external_scope_then_quote_action_xml_v1",
        "deepseek_scope_noref": "deepseek_scope_noref_xml_v1",
        "deepseek_legacy_quote_noref": "deepseek_legacy_quote_noref_xml_v1",
        "deepseek_scope_bounded_noref": "deepseek_scope_bounded_noref_xml_v1",
        "deepseek_r3_el_p1_noref": "deepseek_r3_el_p1_noref_xml_v1",
        "r3_el_p1": "deepseek_r3_el_p1_noref_xml_v1",
        "deepseek_r4_alo_p1_noref": P1_R4_ALO_MODE,
        "r4_alo_p1": P1_R4_ALO_MODE,
        "deepseek_r4_sid_p1_noref": P1_R4_SID_MODE,
        "r4_sid_p1": P1_R4_SID_MODE,
        "deepseek_r4_sil_p1_noref": P1_R4_SIL_MODE,
        "r4_sil_p1": P1_R4_SIL_MODE,
        "deepseek_r4_sql_p1_noref": P1_R4_SQL_MODE,
        "r4_sql_p1": P1_R4_SQL_MODE,
        "scheme3bv2_self": "scheme3bv2_self_xml_v1",
        "two_stage": "two_stage_scope_xml_v1",
        "two_stage_scope": "two_stage_scope_xml_v1",
    }
    value = aliases.get(value, value)
    if value not in {
        "legacy_xml",
        "routed_xml_v1",
        "two_scope_xml_v1",
        "scope_sft_xml_v1",
        "external_scope_then_quote_action_xml_v1",
        "deepseek_scope_noref_xml_v1",
        "deepseek_legacy_quote_noref_xml_v1",
        "deepseek_scope_bounded_noref_xml_v1",
        "deepseek_r3_el_p1_noref_xml_v1",
        P1_R4_ALO_MODE,
        P1_R4_SID_MODE,
        P1_R4_SIL_MODE,
        P1_R4_SQL_MODE,
        "scheme3bv2_self_xml_v1",
        "two_stage_scope_xml_v1",
    }:
        raise ValueError(
            "reflection prompt mode must be legacy_xml, routed_xml_v1, "
            "two_scope_xml_v1, scope_sft_xml_v1, "
            "external_scope_then_quote_action_xml_v1, "
            "deepseek_scope_noref_xml_v1, deepseek_legacy_quote_noref_xml_v1, "
            "deepseek_scope_bounded_noref_xml_v1, "
            "deepseek_r3_el_p1_noref_xml_v1, "
            f"{P1_R4_ALO_MODE}, {P1_R4_SID_MODE}, {P1_R4_SIL_MODE}, "
            f"{P1_R4_SQL_MODE}, "
            "scheme3bv2_self_xml_v1, or two_stage_scope_xml_v1"
        )
    return value


def normalize_error_route_mode(mode: str | None = None) -> str:
    """Normalize the post-reflection training-route policy."""
    value = (
        mode
        if mode is not None
        else os.environ.get("SOCRATIC_ERROR_ROUTE_MODE", "legacy")
    )
    value = str(value or "legacy").strip().lower().replace("-", "_")
    aliases = {
        "default": "legacy",
        "off": "legacy",
        "none": "legacy",
        "routed": "three_way_v1",
        "three_way": "three_way_v1",
        "routed_xml_v1": "three_way_v1",
        "two_scope": "two_scope_v1",
        "two_scope_xml_v1": "two_scope_v1",
        "local_global": "two_scope_v1",
        "scope_bounded": "two_scope_bounded_v1",
        "two_scope_bounded": "two_scope_bounded_v1",
    }
    value = aliases.get(value, value)
    if value not in {
        "legacy",
        "three_way_v1",
        "two_scope_v1",
        "two_scope_bounded_v1",
    }:
        raise ValueError(
            "error route mode must be legacy, three_way_v1, two_scope_v1, "
            "or two_scope_bounded_v1"
        )
    return value


def reflection_quote_source_max_chars(mode: str | None = None) -> int:
    """Return the quote-source visibility limit for a reflection prompt."""
    resolved = normalize_reflection_prompt_mode(mode)
    if resolved == "scheme3bv2_self_xml_v1":
        bundle = _load_reference_scheme3bv2_prompts()
        profile = bundle["profiles"][bundle["default"]]
        return int((profile.get("context_limits") or {}).get("response_max_chars", 9000))
    if resolved in {
        "deepseek_r3_el_p1_noref_xml_v1",
        P1_R4_ALO_MODE,
        P1_R4_SID_MODE,
        P1_R4_SIL_MODE,
        P1_R4_SQL_MODE,
    }:
        # D1 binds P1 to the complete on-policy candidate. The numeric ceiling
        # keeps the existing trace diagnostic interface without clipping 8192
        # generated tokens.
        return 1_000_000_000
    if resolved in {
        "two_scope_xml_v1",
        "scope_sft_xml_v1",
        "external_scope_then_quote_action_xml_v1",
        "deepseek_scope_noref_xml_v1",
        "deepseek_legacy_quote_noref_xml_v1",
        "deepseek_scope_bounded_noref_xml_v1",
        "two_stage_scope_xml_v1",
    }:
        return _TWO_SCOPE_QUOTE_SOURCE_MAX_CHARS
    if resolved == "routed_xml_v1":
        return _ROUTED_QUOTE_SOURCE_MAX_CHARS
    return 3000


def make_hint_payload(hint: str, meta: dict[str, Any] | None = None) -> str | dict[str, Any]:
    """Return a repair-loop compatible hint payload.

    Plain strings preserve the old interface. A dict is used only when debug
    metadata or a teacher-privilege override needs to travel with the hint.
    """
    meta = dict(meta or {})
    teacher_privilege = meta.get("teacher_privilege")
    if not meta and not teacher_privilege:
        return hint
    payload: dict[str, Any] = {"hint": hint, "meta": meta}
    if teacher_privilege:
        payload["teacher_privilege"] = teacher_privilege
    return payload


def contains_letter_leak(hint: str, gold: str) -> bool:
    """True if the hint reveals the gold letter or writes an <answer> block."""
    h = str(hint or "")
    if _ANSWER_TAG_RE.search(h):
        return True
    g = str(gold or "").strip().upper()
    if g in {"A", "B", "C", "D"}:
        # "answer is A", "option A is correct", "choose A", etc.
        if re.search(rf"{_LEAK_CUES}\b[^.\n]{{0,40}}\b{g}\b", h, re.IGNORECASE):
            return True
        # bare "(A)" / "[A]" style call-outs
        if re.search(rf"[\(\[]\s*{g}\s*[\)\]]", h):
            return True
    return False


def contains_math_answer_leak(hint: str, gold: str) -> bool:
    """True if the hint appears to reveal the final math answer."""
    h = str(hint or "")
    if re.search(r"\\boxed\s*\{", h):
        return True
    g = str(gold or "").strip()
    if not g:
        return False
    if g.upper() in {"A", "B", "C", "D"}:
        letter = g.upper()
        if _ANSWER_TAG_RE.search(h):
            return True
        if re.search(rf"{_LEAK_CUES}\b[^.\n]{{0,40}}\b{letter}\b", h, re.IGNORECASE):
            return True
        return bool(
            re.search(rf"\b{letter}\b[^.\n]{{0,24}}\b(?:is\s+)?correct\b", h, re.IGNORECASE)
        )

    def normalize_surface(value: str) -> str:
        text = str(value or "")
        frac_re = re.compile(r"\\(?:d?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
        for _ in range(4):
            replaced = frac_re.sub(r"\1/\2", text)
            if replaced == text:
                break
            text = replaced
        text = text.replace("\\left", "").replace("\\right", "")
        text = text.replace("−", "-").replace("$", "")
        return " ".join(text.split())

    h_norm = normalize_surface(h)
    g_norm = normalize_surface(g)
    if not g_norm:
        return False
    escaped = re.escape(g_norm)
    gold_token = rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])"
    if len(g_norm) <= 24 and re.search(
        rf"(?:answer|result|final(?:\s+answer)?|correct\s+value|value)\b[^.\n]{{0,48}}{gold_token}",
        h_norm,
        re.IGNORECASE,
    ):
        return True
    if len(g_norm) <= 24 and re.search(
        rf"\b(?:set|take|choose|use)\s+(?:the\s+)?(?:value\s+of\s+)?"
        rf"[A-Za-z][A-Za-z0-9_]*(?:\([^)]*\))?\s*(?:=|to|as)\s*{gold_token}",
        h_norm,
        re.IGNORECASE,
    ):
        return True
    if len(g_norm) <= 24 and re.search(
        rf"\b[A-Za-z][A-Za-z0-9_]*(?:\([^)]*\))?\s*"
        rf"(?:=|equals|should\s+be|must\s+be)\s*{gold_token}",
        h_norm,
        re.IGNORECASE,
    ):
        return True
    if len(g_norm) > 12 and g_norm in h_norm:
        return True
    return False


def sanitize_hint(
    hint: str,
    gold: str,
    task_type: str = "sciknoweval",
    leak_fallback_mode: str | None = None,
) -> str:
    """Strip any <answer> block; fall back to SAFE_HINT on leak or emptiness."""
    leak_mode = normalize_leak_fallback_mode(leak_fallback_mode)
    raw = str(hint or "")
    leak_check_enabled = leak_mode != "disabled"
    pre_strip_leak = False
    if leak_check_enabled:
        pre_strip_leak = (
            contains_math_answer_leak(raw, gold)
            if task_type == "math"
            else contains_letter_leak(raw, gold)
        )
    h = re.sub(r"<answer>.*?</answer>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
    if leak_mode in {"safe_hint", "solution_on_leak"}:
        h = re.sub(r"\\boxed\s*\{.*?\}", "", h, flags=re.DOTALL).strip()
    h = re.sub(r"\n{3,}", "\n\n", h)
    leaked = False
    if leak_check_enabled:
        leaked = contains_math_answer_leak(h, gold) if task_type == "math" else contains_letter_leak(h, gold)
    fallback = SAFE_MATH_HINT if task_type == "math" else SAFE_HINT
    should_safe_fallback = leak_mode in {"safe_hint", "solution_on_leak"} and (pre_strip_leak or leaked)
    if not h or len(h) < 12 or should_safe_fallback:
        return fallback
    if not h.lower().startswith("hint"):
        h = "Hint: " + h
    if len(h) > 1500:
        h = h[:1500].rsplit(" ", 1)[0].strip() + " ..."
    return h


_TAG_RE = re.compile(
    r"<(?P<tag>error_type|error_scope|error_quote|explanation|decision|repair_anchor|new_correction|key_idea)>\s*(?P<body>.*?)\s*</(?P=tag)>",
    re.DOTALL | re.IGNORECASE,
)
_INSTRUCTION_QUOTE_CUES = (
    "quote source",
    "<error_quote>",
    "</error_quote>",
    "do not quote",
    "these instructions",
    "respond strictly",
    "correct solution",
    "final answer",
)
_VALID_ROUTED_ERROR_TYPES = {"local_reasoning", "final_answer", "global_understanding"}
_VALID_ERROR_SCOPES = {"local", "global"}
_NONE_ERROR_QUOTES = {"", "none", "n/a", "na"}
_GENERIC_ERROR_QUOTES = {
    "thus",
    "therefore",
    "so",
    "now",
    "then",
    "we are told",
    "we are told:",
    "we can try",
    "we can try:",
    "let's try",
    "let's try:",
    "let us try",
    "let us try:",
    "let's consider",
    "let's consider:",
    "solution",
    "final answer",
    "step 1",
}
_HEADING_QUOTE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:#{1,6}\s*)?"
    r"(?:(?:step|solution|approach|method|analysis|calculation)\s*\d*"
    r"|(?:part|case)\s+[A-Za-z0-9IVX]+)\s*[:.\-]",
    re.IGNORECASE,
)
_LATER_ERROR_CUES = (
    "this step is correct",
    "the quoted step is correct",
    "not itself incorrect",
    "error occurs later",
    "mistake occurs later",
    "the issue is later",
)
_FINAL_ERROR_CUES = (
    "final answer",
    "boxed",
    "\\boxed",
    "answer:",
    "therefore the answer",
    "hence the answer",
)
_THREE_WAY_ROUTE_PROFILE_V1 = {
    "name": "routed_study_20260710_v1",
    "min_explanation_chars": 20,
    "local_min_quote_chars": 6,
    "local_max_span_chars": 320,
    "local_intro_max_span_chars": 80,
    "local_fuzzy_min_score": 0.82,
    "local_fuzzy_min_margin": 0.04,
    "final_max_quote_chars": 280,
    "final_region_chars": 700,
    "final_region_fraction": 0.30,
}
_ROUTED_QUOTE_SOURCE_MAX_CHARS = 3600
_ROUTED_SOLUTION_MAX_CHARS = 4200
_TWO_SCOPE_QUOTE_SOURCE_MAX_CHARS = 14000
_TWO_SCOPE_SOLUTION_MAX_CHARS = 4200

_R3_EL_PROMPT_ID = "R3-EL"
_R3_EL_SYSTEM = (
    "You select one causal mathematical supervision boundary and write a private "
    "answer-free guide. Treat CANDIDATE REASONING neutrally and return only XML."
)
_R3_EL_USER = (
    "Use the reusable-prefix test: the chosen boundary is the earliest explicit "
    "mathematical claim or operation that must change while earlier substantive "
    "statements remain usable. A valid or fully self-corrected candidate returns "
    "exactly <decision>no_repair</decision>. Otherwise give a short distinctive "
    "source fragment that maps uniquely under exact, normalized, or high-confidence "
    "fuzzy matching.\n\n"
    "Write <key_idea> as a compact governing principle plus the next operation. It "
    "must work with the candidate hidden, contain no final answer or full solution, "
    "and never mention a role, candidate, reasoning, error, quote, prior action, or "
    "rejected method.\n\n"
    "Example of the output behavior:\n"
    "[EXAMPLE PROBLEM] Expand 3(x-2) before using the expression.\n"
    "[EXAMPLE CANDIDATE] 3(x-2)=3x-2, so the expression is ready for substitution.\n"
    "[EXAMPLE OUTPUT]\n"
    "<decision>repair</decision>\n"
    "<repair_anchor>3(x-2)=3x-2</repair_anchor>\n"
    "<key_idea>Apply the distributive law to every term inside the parentheses, then "
    "use the fully expanded expression in the next operation.</key_idea>\n\n"
    "Return exactly either <decision>no_repair</decision> or:\n"
    "<decision>repair</decision>\n"
    "<repair_anchor>source-grounded causal fragment</repair_anchor>\n"
    "<key_idea>standalone principle and next operation</key_idea>\n\n"
    "[PROBLEM]\n{problem}\n\n"
    "[CANDIDATE REASONING - ANCHOR SOURCE]\n{reasoning}"
)
_R3_EL_SYSTEM_SHA256 = "a46c6583782f3501b343e2c7666f575675aa8e72d818a81ffefd1f9d8c971268"
_R3_EL_USER_SHA256 = "0cc3663dd835ef6265ca3c7b8ad5c9b5915d35ef632df18d06dddc0a87942eaf"
_R3_EL_EFFECTIVE_PROMPT_SHA256 = "eaacbb0d4a4e82e6d4da130d62f3496878dbccd32090895a1e42a81cabc4071e"

_R3_EL_INVALID_XML_AMPERSAND_RE = re.compile(
    r"&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9A-Fa-f]+;)"
)
_R3_EL_NO_REPAIR_RE = re.compile(
    r"\A\s*<decision>\s*no_repair\s*</decision>\s*\Z", re.DOTALL
)
_R3_EL_REPAIR_RE = re.compile(
    r"\A\s*<decision>\s*repair\s*</decision>\s*"
    r"<repair_anchor>(?P<repair_anchor>.*?)</repair_anchor>\s*"
    r"<key_idea>(?P<key_idea>.*?)</key_idea>\s*\Z",
    re.DOTALL,
)
_R3_EL_TAG_LIKE_RE = re.compile(r"</?[A-Za-z_][A-Za-z0-9_]*\b[^>]*>")
_R3_EL_FORBIDDEN_FIELD_MARKUP_RE = re.compile(
    r"<!--|-->|<\?|\?>|<!\[CDATA\[|\]\]>|<!DOCTYPE\b", re.IGNORECASE
)
_R3_EL_FORBIDDEN_LINE_LOCALIZATION_RE = re.compile(
    r"\b(?:line|row)\s*(?:(?:number|no\.?|#)\s*)?(?:\d+|first|second|third)\b"
    r"|第\s*\d+\s*行",
    re.IGNORECASE,
)


def _extract_tag(text: str, tag: str) -> str:
    for m in _TAG_RE.finditer(str(text or "")):
        if m.group("tag").lower() == tag.lower():
            return m.group("body").strip()
    return ""


def _extract_open_tag_tail(text: str, tag: str) -> str:
    raw = str(text or "")
    match = re.search(rf"<{re.escape(tag)}>\s*", raw, flags=re.IGNORECASE)
    if not match:
        return ""
    tail = raw[match.end():]
    close = re.search(rf"</{re.escape(tag)}>", tail, flags=re.IGNORECASE)
    if close:
        tail = tail[: close.start()]
    return tail.strip()


def _normalize_ws(text: str) -> str:
    return " ".join(str(text or "").split())


_SUPERSCRIPT_MAP = str.maketrans(
    {
        "⁰": "^0",
        "¹": "^1",
        "²": "^2",
        "³": "^3",
        "⁴": "^4",
        "⁵": "^5",
        "⁶": "^6",
        "⁷": "^7",
        "⁸": "^8",
        "⁹": "^9",
        "⁻": "^-",
    }
)


def _normalize_ws_with_offsets(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    offsets: list[int] = []
    in_ws = False
    for idx, ch in enumerate(str(text or "")):
        if ch.isspace():
            if not in_ws:
                chars.append(" ")
                offsets.append(idx)
                in_ws = True
            continue
        chars.append(ch)
        offsets.append(idx)
        in_ws = False
    while chars and chars[0] == " ":
        chars.pop(0)
        offsets.pop(0)
    while chars and chars[-1] == " ":
        chars.pop()
        offsets.pop()
    return "".join(chars), offsets


def _quote_match_chars(ch: str) -> str:
    ch = ch.translate(_SUPERSCRIPT_MAP)
    ch = unicodedata.normalize("NFKC", ch)
    if ch in {"−", "–", "—"}:
        return "-"
    if ch in {"×", "·"}:
        return "*"
    if ch == "÷":
        return "/"
    if ch in {"⇒", "⟹"}:
        return "=>"
    if ch in {"→", "⟶"}:
        return "->"
    if ch == "≤":
        return "<="
    if ch == "≥":
        return ">="
    if ch == "≠":
        return "!="
    if ch == "≡":
        return " equiv "
    return ch


def _read_brace_group(raw: str, start: int) -> tuple[str, list[int], int] | None:
    if start >= len(raw) or raw[start] != "{":
        return None
    depth = 0
    chars: list[str] = []
    offsets: list[int] = []
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if ch == "{":
            depth += 1
            if depth > 1:
                chars.append(ch)
                offsets.append(idx)
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars), offsets, idx + 1
            chars.append(ch)
            offsets.append(idx)
            continue
        chars.append(ch)
        offsets.append(idx)
    return None


def _loose_match_text_with_offsets(text: str) -> tuple[str, list[int]]:
    """Normalize math-ish text for matching while preserving source offsets."""
    chars: list[str] = []
    offsets: list[int] = []
    raw = str(text or "")
    i = 0
    def emit(mapped: str, off: int) -> None:
        for out_ch in mapped:
            chars.append(" " if out_ch.isspace() else out_ch)
            offsets.append(off)

    def emit_piece(piece: str, piece_offsets: list[int]) -> None:
        for out_ch, out_off in zip(piece, piece_offsets):
            mapped = _quote_match_chars(out_ch).lower()
            emit(mapped, out_off)

    while i < len(raw):
        ch = raw[i]
        # Drop common LaTeX/markdown delimiters that often differ in model quotes.
        if ch in "${}`*_()[]":
            i += 1
            continue
        if ch == "\\":
            if raw.startswith("\\frac", i):
                first = _read_brace_group(raw, i + len("\\frac"))
                if first:
                    second = _read_brace_group(raw, first[2])
                    if second:
                        emit_piece(first[0], first[1])
                        emit("/", i)
                        emit_piece(second[0], second[1])
                        i = second[2]
                        continue
            if raw.startswith("\\pmod", i):
                group = _read_brace_group(raw, i + len("\\pmod"))
                if group:
                    emit(" mod ", i)
                    emit_piece(group[0], group[1])
                    i = group[2]
                    continue
            command_map = {
                "\\equiv": " equiv ",
                "\\cdot": "*",
                "\\times": "*",
                "\\div": "/",
                "\\leq": "<=",
                "\\le": "<=",
                "\\geq": ">=",
                "\\ge": ">=",
                "\\neq": "!=",
                "\\Rightarrow": "=>",
                "\\rightarrow": "->",
            }
            matched_command = None
            for command in sorted(command_map, key=len, reverse=True):
                if raw.startswith(command, i):
                    matched_command = command
                    break
            if matched_command:
                emit(command_map[matched_command], i)
                i += len(matched_command)
                continue
            m = re.match(r"\\(?:qquad|quad|left|right|text|mathrm|operatorname|[,;:!])", raw[i:])
            if m:
                i += len(m.group(0))
                continue
            # Keep command names as words for fuzzy matching, but drop the slash.
            i += 1
            continue
        mapped = _quote_match_chars(ch).lower()
        emit(mapped, i)
        i += 1
    compact_chars: list[str] = []
    compact_offsets: list[int] = []
    in_ws = False
    for ch, off in zip(chars, offsets):
        if ch.isspace():
            if not in_ws:
                compact_chars.append(" ")
                compact_offsets.append(off)
                in_ws = True
            continue
        compact_chars.append(ch)
        compact_offsets.append(off)
        in_ws = False
    while compact_chars and compact_chars[0] == " ":
        compact_chars.pop(0)
        compact_offsets.pop(0)
    while compact_chars and compact_chars[-1] == " ":
        compact_chars.pop()
        compact_offsets.pop()
    return "".join(compact_chars), compact_offsets


def _span_from_offsets(offsets: list[int], start: int, end: int, source_len: int) -> tuple[int, int] | None:
    if start < 0 or end <= start or start >= len(offsets):
        return None
    last = min(end - 1, len(offsets) - 1)
    return offsets[start], min(source_len, offsets[last] + 1)


def _source_restatement_prefix_len(source: str) -> int:
    """Heuristic: many wrong rollouts start with a problem restatement before reasoning."""
    raw = str(source or "")
    markers = [
        "\n---\n",
        "\n### Step 1",
        "\n### **Step 1",
        "\nStep 1",
    ]
    positions = [raw.find(m) for m in markers if raw.find(m) >= 0]
    if positions:
        return min(positions)
    return min(len(raw), 650)


def _find_all_occurrences(text: str, needle: str) -> list[int]:
    """Return every possibly overlapping occurrence for ambiguity checks."""
    if not needle:
        return []
    positions: list[int] = []
    start = 0
    while start <= len(text) - len(needle):
        pos = text.find(needle, start)
        if pos < 0:
            break
        positions.append(pos)
        start = pos + 1
    return positions


def locate_error_quote(
    quote: str,
    source: str | None,
    mode: str | None = None,
    *,
    min_fuzzy_score: float = 0.82,
    min_margin: float = 0.04,
) -> dict[str, Any]:
    """Locate a generated error quote in its wrong-rollout source.

    ``locatable`` means the quote can be mapped to a human-readable source span.
    ``usable_for_loss`` is stricter and rejects very short, ambiguous, too-wide,
    or likely problem-restatement spans.
    """
    locator_mode = normalize_error_quote_locator_mode(mode)
    raw_quote = str(quote or "").strip()
    raw_source = "" if source is None else str(source or "")
    base: dict[str, Any] = {
        "locatable": False,
        "usable_for_loss": False,
        "mode": locator_mode,
        "method": None,
        "score": None,
        "margin": None,
        "start_char": None,
        "end_char": None,
        "matched_text": "",
        "occurrence_count": 0,
        "ambiguous": False,
        "reject_reason": None,
    }
    if source is None:
        return {
            **base,
            "locatable": bool(raw_quote),
            "usable_for_loss": bool(raw_quote),
            "method": "no_source",
        }
    if not raw_quote:
        return {**base, "reject_reason": "empty_quote"}
    if not raw_source:
        return {**base, "reject_reason": "empty_source"}

    candidates: list[dict[str, Any]] = []

    def add_candidate(
        method: str,
        start: int,
        end: int,
        score: float,
        margin: float | None = None,
        occurrence_count: int = 1,
    ) -> None:
        if start < 0 or end <= start or start >= len(raw_source):
            return
        end = min(len(raw_source), end)
        candidates.append(
            {
                "method": method,
                "score": float(score),
                "margin": margin,
                "start_char": int(start),
                "end_char": int(end),
                "matched_text": raw_source[start:end],
                "occurrence_count": max(1, int(occurrence_count)),
                "ambiguous": int(occurrence_count) > 1,
            }
        )

    positions = _find_all_occurrences(raw_source, raw_quote)
    if positions:
        add_candidate(
            "exact",
            positions[0],
            positions[0] + len(raw_quote),
            1.0,
            0.0 if len(positions) > 1 else 1.0,
            len(positions),
        )

    if not candidates:
        norm_source, offsets = _normalize_ws_with_offsets(raw_source)
        norm_quote = _normalize_ws(raw_quote)
        positions = _find_all_occurrences(norm_source, norm_quote)
        pos = positions[0] if positions else -1
        span = _span_from_offsets(offsets, pos, pos + len(norm_quote), len(raw_source)) if pos >= 0 else None
        if span:
            add_candidate(
                "whitespace",
                span[0],
                span[1],
                0.99,
                0.0 if len(positions) > 1 else 0.99,
                len(positions),
            )

    if locator_mode != "strict" and not candidates:
        loose_source, loose_offsets = _loose_match_text_with_offsets(raw_source)
        loose_quote, _ = _loose_match_text_with_offsets(raw_quote)
        if loose_quote:
            positions = _find_all_occurrences(loose_source, loose_quote)
            pos = positions[0] if positions else -1
            span = _span_from_offsets(loose_offsets, pos, pos + len(loose_quote), len(raw_source)) if pos >= 0 else None
            if span:
                add_candidate(
                    "loose_exact",
                    span[0],
                    span[1],
                    0.97,
                    0.0 if len(positions) > 1 else 0.97,
                    len(positions),
                )

        if loose_quote and not candidates:
            q_len = len(loose_quote)
            win = max(24, min(max(q_len, 1), 260))
            step = max(8, win // 4)
            scored: list[tuple[float, int, int]] = []
            if len(loose_source) <= win:
                ratio = difflib.SequenceMatcher(None, loose_quote, loose_source).ratio()
                scored.append((ratio, 0, len(loose_source)))
            else:
                for start in range(0, max(1, len(loose_source) - win + 1), step):
                    end = min(len(loose_source), start + win)
                    ratio = difflib.SequenceMatcher(None, loose_quote, loose_source[start:end]).ratio()
                    scored.append((ratio, start, end))
                if scored and scored[-1][2] < len(loose_source):
                    start = max(0, len(loose_source) - win)
                    ratio = difflib.SequenceMatcher(None, loose_quote, loose_source[start:]).ratio()
                    scored.append((ratio, start, len(loose_source)))
            scored.sort(reverse=True, key=lambda x: x[0])
            if scored:
                best = scored[0]
                second = scored[1][0] if len(scored) > 1 else 0.0
                margin = best[0] - second
                if best[0] >= min_fuzzy_score and margin >= min_margin:
                    span = _span_from_offsets(loose_offsets, best[1], best[2], len(raw_source))
                    if span:
                        add_candidate("loose_fuzzy", span[0], span[1], best[0], margin)

    if not candidates:
        return {**base, "reject_reason": "unmatched_quote"}

    best = sorted(candidates, key=lambda c: (c["score"], c.get("margin") or 0.0), reverse=True)[0]
    start = int(best["start_char"])
    end = int(best["end_char"])
    span_len = end - start
    quote_len = len(raw_quote)
    prefix_len = _source_restatement_prefix_len(raw_source)
    reject_reason = None
    if best.get("ambiguous"):
        reject_reason = "ambiguous_match"
    elif quote_len < 6 and span_len < 8:
        reject_reason = "too_short"
    elif span_len > 520:
        reject_reason = "too_wide"
    elif start < prefix_len and span_len > 120:
        reject_reason = "problem_restatement_or_intro"
    elif best.get("margin") is not None and best["method"] == "loose_fuzzy" and best["margin"] < min_margin:
        reject_reason = "ambiguous_match"

    locatable = True
    usable = reject_reason is None
    if locator_mode == "usable_for_loss" and not usable:
        locatable = False
    return {
        **base,
        **best,
        "locatable": locatable,
        "usable_for_loss": usable,
        "reject_reason": reject_reason,
    }


def enhanced_locate(
    quote: str,
    source: str | None,
    min_fuzzy_score: float = 0.78,
    min_margin: float = 0.02,
) -> dict[str, Any]:
    """Enhanced locator from S02-8k benchmark: relaxed fuzzy + LaTeX-normalized fallback."""
    if not quote or not source:
        return {"locatable": False, "usable_for_loss": False, "method": None, "reject_reason": "empty"}

    res = locate_error_quote(quote, source, mode="locatable", min_fuzzy_score=min_fuzzy_score, min_margin=min_margin)
    if res.get("locatable") and res.get("usable_for_loss"):
        return res

    # LaTeX-normalized exact check fallback
    def norm_math(s: str) -> str:
        s = re.sub(r"\\(?:left|right|text|mathrm|displaystyle|mathbf|bm)\b", "", s)
        s = s.replace(r"\cup", "∪").replace(r"\cap", "∩").replace(r"\times", "×").replace(r"\cdot", "·")
        s = s.replace(r"\le", "≤").replace(r"\ge", "≥").replace(r"\ne", "≠").replace(r"\neq", "≠")
        s = s.replace(r"\to", "→").replace(r"\infty", "∞").replace(r"\pmod", "mod")
        s = s.replace("$", "").replace(" ", "")
        return s.strip()

    raw_quote = str(quote or "")
    raw_source = str(source or "")
    nq = norm_math(raw_quote)
    nsrc = norm_math(raw_source)
    if nq and nq in nsrc:
        pos = nsrc.find(nq)
        ratio_pos = pos / max(1, len(nsrc))
        start_char = max(0, int(ratio_pos * len(raw_source)) - 10)
        end_char = min(len(raw_source), start_char + len(raw_quote) + 50)
        return {
            "locatable": True,
            "usable_for_loss": True,
            "method": "latex_norm_exact",
            "score": 0.95,
            "margin": 0.05,
            "start_char": start_char,
            "end_char": end_char,
            "matched_text": raw_source[start_char:end_char],
            "ambiguous": False,
            "reject_reason": None,
        }

    return res if res.get("locatable") else {"locatable": False, "usable_for_loss": False, "method": None, "reject_reason": res.get("reject_reason") or "unmatched_quote"}


def _normalize_error_type(value: str) -> str:
    text = re.sub(r"[^a-z]+", "_", str(value or "").strip().lower()).strip("_")
    aliases = {
        "local": "local_reasoning",
        "reasoning": "local_reasoning",
        "local_error": "local_reasoning",
        "final": "final_answer",
        "answer": "final_answer",
        "finalization": "final_answer",
        "global": "global_understanding",
        "global_error": "global_understanding",
    }
    return aliases.get(text, text)


def _normalize_error_scope(value: str) -> str:
    text = re.sub(r"[^a-z]+", "_", str(value or "").strip().lower()).strip("_")
    aliases = {
        "local_error": "local",
        "local_reasoning": "local",
        "global_error": "global",
        "global_understanding": "global",
    }
    return aliases.get(text, text)


def _two_scope_route_meta(
    *,
    error_scope_raw: str,
    error_quote: str,
    explanation: str,
    location: dict[str, Any] | None,
    leaked: bool,
    fallback_reason: str | None,
) -> dict[str, Any]:
    """Route the deliberately minimal local/global reflection contract."""
    scope = _normalize_error_scope(error_scope_raw)
    valid_scope = scope in _VALID_ERROR_SCOPES
    loc = location if isinstance(location, dict) else {}
    reasons: list[str] = []
    training_route = "solution"
    if fallback_reason:
        reasons.append(f"key_idea_fallback:{fallback_reason}")
    if leaked:
        reasons.append("answer_leak")
    if not fallback_reason and not leaked:
        if not valid_scope:
            reasons.append("invalid_error_scope")
        elif not str(explanation or "").strip():
            reasons.append("missing_explanation")
        elif scope == "global":
            training_route = "global_full"
        elif not str(error_quote or "").strip():
            reasons.append("missing_local_quote")
        elif not bool(loc.get("locatable")):
            reasons.append("local_quote_unlocatable")
        else:
            training_route = "local_suffix"
    if training_route == "solution" and not reasons:
        reasons.append("route_fail_closed")
    return {
        "error_route_mode": "two_scope_v1",
        "error_scope_raw": str(error_scope_raw or "").strip(),
        "error_scope": scope,
        "error_scope_valid": valid_scope,
        "training_route": training_route,
        "route_reasons": reasons,
        "routing_fallback_reason": reasons[0] if training_route == "solution" else None,
        "local_loss_eligible": training_route == "local_suffix",
        "global_loss_eligible": training_route == "global_full",
    }


def _two_scope_bounded_route_meta(
    *,
    error_scope_raw: str,
    error_quote: str,
    explanation: str,
    location: dict[str, Any] | None,
    leaked: bool,
    fallback_reason: str | None,
) -> dict[str, Any]:
    """Route local suffixes and global methodological prefixes.

    Both scopes require one exact quote.  Local supervision begins at the
    first causal error; global supervision starts at the response beginning
    and ends when the key mistaken method has been fully expressed.
    """
    scope = _normalize_error_scope(error_scope_raw)
    valid_scope = scope in _VALID_ERROR_SCOPES
    loc = location if isinstance(location, dict) else {}
    reasons: list[str] = []
    training_route = "solution"
    if fallback_reason:
        reasons.append(f"key_idea_fallback:{fallback_reason}")
    if leaked:
        reasons.append("answer_leak")
    if not fallback_reason and not leaked:
        if not valid_scope:
            reasons.append("invalid_error_scope")
        elif not str(explanation or "").strip():
            reasons.append("missing_explanation")
        elif not str(error_quote or "").strip():
            reasons.append(f"missing_{scope or 'unknown'}_quote")
        elif not bool(loc.get("locatable")):
            reasons.append(f"{scope or 'unknown'}_quote_unlocatable")
        elif scope == "local":
            training_route = "local_suffix"
        else:
            training_route = "global_prefix"
    if training_route == "solution" and not reasons:
        reasons.append("route_fail_closed")
    return {
        "error_route_mode": "two_scope_bounded_v1",
        "error_scope_raw": str(error_scope_raw or "").strip(),
        "error_scope": scope,
        "error_scope_valid": valid_scope,
        "training_route": training_route,
        "route_reasons": reasons,
        "routing_fallback_reason": reasons[0] if training_route == "solution" else None,
        "local_loss_eligible": training_route == "local_suffix",
        "global_loss_eligible": training_route == "global_prefix",
    }


def _is_none_error_quote(value: str) -> bool:
    return " ".join(str(value or "").strip().lower().split()) in _NONE_ERROR_QUOTES


def _is_generic_error_quote(value: str) -> bool:
    raw = str(value or "")
    if re.match(r"^\s*#{1,6}\s+\S", raw) or _HEADING_QUOTE_RE.search(raw):
        return True
    norm = " ".join(raw.lower().split()).strip(" .,:;#*`")
    if norm in _GENERIC_ERROR_QUOTES:
        return True
    words = re.findall(r"[a-zA-Z0-9]+", norm)
    return len(words) <= 2 and not re.search(r"[=<>+\-*/^]", norm)


def _three_way_route_meta(
    *,
    error_type_raw: str,
    error_quote: str,
    explanation: str,
    quote_source: str | None,
    location: dict[str, Any] | None,
    leaked: bool,
    fallback_reason: str | None,
) -> dict[str, Any]:
    """Return the versioned global/final/local training-route decision."""
    profile = dict(_THREE_WAY_ROUTE_PROFILE_V1)
    error_type = _normalize_error_type(error_type_raw)
    error_type_valid = error_type in _VALID_ROUTED_ERROR_TYPES
    quote_none = _is_none_error_quote(error_quote)
    loc = location if isinstance(location, dict) else {}
    locatable = bool(loc.get("locatable"))
    start = loc.get("start_char")
    end = loc.get("end_char")
    span_chars = end - start if isinstance(start, int) and isinstance(end, int) else None
    explanation_text = str(explanation or "").strip()
    generic_quote = False if quote_none else _is_generic_error_quote(error_quote)
    later_error_cue = any(cue in explanation_text.lower() for cue in _LATER_ERROR_CUES)
    source = str(quote_source or "")
    final_region_start = max(
        0,
        len(source) - max(
            int(profile["final_region_chars"]),
            int(len(source) * float(profile["final_region_fraction"])),
        ),
    )
    final_answer_cue = any(
        cue in str(error_quote or "").lower() for cue in _FINAL_ERROR_CUES
    )
    final_region = bool(isinstance(start, int) and start >= final_region_start)

    reasons: list[str] = []
    warnings: list[str] = []
    training_route = "solution"
    local_loss_eligible = False
    final_loss_eligible = False
    if fallback_reason:
        reasons.append(f"key_idea_fallback:{fallback_reason}")
    if leaked:
        reasons.append("answer_leak")
    if fallback_reason or leaked:
        # Sanitizer fallbacks take precedence over any fine-grained route.
        # The caller will replace the key idea with solution-derived context.
        pass
    elif not error_type_valid:
        reasons.append("invalid_error_type")
    elif error_type == "global_understanding":
        reasons.append("global_understanding")
    elif error_type == "final_answer":
        if quote_none:
            reasons.append("missing_or_truncated_final_answer")
        elif not locatable:
            reasons.append("final_quote_unlocatable")
        elif loc.get("ambiguous"):
            reasons.append("ambiguous_final_quote")
        elif generic_quote:
            reasons.append("generic_final_quote")
        elif len(error_quote) > int(profile["final_max_quote_chars"]):
            reasons.append("final_quote_too_wide")
        elif not final_region:
            reasons.append("final_quote_not_in_terminal_region")
        else:
            training_route = "final_answer"
            final_loss_eligible = True
    elif error_type == "local_reasoning":
        if quote_none:
            reasons.append("missing_local_quote")
        elif not locatable:
            reasons.append("local_quote_unlocatable")
        elif loc.get("ambiguous"):
            reasons.append("ambiguous_local_quote")
        elif generic_quote:
            reasons.append("generic_local_quote")
        elif len(error_quote) < int(profile["local_min_quote_chars"]):
            reasons.append("local_quote_too_short")
        elif span_chars is None or span_chars > int(profile["local_max_span_chars"]):
            reasons.append("local_span_too_wide")
        elif (
            isinstance(start, int)
            and start < _source_restatement_prefix_len(source)
            and span_chars > int(profile["local_intro_max_span_chars"])
        ):
            reasons.append("problem_restatement_or_intro")
        elif later_error_cue:
            reasons.append("explanation_points_later")
        elif loc.get("method") == "loose_fuzzy" and (
            float(loc.get("score") or 0.0) < float(profile["local_fuzzy_min_score"])
            or float(loc.get("margin") or 0.0) < float(profile["local_fuzzy_min_margin"])
        ):
            reasons.append("weak_fuzzy_match")
        else:
            training_route = "local_window"
            local_loss_eligible = True

    if explanation_text and len(explanation_text) < int(profile["min_explanation_chars"]):
        warnings.append("explanation_too_short")
    if training_route == "solution" and not reasons:
        reasons.append("route_fail_closed")
    return {
        "error_route_mode": "three_way_v1",
        "eligibility_policy": profile["name"],
        "eligibility_profile": profile,
        "error_type_raw": str(error_type_raw or "").strip(),
        "error_type": error_type,
        "error_type_valid": error_type_valid,
        "error_quote_intentional_none": bool(error_type == "global_understanding" and quote_none),
        "generic_error_quote": generic_quote,
        "explanation_later_error_cue": later_error_cue,
        "span_chars": span_chars,
        "final_region": final_region,
        "final_answer_cue": final_answer_cue,
        "local_loss_eligible": local_loss_eligible,
        "final_loss_eligible": final_loss_eligible,
        "training_route": training_route,
        "route_reasons": reasons,
        "route_warnings": warnings,
        "routing_fallback_reason": reasons[0] if training_route == "solution" else None,
    }


def _quote_in_source(quote: str, source: str | None, mode: str | None = None) -> bool:
    loc = locate_error_quote(quote, source, mode=mode)
    return bool(loc.get("locatable"))


def _looks_like_instruction_quote(quote: str, *, allow_final_answer: bool = False) -> bool:
    q = str(quote or "").lower()
    cues = (
        cue for cue in _INSTRUCTION_QUOTE_CUES
        if not (allow_final_answer and cue == "final answer")
    )
    return any(cue in q for cue in cues)


def _fallback_error_quote_from_source(source: str | None, max_chars: int = 220) -> str:
    raw = str(source or "")
    if not raw.strip():
        return ""
    start = 0
    m = re.search(r"<reasoning>\s*", raw, flags=re.IGNORECASE)
    if m:
        start = m.end()
    while start < len(raw) and raw[start].isspace():
        start += 1
    tail = raw[start:]
    if not tail.strip():
        return ""
    window = tail[:max_chars]
    end = None
    for m in re.finditer(r"(?<=[.!?])\s+", window):
        if m.end() >= 12:
            end = m.start() + 1
            break
    if end is None:
        newline = window.find("\n", 12)
        if newline >= 0:
            end = newline
    if end is None:
        end = len(window)
    return tail[:end].strip()


def parse_r3_el_p1_output(text: str) -> tuple[dict[str, str] | None, str | None]:
    """Strictly parse the frozen R3-EL output contract.

    The prompt-search proof used a full-string schema: extra prose, duplicate
    tags, nested markup, empty fields, and positional line locators are invalid.
    D1 deliberately preserves that fail-closed boundary.
    """
    raw = str(text or "")
    if _R3_EL_INVALID_XML_AMPERSAND_RE.search(raw):
        return None, "malformed_xml_entity_or_bare_ampersand"
    if _R3_EL_NO_REPAIR_RE.fullmatch(raw):
        return {"decision": "no_repair"}, None
    match = _R3_EL_REPAIR_RE.fullmatch(raw)
    if match is None:
        return None, "output_does_not_fullmatch_documented_schema"
    fields = {"decision": "repair"}
    for key, value in match.groupdict().items():
        normalized = str(value or "").strip()
        if (
            not normalized
            or _R3_EL_TAG_LIKE_RE.search(normalized)
            or _R3_EL_FORBIDDEN_FIELD_MARKUP_RE.search(normalized)
        ):
            return None, f"invalid_or_nested_{key}"
        if key == "repair_anchor" and _R3_EL_FORBIDDEN_LINE_LOCALIZATION_RE.search(
            normalized
        ):
            return None, "line_number_localization_in_repair_anchor"
        fields[key] = normalized
    return fields, None


def sanitize_r3_el_p1_with_meta(
    text: str,
    gold: str,
    *,
    task_type: str,
    leak_fallback_mode: str | None,
    quote_source: str | None,
    error_quote_locator_mode: str | None,
) -> tuple[str, dict[str, Any]]:
    """Map frozen R3-EL into the existing causal-anchor suffix-loss contract."""
    raw = str(text or "")
    leak_mode = normalize_leak_fallback_mode(leak_fallback_mode)
    locator_mode = normalize_error_quote_locator_mode(error_quote_locator_mode)
    fields, schema_reject_reason = parse_r3_el_p1_output(raw)
    decision = str((fields or {}).get("decision") or "invalid")
    anchor = str((fields or {}).get("repair_anchor") or "")
    key_idea = str((fields or {}).get("key_idea") or "")
    base_meta: dict[str, Any] = {
        "p1_prompt_id": _R3_EL_PROMPT_ID,
        "p1_prompt_system_sha256": _R3_EL_SYSTEM_SHA256,
        "p1_prompt_user_sha256": _R3_EL_USER_SHA256,
        "p1_effective_prompt_sha256": _R3_EL_EFFECTIVE_PROMPT_SHA256,
        "p1_schema": "sopsd027r4b_r3_el_p1_output_v1",
        "p1_schema_valid": fields is not None,
        "p1_schema_reject_reason": schema_reject_reason,
        "decision": decision,
        "repair_anchor": anchor,
        "extracted_key_idea": bool(key_idea),
        # The no-repair gate ledger historically calls this field
        # ``extracted_explanation``.  Treat a strict Key Idea as its semantic
        # successor so the ledger remains comparable across campaigns.
        "extracted_explanation": bool(key_idea),
        "explanation_open_no_close": False,
        "extracted_error_quote": bool(anchor),
        "error_quote": anchor,
        "raw_error_quote": anchor,
        "error_quote_fallback_mode": "solution_on_invalid_quote",
        "error_quote_locator_mode": locator_mode,
        "error_quote_location": None,
        "error_quote_source_valid": False,
        "error_quote_fallback": None,
        "error_quote_invalid_reason": None,
        "error_route_mode": "legacy",
        "raw_chars": len(raw),
        "idea_chars": len(key_idea),
        "leak_fallback_mode": leak_mode,
        "boxed_removed": False,
        "leaked": False,
        "fallback_reason": None,
        "truncated": False,
        "reference_solution_used_for_reflection": False,
    }
    if fields is None:
        return SAFE_KEY_IDEA, {
            **base_meta,
            "fallback_reason": "invalid_p1_schema",
            "teacher_privilege": "solution",
        }
    if decision == "no_repair":
        return SAFE_KEY_IDEA, {
            **base_meta,
            "teacher_privilege": "solution",
        }

    location = locate_error_quote(anchor, quote_source, mode=locator_mode)
    unique_usable = bool(
        quote_source is not None
        and location.get("locatable")
        and location.get("usable_for_loss")
        and not location.get("ambiguous")
        and isinstance(location.get("start_char"), int)
        and isinstance(location.get("end_char"), int)
        and int(location["end_char"]) > int(location["start_char"])
    )
    # Downstream suffix masking consumes ``locatable``.  Tighten it here even
    # if a caller accidentally asks for the historical permissive mode.
    location = dict(location)
    location["locatable"] = unique_usable
    location["accepted_for_loss"] = unique_usable

    leaked = False
    if leak_mode != "disabled":
        leaked = (
            contains_math_answer_leak(key_idea, gold)
            if task_type == "math"
            else contains_letter_leak(key_idea, gold)
        )
    idea = key_idea
    truncated = False
    if not idea.lower().startswith("key idea"):
        idea = "Key idea: " + idea
    if len(idea) > 1500:
        idea = idea[:1500].rsplit(" ", 1)[0].strip() + " ..."
        truncated = True
    meta = {
        **base_meta,
        "error_quote_location": location,
        "error_quote_source_valid": unique_usable,
        "error_quote_fallback": None if unique_usable else "solution_privilege",
        "error_quote_invalid_reason": (
            None
            if unique_usable
            else str(location.get("reject_reason") or "anchor_not_unique_or_usable")
        ),
        "leaked": bool(leaked),
        "truncated": truncated,
    }
    if leaked or not unique_usable:
        meta["teacher_privilege"] = "solution"
    if leaked:
        meta["fallback_reason"] = "answer_leak"
        return SAFE_KEY_IDEA, meta
    return idea, meta


def sanitize_r4_alo_p1_with_meta(
    text: str,
    gold: str,
    *,
    task_type: str,
    leak_fallback_mode: str | None,
    quote_source: str | None,
    error_quote_locator_mode: str | None,
) -> tuple[str, dict[str, Any]]:
    """Validate R4-ALO and preserve its operational repair packet.

    R4-ALO is opt-in and fail-closed.  A repair packet reaches either the
    student repair turn or the profile-aware teacher only when its anchor maps
    to one unique source span accepted by the suffix-loss locator.
    """
    raw = str(text or "")
    leak_mode = normalize_leak_fallback_mode(leak_fallback_mode)
    locator_mode = normalize_error_quote_locator_mode(error_quote_locator_mode)
    fields, schema_reject_reason = parse_p1_r4_alo_output(raw)
    decision = str((fields or {}).get("decision") or "invalid")
    anchor = str((fields or {}).get("repair_anchor") or "")
    key_idea = str((fields or {}).get("key_idea") or "")
    next_operation = str((fields or {}).get("next_operation") or "")
    verification_check = str((fields or {}).get("verification_check") or "")
    preserved_context = str((fields or {}).get("preserved_context") or "")
    error_scope = str((fields or {}).get("error_scope") or "")
    base_meta: dict[str, Any] = {
        "p1_prompt_id": P1_R4_ALO_ID,
        "p1_prompt_system_sha256": P1_R4_ALO_SYSTEM_SHA256,
        "p1_prompt_user_sha256": P1_R4_ALO_USER_SHA256,
        "p1_effective_prompt_sha256": P1_R4_ALO_EFFECTIVE_SHA256,
        "p1_schema": P1_R4_ALO_SCHEMA,
        "p1_schema_valid": fields is not None,
        "p1_schema_reject_reason": schema_reject_reason,
        "decision": decision,
        "repair_anchor": anchor,
        "error_scope": error_scope,
        "preserved_context": preserved_context,
        "key_idea": key_idea,
        "next_operation": next_operation,
        "verification_check": verification_check,
        "extracted_key_idea": bool(key_idea),
        "extracted_explanation": bool(key_idea),
        "explanation_open_no_close": False,
        "extracted_error_quote": bool(anchor),
        "error_quote": anchor,
        "raw_error_quote": anchor,
        "error_quote_fallback_mode": "solution_on_invalid_quote",
        "error_quote_locator_mode": locator_mode,
        "error_quote_location": None,
        "error_quote_source_valid": False,
        "error_quote_fallback": None,
        "error_quote_invalid_reason": None,
        "error_route_mode": "legacy",
        "training_route": (
            "local_suffix" if error_scope == "local" else
            "global_full" if error_scope == "global" else "solution"
        ),
        "raw_chars": len(raw),
        "idea_chars": len(key_idea),
        "leak_fallback_mode": leak_mode,
        "boxed_removed": False,
        "leaked": False,
        "fallback_reason": None,
        "truncated": False,
        "reference_solution_used_for_reflection": False,
    }
    if fields is None:
        return SAFE_KEY_IDEA, {
            **base_meta,
            "fallback_reason": "invalid_p1_schema",
            "teacher_privilege": "solution",
        }
    if decision == "no_repair":
        return SAFE_KEY_IDEA, {
            **base_meta,
            "teacher_privilege": "solution",
        }

    location = dict(locate_error_quote(anchor, quote_source, mode=locator_mode))
    unique_usable = bool(
        quote_source is not None
        and location.get("locatable")
        and location.get("usable_for_loss")
        and not location.get("ambiguous")
        and isinstance(location.get("start_char"), int)
        and isinstance(location.get("end_char"), int)
        and int(location["end_char"]) > int(location["start_char"])
    )
    location["locatable"] = unique_usable
    location["accepted_for_loss"] = unique_usable

    # The source anchor may legitimately contain the student's *wrong* boxed
    # result.  Leak checking therefore covers only newly generated guidance.
    guidance = "\n".join(
        (preserved_context, key_idea, next_operation, verification_check)
    )
    leaked = False
    if leak_mode != "disabled":
        leaked = (
            contains_math_answer_leak(guidance, gold)
            if task_type == "math"
            else contains_letter_leak(guidance, gold)
        )
    hint = (
        f"Key idea: {key_idea}\n"
        f"Next operation: {next_operation}\n"
        f"Verification check: {verification_check}"
    )
    truncated = False
    if len(hint) > 2200:
        hint = hint[:2200].rsplit(" ", 1)[0].strip() + " ..."
        truncated = True
    meta = {
        **base_meta,
        "error_quote_location": location,
        "error_quote_source_valid": unique_usable,
        "error_quote_fallback": None if unique_usable else "solution_privilege",
        "error_quote_invalid_reason": (
            None if unique_usable else
            str(location.get("reject_reason") or "anchor_not_unique_or_usable")
        ),
        "leaked": bool(leaked),
        "truncated": truncated,
    }
    if leaked or not unique_usable:
        meta["teacher_privilege"] = "solution"
    if leaked:
        meta["fallback_reason"] = "answer_leak"
        return SAFE_KEY_IDEA, meta
    if not unique_usable:
        return SAFE_KEY_IDEA, meta
    return hint, meta


def sanitize_r4_sid_p1_with_meta(
    text: str,
    gold: str,
    *,
    task_type: str,
    leak_fallback_mode: str | None,
    quote_source: str | None,
) -> tuple[str, dict[str, Any]]:
    """Validate the same-field R4-SID packet and resolve its opaque locator.

    The model emits only ``decision``, an opaque segment id in
    ``repair_anchor``, and ``key_idea``. The original source fragment is
    recovered deterministically inside metadata for suffix masking and is not
    included in either the P1 response or the default teacher prompt.
    """
    raw = str(text or "")
    leak_mode = normalize_leak_fallback_mode(leak_fallback_mode)
    fields, schema_reject_reason = parse_p1_r4_sid_output(raw)
    decision = str((fields or {}).get("decision") or "invalid")
    anchor_id = str((fields or {}).get("repair_anchor") or "")
    key_idea = str((fields or {}).get("key_idea") or "")
    source = str(quote_source or "")
    segments = segment_candidate_reasoning(source)
    segment_map = {str(item["segment_id"]): item for item in segments}
    segment = segment_map.get(anchor_id)
    source_fragment = str((segment or {}).get("text") or "")
    normalized_fragment = re.sub(r"\s+", " ", source_fragment).strip().lower()
    normalized_idea = re.sub(r"\s+", " ", key_idea).strip().lower()
    source_fragment_copy_detected = bool(
        len(normalized_fragment) >= 12
        and normalized_fragment in normalized_idea
    )
    location = None
    if segment is not None:
        location = {
            "locatable": True,
            "usable_for_loss": True,
            "accepted_for_loss": True,
            "ambiguous": False,
            "match_kind": "segment_id_exact",
            "segment_id": anchor_id,
            "start_char": int(segment["start_char"]),
            "end_char": int(segment["end_char"]),
            "matched_text": source_fragment,
        }
    base_meta: dict[str, Any] = {
        "p1_prompt_id": P1_R4_SID_ID,
        "p1_prompt_system_sha256": P1_R4_SID_SYSTEM_SHA256,
        "p1_prompt_user_sha256": P1_R4_SID_USER_SHA256,
        "p1_effective_prompt_sha256": P1_R4_SID_EFFECTIVE_SHA256,
        "p1_schema": P1_R4_SID_SCHEMA,
        "p1_schema_valid": fields is not None,
        "p1_schema_reject_reason": schema_reject_reason,
        "decision": decision,
        "repair_anchor": anchor_id,
        "repair_anchor_id": anchor_id,
        "extracted_key_idea": bool(key_idea),
        "extracted_explanation": bool(key_idea),
        "explanation_open_no_close": False,
        "extracted_error_quote": bool(anchor_id),
        # Source text is internal lineage for loss masking, not model output.
        "error_quote": source_fragment,
        "raw_error_quote": source_fragment,
        "error_quote_fallback_mode": "solution_on_invalid_segment_id",
        "error_quote_locator_mode": "segment_id_exact",
        "error_quote_location": location,
        "error_quote_source_valid": segment is not None,
        "error_quote_fallback": None if segment is not None else "solution_privilege",
        "error_quote_invalid_reason": (
            None if segment is not None else "segment_id_not_in_source_manifest"
        ),
        "error_route_mode": "legacy",
        "training_route": "local_suffix" if segment is not None else "solution",
        "segment_count": len(segments),
        "source_fragment_copy_detected": source_fragment_copy_detected,
        "original_error_text_exposed_in_p1_output": source_fragment_copy_detected,
        "raw_chars": len(raw),
        "idea_chars": len(key_idea),
        "leak_fallback_mode": leak_mode,
        "boxed_removed": False,
        "leaked": False,
        "fallback_reason": None,
        "truncated": False,
        "reference_solution_used_for_reflection": False,
    }
    if fields is None:
        return SAFE_KEY_IDEA, {
            **base_meta,
            "fallback_reason": "invalid_p1_schema",
            "teacher_privilege": "solution",
        }
    if decision == "no_repair":
        return SAFE_KEY_IDEA, {
            **base_meta,
            "teacher_privilege": "solution",
        }
    if segment is None:
        return SAFE_KEY_IDEA, {
            **base_meta,
            "fallback_reason": "invalid_segment_id",
            "teacher_privilege": "solution",
        }
    if source_fragment_copy_detected:
        return SAFE_KEY_IDEA, {
            **base_meta,
            "fallback_reason": "source_fragment_copied_to_key_idea",
            "teacher_privilege": "solution",
        }

    leaked = False
    if leak_mode != "disabled":
        leaked = (
            contains_math_answer_leak(key_idea, gold)
            if task_type == "math"
            else contains_letter_leak(key_idea, gold)
        )
    idea = key_idea
    truncated = False
    if not idea.lower().startswith("key idea"):
        idea = "Key idea: " + idea
    if len(idea) > 1500:
        idea = idea[:1500].rsplit(" ", 1)[0].strip() + " ..."
        truncated = True
    meta = {
        **base_meta,
        "leaked": bool(leaked),
        "truncated": truncated,
    }
    if leaked:
        meta["fallback_reason"] = "answer_leak"
        meta["teacher_privilege"] = "solution"
        return SAFE_KEY_IDEA, meta
    return idea, meta


def sanitize_r4_sil_p1_with_meta(
    text: str,
    gold: str,
    *,
    task_type: str,
    leak_fallback_mode: str | None,
    quote_source: str | None,
) -> tuple[str, dict[str, Any]]:
    """Validate R4-SIL using the frozen R4-SID three-field mechanics.

    R4-SIL changes the prompt-level content contract, not the parser or masking
    mechanics. Reusing the R4-SID sanitizer keeps that one-factor boundary
    explicit while rebinding the profile identity and hashes in the receipt.
    """
    hint, meta = sanitize_r4_sid_p1_with_meta(
        text,
        gold,
        task_type=task_type,
        leak_fallback_mode=leak_fallback_mode,
        quote_source=quote_source,
    )
    meta.update(
        {
            "p1_prompt_id": P1_R4_SIL_ID,
            "p1_prompt_system_sha256": P1_R4_SIL_SYSTEM_SHA256,
            "p1_prompt_user_sha256": P1_R4_SIL_USER_SHA256,
            "p1_effective_prompt_sha256": P1_R4_SIL_EFFECTIVE_SHA256,
            "p1_schema": P1_R4_SIL_SCHEMA,
            "key_idea_semantic_location_required": True,
        }
    )
    return hint, meta


def sanitize_r4_sql_p1_with_meta(
    text: str,
    gold: str,
    *,
    task_type: str,
    leak_fallback_mode: str | None,
    quote_source: str | None,
    error_quote_locator_mode: str | None,
) -> tuple[str, dict[str, Any]]:
    """Validate R4-SQL with the precise R3-EL locator mechanics.

    The output fields and exact-quote resolution are deliberately unchanged;
    only the prompt-level Key Idea contract is stronger.  The returned repair
    hint contains no anchor, so the answer-generating model receives semantic
    location through ``key_idea`` rather than the student's source wording.
    """
    hint, meta = sanitize_r3_el_p1_with_meta(
        text,
        gold,
        task_type=task_type,
        leak_fallback_mode=leak_fallback_mode,
        quote_source=quote_source,
        error_quote_locator_mode=error_quote_locator_mode,
    )
    meta.update(
        {
            "p1_prompt_id": P1_R4_SQL_ID,
            "p1_prompt_system_sha256": P1_R4_SQL_SYSTEM_SHA256,
            "p1_prompt_user_sha256": P1_R4_SQL_USER_SHA256,
            "p1_effective_prompt_sha256": P1_R4_SQL_EFFECTIVE_SHA256,
            "p1_schema": P1_R4_SQL_SCHEMA,
            "key_idea_semantic_location_required": True,
            "machine_locator_kind": "minimum_exact_source_quote",
            "repair_anchor_exposed_to_teacher": False,
            "reference_solution_used_for_reflection": False,
        }
    )
    return hint, meta


def sanitize_key_idea_with_meta(
    text: str,
    gold: str,
    task_type: str = "sciknoweval",
    leak_fallback_mode: str | None = None,
    quote_source: str | None = None,
    error_quote_fallback_mode: str | None = None,
    error_quote_locator_mode: str | None = None,
    error_route_mode: str | None = None,
    reflection_prompt_mode: str | None = None,
) -> tuple[str, dict]:
    """Extract and sanitize the ROSD key idea, returning debug metadata too."""
    resolved_reflection_mode = (
        normalize_reflection_prompt_mode(reflection_prompt_mode)
        if reflection_prompt_mode is not None
        else None
    )
    if resolved_reflection_mode == "deepseek_r3_el_p1_noref_xml_v1":
        return sanitize_r3_el_p1_with_meta(
            text,
            gold,
            task_type=task_type,
            leak_fallback_mode=leak_fallback_mode,
            quote_source=quote_source,
            error_quote_locator_mode=error_quote_locator_mode,
        )
    if resolved_reflection_mode == P1_R4_ALO_MODE:
        return sanitize_r4_alo_p1_with_meta(
            text,
            gold,
            task_type=task_type,
            leak_fallback_mode=leak_fallback_mode,
            quote_source=quote_source,
            error_quote_locator_mode=error_quote_locator_mode,
        )
    if resolved_reflection_mode == P1_R4_SID_MODE:
        return sanitize_r4_sid_p1_with_meta(
            text,
            gold,
            task_type=task_type,
            leak_fallback_mode=leak_fallback_mode,
            quote_source=quote_source,
        )
    if resolved_reflection_mode == P1_R4_SIL_MODE:
        return sanitize_r4_sil_p1_with_meta(
            text,
            gold,
            task_type=task_type,
            leak_fallback_mode=leak_fallback_mode,
            quote_source=quote_source,
        )
    if resolved_reflection_mode == P1_R4_SQL_MODE:
        return sanitize_r4_sql_p1_with_meta(
            text,
            gold,
            task_type=task_type,
            leak_fallback_mode=leak_fallback_mode,
            quote_source=quote_source,
            error_quote_locator_mode=error_quote_locator_mode,
        )
    leak_mode = normalize_leak_fallback_mode(leak_fallback_mode)
    quote_fallback_mode = normalize_error_quote_fallback_mode(error_quote_fallback_mode)
    quote_locator_mode = normalize_error_quote_locator_mode(error_quote_locator_mode)
    route_mode = normalize_error_route_mode(error_route_mode)
    raw = str(text or "")
    error_type_raw = _extract_tag(raw, "error_type")
    error_type = _normalize_error_type(error_type_raw)
    error_scope_raw = _extract_tag(raw, "error_scope")
    error_scope = _normalize_error_scope(error_scope_raw)
    error_quote = _extract_tag(raw, "error_quote")
    raw_error_quote = error_quote
    intentional_none = bool(
        (
            route_mode == "three_way_v1"
            and error_type == "global_understanding"
            and _is_none_error_quote(error_quote)
        )
        or (route_mode == "two_scope_v1" and error_scope == "global")
    )
    error_quote_source_valid = True
    error_quote_fallback = None
    error_quote_invalid_reason = None
    quote_teacher_privilege = None
    error_quote_location = None
    if quote_source is not None:
        allow_final_answer_quote = bool(
            route_mode == "three_way_v1" and error_type == "final_answer"
        )
        if intentional_none:
            error_quote_source_valid = False
        elif _is_none_error_quote(error_quote):
            error_quote_source_valid = False
            error_quote_invalid_reason = "missing_error_quote"
        elif _looks_like_instruction_quote(
            error_quote,
            allow_final_answer=allow_final_answer_quote,
        ):
            error_quote_source_valid = False
            error_quote_invalid_reason = "instruction_error_quote"
        else:
            locator_kwargs = {}
            if route_mode == "three_way_v1":
                locator_kwargs = {
                    "min_fuzzy_score": float(_THREE_WAY_ROUTE_PROFILE_V1["local_fuzzy_min_score"]),
                    "min_margin": float(_THREE_WAY_ROUTE_PROFILE_V1["local_fuzzy_min_margin"]),
                }
            error_quote_location = locate_error_quote(
                error_quote,
                quote_source,
                mode=quote_locator_mode,
                **locator_kwargs,
            )
            error_quote_source_valid = bool(error_quote_location.get("locatable"))
        if (
            not intentional_none
            and error_quote
            and not _looks_like_instruction_quote(
                error_quote,
                allow_final_answer=allow_final_answer_quote,
            )
            and not error_quote_source_valid
        ):
            error_quote_source_valid = False
            error_quote_invalid_reason = "unmatched_error_quote"
        if not error_quote_source_valid and not intentional_none:
            if quote_fallback_mode == "quote_source_prefix":
                replacement = _fallback_error_quote_from_source(quote_source)
                if replacement:
                    error_quote = replacement
                    error_quote_fallback = "quote_source_prefix"
                    error_quote_location = locate_error_quote(
                        error_quote,
                        quote_source,
                        mode=quote_locator_mode,
                    )
                    error_quote_source_valid = bool(error_quote_location.get("locatable"))
            elif quote_fallback_mode == "solution_on_invalid_quote":
                error_quote_fallback = "solution_privilege"
                quote_teacher_privilege = "solution"
    extracted = _extract_tag(raw, "explanation")
    open_explanation = ""
    if not extracted:
        open_explanation = _extract_open_tag_tail(raw, "explanation")
    explanation_open_no_close = bool(open_explanation and not extracted)
    if route_mode in {"three_way_v1", "two_scope_v1", "two_scope_bounded_v1"}:
        # Routed labels fail closed when the explanation field is absent. The
        # legacy parser keeps its historical raw-text fallback unchanged.
        idea = extracted or open_explanation
    else:
        idea = extracted or open_explanation or raw
    idea = re.sub(
        r"</?(?:error_type|error_scope|error_quote|explanation)>",
        "",
        idea,
        flags=re.IGNORECASE,
    ).strip()
    pre_strip_leak = False
    if leak_mode != "disabled":
        pre_strip_leak = (
            contains_math_answer_leak(idea, gold)
            if task_type == "math"
            else contains_letter_leak(idea, gold)
        )
    boxed_removed = bool(re.search(r"\\boxed\s*\{", idea))
    if leak_mode in {"safe_hint", "solution_on_leak"}:
        idea = re.sub(r"\\boxed\s*\{.*?\}", "", idea, flags=re.DOTALL).strip()
    idea = re.sub(r"\n{3,}", "\n\n", idea)
    leaked = False
    if leak_mode != "disabled":
        leaked = (
            pre_strip_leak
            or (contains_math_answer_leak(idea, gold) if task_type == "math" else contains_letter_leak(idea, gold))
        )
    fallback_reason = None
    truncated = False
    if not idea:
        fallback_reason = "empty"
    elif len(idea) < 12 and route_mode not in {
        "three_way_v1",
        "two_scope_v1",
        "two_scope_bounded_v1",
    }:
        fallback_reason = "too_short"
    elif leaked and leak_mode == "safe_hint":
        fallback_reason = "answer_leak"
    elif leaked and leak_mode == "solution_on_leak":
        fallback_reason = "answer_leak"
    route_meta: dict[str, Any] = {
        "error_route_mode": route_mode,
        "error_type_raw": str(error_type_raw or "").strip(),
        "error_type": error_type,
        "error_type_valid": error_type in _VALID_ROUTED_ERROR_TYPES,
    }
    if route_mode == "three_way_v1":
        route_meta = _three_way_route_meta(
            error_type_raw=error_type_raw,
            error_quote=error_quote,
            explanation=idea,
            quote_source=quote_source,
            location=error_quote_location,
            leaked=bool(leaked),
            fallback_reason=fallback_reason,
        )
        if route_meta.get("training_route") == "solution":
            quote_teacher_privilege = "solution"
    elif route_mode == "two_scope_v1":
        route_meta = _two_scope_route_meta(
            error_scope_raw=error_scope_raw,
            error_quote=error_quote,
            explanation=idea,
            location=error_quote_location,
            leaked=bool(leaked),
            fallback_reason=fallback_reason,
        )
        if route_meta.get("training_route") == "solution":
            quote_teacher_privilege = "solution"
    elif route_mode == "two_scope_bounded_v1":
        route_meta = _two_scope_bounded_route_meta(
            error_scope_raw=error_scope_raw,
            error_quote=error_quote,
            explanation=idea,
            location=error_quote_location,
            leaked=bool(leaked),
            fallback_reason=fallback_reason,
        )
        if route_meta.get("training_route") == "solution":
            quote_teacher_privilege = "solution"
    if fallback_reason:
        meta = {
            "extracted_explanation": bool(extracted),
            "explanation_open_no_close": explanation_open_no_close,
            "extracted_error_quote": bool(raw_error_quote),
            "error_quote": error_quote,
            "raw_error_quote": raw_error_quote,
            "error_quote_source_valid": error_quote_source_valid,
            "error_quote_fallback": error_quote_fallback,
            "error_quote_invalid_reason": error_quote_invalid_reason,
            "error_quote_fallback_mode": quote_fallback_mode,
            "error_quote_locator_mode": quote_locator_mode,
            "error_quote_location": error_quote_location,
            "boxed_removed": boxed_removed,
            "leaked": bool(leaked),
            "fallback_reason": fallback_reason,
            "leak_fallback_mode": leak_mode,
            "raw_chars": len(raw),
            "idea_chars": len(idea),
            "truncated": False,
            **route_meta,
        }
        if leak_mode == "solution_on_leak" and fallback_reason == "answer_leak":
            meta["teacher_privilege"] = "solution"
        if quote_teacher_privilege:
            meta["teacher_privilege"] = quote_teacher_privilege
        return SAFE_KEY_IDEA, meta
    if not idea.lower().startswith("key idea"):
        idea = "Key idea: " + idea
    if len(idea) > 1500:
        idea = idea[:1500].rsplit(" ", 1)[0].strip() + " ..."
        truncated = True
    meta = {
        "extracted_explanation": bool(extracted),
        "explanation_open_no_close": explanation_open_no_close,
        "extracted_error_quote": bool(raw_error_quote),
        "error_quote": error_quote,
        "raw_error_quote": raw_error_quote,
        "error_quote_source_valid": error_quote_source_valid,
        "error_quote_fallback": error_quote_fallback,
        "error_quote_invalid_reason": error_quote_invalid_reason,
        "error_quote_fallback_mode": quote_fallback_mode,
        "error_quote_locator_mode": quote_locator_mode,
        "error_quote_location": error_quote_location,
        "boxed_removed": boxed_removed,
        "leaked": bool(leaked),
        "fallback_reason": None,
        "leak_fallback_mode": leak_mode,
        "raw_chars": len(raw),
        "idea_chars": len(idea),
        "truncated": truncated,
        **route_meta,
    }
    if quote_teacher_privilege:
        meta["teacher_privilege"] = quote_teacher_privilege
    return idea, meta


def sanitize_key_idea(
    text: str,
    gold: str,
    task_type: str = "sciknoweval",
    leak_fallback_mode: str | None = None,
) -> str:
    """Extract and sanitize the ROSD ``<explanation>`` key corrective idea."""
    hint, _ = sanitize_key_idea_with_meta(
        text, gold, task_type=task_type, leak_fallback_mode=leak_fallback_mode
    )
    return hint


def build_hint_messages(
    problem: str,
    gold: str,
    last_attempt: str,
    prior_hints: Sequence[str],
    teacher_context: str | None = None,
    task_type: str = "sciknoweval",
) -> list:
    prior = "\n".join(f"- {x}" for x in prior_hints) if prior_hints else "(none yet)"
    if task_type == "math":
        system = (
            "You are a Socratic tutor for math problems. You see the correct "
            "answer and reference solution as PRIVATE information. Give one "
            "short corrective hint that helps the student repair the first "
            "important mistake in their attempt. Absolute rules: do NOT reveal "
            "the final answer, do NOT write a boxed answer, and do NOT solve the "
            "whole problem. Point to what to reconsider."
        )
        user = (
            f"Problem:\n{str(problem).strip()}\n\n"
            f"Final answer (PRIVATE - never reveal directly): {str(gold).strip()}\n\n"
            f"Reference solution (PRIVATE):\n{str(teacher_context or '').strip()[:3200]}\n\n"
            f"Hints already given:\n{prior}\n\n"
            f"Student's incorrect attempt:\n{str(last_attempt)[:2600]}\n\n"
            "Return only the hint text (one short paragraph)."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    system = (
        "You are a Socratic tutor for multiple-choice science questions. You see "
        "the correct option as PRIVATE information. Give one short corrective hint "
        "that guides the student to reconsider their mistake. Absolute rules: do "
        "NOT reveal or name the correct option letter, do NOT write an <answer> "
        "block, and do NOT just give the answer. Point at the first flawed step "
        "and suggest what to reconsider."
    )
    user = (
        f"Question:\n{str(problem).strip()}\n\n"
        f"Correct option (PRIVATE — never reveal): {str(gold).strip()}\n\n"
        f"Hints already given:\n{prior}\n\n"
        f"Student's incorrect attempt:\n{str(last_attempt)[:2200]}\n\n"
        "Return only the hint text (one short paragraph)."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_reflection_messages(
    problem: str,
    gold: str,
    wrong_rollout: str,
    wrong_rollouts: Sequence[str] | None = None,
    correct_rollout: str | None = None,
    task_type: str = "math",
    prompt_mode: str | None = None,
) -> list:
    """ROSD-style reflector prompt.

    The paper uses a self-reflector to compare a wrong rollout with a correct
    solution/rollout and emit ``<error_quote>`` plus ``<explanation>``. This
    project consumes only ``<explanation>`` as the key corrective idea.
    """
    resolved_prompt_mode = normalize_reflection_prompt_mode(prompt_mode)
    if resolved_prompt_mode == "two_stage_scope_xml_v1":
        raise ValueError(
            "two_stage_scope_xml_v1 requires build_scope_classifier_messages followed by "
            "build_conditional_reflection_messages"
        )
    if resolved_prompt_mode == "deepseek_r3_el_p1_noref_xml_v1":
        latest_wrong = str(wrong_rollout or "").strip()
        rollouts = [
            str(value or "").strip()
            for value in (wrong_rollouts or [])
            if str(value or "").strip()
        ]
        if rollouts:
            latest_wrong = rollouts[-1]
        # This is the frozen R3-EL prompt.  Gold and correct_rollout are
        # deliberately ignored; the complete candidate is the only anchor
        # source and is never clipped.
        user = _R3_EL_USER.format(
            problem=str(problem).strip(),
            reasoning=latest_wrong,
        )
        return [
            {"role": "system", "content": _R3_EL_SYSTEM},
            {"role": "user", "content": user},
        ]
    if resolved_prompt_mode == P1_R4_ALO_MODE:
        latest_wrong = str(wrong_rollout or "").strip()
        rollouts = [
            str(value or "").strip()
            for value in (wrong_rollouts or [])
            if str(value or "").strip()
        ]
        if rollouts:
            latest_wrong = rollouts[-1]
        # Gold and correct_rollout are intentionally excluded from R4-ALO.
        return build_p1_r4_alo_messages(problem, latest_wrong)
    if resolved_prompt_mode == P1_R4_SID_MODE:
        latest_wrong = str(wrong_rollout or "")
        rollouts = [
            str(value or "")
            for value in (wrong_rollouts or [])
            if str(value or "").strip()
        ]
        if rollouts:
            latest_wrong = rollouts[-1]
        # Gold and correct_rollout are intentionally excluded. Candidate text
        # is rendered as deterministic opaque-id segments so the response need
        # not reproduce the source error.
        return build_p1_r4_sid_messages(problem, latest_wrong)
    if resolved_prompt_mode == P1_R4_SIL_MODE:
        latest_wrong = str(wrong_rollout or "")
        rollouts = [
            str(value or "")
            for value in (wrong_rollouts or [])
            if str(value or "").strip()
        ]
        if rollouts:
            latest_wrong = rollouts[-1]
        # Gold and correct_rollout are intentionally excluded. The opaque id
        # remains machine-facing, while the Key Idea carries a semantic point
        # of application that is usable with this candidate hidden.
        return build_p1_r4_sil_messages(problem, latest_wrong)
    if resolved_prompt_mode == P1_R4_SQL_MODE:
        latest_wrong = str(wrong_rollout or "")
        rollouts = [
            str(value or "")
            for value in (wrong_rollouts or [])
            if str(value or "").strip()
        ]
        if rollouts:
            latest_wrong = rollouts[-1]
        # Gold and correct_rollout are intentionally excluded. The minimum
        # exact quote remains machine-facing; the Key Idea supplies the
        # semantic point of application to a solver with this source hidden.
        return build_p1_r4_sql_messages(problem, latest_wrong)
    if resolved_prompt_mode in {
        "deepseek_scope_noref_xml_v1",
        "deepseek_legacy_quote_noref_xml_v1",
        "deepseek_scope_bounded_noref_xml_v1",
    }:
        latest_wrong = str(wrong_rollout or "").strip()
        rollouts = [
            str(value or "").strip()
            for value in (wrong_rollouts or [])
            if str(value or "").strip()
        ]
        if rollouts:
            latest_wrong = rollouts[-1]
        system = (
            "You are a careful mathematical error analyst. Diagnose a student's "
            "incorrect solution without access to a reference solution. Do not "
            "solve the whole problem, reveal the final answer, or write a corrected "
            "full solution. Return only the requested XML fields."
        )
        common = (
            f"[PROBLEM]\n{str(problem).strip()}\n\n"
            "[STUDENT ATTEMPT - ONLY QUOTE SOURCE]\n"
            f"{latest_wrong[:_TWO_SCOPE_QUOTE_SOURCE_MAX_CHARS]}"
        )
        if resolved_prompt_mode == "deepseek_legacy_quote_noref_xml_v1":
            user = (
                "Identify the first causal mathematical error in the student's "
                "attempt. Copy the smallest exact, contiguous substring that contains "
                "that error. Prefer the cause over a downstream symptom or final-answer "
                "line. Explain why it is wrong and give an answer-free correction "
                "direction. The quote must preserve the source characters exactly.\n\n"
                "Return exactly:\n"
                "<error_quote>exact substring from STUDENT ATTEMPT</error_quote>\n"
                "<explanation>answer-free diagnosis and correction direction</explanation>\n\n"
                f"{common}"
            )
        elif resolved_prompt_mode == "deepseek_scope_noref_xml_v1":
            user = (
                "Classify the first causal error in the student's attempt.\n\n"
                "LOCAL means the mathematical representation and core strategy are "
                "still usable and there is a valid prefix before one first causal "
                "error. For LOCAL, copy the smallest exact contiguous substring "
                "containing that error; supervision will begin at the quote and cover "
                "the remaining suffix.\n\n"
                "GLOBAL means a foundational premise, representation, invariant, case "
                "decomposition, or core strategy is methodologically wrong, so no "
                "stable prefix supports suffix-only repair. GLOBAL supervises the full "
                "attempt and therefore must not emit an error quote.\n\n"
                "Give an answer-free explanation of the error and correction direction. "
                "Do not use GLOBAL merely because one local error causes many later "
                "mistakes; prefer the first causal error.\n\n"
                "Return exactly one schema:\n"
                "Local:\n<error_scope>local</error_scope>\n"
                "<error_quote>exact substring from STUDENT ATTEMPT</error_quote>\n"
                "<explanation>answer-free diagnosis and correction direction</explanation>\n\n"
                "Global:\n<error_scope>global</error_scope>\n"
                "<explanation>answer-free diagnosis and correction direction</explanation>\n\n"
                f"{common}"
            )
        else:
            user = (
                "Classify the first causal error in the student's attempt.\n\n"
                "LOCAL means that the mathematical representation and core strategy "
                "are still usable and there is a valid prefix before one first causal "
                "error. For LOCAL, copy the smallest exact, contiguous substring that "
                "contains that first causal error. The quote start will supervise that "
                "location and every later token, including downstream errors.\n\n"
                "GLOBAL means that a foundational premise, representation, invariant, "
                "case decomposition, or core strategy is methodologically wrong. For "
                "GLOBAL, copy the earliest compact exact, contiguous region that fully "
                "states the key mistaken method or strategic commitment. End the quote "
                "exactly where that methodological error has been fully expressed; do "
                "not extend it into downstream calculations, symptoms, or the final "
                "answer. Supervision will cover the response beginning through the end "
                "of this quote.\n\n"
                "For both scopes, first select one span in STUDENT ATTEMPT and then "
                "copy-paste it into error_quote character for character; do not "
                "normalize LaTeX, Unicode, punctuation, whitespace, or markdown. "
                "Prefer 15-300 characters; if a short quote repeats, extend it with "
                "adjacent source text until unique. Prefer the causal error over a "
                "later symptom. Keep the answer-free explanation under 80 words and "
                "the entire response under 250 words.\n\n"
                "Return exactly:\n"
                "<error_scope>local|global</error_scope>\n"
                "<error_quote>exact substring from STUDENT ATTEMPT</error_quote>\n"
                "<explanation>answer-free diagnosis and correction direction</explanation>\n\n"
                f"{common}"
            )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if resolved_prompt_mode == "external_scope_then_quote_action_xml_v1":
        system = (
            "You are a careful math error analyst. Use the private reference only to diagnose the incorrect attempt. "
            "Classify the supervision scope and produce only the requested XML. Do not reveal the private final answer "
            "or write a corrected full solution."
        )
        latest_wrong = str(wrong_rollout or "").strip()
        rollouts = [str(x or "").strip() for x in (wrong_rollouts or []) if str(x or "").strip()]
        if rollouts:
            latest_wrong = rollouts[-1]
        rubric = (
            "Use this local-repair certificate silently. Mark LOCAL if and only if all three statements\n"
            "are true: (1) every substantive claim before the first causal error can remain unchanged; (2) the same mathematical\n"
            "representation and core strategy can continue; and (3) replacing one exact ATTEMPT span is sufficient before\n"
            "regenerating the suffix. An error may occur early and still be LOCAL when the valid setup before it survives.\n\n"
            "Mark GLOBAL only when you can identify a premise, representation, or strategy that fails before any such repairable\n"
            "span, or when no stable derivation exists. Do not default to GLOBAL merely because the solution needs substantial\n"
            "downstream recomputation. If the diagnosis identifies one concrete first operation or claim as wrong and the prefix\n"
            "before it is valid, prefer LOCAL.\n\n"
            "Boundary cards:\n"
            "- LOCAL: the reasoning before one first causal error is valid and the same representation and strategy remain usable\n"
            "  after replacing that error. Many wrong later lines may all be consequences of this one local error.\n"
            "- GLOBAL: no valid reusable prefix leads into the attempted method because a premise, representation, or core\n"
            "  strategy is already wrong, or the attempt repeatedly restarts/guesses without a stable derivation.\n"
            "A quotable false sentence alone does not prove LOCAL; many downstream errors alone do not prove GLOBAL.\n\n"
            "Decide scope before selecting any quote. If and only if the decision is LOCAL, locate the\n"
            "earliest causal ATTEMPT span after the reusable prefix. Copy a minimal exact substring containing the false claim or\n"
            "operation, not a heading, restatement, generic setup sentence, downstream symptom, or final-answer line when an\n"
            "earlier causal error exists.\n\n"
            "The explanation must be mathematically specific and actionable. Write exactly two labeled clauses:\n"
            "'Error: <the first causal mathematical error and why it is wrong>. Next operation: <the next concrete mathematical\n"
            "operation, invariant, case split, or constraint to apply without giving the final answer>.' Vague advice such as\n"
            "'rethink', 'rebuild correctly', 'recompute carefully', or 'follow the reference' is invalid by itself."
        )
        schema = (
            "Return exactly one schema and no surrounding prose.\n"
            "Global:\n"
            "<error_scope>global</error_scope>\n"
            "<explanation>diagnosis and correction direction</explanation>\n\n"
            "Local:\n"
            "<error_scope>local</error_scope>\n"
            "<error_quote>exact substring from ATTEMPT</error_quote>\n"
            "<explanation>diagnosis and correction direction</explanation>"
        )
        user = (
            f"{rubric}\n\n"
            f"[Problem]\n{str(problem).strip()}\n\n"
            f"[Private Reference Solution]\n"
            f"{str(correct_rollout or '').strip()[:_TWO_SCOPE_SOLUTION_MAX_CHARS]}\n\n"
            f"[ATTEMPT - ONLY QUOTE SOURCE]\n"
            f"{latest_wrong[:_TWO_SCOPE_QUOTE_SOURCE_MAX_CHARS]}\n\n"
            f"Now classify this target attempt, not an example.\n\n{schema}"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if resolved_prompt_mode == "scope_sft_xml_v1":
        system = (
            "Diagnose the incorrect mathematical attempt using the private reference. Return only the "
            "structured global/local key idea. Never reveal the private final answer or a full solution."
        )
        latest_wrong = str(wrong_rollout or "").strip()
        rollouts = [str(x or "").strip() for x in (wrong_rollouts or []) if str(x or "").strip()]
        if rollouts:
            latest_wrong = rollouts[-1]
        rubric = (
            "Choose exactly one supervision scope by applying this prefix-preservation test silently:\n"
            "Imagine replacing one first causal error while keeping every substantive claim before it unchanged. "
            "If the same strategy could then continue toward a correct solution, choose local. If any earlier premise, "
            "interpretation, representation, or strategy must also change, choose global.\n\n"
            "GLOBAL means the whole erroneous trajectory is the training target. Choose global for a wrong problem "
            "model or strategy, repeated restarts or guesses, contradictions, an abandoned derivation, or no stable "
            "argument. A global attempt can contain many quotable false sentences: merely finding one false sentence "
            "is NOT evidence for local. If uncertain, choose global. Global emits no error_quote.\n\n"
            "LOCAL means only the suffix from the first causal error is the training target. Choose local only when "
            "the earlier reasoning is coherent and reusable. Copy one minimal continuous 12-200 character substring "
            "from the STUDENT ATTEMPT as error_quote. Copy characters exactly; never paraphrase, normalize, repair, "
            "or copy from the reference. Before returning local, silently verify that the quote appears verbatim in "
            "the student attempt.\n\n"
            "For either scope, explanation must state the error and the correction direction in 1-3 concise sentences "
            "without disclosing the final answer."
        )
        user = (
            f"{rubric}\n\n"
            f"[Problem]\n{str(problem).strip()}\n\n"
            f"[Private Reference Solution]\n{str(correct_rollout or '').strip()}\n\n"
            f"[ATTEMPT - ONLY QUOTE SOURCE]\n{latest_wrong}\n\n"
            "Return only the XML schema. For global output <error_scope>global</error_scope> and "
            "<explanation>. For local also output an exact <error_quote> from the attempt."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if resolved_prompt_mode == "two_scope_xml_v1":
        system = (
            "You are a careful math error analyst. The correct answer and reference solution are private "
            "diagnostic context. Classify only genuine model errors as local or global and return only the "
            "requested XML fields. Do not reveal the final answer, a corrected final value, or a complete "
            "solution. Treat global as the default; use local only after all local conditions are verified."
        )
        latest_wrong = str(wrong_rollout or "").strip()
        rollouts = [str(x or "").strip() for x in (wrong_rollouts or []) if str(x or "").strip()]
        if rollouts:
            latest_wrong = rollouts[-1]
        rubric = (
            "Choose exactly one scope. Start from global. Change the decision to local only when ALL three tests "
            "are true: (1) the prefix before one clause is mathematically coherent and reusable; (2) that clause "
            "is the first causal error, not merely a later symptom; and (3) supervision from that clause to the end "
            "is a sensible repair target. Finding any incorrect sentence is not enough.\n"
            "- global: keep this default when any local test fails. In particular, use global for repeated guesses, "
            "contradictory candidates, cycling approaches, an unresolved chain with no stable derivation, or a "
            "fundamentally wrong interpretation/strategy that requires rebuilding the reasoning. A global response "
            "may contain quotable false sentences; do not emit <error_quote> for global.\n"
            "- local: use only after all three tests pass. Then copy the smallest useful continuous clause that "
            "contains the first causal error. The quote is a mechanical copy task: copy characters only from the "
            "QUOTE SOURCE, do not paraphrase, repair, normalize, or copy from the private reference. Verify the "
            "exact substring before returning it.\n\n"
            "Scope archetypes: a response that tries unrelated values and ends without a consistent derivation is "
            "global; a correct derivation that first drops one denominator and then continues from that mistake is "
            "local.\n\n"
            "Synthetic format examples (they are not related to the actual problem):\n"
            "- If a source says 'Try x=1: contradiction. Maybe x=2. No. Perhaps another formula...' and never "
            "forms a consistent derivation, return <error_scope>global</error_scope><explanation>The attempt cycles "
            "through guesses without establishing a valid governing relation; rebuild the argument from a stable "
            "invariant.</explanation> Do not add an error_quote.\n"
            "- If a coherent source first says 'Dividing by d gives q = p' when it should retain the denominator, "
            "return <error_scope>local</error_scope><error_quote>Dividing by d gives q = p</error_quote> followed "
            "by a concise explanation. The quote is copied exactly.\n\n"
            "The explanation must state what is wrong and the corrective reasoning direction in 1-3 concise "
            "sentences. It must not state the private final answer or a corrected final value."
        )
        user = (
            f"{rubric}\n\n"
            f"[Problem]\n{str(problem).strip()}\n\n"
            f"[Private Reference Solution]\n{str(correct_rollout or '').strip()[:_TWO_SCOPE_SOLUTION_MAX_CHARS]}\n\n"
            f"[QUOTE SOURCE - Student Incorrect Attempt]\n{latest_wrong[:_TWO_SCOPE_QUOTE_SOURCE_MAX_CHARS]}\n\n"
            "For local return exactly:\n"
            "<error_scope>local</error_scope>\n"
            "<error_quote>verbatim clause copied from QUOTE SOURCE</error_quote>\n"
            "<explanation>concise diagnosis and correction direction</explanation>\n\n"
            "For global return exactly:\n"
            "<error_scope>global</error_scope>\n"
            "<explanation>concise diagnosis and correction direction</explanation>"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    if resolved_prompt_mode == "routed_xml_v1":
        system = (
            "You are a careful math error analyst. Use the private correct solution only to diagnose "
            "the student. Return exactly the three XML fields requested and no other text. "
            "The error quote is a copy task."
        )
        latest_wrong = str(wrong_rollout or "").strip()
        rollouts = [str(x or "").strip() for x in (wrong_rollouts or []) if str(x or "").strip()]
        if rollouts:
            latest_wrong = rollouts[-1]
        rubric = (
            "Classify the student's first consequential error into exactly one type:\n"
            "- local_reasoning: a specific algebraic, arithmetic, logical, or factual step is the first causal error.\n"
            "- final_answer: the reasoning is substantially correct, but the final answer has a sign, option, "
            "unit, formatting, copying, or completeness error.\n"
            "- global_understanding: the attempt solves the wrong task, relies on a globally invalid "
            "interpretation/strategy, or has no trustworthy local repair point.\n\n"
            "For local_reasoning or final_answer, copy one minimal but complete mathematical clause from the "
            "QUOTE SOURCE. Prefer 12-180 characters. Do not copy a heading, transition word, problem "
            "restatement, or a statement that is actually correct while the error occurs later. "
            "For global_understanding, output NONE as the error quote.\n\n"
            "The explanation must be a forward-looking correction in 1-3 sentences: state what is wrong and "
            "what operation or reasoning should replace it. Do not reveal the private final answer, write a "
            "boxed answer, or solve the whole problem."
        )
        user = (
            f"{rubric}\n\n"
            f"[Problem]\n{str(problem).strip()}\n\n"
            f"[Private Correct Final Answer]\n{str(gold).strip()}\n\n"
            f"[Private Correct Solution]\n{str(correct_rollout or '').strip()[:_ROUTED_SOLUTION_MAX_CHARS]}\n\n"
            f"[QUOTE SOURCE - Student Incorrect Attempt]\n{latest_wrong[:_ROUTED_QUOTE_SOURCE_MAX_CHARS]}\n\n"
            "Analyze the actual case above. Return exactly:\n"
            "<error_type>local_reasoning|final_answer|global_understanding</error_type>\n"
            "<error_quote>verbatim source clause, or NONE for global_understanding</error_quote>\n"
            "<explanation>concise forward-looking correction without the final answer</explanation>"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    system = (
        "You are a careful tutor. Respond strictly in the required format and nothing else. "
        "The <error_quote> field is a copy task: copy text only from the quoted incorrect attempt."
    )
    rollouts = [str(x or "").strip() for x in (wrong_rollouts or []) if str(x or "").strip()]
    if len(rollouts) >= 2:
        prior_blocks = "\n\n".join(
            f"[Earlier Incorrect Attempt {i}]\n{text[:2200]}"
            for i, text in enumerate(rollouts[:-1], start=1)
        )
        source_label = "[QUOTE SOURCE - Most Recent Incorrect Attempt]"
        wrong_block = (
            f"{prior_blocks}\n\n" if prior_blocks else ""
        ) + f"{source_label}\n{rollouts[-1][:3000]}"
        diagnose = (
            "Use the earlier incorrect attempts only as context for the evolving mistake. "
            f"The <error_quote> MUST contain only a short verbatim substring from {source_label}. "
            "Do not quote the problem, correct solution, final answer, or these instructions. "
            "If the first flawed step is unclear, quote the earliest local claim in the source "
            "whose logic should be checked."
        )
    else:
        source_label = "[QUOTE SOURCE - Incorrect Solution]"
        wrong_block = f"{source_label}\n{str(wrong_rollout or '').strip()[:3000]}"
        diagnose = (
            f"Diagnose {source_label}. The <error_quote> MUST contain only a short verbatim "
            f"substring from {source_label}. Do not quote the problem, correct solution, "
            "final answer, or these instructions. If the first flawed step is unclear, "
            "quote the earliest local claim in the source whose logic should be checked."
        )
    user = (
        f"[Problem]\n{str(problem).strip()}\n\n"
        f"[Private Correct Final Answer]\n{str(gold).strip()}\n\n"
        f"[Correct Solution]\n{str(correct_rollout or '').strip()[:3600]}\n\n"
        f"{wrong_block}\n\n"
        f"{diagnose} Respond strictly in the following format:\n"
        "<error_quote>\n"
        "(A short exact substring copied only from the QUOTE SOURCE block)\n"
        "</error_quote>\n"
        "<explanation>\n"
        "(Explain the exact mistake in the quote, how to fix it, and what the correct logic is)\n"
        "</explanation>"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_scope_classifier_messages(
    problem: str,
    wrong_rollout: str,
    correct_rollout: str | None = None,
) -> list:
    """Stage A: decide whether the prefix before one error remains reusable.

    The deliberately binary output prevents quote generation from biasing the
    local/global decision. ``yes`` maps to local and ``no`` maps to global.
    """
    system = (
        "You are a math error analyst. Compare the student's incorrect attempt with the private "
        "reference. Return exactly one XML tag and no other text."
    )
    user = (
        "Choose between two equally weighted definitions. YES means: valid substantive progress, then "
        "one quotable first local error, with the overall method retained. NO means: wrong problem model "
        "or strategy, non-convergent guessing, contradictions, or no stable derivation before one error.\n\n"
        f"[Problem]\n{str(problem).strip()}\n\n"
        f"[Private Reference Solution]\n{str(correct_rollout or '').strip()[:_TWO_SCOPE_SOLUTION_MAX_CHARS]}\n\n"
        f"[Student Incorrect Attempt]\n{str(wrong_rollout or '').strip()[:_TWO_SCOPE_QUOTE_SOURCE_MAX_CHARS]}\n\n"
        "Return exactly <prefix_reusable>yes</prefix_reusable> or "
        "<prefix_reusable>no</prefix_reusable>."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_prefix_reusable(text: str) -> dict[str, Any]:
    """Strictly parse the Stage-A binary decision."""
    raw = str(text or "")
    matches = re.findall(
        r"<prefix_reusable>\s*(yes|no)\s*</prefix_reusable>",
        raw,
        flags=re.IGNORECASE,
    )
    residual = re.sub(
        r"<prefix_reusable>\s*(?:yes|no)\s*</prefix_reusable>",
        "",
        raw,
        flags=re.IGNORECASE,
    ).strip()
    valid = len(matches) == 1 and not residual
    value = matches[0].lower() if valid else ""
    return {
        "prefix_reusable_raw": raw,
        "prefix_reusable": value,
        "prefix_reusable_valid": valid,
        "predicted_scope": "local" if value == "yes" else "global" if value == "no" else "",
    }


def build_conditional_reflection_messages(
    problem: str,
    wrong_rollout: str,
    correct_rollout: str | None,
    error_scope: str,
) -> list:
    """Stage B: generate only the artifact required by the fixed scope."""
    scope = _normalize_error_scope(error_scope)
    if scope not in _VALID_ERROR_SCOPES:
        raise ValueError("error_scope must be local or global")
    system = (
        "You are a careful math error analyst. The scope was already fixed by a separate classifier. "
        "Use the private reference only for diagnosis. Never state, compare, or imply the correct final "
        "answer, any corrected final expression, or any evaluated result taken from the private reference. "
        "Give only the error mechanism and a qualitative correction direction. Return only the requested "
        "XML fields."
    )
    common = (
        f"[Problem]\n{str(problem).strip()}\n\n"
        f"[Private Reference Solution]\n{str(correct_rollout or '').strip()[:_TWO_SCOPE_SOLUTION_MAX_CHARS]}\n\n"
        f"[QUOTE SOURCE - Student Incorrect Attempt]\n{str(wrong_rollout or '').strip()[:_TWO_SCOPE_QUOTE_SOURCE_MAX_CHARS]}\n\n"
    )
    if scope == "global":
        task = (
            "The fixed scope is global. Explain in 1-3 concise sentences why the reasoning needs a "
            "global rebuild and name the kind of setup or invariant that should replace it. Do not emit an "
            "error quote, equations, evaluated quantities, or a correct answer.\n\n"
            "Return exactly:\n<explanation>concise diagnosis and correction direction</explanation>"
        )
    else:
        task = (
            "The fixed scope is local. First locate the first causal error in the QUOTE SOURCE, then copy a "
            "single short continuous substring containing it character-for-character, ideally 8-100 "
            "characters and never more than 180 characters. Stop the quote immediately after the defective "
            "clause. Preserve every symbol, space, and LaTeX command; do not clean, reformat, paraphrase, or "
            "reconstruct an equivalent statement. Then explain in 1-3 concise sentences what is wrong and "
            "the qualitative corrective direction, without giving any correct value or final expression.\n\n"
            "Return exactly:\n<error_quote>verbatim clause copied from QUOTE SOURCE</error_quote>\n"
            "<explanation>concise diagnosis and correction direction</explanation>"
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": common + task},
    ]


class ExhaustedEmptyHintContentError(RuntimeError):
    """All bounded provider attempts completed without message content."""

    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = copy.deepcopy(diagnostics or {})


class RequiredRecoveryCacheMiss(RuntimeError):
    """A known completed provider identity did not reproduce its frozen key."""


def build_reference_multiround_messages(item: dict[str, Any]) -> list[dict[str, str]]:
    """Build a frozen reference-aware KI1 or state-revision request."""
    protocol = str(item.get("reference_protocol") or "reference_p1_p2_v1")
    scheme3bv2 = protocol in {
        "reference_scheme3bv2_replacement_v1",
        "reference_scheme3bv2_anchor_replacement_v2",
    }
    prompts = _load_reference_multiround_prompts()
    problem = str(item.get("problem") or "").strip()
    reasoning = str(item.get("last_attempt") or "").strip()
    reference = str(item.get("teacher_context") or "").strip()
    round_number = int(item.get("round") or 1)
    if not problem or not reasoning or not reference:
        raise ValueError("reference multiround request requires problem, reasoning, and reference solution")
    if scheme3bv2:
        bundle = _load_reference_scheme3bv2_prompts()
        profile = bundle["profiles"][bundle["default"]]
        context_limits = profile.get("context_limits") or {}
        try:
            response_max_chars = int(
                context_limits.get("response_max_chars", 1_000_000_000)
            )
            reference_max_chars = int(
                context_limits.get("private_reference_solution_max_chars", 1_000_000_000)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Scheme3b-v2 context limits must be integers") from exc
        if response_max_chars <= 0 or reference_max_chars <= 0:
            raise ValueError("Scheme3b-v2 context limits must be positive")
        slice_mode = str(context_limits.get("slice_mode") or "strip_then_prefix")
        if slice_mode != "strip_then_prefix":
            raise ValueError(
                "Scheme3b-v2 supports only strip_then_prefix context slicing"
            )
        # Keep the full response/reference in the trace item, but bound only the
        # teacher-visible prompt fields.  The child release uses the LONGGEN
        # adapted 5k/5k character visibility contract for every G1/G2/G3 call.
        visible_reasoning = reasoning[:response_max_chars]
        visible_reference = reference[:reference_max_chars]
        if round_number not in {1, 2, 3}:
            raise ValueError(
                f"Scheme3b-v2 supports exactly rounds 1-3, got {round_number}"
            )
        prior = [
            str(value or "").strip()
            for value in (item.get("hints") or [])
            if str(value or "").strip()
        ]
        round_key = f"g{round_number}"
        template = str(profile["rounds"][round_key]["user_template"])
        if round_number == 1:
            if prior:
                raise ValueError("Scheme3b-v2 G1 must not receive prior hints")
            user = template.format(
                problem=problem,
                incorrect_response=visible_reasoning,
                incorrect_response_truncated=visible_reasoning,
                private_answer=str(item.get("gold") or "").strip(),
                private_reference_solution=visible_reference,
                private_reference_solution_truncated=visible_reference,
            )
        else:
            if len(prior) != 1:
                raise ValueError(
                    "Scheme3b-v2 G2/G3 require only the current replacement hint"
                )
            user = template.format(
                problem=problem,
                current_repair_hint=prior[0],
                current_repair_guidance=prior[0],
                latest_incorrect_response=visible_reasoning,
                latest_incorrect_response_truncated=visible_reasoning,
                private_answer=str(item.get("gold") or "").strip(),
                private_reference_solution=visible_reference,
                private_reference_solution_truncated=visible_reference,
            )
        system = str(profile["system"])
    elif round_number == 1:
        user = str(prompts["ki1_user"]).format(
            problem=problem,
            reasoning=reasoning,
            reference_solution=reference,
        )
        system = str(prompts["ki1_system"])
    elif round_number == 2:
        prior = [str(value or "").strip() for value in (item.get("hints") or []) if str(value or "").strip()]
        if len(prior) != 1:
            raise ValueError("reference revision requires exactly one previous repair state")
        user_key = "p2_user"
        system_key = "p2_system"
        user = str(prompts[user_key]).format(
            round_number=round_number,
            problem=problem,
            latest_wrong=reasoning,
            previous_repair_state=prior[0],
            previous_key_idea=prior[0],
            reference_solution=reference,
        )
        system = str(prompts[system_key])
    else:
        raise ValueError(f"reference multiround supports only rounds 1 and 2, got {round_number}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


_SCHEME3BV2_NO_REPAIR_RE = re.compile(
    r"\A\s*<decision>no_repair</decision>\s*\Z",
    re.DOTALL,
)
_SCHEME3BV2_REPAIR_RE = re.compile(
    r"\A\s*<decision>repair</decision>\s*"
    r"(?:<error_quote>|<repair_anchor>)(?P<anchor>.*?)(?:</error_quote>|</repair_anchor>)\s*"
    r"(?:<repair_guidance>|<repair_hint>)(?P<hint>.*?)(?:</repair_guidance>|</repair_hint>)\s*\Z",
    re.DOTALL,
)
_SCHEME3BV2_NODECISION_REPAIR_RE = re.compile(
    r"\A\s*(?:<error_quote>|<repair_anchor>)(?P<anchor>.*?)(?:</error_quote>|</repair_anchor>)\s*"
    r"(?:<repair_guidance>|<repair_hint>)(?P<hint>.*?)(?:</repair_guidance>|</repair_hint>)\s*\Z",
    re.DOTALL,
)
_SCHEME3BV2_SELF_MISSING_FINAL_CLOSE_RE = re.compile(
    r"\A\s*<decision>repair</decision>\s*"
    r"(?:<error_quote>|<repair_anchor>)(?P<anchor>[^<>]+?)(?:</error_quote>|</repair_anchor>)\s*"
    r"(?P<hint_tag><repair_guidance>|<repair_hint>)(?P<hint>[^<>]+?)\s*\Z",
    re.DOTALL,
)
_SCHEME3BV2_NODECISION_MISSING_FINAL_CLOSE_RE = re.compile(
    r"\A\s*(?:<error_quote>|<repair_anchor>)(?P<anchor>[^<>]+?)(?:</error_quote>|</repair_anchor>)\s*"
    r"(?P<hint_tag><repair_guidance>|<repair_hint>)(?P<hint>[^<>]+?)\s*\Z",
    re.DOTALL,
)


def normalize_scheme3bv2_self_terminal_close(
    text: str,
    reference_protocol: str,
) -> tuple[str, bool]:
    """Close only an otherwise complete self-generated Scheme3b-v2 packet.

    Local Qwen generations occasionally stop at the token budget after writing
    the Repair Hint body but before the final ``</repair_hint>``.  This helper
    accepts exactly that terminal omission.  It never invents or edits a
    decision, anchor, hint body, nested tag, or surrounding prose.
    """
    raw = str(text or "")
    if str(reference_protocol) not in {
        "reference_scheme3bv2_replacement_v1",
        "reference_scheme3bv2_anchor_replacement_v2",
    }:
        return raw, False

    think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if think_match:
        prefix = raw[:think_match.end()]
        body = raw[think_match.end():].lstrip()
    else:
        prefix = ""
        body = raw

    if (_SCHEME3BV2_NO_REPAIR_RE.fullmatch(body)
            or _SCHEME3BV2_REPAIR_RE.fullmatch(body)
            or _SCHEME3BV2_NODECISION_REPAIR_RE.fullmatch(body)):
        return raw, False
    match = _SCHEME3BV2_SELF_MISSING_FINAL_CLOSE_RE.fullmatch(body)
    if not match:
        match = _SCHEME3BV2_NODECISION_MISSING_FINAL_CLOSE_RE.fullmatch(body)
    if not match:
        return raw, False
    if not str(match.group("anchor") or "").strip() or not str(match.group("hint") or "").strip():
        return raw, False
    suffix = "</repair_hint>" if match.group("hint_tag") == "<repair_hint>" else "</repair_guidance>"
    repaired_body = body.rstrip() + suffix
    return (f"{prefix}\n{repaired_body}" if prefix else repaired_body), True
_SCHEME3BV2_ERROR_CUE_RE = re.compile(
    r"earliest remaining (?:mathematical )?(?:error|mistake)"
    r"|earliest (?:mathematical |decisive |causal )?(?:error|mistake)"
    r"|(?:mathematical )?(?:error|mistake) (?:occurs|is)"
    r"|\b(?:incorrect|wrong|false|unsupported)\b"
    r"|最早[^。；;]{0,40}(?:错误|问题)"
    r"|(?:错误|有误|不成立|瓶颈)",
    re.IGNORECASE,
)
_SCHEME3BV2_CORRECT_CUE_RE = re.compile(
    r"\b(?:is|are|now|already|remains?) correct\b|正确(?:的|无误|了|是)",
    re.IGNORECASE,
)
_SCHEME3BV2_MARKER_RE = re.compile(
    r"\b(?:Step|Case|Part)\s+[0-9IVX]+(?:\s*[/–-]\s*[0-9IVX]+)?"
    r"|第[一二三四五六七八九十0-9]+(?:步|部分|种|类|种情况)",
    re.IGNORECASE,
)
_SCHEME3BV2_MATH_RE = re.compile(
    r"\\\((?P<inline>.+?)\\\)|\\\[(?P<display>.+?)\\\]|\$(?P<dollar>[^$]+)\$",
    re.DOTALL,
)
_SCHEME3BV2_QUOTE_RE = re.compile(
    r"[\"“](?P<double>[^\"”]{4,220})[\"”]"
    r"|[\'‘](?P<single>[^\'’]{4,220})[\'’]",
    re.DOTALL,
)
_SCHEME3BV2_GENERIC_ANCHORS = {
    "eventually",
    "then for each such",
    "now compute the",
    "matrix exponentiation",
    "the original equation",
}


def _scheme3bv2_error_focus(hint: str) -> str:
    """Drop validation prose that precedes the first remaining-error cue."""
    text = str(hint or "").strip()
    starts: list[int] = []
    for match in _SCHEME3BV2_ERROR_CUE_RE.finditer(text):
        neighborhood = text[max(0, match.start() - 50): match.end() + 25]
        if (
            _SCHEME3BV2_CORRECT_CUE_RE.search(neighborhood)
            and not re.search(
                r"\b(?:but|however|remaining)\b|但|然而|不过",
                neighborhood,
                re.IGNORECASE,
            )
        ):
            continue
        starts.append(match.start())
    return text[min(starts):] if starts else text


def _scheme3bv2_anchor_candidates(
    hint: str,
    quote_source: str,
) -> list[tuple[int, str, str]]:
    """Derive bounded source-grounding candidates without another teacher call."""
    focus = _scheme3bv2_error_focus(hint)[:1000]
    candidates: list[tuple[int, str, str]] = []
    for match in _SCHEME3BV2_MARKER_RE.finditer(focus[:500]):
        candidates.append((0, match.group(0), "round_marker"))
    for match in _SCHEME3BV2_QUOTE_RE.finditer(focus[:700]):
        candidates.append(
            (1, str(match.group("double") or match.group("single")), "quoted_text")
        )
    for match in _SCHEME3BV2_MATH_RE.finditer(focus[:800]):
        candidates.append(
            (
                2,
                str(match.group("inline") or match.group("display") or match.group("dollar")),
                "math_span",
            )
        )

    loose_hint, _ = _loose_match_text_with_offsets(focus)
    loose_source, _ = _loose_match_text_with_offsets(str(quote_source or ""))
    matcher = difflib.SequenceMatcher(
        None,
        loose_hint,
        loose_source,
        autojunk=False,
    )
    for block in sorted(
        matcher.get_matching_blocks(),
        key=lambda value: value.size,
        reverse=True,
    ):
        fragment = loose_hint[block.a: block.a + block.size].strip()
        meaningful = sum(
            char.isalnum() or "\u4e00" <= char <= "\u9fff"
            for char in fragment
        )
        math_like = bool(re.search(r"[0-9=+*/<>]|\\", fragment))
        if (
            block.size >= 10
            and meaningful >= 8
            and (math_like or meaningful >= 14)
            and fragment.lower().strip(" .,:;") not in _SCHEME3BV2_GENERIC_ANCHORS
        ):
            candidates.append((3, fragment, "shared_fragment"))

    unique: list[tuple[int, str, str]] = []
    seen: set[str] = set()
    for priority, candidate, kind in candidates:
        key = re.sub(r"\s+", " ", str(candidate or "").strip()).lower()
        min_chars = 4 if kind == "round_marker" else 6
        if len(re.sub(r"\s+", "", key)) < min_chars or key in seen:
            continue
        seen.add(key)
        unique.append((priority, str(candidate).strip(), kind))
    return unique


def infer_scheme3bv2_error_location(
    repair_hint: str,
    quote_source: str,
) -> tuple[str, dict[str, Any], str | None]:
    """Reuse the original Scheme3b fuzzy locator on one-field Repair Hints."""
    accepted: list[tuple[int, float, float, str, str, dict[str, Any]]] = []
    for priority, candidate, kind in _scheme3bv2_anchor_candidates(
        repair_hint,
        quote_source,
    ):
        location = locate_error_quote(
            candidate,
            quote_source,
            mode="usable_for_loss",
            min_fuzzy_score=0.88,
            min_margin=0.04,
        )
        if (
            location.get("locatable")
            and location.get("usable_for_loss")
            and not location.get("ambiguous")
        ):
            accepted.append(
                (
                    priority,
                    float(location.get("score") or 0.0),
                    float(location.get("margin") or 0.0),
                    candidate,
                    kind,
                    location,
                )
            )
    if not accepted:
        return "", {
            "locatable": False,
            "usable_for_loss": False,
            "mode": "usable_for_loss",
            "method": None,
            "score": None,
            "margin": None,
            "start_char": None,
            "end_char": None,
            "matched_text": "",
            "occurrence_count": 0,
            "ambiguous": False,
            "reject_reason": "no_hint_derived_fuzzy_match",
        }, None
    priority, _, _, anchor, kind, location = sorted(
        accepted,
        key=lambda value: (value[0], -value[1], -value[2]),
    )[0]
    location = dict(location)
    location.update(
        {
            "hint_anchor_priority": int(priority),
            "hint_anchor_kind": kind,
            "hint_anchor_sha256": hashlib.sha256(
                anchor.encode("utf-8")
            ).hexdigest(),
        }
    )
    return anchor, location, kind


def sanitize_scheme3bv2_repair_hint_with_meta(
    text: str,
    gold: str,
    *,
    task_type: str,
    quote_source: str,
    round_number: int,
) -> tuple[str, dict[str, Any]]:
    """Parse LONGGEN error_quote/repair_guidance with the legacy gate."""
    raw = str(text or "")
    thinking = ""
    think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        body = raw[think_match.end():].strip()
    else:
        body = raw
    bundle = _load_reference_scheme3bv2_prompts()
    profile = bundle["profiles"][bundle["default"]]
    decision_mode = str((profile.get("output") or {}).get("decision_mode") or "with_no_repair")
    no_repair = bool(decision_mode == "with_no_repair" and _SCHEME3BV2_NO_REPAIR_RE.fullmatch(body))
    match = (_SCHEME3BV2_REPAIR_RE if decision_mode == "with_no_repair" else _SCHEME3BV2_NODECISION_REPAIR_RE).fullmatch(body)
    decision = "no_repair" if no_repair else "repair" if match else "invalid"
    anchor = str(match.group("anchor") if match else "").strip()
    hint = str(match.group("hint") if match else "").strip()
    nested_tag = bool(
        re.search(
            r"</?[A-Za-z_][A-Za-z0-9_.:-]*(?:\s[^<>]*)?>",
            "\n".join((anchor, hint)),
        )
    )
    schema_ok = bool(no_repair or (match and anchor and hint and not nested_tag))
    leaked = bool(
        contains_math_answer_leak(hint, gold)
        if task_type == "math" and hint
        else contains_letter_leak(hint, gold) if hint else False
    )
    location = (
        enhanced_locate(
            anchor,
            quote_source,
            min_fuzzy_score=0.78,
            min_margin=0.02,
        )
        if match and anchor and not nested_tag
        else {
            "locatable": False,
            "usable_for_loss": False,
            "ambiguous": False,
            "start_char": None,
            "end_char": None,
            "reject_reason": (
                "decision_no_repair" if no_repair else "invalid_repair_packet_schema"
            ),
        }
    )
    location_ok = bool(
        location.get("locatable")
        and isinstance(location.get("start_char"), int)
        and isinstance(location.get("end_char"), int)
        and int(location["end_char"]) > int(location["start_char"])
    )
    repair_ok = bool(
        decision == "repair" and schema_ok and not leaked and location_ok
    )
    if decision == "no_repair":
        fallback_reason = "decision_no_repair"
    elif not schema_ok:
        fallback_reason = "invalid_repair_packet_schema"
    elif leaked:
        fallback_reason = "answer_leak"
    elif not location_ok:
        fallback_reason = str(
            location.get("reject_reason") or "anchor_not_locatable_or_bounded"
        )
    else:
        fallback_reason = None
    meta: dict[str, Any] = {
        "reference_multiround": True,
        "reference_protocol": "reference_scheme3bv2_anchor_replacement_v2",
        "reference_round": int(round_number),
        "reference_solution_used_for_reflection": True,
        "decision": decision,
        "decision_mode": decision_mode,
        "previous_response_correct": decision == "no_repair",
        "repair_hint": hint,
        "repair_guidance": hint,
        "repair_signal": hint,
        "repair_signal_kind": (
            "no_repair" if decision == "no_repair" else "repair_hint"
        ),
        "repair_anchor": anchor,
        "machine_anchor_inferred_from_hint": False,
        "machine_anchor_explicit": bool(anchor),
        "hint_anchor_kind": "explicit_teacher_anchor" if anchor else None,
        "extracted_key_idea": bool(hint),
        "extracted_explanation": bool(hint),
        "explanation_open_no_close": False,
        "extracted_error_quote": bool(anchor),
        "error_quote": anchor,
        "raw_error_quote": anchor,
        "error_quote_location": location,
        "error_quote_source_valid": location_ok,
        "error_quote_fallback": None if repair_ok else "zero_loss",
        "error_quote_invalid_reason": (
            None if location_ok else str(location.get("reject_reason") or "unmatched")
        ),
        "error_quote_locator_mode": "scheme3bv2_explicit_anchor_locatable_v2",
        "error_route_mode": "legacy",
        "training_route": (
            "no_repair_terminal"
            if decision == "no_repair"
            else "local_suffix"
            if repair_ok
            else "zero_loss"
        ),
        "p1_schema_valid": schema_ok,
        "raw_chars": len(raw),
        "idea_chars": len(hint),
        "leaked": leaked,
        "fallback_reason": fallback_reason,
        "training_gate_passed": repair_ok,
        "teacher_privilege": "key_idea" if repair_ok else "solution",
    }
    if thinking:
        meta["thinking"] = thinking
    return (hint, meta) if repair_ok else ("", meta)


def sanitize_reference_multiround_with_meta(
    text: str,
    gold: str,
    *,
    task_type: str,
    quote_source: str,
    round_number: int,
    reference_protocol: str = "reference_p1_p2_v1",
) -> tuple[str, dict[str, Any]]:
    """Apply the lenient anchor/leak gate for reference repair signals."""
    if str(reference_protocol) in {
        "reference_scheme3bv2_replacement_v1",
        "reference_scheme3bv2_anchor_replacement_v2",
    }:
        return sanitize_scheme3bv2_repair_hint_with_meta(
            text,
            gold,
            task_type=task_type,
            quote_source=quote_source,
            round_number=round_number,
        )
    raw = str(text or "")
    decision = str(_extract_tag(raw, "decision") or "").strip().lower()
    anchor = str(_extract_tag(raw, "repair_anchor") or "").strip()
    key_idea = str(_extract_tag(raw, "key_idea") or "").strip()
    new_correction = str(_extract_tag(raw, "new_correction") or "").strip()
    effective_signal = key_idea
    location = locate_error_quote(
        anchor,
        quote_source,
        mode="locatable",
        min_fuzzy_score=0.88,
        min_margin=0.04,
    )
    anchor_ok = bool(
        decision == "repair"
        and location.get("locatable")
        and isinstance(location.get("start_char"), int)
        and isinstance(location.get("end_char"), int)
        and int(location["end_char"]) > int(location["start_char"])
    )
    key_leak = (
        contains_math_answer_leak(key_idea, gold)
        if task_type == "math" and key_idea
        else contains_letter_leak(key_idea, gold) if key_idea else False
    )
    correction_leak = (
        contains_math_answer_leak(new_correction, gold)
        if task_type == "math" and new_correction
        else contains_letter_leak(new_correction, gold) if new_correction else False
    )
    schema_ok = decision in {"repair", "no_repair"}
    repair_ok = bool(
        decision == "repair"
        and anchor_ok
        and effective_signal
        and (not correction_leak if int(round_number) >= 2 else True)
        and not key_leak
        and (
            int(round_number) == 1
            or bool(new_correction and not correction_leak)
        )
    )
    fallback_reason = None
    if decision == "no_repair":
        fallback_reason = "decision_no_repair"
    elif not schema_ok:
        fallback_reason = "invalid_decision"
    elif not anchor_ok:
        fallback_reason = "anchor_not_fuzzy_locatable"
    elif not effective_signal:
        fallback_reason = "missing_key_idea"
    elif key_leak or correction_leak:
        fallback_reason = "answer_leak"
    elif int(round_number) >= 2 and not new_correction:
        fallback_reason = "missing_new_correction"
    meta: dict[str, Any] = {
        "reference_multiround": True,
        "reference_protocol": str(reference_protocol),
        "reference_round": int(round_number),
        "reference_solution_used_for_reflection": True,
        "decision": decision or "invalid",
        "repair_anchor": anchor,
        "new_correction": new_correction,
        "key_idea": key_idea,
        "repair_signal": effective_signal,
        "repair_signal_kind": "key_idea",
        "extracted_key_idea": bool(key_idea),
        "extracted_explanation": bool(key_idea),
        "explanation_open_no_close": False,
        "extracted_error_quote": bool(anchor),
        "error_quote": anchor,
        "raw_error_quote": anchor,
        "error_quote_location": location,
        "error_quote_source_valid": anchor_ok,
        "error_quote_fallback": None if repair_ok else "zero_loss",
        "error_quote_invalid_reason": None if anchor_ok else str(location.get("reject_reason") or "unmatched"),
        "error_quote_locator_mode": "reference_fuzzy_v1",
        "error_route_mode": "legacy",
        "training_route": "local_suffix" if repair_ok else "zero_loss",
        "p1_schema_valid": bool(schema_ok) if int(round_number) == 1 else None,
        "raw_chars": len(raw),
        "idea_chars": len(effective_signal),
        "leaked": bool(key_leak or correction_leak),
        "fallback_reason": fallback_reason,
        "training_gate_passed": repair_ok,
        "teacher_privilege": "key_idea" if repair_ok else "solution",
    }
    if repair_ok:
        return effective_signal, meta
    return "", meta


