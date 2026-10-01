"""Multi-round repair loop (engine-agnostic).

Every student attempt becomes a distillation pair on its own on-policy
trajectory.  A failed attempt is localized to an error anchor, the guidance
generator produces repair guidance for it, and the student retries with that
guidance in context.  Signals that leak the answer, or whose anchor cannot be
located in the response, fail closed and contribute no supervised stage.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

GenerateFn = Callable[[Sequence[str]], list]          # prompts -> completion texts
VerifyFn = Callable[[str, str], bool]                  # (completion, gold) -> correct
MakeHintsFn = Callable[[Sequence[dict]], list]         # items -> hint strings or payload dicts
CompletionTruncatedFn = Callable[[str], bool]


@dataclass
class DistillPair:
    index: int                 # example index in the batch
    round: int                 # 1-based round that produced this on-policy attempt
    problem: str
    gold: str
    correct: bool              # was this attempt correct
    privilege: str             # "solution" | "hint" | "key_idea"
    student_prompt: str        # the on-policy generation context C_t (student side)
    teacher_prompt: str        # C_t + privilege (solution or hint)
    completion: str            # the student's on-policy attempt y_t (distill target)
    hint: str | None = None
    hint_meta: dict | None = None
    include_in_loss: bool = True
    supervision_weight: float = 1.0
    loss_mask_start_char: int | None = None
    loss_mask_end_char: int | None = None
    loss_mask_window_after_tokens: int | None = None
    loss_mask_window_max_tokens: int | None = None
    loss_mask_window_end_mode: str | None = None
    loss_mask_start_token: int | None = None
    loss_mask_end_token: int | None = None
    valid_label_tokens: int | None = None
    teacher_completion: str | None = None
    teacher_completion_start_char: int | None = None


def assign_simple_outcome_raw_weights(pairs: Sequence[DistillPair]) -> dict[str, Any]:
    """Assign normalized 2/1 success weights and a 0.5 failure budget.

    Distributed mass preservation is deliberately left to the trainer. This
    pure helper mutates only pair weights/metadata and returns local masses.
    """
    groups: dict[Any, list[DistillPair]] = {}
    for pair in pairs:
        groups.setdefault(pair.index, []).append(pair)

    baseline_mass = sum(
        float(pair.supervision_weight)
        for pair in pairs
        if pair.include_in_loss and float(pair.supervision_weight) > 0.0
    )
    raw_mass = 0.0
    outcome_counts = {"success": 0, "final_failure": 0, "censored": 0}

    for group in groups.values():
        ordered = sorted(group, key=lambda pair: int(pair.round))
        active = [
            pair
            for pair in ordered
            if pair.include_in_loss and float(pair.supervision_weight) > 0.0
        ]
        success_pair = next((pair for pair in ordered if pair.correct), None)
        censored = any(
            bool((pair.hint_meta or {}).get("attempt_truncated"))
            or bool((pair.hint_meta or {}).get("next_generation_skipped_reason"))
            for pair in ordered
        )

        if success_pair is not None:
            outcome = "success"
            budget = 1.0
            first_success_stage = max(0, int(success_pair.round) - 1)
            success_evidence = str(
                (success_pair.hint_meta or {}).get("success_evidence") or "verifier"
            )
            scores = [
                2.0 if int(pair.round) == first_success_stage else 1.0
                for pair in active
            ]
            outcome_basis = "first_observed_correct_response"
        elif censored:
            outcome = "censored"
            budget = 1.0
            first_success_stage = None
            success_evidence = None
            scores = [1.0 for _ in active]
            outcome_basis = "unobserved_terminal_response"
        else:
            outcome = "final_failure"
            budget = 0.5
            first_success_stage = None
            success_evidence = None
            scores = [1.0 for _ in active]
            outcome_basis = "last_observed_response_without_y3"
        outcome_counts[outcome] += 1

        score_total = sum(scores)
        raw_by_identity = {
            id(pair): (budget * score / score_total if score_total else 0.0)
            for pair, score in zip(active, scores)
        }
        for pair in ordered:
            raw_weight = float(raw_by_identity.get(id(pair), 0.0))
            pair.supervision_weight = raw_weight
            meta = dict(pair.hint_meta or {})
            meta.update(
                {
                    "round_weight_mode": "simple_outcome_2_1_half",
                    "trajectory_outcome": outcome,
                    "trajectory_weight_budget": budget,
                    "terminal_outcome_basis": outcome_basis,
                    "first_success_stage": first_success_stage,
                    "success_evidence": success_evidence,
                    "raw_round_weight": raw_weight,
                    "y3_generated": False,
                    "round3_hint_consumed_by_actor": False,
                }
            )
            pair.hint_meta = meta
            raw_mass += raw_weight

    return {
        "baseline_mass": float(baseline_mass),
        "raw_mass": float(raw_mass),
        "outcome_counts": outcome_counts,
    }


EXP_DECAY_ROUND_WEIGHT_MODE = "exp_decay_g08_terminal_half_postnorm"
EXP_DECAY_ROUND_WEIGHT_GAMMA = 0.8
EXP_DECAY_TERMINAL_WEIGHT_FACTOR = 0.5

# Paper trajectory-outcome weighting: w_t = alpha^(t-1) * rho^s, with a
# per-trajectory unit-budget normalization.  ``s`` is the correctness of the
# immediately following retry; it is zero when that retry is unavailable.
TRAJECTORY_OUTCOME_ROUND_WEIGHT_MODE = "trajectory_outcome_alpha08_rho15"


def _immediate_retry_outcome(pair, by_round) -> tuple[bool, bool]:
    """Return ``(retry_available, retry_correct)`` for one supervised stage.

    The retry that follows a stage is normally the next stage's scored response.
    The last supervised stage has no next stage, so it uses the optional terminal
    retry when one was sampled and otherwise keeps ``s = 0``.
    """
    retry = by_round.get(int(pair.round) + 1)
    if retry is not None:
        return True, bool(
            (retry.hint_meta or {}).get("verifier_attempt_correct", retry.correct)
        )
    meta = pair.hint_meta or {}
    if meta.get("terminal_retry_generated"):
        return True, bool(meta.get("terminal_retry_correct"))
    return False, False


def assign_trajectory_outcome_round_weights(
    pairs: Sequence[DistillPair],
    *,
    alpha: float = 0.8,
    rho: float = 1.5,
) -> dict[str, Any]:
    """Apply alpha/rho weighting within each trajectory. Inactive pairs stay zero; immediate
    verified retries receive the outcome bonus. Normalize active weights per trajectory."""
    alpha = float(alpha)
    rho = float(rho)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    if rho <= 0.0:
        raise ValueError("rho must be positive")
    groups: dict[Any, list[DistillPair]] = {}
    for pair in pairs:
        groups.setdefault(pair.index, []).append(pair)
    baseline_mass = sum(
        float(pair.supervision_weight)
        for pair in pairs
        if pair.include_in_loss and float(pair.supervision_weight) > 0.0
    )
    raw_mass = 0.0
    trajectory_count = 0
    outcome_counts = {"success": 0, "final_failure": 0, "censored": 0}
    for group in groups.values():
        ordered = sorted(group, key=lambda pair: int(pair.round))
        active = [
            pair for pair in ordered
            if pair.include_in_loss and float(pair.supervision_weight) > 0.0
        ]
        if not active:
            for pair in ordered:
                pair.supervision_weight = 0.0
                meta = dict(pair.hint_meta or {})
                meta.update({
                    "round_weight_mode": TRAJECTORY_OUTCOME_ROUND_WEIGHT_MODE,
                    "round_weight_alpha": alpha,
                    "round_weight_rho": rho,
                    "normalized_round_weight": 0.0,
                })
                pair.hint_meta = meta
            continue
        trajectory_count += 1
        success = any(bool(pair.correct) for pair in ordered)
        censored = any(
            bool((pair.hint_meta or {}).get("attempt_truncated"))
            or bool((pair.hint_meta or {}).get("next_generation_skipped_reason"))
            for pair in ordered
        )
        outcome = "success" if success else "censored" if censored else "final_failure"
        outcome_counts[outcome] += 1
        by_round = {int(pair.round): pair for pair in ordered}
        raw_by_id: dict[int, float] = {}
        for pair in active:
            retry_available, retry_correct = _immediate_retry_outcome(pair, by_round)
            stage = int(pair.round)
            raw = (alpha ** max(0, stage - 1)) * (rho if retry_correct else 1.0)
            raw_by_id[id(pair)] = raw
        normalizer = sum(raw_by_id.values())
        for pair in ordered:
            raw = float(raw_by_id.get(id(pair), 0.0))
            normalized = raw / normalizer if normalizer > 0.0 else 0.0
            pair.supervision_weight = normalized
            meta = dict(pair.hint_meta or {})
            retry_available, retry_correct = _immediate_retry_outcome(pair, by_round)
            meta.update({
                "round_weight_mode": TRAJECTORY_OUTCOME_ROUND_WEIGHT_MODE,
                "round_weight_alpha": alpha,
                "round_weight_rho": rho,
                "round_weight_stage_index": max(0, int(pair.round) - 1),
                "immediate_retry_available": retry_available,
                "immediate_retry_correct": retry_correct,
                "raw_round_weight": raw,
                "trajectory_weight_normalizer": normalizer,
                "normalized_round_weight": normalized,
                "trajectory_weight_sum": 1.0 if active else 0.0,
                "terminal_retry_unavailable_s_zero": not retry_available,
            })
            pair.hint_meta = meta
            raw_mass += raw
    return {
        "baseline_mass": float(baseline_mass),
        "raw_mass": float(raw_mass),
        "global_scale": 1.0,
        "trajectory_count": trajectory_count,
        "outcome_counts": outcome_counts,
        "alpha": alpha,
        "rho": rho,
    }


def assign_exponential_decay_round_weights(
    pairs: Sequence[DistillPair],
) -> dict[str, Any]:
    """Assign active-stage weights and return local mass for distributed rescaling."""

    groups: dict[Any, list[DistillPair]] = {}
    for pair in pairs:
        groups.setdefault(pair.index, []).append(pair)

    baseline_mass = sum(
        float(pair.supervision_weight)
        for pair in pairs
        if pair.include_in_loss and float(pair.supervision_weight) > 0.0
    )
    post_mass = 0.0
    outcome_counts = {"success": 0, "final_failure": 0, "censored": 0}
    terminal_failure_penalized = 0

    for group in groups.values():
        ordered = sorted(group, key=lambda pair: int(pair.round))
        active = [
            pair
            for pair in ordered
            if pair.include_in_loss and float(pair.supervision_weight) > 0.0
        ]
        success_pair = next((pair for pair in ordered if pair.correct), None)
        censored = any(
            bool((pair.hint_meta or {}).get("attempt_truncated"))
            or bool((pair.hint_meta or {}).get("next_generation_skipped_reason"))
            for pair in ordered
        )

        if success_pair is not None:
            outcome = "success"
            outcome_basis = "first_observed_correct_response"
            success_evidence = str(
                (success_pair.hint_meta or {}).get("success_evidence") or "verifier"
            )
        elif censored:
            outcome = "censored"
            outcome_basis = "unobserved_terminal_response"
            success_evidence = None
        else:
            outcome = "final_failure"
            success_evidence = None

        outcome_counts[outcome] += 1

        # A terminal penalty is reserved for a genuinely exhausted K=3
        # trajectory.  ``next_response_generated`` is present in runtime
        # records for rounds 1 and 2; absent synthetic/test metadata is
        # treated permissively so the helper remains easy to unit test.
        by_round = {int(pair.round): pair for pair in ordered}
        rounds_complete = all(round_number in by_round for round_number in (1, 2, 3))
        rounds_complete = rounds_complete and all(
            bool(str(by_round[round_number].completion).strip())
            for round_number in (1, 2, 3)
        )
        for round_number in (1, 2, 3):
            meta = by_round.get(round_number).hint_meta if round_number in by_round else {}
            meta = meta or {}
            if meta.get("attempt_truncated") is True or meta.get(
                "next_generation_skipped_reason"
            ):
                rounds_complete = False
        for round_number in (1, 2):
            if round_number in by_round:
                marker = (by_round[round_number].hint_meta or {}).get(
                    "next_response_generated"
                )
                if marker is False:
                    rounds_complete = False

        terminal_pair = by_round.get(3)
        terminal_failure_eligible = bool(
            outcome == "final_failure"
            and rounds_complete
            and terminal_pair is not None
            and terminal_pair.include_in_loss
            and float(terminal_pair.supervision_weight) > 0.0
        )
        if terminal_failure_eligible:
            terminal_failure_penalized += 1
            outcome_basis = "max_rounds_completed_final_response_wrong"
        elif outcome == "final_failure":
            outcome_basis = "last_observed_response_without_max_round_completion"

        stage_count = len(active)
        pre_normalized: dict[int, float] = {}
        normalized: dict[int, float] = {}
        if stage_count:
            latest_stage = stage_count - 1
            for stage_index, pair in enumerate(active):
                value = float(EXP_DECAY_ROUND_WEIGHT_GAMMA) ** (
                    latest_stage - stage_index
                )
                pre_normalized[id(pair)] = value
            normalizer = sum(pre_normalized.values())
            if normalizer <= 0.0:
                raise ValueError("exponential round-weight normalizer must be positive")
            normalized = {
                identity: value / normalizer
                for identity, value in pre_normalized.items()
            }

        trajectory_weight_sum = 0.0
        for stage_index, pair in enumerate(active):
            normalized_weight = float(normalized[id(pair)])
            penalized = bool(terminal_failure_eligible and pair is terminal_pair)
            effective_weight = (
                normalized_weight * EXP_DECAY_TERMINAL_WEIGHT_FACTOR
                if penalized
                else normalized_weight
            )
            pair.supervision_weight = effective_weight
            trajectory_weight_sum += effective_weight
            meta = dict(pair.hint_meta or {})
            meta.update(
                {
                    "round_weight_mode": EXP_DECAY_ROUND_WEIGHT_MODE,
                    "round_weight_gamma": EXP_DECAY_ROUND_WEIGHT_GAMMA,
                    "round_weight_terminal_factor": EXP_DECAY_TERMINAL_WEIGHT_FACTOR,
                    "trajectory_outcome": outcome,
                    "trajectory_weight_budget": 1.0,
                    "trajectory_weight_sum_post_penalty": None,
                    "terminal_outcome_basis": outcome_basis,
                    "success_evidence": success_evidence,
                    "active_round_stage_index": stage_index,
                    "active_round_stage_count": stage_count,
                    "raw_round_weight": pre_normalized[id(pair)],
                    "normalized_round_weight_before_terminal_factor": normalized_weight,
                    "post_normalization_round_weight": effective_weight,
                    "terminal_failure_eligible": terminal_failure_eligible,
                    "terminal_round_weight_penalized": penalized,
                    "y3_generated": False,
                    "round3_hint_consumed_by_actor": False,
                }
            )
            pair.hint_meta = meta

        active_ids = {id(pair) for pair in active}
        for pair in ordered:
            if id(pair) not in active_ids:
                pair.supervision_weight = 0.0
            meta = dict(pair.hint_meta or {})
            meta.update(
                {
                    "round_weight_mode": EXP_DECAY_ROUND_WEIGHT_MODE,
                    "round_weight_gamma": EXP_DECAY_ROUND_WEIGHT_GAMMA,
                    "round_weight_terminal_factor": EXP_DECAY_TERMINAL_WEIGHT_FACTOR,
                    "trajectory_outcome": outcome,
                    "trajectory_weight_budget": 1.0,
                    "trajectory_weight_sum_post_penalty": None,
                    "terminal_outcome_basis": outcome_basis,
                    "success_evidence": success_evidence,
                    "active_round_stage_index": meta.get("active_round_stage_index"),
                    "active_round_stage_count": stage_count,
                    "raw_round_weight": meta.get("raw_round_weight", 0.0),
                    "normalized_round_weight_before_terminal_factor": meta.get(
                        "normalized_round_weight_before_terminal_factor", 0.0
                    ),
                    "post_normalization_round_weight": float(pair.supervision_weight),
                    "terminal_failure_eligible": terminal_failure_eligible,
                    "terminal_round_weight_penalized": bool(
                        meta.get("terminal_round_weight_penalized", False)
                    ),
                    "y3_generated": False,
                    "round3_hint_consumed_by_actor": False,
                }
            )
            pair.hint_meta = meta

        for pair in ordered:
            pair.hint_meta["trajectory_weight_sum_post_penalty"] = trajectory_weight_sum
        post_mass += trajectory_weight_sum

    return {
        "baseline_mass": float(baseline_mass),
        "raw_mass": float(post_mass),
        "outcome_counts": outcome_counts,
        "terminal_failure_penalized": terminal_failure_penalized,
    }


NO_REPAIR_GATE_DECISION_SCHEMA = "socratic_no_repair_gate_decision_v1"


def _dedupe_strings(values: Sequence[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def build_no_repair_gate_decision(
    *,
    gate_mode: str,
    initial_correct: bool,
    initial_truncated: bool,
    reflected: bool,
    raw_hint: str,
    raw_meta: dict[str, Any] | None,
    teacher_privilege: str | None,
    key_idea_clean: bool,
    quote_span: tuple[int, int] | None,
    eligible: bool,
) -> dict[str, Any]:
    """Build a stable, mutually exclusive ledger for no-repair selection.

    This is diagnostic only: ``eligible`` is computed by the existing gate and
    this helper must never change it.  ``all_reasons`` intentionally preserves
    overlapping failures while ``terminal_bucket`` uses a fixed priority so
    batch-level funnel counts sum exactly to the number of trajectories.
    """
    if gate_mode not in {
        "clean_key_idea",
        "locatable_quote",
        "scope_routed",
        "scope_bounded",
    }:
        raise ValueError(f"unsupported no-repair gate mode: {gate_mode!r}")

    meta = dict(raw_meta or {})
    fallback_reason = str(meta.get("fallback_reason") or "").strip() or None
    raw_chars = int(meta.get("raw_chars") or 0)
    leaked = meta.get("leaked") is True
    effective_privilege = str(
        teacher_privilege or meta.get("teacher_privilege") or "key_idea"
    )
    candidate_route = str(meta.get("training_route") or "").strip()
    scope = str(meta.get("error_scope") or "").strip().lower()
    scope_required = gate_mode in {"scope_routed", "scope_bounded", "clean_key_idea"}
    scope_valid = scope in {"local", "global"}
    p1_contract_active = bool(
        meta.get("p1_prompt_id") in {"QUOTE-GUIDANCE", "STRUCTURED-GUIDANCE", "SEGMENT-GUIDANCE", "LOCALIZED-GUIDANCE", "QUOTE-LOCALIZED-GUIDANCE"}
        or str(meta.get("p1_schema") or "").startswith(
            (
                "air_opd_quote_guidance_",
                "air_opd_structured_guidance_",
                "air_opd_segment_guidance_",
                "air_opd_localized_guidance_",
                "air_opd_quote_localized_guidance_",
            )
        )
    )
    p1_decision = str(meta.get("decision") or "").strip().lower() or None
    p1_schema_valid = (
        meta.get("p1_schema_valid") is True if p1_contract_active else None
    )
    quote_required = bool(
        gate_mode in {"locatable_quote", "scope_bounded"}
        or (
            gate_mode in {"scope_routed", "clean_key_idea"}
            and scope == "local"
        )
    ) and not bool(p1_contract_active and p1_decision != "repair")
    error_quote = str(
        meta.get("error_quote") or meta.get("raw_error_quote") or ""
    ).strip()
    location = (
        dict(meta.get("error_quote_location"))
        if isinstance(meta.get("error_quote_location"), dict)
        else {}
    )
    locator_claimed_locatable = bool(location.get("locatable"))
    # The actual training gate consumes the validated character bounds, not
    # the locator's preliminary boolean claim.  Keep both so a malformed span
    # cannot be reported as a passing quote gate.
    quote_locatable = bool(quote_span is not None)

    if not reflected:
        explanation_status = "not_evaluated"
    elif meta.get("extracted_explanation") is True:
        explanation_status = "closed_tag"
    elif meta.get("explanation_open_no_close") is True:
        explanation_status = "open_tag_salvaged"
    elif (
        fallback_reason == "empty"
        or not str(raw_hint or "").strip()
        or (
            p1_contract_active
            and int(meta.get("idea_chars") or 0) <= 0
        )
    ):
        explanation_status = "missing"
    else:
        explanation_status = "unstructured_or_unmarked"

    if not reflected or not scope_required:
        scope_status = "not_evaluated" if not reflected else "not_required"
    elif scope_valid:
        scope_status = scope
    else:
        scope_status = "missing" if not scope else "invalid"

    if not reflected:
        quote_status = "not_evaluated"
    elif not quote_required:
        quote_status = "not_required"
    elif quote_locatable:
        quote_status = "locatable"
    elif not error_quote:
        quote_status = "missing"
    else:
        quote_status = "unlocatable"

    reasons: list[str] = list(meta.get("route_reasons") or [])
    routing_reason = meta.get("routing_fallback_reason")
    if routing_reason:
        reasons.append(str(routing_reason))
    if fallback_reason:
        reasons.append(f"key_idea_fallback:{fallback_reason}")
    if p1_contract_active and not p1_schema_valid:
        reasons.append("invalid_p1_schema")
    if p1_contract_active and p1_decision == "no_repair":
        reasons.append("decision_no_repair")
    if leaked:
        reasons.append("answer_leak")
    if reflected and effective_privilege != "key_idea":
        reasons.append(f"teacher_privilege:{effective_privilege}")
    if reflected and scope_required and not scope_valid:
        reasons.append("invalid_error_scope")
    if reflected and quote_required and not error_quote and not quote_locatable:
        reasons.append("missing_error_quote")
    if reflected and quote_required and error_quote and not quote_locatable:
        reasons.append("error_quote_unlocatable")
    if reflected and gate_mode == "clean_key_idea" and not key_idea_clean:
        reasons.append("clean_key_idea_rejected")
    valid_routes = {
        "clean_key_idea": {"local_suffix", "global_full"},
        "locatable_quote": {"legacy_quote_suffix", ""},
        "scope_routed": {"local_suffix", "global_full"},
        "scope_bounded": {"local_suffix", "global_prefix"},
    }[gate_mode]
    if (
        reflected
        and gate_mode != "locatable_quote"
        and candidate_route not in valid_routes
    ):
        reasons.append("invalid_training_route")

    if initial_correct:
        terminal_bucket = "initial_correct"
        primary_reason = "initial_correct"
    elif initial_truncated:
        terminal_bucket = "initial_truncated"
        primary_reason = "initial_incomplete_or_truncated"
    elif not reflected:
        terminal_bucket = "reflection_missing"
        primary_reason = "reflection_not_requested_or_missing"
    elif p1_contract_active and not p1_schema_valid:
        terminal_bucket = "invalid_p1_schema"
        primary_reason = str(
            meta.get("p1_schema_reject_reason") or "invalid_p1_schema"
        )
    elif p1_contract_active and p1_decision == "no_repair":
        terminal_bucket = "decision_no_repair"
        primary_reason = "decision_no_repair"
    elif fallback_reason == "empty":
        if raw_chars == 0:
            terminal_bucket = "observed_empty_message_content"
            primary_reason = "observed_empty_message_content"
        else:
            terminal_bucket = "missing_explanation"
            primary_reason = "missing_explanation"
    elif not str(raw_hint or "").strip():
        terminal_bucket = "key_idea_empty"
        primary_reason = "key_idea_empty"
    elif leaked:
        terminal_bucket = "answer_leak"
        primary_reason = "answer_leak"
    elif fallback_reason:
        terminal_bucket = "key_idea_fallback"
        primary_reason = f"key_idea_fallback:{fallback_reason}"
    elif scope_required and not scope_valid:
        terminal_bucket = "invalid_scope"
        primary_reason = "invalid_error_scope"
    elif quote_required and not error_quote and not quote_locatable:
        terminal_bucket = "missing_quote"
        primary_reason = "missing_error_quote"
    elif quote_required and not quote_locatable:
        terminal_bucket = "quote_unlocatable"
        primary_reason = "error_quote_unlocatable"
    elif gate_mode == "clean_key_idea" and not key_idea_clean:
        terminal_bucket = "clean_key_idea_rejected"
        primary_reason = "clean_key_idea_rejected"
    elif effective_privilege != "key_idea":
        terminal_bucket = "non_key_idea_privilege"
        primary_reason = f"teacher_privilege:{effective_privilege}"
    elif not eligible:
        terminal_bucket = "invalid_route_or_other_fail_closed"
        primary_reason = (
            str(routing_reason)
            if routing_reason
            else "invalid_route_or_other_fail_closed"
        )
    else:
        selected_route = (
            "legacy_quote_suffix"
            if gate_mode == "locatable_quote"
            else candidate_route
        )
        terminal_bucket = f"selected:{selected_route or 'unknown'}"
        primary_reason = terminal_bucket

    all_reasons = _dedupe_strings([primary_reason, *reasons])
    active_after_initial = not initial_correct and not initial_truncated
    route_valid = bool(eligible)
    stages = {
        "initial_wrong": "fail" if initial_correct else "pass",
        "initial_complete": (
            "skipped"
            if initial_correct
            else "fail"
            if initial_truncated
            else "pass"
        ),
        "reflection_available": (
            "skipped"
            if not active_after_initial
            else "pass"
            if reflected
            else "fail"
        ),
        "key_idea_nonempty": (
            "skipped"
            if not reflected
            else "not_required"
            if p1_contract_active and p1_decision == "no_repair"
            else "fail"
            if (
                fallback_reason == "empty"
                or not str(raw_hint or "").strip()
                or (
                    p1_contract_active
                    and int(meta.get("idea_chars") or 0) <= 0
                )
            )
            else "pass"
        ),
        "answer_leak_free": (
            "skipped" if not reflected else "fail" if leaked else "pass"
        ),
        "fallback_free": (
            "skipped"
            if not reflected
            else "fail"
            if fallback_reason
            else "pass"
        ),
        "scope_valid": (
            "skipped"
            if not reflected
            else "not_required"
            if not scope_required
            else "pass"
            if scope_valid
            else "fail"
        ),
        "quote_locatable": (
            "skipped"
            if not reflected
            else "not_required"
            if not quote_required
            else "pass"
            if quote_locatable
            else "fail"
        ),
        "clean_key_idea": (
            "skipped"
            if not reflected
            else "not_required"
            if gate_mode != "clean_key_idea"
            else "pass"
            if key_idea_clean
            else "fail"
        ),
        "route_eligible": (
            "skipped"
            if not active_after_initial or not reflected
            else "pass"
            if route_valid
            else "fail"
        ),
        "p1_schema_valid": (
            "not_required"
            if not p1_contract_active
            else "pass"
            if p1_schema_valid
            else "fail"
        ),
    }
    provider_response = (
        dict(meta.get("provider_response_diagnostics"))
        if isinstance(meta.get("provider_response_diagnostics"), dict)
        else None
    )
    return {
        "schema": NO_REPAIR_GATE_DECISION_SCHEMA,
        "gate_mode": gate_mode,
        "eligible_before_policy": bool(eligible),
        "selected_after_policy": None,
        "terminal_bucket": terminal_bucket,
        "primary_reason": primary_reason,
        "all_reasons": all_reasons,
        "initial_correct": bool(initial_correct),
        "initial_truncated": bool(initial_truncated),
        "reflection_requested": bool(active_after_initial),
        "reflection_observed": bool(reflected),
        "reflection_raw_chars": raw_chars,
        # ``idea_chars`` is the parser's extracted explanation length.  It is
        # intentionally distinct from the returned safe fallback string,
        # which can be non-empty even when the provider content was empty.
        "key_idea_chars": (
            int(meta["idea_chars"])
            if (
                isinstance(meta.get("idea_chars"), int)
                and not isinstance(meta.get("idea_chars"), bool)
                and int(meta["idea_chars"]) >= 0
            )
            else len(str(raw_hint or "").strip())
        ),
        "key_idea_clean": bool(key_idea_clean),
        "explanation_status": explanation_status,
        "scope_status": scope_status,
        "quote_status": quote_status,
        "locator_claimed_locatable": locator_claimed_locatable,
        "validated_quote_span": (
            [int(quote_span[0]), int(quote_span[1])]
            # A global ``scope_routed`` response may include an extra,
            # perfectly locatable quote even though the selected
            # ``global_full`` route does not consume it.  Record only spans
            # that the active gate actually validated/used; otherwise the
            # ledger would contradict its own ``quote_status=not_required``.
            if quote_status == "locatable" and quote_span is not None
            else None
        ),
        "candidate_route": candidate_route or None,
        "effective_teacher_privilege": effective_privilege,
        "fallback_reason": fallback_reason,
        "leaked": bool(leaked),
        "stages": stages,
        "provider_response_diagnostics": provider_response,
        "p1_contract_active": p1_contract_active,
        "p1_schema_valid": p1_schema_valid,
        "p1_schema_reject_reason": meta.get("p1_schema_reject_reason"),
        "p1_decision": p1_decision,
    }


def apply_supervision_policy(
    pairs: Sequence[DistillPair],
    *,
    supervision_scope: str = "all",
    solution_privilege_mode: str = "allow",
) -> list[DistillPair]:
    """Set auditable loss eligibility while preserving pair identities."""
    if supervision_scope not in {"all", "initial_wrong_only"}:
        raise ValueError("supervision_scope must be 'all' or 'initial_wrong_only'")
    if solution_privilege_mode not in {"allow", "exclude"}:
        raise ValueError("solution_privilege_mode must be 'allow' or 'exclude'")

    out = list(pairs)
    for pair in out:
        selected = bool(pair.include_in_loss)
        reasons: list[str] = []
        if supervision_scope == "initial_wrong_only" and pair.correct:
            selected = False
            reasons.append("initial_correct_excluded")
        if solution_privilege_mode == "exclude" and pair.privilege == "solution":
            selected = False
            reasons.append("solution_privilege_excluded")

        meta = dict(pair.hint_meta or {})
        meta["supervision_policy"] = {
            "scope": supervision_scope,
            "solution_privilege_mode": solution_privilege_mode,
            "selected": bool(selected),
            "reasons": reasons,
            "reference_solution_used_for_loss": bool(
                selected and pair.privilege == "solution"
            ),
        }
        decision = meta.get("gate_decision")
        if isinstance(decision, dict):
            decision = dict(decision)
            decision["selected_after_policy"] = bool(selected)
            decision["supervision_policy_reasons"] = list(reasons)
            meta["gate_decision"] = decision
        pair.hint_meta = meta
        pair.include_in_loss = bool(selected)
        pair.supervision_weight = 1.0 if selected else 0.0
        if not selected:
            pair.teacher_prompt = pair.student_prompt
    return out


def bounded_loss_token_span(
    start_token: int,
    quote_end_token: int,
    completion_tokens: int,
    *,
    window_after_tokens: int | None = None,
    window_max_tokens: int | None = None,
    window_end_mode: str = "fixed",
    completion: str | None = None,
    offsets: Sequence[Sequence[int]] | None = None,
) -> tuple[int, int]:
    """Return a non-empty bounded token span while preserving the full quote."""
    total = max(0, int(completion_tokens))
    start = min(max(0, int(start_token)), total)
    quote_end = min(max(start, int(quote_end_token)), total)
    end = quote_end
    if window_after_tokens is not None:
        target_end = min(total, quote_end + max(0, int(window_after_tokens)))
        cap_end = total
        if window_max_tokens is not None:
            cap_end = min(total, start + max(1, int(window_max_tokens)))
            cap_end = max(quote_end, cap_end)
        end = min(target_end, cap_end)
        if (
            window_end_mode == "next_blank"
            and target_end < total
            and completion is not None
            and offsets is not None
            and target_end > 0
        ):
            target_char = int(offsets[target_end - 1][1])
            separator = re.search(r"\n[ \t]*\n", completion[target_char:])
            if separator is not None:
                boundary_char = target_char + separator.start()
                boundary_end = target_end
                for index, pair in enumerate(offsets[:total]):
                    token_start, token_end = int(pair[0]), int(pair[1])
                    if token_end <= token_start:
                        continue
                    if token_start < boundary_char:
                        boundary_end = index + 1
                end = min(total, max(target_end, boundary_end), cap_end)
            else:
                end = cap_end
        elif window_end_mode != "fixed":
            if window_end_mode != "next_blank":
                raise ValueError(
                    "window_end_mode must be 'fixed' or 'next_blank'"
                )
    return start, min(total, end)


def char_span_to_token_span(
    offsets: Sequence[Sequence[int]],
    start_char: int,
    end_char: int,
) -> tuple[int, int] | None:
    """Map a half-open character span to all overlapping non-empty tokens."""
    start = max(0, int(start_char))
    end = max(start, int(end_char))
    if end <= start:
        return None
    first = None
    last = None
    for idx, pair in enumerate(offsets):
        if len(pair) < 2:
            continue
        token_start, token_end = int(pair[0]), int(pair[1])
        if token_end <= token_start:
            continue
        if token_end <= start or token_start >= end:
            continue
        if first is None:
            first = idx
        last = idx + 1
    if first is None or last is None:
        return None
    return first, last


def _set_filter_meta(
    pair: DistillPair,
    *,
    mode: str,
    selected: bool,
    reason: str,
    quality_reasons: list[str] | None = None,
) -> None:
    meta = dict(pair.hint_meta or {})
    meta["wrong_pair_filter"] = {
        "mode": mode,
        "selected": bool(selected),
        "reason": reason,
        "quality_reasons": list(quality_reasons or []),
    }
    pair.hint_meta = meta
    pair.include_in_loss = bool(selected)


def apply_success_gated_wrong2_filter(
    pairs: Sequence[DistillPair],
    *,
    quality_reasons: Callable[[DistillPair], list[str]] | None = None,
    mode: str = "success_gated_wrong2",
) -> list[DistillPair]:
    """Keep only trajectories validated by a later successful repair.

    Semantics:
    - first-correct examples keep the standard solution-privileged pair;
    - if repair succeeds at round 2 or 3, keep the preceding wrong pairs
      (at most two), using their existing teacher privilege;
    - terminal failures contribute no loss.

    This is intentionally separate from the repair loop: the loop still records
    every attempted trajectory, while this filter decides which trajectories are
    trusted enough for OPSD loss.
    """
    out = list(pairs)
    by_index: dict[int, list[DistillPair]] = {}
    for pair in out:
        by_index.setdefault(pair.index, []).append(pair)

    for group in by_index.values():
        group.sort(key=lambda p: p.round)
        for pair in group:
            pair.include_in_loss = False

        success = next((p for p in group if p.correct and p.privilege == "solution"), None)
        if success is None:
            for pair in group:
                _set_filter_meta(
                    pair,
                    mode=mode,
                    selected=False,
                    reason="terminal_fail_no_repair_success",
                )
            continue

        if success.round <= 1:
            for pair in group:
                _set_filter_meta(
                    pair,
                    mode=mode,
                    selected=pair is success,
                    reason="first_correct_solution" if pair is success else "not_before_repair_success",
                )
            continue

        kept = 0
        for pair in group:
            if pair is success:
                _set_filter_meta(
                    pair,
                    mode=mode,
                    selected=False,
                    reason="repair_success_validation_only",
                )
                continue
            if pair.correct or pair.round >= success.round or kept >= 2:
                _set_filter_meta(pair, mode=mode, selected=False, reason="not_pre_success_wrong_pair")
                continue
            if pair.privilege == "solution":
                _set_filter_meta(pair, mode=mode, selected=True, reason="validated_solution_privilege_wrong_pair")
                kept += 1
                continue
            if pair.privilege in {"key_idea", "hint"}:
                _set_filter_meta(pair, mode=mode, selected=True, reason="validated_wrong_pair_before_repair")
                kept += 1
                continue
            _set_filter_meta(pair, mode=mode, selected=False, reason="unsupported_pre_success_privilege")

    return out


def _decode_hint_payload(payload: Any) -> tuple[str, dict | None, str | None]:
    if isinstance(payload, dict):
        hint = str(payload.get("hint") or payload.get("text") or "")
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        teacher_privilege = payload.get("teacher_privilege") or meta.get("teacher_privilege")
        if teacher_privilege is not None:
            teacher_privilege = str(teacher_privilege)
        return hint, dict(meta), teacher_privilege
    return str(payload), None, None


def _clean_repair_key_idea(
    hint: str,
    meta: dict | None,
    teacher_privilege: str | None,
) -> bool:
    if not isinstance(meta, dict):
        return False
    metadata = meta
    required_sanitizer_fields = {
        "extracted_explanation",
        "explanation_open_no_close",
        "leaked",
        "fallback_reason",
        "raw_chars",
        "idea_chars",
    }
    if not required_sanitizer_fields.issubset(metadata):
        return False
    effective_privilege = teacher_privilege or metadata.get("teacher_privilege")
    two_scope = metadata.get("error_route_mode") == "two_scope_v1"
    routed_clean = (
        metadata.get("training_route") in {"local_suffix", "global_full"}
        if two_scope
        else True
    )
    explanation_present = bool(
        metadata.get("extracted_explanation") is True
        or (two_scope and metadata.get("explanation_open_no_close") is True)
    )
    return bool(
        str(hint or "").strip()
        and effective_privilege in {None, "key_idea"}
        and explanation_present
        and (two_scope or metadata.get("explanation_open_no_close") is False)
        and metadata.get("leaked") is False
        and metadata.get("fallback_reason") is None
        and routed_clean
        and isinstance(metadata.get("raw_chars"), int)
        and not isinstance(metadata.get("raw_chars"), bool)
        and metadata["raw_chars"] > 0
        and isinstance(metadata.get("idea_chars"), int)
        and not isinstance(metadata.get("idea_chars"), bool)
        and metadata["idea_chars"] > 0
    )


def _combine_repair_key_ideas(key_ideas: Sequence[str]) -> str:
    """Render an unambiguous, length-prefixed ordered privilege payload."""
    ideas = [str(key_idea).strip() for key_idea in key_ideas]
    if not ideas or any(not idea for idea in ideas):
        raise ValueError("combined repair key ideas must be non-empty")
    parts = ["AIR-OPD-COMBINED-GUIDANCE-V1", f"count={len(ideas)}"]
    for index, idea in enumerate(ideas, start=1):
        parts.extend((f"idea={index} chars={len(idea)}", idea))
    return "\n".join(parts)


class SocraticRepairLoop:
    def __init__(
        self,
        renderers,
        generate: GenerateFn,
        verify: VerifyFn,
        make_hints: MakeHintsFn,
        max_rounds: int = 3,
        distill_mode: str = "guided",
        vanilla_baseline: bool = False,
        wrong_privilege: str = "hint",
        wrong_pair_student_context: str = "on_policy",
        key_idea_privilege_context: str = "problem_only",
        loss_mask_mode: str = "full",
        local_window_after_tokens: int = 128,
        local_window_max_tokens: int = 256,
        local_window_end_mode: str = "fixed",
        repair_prompt_token_limit: int | None = None,
        is_completion_truncated: CompletionTruncatedFn | None = None,
        terminal_retry: bool = False,
    ):
        if distill_mode != "guided":
            raise ValueError(
                "only the on-policy 'guided' distill_mode is supported "
                "(the distilled trajectory must be the student's on-policy attempt)"
            )
        if int(max_rounds) < 1:
            raise ValueError("max_rounds must be >= 1")
        self.r = renderers
        self.generate = generate
        self.verify = verify
        self.make_hints = make_hints
        self.max_rounds = int(max_rounds)
        self.distill_mode = distill_mode
        self.vanilla_baseline = bool(vanilla_baseline)
        if wrong_privilege not in {"hint", "key_idea"}:
            raise ValueError("wrong_privilege must be 'hint' or 'key_idea'")
        self.wrong_privilege = wrong_privilege
        if wrong_pair_student_context not in {"on_policy", "root_question_after_first"}:
            raise ValueError("wrong_pair_student_context must be 'on_policy' or 'root_question_after_first'")
        self.wrong_pair_student_context = wrong_pair_student_context
        if key_idea_privilege_context not in {
            "problem_only",
            "problem_wrong_attempt",
            "tip_before_error_quote",
        }:
            raise ValueError(
                "key_idea_privilege_context must be 'problem_only', "
                "'problem_wrong_attempt', or 'tip_before_error_quote'"
            )
        self.key_idea_privilege_context = key_idea_privilege_context
        if loss_mask_mode not in {
            "full",
            "after_error_quote",
            "error_quote_only",
            "error_type_routed",
            "two_scope_routed",
            "two_scope_bounded",
        }:
            raise ValueError(
                "loss_mask_mode must be 'full', 'after_error_quote', "
                "'error_quote_only', 'error_type_routed', 'two_scope_routed', "
                "or 'two_scope_bounded'"
            )
        self.loss_mask_mode = loss_mask_mode
        self.local_window_after_tokens = int(local_window_after_tokens)
        self.local_window_max_tokens = int(local_window_max_tokens)
        self.local_window_end_mode = str(local_window_end_mode or "fixed")
        self.repair_prompt_token_limit = (
            None
            if repair_prompt_token_limit is None
            else int(repair_prompt_token_limit)
        )
        self.is_completion_truncated = is_completion_truncated or (lambda _text: False)
        # By default the terminal stage has s=0. Optional terminal retry defines the
        # outcome bonus and is excluded from distillation targets.
        self.terminal_retry = bool(terminal_retry)
        if self.local_window_after_tokens < 0:
            raise ValueError("local_window_after_tokens must be >= 0")
        if self.local_window_max_tokens < 1:
            raise ValueError("local_window_max_tokens must be >= 1")
        if self.local_window_end_mode not in {"fixed", "next_blank"}:
            raise ValueError("local_window_end_mode must be 'fixed' or 'next_blank'")
        if (
            self.repair_prompt_token_limit is not None
            and self.repair_prompt_token_limit < 1
        ):
            raise ValueError("repair_prompt_token_limit must be >= 1")

    def _render_teacher_key_idea(
        self,
        problem: str,
        key_idea: str,
        current_attempt: str | None,
        prompt_meta: dict | None,
    ) -> str:
        """Pass structured prompt metadata to capable renderers; retain the three-argument
        compatibility interface."""
        if getattr(self.r, "supports_teacher_prompt_profiles", False):
            return self.r.render_teacher_key_idea(
                problem,
                key_idea,
                current_attempt,
                prompt_meta=prompt_meta,
            )
        return self.r.render_teacher_key_idea(problem, key_idea, current_attempt)

    @staticmethod
    def _error_quote_start(completion: str, meta: dict | None) -> int | None:
        span = SocraticRepairLoop._error_quote_span(completion, meta)
        return span[0] if span is not None else None

    @staticmethod
    def _error_quote_span(completion: str, meta: dict | None) -> tuple[int, int] | None:
        if not meta:
            return None
        loc = meta.get("error_quote_location") if isinstance(meta, dict) else None
        if isinstance(loc, dict) and loc.get("locatable"):
            start = loc.get("start_char")
            end = loc.get("end_char")
            if isinstance(start, int) and isinstance(end, int):
                text_len = len(str(completion or ""))
                if 0 <= start < end <= text_len:
                    return start, end
        if (
            isinstance(meta, dict)
            and (
                meta.get("p1_prompt_id") in {"QUOTE-GUIDANCE", "STRUCTURED-GUIDANCE", "SEGMENT-GUIDANCE", "LOCALIZED-GUIDANCE", "QUOTE-LOCALIZED-GUIDANCE"}
                or str(meta.get("p1_schema") or "").startswith(
                    (
                        "air_opd_quote_guidance_",
                        "air_opd_structured_guidance_",
                        "air_opd_segment_guidance_",
                        "air_opd_localized_guidance_",
                        "air_opd_quote_localized_guidance_",
                    )
                )
            )
        ):
            # Strict P1 profiles are valid only through their unique, usable
            # locator receipt.
            # Do not restore an ambiguous or rejected anchor through a fallback.
            return None
        quote = str(meta.get("error_quote") or "").strip()
        if not quote:
            return None
        text = str(completion or "")
        pos = text.find(quote)
        if pos >= 0:
            return pos, pos + len(quote)
        norm_text, offsets = SocraticRepairLoop._normalize_ws_with_offsets(text)
        norm_quote = " ".join(quote.split())
        if not norm_quote:
            return None
        pos = norm_text.find(norm_quote)
        if pos < 0 or pos >= len(offsets):
            return None
        end_pos = pos + len(norm_quote) - 1
        if end_pos < pos or end_pos >= len(offsets):
            return None
        return offsets[pos], offsets[end_pos] + 1

    @staticmethod
    def _teacher_tip_prompt(base_prompt: str, completion_prefix: str, key_idea: str) -> str:
        idea = str(key_idea or "").strip()
        if idea.lower().startswith("key idea:"):
            idea = idea.split(":", 1)[1].strip()
        if not idea:
            idea = "re-check the first unsupported step before continuing."
        return (
            f"{str(base_prompt or '')}{str(completion_prefix or '')}"
            f"\n\nTip: {idea}\n\n"
        )

    @staticmethod
    def _normalize_ws_with_offsets(text: str) -> tuple[str, list[int]]:
        chars: list[str] = []
        offsets: list[int] = []
        in_ws = False
        for idx, ch in enumerate(str(text or "")):
            if ch.isspace():
                if chars and not in_ws:
                    chars.append(" ")
                    offsets.append(idx)
                in_ws = True
                continue
            chars.append(ch)
            offsets.append(idx)
            in_ws = False
        if chars and chars[-1] == " ":
            chars.pop()
            offsets.pop()
        return "".join(chars), offsets

    def run(
        self,
        problems: Sequence[str],
        golds: Sequence[str],
        teacher_contexts: Sequence[str | None] | None = None,
        sample_ids: Sequence[str | None] | None = None,
    ) -> list[DistillPair]:
        n = len(problems)
        if len(golds) != n:
            raise ValueError("problems and golds must have equal length")
        if teacher_contexts is None:
            teacher_contexts = [None] * n
        if len(teacher_contexts) != n:
            raise ValueError("teacher_contexts must have the same length as problems")
        if sample_ids is None:
            sample_ids = [None] * n
        if len(sample_ids) != n:
            raise ValueError("sample_ids must have the same length as problems")

        vanilla = self.vanilla_baseline
        attempts: list[list[str]] = [[] for _ in range(n)]   # prior wrong attempts in C_t
        hints: list[list[str]] = [[] for _ in range(n)]      # prior hints in C_t
        pairs: list[DistillPair] = []
        active = list(range(n))
        rnd = 1

        while active and rnd <= self.max_rounds:
            # 1) student attempts on-policy under its current context C_t
            ctx_prompts = [self.r.render_student(problems[i], attempts[i], hints[i]) for i in active]
            comps = self.generate(ctx_prompts)

            round_state: dict[int, tuple] = {}
            wrong: list[int] = []
            for k, i in enumerate(active):
                c = comps[k]
                ok = bool(self.verify(c, golds[i]))
                round_state[i] = (c, ctx_prompts[k], ok)
                if not ok:
                    wrong.append(i)

            # 2) hints for wrong attempts (privilege for the wrong pair AND next context).
            #    Skipped entirely in the vanilla (K=1) baseline.
            hint_map: dict[int, tuple[str, dict | None, str | None]] = {}
            if wrong and not vanilla:
                items = [
                    {
                        "index": i,
                        "round": rnd,
                        "sample_id": sample_ids[i],
                        "problem": problems[i],
                        "gold": golds[i],
                        "teacher_context": teacher_contexts[i],
                        "last_attempt": round_state[i][0],
                        "wrong_attempts": list(attempts[i]) + [round_state[i][0]],
                        "hints": list(hints[i]),
                    }
                    for i in wrong
                ]
                new_hints = self.make_hints(items)
                if len(new_hints) != len(wrong):
                    raise ValueError("make_hints must return one hint per wrong item")
                hint_map = {idx: _decode_hint_payload(payload) for idx, payload in zip(wrong, new_hints)}

            # 3) emit one distillation pair per attempt; update state for wrong ones
            next_active: list[int] = []
            for i in active:
                c, ctx_prompt, ok = round_state[i]
                if ok:
                    teacher = self.r.render_teacher_solution(
                        problems[i], attempts[i], hints[i], golds[i], teacher_contexts[i]
                    )
                    pairs.append(DistillPair(
                        index=i, round=rnd, problem=problems[i], gold=golds[i], correct=True,
                        privilege="solution", student_prompt=ctx_prompt, teacher_prompt=teacher,
                        completion=c, hint=None,
                    ))
                elif vanilla:
                    # vanilla OPSD baseline: wrong first attempt vs solution-teacher, no hint.
                    teacher = self.r.render_teacher_solution(
                        problems[i], attempts[i], hints[i], golds[i], teacher_contexts[i]
                    )
                    pairs.append(DistillPair(
                        index=i, round=rnd, problem=problems[i], gold=golds[i], correct=False,
                        privilege="solution", student_prompt=ctx_prompt, teacher_prompt=teacher,
                        completion=c, hint=None,
                    ))
                else:
                    h, hmeta, teacher_privilege = hint_map[i]
                    if hmeta is not None:
                        hmeta = dict(hmeta)
                    if teacher_privilege is not None and teacher_privilege not in {"hint", "key_idea", "solution"}:
                        raise ValueError(f"unsupported teacher_privilege override: {teacher_privilege!r}")
                    routed_training_route = None
                    routed_error_span = None
                    if self.loss_mask_mode == "error_type_routed":
                        hmeta = hmeta or {}
                        routed_training_route = str(hmeta.get("training_route") or "")
                        if routed_training_route in {"local_window", "final_answer"}:
                            routed_error_span = self._error_quote_span(c, hmeta)
                            if routed_error_span is None:
                                teacher_privilege = "solution"
                                hmeta["training_route"] = "solution"
                                hmeta["route_fail_safe"] = "solution_no_error_quote_span"
                        else:
                            teacher_privilege = "solution"
                            hmeta["route_fail_safe"] = "solution_nonlocal_or_invalid_route"
                    effective_privilege = teacher_privilege or self.wrong_privilege
                    if effective_privilege == "solution":
                        teacher = self.r.render_teacher_solution(
                            problems[i], attempts[i], hints[i], golds[i], teacher_contexts[i]
                        )
                        privilege = "solution"
                    elif effective_privilege == "key_idea":
                        # Render problem and guidance; optionally include the incorrect
                        # attempt as teacher context.
                        current_attempt = (
                            c if self.key_idea_privilege_context == "problem_wrong_attempt" else None
                        )
                        teacher = (
                            None
                            if self.key_idea_privilege_context == "tip_before_error_quote"
                            else self._render_teacher_key_idea(
                                problems[i], h, current_attempt, hmeta
                            )
                        )
                        privilege = "key_idea"
                        hmeta = hmeta or {}
                        hmeta["key_idea_privilege_context"] = self.key_idea_privilege_context
                    else:
                        # teacher sees the incorrect trajectory `c` + this round's hint
                        teacher = self.r.render_teacher_hint(problems[i], attempts[i], hints[i], h, c)
                        privilege = "hint"
                    pair_student_prompt = ctx_prompt
                    if (
                        self.wrong_pair_student_context == "root_question_after_first"
                        and rnd >= 2
                        and privilege == "key_idea"
                    ):
                        pair_student_prompt = self.r.render_student(problems[i], [], [])
                        hmeta = hmeta or {}
                        hmeta["student_context_override"] = "root_question_after_first"
                    loss_mask_start_char = None
                    loss_mask_end_char = None
                    loss_mask_window_after_tokens = None
                    loss_mask_window_max_tokens = None
                    loss_mask_window_end_mode = None
                    teacher_completion = None
                    teacher_completion_start_char = None
                    error_quote_span = None
                    if self.loss_mask_mode in {
                        "after_error_quote",
                        "error_quote_only",
                        "error_type_routed",
                    } and privilege == "key_idea":
                        error_quote_span = routed_error_span or self._error_quote_span(c, hmeta)
                        hmeta = hmeta or {}
                        hmeta["loss_mask_mode"] = self.loss_mask_mode
                        hmeta["loss_mask_start_char"] = error_quote_span[0] if error_quote_span else None
                        hmeta["loss_mask_end_char"] = (
                            error_quote_span[1]
                            if error_quote_span and self.loss_mask_mode in {"error_quote_only", "error_type_routed"}
                            else None
                        )
                        if error_quote_span is None:
                            hmeta["loss_mask_fallback"] = "full_completion_no_error_quote_match"
                        else:
                            loss_mask_start_char = error_quote_span[0]
                            if self.loss_mask_mode in {"error_quote_only", "error_type_routed"}:
                                loss_mask_end_char = error_quote_span[1]
                            if self.loss_mask_mode == "error_type_routed":
                                loss_mask_window_after_tokens = (
                                    self.local_window_after_tokens
                                    if routed_training_route == "local_window"
                                    else 0
                                )
                                loss_mask_window_max_tokens = (
                                    self.local_window_max_tokens
                                    if routed_training_route == "local_window"
                                    else 1
                                )
                                loss_mask_window_end_mode = (
                                    self.local_window_end_mode
                                    if routed_training_route == "local_window"
                                    else "fixed"
                                )
                                hmeta["loss_mask_window_after_tokens"] = loss_mask_window_after_tokens
                                hmeta["loss_mask_window_max_tokens"] = loss_mask_window_max_tokens
                                hmeta["loss_mask_window_end_mode"] = loss_mask_window_end_mode
                    if privilege == "key_idea" and self.key_idea_privilege_context == "tip_before_error_quote":
                        if error_quote_span is None:
                            error_quote_span = self._error_quote_span(c, hmeta)
                        hmeta = hmeta or {}
                        if error_quote_span is None:
                            teacher = self._render_teacher_key_idea(
                                problems[i], h, None, hmeta
                            )
                            hmeta["teacher_tip_fallback"] = "no_error_quote_match"
                        else:
                            start_char = error_quote_span[0]
                            teacher = self._teacher_tip_prompt(pair_student_prompt, c[:start_char], h)
                            teacher_completion = c[start_char:]
                            teacher_completion_start_char = start_char
                            hmeta["teacher_tip_before_error_quote"] = True
                            hmeta["teacher_completion_start_char"] = teacher_completion_start_char
                    if teacher is None:
                        teacher = self._render_teacher_key_idea(
                            problems[i], h, None, hmeta
                        )
                    pairs.append(DistillPair(
                        index=i, round=rnd, problem=problems[i], gold=golds[i], correct=False,
                        privilege=privilege, student_prompt=pair_student_prompt, teacher_prompt=teacher,
                        completion=c, hint=h, hint_meta=hmeta,
                        loss_mask_start_char=loss_mask_start_char,
                        loss_mask_end_char=loss_mask_end_char,
                        loss_mask_window_after_tokens=loss_mask_window_after_tokens,
                        loss_mask_window_max_tokens=loss_mask_window_max_tokens,
                        loss_mask_window_end_mode=loss_mask_window_end_mode,
                        teacher_completion=teacher_completion,
                        teacher_completion_start_char=teacher_completion_start_char,
                    ))
                    attempts[i].append(c)
                    hints[i].append(h)
                    next_active.append(i)

            active = next_active
            rnd += 1

        return pairs





    def run_reference_adaptive_repair_replacement(
        self,
        problems: Sequence[str],
        golds: Sequence[str],
        teacher_contexts: Sequence[str | None] | None = None,
        sample_ids: Sequence[str | None] | None = None,
    ) -> list[DistillPair]:
        """Multi-round causal conversation repair with replacement signals.

        Round one uses the initial key idea.  Every later actor consumes the
        full conversation plus every preceding signal, and only the latest
        feedback hint is replaced while every earlier causal turn is
        preserved.  No further response is sampled once ``max_rounds`` is
        reached.
        """
        n = len(problems)
        if len(golds) != n:
            raise ValueError("problems and golds must have equal length")
        if self.max_rounds < 1:
            raise ValueError("multi-round repair requires max_rounds >= 1")
        if self.wrong_privilege != "key_idea":
            raise ValueError("multi-round repair requires key_idea privilege")
        if self.key_idea_privilege_context != "problem_only":
            raise ValueError("multi-round repair requires problem_only privilege")
        if self.loss_mask_mode != "after_error_quote":
            raise ValueError("multi-round repair requires after_error_quote loss")
        required_teacher_renderer = "render_teacher_repair"
        for method in ("render_student_repair", required_teacher_renderer):
            if not callable(getattr(self.r, method, None)):
                raise ValueError(
                    f"multi-round repair requires renderer method {method}"
                )
        if teacher_contexts is None:
            teacher_contexts = [None] * n
        if sample_ids is None:
            sample_ids = [None] * n
        if len(teacher_contexts) != n or len(sample_ids) != n:
            raise ValueError("teacher_contexts/sample_ids must match problems")
        if any(not str(value or "").strip() for value in teacher_contexts):
            raise ValueError("multi-round repair requires every reference solution")

        selection_mode = "reference_adaptive_repair_replacement"
        reference_protocol = "reference_adaptive_repair_anchor_replacement_v2"
        repair_interface = "conversation_latest_hint_replacement"

        original_prompts = [
            self.r.render_student(problem, (), ()) for problem in problems
        ]
        original_values = [str(value) for value in self.generate(original_prompts)]
        if len(original_values) != n:
            raise ValueError("generate must return one original response per problem")

        current_indices = list(range(n))
        current_attempts = {i: original_values[i] for i in range(n)}
        current_prompts = {i: original_prompts[i] for i in range(n)}
        attempts = {i: [original_values[i]] for i in range(n)}
        signals: dict[int, list[str]] = {i: [] for i in range(n)}
        current_correct = {
            i: bool(self.verify(original_values[i], golds[i])) for i in range(n)
        }
        current_truncated = {
            i: bool(self.is_completion_truncated(original_values[i]))
            for i in range(n)
        }
        pairs: list[DistillPair] = []

        for round_number in range(1, self.max_rounds + 1):
            active = [
                i
                for i in current_indices
                if not current_correct[i] and not current_truncated[i]
            ]
            payloads = self.make_hints(
                [
                    {
                        "index": i,
                        "round": round_number,
                        "sample_id": sample_ids[i],
                        "problem": problems[i],
                        "gold": golds[i],
                        "teacher_context": teacher_contexts[i],
                        "last_attempt": current_attempts[i],
                        "wrong_attempts": list(attempts[i]),
                        "hints": (
                            [] if round_number == 1 else [signals[i][-1]]
                        ),
                        "reference_protocol": reference_protocol,
                    }
                    for i in active
                ]
            ) if active else []
            if len(payloads) != len(active):
                raise ValueError(
                    f"repair-guidance round {round_number} payload count mismatch"
                )
            revisions: dict[int, tuple[str, dict, str | None]] = {}
            for i, payload in zip(active, payloads):
                hint, meta, privilege = _decode_hint_payload(payload)
                revisions[i] = (str(hint).strip(), dict(meta or {}), privilege)

            accepted: dict[int, str] = {}
            pair_for_index: dict[int, DistillPair] = {}
            for i in current_indices:
                signal, raw_meta, raw_privilege = revisions.get(i, ("", {}, None))
                teacher_no_repair = bool(
                    raw_meta.get("decision") == "no_repair"
                    and raw_meta.get("previous_response_correct") is True
                )
                effective_correct = bool(current_correct[i] or teacher_no_repair)
                signal_ok = bool(
                    signal
                    and raw_meta.get("training_gate_passed") is True
                    and (
                        raw_privilege
                        or raw_meta.get("teacher_privilege")
                        or "key_idea"
                    ) == "key_idea"
                )
                meta = dict(raw_meta)
                meta.update(
                    {
                        "selection_mode": selection_mode,
                        "reference_protocol": reference_protocol,
                        "reference_stage": f"g{round_number}_replacement",
                        "repair_interface": repair_interface,
                        "prior_repair_signals": list(signals[i]),
                        "actor_signal_rounds": list(range(1, round_number)),
                        "teacher_added_signal_round": (
                            None if teacher_no_repair else round_number
                        ),
                        "repair_signal": signal,
                        "repair_signal_kind": (
                            "no_repair" if teacher_no_repair else "repair_hint"
                        ),
                        "verifier_attempt_correct": bool(current_correct[i]),
                        "attempt_correct": effective_correct,
                        "no_repair_previous_response_correct": teacher_no_repair,
                        "success_evidence": (
                            "teacher_no_repair"
                            if teacher_no_repair
                            else "verifier"
                            if current_correct[i]
                            else None
                        ),
                        "attempt_truncated": bool(current_truncated[i]),
                        "next_response_generated": False,
                    }
                )
                if signal_ok:
                    span = self._error_quote_span(current_attempts[i], raw_meta)
                    if span is None:
                        raise ValueError(
                            f"accepted repair signal has no anchor in round {round_number}"
                        )
                    if round_number == 1:
                        teacher_prompt = self._render_teacher_key_idea(
                            problems[i], signal, None, raw_meta
                        )
                    else:
                        teacher_prompt = self.r.render_teacher_repair(
                            problems[i],
                            attempts[i][:-1],
                            signals[i][:-1] + [signal],
                        )
                    accepted[i] = signal
                    include = True
                    mask_start = int(span[0])
                    hint = signal
                else:
                    teacher_prompt = current_prompts[i]
                    include = False
                    mask_start = None
                    hint = None
                pair = DistillPair(
                    index=i,
                    round=round_number,
                    problem=str(problems[i]),
                    gold=str(golds[i]),
                    correct=effective_correct,
                    privilege="key_idea",
                    student_prompt=current_prompts[i],
                    teacher_prompt=teacher_prompt,
                    completion=current_attempts[i],
                    hint=hint,
                    hint_meta=meta,
                    include_in_loss=include,
                    loss_mask_start_char=mask_start,
                )
                pairs.append(pair)
                pair_for_index[i] = pair

            if round_number == self.max_rounds:
                if self.terminal_retry:
                    self._apply_terminal_retry_outcomes(
                        accepted=accepted,
                        pair_for_index=pair_for_index,
                        problems=problems,
                        golds=golds,
                        attempts=attempts,
                        signals=signals,
                    )
                break

            next_indices: list[int] = []
            next_prompts: dict[int, str] = {}
            for i, signal in accepted.items():
                prompt = self.r.render_student_repair(
                    problems[i],
                    attempts[i],
                    signals[i] + [signal],
                )
                prompt_tokens = None
                count_tokens = getattr(self.r, "count_tokens", None)
                if callable(count_tokens):
                    prompt_tokens = int(count_tokens(prompt))
                if self.repair_prompt_token_limit is not None and (
                    prompt_tokens is None
                    or prompt_tokens > self.repair_prompt_token_limit
                ):
                    pair_for_index[i].hint_meta.update(
                        {
                            "next_generation_skipped_reason": "repair_prompt_too_long",
                            "next_prompt_tokens": prompt_tokens,
                            "repair_prompt_token_limit": self.repair_prompt_token_limit,
                        }
                    )
                    continue
                next_indices.append(i)
                next_prompts[i] = prompt
                pair_for_index[i].hint_meta.update(
                    {
                        "next_response_generated": True,
                        "next_prompt_tokens": prompt_tokens,
                        "repair_prompt_token_limit": self.repair_prompt_token_limit,
                    }
                )

            next_values = [
                str(value)
                for value in self.generate([next_prompts[i] for i in next_indices])
            ] if next_indices else []
            if len(next_values) != len(next_indices):
                raise ValueError(
                    f"generate must return one response for repair round {round_number}"
                )
            current_indices = next_indices
            current_attempts = dict(zip(next_indices, next_values))
            current_prompts = next_prompts
            for i, value in current_attempts.items():
                attempts[i].append(value)
                signals[i].append(accepted[i])
            current_correct = {
                i: bool(self.verify(current_attempts[i], golds[i]))
                for i in next_indices
            }
            current_truncated = {
                i: bool(self.is_completion_truncated(current_attempts[i]))
                for i in next_indices
            }

        return pairs

    def _apply_terminal_retry_outcomes(
        self,
        *,
        accepted: dict[int, str],
        pair_for_index: dict[int, Any],
        problems: Sequence[str],
        golds: Sequence[str],
        attempts: dict[int, list[str]],
        signals: dict[int, list[str]],
    ) -> None:
        """Sample the retry that follows the last supervised stage.

        The retry defines that stage's immediate-outcome indicator ``s``; it is
        never appended to the history and never becomes a distillation target,
        so the supervised stage count is unchanged.  Its fields are named apart
        from ``next_response_generated`` because that field marks a retry that
        the loop did consume as the next stage's scored response.
        """
        if not accepted:
            return
        indices: list[int] = []
        prompts: list[str] = []
        for i, signal in accepted.items():
            prompt = self.r.render_student_repair(
                problems[i],
                attempts[i],
                signals[i] + [signal],
            )
            prompt_tokens = None
            count_tokens = getattr(self.r, "count_tokens", None)
            if callable(count_tokens):
                prompt_tokens = int(count_tokens(prompt))
            if self.repair_prompt_token_limit is not None and (
                prompt_tokens is None or prompt_tokens > self.repair_prompt_token_limit
            ):
                pair_for_index[i].hint_meta.update({
                    "terminal_retry_generated": False,
                    "terminal_retry_skipped_reason": "repair_prompt_too_long",
                    "terminal_retry_prompt_tokens": prompt_tokens,
                    "terminal_retry_correct": False,
                })
                continue
            indices.append(i)
            prompts.append(prompt)
            pair_for_index[i].hint_meta.update({
                "terminal_retry_prompt_tokens": prompt_tokens,
            })
        if not indices:
            return
        values = [str(value) for value in self.generate(prompts)]
        if len(values) != len(indices):
            raise ValueError("generate must return one terminal retry per accepted stage")
        for i, value in zip(indices, values):
            truncated = bool(self.is_completion_truncated(value))
            correct = bool(self.verify(value, golds[i])) and not truncated
            pair_for_index[i].hint_meta.update({
                "terminal_retry_generated": True,
                "terminal_retry_truncated": truncated,
                # A truncated or malformed retry counts as s=0, matching the
                # treatment of an unavailable retry.
                "terminal_retry_correct": correct,
            })






def summarize(pairs: Sequence[DistillPair]) -> dict:
    """Batch-level diagnostics for logging."""
    out = {
        "pairs": len(pairs), "correct_solution": 0, "wrong_hint": 0, "wrong_key_idea": 0,
        "wrong_solution": 0, "vanilla_unsolved": 0,
        "examples": 0, "solved_examples": 0, "by_round": {},
    }
    seen: set = set()
    solved: set = set()
    for p in pairs:
        seen.add(p.index)
        if p.correct:
            out["correct_solution"] += 1
            solved.add(p.index)
        elif p.privilege == "solution":
            out["wrong_solution"] += 1
            if p.hint is None:
                out["vanilla_unsolved"] += 1
        elif p.privilege == "key_idea":
            out["wrong_key_idea"] += 1
        else:
            out["wrong_hint"] += 1
        out["by_round"][p.round] = out["by_round"].get(p.round, 0) + 1
    out["examples"] = len(seen)
    out["solved_examples"] = len(solved)
    return out
