"""Socratic-OPSD trainer.

Generalizes hint-opsd's single-round online hint-OPSD to SocraticPO-style
*multi-round* repair with *guided regeneration*, then distills the repaired
trajectories with OPSD JSD and an optional trajectory-outcome round weight.

Per training step:
  1. Run the multi-round repair loop (vLLM on-policy student rollouts + external
     or self-model hinter + SciKnowEval verifier) -> one distillation pair per
     solved example (first_correct | guided_repair); terminal_fails excluded.
  2. Assemble student/teacher input_ids (+labels) exactly like OPSD expects.
  3. Call SFTTrainer.training_step (skipping OPSDTrainer's own re-rollout) so the
     standard OPSD compute_loss / generalized_jsd_loss runs unchanged.

Requires vLLM colocate with tensor_parallel_size=1 (matches hint-opsd P370).
Only syntax-checkable off-node; validated by the Phase-2 smoke.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from accelerate.utils import gather_object
from torch.utils.checkpoint import checkpoint
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

from air_opd.opsd.jsd_loss import _elementwise_jsd
from air_opd.opsd.trainer import GOLDVLLMSyncCallback, OPSDTrainer
from air_opd.core import math_verifier, math_verifier_binary, sci_verifier
from air_opd.core.endpoint import DEFAULT_COVERAGE_EPSILON, blank_line_block_starts
from air_opd.core.endpoint_loss import coverage_masked_mean, per_token_correction_mass
from air_opd.core.render import Renderers
from air_opd.core.repair import (
    SocraticRepairLoop,
    apply_supervision_policy,
    apply_success_gated_wrong2_filter,
    assign_exponential_decay_round_weights,
    assign_trajectory_outcome_round_weights,
    assign_simple_outcome_raw_weights,
    bounded_loss_token_span,
    char_span_to_token_span,
    summarize,
)
from air_opd.core.trace_replay import ForbiddenHintClient, TraceReplayStore, trace_replay_enabled
from air_opd.core.hint_client import (
    build_conditional_reflection_messages,
    build_hint_messages,
    build_reference_multiround_messages,
    build_reflection_messages,
    build_scope_classifier_messages,
    make_hint_payload,
    normalize_error_route_mode,
    normalize_error_quote_fallback_mode,
    normalize_error_quote_locator_mode,
    normalize_leak_fallback_mode,
    normalize_reflection_prompt_mode,
    normalize_scheme3bv2_self_terminal_close,
    parse_prefix_reusable,
    reference_multiround_enabled,
    reflection_quote_source_max_chars,
    sanitize_hint,
    sanitize_key_idea_with_meta,
    sanitize_reference_multiround_with_meta,
)

try:
    from vllm import LLM, SamplingParams
except Exception:  # pragma: no cover
    LLM = None
    SamplingParams = None


def compact_zero_weight_stage_pair(pair):
    """Keep a fixed DDP stage slot without forwarding its long excluded text."""
    if float(pair.supervision_weight) != 0.0:
        return pair

    padding = copy.deepcopy(pair)
    # OPSD predicts the completion from logits starting at prompt_len - 1.
    # Keep one real prompt token so the next-token slice has one position.
    padding.student_prompt = "0"
    padding.teacher_prompt = "0"
    padding.completion = "0"
    padding.teacher_completion = "0"
    padding.hint = None
    padding.include_in_loss = False
    padding.supervision_weight = 0.0
    padding.loss_mask_start_char = 0
    padding.loss_mask_end_char = None
    padding.loss_mask_window_after_tokens = None
    padding.loss_mask_window_max_tokens = None
    padding.loss_mask_window_end_mode = None
    padding.loss_mask_start_token = 0
    padding.loss_mask_end_token = 1
    padding.valid_label_tokens = 1
    padding.teacher_completion_start_char = None
    padding.hint_meta = dict(padding.hint_meta or {})
    padding.hint_meta["zero_weight_stage_compacted"] = True
    # The compacted copy has a one-token completion, so the original block
    # partition no longer describes it.
    padding.endpoint_plan = None
    return padding


class TraceReplayCommitCallback(TrainerCallback):
    """Commit a replay group only after its optimizer step succeeds."""

    def __init__(self, trainer: "SocraticOPSDTrainer", stop_after_steps: int):
        self.trainer = trainer
        self.stop_after_steps = int(stop_after_steps)

    def on_step_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        del args, kwargs
        store = self.trainer.trace_replay_store
        if store is None:
            raise RuntimeError("trace replay callback has no store")
        store.commit_optimizer_step(completed_global_step=int(state.global_step))
        if int(state.global_step) == self.stop_after_steps:
            control.should_save = True
            control.should_training_stop = True
        elif int(state.global_step) > self.stop_after_steps:
            raise RuntimeError("trace replay advanced beyond its guarded stop step")
        return control


class SelfModelHintClient:
    """Self-model hinter backend: the student model itself generates hints via
    the trainer's vLLM engine (the 'hinter can be the student' variant)."""

    def __init__(
        self,
        trainer,
        max_new_tokens=320,
        scope_classifier_max_new_tokens=32,
        scope_classifier_temperature=0.0,
        temperature=0.7,
        top_p=0.9,
        task_type="sciknoweval",
        hint_kind="hint",
        debug_log_path: str | None = None,
        debug_sample_rate: float | None = None,
        debug_max_rows: int | None = None,
        debug_text_limit: int | None = None,
        leak_fallback_mode: str | None = None,
        error_quote_fallback_mode: str | None = None,
        error_quote_locator_mode: str | None = None,
        reflection_prompt_mode: str = "legacy_xml",
        error_route_mode: str = "legacy",
        reflection_context_mode: str = "last_wrong",
        reflector_model_path: str | None = None,
        reflector_gpu_memory_utilization: float = 0.18,
        reflector_max_model_len: int = 12000,
        reflector_max_num_seqs: int = 16,
        reflector_enforce_eager: bool = True,
        enable_thinking: bool | None = None,
    ):
        self.t = trainer
        if enable_thinking is not None:
            self.enable_thinking = bool(enable_thinking)
        else:
            self.enable_thinking = bool(
                getattr(self.t, "socratic_self_hint_thinking", False)
                or int(os.environ.get("SOCRATIC_SELF_HINT_THINKING", "0"))
            )
        self.max_new_tokens = max_new_tokens
        self.scope_classifier_max_new_tokens = int(scope_classifier_max_new_tokens)
        if self.scope_classifier_max_new_tokens < 1:
            raise ValueError("scope_classifier_max_new_tokens must be >= 1")
        self.scope_classifier_temperature = float(scope_classifier_temperature)
        if self.scope_classifier_temperature < 0:
            raise ValueError("scope_classifier_temperature must be >= 0")
        self.temperature = temperature
        self.top_p = top_p
        self.task_type = task_type
        if hint_kind not in {"hint", "rosd_key_idea"}:
            raise ValueError("hint_kind must be 'hint' or 'rosd_key_idea'")
        self.hint_kind = hint_kind
        self.leak_fallback_mode = normalize_leak_fallback_mode(leak_fallback_mode)
        self.error_quote_fallback_mode = normalize_error_quote_fallback_mode(error_quote_fallback_mode)
        self.error_quote_locator_mode = normalize_error_quote_locator_mode(error_quote_locator_mode)
        self.reflection_prompt_mode = normalize_reflection_prompt_mode(reflection_prompt_mode)
        self.error_route_mode = normalize_error_route_mode(error_route_mode)
        self.reflection_context_mode = str(reflection_context_mode or "last_wrong")
        if self.reflection_context_mode not in {"last_wrong", "all_wrong"}:
            raise ValueError("reflection_context_mode must be 'last_wrong' or 'all_wrong'")
        self.reflector_model_path = str(reflector_model_path or "").strip()
        self.reflector_engine = None
        self.reflector_gpu_memory_utilization = float(reflector_gpu_memory_utilization)
        self.reflector_max_model_len = int(reflector_max_model_len)
        self.reflector_max_num_seqs = int(reflector_max_num_seqs)
        self.reflector_enforce_eager = bool(reflector_enforce_eager)
        if self.reflector_model_path:
            if LLM is None:
                raise ImportError("vLLM is required for socratic_reflector_model_path.")
            print(
                "[self_hinter] loading independent reflector model "
                f"path={self.reflector_model_path} gpu_mem={self.reflector_gpu_memory_utilization} "
                f"max_model_len={self.reflector_max_model_len} max_num_seqs={self.reflector_max_num_seqs}",
                flush=True,
            )
            self.reflector_engine = LLM(
                model=self.reflector_model_path,
                trust_remote_code=True,
                tensor_parallel_size=1,
                gpu_memory_utilization=self.reflector_gpu_memory_utilization,
                max_model_len=self.reflector_max_model_len,
                max_num_seqs=self.reflector_max_num_seqs,
                enforce_eager=self.reflector_enforce_eager,
            )
        self.debug_log_path = Path(debug_log_path) if debug_log_path else None
        self.debug_sample_rate = float(
            os.environ.get("SOCRATIC_HINT_DEBUG_SAMPLE_RATE", debug_sample_rate if debug_sample_rate is not None else 0.05)
        )
        self.debug_max_rows = int(
            os.environ.get("SOCRATIC_HINT_DEBUG_MAX_ROWS", debug_max_rows if debug_max_rows is not None else 1024)
        )
        self.debug_text_limit = int(
            os.environ.get("SOCRATIC_HINT_DEBUG_TEXT_LIMIT", debug_text_limit if debug_text_limit is not None else 12000)
        )
        self._debug_rows_written = 0
        if self.debug_log_path and self.t.accelerator.is_main_process:
            self.debug_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _clip_debug_text(self, text):
        s = str(text or "")
        limit = max(0, int(self.debug_text_limit))
        if limit and len(s) > limit:
            return s[:limit], True, len(s)
        return s, False, len(s)

    def _should_debug_log(self, item: dict, meta: dict) -> bool:
        if not self.debug_log_path or not self.t.accelerator.is_main_process:
            return False
        if self.debug_max_rows >= 0 and self._debug_rows_written >= self.debug_max_rows:
            return False
        if meta.get("fallback_reason") or meta.get("leaked"):
            return True
        if self.debug_sample_rate <= 0:
            return False
        if self.debug_sample_rate >= 1:
            return True
        key = "|".join(
            str(x)
            for x in (
                item.get("sample_id"),
                item.get("index"),
                item.get("round"),
                item.get("gold"),
                str(item.get("last_attempt", ""))[:128],
            )
        )
        value = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:12], 16) / float(16**12)
        return value < self.debug_sample_rate

    def _write_debug_rows(self, items, raw_texts, hints, metas):
        if not self.debug_log_path or self.hint_kind != "rosd_key_idea":
            return
        rows = []
        for item, raw, hint, meta in zip(items, raw_texts, hints, metas):
            if not self._should_debug_log(item, meta):
                continue
            problem, problem_trunc, problem_chars = self._clip_debug_text(item.get("problem", ""))
            wrong, wrong_trunc, wrong_chars = self._clip_debug_text(item.get("last_attempt", ""))
            raw_out, raw_trunc, raw_chars = self._clip_debug_text(raw)
            sanitized, sanitized_trunc, sanitized_chars = self._clip_debug_text(hint)
            rows.append({
                "step": int(getattr(self.t.state, "global_step", 0)),
                "sample_id": item.get("sample_id"),
                "index": item.get("index"),
                "round": item.get("round"),
                "task_type": self.task_type,
                "hint_kind": self.hint_kind,
                "gold": item.get("gold"),
                "problem": problem,
                "wrong_rollout": wrong,
                "raw_reflection": raw_out,
                "sanitized_hint": sanitized,
                "sanitize_meta": meta,
                "reflector_model_path": self.reflector_model_path or None,
                "text_chars": {
                    "problem": problem_chars,
                    "wrong_rollout": wrong_chars,
                    "raw_reflection": raw_chars,
                    "sanitized_hint": sanitized_chars,
                    "teacher_context": len(str(item.get("teacher_context") or "")),
                },
                "truncated": {
                    "problem": problem_trunc,
                    "wrong_rollout": wrong_trunc,
                    "raw_reflection": raw_trunc,
                    "sanitized_hint": sanitized_trunc,
                },
            })
            self._debug_rows_written += 1
        if not rows or not self.t.accelerator.is_main_process:
            return
        with self.debug_log_path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _generate_texts(self, prompts, max_new_tokens=None, temperature=None):
        generation_limit = int(max_new_tokens or self.max_new_tokens)
        generation_temperature = self.temperature if temperature is None else float(temperature)
        generation_top_p = self.top_p if generation_temperature > 0 else 1.0
        if self.reflector_engine is None:
            return self.t._vllm_generate(
                prompts, generation_limit, generation_temperature, generation_top_p
            )
        top_k = self.t.args.top_k if getattr(self.t.args, "top_k", None) else -1
        params = SamplingParams(
            n=1,
            temperature=generation_temperature,
            top_p=generation_top_p,
            top_k=top_k,
            max_tokens=generation_limit,
        )
        outputs = self.reflector_engine.generate(list(prompts), sampling_params=params, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]

    def _render_messages(self, messages):
        return [
            self.t.processing_class.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            for msg in messages
        ]

    def _make_two_stage_key_ideas(self, items):
        classifier_messages = [
            build_scope_classifier_messages(
                it["problem"],
                it.get("last_attempt", ""),
                correct_rollout=it.get("teacher_context"),
            )
            for it in items
        ]
        classifier_prompts = self._render_messages(classifier_messages)
        classifier_texts = self._generate_texts(
            classifier_prompts,
            max_new_tokens=self.scope_classifier_max_new_tokens,
            temperature=self.scope_classifier_temperature,
        )
        classifier_meta = [parse_prefix_reusable(text) for text in classifier_texts]

        valid_indices = [
            index for index, meta in enumerate(classifier_meta)
            if bool(meta.get("prefix_reusable_valid"))
        ]
        artifact_messages = [
            build_conditional_reflection_messages(
                items[index]["problem"],
                items[index].get("last_attempt", ""),
                items[index].get("teacher_context"),
                classifier_meta[index]["predicted_scope"],
            )
            for index in valid_indices
        ]
        artifact_texts = self._generate_texts(self._render_messages(artifact_messages)) if artifact_messages else []
        artifacts_by_index = dict(zip(valid_indices, artifact_texts))

        composed_texts = []
        for index, meta in enumerate(classifier_meta):
            scope = str(meta.get("predicted_scope") or "")
            artifact = str(artifacts_by_index.get(index) or "")
            if meta.get("prefix_reusable_valid"):
                composed = f"<error_scope>{scope}</error_scope>\n{artifact}"
            else:
                composed = "<error_scope>invalid</error_scope>\n<explanation></explanation>"
            composed_texts.append(composed)

        pairs = [
            sanitize_key_idea_with_meta(
                text,
                it["gold"],
                task_type=self.task_type,
                leak_fallback_mode=self.leak_fallback_mode,
                quote_source=it.get("last_attempt", ""),
                error_quote_fallback_mode=self.error_quote_fallback_mode,
                error_quote_locator_mode=self.error_quote_locator_mode,
                error_route_mode=self.error_route_mode,
                reflection_prompt_mode=self.reflection_prompt_mode,
            )
            for text, it in zip(composed_texts, items)
        ]
        for index, ((_, meta), it) in enumerate(zip(pairs, items)):
            source_chars = len(str(it.get("last_attempt") or ""))
            visible_chars = min(source_chars, reflection_quote_source_max_chars(self.reflection_prompt_mode))
            meta.update(classifier_meta[index])
            meta["scope_classifier_prompt"] = classifier_prompts[index]
            meta["conditional_artifact_raw"] = artifacts_by_index.get(index)
            meta["reflection_prompt_mode"] = self.reflection_prompt_mode
            meta["quote_source_total_chars"] = source_chars
            meta["quote_source_visible_chars"] = visible_chars
            meta["quote_source_truncated"] = visible_chars < source_chars
            if not classifier_meta[index].get("prefix_reusable_valid"):
                meta["teacher_privilege"] = "solution"
                meta["routing_fallback_reason"] = "invalid_prefix_reusable"
                reasons = list(meta.get("route_reasons") or [])
                if "invalid_prefix_reusable" not in reasons:
                    reasons.insert(0, "invalid_prefix_reusable")
                meta["route_reasons"] = reasons
            if os.environ.get("SOCRATIC_FULL_TRACE_ENABLED", "0") == "1":
                meta["raw_reflection"] = composed_texts[index]
                meta["reflection_quote_source"] = str(it.get("last_attempt") or "")
                meta["reference_solution"] = str(it.get("teacher_context") or "")
        hints = [hint for hint, _ in pairs]
        metas = [meta for _, meta in pairs]
        self._write_debug_rows(items, composed_texts, hints, metas)
        return [make_hint_payload(hint, meta) for hint, meta in pairs]

    def _make_reference_multiround_key_ideas(self, items):
        """Generate Scheme3b-v2 G1/G2/G3 packets with the current student."""
        messages = [build_reference_multiround_messages(it) for it in items]
        prompts = self._render_messages(messages)
        raw_texts = self._generate_texts(prompts)
        normalized_rows = [
            normalize_scheme3bv2_self_terminal_close(
                raw,
                str(it.get("reference_protocol") or "reference_p1_p2_v1"),
            )
            for raw, it in zip(raw_texts, items)
        ]
        pairs = [
            sanitize_reference_multiround_with_meta(
                normalized,
                it["gold"],
                task_type=self.task_type,
                quote_source=it.get("last_attempt", ""),
                round_number=int(it.get("round") or 1),
                reference_protocol=str(
                    it.get("reference_protocol") or "reference_p1_p2_v1"
                ),
            )
            for (normalized, _), it in zip(normalized_rows, items)
        ]
        for index, ((_, meta), it, messages_for_item) in enumerate(
            zip(pairs, items, messages)
        ):
            raw = str(raw_texts[index])
            normalized, appended = normalized_rows[index]
            source_chars = len(str(it.get("last_attempt") or ""))
            visible_chars = min(
                source_chars,
                reflection_quote_source_max_chars(self.reflection_prompt_mode),
            )
            meta["reflection_prompt_mode"] = self.reflection_prompt_mode
            meta["hint_generator_backend"] = "self_model_current_policy"
            meta["self_terminal_close_appended"] = bool(appended)
            meta["quote_source_total_chars"] = source_chars
            meta["quote_source_visible_chars"] = visible_chars
            meta["quote_source_truncated"] = visible_chars < source_chars
            if os.environ.get("SOCRATIC_FULL_TRACE_ENABLED", "0") == "1":
                canonical_messages = json.dumps(
                    messages_for_item,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                meta["raw_reflection"] = raw
                if appended:
                    meta["normalized_reflection"] = normalized
                meta["reflection_messages"] = messages_for_item
                meta["reflection_messages_sha256"] = hashlib.sha256(
                    canonical_messages.encode("utf-8")
                ).hexdigest()
                meta["reflection_quote_source"] = str(
                    it.get("last_attempt") or ""
                )
                meta["reference_solution"] = str(
                    it.get("teacher_context") or ""
                )
        hints = [hint for hint, _ in pairs]
        metas = [meta for _, meta in pairs]
        self._write_debug_rows(items, raw_texts, hints, metas)
        return [make_hint_payload(hint, meta) for hint, meta in pairs]

    def make_hints(self, items):
        if self.hint_kind == "rosd_key_idea" and reference_multiround_enabled():
            return self._make_reference_multiround_key_ideas(items)
        if (
            self.hint_kind == "rosd_key_idea"
            and self.reflection_prompt_mode == "two_stage_scope_xml_v1"
        ):
            return self._make_two_stage_key_ideas(items)
        messages = []
        for it in items:
            if self.hint_kind == "rosd_key_idea":
                msg = build_reflection_messages(
                    it["problem"],
                    it["gold"],
                    wrong_rollout=it.get("last_attempt", ""),
                    wrong_rollouts=it.get("wrong_attempts") if self.reflection_context_mode == "all_wrong" else None,
                    correct_rollout=it.get("teacher_context"),
                    task_type=self.task_type,
                    prompt_mode=self.reflection_prompt_mode,
                )
            else:
                msg = build_hint_messages(
                    it["problem"],
                    it["gold"],
                    it.get("last_attempt", ""),
                    it.get("hints", []),
                    teacher_context=it.get("teacher_context"),
                    task_type=self.task_type,
                )
            messages.append(msg)
        prompts = self._render_messages(messages)
        texts = self._generate_texts(prompts)
        if self.hint_kind == "rosd_key_idea":
            pairs = [
                sanitize_key_idea_with_meta(
                    t,
                    it["gold"],
                    task_type=self.task_type,
                    leak_fallback_mode=self.leak_fallback_mode,
                    quote_source=it.get("last_attempt", ""),
                    error_quote_fallback_mode=self.error_quote_fallback_mode,
                    error_quote_locator_mode=self.error_quote_locator_mode,
                    error_route_mode=self.error_route_mode,
                    reflection_prompt_mode=self.reflection_prompt_mode,
                )
                for t, it in zip(texts, items)
            ]
            for (_, meta), it in zip(pairs, items):
                source_chars = len(str(it.get("last_attempt") or ""))
                visible_chars = min(
                    source_chars,
                    reflection_quote_source_max_chars(self.reflection_prompt_mode),
                )
                meta["reflection_prompt_mode"] = self.reflection_prompt_mode
                meta["quote_source_total_chars"] = source_chars
                meta["quote_source_visible_chars"] = visible_chars
                meta["quote_source_truncated"] = visible_chars < source_chars
            if os.environ.get("SOCRATIC_FULL_TRACE_ENABLED", "0") == "1":
                for (_, meta), raw, it in zip(pairs, texts, items):
                    meta["raw_reflection"] = str(raw)
                    meta["reflection_quote_source"] = str(it.get("last_attempt") or "")
                    meta["reference_solution"] = str(it.get("teacher_context") or "")
            hints = [h for h, _ in pairs]
            metas = [m for _, m in pairs]
            self._write_debug_rows(items, texts, hints, metas)
            return [make_hint_payload(h, m) for h, m in pairs]
        return [
            sanitize_hint(t, it["gold"], task_type=self.task_type, leak_fallback_mode=self.leak_fallback_mode)
            for t, it in zip(texts, items)
        ]




class SocraticOPSDTrainer(OPSDTrainer):
    def __init__(
        self,
        *args,
        socratic_max_rounds: int = 3,
        socratic_distill_mode: str = "guided",
        socratic_privilege_mode: str = "auto",
        hinter_backend: str = "self_model",
        hint_max_new_tokens: int = 320,
        scope_classifier_max_new_tokens: int = 32,
        scope_classifier_temperature: float = 0.0,
        hint_temperature: float = 0.7,
        hint_top_p: float = 0.9,
        hint_concurrency: int = 8,
        hint_log_path: str | None = None,
        task_type: str = "sciknoweval",
        math_verifier_profile: str = "legacy",
        socratic_leak_fallback_mode: str = "safe_hint",
        socratic_reflection_context_mode: str = "last_wrong",
        socratic_wrong_pair_student_context: str = "on_policy",
        socratic_key_idea_privilege_context: str = "problem_only",
        socratic_loss_mask_mode: str = "full",
        socratic_error_quote_fallback_mode: str = "quote_source_prefix",
        socratic_error_quote_locator_mode: str = "strict",
        socratic_reflection_prompt_mode: str = "legacy_xml",
        socratic_error_route_mode: str = "legacy",
        socratic_local_window_after_tokens: int = 128,
        socratic_local_window_max_tokens: int = 256,
        socratic_local_window_end_mode: str = "fixed",
        socratic_wrong_pair_filter_mode: str = "none",
        socratic_supervision_scope: str = "all",
        socratic_solution_privilege_mode: str = "allow",
        socratic_repair_selection_mode: str = "legacy_every_trajectory",
        socratic_round_weight_mode: str = "uniform",
        socratic_round_weight_alpha: float = 0.8,
        socratic_round_weight_rho: float = 1.5,
        socratic_endpoint_mode: str = "full_suffix",
        socratic_endpoint_epsilon: float = DEFAULT_COVERAGE_EPSILON,
        socratic_terminal_retry: bool = False,
        socratic_max_repair_attempts: int = 2,
        socratic_reflector_model_path: str | None = None,
        socratic_reflector_gpu_memory_utilization: float = 0.18,
        socratic_reflector_max_model_len: int = 12000,
        socratic_reflector_max_num_seqs: int = 16,
        socratic_reflector_enforce_eager: bool = True,
        socratic_self_hint_thinking: bool | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.trace_replay_mode = trace_replay_enabled()
        self.guarded_resume_smoke_mode = (
            os.environ.get("SOCRATIC_GUARDED_RESUME_SMOKE", "0").strip() == "1"
        )
        self.trace_replay_store: TraceReplayStore | None = None
        if not self.use_vllm:
            raise ValueError("SocraticOPSDTrainer requires use_vllm=True (colocate, TP=1).")
        # Socratic generation synchronizes at the repair-loop boundary. The
        # parent step-end callback would leave weights mapped while KV remains
        # asleep, making the next boundary preparation ambiguous.
        self.remove_callback(GOLDVLLMSyncCallback)
        if self.reason_first:
            raise ValueError("SocraticOPSDTrainer does not support reason_first.")
        if SamplingParams is None:
            raise ImportError("vLLM is required for SocraticOPSDTrainer.")

        self.socratic_max_rounds = int(socratic_max_rounds)
        self.socratic_distill_mode = socratic_distill_mode
        self.socratic_leak_fallback_mode = normalize_leak_fallback_mode(socratic_leak_fallback_mode)
        self.socratic_reflection_context_mode = str(socratic_reflection_context_mode or "last_wrong")
        if self.socratic_reflection_context_mode not in {"last_wrong", "all_wrong"}:
            raise ValueError("socratic_reflection_context_mode must be 'last_wrong' or 'all_wrong'")
        self.socratic_wrong_pair_student_context = str(socratic_wrong_pair_student_context or "on_policy")
        if self.socratic_wrong_pair_student_context not in {"on_policy", "root_question_after_first"}:
            raise ValueError("socratic_wrong_pair_student_context must be 'on_policy' or 'root_question_after_first'")
        self.socratic_key_idea_privilege_context = str(socratic_key_idea_privilege_context or "problem_only")
        if self.socratic_key_idea_privilege_context not in {
            "problem_only",
            "problem_wrong_attempt",
            "tip_before_error_quote",
        }:
            raise ValueError(
                "socratic_key_idea_privilege_context must be 'problem_only', "
                "'problem_wrong_attempt', or 'tip_before_error_quote'"
            )
        self.socratic_loss_mask_mode = str(socratic_loss_mask_mode or "full")
        if self.socratic_loss_mask_mode not in {
            "full",
            "after_error_quote",
            "error_quote_only",
            "error_type_routed",
            "two_scope_routed",
            "two_scope_bounded",
        }:
            raise ValueError(
                "socratic_loss_mask_mode must be 'full', 'after_error_quote', "
                "'error_quote_only', 'error_type_routed', 'two_scope_routed', "
                "or 'two_scope_bounded'"
            )
        self.socratic_error_quote_fallback_mode = normalize_error_quote_fallback_mode(
            socratic_error_quote_fallback_mode
        )
        self.socratic_error_quote_locator_mode = normalize_error_quote_locator_mode(
            socratic_error_quote_locator_mode
        )
        self.socratic_reflection_prompt_mode = normalize_reflection_prompt_mode(
            socratic_reflection_prompt_mode
        )
        self.scope_classifier_max_new_tokens = int(scope_classifier_max_new_tokens)
        if self.scope_classifier_max_new_tokens < 1:
            raise ValueError("scope_classifier_max_new_tokens must be >= 1")
        self.scope_classifier_temperature = float(scope_classifier_temperature)
        if self.scope_classifier_temperature < 0:
            raise ValueError("scope_classifier_temperature must be >= 0")
        self.socratic_error_route_mode = normalize_error_route_mode(socratic_error_route_mode)
        external_hint_top_p = (
            hint_top_p if self.socratic_reflection_prompt_mode != "legacy_xml" else None
        )
        self.socratic_local_window_after_tokens = int(socratic_local_window_after_tokens)
        self.socratic_local_window_max_tokens = int(socratic_local_window_max_tokens)
        self.socratic_local_window_end_mode = str(
            socratic_local_window_end_mode or "fixed"
        )
        if self.socratic_local_window_after_tokens < 0:
            raise ValueError("socratic_local_window_after_tokens must be >= 0")
        if self.socratic_local_window_max_tokens < 1:
            raise ValueError("socratic_local_window_max_tokens must be >= 1")
        if self.socratic_local_window_end_mode not in {"fixed", "next_blank"}:
            raise ValueError(
                "socratic_local_window_end_mode must be 'fixed' or 'next_blank'"
            )
        if self.socratic_loss_mask_mode == "error_type_routed":
            if self.socratic_reflection_prompt_mode != "routed_xml_v1":
                raise ValueError("error_type_routed requires socratic_reflection_prompt_mode=routed_xml_v1")
            if self.socratic_error_route_mode != "three_way_v1":
                raise ValueError("error_type_routed requires socratic_error_route_mode=three_way_v1")
        if self.socratic_loss_mask_mode == "two_scope_routed":
            if self.socratic_reflection_prompt_mode not in {
                "two_scope_xml_v1",
                "scope_sft_xml_v1",
                "external_scope_then_quote_action_xml_v1",
                "deepseek_scope_noref_xml_v1",
                "two_stage_scope_xml_v1",
            }:
                raise ValueError(
                    "two_scope_routed requires socratic_reflection_prompt_mode="
                    "two_scope_xml_v1, scope_sft_xml_v1, "
                    "external_scope_then_quote_action_xml_v1, "
                    "deepseek_scope_noref_xml_v1, or two_stage_scope_xml_v1"
                )
            if self.socratic_error_route_mode != "two_scope_v1":
                raise ValueError(
                    "two_scope_routed requires socratic_error_route_mode=two_scope_v1"
                )
        if self.socratic_loss_mask_mode == "two_scope_bounded":
            if (
                self.socratic_reflection_prompt_mode
                != "deepseek_scope_bounded_noref_xml_v1"
            ):
                raise ValueError(
                    "two_scope_bounded requires "
                    "socratic_reflection_prompt_mode="
                    "deepseek_scope_bounded_noref_xml_v1"
                )
            if self.socratic_error_route_mode != "two_scope_bounded_v1":
                raise ValueError(
                    "two_scope_bounded requires "
                    "socratic_error_route_mode=two_scope_bounded_v1"
                )
        self.socratic_wrong_pair_filter_mode = str(socratic_wrong_pair_filter_mode or "none")
        if self.socratic_wrong_pair_filter_mode not in {
            "none",
            "select_one_r2_quality",
            "success_gated_wrong2",
        }:
            raise ValueError(
                "socratic_wrong_pair_filter_mode must be 'none', "
                "'select_one_r2_quality', or 'success_gated_wrong2'"
            )
        self.socratic_supervision_scope = str(socratic_supervision_scope or "all")
        if self.socratic_supervision_scope not in {"all", "initial_wrong_only"}:
            raise ValueError(
                "socratic_supervision_scope must be 'all' or 'initial_wrong_only'"
            )
        self.socratic_solution_privilege_mode = str(
            socratic_solution_privilege_mode or "allow"
        )
        if self.socratic_solution_privilege_mode not in {"allow", "exclude"}:
            raise ValueError(
                "socratic_solution_privilege_mode must be 'allow' or 'exclude'"
            )
        self.socratic_repair_selection_mode = str(
            socratic_repair_selection_mode or "legacy_every_trajectory"
        )
        if self.socratic_repair_selection_mode not in {
            "legacy_every_trajectory",
            "reference_scheme3bv2_replacement",
        }:
            raise ValueError(
                "socratic_repair_selection_mode must be legacy_every_trajectory "
                "or reference_scheme3bv2_replacement"
            )
        self.socratic_round_weight_mode = str(
            socratic_round_weight_mode or "uniform"
        )
        self.socratic_round_weight_alpha = float(socratic_round_weight_alpha)
        self.socratic_round_weight_rho = float(socratic_round_weight_rho)
        if self.socratic_round_weight_mode not in {
            "uniform",
            "simple_outcome_2_1_half",
            "exp_decay_g08_terminal_half_postnorm",
            "trajectory_outcome_alpha08_rho15",
        }:
            raise ValueError(
                "socratic_round_weight_mode must be uniform, "
                "simple_outcome_2_1_half, or "
                "exp_decay_g08_terminal_half_postnorm, or trajectory_outcome_alpha08_rho15"
            )
        if (
            self.socratic_round_weight_mode in {
                "simple_outcome_2_1_half",
                "exp_decay_g08_terminal_half_postnorm",
            }
            and self.socratic_repair_selection_mode
            != "reference_scheme3bv2_replacement"
        ):
            raise ValueError(
                "Scheme3b-v2 outcome round weighting requires "
                "reference_scheme3bv2_replacement"
            )
        if self.socratic_round_weight_mode == "trajectory_outcome_alpha08_rho15":
            if not (0.0 < self.socratic_round_weight_alpha <= 1.0):
                raise ValueError("socratic_round_weight_alpha must be in (0,1]")
            if self.socratic_round_weight_rho <= 0.0:
                raise ValueError("socratic_round_weight_rho must be positive")
        if (
            self.socratic_round_weight_mode == "exp_decay_g08_terminal_half_postnorm"
            and self.socratic_max_rounds != 3
        ):
            raise ValueError(
                "exp_decay_g08_terminal_half_postnorm requires socratic_max_rounds=3"
            )
        # Supervision endpoint.  ``full_suffix`` is the historical behavior of
        # ``after_error_quote`` and keeps every frozen run reproducible;
        # ``cumulative_correction_coverage`` truncates the suffix at the earliest
        # block end retaining 1-epsilon of its corrective signal.  The rule needs
        # both scored distributions over the whole suffix, so it can only be
        # applied where the pre-mask already spans [b, T).
        self.socratic_endpoint_mode = str(socratic_endpoint_mode or "full_suffix")
        self.socratic_endpoint_epsilon = float(socratic_endpoint_epsilon)
        if self.socratic_endpoint_mode not in {
            "full_suffix",
            "cumulative_correction_coverage",
        }:
            raise ValueError(
                "socratic_endpoint_mode must be 'full_suffix' or "
                "'cumulative_correction_coverage'"
            )
        if self.socratic_endpoint_mode == "cumulative_correction_coverage":
            if self.socratic_loss_mask_mode != "after_error_quote":
                raise ValueError(
                    "cumulative_correction_coverage requires "
                    "socratic_loss_mask_mode=after_error_quote"
                )
            if not 0.0 < self.socratic_endpoint_epsilon < 1.0:
                raise ValueError("socratic_endpoint_epsilon must be in (0,1)")
            if getattr(self, "top_k_loss", None):
                # Top-k restriction renormalizes both branches over the
                # teacher's top-k and can drop the observed token, so the
                # corrective mass would no longer be the quantity the endpoint
                # rule is defined on.  Fail closed rather than silently differ.
                raise ValueError(
                    "cumulative_correction_coverage is incompatible with top_k_loss"
                )
            if self.use_thinking_machines_loss:
                raise ValueError(
                    "cumulative_correction_coverage requires the JSD objective"
                )
        self._socratic_current_endpoint_plan = None
        self._socratic_last_endpoint_record = None
        self._socratic_endpoint_records: list[dict[str, Any]] = []
        self.socratic_self_hint_thinking = bool(
            socratic_self_hint_thinking
            if socratic_self_hint_thinking is not None
            else int(os.environ.get("SOCRATIC_SELF_HINT_THINKING", "0"))
        )
        self.socratic_terminal_retry = bool(socratic_terminal_retry)
        if self.socratic_terminal_retry and self.socratic_repair_selection_mode not in {
            "reference_scheme3bv2_replacement",
        }:
            raise ValueError(
                "socratic_terminal_retry is implemented for the Scheme3b conversation "
                "paths only"
            )
        self.socratic_max_repair_attempts = int(socratic_max_repair_attempts)
        if self.socratic_max_repair_attempts not in {1, 2}:
            raise ValueError("socratic_max_repair_attempts must be 1 or 2")
        if socratic_privilege_mode == "auto":
            resolved_privilege_mode = (
                "vanilla" if self.socratic_max_rounds == 1 and hinter_backend == "self_model" else "hint"
            )
        else:
            resolved_privilege_mode = socratic_privilege_mode
        if resolved_privilege_mode not in {"vanilla", "hint", "rosd_key_idea"}:
            raise ValueError("socratic_privilege_mode must be auto, vanilla, hint, or rosd_key_idea")
        if (
            self.socratic_loss_mask_mode
            in {"error_type_routed", "two_scope_routed", "two_scope_bounded"}
            and resolved_privilege_mode != "rosd_key_idea"
        ):
            raise ValueError(
                "routed loss requires socratic_privilege_mode=rosd_key_idea"
            )
        self.socratic_privilege_mode = resolved_privilege_mode
        if (
            self.socratic_reflection_prompt_mode == "two_stage_scope_xml_v1"
            and hinter_backend != "self_model"
        ):
            raise ValueError("two_stage_scope_xml_v1 currently requires hinter_backend=self_model")
        self.socratic_vanilla_baseline = resolved_privilege_mode == "vanilla"
        if self.socratic_repair_selection_mode in {
            "reference_scheme3bv2_replacement",
        }:
            mode_name = self.socratic_repair_selection_mode
            if mode_name == "reference_scheme3bv2_replacement" and self.socratic_max_rounds < 1:
                raise ValueError(
                    f"{mode_name} requires socratic_max_rounds>=1"
                )
            if self.socratic_privilege_mode != "rosd_key_idea":
                raise ValueError(f"{mode_name} requires rosd_key_idea privilege")
            if self.socratic_wrong_pair_filter_mode != "none":
                raise ValueError(f"{mode_name} requires wrong_pair_filter_mode=none")
            if self.socratic_key_idea_privilege_context != "problem_only":
                raise ValueError(f"{mode_name} requires problem_only key-idea context")
            if self.socratic_loss_mask_mode != "after_error_quote":
                raise ValueError(f"{mode_name} requires after_error_quote loss")
            if self.socratic_supervision_scope != "initial_wrong_only":
                raise ValueError(f"{mode_name} requires initial_wrong_only supervision")
            if self.socratic_solution_privilege_mode != "exclude":
                raise ValueError(f"{mode_name} requires solution privilege exclusion")
            if hinter_backend not in {"self_model", "external_model"}:
                raise ValueError(
                    f"{mode_name} requires a self_model or external_model hinter"
                )
            if (
                mode_name == "reference_scheme3bv2_replacement"
                and hinter_backend == "self_model"
                and str(socratic_reflector_model_path or "").strip()
            ):
                raise ValueError(
                    "Scheme3b-v2 self_model requires the current student; "
                    "socratic_reflector_model_path must be empty"
                )
            if (
                mode_name == "reference_scheme3bv2_replacement"
                and hinter_backend == "external_model"
                and not str(socratic_reflector_model_path or "").strip()
            ):
                raise ValueError(
                    "Scheme3b-v2 external_model generates the repair guidance "
                    "with a separate model; socratic_reflector_model_path is "
                    "required"
                )
            if os.environ.get("SOCRATIC_REFERENCE_MULTIROUND", "0") != "1":
                raise ValueError(f"{mode_name} requires SOCRATIC_REFERENCE_MULTIROUND=1")
        hint_kind = "rosd_key_idea" if resolved_privilege_mode == "rosd_key_idea" else "hint"
        hint_debug_log_path = os.environ.get("SOCRATIC_HINT_DEBUG_LOG_PATH")
        if not hint_debug_log_path and hint_log_path and hint_kind == "rosd_key_idea":
            hint_debug_log_path = str(Path(hint_log_path).with_name("hint_context_debug.jsonl"))
        if task_type not in {"sciknoweval", "math"}:
            raise ValueError("task_type must be 'sciknoweval' or 'math'")
        self.task_type = task_type
        self.math_verifier_profile = str(math_verifier_profile or "legacy")
        if self.math_verifier_profile not in {
            "legacy",
            "binary_math_verify_v1",
            "binary_math_verify_compat_v2",
        }:
            raise ValueError(
                "math_verifier_profile must be legacy, binary_math_verify_v1, "
                "or binary_math_verify_compat_v2"
            )
        if self.trace_replay_mode:
            self.math_verifier = None
            self.verify_fn = None
        elif task_type == "math":
            self.math_verifier = (
                math_verifier_binary
                if self.math_verifier_profile
                in {"binary_math_verify_v1", "binary_math_verify_compat_v2"}
                else math_verifier
            )
            if self.math_verifier_profile in {
                "binary_math_verify_v1",
                "binary_math_verify_compat_v2",
            }:
                math_verifier_binary.validate_backend()
            self.verify_fn = (
                math_verifier_binary.grade_compatible
                if self.math_verifier_profile == "binary_math_verify_compat_v2"
                else self.math_verifier.grade
            )
        else:
            self.math_verifier = None
            self.verify_fn = sci_verifier.grade
        self.renderers = None if self.trace_replay_mode else Renderers(
            self.processing_class,
            student_thinking=self.student_thinking if hasattr(self, "student_thinking") else False,
            teacher_thinking=self.teacher_thinking if hasattr(self, "teacher_thinking") else True,
            task_type=task_type,
        )
        # hint backend
        if self.trace_replay_mode:
            trace_path = os.environ.get("SOCRATIC_TRACE_REPLAY_PATH", "").strip()
            receipt_dir = os.environ.get("SOCRATIC_TRACE_REPLAY_RECEIPT_DIR", "").strip()
            expected_steps = int(os.environ.get("SOCRATIC_TRACE_REPLAY_EXPECTED_STEPS", "0"))
            start_step = int(os.environ.get("SOCRATIC_TRACE_REPLAY_START_STEP", "0"))
            rows_per_group = int(
                os.environ.get("SOCRATIC_TRACE_REPLAY_ROWS_PER_GROUP", "0")
            )
            source_sha = os.environ.get("SOCRATIC_TRACE_REPLAY_SHA256", "").strip() or None
            expected_micro_rows = int(self.args.gradient_accumulation_steps) * int(
                self.args.per_device_train_batch_size
            )
            if int(self.args.per_device_train_batch_size) != 1:
                raise ValueError("trace replay requires frozen per_device_train_batch_size=1")
            if rows_per_group != expected_micro_rows:
                raise ValueError(
                    "trace replay rows/group must equal gradient_accumulation_steps * "
                    "per_device_train_batch_size: "
                    f"rows={rows_per_group} expected={expected_micro_rows}"
                )
            self.trace_replay_store = TraceReplayStore(
                trace_path=trace_path,
                receipt_dir=receipt_dir,
                rank=int(self.accelerator.process_index),
                world_size=int(self.accelerator.num_processes),
                start_step=start_step,
                expected_steps=expected_steps,
                expected_rows_per_group=rows_per_group,
                expected_sha256=source_sha,
            )
            self.trace_replay_store.validate_token_masks(
                self.processing_class, int(self.args.max_completion_length)
            )
            self.hint_client = ForbiddenHintClient()
            self.add_callback(
                TraceReplayCommitCallback(self, start_step + expected_steps)
            )
        elif self.guarded_resume_smoke_mode:
            # The resume-init smoke must reach Trainer checkpoint restoration
            # without constructing a network-capable hint client.  A callback
            # stops at on_train_begin, before the first training batch.
            self.hint_client = ForbiddenHintClient()
        elif hinter_backend in {"self_model", "external_model"}:
            # Both backends share the local generation path.  With
            # external_model, SelfModelHintClient loads socratic_reflector_model_path
            # as an independent engine and generates the guidance with it
            # instead of with the current student policy.
            self.hint_client = SelfModelHintClient(
                self, max_new_tokens=hint_max_new_tokens,
                scope_classifier_max_new_tokens=self.scope_classifier_max_new_tokens,
                scope_classifier_temperature=self.scope_classifier_temperature,
                temperature=hint_temperature, top_p=hint_top_p, task_type=task_type,
                hint_kind=hint_kind, debug_log_path=hint_debug_log_path,
                leak_fallback_mode=self.socratic_leak_fallback_mode,
                error_quote_fallback_mode=self.socratic_error_quote_fallback_mode,
                error_quote_locator_mode=self.socratic_error_quote_locator_mode,
                reflection_prompt_mode=self.socratic_reflection_prompt_mode,
                error_route_mode=self.socratic_error_route_mode,
                reflection_context_mode=self.socratic_reflection_context_mode,
                reflector_model_path=socratic_reflector_model_path,
                reflector_gpu_memory_utilization=socratic_reflector_gpu_memory_utilization,
                reflector_max_model_len=socratic_reflector_max_model_len,
                reflector_max_num_seqs=socratic_reflector_max_num_seqs,
                reflector_enforce_eager=socratic_reflector_enforce_eager,
                enable_thinking=self.socratic_self_hint_thinking,
            )
        else:
            raise ValueError(
                "hinter_backend must be 'self_model' or 'external_model', "
                f"got {hinter_backend!r}"
            )

        self.hint_log_path = Path(hint_log_path) if hint_log_path else None
        if self.hint_log_path and self.accelerator.is_main_process:
            self.hint_log_path.parent.mkdir(parents=True, exist_ok=True)

        trace_enabled = os.environ.get("SOCRATIC_SEMANTIC_TRACE_ENABLED", "1") != "0"
        trace_path = os.environ.get("SOCRATIC_SEMANTIC_TRACE_LOG_PATH")
        if trace_enabled and not trace_path and self.hint_log_path:
            trace_path = str(self.hint_log_path.with_name("semantic_trace.jsonl"))
        self.semantic_trace_log_path = Path(trace_path) if trace_path else None
        self.semantic_trace_first_steps = max(
            0, int(os.environ.get("SOCRATIC_SEMANTIC_TRACE_FIRST_STEPS", "2"))
        )
        self.semantic_trace_sample_rate = min(
            1.0,
            max(0.0, float(os.environ.get("SOCRATIC_SEMANTIC_TRACE_SAMPLE_RATE", "0.005"))),
        )
        self.semantic_trace_max_rows = max(
            0, int(os.environ.get("SOCRATIC_SEMANTIC_TRACE_MAX_ROWS", "256"))
        )
        self.semantic_trace_text_limit = max(
            0, int(os.environ.get("SOCRATIC_SEMANTIC_TRACE_TEXT_LIMIT", "4000"))
        )
        self.semantic_trace_stdout_rows = max(
            0, int(os.environ.get("SOCRATIC_SEMANTIC_TRACE_STDOUT_ROWS", "2"))
        )
        self.semantic_trace_stdout_chars = max(
            0, int(os.environ.get("SOCRATIC_SEMANTIC_TRACE_STDOUT_CHARS", "240"))
        )
        self._semantic_trace_rows_written = 0
        if self.semantic_trace_log_path and self.accelerator.is_main_process:
            self.semantic_trace_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _prepare_dataset(
        self,
        dataset,
        processing_class,
        args,
        packing,
        formatting_func,
        dataset_name,
    ):
        """Preserve raw Socratic columns when TRL enables Liger kernels.

        TRL 0.26's ``SFTTrainer._prepare_dataset`` selects only token columns
        whenever ``use_liger_kernel`` is true.  This trainer intentionally uses
        a custom collator and dummy ``input_ids`` while the real on-policy
        prompts are assembled inside ``training_step``.  Dropping ``problem``,
        ``gold``, ``reference_solution`` and ``sample_id`` therefore turns the
        batch into empty examples before any Liger model kernel runs.

        The non-Liger preparation path already preserves those raw columns.
        Temporarily disabling only the preparation-time Liger branch keeps that
        established dataset behavior; the argument is restored before
        ``transformers.Trainer`` applies Liger to the model instance.
        """
        use_liger_kernel = getattr(args, "use_liger_kernel", False)
        if not use_liger_kernel:
            return super()._prepare_dataset(
                dataset,
                processing_class,
                args,
                packing,
                formatting_func,
                dataset_name,
            )

        args.use_liger_kernel = False
        try:
            prepared = super()._prepare_dataset(
                dataset,
                processing_class,
                args,
                packing,
                formatting_func,
                dataset_name,
            )
        finally:
            args.use_liger_kernel = use_liger_kernel

        self._validate_prepared_dataset_columns(prepared, dataset_name)
        return prepared

    @staticmethod
    def _validate_prepared_dataset_columns(dataset, dataset_name: str) -> None:
        """Fail before rollout if TRL removed columns required by our collator."""
        required = {
            "sample_id",
            "problem",
            "gold",
            "reference_solution",
            "task_type",
            "input_ids",
            "text",
        }
        column_names = getattr(dataset, "column_names", None)
        if isinstance(column_names, dict):
            missing_by_split = {
                split: sorted(required - set(columns))
                for split, columns in column_names.items()
                if not required.issubset(set(columns))
            }
            if missing_by_split:
                raise ValueError(
                    "Socratic raw-column integrity check failed after TRL dataset "
                    f"preparation for {dataset_name!r}: {missing_by_split}. "
                    "The custom collator cannot reconstruct on-policy examples."
                )
            return
        missing = sorted(required - set(column_names or []))
        if missing:
            raise ValueError(
                "Socratic raw-column integrity check failed after TRL dataset "
                f"preparation for {dataset_name!r}; missing={missing}. "
                "This commonly indicates Liger-triggered select_columns() pruning."
            )

    # -- keep raw text columns through SFTTrainer's column pruning --
    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        for col in ("problem", "question", "gold", "answer", "ground_truth",
                    "reference_solution", "teacher_context", "task_type",
                    "sample_id", "id", "input_ids", "text"):
            if self._signature_columns is not None and col not in self._signature_columns:
                self._signature_columns.append(col)

    # -- vLLM text-in/text-out generation (colocate, TP=1) --
    def _vllm_generate(self, prompt_texts, max_new_tokens, temperature, top_p):
        top_k = self.args.top_k if getattr(self.args, "top_k", None) else -1
        params = SamplingParams(
            n=1, temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_new_tokens,
        )
        outputs = self.vllm_engine.generate(list(prompt_texts), sampling_params=params, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]

    def _run_repair_with_vllm_lifecycle(self, inputs):
        # Level-2 sleep discards vLLM parameters. Reload current student
        # weights first, map KV once, keep the engine awake for all repair
        # generations, then release it before the training forward pass.
        self._move_model_to_vllm()
        self._wake_vllm_if_needed()
        try:
            return self._run_repair(inputs)
        finally:
            self._sleep_vllm_if_needed()

    def _student_generate(self, prompt_texts):
        if self.trace_replay_mode:
            raise RuntimeError("trace-fed replay forbids student generation")
        return self._vllm_generate(
            prompt_texts, self.args.max_completion_length, self.temperature, self.top_p,
        )

    def _completion_hit_token_limit(self, completion: str) -> bool:
        """Detect generation-limit truncation from the emitted completion alone."""
        token_ids = self.processing_class(
            str(completion or ""),
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )["input_ids"]
        return len(token_ids) >= max(1, int(self.args.max_completion_length) - 1)

    def _run_repair(self, inputs):
        self._validate_raw_batch(inputs)
        problems = [str(x) for x in inputs["problem_texts"]]
        golds = [str(x) for x in inputs["gold_texts"]]
        teacher_contexts = [str(x) if x is not None else "" for x in inputs.get("reference_texts", [""] * len(problems))]
        repair_prompt_token_limit = None
        if self.socratic_repair_selection_mode in {
            "reference_scheme3bv2_replacement",
        }:
            repair_prompt_token_limit = int(self.args.max_length) - int(
                self.args.max_completion_length
            )
            if repair_prompt_token_limit < 1:
                raise ValueError(
                    "max_length must exceed max_completion_length for repair generation"
                )
        loop = SocraticRepairLoop(
            renderers=self.renderers,
            generate=self._student_generate,
            verify=self.verify_fn,
            make_hints=self.hint_client.make_hints,
            max_rounds=self.socratic_max_rounds,
            distill_mode=self.socratic_distill_mode,
            vanilla_baseline=self.socratic_vanilla_baseline,
            wrong_privilege="key_idea" if self.socratic_privilege_mode == "rosd_key_idea" else "hint",
            wrong_pair_student_context=self.socratic_wrong_pair_student_context,
            key_idea_privilege_context=self.socratic_key_idea_privilege_context,
            loss_mask_mode=self.socratic_loss_mask_mode,
            local_window_after_tokens=self.socratic_local_window_after_tokens,
            local_window_max_tokens=self.socratic_local_window_max_tokens,
            local_window_end_mode=self.socratic_local_window_end_mode,
            repair_prompt_token_limit=repair_prompt_token_limit,
            is_completion_truncated=self._completion_hit_token_limit,
            terminal_retry=self.socratic_terminal_retry,
        )
        sample_ids = inputs.get("sample_ids")
        if self.socratic_repair_selection_mode == "reference_scheme3bv2_replacement":
            return loop.run_reference_scheme3bv2_replacement(
                problems,
                golds,
                teacher_contexts=teacher_contexts,
                sample_ids=sample_ids,
            )
        return loop.run(
            problems,
            golds,
            teacher_contexts=teacher_contexts,
            sample_ids=sample_ids,
        )

    @staticmethod
    def _validate_raw_batch(inputs: dict[str, Any]) -> None:
        """Reject collator fallbacks that would silently train on empty examples."""
        required = ("problem_texts", "gold_texts", "sample_ids")
        missing = [name for name in required if name not in inputs]
        if missing:
            raise ValueError(
                "Socratic raw-batch integrity check failed; missing collator fields="
                f"{missing}. Check TRL column pruning and the custom data collator."
            )
        lengths = {name: len(inputs[name]) for name in required}
        if len(set(lengths.values())) != 1 or not next(iter(lengths.values()), 0):
            raise ValueError(
                f"Socratic raw-batch integrity check failed; invalid lengths={lengths}."
            )
        for name in required:
            bad = [idx for idx, value in enumerate(inputs[name]) if not str(value).strip()]
            if bad:
                raise ValueError(
                    "Socratic raw-batch integrity check failed; "
                    f"{name} has blank values at local indices={bad[:8]}. "
                    "This is consistent with raw dataset columns being removed."
                )

    def _wrong_pair_quality_reasons(self, pair) -> list[str]:
        """Return reasons a wrong key-idea/hint pair is too noisy for loss."""
        reasons: list[str] = []
        meta = pair.hint_meta or {}
        hint = str(pair.hint or "").strip()
        if len(hint) < 24:
            reasons.append("short_hint")
        fallback = meta.get("fallback_reason")
        if fallback:
            reasons.append(f"hint_fallback:{fallback}")
        if meta.get("leaked"):
            reasons.append("leaked_hint")
        if meta.get("loss_mask_fallback"):
            reasons.append(f"loss_mask_fallback:{meta.get('loss_mask_fallback')}")
        if (
            self.task_type == "math"
            and self.math_verifier.extract_boxed_answer(pair.completion) is None
        ):
            reasons.append("no_boxed_answer")
        return reasons

    def _mark_filter_meta(self, pair, *, selected: bool, reason: str, quality_reasons: list[str] | None = None) -> None:
        meta = dict(pair.hint_meta or {})
        meta["wrong_pair_filter"] = {
            "mode": self.socratic_wrong_pair_filter_mode,
            "selected": bool(selected),
            "reason": reason,
            "quality_reasons": list(quality_reasons or []),
        }
        pair.hint_meta = meta
        pair.include_in_loss = bool(selected)

    def _apply_wrong_pair_filter(self, pairs):
        if self.socratic_wrong_pair_filter_mode == "none":
            return pairs
        if self.socratic_wrong_pair_filter_mode == "success_gated_wrong2":
            return apply_success_gated_wrong2_filter(
                pairs,
                quality_reasons=self._wrong_pair_quality_reasons,
                mode=self.socratic_wrong_pair_filter_mode,
            )

        by_index: dict[int, list] = {}
        for pair in pairs:
            by_index.setdefault(pair.index, []).append(pair)

        for group in by_index.values():
            group.sort(key=lambda p: p.round)
            for pair in group:
                pair.include_in_loss = False

            selected = None
            selected_reason = ""
            selected_quality: list[str] = []

            correct_pairs = [p for p in group if p.correct and p.privilege == "solution"]
            if correct_pairs:
                selected = correct_pairs[-1]
                selected_reason = "correct_solution"
            else:
                solution_pairs = [p for p in group if p.privilege == "solution"]
                if solution_pairs:
                    selected = solution_pairs[-1]
                    selected_reason = "solution_privilege_fallback"
                else:
                    quality_by_pair = {id(p): self._wrong_pair_quality_reasons(p) for p in group}
                    r2_candidates = [
                        p for p in reversed(group)
                        if p.round >= 2 and p.privilege in {"key_idea", "hint"} and not quality_by_pair[id(p)]
                    ]
                    if r2_candidates:
                        selected = r2_candidates[0]
                        selected_reason = "r2_quality_pass"
                    else:
                        earlier_candidates = [
                            p for p in reversed(group)
                            if p.privilege in {"key_idea", "hint"} and not quality_by_pair[id(p)]
                        ]
                        if earlier_candidates:
                            selected = earlier_candidates[0]
                            selected_reason = "earlier_quality_pass"
                        elif group:
                            selected = group[0]
                            selected_reason = "fallback_first_pair"
                            selected_quality = quality_by_pair.get(id(selected), [])

            for pair in group:
                if pair is selected:
                    self._mark_filter_meta(pair, selected=True, reason=selected_reason, quality_reasons=selected_quality)
                elif pair.privilege in {"key_idea", "hint"} and not pair.correct:
                    self._mark_filter_meta(
                        pair,
                        selected=False,
                        reason="not_selected",
                        quality_reasons=self._wrong_pair_quality_reasons(pair),
                    )
                else:
                    pair.include_in_loss = False
        return pairs

    def _apply_supervision_policy(self, pairs):
        """Apply loss eligibility without changing rollout identities.

        Error-only formal runs keep one generated pair per input.  Correct or
        solution-privileged pairs are retained as structural DDP placeholders
        but are excluded from the distillation loss.  Scrubbing the teacher
        prompt prevents reference-solution text from reaching either the
        forward path or semantic logs.
        """
        return apply_supervision_policy(
            pairs,
            supervision_scope=self.socratic_supervision_scope,
            solution_privilege_mode=self.socratic_solution_privilege_mode,
        )

    def _apply_round_weight_policy(self, pairs) -> dict[str, Any]:
        """Apply the opt-in outcome weighting and preserve global loss mass."""
        if self.socratic_round_weight_mode == "uniform":
            return {
                "mode": "uniform",
                "global_scale": 1.0,
                "outcome_counts": {},
            }

        if self.socratic_round_weight_mode == "simple_outcome_2_1_half":
            local = assign_simple_outcome_raw_weights(pairs)
        elif self.socratic_round_weight_mode == "exp_decay_g08_terminal_half_postnorm":
            local = assign_exponential_decay_round_weights(pairs)
        elif self.socratic_round_weight_mode == "trajectory_outcome_alpha08_rho15":
            local = assign_trajectory_outcome_round_weights(
                pairs,
                alpha=self.socratic_round_weight_alpha,
                rho=self.socratic_round_weight_rho,
            )
        else:  # pragma: no cover - guarded by constructor validation
            raise ValueError(
                f"unsupported round weight mode: {self.socratic_round_weight_mode}"
            )
        if self.socratic_round_weight_mode == "trajectory_outcome_alpha08_rho15":
            return {
                "mode": self.socratic_round_weight_mode,
                "global_baseline_mass": float(local["baseline_mass"]),
                "global_raw_mass": float(local["raw_mass"]),
                "global_scale": 1.0,
                "outcome_counts": local["outcome_counts"],
                "alpha": self.socratic_round_weight_alpha,
                "rho": self.socratic_round_weight_rho,
            }
        masses = torch.tensor(
            [local["baseline_mass"], local["raw_mass"]],
            dtype=torch.float64,
            device=self.accelerator.device,
        )
        global_masses = self.accelerator.reduce(masses, reduction="sum")
        global_baseline = float(global_masses[0].item())
        global_raw = float(global_masses[1].item())
        global_scale = global_baseline / global_raw if global_raw > 0.0 else 1.0

        for pair in pairs:
            pair.supervision_weight = float(pair.supervision_weight) * global_scale
            meta = dict(pair.hint_meta or {})
            meta.update(
                {
                    "round_weight_global_baseline_mass": global_baseline,
                    "round_weight_global_raw_mass": global_raw,
                    "round_weight_global_scale": global_scale,
                    "normalized_round_weight": float(pair.supervision_weight),
                }
            )
            pair.hint_meta = meta
        return {
            "mode": self.socratic_round_weight_mode,
            "global_baseline_mass": global_baseline,
            "global_raw_mass": global_raw,
            "global_scale": global_scale,
            "outcome_counts": local["outcome_counts"],
        }

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        """Scale the normal OPSD loss without skipping its model forwards.

        A mixed active/zero DDP microbatch must execute the same student and
        teacher forwards on every rank.  Skipping the forward only on excluded
        ranks leaves DDP reducer state inconsistent.  The excluded teacher
        prompt has already been scrubbed by the supervision policy, so this
        zero-weight forward cannot consume reference-solution text.
        """
        result = super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
        weight = float(getattr(self, "_socratic_current_supervision_weight", 1.0))
        if return_outputs:
            loss, outputs = result
            return loss * weight, outputs
        return result * weight

    def _annotate_loss_token_bounds(self, pairs) -> None:
        """Resolve character anchors and routed windows to auditable token bounds."""
        tok = self.processing_class

        def token_len(text: str) -> int:
            return len(tok(
                text,
                padding=False,
                truncation=True,
                max_length=self.args.max_completion_length,
                add_special_tokens=False,
            )["input_ids"])

        for pair in pairs:
            completion = str(pair.completion or "")
            completion_tokens = token_len(completion)
            start_char = getattr(pair, "loss_mask_start_char", None)
            end_char = getattr(pair, "loss_mask_end_char", None)
            alignment_method = "full_completion"
            start_token = 0
            quote_end_token = completion_tokens
            if start_char is not None:
                mapped = None
                offsets_available = False
                offsets = None
                try:
                    encoded = tok(
                        completion,
                        padding=False,
                        truncation=True,
                        max_length=self.args.max_completion_length,
                        add_special_tokens=False,
                        return_offsets_mapping=True,
                    )
                    offsets = encoded.get("offset_mapping")
                    if offsets is not None:
                        offsets_available = True
                        mapped = char_span_to_token_span(
                            offsets,
                            max(0, int(start_char)),
                            max(0, int(end_char)) if end_char is not None else len(completion),
                        )
                except (NotImplementedError, TypeError, ValueError):
                    mapped = None
                if mapped is not None:
                    start_token = mapped[0]
                    quote_end_token = mapped[1] if end_char is not None else completion_tokens
                    alignment_method = "offset_mapping"
                elif offsets_available:
                    raise ValueError(
                        "loss character span does not overlap any retained completion token"
                    )
                else:
                    # Compatibility path for slow tokenizers without offsets.
                    start_token = token_len(completion[:max(0, int(start_char))])
                    quote_end_token = (
                        token_len(completion[:max(0, int(end_char))])
                        if end_char is not None
                        else completion_tokens
                    )
                    alignment_method = "prefix_retokenize_fallback"
            start_token, end_token = bounded_loss_token_span(
                start_token,
                quote_end_token,
                completion_tokens,
                window_after_tokens=getattr(pair, "loss_mask_window_after_tokens", None),
                window_max_tokens=getattr(pair, "loss_mask_window_max_tokens", None),
                window_end_mode=getattr(pair, "loss_mask_window_end_mode", None) or "fixed",
                completion=completion,
                offsets=offsets if start_char is not None else None,
            )
            valid_tokens = max(0, end_token - start_token)
            pair.loss_mask_start_token = start_token
            pair.loss_mask_end_token = end_token
            pair.valid_label_tokens = valid_tokens
            if pair.hint_meta is not None:
                meta = dict(pair.hint_meta)
                meta["loss_mask_start_token"] = start_token
                meta["loss_mask_end_token"] = end_token
                meta["completion_tokens"] = completion_tokens
                meta["valid_label_tokens"] = valid_tokens
                meta["loss_token_alignment"] = alignment_method
                meta["loss_mask_window_end_mode"] = (
                    getattr(pair, "loss_mask_window_end_mode", None) or "fixed"
                )
                pair.hint_meta = meta
            if pair.include_in_loss and valid_tokens <= 0:
                raise ValueError(
                    "included Socratic pair has no valid label tokens "
                    f"(route={getattr(pair, 'hint_meta', None)})"
                )

    def _annotate_endpoint_blocks(self, pairs) -> None:
        """Attach the blank-line block partition of each pre-masked region.

        Runs after ``_annotate_loss_token_bounds`` so the region is already the
        auditable ``[loss_mask_start_token, loss_mask_end_token)`` span.  The
        block partition is a property of the response text alone, so it is fixed
        here; only the endpoint choice needs the scored distributions and is
        therefore deferred to the loss.
        """
        if self.socratic_endpoint_mode != "cumulative_correction_coverage":
            return
        tok = self.processing_class
        for pair in pairs:
            pair.endpoint_plan = None
            if not pair.include_in_loss:
                continue
            start_token = getattr(pair, "loss_mask_start_token", None)
            end_token = getattr(pair, "loss_mask_end_token", None)
            if start_token is None or end_token is None:
                continue
            completion = str(pair.completion or "")
            reason = None
            plan = None
            try:
                encoded = tok(
                    completion,
                    padding=False,
                    truncation=True,
                    max_length=self.args.max_completion_length,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                offsets = encoded.get("offset_mapping")
                if offsets is None:
                    reason = "tokenizer_without_offsets"
                else:
                    starts = blank_line_block_starts(
                        completion,
                        offsets,
                        start_token=int(start_token),
                        total_tokens=int(end_token),
                    )
                    plan = {
                        "mode": self.socratic_endpoint_mode,
                        "epsilon": self.socratic_endpoint_epsilon,
                        "region_start_token": int(start_token),
                        "region_end_token": int(end_token),
                        "completion_tokens": len(encoded["input_ids"]),
                        "block_start_tokens": starts,
                        "blocks": len(starts),
                        "observed_token_ids": list(encoded["input_ids"]),
                    }
            except (NotImplementedError, TypeError, ValueError) as exc:
                reason = f"block_partition_failed: {exc}"
            pair.endpoint_plan = plan
            if pair.hint_meta is not None:
                meta = dict(pair.hint_meta)
                meta["endpoint_mode"] = self.socratic_endpoint_mode
                meta["endpoint_epsilon"] = self.socratic_endpoint_epsilon
                meta["endpoint_blocks"] = None if plan is None else plan["blocks"]
                if reason is not None:
                    # Fail open: keep the full suffix and say why, rather than
                    # inventing a truncation from an unusable partition.
                    meta["endpoint_fallback_reason"] = reason
                pair.hint_meta = meta

    def generalized_jsd_loss(
        self,
        student_logits,
        teacher_logits,
        labels=None,
        beta=0.5,
        temperature=1.0,
        reduction="batchmean",
        logits_are_probs=False,
        top_k=None,
        token_clip=None,
        sequence_chunk_size=None,
    ):
        """OPSD's JSD, optionally truncated by the coverage endpoint.

        The endpoint needs the teacher and student distributions over the whole
        error suffix, which this call already receives.  Deriving it here costs
        no extra model forward; the resulting mask is detached, so gradients are
        exactly OPSD's over the retained region.
        """
        plan = self._socratic_current_endpoint_plan
        self._socratic_last_endpoint_record = None
        if (
            self.socratic_endpoint_mode != "cumulative_correction_coverage"
            or plan is None
            or reduction != "batchmean"
            or logits_are_probs
            or labels is None
        ):
            return OPSDTrainer.generalized_jsd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                beta=beta,
                temperature=temperature,
                reduction=reduction,
                logits_are_probs=logits_are_probs,
                top_k=top_k,
                token_clip=token_clip,
                sequence_chunk_size=sequence_chunk_size,
            )
        if top_k:
            # Also guarded at construction.  Ignoring it here would redefine the
            # corrective mass the endpoint is selected on, so fail closed.
            raise ValueError("coverage endpoint is incompatible with a top_k restriction")
        return self._coverage_endpoint_jsd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            plan=plan,
            beta=beta,
            temperature=temperature,
            token_clip=token_clip,
            sequence_chunk_size=sequence_chunk_size,
        )

    def _coverage_endpoint_jsd_loss(
        self,
        *,
        student_logits,
        teacher_logits,
        labels,
        plan,
        beta,
        temperature,
        token_clip,
        sequence_chunk_size,
    ):
        if student_logits.shape != teacher_logits.shape:
            raise ValueError("coverage endpoint requires aligned student/teacher logits")
        if student_logits.ndim != 3 or student_logits.shape[0] != 1:
            raise ValueError(
                "coverage endpoint expects one pair per micro-step "
                f"(got batch {tuple(student_logits.shape)})"
            )
        sequence_length = int(student_logits.shape[1])
        if int(plan["completion_tokens"]) != sequence_length:
            raise ValueError(
                "endpoint plan token count does not match the assembled batch "
                f"({plan['completion_tokens']} vs {sequence_length})"
            )
        region_start = int(plan["region_start_token"])
        region_end = int(plan["region_end_token"])
        mask = labels[0] != -100
        expected = torch.zeros_like(mask)
        expected[region_start:region_end] = True
        if not torch.equal(mask, expected):
            raise ValueError("coverage endpoint requires the full-suffix pre-mask")
        observed = torch.tensor(
            plan["observed_token_ids"], device=student_logits.device, dtype=torch.long
        )

        chunk_size = int(sequence_chunk_size or 0) or sequence_length
        divergence_chunks = []
        correction_chunks = []
        for start in range(0, sequence_length, chunk_size):
            end = min(sequence_length, start + chunk_size)
            student_chunk = student_logits[:, start:end, :]
            teacher_chunk = teacher_logits[:, start:end, :]

            correction_chunks.append(
                per_token_correction_mass(
                    student_chunk, teacher_chunk, observed[start:end], temperature
                )
            )

            def per_token(current_student, current_teacher):
                values = _elementwise_jsd(
                    current_student,
                    current_teacher,
                    beta=beta,
                    temperature=temperature,
                    logits_are_probs=False,
                    top_k=None,
                )
                if token_clip is not None:
                    values = values.clamp(max=token_clip)
                return values.sum(dim=-1)

            if student_chunk.requires_grad or teacher_chunk.requires_grad:
                # Same reason as the upstream chunked reduction: checkpointing
                # keeps only the chunk inputs alive instead of every chunk's
                # elementwise divergence.  Retaining the per-token vector rather
                # than its sum costs one length-``chunk`` tensor and is what lets
                # the endpoint mask be applied after the fact.
                chunk_values = checkpoint(
                    per_token, student_chunk, teacher_chunk, use_reentrant=False
                )
            else:
                chunk_values = per_token(student_chunk, teacher_chunk)
            divergence_chunks.append(chunk_values[0])

        loss, record = coverage_masked_mean(
            torch.cat(divergence_chunks),
            torch.cat(correction_chunks),
            block_start_tokens=plan["block_start_tokens"],
            region_start=region_start,
            region_end=region_end,
            epsilon=self.socratic_endpoint_epsilon,
        )
        self._socratic_last_endpoint_record = {
            "endpoint_mode": self.socratic_endpoint_mode,
            **record,
        }
        return loss

    def _assemble_inputs(self, solved):
        tok = self.processing_class
        device = self.accelerator.device
        pad_id = tok.pad_token_id

        def ids(texts, max_len):
            return tok(list(texts), padding=False, truncation=True, max_length=max_len,
                       add_special_tokens=False)["input_ids"]

        sp = ids([r.student_prompt for r in solved], self.args.max_length)
        tp = ids([r.teacher_prompt for r in solved], self.args.max_length)
        comp = ids([r.completion for r in solved], self.args.max_completion_length)
        teacher_comp_texts = [
            r.teacher_completion if getattr(r, "teacher_completion", None) is not None else r.completion
            for r in solved
        ]
        tcomp = ids(teacher_comp_texts, self.args.max_completion_length)
        comp = [c if len(c) > 0 else [pad_id] for c in comp]
        tcomp = [c if len(c) > 0 else [pad_id] for c in tcomp]

        s_len = [len(x) for x in sp]
        S, T, C = max(s_len), max(len(x) for x in tp), max(len(x) for x in comp)
        TC = max(len(x) for x in tcomp)

        def rp(seq, L):
            return seq + [pad_id] * (L - len(seq))

        student_input_ids = torch.tensor([rp(s, S) + rp(c, C) for s, c in zip(sp, comp)], device=device)
        teacher_input_ids = torch.tensor([rp(t, T) + rp(c, TC) for t, c in zip(tp, tcomp)], device=device)
        student_attn = (student_input_ids != pad_id).long()
        teacher_attn = (teacher_input_ids != pad_id).long()

        labels = student_input_ids.clone()
        labels[:, :S] = -100
        labels[labels == pad_id] = -100
        teacher_completion_offsets = []
        teacher_completion_lengths = []
        for row_idx, r in enumerate(solved):
            annotated_start = getattr(r, "loss_mask_start_token", None)
            if annotated_start is not None:
                prefix_len = min(max(0, int(annotated_start)), C)
            else:
                start_char = getattr(r, "loss_mask_start_char", None)
                if start_char is None or start_char <= 0:
                    prefix_len = 0
                else:
                    prefix = str(r.completion or "")[:start_char]
                    prefix_ids = tok(
                        prefix,
                        padding=False,
                        truncation=True,
                        max_length=self.args.max_completion_length,
                        add_special_tokens=False,
                    )["input_ids"]
                    prefix_len = min(len(prefix_ids), C)
            if prefix_len > 0:
                labels[row_idx, S:S + prefix_len] = -100

            annotated_end = getattr(r, "loss_mask_end_token", None)
            if annotated_end is not None:
                end_len = min(max(0, int(annotated_end)), C)
            else:
                end_char = getattr(r, "loss_mask_end_char", None)
                if end_char is None:
                    end_len = C
                else:
                    end_prefix = str(r.completion or "")[:max(0, int(end_char))]
                    end_ids = tok(
                        end_prefix,
                        padding=False,
                        truncation=True,
                        max_length=self.args.max_completion_length,
                        add_special_tokens=False,
                    )["input_ids"]
                    end_len = min(len(end_ids), C)
            if end_len < C:
                labels[row_idx, S + end_len:S + C] = -100
            if not torch.any(labels[row_idx, S:S + C] != -100):
                raise ValueError("assembled Socratic pair has no valid label tokens")

            teacher_start_char = getattr(r, "teacher_completion_start_char", None)
            if teacher_start_char is None:
                teacher_offset = 0
            else:
                teacher_prefix = str(r.completion or "")[:max(0, int(teacher_start_char))]
                teacher_prefix_ids = tok(
                    teacher_prefix, padding=False, truncation=True,
                    max_length=self.args.max_completion_length, add_special_tokens=False,
                )["input_ids"]
                teacher_offset = min(len(teacher_prefix_ids), C)
            teacher_completion_offsets.append(teacher_offset)
            teacher_completion_lengths.append(min(len(tcomp[row_idx]), TC))

        return {
            "student_input_ids": student_input_ids,
            "student_attention_mask": student_attn,
            "student_prompt_length": S,
            "student_prompt_lengths_per_example": torch.tensor(s_len, device=device),
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attn,
            "teacher_prompt_length": T,
            "teacher_completion_offsets": torch.tensor(teacher_completion_offsets, device=device),
            "teacher_completion_lengths": torch.tensor(teacher_completion_lengths, device=device),
            "labels": labels,
        }

    def _log_hints(self, inputs, pairs):
        if not self.hint_log_path:
            return
        sample_ids = inputs.get("sample_ids") or []
        rank = int(getattr(self.accelerator, "process_index", 0))
        rows = []
        for p in pairs:
            sid = sample_ids[p.index] if p.index < len(sample_ids) else str(p.index)
            problem = str(p.problem or "")
            gold = str(p.gold or "")
            completion = str(p.completion or "")
            hint = str(p.hint or "")
            row = {
                "step": int(self.state.global_step),
                "rank": rank,
                "sample_id": sid,
                "round": p.round,
                "correct": bool(p.correct),
                "privilege": p.privilege,
                "privilege_mode": self.socratic_privilege_mode,
                "include_in_loss": bool(p.include_in_loss),
                "supervision_weight": float(p.supervision_weight),
                "supervision_policy": (p.hint_meta or {}).get("supervision_policy"),
                "gate_decision": (p.hint_meta or {}).get("gate_decision"),
                "gate_terminal_bucket": (
                    ((p.hint_meta or {}).get("gate_decision") or {}).get(
                        "terminal_bucket"
                    )
                    if isinstance((p.hint_meta or {}).get("gate_decision"), dict)
                    else None
                ),
                "gate_primary_reason": (
                    ((p.hint_meta or {}).get("gate_decision") or {}).get(
                        "primary_reason"
                    )
                    if isinstance((p.hint_meta or {}).get("gate_decision"), dict)
                    else None
                ),
                "provider_empty_content_exhausted": bool(
                    (p.hint_meta or {}).get(
                        "provider_empty_content_exhausted", False
                    )
                ),
                "provider_skip_reason": (p.hint_meta or {}).get(
                    "provider_skip_reason"
                ),
                "provider_response_diagnostics": (p.hint_meta or {}).get(
                    "provider_response_diagnostics"
                ),
                "distill_mode": self.socratic_distill_mode,
                "task_type": self.task_type,
                "hint": p.hint,
                "hint_meta": p.hint_meta,
                "problem_chars": len(problem),
                "gold_chars": len(gold),
                "completion_chars": len(completion),
                "hint_chars": len(hint),
                "problem_sha256": hashlib.sha256(problem.encode("utf-8")).hexdigest(),
                "gold_sha256": hashlib.sha256(gold.encode("utf-8")).hexdigest(),
                "completion_sha256": hashlib.sha256(completion.encode("utf-8")).hexdigest(),
                "hint_sha256": hashlib.sha256(hint.encode("utf-8")).hexdigest(),
                # Independent observation of the exact prompt consumed by the
                # teacher path.  The repair loop records the same hash in its
                # metadata; the post-run auditor requires both sources to agree.
                "teacher_prompt_sha256": hashlib.sha256(
                    str(p.teacher_prompt).encode("utf-8")
                ).hexdigest(),
                "error_type": (p.hint_meta or {}).get("error_type"),
                "training_route": (p.hint_meta or {}).get("training_route"),
                "loss_mask_start_token": p.loss_mask_start_token,
                "loss_mask_end_token": p.loss_mask_end_token,
                "valid_label_tokens": p.valid_label_tokens,
            }
            if os.environ.get("SOCRATIC_FULL_TRACE_ENABLED", "0") == "1":
                # Keep one self-contained attribution record per generated
                # trajectory.  This is intentionally untruncated: formal
                # trace profiles opt into the storage cost so post-run audits
                # do not depend on an expiring scratch filesystem or a second
                # dataset join.
                row.update(
                    {
                        "schema": "socratic_hint_trace_v3",
                        "problem": problem,
                        "gold": gold,
                        "student_prompt": str(p.student_prompt or ""),
                        "student_completion": completion,
                        "on_policy_trajectory": completion,
                        "wrong_trajectory": "" if p.correct else completion,
                        "teacher_prompt": str(p.teacher_prompt or ""),
                        "generated_key_idea": str(
                            (p.hint_meta or {}).get("generated_key_idea")
                            or (p.hint_meta or {}).get("key_idea")
                            or p.hint
                            or ""
                        ),
                        "new_correction": str(
                            (p.hint_meta or {}).get("new_correction") or ""
                        ),
                        "repair_signal": str(
                            (p.hint_meta or {}).get("repair_signal")
                            or p.hint
                            or ""
                        ),
                        "repair_signal_kind": str(
                            (p.hint_meta or {}).get("repair_signal_kind") or ""
                        ),
                        "prior_repair_signals": list(
                            (p.hint_meta or {}).get("prior_repair_signals") or []
                        ),
                        "raw_key_idea_response": str(
                            (p.hint_meta or {}).get("raw_reflection") or ""
                        ),
                        "key_idea_request_messages": (
                            (p.hint_meta or {}).get("reflection_messages") or []
                        ),
                    }
                )
            rows.append(row)
        try:
            rows = gather_object(rows)
        except Exception:
            pass
        if not self.accelerator.is_main_process:
            return
        with self.hint_log_path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _record_endpoint_selection(self, pair) -> None:
        """Fold the loss-time endpoint back into the pair and the step ledger."""
        record = self._socratic_last_endpoint_record
        self._socratic_last_endpoint_record = None
        if record is None:
            return
        if pair.hint_meta is not None:
            meta = dict(pair.hint_meta)
            meta.update(record)
            pair.hint_meta = meta
        self._socratic_endpoint_records.append(
            {
                "step": int(self.state.global_step),
                "rank": int(getattr(self.accelerator, "process_index", 0)),
                "round": int(pair.round),
                "supervision_weight": float(pair.supervision_weight),
                **record,
            }
        )

    def _flush_endpoint_ledger(self) -> dict[str, Any]:
        """Aggregate this step's endpoint decisions and append them to disk.

        Kept separate from the hint log because the hint log is written before
        the forward, when the endpoint is not yet defined.  Returns the
        step-level summary used for training statistics.
        """
        records = self._socratic_endpoint_records
        self._socratic_endpoint_records = []
        if self.socratic_endpoint_mode != "cumulative_correction_coverage":
            return {}
        try:
            gathered = gather_object(records)
        except Exception:
            gathered = records
        if self.hint_log_path and self.accelerator.is_main_process and gathered:
            path = self.hint_log_path.with_name("endpoint_ledger.jsonl")
            with path.open("a", encoding="utf-8") as f:
                for row in gathered:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if not gathered:
            return {
                "endpoint_rows": 0,
                "endpoint_truncated_rows": 0,
                "endpoint_supervised_tokens": 0,
                "endpoint_removed_tokens": 0,
            }
        supervised = sum(int(row["endpoint_supervised_tokens"]) for row in gathered)
        removed = sum(int(row["endpoint_removed_tokens"]) for row in gathered)
        return {
            "endpoint_rows": len(gathered),
            "endpoint_truncated_rows": sum(bool(row["endpoint_truncated"]) for row in gathered),
            "endpoint_supervised_tokens": supervised,
            "endpoint_removed_tokens": removed,
            "endpoint_removed_token_fraction": (
                removed / (supervised + removed) if supervised + removed else 0.0
            ),
            "endpoint_max_discarded_correction_fraction": max(
                float(row["endpoint_discarded_correction_fraction"]) for row in gathered
            ),
        }

    def _semantic_trace_selected(self, *, step: int, rank: int, sample_id: str, pair) -> bool:
        if step < self.semantic_trace_first_steps:
            return True
        if self.semantic_trace_sample_rate <= 0:
            return False
        if self.semantic_trace_sample_rate >= 1:
            return True
        key = f"{step}|{rank}|{sample_id}|{pair.index}|{pair.round}"
        value = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:12], 16) / float(16**12)
        return value < self.semantic_trace_sample_rate

    def _semantic_trace_text(self, value: Any) -> dict[str, Any]:
        text = str(value or "")
        limit = self.semantic_trace_text_limit
        clipped = text[:limit] if limit and len(text) > limit else text
        return {
            "text": clipped,
            "chars": len(text),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "truncated": len(clipped) < len(text),
        }

    def _log_semantic_trace(self, inputs, pairs) -> None:
        """Write bounded, deterministic examples linking data, rollout and hint."""
        if not self.semantic_trace_log_path or self.semantic_trace_max_rows <= 0:
            return
        step = int(self.state.global_step)
        rank = int(getattr(self.accelerator, "process_index", 0))
        sample_ids = inputs.get("sample_ids") or []
        rows = []
        for pair in pairs:
            sid = str(sample_ids[pair.index]) if pair.index < len(sample_ids) else str(pair.index)
            if not self._semantic_trace_selected(
                step=step, rank=rank, sample_id=sid, pair=pair
            ):
                continue
            rows.append({
                "schema": "socratic_semantic_trace_v1",
                "step": step,
                "rank": rank,
                "sample_id": sid,
                "pair_index": int(pair.index),
                "round": int(pair.round),
                "correct": bool(pair.correct),
                "privilege": pair.privilege,
                "include_in_loss": bool(pair.include_in_loss),
                "supervision_weight": float(pair.supervision_weight),
                "supervision_policy": (pair.hint_meta or {}).get("supervision_policy"),
                "gate_decision": (pair.hint_meta or {}).get("gate_decision"),
                "training_route": (pair.hint_meta or {}).get("training_route"),
                "provider_empty_content_exhausted": bool(
                    (pair.hint_meta or {}).get(
                        "provider_empty_content_exhausted", False
                    )
                ),
                "provider_skip_reason": (pair.hint_meta or {}).get(
                    "provider_skip_reason"
                ),
                "provider_response_diagnostics": (pair.hint_meta or {}).get(
                    "provider_response_diagnostics"
                ),
                "problem": self._semantic_trace_text(pair.problem),
                "gold": self._semantic_trace_text(pair.gold),
                "student_prompt": self._semantic_trace_text(pair.student_prompt),
                "student_completion": self._semantic_trace_text(pair.completion),
                "hint_or_key_idea": self._semantic_trace_text(pair.hint),
                "teacher_prompt": self._semantic_trace_text(pair.teacher_prompt),
            })
        try:
            rows = gather_object(rows)
        except Exception:
            pass
        if not self.accelerator.is_main_process or not rows:
            return
        remaining = max(0, self.semantic_trace_max_rows - self._semantic_trace_rows_written)
        rows = rows[:remaining]
        if not rows:
            return
        with self.semantic_trace_log_path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._semantic_trace_rows_written += len(rows)

        for row in rows[: self.semantic_trace_stdout_rows]:
            preview = {
                "step": row["step"],
                "rank": row["rank"],
                "sample_id": row["sample_id"],
                "round": row["round"],
                "correct": row["correct"],
                "privilege": row["privilege"],
                "problem": row["problem"]["text"][: self.semantic_trace_stdout_chars],
                "student_completion": row["student_completion"]["text"][: self.semantic_trace_stdout_chars],
                "hint_or_key_idea": row["hint_or_key_idea"]["text"][: self.semantic_trace_stdout_chars],
            }
            print(f"[socratic_trace] {json.dumps(preview, ensure_ascii=False)}", flush=True)

    def training_step(
        self, model: nn.Module, inputs: dict[str, Any], num_items_in_batch: int | None = None
    ) -> torch.Tensor:
        if self.trace_replay_mode:
            self._validate_raw_batch(inputs)
            if self.trace_replay_store is None:
                raise RuntimeError("trace replay mode has no validated store")
            pairs = self.trace_replay_store.pairs_for_batch(
                step=int(self.state.global_step),
                sample_ids=inputs.get("sample_ids", []),
                problems=inputs.get("problem_texts", []),
                golds=inputs.get("gold_texts", []),
            )
        else:
            pairs = self._run_repair_with_vllm_lifecycle(inputs)
            pairs = self._apply_wrong_pair_filter(pairs)
            pairs = self._apply_supervision_policy(pairs)
        round_weight_stats = self._apply_round_weight_policy(pairs)
        if not self.trace_replay_mode:
            self._annotate_loss_token_bounds(pairs)
            self._annotate_endpoint_blocks(pairs)
            self._log_hints(inputs, pairs)
            self._log_semantic_trace(inputs, pairs)
        stats = summarize(pairs)

        included = [p for p in pairs if p.include_in_loss]
        stats["included_pairs"] = len(included)
        stats["excluded_pairs"] = len(pairs) - len(included)
        stats["wrong_pair_filter_mode"] = self.socratic_wrong_pair_filter_mode
        stats["supervision_scope"] = self.socratic_supervision_scope
        stats["solution_privilege_mode"] = self.socratic_solution_privilege_mode
        stats["round_weight_mode"] = round_weight_stats["mode"]
        stats["round_weight_global_scale"] = round_weight_stats["global_scale"]
        stats["round_weight_outcomes"] = round_weight_stats["outcome_counts"]
        stats["zero_supervision_pairs"] = sum(
            1 for pair in pairs if pair.supervision_weight == 0.0
        )
        if self.accelerator.is_main_process and (self.state.global_step % 10 == 0):
            print(f"[socratic] step={self.state.global_step} pairs={len(included)} {stats}", flush=True)
        error_only_mode = (
            self.socratic_supervision_scope != "all"
            or self.socratic_solution_privilege_mode != "allow"
        )
        scheduled = pairs if (self.trace_replay_mode or error_only_mode) else included
        if not scheduled:
            # no distill pairs this micro-batch (only on empty batch); no gradient
            return torch.tensor(0.0, device=self.accelerator.device)

        fixed_stage_multiround = self.socratic_repair_selection_mode in {
            "reference_scheme3bv2_replacement",
        }
        if fixed_stage_multiround:
            stage_count = self.socratic_max_rounds
            if len(scheduled) > stage_count:
                raise RuntimeError(
                    "reference multiround mode exceeded its fixed stage slots; "
                    f"got {len(scheduled)}"
                )
            while len(scheduled) < stage_count:
                padding = copy.deepcopy(scheduled[0])
                padding.teacher_prompt = padding.student_prompt
                padding.include_in_loss = False
                padding.supervision_weight = 0.0
                padding.round = len(scheduled) + 1
                padding.hint_meta = dict(padding.hint_meta or {})
                padding.hint_meta.update({
                    "ddp_stage_padding": True,
                    "training_gate_passed": False,
                })
                scheduled.append(padding)
            if len(scheduled) != stage_count:
                raise RuntimeError("failed to construct fixed DDP stage slots")
            # Traces above retain the complete trajectory.  Fixed-stage DDP still
            # needs one collective slot per round on every rank, but an excluded
            # zero-weight round has no scientific target and need not forward its
            # potentially 8k-token completion.  Replace only the training copy
            # with a one-token, equal-context dummy so it keeps the same graph and
            # collective schedule while contributing exactly zero loss.
            scheduled = [compact_zero_weight_stage_pair(p) for p in scheduled]
        additive_multiround = fixed_stage_multiround

        # Distill each pair as its own batch-of-1 micro-step. With the
        # every-trajectory design, one example yields pairs of very different
        # prompt lengths (short bare round-1 vs long guided round-t). Batching
        # them would right-pad the short prompts and shift the completion's RoPE
        # positions (degrading exactly the hint-privileged pairs) and multiply
        # peak memory by the number of rounds. Per-pair micro-steps keep each
        # student/teacher prompt self-consistent, bound memory to one pair, and
        # give every included pair uniform weight. Skipping OPSDTrainer's own
        # training_step avoids a second (fresh) rollout; we reuse OPSD's
        # compute_loss (uniform JSD) via SFTTrainer.training_step.
        losses = []
        for p in scheduled:
            # The number of Socratic distill pairs can differ by rank, so
            # per-pair collectives would deadlock or cross-match in DDP.
            if self.accelerator.is_main_process:
                self._textual_logs["prompt"].append(p.student_prompt)
                self._textual_logs["completion"].append(p.completion)
            self._socratic_current_supervision_weight = float(p.supervision_weight)
            self._socratic_current_endpoint_plan = (
                getattr(p, "endpoint_plan", None) if p.include_in_loss else None
            )
            try:
                one = self._assemble_inputs([p])
                loss_p = super(OPSDTrainer, self).training_step(
                    model, one, num_items_in_batch
                )
            finally:
                self._socratic_current_supervision_weight = 1.0
                self._socratic_current_endpoint_plan = None
            # The endpoint is only knowable once the scored distributions exist,
            # so it is folded back into the pair after its micro-step instead of
            # being written by the pre-forward annotation.
            self._record_endpoint_selection(p)
            losses.append(loss_p.detach())

        endpoint_stats = self._flush_endpoint_ledger()
        if endpoint_stats and self.accelerator.is_main_process and (
            self.state.global_step % 10 == 0
        ):
            print(
                f"[socratic_endpoint] step={self.state.global_step} {endpoint_stats}",
                flush=True,
            )
        loss = torch.stack(losses).sum() if additive_multiround else torch.stack(losses).mean()
        ga = max(1, int(self.args.gradient_accumulation_steps))
        self._on_policy_loss_total += float(loss)
        self._on_policy_step_equiv += 1.0 / ga
        return loss
