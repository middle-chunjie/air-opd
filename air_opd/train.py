"""AIR-OPD training entry point. Normalize problems, answers, and reference solutions, then
train with iterative repair and on-policy distillation."""
from __future__ import annotations

import dataclasses
import glob
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoTokenizer, TrainerCallback

from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from trl.experimental.gold import GOLDConfig

from air_opd.collator import SocraticDataCollator
from air_opd.trainer import SocraticOPSDTrainer
from air_opd.core.trace_replay import trace_replay_enabled, validate_trace_replay_env
from air_opd.core.step_checkpoint_callback import StepCheckpointCallback

os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


class GuardedResumeInitCallback(TrainerCallback):
    """Audit checkpoint restoration and stop before any resumed batch runs."""

    def on_train_begin(self, args, state, control, **kwargs):
        expected = int(os.environ["SOCRATIC_RESUME_EXPECTED_GLOBAL_STEP"])
        if int(state.global_step) != expected:
            raise RuntimeError(
                f"guarded resume global_step mismatch: expected={expected} actual={state.global_step}"
            )
        scheduler = kwargs.get("lr_scheduler")
        optimizer = kwargs.get("optimizer")
        if scheduler is None or optimizer is None:
            raise RuntimeError("guarded resume callback requires restored optimizer and scheduler")
        observed_lr = float(scheduler.get_last_lr()[0])
        expected_lr = float(os.environ["SOCRATIC_RESUME_EXPECTED_SCHEDULER_LR"])
        if abs(observed_lr - expected_lr) > max(1e-15, abs(expected_lr) * 1e-9):
            raise RuntimeError(
                f"guarded resume scheduler mismatch: expected={expected_lr} actual={observed_lr}"
            )
        receipt_dir = Path(os.environ["SOCRATIC_GUARDED_RESUME_RECEIPT_DIR"])
        receipt_dir.mkdir(parents=True, exist_ok=True)
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        payload = {
            "schema": "socratic_guarded_resume_init_v1",
            "status": "completed-valid",
            "rank": rank,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "global_step": expected,
            "next_optimizer_step": expected + 1,
            "max_steps": int(state.max_steps),
            "scheduler_last_lr": observed_lr,
            "optimizer_state_entries": len(optimizer.state),
            "generation_calls": 0,
            "hint_calls": 0,
            "provider_calls": 0,
        }
        path = receipt_dir / f"rank{rank}.json"
        tmp = path.with_suffix(f".json.tmp.{os.getpid()}")
        tmp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)
        control.should_training_stop = True
        return control


class TaskSmokeStopCallback(TrainerCallback):
    """Run exactly one real resumed optimizer step under the formal schedule."""

    def on_train_begin(self, args, state, control, **kwargs):
        del args, kwargs
        expected = int(os.environ["SOCRATIC_TASK_SMOKE_START_STEP"])
        if int(state.global_step) != expected:
            raise RuntimeError(
                f"task smoke start mismatch: expected={expected} actual={state.global_step}"
            )
        return control

    def on_step_end(self, args, state, control, **kwargs):
        del args, kwargs
        stop = int(os.environ["SOCRATIC_TASK_SMOKE_STOP_STEP"])
        if int(state.global_step) >= stop:
            control.should_save = True
            control.should_training_stop = True
        return control

try:
    import wandb
except Exception:  # pragma: no cover
    class _NoopWandb:
        @staticmethod
        def init(*a, **k):
            print("wandb not installed; continuing without it")
    wandb = _NoopWandb()


@dataclass
class SocraticScriptArguments(ScriptArguments):
    use_tinker_loss: bool = field(default=False)
    fixed_teacher: bool = field(default=False)
    run_config: str | None = field(default=None)
    presence_penalty: float = field(default=0.0)
    top_k_loss: int = field(default=0)
    jsd_token_clip: float = field(default=0.05)
    jsd_sequence_chunk_size: int = field(
        default=0,
        metadata={
            "help": (
                "If positive, evaluate scalar JSD reductions in sequence chunks of this size. "
                "This is an exact objective-preserving memory bound; 0 keeps the legacy path."
            )
        },
    )
    use_ema_teacher: bool = field(default=False)
    ema_decay: float = field(default=0.999)
    student_thinking: bool = field(default=False)
    teacher_thinking: bool = field(default=True)
    # data
    sciknoweval_root: str | None = field(
        default=None, metadata={"help": "Dir with <domain>/train.parquet (SocraticPO without_solution)."}
    )
    sciknoweval_domains: str = field(default="chemistry", metadata={"help": "comma list or 'all'"})
    dataset_kind: str = field(default="sciknoweval", metadata={"help": "sciknoweval or openthoughts_math"})
    openthoughts_jsonl: str | None = field(default=None, metadata={"help": "OPSD/OpenThoughts train JSONL."})
    max_train_rows: int = field(default=-1)
    # socratic
    socratic_max_rounds: int = field(default=3)
    socratic_distill_mode: str = field(default="guided")
    socratic_privilege_mode: str = field(default="auto")
    hinter_backend: str = field(default="self_model")
    hint_max_new_tokens: int = field(default=320)
    scope_classifier_max_new_tokens: int = field(
        default=32,
        metadata={"help": "Stage-A prefix_reusable classifier token limit."},
    )
    scope_classifier_temperature: float = field(
        default=0.0,
        metadata={"help": "Stage-A prefix_reusable classifier temperature."},
    )
    hint_temperature: float = field(default=0.7)
    hint_top_p: float = field(default=0.9)
    hint_concurrency: int = field(default=8)
    hint_log_path: str | None = field(default=None)
    socratic_leak_fallback_mode: str = field(
        default="safe_hint",
        metadata={"help": "safe_hint, allow_leak, disabled, or solution_on_leak for hint answer-leak behavior."},
    )
    socratic_reflection_context_mode: str = field(
        default="last_wrong",
        metadata={"help": "last_wrong or all_wrong for repair-guidance reflection prompt context."},
    )
    socratic_wrong_pair_student_context: str = field(
        default="on_policy",
        metadata={"help": "on_policy or root_question_after_first for wrong key-idea pairs after round 1."},
    )
    socratic_key_idea_privilege_context: str = field(
        default="problem_only",
        metadata={"help": "problem_only or problem_wrong_attempt for repair-guidance teacher privilege context."},
    )
    socratic_loss_mask_mode: str = field(
        default="full",
        metadata={
            "help": (
                "full, after_error_quote, error_quote_only, error_type_routed, or two_scope_routed "
                "for wrong key-idea distillation labels."
            )
        },
    )
    socratic_error_quote_fallback_mode: str = field(
        default="quote_source_prefix",
        metadata={
            "help": (
                "quote_source_prefix or solution_on_invalid_quote for missing, instruction-like, "
                "or unmatched <error_quote>."
            )
        },
    )
    socratic_error_quote_locator_mode: str = field(
        default="strict",
        metadata={
            "help": (
                "strict, locatable, or usable_for_loss for matching <error_quote> "
                "against the wrong rollout."
            )
        },
    )
    socratic_reflection_prompt_mode: str = field(
        default="legacy_xml",
        metadata={
            "help": (
                "Repair-guidance generation profile used by the released recipe."
            )
        },
    )
    socratic_error_route_mode: str = field(
        default="legacy",
        metadata={
            "help": "Post-reflection routing policy."
        },
    )
    socratic_local_window_after_tokens: int = field(
        default=128,
        metadata={"help": "Tokens after the located local error quote retained for routed loss."},
    )
    socratic_local_window_max_tokens: int = field(
        default=256,
        metadata={"help": "Soft total-token cap for routed local loss; the full quote is always retained."},
    )
    socratic_local_window_end_mode: str = field(
        default="fixed",
        metadata={
            "help": (
                "fixed or next_blank. next_blank extends past the minimum suffix "
                "to the first following blank line, subject to the total-token cap."
            )
        },
    )
    socratic_wrong_pair_filter_mode: str = field(
        default="none",
        metadata={
            "help": (
                "Optional loss-pair selection. The released recipe keeps all validated failed stages."
            )
        },
    )
    socratic_supervision_scope: str = field(
        default="all",
        metadata={
            "help": (
                "all or initial_wrong_only. initial_wrong_only keeps rollout "
                "identities fixed but gives initially correct targets zero loss."
            )
        },
    )
    socratic_solution_privilege_mode: str = field(
        default="allow",
        metadata={
            "help": (
                "allow or exclude. exclude gives every solution-privileged pair "
                "zero loss and scrubs its teacher prompt before logging/forward."
            )
        },
    )
    socratic_repair_selection_mode: str = field(
        default="legacy_every_trajectory",
        metadata={
            "help": (
                "legacy_every_trajectory or reference_adaptive_repair_replacement. "
                "The latter runs the multi-round adaptive repair loop and is "
                "the setting used by the released recipes."
            )
        },
    )
    socratic_round_weight_mode: str = field(
        default="uniform",
        metadata={
            "help": (
                "Stage weighting: uniform or trajectory-normalized temporal and immediate-outcome weighting; the paper uses gamma=0.8 and rho=1.5."
            )
        },
    )
    socratic_round_weight_alpha: float = field(default=0.8)
    socratic_round_weight_rho: float = field(default=1.5)
    socratic_endpoint_mode: str = field(
        default="full_suffix",
        metadata={
            "help": (
                "Supervision endpoint of the repair region. 'full_suffix' keeps "
                "the error suffix (b to the end of the "
                "response). 'cumulative_correction_coverage' ends the region at "
                "the earliest reasoning-block end retaining at least "
                "1-socratic_endpoint_epsilon of the suffix corrective signal."
            )
        },
    )
    socratic_endpoint_epsilon: float = field(
        default=0.10,
        metadata={
            "help": (
                "Fraction of the error-suffix corrective signal the endpoint may "
                "discard. Used only by cumulative_correction_coverage."
            )
        },
    )
    socratic_terminal_retry: bool = field(
        default=False,
        metadata={
            "help": (
                "Sample one extra on-policy retry after the last supervised "
                "stage, used only to define that stage's outcome bonus s. Off by "
                "default, which keeps s=0 for the terminal stage."
            )
        },
    )
    socratic_max_repair_attempts: int = field(
        default=2,
        metadata={"help": "Number of user-feedback repair retries per round (1 or 2)."},
    )
    socratic_reflector_model_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "Optional independent vLLM model path for self-model repair-guidance generation. "
                "If unset, hints are generated by the trainer's existing student vLLM engine."
            )
        },
    )
    socratic_reflector_gpu_memory_utilization: float = field(default=0.18)
    socratic_reflector_max_model_len: int = field(default=12000)
    socratic_reflector_max_num_seqs: int = field(default=16)
    socratic_reflector_enforce_eager: bool = field(default=True)
    socratic_self_hint_thinking: bool | None = field(
        default=None,
        metadata={"help": "Enable thinking tokens during self-model hint generation."},
    )
    task_type: str | None = field(default=None)
    math_verifier_profile: str = field(
        default="legacy",
        metadata={
            "help": (
                "legacy, binary_math_verify_v1, or binary_math_verify_compat_v2. "
                "The binary profiles require pinned math-verify==0.9.0 and "
                "return only correct/incorrect; compat_v2 preserves every "
                "legacy positive and applies enhanced matching only to legacy "
                "negatives."
            )
        },
    )


_DOMAINS = ["chemistry", "physics", "material", "biology"]


def _extract_problem(row: dict) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, list) and prompt:
        for m in reversed(prompt):
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content", ""))
        last = prompt[-1]
        return str(last.get("content", "") if isinstance(last, dict) else last)
    return str(row.get("problem") or row.get("question") or "")


def _extract_gold(row: dict) -> str:
    rm = row.get("reward_model")
    if isinstance(rm, dict) and rm.get("ground_truth") is not None:
        return str(rm["ground_truth"])
    for k in ("ground_truth", "answer", "label", "gold"):
        if row.get(k) is not None:
            return str(row[k])
    return ""


def _normalize(row: dict, idx: int) -> dict:
    problem = _extract_problem(row)
    gold = _extract_gold(row)
    if not problem or not gold:
        raise ValueError(f"row {idx}: missing problem/gold (keys={list(row.keys())})")
    return {
        "sample_id": str(row.get("sample_id") or row.get("id") or idx),
        "problem": problem,
        "gold": gold,
        "reference_solution": "",
        "task_type": "sciknoweval",
        # dummy columns so TRL SFTTrainer treats rows as ready
        "input_ids": [0],
        "text": problem,
    }


def _strip_boxed_prefix(question: str) -> str:
    prefix = r"Return your final response within \boxed{}."
    q = str(question or "").strip()
    if q.startswith(prefix):
        q = q[len(prefix):].strip()
    return q


def _normalize_openthoughts(row: dict, idx: int) -> dict | None:
    problem = str(row.get("problem") or _strip_boxed_prefix(row.get("Question") or "")).strip()
    gold = str(row.get("Answer") or row.get("answer") or "").strip()
    reference = str(row.get("solution") or row.get("COT_Reason") or "").strip()
    if not problem or not gold or not reference:
        return None
    return {
        "sample_id": str(row.get("sample_id") or row.get("id") or f"openthoughts30k-{idx}"),
        "problem": problem,
        "gold": gold,
        "reference_solution": reference,
        "task_type": "math",
        "source": str(row.get("source") or "openthoughts_math_30k_opsd"),
        "input_ids": [0],
        "text": problem,
    }


def build_sciknoweval_dataset(args: SocraticScriptArguments) -> Dataset:
    if not args.sciknoweval_root:
        raise ValueError("--sciknoweval_root is required")
    domains = _DOMAINS if args.sciknoweval_domains.strip() == "all" else [
        d.strip() for d in args.sciknoweval_domains.split(",") if d.strip()
    ]
    parts = []
    for d in domains:
        path = os.path.join(args.sciknoweval_root, d, "train.parquet")
        files = glob.glob(path)
        if not files:
            raise FileNotFoundError(f"no train.parquet for domain {d} at {path}")
        parts.append(load_dataset("parquet", data_files=files)["train"])
    raw = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    rows = []
    for idx, row in enumerate(raw):
        rows.append(_normalize(row, idx))
        if args.max_train_rows > 0 and len(rows) >= args.max_train_rows:
            break
    print(f"[data] domains={domains} normalized_rows={len(rows)}")
    return Dataset.from_list(rows)


def build_openthoughts_math_dataset(args: SocraticScriptArguments) -> Dataset:
    if not args.openthoughts_jsonl:
        raise ValueError("--openthoughts_jsonl is required for dataset_kind=openthoughts_math")
    path = Path(args.openthoughts_jsonl)
    if not path.exists():
        raise FileNotFoundError(f"missing OPSD/OpenThoughts JSONL: {path}")
    rows = []
    skipped = 0
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            normalized = _normalize_openthoughts(row, idx)
            if normalized is None:
                skipped += 1
                continue
            rows.append(normalized)
            if args.max_train_rows > 0 and len(rows) >= args.max_train_rows:
                break
    if not rows:
        raise ValueError(f"no usable rows found in {path}")
    print(f"[data] kind=openthoughts_math path={path} normalized_rows={len(rows)} skipped={skipped}")
    return Dataset.from_list(rows)


def build_dataset(args: SocraticScriptArguments) -> Dataset:
    if args.dataset_kind == "sciknoweval":
        return build_sciknoweval_dataset(args)
    if args.dataset_kind == "openthoughts_math":
        return build_openthoughts_math_dataset(args)
    raise ValueError("dataset_kind must be 'sciknoweval' or 'openthoughts_math'")


def ensure_defaults(training_args: GOLDConfig) -> None:
    defaults = {
        "steps_per_generation": 1, "log_completions": False, "log_completions_steps": 10,
        "wandb_log_unique_prompts": False, "num_completions_to_print": 0, "seq_kd": False,
        "vllm_enable_sleep_mode": False, "vllm_server_host": "0.0.0.0", "vllm_server_port": 8000,
        "vllm_server_timeout": 240.0, "vllm_guided_decoding_regex": None, "vllm_sync_frequency": 1,
        "repetition_penalty": 1.0, "min_p": 0.0,
    }
    for name, value in defaults.items():
        if not hasattr(training_args, name):
            setattr(training_args, name, value)


def disable_deepspeed_probe_if_requested() -> None:
    """Disable the optional DeepSpeed type probe when no DeepSpeed engine is used, avoiding a
    CUDA-toolkit requirement during import."""
    flag = os.environ.get("SOCRATIC_DISABLE_DEEPSPEED_PROBE", "1").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return
    try:
        import accelerate.utils.imports as accel_imports
        import accelerate.utils.other as accel_other
    except Exception:
        return

    accel_imports.is_deepspeed_available = lambda: False
    accel_other.is_deepspeed_available = lambda: False
    print("[env] disabled Accelerate DeepSpeed type probe (SOCRATIC_DISABLE_DEEPSPEED_PROBE=1)", flush=True)


if __name__ == "__main__":
    disable_deepspeed_probe_if_requested()

    parser = TrlParser((SocraticScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    ensure_defaults(training_args)

    guarded_resume_smoke = os.environ.get("SOCRATIC_GUARDED_RESUME_SMOKE", "0") == "1"
    task_smoke = os.environ.get("SOCRATIC_TASK_SMOKE_MODE", "0") == "1"
    if trace_replay_enabled():
        validate_trace_replay_env()
    elif guarded_resume_smoke:
        if not getattr(training_args, "resume_from_checkpoint", None):
            raise ValueError("guarded resume smoke requires resume_from_checkpoint")

    if script_args.run_config and not training_args.output_dir.endswith(script_args.run_config):
        training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    task_type = script_args.task_type or ("math" if script_args.dataset_kind == "openthoughts_math" else "sciknoweval")

    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError("fixed_teacher=True requires use_peft=True.")

    import torch

    dtype = torch.bfloat16
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    distributed_state = getattr(training_args, "distributed_state", None)
    num_processes = int(getattr(distributed_state, "num_processes", 1) or 1)
    distributed_type = str(getattr(distributed_state, "distributed_type", "NO")).upper()
    if num_processes > 1 or ("NO" not in distributed_type and distributed_type):
        # TRL defaults to device_map="auto"; Accelerate rejects that under DDP.
        model_kwargs["device_map"] = None
    training_args.model_init_kwargs = model_kwargs
    training_args.presence_penalty = script_args.presence_penalty

    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=getattr(training_args, "wandb_entity", None),
            project=getattr(training_args, "wandb_project", None),
            name=script_args.run_config,
            config={
                "model": model_args.model_name_or_path,
                "socratic_max_rounds": script_args.socratic_max_rounds,
                "socratic_distill_mode": script_args.socratic_distill_mode,
                "socratic_privilege_mode": script_args.socratic_privilege_mode,
                "socratic_leak_fallback_mode": script_args.socratic_leak_fallback_mode,
                "socratic_reflection_context_mode": script_args.socratic_reflection_context_mode,
                "socratic_wrong_pair_student_context": script_args.socratic_wrong_pair_student_context,
                "socratic_key_idea_privilege_context": script_args.socratic_key_idea_privilege_context,
                "socratic_loss_mask_mode": script_args.socratic_loss_mask_mode,
                "socratic_error_quote_fallback_mode": script_args.socratic_error_quote_fallback_mode,
                "socratic_error_quote_locator_mode": script_args.socratic_error_quote_locator_mode,
                "socratic_reflection_prompt_mode": script_args.socratic_reflection_prompt_mode,
                "scope_classifier_max_new_tokens": script_args.scope_classifier_max_new_tokens,
                "scope_classifier_temperature": script_args.scope_classifier_temperature,
                "socratic_error_route_mode": script_args.socratic_error_route_mode,
                "socratic_local_window_after_tokens": script_args.socratic_local_window_after_tokens,
                "socratic_local_window_max_tokens": script_args.socratic_local_window_max_tokens,
                "socratic_local_window_end_mode": script_args.socratic_local_window_end_mode,
                "socratic_wrong_pair_filter_mode": script_args.socratic_wrong_pair_filter_mode,
                "socratic_supervision_scope": script_args.socratic_supervision_scope,
                "socratic_solution_privilege_mode": script_args.socratic_solution_privilege_mode,
                "socratic_repair_selection_mode": script_args.socratic_repair_selection_mode,
                "socratic_round_weight_mode": script_args.socratic_round_weight_mode,
                "socratic_round_weight_alpha": script_args.socratic_round_weight_alpha,
                "socratic_round_weight_rho": script_args.socratic_round_weight_rho,
                "socratic_endpoint_mode": script_args.socratic_endpoint_mode,
                "socratic_endpoint_epsilon": script_args.socratic_endpoint_epsilon,
                "socratic_terminal_retry": script_args.socratic_terminal_retry,
                "socratic_max_repair_attempts": script_args.socratic_max_repair_attempts,
                "jsd_sequence_chunk_size": script_args.jsd_sequence_chunk_size,
                "socratic_reflector_model_path": script_args.socratic_reflector_model_path,
                "socratic_reflector_gpu_memory_utilization": script_args.socratic_reflector_gpu_memory_utilization,
                "socratic_reflector_max_model_len": script_args.socratic_reflector_max_model_len,
                "socratic_reflector_max_num_seqs": script_args.socratic_reflector_max_num_seqs,
                "hinter_backend": script_args.hinter_backend,
                "dataset_kind": script_args.dataset_kind,
                "task_type": task_type,
                "math_verifier_profile": script_args.math_verifier_profile,
                "sciknoweval_domains": script_args.sciknoweval_domains,
            },
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = build_dataset(script_args)
    data_collator = SocraticDataCollator(tokenizer=tokenizer)

    trainer = SocraticOPSDTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        data_collator=data_collator,
        peft_config=get_peft_config(model_args),
        use_thinking_machines_loss=script_args.use_tinker_loss,
        fixed_teacher=script_args.fixed_teacher,
        reason_first=False,
        top_k_loss=script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        jsd_token_clip=script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
        jsd_sequence_chunk_size=(
            script_args.jsd_sequence_chunk_size
            if script_args.jsd_sequence_chunk_size > 0
            else None
        ),
        use_ema_teacher=script_args.use_ema_teacher,
        ema_decay=script_args.ema_decay,
        student_thinking=script_args.student_thinking,
        teacher_thinking=script_args.teacher_thinking,
        socratic_max_rounds=script_args.socratic_max_rounds,
        socratic_distill_mode=script_args.socratic_distill_mode,
        socratic_privilege_mode=script_args.socratic_privilege_mode,
        hinter_backend=script_args.hinter_backend,
        hint_max_new_tokens=script_args.hint_max_new_tokens,
        scope_classifier_max_new_tokens=script_args.scope_classifier_max_new_tokens,
        scope_classifier_temperature=script_args.scope_classifier_temperature,
        hint_temperature=script_args.hint_temperature,
        hint_top_p=script_args.hint_top_p,
        hint_concurrency=script_args.hint_concurrency,
        hint_log_path=script_args.hint_log_path,
        socratic_leak_fallback_mode=script_args.socratic_leak_fallback_mode,
        socratic_reflection_context_mode=script_args.socratic_reflection_context_mode,
        socratic_wrong_pair_student_context=script_args.socratic_wrong_pair_student_context,
        socratic_key_idea_privilege_context=script_args.socratic_key_idea_privilege_context,
        socratic_loss_mask_mode=script_args.socratic_loss_mask_mode,
        socratic_error_quote_fallback_mode=script_args.socratic_error_quote_fallback_mode,
        socratic_error_quote_locator_mode=script_args.socratic_error_quote_locator_mode,
        socratic_reflection_prompt_mode=script_args.socratic_reflection_prompt_mode,
        socratic_error_route_mode=script_args.socratic_error_route_mode,
        socratic_local_window_after_tokens=script_args.socratic_local_window_after_tokens,
        socratic_local_window_max_tokens=script_args.socratic_local_window_max_tokens,
        socratic_local_window_end_mode=script_args.socratic_local_window_end_mode,
        socratic_wrong_pair_filter_mode=script_args.socratic_wrong_pair_filter_mode,
        socratic_supervision_scope=script_args.socratic_supervision_scope,
        socratic_solution_privilege_mode=script_args.socratic_solution_privilege_mode,
        socratic_repair_selection_mode=script_args.socratic_repair_selection_mode,
        socratic_round_weight_mode=script_args.socratic_round_weight_mode,
        socratic_round_weight_alpha=script_args.socratic_round_weight_alpha,
        socratic_round_weight_rho=script_args.socratic_round_weight_rho,
        socratic_endpoint_mode=script_args.socratic_endpoint_mode,
        socratic_endpoint_epsilon=script_args.socratic_endpoint_epsilon,
        socratic_terminal_retry=script_args.socratic_terminal_retry,
        socratic_max_repair_attempts=script_args.socratic_max_repair_attempts,
        socratic_reflector_model_path=script_args.socratic_reflector_model_path,
        socratic_reflector_gpu_memory_utilization=script_args.socratic_reflector_gpu_memory_utilization,
        socratic_reflector_max_model_len=script_args.socratic_reflector_max_model_len,
        socratic_reflector_max_num_seqs=script_args.socratic_reflector_max_num_seqs,
        socratic_reflector_enforce_eager=script_args.socratic_reflector_enforce_eager,
        socratic_self_hint_thinking=script_args.socratic_self_hint_thinking,
        task_type=task_type,
        math_verifier_profile=script_args.math_verifier_profile,
    )

    if guarded_resume_smoke:
        trainer.add_callback(GuardedResumeInitCallback())
    if task_smoke:
        trainer.add_callback(TaskSmokeStopCallback())
    if os.environ.get("SOCRATIC_STEP_CHECKPOINT_ENABLED", "0") == "1":
        trainer.add_callback(
            StepCheckpointCallback(
                trainer,
                milestone_interval=int(
                    os.environ.get("SOCRATIC_CHECKPOINT_MILESTONE_INTERVAL", "25")
                ),
                rolling_keep=int(
                    os.environ.get("SOCRATIC_CHECKPOINT_ROLLING_KEEP", "2")
                ),
            )
        )

    resolved_path = os.environ.get("SOCRATIC_RESOLVED_CONFIG_PATH", "").strip()
    if resolved_path and trainer.accelerator.is_main_process:
        resolved = {
            "script_args": dataclasses.asdict(script_args),
            "training_args": training_args.to_dict(),
            "model_args": dataclasses.asdict(model_args),
            "world_size": int(trainer.accelerator.num_processes),
        }
        path = Path(resolved_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(resolved, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    resume = getattr(training_args, "resume_from_checkpoint", None) or None
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(training_args.output_dir)
