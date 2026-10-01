"""Fail-closed replay of frozen Socratic distillation pairs.

This recovery path is deliberately narrower than normal training.  It consumes
previously persisted training traces and never regenerates student completions,
runs a verifier, constructs a hint, or contacts an external provider.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from air_opd.core.repair import DistillPair


TRACE_REPLAY_SCHEMA = "socratic_trace_fed_replay_v1"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return _sha256_text(payload)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


class ForbiddenHintClient:
    """Sentinel that turns accidental hint-provider access into a hard error."""

    def make_hints(self, _items: Sequence[dict[str, Any]]) -> list[Any]:
        raise RuntimeError("trace-fed replay forbids hint generation/provider access")


class TraceReplayStore:
    """Validated per-rank, per-step source for frozen :class:`DistillPair`s."""

    REQUIRED_FIELDS = {
        "step",
        "rank",
        "sample_id",
        "round",
        "problem",
        "gold",
        "correct",
        "privilege",
        "student_prompt",
        "student_completion",
        "teacher_prompt",
        "hint",
        "hint_meta",
        "include_in_loss",
        "supervision_weight",
        "loss_mask_start_token",
        "loss_mask_end_token",
        "valid_label_tokens",
    }

    def __init__(
        self,
        *,
        trace_path: str | Path,
        receipt_dir: str | Path,
        rank: int,
        world_size: int,
        start_step: int = 0,
        expected_steps: int,
        expected_rows_per_group: int,
        expected_sha256: str | None = None,
    ) -> None:
        self.trace_path = Path(trace_path)
        self.receipt_dir = Path(receipt_dir)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.start_step = int(start_step)
        self.expected_steps = int(expected_steps)
        self.expected_rows_per_group = int(expected_rows_per_group)
        self._pending_step: int | None = None
        self._pending_group_sha256: str | None = None
        self._pending_rows: list[dict[str, Any]] = []
        self._pending_pairs: list[DistillPair] = []

        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("invalid trace replay rank/world_size")
        if self.start_step < 0:
            raise ValueError("trace replay start step must be non-negative")
        if self.expected_steps < 1 or self.expected_rows_per_group < 1:
            raise ValueError("trace replay expected counts must be positive")
        raw = self.trace_path.read_bytes()
        actual_sha = hashlib.sha256(raw).hexdigest()
        if expected_sha256 and actual_sha != expected_sha256:
            raise ValueError(
                f"trace replay SHA mismatch: expected={expected_sha256} actual={actual_sha}"
            )
        self.source_sha256 = actual_sha

        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if not line.strip():
                raise ValueError(f"blank trace replay row at line {line_number}")
            row = json.loads(line)
            missing = sorted(self.REQUIRED_FIELDS - set(row))
            if missing:
                raise ValueError(
                    f"trace replay row {line_number} missing fields: {missing}"
                )
            self._validate_row_hashes(row, line_number)
            rows.append(row)

        expected_total = (
            self.expected_steps * self.world_size * self.expected_rows_per_group
        )
        if len(rows) != expected_total:
            raise ValueError(
                f"trace replay row count mismatch: expected={expected_total} actual={len(rows)}"
            )
        self._rows = rows
        identities = [
            (int(row["step"]), int(row["rank"]), str(row["sample_id"]))
            for row in rows
        ]
        if len(set(identities)) != len(identities):
            raise ValueError("trace replay contains duplicate step/rank/sample_id identities")

        grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = (int(row["step"]), int(row["rank"]))
            grouped[key].append(row)
        observed_order = [(int(row["step"]), int(row["rank"])) for row in rows]
        expected_order = [
            (step, rank)
            for step in range(self.start_step, self.start_step + self.expected_steps)
            for _microstep in range(self.expected_rows_per_group)
            for rank in range(self.world_size)
        ]
        if observed_order != expected_order:
            raise ValueError("trace replay global step/microstep/rank order mismatch")
        expected_keys = {
            (step, rank)
            for step in range(self.start_step, self.start_step + self.expected_steps)
            for rank in range(self.world_size)
        }
        if set(grouped) != expected_keys:
            missing = sorted(expected_keys - set(grouped))
            extra = sorted(set(grouped) - expected_keys)
            raise ValueError(f"trace replay group mismatch: missing={missing} extra={extra}")
        for key, group in grouped.items():
            if len(group) != self.expected_rows_per_group:
                raise ValueError(
                    f"trace replay group {key} size mismatch: "
                    f"expected={self.expected_rows_per_group} actual={len(group)}"
                )
        self._groups = {
            step: grouped[(step, self.rank)]
            for step in range(self.start_step, self.start_step + self.expected_steps)
        }

    @staticmethod
    def _validate_row_hashes(row: dict[str, Any], line_number: int) -> None:
        hash_fields = {
            "problem": "problem_sha256",
            "gold": "gold_sha256",
            "student_completion": "completion_sha256",
            "teacher_prompt": "teacher_prompt_sha256",
        }
        for text_field, hash_field in hash_fields.items():
            expected = row.get(hash_field)
            if expected is not None and _sha256_text(str(row[text_field])) != str(expected):
                raise ValueError(
                    f"trace replay row {line_number} {hash_field} mismatch"
                )
        expected_hint_hash = row.get("hint_sha256")
        if expected_hint_hash is not None:
            hint_text = "" if row.get("hint") is None else str(row["hint"])
            if _sha256_text(hint_text) != str(expected_hint_hash):
                raise ValueError(
                    f"trace replay row {line_number} hint_sha256 mismatch"
                )
        start = int(row["loss_mask_start_token"])
        end = int(row["loss_mask_end_token"])
        valid = int(row["valid_label_tokens"])
        if start < 0 or end < start or valid != end - start:
            raise ValueError(f"trace replay row {line_number} has invalid token mask")
        include = bool(row["include_in_loss"])
        weight = float(row["supervision_weight"])
        if include != (weight > 0.0):
            raise ValueError(
                f"trace replay row {line_number} include/weight mismatch"
            )

    def pairs_for_batch(
        self,
        *,
        step: int,
        sample_ids: Sequence[Any],
        problems: Sequence[Any],
        golds: Sequence[Any],
    ) -> list[DistillPair]:
        step = int(step)
        if self._pending_step is not None and self._pending_step != step:
            raise RuntimeError(
                f"trace replay step {self._pending_step} is pending and uncommitted"
            )
        if step not in self._groups:
            raise ValueError(f"trace replay step out of range: {step}")
        receipt = self._receipt_path(step)
        if receipt.exists():
            raise RuntimeError(f"trace replay step {step} already has a receipt")
        group = self._groups[step]
        cursor = len(self._pending_rows)
        batch_size = len(sample_ids)
        if batch_size < 1 or cursor + batch_size > len(group):
            raise ValueError(
                f"trace replay microbatch overflow at step={step} rank={self.rank}: "
                f"cursor={cursor} batch={batch_size} group={len(group)}"
            )
        expected_batch = group[cursor : cursor + batch_size]
        actual_ids = [str(value) for value in sample_ids]
        actual_problems = [str(value) for value in problems]
        actual_golds = [str(value) for value in golds]
        if not (len(actual_ids) == len(actual_problems) == len(actual_golds) == 1):
            raise ValueError("trace replay requires the frozen singleton microbatch")
        expected_ids = [str(row["sample_id"]) for row in expected_batch]
        expected_problems = [str(row["problem"]) for row in expected_batch]
        expected_golds = [str(row["gold"]) for row in expected_batch]
        if actual_ids != expected_ids:
            raise ValueError(
                f"trace replay sample order mismatch at step={step} rank={self.rank}"
            )
        if actual_problems != expected_problems or actual_golds != expected_golds:
            raise ValueError(
                f"trace replay batch content mismatch at step={step} rank={self.rank}"
            )

        pairs = [self._pair_from_row(index, row) for index, row in enumerate(expected_batch)]
        if self._pending_step is None:
            self._pending_step = step
            self._pending_group_sha256 = _canonical_sha256(group)
        self._pending_rows.extend(expected_batch)
        self._pending_pairs.extend(pairs)
        return pairs

    @staticmethod
    def _pair_from_row(index: int, row: dict[str, Any]) -> DistillPair:
        return DistillPair(
            index=index,
            round=int(row["round"]),
            problem=str(row["problem"]),
            gold=str(row["gold"]),
            correct=bool(row["correct"]),
            privilege=str(row["privilege"]),
            student_prompt=str(row["student_prompt"]),
            teacher_prompt=str(row["teacher_prompt"]),
            completion=str(row["student_completion"]),
            hint=None if row.get("hint") is None else str(row["hint"]),
            hint_meta=row.get("hint_meta"),
            include_in_loss=bool(row["include_in_loss"]),
            supervision_weight=float(row["supervision_weight"]),
            loss_mask_start_token=int(row["loss_mask_start_token"]),
            loss_mask_end_token=int(row["loss_mask_end_token"]),
            valid_label_tokens=int(row["valid_label_tokens"]),
        )

    def commit_step(self, *, step: int, pairs: Sequence[DistillPair]) -> Path:
        step = int(step)
        if self._pending_step != step or self._pending_group_sha256 is None:
            raise RuntimeError(f"trace replay step {step} was not claimed")
        group = self._groups[step]
        if len(self._pending_rows) != len(group):
            raise RuntimeError(
                f"trace replay step {step} incomplete at optimizer boundary: "
                f"consumed={len(self._pending_rows)} expected={len(group)}"
            )
        expected_pair_sha = _canonical_sha256(
            [asdict(self._pair_from_row(0, row)) for row in group]
        )
        committed_pairs = list(pairs) if pairs else self._pending_pairs
        actual_pair_sha = _canonical_sha256([asdict(pair) for pair in committed_pairs])
        if actual_pair_sha != expected_pair_sha:
            raise ValueError(f"trace replay pair mutation detected at step={step}")
        path = self._receipt_path(step)
        if path.exists():
            raise RuntimeError(f"trace replay receipt already exists: {path}")
        _atomic_json(
            path,
            {
                "schema": TRACE_REPLAY_SCHEMA,
                "source_sha256": self.source_sha256,
                "group_sha256": self._pending_group_sha256,
                "pair_sha256": actual_pair_sha,
                "step": step,
                "rank": self.rank,
                "rows": len(group),
                "sample_ids": [str(row["sample_id"]) for row in group],
                "generation_calls": 0,
                "hint_calls": 0,
                "provider_calls": 0,
            },
        )
        self._pending_step = None
        self._pending_group_sha256 = None
        self._pending_rows = []
        self._pending_pairs = []
        return path

    def commit_optimizer_step(self, *, completed_global_step: int) -> Path:
        """Commit the just-finished zero-based source step after optimizer success.

        Transformers exposes the incremented ``state.global_step`` to
        ``on_step_end``.  Source step ``s`` therefore commits when the callback
        observes ``completed_global_step == s + 1``.
        """
        source_step = int(completed_global_step) - 1
        return self.commit_step(step=source_step, pairs=self._pending_pairs)

    def _receipt_path(self, step: int) -> Path:
        return self.receipt_dir / f"rank{self.rank}" / f"step{step:04d}.json"

    def validate_token_masks(self, tokenizer: Any, max_completion_length: int) -> None:
        """Recompute the frozen completion token bounds before any optimizer step."""
        max_completion_length = int(max_completion_length)
        if max_completion_length < 1:
            raise ValueError("trace replay max_completion_length must be positive")
        for row in self._rows:
            token_ids = tokenizer(
                str(row["student_completion"]),
                padding=False,
                truncation=True,
                max_length=max_completion_length,
                add_special_tokens=False,
            )["input_ids"]
            start = int(row["loss_mask_start_token"])
            end = int(row["loss_mask_end_token"])
            valid = int(row["valid_label_tokens"])
            if end > len(token_ids) or valid != max(0, end - start):
                raise ValueError(
                    "trace replay tokenizer mask mismatch for "
                    f"step={row['step']} rank={row['rank']} sample={row['sample_id']}"
                )


def trace_replay_enabled() -> bool:
    return os.environ.get("SOCRATIC_TRACE_REPLAY_MODE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def validate_trace_replay_env() -> None:
    """Validate the zero-provider trace-fed contract before model startup."""
    required = {
        "SOCRATIC_TRACE_REPLAY_PATH": os.environ.get("SOCRATIC_TRACE_REPLAY_PATH", ""),
        "SOCRATIC_TRACE_REPLAY_SHA256": os.environ.get("SOCRATIC_TRACE_REPLAY_SHA256", ""),
        "SOCRATIC_TRACE_REPLAY_RECEIPT_DIR": os.environ.get(
            "SOCRATIC_TRACE_REPLAY_RECEIPT_DIR", ""
        ),
        "SOCRATIC_TRACE_REPLAY_EXPECTED_STEPS": os.environ.get(
            "SOCRATIC_TRACE_REPLAY_EXPECTED_STEPS", ""
        ),
        "SOCRATIC_TRACE_REPLAY_START_STEP": os.environ.get(
            "SOCRATIC_TRACE_REPLAY_START_STEP", "0"
        ),
        "SOCRATIC_TRACE_REPLAY_ROWS_PER_GROUP": os.environ.get(
            "SOCRATIC_TRACE_REPLAY_ROWS_PER_GROUP", ""
        ),
    }
    missing = sorted(name for name, value in required.items() if not str(value).strip())
    if missing:
        raise ValueError(f"trace replay missing required environment: {missing}")
    path = Path(required["SOCRATIC_TRACE_REPLAY_PATH"])
    if not path.is_file():
        raise ValueError(f"trace replay bundle is absent: {path}")
    forbidden = [
        "SOCRATIC_TEACHER_BASE_URL",
        "SOCRATIC_TEACHER_API_KEY",
        "SOCRATIC_TEACHER_HOST_IP",
        "SOCRATIC_TEACHER_HOST_PORT",
        "SOCRATIC_EXTERNAL_CACHE_BUNDLE",
        "SOCRATIC_TEACHER_ENV_FILE",
        "SOCRATIC_API_TUNNEL_KEY",
    ]
    present = sorted(name for name in forbidden if os.environ.get(name))
    if present:
        raise ValueError(f"trace replay forbids provider/tunnel environment: {present}")
