"""Atomic step-checkpoint validation and retention for preemptible training."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

SCHEMA = "socratic_step_checkpoint_v1"
MARKER = "socratic_step_checkpoint.json"
_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def checkpoint_step(path: Path) -> int:
    match = _CHECKPOINT_RE.fullmatch(path.name)
    if not match:
        raise ValueError(f"invalid checkpoint directory name: {path}")
    return int(match.group(1))


def required_files(world_size: int) -> tuple[str, ...]:
    return (
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
        *(f"rng_state_{rank}.pth" for rank in range(world_size)),
    )


def finalize_checkpoint(
    path: str | Path,
    *,
    world_size: int,
    milestone_interval: int,
) -> dict[str, Any]:
    checkpoint = Path(path)
    step = checkpoint_step(checkpoint)
    files: dict[str, dict[str, Any]] = {}
    for name in required_files(world_size):
        item = checkpoint / name
        if not item.is_file() or item.stat().st_size <= 0:
            raise ValueError(f"checkpoint{step} missing or empty file: {name}")
        files[name] = {"bytes": item.stat().st_size, "sha256": _sha256(item)}
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if int(state.get("global_step", -1)) != step:
        raise ValueError("checkpoint trainer_state global_step mismatch")
    payload = {
        "schema": SCHEMA,
        "status": "completed-valid",
        "global_step": step,
        "max_steps": int(state.get("max_steps", -1)),
        "world_size": int(world_size),
        "milestone_interval": int(milestone_interval),
        "milestone": step % int(milestone_interval) == 0,
        "files": files,
    }
    _atomic_json(checkpoint / MARKER, payload)
    return payload


def validate_checkpoint(path: str | Path, *, world_size: int) -> dict[str, Any]:
    checkpoint = Path(path)
    step = checkpoint_step(checkpoint)
    marker = checkpoint / MARKER
    if not marker.is_file():
        raise ValueError(f"checkpoint{step} lacks completion marker")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("status") != "completed-valid":
        raise ValueError("checkpoint completion marker schema/status mismatch")
    if int(payload.get("global_step", -1)) != step:
        raise ValueError("checkpoint completion marker step mismatch")
    if int(payload.get("world_size", -1)) != int(world_size):
        raise ValueError("checkpoint completion marker world-size mismatch")
    expected = set(required_files(world_size))
    files = payload.get("files")
    if not isinstance(files, dict) or set(files) != expected:
        raise ValueError("checkpoint completion marker file set mismatch")
    for name in sorted(expected):
        item = checkpoint / name
        meta = files[name]
        if not item.is_file() or item.stat().st_size != int(meta.get("bytes", -1)):
            raise ValueError(f"checkpoint file size mismatch: {name}")
        if _sha256(item) != meta.get("sha256"):
            raise ValueError(f"checkpoint file hash mismatch: {name}")
    return payload


def latest_complete_checkpoint(run_root: str | Path, *, world_size: int) -> Path:
    root = Path(run_root)
    valid: list[tuple[int, Path]] = []
    for path in root.glob("checkpoint-*"):
        if not path.is_dir() or not _CHECKPOINT_RE.fullmatch(path.name):
            continue
        try:
            validate_checkpoint(path, world_size=world_size)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        valid.append((checkpoint_step(path), path))
    if not valid:
        raise ValueError(f"no completed-valid step checkpoint under {root}")
    return max(valid)[1]


def rotate_checkpoints(
    run_root: str | Path,
    *,
    world_size: int,
    milestone_interval: int,
    rolling_keep: int,
) -> list[Path]:
    root = Path(run_root)
    rolling: list[tuple[int, Path]] = []
    for path in root.glob("checkpoint-*"):
        if not path.is_dir() or not _CHECKPOINT_RE.fullmatch(path.name):
            continue
        try:
            validate_checkpoint(path, world_size=world_size)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        step = checkpoint_step(path)
        if step % milestone_interval != 0:
            rolling.append((step, path))
    removed: list[Path] = []
    for _, path in sorted(rolling)[:-rolling_keep] if rolling_keep else sorted(rolling):
        shutil.rmtree(path)
        removed.append(path)
    return removed
