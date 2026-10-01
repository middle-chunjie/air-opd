#!/usr/bin/env python3
"""Resumable GPQA-Diamond Avg@4 evaluation for base and adapted models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from jsonl_utils import read_jsonl_records
from strict_boxed_parser import PARSER_ID, parse_last_boxed_option


SCHEMA = "gpqa_diamond_avg4_qwen_lora_resumable_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stride_index", type=int)
    parser.add_argument("--stride_count", type=int)
    parser.add_argument("--ordinal_file")
    parser.add_argument("--max_tokens", type=int, default=38912)
    parser.add_argument("--max_model_len", type=int, default=40960)
    parser.add_argument("--batch_questions", type=int, default=8)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.88)
    parser.add_argument("--max_num_seqs", type=int, default=16)
    parser.add_argument("--max_num_batched_tokens", type=int)
    parser.add_argument("--eager", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def read_progress(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    records: dict[int, dict] = {}
    for record in read_jsonl_records(path):
        ordinal = int(record["ordinal"])
        if ordinal in records:
            raise ValueError(f"duplicate ordinal in progress: {ordinal}")
        records[ordinal] = record
    return records


def write_progress(path: Path, records: dict[int, dict]) -> None:
    body = "".join(
        json.dumps(records[key], ensure_ascii=False, separators=(",", ":")) + "\n"
        for key in sorted(records)
    )
    atomic_write(path, body)


def build_prompt(row: dict) -> str:
    options = "\n".join(f"{key}) {value}" for key, value in row["options"].items())
    return (
        f'{row["question"]}\n\n{options}\n\n'
        "Solve the problem carefully. Think step by step, then give only the final choice "
        r"in \boxed{A}, \boxed{B}, \boxed{C}, or \boxed{D}."
    )


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    rows = read_jsonl_records(data_path)
    stride_requested = args.stride_index is not None or args.stride_count is not None
    ordinal_requested = args.ordinal_file is not None
    if stride_requested == ordinal_requested:
        raise ValueError("use exactly one of stride selection or --ordinal_file")
    if stride_requested:
        if args.stride_index is None or args.stride_count is None:
            raise ValueError("--stride_index and --stride_count must be used together")
        if args.stride_count <= 0 or not 0 <= args.stride_index < args.stride_count:
            raise ValueError("invalid stride selection")
        selected = [
            (ordinal, row)
            for ordinal, row in enumerate(rows)
            if ordinal % args.stride_count == args.stride_index
        ]
    else:
        ordinal_path = Path(args.ordinal_file)
        ordinals = [int(line) for line in ordinal_path.read_text().splitlines() if line]
        if len(ordinals) != len(set(ordinals)):
            raise ValueError("duplicate ordinal in ordinal file")
        if any(ordinal < 0 or ordinal >= len(rows) for ordinal in ordinals):
            raise ValueError("ordinal outside dataset range")
        selected = [(ordinal, rows[ordinal]) for ordinal in ordinals]
    if not selected:
        raise ValueError("no GPQA rows selected")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.jsonl"
    protocol_path = out_dir / "protocol.json"
    protocol = {
        "schema": SCHEMA,
        "parser": PARSER_ID,
        "base": args.base,
        "adapter": args.adapter,
        "data": str(data_path),
        "data_sha256": sha256_file(data_path),
        "question_ordinals": [ordinal for ordinal, _ in selected],
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": -1,
        "n": 4,
        "seed": 2026071201,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tp,
        "batch_questions": args.batch_questions,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enforce_eager": args.eager,
        "prompt": "gpqa_raw_prompt_v1",
    }
    if stride_requested:
        protocol["stride_index"] = args.stride_index
        protocol["stride_count"] = args.stride_count
    else:
        protocol["ordinal_file"] = str(Path(args.ordinal_file))
        protocol["ordinal_file_sha256"] = sha256_file(Path(args.ordinal_file))
    protocol_text = json.dumps(protocol, ensure_ascii=False, indent=2) + "\n"
    if protocol_path.exists() and protocol_path.read_text() != protocol_text:
        raise ValueError(f"resume protocol mismatch: {protocol_path}")
    atomic_write(protocol_path, protocol_text)

    records = read_progress(progress_path)
    expected_ordinals = {ordinal for ordinal, _ in selected}
    if not set(records) <= expected_ordinals:
        raise ValueError("progress includes an unexpected ordinal")
    llm_kwargs = dict(
        model=args.base,
        tensor_parallel_size=args.tp,
        enable_lora=bool(args.adapter),
        max_lora_rank=64,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=(
            args.max_num_batched_tokens or args.max_model_len
        ),
        enable_prefix_caching=True,
        seed=2026071201,
    )
    if args.eager:
        llm_kwargs.update(
            enforce_eager=True,
            compilation_config={"mode": 0, "custom_ops": []},
        )
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        n=4,
        seed=2026071201,
        temperature=1.0,
        top_p=0.95,
        top_k=-1,
        max_tokens=args.max_tokens,
    )
    lora_request = LoRARequest("adapter", 1, args.adapter) if args.adapter else None

    started = time.monotonic()
    pending = [(ordinal, row) for ordinal, row in selected if ordinal not in records]
    for offset in range(0, len(pending), args.batch_questions):
        batch = pending[offset : offset + args.batch_questions]
        batch_started = time.monotonic()
        generated = llm.generate(
            [build_prompt(row) for _, row in batch],
            sampling,
            lora_request=lora_request,
        )
        for (ordinal, row), result in zip(batch, generated):
            texts = [output.text for output in result.outputs]
            predictions = [parse_last_boxed_option(text) or "" for text in texts]
            records[ordinal] = {
                "schema": SCHEMA,
                "parser": PARSER_ID,
                "ordinal": ordinal,
                "id": row["id"],
                "answer": row["answer"],
                "pred": predictions,
                "correct": [prediction == row["answer"] for prediction in predictions],
                "output_tokens": [len(output.token_ids) for output in result.outputs],
                "finish_reasons": [str(output.finish_reason or "") for output in result.outputs],
                "texts": texts,
            }
        write_progress(progress_path, records)
        print(
            json.dumps(
                {
                    "event": "batch_complete",
                    "completed_questions": len(records),
                    "expected_questions": len(selected),
                    "completed_generations": len(records) * 4,
                    "batch_questions": len(batch),
                    "batch_seconds": time.monotonic() - batch_started,
                }
            ),
            flush=True,
        )

    ordered = [records[ordinal] for ordinal in sorted(expected_ordinals)]
    if len(ordered) != len(selected):
        raise ValueError(f"incomplete GPQA evaluation: {len(ordered)}/{len(selected)}")
    output_tokens = [value for record in ordered for value in record["output_tokens"]]
    correct = sum(sum(record["correct"]) for record in ordered)
    invalid = sum(pred == "" for record in ordered for pred in record["pred"])
    cap_hit = sum(
        reason == "length" for record in ordered for reason in record["finish_reasons"]
    )
    generations = len(ordered) * 4
    summary = {
        "schema": SCHEMA,
        "parser": PARSER_ID,
        "dataset": "GPQA-Diamond",
        "n_problems": len(ordered),
        "expected_problems": len(selected),
        "generations": generations,
        "accuracy_avg4": correct / generations,
        "correct": correct,
        "invalid": invalid,
        "cap_hit": cap_hit,
        "mean_output_tokens": statistics.fmean(output_tokens),
        "p50_output_tokens": percentile(output_tokens, 0.50),
        "p90_output_tokens": percentile(output_tokens, 0.90),
        "p99_output_tokens": percentile(output_tokens, 0.99),
        "elapsed_seconds_this_process": time.monotonic() - started,
    }
    atomic_write(out_dir / "run_summary.json", json.dumps(summary, indent=2) + "\n")
    atomic_write(out_dir / "stage_eval.done", json.dumps(summary, separators=(",", ":")) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
