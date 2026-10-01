#!/usr/bin/env python3
"""Resumable MMLU-Pro-1k Avg@4 evaluator for Qwen3 LoRA checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from jsonl_utils import read_jsonl_records

SCHEMA = "mmlu_pro_1k_avg4_qwen_chat_resumable_v1"
PARSER = "mmlu_pro_last_boxed_then_explicit_answer_v1"
SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--pilot_stratified", type=int)
    parser.add_argument("--stride_index", type=int)
    parser.add_argument("--stride_count", type=int)
    parser.add_argument("--ordinal_file")
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--max_model_len", type=int, default=12288)
    parser.add_argument("--batch_questions", type=int, default=16)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.88)
    parser.add_argument("--max_num_seqs", type=int, default=16)
    parser.add_argument("--max_num_batched_tokens", type=int)
    parser.add_argument("--no_lora", action="store_true")
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
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(fraction * len(ordered))))
    return ordered[index]


def select_stratified(rows: list[dict], count: int) -> list[tuple[int, dict]]:
    by_category: dict[str, list[tuple[int, dict]]] = {}
    for ordinal, row in enumerate(rows):
        by_category.setdefault(row["category"], []).append((ordinal, row))
    selected: list[tuple[int, dict]] = []
    categories = sorted(by_category)
    offset = 0
    while len(selected) < count:
        advanced = False
        for category in categories:
            bucket = by_category[category]
            if offset < len(bucket):
                selected.append(bucket[offset])
                advanced = True
                if len(selected) == count:
                    break
        if not advanced:
            break
        offset += 1
    if len(selected) != count:
        raise ValueError(f"requested {count} pilot rows but selected {len(selected)}")
    return sorted(selected)


def parse_choice(text: str) -> tuple[str, str]:
    boxed = re.findall(r"(?i)\\boxed\s*\{\s*([A-J])\s*\}", text)
    if boxed:
        return boxed[-1].upper(), "boxed"
    explicit = re.findall(
        r"(?i)(?:final\s+answer(?:\s+is)?|answer)\s*[:：]?\s*\(?([A-J])\)?",
        text,
    )
    if explicit:
        return explicit[-1].upper(), "explicit_answer"
    return "", "invalid"


def build_prompt(tokenizer, row: dict) -> str:
    options = "\n".join(
        f"{chr(65 + index)}. {option}" for index, option in enumerate(row["options"])
    )
    choices = ", ".join(
        f"\\boxed{{{chr(65 + index)}}}" for index in range(len(row["options"]))
    )
    user_prompt = (
        f"Question:\n{row['question']}\nOptions:\n{options}\n\n"
        f"Choose the correct option. Give the final choice as one of: {choices}."
    )
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


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


def build_summary(records: list[dict], expected_questions: int, elapsed: float) -> dict:
    output_tokens = [
        token_count for record in records for token_count in record["output_tokens"]
    ]
    correct = sum(sum(record["correct"]) for record in records)
    invalid = sum(
        prediction == "" for record in records for prediction in record["pred"]
    )
    cap_hit = sum(
        finish_reason == "length"
        for record in records
        for finish_reason in record["finish_reasons"]
    )
    generations = len(records) * 4
    return {
        "schema": SCHEMA,
        "parser": PARSER,
        "dataset": "MMLU-Pro-1k-frozen",
        "n_problems": len(records),
        "expected_problems": expected_questions,
        "generations": generations,
        "accuracy_avg4": correct / generations if generations else 0.0,
        "correct": correct,
        "invalid": invalid,
        "cap_hit": cap_hit,
        "mean_output_tokens": statistics.fmean(output_tokens) if output_tokens else 0.0,
        "p50_output_tokens": percentile(output_tokens, 0.50),
        "p90_output_tokens": percentile(output_tokens, 0.90),
        "p99_output_tokens": percentile(output_tokens, 0.99),
        "elapsed_seconds_this_process": elapsed,
    }


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    source_rows = read_jsonl_records(data_path)
    stride_requested = args.stride_index is not None or args.stride_count is not None
    ordinal_file_requested = args.ordinal_file is not None
    if stride_requested and ordinal_file_requested:
        raise ValueError("strided selection cannot be combined with --ordinal_file")
    if ordinal_file_requested:
        if args.pilot_stratified is not None or args.start != 0 or args.end is not None:
            raise ValueError("ordinal-file selection cannot be combined with pilot/start/end")
        ordinal_path = Path(args.ordinal_file)
        ordinal_values = [
            int(line.strip())
            for line in ordinal_path.read_text().splitlines()
            if line.strip()
        ]
        if len(set(ordinal_values)) != len(ordinal_values):
            raise ValueError("duplicate ordinal in --ordinal_file")
        if any(ordinal < 0 or ordinal >= len(source_rows) for ordinal in ordinal_values):
            raise ValueError("ordinal outside dataset range")
        selected = [(ordinal, source_rows[ordinal]) for ordinal in ordinal_values]
    elif stride_requested:
        if args.stride_index is None or args.stride_count is None:
            raise ValueError("--stride_index and --stride_count must be used together")
        if args.pilot_stratified is not None or args.start != 0 or args.end is not None:
            raise ValueError("strided selection cannot be combined with pilot/start/end")
        if args.stride_count <= 0 or not 0 <= args.stride_index < args.stride_count:
            raise ValueError("invalid stride index/count")
        selected = [
            (ordinal, row)
            for ordinal, row in enumerate(source_rows)
            if ordinal % args.stride_count == args.stride_index
        ]
    elif args.pilot_stratified:
        selected = select_stratified(source_rows, args.pilot_stratified)
    else:
        end = args.end if args.end is not None else len(source_rows)
        selected = list(enumerate(source_rows))[args.start:end]
    if not selected:
        raise ValueError("no evaluation rows selected")
    if args.max_model_len <= args.max_tokens:
        raise ValueError("max_model_len must be greater than max_tokens")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.jsonl"
    protocol_path = out_dir / "protocol.json"

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer has no eos_token_id")
    protocol = {
        "schema": SCHEMA,
        "parser": PARSER,
        "base": args.base,
        "adapter": None if args.no_lora else args.adapter,
        "data": str(data_path),
        "data_sha256": sha256_file(data_path),
        "start": args.start,
        "end": args.end,
        "pilot_stratified": args.pilot_stratified,
        "question_ordinals": [ordinal for ordinal, _ in selected],
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": -1,
        "n": 4,
        "seed": 2026071201,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "thinking": True,
        "chat_template": True,
        "system_prompt": SYSTEM_PROMPT,
        "eos_token_id": tokenizer.eos_token_id,
        "tensor_parallel_size": args.tp,
        "batch_questions": args.batch_questions,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enforce_eager": args.eager,
    }
    if stride_requested:
        protocol["stride_index"] = args.stride_index
        protocol["stride_count"] = args.stride_count
    if ordinal_file_requested:
        protocol["ordinal_file"] = str(Path(args.ordinal_file))
        protocol["ordinal_file_sha256"] = sha256_file(Path(args.ordinal_file))
    protocol_text = json.dumps(protocol, ensure_ascii=False, indent=2) + "\n"
    if protocol_path.exists() and protocol_path.read_text() != protocol_text:
        raise ValueError(f"resume protocol mismatch: {protocol_path}")
    atomic_write(protocol_path, protocol_text)

    records = read_progress(progress_path)
    expected_ordinals = {ordinal for ordinal, _ in selected}
    unexpected = set(records) - expected_ordinals
    if unexpected:
        raise ValueError(f"progress contains unexpected ordinals: {sorted(unexpected)}")

    llm_kwargs = dict(
        model=args.base,
        tensor_parallel_size=args.tp,
        enable_lora=not args.no_lora,
        max_lora_rank=64,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=(
            args.max_num_batched_tokens or max(32768, args.max_model_len)
        ),
        enable_prefix_caching=True,
        trust_remote_code=True,
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
        stop_token_ids=[tokenizer.eos_token_id],
    )
    lora_request = None
    if not args.no_lora:
        if not args.adapter:
            raise ValueError("--adapter is required unless --no_lora is set")
        lora_request = LoRARequest("adapter", 1, args.adapter)

    started = time.monotonic()
    pending = [(ordinal, row) for ordinal, row in selected if ordinal not in records]
    for offset in range(0, len(pending), args.batch_questions):
        batch = pending[offset : offset + args.batch_questions]
        prompts = [build_prompt(tokenizer, row) for _, row in batch]
        batch_started = time.monotonic()
        generated = llm.generate(
            prompts,
            sampling,
            lora_request=lora_request,
        )
        for (ordinal, row), request_output in zip(batch, generated):
            predictions: list[str] = []
            parser_routes: list[str] = []
            texts: list[str] = []
            output_tokens: list[int] = []
            finish_reasons: list[str] = []
            stop_reasons: list[object] = []
            for output in request_output.outputs:
                prediction, parser_route = parse_choice(output.text)
                predictions.append(prediction)
                parser_routes.append(parser_route)
                texts.append(output.text)
                output_tokens.append(len(output.token_ids))
                finish_reasons.append(str(output.finish_reason or ""))
                stop_reasons.append(output.stop_reason)
            records[ordinal] = {
                "schema": SCHEMA,
                "parser": PARSER,
                "ordinal": ordinal,
                "question_id": row["question_id"],
                "category": row["category"],
                "answer": row["answer"],
                "pred": predictions,
                "parser_routes": parser_routes,
                "correct": [prediction == row["answer"] for prediction in predictions],
                "output_tokens": output_tokens,
                "finish_reasons": finish_reasons,
                "stop_reasons": stop_reasons,
                "texts": texts,
            }
        write_progress(progress_path, records)
        completed = len(records)
        batch_elapsed = time.monotonic() - batch_started
        print(
            json.dumps(
                {
                    "event": "batch_complete",
                    "completed_questions": completed,
                    "expected_questions": len(selected),
                    "completed_generations": completed * 4,
                    "batch_questions": len(batch),
                    "batch_seconds": batch_elapsed,
                }
            ),
            flush=True,
        )

    final_records = [records[ordinal] for ordinal in sorted(expected_ordinals)]
    if len(final_records) != len(selected):
        raise ValueError(
            f"incomplete evaluation: {len(final_records)}/{len(selected)} questions"
        )
    summary = build_summary(
        final_records,
        expected_questions=len(selected),
        elapsed=time.monotonic() - started,
    )
    atomic_write(
        out_dir / "run_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write(
        out_dir / "stage_eval.done",
        json.dumps(summary, ensure_ascii=False, separators=(",", ":")) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
