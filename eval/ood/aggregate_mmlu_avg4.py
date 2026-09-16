#!/usr/bin/env python3
"""Validate and merge four resumable MMLU-Pro-1k evaluation shards."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from jsonl_utils import read_jsonl_records

SCHEMA = "mmlu_pro_1k_avg4_qwen_chat_resumable_v1"


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected_questions", type=int, default=1000)
    parser.add_argument("--shards", type=int, default=4)
    args = parser.parse_args()
    root = Path(args.root)

    records: dict[int, dict] = {}
    shard_summaries = []
    protocols = []
    for shard_index in range(args.shards):
        shard = root / f"shard-{shard_index}"
        if not (shard / "stage_eval.done").is_file():
            raise ValueError(f"missing shard done marker: {shard}")
        shard_summaries.append(json.loads((shard / "run_summary.json").read_text()))
        protocols.append(json.loads((shard / "protocol.json").read_text()))
        for record in read_jsonl_records(shard / "progress.jsonl"):
            if record.get("schema") != SCHEMA:
                raise ValueError(f"schema mismatch in {shard}")
            ordinal = int(record["ordinal"])
            if ordinal in records:
                raise ValueError(f"duplicate ordinal across shards: {ordinal}")
            if len(record["pred"]) != 4 or len(record["correct"]) != 4:
                raise ValueError(f"non-Avg@4 record at ordinal {ordinal}")
            records[ordinal] = record

    identity_keys = (
        "schema", "parser", "base", "adapter", "data_sha256", "temperature",
        "top_p", "top_k", "n", "seed", "max_tokens", "max_model_len",
        "thinking", "chat_template", "system_prompt", "eos_token_id",
    )
    for protocol in protocols[1:]:
        for key in identity_keys:
            if protocol.get(key) != protocols[0].get(key):
                raise ValueError(f"protocol mismatch for {key}")

    expected_ordinals = set(range(args.expected_questions))
    if set(records) != expected_ordinals:
        missing = sorted(expected_ordinals - set(records))
        extra = sorted(set(records) - expected_ordinals)
        raise ValueError(f"ordinal mismatch missing={missing[:20]} extra={extra[:20]}")

    ordered = [records[ordinal] for ordinal in range(args.expected_questions)]
    question_ids = [record["question_id"] for record in ordered]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("duplicate question_id in merged evaluation")
    output_tokens = [
        value for record in ordered for value in record["output_tokens"]
    ]
    correct = sum(sum(record["correct"]) for record in ordered)
    invalid = sum(
        pred is None or pred == ""
        for record in ordered
        for pred in record["pred"]
    )
    cap_hit = sum(
        reason == "length"
        for record in ordered
        for reason in record["finish_reasons"]
    )
    generations = args.expected_questions * 4
    summary = {
        "schema": SCHEMA,
        "parser": protocols[0]["parser"],
        "dataset": "MMLU-Pro-1k-frozen",
        "base": protocols[0]["base"],
        "adapter": protocols[0]["adapter"],
        "data_sha256": protocols[0]["data_sha256"],
        "n_problems": args.expected_questions,
        "generations": generations,
        "accuracy_avg4": correct / generations,
        "correct": correct,
        "invalid": invalid,
        "cap_hit": cap_hit,
        "mean_output_tokens": statistics.fmean(output_tokens),
        "p50_output_tokens": percentile(output_tokens, 0.50),
        "p90_output_tokens": percentile(output_tokens, 0.90),
        "p99_output_tokens": percentile(output_tokens, 0.99),
        "shards": shard_summaries,
    }
    atomic_write(
        root / "progress.jsonl",
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in ordered),
    )
    atomic_write(
        root / "run_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write(
        root / "stage_eval.done",
        json.dumps(summary, ensure_ascii=False, separators=(",", ":")) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
