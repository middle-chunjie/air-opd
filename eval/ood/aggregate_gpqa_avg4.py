#!/usr/bin/env python3
"""Validate and merge resumable GPQA-Diamond Avg@4 shards."""

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

SUPPORTED_SCHEMAS = {
    "gpqa_diamond_avg4_raw_qwen_resumable_v1",
    "gpqa_diamond_avg4_qwen_lora_resumable_v1",
}
IDENTITY_KEYS = (
    "schema", "parser", "base", "adapter", "data_sha256", "temperature", "top_p",
    "top_k", "n", "seed", "max_tokens", "max_model_len", "prompt",
)


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected_questions", type=int, default=198)
    parser.add_argument("--shards", type=int, default=2)
    args = parser.parse_args()
    root = Path(args.root)
    records: dict[int, dict] = {}
    protocols = []
    summaries = []
    for shard_index in range(args.shards):
        shard = root / f"shard-{shard_index}"
        if not (shard / "stage_eval.done").is_file():
            raise ValueError(f"missing shard done marker: {shard}")
        summaries.append(json.loads((shard / "run_summary.json").read_text()))
        protocols.append(json.loads((shard / "protocol.json").read_text()))
        if protocols[-1].get("schema") not in SUPPORTED_SCHEMAS:
            raise ValueError(f"unsupported GPQA schema in {shard}")
        for record in read_jsonl_records(shard / "progress.jsonl"):
            if record.get("schema") != protocols[-1].get("schema"):
                raise ValueError(f"schema mismatch in {shard}")
            ordinal = int(record["ordinal"])
            if ordinal in records:
                raise ValueError(f"duplicate ordinal: {ordinal}")
            if len(record["pred"]) != 4 or len(record["correct"]) != 4:
                raise ValueError(f"non-Avg@4 record: {ordinal}")
            records[ordinal] = record
    for protocol in protocols[1:]:
        for key in IDENTITY_KEYS:
            if protocols[0].get(key) != protocol.get(key):
                raise ValueError(f"protocol mismatch for {key}")
    schema = protocols[0]["schema"]
    expected = set(range(args.expected_questions))
    if set(records) != expected:
        raise ValueError(
            f"ordinal mismatch missing={sorted(expected-set(records))[:20]} "
            f"extra={sorted(set(records)-expected)[:20]}"
        )
    ordered = [records[ordinal] for ordinal in range(args.expected_questions)]
    ids = [record["id"] for record in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate GPQA id")
    tokens = [value for record in ordered for value in record["output_tokens"]]
    correct = sum(sum(record["correct"]) for record in ordered)
    invalid = sum(pred == "" for record in ordered for pred in record["pred"])
    cap_hit = sum(
        reason == "length" for record in ordered for reason in record["finish_reasons"]
    )
    generations = args.expected_questions * 4
    summary = {
        "schema": schema,
        "parser": protocols[0]["parser"],
        "dataset": "GPQA-Diamond",
        "base": protocols[0]["base"],
        "data_sha256": protocols[0]["data_sha256"],
        "n_problems": args.expected_questions,
        "generations": generations,
        "accuracy_avg4": correct / generations,
        "correct": correct,
        "invalid": invalid,
        "cap_hit": cap_hit,
        "mean_output_tokens": statistics.fmean(tokens),
        "p50_output_tokens": percentile(tokens, 0.50),
        "p90_output_tokens": percentile(tokens, 0.90),
        "p99_output_tokens": percentile(tokens, 0.99),
        "shards": summaries,
    }
    atomic_write(
        root / "progress.jsonl",
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in ordered),
    )
    atomic_write(root / "run_summary.json", json.dumps(summary, indent=2) + "\n")
    atomic_write(root / "stage_eval.done", json.dumps(summary, separators=(",", ":")) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
