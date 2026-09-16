#!/usr/bin/env python3
"""Reproducible competition-math avg@12 evaluation in three seed blocks.

Each block produces ``block_n`` completions per problem.  Blocks are written
atomically and can be resumed independently; the merged result is accepted only
when every problem has exactly ``len(seeds) * block_n`` indexed generations.
The historical avg@4 artifacts are deliberately not read or merged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and limit >= 0 and len(rows) >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def load_rows(dataset: str, path: Path, limit: int | None) -> list[dict[str, str]]:
    rows = []
    for index, raw in enumerate(read_jsonl(path, limit)):
        problem = first_present(raw, ("problem", "Problem", "prompt", "question", "Question"))
        answer = first_present(raw, ("answer", "Answer", "target", "final_answer", "Final Answer"))
        if problem and answer is not None:
            rows.append(
                {
                    "problem_id": str(raw.get("id") or raw.get("ID") or raw.get("problem_idx") or f"{dataset}-{index}"),
                    "problem": str(problem),
                    "ground_truth": str(answer),
                }
            )
    return rows


def build_prompt(tokenizer: Any, problem: str, enable_thinking: bool) -> str:
    user = f"{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def majority_status(predictions: list[str | None], ground_truth: str, grade_answer: Any) -> tuple[bool, bool]:
    formatted = [prediction for prediction in predictions if prediction is not None]
    if not formatted:
        return False, False
    counts = Counter(formatted)
    top = max(counts.values())
    winners = [answer for answer, count in counts.items() if count == top]
    if len(winners) != 1:
        return False, True
    return bool(grade_answer(f"\\boxed{{{winners[0]}}}", ground_truth)), False


def metrics_from_results(results: list[dict[str, Any]], val_n: int, grade_answer: Any) -> dict[str, Any]:
    total_correct = 0
    total_formatted = 0
    total_tokens = 0
    hit_max = 0
    pass_count = 0
    majority_count = 0
    majority_ties = 0
    for row in results:
        generations = row["generations"]
        if len(generations) != val_n:
            raise ValueError(f"{row['problem_id']}: expected {val_n} generations, found {len(generations)}")
        indices = [int(item["sample_index"]) for item in generations]
        if sorted(indices) != list(range(val_n)):
            raise ValueError(f"{row['problem_id']}: invalid sample indices {indices}")
        correct = [bool(item["correct"]) for item in generations]
        predictions = [item.get("predicted_answer_raw") for item in generations]
        majority_ok, majority_tie = majority_status(predictions, row["ground_truth"], grade_answer)
        row["num_correct"] = sum(correct)
        row["pass_at_n"] = any(correct)
        row["majority_vote_correct"] = majority_ok
        row["majority_vote_tie"] = majority_tie
        total_correct += sum(correct)
        total_formatted += sum(bool(item["formatted"]) for item in generations)
        total_tokens += sum(int(item["completion_tokens"]) for item in generations)
        hit_max += sum(bool(item["hit_max_new_tokens"]) for item in generations)
        pass_count += int(any(correct))
        majority_count += int(majority_ok)
        majority_ties += int(majority_tie)
    total_solutions = len(results) * val_n
    return {
        "num_problems": len(results),
        "val_n": val_n,
        "total_solutions": total_solutions,
        "average_at_n": total_correct,
        "average_at_n_pct": 100.0 * total_correct / total_solutions if total_solutions else 0.0,
        "pass_at_n": pass_count,
        "pass_at_n_pct": 100.0 * pass_count / len(results) if results else 0.0,
        "majority_vote_at_n": majority_count,
        "majority_vote_at_n_pct": 100.0 * majority_count / len(results) if results else 0.0,
        "majority_vote_ties": majority_ties,
        "majority_vote_tie_rate_pct": 100.0 * majority_ties / len(results) if results else 0.0,
        "formatted_count": total_formatted,
        "format_rate_pct": 100.0 * total_formatted / total_solutions if total_solutions else 0.0,
        "hit_max_new_tokens_count": hit_max,
        "hit_max_new_tokens_rate": hit_max / total_solutions if total_solutions else 0.0,
        "avg_completion_tokens": total_tokens / total_solutions if total_solutions else 0.0,
    }


def block_contract(args: argparse.Namespace, seed: int, dataset_hash: str, evaluator_hash: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset": args.dataset,
        "dataset_sha256": dataset_hash,
        "base_model": args.base_model,
        "checkpoint_dir": args.checkpoint_dir,
        "checkpoint_sha256": args.checkpoint_sha256,
        "evaluator_sha256": evaluator_hash,
        "seed": seed,
        "block_n": args.block_n,
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "enable_thinking": args.enable_thinking,
    }


def load_valid_block(path: Path, expected_contract: dict[str, Any], expected_ids: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("contract") != expected_contract:
        raise ValueError(f"existing block contract mismatch: {path}")
    results = payload.get("results", [])
    if [row.get("problem_id") for row in results] != expected_ids:
        raise ValueError(f"existing block problem order mismatch: {path}")
    if any(len(row.get("generations", [])) != int(expected_contract["block_n"]) for row in results):
        raise ValueError(f"existing block generation count mismatch: {path}")
    return payload


def make_lora_request(checkpoint_dir: str | None) -> Any:
    if not checkpoint_dir:
        return None
    checkpoint = Path(checkpoint_dir)
    if not ((checkpoint / "adapter_model.safetensors").is_file() or (checkpoint / "adapter_model.bin").is_file()):
        raise FileNotFoundError(f"missing LoRA adapter weights under {checkpoint}")
    from vllm.lora.request import LoRARequest

    return LoRARequest("socratic_opsd_avg12_lora", 1, str(checkpoint))


def generate_block(
    args: argparse.Namespace,
    llm: Any,
    tokenizer: Any,
    lora_request: Any,
    rows: list[dict[str, str]],
    seed: int,
    block_index: int,
    contract: dict[str, Any],
) -> dict[str, Any]:
    from vllm import SamplingParams
    from air_opd.core.math_verifier import extract_boxed_answer, grade_answer

    prompts = [build_prompt(tokenizer, row["problem"], args.enable_thinking) for row in rows]
    sampling = SamplingParams(
        n=args.block_n,
        seed=seed,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
    )
    started = time.time()
    outputs = llm.generate(prompts, sampling, lora_request=lora_request, use_tqdm=True)
    results = []
    for row, output in zip(rows, outputs, strict=True):
        generations = []
        for local_index, sample in enumerate(output.outputs):
            text = sample.text
            token_ids = list(getattr(sample, "token_ids", []) or [])
            predicted = extract_boxed_answer(text)
            finish_reason = getattr(sample, "finish_reason", None)
            generations.append(
                {
                    "seed": seed,
                    "block_index": block_index,
                    "block_sample_index": local_index,
                    "sample_index": block_index * args.block_n + local_index,
                    "predicted_answer": predicted if predicted is not None else "[No boxed answer found]",
                    "predicted_answer_raw": predicted,
                    "correct": bool(grade_answer(text, row["ground_truth"])),
                    "formatted": predicted is not None,
                    "completion_tokens": len(token_ids),
                    "finish_reason": finish_reason,
                    "hit_max_new_tokens": bool(finish_reason == "length" or len(token_ids) >= args.max_new_tokens),
                    "full_generation": text,
                }
            )
        results.append({**row, "generations": generations})
    return {
        "kind": "socratic_opsd_competition_math_avg12_seed_block",
        "contract": contract,
        "wall_seconds": time.time() - started,
        "results": results,
    }


def merge_blocks(blocks: list[dict[str, Any]], seeds: list[int], block_n: int, grade_answer: Any) -> dict[str, Any]:
    if not blocks:
        raise ValueError("no seed blocks to merge")
    merged: list[dict[str, Any]] = []
    first_rows = blocks[0]["results"]
    for row_index, first in enumerate(first_rows):
        generations = []
        for block_index, block in enumerate(blocks):
            row = block["results"][row_index]
            for item in row["generations"]:
                if int(item["block_index"]) != block_index or int(item["seed"]) != seeds[block_index]:
                    raise ValueError(f"seed block metadata mismatch for {first['problem_id']}")
                generations.append(dict(item))
        merged.append(
            {
                "problem_id": first["problem_id"],
                "problem": first["problem"],
                "ground_truth": first["ground_truth"],
                "generations": generations,
            }
        )
    val_n = len(seeds) * block_n
    metrics = metrics_from_results(merged, val_n, grade_answer)
    return {"metrics": metrics, "results": merged}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--checkpoint_sha256", default=None)
    parser.add_argument("--dataset", choices=("aime24", "aime25", "hmmt25", "amc23"), required=True)
    parser.add_argument("--dataset_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seeds", default="2026071201,2026071202,2026071203")
    parser.add_argument("--block_n", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=38912)
    parser.add_argument("--max_model_len", type=int, default=40960)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.88)
    parser.add_argument("--max_num_seqs", type=int, default=16)
    parser.add_argument("--max_num_batched_tokens", type=int, default=40960)
    parser.add_argument("--attention_backend", default="FLASH_ATTN")
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce_eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable_chunked_prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_lora_rank", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("seeds must be a non-empty unique list")
    if args.block_n <= 0:
        raise ValueError("block_n must be positive")

    from transformers import AutoTokenizer
    from air_opd.core.math_verifier import grade_answer

    dataset_path = Path(args.dataset_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.dataset, dataset_path, args.max_samples)
    expected_ids = [row["problem_id"] for row in rows]
    dataset_hash = sha256_file(dataset_path)
    evaluator_hash = sha256_file(Path(__file__))

    blocks: list[dict[str, Any] | None] = []
    contracts = []
    for block_index, seed in enumerate(seeds):
        contract = block_contract(args, seed, dataset_hash, evaluator_hash)
        contracts.append(contract)
        blocks.append(load_valid_block(output_dir / f"block_{block_index}_seed_{seed}.json", contract, expected_ids))

    missing = [index for index, block in enumerate(blocks) if block is None]
    if missing:
        from vllm import LLM

        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        llm_kwargs: dict[str, Any] = {
            "model": args.base_model,
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "seed": 0,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "enforce_eager": args.enforce_eager,
            "enable_chunked_prefill": args.enable_chunked_prefill,
            "disable_log_stats": True,
        }
        if args.attention_backend:
            # Environment selection depends on the installed vLLM runtime.
            os.environ["VLLM_ATTENTION_BACKEND"] = args.attention_backend
        if args.checkpoint_dir:
            llm_kwargs.update({"enable_lora": True, "max_lora_rank": args.max_lora_rank, "max_loras": 1, "max_cpu_loras": 1})
        llm = LLM(**llm_kwargs)
        lora_request = make_lora_request(args.checkpoint_dir)
        for block_index in missing:
            payload = generate_block(
                args,
                llm,
                tokenizer,
                lora_request,
                rows,
                seeds[block_index],
                block_index,
                contracts[block_index],
            )
            block_path = output_dir / f"block_{block_index}_seed_{seeds[block_index]}.json"
            atomic_write_json(block_path, payload)
            blocks[block_index] = payload

    merged = merge_blocks([block for block in blocks if block is not None], seeds, args.block_n, grade_answer)
    summary = {
        "kind": "socratic_opsd_competition_math_avg12_seedblocks",
        "schema_version": 1,
        "dataset": args.dataset,
        "seeds": seeds,
        "block_n": args.block_n,
        "val_n": len(seeds) * args.block_n,
        "base_model": args.base_model,
        "checkpoint_dir": args.checkpoint_dir,
        "checkpoint_sha256": args.checkpoint_sha256,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "evaluator_sha256": evaluator_hash,
        "engine": {
            "vllm_version": __import__("vllm").__version__,
            "torch_version": __import__("torch").__version__,
            "attention_backend_requested": args.attention_backend or None,
            "flashinfer_sampler_enabled": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER", "1") != "0",
        },
        "args": vars(args),
        **merged["metrics"],
        "results": merged["results"],
    }
    atomic_write_json(output_dir / "run_summary.json", summary)
    (output_dir / "stage_eval_avg12.done").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
