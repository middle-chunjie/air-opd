"""Minimal data collator for Socratic-OPSD.

The multi-round repair loop renders and tokenizes all prompts itself inside the
training step (it needs the live model for on-policy rollouts), so the collator
only needs to pass the raw problem text, gold answer, and optional teacher
reference through the batch.
"""
from __future__ import annotations


class SocraticDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # right padding matches the assembly in SocraticOPSDTrainer
        self.tokenizer.padding_side = "right"

    def __call__(self, features: list) -> dict:
        problems, golds, references, sample_ids, task_types = [], [], [], [], []
        for i, f in enumerate(features):
            problems.append(str(f.get("problem") or f.get("question") or ""))
            golds.append(str(f.get("gold") or f.get("answer") or f.get("ground_truth") or ""))
            references.append(str(f.get("reference_solution") or f.get("teacher_context") or ""))
            sample_ids.append(str(f.get("sample_id") or f.get("id") or i))
            task_types.append(str(f.get("task_type") or ""))
        return {
            "problem_texts": problems,
            "gold_texts": golds,
            "reference_texts": references,
            "sample_ids": sample_ids,
            "task_types": task_types,
            # dummy tensor column so TRL SFTTrainer keeps the batch intact
            "labels_placeholder": [0] * len(features),
        }
