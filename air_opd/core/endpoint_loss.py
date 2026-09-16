"""Torch numerics behind the cumulative-correction-coverage endpoint.

Split from ``socratic.endpoint`` so the block partition and the selection rule
stay importable without torch, and so this part is testable without the trainer's
transformers/trl dependencies.

Both pieces operate on quantities the training forward already produced: the
per-token divergence that the OPSD objective sums, and the teacher/student
distributions it summed it from.  Nothing here runs a model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from air_opd.core.endpoint import block_masses, select_coverage_endpoint


def per_token_correction_mass(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    observed_ids: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return ``C_j`` for one sequence slice, shaped ``[sequence]``.

    ``C_j = sum_{v != y_j} [ teacher(v) - student(v) ]_+`` at the same
    temperature the objective uses.  The observed token is excluded so that a
    teacher that is merely more confident in the continuation the student
    already produced does not register as corrective evidence.

    Computed in float32 regardless of the incoming dtype: the quantity is a
    difference of probabilities summed over the vocabulary, where bf16 rounding
    is comparable to the small tail budget the endpoint rule spends.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have identical shapes")
    if student_logits.ndim != 3 or student_logits.shape[0] != 1:
        raise ValueError(
            "per_token_correction_mass expects one sequence "
            f"(got {tuple(student_logits.shape)})"
        )
    if observed_ids.ndim != 1 or observed_ids.shape[0] != student_logits.shape[1]:
        raise ValueError("observed_ids must be one id per scored position")
    with torch.no_grad():
        student_probs = torch.softmax(student_logits.float() / temperature, dim=-1)
        positive = torch.softmax(teacher_logits.float() / temperature, dim=-1)
        # In place: one fewer [1, sequence, vocab] float32 tensor alive.
        positive.sub_(student_probs).clamp_min_(0.0)
        del student_probs
        observed = positive.gather(-1, observed_ids.view(1, -1, 1)).squeeze(-1)
        return (positive.sum(dim=-1) - observed)[0]


def coverage_masked_mean(
    divergence: torch.Tensor,
    correction: torch.Tensor,
    *,
    block_start_tokens: Sequence[int],
    region_start: int,
    region_end: int,
    epsilon: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Average ``divergence`` over ``[region_start, endpoint)``.

    The endpoint comes from the detached corrective signal, so it is a constant
    supervision mask: gradients are exactly OPSD's over the retained region and
    the denominator is that region's token count.
    """
    if divergence.ndim != 1 or correction.ndim != 1:
        raise ValueError("divergence and correction must be per-token vectors")
    if divergence.shape != correction.shape:
        raise ValueError("divergence and correction must cover the same positions")
    if not 0 <= region_start < region_end <= divergence.shape[0]:
        raise ValueError(
            f"region [{region_start},{region_end}) outside {divergence.shape[0]} positions"
        )
    masses = block_masses(
        block_start_tokens,
        region_end,
        correction.detach().double().cpu().tolist(),
        region_start,
    )
    endpoint, discarded = select_coverage_endpoint(
        block_start_tokens, masses, region_end, epsilon
    )
    supervised = int(endpoint) - int(region_start)
    if supervised <= 0:
        raise ValueError("coverage endpoint produced an empty supervision region")
    loss = divergence[region_start:endpoint].sum() / supervised
    record = {
        "endpoint_epsilon": float(epsilon),
        "endpoint_region_start_token": int(region_start),
        "endpoint_region_end_token": int(region_end),
        "endpoint_selected_token": int(endpoint),
        "endpoint_supervised_tokens": supervised,
        "endpoint_removed_tokens": int(region_end) - int(endpoint),
        "endpoint_blocks": len(list(block_start_tokens)),
        "endpoint_total_correction_mass": float(sum(masses)),
        "endpoint_discarded_correction_fraction": float(discarded),
        "endpoint_truncated": bool(endpoint < region_end),
    }
    return loss, record
