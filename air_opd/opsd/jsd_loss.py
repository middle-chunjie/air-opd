"""Memory-bounded generalized JSD used by the OPSD trainer.

Implements the generalized JSD objective of HuggingFace TRL (Apache-2.0) with
a chunked reduction; see LICENSE.

The optional sequence chunking path preserves the existing objective and only
changes how its scalar reduction is evaluated.  It avoids materializing an
additional full ``[batch, sequence, vocabulary]`` KL/JSD tensor for long
completions.  The legacy path remains the default.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _elementwise_jsd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    beta: float,
    temperature: float,
    logits_are_probs: bool,
    top_k: int | None,
) -> torch.Tensor:
    """Return the legacy elementwise generalized JSD for one sequence slice."""

    if logits_are_probs:
        # Preserve the historical behavior: top-k is ignored when probability
        # tensors, rather than logits, are supplied.
        student_log_probs = torch.log(student_logits.clamp_min(1e-8))
        teacher_log_probs = torch.log(teacher_logits.clamp_min(1e-8))
    else:
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        if top_k is not None and top_k > 0:
            _, top_k_indices = torch.topk(teacher_logits, k=top_k, dim=-1)
            student_logits = torch.gather(student_logits, dim=-1, index=top_k_indices)
            teacher_logits = torch.gather(teacher_logits, dim=-1, index=top_k_indices)

        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    if beta == 0:
        return F.kl_div(
            student_log_probs,
            teacher_log_probs,
            reduction="none",
            log_target=True,
        )
    if beta == 1:
        return F.kl_div(
            teacher_log_probs,
            student_log_probs,
            reduction="none",
            log_target=True,
        )

    beta_tensor = torch.tensor(
        beta,
        dtype=student_log_probs.dtype,
        device=student_log_probs.device,
    )
    mixture_log_probs = torch.logsumexp(
        torch.stack(
            [
                student_log_probs + torch.log1p(-beta_tensor),
                teacher_log_probs + torch.log(beta_tensor),
            ]
        ),
        dim=0,
    )
    kl_teacher = F.kl_div(
        mixture_log_probs,
        teacher_log_probs,
        reduction="none",
        log_target=True,
    )
    kl_student = F.kl_div(
        mixture_log_probs,
        student_log_probs,
        reduction="none",
        log_target=True,
    )
    return beta_tensor * kl_teacher + (1 - beta_tensor) * kl_student


def generalized_jsd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor | None = None,
    beta: float = 0.5,
    temperature: float = 1.0,
    reduction: str = "batchmean",
    logits_are_probs: bool = False,
    top_k: int | None = None,
    token_clip: float | None = None,
    sequence_chunk_size: int | None = None,
) -> torch.Tensor:
    """Compute the existing generalized JSD, optionally by sequence chunks.

    Chunking is used only for scalar reductions.  ``reduction='none'`` keeps
    the legacy full-tensor behavior because its public return shape requires
    the full elementwise tensor.
    """

    chunk_size = int(sequence_chunk_size or 0)
    chunk_candidate = chunk_size > 0 and reduction in {"batchmean", "sum", "mean"}
    if chunk_candidate:
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                "student_logits and teacher_logits must have identical shapes, "
                f"got {tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}"
            )
        if student_logits.ndim != 3:
            raise ValueError(
                "student_logits and teacher_logits must be rank-3 "
                "[batch, sequence, vocabulary]"
            )
        if labels is not None and labels.shape != student_logits.shape[:2]:
            raise ValueError(
                "labels must have shape [batch, sequence], "
                f"got {tuple(labels.shape)} for logits {tuple(student_logits.shape)}"
            )
        sequence_length = int(student_logits.shape[1])
        use_chunking = chunk_size < sequence_length
    else:
        sequence_length = 0
        use_chunking = False

    if not use_chunking:
        jsd = _elementwise_jsd(
            student_logits,
            teacher_logits,
            beta=beta,
            temperature=temperature,
            logits_are_probs=logits_are_probs,
            top_k=top_k,
        )
        if token_clip is not None:
            # This intentionally preserves the historical elementwise clamp,
            # even though the option name says "token" clip.
            jsd = jsd.clamp(max=token_clip)
        if labels is not None:
            mask = labels != -100
            jsd = jsd[mask]
        if reduction == "batchmean":
            return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / jsd.size(0)
        if reduction == "sum":
            return jsd.sum()
        if reduction == "mean":
            return jsd.mean()
        return jsd

    total = student_logits.new_zeros(())
    effective_vocab = (
        int(top_k)
        if (not logits_are_probs and top_k is not None and top_k > 0)
        else int(student_logits.shape[-1])
    )
    if labels is not None:
        full_mask = labels != -100
        valid_token_count = int(full_mask.sum().item())
        element_count = valid_token_count * effective_vocab
    else:
        full_mask = None
        valid_token_count = 0
        element_count = int(
            student_logits.shape[0] * student_logits.shape[1] * effective_vocab
        )
    for start in range(0, sequence_length, chunk_size):
        end = min(sequence_length, start + chunk_size)
        student_chunk = student_logits[:, start:end, :]
        teacher_chunk = teacher_logits[:, start:end, :]
        chunk_mask = full_mask[:, start:end] if full_mask is not None else None

        def reduced_chunk(
            current_student: torch.Tensor,
            current_teacher: torch.Tensor,
            current_mask: torch.Tensor | None = chunk_mask,
        ) -> torch.Tensor:
            jsd_chunk = _elementwise_jsd(
                current_student,
                current_teacher,
                beta=beta,
                temperature=temperature,
                logits_are_probs=logits_are_probs,
                top_k=top_k,
            )
            if token_clip is not None:
                jsd_chunk = jsd_chunk.clamp(max=token_clip)
            if current_mask is not None:
                jsd_chunk = jsd_chunk[current_mask]
            return jsd_chunk.sum()

        # A loop alone is insufficient: one outer backward would retain every
        # chunk's log-probability/KL/clamp activations and reconstruct the same
        # aggregate peak.  Non-reentrant checkpointing retains only the chunk
        # inputs and recomputes its elementwise divergence during backward.
        if student_chunk.requires_grad or teacher_chunk.requires_grad:
            chunk_total = checkpoint(
                reduced_chunk,
                student_chunk,
                teacher_chunk,
                use_reentrant=False,
            )
        else:
            chunk_total = reduced_chunk(student_chunk, teacher_chunk)
        total = total + chunk_total

    if reduction == "sum":
        return total
    if reduction == "batchmean":
        denominator = valid_token_count if labels is not None else int(student_logits.shape[0])
        return total / denominator
    # reduction == "mean"
    return total / element_count
