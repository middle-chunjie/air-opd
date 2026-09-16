"""Cumulative-correction-coverage endpoint for the repair supervision region.

The repair region starts at the located error anchor ``b`` and ends at ``e``.
``e`` is the earliest reasoning-block end that still retains at least
``1 - epsilon`` of the token-summed corrective signal in the error suffix
``[b, T)``:

    e = min { r in E : sum_{j=b}^{r-1} C_j >= (1 - epsilon) * sum_{j=b}^{T-1} C_j }

with ``E`` the set of block end positions (including ``T``).  The per-token
corrective signal is the probability mass the teacher moves toward alternatives
to the observed token,

    C_j = sum_{v != y_j} [ teacher(v | .) - student(v | .) ]_+ ,

so the guarantee is per trajectory and holds on the measured signal alone:
``sum_{j=e}^{T-1} C_j <= epsilon * sum_{j=b}^{T-1} C_j``.  It says nothing about
retained semantic repair opportunity, training gain, or token length.

The adopted tolerance is ``epsilon = 0.10``.
"""

from __future__ import annotations

import bisect
import math
import re
from collections.abc import Sequence

# A reasoning block boundary is a blank line.  ``+`` folds runs of blank lines
# into one boundary so that repeated newlines cannot create empty blocks.
BLANK_LINE_PATTERN = re.compile(r"\n[ \t]*\n+")

DEFAULT_COVERAGE_EPSILON = 0.10


def blank_line_block_starts(
    completion: str,
    offsets: Sequence[Sequence[int]],
    start_token: int,
    total_tokens: int,
) -> list[int]:
    """Return the block start tokens of ``[start_token, total_tokens)``.

    ``offsets`` is the tokenizer offset mapping of the retained completion, so
    the boundaries are the same ones the frozen answer-reachability validation
    asserted against the tokenizer (a blank-line match's end character maps to
    the first token that starts at or after it).  The first block always starts
    at ``start_token``, so the result is never empty when the region is.
    """
    start = int(start_token)
    total = int(total_tokens)
    if not 0 <= start < total:
        raise ValueError(f"empty or inverted supervision region [{start},{total})")
    token_start_chars = [int(pair[0]) for pair in offsets[:total]]
    if len(token_start_chars) != total:
        raise ValueError("offset mapping shorter than the retained completion")
    starts = {start}
    for match in BLANK_LINE_PATTERN.finditer(completion):
        boundary = bisect.bisect_left(token_start_chars, match.end())
        if start < boundary < total:
            starts.add(boundary)
    return sorted(starts)


def block_masses(
    block_starts: Sequence[int],
    total_tokens: int,
    token_masses: Sequence[float],
    region_start: int,
) -> list[float]:
    """Sum per-token corrective signal within each block.

    ``token_masses`` is indexed by completion token, so ``region_start`` shifts
    block positions into that indexing.  Blocks are contiguous and cover the
    whole region, hence the block sums add up to the region total exactly.
    """
    starts = [int(value) for value in block_starts]
    total = int(total_tokens)
    ends = starts[1:] + [total]
    masses = []
    for start, end in zip(starts, ends):
        if not region_start <= start < end <= total:
            raise ValueError(f"block [{start},{end}) outside [{region_start},{total})")
        masses.append(math.fsum(float(value) for value in token_masses[start:end]))
    return masses


def select_coverage_endpoint(
    block_starts: Sequence[int],
    masses: Sequence[float],
    total_tokens: int,
    epsilon: float,
) -> tuple[int, float]:
    """Return ``(endpoint, discarded_mass_fraction)`` for one trajectory.

    Accumulating from the last block backwards keeps the small tail budget free
    of the subtraction cancellation a forward cumulative sum would introduce.
    The first block is never dropped, so the endpoint always exceeds the error
    anchor; a zero-signal region keeps the full suffix.
    """
    starts = [int(value) for value in block_starts]
    values = [float(value) for value in masses]
    total = int(total_tokens)
    epsilon = float(epsilon)
    if len(starts) != len(values) or not starts:
        raise ValueError("block starts and masses must be a non-empty aligned pair")
    if starts != sorted(set(starts)):
        raise ValueError("block starts must be strictly increasing")
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("block masses must be finite and non-negative")
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be in (0, 1)")
    mass_total = math.fsum(values)
    if mass_total == 0.0:
        return total, 0.0
    deleted = 0.0
    endpoint = total
    for index in range(len(values) - 1, 0, -1):
        next_deleted = math.fsum([deleted, values[index]])
        if next_deleted > epsilon * mass_total:
            break
        deleted = next_deleted
        endpoint = starts[index]
    return endpoint, deleted / mass_total
