"""Shared utilities for random non-overlapping segment selection."""

from __future__ import annotations

import numpy as np


def pick_non_overlapping_segments(
    n_total: int,
    seg_len: int,
    n_segs: int,
    rng: np.random.Generator,
    max_attempts: int = 10_000,
) -> list[int]:
    """Pick non-overlapping segment start indices from one recording.

    Returns sorted start indices in ``[0, n_total - seg_len]``.
    If the recording is shorter than one segment, returns an empty list.
    Returns the requested count whenever capacity permits. ``max_attempts`` is
    retained for call compatibility; the constructive algorithm needs no retry.
    """
    if n_total <= 0 or seg_len <= 0 or n_segs <= 0:
        return []

    # Distribute slack around k non-overlapping windows. Sampling sorted
    # offsets is uniform over feasible ordered start configurations.
    k = min(n_segs, n_total // seg_len)
    if k == 0:
        return []
    slack = n_total - k * seg_len
    bars = np.sort(rng.choice(slack + k, size=k, replace=False))
    return (bars - np.arange(k) + np.arange(k) * seg_len).astype(int).tolist()
