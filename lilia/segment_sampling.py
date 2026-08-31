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
    """
    if n_total <= 0 or seg_len <= 0 or n_segs <= 0:
        return []

    max_start = n_total - seg_len
    if max_start < 0:
        return []

    starts: list[int] = []
    attempts = 0
    while len(starts) < n_segs and attempts < max_attempts:
        attempts += 1
        s = int(rng.integers(0, max_start + 1))
        if all(abs(s - prev) >= seg_len for prev in starts):
            starts.append(s)
    starts.sort()
    return starts
