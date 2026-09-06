"""Timestamp contracts and sample windows shared by analysis pipelines."""
from __future__ import annotations

import numpy as np


def continuous_slices(time_us, fs, gap_factor=3.0):
    """Split strictly increasing microsecond timestamps at missing samples.

    Gaps larger than three nominal sample periods start a new segment. Duplicate
    timestamps are rejected rather than silently discarding conflicting samples.
    """
    t = np.asarray(time_us)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError('fs must be finite and positive')
    if t.ndim != 1 or not np.all(np.isfinite(t)):
        raise ValueError('timestamps must be a finite one-dimensional array')
    if len(t) == 0:
        return []
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError('timestamps must be strictly increasing; resolve collisions first')
    cuts = np.r_[0, np.flatnonzero(dt > gap_factor * 1e6 / fs) + 1, len(t)]
    return [slice(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:])]


def require_continuous(time_us, fs, context='analysis'):
    """Guard legacy algorithms which cannot yet represent disconnected epochs."""
    if len(continuous_slices(time_us, fs)) > 1:
        raise ValueError(f'{context} requires a continuous recording; split at timestamp gaps first')


def window_starts(n, win, step, time_us=None, fs=None):
    """Keep the original sample grid, excluding windows crossing a gap."""
    if win < 1 or step < 1:
        raise ValueError('window and step must contain at least one sample')
    if time_us is None:
        yield from range(0, n - win + 1, step)
        return
    if len(time_us) != n:
        raise ValueError('timestamp and signal lengths differ')
    for sl in continuous_slices(time_us, fs):
        first = ((sl.start + step - 1) // step) * step
        yield from range(first, sl.stop - win + 1, step)


def elapsed_window_times(time_us, starts, win):
    t = np.asarray(time_us, dtype=np.int64)
    centres = np.asarray(starts, dtype=int) + win // 2
    return (t[centres] - t[0]) / 1e6


def time_slice(time_us, lo_us, hi_us):
    """Half-open physical-time interval; never convert wall time using fs."""
    t = np.asarray(time_us)
    return slice(int(np.searchsorted(t, lo_us)), int(np.searchsorted(t, hi_us)))
