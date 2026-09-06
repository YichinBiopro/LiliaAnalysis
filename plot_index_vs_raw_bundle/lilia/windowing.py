"""Timestamp contracts and sample windows shared by analysis pipelines."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class WindowGrid:
    """One sample grid shared by metrics, quality, and exported timestamps.

    End indexes and end timestamps are exclusive. The last sample's nominal
    period defines an end timestamp; centres use the actual centre sample.
    """
    n_samples: int
    fs: float
    win: int
    step: int
    columns: dict

    @property
    def starts(self):
        return self.columns['window_start_idx']

    @property
    def time_s(self):
        return self.columns['time_s']

    def validate(self, n_samples, fs, win, step):
        if (n_samples, fs, win, step) != (self.n_samples, self.fs, self.win, self.step):
            raise ValueError('Window grid does not match signal length or analysis settings')


def build_window_grid(time_us, fs, win_sec, step_sec=None):
    """Create complete windows on the original grid, skipping timestamp gaps."""
    step_sec = win_sec if step_sec is None else step_sec
    if not all(np.isfinite(v) and v > 0 for v in (fs, win_sec, step_sec)):
        raise ValueError('fs, window and step must be finite and positive')
    t = np.asarray(time_us)
    if t.dtype.kind not in 'iu':
        raise ValueError('Window timestamps must be integer microseconds')
    win, step = int(round(win_sec * fs)), int(round(step_sec * fs))
    if win < 8 or step < 1:
        raise ValueError('Analysis needs at least 8 samples per window and a positive step')
    segments = continuous_slices(t, fs)
    starts = np.asarray(list(window_starts(len(t), win, step, t, fs)), dtype=np.int64)
    if not len(starts):
        raise ValueError('No complete analysis window within any continuous segment')
    ends = starts + win
    centres = starts + win // 2
    segment_ids = np.searchsorted([s.stop for s in segments], starts, side='right')
    columns = {
        'window_start_idx': starts, 'window_end_idx': ends,
        'window_start_us': t[starts],
        'window_end_us': t[ends - 1] + int(round(1e6 / fs)),
        'window_center_us': t[centres], 'segment_id': segment_ids,
        'time_s': (t[centres] - t[0]) / 1e6,
    }
    for values in columns.values():
        values.flags.writeable = False
    return WindowGrid(len(t), fs, win, step, columns)


def finite_runs(values, segment_ids=None):
    """Return index runs separated by invalid samples or a recording gap."""
    values = np.asarray(values)
    valid = np.isfinite(values)
    if values.ndim > 1:
        valid = valid.all(axis=tuple(range(1, values.ndim)))
    idx = np.flatnonzero(valid)
    if not len(idx):
        return []
    split = np.diff(idx) != 1
    if segment_ids is not None:
        groups = np.asarray(segment_ids)
        if len(groups) != len(values):
            raise ValueError('Segment IDs and plotted values differ in length')
        split |= groups[idx[1:]] != groups[idx[:-1]]
    return np.split(idx, np.flatnonzero(split) + 1)


def transform_runs(values, transform, segment_ids=None):
    """Smooth each valid run without borrowing samples across a gap."""
    out = np.full(np.shape(values), np.nan, dtype=float)
    for run in finite_runs(values, segment_ids):
        out[run] = transform(np.asarray(values)[run])
    return out


def plot_breaks(x, y, segment_ids=None):
    """Insert plotting-only NaNs between segments, retaining every real point."""
    x, y = np.asarray(x), np.asarray(y, dtype=float)
    if segment_ids is None or len(x) < 2:
        return x, y
    cuts = np.flatnonzero(np.diff(segment_ids) != 0) + 1
    # Repeating the next timestamp works for both numeric and datetime axes.
    return np.insert(x, cuts, x[cuts]), np.insert(y, cuts, np.nan, axis=0)


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
