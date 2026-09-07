"""Physical-time state intervals and auditable, segment-local entropy windows."""
from __future__ import annotations

import numpy as np

from lilia.windowing import continuous_slices


def select_state_windows(time_us, fs, lo_us, hi_us, win_sec, step_sec=None):
    """Restart at the first in-range sample of each segment; never bridge gaps.

    Only complete windows whose nominal exclusive end is <= hi_us contribute.
    Trailing grid candidates are retained as incomplete_window audit rows.
    Missing physical-time spans (including outside the recording) are explicit.
    """
    step_sec = win_sec if step_sec is None else step_sec
    if not all(np.isfinite(v) and v > 0 for v in (fs, win_sec, step_sec)):
        raise ValueError('fs, window and step must be finite and positive')
    if not all(np.isfinite(v) for v in (lo_us, hi_us)) or hi_us <= lo_us:
        raise ValueError('State interval end must be finite and greater than start')
    t = np.asarray(time_us)
    if t.ndim != 1 or t.dtype.kind not in 'iu' or not len(t):
        raise ValueError('State timestamps must be nonempty integer microseconds')
    win, step = int(round(fs * win_sec)), int(round(fs * step_sec))
    if win < 8 or step < 1:
        raise ValueError('Analysis needs at least 8 samples and a positive step')
    period = int(round(1e6 / fs))
    rows, missing = [], []
    cursor = int(lo_us)
    for group, sl in enumerate(continuous_slices(t, fs)):
        left, right = max(int(lo_us), int(t[sl.start])), min(int(hi_us), int(t[sl.stop-1]) + period)
        if right <= left:
            continue
        if left > cursor:
            missing.append({'start_us': cursor, 'end_us': left, 'reason': 'no_recorded_samples'})
        cursor = max(cursor, right)
        a = max(sl.start, int(np.searchsorted(t, lo_us)))
        b = min(sl.stop, int(np.searchsorted(t, hi_us)))
        for start in range(a, b, step):
            end = min(start + win, b)
            end_us = int(t[end-1]) + period
            complete = end-start == win and end_us <= hi_us
            rows.append({
                'window_start_idx': start, 'window_end_idx': end,
                'window_start_us': int(t[start]), 'window_end_us': end_us,
                'window_center_us': int(t[start + (end-start)//2]),
                'segment_id': group, 'complete': bool(complete),
                'status': 'candidate' if complete else 'incomplete_window',
            })
    if cursor < hi_us:
        missing.append({'start_us': cursor, 'end_us': int(hi_us), 'reason': 'no_recorded_samples'})
    return {'lo_us': int(lo_us), 'hi_us': int(hi_us), 'windows': rows, 'missing_spans': missing}


def state_bounds(time_us, interval):
    if not len(time_us):
        raise ValueError('State timestamps must be nonempty')
    if len(interval) != 2 or not all(np.isfinite(v) for v in interval) or interval[1] <= interval[0]:
        raise ValueError('State range end must be finite and greater than start')
    return tuple(int(time_us[0]) + int(round(float(v) * 1e6)) for v in interval)
