"""Hardy_2 five-band metrics with source-local windows and complete periods."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal

from lilia.io import bandpass_filter
from lilia.qeeg import compute_qeeg_indices
from lilia.windowing import build_window_grid, continuous_slices, finite_runs, transform_runs

BANDS = {'delta': (1., 4.), 'theta': (4., 8.), 'alpha': (8., 13.),
         'beta': (13., 30.), 'gamma': (30., 45.)}
INDEX_KEYS = ('focus', 'flow', 'calm', 'relaxation')
METRIC_KEYS = (*BANDS, *INDEX_KEYS)
PERIOD_NAMES = ('Pre-16:18', '16:18-16:21', '16:21-16:36', 'Post-16:36')


def bandpower(freqs, psd, low, high):
    mask = (freqs >= low) & (freqs <= high)
    return float(np.trapezoid(psd[mask], freqs[mask])) if mask.any() else 0.


def window_layout(time_us, fs, win_sec):
    """Audit every source segment, including unanalysed tails and short segments."""
    t = np.asarray(time_us)
    if t.dtype.kind not in 'iu' or not len(t):
        raise ValueError('Hardy_2 needs nonempty integer microsecond timestamps')
    try:
        grid = build_window_grid(t, fs, win_sec, reset_per_segment=True)
    except ValueError as exc:
        if 'No complete analysis window' not in str(exc):
            raise
        grid = None
    win = int(round(fs*win_sec))
    segments = []
    for sid, sl in enumerate(continuous_slices(t, fs)):
        n = sl.stop-sl.start
        segments.append({'segment_id': sid, 'raw_start_idx': sl.start, 'raw_end_idx': sl.stop,
            'raw_start_us': int(t[sl.start]), 'raw_end_us': int(t[sl.stop-1])+int(round(1e6/fs)),
            'complete_windows': n//win, 'tail_start_idx': sl.start+(n//win)*win,
            'tail_samples': n % win,
            'metric_rows': [] if grid is None else np.flatnonzero(grid.columns['segment_id']==sid).tolist()})
    return grid, segments


def window_metrics(window, fs):
    """Keep the historical five-band denominator and separate qEEG formulas."""
    values = {k: [] for k in METRIC_KEYS}
    for ch in range(window.shape[1]):
        x = window[:, ch].astype(np.float64)
        f, psd = signal.welch(x, fs=fs, nperseg=min(len(x), int(fs*4)),
                             noverlap=min(len(x)//2, int(fs*2)), window='hann')
        powers = {k: bandpower(f, psd, *bounds) for k, bounds in BANDS.items()}
        denominator = sum(powers.values())+1e-12
        for k in BANDS:
            values[k].append(powers[k]/denominator)
        indices = compute_qeeg_indices(x, fs=fs)
        for k in INDEX_KEYS:
            values[k].append(float(indices[k]))
    return {k: np.asarray(v) for k, v in values.items()}


def analyze_hardy2(time_us, raw, *, fs=500., win_sec=5., use_bandpass=True, low=.5, high=45.):
    t, raw = np.asarray(time_us), np.asarray(raw)
    if raw.ndim != 2 or raw.shape[0] != len(t) or raw.shape[1] < 1:
        raise ValueError('Hardy_2 needs aligned raw channels')
    if use_bandpass and not (np.isfinite(low) and np.isfinite(high) and 0 < low < high < fs/2):
        raise ValueError('Invalid Hardy_2 bandpass bounds')
    grid, segments = window_layout(t, fs, win_sec)
    n = 0 if grid is None else len(grid.starts)
    channel_values = {k: np.full((n, raw.shape[1]), np.nan) for k in METRIC_KEYS}
    windows = [{'metric_row': i, 'status': 'not_computed'} for i in range(n)]
    for row in segments:
        if not row['metric_rows']:
            row['status'] = 'short_segment'
            continue
        part = raw[row['raw_start_idx']:row['raw_end_idx']]
        if use_bandpass:
            if not np.isfinite(part).all():
                row['status'] = 'nonfinite_segment'
            else:
                try:
                    part = bandpass_filter(part, fs=fs, lo=low, hi=high)
                    if not np.isfinite(part).all():
                        raise ValueError('Non-finite bandpass output')
                    row['status'] = 'filtered'
                except ValueError as exc:
                    row.update(status='filter_error', error=str(exc))
        else:
            row['status'] = 'unfiltered'
        for i in row['metric_rows']:
            if row['status'] not in ('filtered', 'unfiltered'):
                windows[i]['status'] = row['status']
                continue
            start = int(grid.starts[i])-row['raw_start_idx']
            window = part[start:start+grid.win]
            if not np.isfinite(window).all():
                windows[i]['status'] = 'nonfinite_window'
                continue
            try:
                values = window_metrics(window, fs)
                if any(v.shape != (raw.shape[1],) or not np.isfinite(v).all() for v in values.values()):
                    raise ValueError('Invalid Hardy_2 metric output')
                for k in METRIC_KEYS:
                    channel_values[k][i] = values[k]
                windows[i]['status'] = 'computed'
            except ValueError as exc:
                windows[i].update(status='metric_error', error=str(exc))
    metrics = {k: np.median(v, axis=1) for k, v in channel_values.items()}
    smooth = smooth_metrics(metrics, None if grid is None else grid.columns['segment_id'])
    return {'grid': grid, 'segments': segments, 'window_audit': windows,
            'channels': channel_values, 'metrics': metrics, 'smooth': smooth,
            'valid': np.array([w['status']=='computed' for w in windows], dtype=bool)}


def smooth_metrics(metrics, groups):
    return {k: transform_runs(v, lambda x: pd.Series(x).rolling(
        5, center=True, min_periods=1).mean().to_numpy(), groups) for k, v in metrics.items()}


def summarize_periods(grid, metrics, event_us):
    """Mean unsmoothed channel medians for windows fully inside half-open periods.

    Spans address valid adjacent windows in one source segment. Unassigned rows
    cross an event boundary; invalid candidate rows stay visible in the audit.
    """
    edges = np.asarray(event_us)
    if edges.shape != (3,) or edges.dtype.kind not in 'iu' or np.any(np.diff(edges)<=0):
        raise ValueError('Hardy_2 periods need three increasing integer event timestamps')
    n = 0 if grid is None else len(grid.starts)
    valid = np.ones(n, dtype=bool)
    for key in METRIC_KEYS:
        values = np.asarray(metrics[key])
        if values.shape != (n,):
            raise ValueError('Hardy_2 metrics do not match window grid')
        valid &= np.isfinite(values)
    assigned = np.zeros(n, dtype=bool)
    periods = []
    for name, lo, hi in zip(PERIOD_NAMES, [None, *map(int, edges)], [*map(int, edges), None]):
        mask = np.ones(n, dtype=bool)
        if grid is not None:
            if lo is not None:
                mask &= grid.columns['window_start_us']>=lo
            if hi is not None:
                mask &= grid.columns['window_end_us']<=hi
        rows = np.flatnonzero(mask)
        accepted = np.flatnonzero(mask & valid)
        assigned |= mask
        spans = []
        if grid is not None:
            for run in finite_runs(np.where(mask & valid, 1., np.nan), grid.columns['segment_id']):
                spans.append({'start_us': int(grid.columns['window_start_us'][run[0]]),
                              'end_us': int(grid.columns['window_end_us'][run[-1]]),
                              'segment_id': int(grid.columns['segment_id'][run[0]]),
                              'metric_rows': run.tolist()})
        periods.append({'name': name, 'start_us': lo, 'end_us': hi,
            'candidate_rows': rows.tolist(), 'accepted_rows': accepted.tolist(),
            'excluded_rows': np.flatnonzero(mask & ~valid).tolist(), 'spans': spans,
            'status': 'computed' if len(accepted) else ('all_invalid' if len(rows) else 'no_complete_windows'),
            'means': {k: float(np.mean(v[accepted])) if len(accepted) else None for k, v in metrics.items()}})
    return {'periods': periods, 'boundary_crossing_rows': np.flatnonzero(~assigned).tolist()}
