"""Single raw-channel qEEG on an auditable, gap-aware source sample grid.

The legacy helper remains timestamp-free. This CLI adapter retains its truncated
window size and original sample lattice, and uses actual centre timestamps.
"""
from __future__ import annotations

import numpy as np

from lilia.qeeg import compute_qeeg_indices, BAND_THETA, BAND_ALPHA, BAND_BETA, EPSILON
from lilia.windowing import build_window_grid, continuous_slices

INDEX_KEYS = ('focus', 'flow', 'calm', 'relaxation')
METRIC_KEYS = ('theta', 'alpha', 'beta', *INDEX_KEYS)


def analysis_parameters(fs, win_sec, channel):
    if not all(np.isfinite(v) and v > 0 for v in (fs, win_sec)):
        raise ValueError('fs and window must be finite and positive')
    if not np.isfinite(fs * win_sec) or int(fs * win_sec) < 8:
        raise ValueError('qEEG needs at least 8 samples per window')
    if not isinstance(channel, (int, np.integer)) or channel < 1:
        raise ValueError('channel must be a positive 1-based integer')
    return {
        'fs': float(fs), 'win_sec': float(win_sec), 'step_sec': float(win_sec),
        'window_samples': int(fs * win_sec), 'channel': int(channel),
        'index_space': 'raw_samples', 'grid_policy': 'original_sample_lattice',
        'window_rounding': 'truncate', 'gap_factor': 3.,
        'time_axis': 'elapsed_from_first_source_sample',
        'quality_state': 'disabled', 'preprocessing': 'none', 'baseline': 'none',
        'bands': {'theta': list(BAND_THETA), 'alpha': list(BAND_ALPHA), 'beta': list(BAND_BETA)},
        'band_edges': 'half_open', 'epsilon': EPSILON,
        'method': 'legacy_qeeg_welch_and_four_indices_v1',
    }


def analyze_raw_qeeg(time_us, raw, *, fs=500., win_sec=5., channel=1):
    """Retain all legal rows, with NaNs and reasons for excluded raw windows."""
    p = analysis_parameters(fs, win_sec, channel)
    t, raw = np.asarray(time_us), np.asarray(raw, dtype=float)
    if t.ndim != 1 or t.dtype.kind not in 'iu' or not len(t):
        raise ValueError('qEEG needs nonempty integer microsecond timestamps')
    if raw.ndim != 2 or raw.shape[0] != len(t):
        raise ValueError('qEEG needs raw channels aligned with timestamps')
    if channel > raw.shape[1]:
        raise ValueError(f'channel {channel} not found (file has {raw.shape[1]} channels)')
    slices = continuous_slices(t, fs)
    win = p['window_samples']
    # WindowGrid rounds its seconds argument; supplying the effective duration
    # preserves the legacy helper's int(win_sec * fs), including odd windows.
    try:
        grid = build_window_grid(t, fs, win / fs)
    except ValueError as exc:
        if 'No complete analysis window' not in str(exc):
            raise
        grid = None
    starts = np.array([], dtype=np.int64) if grid is None else grid.starts
    metrics = {key: np.full(len(starts), np.nan) for key in METRIC_KEYS}
    windows, segments = [], []
    x = raw[:, channel - 1]
    for row, start in enumerate(starts):
        block = x[start:start + win]
        record = {'metric_row': row, 'status': 'computed'}
        if not np.isfinite(block).all():
            record.update(status='nonfinite_window', nonfinite_samples=int((~np.isfinite(block)).sum()))
        else:
            try:
                values = compute_qeeg_indices(block, fs=fs)
                if not all(np.isfinite(values[key]) for key in METRIC_KEYS):
                    record['status'] = 'nonfinite_metric'
                else:
                    for key in METRIC_KEYS:
                        metrics[key][row] = values[key]
            except Exception as exc:
                record.update(status='metric_error', error=f'{type(exc).__name__}: {exc}')
        windows.append(record)
    period_us = int(round(1e6 / fs))
    for sid, sl in enumerate(slices):
        rows = np.flatnonzero((starts >= sl.start) & (starts + win <= sl.stop))
        first = int(starts[rows[0]]) if len(rows) else sl.start
        end = int(starts[rows[-1]]) + win if len(rows) else sl.start
        segments.append({
            'segment_id': sid, 'raw_start_idx': sl.start, 'raw_end_idx': sl.stop,
            'raw_start_us': int(t[sl.start]), 'raw_end_us': int(t[sl.stop - 1]) + period_us,
            'gap_before_us': 0 if sid == 0 else int(t[sl.start] - t[sl.start - 1]) - period_us,
            'metric_rows': rows.tolist(), 'complete_windows': len(rows),
            'status': 'windowed' if len(rows) else ('short_segment' if sl.stop-sl.start < win else 'no_grid_window'),
            'excluded_prefix_start_idx': sl.start, 'excluded_prefix_end_idx': first,
            'tail_start_idx': end, 'tail_end_idx': sl.stop, 'tail_samples': sl.stop - end,
            'unwindowed_samples': sl.stop - sl.start - len(rows) * win,
            'nonfinite_samples': int((~np.isfinite(x[sl])).sum()),
        })
    accepted = set(starts.tolist())
    crossed = [{'window_start_idx': start, 'window_end_idx': start + win,
                'status': 'crosses_gap'} for start in range(0, len(t) - win + 1, win)
               if start not in accepted]
    valid = np.array([w['status'] == 'computed' for w in windows], dtype=bool)
    summary = {key: {'mean': float(metrics[key][valid].mean()),
                     'std': float(metrics[key][valid].std())} if valid.any()
               else {'mean': None, 'std': None} for key in INDEX_KEYS}
    analysis = {
        'status': 'success' if valid.any() else ('no_complete_windows' if grid is None else 'all_windows_excluded'),
        'source_channels': raw.shape[1], 'segments': segments,
        'window_audit': windows, 'excluded_grid_windows': crossed,
        'candidate_windows': len(starts), 'finite_windows': int(valid.sum()),
        'quality_state': 'disabled', 'summary': summary,
    }
    return {'grid': grid, 'metrics': metrics, 'valid': valid, 'analysis': analysis, 'parameters': p}
