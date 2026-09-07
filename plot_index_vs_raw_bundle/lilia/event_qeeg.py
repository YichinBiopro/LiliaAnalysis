"""Segment-local event qEEG and explicit baseline/heatmap selection policies."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lilia.io import bandpass_filter
from lilia.qeeg import compute_qeeg_indices
from lilia.tflite import build_tflite_timeline, run_tflite_recording
from lilia.windowing import build_window_grid, continuous_slices, transform_runs

INDEX_KEYS = ('focus', 'flow', 'calm', 'relaxation')


def raw_quality_mapping(time_us, grid, *, raw_index_space=False):
    """Map each metric's physical interval to raw samples, never by row position."""
    if raw_index_space:
        return {'quality_raw_start_idx': grid.starts,
                'quality_raw_end_idx': grid.columns['window_end_idx']}
    return {key: np.searchsorted(time_us, grid.columns[bound]).astype(np.int64)
            for key, bound in [('quality_raw_start_idx', 'window_start_us'),
                               ('quality_raw_end_idx', 'window_end_us')]}


def score_branch(time_us, raw, data, grid, fs, raw_fs, scorer, params, threshold, *, raw_index_space=False):
    mapping = raw_quality_mapping(time_us, grid, raw_index_space=raw_index_space)
    quality = np.full((len(grid.starts), raw.shape[1]), np.nan)
    scores = {k: np.full((len(grid.starts), data.shape[1]), np.nan) for k in INDEX_KEYS}
    audit = []
    for i, start in enumerate(grid.starts):
        a, b = (int(mapping[k][i]) for k in ('quality_raw_start_idx', 'quality_raw_end_idx'))
        row = {'metric_row': i, 'quality_status': 'not_scored', 'metric_status': 'not_computed'}
        segment = raw[a:b]
        if len(segment) < 8 or not np.isfinite(segment).all():
            row['quality_status'] = 'nonfinite_or_short_raw'
        else:
            try:
                q = np.asarray(scorer(segment.T.astype(np.float64), fs=raw_fs, params=params)['overall'])
                if q.shape != (raw.shape[1],) or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
                    row['quality_status'] = 'invalid_quality'
                else:
                    quality[i] = q
                    row['quality_status'] = 'accepted' if np.median(q) >= threshold else 'low_quality'
            except Exception as exc:
                row.update(quality_status='quality_error', quality_error=str(exc))
        window = data[start:start + grid.win]
        if not np.isfinite(window).all():
            row['metric_status'] = 'nonfinite_filtered_segment'
        else:
            for channel in range(data.shape[1]):
                values = compute_qeeg_indices(window[:, channel].astype(np.float64), fs=fs)
                for key in INDEX_KEYS:
                    scores[key][i, channel] = values[key]
            row['metric_status'] = 'computed'
        audit.append(row)
    valid = np.isfinite(quality).all(axis=1) & (np.median(quality, axis=1) >= threshold)
    for values in scores.values():
        valid &= np.isfinite(values).all(axis=1)
    return {'grid': grid, 'quality': quality, 'scores': scores, 'valid': valid,
            'quality_mapping': mapping, 'window_audit': audit}


def analyze_recording(time_us, raw, *, fs=500., win_sec=5., low=.5, high=45.,
                      scorer, quality_params, threshold=.5, model_path=None,
                      model_fs=200., model_window=400):
    """Compute independent BP/model grids; model windows retain source segment IDs."""
    t, raw = np.asarray(time_us), np.asarray(raw)
    if t.dtype.kind not in 'iu' or raw.ndim != 2 or len(t) != len(raw) or not len(t):
        raise ValueError('Expected nonempty integer timestamps and aligned raw channels')
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Quality threshold must be within [0, 1]')
    segments = continuous_slices(t, fs)
    win = int(round(fs * win_sec))
    result = {'bp': None, 'tflite': None, 'timeline': None, 'errors': [], 'segments': []}
    try:
        grid = build_window_grid(t, fs, win_sec, reset_per_segment=True)
    except ValueError as exc:
        # Validate invalid settings normally; only no-window input is auditable here.
        if 'No complete analysis window' not in str(exc):
            raise
        grid = None
        result['errors'].append(str(exc))
    timeline = build_tflite_timeline(t, fs, model_fs, model_window) if model_path is not None else None
    filtered = np.full(raw.shape, np.nan, dtype=np.float32)
    for sid, sl in enumerate(segments):
        length = sl.stop - sl.start
        row = {'segment_id': sid, 'raw_start_idx': sl.start, 'raw_end_idx': sl.stop,
               'raw_start_us': int(t[sl.start]), 'raw_end_us': int(t[sl.stop-1]) + int(round(1e6/fs)),
               'bp_complete_windows': length // win, 'bp_tail_samples': length % win}
        needed = length >= win or (timeline is not None and timeline.segments[sid]['retained_samples'] > 0)
        if not needed:
            row['status'] = 'short_segment'
        elif not np.isfinite(raw[sl]).all():
            row['status'] = 'nonfinite_segment'
        else:
            try:
                filtered[sl] = bandpass_filter(raw[sl], fs=fs, lo=low, hi=high)
                row['status'] = 'filtered'
            except ValueError as exc:
                row.update(status='filter_error', error=str(exc))
        result['segments'].append(row)
    if grid is not None:
        result['bp'] = score_branch(t, raw, filtered, grid, fs, fs, scorer, quality_params, threshold,
                                     raw_index_space=True)
    if timeline is not None:
        result['timeline'] = timeline
        try:
            timeline, _, output = run_tflite_recording(t, filtered, model_path, fs, model_fs, model_window)
            result['tflite'] = score_branch(t, raw, output, timeline.grid(win_sec), model_fs,
                                             fs, scorer, quality_params, threshold)
        except Exception as exc:
            result['errors'].append(f'TFLite failed: {exc}')
    return result


def complete_mask(starts, ends, lo=None, hi=None):
    mask = np.ones(len(starts), dtype=bool)
    if lo is not None:
        mask &= starts >= lo
    if hi is not None:
        mask &= ends <= hi
    return mask


def summarize_branch(branch, events, baseline_mode='session-start', bin_size=6):
    """Keep mean channel deltas and median heatmaps distinct; missing stays NaN.

    session-start uses the same pre-first-participating-event interval for all
    events. With no participating events it uses the first max(1, n_bins//5)
    complete bins. pre-event-rest never falls back to an earlier event period.
    Both baseline and event selections require complete contained windows/bins.
    """
    if baseline_mode not in ('session-start', 'pre-event-rest'):
        raise ValueError('Unknown baseline mode')
    grid, scores, valid = branch['grid'], branch['scores'], branch['valid']
    groups = grid.columns['segment_id']
    smooth = {}
    for key in INDEX_KEYS:
        series = np.where(valid, np.median(scores[key], axis=1), np.nan)
        smooth[key] = transform_runs(series, lambda x: pd.Series(x).rolling(
            bin_size, center=True, min_periods=1).mean().to_numpy(), groups)
    smooth['restfulness'] = (smooth['calm'] + smooth['relaxation']) / 2
    smooth['engagement'] = smooth['focus'] - smooth['restfulness']
    bins = []
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        for start in range(0, len(rows)-bin_size+1, bin_size):
            chunk = rows[start:start+bin_size]
            lo, hi = int(grid.columns['window_start_us'][chunk[0]]), int(grid.columns['window_end_us'][chunk[-1]])
            bins.append({'segment_id': int(group), 'metric_rows': chunk.tolist(),
                         'start_us': lo, 'end_us': hi, 'center_us': (lo+hi)//2,
                         'valid_rows': chunk[valid[chunk]].tolist()})
    absolute = np.full((len(INDEX_KEYS), len(bins)), np.nan)
    for j, item in enumerate(bins):
        rows = item['valid_rows']
        if rows:
            for i, key in enumerate(INDEX_KEYS):
                absolute[i, j] = np.median(np.median(scores[key][rows], axis=1))
    starts = np.asarray([b['start_us'] for b in bins], dtype=np.int64)
    ends = np.asarray([b['end_us'] for b in bins], dtype=np.int64)
    ws, we = grid.columns['window_start_us'], grid.columns['window_end_us']
    ordered = sorted(events, key=lambda e: e['start_us'])
    participating = [e for e in ordered if e['participates']]
    first = min((e['start_us'] for e in participating), default=None)
    specs, previous_end = [], None
    for event in ordered:
        if event['participates']:
            specs.append((event, previous_end if baseline_mode == 'pre-event-rest' else None,
                          event['start_us'] if baseline_mode == 'pre-event-rest' else first))
        previous_end = max(previous_end or event['end_us'], event['end_us'])
    delta = np.full_like(absolute, np.nan)
    baseline_audit, block_deltas = [], []
    for event, lo, hi in specs:
        bm = complete_mask(ws, we, lo, hi) & valid
        em = complete_mask(ws, we, event['start_us'], event['end_us']) & valid
        bb = complete_mask(starts, ends, lo, hi)
        eb = complete_mask(starts, ends, event['start_us'], event['end_us'])
        refs = []
        for k in range(len(INDEX_KEYS)):
            values = absolute[k, bb]
            ref = float(np.median(values[np.isfinite(values)])) if np.isfinite(values).any() else np.nan
            refs.append(ref)
            target = (bb | eb) if baseline_mode == 'pre-event-rest' else np.ones(len(bins), dtype=bool)
            delta[k, target] = absolute[k, target] - ref
        per_index = {}
        for key in INDEX_KEYS:
            if bm.any() and em.any():
                channel_delta = scores[key][em].mean(axis=0) - scores[key][bm].mean(axis=0)
                per_index[key] = (float(channel_delta.mean()), float(channel_delta.std()))
            else:
                per_index[key] = (np.nan, np.nan)
        block_deltas.append((event['label'], event['color'], per_index))
        baseline_audit.append({'event': event['label'], 'start_us': lo, 'end_us': hi,
            'baseline_metric_rows': np.flatnonzero(bm).tolist(), 'event_metric_rows': np.flatnonzero(em).tolist(),
            'baseline_bins': np.flatnonzero(bb).tolist(), 'event_bins': np.flatnonzero(eb).tolist(),
            'heatmap_reference': refs, 'bar_status': 'accepted' if bm.any() and em.any() else 'missing_baseline_or_event',
            'heatmap_status': 'accepted' if np.isfinite(refs).all() else 'missing_baseline'})
    if not participating:
        selected = np.arange(min(len(bins), max(1, len(bins)//5)))
        refs = []
        for k in range(len(INDEX_KEYS)):
            values = absolute[k, selected]
            ref = float(np.median(values[np.isfinite(values)])) if np.isfinite(values).any() else np.nan
            refs.append(ref)
            delta[k] = absolute[k] - ref
        baseline_audit.append({'event': None, 'policy': 'first_fifth_complete_bins',
            'baseline_bins': selected.tolist(), 'heatmap_reference': refs,
            'heatmap_status': 'accepted' if np.isfinite(refs).all() else 'missing_baseline'})
    return {'smooth': smooth, 'bins': bins, 'heatmap_abs': absolute, 'heatmap_delta': delta,
            'block_deltas': block_deltas, 'baseline_audit': baseline_audit}
