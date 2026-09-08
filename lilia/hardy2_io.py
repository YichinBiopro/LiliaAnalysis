"""Source-verifiable Hardy_2 raw-window metrics and period summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lilia.entropy_io import _write_window_table, _load_window_table, config_id
from lilia.event_qeeg_io import json_safe
from lilia.hardy2 import METRIC_KEYS, BANDS, window_layout, smooth_metrics, summarize_periods
from lilia.io import read_lilia_frame
from lilia.provenance import file_sha256


def write_hardy2_table(path, source, result, parameters, code_id):
    frame = pd.DataFrame(result['metrics'])
    for key in METRIC_KEYS:
        frame[f'{key}_smooth'] = result['smooth'][key]
        for ch in range(result['channels'][key].shape[1]):
            frame[f'{key}_ch{ch+1}'] = result['channels'][key][:, ch]
    frame['metric_status'] = [r['status'] for r in result['window_audit']]
    frame['metric_valid'] = result['valid']
    frame['quality_state'] = 'disabled'
    analysis = json_safe({k: result[k] for k in ('segments', 'window_audit', 'summary')})
    _write_window_table(path, source, frame, result['grid'], parameters, code_id,
                        'hardy2_band_indices', source_info={'analysis': analysis, 'analysis_id': config_id(analysis)})


def _equivalent(actual, expected):
    """Compare reconstructed JSON, allowing only CSV round-trip float precision."""
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and actual.keys()==expected.keys()
                and all(_equivalent(actual[k], v) for k, v in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual)==len(expected)
                and all(_equivalent(a, b) for a, b in zip(actual, expected)))
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and np.isclose(actual, expected, rtol=1e-12, atol=1e-12)
    return actual == expected


def load_hardy2_table(path, raw_csv):
    frame, meta = _load_window_table(path, kind='hardy2_band_indices')
    if meta is None:
        raise ValueError('Hardy_2 source metadata required')
    p = meta['parameters']
    if (p['index_space']!='raw_samples' or p['bands']!={k:list(v) for k,v in BANDS.items()}
            or p['quality_state']!='disabled' or p['period_policy']!='complete_contained_windows'
            or p['step_sec']!=p['win_sec'] or p['smooth_windows']!=5
            or meta['analysis_id']!=config_id(meta['analysis'])):
        raise ValueError('Hardy_2 analysis or policy fingerprint mismatch')
    if file_sha256(raw_csv)!=meta['source_id']:
        raise ValueError('Raw recording differs from Hardy_2 source')
    raw = read_lilia_frame(raw_csv)
    t = raw.iloc[:, 0].to_numpy(dtype=np.int64)
    if len(raw)!=meta['source_samples'] or int(t[0])!=meta['source_epoch_us'] or len(raw.columns)-1!=p['channels']:
        raise ValueError('Hardy_2 source shape or epoch mismatch')
    grid, segments = window_layout(t, p['fs'], p['win_sec'])
    if grid is None or len(frame)!=len(grid.starts):
        raise ValueError('Hardy_2 window count differs from source')
    for key, expected in grid.columns.items():
        actual = frame[key].to_numpy()
        same = np.allclose(actual, expected, rtol=0, atol=1e-9) if key=='time_s' else np.array_equal(actual, expected)
        if not same:
            raise ValueError(f'Hardy_2 {key} mapping differs from source')
    stored = meta['analysis']
    if len(stored['segments'])!=len(segments):
        raise ValueError('Hardy_2 segment count differs from source')
    for structural, saved in zip(segments, stored['segments']):
        if any(saved[k]!=v for k, v in structural.items()):
            raise ValueError('Hardy_2 segment mapping differs from source')
    statuses = [r['status'] for r in stored['window_audit']]
    if (statuses!=frame.metric_status.tolist() or not np.array_equal(frame.metric_valid, frame.metric_status=='computed')
            or [r['metric_row'] for r in stored['window_audit']]!=list(range(len(frame)))
            or not (frame.quality_state=='disabled').all()):
        raise ValueError('Hardy_2 metric status differs from audit')
    metrics = {k: frame[k].to_numpy() for k in METRIC_KEYS}
    for key in METRIC_KEYS:
        channels = frame[[f'{key}_ch{i+1}' for i in range(p['channels'])]].to_numpy()
        if (not np.array_equal(np.isfinite(channels).all(axis=1), frame.metric_valid)
                or not np.allclose(metrics[key], np.median(channels,axis=1), rtol=1e-12, atol=1e-12, equal_nan=True)):
            raise ValueError('Hardy_2 channel medians differ from table')
    smooth = smooth_metrics(metrics, grid.columns['segment_id'])
    for key, values in smooth.items():
        if not np.allclose(values, frame[f'{key}_smooth'], rtol=1e-12, atol=1e-12, equal_nan=True):
            raise ValueError('Hardy_2 smoothing differs from source runs')
    summary = json_safe(summarize_periods(grid, metrics, p['event_us']))
    if not _equivalent(stored['summary'], summary):
        raise ValueError('Hardy_2 period summary differs from source windows')
    return frame, meta
