"""Raw qEEG table provenance, source reconstruction and numerical verification."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lilia.entropy_io import _write_window_table, _load_window_table, config_id
from lilia.io import read_lilia_frame
from lilia.provenance import file_sha256
from lilia.qeeg_raw import analyze_raw_qeeg, analysis_parameters, METRIC_KEYS


def write_qeeg_table(path, source, result, code_id):
    frame = pd.DataFrame(result['metrics'])
    frame['metric_status'] = [r['status'] for r in result['analysis']['window_audit']]
    frame['metric_valid'] = result['valid']
    frame['quality_state'] = 'disabled'
    frame['channel'] = result['parameters']['channel']
    analysis = result['analysis']
    _write_window_table(path, source, frame, result['grid'], result['parameters'], code_id,
                        'raw_qeeg', source_info={'analysis': analysis, 'analysis_id': config_id(analysis)})


def load_qeeg_table(path, raw_csv, channel=None):
    """Require metadata and raw source; reconstruct every window and metric."""
    frame, meta = _load_window_table(path, channel=channel, kind='raw_qeeg')
    if meta is None:
        raise ValueError('Raw qEEG metadata required')
    p = meta['parameters']
    if p != analysis_parameters(p['fs'], p['win_sec'], p['channel']):
        raise ValueError('Raw qEEG policy mismatch')
    if file_sha256(raw_csv) != meta['source_id']:
        raise ValueError('Raw recording differs from qEEG source')
    raw = read_lilia_frame(raw_csv)
    t = raw.iloc[:, 0].to_numpy(dtype=np.int64)
    result = analyze_raw_qeeg(t, raw.iloc[:, 1:].to_numpy(dtype=float),
                             fs=p['fs'], win_sec=p['win_sec'], channel=p['channel'])
    grid = result['grid']
    if (grid is None or len(frame) != len(grid.starts) or len(t) != meta['source_samples']
            or int(t[0]) != meta['source_epoch_us']):
        raise ValueError('Raw qEEG source shape, epoch or window count mismatch')
    for key, expected in grid.columns.items():
        actual = frame[key].to_numpy()
        if key != 'time_s' and actual.dtype.kind not in 'iu':
            raise ValueError(f'Raw qEEG {key} must contain integers')
        same = (np.allclose(actual, expected, rtol=0, atol=1e-9) if key == 'time_s'
                else np.array_equal(actual, expected))
        if not same:
            raise ValueError(f'Raw qEEG {key} differs from source')
    analysis = meta['analysis']
    if meta['analysis_id'] != config_id(analysis):
        raise ValueError('Raw qEEG analysis fingerprint mismatch')
    # Compare structural audit exactly; summary alone permits CSV/BLAS-level
    # floating point round-trip tolerance, like the metrics themselves.
    for key, expected in result['analysis'].items():
        if key == 'summary':
            for metric, stats in expected.items():
                for stat, value in stats.items():
                    saved = analysis[key][metric][stat]
                    if not (saved is None if value is None else
                            saved is not None and np.isclose(saved, value, rtol=1e-12, atol=1e-12)):
                        raise ValueError('Raw qEEG summary differs from source')
        elif analysis.get(key) != expected:
            raise ValueError(f'Raw qEEG {key} audit differs from source')
    if (frame.metric_status.tolist() != [w['status'] for w in analysis['window_audit']]
            or not np.array_equal(frame.metric_valid, result['valid'])
            or not (frame.quality_state == 'disabled').all() or not (frame.channel == p['channel']).all()):
        raise ValueError('Raw qEEG status, quality or channel mismatch')
    for key in METRIC_KEYS:
        if not np.allclose(frame[key], result['metrics'][key], rtol=1e-12, atol=1e-12, equal_nan=True):
            raise ValueError(f'Raw qEEG {key} differs from recomputed source')
    return frame, meta
