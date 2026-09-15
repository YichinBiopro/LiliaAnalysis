"""Source-bound quality windows for plot_index_vs_raw custom markers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lilia.io import load_merged_csv
from lilia.provenance import file_sha256
from lilia.quality_audit import (COLUMNS, POLICY, diagnostic_columns,
                                 json_value, validate_scored_diagnostics)


KIND = 'custom_marker_quality'


def write_custom_marker_quality(path, source, frame, parameters, audit):
    """Save the original scorer scores alongside caller-bound diagnostics."""
    path = Path(path)
    frame.to_csv(path, index=False)
    meta = dict(schema_version=1, kind=KIND, source_sha256=file_sha256(source),
                table_sha256=file_sha256(path), parameters=json_value(parameters),
                quality_diagnostics_version=1, quality_audit=json_value(audit))
    Path(str(path) + '.meta.json').write_text(json.dumps(meta, indent=2) + '\n')


def _column(frame, key, expected, *, exact=False):
    if key not in frame:
        raise ValueError(f'Custom marker quality missing {key}')
    actual = frame[key].to_numpy()
    expected = np.asarray(expected)
    try:
        same = actual.shape == expected.shape and (
            np.array_equal(actual, expected) if exact else
            np.allclose(actual.astype(float), expected.astype(float),
                        rtol=1e-14, atol=1e-14, equal_nan=True))
    except (TypeError, ValueError):
        same = False
    if not same:
        raise ValueError(f'Custom marker quality {key} differs from source/audit')


def load_custom_marker_quality_table(path, raw_csv=None):
    """Validate source, window mapping, legacy mask, and saved diagnostics."""
    path = Path(path)
    meta = json.loads(Path(str(path) + '.meta.json').read_text())
    if (meta.get('schema_version') != 1 or meta.get('kind') != KIND or
            meta.get('table_sha256') != file_sha256(path) or
            meta.get('quality_diagnostics_version') != 1):
        raise ValueError('Custom marker quality schema or hash mismatch')
    frame = pd.read_csv(path, float_precision='round_trip')
    p = meta['parameters']
    audit = meta.get('quality_audit', [])
    if len(audit) != len(frame) or any(key not in frame for key in COLUMNS):
        raise ValueError('Custom marker quality diagnostics coverage mismatch')
    for key, values in diagnostic_columns(audit).items():
        _column(frame, key, values, exact=True)
    from plot_event_markers import QUALITY_PARAMS
    if (p['score_policy'] != POLICY or p['quality_params'] != QUALITY_PARAMS or
            p['stage'] != 'raw' or p['gap_factor'] != 3.0 or
            not np.isfinite(p['fs']) or p['fs'] <= 0 or
            not np.isfinite(p['win_sec']) or p['win_sec'] <= 0 or
            not np.isfinite(p['quality_threshold'])):
        raise ValueError('Custom marker quality caller settings mismatch')
    q_win = int(p['win_sec'] * p['fs'])
    score_win = int(round(p['win_sec'] * p['fs']))
    if q_win < 1 or score_win < 1 or p['quality_win_samples'] != q_win or p['score_win_samples'] != score_win:
        raise ValueError('Custom marker quality window settings mismatch')
    starts = list(range(0, p['source_samples'] - q_win + 1, q_win))
    score_starts = list(range(0, p['source_samples'] - score_win + 1, score_win))
    if len(starts) != len(frame) or len(score_starts) != len(frame):
        raise ValueError('Custom marker quality window coverage mismatch')
    if (type(p['channel']) is not int or p['channel'] < 1 or
            p['channel'] > p['source_channels']):
        raise ValueError('Custom marker quality channel invalid')
    _column(frame, 'window_start_idx', starts, exact=True)
    _column(frame, 'window_end_idx', [a + q_win for a in starts], exact=True)
    _column(frame, 'score_window_start_idx', score_starts, exact=True)
    _column(frame, 'score_window_end_idx', [a + score_win for a in score_starts], exact=True)
    t = raw = None
    if raw_csv is not None:
        if meta.get('source_sha256') != file_sha256(raw_csv):
            raise ValueError('Custom marker quality source hash mismatch')
        t, raw = load_merged_csv(raw_csv)
        if len(t) != p['source_samples'] or raw.shape[1] != p['source_channels']:
            raise ValueError('Custom marker quality source shape mismatch')
    for i, (q_start, score_start) in enumerate(zip(starts, score_starts)):
        saved = audit[i]
        if (saved['window_start_idx'] != q_start or
                saved['window_end_idx'] != q_start + q_win or
                saved['score_window_start_idx'] != score_start or
                saved['score_window_end_idx'] != score_start + score_win or
                saved['score_policy'] != POLICY):
            raise ValueError('Custom marker quality audit mapping mismatch')
        diagnostic = saved['quality_diagnostics']
        if diagnostic['state'] not in ('valid', 'invalid', 'unavailable'):
            raise ValueError('Custom marker quality scored window lacks diagnostics')
        signal = None if raw is None else raw[q_start:q_start + q_win, p['channel'] - 1:p['channel']]
        validate_scored_diagnostics(diagnostic, fs=p['fs'], params=p['quality_params'],
            n_channels=1, n_samples=q_win, stage='raw', signal=signal)
        score = np.nan if saved['legacy_overall'][0] is None else saved['legacy_overall'][0]
        _column(frame.iloc[i:i + 1], 'quality_ch1', [score])
        if diagnostic['result'] is not None:
            result_score = diagnostic['result']['overall'][0]
            _column(frame.iloc[i:i + 1], 'quality_ch1',
                    [np.nan if result_score is None else result_score])
        if t is not None:
            q_mid = int(t[q_start + q_win // 2])
            score_mid = int(t[score_start + score_win // 2])
            gap = bool(np.any(np.diff(t[score_start:score_start + score_win]) >
                                  p['gap_factor'] * 1e6 / p['fs']))
            _column(frame.iloc[i:i + 1], 'window_center_us', [q_mid], exact=True)
            _column(frame.iloc[i:i + 1], 'score_center_us', [score_mid], exact=True)
            _column(frame.iloc[i:i + 1], 'crosses_gap', [gap], exact=True)
        else:
            gap = bool(frame.crosses_gap.iloc[i])
        good = bool(np.isfinite(score) and score >= p['quality_threshold'] and not gap)
        _column(frame.iloc[i:i + 1], 'quality_good', [good], exact=True)
        for key in ('window_center_us', 'score_center_us'):
            if saved[key] != int(frame[key].iloc[i]):
                raise ValueError('Custom marker quality timestamp audit mismatch')
        if saved['crosses_gap'] != gap or saved['quality_good'] != good:
            raise ValueError('Custom marker quality mask audit mismatch')
    return frame, meta
