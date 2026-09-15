"""Read Goertzel metrics and verify filtered quality and artifact input stages."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lilia.goertzel import goertzel_power
from lilia.io import load_merged_csv, bandpass_filter
from lilia.provenance import file_sha256
from lilia.quality_audit import COLUMNS, diagnostic_columns, validate_scored_diagnostics
from lilia.windowing import window_starts

FEATURE_SOURCES = {
    'quality': 'filtered', 'goertzel_power': 'filtered', 'goertzel_db': 'filtered',
    'sat_frac_1950': 'raw', 'peak_to_peak_uv': 'raw', 'max_abs_diff_uv': 'raw',
    'bp_edge_shift_uv': 'filtered',
    'artifact_hard_clip': ['sat_frac_1950', 'peak_to_peak_uv', 'bp_edge_shift_uv'],
    'quality_final': 'zero_if_hard_else_legacy_overall_selected_channel',
}


def _matches(frame, key, expected, *, exact=False):
    if key not in frame:
        raise ValueError(f'Goertzel missing column: {key}')
    values = frame[key].to_numpy()
    expected = np.asarray(expected)
    valid = (values.shape == expected.shape and
             (np.array_equal(values, expected) if exact else
              np.allclose(values.astype(float), expected.astype(float), rtol=1e-14,
                          atol=1e-14, equal_nan=True)))
    if not valid:
        raise ValueError(f'Goertzel {key} differs from source/audit')


def load_goertzel_table(path, raw_csv=None):
    """Validate durable evidence without requiring today's code fingerprint.

    Cache reuse separately checks the code fingerprint. Legacy tables lacking
    the whole diagnostics extension remain readable with their source metadata.
    Historical scorer exceptions use the shared component/fallback validator;
    external scorers remain explicitly unavailable and cannot be recomputed.
    """
    path = Path(path)
    meta = json.loads(Path(str(path) + '.meta.json').read_text())
    if (meta.get('schema_version') != 1 or meta.get('kind', 'goertzel') != 'goertzel'
            or meta.get('table_sha256') != file_sha256(path)):
        raise ValueError('Goertzel metadata or table hash mismatch')
    frame = pd.read_csv(path, float_precision='round_trip')
    for key in ('window_start_idx', 'window_end_idx', 'window_start_us', 'window_end_us',
                'window_center_us', 'time_s', 'goertzel_power', 'goertzel_db', 'quality',
                'quality_final', 'sat_frac_1950', 'peak_to_peak_uv', 'max_abs_diff_uv',
                'bp_edge_shift_uv', 'artifact_hard_clip'):
        if key not in frame:
            raise ValueError(f'Goertzel missing column: {key}')
        frame[key] = pd.to_numeric(frame[key], errors='raise')
    p = meta['parameters']
    rows = meta.get('quality_audit', [])
    version = meta.get('quality_diagnostics_version')
    if version is None:
        if ('quality_audit' in meta or 'feature_sources' in meta or any(c in frame for c in COLUMNS)):
            raise ValueError('Goertzel diagnostics require a declared version')
    else:
        if (type(version) is not int or version != 1 or meta.get('kind') != 'goertzel'
                or meta.get('feature_sources') != FEATURE_SOURCES or len(rows) != len(frame)):
            raise ValueError('Goertzel diagnostic coverage or feature sources mismatch')
        for key, expected in diagnostic_columns(rows).items():
            _matches(frame, key, expected, exact=True)
    ch = p['ch'] - 1
    if type(p['ch']) is not int or ch < 0:
        raise ValueError('Goertzel channel mapping invalid')
    starts = frame.window_start_idx.to_numpy()
    ends = frame.window_end_idx.to_numpy()
    win = max(int(round(p['win_sec'] * p['fs'])), 1)
    if (not np.isfinite(starts).all() or not np.isfinite(ends).all()
            or not np.equal(starts, starts.astype(np.int64)).all()
            or not np.equal(ends, ends.astype(np.int64)).all()
            or (starts < 0).any() or not np.all(ends - starts == win)):
        raise ValueError('Goertzel window indices invalid')
    raw = bp = None
    if raw_csv is not None:
        if meta.get('source_sha256') != file_sha256(raw_csv):
            raise ValueError('Goertzel source hash mismatch')
        t, raw = load_merged_csv(raw_csv)
        if ch >= raw.shape[1]:
            raise ValueError('Goertzel channel outside source')
        bp = bandpass_filter(raw, fs=p['fs'], lo=p['bp_low'], hi=p['bp_high'], time_us=t)
        expected_starts = np.asarray(list(window_starts(len(bp), win,
            max(int(round(p['step_sec'] * p['fs'])), 1), t, p['fs'])), dtype=np.int64)
        _matches(frame, 'window_start_idx', expected_starts, exact=True)
        expected_ends = expected_starts + win
        _matches(frame, 'window_end_idx', expected_ends, exact=True)
        _matches(frame, 'window_start_us', t[expected_starts], exact=True)
        _matches(frame, 'window_end_us', t[expected_ends - 1] + int(round(1e6 / p['fs'])), exact=True)
        center = t[(expected_starts + expected_ends) // 2]
        _matches(frame, 'window_center_us', center, exact=True)
        _matches(frame, 'time_s', (center - int(t[0])) / 1e6)
        features = {key: [] for key in ('goertzel_power', 'sat_frac_1950', 'peak_to_peak_uv',
                                       'max_abs_diff_uv', 'bp_edge_shift_uv')}
        for a in expected_starts:
            x, y = raw[a:a+win, ch], bp[a:a+win, ch]
            edge = min(max(1, int(round(p['bp_shift_sec'] * p['fs']))), max(1, len(y) // 2))
            features['goertzel_power'].append(goertzel_power(y, p['target_freq'], p['fs'], True, True))
            features['sat_frac_1950'].append(float(np.mean(np.abs(x) >= p['sat_uv'])))
            features['peak_to_peak_uv'].append(float(np.ptp(x)))
            features['max_abs_diff_uv'].append(float(np.max(np.abs(np.diff(x)))) if len(x) > 1 else 0.)
            features['bp_edge_shift_uv'].append(float(abs(np.median(y[:edge]) - np.median(y[-edge:]))))
        for key, values in features.items():
            _matches(frame, key, values)
    _matches(frame, 'goertzel_db', 10. * np.log10(np.maximum(frame.goertzel_power.to_numpy(), 1e-12)))
    hard = ((frame.sat_frac_1950.to_numpy() >= p['sat_frac_threshold'])
            | (frame.peak_to_peak_uv.to_numpy() >= p['step_ptp_threshold'])
            | (frame.bp_edge_shift_uv.to_numpy() >= p['bp_shift_threshold']))
    _matches(frame, 'artifact_hard_clip', hard.astype(int), exact=True)
    _matches(frame, 'quality_final', np.where(hard, 0., frame.quality.to_numpy()))
    if version is not None:
        for i, row in enumerate(rows):
            a, b = int(starts[i]), int(ends[i])
            if row['window_start_idx'] != a or row['window_end_idx'] != b:
                raise ValueError('Goertzel diagnostic window mapping mismatch')
            saved = row['quality_diagnostics']
            if saved['state'] not in ('valid', 'invalid', 'unavailable'):
                raise ValueError('Goertzel scored window lacks diagnostics')
            n_channels = raw.shape[1] if raw is not None else saved['request']['n_channels']
            if ch >= n_channels:
                raise ValueError('Goertzel selected channel outside diagnostic input')
            validate_scored_diagnostics(saved, fs=int(p['fs']), params=p['quality_params'],
                n_channels=n_channels, n_samples=b-a, stage='filtered',
                signal=None if bp is None else bp[a:b])
            if saved['result'] is not None:
                score = saved['result']['overall'][ch]
                _matches(frame.iloc[i:i+1], 'quality', [np.nan if score is None else score])
    return frame, meta
