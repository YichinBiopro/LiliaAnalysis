"""Source-bound tables for quality_check sample and anomaly diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lilia.io import load_merged_csv, bandpass_filter
from lilia.provenance import file_sha256
from lilia.quality_audit import (COLUMNS, diagnostic_columns, validate_scored_diagnostics,
                                 POLICY, json_value)


def write_quality_check_table(path, source, frame, kind, parameters, audit):
    if kind not in ('quality_check_samples', 'quality_check_anomalies'):
        raise ValueError('Unknown quality_check table kind')
    frame.to_csv(path, index=False)
    meta = dict(schema_version=1, kind=kind, source_sha256=file_sha256(source),
                source_samples=parameters['source_samples'],
                source_channels=parameters['source_channels'],
                table_sha256=file_sha256(path), parameters=parameters,
                quality_diagnostics_version=1, quality_audit=json_value(audit))
    Path(str(path)+'.meta.json').write_text(json.dumps(meta, indent=2)+'\n')


def _column(frame, key, expected, *, exact=False):
    if key not in frame:
        raise ValueError(f'quality_check missing column: {key}')
    values = frame[key].to_numpy()
    expected = np.asarray(expected)
    try:
        same = (values.shape == expected.shape and
                (np.array_equal(values, expected) if exact else
                 np.allclose(values.astype(float), expected.astype(float),
                             rtol=1e-14, atol=1e-14, equal_nan=True)))
    except (TypeError, ValueError):
        same = False
    if not same:
        raise ValueError(f'quality_check {key} differs from source/audit')


def _load(path, raw_csv, kind):
    path = Path(path)
    meta = json.loads(Path(str(path)+'.meta.json').read_text())
    if (meta.get('schema_version') != 1 or meta.get('kind') != kind
            or meta.get('table_sha256') != file_sha256(path)):
        raise ValueError('quality_check kind or table hash mismatch')
    frame = pd.read_csv(path, float_precision='round_trip')
    audit = meta.get('quality_audit', [])
    version = meta.get('quality_diagnostics_version')
    if version is None:
        if audit or any(key in frame for key in COLUMNS):
            raise ValueError('quality_check diagnostics require a declared version')
    elif type(version) is not int or version != 1 or len(audit) != len(frame):
        raise ValueError('quality_check diagnostic version or row coverage mismatch')
    else:
        for key, expected in diagnostic_columns(audit).items():
            _column(frame, key, expected, exact=True)
    t = raw = None
    if raw_csv is not None:
        if meta.get('source_sha256') != file_sha256(raw_csv):
            raise ValueError('quality_check source hash mismatch')
        t, raw = load_merged_csv(raw_csv)
        if (meta.get('source_samples') != len(t) or
                meta.get('source_channels') != raw.shape[1]):
            raise ValueError('quality_check source shape mismatch')
    return frame, meta, t, raw


def load_quality_samples_table(path, raw_csv=None):
    frame, meta, t, raw = _load(path, raw_csv, 'quality_check_samples')
    p = meta['parameters']
    audit = meta.get('quality_audit', [])
    expected_count = p['n_segments'] * 2
    if len(frame) != expected_count or len(p['starts']) != p['n_segments']:
        raise ValueError('Sample quality stage or segment coverage mismatch')
    starts = p['starts']
    if (any(type(a) is not int or a < 0 or a + p['segment_samples'] > p['source_samples']
            for a in starts) or starts != sorted(starts) or
            any(a+p['segment_samples'] > b for a, b in zip(starts, starts[1:]))):
        raise ValueError('Sample quality segment selection invalid')
    for i in range(expected_count):
        segment = i // 2
        stage = 'raw' if i % 2 == 0 else 'filtered'
        a, b = starts[segment], starts[segment]+p['segment_samples']
        for key, expected in [('segment', segment), ('window_start_idx', a),
                              ('window_end_idx', b), ('quality_stage', stage)]:
            _column(frame.iloc[i:i+1], key, [expected], exact=True)
        row = audit[i] if audit else None
        if row is not None:
            if (row['segment'] != segment or row['window_start_idx'] != a or
                    row['window_end_idx'] != b):
                raise ValueError('Sample quality audit mapping mismatch')
        if t is not None:
            _column(frame.iloc[i:i+1], 'window_start_us', [t[a]], exact=True)
            _column(frame.iloc[i:i+1], 'window_center_us', [t[a+p['segment_samples']//2]], exact=True)
        if row is not None:
            signal = None
            if raw is not None:
                signal = raw[a:b] if stage == 'raw' else bandpass_filter(raw[a:b],
                    fs=p['fs'], lo=p['bp_low'], hi=p['bp_high'])
            saved = row['quality_diagnostics']
            if saved['state'] not in ('valid', 'invalid', 'unavailable'):
                raise ValueError('Scored sample segment lacks quality diagnostics')
            validate_scored_diagnostics(saved, fs=p['fs'], params=p['quality_params'],
                n_channels=p['source_channels'], n_samples=b-a, stage=stage, signal=signal)
            if saved['result'] is not None:
                for ch, score in enumerate(saved['result']['overall']):
                    _column(frame.iloc[i:i+1], f'quality_ch{ch+1}', [np.nan if score is None else score])
    return frame, meta


def _flag_expected(rows, p):
    q = np.asarray([row['qmed'] for row in rows], dtype=float)
    for i, row in enumerate(rows):
        reasons = []
        if row['gap'] > p['gap_sec']:
            reasons.append(f"time-gap {row['gap']:.0f}s")
        if row['clip'] > p['clip_frac']:
            reasons.append(f"clip {row['clip']*100:.0f}%")
        if row['flat'] > p['flat_frac']:
            reasons.append(f"flat {row['flat']*100:.0f}%")
        if not np.isfinite(q[i]):
            reasons.append('invalid-Q non-finite')
        elif q[i] < p['quality_threshold']:
            reasons.append(f"low-Q {q[i]:.2f}")
        if i > 0 and np.isfinite(q[i]) and np.isfinite(q[i-1]) and abs(q[i]-q[i-1]) > p['jump_delta']:
            reasons.append(f"jump {q[i]-q[i-1]:+.2f}")
        row['reasons'] = reasons
        row['severity'] = (min(row['gap']/10., 1.)+row['clip']+row['flat']+
                           (1. if not np.isfinite(q[i]) else max(0., p['quality_threshold']-q[i])))
    return rows


def load_quality_anomalies_table(path, raw_csv=None):
    frame, meta, t, raw = _load(path, raw_csv, 'quality_check_anomalies')
    p = meta['parameters']
    from lilia.constants import QUALITY_THRESHOLD, FS
    from plot_event_markers import QUALITY_PARAMS
    if (p['score_policy'] != POLICY or p['fs'] != FS or
            p['quality_threshold'] != QUALITY_THRESHOLD or
            p['quality_params'] != QUALITY_PARAMS):
        raise ValueError('Anomaly quality caller configuration mismatch')
    win = int(p['win_sec'] * p['fs'])
    if win <= 0:
        raise ValueError('Anomaly quality window invalid')
    starts = list(range(0, p['source_samples']-win+1, win))
    if len(frame) != len(starts):
        raise ValueError('Anomaly quality window coverage mismatch')
    _column(frame, 'window_start_idx', starts, exact=True)
    _column(frame, 'window_end_idx', [a+win for a in starts], exact=True)
    rows = []
    for i, a in enumerate(starts):
        b = a+win
        row = meta['quality_audit'][i]
        if (row['window_start_idx'] != a or row['window_end_idx'] != b or
                row['score_policy'] != POLICY):
            raise ValueError('Anomaly quality audit mapping or policy mismatch')
        saved = row['quality_diagnostics']
        if saved['state'] not in ('valid', 'invalid', 'unavailable'):
            raise ValueError('Anomaly scored window lacks diagnostics')
        validate_scored_diagnostics(saved, fs=p['fs'], params=p['quality_params'],
            n_channels=p['source_channels'], n_samples=win, stage='raw',
            signal=None if raw is None else raw[a:b])
        qmed = frame.qmed.iloc[i]
        if saved['result'] is not None:
            score = np.median(np.asarray([np.nan if v is None else v
                                          for v in saved['result']['overall']], dtype=float))
            _column(frame.iloc[i:i+1], 'qmed', [score])
        if t is not None:
            seg = raw[a:b]
            ts = t[a:b]
            _column(frame.iloc[i:i+1], 'window_center_us', [ts[win//2]], exact=True)
            clip = float(np.mean(np.any(np.abs(seg) >= p['rail_value']-1., axis=1)))
            flat = float(np.mean(np.all(np.diff(seg, axis=0) == 0, axis=1)))
            gap = float(np.max(np.diff(ts))/1e6) if win > 1 else 0.
            for key, expected in [('clip', clip), ('flat', flat), ('gap', gap)]:
                _column(frame.iloc[i:i+1], key, [expected])
        for key in ('qmed', 'clip', 'flat', 'gap', 'severity'):
            _column(frame.iloc[i:i+1], key, [row[key]])
        if row['window_center_us'] != int(frame.window_center_us.iloc[i]):
            raise ValueError('Anomaly quality timestamp audit mismatch')
        rows.append(dict(qmed=float(qmed), clip=float(frame['clip'].iloc[i]),
                         flat=float(frame['flat'].iloc[i]), gap=float(frame['gap'].iloc[i])))
    _flag_expected(rows, p)
    for i, row in enumerate(rows):
        saved = meta['quality_audit'][i]
        reasons = json.dumps(row['reasons'], separators=(',', ':'))
        _column(frame.iloc[i:i+1], 'reasons', [reasons], exact=True)
        if saved['reasons'] != row['reasons'] or saved['severity'] != row['severity']:
            raise ValueError('Anomaly reason/severity audit mismatch')
        _column(frame.iloc[i:i+1], 'severity', [row['severity']])
    return frame, meta
