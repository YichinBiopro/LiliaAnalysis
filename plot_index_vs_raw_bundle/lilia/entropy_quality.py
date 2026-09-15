"""Entropy caller diagnostics, preserving legacy channel-median selection."""
from __future__ import annotations

import json

import numpy as np

from lilia.io import read_lilia_frame, bandpass_filter
from lilia.quality_audit import (COLUMNS, POLICY, diagnostic_columns, diagnostic_summary,
                               diagnostic_label, pending_diagnostics, validate_scored_diagnostics,
                               plot_diagnostic_markers)
from lilia.windowing import continuous_slices, plot_breaks


def disabled_rows(grid, *, fs, n_channels, stage):
    return [dict(window_start_idx=int(a), window_end_idx=int(a + grid.win),
                 quality_diagnostics=pending_diagnostics(fs=fs, params=None, n_channels=n_channels,
                     n_samples=grid.win, stage=stage, reasons=['quality_disabled'])) for a in grid.starts]


def _version(metadata, rows, frame, columns=COLUMNS):
    version = metadata.get('quality_diagnostics_version')
    present = any('quality_diagnostics' in row for row in rows) or any(key in frame for key in columns)
    if version is None:
        if present:
            raise ValueError('Entropy quality diagnostics require a declared version')
        return False
    if type(version) is not int or version != 1 or any('quality_diagnostics' not in row for row in rows):
        raise ValueError('Incomplete or unsupported entropy quality diagnostics')
    return True


def _source_signal(raw_csv, parameters, rows, *, state=False):
    """Rebuild full contributing source segments with the caller's float32 front-end."""
    source = read_lilia_frame(raw_csv)
    t = source.iloc[:, 0].to_numpy(dtype=np.int64)
    raw = source.iloc[:, 1:].to_numpy(dtype=np.float32)
    channels = parameters['quality_channels']
    if channels != (parameters['channels'] if 'channels' in parameters else list(range(1, raw.shape[1] + 1))):
        raise ValueError('Entropy quality channel mapping differs from source/caller')
    if any(type(ch) is not int or not 1 <= ch <= raw.shape[1] for ch in channels):
        raise ValueError('Entropy quality channel outside source')
    filtered = np.full(raw.shape, np.nan, dtype=np.float32)
    failures = {}
    for group, sl in enumerate(continuous_slices(t, parameters['fs'])):
        if not any(sl.start <= row['window_start_idx'] < sl.stop and row.get('complete', True) for row in rows):
            continue
        values = raw[sl]
        if parameters['bandpass'] is None:
            filtered[sl] = values
        elif state and not np.isfinite(values).all():
            failures[group] = 'nonfinite_signal'
        else:
            try:
                filtered[sl] = bandpass_filter(values, fs=parameters['fs'],
                    lo=parameters['bandpass'][0], hi=parameters['bandpass'][1])
            except ValueError:
                if not state:
                    raise
                failures[group] = 'filter_error'
    return raw, filtered[:, np.asarray(channels) - 1], failures


def _score_matches(record, score):
    payload = record['result']
    if payload is not None:
        median = float(np.median(np.asarray(payload['overall'], dtype=float)))
        if not np.isclose(median, float(score), rtol=1e-14, atol=1e-14, equal_nan=True):
            raise ValueError('Entropy quality score differs from diagnostic channel median')


def validate_window_diagnostics(frame, metadata, raw_csv=None):
    rows = metadata.get('quality_audit', [])
    if not _version(metadata, rows, frame):
        return
    if len(rows) != len(frame):
        raise ValueError('Entropy quality diagnostic window coverage mismatch')
    for key, expected in diagnostic_columns(rows).items():
        if key not in frame or frame[key].tolist() != expected:
            raise ValueError('Entropy quality diagnostic CSV differs from audit')
    p = metadata['parameters']
    enabled = p['quality_enabled']
    denoised = metadata['kind'] == 'denoised_joint_mi'
    stage = 'model_output' if denoised else 'raw' if p['bandpass'] is None else 'filtered'
    channels = p['quality_channels']
    if denoised and channels != [1, 2]:
        raise ValueError('Denoised diagnostic channels differ from model output')
    signal = None
    if enabled and raw_csv is not None:
        _, signal, _ = _source_signal(raw_csv, p, rows)
    for i, row in enumerate(rows):
        a, b = int(frame.window_start_idx.iloc[i]), int(frame.window_end_idx.iloc[i])
        if row['window_start_idx'] != a or row['window_end_idx'] != b:
            raise ValueError('Entropy quality diagnostic window mapping mismatch')
        saved = row['quality_diagnostics']
        validate_scored_diagnostics(saved, fs=p['fs'], params=p['quality_params'],
            stage=stage, n_channels=len(channels), n_samples=b-a,
            signal=None if signal is None else signal[a:b])
        if enabled:
            if saved['state'] not in ('valid', 'invalid', 'unavailable'):
                raise ValueError('Scored entropy window lacks scorer diagnostics')
            _score_matches(saved, frame.quality.iloc[i])
        elif saved['state'] != 'not_scored' or saved['reasons'] != ['quality_disabled']:
            raise ValueError('Disabled entropy window contains scoring diagnostics')


STATE_COLUMNS = ('quality_diagnostic_counts', 'quality_stage', 'quality_score_policy')


def state_columns(states, parameters):
    return dict(quality_diagnostic_counts=[json.dumps(diagnostic_summary(states[label]['windows']), sort_keys=True)
                                          for label in ('baseline', 'event')],
                quality_stage='raw' if parameters['bandpass'] is None else 'filtered',
                quality_score_policy=POLICY)


def validate_state_diagnostics(frame, metadata, raw_csv=None):
    states, p = metadata['states'], metadata['parameters']
    rows = [row for state in states.values() for row in state['windows']]
    if not _version(metadata, rows, frame, STATE_COLUMNS):
        return
    for key, expected in state_columns(states, p).items():
        values = expected if isinstance(expected, list) else [expected] * len(frame)
        if key not in frame or frame[key].tolist() != values:
            raise ValueError('State quality diagnostic summary differs from audit')
    raw = signal = None
    failures = {}
    if p['clean'] and raw_csv is not None:
        raw, signal, failures = _source_signal(raw_csv, p, rows, state=True)
    stage = 'raw' if p['bandpass'] is None else 'filtered'
    for row in rows:
        a, b = row['window_start_idx'], row['window_end_idx']
        saved, status = row['quality_diagnostics'], row['status']
        scored = status in ('accepted', 'low_quality', 'invalid_quality') and p['clean']
        precheck = None
        if p['clean'] and raw is not None:
            if not row['complete']:
                precheck = 'incomplete_window'
            elif row['segment_id'] in failures:
                precheck = failures[row['segment_id']]
            elif not np.isfinite(raw[a:b]).all() or not np.isfinite(signal[a:b]).all():
                precheck = 'nonfinite_signal'
            else:
                saturation = float(np.mean(np.any(np.abs(raw[a:b]) >= p['saturation_rail'] - 1., axis=1)))
                if row['saturation_fraction'] != saturation:
                    raise ValueError('State quality raw saturation differs from source')
                if saturation > p['saturation_fraction_max']:
                    precheck = 'raw_saturation'
            if (precheck is not None and status != precheck) or (precheck is None and status in
                    ('incomplete_window', 'filter_error', 'nonfinite_signal', 'raw_saturation', 'raw_unavailable')):
                raise ValueError('State quality precheck differs from source')
        validate_scored_diagnostics(saved, fs=p['fs'], params=p['quality_params'],
            stage=stage, n_channels=len(p['quality_channels']), n_samples=b-a,
            signal=signal[a:b] if scored and signal is not None else None)
        if scored:
            if saved['state'] not in ('valid', 'invalid', 'unavailable'):
                raise ValueError('State scorer diagnostics missing')
            if status != 'invalid_quality':
                _score_matches(saved, row['quality'])
            elif saved['result'] is not None:
                scores = np.asarray(saved['result']['overall'], dtype=float)
                if np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all():
                    raise ValueError('State invalid quality contradicts diagnostic scores')
        elif p['clean'] and status == 'quality_error':
            if saved['state'] != 'error' or len(saved['reasons']) != 1 or not saved['reasons'][0].startswith('scorer_exception:'):
                raise ValueError('State quality error diagnostic mismatch')
        elif saved['state'] != 'not_scored' or saved['reasons'] != [status if p['clean'] else 'quality_disabled']:
            raise ValueError('State quality precheck diagnostic mismatch')


def plot_quality_audit(rows, time_s, scores, outpath, *, title, threshold, groups=None):
    """Show diagnostic failures independently of legacy threshold acceptance."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 3.2))
    scores = np.asarray(scores, dtype=float)
    disabled = bool(rows) and all(row['quality_diagnostics']['reasons'] == ['quality_disabled'] for row in rows)
    ax.plot(*plot_breaks(time_s, scores, groups), '.-', color='black', label='Channel median (legacy overall)')
    if not rows:
        ax.text(.5, .5, 'No candidate windows', ha='center', transform=ax.transAxes)
    elif disabled:
        ax.text(.5, .5, 'Quality scoring disabled', ha='center', transform=ax.transAxes)
    else:
        ax.axhline(threshold, color='red', ls='--', label=f'Threshold {threshold:g}')
        plot_diagnostic_markers(ax, time_s, rows, scores[:, None])
        low = np.isfinite(scores) & (scores < threshold)
        ax.scatter(np.asarray(time_s)[low], scores[low], marker='v', color='red', label='Below threshold')
    if len(time_s):
        lo, hi = float(np.min(time_s)), float(np.max(time_s))
        ax.set_xlim(lo - .1, hi + .1)
    ax.set_ylim(-.03, 1.05)
    ax.set_xlabel('Time since recording start (s)')
    ax.set_ylabel('Quality')
    stages = sorted({r['quality_diagnostics']['request']['stage'] for r in rows})
    ax.set_title(title + '\n' + ', '.join(stages) + ' | ' + diagnostic_label(rows))
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
