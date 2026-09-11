"""Quality provenance for filtered, resampled and model-output TFLite inputs."""
from __future__ import annotations

import numpy as np

from lilia.io import bandpass_filter
from lilia.quality_audit import (POLICY, COLUMNS, capture_diagnostics, diagnostic_columns,
                                json_value, validate_scored_diagnostics)
from lilia.signal import resample_polyphase
from lilia.tflite_baseline import score_baseline_windows

STAGES = {'before': 'filtered_resampled', 'after': 'model_output'}
VERIFICATION = {'before': 'source filter/resample and diagnostic rescoring',
                'baseline': 'source filter, raw saturation and diagnostic rescoring',
                'after': 'model identity and diagnostic consistency; reader does not rerun inference'}


def score_windows(data, starts, win, *, scorer, fs, params, stage):
    scores, rows = [], []
    for i, start in enumerate(starts):
        block = data[start:start + win]
        result = scorer(block.T.astype(np.float64), fs=fs, params=params)
        scores.append(result['overall'])
        rows.append(dict(metric_row=i, start_idx=int(start), end_idx=int(start + win),
                         quality_overall=json_value(result['overall']),
                         quality_diagnostics=capture_diagnostics(result, fs=fs, params=params,
                            n_channels=block.shape[1], n_samples=len(block), stage=stage)))
    return np.array(scores), rows


def attach_columns(frame, analysis):
    for branch in STAGES:
        for key, values in diagnostic_columns(analysis[branch]).items():
            frame[f'{branch}_{key}'] = values


def reconstruct_inputs(raw, timeline, params):
    """Reproduce the entry's full-segment filtering and retained resampling."""
    filtered = np.full(raw.shape, np.nan, dtype=np.float32)
    parts = []
    for segment in timeline.segments:
        if segment['status'] != 'retained':
            continue
        a, b = segment['raw_start_idx'], segment['raw_end_idx']
        filtered[a:b] = bandpass_filter(raw[a:b], fs=params['input_fs'],
                                       lo=params['bandpass'][0], hi=params['bandpass'][1])
        parts.append(resample_polyphase(filtered[a:b], params['input_fs'], params['fs'])[
            :segment['retained_samples']])
    return filtered, np.concatenate(parts)[:, :2]


def validate_quality(frame, meta, *, raw=None, time_us=None, timeline=None):
    version = meta.get('quality_diagnostics_version')
    if version is None:
        if 'quality_analysis' in meta or any(f'{branch}_{key}' in frame for branch in STAGES for key in COLUMNS):
            raise ValueError('TFLite quality diagnostics require a declared version')
        return
    if version != 1 or 'quality_analysis' not in meta:
        raise ValueError('Incomplete TFLite quality diagnostics')
    analysis, params = meta['quality_analysis'], meta['parameters']
    if (params['channels'] != [1, 2] or params['input_channels'] != [1, 2, 3, 4]
            or params['quality_reduction'] != 'two-channel median'):
        raise ValueError('TFLite quality channel mapping or reduction mismatch')
    if analysis['score_policy'] != POLICY or analysis['reader_verification'] != VERIFICATION:
        raise ValueError('TFLite quality policy or verification scope mismatch')
    filtered = before = None
    if raw is not None:
        filtered, before = reconstruct_inputs(raw, timeline, params)
    for branch, stage in STAGES.items():
        rows = analysis[branch]
        if [r['metric_row'] for r in rows] != list(range(len(frame))):
            raise ValueError('TFLite quality row coverage mismatch')
        for key, values in diagnostic_columns(rows).items():
            if f'{branch}_{key}' not in frame or frame[f'{branch}_{key}'].tolist() != values:
                raise ValueError('TFLite quality CSV diagnostics differ from audit')
        for i, row in enumerate(rows):
            a, b = int(frame.window_start_idx.iloc[i]), int(frame.window_end_idx.iloc[i])
            if (row['start_idx'], row['end_idx']) != (a, b):
                raise ValueError('TFLite quality index mapping mismatch')
            record = row['quality_diagnostics']
            if record['state'] not in ('valid', 'invalid', 'unavailable'):
                raise ValueError('TFLite quality window was not scored')
            validate_scored_diagnostics(record, fs=params['fs'], params=params['quality_params'],
                n_channels=2, n_samples=b-a, stage=stage,
                signal=before[a:b] if branch == 'before' and before is not None else None)
            values = np.asarray(row['quality_overall'], dtype=float)
            if values.shape != (2,):
                raise ValueError('TFLite quality score shape mismatch')
            payload = record['result']
            if payload is not None and not np.allclose(values, np.asarray(payload['overall'], dtype=float),
                                                        atol=0, rtol=0, equal_nan=True):
                raise ValueError('TFLite quality scores differ from diagnostics')
            if not np.allclose(float(frame[f'quality_{branch}'].iloc[i]), np.median(values),
                               atol=1e-14, rtol=1e-14, equal_nan=True):
                raise ValueError('TFLite quality median differs from CSV')
    # quality_valid also includes finite qEEG; preserve that additional exclusion.
    after = frame.quality_after.to_numpy(dtype=float)
    if np.any(frame.quality_valid & (~np.isfinite(after) | (after < params['quality_threshold']))):
        raise ValueError('TFLite accepted a below-threshold quality window')
    catalog = analysis['baseline_catalog']
    if len(catalog) != len(meta['inference']['model_windows']):
        raise ValueError('TFLite baseline catalog coverage mismatch')
    for row, window in zip(catalog, meta['inference']['model_windows']):
        if any(row.get(key) != value for key, value in window.items()):
            raise ValueError('TFLite baseline model mapping mismatch')
    if raw is not None:
        # Replay only the saved scorer responses. Rebuild all prechecks, ordering,
        # short circuits and reductions independently from the source slices.
        scored = iter(sub for row in catalog for sub in row['quality_subepochs']
                      if sub['quality_overall'] is not None)

        def replay(data, fs, params):
            try:
                sub = next(scored)
            except StopIteration as exc:
                raise ValueError('TFLite baseline scored-subepoch coverage mismatch') from exc
            record = sub['quality_diagnostics']
            if record['state'] not in ('valid', 'invalid', 'unavailable'):
                raise ValueError('TFLite baseline scored state mismatch')
            validate_scored_diagnostics(record, fs=fs, params=params, n_channels=data.shape[0],
                n_samples=data.shape[1], stage='filtered', signal=data.T)
            result = record['result'] or {'overall': sub['quality_overall']}
            if not np.allclose(np.asarray(result['overall'], dtype=float),
                               np.asarray(sub['quality_overall'], dtype=float),
                               atol=0, rtol=0, equal_nan=True):
                raise ValueError('TFLite baseline scores differ from diagnostics')
            return {**result, 'overall': np.asarray(result['overall'], dtype=float)}

        expected = score_baseline_windows(time_us, filtered, raw, timeline, scorer=replay,
            quality_params=params['quality_params'], threshold=params['quality_threshold'],
            **params['baseline_screening'])
        if next(scored, None) is not None or json_value(expected) != catalog:
            raise ValueError('TFLite baseline diagnostics or screening differ from source')
    for baseline in analysis['baselines']:
        if baseline.get('status') == 'excluded':
            continue
        candidates = [r for r in catalog if r['window_start_us'] >= baseline['search_start_us']
                      and r['window_end_us'] <= baseline['search_end_us']]
        eligible = [r for r in candidates if r['eligible']]
        need = int(np.ceil(baseline['required_sec'] / (params['model_window'] / params['fs'])))
        if (need > len(eligible) or baseline['n_selected'] != need
                or baseline['n_candidates'] != len(candidates) or baseline['n_eligible'] != len(eligible)):
            raise ValueError('TFLite baseline selection counts mismatch')
        indices = np.sort(np.random.default_rng(baseline['random_seed']).choice(len(eligible), need, replace=False))
        if baseline['selected'] != [eligible[int(i)] for i in indices]:
            raise ValueError('TFLite baseline selection differs from catalog and seed')
