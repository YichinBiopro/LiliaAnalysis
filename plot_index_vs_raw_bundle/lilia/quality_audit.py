"""JSON quality diagnostics for callers that preserve legacy score selection."""
from __future__ import annotations

import copy
import json

import numpy as np

from lilia.entropy_io import config_id
from lilia.quality import get_eeg_quality_index_v2_parametric, _weighted_geometric_quality

POLICY = 'legacy_overall'
COLUMNS = ('quality_diagnostic_state', 'quality_diagnostic_reason', 'quality_stage', 'quality_score_policy')


def json_value(value):
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not np.isfinite(value) else value


def pending_diagnostics(*, fs, params, n_channels, n_samples, stage='raw', state='not_scored', reasons=()):
    return {'schema_version': 1, 'state': state, 'score_policy': POLICY,
            'request': json_value({'fs': float(fs), 'parameters': params, 'stage': stage,
                                   'n_channels': n_channels, 'n_samples': n_samples}),
            'reasons': list(reasons), 'result': None}


def capture_diagnostics(result, *, fs, params, n_channels, n_samples, stage='raw'):
    record = pending_diagnostics(fs=fs, params=params, n_channels=n_channels, n_samples=n_samples, stage=stage)
    if 'context' not in result:
        record.update(state='unavailable', reasons=['scorer_diagnostics_unavailable'])
        return record
    keys = ('overall', 'detail', 'valid', 'invalid_reasons', 'component_valid', 'component_reasons', 'context')
    payload = copy.deepcopy({key: result[key] for key in keys})
    context = payload['context']
    if (context['fs'] != float(fs) or context['n_channels'] != n_channels
            or context['n_samples'] != n_samples or context['stage'] not in ('unspecified', stage)):
        raise ValueError('Quality diagnostic context differs from caller input')
    # The caller knows the signal stage; old injected scorer signatures stay intact.
    context['stage'] = stage
    context['config_id'] = config_id({key: context[key] for key in ('profile', 'preset', 'stage', 'fs', 'parameters')})
    record.update(state='valid' if np.asarray(payload['valid']).all() else 'invalid', result=json_value(payload))
    record['reasons'] = sorted({reason for channel in payload['invalid_reasons'] for reason in channel})
    return record


def diagnostic_columns(rows):
    diagnostics = [row['quality_diagnostics'] for row in rows]
    return dict(zip(COLUMNS, ([d['state'] for d in diagnostics],
                              [json.dumps(d['reasons'], separators=(',', ':')) for d in diagnostics],
                              [d['request']['stage'] for d in diagnostics],
                              [d['score_policy'] for d in diagnostics])))


def attach_diagnostic_columns(frame, rows):
    for key, value in diagnostic_columns(rows).items():
        frame[key] = value


def validate_scored_diagnostics(saved, *, signal=None, **request):
    """Check a caller-bound record; optional input enables numerical rescoring.

    Without the scoring input, validate core context and internal consistency
    only. Historical exception occurrence cannot be reproduced from metadata.
    Precheck exclusions must additionally be verified by the caller's reader.
    """
    base = pending_diagnostics(**request)
    if (saved.get('schema_version') != 1 or saved.get('request') != base['request']
            or saved.get('score_policy') != POLICY):
        raise ValueError('Quality diagnostic source request or policy mismatch')
    state, payload = saved['state'], saved['result']
    if state in ('unavailable', 'not_scored', 'error'):
        if payload is not None or not saved['reasons']:
            raise ValueError('Quality diagnostic unavailable/error record is incomplete')
        if state == 'unavailable' and saved['reasons'] != ['scorer_diagnostics_unavailable']:
            raise ValueError('External quality diagnostic reason mismatch')
        return
    if state not in ('valid', 'invalid') or payload is None:
        raise ValueError('Unknown quality diagnostic state')
    n = request['n_channels']
    if (len(payload['valid']) != n or any(type(v) is not bool for v in payload['valid'])
            or len(payload['invalid_reasons']) != n or len(payload['overall']) != n):
        raise ValueError('Quality diagnostic channel shape mismatch')
    if payload['valid'] != [not r for r in payload['invalid_reasons']]:
        raise ValueError('Quality diagnostic validity/reasons mismatch')
    if state != ('valid' if all(payload['valid']) else 'invalid'):
        raise ValueError('Quality diagnostic aggregate state mismatch')
    if saved['reasons'] != sorted({reason for channel in payload['invalid_reasons'] for reason in channel}):
        raise ValueError('Quality diagnostic reasons differ from channels')
    # Capture normal context independently, even if the saved run had an exception.
    from warnings import catch_warnings, simplefilter
    with catch_warnings():
        simplefilter('ignore', RuntimeWarning)
        actual = get_eeg_quality_index_v2_parametric(np.asarray(signal).T.astype(float) if signal is not None else np.zeros((n, request['n_samples'])),
            fs=request['fs'], params=request['params'], stage=request['stage'])
    expected = capture_diagnostics(actual, **request)
    if payload['context'] != expected['result']['context']:
        raise ValueError('Quality diagnostic preset/context differs from source')
    exceptions = any(':exception:' in reason for reason in saved['reasons'])
    if signal is not None and not exceptions and saved != expected:
        raise ValueError('Quality diagnostics differ from input rescoring')
    components = payload['context']['active_components']
    if any(set(payload[key]) != set(components) for key in ('detail', 'component_valid', 'component_reasons')):
        raise ValueError('Quality diagnostic component coverage mismatch')
    weights = {key: max(float(payload['context']['parameters'][key + '_weight']), 0.) for key in components}
    for component in components:
        for key in ('detail', 'component_valid', 'component_reasons'):
            if len(payload[key][component]) != n:
                raise ValueError('Quality diagnostic component shape mismatch')
        for channel, reasons in enumerate(payload['component_reasons'][component]):
            invalid_input = any(reason in ('empty_input', 'nonfinite_input') for reason in payload['invalid_reasons'][channel])
            if payload['component_valid'][component][channel] is not (not invalid_input and not reasons):
                raise ValueError('Quality diagnostic component validity mismatch')
            expected_reasons = [f'{component}:{reason}' for reason in reasons]
            if any(reason not in payload['invalid_reasons'][channel] for reason in expected_reasons):
                raise ValueError('Quality diagnostic component reasons mismatch')
            if any(reason.startswith('exception:') for reason in reasons):
                if component not in ('spectrum', 'kurtosis', 'corr'):
                    raise ValueError('Unsupported quality exception fallback')
                fallback = payload['context']['parameters']['corr_floor'] if component == 'corr' else .5
                if payload['detail'][component][channel] != fallback:
                    raise ValueError('Quality exception fallback score mismatch')
            elif signal is not None:
                observed = np.asarray([payload['detail'][component][channel]], dtype=float)
                computed = np.asarray([actual['detail'][component][channel]], dtype=float)
                if not np.allclose(observed, computed, rtol=1e-14, atol=1e-14, equal_nan=True):
                    raise ValueError('Quality non-exception component differs from scoring input')
    computed_overall = _weighted_geometric_quality(
        {key: np.asarray(value, dtype=float) for key, value in payload['detail'].items()}, weights)
    if not np.allclose(computed_overall, np.asarray(payload['overall'], dtype=float), rtol=1e-14, atol=1e-14, equal_nan=True):
        raise ValueError('Quality diagnostic overall differs from components')


def validate_diagnostics(frame, metadata, raw=None):
    """Validate records and raw mapping; rerun only diagnostic-aware raw scoring.

    Legacy/external scorers are explicitly unavailable, never passed off as v2.
    Exception fallbacks cannot be reproduced on demand: their recorded reasons,
    shapes and core context are validated without claiming they recur.
    """
    version = metadata.get('quality_diagnostics_version')
    if version is None:
        if any(key in frame for key in COLUMNS) or any('quality_diagnostics' in r
                for r in metadata['analysis']['window_audit']):
            raise ValueError('Quality diagnostics require a declared version')
        return
    if version != 1:
        raise ValueError('Unsupported quality diagnostics version')
    rows = metadata['analysis']['window_audit']
    if [row['metric_row'] for row in rows] != list(range(len(frame))):
        raise ValueError('Quality diagnostic row coverage mismatch')
    expected_columns = diagnostic_columns(rows)
    for key, expected in expected_columns.items():
        if key not in frame or frame[key].tolist() != expected:
            raise ValueError(f'Quality diagnostic {key} differs from audit')
    params = metadata['parameters']
    for i, row in enumerate(rows):
        saved = row['quality_diagnostics']
        a, b = int(frame.quality_raw_start_idx.iloc[i]), int(frame.quality_raw_end_idx.iloc[i])
        request = dict(fs=float(params['input_fs']), params=params['quality_params'],
                       n_channels=params['quality_channels'], n_samples=b-a, stage='raw')
        base = pending_diagnostics(**request)
        if (saved.get('schema_version') != 1 or saved.get('request') != base['request']
                or saved.get('score_policy') != POLICY):
            raise ValueError('Quality diagnostic source request or policy mismatch')
        state, payload = saved['state'], saved['result']
        if state in ('unavailable', 'not_scored', 'error'):
            if payload is not None or not saved['reasons']:
                raise ValueError('Quality diagnostic unavailable/error record is incomplete')
            if state == 'unavailable' and saved['reasons'] != ['scorer_diagnostics_unavailable']:
                raise ValueError('External quality diagnostic reason mismatch')
            if state == 'error' and row['quality_status'] != 'quality_error':
                raise ValueError('Quality diagnostic error status mismatch')
            if state == 'not_scored':
                reasons = []
                if b-a < 8:
                    reasons.append('short_raw')
                if raw is not None and not np.isfinite(raw[a:b]).all():
                    reasons.append('nonfinite_raw')
                if row['quality_status'] != 'nonfinite_or_short_raw' or (raw is not None and saved['reasons'] != reasons):
                    raise ValueError('Quality diagnostic exclusion differs from raw')
            continue
        validate_scored_diagnostics(saved, signal=None if raw is None else raw[a:b], **request)
        if row['quality_status'] in ('accepted', 'low_quality'):
            n = request['n_channels']
            values = frame.iloc[i][[f'quality_ch{ch+1}' for ch in range(n)]].to_numpy(dtype=float)
            if not np.allclose(values, payload['overall'], rtol=1e-14, atol=1e-14):
                raise ValueError('Quality CSV scores differ from diagnostics')


def plot_diagnostic_markers(ax, x, rows, quality, *, label_prefix='', invalid_marker='x'):
    """Mark unavailable computation separately from a threshold-based low score."""
    states = [row['quality_diagnostics']['state'] for row in rows]
    for selected, marker, color, label in [
            ({'invalid', 'error', 'not_scored'}, invalid_marker, 'darkorange', 'Diagnostic invalid (legacy scores)'),
            ({'unavailable'}, '|', 'gray', 'Diagnostics unavailable')]:
        indices = [i for i, state in enumerate(states) if state in selected]
        if indices:
            medians = np.median(np.asarray(quality)[indices], axis=1)
            y = np.where(np.isfinite(medians), medians, .02)
            ax.scatter(np.asarray(x)[indices], y, marker=marker, color=color, s=35, zorder=5, label=label_prefix + label)


def diagnostic_summary(rows):
    states = [row['quality_diagnostics']['state'] for row in rows]
    return {state: states.count(state) for state in ('valid', 'invalid', 'unavailable', 'error', 'not_scored')}


def diagnostic_label(rows):
    labels = set()
    for row in rows:
        payload = row['quality_diagnostics']['result']
        if payload is not None:
            context = payload['context']
            labels.add('+'.join(context['active_components']) + ' [' + context['preset'] + ']')
    if len(labels) == 1:
        return next(iter(labels))
    return 'mixed scorer contexts' if labels else 'scorer diagnostics unavailable'
