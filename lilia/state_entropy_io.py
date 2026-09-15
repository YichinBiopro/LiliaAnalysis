"""State entropy summaries with source-verified interval/window audit sidecars."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lilia.entropy_io import config_id
from lilia.io import read_lilia_frame
from lilia.provenance import file_sha256
from lilia.state_windows import select_state_windows, state_bounds


def _validate_state_quality(frame, states, parameters):
    """Verify saved prechecks and quality decisions, without rescoring input."""
    clean = parameters.get('clean')
    if type(clean) is not bool or 'quality_state' not in frame:
        raise ValueError('State entropy quality configuration is incomplete')
    version = parameters.get('quality_state_version')
    if version is not None and (type(version) is not int or version != 1):
        raise ValueError('Unsupported state quality version')
    if not (frame.quality_state == ('enabled' if clean else 'disabled')).all():
        raise ValueError('State entropy quality state differs from clean setting')
    if not (frame.sampling == ('clean_epochs' if clean else 'segment_windows')).all():
        raise ValueError('State entropy sampling differs from clean setting')
    threshold = parameters['quality_threshold']
    if clean and (not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('State quality threshold must be within [0, 1]')
    for label, audit in states.items():
        for row in audit['windows']:
            status, state, score = row['status'], row.get('quality_state'), row.get('quality')
            if not clean:
                if state != 'disabled' or score is not None or status in (
                        'raw_unavailable', 'raw_saturation', 'quality_error', 'invalid_quality', 'low_quality'):
                    raise ValueError('Disabled state quality contains scoring decisions')
                continue
            expected = ('scored' if status in ('accepted', 'low_quality') else
                        'error' if status == 'quality_error' else
                        'invalid' if status == 'invalid_quality' else 'not_scored')
            if state != expected:
                raise ValueError('State window quality state differs from inclusion status')
            if expected == 'scored':
                if (score is None or not np.isfinite(score) or not 0 <= score <= 1
                        or (score >= threshold) != (status == 'accepted')):
                    raise ValueError('State window quality differs from threshold decision')
            elif score is not None:
                raise ValueError('Unscored state window contains a quality value')
            saturation = row.get('saturation_fraction')
            if status == 'raw_saturation' or expected in ('scored', 'error', 'invalid'):
                if saturation is None or not np.isfinite(saturation) or not 0 <= saturation <= 1:
                    raise ValueError('State quality screening lacks saturation precheck')
                if (saturation > parameters['saturation_fraction_max']) != (status == 'raw_saturation'):
                    raise ValueError('State saturation contradicts quality screening')
        statuses = [row['status'] for row in audit['windows']]
        for key, status in [('n_saturated_epochs', 'raw_saturation'), ('n_lowquality_epochs', 'low_quality'),
                            ('n_nonfinite_epochs', 'nonfinite_signal'), ('n_invalid_quality_epochs', 'invalid_quality')]:
            if audit[key] != statuses.count(status):
                raise ValueError('State quality summary counts differ from audit')
        summary = frame.loc[frame.state == label].iloc[0]
        if summary.status != ('accepted' if statuses.count('accepted') else 'excluded_no_usable_windows'):
            raise ValueError('State quality summary status differs from audit')


def write_state_entropy_table(path, source, frame, parameters, states, code_sha256):
    path = Path(path)
    diagnostics = any('quality_diagnostics' in row for audit in states.values() for row in audit['windows'])
    diagnostics = diagnostics or parameters.get('quality_diagnostics_version') == 1
    if diagnostics:
        from lilia.entropy_quality import state_columns
        frame = frame.assign(**state_columns(states, parameters))
    frame.to_csv(path, index=False)
    metadata = {
        'kind': 'state_band_entropy', 'schema_version': 1, 'index_space': 'raw_samples',
        'source_path': str(Path(source).resolve()), 'source_id': file_sha256(source),
        'parameters': parameters, 'config_id': config_id(parameters),
        'states': states, 'audit_id': config_id(states),
        'code_sha256': code_sha256, 'table_sha256': file_sha256(path),
    }
    if diagnostics:
        metadata['quality_diagnostics_version'] = 1
    Path(str(path) + '.meta.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')


def load_state_entropy_table(path, raw_csv=None, channel=None):
    """Verify the pair and rebuild every candidate and missing span from raw time.

    This checks provenance and selection geometry, not a recomputation of PSD/QC.
    Excluded/all-empty states remain readable for failure auditing.
    """
    meta = json.loads(Path(str(path) + '.meta.json').read_text())
    p = meta['parameters']
    if (meta.get('kind') != 'state_band_entropy' or meta.get('schema_version') != 1
            or meta.get('index_space') != 'raw_samples' or meta['config_id'] != config_id(p)
            or meta.get('audit_id') != config_id(meta['states'])
            or meta['table_sha256'] != file_sha256(path)):
        raise ValueError('State entropy table/configuration fingerprint mismatch')
    if channel is not None and channel != p['channel']:
        raise ValueError('Requested channel differs from state entropy table')
    frame = pd.read_csv(path, float_precision='round_trip')
    if list(frame['state']) != ['baseline', 'event'] or set(meta['states']) != {'baseline', 'event'}:
        raise ValueError('State entropy summary must contain baseline and event')
    t = None
    if raw_csv is not None:
        if file_sha256(raw_csv) != meta['source_id']:
            raise ValueError('Raw recording differs from state entropy source')
        t = read_lilia_frame(raw_csv).iloc[:, 0].to_numpy(dtype=np.int64)
    allowed = {'accepted', 'incomplete_window', 'nonfinite_signal', 'raw_saturation',
               'raw_unavailable', 'quality_error', 'invalid_quality', 'low_quality', 'filter_error'}
    for i, label in enumerate(('baseline', 'event')):
        audit = meta['states'][label]
        rows = audit['windows']
        if any(r['status'] not in allowed or (not r['complete'] and r['status'] != 'incomplete_window')
               for r in rows):
            raise ValueError('Invalid state window inclusion status')
        accepted = sum(r['status'] == 'accepted' for r in rows)
        if (accepted != int(frame.iloc[i]['n_windows']) or accepted != audit['n_accepted_epochs']
                or len(audit.get('per_window_entropy', [])) != accepted):
            raise ValueError('State entropy accepted window counts differ')
        if (list(frame.iloc[i][['range_start_s', 'range_end_s']]) != p['ranges'][label]
                or not np.isfinite(audit['per_window_entropy']).all()):
            raise ValueError('State summary range or entropy differs from audit')
        if t is not None:
            lo, hi = state_bounds(t, p['ranges'][label])
            expected = select_state_windows(t, p['fs'], lo, hi, p['win_sec'], p['step_sec'])
            if any(audit[key] != expected[key] for key in ('lo_us', 'hi_us', 'missing_spans')) or len(rows) != len(expected['windows']):
                raise ValueError('State interval mapping differs from raw timestamps')
            for actual, want in zip(rows, expected['windows']):
                if any(actual.get(k) != v for k, v in want.items() if k != 'status'):
                    raise ValueError('State window mapping differs from raw timestamps')
    _validate_state_quality(frame, meta['states'], p)
    from lilia.entropy_quality import validate_state_diagnostics
    validate_state_diagnostics(frame, meta, raw_csv)
    return frame, meta
