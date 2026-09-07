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


def write_state_entropy_table(path, source, frame, parameters, states, code_sha256):
    path = Path(path)
    frame.to_csv(path, index=False)
    metadata = {
        'kind': 'state_band_entropy', 'schema_version': 1, 'index_space': 'raw_samples',
        'source_path': str(Path(source).resolve()), 'source_id': file_sha256(source),
        'parameters': parameters, 'config_id': config_id(parameters),
        'states': states, 'audit_id': config_id(states),
        'code_sha256': code_sha256, 'table_sha256': file_sha256(path),
    }
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
    frame = pd.read_csv(path)
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
    return frame, meta
