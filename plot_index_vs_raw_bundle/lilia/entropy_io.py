"""Versioned entropy and joint-MI tables with verifiable sample/time alignment."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from lilia.io import read_lilia_frame
from lilia.provenance import file_sha256
from lilia.windowing import build_window_grid


SCHEMA_VERSION = 1
WINDOW_COLUMNS = (
    'window_start_idx', 'window_end_idx', 'window_start_us', 'window_end_us',
    'window_center_us', 'segment_id', 'time_s',
)


def config_id(parameters):
    return hashlib.sha256(json.dumps(parameters, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _write_window_table(path, source, frame, grid, parameters, code_sha256, kind, *, source_info=None):
    """Publish a table and sidecar; interrupted pairs fail validation on read."""
    frame = frame.copy()
    if len(frame) != len(grid.starts):
        raise ValueError('Metric rows do not match the shared window grid')
    source_id = file_sha256(source)
    identity = config_id(parameters)
    for key, values in grid.columns.items():
        frame[key] = values
    frame['window_schema_version'] = SCHEMA_VERSION
    frame['source_id'] = source_id
    frame['config_id'] = identity
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.csv.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            frame.to_csv(handle, index=False)
        metadata = {
            'kind': kind, 'schema_version': SCHEMA_VERSION,
            'source_id': source_id, 'source_path': str(Path(source).resolve()),
            'source_samples': grid.n_samples, 'source_epoch_us': int(grid.columns['window_center_us'][0]
                                                                        - round(grid.time_s[0] * 1e6)),
            'config_id': identity, 'parameters': parameters,
            'code_sha256': code_sha256, 'table_sha256': file_sha256(temporary),
        }
        if source_info is not None:
            metadata.update(source_info)
        os.replace(temporary, path)
        Path(str(path) + '.meta.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_window_table(path, raw_csv=None, channel=None, *, kind):
    """Read legacy tables, or fully validate new tables against their raw file.

    New tables must carry a complete sidecar and explicit integer indexes; row
    sorting/filtering or source edits are never silently interpreted as new times.
    """
    label = kind.replace('_', '-')
    # SHA-256 identities are opaque text. Numeric inference can strip leading
    # zeros or parse a digit/e prefix as an enormous scientific exponent (some
    # pandas versions crash in that conversion before validation can run).
    frame = pd.read_csv(path, dtype={'source_id': str, 'config_id': str})
    sidecar = Path(str(path) + '.meta.json')
    is_new = any(key in frame for key in WINDOW_COLUMNS if key != 'time_s') or 'window_schema_version' in frame
    if not is_new and not sidecar.exists():
        return frame, None
    required = set(WINDOW_COLUMNS) | {'window_schema_version', 'source_id', 'config_id'}
    if not required.issubset(frame) or not sidecar.exists() or frame.empty:
        raise ValueError(f'Incomplete {label} window metadata; regenerate the analysis CSV')
    metadata = json.loads(sidecar.read_text(encoding='utf-8'))
    parameters = metadata.get('parameters', {})
    if (metadata.get('kind') != kind or metadata.get('schema_version') != SCHEMA_VERSION
            or metadata.get('config_id') != config_id(parameters)
            or metadata.get('table_sha256') != file_sha256(path)):
        raise ValueError(f'{label.capitalize()} table or configuration fingerprint mismatch')
    for key, expected in [('window_schema_version', SCHEMA_VERSION),
                          ('source_id', metadata['source_id']), ('config_id', metadata['config_id'])]:
        if not (frame[key] == expected).all():
            raise ValueError(f'Inconsistent {label} {key}')
    if channel is not None and channel != parameters.get('channel'):
        raise ValueError(f'Requested channel differs from the {label} table')
    if raw_csv is not None:
        if file_sha256(raw_csv) != metadata['source_id']:
            raise ValueError(f'Raw recording differs from the {label} source')
        t = read_lilia_frame(raw_csv).iloc[:, 0].to_numpy(dtype=np.int64)
        grid = build_window_grid(t, parameters['fs'], parameters['win_sec'], parameters['step_sec'])
        if (len(t) != metadata['source_samples'] or len(frame) != len(grid.starts)
                or int(t[0]) != metadata['source_epoch_us']):
            raise ValueError(f'Raw recording and {label} window counts differ')
        for key, expected in grid.columns.items():
            actual = frame[key].to_numpy()
            if key == 'time_s':
                valid = np.allclose(actual, expected, rtol=0, atol=1e-9)
            else:
                valid = np.array_equal(actual, expected)
            if not valid:
                raise ValueError(f'{label.capitalize()} {key} does not match raw timestamps')
    return frame, metadata


def write_entropy_table(path, source, frame, grid, parameters, code_sha256):
    """Publish the band-entropy table with its verified window sidecar."""
    _write_window_table(path, source, frame, grid, parameters, code_sha256, 'band_entropy')


def load_entropy_table(path, raw_csv=None, channel=None):
    """Read legacy entropy or verify versioned entropy against the raw source."""
    return _load_window_table(path, raw_csv, channel, kind='band_entropy')


def write_joint_mi_table(path, source, frame, grid, parameters, code_sha256):
    """Publish non-denoised joint MI with a distinct kind and the shared grid."""
    _write_window_table(path, source, frame, grid, parameters, code_sha256, 'joint_mi')


def load_joint_mi_table(path, raw_csv=None, channels=None):
    """Verify a joint-MI table, including the ordered pair of raw channels."""
    frame, metadata = _load_window_table(path, raw_csv, kind='joint_mi')
    if metadata is None:
        raise ValueError('Joint-MI window metadata is required; regenerate the analysis CSV')
    if channels is not None and list(channels) != metadata['parameters'].get('channels'):
        raise ValueError('Requested channels differ from the joint-MI table')
    return frame, metadata
