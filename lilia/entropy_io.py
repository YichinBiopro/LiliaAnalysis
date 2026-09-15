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


def validate_window_quality(frame, parameters, *, joint=False):
    """Verify saved scoring state and masking without recomputing the scorer.

    Pre-contract tables with neither state column remain readable. A declared
    contract or either state column requires the complete state/mask pair.
    Diagnostic validity remains separate from the legacy score threshold.
    """
    version = parameters.get('quality_state_version')
    present = {'quality_state', 'quality_valid'} & set(frame)
    if version is None and not present:
        return
    if version is not None and (type(version) is not int or version != 1):
        raise ValueError('Unsupported window quality state version')
    if present != {'quality_state', 'quality_valid'}:
        raise ValueError('Incomplete window quality state/mask')
    enabled = parameters.get('quality_enabled')
    if type(enabled) is not bool:
        raise ValueError('Window quality_enabled must be boolean')
    if not (frame.quality_state == ('scored' if enabled else 'disabled')).all():
        raise ValueError('Window quality state differs from quality_enabled')
    if frame.quality_valid.dtype.kind != 'b':
        raise ValueError('Window quality_valid must contain booleans')
    valid = np.ones(len(frame), dtype=bool)
    if enabled:
        if 'quality' not in frame or 'quality_threshold' not in parameters:
            raise ValueError('Scored windows require quality and threshold')
        threshold = float(parameters['quality_threshold'])
        if not np.isfinite(threshold):
            raise ValueError('Window quality threshold must be finite')
        scores = frame.quality.to_numpy(dtype=float)
        valid = np.isfinite(scores) & (scores >= threshold)
    elif 'quality' in frame and frame.quality.notna().any():
        raise ValueError('Disabled quality cannot contain scored values')
    if not np.array_equal(frame.quality_valid, valid):
        raise ValueError('Window quality mask differs from scores and threshold')
    if joint:
        metrics = ['joint_mi', 'joint_mi_norm']
        if 'signal_valid' not in frame or frame.signal_valid.dtype.kind != 'b':
            raise ValueError('Joint-MI signal_valid must contain booleans')
        if not set(metrics).issubset(frame):
            raise ValueError('Incomplete joint-MI metrics')
        # signal_valid describes the original MI computation, before masking.
        signal_valid = frame.signal_valid.to_numpy()
        for metric in metrics:
            observed = np.isfinite(frame[metric].to_numpy(dtype=float))
            if not np.array_equal(observed, valid & signal_valid):
                raise ValueError('Joint-MI finite values differ from quality/signal masks')
        excluded = ~valid | ~signal_valid
    else:
        metrics = [key for key in frame if key.startswith(('E_', 'p_', 'lagged_mi_'))
                   or key in ('band_entropy', 'band_entropy_norm')]
        excluded = ~valid
    if frame.loc[excluded, metrics].notna().any().any():
        raise ValueError('Quality-excluded windows contain unmasked metrics')


def _write_window_table(path, source, frame, grid, parameters, code_sha256, kind, *, source_info=None, quality_audit=None):
    """Publish a table and sidecar; interrupted pairs fail validation on read."""
    frame = frame.copy()
    if quality_audit is not None:
        from lilia.quality_audit import attach_diagnostic_columns
        attach_diagnostic_columns(frame, quality_audit)
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
        if quality_audit is not None:
            metadata.update(quality_diagnostics_version=1, quality_audit=quality_audit)
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
    frame = pd.read_csv(path, dtype={'source_id': str, 'config_id': str}, float_precision='round_trip')
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


def write_entropy_table(path, source, frame, grid, parameters, code_sha256, *, quality_audit=None):
    """Publish the band-entropy table with its verified window sidecar."""
    _write_window_table(path, source, frame, grid, parameters, code_sha256, 'band_entropy', quality_audit=quality_audit)


def load_entropy_table(path, raw_csv=None, channel=None):
    """Read legacy entropy or verify versioned entropy against the raw source."""
    frame, metadata = _load_window_table(path, raw_csv, channel, kind='band_entropy')
    if metadata is not None:
        validate_window_quality(frame, metadata['parameters'])
        from lilia.entropy_quality import validate_window_diagnostics
        validate_window_diagnostics(frame, metadata, raw_csv)
    return frame, metadata


def write_joint_mi_table(path, source, frame, grid, parameters, code_sha256, *, quality_audit=None):
    """Publish non-denoised joint MI with a distinct kind and the shared grid."""
    _write_window_table(path, source, frame, grid, parameters, code_sha256, 'joint_mi', quality_audit=quality_audit)


def load_joint_mi_table(path, raw_csv=None, channels=None):
    """Verify a joint-MI table, including the ordered pair of raw channels."""
    frame, metadata = _load_window_table(path, raw_csv, kind='joint_mi')
    if metadata is None:
        raise ValueError('Joint-MI window metadata is required; regenerate the analysis CSV')
    if channels is not None and list(channels) != metadata['parameters'].get('channels'):
        raise ValueError('Requested channels differ from the joint-MI table')
    validate_window_quality(frame, metadata['parameters'], joint=True)
    from lilia.entropy_quality import validate_window_diagnostics
    validate_window_diagnostics(frame, metadata, raw_csv)
    return frame, metadata


def write_joint_mi_summary(path, frame, series_path):
    """Bind the unmasked population summary to its scored/disabled series."""
    path, series_path = Path(path), Path(series_path)
    series_meta_path = Path(str(series_path) + '.meta.json')
    series_meta = json.loads(series_meta_path.read_text())
    frame.to_csv(path, index=False)
    metadata = dict(schema_version=1, kind='joint_mi_summary', source_id=series_meta['source_id'],
                    series_path=os.path.relpath(series_path, path.parent),
                    series_sha256=file_sha256(series_path), series_meta_sha256=file_sha256(series_meta_path),
                    table_sha256=file_sha256(path))
    Path(str(path) + '.meta.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')


def load_joint_mi_summary(path, raw_csv=None, model_path=None):
    """Verify population/series scope; numerical MI is validated separately."""
    path = Path(path)
    metadata = json.loads(Path(str(path) + '.meta.json').read_text())
    if (metadata.get('kind') != 'joint_mi_summary' or metadata.get('schema_version') != 1
            or metadata.get('table_sha256') != file_sha256(path)):
        raise ValueError('Joint-MI summary fingerprint mismatch')
    series_path = path.parent / metadata['series_path']
    sidecar = Path(str(series_path) + '.meta.json')
    if (file_sha256(series_path) != metadata['series_sha256']
            or file_sha256(sidecar) != metadata['series_meta_sha256']):
        raise ValueError('Joint-MI summary series fingerprint mismatch')
    series_metadata = json.loads(sidecar.read_text())
    if series_metadata['kind'] == 'denoised_joint_mi':
        from lilia.neural_io import load_denoised_joint_mi_table
        series, series_metadata = load_denoised_joint_mi_table(series_path, raw_csv, model_path)
    else:
        series, series_metadata = load_joint_mi_table(series_path, raw_csv)
    if metadata['source_id'] != series_metadata['source_id']:
        raise ValueError('Joint-MI summary source differs from series')
    frame = pd.read_csv(path, float_precision='round_trip')
    expected = {'population_quality_state': 'disabled', 'population_quality_masked': False,
                'series_quality_state': 'scored' if series_metadata['parameters']['quality_enabled'] else 'disabled',
                'series_windows': len(series), 'series_quality_valid_windows': int(series.quality_valid.sum())}
    if len(frame) != 1 or not set(expected).issubset(frame):
        raise ValueError('Incomplete joint-MI summary quality scope')
    if frame.population_quality_masked.dtype.kind != 'b':
        raise ValueError('Joint-MI population mask flag must be boolean')
    if any(frame[key].iloc[0] != value for key, value in expected.items()):
        raise ValueError('Joint-MI summary quality scope differs from series/population policy')
    return frame, metadata
