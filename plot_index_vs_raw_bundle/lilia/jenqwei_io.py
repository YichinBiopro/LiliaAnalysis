"""Source-verifiable, independently indexed Jenqwei Before and After tables."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lilia.entropy_io import config_id
from lilia.io import read_lilia_frame
from lilia.provenance import file_sha256
from lilia.signal import resample_segment_time_us
from lilia.tflite import build_tflite_timeline


def signal_parameters(model_path, max_samples=None):
    if max_samples is not None and (isinstance(max_samples, bool)
                                   or not isinstance(max_samples, int) or max_samples <= 0):
        raise ValueError('max_samples must be a positive integer or None')
    return {'input_fs': 500, 'fs': 200, 'input_channels': [1, 2, 3, 4], 'output_channels': [1, 2],
            'bandpass': [.5, 45.], 'filter_order': 4, 'filter_dtype': 'float32',
            'resampling': 'scipy.signal.resample_poly default', 'model_window': 400,
            'model_sha256': file_sha256(model_path), 'normalization': 'window RMS(float64) + 1e-8; scale float32',
            'model_padding': 'none', 'before_tail': 'retain_all_resampled_samples_of_retained_segments',
            'after_tail': 'trim_each_segment_to_complete_400_sample_windows',
            'short_segments': 'exclude_from_both_branches', 'nonfinite_input': 'fail_including_excluded',
            'max_samples': max_samples, 'truncation': 'filter_full_source_segment_then_truncate_raw_prefix',
            'quality_state': 'disabled'}


def sample_contract(time_us, max_samples=None):
    """Rebuild all row coordinates without filtering or model inference."""
    full = build_tflite_timeline(time_us, 500, 200, 400)
    n = len(time_us) if max_samples is None else min(len(time_us), max_samples)
    timeline = full if n == len(time_us) else build_tflite_timeline(time_us[:n], 500, 200, 400)
    if not len(timeline.time_us):
        raise ValueError('No complete Jenqwei model windows')
    times, groups, positions, after_map, records = [], [], [], [], []
    raw_offset = before_offset = 0
    for segment in timeline.segments:
        record = dict(segment)
        original = full.segments[segment['segment_id']]
        record.update(filter_context_start_idx=original['raw_start_idx'],
                      filter_context_end_idx=original['raw_end_idx'])
        records.append(record)
        if segment['status'] != 'retained':
            continue
        a, b = segment['raw_start_idx'], segment['raw_end_idx']
        count = segment['resampled_samples']
        times.append(resample_segment_time_us(time_us[a:b], 500, 200))
        groups.append(np.full(count, segment['segment_id'], dtype=np.int64))
        positions.append(a + np.arange(count) * 2.5)
        after_map.append(before_offset + np.arange(segment['retained_samples'], dtype=np.int64))
        record.update(packed_raw_start_idx=raw_offset, packed_raw_end_idx=raw_offset + b - a,
                      before_start_idx=before_offset, before_end_idx=before_offset + count)
        raw_offset += b - a
        before_offset += count
    before = {'sample_idx': np.arange(before_offset, dtype=np.int64), 'Time[us]': np.concatenate(times),
              'segment_id': np.concatenate(groups), 'raw_fractional_idx': np.concatenate(positions)}
    idx = np.concatenate(after_map)
    after = {'sample_idx': np.arange(len(idx), dtype=np.int64), 'Time[us]': timeline.time_us,
             'segment_id': timeline.segment_ids, 'raw_fractional_idx': before['raw_fractional_idx'][idx],
             'before_idx': idx}
    return {'inference': timeline.metadata(), 'segments': records, 'before': before, 'after': after,
            'full_source_samples': len(time_us), 'used_source_samples': n}


def table_columns(branch):
    if branch not in ('before', 'after'):
        raise ValueError('Unknown Jenqwei branch')
    return ['sample_idx', 'Time[us]', 'segment_id', 'raw_fractional_idx'] + (
        ['before_idx', 'ch1', 'ch2'] if branch == 'after' else ['ch1', 'ch2', 'ch3', 'ch4'])


def write_signal_tables(outdir, source, result, model_path, code_hashes):
    source = Path(source)
    outdir = Path(outdir)
    parameters = signal_parameters(model_path, result['max_samples'])
    paths = [outdir / f'{source.stem}_{branch}.csv' for branch in ('before', 'after')]
    for path in paths:
        if path.exists() or Path(str(path) + '.meta.json').exists():
            raise FileExistsError(f'Jenqwei output already exists: {path}')
    raw = read_lilia_frame(source)
    contract = sample_contract(raw.iloc[:, 0].to_numpy(dtype=np.int64), result['max_samples'])
    if result['segments'] != contract['segments'] or result['timeline'].metadata() != contract['inference']:
        raise ValueError('Jenqwei processing mapping differs from source')
    mappings = {
        'before': {'Time[us]': result['time_us_200'], 'segment_id': result['before_segment_ids'],
                   'raw_fractional_idx': result['before_raw_fractional_idx']},
        'after': {'Time[us]': result['tfl_time_us_200'], 'segment_id': result['timeline'].segment_ids,
                  'raw_fractional_idx': result['model_raw_fractional_idx'], 'before_idx': result['model_to_before_idx']}}
    artifacts = []
    for branch, path in zip(('before', 'after'), paths):
        values = np.asarray(result['pre_data_200' if branch == 'before' else 'tfl_data_200'])
        mapping = contract[branch]
        for key, actual in mappings[branch].items():
            if not np.array_equal(actual, mapping[key]):
                raise ValueError(f'Jenqwei {branch} {key} differs from source')
        n_ch = 4 if branch == 'before' else 2
        if values.shape != (len(mapping['sample_idx']), n_ch) or not np.isfinite(values).all():
            raise ValueError(f'Invalid Jenqwei {branch} signal values')
        columns = table_columns(branch)
        with path.open('x', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            for i, row in enumerate(values):
                coords = [float(mapping[key][i]) if key == 'raw_fractional_idx' else int(mapping[key][i])
                          for key in columns[:-n_ch]]
                writer.writerow([*coords, *map(float, row)])
        metadata = {'schema_version': 1, 'kind': 'jenqwei_signal', 'branch': branch,
                    'source_id': file_sha256(source), 'source_path': str(source.resolve()),
                    'parameters': parameters, 'config_id': config_id(parameters), 'code_sha256': code_hashes,
                    'columns': columns, 'index_space': f'packed_{branch}', 'quality_state': 'disabled',
                    **{k: contract[k] for k in ('inference', 'segments', 'full_source_samples', 'used_source_samples')},
                    'table_sha256': file_sha256(path)}
        sidecar = Path(str(path) + '.meta.json')
        with sidecar.open('x', encoding='utf-8') as handle:
            handle.write(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
        artifacts.extend([path, sidecar])
    return artifacts


def load_signal_table(path, raw_csv, *, model_path):
    """Verify source/model/settings, every row and timeline; do not rerun inference."""
    meta = json.loads(Path(str(path) + '.meta.json').read_text())
    if (meta.get('schema_version') != 1 or meta.get('kind') != 'jenqwei_signal'
            or meta.get('table_sha256') != file_sha256(path)):
        raise ValueError('Jenqwei schema, kind or table fingerprint mismatch')
    params = signal_parameters(model_path, meta.get('parameters', {}).get('max_samples'))
    if meta.get('parameters') != params or meta.get('config_id') != config_id(params):
        raise ValueError('Jenqwei model or configuration mismatch')
    if meta.get('source_id') != file_sha256(raw_csv):
        raise ValueError('Jenqwei source fingerprint mismatch')
    raw = read_lilia_frame(raw_csv)
    if raw.shape[1] < 5 or not np.isfinite(raw.iloc[:, 1:5].to_numpy(dtype=np.float32)).all():
        raise ValueError('Jenqwei source needs four finite channels including excluded segments')
    contract = sample_contract(raw.iloc[:, 0].to_numpy(dtype=np.int64), params['max_samples'])
    for key in ('inference', 'segments', 'full_source_samples', 'used_source_samples'):
        if meta.get(key) != contract[key]:
            raise ValueError(f'Jenqwei {key} differs from source')
    branch = meta.get('branch')
    columns = table_columns(branch)
    if (meta.get('columns') != columns or meta.get('index_space') != f'packed_{branch}'
            or meta.get('quality_state') != 'disabled'):
        raise ValueError('Jenqwei columns, index space or quality mismatch')
    mapping = contract[branch]
    coordinate_columns = list(mapping)
    data = {key: [] for key in columns}
    with open(path, newline='', encoding='utf-8') as handle:
        reader = csv.reader(handle)
        if next(reader, None) != columns:
            raise ValueError('Jenqwei CSV columns differ from metadata')
        for row in reader:
            if len(row) != len(columns):
                raise ValueError('Jenqwei CSV row width mismatch')
            for key, value in zip(columns, row):
                data[key].append(int(value) if key in coordinate_columns and key != 'raw_fractional_idx'
                                 else float(value))
    for key, expected in mapping.items():
        if not np.array_equal(data[key], expected):
            raise ValueError(f'Jenqwei {branch} {key} differs from source')
    frame = pd.DataFrame(data)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError('Non-finite Jenqwei signal values')
    return frame, meta
