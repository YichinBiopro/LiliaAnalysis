"""Event-marker qEEG tables with raw quality and metric source mapping."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lilia.entropy_io import _write_window_table, _load_window_table, config_id
from lilia.event_qeeg import INDEX_KEYS, raw_quality_mapping, summarize_branch
from lilia.io import read_lilia_frame
from lilia.provenance import file_sha256
from lilia.quality_audit import attach_diagnostic_columns, validate_diagnostics
from lilia.tflite import build_tflite_timeline
from lilia.windowing import build_window_grid


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_event_qeeg_table(path, source, branch, parameters, code_id, timeline=None):
    grid = branch['grid']
    frame = pd.DataFrame(branch['quality_mapping'])
    frame['quality_valid'] = branch['valid']
    for channel in range(branch['quality'].shape[1]):
        frame[f'quality_ch{channel+1}'] = branch['quality'][:, channel]
    for key in INDEX_KEYS:
        for channel in range(branch['scores'][key].shape[1]):
            frame[f'{key}_ch{channel+1}'] = np.where(branch['valid'], branch['scores'][key][:, channel], np.nan)
    for key in ('quality_status', 'metric_status'):
        frame[key] = [row[key] for row in branch['window_audit']]
    attach_diagnostic_columns(frame, branch['window_audit'])
    analysis = json_safe({'summary': branch['summary'], 'window_audit': branch['window_audit']})
    info = {'analysis': analysis, 'analysis_id': config_id(analysis), 'quality_diagnostics_version': 1}
    kind = 'event_marker_bp' if timeline is None else 'event_marker_tflite'
    if timeline is not None:
        info.update(inference=timeline.metadata(), source_samples=timeline.source_samples,
                    source_epoch_us=timeline.source_epoch_us)
    _write_window_table(path, source, frame, grid, parameters, code_id, kind, source_info=info)


def load_event_qeeg_table(path, raw_csv=None, model_path=None):
    metadata = json.loads(Path(str(path)+'.meta.json').read_text())
    kind = metadata.get('kind')
    if kind not in ('event_marker_bp', 'event_marker_tflite'):
        raise ValueError('Expected event-marker qEEG metadata')
    frame, meta = _load_window_table(path, kind=kind)
    params = meta['parameters']
    expected_space = 'raw_samples' if kind == 'event_marker_bp' else 'retained_tflite_output'
    if params['index_space'] != expected_space or meta['analysis_id'] != config_id(meta['analysis']):
        raise ValueError('Event qEEG index space or analysis fingerprint mismatch')
    if model_path is not None and params.get('model_sha256') != file_sha256(model_path):
        raise ValueError('TFLite model differs from event qEEG source')
    raw = None
    if raw_csv is not None:
        if file_sha256(raw_csv) != meta['source_id']:
            raise ValueError('Raw recording differs from event qEEG source')
        raw = read_lilia_frame(raw_csv)
        t = raw.iloc[:, 0].to_numpy(dtype=np.int64)
        if (len(t) != meta['source_samples'] or int(t[0]) != meta['source_epoch_us']
                or len(raw.columns)-1 != params['quality_channels']):
            raise ValueError('Event qEEG source shape or epoch mismatch')
        if kind == 'event_marker_bp':
            grid = build_window_grid(t, params['fs'], params['win_sec'], reset_per_segment=True)
        else:
            timeline = build_tflite_timeline(t, params['input_fs'], params['fs'], params['model_window'])
            if meta['inference'] != timeline.metadata():
                raise ValueError('Event qEEG inference mapping differs from raw')
            grid = timeline.grid(params['win_sec'])
        for key, expected in {**grid.columns, **raw_quality_mapping(t, grid, raw_index_space=kind == 'event_marker_bp')}.items():
            actual = frame[key].to_numpy()
            valid = len(actual) == len(expected) and (np.allclose(actual, expected, rtol=0, atol=1e-9)
                       if key == 'time_s' else np.array_equal(actual, expected))
            if not valid:
                raise ValueError(f'Event qEEG {key} mapping differs from source')
        scores = {k: frame[[f'{k}_ch{ch+1}' for ch in range(params['metric_channels'])]].to_numpy()
                  for k in INDEX_KEYS}
        reconstructed = summarize_branch({'grid': grid, 'scores': scores,
            'valid': frame.quality_valid.to_numpy(dtype=bool)}, params['events'], params['baseline_mode'])
        stored = meta['analysis']['summary']
        if reconstructed['bins'] != stored['bins']:
            raise ValueError('Event qEEG heatmap mapping differs from source')
        if len(reconstructed['baseline_audit']) != len(stored['baseline_audit']):
            raise ValueError('Event qEEG baseline count differs from source')
        for actual, expected in zip(reconstructed['baseline_audit'], stored['baseline_audit']):
            for key in ('baseline_metric_rows', 'event_metric_rows', 'baseline_bins', 'event_bins'):
                if actual.get(key) != expected.get(key):
                    raise ValueError('Event qEEG baseline mapping differs from source')
    validate_diagnostics(frame, meta, None if raw is None else raw.iloc[:, 1:].to_numpy(dtype=np.float32))
    return frame, meta
