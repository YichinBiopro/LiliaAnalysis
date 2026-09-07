"""Verify qEEG windows against a complete-window TFLite source mapping."""
from __future__ import annotations

import numpy as np
from lilia.entropy_io import _write_window_table, _load_window_table
from lilia.io import read_lilia_frame
from lilia.provenance import file_sha256
from lilia.tflite import build_tflite_timeline


def write_tflite_table(path, source, frame, grid, parameters, code_sha256, timeline):
    _write_window_table(path, source, frame, grid, parameters, code_sha256, 'tflite_qeeg',
                        source_info={'source_samples': timeline.source_samples,
                                     'source_epoch_us': timeline.source_epoch_us,
                                     'inference': timeline.metadata()})


def load_tflite_table(path, raw_csv=None, model_path=None):
    frame, meta = _load_window_table(path, kind='tflite_qeeg')
    if meta is None or 'inference' not in meta:
        raise ValueError('TFLite inference metadata is required')
    params = meta['parameters']
    if params.get('index_space') != 'retained_tflite_output':
        raise ValueError('Incorrect TFLite index space')
    if model_path is not None and file_sha256(model_path) != params['model_sha256']:
        raise ValueError('TFLite model differs from metadata')
    if raw_csv is not None:
        if file_sha256(raw_csv) != meta['source_id']:
            raise ValueError('Raw recording differs from TFLite source')
        raw = read_lilia_frame(raw_csv)
        t = raw.iloc[:, 0].to_numpy(dtype=np.int64)
        if len(raw.columns) != 5 or len(t) != meta['source_samples'] or int(t[0]) != meta['source_epoch_us']:
            raise ValueError('Raw source shape or epoch differs from TFLite metadata')
        timeline = build_tflite_timeline(t, params['input_fs'], params['fs'], params['model_window'])
        if timeline.metadata() != meta['inference']:
            raise ValueError('TFLite inference mapping does not match raw timestamps')
        grid = timeline.grid(params['win_sec'], params['step_sec'])
        for key, expected in grid.columns.items():
            actual = frame[key].to_numpy()
            valid = (len(actual) == len(expected) and
                     (np.allclose(actual, expected, rtol=0, atol=1e-9) if key == 'time_s'
                      else np.array_equal(actual, expected)))
            if not valid:
                raise ValueError(f'TFLite {key} does not match inference timestamps')
    return frame, meta
