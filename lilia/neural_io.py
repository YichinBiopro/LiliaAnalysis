"""Verified denoised MI tables: raw source -> inference axis -> MI windows."""
from __future__ import annotations

import numpy as np

from lilia.entropy_io import _write_window_table, _load_window_table
from lilia.io import read_lilia_frame
from lilia.neural import build_inference_timeline
from lilia.provenance import file_sha256


def write_denoised_joint_mi_table(path, source, frame, grid, parameters, code_sha256, timeline):
    _write_window_table(path, source, frame, grid, parameters, code_sha256, 'denoised_joint_mi',
                        source_info={'source_samples': timeline.source_samples,
                                     'source_epoch_us': timeline.source_epoch_us,
                                     'inference': timeline.metadata()})


def load_denoised_joint_mi_table(path, raw_csv=None, model_path=None):
    """Check fingerprints and reconstruct the output axis without running a model.

    Raw and denoised indexes have distinct kinds. If supplied, model_path also
    verifies the checkpoint; reading a table never loads an executable model.
    """
    frame, metadata = _load_window_table(path, kind='denoised_joint_mi')
    if metadata is None or 'inference' not in metadata:
        raise ValueError('Denoised joint-MI inference metadata is required')
    params = metadata['parameters']
    if params.get('index_space') != 'resampled_model_output' or not params.get('denoise'):
        raise ValueError('Incorrect denoised joint-MI index space')
    if model_path is not None and file_sha256(model_path) != params['model']['checkpoint_sha256']:
        raise ValueError('Model checkpoint differs from denoised joint-MI source')
    if raw_csv is not None:
        if file_sha256(raw_csv) != metadata['source_id']:
            raise ValueError('Raw recording differs from denoised joint-MI source')
        raw = read_lilia_frame(raw_csv)
        t = raw.iloc[:, 0].to_numpy(dtype=np.int64)
        if len(raw.columns) < 5 or len(t) != metadata['source_samples'] or int(t[0]) != metadata['source_epoch_us']:
            raise ValueError('Raw source shape or epoch differs from inference metadata')
        timeline = build_inference_timeline(t, params['input_fs'], params['fs'],
                                           params['model_window'], params['model_hop'])
        if timeline.metadata() != metadata['inference']:
            raise ValueError('Inference mapping does not match raw timestamps')
        grid = timeline.grid(params['win_sec'], params['step_sec'])
        for key, expected in grid.columns.items():
            actual = frame[key].to_numpy()
            valid = (len(actual) == len(expected) and
                     (np.allclose(actual, expected, rtol=0, atol=1e-9) if key == 'time_s'
                      else np.array_equal(actual, expected)))
            if not valid:
                raise ValueError(f'Denoised joint-MI {key} does not match inference timestamps')
    return frame, metadata
