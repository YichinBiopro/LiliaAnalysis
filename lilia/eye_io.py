"""Source-verifiable eye open/close model output with the legacy CSV layout."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lilia.entropy_io import config_id
from lilia.io import read_lilia_frame
from lilia.neural import build_inference_timeline, model_provenance
from lilia.provenance import file_sha256

OUTPUT_CHANNELS = [1, 2, 5, 6]
COLUMNS = ['Time[us]', 'ch1', 'ch2', 'ch5', 'ch6']


def signal_parameters():
    """Describe the eye filter, without inheriting the general denoiser filters."""
    import data_analysis as da

    model = model_provenance()
    model['filters'] = {'bandpass': [da.BANDPASS_LOW, da.BANDPASS_HIGH]}
    return {
        'index_space': 'resampled_model_output', 'input_fs': da.FS,
        'fs': da.DOWNSAMPLED_FS, 'model_window': da.MODEL_WINDOW,
        'model_hop': da.MODEL_WINDOW // 2, 'input_channels': list(range(1, 9)),
        'model_input_groups': [[1, 2, 3, 4], [5, 6, 7, 8]],
        'output_channels': OUTPUT_CHANNELS, 'filter_dtype': 'float32',
        'filter_order': 4, 'resampling': 'scipy.signal.resample_poly default',
        'short_segments': 'exclude_if_resampled_samples_less_than_model_window',
        'nonfinite_input': 'fail_entire_analysis_including_excluded_segments',
        'model_failure': 'fail_entire_analysis', 'quality_state': 'disabled',
        'model': model,
    }


def sample_mapping(timeline):
    """Packed output indexes and source positions; model padding is in inference."""
    fractional = [s['raw_start_idx'] + i * s['ratio_down'] / s['ratio_up']
                  for s in timeline.segments if s['status'] == 'retained'
                  for i in range(s['resampled_samples'])]
    return {'output_idx': list(range(len(timeline.time_us))),
            'segment_id': timeline.segment_ids.tolist(),
            'raw_fractional_idx': fractional}


def output_header(source, fs):
    with open(source, encoding='utf-8-sig') as handle:
        lines = [handle.readline().rstrip('\r\n') for _ in range(4)]
    lines[0] += ',Processing,TinyUNetV4,Output source channels,1/2/5/6'
    lines[2] = 'Channels,1,2,5,6'
    lines[3] = 'Sample Rate (per channel),' + ','.join([str(fs)] * 4)
    return lines


def write_signal_table(path, source, timeline, output, parameters, code_hashes):
    """Write finite outputs with exact integer microseconds; never replace files."""
    path = Path(path)
    sidecar = Path(str(path) + '.meta.json')
    if path.exists() or sidecar.exists():
        raise FileExistsError(f'Eye output already exists: {path} or {sidecar}')
    output = np.asarray(output)
    if output.shape != (len(timeline.time_us), 4) or not len(output) or not np.isfinite(output).all():
        raise ValueError('Eye output must contain four finite channels aligned to the timeline')
    meta = {
        'schema_version': 1, 'kind': 'eye_model_signal',
        'source_path': str(Path(source).resolve()), 'source_id': file_sha256(source),
        'parameters': parameters, 'config_id': config_id(parameters),
        'code_sha256': code_hashes, 'inference': timeline.metadata(),
        'sample_mapping': sample_mapping(timeline), 'columns': COLUMNS,
        'quality_state': 'disabled',
        'model_input_index_space': 'segment_local_resampled_with_mirrored_padding',
    }
    with path.open('x', encoding='utf-8', newline='') as handle:
        for line in output_header(source, parameters['fs']):
            handle.write(line + '\n')
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for timestamp, row in zip(timeline.time_us, output):
            writer.writerow([int(timestamp), *map(float, row)])
    meta['table_sha256'] = file_sha256(path)
    with sidecar.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(meta, indent=2, allow_nan=False) + '\n')
    return [path, sidecar]


def load_signal_table(path, raw_csv, *, model_path):
    """Verify source/model/config, all sample mappings and the complete timeline.

    The table hash checks stored signal values; this reader does not repeat model
    inference. Independent numerical evidence remains necessary for equivalence.
    """
    meta = json.loads(Path(str(path) + '.meta.json').read_text(encoding='utf-8'))
    if (meta.get('schema_version') != 1 or meta.get('kind') != 'eye_model_signal'
            or meta.get('table_sha256') != file_sha256(path)):
        raise ValueError('Eye table kind, schema or fingerprint mismatch')
    parameters = signal_parameters()
    if file_sha256(model_path) != parameters['model']['checkpoint_sha256']:
        raise ValueError('Eye checkpoint differs from the processing model')
    if meta.get('parameters') != parameters or meta.get('config_id') != config_id(parameters):
        raise ValueError('Eye configuration or model fingerprint mismatch')
    if meta.get('source_id') != file_sha256(raw_csv):
        raise ValueError('Raw recording differs from eye source')
    raw = read_lilia_frame(raw_csv)
    if raw.shape[1] < 9 or not np.isfinite(raw.iloc[:, 1:9].to_numpy(dtype=np.float32)).all():
        raise ValueError('Eye source must contain eight finite input channels')
    timeline = build_inference_timeline(raw.iloc[:, 0].to_numpy(dtype=np.int64),
                                        parameters['input_fs'], parameters['fs'],
                                        parameters['model_window'], parameters['model_hop'])
    if not len(timeline.time_us) or meta.get('inference') != timeline.metadata():
        raise ValueError('Eye inference mapping differs from raw timestamps')
    if meta.get('sample_mapping') != sample_mapping(timeline):
        raise ValueError('Eye sample mapping differs from source')
    if (meta.get('columns') != COLUMNS or meta.get('quality_state') != 'disabled'
            or meta.get('model_input_index_space') != 'segment_local_resampled_with_mirrored_padding'):
        raise ValueError('Eye channel, quality or index space mismatch')
    with open(path, encoding='utf-8', newline='') as handle:
        header = [handle.readline().rstrip('\r\n') for _ in range(4)]
        rows = csv.reader(handle)
        if header != output_header(raw_csv, parameters['fs']) or next(rows, None) != COLUMNS:
            raise ValueError('Eye CSV header or channel mapping mismatch')
        timestamps, values = [], []
        for row in rows:
            if len(row) != len(COLUMNS):
                raise ValueError('Eye CSV row width mismatch')
            timestamps.append(int(row[0]))
            values.append([float(value) for value in row[1:]])
    if not np.array_equal(timestamps, timeline.time_us):
        raise ValueError('Eye output timestamps differ from source')
    values = np.asarray(values)
    if values.shape != (len(timeline.time_us), 4) or not np.isfinite(values).all():
        raise ValueError('Eye output contains invalid model values')
    frame = pd.DataFrame(values, columns=COLUMNS[1:])
    frame.insert(0, COLUMNS[0], np.asarray(timestamps, dtype=np.int64))
    return frame, meta
