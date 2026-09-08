"""Source/crop/model-verifiable meditation metrics without a quality scorer."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lilia.entropy_io import _write_window_table, _load_window_table, config_id
from lilia.event_qeeg import INDEX_KEYS
from lilia.event_qeeg_io import json_safe
from lilia.hardy2_io import _equivalent
from lilia.io import read_lilia_frame
from lilia.meditation import crop_layout, bp_grid, model_timeline, summarize_meditation, meditation_display
from lilia.provenance import file_sha256


def write_meditation_table(path, source, branch, parameters, code_id, result, tag):
    frame = pd.DataFrame({'metric_valid': branch['valid'], 'quality_state': 'disabled',
                          'metric_status': [r['metric_status'] for r in branch['window_audit']]})
    for k in INDEX_KEYS:
        for ch in range(branch['scores'][k].shape[1]):
            frame[f'{k}_ch{ch+1}'] = branch['scores'][k][:, ch]
    audit = json_safe({'crop': result['crop'], 'segments': result['segments'], 'display': result['display'],
                       'summary': branch['summary'], 'window_audit': branch['window_audit']})
    info = {'analysis': audit, 'analysis_id': config_id(audit),
            'source_samples': result['source_samples'], 'source_epoch_us': result['source_epoch_us']}
    if tag == 'tflite':
        info['inference'] = result['timeline'].metadata()
    _write_window_table(path, source, frame, branch['grid'], parameters, code_id, 'meditation_'+tag, source_info=info)


def load_meditation_table(path, raw_csv, model_path=None):
    import json
    from pathlib import Path
    kind = json.loads(Path(str(path)+'.meta.json').read_text()).get('kind')
    if kind not in ('meditation_bp', 'meditation_tflite'):
        raise ValueError('Expected meditation metadata')
    f, m = _load_window_table(path, kind=kind)
    p = m['parameters']
    is_bp = kind == 'meditation_bp'
    if (p['quality_state'] != 'disabled' or p['heatmap_bin_windows'] != 6
            or p['index_space'] != ('raw_samples' if is_bp else 'retained_tflite_output')
            or p['step_sec'] != p['win_sec'] or m['analysis_id'] != config_id(m['analysis'])):
        raise ValueError('Meditation policy or analysis fingerprint mismatch')
    if file_sha256(raw_csv) != m['source_id']:
        raise ValueError('Raw recording differs from meditation source')
    if not is_bp and model_path is not None and file_sha256(model_path) != p['model_sha256']:
        raise ValueError('Meditation model differs from checkpoint')
    raw = read_lilia_frame(raw_csv)
    t = raw.iloc[:, 0].to_numpy(dtype=np.int64)
    if len(t) != m['source_samples'] or int(t[0]) != m['source_epoch_us'] or raw.shape[1]-1 != p['source_channels']:
        raise ValueError('Meditation source shape or epoch mismatch')
    if p['metric_channels'] != (raw.shape[1]-1 if is_bp else 2):
        raise ValueError('Meditation metric channel selection differs')
    crop = crop_layout(t, p['input_fs'], p['crop_start_us'], p['crop_end_us'])
    if crop != m['analysis']['crop']:
        raise ValueError('Meditation source crop mapping differs')
    if p['display_channels'] != [0, 1] or not _equivalent(m['analysis']['display'],
            json_safe(meditation_display(t, p, p['display_downsample']))):
        raise ValueError('Meditation display crop differs from source')
    if is_bp:
        grid = bp_grid(t, crop, p['fs'], p['win_sec'])
    else:
        timeline = model_timeline(t, crop, p['input_fs'], p['fs'], p['model_window'])
        if timeline.metadata() != m['inference']:
            raise ValueError('Meditation inference mapping differs from source')
        grid = timeline.grid(p['win_sec'])
    if len(grid.starts) != len(f):
        raise ValueError('Meditation source window count differs')
    for k, expected in grid.columns.items():
        actual = f[k].to_numpy()
        if not (np.allclose(actual, expected, rtol=0, atol=1e-9) if k == 'time_s' else np.array_equal(actual, expected)):
            raise ValueError(f'Meditation {k} mapping differs from source')
    scores = {k: f[[f'{k}_ch{i+1}' for i in range(p['metric_channels'])]].to_numpy() for k in INDEX_KEYS}
    valid = np.logical_and.reduce([np.isfinite(v).all(axis=1) for v in scores.values()])
    if not np.array_equal(valid, f.metric_valid) or not (f.quality_state == 'disabled').all():
        raise ValueError('Meditation validity or quality state differs')
    branch = {'grid': grid, 'scores': scores, 'valid': valid}
    summary = summarize_meditation(branch, p['baseline_end_us'], p['view_start_us'], p['view_end_us'])
    if not _equivalent(m['analysis']['summary'], json_safe(summary)):
        raise ValueError('Meditation baseline or heatmap differs from source windows')
    saved = m['analysis']['segments']
    if len(saved) != len(crop['segments']):
        raise ValueError('Meditation segment count differs')
    for row, actual in zip(crop['segments'], saved):
        n = row['raw_end_idx']-row['raw_start_idx']
        expected = {**row, 'bp_complete_windows': n//round(p['input_fs']*p['win_sec']),
                    'bp_tail_samples': n%round(p['input_fs']*p['win_sec'])}
        if any(actual[k] != v for k, v in expected.items()):
            raise ValueError('Meditation segment tail mapping differs')
    audit = m['analysis']['window_audit']
    if ([r['metric_row'] for r in audit] != list(range(len(f)))
            or [r['metric_status'] for r in audit] != f.metric_status.tolist()
            or any(r['quality_state'] != 'disabled' for r in audit)):
        raise ValueError('Meditation metric audit differs')
    return f, m
