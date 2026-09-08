"""Persist comparison windows, raw quality mappings and baseline selections."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lilia.entropy_io import _write_window_table, _load_window_table, config_id
from lilia.event_qeeg import INDEX_KEYS, raw_quality_mapping
from lilia.event_qeeg_io import json_safe
from lilia.hardy2_io import _equivalent
from lilia.io import read_lilia_frame
from lilia.provenance import file_sha256
from lilia.subject_comparison import summarize_comparison
from lilia.tflite import build_tflite_timeline
from lilia.windowing import build_window_grid, continuous_slices


def write_comparison_table(path, source, branch, parameters, code_id, segments, timeline=None):
    frame = pd.DataFrame(branch['quality_mapping'])
    frame['metric_valid'] = branch['valid']
    for ch in range(branch['quality'].shape[1]):
        frame[f'quality_ch{ch+1}'] = branch['quality'][:,ch]
    for k in INDEX_KEYS:
        for ch in range(branch['scores'][k].shape[1]):
            frame[f'{k}_ch{ch+1}'] = branch['scores'][k][:,ch]
    for key in ('quality_status','metric_status'):
        frame[key] = [r[key] for r in branch['window_audit']]
    analysis = json_safe({'summary':branch['summary'],'window_audit':branch['window_audit'],'segments':segments})
    info = {'analysis':analysis,'analysis_id':config_id(analysis)}
    if timeline is not None:
        info.update(inference=timeline.metadata(),source_samples=timeline.source_samples,source_epoch_us=timeline.source_epoch_us)
    kind = 'subject_comparison_bp' if timeline is None else 'subject_comparison_tflite'
    _write_window_table(path,source,frame,branch['grid'],parameters,code_id,kind,source_info=info)


def load_comparison_table(path, raw_csv, model_path=None):
    from pathlib import Path
    import json
    kind = json.loads(Path(str(path)+'.meta.json').read_text()).get('kind')
    if kind not in ('subject_comparison_bp','subject_comparison_tflite'):
        raise ValueError('Expected subject comparison metadata')
    frame,meta = _load_window_table(path,kind=kind)
    p = meta['parameters']
    is_bp = kind=='subject_comparison_bp'
    if (p['index_space']!=('raw_samples' if is_bp else 'retained_tflite_output')
            or meta['analysis_id']!=config_id(meta['analysis']) or p['step_sec']!=p['win_sec']
            or not np.isfinite(p['quality_threshold']) or not 0<=p['quality_threshold']<=1):
        raise ValueError('Comparison analysis or index space mismatch')
    if file_sha256(raw_csv)!=meta['source_id']:
        raise ValueError('Raw recording differs from comparison source')
    if model_path is not None and not is_bp and file_sha256(model_path)!=p['model_sha256']:
        raise ValueError('Comparison model differs from checkpoint')
    raw = read_lilia_frame(raw_csv)
    t = raw.iloc[:,0].to_numpy(dtype=np.int64)
    if len(t)!=meta['source_samples'] or int(t[0])!=meta['source_epoch_us'] or len(raw.columns)-1!=p['quality_channels']:
        raise ValueError('Comparison source shape or epoch mismatch')
    if is_bp:
        grid = build_window_grid(t,p['fs'],p['win_sec'],reset_per_segment=True)
    else:
        timeline = build_tflite_timeline(t,p['input_fs'],p['fs'],p['model_window'])
        if timeline.metadata()!=meta['inference']:
            raise ValueError('Comparison inference mapping differs from source')
        grid = timeline.grid(p['win_sec'])
    if len(frame)!=len(grid.starts):
        raise ValueError('Comparison window count differs from source')
    for key,expected in {**grid.columns,**raw_quality_mapping(t,grid,raw_index_space=is_bp)}.items():
        actual = frame[key].to_numpy()
        same = np.allclose(actual,expected,rtol=0,atol=1e-9) if key=='time_s' else np.array_equal(actual,expected)
        if not same:
            raise ValueError(f'Comparison {key} mapping differs from source')
    segments = continuous_slices(t,p['input_fs'])
    saved = meta['analysis']['segments']
    win = round(p['input_fs']*p['win_sec'])
    if len(saved)!=len(segments):
        raise ValueError('Comparison source segment count differs')
    for sid,(sl,row) in enumerate(zip(segments,saved)):
        expected = {'segment_id':sid,'raw_start_idx':sl.start,'raw_end_idx':sl.stop,
            'raw_start_us':int(t[sl.start]),'raw_end_us':int(t[sl.stop-1])+round(1e6/p['input_fs']),
            'bp_complete_windows':(sl.stop-sl.start)//win,'bp_tail_samples':(sl.stop-sl.start)%win}
        if any(row[k]!=v for k,v in expected.items()):
            raise ValueError('Comparison source segment mapping differs')
    quality = frame[[f'quality_ch{i+1}' for i in range(p['quality_channels'])]].to_numpy()
    valid = np.isfinite(quality).all(axis=1) & ((quality>=0)&(quality<=1)).all(axis=1) & (np.median(quality,axis=1)>=p['quality_threshold'])
    scores = {k:frame[[f'{k}_ch{i+1}' for i in range(p['metric_channels'])]].to_numpy() for k in INDEX_KEYS}
    for values in scores.values():
        valid &= np.isfinite(values).all(axis=1)
    if not np.array_equal(frame.metric_valid,valid):
        raise ValueError('Comparison validity differs from quality and metrics')
    rows = meta['analysis']['window_audit']
    if [r['metric_row'] for r in rows]!=list(range(len(frame))):
        raise ValueError('Comparison window audit mapping differs')
    for key in ('quality_status','metric_status'):
        if frame[key].tolist()!=[r[key] for r in rows]:
            raise ValueError('Comparison status differs from audit')
    summary = summarize_comparison({'grid':grid,'valid':valid,'scores':scores},p['events'])
    if not _equivalent(meta['analysis']['summary'],json_safe(summary)):
        raise ValueError('Comparison baseline or delta summary differs from source windows')
    return frame,meta
