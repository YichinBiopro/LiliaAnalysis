"""Persist and reconstruct the source windows and display crop for event zoom."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lilia.entropy_io import config_id
from lilia.event_qeeg import INDEX_KEYS, summarize_branch
from lilia.event_qeeg_io import json_safe, write_event_qeeg_table, load_event_qeeg_table
from lilia.event_zoom import select_zoom
from lilia.hardy2_io import _equivalent
from lilia.io import read_lilia_frame
from lilia.windowing import build_window_grid, continuous_slices


def write_zoom_table(path, source, branch, parameters, code_id, selection, segments):
    write_event_qeeg_table(path, source, branch, parameters, code_id)
    side = Path(str(path)+'.meta.json')
    meta = json.loads(side.read_text())
    meta['zoom'] = json_safe({'selection': selection, 'segments': segments})
    meta['zoom_id'] = config_id(meta['zoom'])
    side.write_text(json.dumps(meta, indent=2, allow_nan=False)+'\n')


def load_zoom_table(path, raw_csv):
    frame, meta = load_event_qeeg_table(path, raw_csv)
    p = meta['parameters']
    if (meta['kind'] != 'event_marker_bp' or p.get('scope') != 'event_zoom'
            or p['baseline_mode'] != 'pre-event-rest' or p['heatmap_bin_windows'] != 6
            or p['step_sec'] != p['win_sec'] or meta['zoom_id'] != config_id(meta['zoom'])):
        raise ValueError('Event zoom policy or fingerprint mismatch')
    raw = read_lilia_frame(raw_csv)
    t = raw.iloc[:, 0].to_numpy(dtype=np.int64)
    grid = build_window_grid(t, p['fs'], p['win_sec'], reset_per_segment=True)
    quality = frame[[f'quality_ch{i+1}' for i in range(p['quality_channels'])]].to_numpy()
    scores = {k: frame[[f'{k}_ch{i+1}' for i in range(p['metric_channels'])]].to_numpy() for k in INDEX_KEYS}
    valid = (np.isfinite(quality).all(axis=1) & ((quality >= 0) & (quality <= 1)).all(axis=1)
             & (np.median(quality, axis=1) >= p['quality_threshold']))
    for values in scores.values():
        valid &= np.isfinite(values).all(axis=1)
    if not np.array_equal(frame.quality_valid, valid):
        raise ValueError('Event zoom validity differs from quality and metrics')
    branch = {'grid': grid, 'valid': valid, 'scores': scores}
    branch['summary'] = summarize_branch(branch, p['events'], 'pre-event-rest')
    if not _equivalent(meta['analysis']['summary'], json_safe(branch['summary'])):
        raise ValueError('Event zoom summary differs from source windows')
    selection = select_zoom(t, branch, p['events'], p['target_event'], p['display_downsample'], p['fs'])
    if not _equivalent(meta['zoom']['selection'], json_safe(selection)):
        raise ValueError('Event zoom crop differs from source windows')
    segments = continuous_slices(t, p['fs'])
    saved = meta['zoom']['segments']
    if len(segments) != len(saved):
        raise ValueError('Event zoom segment count differs from source')
    for sid, (sl, row) in enumerate(zip(segments, saved)):
        expected = {'segment_id': sid, 'raw_start_idx': sl.start, 'raw_end_idx': sl.stop,
                    'raw_start_us': int(t[sl.start]), 'raw_end_us': int(t[sl.stop-1])+round(1e6/p['fs']),
                    'bp_complete_windows': (sl.stop-sl.start)//grid.win,
                    'bp_tail_samples': (sl.stop-sl.start)%grid.win}
        if any(row[k] != v for k, v in expected.items()):
            raise ValueError('Event zoom segment mapping differs from source')
    rows = meta['analysis']['window_audit']
    if [r['metric_row'] for r in rows] != list(range(len(frame))):
        raise ValueError('Event zoom window audit mapping differs')
    for key in ('quality_status', 'metric_status'):
        if frame[key].tolist() != [r[key] for r in rows]:
            raise ValueError('Event zoom window status differs from audit')
    return frame, meta
