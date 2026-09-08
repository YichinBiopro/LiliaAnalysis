"""Cropped meditation BP/model timelines with explicit, unscored baselines."""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from lilia.event_qeeg import INDEX_KEYS
from lilia.event_zoom import select_zoom
from lilia.io import bandpass_filter
from lilia.qeeg import compute_qeeg_indices
from lilia.tflite import build_tflite_timeline, run_tflite_recording
from lilia.windowing import build_window_grid, continuous_slices


def meditation_display(time_us, parameters, ds):
    return select_zoom(time_us, None, [{'label': 'View',
        'start_us': max(parameters['view_start_us'], parameters['crop_start_us']),
        'end_us': min(parameters['view_end_us'], parameters['crop_end_us']),
        'color': '#911eb4', 'participates': True}], 'View', ds, parameters.get('input_fs', 500.))


def crop_layout(time_us, fs, start_us, end_us):
    t = np.asarray(time_us)
    if start_us >= end_us:
        raise ValueError('Analysis crop must have positive duration')
    segments = continuous_slices(t, fs)
    a, b = (int(np.searchsorted(t, v)) for v in (start_us, end_us))
    rows = []
    for sid, sl in enumerate(segments):
        lo, hi = max(a, sl.start), min(b, sl.stop)
        if lo < hi:
            rows.append({'segment_id': sid, 'source_start_idx': sl.start, 'source_end_idx': sl.stop,
                         'raw_start_idx': lo, 'raw_end_idx': hi, 'raw_start_us': int(t[lo]),
                         'raw_end_us': min(int(end_us), int(t[hi-1])+round(1e6/fs))})
    return {'start_us': int(start_us), 'end_us': int(end_us), 'raw_start_idx': a,
            'raw_end_idx': b, 'segments': rows}


def bp_grid(time_us, crop, fs, win_sec):
    t = np.asarray(time_us)
    a, b = crop['raw_start_idx'], crop['raw_end_idx']
    ids = np.concatenate([np.full(r['raw_end_idx']-r['raw_start_idx'], r['segment_id'], dtype=np.int64)
                          for r in crop['segments']])
    grid = build_window_grid(t[a:b], fs, win_sec, segment_ids=ids, epoch_us=int(t[0]), reset_per_segment=True)
    return replace(grid, n_samples=len(t), columns={**grid.columns,
        'window_start_idx': grid.starts+a, 'window_end_idx': grid.columns['window_end_idx']+a,
        'window_end_us': np.minimum(grid.columns['window_end_us'], crop['end_us'])})


def global_timeline(timeline, crop, source_samples, source_epoch_us):
    """Translate cropped adapter mappings back to the complete raw recording."""
    offset = crop['raw_start_idx']
    original_ids = [r['segment_id'] for r in crop['segments']]
    def translate(row):
        mapped = {**row, 'segment_id': original_ids[row['segment_id']],
                  'raw_start_idx': row['raw_start_idx']+offset, 'raw_end_idx': row['raw_end_idx']+offset}
        for key in ('raw_end_us', 'window_end_us'):
            if key in mapped:
                mapped[key] = min(mapped[key], crop['end_us'])
        return mapped
    return replace(timeline, source_samples=source_samples, source_epoch_us=source_epoch_us,
        segment_ids=np.asarray(original_ids, dtype=np.int64)[timeline.segment_ids],
        segments=[translate(r) for r in timeline.segments], model_windows=[translate(r) for r in timeline.model_windows])


def model_timeline(time_us, crop, fs=500., model_fs=200., model_window=400):
    t = np.asarray(time_us)
    local = build_tflite_timeline(t[crop['raw_start_idx']:crop['raw_end_idx']], fs, model_fs, model_window)
    return global_timeline(local, crop, len(t), int(t[0]))


def metric_branch(data, grid, fs, *, index_offset=0):
    scores = {k: np.full((len(grid.starts), data.shape[1]), np.nan) for k in INDEX_KEYS}
    audit = []
    for i, start in enumerate(grid.starts):
        window = data[start-index_offset:start-index_offset+grid.win]
        row = {'metric_row': i, 'metric_status': 'nonfinite_filtered_segment', 'quality_state': 'disabled'}
        if np.isfinite(window).all():
            try:
                for ch in range(data.shape[1]):
                    values = compute_qeeg_indices(window[:, ch].astype(np.float64), fs=fs)
                    for k in INDEX_KEYS:
                        scores[k][i, ch] = values[k]
                row['metric_status'] = 'computed' if all(np.isfinite(scores[k][i]).all() for k in INDEX_KEYS) else 'nonfinite_metrics'
            except Exception as exc:
                for k in INDEX_KEYS:
                    scores[k][i] = np.nan
                row.update(metric_status='metric_error', error=str(exc))
        audit.append(row)
    valid = np.logical_and.reduce([np.isfinite(v).all(axis=1) for v in scores.values()])
    return {'grid': grid, 'scores': scores, 'valid': valid, 'window_audit': audit}


def summarize_meditation(branch, baseline_end_us, view_start_us, view_end_us, bin_size=6):
    if view_start_us >= view_end_us:
        raise ValueError('Display interval must have positive duration')
    grid, scores, valid = branch['grid'], branch['scores'], branch['valid']
    bins, tails = [], []
    groups = grid.columns['segment_id']
    for sid in np.unique(groups):
        rows = np.flatnonzero(groups == sid)
        keep = len(rows)//bin_size*bin_size
        tails.append({'segment_id': int(sid), 'unbinned_metric_rows': rows[keep:].tolist()})
        for a in range(0, keep, bin_size):
            chunk = rows[a:a+bin_size]
            lo, hi = int(grid.columns['window_start_us'][chunk[0]]), int(grid.columns['window_end_us'][chunk[-1]])
            bins.append({'segment_id': int(sid), 'metric_rows': chunk.tolist(),
                         'valid_rows': chunk[valid[chunk]].tolist(), 'start_us': lo,
                         'end_us': hi, 'center_us': (lo+hi)//2})
    absolute = np.full((4, len(bins)), np.nan)
    for i, item in enumerate(bins):
        if item['valid_rows']:
            for j, k in enumerate(INDEX_KEYS):
                absolute[j, i] = np.median(np.median(scores[k][item['valid_rows']], axis=1))
    candidates = [i for i, item in enumerate(bins) if item['end_us'] <= baseline_end_us]
    accepted = [i for i in candidates if np.isfinite(absolute[:, i]).all()]
    reference = np.median(absolute[:, accepted], axis=1) if accepted else np.full(4, np.nan)
    delta = absolute-reference[:, None]
    display = [i for i, item in enumerate(bins) if item['start_us'] < view_end_us and item['end_us'] > view_start_us]
    visible = [i for i in display if np.isfinite(delta[:, i]).all()]
    status = ('no_complete_bins' if not bins else 'missing_baseline' if not accepted
              else 'no_display_bins' if not display else 'no_finite_display_bins' if not visible else 'computed')
    return {'bins': bins, 'tails': tails, 'heatmap_abs': absolute, 'heatmap_delta': delta,
            'baseline_candidate_bins': candidates, 'baseline_accepted_bins': accepted,
            'baseline_excluded_bins': [i for i in candidates if i not in accepted],
            'baseline_reference': reference, 'display_bins': display, 'finite_display_bins': visible,
            'status': status, 'quality_state': 'disabled'}


def analyze_meditation(time_us, raw, *, crop_start_us, crop_end_us, baseline_end_us,
                       view_start_us, view_end_us, fs=500., win_sec=5., model_path=None,
                       model_fs=200., model_window=400):
    t, raw = np.asarray(time_us), np.asarray(raw)
    if t.dtype.kind not in 'iu' or raw.ndim != 2 or len(t) != len(raw) or not len(t):
        raise ValueError('Expected nonempty integer timestamps and aligned channels')
    crop = crop_layout(t, fs, crop_start_us, crop_end_us)
    result = {'crop': crop, 'segments': [], 'bp': None, 'tflite': None, 'timeline': None, 'errors': []}
    a, b = crop['raw_start_idx'], crop['raw_end_idx']
    if a == b:
        result['errors'].append('No raw samples inside analysis crop')
        return result
    try:
        grid = bp_grid(t, crop, fs, win_sec)
    except ValueError as exc:
        if 'No complete analysis window' not in str(exc):
            raise
        grid = None
        result['errors'].append(str(exc))
    timeline = model_timeline(t, crop, fs, model_fs, model_window) if model_path is not None else None
    result['timeline'] = timeline
    filtered = np.full(raw[a:b].shape, np.nan, dtype=np.float32)
    for i, row in enumerate(crop['segments']):
        lo, hi = row['raw_start_idx'], row['raw_end_idx']
        length = hi-lo
        item = {**row, 'bp_complete_windows': length//round(fs*win_sec),
                'bp_tail_samples': length%round(fs*win_sec)}
        needed = item['bp_complete_windows'] > 0 or (timeline is not None and timeline.segments[i]['retained_samples'] > 0)
        if not needed:
            item['status'] = 'short_segment'
        elif not np.isfinite(raw[lo:hi]).all():
            item['status'] = 'nonfinite_segment'
        else:
            try:
                filtered[lo-a:hi-a] = bandpass_filter(raw[lo:hi], fs=fs)
                item['status'] = 'filtered'
            except ValueError as exc:
                item.update(status='filter_error', error=str(exc))
        result['segments'].append(item)
    if grid is not None:
        result['bp'] = metric_branch(filtered, grid, fs, index_offset=a)
    if timeline is not None:
        try:
            _, _, output = run_tflite_recording(t[a:b], filtered, model_path, fs, model_fs, model_window)
            result['tflite'] = metric_branch(output, timeline.grid(win_sec), model_fs)
        except Exception as exc:
            result['errors'].append(f'TFLite failed: {exc}')
    for tag in ('bp', 'tflite'):
        branch = result[tag]
        if branch is not None:
            branch['summary'] = summarize_meditation(branch, baseline_end_us, view_start_us, view_end_us)
            if branch['summary']['status'] != 'computed':
                result['errors'].append(f'{tag}: {branch["summary"]["status"]}')
    return result
