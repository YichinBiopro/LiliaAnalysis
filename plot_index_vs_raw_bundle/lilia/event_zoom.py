"""Select one event from the shared pre-event-rest BP analysis."""
from __future__ import annotations

import numpy as np

from lilia.event_qeeg import complete_mask
from lilia.windowing import continuous_slices


def select_zoom(time_us, branch, events, label, ds=10, fs=500.):
    """Keep full-session baselines and crop display samples within source runs.

    Overlapping boundary bins remain visible as missing; only bins completely
    inside the target event may display a delta. No nearest/all-bin fallback.
    """
    if not isinstance(ds, int) or isinstance(ds, bool) or ds < 1:
        raise ValueError('Display downsample factor must be a positive integer')
    if len({e['label'] for e in events}) != len(events):
        raise ValueError('Event labels must be unique')
    if any(e['start_us'] >= e['end_us'] for e in events):
        raise ValueError('Events must have positive duration')
    matches = [e for e in events if e['label'] == label]
    if not matches:
        raise ValueError(f'Unknown event: {label}')
    target = matches[0]
    lo, hi = target['start_us'], target['end_us']
    t = np.asarray(time_us)
    segments = continuous_slices(t, fs)
    runs, gaps = [], []
    cursor = lo
    for sid, sl in enumerate(segments):
        a = max(sl.start, int(np.searchsorted(t, lo)))
        b = min(sl.stop, int(np.searchsorted(t, hi)))
        if a >= b:
            continue
        indices = np.unique(np.r_[np.arange(a, b, ds), b-1])
        start, end = int(t[a]), min(hi, int(t[b-1])+round(1e6/fs))
        if start > cursor:
            gaps.append({'start_us': cursor, 'end_us': start})
        cursor = max(cursor, end)
        runs.append({'segment_id': sid, 'raw_start_idx': a, 'raw_end_idx': b,
                     'start_us': start, 'end_us': end, 'display_raw_indices': indices.tolist()})
    if cursor < hi:
        gaps.append({'start_us': cursor, 'end_us': hi})
    selection = {'target': target, 'display_runs': runs, 'missing_raw_intervals': gaps,
                 'display_bins': [], 'complete_event_bins': [], 'valid_delta_bins': [],
                 'boundary_bins': [], 'baseline': None, 'baseline_candidate_rows': [],
                 'event_candidate_rows': [], 'baseline_excluded_rows': [],
                 'event_excluded_rows': [], 'boundary_metric_rows': []}
    if branch is not None:
        summary, grid = branch['summary'], branch['grid']
        bins = summary['bins']
        display = [i for i, b in enumerate(bins) if b['start_us'] < hi and b['end_us'] > lo]
        complete = [i for i in display if bins[i]['start_us'] >= lo and bins[i]['end_us'] <= hi]
        selection.update(display_bins=display, complete_event_bins=complete,
                         boundary_bins=[i for i in display if i not in complete])
        baseline = next((r for r in summary['baseline_audit'] if r['event'] == label), None)
        selection['baseline'] = baseline
        starts, ends = grid.columns['window_start_us'], grid.columns['window_end_us']
        event_mask = complete_mask(starts, ends, lo, hi)
        selection['event_candidate_rows'] = np.flatnonzero(event_mask).tolist()
        selection['event_excluded_rows'] = np.flatnonzero(event_mask & ~branch['valid']).tolist()
        selection['boundary_metric_rows'] = np.flatnonzero((starts < hi) & (ends > lo) & ~event_mask).tolist()
        if baseline is not None:
            mask = complete_mask(starts, ends, baseline['start_us'], baseline['end_us'])
            selection['baseline_candidate_rows'] = np.flatnonzero(mask).tolist()
            selection['baseline_excluded_rows'] = np.flatnonzero(mask & ~branch['valid']).tolist()
            selection['valid_delta_bins'] = [i for i in complete if np.isfinite(summary['heatmap_delta'][:, i]).all()]
    if not target['participates']:
        status = 'not_participating'
    elif not runs:
        status = 'no_raw_samples'
    elif branch is None:
        status = 'no_complete_metric_windows'
    elif not selection['complete_event_bins']:
        status = 'no_complete_event_bins'
    elif not selection['baseline'] or selection['baseline']['heatmap_status'] != 'accepted':
        status = 'missing_baseline'
    elif not selection['valid_delta_bins']:
        status = 'no_accepted_event_bins'
    else:
        status = 'computed'
    selection['status'] = status
    return selection


def zoom_values(branch, selection):
    """Return only complete target deltas, leaving crop-boundary bins missing."""
    values = np.full((4, len(selection['display_bins'])), np.nan)
    if selection['status'] == 'computed':
        for j, i in enumerate(selection['display_bins']):
            if i in selection['valid_delta_bins']:
                values[:, j] = branch['summary']['heatmap_delta'][:, i]
    return values
