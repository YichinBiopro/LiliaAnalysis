"""Subject deltas on source-local BP/TFLite grids with explicit selections."""
from __future__ import annotations

import numpy as np

from lilia.event_qeeg import INDEX_KEYS, complete_mask


def summarize_comparison(branch, events=None):
    """Preserve channel mean/SD and the two distinct historical baselines.

    Event mode uses complete windows before the first participating event.
    Session mode uses the first max(1, N//5) candidate metric rows, then the
    remaining rows. This is a window-count baseline, not 20% of elapsed time.
    Quality is applied after selecting either baseline; no fallback is allowed.
    """
    grid, scores = branch['grid'], branch['scores']
    n = len(grid.starts)
    valid = np.asarray(branch['valid'], dtype=bool).copy()
    if valid.shape != (n,):
        raise ValueError('Comparison validity does not match metric grid')
    for k in INDEX_KEYS:
        values = np.asarray(scores[k])
        if values.ndim != 2 or len(values)!=n or values.shape[1]<1:
            raise ValueError('Comparison scores do not match metric grid')
        valid &= np.isfinite(values).all(axis=1)
    starts, ends = grid.columns['window_start_us'], grid.columns['window_end_us']
    if events is None:
        cut = max(1, n//5)
        baseline = np.arange(n)<cut
        specs = [('Session', True, ~baseline, None, None)]
        policy = 'first_max_1_floor_N_over_5_candidate_rows'
    else:
        labels = [e['label'] for e in events]
        if len(labels)!=len(set(labels)):
            raise ValueError('Comparison event labels must be unique')
        for e in events:
            if e['start_us']>=e['end_us']:
                raise ValueError('Comparison events need positive duration')
        first = min((e['start_us'] for e in events if e['participates']), default=None)
        baseline = complete_mask(starts,ends,hi=first) if first is not None else np.zeros(n,dtype=bool)
        specs = [(e['label'],e['participates'],complete_mask(starts,ends,e['start_us'],e['end_us']),
                  e['start_us'],e['end_us']) for e in events]
        policy = 'complete_windows_before_first_participating_event'
    summaries = []
    for label, participates, target, lo, hi in specs:
        bm = baseline if participates else np.zeros(n,dtype=bool)
        tm = target if participates else np.zeros(n,dtype=bool)
        accepted_b, accepted_t = bm & valid, tm & valid
        if not participates:
            status = 'not_participating'
        elif not bm.any():
            status = 'no_baseline_windows'
        elif not accepted_b.any():
            status = 'no_accepted_baseline'
        elif not tm.any():
            status = 'no_comparison_windows'
        elif not accepted_t.any():
            status = 'no_accepted_comparison'
        else:
            status = 'computed'
        metrics = {}
        for k in INDEX_KEYS:
            if status == 'computed':
                b = np.mean(scores[k][accepted_b],axis=0)
                target_mean = np.mean(scores[k][accepted_t],axis=0)
                delta = target_mean-b
                metrics[k] = {'baseline_channel_mean':b.tolist(),'comparison_channel_mean':target_mean.tolist(),
                              'channel_delta':delta.tolist(),'mean':float(delta.mean()),'channel_sd':float(delta.std())}
            else:
                metrics[k] = {'baseline_channel_mean':None,'comparison_channel_mean':None,
                              'channel_delta':None,'mean':None,'channel_sd':None}
        summaries.append({'label':label,'participates':bool(participates),'status':status,
            'start_us':lo,'end_us':hi,'baseline_policy':policy,
            'baseline_candidate_rows':np.flatnonzero(bm).tolist(),
            'comparison_candidate_rows':np.flatnonzero(tm).tolist(),
            'baseline_accepted_rows':np.flatnonzero(accepted_b).tolist(),
            'comparison_accepted_rows':np.flatnonzero(accepted_t).tolist(),
            'baseline_excluded_rows':np.flatnonzero(bm & ~valid).tolist(),
            'comparison_excluded_rows':np.flatnonzero(tm & ~valid).tolist(),
            'metrics':metrics})
    return summaries


def legacy_results(summary):
    """Keep the historical dict-of-tuples interface; missing SD stays missing."""
    return {r['label']:{k:(m['mean'],m['channel_sd']) if r['status']=='computed' else (np.nan,np.nan)
                       for k,m in r['metrics'].items()} for r in summary}
