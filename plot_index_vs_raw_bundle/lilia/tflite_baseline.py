"""Select complete, already inferred model windows for baseline references."""
from __future__ import annotations

import math
import numpy as np
from lilia.qeeg import compute_qeeg_indices
from lilia.quality_audit import capture_diagnostics, pending_diagnostics, json_value

KEYS = ('relaxation', 'calm', 'flow', 'focus')


def score_baseline_windows(time_us, filtered, raw, timeline, *, scorer, quality_params,
                           threshold=.5, epoch_sec=1., rail=2048., max_saturation=.02):
    """A model window is eligible only if all its raw sub-epochs pass screening."""
    t = np.asarray(time_us)
    if len(t) != len(filtered) or np.shape(raw) != np.shape(filtered):
        raise ValueError('Baseline raw, filtered and timestamp shapes differ')
    unit_sec = timeline.model_window / timeline.fs_out
    if not np.isfinite(epoch_sec) or epoch_sec <= 0 or not np.isfinite(threshold):
        raise ValueError('Baseline epoch duration and quality threshold must be finite')
    n_sub = int(round(unit_sec / epoch_sec))
    if n_sub < 1 or not np.isclose(n_sub * epoch_sec, unit_sec):
        raise ValueError('Quality epoch_sec must divide a complete model window')
    rows = []
    for window in timeline.model_windows:
        row = dict(window, eligible=True, reason='', quality_min=None, saturated_subepochs=0,
                   quality_subepochs=[], n_subepochs=n_sub)
        scores = []
        for i in range(n_sub):
            lo = window['window_start_us'] + int(round(i * epoch_sec * 1e6))
            hi = min(window['window_end_us'], lo + int(round(epoch_sec * 1e6)))
            a, b = int(np.searchsorted(t, lo)), int(np.searchsorted(t, hi))
            request = dict(fs=timeline.fs_in, params=quality_params, n_channels=filtered.shape[1],
                           n_samples=b-a, stage='filtered')
            audit = dict(subepoch_index=i, raw_start_idx=a, raw_end_idx=b,
                         window_start_us=lo, window_end_us=hi, quality_overall=None,
                         quality_median=None, saturation_fraction=None, screening_status='accepted')
            row['quality_subepochs'].append(audit)
            if b - a < 8:
                row.update(eligible=False, reason='insufficient_raw_subepoch')
                audit.update(screening_status=row['reason'], quality_diagnostics=pending_diagnostics(
                    **request, reasons=[row['reason']]))
                break
            if not np.isfinite(filtered[a:b]).all() or not np.isfinite(raw[a:b]).all():
                row.update(eligible=False, reason='nonfinite_signal')
                audit.update(screening_status=row['reason'], quality_diagnostics=pending_diagnostics(
                    **request, reasons=[row['reason']]))
                break
            saturation = float(np.mean(np.any(np.abs(raw[a:b]) >= rail - 1, axis=1)))
            audit['saturation_fraction'] = saturation
            if saturation > max_saturation:
                row.update(eligible=False, reason='raw_saturation', saturated_subepochs=1)
                audit.update(screening_status=row['reason'], quality_diagnostics=pending_diagnostics(
                    **request, reasons=[row['reason']]))
                break
            result = scorer(filtered[a:b].T.astype(np.float64), fs=timeline.fs_in, params=quality_params)
            quality = float(np.median(result['overall']))
            audit.update(quality_overall=json_value(result['overall']), quality_median=json_value(quality),
                         quality_diagnostics=capture_diagnostics(result, **request))
            if not np.isfinite(quality) or quality < threshold:
                row.update(eligible=False, reason='nonfinite_quality' if not np.isfinite(quality) else 'low_quality')
                audit['screening_status'] = row['reason']
                break
            scores.append(quality)
        if scores:
            row['quality_min'] = min(scores)
        rows.append(row)
    return rows


def select_tflite_baseline(output, timeline, catalog, lo_us, hi_us, *, required_sec=10., seed=42, label='session'):
    """Sample complete model outputs, computing each block's qEEG separately.

    The reference is the arithmetic mean of per-model-window indices. Neither
    inference, resampling nor Welch sees concatenated, nonadjacent epochs.
    """
    output = np.asarray(output)
    if output.shape != (len(timeline.time_us), 2):
        raise ValueError('Baseline model output does not match its timeline')
    if not np.isfinite(required_sec) or required_sec <= 0 or hi_us <= lo_us:
        raise ValueError('Baseline duration and physical search interval must be positive')
    duration = timeline.model_window / timeline.fs_out
    need = int(math.ceil(required_sec / duration))
    candidates = [r for r in catalog if r['window_start_us'] >= lo_us and r['window_end_us'] <= hi_us]
    eligible = [r for r in candidates if r['eligible'] and
                np.isfinite(output[r['output_start_idx']:r['output_end_idx']]).all()]
    if len(eligible) < need:
        raise ValueError(f'Insufficient complete baseline model windows for {label}: '
                         f'{len(eligible)}/{len(candidates)} eligible, need {need} ({required_sec:g}s)')
    chosen = np.sort(np.random.default_rng(seed).choice(len(eligible), size=need, replace=False))
    selected = [eligible[int(i)] for i in chosen]
    values = {k: [] for k in KEYS}
    for row in selected:
        block = output[row['output_start_idx']:row['output_end_idx']]
        scores = [compute_qeeg_indices(block[:, ch].astype(float), fs=timeline.fs_out) for ch in range(2)]
        for key in KEYS:
            values[key].append([s[key] for s in scores])
    reference = {k: np.mean(v, axis=0) for k, v in values.items()}
    meta = {'label': label, 'search_start_us': int(lo_us), 'search_end_us': int(hi_us),
            'n_candidates': len(candidates), 'n_eligible': len(eligible), 'n_selected': need,
            'required_sec': required_sec, 'selected_nominal_sec': need * duration,
            'model_window_sec': duration, 'random_seed': seed,
            'reference_method': 'mean of per-complete-model-window qEEG indices',
            'selected': selected, 'reference': {k: v.tolist() for k, v in reference.items()}}
    return reference, meta
