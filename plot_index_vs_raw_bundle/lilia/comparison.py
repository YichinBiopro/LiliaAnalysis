"""APP/NUC segmented preprocessing, elapsed-clock pairing, and spectra."""
from __future__ import annotations

import numpy as np
from scipy import signal

from lilia.neural import build_inference_timeline
from lilia.signal import resample_polyphase
from lilia.qeeg import compute_qeeg_indices
from lilia.windowing import continuous_slices


def prepare_recording(time_us, raw, model):
    """Preserve the CLI's filter -> per-segment MAD repair -> resample -> OLA.

    MAD repair is intentionally absent from the existing denoised-MI pipeline.
    A corrupt retained segment fails this source; short segments are audited.
    """
    import data_analysis as da
    t, raw = np.asarray(time_us), np.asarray(raw, dtype=float)
    if raw.shape != (len(t), da.N_CH):
        raise ValueError('APP/NUC inference requires exactly four selected raw channels')
    timeline = build_inference_timeline(t, da.FS, da.DOWNSAMPLED_FS, da.MODEL_WINDOW, da.MODEL_WINDOW//2)
    if not len(timeline.time_us):
        raise ValueError('No continuous segment contains a complete model window after resampling')
    filtered, cleaned = np.full_like(raw, np.nan), np.full_like(raw, np.nan)
    down_parts, predictions, repair = [], [], []
    for row in timeline.segments:
        if row['status'] != 'retained':
            continue
        sl = slice(row['raw_start_idx'], row['raw_end_idx'])
        try:
            filtered[sl] = da.apply_filters(raw[sl], fs=da.FS)
            clean, counts, masks = da.remove_artifacts(filtered[sl], fs=da.FS, return_masks=True)
            cleaned[sl] = clean
            down = resample_polyphase(clean, da.FS, da.DOWNSAMPLED_FS)
            pred = da.run_model(model, down, window=timeline.model_window, hop=timeline.model_hop)
        except ValueError as exc:
            raise ValueError(f'Source segment {row["segment_id"]} failed: {exc}') from exc
        if pred.shape != (row['resampled_samples'], da.N_CH_OUT) or not np.isfinite(pred).all():
            raise ValueError('Model output does not match inference timeline')
        repair.append({'segment_id': row['segment_id'], 'counts': counts,
                       'raw_ranges_by_channel': [mask_ranges(m, sl.start) for m in masks.T]})
        down_parts.append(down)
        predictions.append(pred)
    output = np.concatenate(predictions)
    try:
        grid = timeline.grid(5.)
    except ValueError as exc:
        if 'No complete analysis window' not in str(exc):
            raise
        grid = None
    metrics = {}
    if grid is not None:
        for ch in range(da.N_CH_OUT):
            rows = [compute_qeeg_indices(output[start:start+grid.win, ch].astype(float), fs=timeline.fs_out)
                    for start in grid.starts]
            for key in rows[0]:
                metrics[f'{key}_ch{ch+1}'] = np.array([r[key] for r in rows])
    return {'timeline': timeline, 'filtered_raw': filtered, 'cleaned_raw': cleaned,
            'input': np.concatenate(down_parts), 'output': output, 'repair': repair,
            'qeeg_grid': grid, 'qeeg': metrics}


def mask_ranges(mask, offset=0):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return [[int(a)+offset, int(b)+offset] for a, b in zip(np.flatnonzero(edges==1), np.flatnonzero(edges==-1))]


def pair_by_elapsed(timeline_a, timeline_b, lag=0):
    """Pair nearest elapsed timestamps within half an output period, one-to-one."""
    if timeline_a.fs_out != timeline_b.fs_out:
        raise ValueError('APP/NUC output sample rates differ')
    if not isinstance(lag, (int, np.integer)):
        raise ValueError('APP/NUC lag must be an integer number of samples')
    if not len(timeline_a.time_us) or not len(timeline_b.time_us):
        return tuple(np.empty(0, dtype=np.int64) for _ in range(3))
    ta = timeline_a.time_us - timeline_a.source_epoch_us
    tb = timeline_b.time_us - timeline_b.source_epoch_us + int(round(lag*1e6/timeline_a.fs_out))
    right = np.clip(np.searchsorted(tb, ta), 0, len(tb)-1)
    left = np.maximum(right-1, 0)
    best = np.where(np.abs(tb[left]-ta) <= np.abs(tb[right]-ta), left, right)
    keep = np.abs(tb[best]-ta) <= .5e6/timeline_a.fs_out
    ia = np.flatnonzero(keep)
    ib = best[keep]
    # Timestamp jitter must not pair the same NUC sample more than once.
    unique = np.r_[True, np.diff(ib)>0] if len(ib) else np.empty(0,dtype=bool)
    ia, ib = ia[unique], ib[unique]
    if not len(ia):
        return ia, ib, np.empty(0,dtype=np.int64)
    breaks = ((np.diff(ia)!=1) | (np.diff(ib)!=1)
              | (np.diff(timeline_a.segment_ids[ia])!=0) | (np.diff(timeline_b.segment_ids[ib])!=0))
    return ia, ib, np.r_[0, np.cumsum(breaks)].astype(np.int64)


def align_recordings(a, b):
    """Estimate one signal lag from the longest shared continuous elapsed run.

    This is waveform alignment, not absolute acquisition-time synchronization.
    Original APP/NUC source and output indexes remain available after pairing.
    """
    import data_analysis as da
    ia, ib, groups = pair_by_elapsed(a['timeline'], b['timeline'])
    runs = [np.flatnonzero(groups==g) for g in np.unique(groups)]
    if not runs or max(map(len,runs)) < da.MODEL_WINDOW:
        raise ValueError('No shared continuous elapsed interval long enough to estimate lag')
    reference = max(runs, key=len)
    lag = da.estimate_lag(a['input'][ia[reference]], b['input'][ib[reference]], fs=a['timeline'].fs_out)
    audit = {'policy': 'recording_elapsed_nearest_sample_then_single_signal_lag',
        'lag_samples': lag, 'lag_us': int(round(lag*1e6/a['timeline'].fs_out)),
        'reference_app_output_range': [int(ia[reference[0]]), int(ia[reference[-1]])+1],
        'reference_nuc_output_range': [int(ib[reference[0]]), int(ib[reference[-1]])+1],
        'reference_samples': len(reference), 'reference_channel': 1}
    ia, ib, groups = pair_by_elapsed(a['timeline'], b['timeline'], lag)
    if not len(ia):
        raise ValueError('Estimated lag leaves no paired elapsed samples')
    audit.update(paired_samples=len(ia), paired_runs=int(groups[-1])+1,
                 unmatched_app_samples=len(a['input'])-len(ia), unmatched_nuc_samples=len(b['input'])-len(ib))
    return ia, ib, groups, audit


def segmented_psd(values, fs, segments):
    """Average linear segment Welch spectra, weighted by Welch subwindow count.

    nfft is four seconds. Short segments use shorter windows/overlap and zero
    padding to that frequency grid; none of the Welch windows cross a gap.
    """
    nfft = int(round(fs*4))
    total, weight, audit = None, 0, []
    for sl in segments:
        part = np.asarray(values[sl])
        if len(part)<8 or not np.isfinite(part).all():
            continue
        win = min(len(part), nfft)
        overlap = min(int(round(fs*2)), win//2)
        freqs, power = signal.welch(part, fs=fs, nperseg=win, noverlap=overlap, nfft=nfft, window='hann')
        count = 1+(len(part)-win)//(win-overlap)
        total = power*count if total is None else total+power*count
        weight += count
        audit.append({'start_idx':sl.start,'end_idx':sl.stop,'nperseg':win,'noverlap':overlap,'welch_windows':count})
    if total is None:
        raise ValueError('No finite segment long enough for PSD')
    return freqs, 10*np.log10(total/weight+1e-12), audit


def stft_parts(time_s, values, fs, segment_ids=None):
    """Independent STFT per source/pair run with centers on the supplied clock."""
    import data_analysis as da
    time_s = np.asarray(time_s)
    parts = []
    slices = continuous_slices(time_s*1e6, fs, segment_ids=segment_ids)
    for sl in slices:
        if not np.isfinite(values[sl]).all() or sl.stop-sl.start<8:
            continue
        f, seconds, db = da.compute_stft(values[sl], fs=fs)
        pos = seconds*fs
        rel = time_s[sl]-time_s[sl.start]
        mapped = time_s[sl.start]+np.interp(pos,np.arange(len(rel)),rel)
        inside = pos <= len(rel)-1
        if inside.sum()>=2:
            parts.append((f,mapped[inside],db[:,inside]))
    return parts
