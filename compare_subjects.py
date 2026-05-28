"""
compare_subjects.py
===================
Inter-subject comparison of BP delta index and TFLite delta index.

Two groups are analysed independently:
  • iBrainCenter: Ann, Hsin, Hardy, TYY, James  (8 event blocks)
  • YoGa:         James, Jammie, TYY            (no events → whole-session delta)

Outputs (per group, written to <group>/comparison/):
  {group}_bp_delta_comparison.png
  {group}_tflite_delta_comparison.png
  {group}_combined_comparison.png   — BP (solid) vs TFLite (hatched) per subject

Usage
-----
    python compare_subjects.py [--ibrain-outdir DIR] [--yoga-outdir DIR]
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from math import gcd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.signal import resample_poly

from eeg_utils import load_merged_csv, bandpass_filter
from qeeg_indices import compute_qeeg_indices
from plot_event_markers import (
    SUBJECTS, YOGA_SUBJECTS, EVENTS, EVT_COLORS,
    IBRAIN_DIR, YOGA_DIR,
    FS, TFLITE_FS, TFLITE_WIN, TFLITE_MODEL_PATH,
    QEEG_WIN_SEC, QUALITY_WIN_SEC, QUALITY_THRESHOLD,
    us_to_local_dt, hhmm_to_dt,
    compute_quality_windowed, compute_qeeg_windowed,
    apply_tflite_windowed,
)

# ── Constants ──────────────────────────────────────────────────────────────────
INDEX_KEYS  = ['focus', 'flow', 'calm', 'relaxation']
IDX_LABELS  = ['Focus', 'Flow', 'Calm', 'Relaxation']
IDX_COLORS  = {
    'focus':       '#e6194b',
    'flow':        '#3cb44b',
    'calm':        '#4363d8',
    'relaxation':  '#f58231',
}

SUBJ_COLORS_IBRAIN = {
    'Ann':   '#1f77b4',
    'Hsin':  '#ff7f0e',
    'Hardy': '#2ca02c',
    'TYY':   '#d62728',
    'James': '#9467bd',
}

SUBJ_COLORS_YOGA = {
    'James':  '#1f77b4',
    'Jammie': '#ff7f0e',
    'TYY':    '#2ca02c',
}

# ── Event helpers ──────────────────────────────────────────────────────────────

def _build_participating_events(name: str) -> list[tuple]:
    """Return list of (start_dt, end_dt, label, color) for events *name* joins."""
    events = []
    for idx, (label, start_hhmm, dur_min, participants) in enumerate(EVENTS):
        if participants is None or name in participants:
            start_dt = hhmm_to_dt(start_hhmm)
            end_dt   = start_dt + datetime.timedelta(minutes=dur_min)
            color    = EVT_COLORS[idx % len(EVT_COLORS)]
            events.append((start_dt, end_dt, label, color))
    return events


def _all_event_labels() -> list[str]:
    return [label for label, *_ in EVENTS]


# ── Per-subject delta computation ──────────────────────────────────────────────

def _compute_block_deltas(
    qeeg_dt: list,
    qeeg_scores: dict,
    qual_mask: np.ndarray,
    participating_events: list,
) -> dict[str, dict[str, tuple[float, float]]]:
    """
    For each participating event block compute Δ vs baseline.

    Returns { event_label -> { index_key -> (mean_delta, std_delta) } }
    """
    if not participating_events or not qeeg_dt:
        return {}

    t_arr     = np.array(qeeg_dt)
    good_qual = ~qual_mask

    first_evt_dt  = min(e[0] for e in participating_events)
    baseline_mask = (t_arr < first_evt_dt) & good_qual

    result = {}
    for (start_dt, end_dt, label, _color) in participating_events:
        block_mask = (t_arr >= start_dt) & (t_arr < end_dt) & good_qual
        per_index  = {}
        for k in INDEX_KEYS:
            scores = qeeg_scores[k]   # shape (n_windows, n_ch)
            if baseline_mask.sum() > 0 and block_mask.sum() > 0:
                bl   = scores[baseline_mask].mean(axis=0)   # (n_ch,)
                blk  = scores[block_mask].mean(axis=0)
                ch_d = blk - bl
                per_index[k] = (float(ch_d.mean()), float(ch_d.std()))
            else:
                per_index[k] = (float('nan'), 0.0)
        result[label] = per_index

    return result


def _compute_session_delta(
    qeeg_dt: list,
    qeeg_scores: dict,
    qual_mask: np.ndarray,
) -> dict[str, tuple[float, float]]:
    """
    Whole-session delta using first 20 % as baseline.
    Returns { index_key -> (mean_delta, std_delta) }
    """
    if not qeeg_dt:
        return {}

    t_arr     = np.array(qeeg_dt)
    n         = len(t_arr)
    good_qual = ~qual_mask

    bl_mask  = np.zeros(n, dtype=bool)
    bl_mask[:max(1, n // 5)] = True
    bl_mask &= good_qual
    sess_mask = ~bl_mask & good_qual

    result = {}
    for k in INDEX_KEYS:
        scores = qeeg_scores[k]
        if bl_mask.sum() > 0 and sess_mask.sum() > 0:
            bl   = scores[bl_mask].mean(axis=0)
            sess = scores[sess_mask].mean(axis=0)
            ch_d = sess - bl
            result[k] = (float(ch_d.mean()), float(ch_d.std()))
        else:
            result[k] = (float('nan'), 0.0)
    return result


def extract_subject_deltas(
    name: str,
    info: dict,
    base_dir: str,
    participating_events: list | None,
) -> tuple[dict, dict]:
    """
    Load and process one subject.

    Returns (bp_result, tfl_result) where each is either:
      • iBrainCenter: { event_label -> { index_key -> (mean, std) } }
      • YoGa:        { 'Session'   -> { index_key -> (mean, std) } }
    None is returned for a signal type if data is unavailable.
    """
    merged = os.path.join(base_dir, info['dir'], 'merged.csv')
    if not os.path.isfile(merged):
        print(f'  [{name}] merged.csv not found — skipping')
        return None, None

    print(f'  [{name}] loading…', end=' ', flush=True)
    time_us, data = load_merged_csv(merged)
    print(f'{len(time_us)} samples ({(time_us[-1]-time_us[0])/1e6:.0f} s)')

    data_filt = bandpass_filter(data)

    # Quality mask
    print(f'  [{name}] quality…', end=' ', flush=True)
    q_dt, q_overall = compute_quality_windowed(time_us, data, win_sec=QUALITY_WIN_SEC)
    q_median  = np.median(q_overall, axis=1)
    low_qual  = q_median < QUALITY_THRESHOLD
    print(f'{len(q_dt)} windows')

    # BP qEEG indices
    print(f'  [{name}] BP qEEG…', end=' ', flush=True)
    qeeg_dt, qeeg_scores = compute_qeeg_windowed(time_us, data_filt)
    n_q   = len(q_dt)
    n_qeeg = len(qeeg_dt)
    qual_mask = low_qual[:n_qeeg] if n_q >= n_qeeg else np.zeros(n_qeeg, dtype=bool)
    print(f'{n_qeeg} windows')

    if participating_events is not None:
        bp_result = _compute_block_deltas(qeeg_dt, qeeg_scores, qual_mask,
                                          participating_events)
    else:
        sess = _compute_session_delta(qeeg_dt, qeeg_scores, qual_mask)
        bp_result = {'Session': sess} if sess else {}

    # TFLite qEEG indices
    tfl_result = None
    if os.path.isfile(TFLITE_MODEL_PATH):
        try:
            print(f'  [{name}] TFLite…', end=' ', flush=True)
            _g        = gcd(int(TFLITE_FS), int(FS))
            _up, _dn  = int(TFLITE_FS) // _g, int(FS) // _g
            data_200  = resample_poly(data_filt, _up, _dn, axis=0).astype(np.float32)
            t_orig    = np.arange(len(data_filt))
            t_new     = np.arange(len(data_200)) * (_dn / _up)
            tfl_time  = np.interp(t_new, t_orig,
                                  time_us[:len(data_filt)]).astype(np.int64)
            tfl_raw   = apply_tflite_windowed(data_200)
            N_tfl     = len(tfl_raw)
            tfl_time  = tfl_time[:N_tfl]
            print(f'{N_tfl} samples')

            print(f'  [{name}] TFLite qEEG…', end=' ', flush=True)
            tfl_dt, tfl_scores = compute_qeeg_windowed(tfl_time, tfl_raw,
                                                       fs=TFLITE_FS)
            n_tfl = len(tfl_dt)
            tfl_mask = low_qual[:n_tfl] if n_q >= n_tfl else np.zeros(n_tfl, dtype=bool)
            print(f'{n_tfl} windows')

            if participating_events is not None:
                tfl_result = _compute_block_deltas(tfl_dt, tfl_scores, tfl_mask,
                                                   participating_events)
            else:
                sess_t = _compute_session_delta(tfl_dt, tfl_scores, tfl_mask)
                tfl_result = {'Session': sess_t} if sess_t else {}

        except Exception as exc:
            print(f'\n  [{name}] TFLite skipped: {exc}')

    return bp_result, tfl_result


# ── Plotting ───────────────────────────────────────────────────────────────────

def _bar_cell(
    ax: plt.Axes,
    subjects: list[str],
    evt_label: str,
    idx_name: str,
    signal_data: dict,        # { subj -> { evt_label -> { idx -> (m, s) } } }
    subject_colors: dict,
    participation: dict | None,   # { subj -> set_of_event_labels } or None
    show_xticklabels: bool = True,
) -> None:
    """Draw one grouped-bar cell (subjects on x-axis, one bar per subject)."""
    vals, errs, colors, alphas = [], [], [], []
    for subj in subjects:
        evt_dict = signal_data.get(subj) or {}
        per_idx  = evt_dict.get(evt_label) or {}
        m, s = per_idx.get(idx_name, (float('nan'), 0.0))
        participated = (participation is None
                        or subj not in participation
                        or evt_label in participation[subj])
        vals.append(m)
        errs.append(s)
        colors.append(subject_colors.get(subj, '#888888'))
        alphas.append(0.85 if participated else 0.20)

    x = np.arange(len(subjects))
    for xi, (v, e, c, a) in enumerate(zip(vals, errs, colors, alphas)):
        if np.isnan(v):
            ax.bar(xi, 0, width=0.6, color=c, alpha=0.12, zorder=2)
        else:
            ax.bar(xi, v, width=0.6, color=c, alpha=a,
                   yerr=e, capsize=3, error_kw={'lw': 0.8}, zorder=3)
    ax.axhline(0, color='k', lw=0.7, zorder=4)
    ax.set_xticks(x)
    if show_xticklabels:
        ax.set_xticklabels(subjects, rotation=35, ha='right', fontsize=7)
    else:
        ax.set_xticklabels([])
    ax.grid(True, alpha=0.2, axis='y')


def plot_single_signal_comparison(
    group_label: str,
    subjects: list[str],
    event_labels: list[str],
    signal_data: dict,
    subject_colors: dict,
    participation: dict | None,
    signal_tag: str,           # 'BP' or 'TFLite'
    ylabel_prefix: str,
    outpath: str,
) -> None:
    """
    4 rows (indices) × N_evt columns.
    Each cell: bars per subject for that (index, event) combination.
    """
    n_idx = len(INDEX_KEYS)
    n_evt = len(event_labels)

    cell_w = max(1.6, len(subjects) * 0.45 + 0.8)
    fig, axes = plt.subplots(
        n_idx, n_evt,
        figsize=(cell_w * n_evt + 1.0, 3.2 * n_idx + 1.2),
        sharey='row',
    )
    if n_idx == 1:
        axes = axes[np.newaxis, :]
    if n_evt == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(
        f'{group_label} — Inter-subject Comparison  [{signal_tag} Δ vs baseline]\n'
        f'Each column = event block  |  Each bar = one subject',
        fontsize=12, fontweight='bold',
    )

    for row_i, (idx_name, idx_label) in enumerate(zip(INDEX_KEYS, IDX_LABELS)):
        for col_j, evt_label in enumerate(event_labels):
            ax = axes[row_i, col_j]
            show_xtick = (row_i == n_idx - 1)
            _bar_cell(ax, subjects, evt_label, idx_name,
                      signal_data, subject_colors, participation,
                      show_xticklabels=show_xtick)
            if col_j == 0:
                ax.set_ylabel(f'{idx_label}\n{ylabel_prefix} Δ', fontsize=8)
            if row_i == 0:
                ax.set_title(evt_label.replace(' ', '\n'), fontsize=8,
                             fontweight='bold')

    # Subject legend
    legend_handles = [
        mpatches.Patch(color=subject_colors.get(s, '#888'), label=s)
        for s in subjects
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=len(subjects),
               fontsize=9, bbox_to_anchor=(0.5, 0.0), framealpha=0.9,
               title='Subjects')

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {outpath}')


def plot_combined_comparison(
    group_label: str,
    subjects: list[str],
    event_labels: list[str],
    bp_data: dict,
    tfl_data: dict,
    subject_colors: dict,
    participation: dict | None,
    outpath: str,
) -> None:
    """
    N_events rows × 4 columns (one per qEEG index).
    Each cell: grouped bars where each subject has a pair of bars:
      solid  = BP delta
      hatched = TFLite delta
    """
    n_evt = len(event_labels)
    n_idx = len(INDEX_KEYS)
    n_subj = len(subjects)

    cell_w = max(2.0, n_subj * 0.7 + 0.8)
    fig, axes = plt.subplots(
        n_evt, n_idx,
        figsize=(cell_w * n_idx + 1.0, 2.8 * n_evt + 1.4),
        sharey='row',
    )
    if n_evt == 1:
        axes = axes[np.newaxis, :]
    if n_idx == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(
        f'{group_label} — BP vs TFLite Inter-subject Comparison\n'
        f'Solid = BP Δ  |  Hatched = TFLite Δ  |  Each colour = one subject',
        fontsize=12, fontweight='bold',
    )

    bar_w = 0.35
    x_off = np.array([-bar_w / 2, bar_w / 2])   # BP left, TFLite right

    for row_i, evt_label in enumerate(event_labels):
        for col_j, (idx_name, idx_label) in enumerate(zip(INDEX_KEYS, IDX_LABELS)):
            ax = axes[row_i, col_j]
            x_pos = np.arange(n_subj) * (2 * bar_w * 1.6)

            for s_i, subj in enumerate(subjects):
                clr = subject_colors.get(subj, '#888888')
                participated = (participation is None
                                or subj not in participation
                                or evt_label in participation[subj])
                alpha = 0.85 if participated else 0.18

                bp_val,  bp_err  = ((bp_data or {}).get(subj) or {}).get(evt_label, {}).get(idx_name, (float('nan'), 0.0))
                tfl_val, tfl_err = ((tfl_data or {}).get(subj) or {}).get(evt_label, {}).get(idx_name, (float('nan'), 0.0))

                if not np.isnan(bp_val):
                    ax.bar(x_pos[s_i] + x_off[0], bp_val, width=bar_w,
                           color=clr, alpha=alpha, yerr=bp_err, capsize=2.5,
                           error_kw={'lw': 0.8}, zorder=3)
                if not np.isnan(tfl_val):
                    ax.bar(x_pos[s_i] + x_off[1], tfl_val, width=bar_w,
                           color=clr, alpha=alpha * 0.6, hatch='//',
                           edgecolor=clr, linewidth=0.5,
                           yerr=tfl_err, capsize=2.5,
                           error_kw={'lw': 0.8}, zorder=3)

            ax.axhline(0, color='k', lw=0.7, zorder=4)
            show_xtick = (row_i == n_evt - 1)
            ax.set_xticks(x_pos)
            if show_xtick:
                ax.set_xticklabels(subjects, rotation=35, ha='right', fontsize=7)
            else:
                ax.set_xticklabels([])
            ax.grid(True, alpha=0.2, axis='y')

            if col_j == 0:
                ax.set_ylabel(
                    evt_label.replace(' ', '\n') + '\nΔ Index', fontsize=7.5)
            if row_i == 0:
                ax.set_title(idx_label, fontsize=9, fontweight='bold',
                             color=IDX_COLORS[idx_name])

    # Legend: subjects + BP/TFLite indicator
    legend_handles = [
        mpatches.Patch(color=subject_colors.get(s, '#888'), label=s)
        for s in subjects
    ]
    legend_handles += [
        mpatches.Patch(facecolor='grey', alpha=0.85, label='BP (solid)'),
        mpatches.Patch(facecolor='grey', alpha=0.5, hatch='//',
                       edgecolor='grey', label='TFLite (hatched)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=len(subjects) + 2, fontsize=8,
               bbox_to_anchor=(0.5, 0.0), framealpha=0.9)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {outpath}')


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Inter-subject comparison of BP / TFLite qEEG delta indices.')
    p.add_argument('--ibrain-outdir',
                   default=os.path.join(IBRAIN_DIR, 'comparison'),
                   metavar='DIR')
    p.add_argument('--yoga-outdir',
                   default=os.path.join(YOGA_DIR, 'comparison'),
                   metavar='DIR')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    for group_label, subjects_registry, base_dir, outdir, use_events in [
        ('iBrainCenter', SUBJECTS,      IBRAIN_DIR, args.ibrain_outdir, True),
        ('YoGa',         YOGA_SUBJECTS, YOGA_DIR,   args.yoga_outdir,   False),
    ]:
        os.makedirs(outdir, exist_ok=True)
        print(f'\n══ {group_label} → {outdir}')

        # Build participation map and event label list
        if use_events:
            all_evt_labels = _all_event_labels()
            participation  = {
                name: {lbl for lbl, _start, _dur, parts in EVENTS
                       if parts is None or name in parts}
                for name in subjects_registry
            }
        else:
            all_evt_labels = ['Session']
            participation  = None

        # Process each subject
        bp_all  = {}   # { subj_name -> { evt_label -> { idx -> (m, s) } } }
        tfl_all = {}

        for name, info in subjects_registry.items():
            if use_events:
                part_evts = _build_participating_events(name)
            else:
                part_evts = None

            bp_res, tfl_res = extract_subject_deltas(
                name, info, base_dir, part_evts)

            bp_all[name]  = bp_res  or {}
            tfl_all[name] = tfl_res or {}

        subjects = list(subjects_registry.keys())
        subject_colors = (SUBJ_COLORS_IBRAIN if group_label == 'iBrainCenter'
                          else SUBJ_COLORS_YOGA)

        # Plot 1: BP delta comparison
        plot_single_signal_comparison(
            group_label, subjects, all_evt_labels, bp_all,
            subject_colors, participation,
            signal_tag='BP', ylabel_prefix='BP',
            outpath=os.path.join(outdir, f'{group_label.lower()}_bp_delta_comparison.png'),
        )

        # Plot 2: TFLite delta comparison
        if any(tfl_all[s] for s in subjects):
            plot_single_signal_comparison(
                group_label, subjects, all_evt_labels, tfl_all,
                subject_colors, participation,
                signal_tag='TFLite', ylabel_prefix='TFLite',
                outpath=os.path.join(outdir, f'{group_label.lower()}_tflite_delta_comparison.png'),
            )
        else:
            print(f'  [{group_label}] no TFLite data — skipping tflite comparison plot')

        # Plot 3: Combined BP vs TFLite
        if any(tfl_all[s] for s in subjects):
            plot_combined_comparison(
                group_label, subjects, all_evt_labels,
                bp_all, tfl_all,
                subject_colors, participation,
                outpath=os.path.join(outdir, f'{group_label.lower()}_combined_comparison.png'),
            )

    print('\nDone.')


if __name__ == '__main__':
    main()
