"""
sample_quality_check.py
=======================
Randomly sample 2 non-overlapping 30-second segments from every merged.csv
(iBrainCenter + YoGa), apply a 0.5–45 Hz bandpass filter, compute EEG quality
(flat+spectrum only, 500 Hz), and plot raw EEG + PSD + quality side-by-side.

Usage
-----
    python sample_quality_check.py [--outdir <dir>] [--seed <int>]

Output: one PNG per merged.csv → <outdir>/<group>_<subject>_sample_quality.png
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.signal import welch

from eeg_quality_v2 import (
    get_eeg_quality_index_v2_parametric,
    get_best_eeg_quality_v2_flat_spectrum_only_params,
)
from qeeg_indices import compute_qeeg_indices
from eeg_utils import load_merged_csv, bandpass_filter

# ── Constants ──────────────────────────────────────────────────────────────────
FS            = 500          # Hz
SEG_SEC       = 30.0         # segment length (seconds)
N_SEGS        = 2            # number of random segments per file
SEG_SAMPLES   = int(SEG_SEC * FS)

QUALITY_PARAMS    = get_best_eeg_quality_v2_flat_spectrum_only_params()
QUALITY_THRESHOLD = 0.5

BP_LOW  = 0.5    # Hz — bandpass lower cutoff
BP_HIGH = 45.0   # Hz — bandpass upper cutoff

QEEG_WIN_SEC    = 5.0          # qEEG index window length (seconds)
QEEG_WIN_SAMPLES = int(QEEG_WIN_SEC * FS)   # samples per qEEG window
QEEG_INDICES = ['focus', 'flow', 'calm', 'relaxation']

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CH_COLORS  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# ── CSV helpers ────────────────────────────────────────────────────────────────

def pick_segments(n_total: int, seg_len: int, n_segs: int, rng: np.random.Generator):
    """
    Pick *n_segs* non-overlapping start indices uniformly at random from
    [0, n_total - seg_len].  Returns sorted list of start indices.
    """
    starts = []
    max_start = n_total - seg_len
    attempts  = 0
    while len(starts) < n_segs and attempts < 10_000:
        attempts += 1
        s = int(rng.integers(0, max_start + 1))
        # check non-overlapping with already chosen segments
        if all(abs(s - prev) >= seg_len for prev in starts):
            starts.append(s)
    starts.sort()
    return starts


def compute_qeeg_windowed_seg(seg: np.ndarray, fs: float = FS,
                               win_sec: float = QEEG_WIN_SEC):
    """
    Slide non-overlapping *win_sec* windows over *seg* (N, n_ch) and
    compute qEEG indices per channel.

    Returns
    -------
    t_mid   : (n_wins,) relative midpoint times in seconds
    scores  : dict { index_name -> (n_wins, n_ch) }
    """
    win    = int(win_sec * fs)
    n, n_ch = seg.shape
    t_mid  = []
    accum  = {k: [] for k in QEEG_INDICES}
    for s in range(0, n - win + 1, win):
        t_mid.append((s + win / 2) / fs)
        row = {k: [] for k in QEEG_INDICES}
        for ch_i in range(n_ch):
            res = compute_qeeg_indices(seg[s : s + win, ch_i].astype(np.float64), fs=fs)
            for k in QEEG_INDICES:
                row[k].append(res[k])
        for k in QEEG_INDICES:
            accum[k].append(row[k])
    return np.array(t_mid), {k: np.array(accum[k]) for k in QEEG_INDICES}


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_segments(path: str, group: str, subject: str, outdir: str,
                  rng: np.random.Generator):
    print(f'  [{group}/{subject}] loading…', end=' ', flush=True)
    time_us, data = load_merged_csv(path)
    n_total, n_ch = data.shape
    print(f'{n_total} pts, {n_ch} ch')

    starts = pick_segments(n_total, SEG_SAMPLES, N_SEGS, rng)
    if len(starts) < N_SEGS:
        print(f'    WARNING: only {len(starts)} segment(s) found, skipping')
        return

    # ── Figure layout ─────────────────────────────────────────────────────────
    # Rows per segment column:
    #   n_ch  raw EEG traces
    #   1     quality bar chart (raw vs filtered)
    #   n_ch  PSD panels
    #   4     qEEG index bar charts (Focus / Flow / Calm / Relaxation)
    n_qeeg = len(QEEG_INDICES)
    n_rows = n_ch + 1 + n_ch + n_qeeg
    height_ratios = [2.5] * n_ch + [1.8] + [1.8] * n_ch + [1.5] * n_qeeg
    fig = plt.figure(figsize=(10 * N_SEGS, sum(hr * 0.82 for hr in height_ratios) + 1.5))
    fig.suptitle(
        f'{group} — {subject}  |  {N_SEGS} random {SEG_SEC:.0f}s segments  '
        f'[BP {BP_LOW}–{BP_HIGH} Hz]  (quality: flat+spectrum only @ {FS}Hz)\n'
        f'qEEG indices: raw (dashed) vs BP-filtered (solid) — {QEEG_WIN_SEC:.0f}s non-overlapping windows',
        fontsize=13, fontweight='bold',
    )

    outer = gridspec.GridSpec(1, N_SEGS, figure=fig, hspace=0.05, wspace=0.18)

    for col, start in enumerate(starts):
        seg_raw  = data[start : start + SEG_SAMPLES]           # (SEG_SAMPLES, n_ch)
        seg_t    = time_us[start : start + SEG_SAMPLES]
        t_s      = (seg_t - seg_t[0]) / 1e6                   # relative seconds

        # Bandpass filter (0.5–45 Hz) — used for quality & PSD
        seg_data = bandpass_filter(seg_raw)

        # qEEG indices — 5s non-overlapping windows, raw and filtered
        qeeg_t,    qeeg_raw_w  = compute_qeeg_windowed_seg(seg_raw)
        _,         qeeg_filt_w = compute_qeeg_windowed_seg(seg_data)

        # Quality on RAW signal
        result_raw = get_eeg_quality_index_v2_parametric(
            seg_raw.T.astype(np.float64),
            fs=FS,
            params=QUALITY_PARAMS,
        )
        overall_raw = result_raw["overall"]       # (n_ch,)

        # Quality on FILTERED signal
        result = get_eeg_quality_index_v2_parametric(
            seg_data.T.astype(np.float64),
            fs=FS,
            params=QUALITY_PARAMS,
        )
        overall = result["overall"]       # (n_ch,)

        # PSD via Welch — raw and filtered
        psd_freqs, psd_powers_raw, psd_powers_filt = [], [], []
        for ch_i in range(n_ch):
            f, pxx_raw  = welch(seg_raw[:, ch_i].astype(np.float64),
                                fs=FS, nperseg=FS * 4, noverlap=FS * 2)
            _, pxx_filt = welch(seg_data[:, ch_i].astype(np.float64),
                                fs=FS, nperseg=FS * 4, noverlap=FS * 2)
            psd_freqs.append(f)
            psd_powers_raw.append(pxx_raw)
            psd_powers_filt.append(pxx_filt)

        inner = gridspec.GridSpecFromSubplotSpec(
            n_rows, 1, subplot_spec=outer[col],
            hspace=0.30,
            height_ratios=height_ratios,
        )

        # ── EEG raw + filtered panels ────────────────────────────────────────
        axes_eeg = []
        for ch_i in range(n_ch):
            ax = fig.add_subplot(inner[ch_i],
                                 sharex=axes_eeg[0] if axes_eeg else None)
            ax.plot(t_s, seg_raw[:, ch_i],
                    color='#aaaaaa', lw=0.4, alpha=0.5, label='raw')
            ax.plot(t_s, seg_data[:, ch_i],
                    color=CH_COLORS[ch_i % len(CH_COLORS)], lw=0.6,
                    label=f'BP {BP_LOW}–{BP_HIGH}Hz')
            ax.set_ylabel(f'ch{ch_i+1}\n(µV)', fontsize=8)
            ax.grid(True, alpha=0.2)
            if ch_i == 0:
                abs_sec = int((seg_t[0] - time_us[0]) / 1e6)
                h, rem  = divmod(abs_sec, 3600)
                m, s_   = divmod(rem, 60)
                ax.set_title(
                    f'Segment {col+1}  |  start +{h:02d}:{m:02d}:{s_:02d} from recording',
                    fontsize=9,
                )
                ax.legend(loc='upper right', fontsize=6, framealpha=0.7)
            if ch_i < n_ch - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
            axes_eeg.append(ax)
        axes_eeg[-1].set_xlabel('Time within segment (s)', fontsize=8)

        # ── Quality comparison bar chart (raw vs filtered) ──────────────────
        ax_q = fig.add_subplot(inner[n_ch])   # independent x-axis
        bar_x     = np.arange(n_ch)
        bar_w     = 0.35
        bars_raw  = ax_q.bar(bar_x - bar_w/2, overall_raw, width=bar_w,
                             color=[CH_COLORS[i % len(CH_COLORS)] for i in range(n_ch)],
                             alpha=0.40, hatch='//', label='raw', zorder=3)
        bars_filt = ax_q.bar(bar_x + bar_w/2, overall, width=bar_w,
                             color=[CH_COLORS[i % len(CH_COLORS)] for i in range(n_ch)],
                             alpha=0.85, label=f'BP {BP_LOW}–{BP_HIGH}Hz', zorder=3)
        ax_q.axhline(QUALITY_THRESHOLD, color='red', lw=1.2, ls='--',
                     label=f'threshold {QUALITY_THRESHOLD:.2f}')
        for i in range(n_ch):
            ax_q.text(i - bar_w/2, overall_raw[i] + 0.015, f'{overall_raw[i]:.2f}',
                      ha='center', va='bottom', fontsize=6.5, color='#555555')
            ax_q.text(i + bar_w/2, overall[i] + 0.015, f'{overall[i]:.2f}',
                      ha='center', va='bottom', fontsize=6.5, fontweight='bold')
            # delta annotation
            delta = overall[i] - overall_raw[i]
            sign  = '+' if delta >= 0 else ''
            ax_q.text(i, max(overall_raw[i], overall[i]) + 0.065,
                      f'Δ{sign}{delta:.2f}',
                      ha='center', va='bottom', fontsize=6.5,
                      color='green' if delta >= 0 else 'red')
        ax_q.set_ylim(0, 1.22)
        ax_q.set_xticks(bar_x)
        ax_q.set_xticklabels([f'ch{i+1}' for i in range(n_ch)], fontsize=8)
        ax_q.set_ylabel('Overall\nQuality', fontsize=8)
        ax_q.legend(loc='upper right', fontsize=7)
        ax_q.grid(True, alpha=0.2, axis='y')
        ax_q.set_xlim(-0.5, n_ch - 0.5)

        # ── PSD panels (one per channel, shared x and y axes) ────────────────
        BANDS = [('δ', 0.5, 4,  '#a8d8ea'),
                 ('θ', 4,   8,  '#a8e6cf'),
                 ('α', 8,   13, '#ffd3b6'),
                 ('β', 13,  30, '#ffaaa5'),
                 ('γ', 30,  50, '#d4a5ff')]
        axes_psd = []
        for ch_i in range(n_ch):
            row   = n_ch + 1 + ch_i
            share = axes_psd[0] if axes_psd else None
            ax_p  = fig.add_subplot(inner[row], sharex=share, sharey=share)

            f         = psd_freqs[ch_i]
            mask      = f <= 50
            pxx_raw   = psd_powers_raw[ch_i]
            pxx_filt  = psd_powers_filt[ch_i]

            ax_p.semilogy(f[mask], pxx_raw[mask],
                          color='#aaaaaa', lw=0.9, alpha=0.7, ls='--', label='raw')
            ax_p.semilogy(f[mask], pxx_filt[mask],
                          color=CH_COLORS[ch_i % len(CH_COLORS)], lw=1.2,
                          label=f'BP {BP_LOW}–{BP_HIGH}Hz')

            for bname, blo, bhi, bcol in BANDS:
                ax_p.axvspan(blo, bhi, alpha=0.12, color=bcol)

            ax_p.set_ylabel(f'ch{ch_i+1}\n(µV²/Hz)', fontsize=7)
            ax_p.grid(True, alpha=0.2, which='both')
            if ch_i < n_ch - 1:
                plt.setp(ax_p.get_xticklabels(), visible=False)
            else:
                ax_p.set_xlabel('Frequency (Hz)', fontsize=8)
            if ch_i == 0:
                ax_p.set_title('PSD (Welch, 4 s window)', fontsize=8, pad=3)
                ax_p.legend(loc='upper right', fontsize=6, framealpha=0.8)
            axes_psd.append(ax_p)

        # annotate band labels on last PSD panel after shared ylim is settled
        ylim = axes_psd[-1].get_ylim()
        label_y = 10 ** (np.log10(ylim[0]) + 0.85 * (np.log10(ylim[1]) - np.log10(ylim[0])))
        for ax_p in axes_psd:
            for bname, blo, bhi, _ in BANDS:
                ax_p.text((blo + bhi) / 2, label_y, bname,
                          ha='center', fontsize=6, alpha=0.65, clip_on=True)

        # ── qEEG index line plots (5s windows, raw dashed vs filtered solid) ──
        for idx_i, idx_name in enumerate(QEEG_INDICES):
            row   = n_ch + 1 + n_ch + idx_i
            ax_qi = fig.add_subplot(inner[row])
            for ch_i in range(n_ch):
                c = CH_COLORS[ch_i % len(CH_COLORS)]
                ax_qi.plot(qeeg_t, qeeg_raw_w[idx_name][:, ch_i],
                           color=c, lw=0.9, ls='--', alpha=0.55,
                           label=f'ch{ch_i+1} raw' if idx_i == 0 else None)
                ax_qi.plot(qeeg_t, qeeg_filt_w[idx_name][:, ch_i],
                           color=c, lw=1.4, alpha=0.85,
                           label=f'ch{ch_i+1} filt' if idx_i == 0 else None)
            ax_qi.axhline(0, color='k', lw=0.5, ls=':')
            ax_qi.set_ylim(-1.15, 1.15)
            ax_qi.set_xlim(0, SEG_SEC)
            ax_qi.set_ylabel(f'{idx_name.capitalize()}\nIndex', fontsize=7)
            ax_qi.grid(True, alpha=0.2)
            if idx_i < n_qeeg - 1:
                plt.setp(ax_qi.get_xticklabels(), visible=False)
            else:
                ax_qi.set_xlabel('Time within segment (s)', fontsize=8)
            if idx_i == 0:
                ax_qi.legend(loc='upper right', fontsize=6, framealpha=0.7,
                             ncol=n_ch, title='── filt  -- raw', title_fontsize=5)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    fname = f'{group}_{subject}_sample_quality.png'
    outpath = os.path.join(outdir, fname)
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'       → {outpath}')


# ── Discovery ──────────────────────────────────────────────────────────────────

def discover_merged_csvs():
    """Yield (group, subject, path) for every merged.csv found."""
    for group_dir in ['iBrainCenter', 'YoGa']:
        root = os.path.join(BASE_DIR, group_dir)
        if not os.path.isdir(root):
            continue
        for subj_dir in sorted(os.listdir(root)):
            csv_path = os.path.join(root, subj_dir, 'merged.csv')
            if os.path.isfile(csv_path):
                yield group_dir, subj_dir, csv_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Random-sample quality check on all merged.csv files.')
    p.add_argument('--outdir', default=os.path.join(BASE_DIR, 'sample_quality'),
                   metavar='DIR', help='Output directory for PNGs.')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for reproducibility (default 42).')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f'Sampling {N_SEGS}×{SEG_SEC:.0f}s segments from all merged.csv '
          f'(seed={args.seed}) → {args.outdir}\n')

    for group, subject, path in discover_merged_csvs():
        plot_segments(path, group, subject, args.outdir, rng)

    print('\nDone.')


if __name__ == '__main__':
    main()
