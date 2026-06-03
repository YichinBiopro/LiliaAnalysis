"""
plot_raw_eeg.py
===============
For each iBrainCenter merged.csv, randomly pick N_SEGS non-overlapping
30-second segments and plot the raw (no downsampling, no filtering) EEG
waveforms for all 4 channels.

Usage
-----
    python plot_raw_eeg.py [--outdir <dir>] [--seed <int>] [--seg_sec <float>]

Output: one PNG per merged.csv → <outdir>/raw_<group>_<subject>.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from eeg_utils import load_merged_csv

# ── Constants ──────────────────────────────────────────────────────────────────
FS       = 500          # Hz
SEG_SEC  = 30.0
N_SEGS   = 2
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CH_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

SUBJECTS = {
    'iBrainCenter': [
        d for d in sorted(os.listdir(os.path.join(BASE_DIR, 'iBrainCenter')))
        if os.path.isfile(os.path.join(BASE_DIR, 'iBrainCenter', d, 'merged.csv'))
    ]
}


def pick_segments(n_total: int, seg_len: int, n_segs: int, rng: np.random.Generator):
    starts, attempts = [], 0
    max_start = n_total - seg_len
    while len(starts) < n_segs and attempts < 10_000:
        attempts += 1
        s = int(rng.integers(0, max_start + 1))
        if all(abs(s - p) >= seg_len for p in starts):
            starts.append(s)
    return sorted(starts)


def plot_subject(path: str, group: str, subject: str, outdir: str,
                 rng: np.random.Generator, seg_sec: float = SEG_SEC):
    seg_len = int(seg_sec * FS)
    print(f'  [{group}/{subject}] loading…', end=' ', flush=True)
    time_us, data = load_merged_csv(path, downsample=1)   # no downsampling
    n_total, n_ch = data.shape
    duration_min = n_total / FS / 60
    print(f'{n_total} pts ({duration_min:.1f} min), {n_ch} ch')

    starts = pick_segments(n_total, seg_len, N_SEGS, rng)
    if not starts:
        print(f'    WARNING: not enough data for even 1 segment, skipping')
        return

    t_rel = np.arange(seg_len) / FS  # 0 … seg_sec

    fig, axes = plt.subplots(
        n_ch, len(starts),
        figsize=(10 * len(starts), 2.5 * n_ch + 1.2),
        sharex='col', sharey='row',
        squeeze=False,
    )
    fig.suptitle(
        f'{group} — {subject}  |  {len(starts)} random {seg_sec:.0f}s segments  '
        f'(raw, no downsampling @ {FS} Hz)',
        fontsize=13, fontweight='bold',
    )

    for col, start in enumerate(starts):
        seg      = data[start : start + seg_len]   # (seg_len, n_ch)
        t_abs_s  = (time_us[start] - time_us[0]) / 1e6   # seconds from file start
        t_label  = f't={t_abs_s/60:.1f} min'

        for ch in range(n_ch):
            ax = axes[ch][col]
            ax.plot(t_rel, seg[:, ch], color=CH_COLORS[ch % len(CH_COLORS)],
                    lw=0.4, rasterized=True)
            ax.set_ylabel(f'ch{ch+1} (µV)')
            if ch == 0:
                ax.set_title(f'Segment {col+1}  [{t_label}]', fontsize=10)
            if ch == n_ch - 1:
                ax.set_xlabel('Time (s)')
            ax.set_ylim(-100, 100)
            ax.grid(True, lw=0.3, alpha=0.5)

    fig.tight_layout()
    fname = f'raw_{group}_{subject}.png'
    out_path = os.path.join(outdir, fname)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'    → saved {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Plot raw EEG waveforms.')
    parser.add_argument('--outdir',  default=os.path.join(BASE_DIR, 'sample_quality'))
    parser.add_argument('--seed',    type=int, default=42)
    parser.add_argument('--seg_sec', type=float, default=SEG_SEC)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for group, subjects in SUBJECTS.items():
        print(f'\n=== {group} ===')
        for subj in subjects:
            path = os.path.join(BASE_DIR, group, subj, 'merged.csv')
            plot_subject(path, group, subj, args.outdir, rng, seg_sec=args.seg_sec)

    print('\nDone.')


if __name__ == '__main__':
    main()
