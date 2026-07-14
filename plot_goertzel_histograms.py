#!/usr/bin/env python3
"""Plot quality-filtered Goertzel distributions as histograms.

For each subject/channel (and overall aggregates), this script:
- keeps windows with quality > threshold
- plots histogram of Goertzel power in dB
- marks mean and mean±1 std with vertical lines and corresponding points
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _iter_subject_dirs(root: str):
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, 'merged.csv')):
            yield name


def _build_csv_path(root: str, subject_dir: str, stem: str, ch: int, target_freq: float) -> str:
    hz_txt = f"{target_freq:g}Hz"
    return os.path.join(root, subject_dir, f"{stem}_ch{ch}_{hz_txt}.csv")


def _hist_y_at_x(edges: np.ndarray, counts: np.ndarray, x: float) -> float:
    if counts.size == 0:
        return 0.0
    idx = np.searchsorted(edges, x, side='right') - 1
    idx = int(np.clip(idx, 0, counts.size - 1))
    return float(counts[idx])


def _plot_hist(values_db: np.ndarray, title: str, out_png: str, bins: int) -> None:
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    if values_db.size == 0:
        ax.text(0.5, 0.5, 'No samples above quality threshold', ha='center', va='center')
        ax.set_title(title)
        ax.set_xlabel('Goertzel power (dB)')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
        return

    counts, edges, _ = ax.hist(values_db, bins=bins, color='#7aa6c2', alpha=0.8,
                               edgecolor='white', linewidth=0.5)

    mu = float(np.mean(values_db))
    sigma = float(np.std(values_db, ddof=0))
    x_left = mu - sigma
    x_right = mu + sigma

    y_mu = _hist_y_at_x(edges, counts, mu)
    y_left = _hist_y_at_x(edges, counts, x_left)
    y_right = _hist_y_at_x(edges, counts, x_right)

    ax.axvline(mu, color='#d62728', lw=2.0, ls='-', label=f'mean={mu:.2f} dB')
    ax.axvline(x_left, color='#1f77b4', lw=1.6, ls='--', label=f'mean-1σ={x_left:.2f} dB')
    ax.axvline(x_right, color='#1f77b4', lw=1.6, ls='--', label=f'mean+1σ={x_right:.2f} dB')

    ax.scatter([mu], [y_mu], color='#d62728', s=45, zorder=5)
    ax.scatter([x_left, x_right], [y_left, y_right], color='#1f77b4', s=42, zorder=5)

    ax.set_title(title)
    ax.set_xlabel('Goertzel power (dB)')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right', fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Plot histogram (with ±1 std markers) of quality-filtered Goertzel dB distributions.'
    )
    p.add_argument('--root', default='iBrainCenter', help='Root folder with subject subfolders.')
    p.add_argument('--channels', type=int, nargs='+', default=[1, 2],
                   help='Channels to plot (default: 1 2).')
    p.add_argument('--target-freq', type=float, default=60.0,
                   help='Goertzel target frequency used in filenames (default 60).')
    p.add_argument('--threshold', type=float, default=0.5,
                   help='Keep rows with quality > threshold (default 0.5).')
    p.add_argument('--exclude-hard-artifact', dest='exclude_hard_artifact',
                   action='store_true', default=True,
                   help='Exclude rows where artifact_hard_clip==1 when available (default on).')
    p.add_argument('--no-exclude-hard-artifact', dest='exclude_hard_artifact',
                   action='store_false',
                   help='Ignore artifact_hard_clip and filter by quality only.')
    p.add_argument('--stem', default='index_vs_raw_goertzel',
                   help='Input CSV filename stem (default index_vs_raw_goertzel).')
    p.add_argument('--bins', type=int, default=36, help='Histogram bin count (default 36).')
    p.add_argument('--outdir', default='iBrainCenter/comparison/goertzel_histograms',
                   help='Output folder for histogram PNGs.')
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    groups: dict[tuple[str, int], np.ndarray] = {}

    for sub in _iter_subject_dirs(args.root):
        for ch in args.channels:
            csv_path = _build_csv_path(args.root, sub, args.stem, ch, args.target_freq)
            if not os.path.isfile(csv_path):
                print(f'Skip missing file: {csv_path}')
                continue

            df = pd.read_csv(csv_path)
            if not {'goertzel_db', 'quality'}.issubset(df.columns):
                print(f'Skip invalid columns: {csv_path}')
                continue

            quality_col = 'quality_final' if 'quality_final' in df.columns else 'quality'
            q = df[quality_col].to_numpy(dtype=float)
            keep = np.isfinite(q) & (q > args.threshold)
            if args.exclude_hard_artifact and 'artifact_hard_clip' in df.columns:
                keep &= (df['artifact_hard_clip'].to_numpy(dtype=float) < 0.5)
            db = df['goertzel_db'].to_numpy(dtype=float)[keep]
            groups[(sub, ch)] = db

    if not groups:
        raise SystemExit('No valid subject/channel data found.')

    # Per-subject, per-channel histograms.
    for (sub, ch), db in groups.items():
        title = f'{sub}  ch{ch}  |  Goertzel {args.target_freq:g}Hz (quality>{args.threshold:g})'
        out_png = os.path.join(args.outdir, f'{sub}_ch{ch}_{args.target_freq:g}Hz_thr_{args.threshold:g}.png')
        _plot_hist(db, title, out_png, bins=args.bins)
        print(f'Saved: {out_png}')

    # Overall by channel.
    for ch in args.channels:
        collect = [v for (s, c), v in groups.items() if c == ch and v.size > 0]
        overall = np.concatenate(collect) if collect else np.array([], dtype=float)
        title = f'ALL_SUBJECTS  ch{ch}  |  Goertzel {args.target_freq:g}Hz (quality>{args.threshold:g})'
        out_png = os.path.join(args.outdir, f'ALL_SUBJECTS_ch{ch}_{args.target_freq:g}Hz_thr_{args.threshold:g}.png')
        _plot_hist(overall, title, out_png, bins=args.bins)
        print(f'Saved: {out_png}')

    # Overall all channels.
    all_values = [v for v in groups.values() if v.size > 0]
    overall_all = np.concatenate(all_values) if all_values else np.array([], dtype=float)
    title = f'ALL_SUBJECTS  ALL_CHANNELS  |  Goertzel {args.target_freq:g}Hz (quality>{args.threshold:g})'
    out_png = os.path.join(args.outdir, f'ALL_SUBJECTS_ALL_CHANNELS_{args.target_freq:g}Hz_thr_{args.threshold:g}.png')
    _plot_hist(overall_all, title, out_png, bins=args.bins)
    print(f'Saved: {out_png}')


if __name__ == '__main__':
    main()
