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

from lilia.goertzel_distribution import collect_group_data, with_aggregates


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
        fig.savefig(os.path.splitext(out_png)[0] + '.svg')
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
    fig.savefig(os.path.splitext(out_png)[0] + '.svg')
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

    try:
        base_groups, missing = collect_group_data(
            root=args.root,
            channels=args.channels,
            stem=args.stem,
            target_freq=args.target_freq,
            threshold=args.threshold,
            exclude_hard_artifact=args.exclude_hard_artifact,
            require_power=False,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if missing:
        for path in missing:
            print(f'Skip missing file: {path}')

    if not base_groups:
        raise SystemExit('No valid subject/channel data found.')

    groups = with_aggregates(base_groups, args.channels)

    # Per-subject, per-channel histograms.
    for g in groups:
        if g.subject == 'ALL_SUBJECTS' and g.channel == 'ALL_CHANNELS':
            name = f'ALL_SUBJECTS_ALL_CHANNELS_{args.target_freq:g}Hz_thr_{args.threshold:g}.png'
        else:
            name = f'{g.subject}_{g.channel}_{args.target_freq:g}Hz_thr_{args.threshold:g}.png'
        title = f'{g.subject}  {g.channel}  |  Goertzel {args.target_freq:g}Hz (quality>{args.threshold:g})'
        out_png = os.path.join(args.outdir, name)
        _plot_hist(g.power_db, title, out_png, bins=args.bins)
        print(f'Saved: {out_png}')


if __name__ == '__main__':
    main()
