#!/usr/bin/env python3
"""Filter iBrainCenter windows by quality + hard-artifact criteria.

Outputs per subject/channel:
- filtered CSV: quality > threshold and artifact_hard_clip == 0
- artifact CSV: artifact_hard_clip == 1
- timeline PNG with hard-artifact windows highlighted
- hard-artifact snippet gallery PNG
"""

from __future__ import annotations

import argparse
import math
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lilia.io import bandpass_filter, load_merged_csv
import plot_event_markers as pem


def _iter_subject_dirs(root: str):
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, 'merged.csv')):
            yield name


def _load_decimated(time_us: np.ndarray, y: np.ndarray, max_points: int = 16000):
    t_s = (time_us - int(time_us[0])) / 1e6
    stride = max(1, len(t_s) // max_points)
    return t_s[::stride], y[::stride]


def _plot_timeline(
    out_png: str,
    subject: str,
    ch: int,
    df: pd.DataFrame,
    t_raw_s: np.ndarray,
    y_raw: np.ndarray,
    quality_threshold: float,
    win_sec: float,
) -> None:
    t = df['time_s'].to_numpy(dtype=float)
    db = df['goertzel_db'].to_numpy(dtype=float)
    q = df['quality'].to_numpy(dtype=float)
    hard = df['artifact_hard_clip'].to_numpy(dtype=float) > 0.5
    keep = (q > quality_threshold) & (~hard)

    fig, axes = plt.subplots(
        3, 1, figsize=(13.8, 8.0), sharex=True,
        gridspec_kw={'height_ratios': [1.8, 1.2, 1.2]},
    )

    fig.suptitle(
        f'{subject} ch{ch} | filtered windows and hard artifacts',
        fontsize=12,
        fontweight='bold',
    )

    ax0, ax1, ax2 = axes
    ax0.plot(t / 60.0, db, color='0.78', lw=0.8, label='all windows')
    ax0.scatter((t[keep] / 60.0), db[keep], s=10, color='#1f77b4', alpha=0.85,
                label='kept (quality + no hard clip)')
    ax0.scatter((t[hard] / 60.0), db[hard], s=16, color='#d62728', alpha=0.9,
                marker='x', label='artifact_hard_clip=1')
    ax0.set_ylabel('Goertzel dB')
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc='upper right', fontsize=8)

    ax1.plot(t / 60.0, q, color='#2ca02c', lw=1.2, label='quality')
    ax1.axhline(quality_threshold, color='#d62728', ls='--', lw=1.0,
                label=f'threshold={quality_threshold:g}')
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel('Quality')
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc='upper right', fontsize=8)

    ax2.plot(t_raw_s / 60.0, y_raw, color='#444444', lw=0.45, alpha=0.85)
    ax2.set_ylabel('Raw (uV)')
    ax2.set_xlabel('Time (min)')
    ax2.grid(True, alpha=0.25)

    half = win_sec / 2.0
    for tb in t[hard]:
        lo = (tb - half) / 60.0
        hi = (tb + half) / 60.0
        ax0.axvspan(lo, hi, color='#d62728', alpha=0.08, lw=0)
        ax2.axvspan(lo, hi, color='#d62728', alpha=0.10, lw=0)

    for tb in t[(q <= quality_threshold) & (~hard)]:
        lo = (tb - half) / 60.0
        hi = (tb + half) / 60.0
        ax0.axvspan(lo, hi, color='0.55', alpha=0.06, lw=0)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_hard_gallery(
    out_png: str,
    subject: str,
    ch: int,
    hard_rows: pd.DataFrame,
    time_us: np.ndarray,
    raw_ch: np.ndarray,
    bp_ch: np.ndarray,
    fs: float,
    seg_sec: float,
    max_panels: int,
) -> None:
    if hard_rows.empty:
        return

    rows = hard_rows.sort_values('time_s').head(max_panels).reset_index(drop=True)
    n = len(rows)
    ncol = 3
    nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 2.9 * nrow), sharex=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(-1)

    total_s = (len(raw_ch) - 1) / fs
    half = seg_sec / 2.0

    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis('off')
            continue

        r = rows.iloc[i]
        center = float(r['time_s'])
        start = max(0.0, center - half)
        end = min(total_s, center + half)
        i0 = int(round(start * fs))
        i1 = max(i0 + 1, int(round(end * fs)))
        i0 = max(0, min(i0, len(raw_ch) - 1))
        i1 = max(i0 + 1, min(i1, len(raw_ch)))

        x = np.arange(i0, i1) / fs - center
        yr = raw_ch[i0:i1]
        yb = bp_ch[i0:i1]
        if yr.size == 0 or yb.size == 0:
            ax.axis('off')
            continue

        ax.plot(x, yr, color='0.45', lw=0.7, label='raw')
        ax.plot(x, yb, color='#1f77b4', lw=0.7, alpha=0.9, label='bp 0.5-45')
        ax.axvline(0.0, color='#d62728', ls='--', lw=0.9)
        ax.set_title(
            f"t={center:.1f}s db={float(r['goertzel_db']):.1f}\nq={float(r['quality']):.3f}",
            fontsize=8,
        )
        ax.grid(True, alpha=0.22)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=8)
    fig.suptitle(f'{subject} ch{ch} hard-artifact snippets', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Filter by criteria and plot hard artifacts for iBrainCenter.')
    p.add_argument('--root', default='iBrainCenter', help='Root subject folder.')
    p.add_argument('--channels', type=int, nargs='+', default=[1, 2], help='Channels to process.')
    p.add_argument('--target-freq', type=float, default=60.0, help='Goertzel frequency in filename.')
    p.add_argument('--stem', default='index_vs_raw_goertzel', help='Input filename stem.')
    p.add_argument('--quality-threshold', type=float, default=0.5, help='quality > threshold to keep.')
    p.add_argument('--win-sec', type=float, default=5.0, help='Window length (sec) for span display.')
    p.add_argument('--fs', type=float, default=500.0, help='Sampling rate.')
    p.add_argument('--outdir', default='iBrainCenter/comparison/hard_clip_filter', help='Output folder.')
    p.add_argument('--max-gallery', type=int, default=12, help='Max snippets in each gallery figure.')
    p.add_argument('--gallery-seg-sec', type=float, default=5.0, help='Snippet length for gallery.')
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    report_rows = []

    for subject in _iter_subject_dirs(args.root):
        merged = os.path.join(args.root, subject, 'merged.csv')
        time_us, data = load_merged_csv(merged)
        bp = bandpass_filter(data, fs=args.fs, lo=pem.BP_LOW, hi=pem.BP_HIGH)

        for ch in args.channels:
            in_csv = os.path.join(
                args.root,
                subject,
                f'{args.stem}_ch{ch}_{args.target_freq:g}Hz.csv',
            )
            if not os.path.isfile(in_csv):
                continue

            df = pd.read_csv(in_csv)
            needed = {'time_s', 'goertzel_db', 'quality', 'artifact_hard_clip'}
            if not needed.issubset(df.columns):
                print(f'Skip (missing columns): {in_csv}')
                continue

            hard = df['artifact_hard_clip'].to_numpy(dtype=float) > 0.5
            quality_col = 'quality_final' if 'quality_final' in df.columns else 'quality'
            q = df[quality_col].to_numpy(dtype=float)
            keep = (q > args.quality_threshold) & (~hard)

            filtered = df.loc[keep].reset_index(drop=True)
            hard_df = df.loc[hard].reset_index(drop=True)

            base = f'{subject}_ch{ch}_{args.target_freq:g}Hz_thr_{args.quality_threshold:g}'
            out_filtered = os.path.join(args.outdir, f'{base}_filtered.csv')
            out_hard = os.path.join(args.outdir, f'{base}_artifact_hard_clip.csv')
            out_timeline = os.path.join(args.outdir, f'{base}_timeline.png')
            out_gallery = os.path.join(args.outdir, f'{base}_hard_gallery.png')

            filtered.to_csv(out_filtered, index=False)
            hard_df.to_csv(out_hard, index=False)

            t_raw_s, y_raw = _load_decimated(time_us, data[:, ch - 1])
            _plot_timeline(
                out_png=out_timeline,
                subject=subject,
                ch=ch,
                df=df,
                t_raw_s=t_raw_s,
                y_raw=y_raw,
                quality_threshold=args.quality_threshold,
                win_sec=args.win_sec,
            )
            _plot_hard_gallery(
                out_png=out_gallery,
                subject=subject,
                ch=ch,
                hard_rows=hard_df,
                time_us=time_us,
                raw_ch=data[:, ch - 1],
                bp_ch=bp[:, ch - 1],
                fs=args.fs,
                seg_sec=args.gallery_seg_sec,
                max_panels=args.max_gallery,
            )

            report_rows.append(
                {
                    'subject': subject,
                    'channel': ch,
                    'n_total': int(len(df)),
                    'n_hard_artifact': int(hard.sum()),
                    'n_kept_after_filter': int(keep.sum()),
                    'kept_ratio': float(keep.mean()) if len(df) else np.nan,
                    'filtered_csv': out_filtered,
                    'artifact_csv': out_hard,
                    'timeline_png': out_timeline,
                    'hard_gallery_png': out_gallery,
                }
            )

            print(f'Saved: {out_filtered}')
            print(f'Saved: {out_hard}')
            print(f'Saved: {out_timeline}')
            if not hard_df.empty:
                print(f'Saved: {out_gallery}')

    if report_rows:
        report = pd.DataFrame(report_rows)
        out_report = os.path.join(args.outdir, f'filter_report_{args.target_freq:g}Hz_thr_{args.quality_threshold:g}.csv')
        report.to_csv(out_report, index=False)
        print(f'Saved: {out_report}')
    else:
        print('No subject/channel data processed.')


if __name__ == '__main__':
    main()
