#!/usr/bin/env python3
"""Resample clean Goertzel windows per subject after hard-artifact exclusion.

For each subject, this script:
- loads ch1/ch2 Goertzel window CSVs
- keeps windows with quality > threshold and artifact_hard_clip == 0
- prefers windows near a target dB value (e.g. 67.15 dB)
- randomly samples N windows per subject without replacement
- plots a small gallery of sampled raw + bandpass snippets
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


def _csv_path(root: str, subject: str, stem: str, ch: int, target_freq: float) -> str:
    return os.path.join(root, subject, f"{stem}_ch{ch}_{target_freq:g}Hz.csv")


def _load_candidates(
    root: str,
    subject: str,
    channels: list[int],
    stem: str,
    target_freq: float,
    target_db: float,
    quality_threshold: float,
    fs: float,
) -> pd.DataFrame:
    frames = []
    for ch in channels:
        path = _csv_path(root, subject, stem, ch, target_freq)
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path)
        need = {'time_s', 'goertzel_db', 'quality'}
        if not need.issubset(df.columns):
            continue

        quality_col = 'quality_final' if 'quality_final' in df.columns else 'quality'
        q = df[quality_col].to_numpy(dtype=float)
        keep = np.isfinite(q) & (q > quality_threshold)
        if 'artifact_hard_clip' in df.columns:
            keep &= (df['artifact_hard_clip'].to_numpy(dtype=float) < 0.5)

        if not np.any(keep):
            continue

        sub = df.loc[keep].copy()
        sub['subject'] = subject
        sub['channel'] = ch
        if 'window_start_idx' in sub.columns and 'window_end_idx' in sub.columns:
            sub['window_start_idx'] = sub['window_start_idx'].astype(int)
            sub['window_end_idx'] = sub['window_end_idx'].astype(int)
            sub['window_start_s'] = sub['window_start_idx'].astype(float) / fs
            sub['window_end_s'] = sub['window_end_idx'].astype(float) / fs
            sub['window_center_s'] = 0.5 * (sub['window_start_s'] + sub['window_end_s'])
            sub['window_duration_s'] = sub['window_end_s'] - sub['window_start_s']
        else:
            sub['window_start_s'] = sub['time_s'].astype(float) - 2.5
            sub['window_end_s'] = sub['time_s'].astype(float) + 2.5
            sub['window_center_s'] = sub['time_s'].astype(float)
            sub['window_duration_s'] = 5.0
        sub['distance_db'] = np.abs(sub['goertzel_db'].to_numpy(dtype=float) - target_db)
        frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=['subject', 'channel', 'time_s', 'goertzel_db', 'quality', 'distance_db'])
    return pd.concat(frames, ignore_index=True)


def _plot_gallery(
    out_png: str,
    subject: str,
    sampled: pd.DataFrame,
    merged_csv: str,
    fs: float,
    seg_sec: float,
) -> None:
    if sampled.empty:
        return

    time_us, data = load_merged_csv(merged_csv)
    bp = bandpass_filter(data, fs=fs, lo=pem.BP_LOW, hi=pem.BP_HIGH)

    n = len(sampled)
    ncol = 5
    nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2 * ncol, 2.8 * nrow), sharex=False)
    axes = np.atleast_1d(axes).reshape(-1)
    total_s = (len(time_us) - 1) / fs
    half = seg_sec / 2.0

    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis('off')
            continue

        r = sampled.iloc[i]
        ch = int(r['channel'])
        center_label = float(r['time_s'])
        if 'window_start_idx' in sampled.columns and 'window_end_idx' in sampled.columns:
            i0 = int(r['window_start_idx'])
            i1 = int(r['window_end_idx'])
            center_idx = 0.5 * (i0 + i1)
        else:
            center = float(r['window_center_s']) if 'window_center_s' in sampled.columns else float(r['time_s'])
            start = max(0.0, center - half)
            end = min(total_s, center + half)
            i0 = int(round(start * fs))
            i1 = max(i0 + 1, int(round(end * fs)))
            center_idx = center * fs
        i0 = max(0, min(i0, len(time_us) - 1))
        i1 = max(i0 + 1, min(i1, len(time_us)))

        x = (np.arange(i0, i1) - center_idx) / fs
        raw = data[i0:i1, ch - 1]
        bps = bp[i0:i1, ch - 1]

        if raw.size == 0 or bps.size == 0:
            ax.axis('off')
            continue

        ax.plot(x, raw, color='0.45', lw=0.8, label='raw')
        ax.plot(x, bps, color='#1f77b4', lw=0.8, alpha=0.9, label='bp 0.5-45')
        ax.axvline(0.0, color='#d62728', ls='--', lw=1.0)
        ax.set_title(
            f"#{i+1} ch{ch} | t={center_label:.1f}s\n{float(r['goertzel_db']):.2f} dB | qf={float(r['quality_final']) if 'quality_final' in sampled.columns else float(r['quality']):.3f}",
            fontsize=8,
        )
        ax.grid(True, alpha=0.22)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=8)
    fig.suptitle(f'{subject} resampled clean Goertzel windows', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Resample clean Goertzel windows per subject.')
    p.add_argument('--root', default='iBrainCenter', help='Root folder with subject dirs.')
    p.add_argument('--channels', type=int, nargs='+', default=[1, 2], help='Channels to use.')
    p.add_argument('--stem', default='index_vs_raw_goertzel', help='Input CSV stem.')
    p.add_argument('--target-freq', type=float, default=60.0, help='Target frequency in Hz.')
    p.add_argument('--target-db', type=float, default=67.15, help='Target dB value to prioritize.')
    p.add_argument('--quality-threshold', type=float, default=0.5, help='Keep windows with quality > threshold.')
    p.add_argument('--n', type=int, default=10, help='Number of windows to sample per subject.')
    p.add_argument('--pool-size', type=int, default=30, help='Take the nearest pool-size windows before random sampling.')
    p.add_argument('--seed', type=int, default=42, help='Random seed.')
    p.add_argument('--fs', type=float, default=500.0, help='Sampling rate.')
    p.add_argument('--segment-sec', type=float, default=5.0, help='Snippet length in seconds.')
    p.add_argument('--outdir', default='iBrainCenter/comparison/goertzel_samples_67dB_resampled',
                   help='Output directory.')
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    report_rows = []

    for subject in _iter_subject_dirs(args.root):
        candidates = _load_candidates(
            root=args.root,
            subject=subject,
            channels=args.channels,
            stem=args.stem,
            target_freq=args.target_freq,
            target_db=args.target_db,
            quality_threshold=args.quality_threshold,
            fs=args.fs,
        )

        if candidates.empty:
            print(f'Skip {subject}: no clean candidates.')
            continue

        candidates = candidates.sort_values(['distance_db', 'goertzel_db', 'time_s']).reset_index(drop=True)
        pool = candidates.head(min(args.pool_size, len(candidates))).copy()

        take = min(args.n, len(pool))
        pick_idx = rng.choice(len(pool), size=take, replace=False)
        sampled = pool.iloc[np.sort(pick_idx)].reset_index(drop=True)
        sampled = sampled.sort_values(['distance_db', 'channel', 'time_s']).reset_index(drop=True)
        sampled.insert(0, 'sample_id', [f'{subject}_S{i+1:02d}' for i in range(len(sampled))])
        sampled['selected_rank'] = np.arange(1, len(sampled) + 1)

        sub_out = os.path.join(args.outdir, subject)
        os.makedirs(sub_out, exist_ok=True)
        sampled_csv = os.path.join(sub_out, f'sampled_{args.target_freq:g}Hz_{args.target_db:g}dB.csv')
        sampled.to_csv(sampled_csv, index=False)

        gallery_png = os.path.join(sub_out, f'sampled_{args.target_freq:g}Hz_{args.target_db:g}dB_gallery.png')
        _plot_gallery(
            out_png=gallery_png,
            subject=subject,
            sampled=sampled,
            merged_csv=os.path.join(args.root, subject, 'merged.csv'),
            fs=args.fs,
            seg_sec=args.segment_sec,
        )

        report_rows.append(
            {
                'subject': subject,
                'n_candidates_clean': int(len(candidates)),
                'n_pool': int(len(pool)),
                'n_sampled': int(len(sampled)),
                'min_db': float(sampled['goertzel_db'].min()) if not sampled.empty else np.nan,
                'max_db': float(sampled['goertzel_db'].max()) if not sampled.empty else np.nan,
                'mean_db': float(sampled['goertzel_db'].mean()) if not sampled.empty else np.nan,
                'sampled_csv': sampled_csv,
                'gallery_png': gallery_png,
            }
        )

        print(f'Saved: {sampled_csv}')
        print(f'Saved: {gallery_png}')

    if report_rows:
        report = pd.DataFrame(report_rows)
        report_path = os.path.join(
            args.outdir,
            f'resample_report_{args.target_freq:g}Hz_{args.target_db:g}dB_thr_{args.quality_threshold:g}.csv',
        )
        report.to_csv(report_path, index=False)
        print(f'Saved: {report_path}')


if __name__ == '__main__':
    main()
