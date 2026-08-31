#!/usr/bin/env python3
"""Randomly sample signal segments near a target Goertzel dB value.

Scans iBrainCenter subject CSV outputs from plot_goertzel_vs_raw.py and selects
windows whose goertzel_db is close to a target (e.g., 67.15 dB at 60 Hz).
For each sampled window, plot raw and bandpass (0.5-45 Hz) signal segments.
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

from lilia.goertzel_sampling import (
    collect_candidates_per_subject,
    collect_matches_global,
    sample_rows,
)
from lilia.io import bandpass_filter, load_merged_csv
from lilia.subject_paths import iter_subject_dirs
import plot_event_markers as pem


def _plot_one_segment(
    root: str,
    subject: str,
    ch: int,
    center_s: float,
    db: float,
    quality: float,
    fs: float,
    seg_sec: float,
    out_png: str,
) -> None:
    merged_csv = os.path.join(root, subject, 'merged.csv')
    t_us, data = load_merged_csv(merged_csv)
    bp = bandpass_filter(data, fs=fs, lo=pem.BP_LOW, hi=pem.BP_HIGH)

    ch_idx = ch - 1
    n = data.shape[0]

    half = seg_sec / 2.0
    start_s = max(0.0, center_s - half)
    end_s = min((n - 1) / fs, center_s + half)

    i0 = int(round(start_s * fs))
    i1 = int(round(end_s * fs))
    if i1 <= i0:
        i1 = min(i0 + 1, n)

    x = np.arange(i0, i1) / fs
    x_rel = x - center_s
    raw = data[i0:i1, ch_idx]
    bps = bp[i0:i1, ch_idx]

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.6), sharex=True)
    fig.suptitle(
        f'{subject} ch{ch} | center={center_s:.2f}s | 60Hz={db:.2f} dB | quality={quality:.3f}',
        fontsize=11,
        fontweight='bold',
    )

    axes[0].plot(x_rel, raw, color='#444444', lw=0.8)
    axes[0].axvline(0.0, color='#d62728', ls='--', lw=1.0, alpha=0.7)
    axes[0].set_ylabel('Raw (uV)')
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(x_rel, bps, color='#1f77b4', lw=0.8)
    axes[1].axvline(0.0, color='#d62728', ls='--', lw=1.0, alpha=0.7)
    axes[1].set_ylabel('Bandpass 0.5-45 Hz (uV)')
    axes[1].set_xlabel('Time relative to window center (s)')
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=170)
    fig.savefig(os.path.splitext(out_png)[0] + '.svg')
    plt.close(fig)


def _plot_subject_gallery(
    out_png: str,
    root: str,
    subject: str,
    sampled: pd.DataFrame,
    fs: float,
    seg_sec: float,
) -> None:
    if sampled.empty:
        return

    merged_csv = os.path.join(root, subject, 'merged.csv')
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
        qf = float(r['quality_final']) if 'quality_final' in sampled.columns else float(r['quality'])
        ax.set_title(
            f"#{i+1} ch{ch} | t={center_label:.1f}s\\n{float(r['goertzel_db']):.2f} dB | qf={qf:.3f}",
            fontsize=8,
        )
        ax.grid(True, alpha=0.22)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=8)
    fig.suptitle(f'{subject} resampled clean Goertzel windows', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=160)
    fig.savefig(os.path.splitext(out_png)[0] + '.svg')
    plt.close(fig)


def run_global_sampling(args: argparse.Namespace) -> None:
    q_th = None if args.quality_threshold < 0 else args.quality_threshold
    matches = collect_matches_global(
        root=args.root,
        channels=args.channels,
        stem=args.stem,
        target_freq=args.target_freq,
        target_db=args.target_db,
        tol_db=args.tol_db,
        quality_threshold=q_th,
    )

    if matches.empty:
        raise SystemExit('No matching windows found. Try increasing --tol-db or lowering threshold.')

    picked = sample_rows(matches, n=args.n, seed=args.seed)
    take = len(picked)

    os.makedirs(args.outdir, exist_ok=True)
    picked_csv = os.path.join(args.outdir, f'sampled_windows_{args.target_freq:g}Hz_{args.target_db:g}dB.csv')
    picked.to_csv(picked_csv, index=False)
    print(f'Saved: {picked_csv}')

    for i, r in picked.iterrows():
        out_png = os.path.join(
            args.outdir,
            f"sample_{i+1:02d}_{r['subject']}_ch{int(r['channel'])}_t{r['time_s']:.2f}s_db{r['goertzel_db']:.2f}.png",
        )
        _plot_one_segment(
            root=args.root,
            subject=str(r['subject']),
            ch=int(r['channel']),
            center_s=float(r['time_s']),
            db=float(r['goertzel_db']),
            quality=float(r['quality']),
            fs=args.fs,
            seg_sec=args.segment_sec,
            out_png=out_png,
        )
        print(f'Saved: {out_png}')

    print(f'Matches found: {len(matches)} | Sampled: {take}')


def run_per_subject_resample(args: argparse.Namespace) -> None:
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    report_rows: list[dict] = []

    for subject in iter_subject_dirs(args.root):
        candidates = collect_candidates_per_subject(
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
        _plot_subject_gallery(
            out_png=gallery_png,
            root=args.root,
            subject=subject,
            sampled=sampled,
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Sample and plot segments near target Goertzel dB.')
    p.add_argument('--mode', choices=['global', 'per-subject'], default='global',
                   help='Sampling mode: global pooled sampling or per-subject resampling.')
    p.add_argument('--root', default='iBrainCenter', help='Root folder (default iBrainCenter).')
    p.add_argument('--channels', type=int, nargs='+', default=[1, 2], help='Channels (default 1 2).')
    p.add_argument('--stem', default='index_vs_raw_goertzel', help='Input CSV stem.')
    p.add_argument('--target-freq', type=float, default=60.0, help='Target frequency in Hz.')
    p.add_argument('--target-db', type=float, default=67.15, help='Target dB value to sample around.')
    p.add_argument('--tol-db', type=float, default=0.6, help='Absolute dB tolerance around target.')
    p.add_argument('--quality-threshold', type=float, default=0.5,
                   help='Only keep windows with quality > threshold. Use negative to disable.')
    p.add_argument('--n', type=int, default=6, help='Number of random segments to sample.')
    p.add_argument('--pool-size', type=int, default=30,
                   help='In per-subject mode, nearest-N pool size before random sampling.')
    p.add_argument('--seed', type=int, default=42, help='Random seed for reproducible sampling.')
    p.add_argument('--fs', type=float, default=500.0, help='Sampling rate (Hz).')
    p.add_argument('--segment-sec', type=float, default=5.0, help='Plotted segment length (seconds).')
    p.add_argument('--outdir', default='iBrainCenter/comparison/goertzel_samples_67dB',
                   help='Output folder for sample plots and CSV list.')
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == 'global':
        run_global_sampling(args)
        return
    run_per_subject_resample(args)


if __name__ == '__main__':
    main()
