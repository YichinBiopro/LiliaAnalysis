#!/usr/bin/env python3
"""Randomly sample signal segments near a target Goertzel dB value.

Scans iBrainCenter subject CSV outputs from plot_goertzel_vs_raw.py and selects
windows whose goertzel_db is close to a target (e.g., 67.15 dB at 60 Hz).
For each sampled window, plot raw and bandpass (0.5-45 Hz) signal segments.
"""

from __future__ import annotations

import argparse
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


def _collect_matches(
    root: str,
    channels: list[int],
    stem: str,
    target_freq: float,
    target_db: float,
    tol_db: float,
    quality_threshold: float | None,
) -> pd.DataFrame:
    rows = []
    for sub in _iter_subject_dirs(root):
        for ch in channels:
            path = _csv_path(root, sub, stem, ch, target_freq)
            if not os.path.isfile(path):
                continue
            df = pd.read_csv(path)
            need = {'time_s', 'goertzel_db', 'quality'}
            if not need.issubset(df.columns):
                continue

            db = df['goertzel_db'].to_numpy(dtype=float)
            q = df['quality'].to_numpy(dtype=float)
            keep = np.isfinite(db)
            keep &= np.abs(db - target_db) <= tol_db
            if quality_threshold is not None:
                keep &= np.isfinite(q) & (q > quality_threshold)

            if not np.any(keep):
                continue

            sub_df = pd.DataFrame(
                {
                    'subject': sub,
                    'channel': ch,
                    'time_s': df.loc[keep, 'time_s'].to_numpy(dtype=float),
                    'goertzel_db': db[keep],
                    'quality': q[keep],
                }
            )
            rows.append(sub_df)

    if not rows:
        return pd.DataFrame(columns=['subject', 'channel', 'time_s', 'goertzel_db', 'quality'])
    return pd.concat(rows, ignore_index=True)


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
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Sample and plot segments near target Goertzel dB.')
    p.add_argument('--root', default='iBrainCenter', help='Root folder (default iBrainCenter).')
    p.add_argument('--channels', type=int, nargs='+', default=[1, 2], help='Channels (default 1 2).')
    p.add_argument('--stem', default='index_vs_raw_goertzel', help='Input CSV stem.')
    p.add_argument('--target-freq', type=float, default=60.0, help='Target frequency in Hz.')
    p.add_argument('--target-db', type=float, default=67.15, help='Target dB value to sample around.')
    p.add_argument('--tol-db', type=float, default=0.6, help='Absolute dB tolerance around target.')
    p.add_argument('--quality-threshold', type=float, default=0.5,
                   help='Only keep windows with quality > threshold. Use negative to disable.')
    p.add_argument('--n', type=int, default=6, help='Number of random segments to sample.')
    p.add_argument('--seed', type=int, default=42, help='Random seed for reproducible sampling.')
    p.add_argument('--fs', type=float, default=500.0, help='Sampling rate (Hz).')
    p.add_argument('--segment-sec', type=float, default=5.0, help='Plotted segment length (seconds).')
    p.add_argument('--outdir', default='iBrainCenter/comparison/goertzel_samples_67dB',
                   help='Output folder for sample plots and CSV list.')
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    q_th = None if args.quality_threshold < 0 else args.quality_threshold
    matches = _collect_matches(
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

    rng = np.random.default_rng(args.seed)
    take = min(args.n, len(matches))
    idx = rng.choice(len(matches), size=take, replace=False)
    picked = matches.iloc[np.sort(idx)].reset_index(drop=True)

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


if __name__ == '__main__':
    main()
