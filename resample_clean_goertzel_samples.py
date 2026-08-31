#!/usr/bin/env python3
"""Compatibility wrapper for per-subject Goertzel resampling.

The implementation has been consolidated in
``sample_segments_by_goertzel_db.py --mode per-subject``.
This wrapper preserves the legacy CLI.
"""

from __future__ import annotations

import argparse

from sample_segments_by_goertzel_db import run_per_subject_resample


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Resample clean Goertzel windows per subject.')
    p.add_argument('--root', default='iBrainCenter', help='Root folder with subject dirs.')
    p.add_argument('--channels', type=int, nargs='+', default=[1, 2], help='Channels to use.')
    p.add_argument('--stem', default='index_vs_raw_goertzel', help='Input CSV stem.')
    p.add_argument('--target-freq', type=float, default=60.0, help='Target frequency in Hz.')
    p.add_argument('--target-db', type=float, default=67.15, help='Target dB value to prioritize.')
    p.add_argument('--quality-threshold', type=float, default=0.5, help='Keep windows with quality > threshold.')
    p.add_argument('--n', type=int, default=10, help='Number of windows to sample per subject.')
    p.add_argument('--pool-size', type=int, default=30, help='Take nearest pool-size windows before random sampling.')
    p.add_argument('--seed', type=int, default=42, help='Random seed.')
    p.add_argument('--fs', type=float, default=500.0, help='Sampling rate.')
    p.add_argument('--segment-sec', type=float, default=5.0, help='Snippet length in seconds.')
    p.add_argument('--outdir', default='iBrainCenter/comparison/goertzel_samples_67dB_resampled',
                   help='Output directory.')
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    args.mode = 'per-subject'
    run_per_subject_resample(args)


if __name__ == '__main__':
    main()
