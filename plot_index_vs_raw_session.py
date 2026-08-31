#!/usr/bin/env python3
"""Compatibility wrapper for session-baseline qEEG-vs-raw plotting.

The implementation now lives in ``plot_index_vs_raw.py`` as
``plot_index_vs_raw_session`` and can also be called directly with
``python plot_index_vs_raw.py --session-baseline ...``.
"""

from __future__ import annotations

import argparse

import plot_event_markers as pem
from plot_index_vs_raw import plot_index_vs_raw_session


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--be-csv', required=True, metavar='PATH',
                   help='A *_band_entropy_ch*.csv produced by spectral_entropy.py.')
    p.add_argument('--raw-csv', required=True, metavar='PATH',
                   help='The corresponding merged.csv (lilia format).')
    p.add_argument('--ch', type=int, default=1, help='1-based channel (default 1).')
    p.add_argument('--label', required=True, help='Subject/session label for titles.')
    p.add_argument('--out', required=True, metavar='PATH', help='Output PNG path.')
    p.add_argument('--seconds', action='store_true',
                   help='Use seconds instead of minutes for the x-axis.')
    p.add_argument('--smooth-win', type=int, default=5,
                   help='Rolling-median smoothing window, in analysis windows (default 5).')
    p.add_argument('--raw-ylim', type=float, nargs=2, default=(-150.0, 150.0),
                   metavar=('LO', 'HI'), help='Raw EEG panel y-limits (default -150 150).')
    p.add_argument('--quality-threshold', type=float,
                   default=pem.QUALITY_THRESHOLD,
                   help=f'Quality mask threshold (default {pem.QUALITY_THRESHOLD:g}).')
    p.add_argument('--win-sec', type=float, default=2.0,
                   help='Analysis window length used by spectral_entropy.py (default 2.0).')
    p.add_argument('--step-sec', type=float, default=2.0,
                   help='Analysis step used by spectral_entropy.py (default 2.0).')
    p.add_argument('--fs', type=float, default=500.0,
                   help='Sampling rate in Hz (default 500).')
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    plot_index_vs_raw_session(
        args.be_csv, args.raw_csv, args.label, args.ch, args.out,
        win_sec=args.win_sec, step_sec=args.step_sec, fs=args.fs,
        use_minutes=not args.seconds, smooth_win=args.smooth_win,
        raw_ylim=tuple(args.raw_ylim), quality_threshold=args.quality_threshold,
    )


if __name__ == '__main__':
    main()
