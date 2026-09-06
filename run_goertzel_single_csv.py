#!/usr/bin/env python3
"""Run the windowed Goertzel power pipeline (plot_goertzel_vs_raw) on a single
arbitrary merged-format CSV that doesn't live under a <root>/<subject>/merged.csv
layout (e.g. ad-hoc ECEO_60hz recordings).

Reuses plot_goertzel_vs_raw._plot_subject as-is so output (dB trend + quality +
BP/raw EEG panels, plus per-window CSV) matches the iBrainCenter pipeline.
"""

from __future__ import annotations

import argparse
import os

import plot_goertzel_vs_raw as pgv
from lilia.provenance import validate_goertzel_cache


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv', help='Path to the merged-format CSV (4-row header, Time[us]+ch1..N).')
    p.add_argument('--channels', type=int, nargs='+', default=[1, 2, 3, 4],
                    help='1-based channel indices (default 1 2 3 4).')
    p.add_argument('--fs', type=float, default=500.0)
    p.add_argument('--target-freq', type=float, default=60.0)
    p.add_argument('--win-sec', type=float, default=5.0)
    p.add_argument('--step-sec', type=float, default=5.0)
    p.add_argument('--smooth-win', type=int, default=5)
    p.add_argument('--quality-threshold', type=float, default=pgv.pem.QUALITY_THRESHOLD)
    p.add_argument('--no-exclude-hard-artifact', dest='exclude_hard_artifact',
                    action='store_false', default=True)
    p.add_argument('--raw-ylim', type=float, nargs=2, default=(-150.0, 150.0))
    p.add_argument('--outdir', default=None,
                    help='Output dir (default: same dir as the input CSV).')
    p.add_argument('--out-stem', default=None,
                    help='Filename stem (default: input CSV basename without extension).')
    p.add_argument('--linear', action='store_true',
                    help='Plot Goertzel power on a linear scale instead of dB '
                         '(the per-window CSV always has both goertzel_power '
                         'and goertzel_db columns regardless of this flag).')
    p.add_argument('--reuse-csv', action='store_true',
                    help='If the per-window CSV from a previous run already exists, '
                         'reuse it and only re-render the PNG (skips the slow '
                         'Goertzel/quality recomputation).')
    p.add_argument('--ref-line', nargs=2, action='append', default=None,
                    metavar=('VALUE', 'LABEL'),
                    help='Draw a horizontal reference line at VALUE on the top '
                         '(Goertzel power) panel, annotated with LABEL. Repeatable, '
                         'e.g. --ref-line 100000 "goertzel 60".')
    p.add_argument('--power-ymin', type=float, default=None,
                    help='Y-axis minimum for the top (Goertzel power) panel.')
    p.add_argument('--power-ymax', type=float, default=None,
                    help='Y-axis maximum for the top (Goertzel power) panel, '
                         'e.g. --power-ymax 5e5.')
    args = p.parse_args()

    ref_lines = ([(float(v), lbl) for v, lbl in args.ref_line]
                 if args.ref_line else None)
    power_ylim = None
    if args.power_ymax is not None or args.power_ymin is not None:
        power_ylim = (args.power_ymin if args.power_ymin is not None else 0.0,
                      args.power_ymax)

    csv_path = os.path.abspath(args.csv)
    outdir = args.outdir or os.path.dirname(csv_path)
    stem = args.out_stem or os.path.splitext(os.path.basename(csv_path))[0]
    os.makedirs(outdir, exist_ok=True)

    for ch in args.channels:
        suffix = f'ch{ch}_{args.target_freq:g}Hz'
        scale_tag = '_linear' if args.linear else ''
        out_png = os.path.join(outdir, f'{stem}_goertzel_{suffix}{scale_tag}.png')
        out_csv = os.path.join(outdir, f'{stem}_goertzel_{suffix}.csv')

        if args.reuse_csv and os.path.isfile(out_csv):
            validate_goertzel_cache(out_csv, csv_path, {
                'ch': ch, 'fs': args.fs, 'target_freq': args.target_freq,
                'win_sec': args.win_sec, 'step_sec': args.step_sec,
                'sat_uv': 1950.0, 'sat_frac_threshold': 0.12,
                'step_ptp_threshold': 1000.0, 'bp_shift_sec': 1.0,
                'bp_shift_threshold': 80.0,
                'bp_low': pgv.pem.BP_LOW, 'bp_high': pgv.pem.BP_HIGH,
                'quality_params': pgv.get_ibrain_device_eeg_quality_v2_params(),
            })
            print(f'Re-rendering from existing CSV: {out_csv} (ch{ch})')
            pgv._plot_subject_from_csv(
                merged_csv=csv_path,
                in_csv=out_csv,
                out_png=out_png,
                ch=ch,
                fs=args.fs,
                target_freq=args.target_freq,
                smooth_win=args.smooth_win,
                quality_threshold=args.quality_threshold,
                raw_ylim=tuple(args.raw_ylim),
                exclude_hard_artifact=args.exclude_hard_artifact,
                use_db=not args.linear,
                ref_lines=ref_lines,
                power_ylim=power_ylim,
            )
        else:
            print(f'Processing: {csv_path} (ch{ch})')
            pgv._plot_subject(
                merged_csv=csv_path,
                out_png=out_png,
                out_csv=out_csv,
                ch=ch,
                fs=args.fs,
                target_freq=args.target_freq,
                win_sec=args.win_sec,
                step_sec=args.step_sec,
                smooth_win=args.smooth_win,
                quality_threshold=args.quality_threshold,
                raw_ylim=tuple(args.raw_ylim),
                exclude_hard_artifact=args.exclude_hard_artifact,
                sat_uv=1950.0,
                sat_frac_threshold=0.12,
                step_ptp_threshold=1000.0,
                bp_shift_sec=1.0,
                bp_shift_threshold=80.0,
                use_db=not args.linear,
                ref_lines=ref_lines,
                power_ylim=power_ylim,
            )
        print(f'Saved: {out_png}')
        print(f'Saved: {out_csv}')


if __name__ == '__main__':
    main()
