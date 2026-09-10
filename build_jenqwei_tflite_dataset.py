#!/usr/bin/env python3
"""Build TFLite-ready EEG dataset CSVs from jenqwei sessions.

Pipeline per input CSV:
1) Load 4-row-header Lilia CSV.
2) Keep first 4 channels (TFLite input requirement).
3) Bandpass filter (default 0.5-45 Hz) at 500 Hz.
4) Downsample to 200 Hz via polyphase resampling.
5) Split each continuous source segment into N equal parts.
6) Trim each part to a multiple of 400 samples (2 s @ 200 Hz).
7) Save each split as CSV.

Inputs are discovered from jenqwei/*/*.csv and files named time_marker.csv
are excluded (case-insensitive).
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np

from lilia.jenqwei_dataset import (
    DatasetPlanError, SegmentMeta, parameters, process_source,
    write_json, write_manifest,
)
from lilia.jenqwei_dataset import save_segment_csv as save_segment_csv, split_bounds as split_bounds
from lilia.signal import resample_with_time


__all__ = ['SegmentMeta', 'save_segment_csv', 'split_bounds', 'process_one_csv',
           'find_input_csvs', 'downsample_with_time', 'parse_args', 'main']

DEFAULT_FS_IN = 500.0
DEFAULT_FS_OUT = 200.0
DEFAULT_BP_LOW = 0.5
DEFAULT_BP_HIGH = 45.0
DEFAULT_N_CH = 4
DEFAULT_TFLITE_WIN = 400


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-glob",
        default="jenqwei/*/*.csv",
        help="Input CSV glob relative to repo root (default: jenqwei/*/*.csv)",
    )
    p.add_argument(
        "--exclude-name",
        default="time_marker.csv",
        help="Filename to exclude case-insensitively (default: time_marker.csv)",
    )
    p.add_argument(
        "--outdir",
        default="jenqwei_tflite_dataset",
        help="Output root directory for split CSVs",
    )
    p.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of equal splits per continuous source segment (default: 5)",
    )
    p.add_argument("--fs-in", type=float, default=DEFAULT_FS_IN)
    p.add_argument("--fs-out", type=float, default=DEFAULT_FS_OUT)
    p.add_argument("--bp-low", type=float, default=DEFAULT_BP_LOW)
    p.add_argument("--bp-high", type=float, default=DEFAULT_BP_HIGH)
    p.add_argument(
        "--n-ch",
        type=int,
        default=DEFAULT_N_CH,
        help="Number of input channels kept from the CSV (default: 4)",
    )
    p.add_argument(
        "--tflite-win",
        type=int,
        default=DEFAULT_TFLITE_WIN,
        help="TFLite window size for trimming (default: 400)",
    )
    p.add_argument(
        "--allow-short-drop",
        action="store_true",
        help="Allow excluding short splits or source segments too short to filter",
    )
    p.add_argument(
        "--one-window-per-file",
        action="store_true",
        help=(
            "If set, each output CSV contains exactly one TFLite window "
            f"({DEFAULT_TFLITE_WIN} samples by default)."
        ),
    )
    return p.parse_args()


def find_input_csvs(input_glob: str, exclude_name: str) -> list[str]:
    paths = sorted(glob.glob(input_glob))
    exclude_lower = exclude_name.lower()
    return [p for p in paths if os.path.basename(p).lower() != exclude_lower]


def downsample_with_time(
    time_us: np.ndarray,
    data: np.ndarray,
    fs_in: float,
    fs_out: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper for the shared microsecond resampler."""
    return resample_with_time(time_us, data, fs_in, fs_out)


def process_one_csv(
    csv_path: str,
    out_root: str,
    n_splits: int,
    fs_in: float,
    fs_out: float,
    bp_low: float,
    bp_high: float,
    n_ch: int,
    tflite_win: int,
    allow_short_drop: bool,
    one_window_per_file: bool,
) -> list[SegmentMeta]:
    return process_source(csv_path, out_root, parameters(
        n_splits, fs_in, fs_out, bp_low, bp_high, n_ch, tflite_win,
        allow_short_drop, one_window_per_file))


def main() -> None:
    args = parse_args()
    params = parameters(**{key: getattr(args, key) for key in parameters()})
    csv_paths = find_input_csvs(args.input_glob, args.exclude_name)
    if not csv_paths:
        raise ValueError(f'No input CSV matched: {args.input_glob}')
    out = Path(args.outdir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory must be new or empty: {out}')
    out.mkdir(parents=True, exist_ok=True)
    audit = dict(schema_version=1, status='running', parameters=params,
                 sources=[str(Path(p).resolve()) for p in csv_paths], completed=[], failed=None)
    all_rows = []
    try:
        for source in csv_paths:
            print(f'[PROCESS] {source}', flush=True)
            rows = process_source(source, out, params)
            all_rows.extend(rows)
            audit['completed'].append(dict(source=str(Path(source).resolve()), fragments=len(rows)))
            print(f'  saved {len(rows)} files, {sum(r.model_window_count for r in rows)} windows', flush=True)
        manifest = write_manifest(out, all_rows, csv_paths, params)
        audit['status'] = 'complete'
    except Exception as exc:
        audit['status'] = 'failed'
        audit['failed'] = dict(error_type=type(exc).__name__, reason=str(exc))
        if isinstance(exc, DatasetPlanError):
            audit['failed']['plan'] = exc.plan
        raise
    finally:
        write_json(out / 'run_audit.json', audit)
    print(f'Manifest: {manifest} ({len(all_rows)} files)')


if __name__ == '__main__':
    main()
