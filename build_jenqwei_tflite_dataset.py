#!/usr/bin/env python3
"""Build TFLite-ready EEG dataset CSVs from jenqwei sessions.

Pipeline per input CSV:
1) Load 4-row-header Lilia CSV.
2) Keep first 4 channels (TFLite input requirement).
3) Bandpass filter (default 0.5-45 Hz) at 500 Hz.
4) Downsample to 200 Hz via polyphase resampling.
5) Split each session into N equal parts.
6) Trim each part to a multiple of 400 samples (2 s @ 200 Hz).
7) Save each split as CSV.

Inputs are discovered from jenqwei/*/*.csv and files named time_marker.csv
are excluded (case-insensitive).
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from lilia.io import bandpass_filter, load_merged_csv


DEFAULT_FS_IN = 500.0
DEFAULT_FS_OUT = 200.0
DEFAULT_BP_LOW = 0.5
DEFAULT_BP_HIGH = 45.0
DEFAULT_N_CH = 4
DEFAULT_TFLITE_WIN = 400


@dataclass
class SegmentMeta:
    source_csv: str
    split_index: int
    split_count: int
    window_index_in_split: int
    window_count_in_split: int
    start_idx_200hz: int
    end_idx_200hz: int
    n_samples_200hz: int
    duration_sec: float
    out_csv: str


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
        help="Number of equal splits per input file (default: 5)",
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
        help="Allow dropping splits that become empty after window trimming",
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
    if fs_in == fs_out:
        return time_us.astype(np.int64), data.astype(np.float32)

    up = int(round(fs_out))
    dn = int(round(fs_in))
    if not math.isclose(fs_in, float(dn), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"fs_in must be integer-like. Got {fs_in}")
    if not math.isclose(fs_out, float(up), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"fs_out must be integer-like. Got {fs_out}")

    g = gcd(up, dn)
    up //= g
    dn //= g

    data_ds = resample_poly(data, up, dn, axis=0).astype(np.float32)
    t_orig = np.arange(len(data), dtype=np.float64)
    t_new = np.arange(len(data_ds), dtype=np.float64) * (dn / up)
    time_ds = np.interp(t_new, t_orig, time_us.astype(np.float64)).astype(np.int64)
    return time_ds, data_ds


def split_bounds(n_samples: int, n_splits: int) -> list[tuple[int, int]]:
    # Equal split boundaries with minimal size mismatch across splits.
    bounds: list[tuple[int, int]] = []
    for i in range(n_splits):
        s = (i * n_samples) // n_splits
        e = ((i + 1) * n_samples) // n_splits
        bounds.append((s, e))
    return bounds


def save_segment_csv(
    out_csv: str,
    time_us_seg: np.ndarray,
    data_seg: np.ndarray,
) -> None:
    cols = {"time_us": time_us_seg.astype(np.int64)}
    for ch in range(data_seg.shape[1]):
        cols[f"ch{ch + 1}"] = data_seg[:, ch].astype(np.float32)
    df = pd.DataFrame(cols)
    df.to_csv(out_csv, index=False)


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
    time_us, data = load_merged_csv(csv_path)
    if data.shape[1] < n_ch:
        raise ValueError(
            f"{csv_path} has {data.shape[1]} channels, but --n-ch={n_ch} is required"
        )
    data = data[:, :n_ch].astype(np.float32)

    data_bp = bandpass_filter(data, fs=fs_in, lo=bp_low, hi=bp_high)
    time_200, data_200 = downsample_with_time(time_us, data_bp, fs_in=fs_in, fs_out=fs_out)

    sess_name = Path(csv_path).stem
    sess_outdir = os.path.join(out_root, sess_name)
    os.makedirs(sess_outdir, exist_ok=True)

    metas: list[SegmentMeta] = []
    for split_idx, (s, e) in enumerate(split_bounds(len(data_200), n_splits), start=1):
        seg_time = time_200[s:e]
        seg_data = data_200[s:e]

        keep_n = (len(seg_data) // tflite_win) * tflite_win
        if keep_n <= 0:
            msg = (
                f"Split {split_idx}/{n_splits} of {csv_path} has {len(seg_data)} samples "
                f"(< {tflite_win}); cannot form a TFLite window"
            )
            if allow_short_drop:
                print(f"[WARN] {msg}; dropped")
                continue
            raise ValueError(msg)

        seg_time = seg_time[:keep_n]
        seg_data = seg_data[:keep_n]

        n_windows = keep_n // tflite_win
        if one_window_per_file:
            for w_idx in range(n_windows):
                ws = w_idx * tflite_win
                we = ws + tflite_win
                win_time = seg_time[ws:we]
                win_data = seg_data[ws:we]
                out_csv = os.path.join(
                    sess_outdir,
                    (
                        f"{sess_name}_split{split_idx:02d}-of-{n_splits:02d}_"
                        f"w{w_idx + 1:04d}-of-{n_windows:04d}_{int(fs_out)}Hz_bp.csv"
                    ),
                )
                save_segment_csv(out_csv, win_time, win_data)
                metas.append(
                    SegmentMeta(
                        source_csv=os.path.abspath(csv_path),
                        split_index=split_idx,
                        split_count=n_splits,
                        window_index_in_split=w_idx + 1,
                        window_count_in_split=n_windows,
                        start_idx_200hz=s + ws,
                        end_idx_200hz=s + we,
                        n_samples_200hz=tflite_win,
                        duration_sec=tflite_win / fs_out,
                        out_csv=os.path.abspath(out_csv),
                    )
                )
        else:
            out_csv = os.path.join(
                sess_outdir,
                f"{sess_name}_split{split_idx:02d}-of-{n_splits:02d}_{int(fs_out)}Hz_bp.csv",
            )
            save_segment_csv(out_csv, seg_time, seg_data)
            metas.append(
                SegmentMeta(
                    source_csv=os.path.abspath(csv_path),
                    split_index=split_idx,
                    split_count=n_splits,
                    window_index_in_split=1,
                    window_count_in_split=1,
                    start_idx_200hz=s,
                    end_idx_200hz=s + keep_n,
                    n_samples_200hz=keep_n,
                    duration_sec=keep_n / fs_out,
                    out_csv=os.path.abspath(out_csv),
                )
            )
    return metas


def main() -> None:
    args = parse_args()
    if args.n_splits <= 0:
        raise ValueError("--n-splits must be > 0")
    if args.tflite_win <= 0:
        raise ValueError("--tflite-win must be > 0")
    if args.bp_low <= 0 or args.bp_high <= args.bp_low:
        raise ValueError("Invalid bandpass range. Require 0 < bp_low < bp_high")

    csv_paths = find_input_csvs(args.input_glob, args.exclude_name)
    if not csv_paths:
        print(f"[WARN] No input CSV matched: {args.input_glob}")
        return

    os.makedirs(args.outdir, exist_ok=True)

    all_rows: list[SegmentMeta] = []
    print(f"Found {len(csv_paths)} input CSV(s)")
    for p in csv_paths:
        print(f"  - {p}")

    for p in csv_paths:
        print(f"\n[PROCESS] {p}")
        rows = process_one_csv(
            csv_path=p,
            out_root=args.outdir,
            n_splits=args.n_splits,
            fs_in=args.fs_in,
            fs_out=args.fs_out,
            bp_low=args.bp_low,
            bp_high=args.bp_high,
            n_ch=args.n_ch,
            tflite_win=args.tflite_win,
            allow_short_drop=args.allow_short_drop,
            one_window_per_file=args.one_window_per_file,
        )
        all_rows.extend(rows)
        for row in rows:
            print(
                f"  saved split {row.split_index}/{row.split_count}, "
                f"window {row.window_index_in_split}/{row.window_count_in_split}: "
                f"{row.n_samples_200hz} samples ({row.duration_sec:.1f}s)"
            )

    manifest_path = os.path.join(args.outdir, "manifest.csv")
    pd.DataFrame([r.__dict__ for r in all_rows]).to_csv(manifest_path, index=False)

    print("\n=== Done ===")
    print(f"Total segments: {len(all_rows)}")
    print(f"Manifest: {os.path.abspath(manifest_path)}")
    print(f"Output root: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
