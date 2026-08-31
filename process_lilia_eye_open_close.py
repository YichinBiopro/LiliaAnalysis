#!/usr/bin/env python3
"""Process 2026-07-03-lilia-eye-open-close.csv with TinyUNetV4.

Workflow
--------
1. Load the 8-channel Lilia CSV.
2. Apply a 0.5-45 Hz zero-phase bandpass filter to channels 1-8.
3. Downsample 500 Hz -> 200 Hz to match TinyUNetV4's operating rate.
4. Run tiny_v4_optimized.pth twice:
   - input channels 1-4  -> output channels 1-2
   - input channels 5-8  -> output channels 5-6
5. Save the processed signals and plot the denoised output channels 1-2
    and 5-6 in both time domain and STFT form.
6. Also generate before/after comparison figures for output channels 1-2
    and 5-6, where "before" is the model input (bandpass-filtered + downsampled).

The script keeps the existing project conventions and reuses the model loading
and overlap-add inference helpers from data_analysis.py.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import stft

from lilia.io import bandpass_filter

from data_analysis import (
    BANDPASS_HIGH,
    BANDPASS_LOW,
    DOWNSAMPLED_FS,
    FS,
    downsample_data,
    load_model,
    run_model,
)


DEFAULT_CSV = "2026-07-03-lilia-eye-open-close.csv"
DEFAULT_OUTDIR = "eye_open_close_output"
STFT_FMAX = 50.0
STFT_NPERSEG = 256
STFT_NOVERLAP = 192


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process Lilia eye open/close EEG CSV with TinyUNetV4",
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help="Input Lilia CSV file (default: 2026-07-03-lilia-eye-open-close.csv)",
    )
    parser.add_argument(
        "--outdir",
        default=DEFAULT_OUTDIR,
        help="Output directory for CSV and figures",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=STFT_FMAX,
        help="Maximum frequency shown in the STFT plot (default: 100 Hz)",
    )
    return parser.parse_args()


def load_lilia_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, skiprows=4)
    if df.shape[1] < 9:
        raise ValueError(f"Expected at least 8 channels in {path}, got {df.shape[1] - 1}")

    time_us = df.iloc[:, 0].to_numpy(dtype=np.int64)
    data = df.iloc[:, 1:9].to_numpy(dtype=np.float32)
    return time_us, data


def read_lilia_header_lines(path: str, n_lines: int = 4) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        return [handle.readline().rstrip("\n") for _ in range(n_lines)]


def compute_stft_db(sig: np.ndarray, fs: float, fmax: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f, t, zxx = stft(
        sig.astype(np.float64),
        fs=fs,
        nperseg=STFT_NPERSEG,
        noverlap=STFT_NOVERLAP,
        window="hann",
    )
    mask = f <= fmax
    return f[mask], t, 20.0 * np.log10(np.abs(zxx[mask]) + 1e-8)


def plot_output_channels(time_s: np.ndarray,
                         data_200: np.ndarray,
                         outdir: str,
                         stem: str,
                         fmax: float,
                         ch_offset: int,
                         suffix: str) -> str:
    out_sig = data_200[:, ch_offset:ch_offset + 2]

    stft_panels = [compute_stft_db(out_sig[:, idx], fs=DOWNSAMPLED_FS, fmax=fmax)
                   for idx in range(2)]
    vmin = min(panel[2].min() for panel in stft_panels)
    vmax = max(panel[2].max() for panel in stft_panels)

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    fig.suptitle(
        f"{stem} - TinyUNetV4 output channels {ch_offset + 1}-{ch_offset + 2}\n"
        f"Bandpass {BANDPASS_LOW:.1f}-{BANDPASS_HIGH:.1f} Hz, 500 Hz -> {DOWNSAMPLED_FS} Hz",
        fontsize=12,
    )

    for row in range(2):
        sig = out_sig[:, row]
        f, t, zdb = stft_panels[row]

        ax_td = axes[row, 0]
        ax_td.plot(time_s, sig, color="#1f77b4", lw=0.7)
        ax_td.set_title(f"Output Ch{ch_offset + row + 1} - Time Domain")
        ax_td.set_xlabel("Time (s)")
        ax_td.set_ylabel("Amplitude")
        ax_td.grid(True, alpha=0.25)

        ax_stft = axes[row, 1]
        pcm = ax_stft.pcolormesh(
            t,
            f,
            zdb,
            shading="gouraud",
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
        )
        ax_stft.set_title(f"Output Ch{ch_offset + row + 1} - STFT (0-{fmax:.0f} Hz)")
        ax_stft.set_xlabel("Time (s)")
        ax_stft.set_ylabel("Frequency (Hz)")
        ax_stft.set_ylim(0, fmax)
        fig.colorbar(pcm, ax=ax_stft, label="dB", pad=0.01)

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{stem}_tinyv4_output_{suffix}_time_stft.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_before_after_channels(time_s: np.ndarray,
                               before_200: np.ndarray,
                               after_200: np.ndarray,
                               outdir: str,
                               stem: str,
                               fmax: float,
                               ch_offset: int,
                               suffix: str) -> str:
    before_sig = before_200[:, ch_offset:ch_offset + 2]
    after_sig = after_200[:, ch_offset:ch_offset + 2]

    before_stft = [compute_stft_db(before_sig[:, idx], fs=DOWNSAMPLED_FS, fmax=fmax)
                   for idx in range(2)]
    after_stft = [compute_stft_db(after_sig[:, idx], fs=DOWNSAMPLED_FS, fmax=fmax)
                  for idx in range(2)]
    vmin = min(panel[2].min() for panel in before_stft + after_stft)
    vmax = max(panel[2].max() for panel in before_stft + after_stft)

    fig, axes = plt.subplots(2, 4, figsize=(22, 9), constrained_layout=True)
    fig.suptitle(
        f"{stem} - TinyUNetV4 before/after comparison for channels {ch_offset + 1}-{ch_offset + 2}\n"
        f"Before: bandpass-filtered + downsampled, After: model output, {DOWNSAMPLED_FS} Hz",
        fontsize=12,
    )

    for row in range(2):
        f_b, t_b, zdb_b = before_stft[row]
        f_a, t_a, zdb_a = after_stft[row]

        ax_td_b = axes[row, 0]
        ax_td_a = axes[row, 1]
        ax_stft_b = axes[row, 2]
        ax_stft_a = axes[row, 3]

        ax_td_b.plot(time_s, before_sig[:, row], color="#7f7f7f", lw=0.7)
        ax_td_b.set_title(f"Ch{ch_offset + row + 1} - Before (Time)")
        ax_td_b.set_xlabel("Time (s)")
        ax_td_b.set_ylabel("Amplitude")
        ax_td_b.grid(True, alpha=0.25)

        ax_td_a.plot(time_s, after_sig[:, row], color="#d62728", lw=0.7)
        ax_td_a.set_title(f"Ch{ch_offset + row + 1} - After (Time)")
        ax_td_a.set_xlabel("Time (s)")
        ax_td_a.set_ylabel("Amplitude")
        ax_td_a.grid(True, alpha=0.25)

        pcm_b = ax_stft_b.pcolormesh(
            t_b,
            f_b,
            zdb_b,
            shading="gouraud",
            cmap="Blues",
            vmin=vmin,
            vmax=vmax,
        )
        ax_stft_b.set_title(f"Ch{ch_offset + row + 1} - Before (STFT 0-{fmax:.0f} Hz)")
        ax_stft_b.set_xlabel("Time (s)")
        ax_stft_b.set_ylabel("Frequency (Hz)")
        ax_stft_b.set_ylim(0, fmax)
        fig.colorbar(pcm_b, ax=ax_stft_b, label="dB", pad=0.01)

        pcm_a = ax_stft_a.pcolormesh(
            t_a,
            f_a,
            zdb_a,
            shading="gouraud",
            cmap="Reds",
            vmin=vmin,
            vmax=vmax,
        )
        ax_stft_a.set_title(f"Ch{ch_offset + row + 1} - After (STFT 0-{fmax:.0f} Hz)")
        ax_stft_a.set_xlabel("Time (s)")
        ax_stft_a.set_ylabel("Frequency (Hz)")
        ax_stft_a.set_ylim(0, fmax)
        fig.colorbar(pcm_a, ax=ax_stft_a, label="dB", pad=0.01)

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{stem}_tinyv4_before_after_{suffix}.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outpath


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading: {csv_path}")
    time_us, data_raw = load_lilia_csv(str(csv_path))
    print(f"Loaded {len(data_raw)} samples x {data_raw.shape[1]} channels")

    print(f"Bandpass filtering channels 1-8 with {BANDPASS_LOW:.1f}-{BANDPASS_HIGH:.1f} Hz")
    data_filt = bandpass_filter(data_raw, fs=FS, lo=BANDPASS_LOW, hi=BANDPASS_HIGH)

    time_s = time_us.astype(np.float64) / 1e6
    time_200, data_200 = downsample_data(time_s, data_filt, fs_in=FS, fs_out=DOWNSAMPLED_FS)
    print(f"Downsampled to {len(data_200)} samples at {DOWNSAMPLED_FS} Hz")

    print("Loading TinyUNetV4 model")
    model = load_model()

    print("Running model on channels 1-4 -> output channels 1-2")
    out_12 = run_model(model, data_200[:, :4])
    print("Running model on channels 5-8 -> output channels 5-6")
    out_56 = run_model(model, data_200[:, 4:8])

    processed = np.concatenate([out_12, out_56], axis=1)
    stem = csv_path.stem
    out_csv = os.path.join(args.outdir, f"{stem}_tinyv4_output.csv")
    header_lines = read_lilia_header_lines(str(csv_path))
    with open(out_csv, "w", encoding="utf-8", newline="") as handle:
        for line in header_lines:
            handle.write(f"{line}\n")
        writer = csv.writer(handle)
        writer.writerow(["Time[us]", "value", "value", "value", "value"])
        out_time_us = np.round(time_200[: len(processed)] * 1e6).astype(np.int64)
        for time_us_row, row in zip(out_time_us, processed):
            writer.writerow([int(time_us_row), *[float(v) for v in row]])
    print(f"Saved processed CSV: {out_csv}")

    fig_path_12 = plot_output_channels(
        time_200[: len(processed)], processed, args.outdir, stem, args.fmax, 0, "ch1_2"
    )
    print(f"Saved figure: {fig_path_12}")
    fig_path_56 = plot_output_channels(
        time_200[: len(processed)], processed, args.outdir, stem, args.fmax, 2, "ch5_6"
    )
    print(f"Saved figure: {fig_path_56}")

    compare_path_12 = plot_before_after_channels(
        time_200[: len(processed)],
        data_200[: len(processed)],
        processed,
        args.outdir,
        stem,
        args.fmax,
        0,
        "ch1_2",
    )
    print(f"Saved comparison figure: {compare_path_12}")
    compare_path_56 = plot_before_after_channels(
        time_200[: len(processed)],
        data_200[: len(processed)],
        processed,
        args.outdir,
        stem,
        args.fmax,
        2,
        "ch5_6",
    )
    print(f"Saved comparison figure: {compare_path_56}")


if __name__ == "__main__":
    main()