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
import json
import os
from pathlib import Path
import uuid

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from lilia.windowing import require_continuous
import numpy as np
from lilia.io import read_lilia_frame
from scipy.signal import stft

from lilia.io import bandpass_filter
from lilia.neural import build_inference_timeline
from lilia.eye_io import load_signal_table, signal_parameters, write_signal_table
from lilia.provenance import file_sha256
from lilia.signal import resample_polyphase

from data_analysis import (
    BANDPASS_HIGH,
    BANDPASS_LOW,
    DOWNSAMPLED_FS,
    FS,
    MODEL_WINDOW,
    MODEL_PATH,
    load_model,
    run_model,
)


DEFAULT_CSV = "2026-07-03-lilia-eye-open-close.csv"
DEFAULT_OUTDIR = "eye_open_close_output"
STFT_FMAX = 50.0
STFT_NPERSEG = 256
STFT_NOVERLAP = 192


def process_segments(time_us: np.ndarray, data_raw: np.ndarray, *, model=None):
    """Return (timeline, eight-channel input, packed output ch1/2/5/6).

    Stage 15 processing adapter; CLI remains guarded until segmented output and
    plotting validation is complete. Filtering and inference are independent for
    every retained source segment. Short segments are recorded in the timeline;
    non-finite input or any failed model group aborts the entire analysis.

    Keep the eye entry's float32 bandpass and polyphase convention. The general
    denoise_with_time adapter adds notch/bandstop and is not equivalent here.
    """
    data = np.asarray(data_raw, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 8 or len(data) != len(time_us):
        raise ValueError('Eye processing needs aligned timestamps and exactly eight channels')
    if not np.isfinite(data).all():
        raise ValueError('Eye processing input contains non-finite samples')
    timeline = build_inference_timeline(time_us, FS, DOWNSAMPLED_FS, MODEL_WINDOW, MODEL_WINDOW // 2)
    if not len(timeline.time_us):
        raise ValueError('No continuous segment contains a complete model window after resampling')
    if model is None:
        model = load_model()
    inputs, outputs = [], []
    for segment in timeline.segments:
        if segment['status'] != 'retained':
            continue
        start, end = segment['raw_start_idx'], segment['raw_end_idx']
        label = f"segment {segment['segment_id']} raw [{start}:{end}]"
        try:
            filtered = bandpass_filter(data[start:end], fs=FS, lo=BANDPASS_LOW, hi=BANDPASS_HIGH)
            before = resample_polyphase(filtered, FS, DOWNSAMPLED_FS)
            if before.shape != (segment['resampled_samples'], 8) or not np.isfinite(before).all():
                raise ValueError('Resampled input does not match inference timestamps')
            groups = []
            for offset in (0, 4):
                try:
                    predicted = np.asarray(run_model(model, before[:, offset:offset + 4],
                                                     window=timeline.model_window,
                                                     hop=timeline.model_hop))
                    if predicted.shape != (len(before), 2) or not np.isfinite(predicted).all():
                        raise ValueError('Model output does not match inference timestamps')
                except Exception as exc:
                    raise ValueError(f'input channels {offset + 1}-{offset + 4}: {exc}') from exc
                groups.append(predicted)
        except Exception as exc:
            raise ValueError(f'Eye processing failed in {label}: {exc}') from exc
        inputs.append(before)
        outputs.append(np.concatenate(groups, axis=1))
    return timeline, np.concatenate(inputs), np.concatenate(outputs)


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
        help="Maximum frequency shown in the STFT plot (default: 50 Hz)",
    )
    return parser.parse_args()


def load_lilia_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    df = read_lilia_frame(path)
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


def run_analysis(args) -> dict:
    """Run the guarded CLI and preserve an audit on success or failure."""
    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem
    out_csv = outdir / f'{stem}_tinyv4_output.csv'
    audit_path = outdir / f'{stem}_analysis_audit.json'
    expected = [out_csv, Path(str(out_csv) + '.meta.json'), audit_path]
    for suffix in ('ch1_2', 'ch5_6'):
        expected.extend([outdir / f'{stem}_tinyv4_output_{suffix}_time_stft.png',
                         outdir / f'{stem}_tinyv4_before_after_{suffix}.png'])
    collisions = [str(path) for path in expected if path.exists()]
    if audit_path.exists():
        audit_path = outdir / f'{stem}_analysis_audit_{uuid.uuid4().hex}.json'
    audit = {'schema_version': 1, 'kind': 'eye_analysis', 'status': 'failed',
             'source_path': str(csv_path.resolve()), 'quality_state': 'disabled',
             'model_path': str(Path(MODEL_PATH).resolve()), 'table_verified': False,
             'continuity_guard': 'enabled_pending_segmented_plot_validation',
             'plotting_status': 'legacy_pending_segment_and_ch5_6_mapping_fix',
             'artifacts': [], 'stage': 'preflight'}

    def record_artifacts(paths):
        for path in paths:
            audit['artifacts'].append({'path': str(Path(path).resolve()), 'sha256': file_sha256(path)})

    try:
        if collisions:
            raise FileExistsError(f'Eye output already exists; use a new output directory: {collisions}')
        if not np.isfinite(args.fmax) or not 0 < args.fmax <= DOWNSAMPLED_FS / 2:
            raise ValueError('fmax must be finite and within (0, model Nyquist]')
        audit['plot_parameters'] = {'fmax': args.fmax}
        audit['stage'] = 'load_source'
        audit['source_id'] = file_sha256(csv_path)
        time_us, data_raw = load_lilia_csv(str(csv_path))
        timeline = build_inference_timeline(time_us, FS, DOWNSAMPLED_FS, MODEL_WINDOW, MODEL_WINDOW // 2)
        audit['inference'] = timeline.metadata()
        audit['inference_status'] = 'planned_not_processed'
        audit['nonfinite_input'] = [
            {'segment_id': segment['segment_id'], 'input_channel': channel + 1,
             'count': int(count)}
            for segment in timeline.segments
            for channel, count in enumerate(np.count_nonzero(
                ~np.isfinite(data_raw[segment['raw_start_idx']:segment['raw_end_idx']]), axis=0))
            if count]
        audit['stage'] = 'continuity_guard'
        require_continuous(time_us, FS, 'process_lilia_eye_open_close.py')
        audit['stage'] = 'model_provenance'
        parameters = signal_parameters()
        audit['parameters'] = parameters
        root = Path(__file__).resolve().parent
        code = {name: file_sha256(root / name) for name in (
            'process_lilia_eye_open_close.py', 'lilia/eye_io.py', 'data_analysis.py',
            'lilia/neural.py', 'lilia/signal.py', 'lilia/io.py', 'lilia/windowing.py')}
        audit['code_sha256'] = code
        audit['stage'] = 'process_segments'
        timeline, before, processed = process_segments(time_us, data_raw)
        audit['inference'] = timeline.metadata()
        audit['inference_status'] = 'completed'
        audit['stage'] = 'write_signal'
        paths = write_signal_table(out_csv, csv_path, timeline, processed, parameters, code)
        record_artifacts(paths)
        audit['stage'] = 'verify_signal'
        load_signal_table(out_csv, csv_path, model_path=MODEL_PATH)
        audit['table_verified'] = True
        print(f'Saved and verified processed CSV: {out_csv}')
        audit['stage'] = 'legacy_plots'
        # Plot migration remains a separate stage-15 task; guard above prevents
        # packed segments being passed to the legacy whole-recording STFT.
        time_s = timeline.time_us.astype(np.float64) / 1e6
        for offset, suffix in ((0, 'ch1_2'), (2, 'ch5_6')):
            record_artifacts([plot_output_channels(time_s, processed, str(outdir), stem,
                                                   args.fmax, offset, suffix)])
            record_artifacts([plot_before_after_channels(time_s, before, processed, str(outdir),
                                                        stem, args.fmax, offset, suffix)])
        if file_sha256(csv_path) != audit['source_id'] or signal_parameters() != parameters:
            raise ValueError('Eye source or model changed during analysis')
        audit.update(status='success', stage='complete', output_samples=len(processed))
    except Exception as exc:
        audit['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        # A collision gets a new audit name, preserving earlier research output.
        with audit_path.open('x', encoding='utf-8') as handle:
            handle.write(json.dumps(audit, indent=2, allow_nan=False) + '\n')
        print(f'Saved analysis audit: {audit_path}')
    return audit


def main() -> None:
    run_analysis(parse_args())


if __name__ == "__main__":
    main()
