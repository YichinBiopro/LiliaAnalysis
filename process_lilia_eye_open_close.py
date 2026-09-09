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

    Filtering and inference are independent for
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


def plot_coordinates(timeline):
    """Project each source segment onto elapsed time without joining padding.

    STFT centers refer to segment-local sample positions. Interpolate timestamps
    at those positions (including source jitter); extrapolate only padded centers
    at the nominal sample period. Rendering is clipped to the raw segment end.
    """
    hop = STFT_NPERSEG - STFT_NOVERLAP
    records = []
    for segment in timeline.segments:
        if segment['status'] != 'retained':
            continue
        a, b = segment['output_start_idx'], segment['output_end_idx']
        elapsed = (timeline.time_us[a:b] - timeline.source_epoch_us) / 1e6
        positions = np.arange((b - a + hop - 1) // hop + 1) * hop
        centers = np.interp(positions, np.arange(b - a), elapsed)
        beyond = positions > b - a - 1
        centers[beyond] = elapsed[-1] + (positions[beyond] - (b - a - 1)) / timeline.fs_out
        records.append({'segment_id': segment['segment_id'], 'output_start_idx': a,
                        'output_end_idx': b, 'sample_time_s': elapsed,
                        'stft_sample_positions': positions, 'stft_time_s': centers,
                        'clip_start_s': (segment['raw_start_us'] - timeline.source_epoch_us) / 1e6,
                        'clip_end_s': (segment['raw_end_us'] - timeline.source_epoch_us) / 1e6})
    return records


def segmented_stft(timeline, values, fmax):
    values = np.asarray(values)
    if (values.ndim != 2 or values.shape[0] != len(timeline.time_us)
            or not values.shape[1] or not np.isfinite(values).all()):
        raise ValueError('Plot signals must be finite and aligned to the model timeline')
    panels = []
    for record in plot_coordinates(timeline):
        a, b = record['output_start_idx'], record['output_end_idx']
        channels = []
        for col in range(values.shape[1]):
            f, t, db = compute_stft_db(values[a:b, col], timeline.fs_out, fmax)
            if not np.array_equal(np.rint(t * timeline.fs_out).astype(int), record['stft_sample_positions']):
                raise ValueError('STFT centers differ from the source segment mapping')
            channels.append(db)
        panels.append({**record, 'frequency_hz': f, 'db': np.asarray(channels)})
    if not panels:
        raise ValueError('No retained segment to plot')
    return panels


def plot_metadata(timeline, fmax):
    excluded, gaps = [], []
    for i, segment in enumerate(timeline.segments):
        lo = (segment['raw_start_us'] - timeline.source_epoch_us) / 1e6
        hi = (segment['raw_end_us'] - timeline.source_epoch_us) / 1e6
        if segment['status'] == 'excluded':
            excluded.append({'segment_id': segment['segment_id'], 'start_s': lo,
                             'end_s': hi, 'reason': segment['reason']})
        if i:
            prev = timeline.segments[i - 1]
            gaps.append({'start_s': (prev['raw_end_us'] - timeline.source_epoch_us) / 1e6,
                         'end_s': lo})
    return {'time_axis': 'elapsed_from_source_epoch', 'source_epoch_us': timeline.source_epoch_us,
            'xlim_s': [0., (timeline.segments[-1]['raw_end_us'] - timeline.source_epoch_us) / 1e6],
            'source_channels': [1, 2, 5, 6], 'before_columns': [0, 1, 4, 5],
            'after_columns': [0, 1, 2, 3], 'excluded_spans': excluded, 'missing_spans': gaps,
            'stft': {'fs': timeline.fs_out, 'nperseg': STFT_NPERSEG, 'noverlap': STFT_NOVERLAP,
                     'window': 'hann', 'boundary': 'zeros', 'padded': True, 'fmax': fmax,
                     'scale': '20*log10(abs(z)+1e-8)', 'display': 'clip_to_source_segment'},
            'segments': [{k: v.tolist() if isinstance(v, np.ndarray) else v
                          for k, v in record.items() if k != 'sample_time_s'}
                         for record in plot_coordinates(timeline)]}


def _draw_eye_panels(ax_td, ax_stft, timeline, values, panels, col, channel, label,
                     color, cmap, vmin, vmax, metadata):
    from matplotlib.patches import Rectangle

    for panel in panels:
        a, b = panel['output_start_idx'], panel['output_end_idx']
        ax_td.plot(panel['sample_time_s'], values[a:b, col], color=color, lw=.7)
        pcm = ax_stft.pcolormesh(panel['stft_time_s'], panel['frequency_hz'], panel['db'][col],
                                shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax)
        # Keep the legacy padded STFT values; never paint beyond the source span.
        clip = Rectangle((panel['clip_start_s'], 0), panel['clip_end_s'] - panel['clip_start_s'],
                         metadata['stft']['fmax'], transform=ax_stft.transData)
        pcm.set_clip_path(clip)
    for ax in (ax_td, ax_stft):
        for excluded in metadata['excluded_spans']:
            ax.axvspan(excluded['start_s'], excluded['end_s'], facecolor='.9',
                       edgecolor='.6', hatch='///', linewidth=.6)
        for gap in metadata['missing_spans']:
            for boundary in (gap['start_s'], gap['end_s']):
                ax.axvline(boundary, color='.65', lw=.6, linestyle=':')
        ax.set_xlim(metadata['xlim_s'])
        ax.set_xlabel('Elapsed time (s)')
    ax_td.set_title(f'Ch{channel} - {label} (Time)')
    ax_td.set_ylabel('Amplitude')
    ax_td.grid(True, alpha=.25)
    ax_stft.set_title(f'Ch{channel} - {label} (STFT)')
    ax_stft.set_ylabel('Frequency (Hz)')
    ax_stft.set_ylim(0, metadata['stft']['fmax'])
    return pcm


def _plot_eye_channels(timeline, before, after, outdir, stem, fmax, ch_offset, suffix):
    if ch_offset not in (0, 2):
        raise ValueError('Eye plot output offset must be 0 or 2')
    source_channels = [1, 2, 5, 6][ch_offset:ch_offset + 2]
    after = np.asarray(after)
    if after.shape != (len(timeline.time_us), 4):
        raise ValueError('Eye plot needs packed output channels 1/2/5/6')
    output = after[:, ch_offset:ch_offset + 2]
    branches = [('Output', output, '#1f77b4', 'inferno')]
    if before is not None:
        before = np.asarray(before)
        if before.shape != (len(timeline.time_us), 8):
            raise ValueError('Eye comparison needs eight source input channels')
        branches = [('Before', before[:, np.asarray(source_channels) - 1], '#7f7f7f', 'Blues'),
                    ('After', output, '#d62728', 'Reds')]
    panels = [segmented_stft(timeline, branch[1], fmax) for branch in branches]
    vmin = min(panel['db'].min() for branch in panels for panel in branch)
    vmax = max(panel['db'].max() for branch in panels for panel in branch)
    metadata = plot_metadata(timeline, fmax)
    comparison = before is not None
    # Dedicated colorbar columns keep all TD/STFT time axes equally wide.
    ratios = [1, 1, 1, .035, 1, .035] if comparison else [1, 1, .035]
    fig = plt.figure(figsize=(23 if comparison else 16, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, len(ratios), width_ratios=ratios)
    count = len(panels[0])
    fig.suptitle(f'{stem} - TinyUNetV4 {"before/after" if comparison else "output"} '
                 f'channels {source_channels[0]}-{source_channels[1]}\n'
                 f'Bandpass {BANDPASS_LOW:.1f}-{BANDPASS_HIGH:.1f} Hz, '
                 f'{FS} Hz -> {DOWNSAMPLED_FS} Hz; {count} retained segments; '
                 'gaps blank, excluded short segments hatched', fontsize=12)
    try:
        for row, channel in enumerate(source_channels):
            for branch_idx, (label, values, color, cmap) in enumerate(branches):
                td_col, stft_col, bar_col = ((branch_idx, 2 + 2 * branch_idx, 3 + 2 * branch_idx)
                                            if comparison else (0, 1, 2))
                ax_td, ax_stft = fig.add_subplot(grid[row, td_col]), fig.add_subplot(grid[row, stft_col])
                pcm = _draw_eye_panels(ax_td, ax_stft, timeline, values, panels[branch_idx], row,
                                       channel, label, color, cmap, vmin, vmax, metadata)
                fig.colorbar(pcm, cax=fig.add_subplot(grid[row, bar_col]), label='dB')
        os.makedirs(outdir, exist_ok=True)
        name = f'{stem}_tinyv4_before_after_{suffix}.png' if comparison else f'{stem}_tinyv4_output_{suffix}_time_stft.png'
        path = os.path.join(outdir, name)
        fig.savefig(path, dpi=150, bbox_inches='tight')
    finally:
        plt.close(fig)
    return path


def plot_output_channels(timeline, data_200, outdir, stem, fmax, ch_offset, suffix):
    return _plot_eye_channels(timeline, None, data_200, outdir, stem, fmax, ch_offset, suffix)


def plot_before_after_channels(timeline, before_200, after_200, outdir, stem, fmax, ch_offset, suffix):
    return _plot_eye_channels(timeline, before_200, after_200, outdir, stem, fmax, ch_offset, suffix)


def run_analysis(args) -> dict:
    """Run the segmented CLI and preserve an audit on success or failure."""
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
             'continuity_guard': 'not_required_segmented_pipeline',
             'plotting_status': 'not_started',
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
        audit['stage'] = 'segmented_plots'
        audit['plot_timeline'] = plot_metadata(timeline, args.fmax)
        for offset, suffix in ((0, 'ch1_2'), (2, 'ch5_6')):
            record_artifacts([plot_output_channels(timeline, processed, str(outdir), stem,
                                                   args.fmax, offset, suffix)])
            record_artifacts([plot_before_after_channels(timeline, before, processed, str(outdir),
                                                        stem, args.fmax, offset, suffix)])
        if file_sha256(csv_path) != audit['source_id'] or signal_parameters() != parameters:
            raise ValueError('Eye source or model changed during analysis')
        audit.update(status='success', stage='complete', output_samples=len(processed),
                     plotting_status='complete')
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
