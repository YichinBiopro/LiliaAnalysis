"""Analyze Jenqwei sources with independent Before/After timelines.

Before: first four raw channels, float32 0.5–45 Hz bandpass, 500→200 Hz,
all resampled samples in retained source segments. After: source ch1/2 only,
TFLite reconstruction with nonoverlapping 400-sample windows and per-segment
trimming. Display channels 2/3 (zero-based) show Before only.

Usage: python analyze_jenqwei_pipeline.py [--csv source.csv ...]
       [--outdir new_directory] [--channels 0 1] [--max-sec 60]
Without --csv, read non-marker CSV files recursively from jenqwei/.
"""

from __future__ import annotations

import argparse
import json
import uuid
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from lilia.windowing import continuous_slices, require_continuous
import numpy as np
from lilia.signal import resample_with_time
from lilia.tflite import build_tflite_timeline
from lilia.jenqwei_io import load_signal_table, signal_parameters, write_signal_tables
from lilia.jenqwei_plot import compute_panels, plot_metadata, plot_result, validate_display
from lilia.provenance import file_sha256

# ── 重用既有模組（單一事實來源）─────────────────────────────────────────────────
from lilia.io import bandpass_filter, read_lilia_frame
from plot_event_markers import (
    FS, TFLITE_FS, TFLITE_WIN,
    BP_LOW, BP_HIGH,
    apply_tflite_windowed, TFLITE_MODEL_PATH,
)

# =============================================================================
# Pipeline 步驟說明（明確列出，方便閱讀者對照程式碼中的 "# ── Step N" 標記）
# =============================================================================
PIPELINE_STEPS = [
    ("Step 1", "Load raw CSV",
     "Read 4-row-header Lilia CSV; select first 4 channels (TFLite input requirement). "
     "Preserve Time[us] timestamp column for downstream alignment."),
    ("Step 2", f"Bandpass filter  {BP_LOW}–{BP_HIGH} Hz",
     f"Apply 4th-order zero-phase Butterworth bandpass filter ({BP_LOW}–{BP_HIGH} Hz) "
     "column-wise at 500 Hz. Removes DC drift, line noise, and high-frequency EMG artefacts."),
    ("Step 3", f"Resample {int(FS)} Hz -> {int(TFLITE_FS)} Hz",
     "Polyphase anti-aliasing downsample via resample_poly (up=2, down=5) to match "
     f"TFLite model input rate ({int(TFLITE_FS)} Hz). "
     "This resampled-only signal is the apples-to-apples 'Before' baseline."),
    ("Step 4", "TFLite reconstruction  (RMS-normalise → inference → de-normalise)",
     f"For each non-overlapping {TFLITE_WIN}-sample (2 s) window: "
     "(a) RMS-normalise input (divide by per-window RMS + 1e-8 epsilon); "
     "(b) run tiny_v4_optimized.tflite for denoising / artefact removal; "
     "(c) de-normalise output by multiplying the saved RMS scalar to restore "
     "original EEG amplitude scale. Produces the final 'After' signal."),
]

# =============================================================================
# 其他常數
# =============================================================================
JENQWEI_DIR = os.path.join(os.path.dirname(__file__), "jenqwei")
N_TFLITE_CH = 4        # TFLite 模型固定使用前 4 通道
PSD_FMAX    = 50.0     # PSD / STFT 顯示頻率上限（Hz）
STFT_NPERSEG = 256     # STFT 每段點數（@200 Hz → 頻率解析度 ≈ 0.78 Hz）
CMAP_STFT   = "viridis"


# =============================================================================
# Step 1  Load raw CSV
# =============================================================================

def load_raw_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """讀取 Lilia 裝置原始 CSV（4-row header），回傳時間軸與前 4 通道資料。

    Lilia CSV 格式：
        Row 0 : Device info
        Row 1 : Amp Gain / Abs Time Offset
        Row 2 : Channels
        Row 3 : Sample Rate per channel
        Row 4+: Time[us], value, value, ...（資料本體）

    Parameters
    ----------
    path : CSV 檔路徑

    Returns
    -------
    time_us : (N,) int64  — 絕對 UTC µs 時間戳
    data    : (N, min(n_ch, N_TFLITE_CH)) float32  — 前 N_TFLITE_CH 通道
    """

    # ── Step 1 ────────────────────────────────────────────────────────────────
    df = read_lilia_frame(path)
    time_us = df.iloc[:, 0].values.astype(np.int64)
    data_all = df.iloc[:, 1:].values.astype(np.float32)

    # 只取前 N_TFLITE_CH 個通道
    n_ch = min(data_all.shape[1], N_TFLITE_CH)
    data = data_all[:, :n_ch]
    return time_us, data


# =============================================================================
# Step 2–4  Data pipeline
# =============================================================================

def run_pipeline(time_us: np.ndarray,
                 data_raw: np.ndarray,
                 max_samples: int | None = None
                 ) -> dict:
    """將原始 EEG 資料跑完完整 pipeline，回傳各階段中間結果。

    Returns
    -------
    dict with keys:
        time_us_500   : (N,)   原始 500 Hz 時間軸（µs）
        data_raw      : (N, 4) 原始 ADC 訊號
        data_filt_500 : (N, 4) 帶通濾波後、仍 500 Hz
        time_us_200   : (M,)   200 Hz 時間軸（µs）
        pre_data_200  : (M, 4) 僅降採樣（Step 3 輸出）→ pipeline 「處理前」基準
        tfl_data_200  : (K, n_out) TFLite 重建後（Step 4 輸出）→ pipeline「處理後」
    """

    # ── Step 2 : Bandpass filter 0.5–45 Hz ───────────────────────────────────
    require_continuous(time_us, FS, 'analyze_jenqwei_pipeline.py')
    data_filt_500 = bandpass_filter(data_raw, fs=FS, lo=BP_LOW, hi=BP_HIGH)

    # Optional truncation to limit memory usage for long sessions
    if max_samples is not None:
        n = min(len(data_filt_500), max_samples)
        data_filt_500 = data_filt_500[:n]
        time_us = time_us[:n]
        data_raw = data_raw[:n]

    # ── Step 3 : Resample 500 Hz → 200 Hz ────────────────────────────────────
    time_us_200, pre_data_200 = resample_with_time(
        time_us[:len(data_filt_500)], data_filt_500, FS, TFLITE_FS)

    # ── Step 4 : TFLite reconstruction (RMS-normalise → inference → de-normalise)
    # apply_tflite_windowed already performs all three sub-steps internally:
    #   (a) per-window RMS normalisation  (seg / seg_rms)
    #   (b) model inference
    #   (c) amplitude de-normalisation    (pred * seg_rms)
    tfl_data_200 = apply_tflite_windowed(pre_data_200)   # (K, n_out) float32

    return {
        "time_us_500"   : time_us,
        "data_raw"      : data_raw,
        "data_filt_500" : data_filt_500,
        "time_us_200"   : time_us_200,
        "pre_data_200"  : pre_data_200,
        "tfl_data_200"  : tfl_data_200,
    }


def process_segments(time_us: np.ndarray, data_raw: np.ndarray,
                     max_samples: int | None = None) -> dict:
    """Process separate source segments; the legacy run_pipeline remains guarded.

    Keep all Before samples for retained segments and trim only After to complete
    model windows. Raw, Before and After have independent packed index spaces;
    explicit maps refer to original source positions, including fractional
    resampling positions beyond the final raw sample center. Model-short segments
    are excluded from all three packed arrays and recorded in the timeline.

    Preserve legacy max_samples semantics: filter each full source segment first,
    then truncate the raw prefix and resample. No filtering crosses source gaps.
    """
    t = np.asarray(time_us)
    data = np.asarray(data_raw, dtype=np.float32)
    if data.ndim != 2 or data.shape != (len(t), N_TFLITE_CH):
        raise ValueError('Jenqwei processing needs aligned timestamps and exactly four channels')
    if not np.isfinite(data).all():
        raise ValueError('Jenqwei input contains non-finite samples, including excluded segments')
    if max_samples is not None and (isinstance(max_samples, bool)
                                   or not isinstance(max_samples, int) or max_samples <= 0):
        raise ValueError('max_samples must be a positive integer or None')
    # Validate the complete source, including the part outside a requested prefix.
    full_timeline = build_tflite_timeline(t, FS, TFLITE_FS, TFLITE_WIN)
    n = len(t) if max_samples is None else min(len(t), max_samples)
    timeline = (full_timeline if n == len(t)
                else build_tflite_timeline(t[:n], FS, TFLITE_FS, TFLITE_WIN))
    if not len(timeline.time_us):
        raise ValueError('No source segment contains a complete TFLite window after resampling')
    source_slices = continuous_slices(t, FS)
    filtered_parts, raw_indices, before_parts, before_times = [], [], [], []
    before_groups, before_raw_positions, model_to_before, outputs, records = [], [], [], [], []
    raw_offset = before_offset = 0
    for segment in timeline.segments:
        record = dict(segment)
        context = source_slices[segment['segment_id']]
        record.update(filter_context_start_idx=context.start, filter_context_end_idx=context.stop)
        records.append(record)
        if segment['status'] != 'retained':
            continue
        a, b = segment['raw_start_idx'], segment['raw_end_idx']
        try:
            filtered = bandpass_filter(data[context], fs=FS, lo=BP_LOW, hi=BP_HIGH)[:b - a]
            pre_t, before = resample_with_time(t[a:b], filtered, FS, TFLITE_FS)
            if before.shape != (segment['resampled_samples'], 4) or not np.isfinite(before).all():
                raise ValueError('Resampled Before does not match source segment')
            after = np.asarray(apply_tflite_windowed(before))
            keep = segment['retained_samples']
            if after.shape != (keep, 2) or not np.isfinite(after).all():
                raise ValueError('Model output does not match complete TFLite windows')
            oa, ob = segment['output_start_idx'], segment['output_end_idx']
            if not np.array_equal(pre_t[:keep], timeline.time_us[oa:ob]):
                raise ValueError('Model timestamps differ from retained Before timestamps')
        except Exception as exc:
            raise ValueError(f"Jenqwei segment {segment['segment_id']} raw [{a}:{b}] failed: {exc}") from exc
        record.update(packed_raw_start_idx=raw_offset, packed_raw_end_idx=raw_offset + b - a,
                      before_start_idx=before_offset, before_end_idx=before_offset + len(before))
        filtered_parts.append(filtered)
        raw_indices.append(np.arange(a, b, dtype=np.int64))
        before_parts.append(before)
        before_times.append(pre_t)
        before_groups.append(np.full(len(before), segment['segment_id'], dtype=np.int64))
        before_raw_positions.append(a + np.arange(len(before)) * segment['ratio_down'] / segment['ratio_up'])
        model_to_before.append(before_offset + np.arange(keep, dtype=np.int64))
        outputs.append(after)
        raw_offset += b - a
        before_offset += len(before)
    raw_idx = np.concatenate(raw_indices)
    before_positions = np.concatenate(before_raw_positions)
    after_map = np.concatenate(model_to_before)
    return {'time_us_500': t[raw_idx], 'data_raw': data[raw_idx],
            'data_filt_500': np.concatenate(filtered_parts), 'raw_sample_idx': raw_idx,
            'time_us_200': np.concatenate(before_times), 'pre_data_200': np.concatenate(before_parts),
            'before_segment_ids': np.concatenate(before_groups),
            'before_raw_fractional_idx': before_positions,
            'tfl_time_us_200': timeline.time_us, 'tfl_data_200': np.concatenate(outputs),
            'model_to_before_idx': after_map, 'model_raw_fractional_idx': before_positions[after_map],
            'timeline': timeline, 'segments': records, 'full_source_samples': len(t),
            'used_source_samples': n, 'max_samples': max_samples, 'quality_state': 'disabled'}



def analyze_recording(csv_path, outdir, channels=(0, 1), max_display_sec=60.):
    """Infer each source once; publish tables/plots with a success or failure audit."""
    source, directory = Path(csv_path), Path(outdir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    audit_path = directory / f'{stem}_analysis_audit.json'
    previous_audit = audit_path.exists()
    if previous_audit:
        audit_path = directory / f'{stem}_analysis_audit_{uuid.uuid4().hex}.json'
    audit = {'schema_version': 1, 'kind': 'jenqwei_analysis', 'status': 'failed', 'stage': 'preflight',
             'source_path': str(source.resolve()), 'model_path': str(Path(TFLITE_MODEL_PATH).resolve()),
             'quality_state': 'disabled', 'continuity_guard': 'segmented_main; legacy_run_pipeline_guard_retained',
             'inference_status': 'not_started', 'tables_verified': False, 'plotting_status': 'not_started',
             'artifacts': [], 'requested_channels': list(channels), 'max_display_sec': max_display_sec}
    expected = []
    try:
        validate_display(list(channels), max_display_sec)
        for branch in ('before', 'after'):
            path = directory / f'{stem}_{branch}.csv'
            expected.extend([path, Path(str(path) + '.meta.json')])
        expected.extend(directory / f'{stem}_ch{ch + 1}_pipeline_comparison.png' for ch in channels)
        if previous_audit or any(path.exists() for path in expected):
            raise FileExistsError('Jenqwei output exists; use a new output directory')
        audit['stage'] = 'load_source'
        audit['source_id'] = file_sha256(source)
        t, raw = load_raw_csv(str(source))
        timeline = build_tflite_timeline(t, FS, TFLITE_FS, TFLITE_WIN)
        audit['inference'] = timeline.metadata()
        audit['inference_status'] = 'planned_not_processed'
        audit['nonfinite_input'] = [{'segment_id': s['segment_id'], 'input_channel': ch + 1, 'count': int(count)}
            for s in timeline.segments for ch, count in enumerate(np.count_nonzero(
                ~np.isfinite(raw[s['raw_start_idx']:s['raw_end_idx']]), axis=0)) if count]
        audit['stage'] = 'model_provenance'
        params = signal_parameters(TFLITE_MODEL_PATH)
        audit['parameters'] = params
        root = Path(__file__).resolve().parent
        code = {name: file_sha256(root / name) for name in (
            'analyze_jenqwei_pipeline.py', 'lilia/jenqwei_io.py', 'lilia/jenqwei_plot.py',
            'lilia/tflite.py', 'lilia/signal.py', 'lilia/io.py', 'lilia/windowing.py', 'plot_event_markers.py')}
        audit['code_sha256'] = code
        audit['stage'] = 'process_segments'
        result = process_segments(t, raw)
        audit.update(inference_status='completed', inference=result['timeline'].metadata(), segments=result['segments'])
        audit['stage'] = 'write_signal'
        written = write_signal_tables(directory, source, result, TFLITE_MODEL_PATH, code)
        audit['stage'] = 'verify_signal'
        for path in written:
            if path.suffix == '.csv':
                load_signal_table(path, source, model_path=TFLITE_MODEL_PATH)
        audit['tables_verified'] = True
        audit['stage'] = 'segmented_plots'
        audit['plotting_status'] = 'in_progress'
        audit['plot_timeline'] = plot_metadata(result, list(channels), max_display_sec)
        audit['spectral_coverage'] = {}
        for ch in channels:
            panels = compute_panels(result, ch, max_display_sec)
            audit['spectral_coverage'][str(ch)] = [{k: p[k] for k in
                ('segment_id', 'branch', 'stft_samples', 'stft_status', 'clip_start_s', 'clip_end_s')} for p in panels]
            plot_result(result, directory, stem, ch, max_display_sec)
        if file_sha256(source) != audit['source_id'] or signal_parameters(TFLITE_MODEL_PATH) != params:
            raise ValueError('Jenqwei source/model changed during analysis')
        audit.update(status='success', stage='complete', plotting_status='complete',
                     before_samples=len(result['pre_data_200']), after_samples=len(result['tfl_data_200']))
    except Exception as exc:
        audit['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        # Hash partial output as well; failed audit never labels it a complete set.
        if audit['stage'] != 'preflight':
            audit['artifacts'] = [{'path': str(path.resolve()), 'sha256': file_sha256(path)}
                                  for path in expected if path.is_file()]
        # Invalid nonfinite display arguments still need a valid JSON failure audit.
        if not np.isfinite(audit['max_display_sec']):
            audit['max_display_sec'] = repr(audit['max_display_sec'])
        with audit_path.open('x', encoding='utf-8') as handle:
            handle.write(json.dumps(audit, indent=2, allow_nan=False) + '\n')
        print(f'{audit["status"]}: {source} — audit {audit_path}', flush=True)
    return audit


def analyze_csv(csv_path: str, outdir: str, ch: int = 0, max_display_sec: float = 60.) -> str | None:
    """Compatibility single-channel wrapper; failures return None with a saved audit."""
    try:
        analyze_recording(csv_path, outdir, [ch], max_display_sec)
    except Exception as exc:
        print(f'Analysis failed: {exc}', flush=True)
        return None
    return str(Path(outdir) / f'{Path(csv_path).stem}_ch{ch + 1}_pipeline_comparison.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', nargs='+', help='Explicit input CSV files; otherwise scan jenqwei/')
    parser.add_argument('--outdir', default='jenqwei_pipeline_output')
    parser.add_argument('--channels', type=int, nargs='+', default=[0, 1],
                        help='Display 0..3; channels 2/3 have Before only (no corresponding model output)')
    parser.add_argument('--max-sec', type=float, default=60., help='Elapsed display limit for TD/STFT; PSD uses full segments')
    args = parser.parse_args()
    sources = args.csv if args.csv is not None else [str(p) for p in sorted(Path(JENQWEI_DIR).rglob('*.csv'))
                                                    if 'time_marker' not in p.name.lower()]
    if not sources:
        print(f'No raw CSV found under {JENQWEI_DIR}')
        return 1
    failed = 0
    for source in sources:
        try:
            analyze_recording(source, args.outdir, args.channels, args.max_sec)
        except Exception as exc:
            failed += 1
            print(f'FAILED {source}: {exc}', flush=True)
    print(f'{len(sources) - failed} sources succeeded, {failed} failed.', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
