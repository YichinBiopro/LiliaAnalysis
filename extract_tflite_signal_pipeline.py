#!/usr/bin/env python3
"""Extract and run the real tiny_v4_optimized.tflite signal pipeline.

This script reproduces the same practical preprocessing/inference chain used
across this repository:

1) Load EEG CSV (4-row header format)
2) Keep first 4 channels (model input requirement)
3) Bandpass filter (0.5-45 Hz, 4th-order zero-phase Butterworth)
4) Resample 500 Hz -> 200 Hz (polyphase anti-aliasing)
5) TFLite windowed inference (400 samples / 2 s, non-overlapping)
6) Per-window RMS normalize -> inference -> de-normalize

Outputs:
- pipeline_report.json : step-by-step metadata
- pipeline_arrays.npz  : intermediate arrays (time axis + signals)
- Optional stage CSVs  : selected stages via --csv-stages
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from lilia.io import read_lilia_frame

from lilia.constants import BP_HIGH, BP_LOW, FS, N_CH, TFLITE_FS, TFLITE_WIN
from lilia.io import bandpass_filter
from lilia.pathing import get_project_root
from lilia.signal import resample_with_time
from lilia.tflite import apply_tflite_with_time


CSV_STAGE_KEYS = [
    "step1_raw500",
    "step2_bp500",
    "step3_bp200",
    "step4_tflite200",
]


def load_lilia_csv_first4(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load 4-row-header EEG CSV and return (time_us, first_4_channels)."""
    df = read_lilia_frame(csv_path)
    if df.shape[1] < 2:
        raise ValueError("CSV has no EEG channels after Time[us] column.")

    time_us = df.iloc[:, 0].values.astype(np.int64)
    data_all = df.iloc[:, 1:].values.astype(np.float32)
    if data_all.shape[1] < N_CH:
        raise ValueError(f"Model requires {N_CH} input channels")
    n_ch = N_CH
    data = data_all[:, :n_ch]
    return time_us, data


def run_pipeline(csv_path: str, model_path: str, max_sec: float | None = None) -> dict:
    time_us_500, data_raw_500 = load_lilia_csv_first4(csv_path)

    if max_sec is not None and max_sec > 0:
        max_n = int(round(max_sec * FS))
        max_n = min(max_n, len(time_us_500))
        time_us_500 = time_us_500[:max_n]
        data_raw_500 = data_raw_500[:max_n]

    # Step 2: bandpass (0.5-45 Hz)
    data_bp_500 = bandpass_filter(data_raw_500, fs=FS, lo=BP_LOW, hi=BP_HIGH, time_us=time_us_500)

    # Step 3: 500 -> 200 Hz
    time_us_200, data_bp_200, segment_ids_200 = resample_with_time(
        time_us_500, data_bp_500, FS, TFLITE_FS, return_segment_ids=True)

    # Step 4: TFLite inference (RMS normalize -> infer -> de-normalize)
    time_us_tfl, data_tfl_200 = apply_tflite_with_time(
        time_us_200, data_bp_200,
        tflite_path=model_path,
        tflite_win=TFLITE_WIN, segment_ids=segment_ids_200,
    )

    duration_500 = float((time_us_500[-1] - time_us_500[0]) / 1e6) if len(time_us_500) > 1 else 0.0
    duration_200 = float((time_us_200[-1] - time_us_200[0]) / 1e6) if len(time_us_200) > 1 else 0.0
    duration_tfl = float((time_us_tfl[-1] - time_us_tfl[0]) / 1e6) if len(time_us_tfl) > 1 else 0.0

    report = {
        "input_csv": os.path.abspath(csv_path),
        "model_path": os.path.abspath(model_path),
        "constants": {
            "FS": FS,
            "TFLITE_FS": TFLITE_FS,
            "TFLITE_WIN": TFLITE_WIN,
            "BP_LOW": BP_LOW,
            "BP_HIGH": BP_HIGH,
            "N_CH_IN": N_CH,
        },
        "steps": [
            {
                "step": 1,
                "name": "load_csv_first4_channels",
                "shape": list(data_raw_500.shape),
                "sample_rate_hz": FS,
                "duration_sec": duration_500,
            },
            {
                "step": 2,
                "name": "bandpass_filter_0p5_45hz_zero_phase",
                "shape": list(data_bp_500.shape),
                "sample_rate_hz": FS,
                "duration_sec": duration_500,
            },
            {
                "step": 3,
                "name": "resample_poly_500_to_200",
                "shape": list(data_bp_200.shape),
                "sample_rate_hz": TFLITE_FS,
                "duration_sec": duration_200,
            },
            {
                "step": 4,
                "name": "tflite_windowed_rmsnorm_infer_denorm",
                "shape": list(data_tfl_200.shape),
                "sample_rate_hz": TFLITE_FS,
                "duration_sec": duration_tfl,
                "notes": {
                    "window_samples": TFLITE_WIN,
                    "window_sec": float(TFLITE_WIN / TFLITE_FS),
                    "n_windows_used": int(len(data_tfl_200) // TFLITE_WIN),
                    "tail_policy": "discard incomplete window at each timestamp segment",
                    "n_samples_discarded_tail": int(len(data_bp_200) - len(data_tfl_200)),
                },
            },
        ],
        "arrays": {
            "time_us_500": "(N,)",
            "data_raw_500": "(N, <=4)",
            "data_bp_500": "(N, <=4)",
            "time_us_200": "(M,)",
            "data_bp_200": "(M, <=4)",
            "time_us_tfl": "(K,)",
            "data_tfl_200": "(K, 2 usually)",
        },
    }

    return {
        "report": report,
        "time_us_500": time_us_500,
        "data_raw_500": data_raw_500,
        "data_bp_500": data_bp_500,
        "time_us_200": time_us_200,
        "data_bp_200": data_bp_200,
        "time_us_tfl": time_us_tfl,
        "data_tfl_200": data_tfl_200,
    }


def _default_model_path() -> str:
    return os.path.join(get_project_root(), "tiny_v4_optimized.tflite")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run and export the actual tiny_v4_optimized.tflite EEG signal pipeline.",
    )
    p.add_argument("csv", help="Input EEG CSV path (4-row-header Lilia format)")
    p.add_argument("--outdir", default="pipeline_step_output", help="Output directory")
    p.add_argument(
        "--model",
        default=_default_model_path(),
        help="Path to tiny_v4_optimized.tflite",
    )
    p.add_argument(
        "--max-sec",
        type=float,
        default=None,
        help="Optional truncation duration in seconds at 500 Hz before processing",
    )
    p.add_argument(
        "--csv-stages",
        nargs="+",
        default=[],
        metavar="STAGE",
        help=(
            "Export selected pipeline stages to CSV. "
            "Choices: step1_raw500 step2_bp500 step3_bp200 step4_tflite200 all"
        ),
    )
    return p


def _resolve_csv_stages(values: list[str]) -> list[str]:
    """Normalize and validate --csv-stages values."""
    if not values:
        return []

    norm = [v.strip().lower() for v in values if v.strip()]
    if not norm:
        return []

    if "all" in norm:
        return CSV_STAGE_KEYS.copy()

    invalid = [v for v in norm if v not in CSV_STAGE_KEYS]
    if invalid:
        raise ValueError(
            "Invalid --csv-stages value(s): "
            + ", ".join(invalid)
            + ". Valid values: "
            + ", ".join(CSV_STAGE_KEYS)
            + ", all"
        )

    # Keep user-specified order while removing duplicates.
    dedup = []
    seen = set()
    for v in norm:
        if v not in seen:
            dedup.append(v)
            seen.add(v)
    return dedup


def _save_stage_csv(outdir: Path, stage_key: str, time_us: np.ndarray, data: np.ndarray) -> Path:
    """Save one stage array as CSV with Time[us] + channel columns."""
    n_ch = int(data.shape[1]) if data.ndim == 2 else 1
    cols = ["Time[us]"] + [f"Ch{i + 1}" for i in range(n_ch)]

    if data.ndim == 1:
        data2 = data[:, np.newaxis]
    else:
        data2 = data

    frame = np.column_stack([time_us, data2])
    df = pd.DataFrame(frame, columns=cols)
    df["Time[us]"] = df["Time[us]"].astype(np.int64)

    out_path = outdir / f"{stage_key}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def main() -> int:
    args = _build_parser().parse_args()

    csv_path = os.path.abspath(args.csv)
    model_path = os.path.abspath(args.model)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"TFLite model not found: {model_path}")

    csv_stages = _resolve_csv_stages(args.csv_stages)

    result = run_pipeline(csv_path, model_path, max_sec=args.max_sec)

    report_path = outdir / "pipeline_report.json"
    npz_path = outdir / "pipeline_arrays.npz"

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(result["report"], f, ensure_ascii=False, indent=2)

    np.savez_compressed(
        npz_path,
        time_us_500=result["time_us_500"],
        data_raw_500=result["data_raw_500"],
        data_bp_500=result["data_bp_500"],
        time_us_200=result["time_us_200"],
        data_bp_200=result["data_bp_200"],
        time_us_tfl=result["time_us_tfl"],
        data_tfl_200=result["data_tfl_200"],
    )

    csv_written = []
    if csv_stages:
        stage_arrays = {
            "step1_raw500": (result["time_us_500"], result["data_raw_500"]),
            "step2_bp500": (result["time_us_500"], result["data_bp_500"]),
            "step3_bp200": (result["time_us_200"], result["data_bp_200"]),
            "step4_tflite200": (result["time_us_tfl"], result["data_tfl_200"]),
        }
        for stage in csv_stages:
            t_arr, d_arr = stage_arrays[stage]
            csv_written.append(_save_stage_csv(outdir, stage, t_arr, d_arr))

    print("Pipeline completed.")
    print(f"- Report : {report_path}")
    print(f"- Arrays : {npz_path}")
    if csv_written:
        print("- CSV stages:")
        for p in csv_written:
            print(f"  * {p}")

    step4 = result["report"]["steps"][3]
    n_win = step4["notes"]["n_windows_used"]
    n_tail = step4["notes"]["n_samples_discarded_tail"]
    print(f"- TFLite windows used: {n_win}")
    print(f"- Tail discarded (< window): {n_tail} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
