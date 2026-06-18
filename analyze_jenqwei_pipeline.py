"""
analyze_jenqwei_pipeline.py
============================
對 jenqwei/ 資料夾下所有含原始 EEG 資料的 .csv 檔，套用與
``plot_tflite_summary.py`` 完全相同的 data pipeline，並視覺化處理前後的差異。

每份 CSV 輸出一張圖，包含：
  1. **Time Domain**  — 處理前（原始 ADC）vs 處理後（TFLite 重建）
  2. **PSD**          — 處理前 vs 處理後（Welch method）
  3. **STFT**         — 處理前 vs 處理後（短時傅立葉轉換頻譜圖）

「處理前」定義（apples-to-apples，同 plot_tflite_summary.py）：
  帶通濾波後僅做降採樣（500Hz → 200Hz），尚未通過 TFLite 模型。

「處理後」定義：
  帶通濾波 → 降採樣（500Hz → 200Hz） → TFLite 重建（降噪/去假影）。

Pipeline 步驟（下方 PIPELINE_STEPS 常數有文字說明，程式碼中以
"# ── Step N" 標記各步驟的起始位置）：
  Step 1 : Load raw CSV            (load_raw_csv)
  Step 2 : Bandpass filter         (bandpass_filter, 0.5–45 Hz, 4th-order zero-phase)
  Step 3 : Resample 500 Hz → 200 Hz (resample_poly, polyphase anti-alias)
  Step 4 : TFLite reconstruction    (apply_tflite_windowed, tiny_v4_optimized.tflite)

Usage
-----
    python analyze_jenqwei_pipeline.py [--outdir <dir>] [--channel <0-3>]
                                       [--max-sec <seconds>]

預設：
  --outdir   jenqwei_pipeline_output/
  --channel  0  (顯示第一個通道；TFLite 只重建前 4 通道)
  --max-sec  60 (時域圖最多顯示 60 秒，避免長 session 圖太擠)
"""

from __future__ import annotations

import argparse
import os
import textwrap
from math import gcd
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.signal import resample_poly, welch, stft

# ── 重用既有模組（單一事實來源）─────────────────────────────────────────────────
from eeg_utils import bandpass_filter
from plot_event_markers import (
    FS, TFLITE_FS, TFLITE_WIN, TFLITE_MODEL_PATH,
    BP_LOW, BP_HIGH,
    apply_tflite_windowed,
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
    import pandas as pd

    # ── Step 1 ────────────────────────────────────────────────────────────────
    df = pd.read_csv(path, skiprows=4, header=0)
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
    data_filt_500 = bandpass_filter(data_raw, fs=FS, lo=BP_LOW, hi=BP_HIGH)

    # Optional truncation to limit memory usage for long sessions
    if max_samples is not None:
        n = min(len(data_filt_500), max_samples)
        data_filt_500 = data_filt_500[:n]
        time_us = time_us[:n]
        data_raw = data_raw[:n]

    # ── Step 3 : Resample 500 Hz → 200 Hz ────────────────────────────────────
    _g  = gcd(int(TFLITE_FS), int(FS))
    _up = int(TFLITE_FS) // _g   # up-factor   (2)
    _dn = int(FS) // _g          # down-factor  (5)
    pre_data_200 = resample_poly(data_filt_500, _up, _dn, axis=0).astype(np.float32)

    # Interpolated 200 Hz timestamp axis
    t_orig = np.arange(len(data_filt_500), dtype=np.float64)
    t_new  = np.arange(len(pre_data_200),  dtype=np.float64) * (_dn / _up)
    time_us_200 = np.interp(t_new, t_orig, time_us[:len(data_filt_500)]).astype(np.int64)

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


def _time_axis_sec(time_us: np.ndarray) -> np.ndarray:
    """Convert absolute µs timestamps to seconds from start (for display)."""
    return (time_us - time_us[0]) / 1e6


def _plot_time_domain(ax: plt.Axes, result: dict, ch: int,
                      max_display_sec: float = 60.0) -> None:
    # Before: bandpass-filtered + resampled to 200 Hz (Step 3 output)
    # After:  TFLite reconstruction at 200 Hz (Step 4 output)
    # Both on same axis for direct comparison (apples-to-apples).
    pre  = result["pre_data_200"]
    tfl  = result["tfl_data_200"]
    t200 = _time_axis_sec(result["time_us_200"])

    ch_pre = min(ch, pre.shape[1] - 1)
    ch_tfl = min(ch, tfl.shape[1] - 1)

    mask = t200 <= max_display_sec
    n_tfl = len(tfl)
    mask_tfl = (t200[:n_tfl]) <= max_display_sec

    ax.plot(t200[mask], pre[mask, ch_pre],
            lw=0.6, color="#2166ac", alpha=0.8, label="Before", rasterized=True)
    ax.plot(t200[:n_tfl][mask_tfl], tfl[mask_tfl, ch_tfl],
            lw=0.6, color="#d6604d", alpha=0.8, label="After", rasterized=True)
    ax.set_ylabel("ADC Value", fontsize=8)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.6)
    ax.grid(True, alpha=0.2)


def _plot_psd(ax: plt.Axes, result: dict, ch: int) -> None:
    # Welch PSD overlaid: before (Step 3) and after (Step 4) on the same axis.
    # EEG band spans shown as background shading.
    pre = result["pre_data_200"]
    tfl = result["tfl_data_200"]

    ch_pre = min(ch, pre.shape[1] - 1)
    ch_tfl = min(ch, tfl.shape[1] - 1)
    fs = float(TFLITE_FS)

    nperseg = min(int(4 * fs), len(pre))
    f_pre, pxx_pre = welch(pre[:, ch_pre].astype(np.float64), fs=fs,
                           nperseg=nperseg, scaling="density")
    f_tfl, pxx_tfl = welch(tfl[:, ch_tfl].astype(np.float64), fs=fs,
                           nperseg=min(nperseg, len(tfl)), scaling="density")

    ax.semilogy(f_pre[f_pre <= PSD_FMAX], pxx_pre[f_pre <= PSD_FMAX],
                color="#2166ac", lw=1.2, label="Before")
    ax.semilogy(f_tfl[f_tfl <= PSD_FMAX], pxx_tfl[f_tfl <= PSD_FMAX],
                color="#d6604d", lw=1.2, label="After")
    ax.set_xlabel("Frequency (Hz)", fontsize=8)
    ax.set_ylabel("Power (µV²/Hz)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.6)
    ax.grid(True, alpha=0.2, which="both")
    _add_eeg_bands(ax)


def _add_eeg_bands(ax: plt.Axes) -> None:
    # Shade canonical EEG frequency bands (δ/θ/α/β/γ) as background colour.
    bands = [
        ("δ",  0.5,  4.0,  "#a8d8ea"),
        ("θ",  4.0,  8.0,  "#aa96da"),
        ("α",  8.0,  13.0, "#fcbad3"),
        ("β", 13.0,  30.0, "#ffffd2"),
        ("γ", 30.0,  45.0, "#d4f1a1"),
    ]
    for label, lo, hi, color in bands:
        ax.axvspan(lo, hi, color=color, alpha=0.22, zorder=0)
        ax.text((lo + hi) / 2, ax.get_ylim()[0],
                label, ha="center", va="bottom", fontsize=6, color="#555555")


def _plot_stft(ax_before: plt.Axes, ax_after: plt.Axes,
               result: dict, ch: int,
               max_display_sec: float = 60.0) -> None:
    # STFT (short-time Fourier transform) spectrograms before and after,
    # placed side-by-side — colour maps cannot be meaningfully overlaid.
    pre  = result["pre_data_200"]
    tfl  = result["tfl_data_200"]
    t200 = _time_axis_sec(result["time_us_200"])

    ch_pre = min(ch, pre.shape[1] - 1)
    ch_tfl = min(ch, tfl.shape[1] - 1)
    fs = float(TFLITE_FS)

    n_pre = min(len(pre), int(max_display_sec * fs))
    n_tfl = min(len(tfl), int(max_display_sec * fs))

    # Shared colour scale so before/after are directly comparable
    def _compute_stft_db(sig):
        f, t, Zxx = stft(sig.astype(np.float64), fs=fs,
                         nperseg=STFT_NPERSEG, noverlap=STFT_NPERSEG // 2,
                         window="hann")
        freq_mask = f <= PSD_FMAX
        return f[freq_mask], t, 20 * np.log10(np.abs(Zxx[freq_mask]) + 1e-8)

    f_b, t_b, Zdb_b = _compute_stft_db(pre[:n_pre, ch_pre])
    f_a, t_a, Zdb_a = _compute_stft_db(tfl[:n_tfl, ch_tfl])
    vmin = min(Zdb_b.min(), Zdb_a.min())
    vmax = max(Zdb_b.max(), Zdb_a.max())

    def _draw(ax, f, t, Zdb, title):
        pcm = ax.pcolormesh(t, f, Zdb, shading="gouraud",
                            cmap=CMAP_STFT, vmin=vmin, vmax=vmax)
        plt.colorbar(pcm, ax=ax, label="dB", pad=0.02).ax.tick_params(labelsize=6)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Hz", fontsize=8)
        ax.tick_params(labelsize=7)
        for lo in [4, 8, 13, 30]:
            ax.axhline(lo, color="white", lw=0.5, alpha=0.5, ls="--")

    _draw(ax_before, f_b, t_b, Zdb_b, "Before")
    _draw(ax_after,  f_a, t_a, Zdb_a, "After")


# =============================================================================
# 主分析函式：單一 CSV
# =============================================================================

def analyze_csv(csv_path: str, outdir: str, ch: int = 0,
                max_display_sec: float = 60.0) -> str | None:
    """對單一 CSV 執行 pipeline 並輸出比較圖。

    Parameters
    ----------
    csv_path        : 原始 CSV 路徑
    outdir          : 輸出資料夾
    ch              : 要顯示的通道索引（0-based）
    max_display_sec : 時域圖 / STFT 最多顯示的秒數

    Returns
    -------
    輸出 PNG 路徑，或 None（若處理失敗）
    """
    session_name = Path(csv_path).stem
    print(f"\n{'='*60}")
    print(f"  Processing: {session_name}")
    print(f"  File: {csv_path}")

    # ── Step 1 : Load ─────────────────────────────────────────────────────────
    print("  [Step 1] Loading raw CSV …")
    try:
        time_us, data_raw = load_raw_csv(csv_path)
    except Exception as exc:
        print(f"  [ERROR] Failed to load: {exc}")
        return None

    n_samples = len(data_raw)
    duration_s = n_samples / FS
    print(f"           Loaded {n_samples} samples ({duration_s:.1f} s) "
          f"× {data_raw.shape[1]} channels (using first {N_TFLITE_CH})")

    # ── Step 2–4 : Pipeline ──────────────────────────────────────────────────
    print("  [Step 2] Bandpass filter 0.5–45 Hz (4th-order zero-phase Butterworth) …")
    print(f"  [Step 3] Resample {FS:.0f} Hz → {TFLITE_FS:.0f} Hz (resample_poly) …")
    print("  [Step 4] TFLite reconstruction: RMS-normalise → inference → de-normalise …")
    try:
        result = run_pipeline(time_us, data_raw)
    except Exception as exc:
        print(f"  [ERROR] Pipeline failed: {exc}")
        return None

    tfl_shape = result["tfl_data_200"].shape
    print(f"           Pipeline done. TFLite output: {tfl_shape}")

    # Layout:
    #   Row 0 : Time Domain  (single axis, before+after overlaid)
    #   Row 1 : PSD          (single axis, before+after overlaid)
    #   Row 2 : STFT Before | STFT After  (side-by-side; colourmaps can't overlay)
    print("  [Plot] Generating figure …")
    fig = plt.figure(figsize=(12, 12))
    fig.suptitle(f"{session_name} — Ch{ch+1}", fontsize=11, y=0.99)

    gs = gridspec.GridSpec(3, 2,
                           top=0.95, bottom=0.06,
                           hspace=0.40, wspace=0.30,
                           left=0.08, right=0.97)

    ax_td   = fig.add_subplot(gs[0, :])    # spans both columns
    ax_psd  = fig.add_subplot(gs[1, :])    # spans both columns
    ax_st_b = fig.add_subplot(gs[2, 0])
    ax_st_a = fig.add_subplot(gs[2, 1])

    ax_td.set_title("Time Domain", fontsize=9, loc="left")
    ax_psd.set_title("PSD (Welch)", fontsize=9, loc="left")

    _plot_time_domain(ax_td,  result, ch, max_display_sec)
    _plot_psd(ax_psd, result, ch)
    _plot_stft(ax_st_b, ax_st_a, result, ch, max_display_sec)

    # ── 儲存 ─────────────────────────────────────────────────────────────────
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{session_name}_ch{ch+1}_pipeline_comparison.png")
    fig.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Done] Saved → {outpath}")
    return outpath


# =============================================================================
# 入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze jenqwei CSVs with the plot_tflite_summary.py pipeline")
    parser.add_argument("--outdir", default="jenqwei_pipeline_output",
                        help="Output directory for PNG files")
    parser.add_argument("--channels", type=int, nargs="+", default=[0, 1],
                        help="EEG channel indices to display (0-based). Default: 0 1")
    parser.add_argument("--max-sec", type=float, default=60.0,
                        help="Max seconds to display in time domain & STFT (default 60)")
    args = parser.parse_args()

    # 列印 Pipeline 說明
    print("\n" + "="*60)
    print("  EEG Data Pipeline (same as plot_tflite_summary.py)")
    print("="*60)
    for step, name, desc in PIPELINE_STEPS:
        print(f"\n  {step}: {name}")
        for line in textwrap.wrap(desc, width=68):
            print(f"    {line}")

    # 找出 jenqwei/ 下所有 raw CSV（排除 time_marker.csv）
    jenqwei_path = Path(JENQWEI_DIR)
    raw_csvs = [
        str(p) for p in sorted(jenqwei_path.rglob("*.csv"))
        if "time_marker" not in p.name.lower()
    ]

    if not raw_csvs:
        print(f"\n[WARN] No raw CSV found under {JENQWEI_DIR}")
        return

    print(f"\n\nFound {len(raw_csvs)} raw CSV(s) in jenqwei/:")
    for p in raw_csvs:
        print(f"  {p}")

    results = []
    for csv_path in raw_csvs:
        for ch in args.channels:
            out = analyze_csv(csv_path,
                              outdir=args.outdir,
                              ch=ch,
                              max_display_sec=args.max_sec)
            results.append((csv_path, ch, out))

    # 總結
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    ok  = [(p, c, o) for p, c, o in results if o]
    err = [(p, c, o) for p, c, o in results if not o]
    for _, _c, outpath in ok:
        print(f"  ✓  {outpath}")
    for csv_path, ch, _ in err:
        print(f"  ✗  {csv_path}  ch{ch+1}  (failed)")
    print(f"\n  {len(ok)} succeeded, {len(err)} failed.")
    print(f"  Output directory: {os.path.abspath(args.outdir)}\n")


if __name__ == "__main__":
    main()
