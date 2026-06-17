"""
check_quality_anomalies.py
===========================
診斷工具：找出 EEG Quality 曲線上的「異常線段」，並與原始時域 (raw time-domain)
資料逐一對照，以判別每段異常的真正成因。

動機
----
在 ``plot_tflite_summary.py`` / ``plot_event_markers.py`` 的品質面板中，常會看到
一些「異常線段」。它們其實來自兩類完全不同的成因，必須分開處理：

  1. **資料斷點造成的連線假影 (gap-bridging artifact)**：``merged.csv`` 是由多個
     錄製檔串接而成，檔與檔之間可能有數十秒甚至數百秒的時間缺口。matplotlib 會
     用一條直線把缺口兩端的品質點連起來，形成一條「斜向長直線」——這並非真實的
     品質變化，而是繪圖把不連續資料硬接起來的假影。

  2. **訊號本身的真實異常**：例如 ADC 飽和削波 (clipping，數值貼在 ±2048 滿格)、
     斷線/平線 (flat-lining，連續樣本完全相同)、或大幅雜訊使品質分數驟降。這些
     是「真的壞掉」的資料段，品質分數低是正確反映。

本工具會逐視窗計算品質分數與原始訊號診斷量，將異常視窗分類標記，並輸出一張
對照圖：上方為（已斷開缺口的）品質時間軸，下方列出最嚴重的數個異常視窗之原始
4-ch 波形，讓使用者一眼看出每段異常到底是「連線假影」還是「真實壞訊號」。

Usage
-----
    python check_quality_anomalies.py [--subject Hardy] [--outdir <dir>]
    python check_quality_anomalies.py --all
"""

from __future__ import annotations

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import numpy as np

from eeg_utils import load_merged_csv
from eeg_quality_v2 import get_eeg_quality_index_v2_parametric
from plot_event_markers import (
    FS, QUALITY_WIN_SEC, QUALITY_PARAMS, QUALITY_THRESHOLD,
    SUBJECTS, IBRAIN_DIR, us_to_local_dt, _series_with_gaps,
)

# ── 偵測門檻（皆可由 CLI 覆寫） ─────────────────────────────────────────────────
RAIL_VALUE     = 2048.0   # 12-bit ADC 滿格值；|x| >= RAIL-1 視為削波 (clipping)
CLIP_FRAC      = 0.01     # 視窗內削波樣本比例 > 1% → 標記 saturation
FLAT_FRAC      = 0.05     # 視窗內「四通道同時零變化」樣本比例 > 5% → 標記 flat-line
GAP_SEC        = 1.0      # 視窗內任一相鄰樣本時間差 > 1s → 視窗橫跨資料斷點
JUMP_DELTA     = 0.40     # 相鄰視窗品質中位數跳動 > 0.40 → 標記 quality jump
MAX_RAW_PANELS = 6        # 對照圖最多顯示幾個原始波形視窗
CH_COLORS      = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


def analyze_quality_windows(time_us: np.ndarray, data: np.ndarray,
                            fs: float = FS, win_sec: float = QUALITY_WIN_SEC,
                            params=QUALITY_PARAMS) -> list[dict]:
    """逐個非重疊視窗計算品質分數 + 原始訊號診斷量。

    對每個視窗回傳：
      s, e        : 原始資料的起訖樣本索引（方便回頭抓原始波形）
      mid_us      : 視窗中點絕對時間
      qmed        : 四通道 overall 品質的中位數
      clip        : 削波樣本比例（任一通道貼在 ±RAIL）
      flat        : 四通道同時零變化的樣本比例（平線/斷線指標）
      gap         : 視窗內最大相鄰樣本時間差（秒）——揭露資料斷點
    """
    win = int(win_sec * fs)
    rows = []
    for s in range(0, len(data) - win + 1, win):
        seg = data[s:s + win]
        t_seg = time_us[s:s + win]
        res = get_eeg_quality_index_v2_parametric(
            seg.T.astype(np.float64), fs=fs, params=params)
        q = np.asarray(res["overall"], dtype=float)
        clip = float(np.mean(np.any(np.abs(seg) >= RAIL_VALUE - 1.0, axis=1)))
        flat = float(np.mean(np.all(np.diff(seg, axis=0) == 0, axis=1)))
        gap  = float(np.max(np.diff(t_seg)) / 1e6) if len(t_seg) > 1 else 0.0
        rows.append(dict(s=int(s), e=int(s + win), mid_us=int(t_seg[len(t_seg) // 2]),
                         qmed=float(np.median(q)), clip=clip, flat=flat, gap=gap))
    return rows


def flag_anomalies(rows: list[dict],
                   q_thresh: float = QUALITY_THRESHOLD,
                   clip_frac: float = CLIP_FRAC,
                   flat_frac: float = FLAT_FRAC,
                   gap_sec: float = GAP_SEC,
                   jump_delta: float = JUMP_DELTA) -> list[dict]:
    """替每個視窗標記異常原因 (reasons) 與嚴重度分數 (severity)。

    嚴重度用於挑選最值得人工檢視的視窗：削波/平線/斷點各自歸一化後相加，再加上
    低於品質門檻的程度。reasons 為人類可讀的字串清單，空清單代表正常視窗。
    """
    qmed = np.array([r["qmed"] for r in rows])
    for i, r in enumerate(rows):
        reasons = []
        if r["gap"] > gap_sec:
            reasons.append(f"time-gap {r['gap']:.0f}s")          # 資料斷點 → 連線假影
        if r["clip"] > clip_frac:
            reasons.append(f"clip {r['clip'] * 100:.0f}%")        # ADC 飽和削波
        if r["flat"] > flat_frac:
            reasons.append(f"flat {r['flat'] * 100:.0f}%")        # 平線/斷線
        if r["qmed"] < q_thresh:
            reasons.append(f"low-Q {r['qmed']:.2f}")              # 品質低落
        if i > 0 and abs(qmed[i] - qmed[i - 1]) > jump_delta:
            reasons.append(f"jump {qmed[i] - qmed[i - 1]:+.2f}")  # 品質驟跳
        r["reasons"] = reasons
        r["severity"] = (min(r["gap"] / 10.0, 1.0) + r["clip"] + r["flat"]
                         + max(0.0, q_thresh - r["qmed"]))
    return rows


def plot_quality_anomaly_report(name: str, info: dict, outdir: str,
                                base_dir: str = None,
                                win_sec: float = QUALITY_WIN_SEC) -> str:
    """產生「品質異常 vs 原始時域」對照圖並輸出 PNG。"""
    if base_dir is None:
        base_dir = IBRAIN_DIR
    merged = os.path.join(base_dir, info["dir"], "merged.csv")
    if not os.path.isfile(merged):
        print(f"  [{name}] merged.csv not found — skipping")
        return ""

    print(f"  [{name}] loading & scoring quality windows…", flush=True)
    time_us, data = load_merged_csv(merged)
    rows = flag_anomalies(analyze_quality_windows(time_us, data, win_sec=win_sec))

    flagged = [r for r in rows if r["reasons"]]
    print(f"  [{name}] {len(flagged)}/{len(rows)} windows flagged as abnormal:")
    for r in flagged:
        ts = us_to_local_dt(r["mid_us"]).strftime("%H:%M:%S")
        print(f"      {ts}  " + "  ·  ".join(r["reasons"]))

    # 挑選最嚴重的數個視窗做原始波形對照
    show = sorted(flagged, key=lambda r: r["severity"], reverse=True)[:MAX_RAW_PANELS]
    show = sorted(show, key=lambda r: r["s"])           # 依時間排序便於閱讀
    n_raw = len(show)

    q_dt  = np.array([us_to_local_dt(r["mid_us"]) for r in rows])
    q_med = np.array([r["qmed"] for r in rows])

    # ── 圖面：上=品質時間軸（缺口斷開），下=原始波形格 ──────────────────────
    n_cols = min(3, max(1, n_raw))
    n_rows_raw = int(np.ceil(n_raw / n_cols)) if n_raw else 0
    fig = plt.figure(figsize=(16, 3.2 + 2.4 * n_rows_raw))
    gs = gridspec.GridSpec(1 + n_rows_raw, n_cols, figure=fig,
                           height_ratios=[2.0] + [2.2] * n_rows_raw,
                           hspace=0.55, wspace=0.22)

    # 上：品質時間軸。用 _series_with_gaps 在大缺口插入 NaN → 不再畫出斜向連線假影。
    ax_q = fig.add_subplot(gs[0, :])
    t_brk, y_brk = _series_with_gaps(q_dt, q_med, gap_sec=win_sec * 2.5)
    ax_q.plot(t_brk, y_brk, color="#1a1a1a", lw=1.3, label="ch-median quality (gap-aware)")
    ax_q.axhline(QUALITY_THRESHOLD, color="k", lw=0.8, ls="--", alpha=0.6,
                 label=f"threshold {QUALITY_THRESHOLD:.2f}")
    # 各類異常以不同標記疊在品質曲線上
    cat_style = {"time-gap": ("v", "#9467bd"), "clip": ("s", "#d62728"),
                 "flat": ("D", "#ff7f0e"), "low-Q": ("o", "#1f77b4"),
                 "jump": ("^", "#2ca02c")}
    seen = set()
    for r in flagged:
        for reason in r["reasons"]:
            cat = reason.split()[0]
            mk, col = cat_style.get(cat, ("x", "#000000"))
            ax_q.plot(us_to_local_dt(r["mid_us"]), r["qmed"], mk, color=col,
                      ms=7, mec="k", mew=0.4,
                      label=cat if cat not in seen else None)
            seen.add(cat)
    # 標出將在下方展示原始波形的視窗
    for j, r in enumerate(show):
        ax_q.annotate(f"#{j + 1}", (us_to_local_dt(r["mid_us"]), r["qmed"]),
                      textcoords="offset points", xytext=(0, 10),
                      ha="center", fontsize=8, fontweight="bold", color="#b00")
    ax_q.set_ylim(0, 1.05)
    ax_q.set_ylabel("EEG Quality")
    ax_q.set_xlabel("Local Time (UTC+8, HH:MM)")
    ax_q.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_q.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 5)))
    ax_q.legend(loc="lower right", fontsize=8, ncol=4, framealpha=0.85)
    ax_q.set_title(f"{name} ({info['sn']}) — EEG quality anomalies vs raw time-domain",
                   fontsize=12, fontweight="bold")
    ax_q.grid(True, alpha=0.2)

    # 下：每個被選視窗的原始 4-ch 波形（含 ±1 視窗的前後文）
    win = int(win_sec * FS)
    for j, r in enumerate(show):
        ax = fig.add_subplot(gs[1 + j // n_cols, j % n_cols])
        ctx = win                                        # 前後各延伸一個視窗作 context
        lo = max(0, r["s"] - ctx)
        hi = min(len(data), r["e"] + ctx)
        seg = data[lo:hi]
        t_rel = (time_us[lo:hi] - time_us[r["s"]]) / 1e6  # 相對視窗起點的秒數
        for ch in range(seg.shape[1]):
            ax.plot(t_rel, seg[:, ch], lw=0.5, color=CH_COLORS[ch % 4],
                    label=f"ch{ch + 1}")
        # 視窗本體範圍 + ADC 滿格參考線
        ax.axvspan(0, win_sec, color="grey", alpha=0.12)
        ax.axhline(RAIL_VALUE, color="r", lw=0.6, ls=":", alpha=0.7)
        ax.axhline(-RAIL_VALUE, color="r", lw=0.6, ls=":", alpha=0.7)
        # 若視窗內有資料斷點，標出斷點位置
        dt_local = np.diff(time_us[r["s"]:r["e"]]) / 1e6
        if dt_local.size and dt_local.max() > GAP_SEC:
            k = int(np.argmax(dt_local))
            gx = (time_us[r["s"] + k] - time_us[r["s"]]) / 1e6
            ax.axvline(gx, color="#9467bd", lw=1.2, ls="--",
                       label=f"gap {dt_local.max():.0f}s")
        ts = us_to_local_dt(r["mid_us"]).strftime("%H:%M:%S")
        ax.set_title(f"#{j + 1}  {ts}\n" + " · ".join(r["reasons"]),
                     fontsize=8.5)
        ax.set_xlabel("Time within window (s)", fontsize=8)
        ax.set_ylabel("µV (raw)", fontsize=8)
        ax.set_ylim(-RAIL_VALUE * 1.1, RAIL_VALUE * 1.1)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(loc="upper right", fontsize=6, ncol=2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{name}_{info['sn']}_quality_anomalies.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"       → {outpath}")
    return outpath


# ── CLI ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="檢查 EEG Quality 異常線段並與原始時域資料對照。")
    p.add_argument("--subject", default="Hardy",
                   help="受試者名稱（預設 Hardy；缺口最多、最具代表性）。")
    p.add_argument("--all", action="store_true", help="對全部 iBrainCenter 受試者執行。")
    p.add_argument("--outdir", default=os.path.join(IBRAIN_DIR, "quality_anomalies"),
                   metavar="DIR", help="PNG 輸出目錄。")
    return p.parse_args()


def main():
    args = parse_args()
    targets = SUBJECTS.items() if args.all else [(args.subject, SUBJECTS[args.subject])]
    print(f"── Quality anomaly check → {args.outdir}")
    for name, info in targets:
        plot_quality_anomaly_report(name, info, args.outdir)
    print("\nDone.")


# ── 可直接執行的範例 (ready-to-run example) ──────────────────────────────────────
if __name__ == "__main__":
    # 範例：檢查 Hardy（其 merged.csv 含 5 個時間斷點，最能示範兩類異常）。
    #   $ python check_quality_anomalies.py --subject Hardy
    #   $ python check_quality_anomalies.py --all
    main()
