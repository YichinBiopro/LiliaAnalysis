"""
plot_tflite_summary.py
======================
依據 ``plot_event_markers.py`` 中 ``*_session_start.png`` / ``*_pre_event_rest.png``
「Summary」子圖的呈現風格改寫，產生新的分析圖。本腳本聚焦於：

  1. **Summary 趨勢圖**：沿用 Summary 子圖的視覺風格，但繪製的指標改為
     Relax（放鬆）、Calm（平靜）、Flow（心流）、Focus（專注）四項。
  2. **僅繪製 TFLite 處理後的指標**：趨勢圖的資料來源為 ``tiny_v4_optimized.tflite``
     重建（降噪 / 去假影）後的訊號所計算之 qEEG 指標。
  3. **訊號品質前後比較（apples-to-apples）**：TFLite 處理「前」與「後」皆在
     200 Hz 評分——「前」為僅降採樣（未經 TFLite）的訊號、「後」為 TFLite 重建
     訊號，使兩者唯一差別是模型重建本身，而非降採樣，檢視模型是否劣化訊號品質。
  4. **特定模式的基線處理**：在 ``pre_event_rest`` 模式下，「Color Agility Ladder」
     之前緊接著「Agility Ladder」，沒有獨立的事件前靜息段，會使 delta 失效。
     因此「Color Agility Ladder」改用與標準「Agility Ladder」完全相同的基線區間。
  5. **緩衝區 (Buffer Zone)**：以 ``t_buffer = -3.0`` 秒嚴格捨棄觸發點前 3 秒內的資料。
  6. **微分段 (Micro-epoching)**：將潛在的事件前基線區（-15 ~ -3 秒）切成 1 秒微分段。
  7. **盲抽樣與可重現性**：從通過假影/雜訊篩選的乾淨微分段中以「固定亂數種子」隨機抽樣。
  8. **錯誤處理**：若乾淨微分段總時長不足所需基線長度，丟出明確錯誤訊息。

設計上盡量重用 ``plot_event_markers.py`` 既有的常數與函式，避免邏輯重複。

Usage
-----
    python plot_tflite_summary.py [--outdir <dir>] [--seed 42]

預設對 iBrainCenter 所有受試者各輸出一張 PNG。
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from math import gcd
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import resample_poly

# ── 重用既有模組 ────────────────────────────────────────────────────────────────
from lilia.io import load_merged_csv, bandpass_filter
from lilia.qeeg import compute_qeeg_indices
from lilia.quality import get_eeg_quality_index_v2_parametric

# 直接沿用 plot_event_markers 的常數與工具函式（單一事實來源 single source of truth）
from plot_event_markers import (
    FS, TFLITE_FS,
    BP_LOW, BP_HIGH, QEEG_WIN_SEC, QUALITY_WIN_SEC,
    QUALITY_PARAMS, QUALITY_THRESHOLD,
    EVENTS, CONE_STAGES, SUBJECTS, IBRAIN_DIR,
    hhmm_to_us, hhmm_to_dt, us_to_local_dt,
    apply_tflite_windowed, compute_qeeg_windowed,
    _overlay_events, _series_with_gaps,
)

# ── 本腳本專屬設定 ──────────────────────────────────────────────────────────────
# 要繪製的四項指標：依使用者要求的順序 Relax → Calm → Flow → Focus。
# 內部鍵沿用 qeeg_indices 的命名（relaxation/calm/flow/focus）。
SUMMARY_KEYS   = ['relaxation', 'calm', 'flow', 'focus']
SUMMARY_LABELS = {'relaxation': 'Relax', 'calm': 'Calm',
                  'flow': 'Flow', 'focus': 'Focus'}
SUMMARY_COLORS = {'relaxation': '#f58231', 'calm': '#4363d8',
                  'flow': '#3cb44b', 'focus': '#e6194b'}

# 熱圖 Δ 色階半幅（±值）：colorbar 範圍為 −HEATMAP_DELTA_VABS ~ +HEATMAP_DELTA_VABS。
# qEEG 指標本身有界於 −1~+1，故 Δ 理論最大為 ±2，這裡用 ±2 涵蓋完整動態範圍。
HEATMAP_DELTA_VABS = 2.0

# 基線錨點對照表：某些任務缺乏自身的事件前靜息段，改借用其他任務的基線區間。
#   神經科學/實驗設計原因：「Color Agility Ladder」緊接在「Agility Ladder」之後，
#   兩者之間沒有真正的靜息（rest）區段，若硬取其前一段資料，取到的其實是上一個
#   高強度運動任務的尾段，基準（baseline）會被嚴重高估，導致 delta 失真甚至無效。
#   因此改用標準「Agility Ladder」的事件前靜息區間作為共同基線。
BASELINE_ANCHOR = {'Color Agility Ladder': 'Agility Ladder'}

# 基線微分段預設參數（皆可由呼叫端覆寫）
DEFAULT_T_BUFFER      = -3.0    # 緩衝區：觸發點前 3 秒一律捨棄（秒，負值代表觸發前）
DEFAULT_T_SEARCH      = -15.0   # 基線搜尋窗起點：觸發點前 15 秒（秒）
DEFAULT_EPOCH_SEC     = 1.0     # 微分段長度（秒）
DEFAULT_REQUIRED_SEC  = 10.0    # 所需乾淨基線總長度（秒）
DEFAULT_RANDOM_SEED   = 42      # 固定亂數種子，確保可重現

# ── ADC 飽和（削波）偵測 ─────────────────────────────────────────────────────────
# 重要：品質評分跑在「帶通濾波後」的訊號上，但帶通會把貼在 ±2048 滿格的削波
# 平滑掉（實測：64% 削波的視窗在濾波前評分 0.19、濾波後升到 0.42），使真正的
# ADC 飽和被低估甚至漏抓。故在「原始（未濾波）」訊號上另做一道飽和偵測，與品質
# 門檻一起把關：任一時點若任一通道貼在滿格即計入，超過比例門檻的視窗直接判為髒。
RAIL_VALUE   = 2048.0   # 12-bit ADC 滿格值（資料範圍 −2048 ~ +2047）
SAT_FRAC_MAX = 0.02     # 視窗內削波樣本比例 > 2% → 視窗判定為飽和（髒）


def _saturation_frac(seg_raw: np.ndarray) -> float:
    """回傳原始視窗 (n_samples, n_ch) 內貼在 ±ADC 滿格的樣本比例（任一通道）。"""
    if seg_raw.size == 0:
        return 0.0
    return float(np.mean(np.any(np.abs(seg_raw) >= RAIL_VALUE - 1.0, axis=1)))


# ── 訊號處理小工具 ──────────────────────────────────────────────────────────────

def _resample_500_to_200(data: np.ndarray) -> np.ndarray:
    """將 (N, n_ch) 由 FS(500Hz) 多相位重採樣到 TFLITE_FS(200Hz)。

    TFLite 模型在 200 Hz 下訓練/部署，故推論前必須先降採樣，且使用
    ``resample_poly``（多相位濾波）以避免單純抽取造成的混疊 (aliasing)。
    """
    g  = gcd(int(TFLITE_FS), int(FS))
    up = int(TFLITE_FS) // g
    dn = int(FS) // g
    return resample_poly(data, up, dn, axis=0).astype(np.float32)


def _session_tflite(time_us_full: np.ndarray,
                    data_filt_full: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """對整段資料跑 TFLite。

    流程與 plot_event_markers 一致：先降採樣到 200 Hz，再以非重疊
    ``TFLITE_WIN`` 視窗逐段推論並還原振幅，同時以線性內插建立對應的
    絕對時間軸，方便後續對齊事件標記。

    Returns
    -------
    tfl_time_us, tfl_data   : TFLite 重建訊號（後）及其時間軸（200Hz）
    pre200_time_us, pre200  : 僅降採樣、未經 TFLite 的訊號（前）及其時間軸（200Hz）
        兩者皆在 200Hz，使「前 vs 後」的品質比較只差在 TFLite 重建本身，
        而非降採樣（apples-to-apples）。
    """
    g  = gcd(int(TFLITE_FS), int(FS))
    up = int(TFLITE_FS) // g
    dn = int(FS) // g
    data_200 = resample_poly(data_filt_full, up, dn, axis=0).astype(np.float32)

    # 建立 200Hz 的時間軸（以原始 µs 時間做線性內插）
    t_orig = np.arange(len(data_filt_full))
    t_new  = np.arange(len(data_200)) * (dn / up)
    tfl_time_us_full = np.interp(
        t_new, t_orig, time_us_full[:len(data_filt_full)]).astype(np.int64)

    tfl_data = apply_tflite_windowed(data_200)          # (M, n_ch)
    tfl_time_us = tfl_time_us_full[:len(tfl_data)]
    return tfl_time_us, tfl_data, tfl_time_us_full, data_200


def compute_quality_windowed_fs(time_us: np.ndarray, data: np.ndarray,
                                fs: float,
                                win_sec: float = QUALITY_WIN_SEC):
    """以非重疊視窗計算 EEG 品質指標（可指定取樣率 *fs*）。

    plot_event_markers 內建的 ``compute_quality_windowed`` 將 fs 寫死為 500，
    而本腳本要對 TFLite 輸出（200 Hz）也評分，故另外提供可帶 fs 的版本，
    使「處理前(500Hz)」與「處理後(200Hz)」採用一致的演算法與視窗長度，
    才能做公平的品質比較。

    Returns
    -------
    q_dt      : list[datetime] 視窗中點（本地時間）
    q_overall : (n_windows, n_ch) 每通道 overall 品質分數
    """
    win = int(win_sec * fs)
    n   = len(data)
    q_dt, q_overall = [], []
    for start in range(0, n - win + 1, win):
        seg = data[start:start + win]
        res = get_eeg_quality_index_v2_parametric(
            seg.T.astype(np.float64), fs=fs, params=QUALITY_PARAMS)
        q_dt.append(us_to_local_dt(int(time_us[start + win // 2])))
        q_overall.append(res["overall"])
    return q_dt, np.array(q_overall)


# ── 核心：事件前基線微分段建構 ──────────────────────────────────────────────────

def build_baseline_epochs(time_us_full: np.ndarray,
                          data_filt_full: np.ndarray,
                          trigger_us: int,
                          fs: float = FS,
                          t_buffer: float = DEFAULT_T_BUFFER,
                          t_search_start: float = DEFAULT_T_SEARCH,
                          epoch_sec: float = DEFAULT_EPOCH_SEC,
                          required_sec: float = DEFAULT_REQUIRED_SEC,
                          quality_threshold: float = QUALITY_THRESHOLD,
                          random_seed: int = DEFAULT_RANDOM_SEED,
                          label: str = '',
                          data_raw_full: np.ndarray = None) -> tuple[np.ndarray, dict]:
    """建構「乾淨」事件前基線：緩衝區 → 微分段 → 品質篩選 → 盲抽樣。

    神經科學與方法學原理
    --------------------
    * **緩衝區 (Buffer Zone, ``t_buffer``)**：嚴格捨棄觸發點前 ``|t_buffer|`` 秒
      內的資料。受試者在「預期」任務即將開始時，會出現預期焦慮波、關聯性負
      變化 (CNV, Contingent Negative Variation) 以及 α 去同步化
      (alpha desynchronization / event-related desynchronization, ERD)。
      這些是「任務預備」而非「靜息」狀態，若納入基線將系統性地污染基準值
      （例如低估 Relax/Calm），使後續 delta 失真。

    * **微分段 (Micro-epoching)**：將 [``t_search_start``, ``t_buffer``] 區間
      切成 ``epoch_sec`` 秒（預設 1 秒）的非重疊微分段。逐段評分可僅剔除含
      眨眼、肌電 (EMG)、移動假影的片段，而非「整段全取或全棄」，最大化可用
      的乾淨靜息資料。

    * **盲抽樣 + 固定亂數種子 (Blind selection, fixed seed)**：從通過品質
      篩選的乾淨微分段中「隨機」抽取至目標總長度，避免研究者（有意或無意）
      挑選對假設有利的片段所造成的選樣偏差 (selection bias)。固定
      ``random_seed`` 確保任何人重跑都得到完全相同的抽樣結果，符合同行審查
      (peer review) 對科學可重現性 (reproducibility) 的要求。

    Parameters
    ----------
    time_us_full   : (N,) 絕對 UTC µs 時間軸
    data_filt_full : (N, n_ch) 已帶通濾波之 EEG
    trigger_us     : 任務觸發點（UTC µs）——基線錨點
    其餘            : 見模組層級預設常數說明

    Returns
    -------
    baseline_data : (required_samples, n_ch) 隨機抽樣串接後的乾淨基線資料(500Hz)
    meta          : dict，含篩選/抽樣統計（供日誌與重現性紀錄）

    Raises
    ------
    ValueError : 當乾淨微分段總時長 < ``required_sec`` 時，丟出明確的錯誤指示，
                 建議使用者放寬品質門檻或擴大搜尋窗。
    """
    # 搜尋窗（µs）：[trigger + t_search_start, trigger + t_buffer)
    search_lo_us = trigger_us + int(t_search_start * 1e6)
    search_hi_us = trigger_us + int(t_buffer * 1e6)
    hint = (f"    (1) 放寬品質門檻 quality_threshold（目前 {quality_threshold:.2f}）；\n"
            f"    (2) 擴大搜尋窗 t_search_start（目前 {t_search_start:.0f}s，可往更早）；\n"
            f"    (3) 縮短所需基線長度 required_sec（目前 {required_sec:.0f}s）。")
    return _sample_clean_epochs(
        time_us_full, data_filt_full, search_lo_us, search_hi_us,
        fs=fs, epoch_sec=epoch_sec, required_sec=required_sec,
        quality_threshold=quality_threshold, random_seed=random_seed,
        label=label, where=f"事件「{label or '?'}」事件前", hint=hint,
        data_raw_full=data_raw_full)


def _sample_clean_epochs(time_us_full: np.ndarray, data_filt_full: np.ndarray,
                         lo_us: float, hi_us: float, *,
                         fs: float, epoch_sec: float, required_sec: float,
                         quality_threshold: float, random_seed: int,
                         label: str, where: str, hint: str,
                         data_raw_full: np.ndarray = None) -> tuple[np.ndarray, dict]:
    """共用核心：在 [lo_us, hi_us) 範圍內切微分段 → 品質篩選 → 盲抽樣。

    供 ``build_baseline_epochs``（事件前窗）與 ``build_session_baseline``
    （整段 session）共用，避免邏輯重複；兩者差別僅在搜尋範圍。

    若提供 ``data_raw_full``（未濾波原始訊號），會先在原始視窗上做 ADC 飽和
    偵測：帶通濾波會把削波平滑掉而低估飽和，故在此用原始訊號把關，飽和視窗
    直接剔除，再進行品質評分。基線是後續所有 delta 的基準，必須最乾淨。
    """
    epoch_n = int(round(epoch_sec * fs))                 # 每個微分段的取樣點數
    need_ep = int(math.ceil(required_sec / epoch_sec))   # 需要的乾淨微分段數
    in_win  = np.where((time_us_full >= lo_us) & (time_us_full < hi_us))[0]

    clean_segments, clean_starts, n_total_ep, n_saturated = [], [], 0, 0
    if in_win.size >= epoch_n:
        i0, i1 = in_win[0], in_win[-1] + 1
        for s in range(i0, i1 - epoch_n + 1, epoch_n):
            seg = data_filt_full[s:s + epoch_n]          # (epoch_n, n_ch)
            n_total_ep += 1
            # (1) ADC 飽和篩選：在原始（未濾波）視窗上偵測，避免帶通遮蔽削波。
            if data_raw_full is not None:
                if _saturation_frac(data_raw_full[s:s + epoch_n]) > SAT_FRAC_MAX:
                    n_saturated += 1
                    continue
            # (2) 假影/雜訊篩選：以 eeg_quality_v2 評分，通道中位數 >= 門檻才算乾淨
            res = get_eeg_quality_index_v2_parametric(
                seg.T.astype(np.float64), fs=fs, params=QUALITY_PARAMS)
            if float(np.median(res["overall"])) >= quality_threshold:
                clean_segments.append(seg)
                clean_starts.append(int(s))

    n_clean = len(clean_segments)
    if n_clean < need_ep:                                # 錯誤處理：乾淨資料不足
        raise ValueError(
            f"[基線建構失敗] {where}可用之乾淨基線僅 {n_clean * epoch_sec:.0f}s "
            f"（{n_clean}/{n_total_ep} 個 {epoch_sec:.0f}s 微分段通過品質門檻 "
            f"{quality_threshold:.2f}），不足所需的 {required_sec:.0f}s。\n"
            f"  請採取下列其一後重試：\n{hint}")

    # 盲抽樣：固定種子，從乾淨微分段中隨機選取 need_ep 段（可重現）
    rng = np.random.default_rng(random_seed)
    sel = np.sort(rng.choice(n_clean, size=need_ep, replace=False))
    baseline_data = np.concatenate([clean_segments[i] for i in sel], axis=0)
    meta = {
        'label': label,
        'n_total_epochs': n_total_ep,
        'n_saturated_epochs': n_saturated,
        'n_clean_epochs': n_clean,
        'n_selected': int(need_ep),
        'required_sec': required_sec,
        'selected_starts_us': [int(time_us_full[clean_starts[i]]) for i in sel],
        'random_seed': random_seed,
    }
    return baseline_data.astype(np.float32), meta


def build_session_baseline(time_us_full: np.ndarray, data_filt_full: np.ndarray,
                           fs: float = FS,
                           epoch_sec: float = DEFAULT_EPOCH_SEC,
                           required_sec: float = DEFAULT_REQUIRED_SEC,
                           quality_threshold: float = QUALITY_THRESHOLD,
                           random_seed: int = DEFAULT_RANDOM_SEED,
                           data_raw_full: np.ndarray = None) -> tuple[np.ndarray, dict]:
    """建構「整段 session」基線：在整段錄製中隨機抽樣乾淨 1s 微分段。

    與 ``build_baseline_epochs`` 不同，搜尋範圍是「整段錄製」而非事件前窗。
    意義：以全段資料的隨機乾淨樣本作為「session 平均參考」，使 heatmap 的
    Δ 反映「相對於整段平均狀態」的偏離，並讓整條時間軸（含事件之間的過渡段）
    都能著色，大幅減少灰格。同樣採固定亂數種子確保可重現。
    """
    lo_us, hi_us = float(time_us_full[0]), float(time_us_full[-1]) + 1.0
    hint = (f"    (1) 放寬品質門檻 quality_threshold（目前 {quality_threshold:.2f}）；\n"
            f"    (2) 縮短所需基線長度 required_sec（目前 {required_sec:.0f}s）。")
    return _sample_clean_epochs(
        time_us_full, data_filt_full, lo_us, hi_us,
        fs=fs, epoch_sec=epoch_sec, required_sec=required_sec,
        quality_threshold=quality_threshold, random_seed=random_seed,
        label='session', where='整段 session', hint=hint,
        data_raw_full=data_raw_full)


def _tflite_qeeg_reference(baseline_500: np.ndarray) -> dict:
    """將基線資料經 TFLite 處理後，計算每通道的 qEEG 指標基準值。

    為與「TFLite 處理後」的趨勢做公平比較 (apples-to-apples)，基線參考值
    亦須來自相同的 TFLite 處理鏈：降採樣 → TFLite 重建 → qEEG。

    Returns
    -------
    dict : { index_key -> (n_ch,) ndarray }  每通道的基準指標值
    """
    b200 = _resample_500_to_200(baseline_500)
    tfl  = apply_tflite_windowed(b200)                   # (M, n_ch)
    if len(tfl) == 0:
        raise ValueError('基線經 TFLite 後長度為 0，請增加 required_sec（需 ≥ 2s）。')
    n_ch = tfl.shape[1]
    ref = {k: [] for k in SUMMARY_KEYS}
    for ch in range(n_ch):
        r = compute_qeeg_indices(tfl[:, ch].astype(np.float64), fs=TFLITE_FS)
        for k in SUMMARY_KEYS:
            ref[k].append(float(r[k]))
    return {k: np.array(v) for k, v in ref.items()}      # (n_ch,)


# ── 主繪圖函式 ─────────────────────────────────────────────────────────────────

def plot_subject_tflite_summary(name: str, info: dict, outdir: str,
                                base_dir: str = None,
                                t_buffer: float = DEFAULT_T_BUFFER,
                                t_search_start: float = DEFAULT_T_SEARCH,
                                epoch_sec: float = DEFAULT_EPOCH_SEC,
                                required_sec: float = DEFAULT_REQUIRED_SEC,
                                random_seed: int = DEFAULT_RANDOM_SEED,
                                quality_ratio: float = QUALITY_THRESHOLD,
                                heatmap_baseline_mode: str = 'session',
                                on_insufficient: str = 'raise') -> list[str]:
    """對單一受試者產生 TFLite Summary 分析圖。

    圖面（沿用 Summary 子圖風格，但採「小倍數 small multiples」呈現）：
      1. 訊號品質前後比較（TFLite 前 500Hz vs 後 200Hz）。
      2. 每個指標各一條 strip：Relax / Calm / Flow / Focus 的 raw 原始值（填色
         面積 + 基線參考虛線），比「四線疊在一起」更易讀，且各自用滿縱軸。
      3. qEEG Δ heatmap（vs baseline，模式見下）。
      4. 各事件區塊的平均 Δ 長條圖（誤差棒 = 跨通道標準差）。

    Parameters
    ----------
    quality_ratio : EEG 品質分數 (0~1) 的門檻；視窗品質低於此值即視為「無效/無訊號」
        而被遮罩捨棄，同時也作為事件前/整段基線微分段篩選的門檻。調高 → 更嚴格、
        捨棄更多疑似假影資料；調低 → 保留更多但可能納入雜訊。預設沿用 0.5。
    heatmap_baseline_mode : heatmap / bar 的 Δ 基線模式：
        * 'session'   — 以「整段 session 隨機抽樣乾淨微分段」為單一基線，Δ 反映
                        相對於 session 平均狀態的偏離，**整條時間軸都會著色**
                        （灰格僅剩真正低品質的 bin），可大幅減少灰格。
        * 'pre-event' — 沿用「每個事件各自的事件前微分段基線」，僅事件視窗內著色，
                        事件之間維持灰色（聚焦於各任務 vs 其事前靜息的變化）。
        * 'both'      — 兩種模式各輸出一張圖，檔名以模式標記區分。
    on_insufficient : 'raise'（預設，符合嚴格科學要求）→ 基線不足時丟錯並中止；
                      'skip' → 僅警告並略過（事件或 session 模式），方便批次掃描。

    Returns
    -------
    list[str] : 實際輸出的 PNG 路徑（每個模式一張）。
    """
    if base_dir is None:
        base_dir = IBRAIN_DIR
    merged = os.path.join(base_dir, info['dir'], 'merged.csv')
    if not os.path.isfile(merged):
        print(f'  [{name}] merged.csv not found — skipping')
        return ''

    print(f'  [{name}] loading…', flush=True)
    time_us_full, data_full = load_merged_csv(merged)
    data_filt = bandpass_filter(data_full, fs=FS, lo=BP_LOW, hi=BP_HIGH)

    # ── TFLite 處理（整段）→ 同時取得「前(僅降採樣)」與「後(TFLite 重建)」 ────────
    print(f'  [{name}] applying TFLite ({FS}→{TFLITE_FS}Hz)…', flush=True)
    tfl_time_us, tfl_data, pre_time_us, pre_data = _session_tflite(time_us_full, data_filt)

    # ── 訊號品質：TFLite 前 vs 後（兩者皆 200Hz，apples-to-apples） ───────────────
    # 公平性：原本「前」在 500Hz、「後」在 200Hz，差異會混入「降採樣」本身的影響。
    # 改為「前」= 僅降採樣到 200Hz、未經 TFLite 的訊號，使前後唯一差別是 TFLite 重建。
    print(f'  [{name}] quality before TFLite (200Hz, resampled only)…', flush=True)
    qb_dt, qb_overall = compute_quality_windowed_fs(pre_time_us, pre_data, fs=TFLITE_FS)
    qb_dt  = np.array(qb_dt)
    qb_med = np.median(qb_overall, axis=1) if len(qb_overall) else np.array([])

    print(f'  [{name}] quality after TFLite (200Hz)…', flush=True)
    qa_dt, qa_overall = compute_quality_windowed_fs(tfl_time_us, tfl_data, fs=TFLITE_FS)
    qa_dt  = np.array(qa_dt)
    qa_med = np.median(qa_overall, axis=1) if len(qa_overall) else np.array([])

    print(f'  [{name}] computing TFLite qEEG ({QEEG_WIN_SEC:.0f}s windows)…', flush=True)
    qeeg_tfl_dt, qeeg_tfl = compute_qeeg_windowed(tfl_time_us, tfl_data, fs=TFLITE_FS)
    tfl_t_arr = np.array(qeeg_tfl_dt)
    n_win = qeeg_tfl[SUMMARY_KEYS[0]].shape[0] if len(qeeg_tfl_dt) else 0

    # 品質遮罩：TFLite qEEG 視窗與 TFLite 品質視窗皆為 5s@200Hz，可一對一對齊。
    # 門檻採 quality_ratio（可由呼叫端控制）。
    low_after = qa_med < quality_ratio if len(qa_med) else np.zeros(0, bool)
    qual_mask = np.zeros(n_win, dtype=bool)
    m = min(len(low_after), n_win)
    qual_mask[:m] = low_after[:m]

    # ── 建立事件清單（沿用 plot_event_markers 的規則） ─────────────────────────
    evt_list, participating_events = [], []
    for idx, (label, start_hhmm, dur_min, participants) in enumerate(EVENTS):
        participates = (participants is None) or (name in participants)
        start_dt = hhmm_to_dt(start_hhmm)
        end_dt   = start_dt + datetime.timedelta(minutes=dur_min)
        evt_list.append((start_dt, end_dt, label, '#888888', participates))
        if participates:
            participating_events.append((label, start_hhmm, start_dt, end_dt))
    cone_stage_dt = [hhmm_to_dt(t) for t in CONE_STAGES]

    # 要輸出哪些 heatmap 基線模式
    modes = (['session', 'pre-event'] if heatmap_baseline_mode == 'both'
             else [heatmap_baseline_mode])

    # ── 為每個參與事件建立事件前基線（pre-event 模式需要；含 Color Agility Ladder 特例）
    event_baseline_ref = {}    # label -> {index_key: (n_ch,) ref}
    if 'pre-event' in modes:
        print(f'  [{name}] building pre-event micro-epoch baselines '
              f'(buffer {t_buffer:.0f}s, search {t_search_start:.0f}s, '
              f'need {required_sec:.0f}s, seed {random_seed})…', flush=True)
        for label, start_hhmm, start_dt, end_dt in participating_events:
            # 基線錨點：Color Agility Ladder 借用 Agility Ladder 的事件前靜息區間。
            anchor_label = BASELINE_ANCHOR.get(label, label)
            anchor_hhmm  = next((e[1] for e in EVENTS if e[0] == anchor_label), start_hhmm)
            trigger_us   = hhmm_to_us(anchor_hhmm)
            try:
                bl_data, meta = build_baseline_epochs(
                    time_us_full, data_filt, trigger_us,
                    fs=FS, t_buffer=t_buffer, t_search_start=t_search_start,
                    epoch_sec=epoch_sec, required_sec=required_sec,
                    quality_threshold=quality_ratio,
                    random_seed=random_seed, label=label,
                    data_raw_full=data_full)
                event_baseline_ref[label] = _tflite_qeeg_reference(bl_data)
                tag = f'(anchor={anchor_label})' if anchor_label != label else ''
                print(f'      {label:<24s}{tag}  clean {meta["n_clean_epochs"]}/'
                      f'{meta["n_total_epochs"]} ep → sampled {meta["n_selected"]} ep')
            except ValueError as exc:
                if on_insufficient == 'skip':
                    print(f'      [WARN] {exc}')
                    continue
                raise

    # ── 建立整段 session 基線（session 模式需要） ─────────────────────────────
    session_ref = None
    if 'session' in modes:
        print(f'  [{name}] building session baseline '
              f'(blind-sample {required_sec:.0f}s across whole session, '
              f'seed {random_seed})…', flush=True)
        try:
            sb_data, sb_meta = build_session_baseline(
                time_us_full, data_filt, fs=FS, epoch_sec=epoch_sec,
                required_sec=required_sec, quality_threshold=quality_ratio,
                random_seed=random_seed, data_raw_full=data_full)
            session_ref = _tflite_qeeg_reference(sb_data)
            print(f'      session  clean {sb_meta["n_clean_epochs"]}/'
                  f'{sb_meta["n_total_epochs"]} ep → sampled {sb_meta["n_selected"]} ep')
        except ValueError as exc:
            if on_insufficient == 'skip':
                print(f'      [WARN] {exc}')
            else:
                raise
        if session_ref is None:
            modes = [m for m in modes if m != 'session']

    # ── (A) 原始趨勢 (raw absolute index，不減基線) ───────────────────────────
    # 設計考量：基線相減 (Δ) 只在「比較事件 vs 靜息」時有意義；要看受試者「當下
    # 的絕對狀態水準」應直接看原始指標。qEEG 指標本身已是有界比值 (−1~+1)，可
    # 跨時段直接判讀，故 Summary 趨勢改畫原始值，不參照基線。基線僅保留給下方
    # 的 Δ heatmap 與長條圖（兩者本質上是「相對於靜息的變化量」）。
    SMOOTH_WIN = max(1, int(30.0 / QEEG_WIN_SEC))        # 30s 平滑（6×5s）
    smooth_abs = {}
    for k in SUMMARY_KEYS:
        series = np.where(qual_mask, np.nan, np.median(qeeg_tfl[k], axis=1))
        sm = (pd.Series(series)
              .rolling(SMOOTH_WIN, center=True, min_periods=1)
              .mean().to_numpy())
        # 重要：低品質/無訊號視窗一律捨棄。rolling(min_periods=1) 會用鄰近有效點
        # 把空洞補回，導致「無訊號區段」仍畫出指標值；故平滑後再把這些視窗設回
        # NaN，確保無效區段不顯示任何指標。
        sm[qual_mask] = np.nan
        smooth_abs[k] = sm

    good_qual = ~qual_mask

    # ── (B) qEEG Δ heatmap 的 30s bin 絕對值（與基線模式無關，先算一次共用） ────
    HEATMAP_BIN_SEC = 30
    bin_size = max(1, int(HEATMAP_BIN_SEC / QEEG_WIN_SEC))   # = 6 windows
    n_bins = n_win // bin_size
    bin_t = []
    heatmap_abs = np.full((len(SUMMARY_KEYS), n_bins), np.nan)
    for b in range(n_bins):
        sl   = slice(b * bin_size, (b + 1) * bin_size)
        good = good_qual[sl]
        bin_t.append(tfl_t_arr[b * bin_size + bin_size // 2])
        for idx_i, k in enumerate(SUMMARY_KEYS):
            vals = np.median(qeeg_tfl[k][sl], axis=1)        # per-window ch median
            gv   = vals[good]
            if gv.size > 0:
                heatmap_abs[idx_i, b] = float(np.median(gv))
    bin_t_arr = np.array(bin_t)

    # ── 依模式渲染（'session' / 'pre-event'，'both' 則兩張都出） ────────────────
    os.makedirs(outdir, exist_ok=True)
    outpaths = []
    for mode in modes:
        # 該模式的事件視窗清單與基線取得方式
        if mode == 'session':
            # 整段共用單一 session 基線；所有參與事件都納入長條圖
            mode_windows = [(lbl, s, e) for (lbl, _h, s, e) in participating_events]
            baseline_desc = f'session baseline (blind-sample {required_sec:.0f}s)'
            ylab_hm, ylab_bar = 'qEEG Δ\n(vs session)', 'Mean Δ Index\n(vs session)'
        else:  # pre-event
            mode_windows = [(lbl, s, e) for (lbl, _h, s, e) in participating_events
                            if lbl in event_baseline_ref]
            baseline_desc = (f'pre-event baseline (buffer {t_buffer:.0f}s, '
                             f'{required_sec:.0f}s)')
            ylab_hm, ylab_bar = 'qEEG Δ\n(vs pre-event)', 'Mean Δ Index\n(vs pre-event)'

        def _ref_for(label):
            return session_ref if mode == 'session' else event_baseline_ref[label]

        # ── (C) heatmap Δ ───────────────────────────────────────────────────────
        heatmap_delta = np.full_like(heatmap_abs, np.nan)
        if mode == 'session':
            # 整段時間軸都以同一 session 基線相減 → 每個有效 bin 都著色，灰格大減
            for idx_i, k in enumerate(SUMMARY_KEYS):
                ref_scalar = float(np.median(session_ref[k]))
                col = heatmap_abs[idx_i]
                heatmap_delta[idx_i] = np.where(np.isnan(col), np.nan, col - ref_scalar)
        else:
            # 僅事件視窗內以該事件的事件前基線相減；窗外留 NaN(灰)
            for b in range(n_bins):
                bt = bin_t_arr[b]
                for (label, s_dt, e_dt) in mode_windows:
                    if s_dt <= bt < e_dt:
                        ref_ch = _ref_for(label)
                        for idx_i, k in enumerate(SUMMARY_KEYS):
                            if not np.isnan(heatmap_abs[idx_i, b]):
                                heatmap_delta[idx_i, b] = (
                                    heatmap_abs[idx_i, b] - float(np.median(ref_ch[k])))
                        break
        heatmap_delta_ma = np.ma.masked_invalid(heatmap_delta)

        # ── (D) 各事件平均 Δ 長條圖 ──────────────────────────────────────────────
        block_deltas = []                                    # (label, {k:(mean,std)})
        for (label, s_dt, e_dt) in mode_windows:
            ref_ch = _ref_for(label)
            blk_mask = (tfl_t_arr >= s_dt) & (tfl_t_arr < e_dt) & good_qual
            per_index = {}
            for k in SUMMARY_KEYS:
                if blk_mask.sum() > 0:
                    blk_ch = qeeg_tfl[k][blk_mask].mean(axis=0)   # (n_ch,)
                    ch_d   = blk_ch - ref_ch[k]
                    per_index[k] = (float(ch_d.mean()), float(ch_d.std()))
                else:
                    per_index[k] = (0.0, 0.0)
            block_deltas.append((label, per_index))

        # 每個指標的「基線參考水準」(供 strip 的虛線)：session 模式為單一常數；
        # pre-event 模式為各事件視窗內的基線值、窗外 NaN。填色線與虛線之間的距離
        # 即為 Δ，使單一 strip 同時呈現「絕對值」與「相對基線變化」。
        ref_level = {k: np.full(n_win, np.nan) for k in SUMMARY_KEYS}
        if mode == 'session':
            for k in SUMMARY_KEYS:
                ref_level[k][:] = float(np.median(session_ref[k]))
        else:
            for (label, s_dt, e_dt) in mode_windows:
                ref_ch = _ref_for(label)
                evt_mask = (tfl_t_arr >= s_dt) & (tfl_t_arr < e_dt)
                for k in SUMMARY_KEYS:
                    ref_level[k][evt_mask] = float(np.median(ref_ch[k]))

        # ── 圖面組裝（小倍數 small multiples：每個指標一條 strip） ────────────────
        has_hm  = n_bins > 0
        has_bar = bool(block_deltas)
        n_strip = len(SUMMARY_KEYS)
        height_ratios = ([1.6] + [1.0] * n_strip
                         + ([2.2] if has_hm else [])
                         + ([2.6] if has_bar else []))
        fig = plt.figure(figsize=(18, sum(height_ratios) * 1.7 + 2.0))
        gs = gridspec.GridSpec(len(height_ratios), 1, figure=fig,
                               height_ratios=height_ratios, hspace=0.32)
        _r = 0
        ax_qual = fig.add_subplot(gs[_r]); _r += 1
        ax_strips = []
        for _i in range(n_strip):
            ax_strips.append(fig.add_subplot(gs[_r], sharex=ax_qual)); _r += 1
        ax_hm  = fig.add_subplot(gs[_r], sharex=ax_qual) if has_hm else None
        if has_hm:
            _r += 1
        ax_bar = fig.add_subplot(gs[_r]) if has_bar else None

        fig.suptitle(
            f'iBrainCenter — {info["sn"]} ({name})  |  TFLite Summary '
            f'(Relax · Calm · Flow · Focus)\n'
            f'Trend = raw index · Heatmap/Bar = Δ vs {baseline_desc}  |  '
            f'micro-epoch {epoch_sec:.0f}s @ seed {random_seed}',
            fontsize=13, fontweight='bold')

        # 1) 品質前後比較
        # 注意：merged.csv 由多個錄製檔串接而成，檔間可能有數十~數百秒的時間斷點。
        # 直接連線會在斷點兩端畫出「斜向長直線」假影，誤導判讀。故以 _series_with_gaps
        # 在大缺口插入 NaN，讓 matplotlib 自動斷開、不跨缺口連線。
        gap_break = QUALITY_WIN_SEC * 2.5
        if len(qb_dt):
            tb, yb = _series_with_gaps(qb_dt, qb_med, gap_sec=gap_break)
            ax_qual.plot(tb, yb, color='#1a1a1a', lw=1.5,
                         label=f'Before TFLite ({TFLITE_FS}Hz resampled, ch median)')
        if len(qa_dt):
            ta, ya = _series_with_gaps(qa_dt, qa_med, gap_sec=gap_break)
            ax_qual.plot(ta, ya, color='#1f77b4', lw=1.5, ls='--',
                         label=f'After TFLite ({TFLITE_FS}Hz, ch median)')
        ax_qual.axhline(quality_ratio, color='k', lw=0.8, ls=':',
                        alpha=0.6, label=f'quality_ratio {quality_ratio:.2f}')
        _overlay_events(ax_qual, evt_list, cone_stage_dt)
        ax_qual.set_ylim(0, 1.05)
        ax_qual.set_ylabel('EEG Quality\n(before vs after)', fontsize=9)
        ax_qual.set_title('Signal-quality comparison: TFLite processing before vs after',
                          fontsize=10, loc='left')
        ax_qual.legend(loc='lower right', fontsize=8, ncol=3, framealpha=0.85)
        plt.setp(ax_qual.get_xticklabels(), visible=False)

        # 2) TFLite Summary：每個指標一條 strip（小倍數）
        # 填色面積 = raw 原始指標；黑色虛線 = 基線水準（填色與虛線的落差即 Δ）。
        # 同品質面板：用 _series_with_gaps 在大時間斷點斷開；低品質視窗已於 smooth_abs
        # 設為 NaN 而捨棄，故「無訊號區段」自然留白、不畫出指標。
        gap_q = QEEG_WIN_SEC * 2.5
        for si, k in enumerate(SUMMARY_KEYS):
            ax = ax_strips[si]
            tt, yy = _series_with_gaps(tfl_t_arr, smooth_abs[k], gap_sec=gap_q)
            ax.fill_between(tt, 0.0, yy, color=SUMMARY_COLORS[k], alpha=0.30,
                            linewidth=0)
            ax.plot(tt, yy, color=SUMMARY_COLORS[k], lw=1.7)
            rt, rv = _series_with_gaps(tfl_t_arr, ref_level[k], gap_sec=gap_q)
            ax.plot(rt, rv, color='#222222', lw=1.0, ls='--', alpha=0.7)
            ax.axhline(0, color='k', lw=0.4, ls=':', alpha=0.5)
            _overlay_events(ax, evt_list, cone_stage_dt)
            ax.set_ylim(-1.05, 1.05)
            ax.set_yticks([-1, 0, 1])
            ax.tick_params(labelsize=8)
            ax.set_ylabel(SUMMARY_LABELS[k], color=SUMMARY_COLORS[k],
                          fontweight='bold', fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if si == 0:
                ax.set_title('Raw qEEG index per metric (filled = raw · '
                             'dashed = baseline level · gaps = discarded low-quality)',
                             fontsize=10, loc='left')
            plt.setp(ax.get_xticklabels(), visible=False)

        # 3) qEEG Δ heatmap（沿用 plot_event_markers 的 Δ-vs-baseline 熱圖）
        if ax_hm is not None:
            cmap_hm = LinearSegmentedColormap.from_list(
                'OrgPur', ['#5e3c99', '#f7f7f7', '#e66101'])
            cmap_hm.set_bad(color='#aaaaaa')          # 灰 = 無法計算(低品質/窗外)
            bin_t_num = mdates.date2num(bin_t_arr)
            dt_h = (float(np.diff(bin_t_num).mean()) / 2) if n_bins > 1 \
                   else (HEATMAP_BIN_SEC / 86400 / 2)
            t_edges = np.concatenate([[bin_t_num[0] - dt_h],
                                      (bin_t_num[:-1] + bin_t_num[1:]) / 2,
                                      [bin_t_num[-1] + dt_h]])
            y_edges = np.arange(len(SUMMARY_KEYS) + 1) - 0.5
            v_abs = HEATMAP_DELTA_VABS
            pcm = ax_hm.pcolormesh(t_edges, y_edges, heatmap_delta_ma,
                                   cmap=cmap_hm, vmin=-v_abs, vmax=v_abs,
                                   shading='flat')
            cbar = plt.colorbar(pcm, ax=ax_hm, pad=0.012, fraction=0.015)
            cbar.set_label(f'Δ Index  (−{v_abs:.1f} → +{v_abs:.1f})', labelpad=12)
            cbar.ax.tick_params(pad=4)
            for (label, s_dt, e_dt) in mode_windows:
                ax_hm.axvline(mdates.date2num(s_dt), color='#444444',
                              lw=1.0, ls='--', alpha=0.7)
            ax_hm.set_yticks(range(len(SUMMARY_KEYS)))
            ax_hm.set_yticklabels([SUMMARY_LABELS[k] for k in SUMMARY_KEYS], fontsize=9)
            ax_hm.set_ylabel(ylab_hm, fontsize=9)
            ax_hm.xaxis_date()
            ax_hm.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax_hm.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 5)))
            ax_hm.grid(False)
            ax_hm.set_xlabel('Local Time (UTC+8, HH:MM)', fontsize=10)

        # 4) 事件區塊平均 Δ 長條圖
        if ax_bar is not None and block_deltas:
            n_evt = len(block_deltas)
            n_idx = len(SUMMARY_KEYS)
            bar_w = 0.8 / n_idx
            for idx_i, k in enumerate(SUMMARY_KEYS):
                x_off = (idx_i - n_idx / 2 + 0.5) * bar_w
                means = [block_deltas[e][1][k][0] for e in range(n_evt)]
                stds  = [block_deltas[e][1][k][1] for e in range(n_evt)]
                ax_bar.bar(np.arange(n_evt) + x_off, means, width=bar_w,
                           color=SUMMARY_COLORS[k], alpha=0.85,
                           label=SUMMARY_LABELS[k], yerr=stds, capsize=3,
                           error_kw={'lw': 1.0}, zorder=3)
            ax_bar.axhline(0, color='k', lw=0.8)
            ax_bar.set_xticks(np.arange(n_evt))
            ax_bar.set_xticklabels([bd[0] for bd in block_deltas],
                                   rotation=25, ha='right', fontsize=8)
            ax_bar.set_ylabel(ylab_bar, fontsize=9)
            ax_bar.set_xlabel('Event Block', fontsize=9)
            ax_bar.set_title('Per-event mean Δ (error bars = across-channel SD)',
                             fontsize=10, loc='left')
            ax_bar.legend(loc='upper right', fontsize=8, ncol=n_idx)
            ax_bar.grid(True, alpha=0.2, axis='y')
            ax_bar.set_xlim(-0.5, n_evt - 0.5)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            fig.tight_layout(rect=[0, 0.0, 1, 0.97])
        if len(qb_dt):
            ax_qual.set_xlim(qb_dt[0], qb_dt[-1])

        baseline_tag = 'session' if mode == 'session' else 'pre_event'
        outpath = os.path.join(
            outdir, f'{name}_{info["sn"]}_tflite_summary_{baseline_tag}.png')
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'       → {outpath}')
        outpaths.append(outpath)
    return outpaths


# ── CLI ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='TFLite Summary 趨勢圖（Relax/Calm/Flow/Focus）+ 品質前後比較。')
    p.add_argument('--outdir',
                   default=os.path.join(IBRAIN_DIR, 'tflite_summary'),
                   metavar='DIR', help='輸出資料夾。')
    p.add_argument('--heatmap-baseline', choices=['session', 'pre-event', 'both'],
                   default='session',
                   help="heatmap/bar 的 Δ 基線模式："
                        "'session'（整段抽樣，灰格最少，預設）/"
                        "'pre-event'（每事件事件前基線，僅事件內著色）/'both'（兩張都出）。")
    p.add_argument('--t-buffer', type=float, default=DEFAULT_T_BUFFER,
                   help='緩衝區秒數（負值；觸發前此區間一律捨棄）。')
    p.add_argument('--t-search', type=float, default=DEFAULT_T_SEARCH,
                   help='基線搜尋窗起點（負值；觸發前幾秒開始找基線）。')
    p.add_argument('--required-sec', type=float, default=DEFAULT_REQUIRED_SEC,
                   help='所需乾淨基線總長度（秒）。')
    p.add_argument('--quality-ratio', type=float, default=QUALITY_THRESHOLD,
                   help='EEG 品質分數門檻 (0~1)；低於此值的視窗視為無效並捨棄，'
                        '亦作為基線微分段篩選門檻（預設 0.5；調高更嚴格）。')
    p.add_argument('--seed', type=int, default=DEFAULT_RANDOM_SEED,
                   help='固定亂數種子（確保可重現）。')
    p.add_argument('--on-insufficient', choices=['raise', 'skip'], default='raise',
                   help="基線不足時：'raise' 丟錯中止（預設）/ 'skip' 警告略過。")
    return p.parse_args()


def main():
    args = parse_args()
    print(f'── TFLite Summary → {args.outdir}')
    for name, info in SUBJECTS.items():
        try:
            plot_subject_tflite_summary(
                name, info, args.outdir,
                base_dir=IBRAIN_DIR,
                heatmap_baseline_mode=args.heatmap_baseline,
                t_buffer=args.t_buffer,
                t_search_start=args.t_search,
                required_sec=args.required_sec,
                quality_ratio=args.quality_ratio,
                random_seed=args.seed,
                on_insufficient=args.on_insufficient)
        except ValueError as exc:
            print(f'  [{name}] 中止：{exc}')
    print('\nDone.')


# ── 可直接執行的範例 (ready-to-run example) ──────────────────────────────────────
if __name__ == '__main__':
    # 範例 1：以 CLI 參數對 iBrainCenter 全部受試者產圖（最常用）。
    #   $ python plot_tflite_summary.py                       # 預設 session 基線（灰格最少）
    #   $ python plot_tflite_summary.py --heatmap-baseline pre-event   # 僅事件內著色
    #   $ python plot_tflite_summary.py --heatmap-baseline both --on-insufficient skip
    #
    # 範例 2：在程式中針對單一受試者呼叫（示範完整參數）。
    #   from plot_tflite_summary import plot_subject_tflite_summary, SUBJECTS
    #   plot_subject_tflite_summary(
    #       'Hardy', SUBJECTS['Hardy'],
    #       outdir='iBrainCenter/tflite_summary',
    #       heatmap_baseline_mode='session',  # 'session' / 'pre-event' / 'both'
    #       t_buffer=-3.0,        # 緩衝區：捨棄觸發前 3 秒（CNV / alpha ERD）
    #       t_search_start=-15.0, # 從觸發前 15 秒開始找基線（pre-event 用）
    #       required_sec=10.0,    # 需要 10 秒乾淨基線
    #       random_seed=42,       # 固定種子 → 可重現
    #       on_insufficient='skip')
    main()
