"""
plot_tyy_meditation.py
======================
Focused figure for TYY (SN041) — Mindfulness Meditation segment.

Panels (top → bottom):
  1. Ch1 EEG trace
  2. Ch2 EEG trace
  3. qEEG Δ vs baseline heatmap  (grayscale, BP-filtered)
  4. TFLite qEEG Δ vs baseline heatmap (grayscale, if model available)

X-axis is restricted to a configurable pre-baseline + meditation window.
Saves both PNG (150 dpi) and SVG to the output directory.

Usage
-----
    python plot_tyy_meditation.py [--outdir <dir>] [--ds <factor>] [--no-tflite]
"""

import argparse
import datetime
import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Locate project root and import shared utilities ───────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from eeg_utils import load_merged_csv, bandpass_filter          # noqa: E402
from qeeg_indices import compute_qeeg_indices                   # noqa: E402

# ── Session / subject constants ────────────────────────────────────────────────
SESSION_DATE   = datetime.date(2026, 5, 12)
TZ_OFFSET_H    = 8                        # Asia/Taipei = UTC+8
EPOCH          = datetime.datetime(1970, 1, 1)

SUBJECT_NAME   = 'TYY'
SUBJECT_SN     = 'SN041'
SUBJECT_DIR    = os.path.join(BASE_DIR, 'iBrainCenter', 'TYY(SN041)')
MERGED_CSV     = os.path.join(SUBJECT_DIR, 'merged.csv')

DEFAULT_OUTDIR = os.path.join(BASE_DIR, 'iBrainCenter', 'TYY_meditation')

# ── Signal parameters ─────────────────────────────────────────────────────────
FS             = 500      # Hz
TFLITE_FS      = 200      # model input sample rate
TFLITE_WIN     = 400      # model input window samples (= 2 s @ 200 Hz)
TFLITE_PATH    = os.path.join(BASE_DIR, 'tiny_v4_optimized.tflite')

QEEG_WIN_SEC   = 5.0      # non-overlapping qEEG window length
HEATMAP_BIN_SEC = 30      # bin width for the heatmap (seconds)

# ── Channels to display ───────────────────────────────────────────────────────
SHOW_CHS       = [0, 1]   # 0-based → Ch1, Ch2

# ── Event window ─────────────────────────────────────────────────────────────
MEDITATION_START_HHMM = '15:06'
MEDITATION_DUR_MIN    = 11

# Baseline = before first event TYY participates in (Push-ups 14:24),
# matching plot_event_markers.py logic (bins before first participating event).
BASELINE_END_HHMM  = '14:24'   # first TYY-participating event (Push-ups)
BASELINE_LOAD_HHMM = '14:10'   # start loading data for baseline computation

# ── Colours ───────────────────────────────────────────────────────────────────
CH_COLORS    = ['#1f77b4', '#ff7f0e']
IDX_COLORS   = {'focus': '#e6194b', 'flow': '#3cb44b',
                'calm':  '#4363d8', 'relaxation': '#f58231'}
INDEX_KEYS   = ['focus', 'flow', 'calm', 'relaxation']
IDX_LABELS   = ['Focus', 'Flow', 'Calm', 'Relax']
MEDITATION_COLOR = '#911eb4'


# ── Time helpers ──────────────────────────────────────────────────────────────

def hhmm_to_dt(hhmm: str) -> datetime.datetime:
    h, m = map(int, hhmm.split(':'))
    return datetime.datetime(SESSION_DATE.year, SESSION_DATE.month,
                             SESSION_DATE.day, h, m)


def us_to_local_dt(us: int) -> datetime.datetime:
    utc_dt = EPOCH + datetime.timedelta(microseconds=int(us))
    return utc_dt + datetime.timedelta(hours=TZ_OFFSET_H)


# ── qEEG windowed (multi-channel) ─────────────────────────────────────────────

def _compute_qeeg_windowed(time_us: np.ndarray, data: np.ndarray,
                           win_sec: float = QEEG_WIN_SEC,
                           fs: float = FS) -> tuple:
    """Return (q_dt_list, scores_dict) for the given (N, n_ch) data."""
    win   = int(win_sec * fs)
    n, n_ch = data.shape
    q_dt  = []
    accum = {k: [] for k in INDEX_KEYS}

    for start in range(0, n - win + 1, win):
        mid_us = int(time_us[start + win // 2])
        q_dt.append(us_to_local_dt(mid_us))
        row = {k: [] for k in INDEX_KEYS}
        for ch_i in range(n_ch):
            res = compute_qeeg_indices(
                data[start : start + win, ch_i].astype(np.float64), fs=fs)
            for k in INDEX_KEYS:
                row[k].append(res[k])
        for k in INDEX_KEYS:
            accum[k].append(row[k])

    return q_dt, {k: np.array(accum[k]) for k in INDEX_KEYS}  # (n_win, n_ch)


# ── TFLite inference ──────────────────────────────────────────────────────────

def _apply_tflite(data: np.ndarray) -> np.ndarray:
    """Run TFLite model on (N, 4) float32 data; returns (M, 2)."""
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    n_win  = len(data) // TFLITE_WIN
    chunks = []
    for i in range(n_win):
        seg = data[i * TFLITE_WIN : (i + 1) * TFLITE_WIN][np.newaxis].astype(np.float32)
        interp.set_tensor(inp['index'], seg)
        interp.invoke()
        chunks.append(interp.get_tensor(out['index'])[0])
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 2), np.float32)


# ── Heatmap builder ───────────────────────────────────────────────────────────

def _build_heatmap(q_dt: list, scores: dict, bl_end_dt: datetime.datetime,
                   bin_sec: float = HEATMAP_BIN_SEC,
                   win_sec: float = QEEG_WIN_SEC) -> tuple:
    """
    Bin qEEG scores into *bin_sec* medians and compute Δ vs pre-event baseline.

    Returns (bin_t_arr, heatmap_delta_ma) — shape (4, n_bins).
    """
    t_arr    = np.array(q_dt)
    n_hm     = scores[INDEX_KEYS[0]].shape[0]
    bin_size = max(1, int(bin_sec / win_sec))
    n_bins   = n_hm // bin_size

    heatmap_abs = np.full((4, n_bins), np.nan)
    bin_t_list  = []

    for b in range(n_bins):
        sl  = slice(b * bin_size, (b + 1) * bin_size)
        mid = t_arr[b * bin_size + bin_size // 2]
        bin_t_list.append(mid)
        for idx_i, k in enumerate(INDEX_KEYS):
            vals = np.median(scores[k][sl], axis=1)  # (bin_size,)
            if vals.size > 0:
                heatmap_abs[idx_i, b] = float(np.nanmedian(vals))

    bin_t_arr = np.array(bin_t_list)
    bl_mask   = bin_t_arr < bl_end_dt

    heatmap_delta = np.full_like(heatmap_abs, np.nan)
    for idx_i in range(4):
        bl_vals = heatmap_abs[idx_i, bl_mask]
        bl_ref  = float(np.nanmedian(bl_vals)) if np.any(~np.isnan(bl_vals)) else 0.0
        heatmap_delta[idx_i] = heatmap_abs[idx_i] - bl_ref

    return bin_t_arr, np.ma.masked_invalid(heatmap_delta)


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _fmt_xaxis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 2)))
    ax.grid(True, alpha=0.2)


def _draw_meditation_span(ax, start_dt, end_dt, label=True):
    ax.axvspan(start_dt, end_dt, color=MEDITATION_COLOR, alpha=0.15)
    ax.axvline(start_dt, color=MEDITATION_COLOR, lw=1.2, ls='--', alpha=0.8)
    if label:
        ylim = ax.get_ylim()
        ax.text(start_dt, ylim[1] * 0.92,
                'Mindfulness\nMeditation',
                fontsize=7, color=MEDITATION_COLOR, va='top', clip_on=True)


def _draw_heatmap_panel(ax, bin_t_arr, heatmap_delta_ma,
                        med_start_dt, med_end_dt, ylabel):
    n_bins = len(bin_t_arr)
    if n_bins == 0:
        ax.set_visible(False)
        return

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        'OrgPur', ['#5e3c99', '#f7f7f7', '#e66101'])  # purple–white–orange diverging
    cmap.set_bad(color='#aaaaaa')

    bin_t_num = mdates.date2num(bin_t_arr)
    dt_h = (float(np.diff(bin_t_num).mean()) / 2) if n_bins > 1 \
           else (HEATMAP_BIN_SEC / 86400 / 2)
    t_edges = np.concatenate([[bin_t_num[0] - dt_h],
                               (bin_t_num[:-1] + bin_t_num[1:]) / 2,
                               [bin_t_num[-1] + dt_h]])
    y_edges = np.arange(5) - 0.5
    v_abs   = max(0.3, float(np.nanpercentile(np.abs(heatmap_delta_ma.data), 95)))

    pcm = ax.pcolormesh(t_edges, y_edges, heatmap_delta_ma,
                        cmap=cmap, vmin=-v_abs, vmax=v_abs, shading='flat')
    plt.colorbar(pcm, ax=ax, pad=0.005, fraction=0.015,
                 label=f'Δ Index  (−{v_abs:.1f} → +{v_abs:.1f})')

    ax.axvline(mdates.date2num(med_start_dt),
               color=MEDITATION_COLOR, lw=1.2, ls='--', alpha=0.8)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(IDX_LABELS, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 2)))
    ax.grid(False)


# ── Main plot function ────────────────────────────────────────────────────────

def plot_tyy_meditation(outdir: str, ds: int = 500, use_tflite: bool = True):
    os.makedirs(outdir, exist_ok=True)

    if not os.path.isfile(MERGED_CSV):
        sys.exit(f'merged.csv not found: {MERGED_CSV}')

    print(f'Loading {MERGED_CSV}…')
    time_us_ds,   data_ds   = load_merged_csv(MERGED_CSV, downsample=ds)
    time_us_full, data_full = load_merged_csv(MERGED_CSV)

    # ── Time window ──────────────────────────────────────────────────────────
    med_start_dt  = hhmm_to_dt(MEDITATION_START_HHMM)
    med_end_dt    = med_start_dt + datetime.timedelta(minutes=MEDITATION_DUR_MIN)
    bl_end_dt     = hhmm_to_dt(BASELINE_END_HHMM)   # first participating event
    win_start_dt  = hhmm_to_dt(BASELINE_LOAD_HHMM)  # load from here for baseline
    win_end_dt    = med_end_dt + datetime.timedelta(minutes=2)
    plot_start_dt = med_start_dt - datetime.timedelta(minutes=6)  # show 6 min pre-meditation

    # Downsampled time array (for EEG traces)
    t_dt = np.array([us_to_local_dt(u) for u in time_us_ds])

    # ── Restrict full-res data to window for efficiency ───────────────────────
    def _dt_to_us(dt):
        delta = dt - EPOCH
        return int(delta.total_seconds() * 1e6) - TZ_OFFSET_H * 3600 * int(1e6)

    win_start_us = _dt_to_us(win_start_dt)
    win_end_us   = _dt_to_us(win_end_dt)
    sel_full     = (time_us_full >= win_start_us) & (time_us_full <= win_end_us)
    time_us_win  = time_us_full[sel_full]
    data_win     = data_full[sel_full]

    # ── Band-pass filter (full window) ───────────────────────────────────────
    print('Bandpass filtering…')
    data_filt = bandpass_filter(data_win)

    # ── qEEG (Ch1 + Ch2 only from the windowed data) ─────────────────────────
    print(f'Computing qEEG indices ({QEEG_WIN_SEC:.0f}s windows)…')
    qeeg_dt, qeeg_filt = _compute_qeeg_windowed(time_us_win, data_filt)
    print(f'  {len(qeeg_dt)} windows')

    # ── TFLite ───────────────────────────────────────────────────────────────
    tfl_qeeg_dt, tfl_qeeg = None, None
    if use_tflite and os.path.isfile(TFLITE_PATH):
        print(f'Resampling {FS}→{TFLITE_FS}Hz and running TFLite…')
        try:
            from scipy.signal import resample_poly
            from math import gcd
            g    = gcd(TFLITE_FS, FS)
            up_, dn_ = TFLITE_FS // g, FS // g
            data_200 = resample_poly(data_filt, up_, dn_, axis=0).astype(np.float32)
            t_orig   = np.arange(len(data_filt))
            t_new    = np.arange(len(data_200)) * (dn_ / up_)
            tfl_time = np.interp(t_new, t_orig,
                                 time_us_win[:len(data_filt)]).astype(np.int64)
            tfl_raw  = _apply_tflite(data_200)
            N_tfl    = len(tfl_raw)
            tfl_time = tfl_time[:N_tfl]
            print(f'  TFLite: {N_tfl} samples ({N_tfl / TFLITE_FS:.0f}s)')
            print(f'  Computing TFLite qEEG…')
            tfl_qeeg_dt, tfl_qeeg = _compute_qeeg_windowed(
                tfl_time, tfl_raw, fs=TFLITE_FS)
            print(f'  {len(tfl_qeeg_dt)} windows')
        except Exception as exc:
            print(f'  TFLite skipped: {exc}')

    # ── Build heatmaps ───────────────────────────────────────────────────────
    bin_t_arr, heatmap_dm = _build_heatmap(qeeg_dt, qeeg_filt, bl_end_dt)

    tfl_bin_t_arr, tfl_heatmap_dm = None, None
    if tfl_qeeg is not None:
        tfl_bin_t_arr, tfl_heatmap_dm = _build_heatmap(
            tfl_qeeg_dt, tfl_qeeg, bl_end_dt,
            bin_sec=HEATMAP_BIN_SEC, win_sec=QEEG_WIN_SEC)

    has_tflite = (tfl_heatmap_dm is not None)

    # ── Figure layout ────────────────────────────────────────────────────────
    n_ch_shown = len(SHOW_CHS)
    height_ratios = ([2.5] * n_ch_shown
                     + [2.2]                         # qEEG Δ heatmap
                     + ([2.2] if has_tflite else []))  # TFLite Δ heatmap
    n_rows = len(height_ratios)
    fig_h  = max(6, sum(hr * 0.9 for hr in height_ratios) + 1.5)
    fig    = plt.figure(figsize=(14, fig_h))
    gs     = gridspec.GridSpec(n_rows, 1, figure=fig,
                               height_ratios=height_ratios, hspace=0.30)

    ax_ch   = []
    for i, ch_i in enumerate(SHOW_CHS):
        ax = fig.add_subplot(gs[i], sharex=ax_ch[0] if ax_ch else None)
        ax_ch.append(ax)

    _row          = n_ch_shown
    ax_hm         = fig.add_subplot(gs[_row], sharex=ax_ch[0]); _row += 1
    ax_tfl_hm     = (fig.add_subplot(gs[_row], sharex=ax_ch[0])
                     if has_tflite else None)

    # ── EEG trace panels ─────────────────────────────────────────────────────
    sel_ds = (t_dt >= win_start_dt) & (t_dt <= win_end_dt)
    t_ds_w = t_dt[sel_ds]

    for i, ch_i in enumerate(SHOW_CHS):
        ax = ax_ch[i]
        ax.plot(t_ds_w, data_ds[sel_ds, ch_i],
                color=CH_COLORS[i], lw=0.5, alpha=0.85)
        _draw_meditation_span(ax, med_start_dt, med_end_dt, label=(i == 0))
        ax.set_ylim(-100, 100)
        ax.set_ylabel(f'Ch{ch_i + 1}\n(µV)', fontsize=8)
        _fmt_xaxis(ax)
        plt.setp(ax.get_xticklabels(), visible=False)

    # ── qEEG Δ heatmap ───────────────────────────────────────────────────────
    _draw_heatmap_panel(ax_hm, bin_t_arr, heatmap_dm,
                        med_start_dt, med_end_dt,
                        'qEEG Δ\n(vs baseline)')
    plt.setp(ax_hm.get_xticklabels(),
             visible=(not has_tflite))
    if has_tflite:
        plt.setp(ax_hm.get_xticklabels(), visible=False)

    # ── TFLite Δ heatmap ─────────────────────────────────────────────────────
    if ax_tfl_hm is not None:
        _draw_heatmap_panel(ax_tfl_hm, tfl_bin_t_arr, tfl_heatmap_dm,
                            med_start_dt, med_end_dt,
                            'TFLite qEEG Δ\n(vs baseline)')
        ax_tfl_hm.set_xlabel('Local Time (UTC+8, HH:MM)', fontsize=10)

    if not has_tflite:
        ax_hm.set_xlabel('Local Time (UTC+8, HH:MM)', fontsize=10)

    # ── Title + limits ────────────────────────────────────────────────────────
    tflite_note = '  +TFLite' if has_tflite else ''
    fig.suptitle(
        f'{SUBJECT_SN} — Mindfulness Meditation  '
        f'({MEDITATION_START_HHMM}, {MEDITATION_DUR_MIN} min){tflite_note}\n'
        f'Ch1 & Ch2  |  qEEG Δ heatmap (orange–purple, BP 0.5–45 Hz, 5 s windows, baseline < {BASELINE_END_HHMM})',
        fontsize=12, fontweight='bold',
    )

    # Event legend patch
    med_patch = mpatches.Patch(facecolor=MEDITATION_COLOR, alpha=0.35,
                               label=f'Mindfulness Meditation ({MEDITATION_START_HHMM})')
    fig.legend(handles=[med_patch], loc='lower center', ncol=1,
               fontsize=9, bbox_to_anchor=(0.5, -0.01), framealpha=0.9)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        fig.tight_layout(rect=[0, 0.03, 1, 1])

    # Lock x-axis to meditation window only (baseline used for Δ calc, not shown)
    ax_ch[0].set_xlim(plot_start_dt, win_end_dt)

    # ── Save ─────────────────────────────────────────────────────────────────
    suffix = '_tflite' if has_tflite else ''
    stem   = os.path.join(outdir, f'TYY_{SUBJECT_SN}_meditation{suffix}')
    for ext in ('png', 'svg'):
        path = f'{stem}.{ext}'
        dpi  = 150 if ext == 'png' else None
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        print(f'Saved: {path}')

    plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description='Plot TYY (SN041) Mindfulness Meditation — Ch1, Ch2, qEEG Δ heatmaps.')
    p.add_argument('--outdir', default=DEFAULT_OUTDIR, metavar='DIR',
                   help=f'Output directory (default: {DEFAULT_OUTDIR})')
    p.add_argument('--ds', type=int, default=500, metavar='N',
                   help='EEG trace downsample factor (default 500 → 1 pt/s)')
    p.add_argument('--no-tflite', action='store_true',
                   help='Skip TFLite model processing')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    plot_tyy_meditation(
        outdir=args.outdir,
        ds=args.ds,
        use_tflite=not args.no_tflite,
    )
