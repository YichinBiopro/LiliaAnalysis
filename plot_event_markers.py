"""
plot_event_markers.py
=====================
Overlay iBrainCenter session event markers (from evt_time.docx) onto each
subject's merged.csv EEG signal for visual verification.

Events are defined inline (parsed from evt_time.docx) with English names.
Time alignment uses the Abs Time Offset[us] header field (UTC Unix µs) and
the session date 2026-05-12, Asia/Taipei (UTC+8).

Each figure shows:
  • 4 EEG channel traces (downsampled for speed)
  • 1 EEG quality panel (flat+spectrum only, 5 s sliding window)
  All panels share the absolute HH:MM x-axis.

Usage
-----
    python plot_event_markers.py [--outdir <dir>] [--ds <factor>]

Outputs one PNG per subject to <outdir> (default: iBrainCenter/event_verification/).
"""

import argparse
import datetime
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import signal
from eeg_quality_v2 import (
    get_eeg_quality_index_v2_parametric,
    get_best_eeg_quality_v2_flat_spectrum_only_params,
)
from qeeg_indices import compute_qeeg_indices_windowed
from qeeg_indices import compute_qeeg_indices
from eeg_utils import load_merged_csv, bandpass_filter

# ── Session metadata ───────────────────────────────────────────────────────────
SESSION_DATE = datetime.date(2026, 5, 12)
TZ_OFFSET_H  = 8          # Asia/Taipei = UTC+8
EPOCH        = datetime.datetime(1970, 1, 1)

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
IBRAIN_DIR   = os.path.join(BASE_DIR, 'iBrainCenter')
YOGA_DIR     = os.path.join(BASE_DIR, 'YoGa')

FS               = 500          # Hz
QUALITY_WIN_SEC  = 5.0          # window length for quality scorer (5 s)
QUALITY_STEP_SEC = 5.0          # non-overlapping windows (matches app refresh)
QUALITY_WIN_SEC_LONG  = 30.0   # second quality scorer window (30 s)
QUALITY_PARAMS    = get_best_eeg_quality_v2_flat_spectrum_only_params()
QUALITY_THRESHOLD = 0.5

TFLITE_MODEL_PATH = os.path.join(BASE_DIR, 'tiny_v4_optimized.tflite')
TFLITE_FS         = 200   # model sample rate (Hz)
TFLITE_WIN        = 400   # model input window size at TFLITE_FS (400 samples = 2 s)

BP_LOW       = 0.5    # Hz — bandpass lower cutoff
BP_HIGH      = 45.0   # Hz — bandpass upper cutoff
QEEG_WIN_SEC = 5.0    # window for qEEG indices (non-overlapping, seconds)

# ── Event table (from evt_time.docx) ──────────────────────────────────────────
# Each entry: (english_name, start_HH_MM, duration_min, [participant_keys])
# participant_keys: subset of subject keys below; None = all
EVENTS = [
    ('Single Cycling',        '14:13',  3,   ['Hsin', 'Hardy', 'James']),
    ('Cycling Boxing',        '14:17',  3,   ['Hsin', 'Hardy', 'James']),
    ('Push-ups',              '14:24',  5,   None),   # 3-5 min, use 5
    ('Machine Chest Press',   '14:35',  3,   None),
    ('Agility Ladder',        '14:43',  7,   None),
    ('Color Agility Ladder',  '14:50',  5,   ['Hardy', 'Ann', 'Hsin', 'James']),
    ('Cone Rotation',         '14:58',  5,   None),
    ('Mindfulness Meditation','15:06', 11,   None),
]

# Sub-events inside Cone Rotation (stage escalation)
CONE_STAGES = ['14:58', '15:00', '15:03']

# ── Subject registry ───────────────────────────────────────────────────────────
SUBJECTS = {
    'Ann':   {'sn': 'SN027', 'dir': 'Ann(SN027)'},
    'Hsin':  {'sn': 'SN032', 'dir': 'Hsin(SN032)'},
    'Hardy': {'sn': 'SN036', 'dir': 'Hardy(SN036)'},
    'TYY':   {'sn': 'SN041', 'dir': 'TYY(SN041)'},
    'James': {'sn': 'SN035', 'dir': 'James(SN035)'},
}

YOGA_SUBJECTS = {
    'James':  {'sn': 'SN035', 'dir': 'James(SN035)'},
    'Jammie': {'sn': 'SN036', 'dir': 'Jammie(SN036)'},
    'TYY':    {'sn': 'SN041', 'dir': 'TYY(SN041)'},
}

# ── Colours per event (cycle if more events added) ────────────────────────────
EVT_COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
]

CH_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

HARDY2_EVENTS = [
    ('Wear Device', '16:18', '#1f77b4'),
    ('Light On', '16:21', '#d62728'),
    ('End', '16:36', '#2ca02c'),
]

HARDY2_BANDS = {
    'delta': (1.0, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta':  (13.0, 30.0),
    'gamma': (30.0, 45.0),
}

HARDY2_BAND_COLORS = {
    'delta': '#6a3d9a',
    'theta': '#1f78b4',
    'alpha': '#33a02c',
    'beta':  '#ff7f00',
    'gamma': '#e31a1c',
}

HARDY2_INDEX_KEYS = ['focus', 'flow', 'calm', 'relaxation']
HARDY2_INDEX_LABELS = {
    'focus': 'Focus',
    'flow': 'Flow',
    'calm': 'Calm',
    'relaxation': 'Relax',
}
HARDY2_INDEX_COLORS = {
    'focus': '#e6194b',
    'flow': '#3cb44b',
    'calm': '#4363d8',
    'relaxation': '#f58231',
}


# ── Time helpers ───────────────────────────────────────────────────────────────

def hhmm_to_us(hhmm: str) -> int:
    """Convert 'HH:MM' on SESSION_DATE (local TZ) to UTC Unix microseconds."""
    h, m = map(int, hhmm.split(':'))
    local_dt = datetime.datetime(
        SESSION_DATE.year, SESSION_DATE.month, SESSION_DATE.day, h, m,
        tzinfo=datetime.timezone(datetime.timedelta(hours=TZ_OFFSET_H)))
    utc_dt = local_dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return int((utc_dt - EPOCH).total_seconds() * 1_000_000)


def us_to_local_dt(us: int) -> datetime.datetime:
    """Convert UTC Unix µs → local datetime (UTC+8, naive for matplotlib)."""
    utc_dt = EPOCH + datetime.timedelta(microseconds=int(us))
    return utc_dt + datetime.timedelta(hours=TZ_OFFSET_H)


def hhmm_to_dt(hhmm: str) -> datetime.datetime:
    """'HH:MM' on SESSION_DATE (local) → naive local datetime."""
    h, m = map(int, hhmm.split(':'))
    return datetime.datetime(SESSION_DATE.year, SESSION_DATE.month,
                             SESSION_DATE.day, h, m)


# ── CSV loading ────────────────────────────────────────────────────────────────

def read_abs_time_offset(path: str) -> int:
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 1:
                return int(line.strip().split(',')[3])
    raise ValueError(f'Cannot read offset from {path}')


# ── EEG quality (windowed) ─────────────────────────────────────────────────────

def compute_quality_windowed(time_us: np.ndarray, data: np.ndarray,
                             win_sec: float = QUALITY_WIN_SEC):
    """
    Slide a non-overlapping window over data and score each window.

    Parameters
    ----------
    time_us : (N,) absolute UTC Unix µs timestamps
    data    : (N, n_ch) EEG samples at FS
    win_sec : window length in seconds (default QUALITY_WIN_SEC=5 s)

    Returns
    -------
    q_dt      : list of local datetime objects (window midpoints)
    q_overall : (n_windows, n_ch) float array, overall quality per channel
    """
    win  = int(win_sec * FS)
    step = win   # non-overlapping
    n    = len(data)

    q_dt      = []
    q_overall = []

    for start in range(0, n - win + 1, step):
        seg = data[start : start + win]            # (win, n_ch)
        mid_us = int(time_us[start + win // 2])
        result = get_eeg_quality_index_v2_parametric(
            seg.T.astype(np.float64),              # (n_ch, win)
            fs=FS,
            params=QUALITY_PARAMS,
        )
        q_dt.append(us_to_local_dt(mid_us))
        q_overall.append(result["overall"])        # (n_ch,)

    return q_dt, np.array(q_overall)              # (n_windows, n_ch)


# ── qEEG windowed computation ──────────────────────────────────────────────────

QEEG_INDICES = ['focus', 'flow', 'calm', 'relaxation']
QEEG_COLORS  = ['#e6194b', '#3cb44b', '#4363d8', '#f58231']   # one per index

def compute_qeeg_windowed(time_us: np.ndarray, data: np.ndarray,
                          win_sec: float = QEEG_WIN_SEC,
                          fs: float = FS):
    """
    Slide non-overlapping windows over *data* (N, n_ch) and compute the four
    qEEG wellness indices for each channel.

    Returns
    -------
    q_dt    : list of local datetime objects (window midpoints)
    scores  : dict { index_name -> np.ndarray (n_windows, n_ch) }
    """
    win  = int(win_sec * fs)
    n    = len(data)
    n_ch = data.shape[1]

    q_dt   = []
    accum  = {k: [] for k in QEEG_INDICES}

    for start in range(0, n - win + 1, win):
        mid_us = int(time_us[start + win // 2])
        q_dt.append(us_to_local_dt(mid_us))
        row = {k: [] for k in QEEG_INDICES}
        for ch_i in range(n_ch):
            res = compute_qeeg_indices(data[start : start + win, ch_i].astype(np.float64),
                                       fs=fs)
            for k in QEEG_INDICES:
                row[k].append(res[k])
        for k in QEEG_INDICES:
            accum[k].append(row[k])

    return q_dt, {k: np.array(accum[k]) for k in QEEG_INDICES}  # (n_windows, n_ch)


# ── TFLite model inference ─────────────────────────────────────────────────────

def apply_tflite_windowed(data: np.ndarray,
                          tflite_path: str = TFLITE_MODEL_PATH) -> np.ndarray:
    """Run tiny_v4_optimized.tflite on (N, 4) data in non-overlapping windows.

    Parameters
    ----------
    data        : (N, 4) float32 bandpass-filtered EEG
    tflite_path : path to the .tflite model file

    Returns
    -------
    out : (M, 2) float32, M = (N // TFLITE_WIN) * TFLITE_WIN
    """
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    n_win  = len(data) // TFLITE_WIN
    chunks = []
    for i in range(n_win):
        seg = data[i * TFLITE_WIN : (i + 1) * TFLITE_WIN][np.newaxis].astype(np.float32)
        interp.set_tensor(inp_det['index'], seg)
        interp.invoke()
        chunks.append(interp.get_tensor(out_det['index'])[0])   # (400, 2)
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 2), np.float32)


# ── Event overlay helper ───────────────────────────────────────────────────────

def _overlay_events(
    ax: plt.Axes,
    evt_list: list,
    cone_stage_dt: list,
) -> None:
    """Overlay event spans, start lines, and Cone stage markers on *ax*.

    Parameters
    ----------
    ax             : matplotlib Axes to annotate
    evt_list       : list of (start_dt, end_dt, label, color, participates) tuples
    cone_stage_dt  : list of datetime objects for Cone Rotation sub-stages
    """
    for start_dt, end_dt, label, color, participates in evt_list:
        ax.axvspan(start_dt, end_dt, color=color,
                   alpha=0.20 if participates else 0.06)
        if participates:
            ax.axvline(start_dt, color=color, lw=1.2, ls='--', alpha=0.7)
    for cdt in cone_stage_dt:
        ax.axvline(cdt, color='#ff1493', lw=1.0, ls=':', alpha=0.6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 5)))
    ax.grid(True, alpha=0.2)


def _bandpower_from_psd(freqs: np.ndarray, psd: np.ndarray,
                        fmin: float, fmax: float) -> float:
    """Integrate PSD over [fmin, fmax] using trapezoidal rule."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def _compute_hardy2_windowed_metrics(time_us: np.ndarray,
                                     data: np.ndarray,
                                     win_sec: float = 5.0,
                                     fs: float = FS):
    """Compute 5-band relative powers + qEEG indices in non-overlapping windows."""
    win = int(win_sec * fs)
    n = len(data)
    n_ch = data.shape[1]

    time_dt = []
    band_acc = {k: [] for k in HARDY2_BANDS}
    idx_acc = {k: [] for k in HARDY2_INDEX_KEYS}

    for start in range(0, n - win + 1, win):
        seg = data[start:start + win]
        mid_us = int(time_us[start + win // 2])
        time_dt.append(us_to_local_dt(mid_us))

        per_ch_band = {k: [] for k in HARDY2_BANDS}
        per_ch_idx = {k: [] for k in HARDY2_INDEX_KEYS}
        for ch_i in range(n_ch):
            x = seg[:, ch_i].astype(np.float64)
            freqs, psd = signal.welch(
                x,
                fs=fs,
                nperseg=min(len(x), int(fs * 4)),
                noverlap=min(len(x) // 2, int(fs * 2)),
                window='hann',
            )
            p = {k: _bandpower_from_psd(freqs, psd, *fr)
                 for k, fr in HARDY2_BANDS.items()}
            p_sum = sum(p.values()) + 1e-12
            for k in HARDY2_BANDS:
                per_ch_band[k].append(p[k] / p_sum)

            idx = compute_qeeg_indices(x, fs=fs)
            for k in HARDY2_INDEX_KEYS:
                per_ch_idx[k].append(float(idx[k]))

        for k in HARDY2_BANDS:
            band_acc[k].append(float(np.median(per_ch_band[k])))
        for k in HARDY2_INDEX_KEYS:
            idx_acc[k].append(float(np.median(per_ch_idx[k])))

    return np.array(time_dt), {k: np.array(v) for k, v in band_acc.items()}, \
        {k: np.array(v) for k, v in idx_acc.items()}


def _annotate_hardy2_events(ax: plt.Axes) -> None:
    """Add Hardy_2 event lines and labels."""
    y_positions = [0.965, 0.925, 0.965]
    for (label, hhmm, color), ypos in zip(HARDY2_EVENTS, y_positions):
        dt = hhmm_to_dt(hhmm)
        ax.axvline(dt, color=color, ls='--', lw=1.5, alpha=0.9, zorder=4)
        ax.text(dt, ypos, f'{hhmm} {label}', color=color, fontsize=10,
                fontweight='bold', ha='center', va='top',
                transform=ax.get_xaxis_transform(),
                bbox=dict(boxstyle='round,pad=0.22', fc='white', ec='none', alpha=0.82))


def _summarize_hardy2_periods(time_dt: np.ndarray, series_dict: dict,
                              title: str) -> None:
    """Print period-wise means around Hardy_2 events for quick inspection."""
    t_wear = hhmm_to_dt('16:18')
    t_light = hhmm_to_dt('16:21')
    t_end = hhmm_to_dt('16:36')
    masks = [
        ('Pre-16:18', time_dt < t_wear),
        ('16:18-16:21', (time_dt >= t_wear) & (time_dt < t_light)),
        ('16:21-16:36', (time_dt >= t_light) & (time_dt < t_end)),
        ('Post-16:36', time_dt >= t_end),
    ]
    print(f'\n[{title}] period means:')
    for name, mask in masks:
        if mask.sum() == 0:
            print(f'  {name:<12s} no samples')
            continue
        vals = []
        for k, arr in series_dict.items():
            vals.append(f'{k}:{float(np.nanmean(arr[mask])):+.3f}')
        print(f'  {name:<12s} ' + '  '.join(vals))


def _hardy2_period_masks(time_dt: np.ndarray):
    """Return period masks for Hardy_2 event timeline."""
    t_wear = hhmm_to_dt('16:18')
    t_light = hhmm_to_dt('16:21')
    t_end = hhmm_to_dt('16:36')
    return [
        ('Pre-16:18', time_dt < t_wear, '#dddddd'),
        ('16:18-16:21', (time_dt >= t_wear) & (time_dt < t_light), '#d9ecff'),
        ('16:21-16:36', (time_dt >= t_light) & (time_dt < t_end), '#ffe3d9'),
        ('Post-16:36', time_dt >= t_end, '#e6f7e6'),
    ]


def _series_with_gaps(time_dt: np.ndarray, values: np.ndarray,
                      gap_sec: float) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaNs across large gaps so matplotlib does not connect segments."""
    if len(time_dt) == 0:
        return time_dt, values

    out_t = [time_dt[0]]
    out_v = [values[0]]
    for idx in range(1, len(time_dt)):
        gap = (time_dt[idx] - time_dt[idx - 1]).total_seconds()
        if gap > gap_sec:
            out_t.append(time_dt[idx - 1] + datetime.timedelta(seconds=1))
            out_v.append(np.nan)
        out_t.append(time_dt[idx])
        out_v.append(values[idx])
    return np.array(out_t, dtype=object), np.array(out_v, dtype=float)


def _smooth_series(values: np.ndarray, win_points: int = 5) -> np.ndarray:
    """Centered rolling mean for presentation-friendly trend lines."""
    return (pd.Series(values)
            .rolling(win_points, center=True, min_periods=1)
            .mean()
            .to_numpy())


def _style_presentation_axis(ax: plt.Axes) -> None:
    """Apply a cleaner presentation-oriented visual style."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', alpha=0.22)
    ax.grid(True, axis='x', alpha=0.10)
    ax.tick_params(labelsize=10)


def plot_hardy2_band_and_indices(outdir: str,
                                 win_sec: float = QEEG_WIN_SEC,
                                 use_bandpass: bool = True):
    """Generate Hardy_2 band-power and wellness-index plots with event markers."""
    merged = os.path.join(IBRAIN_DIR, 'Hardy_2(SN036)', 'merged.csv')
    if not os.path.isfile(merged):
        raise FileNotFoundError(f'merged.csv not found: {merged}')

    os.makedirs(outdir, exist_ok=True)
    print(f'\n[Hardy_2] loading: {merged}')
    time_us, data = load_merged_csv(merged)
    if use_bandpass:
        data = bandpass_filter(data, fs=FS, lo=BP_LOW, hi=BP_HIGH)

    print(f'[Hardy_2] computing {win_sec:.1f}s windowed metrics...')
    time_dt, band_power, indices = _compute_hardy2_windowed_metrics(
        time_us, data, win_sec=win_sec, fs=FS)

    period_defs = _hardy2_period_masks(time_dt)
    gap_sec = max(15.0, win_sec * 2.5)
    for band_key in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        fig1, ax1 = plt.subplots(figsize=(16, 6.3))
        y = band_power[band_key]
        y_smooth = _smooth_series(y, win_points=5)
        raw_t, raw_y = _series_with_gaps(time_dt, y, gap_sec=gap_sec)
        smooth_t, smooth_y = _series_with_gaps(time_dt, y_smooth, gap_sec=gap_sec)

        for period_name, mask, shade_color in period_defs:
            if mask.sum() == 0:
                continue
            t0 = time_dt[np.argmax(mask)]
            t1 = time_dt[len(mask) - 1 - np.argmax(mask[::-1])]
            ax1.axvspan(t0, t1, color=shade_color, alpha=0.26, zorder=0)
            mean_ratio = float(np.nanmean(y[mask]))
            xm = t0 + (t1 - t0) / 2
            ax1.text(
                xm, 0.895,
                f'{period_name}: {mean_ratio * 100:.1f}%',
                ha='center', va='top', fontsize=10, fontweight='bold',
                transform=ax1.get_xaxis_transform(),
                color='#111111',
                bbox=dict(boxstyle='round,pad=0.24', fc='white', ec='none', alpha=0.88),
            )

        ax1.plot(raw_t, raw_y, lw=1.1,
                 color=HARDY2_BAND_COLORS[band_key], alpha=0.22)
        ax1.plot(smooth_t, smooth_y, lw=3.0,
                 color=HARDY2_BAND_COLORS[band_key],
                 label=f'{band_key.capitalize()} ratio (25s smooth)')
        ax1.fill_between(raw_t, 0, raw_y,
                         color=HARDY2_BAND_COLORS[band_key], alpha=0.08)

        ax1.set_title(
            f'Hardy_2(SN036) {band_key.capitalize()} Band Ratio vs Time',
            fontsize=17, fontweight='bold', pad=12)
        ax1.text(0.0, 1.02,
                 'Median across 4 channels, 5 s windows, bold line = smoothed trend',
                 transform=ax1.transAxes, fontsize=10.5, color='#444444')
        ax1.set_ylabel('Band Ratio (%)')
        ax1.set_xlabel('Local Time (UTC+8, HH:MM)')
        ax1.set_ylim(0, 1)
        ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 3)))
        _style_presentation_axis(ax1)
        ax1.legend(loc='lower right', frameon=False, fontsize=10)
        _annotate_hardy2_events(ax1)
        fig1.tight_layout()
        out_band = os.path.join(
            outdir, f'Hardy_2_SN036_{band_key}_band_ratio_vs_time.png')
        fig1.savefig(out_band, dpi=180)
        plt.close(fig1)
        print(f'[Hardy_2] saved: {out_band}')

    fig2, ax2 = plt.subplots(figsize=(16, 6.8))
    for period_name, mask, shade_color in period_defs:
        if mask.sum() == 0:
            continue
        t0 = time_dt[np.argmax(mask)]
        t1 = time_dt[len(mask) - 1 - np.argmax(mask[::-1])]
        ax2.axvspan(t0, t1, color=shade_color, alpha=0.22, zorder=0)
    for k in HARDY2_INDEX_KEYS:
        y_raw = indices[k]
        y_smooth = _smooth_series(y_raw, win_points=5)
        raw_t, raw_y = _series_with_gaps(time_dt, y_raw, gap_sec=gap_sec)
        smooth_t, smooth_y = _series_with_gaps(time_dt, y_smooth, gap_sec=gap_sec)
        ax2.plot(raw_t, raw_y, lw=1.0, alpha=0.20,
                 color=HARDY2_INDEX_COLORS[k])
        ax2.plot(smooth_t, smooth_y, lw=2.7,
                 color=HARDY2_INDEX_COLORS[k], label=HARDY2_INDEX_LABELS[k])
    ax2.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax2.set_title('Hardy_2(SN036) Focus / Flow / Calm / Relax vs Time',
                  fontsize=17, fontweight='bold', pad=12)
    ax2.text(0.0, 1.02,
             'Thin line = raw 5 s windows, bold line = 25 s smoothed trend',
             transform=ax2.transAxes, fontsize=10.5, color='#444444')
    ax2.set_ylabel('Index (−1 to +1)')
    ax2.set_xlabel('Local Time (UTC+8, HH:MM)')
    ax2.set_ylim(-1.1, 1.1)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 3)))
    _style_presentation_axis(ax2)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=4,
               frameon=False, fontsize=11)
    _annotate_hardy2_events(ax2)
    fig2.tight_layout(rect=[0, 0.05, 1, 1])
    out_idx = os.path.join(outdir, 'Hardy_2_SN036_focus_flow_calm_relax_vs_time.png')
    fig2.savefig(out_idx, dpi=180)
    plt.close(fig2)
    print(f'[Hardy_2] saved: {out_idx}')

    _summarize_hardy2_periods(time_dt, band_power, 'Band Power')
    _summarize_hardy2_periods(time_dt, {
        'focus': indices['focus'],
        'flow': indices['flow'],
        'calm': indices['calm'],
        'relax': indices['relaxation'],
    }, 'Indices')


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_subject(name: str, info: dict, outdir: str, ds: int,
                 base_dir: str = None, with_events: bool = True,
                 group_label: str = 'iBrainCenter',
                 use_tflite: bool = True):
    """
    Plot EEG traces + quality panel for one subject.

    Parameters
    ----------
    base_dir    : root directory that contains info['dir'] (default: IBRAIN_DIR)
    with_events : if True, overlay iBrainCenter event spans/lines and legend
    group_label : string used in the figure title and output filename prefix
    """
    if base_dir is None:
        base_dir = IBRAIN_DIR
    merged = os.path.join(base_dir, info['dir'], 'merged.csv')
    if not os.path.isfile(merged):
        print(f'  [{name}] merged.csv not found — skipping')
        return

    print(f'  [{name}] loading…', end=' ', flush=True)
    time_us_ds, data_ds = load_merged_csv(merged, downsample=ds)
    time_us_full, data_full = load_merged_csv(merged)

    t_dt = np.array([us_to_local_dt(u) for u in time_us_ds])
    n_ch = data_ds.shape[1]
    dur_s = (time_us_ds[-1] - time_us_ds[0]) / 1e6
    print(f'{len(t_dt)} ds-pts, {dur_s:.0f} s  '
          f'({t_dt[0].strftime("%H:%M:%S")} – {t_dt[-1].strftime("%H:%M:%S")})')

    print(f'  [{name}] computing quality ({QUALITY_WIN_SEC:.0f}s windows)…',
          end=' ', flush=True)
    q_dt, q_overall = compute_quality_windowed(time_us_full, data_full,
                                               win_sec=QUALITY_WIN_SEC)
    print(f'{len(q_dt)} windows')

    print(f'  [{name}] computing qEEG indices (BP {BP_LOW}–{BP_HIGH}Hz, {QEEG_WIN_SEC:.0f}s)…',
          end=' ', flush=True)
    data_filt = bandpass_filter(data_full)
    qeeg_dt, qeeg_filt = compute_qeeg_windowed(time_us_full, data_filt)
    print(f'{len(qeeg_dt)} windows')

    # ── TFLite processing (optional) ─────────────────────────────────────────────
    tfl_data, tfl_time_us   = None, None
    qeeg_tfl_dt, qeeg_tfl  = None, None
    if use_tflite and os.path.isfile(TFLITE_MODEL_PATH):
        print(f'  [{name}] applying TFLite model (resample {FS}→{TFLITE_FS}Hz)…',
              end=' ', flush=True)
        try:
            from scipy.signal import resample_poly
            from math import gcd
            _g   = gcd(int(TFLITE_FS), int(FS))
            _up, _dn = int(TFLITE_FS) // _g, int(FS) // _g
            # Downsample data_filt (N_500, 4) → (N_200, 4)
            data_filt_200 = resample_poly(data_filt, _up, _dn, axis=0).astype(np.float32)
            # Build matching time_us at TFLITE_FS by linear interpolation
            t_orig = np.arange(len(data_filt))
            t_new  = np.arange(len(data_filt_200)) * (_dn / _up)
            tfl_time_us_full = np.interp(t_new, t_orig,
                                         time_us_full[:len(data_filt)]).astype(np.int64)
            tfl_raw    = apply_tflite_windowed(data_filt_200)
            N_tfl      = len(tfl_raw)
            tfl_time_us = tfl_time_us_full[:N_tfl]
            tfl_data   = tfl_raw
            print(f'{N_tfl} samples @ {TFLITE_FS}Hz ({N_tfl / TFLITE_FS:.0f}s)')
            print(f'  [{name}] computing TFLite qEEG…', end=' ', flush=True)
            qeeg_tfl_dt, qeeg_tfl = compute_qeeg_windowed(tfl_time_us, tfl_data,
                                                           fs=TFLITE_FS)
            print(f'{len(qeeg_tfl_dt)} windows')
        except Exception as _exc:
            print(f'\n  [{name}] TFLite skipped: {_exc}')
            tfl_data = None

    # ── Build event lists ────────────────────────────────────────────────────────
    evt_patches, evt_list, cone_stage_dt = [], [], []
    participating_events = []
    if with_events:
        for idx, (label, start_hhmm, dur_min, participants) in enumerate(EVENTS):
            participates = (participants is None) or (name in participants)
            start_dt = hhmm_to_dt(start_hhmm)
            end_dt   = start_dt + datetime.timedelta(minutes=dur_min)
            color    = EVT_COLORS[idx % len(EVT_COLORS)]
            evt_list.append((start_dt, end_dt, label, color, participates))
            evt_patches.append(mpatches.Patch(
                facecolor=color,
                alpha=0.35 if participates else 0.10,
                label=label + ('' if participates else ' (not participant)')))
            if participates:
                participating_events.append((start_dt, end_dt, label, color))
        cone_stage_dt = [hhmm_to_dt(t) for t in CONE_STAGES]

    # ── Derived data ─────────────────────────────────────────────────────────────
    # Quality arrays
    q_arr    = np.array(q_dt) if q_dt else np.array([])
    q_median = np.median(q_overall, axis=1)
    q_p25    = np.percentile(q_overall, 25, axis=1)
    q_p75    = np.percentile(q_overall, 75, axis=1)
    low_qual = q_median < QUALITY_THRESHOLD

    INDEX_KEYS  = ['focus', 'flow', 'calm', 'relaxation']
    IDX_LABELS  = ['Focus', 'Flow', 'Calm', 'Relax']
    IDX_COLORS  = {'focus':       '#e6194b', 'flow':        '#3cb44b',
                   'calm':        '#4363d8', 'relaxation':  '#f58231',
                   'restfulness': '#6f42c1', 'engagement':  '#00a6c8'}
    qeeg_t_arr = np.array(qeeg_dt)

    # quality mask (1-to-1 with 5s qEEG windows)
    n_q, n_hm = len(q_median), qeeg_filt[INDEX_KEYS[0]].shape[0]
    qual_mask  = low_qual if n_q == n_hm else np.zeros(n_hm, dtype=bool)

    # ── 5s channel-median series (quality-masked) for smooth trend ────────────
    SMOOTH_WIN = max(1, int(30.0 / QEEG_WIN_SEC))   # 6 × 5s = 30s
    smooth_trend = {}
    for k in INDEX_KEYS:
        series = np.where(qual_mask, np.nan, np.median(qeeg_filt[k], axis=1))
        smooth_trend[k] = (pd.Series(series)
                           .rolling(SMOOTH_WIN, center=True, min_periods=1)
                           .mean().to_numpy())
    smooth_trend['restfulness'] = (smooth_trend['calm'] + smooth_trend['relaxation']) / 2
    smooth_trend['engagement']  = smooth_trend['focus'] - smooth_trend['restfulness']

    # ── Heatmap: bin into 30s medians, then Δ vs baseline ────────────────────
    HEATMAP_BIN_SEC = 30
    bin_size = max(1, int(HEATMAP_BIN_SEC / QEEG_WIN_SEC))   # = 6 windows
    n_bins   = n_hm // bin_size
    bin_t    = []
    heatmap_abs = np.full((4, n_bins), np.nan)
    for b in range(n_bins):
        sl    = slice(b * bin_size, (b + 1) * bin_size)
        good  = ~qual_mask[sl]
        mid   = qeeg_t_arr[b * bin_size + bin_size // 2]
        bin_t.append(mid)
        for idx_i, k in enumerate(INDEX_KEYS):
            vals = np.median(qeeg_filt[k][sl], axis=1)   # per-window ch median
            gv   = vals[good]
            if gv.size > 0:
                heatmap_abs[idx_i, b] = float(np.median(gv))

    bin_t_arr = np.array(bin_t)
    # baseline = bins before first participating event (or first 20% if no events)
    if with_events and participating_events:
        first_evt_dt  = min(e[0] for e in participating_events)
        bl_mask_bin   = bin_t_arr < first_evt_dt
    else:
        bl_mask_bin   = np.zeros(n_bins, dtype=bool)
        bl_mask_bin[:max(1, n_bins // 5)] = True
    heatmap_delta = np.full_like(heatmap_abs, np.nan)
    for idx_i in range(4):
        bl_vals = heatmap_abs[idx_i, bl_mask_bin]
        bl_ref  = float(np.nanmedian(bl_vals)) if np.any(~np.isnan(bl_vals)) else 0.0
        heatmap_delta[idx_i] = heatmap_abs[idx_i] - bl_ref
    heatmap_delta_ma = np.ma.masked_invalid(heatmap_delta)

    # ── Block-level delta bar chart (quality-masked) ──────────────────────────
    block_deltas = []
    if with_events and len(participating_events) > 0 and len(qeeg_t_arr) > 0:
        good_qual     = ~qual_mask
        first_evt_dt  = min(e[0] for e in participating_events)
        baseline_mask = (qeeg_t_arr < first_evt_dt) & good_qual
        for (start_dt, end_dt, label, color) in participating_events:
            block_mask = (qeeg_t_arr >= start_dt) & (qeeg_t_arr < end_dt) & good_qual
            per_index  = {}
            for k in INDEX_KEYS:
                scores = qeeg_filt[k]
                if baseline_mask.sum() > 0 and block_mask.sum() > 0:
                    bl   = scores[baseline_mask].mean(axis=0)
                    blk  = scores[block_mask].mean(axis=0)
                    ch_d = blk - bl
                    per_index[k] = (float(ch_d.mean()), float(ch_d.std()))
                else:
                    per_index[k] = (0.0, 0.0)
            block_deltas.append((label, color, per_index))

    # ── TFLite derived data ───────────────────────────────────────────────────────
    has_tflite        = (qeeg_tfl is not None and len(qeeg_tfl_dt) > 0)
    tfl_t_arr         = np.array(qeeg_tfl_dt) if has_tflite else np.array([])
    tfl_smooth_trend  = {}
    tfl_heatmap_delta = None
    tfl_heatmap_delta_ma = None
    tfl_bin_t_arr     = np.array([])
    n_tfl_bins        = 0
    tfl_block_deltas  = []

    if has_tflite:
        n_tfl_hm = qeeg_tfl[INDEX_KEYS[0]].shape[0]
        n_tfl_q  = len(q_median)
        tfl_qual_mask = (low_qual[:n_tfl_hm] if n_tfl_q >= n_tfl_hm
                         else np.zeros(n_tfl_hm, dtype=bool))

        for k in INDEX_KEYS:
            series = np.where(tfl_qual_mask, np.nan,
                              np.median(qeeg_tfl[k], axis=1))
            tfl_smooth_trend[k] = (pd.Series(series)
                                   .rolling(SMOOTH_WIN, center=True, min_periods=1)
                                   .mean().to_numpy())
        tfl_smooth_trend['restfulness'] = (tfl_smooth_trend['calm'] +
                                           tfl_smooth_trend['relaxation']) / 2
        tfl_smooth_trend['engagement']  = (tfl_smooth_trend['focus'] -
                                           tfl_smooth_trend['restfulness'])

        n_tfl_bins    = n_tfl_hm // bin_size
        tfl_bin_t_list = []
        tfl_heatmap_abs = np.full((4, n_tfl_bins), np.nan)
        for b in range(n_tfl_bins):
            sl   = slice(b * bin_size, (b + 1) * bin_size)
            good = ~tfl_qual_mask[sl]
            mid  = tfl_t_arr[b * bin_size + bin_size // 2]
            tfl_bin_t_list.append(mid)
            for idx_i, k in enumerate(INDEX_KEYS):
                vals = np.median(qeeg_tfl[k][sl], axis=1)
                gv   = vals[good]
                if gv.size > 0:
                    tfl_heatmap_abs[idx_i, b] = float(np.median(gv))
        tfl_bin_t_arr = np.array(tfl_bin_t_list)

        if with_events and participating_events:
            first_evt_dt = min(e[0] for e in participating_events)
            tfl_bl_mask  = tfl_bin_t_arr < first_evt_dt
        else:
            tfl_bl_mask = np.zeros(n_tfl_bins, dtype=bool)
            tfl_bl_mask[:max(1, n_tfl_bins // 5)] = True
        tfl_heatmap_delta = np.full_like(tfl_heatmap_abs, np.nan)
        for idx_i in range(4):
            bl_vals = tfl_heatmap_abs[idx_i, tfl_bl_mask]
            bl_ref  = float(np.nanmedian(bl_vals)) if np.any(~np.isnan(bl_vals)) else 0.0
            tfl_heatmap_delta[idx_i] = tfl_heatmap_abs[idx_i] - bl_ref
        tfl_heatmap_delta_ma = np.ma.masked_invalid(tfl_heatmap_delta)

        if with_events and len(participating_events) > 0:
            good_qual_tfl     = ~tfl_qual_mask
            first_evt_dt      = min(e[0] for e in participating_events)
            baseline_mask_tfl = (tfl_t_arr < first_evt_dt) & good_qual_tfl
            for (start_dt, end_dt, label, color) in participating_events:
                block_mask_tfl = ((tfl_t_arr >= start_dt) & (tfl_t_arr < end_dt)
                                  & good_qual_tfl)
                per_index = {}
                for k in INDEX_KEYS:
                    scores = qeeg_tfl[k]
                    if baseline_mask_tfl.sum() > 0 and block_mask_tfl.sum() > 0:
                        bl   = scores[baseline_mask_tfl].mean(axis=0)
                        blk  = scores[block_mask_tfl].mean(axis=0)
                        ch_d = blk - bl
                        per_index[k] = (float(ch_d.mean()), float(ch_d.std()))
                    else:
                        per_index[k] = (0.0, 0.0)
                tfl_block_deltas.append((label, color, per_index))

    # ── Figure layout (GridSpec) ──────────────────────────────────────────────
    has_bar     = bool(block_deltas)
    has_tfl_bar = has_tflite and bool(tfl_block_deltas)
    height_ratios = ([2.5] * n_ch
                     + [1.5]            # quality
                     + [2.2]            # heatmap BP (Δ vs baseline, 30s bins)
                     + [2.1]            # trend BP summary panel
                     + ([2.2] if has_tflite else [])   # heatmap TFLite
                     + ([2.1] if has_tflite else [])   # trend TFLite
                     + ([2.5] if has_bar else [])
                     + ([2.5] if has_tfl_bar else []))
    n_rows = len(height_ratios)
    fig = plt.figure(figsize=(22, sum(hr * 0.85 for hr in height_ratios) + 1.8))
    gs  = gridspec.GridSpec(n_rows, 1, figure=fig,
                            height_ratios=height_ratios, hspace=0.32)

    ax_eeg = []
    for ch_i in range(n_ch):
        ax = fig.add_subplot(gs[ch_i], sharex=ax_eeg[0] if ax_eeg else None)
        ax_eeg.append(ax)
    _row = n_ch
    ax_quality     = fig.add_subplot(gs[_row], sharex=ax_eeg[0]); _row += 1
    ax_heatmap     = fig.add_subplot(gs[_row], sharex=ax_eeg[0]); _row += 1
    ax_trend       = fig.add_subplot(gs[_row], sharex=ax_eeg[0]); _row += 1
    ax_tfl_heatmap = (fig.add_subplot(gs[_row], sharex=ax_eeg[0])
                      if has_tflite else None)
    if has_tflite:
        _row += 1
    ax_tfl_trend   = (fig.add_subplot(gs[_row], sharex=ax_eeg[0])
                      if has_tflite else None)
    if has_tflite:
        _row += 1
    ax_bar     = fig.add_subplot(gs[_row]) if has_bar     else None
    if has_bar:
        _row += 1
    ax_tfl_bar = fig.add_subplot(gs[_row]) if has_tfl_bar else None

    tflite_note = '  |  +TFLite comparison' if has_tflite else ''
    evt_note = 'Event Marker Verification' if with_events else 'EEG Overview'
    fig.suptitle(
        f'{group_label} — {info["sn"]}  |  {evt_note}{tflite_note}\n'
        f'EEG (ds×{ds}=1 pt/s)  |  Quality: flat+spectrum, '
        f'{QUALITY_WIN_SEC:.0f}s windows @ {FS}Hz  |  '
        f'qEEG: BP {BP_LOW}–{BP_HIGH}Hz, ch median, {QEEG_WIN_SEC:.0f}s windows',
        fontsize=12, fontweight='bold',
    )

    # ── EEG panels ───────────────────────────────────────────────────────────────
    for ch_i, ax in enumerate(ax_eeg):
        ax.plot(t_dt, data_ds[:, ch_i], color='#444444', lw=0.4, alpha=0.8)
        _overlay_events(ax, evt_list, cone_stage_dt)
        ax.set_ylim(-100, 100)
        ax.set_ylabel(f'ch{ch_i+1}\n(µV)', fontsize=8)
        plt.setp(ax.get_xticklabels(), visible=False)
    if with_events:
        for start_dt, end_dt, label, color, participates in evt_list:
            if participates:
                ypos = ax_eeg[0].get_ylim()[1]
                ax_eeg[0].text(start_dt, ypos * 0.92,
                               label.replace(' ', '\n'),
                               fontsize=6, color=color, va='top',
                               rotation=0, clip_on=True)

    # ── Layer 1: Quality panel ────────────────────────────────────────────────────
    if len(q_arr) > 0:
        ax_quality.fill_between(q_arr, q_p25, q_p75,
                                alpha=0.22, color='steelblue',
                                label='IQR (ch 25–75%)')
        ax_quality.plot(q_arr, q_median,
                        color='#1a1a1a', lw=1.5, label='ch median')
        ax_quality.fill_between(q_arr, 0, 1.05, where=low_qual,
                                color='red', alpha=0.18,
                                label='below threshold')
        ax_quality.axhline(QUALITY_THRESHOLD, color='k', lw=0.8,
                           ls='--', alpha=0.5,
                           label=f'threshold {QUALITY_THRESHOLD:.2f}')
    _overlay_events(ax_quality, evt_list, cone_stage_dt)
    ax_quality.set_ylim(0, 1.05)
    ax_quality.set_ylabel('Quality\n(ch median)', fontsize=8)
    ax_quality.legend(loc='upper right', fontsize=7, ncol=4)
    plt.setp(ax_quality.get_xticklabels(), visible=False)

    # ── Layer 2: Index heatmap (30s bins, Δ vs baseline) ─────────────────────────
    if n_bins > 0:
        from matplotlib.colors import LinearSegmentedColormap
        cmap_hm = LinearSegmentedColormap.from_list(
            'OrgPur', ['#5e3c99', '#f7f7f7', '#e66101'])
        cmap_hm.set_bad(color='#aaaaaa')
        bin_t_num  = mdates.date2num(bin_t_arr)
        dt_h       = (float(np.diff(bin_t_num).mean()) / 2) if n_bins > 1 \
                     else (HEATMAP_BIN_SEC / 86400 / 2)
        t_edges    = np.concatenate([[bin_t_num[0] - dt_h],
                                     (bin_t_num[:-1] + bin_t_num[1:]) / 2,
                                     [bin_t_num[-1] + dt_h]])
        y_edges    = np.arange(5) - 0.5
        v_abs      = max(0.3, float(np.nanpercentile(np.abs(heatmap_delta), 95)))
        pcm = ax_heatmap.pcolormesh(t_edges, y_edges, heatmap_delta_ma,
                                    cmap=cmap_hm, vmin=-v_abs, vmax=v_abs,
                                    shading='flat')
        plt.colorbar(pcm, ax=ax_heatmap, pad=0.005, fraction=0.015,
                     label=f'Δ Index  (−{v_abs:.1f} → +{v_abs:.1f})')
        for start_dt, end_dt, label, color, participates in evt_list:
            if participates:
                ax_heatmap.axvline(mdates.date2num(start_dt),
                                   color=color, lw=1.2, ls='--', alpha=0.8)
    ax_heatmap.set_yticks([0, 1, 2, 3])
    ax_heatmap.set_yticklabels(IDX_LABELS, fontsize=8)
    ax_heatmap.set_ylabel('qEEG Δ\n(vs baseline)', fontsize=8)
    ax_heatmap.xaxis_date()
    ax_heatmap.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_heatmap.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 5)))
    ax_heatmap.grid(False)
    plt.setp(ax_heatmap.get_xticklabels(), visible=False)

    # ── Trend panel (30s smooth, reduced summary) ─────────────────────────────
    # Full four-index detail is already shown in the heatmap/bar layers.  Keep
    # this panel focused on the easiest-to-read state trajectory.
    TREND_KEYS   = ['focus', 'restfulness', 'engagement']
    TREND_LABELS = ['Focus', 'Restfulness', 'Engagement']
    TREND_LW     = [1.9, 2.2, 2.0]
    TREND_LS     = ['-', '--', ':']
    if len(qeeg_t_arr) > 0:
        for k, lbl, lw, ls in zip(TREND_KEYS, TREND_LABELS, TREND_LW, TREND_LS):
            ax_trend.plot(qeeg_t_arr, smooth_trend[k],
                          color=IDX_COLORS[k], lw=lw, ls=ls, alpha=0.92,
                          label=lbl)
        ax_trend.axhline(0, color='k', lw=0.5, ls=':')
        if with_events and participating_events:
            first_evt_dt = min(e[0] for e in participating_events)
            ax_trend.axvspan(qeeg_t_arr[0], first_evt_dt,
                             color='grey', alpha=0.08, label='baseline')
    _overlay_events(ax_trend, evt_list, cone_stage_dt)
    ax_trend.set_ylim(-1.1, 1.1)
    ax_trend.set_ylabel('Summary\n(30s smooth)', fontsize=8)
    ax_trend.legend(loc='upper right', fontsize=8, ncol=3,
                    framealpha=0.85, handlelength=2.8)
    ax_trend.set_xlabel('Local Time (UTC+8, HH:MM)', fontsize=10)

    # ── TFLite heatmap (Δ vs baseline, 30s bins) ─────────────────────────────────
    if ax_tfl_heatmap is not None:
        if n_tfl_bins > 0 and tfl_heatmap_delta_ma is not None:
            from matplotlib.colors import LinearSegmentedColormap
            cmap_hm2 = LinearSegmentedColormap.from_list(
                'OrgPur', ['#5e3c99', '#f7f7f7', '#e66101'])
            cmap_hm2.set_bad(color='#aaaaaa')
            tfl_bin_t_num = mdates.date2num(tfl_bin_t_arr)
            dt_h2 = (float(np.diff(tfl_bin_t_num).mean()) / 2) if n_tfl_bins > 1 \
                     else (HEATMAP_BIN_SEC / 86400 / 2)
            t_edges2 = np.concatenate([[tfl_bin_t_num[0] - dt_h2],
                                        (tfl_bin_t_num[:-1] + tfl_bin_t_num[1:]) / 2,
                                        [tfl_bin_t_num[-1] + dt_h2]])
            y_edges2 = np.arange(5) - 0.5
            v_abs2   = max(0.3, float(np.nanpercentile(
                np.abs(tfl_heatmap_delta[~np.isnan(tfl_heatmap_delta)]), 95)))
            pcm2 = ax_tfl_heatmap.pcolormesh(t_edges2, y_edges2, tfl_heatmap_delta_ma,
                                              cmap=cmap_hm2, vmin=-v_abs2, vmax=v_abs2,
                                              shading='flat')
            plt.colorbar(pcm2, ax=ax_tfl_heatmap, pad=0.005, fraction=0.015,
                         label=f'Δ Index  (−{v_abs2:.1f} → +{v_abs2:.1f})')
            for start_dt, end_dt, label, color, participates in evt_list:
                if participates:
                    ax_tfl_heatmap.axvline(mdates.date2num(start_dt),
                                           color=color, lw=1.2, ls='--', alpha=0.8)
        ax_tfl_heatmap.set_yticks([0, 1, 2, 3])
        ax_tfl_heatmap.set_yticklabels(IDX_LABELS, fontsize=8)
        ax_tfl_heatmap.set_ylabel('TFLite qEEG Δ\n(vs baseline)', fontsize=8)
        ax_tfl_heatmap.xaxis_date()
        ax_tfl_heatmap.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax_tfl_heatmap.xaxis.set_major_locator(
            mdates.MinuteLocator(byminute=range(0, 60, 5)))
        ax_tfl_heatmap.grid(False)
        plt.setp(ax_tfl_heatmap.get_xticklabels(), visible=False)

    # ── TFLite trend panel ────────────────────────────────────────────────────────
    if ax_tfl_trend is not None:
        if len(tfl_t_arr) > 0:
            for k, lbl, lw, ls in zip(TREND_KEYS, TREND_LABELS, TREND_LW, TREND_LS):
                ax_tfl_trend.plot(tfl_t_arr, tfl_smooth_trend[k],
                                  color=IDX_COLORS[k], lw=lw, ls=ls, alpha=0.92,
                                  label=lbl)
            ax_tfl_trend.axhline(0, color='k', lw=0.5, ls=':')
            if with_events and participating_events:
                first_evt_dt = min(e[0] for e in participating_events)
                ax_tfl_trend.axvspan(tfl_t_arr[0], first_evt_dt,
                                     color='grey', alpha=0.08, label='baseline')
        _overlay_events(ax_tfl_trend, evt_list, cone_stage_dt)
        ax_tfl_trend.set_ylim(-1.1, 1.1)
        ax_tfl_trend.set_ylabel('TFLite Summary\n(30s smooth)', fontsize=8)
        ax_tfl_trend.legend(loc='upper right', fontsize=8, ncol=3,
                            framealpha=0.85, handlelength=2.8)
        ax_tfl_trend.set_xlabel('Local Time (UTC+8, HH:MM)', fontsize=10)
    # ── Layer 3: Block-level delta bar chart ──────────────────────────────────────
    if ax_bar is not None and block_deltas:
        n_evt  = len(block_deltas)
        n_idx  = len(INDEX_KEYS)
        bar_w  = 0.8 / n_idx
        for idx_i, idx_name in enumerate(INDEX_KEYS):
            x_off = (idx_i - n_idx / 2 + 0.5) * bar_w
            means = [block_deltas[e][2][idx_name][0] for e in range(n_evt)]
            stds  = [block_deltas[e][2][idx_name][1] for e in range(n_evt)]
            ax_bar.bar(np.arange(n_evt) + x_off, means, width=bar_w,
                       color=IDX_COLORS[idx_name], alpha=0.82,
                       label=idx_name.capitalize(),
                       yerr=stds, capsize=3,
                       error_kw={'lw': 1.0}, zorder=3)
        ax_bar.axhline(0, color='k', lw=0.8)
        ax_bar.set_xticks(np.arange(n_evt))
        ax_bar.set_xticklabels([bd[0] for bd in block_deltas],
                               rotation=25, ha='right', fontsize=8)
        ax_bar.set_ylabel('BP Δ Index\n(vs baseline)', fontsize=8)
        ax_bar.set_xlabel('Event Block', fontsize=9)
        ax_bar.legend(loc='upper right', fontsize=7, ncol=n_idx)
        ax_bar.grid(True, alpha=0.2, axis='y')
        ax_bar.set_xlim(-0.5, n_evt - 0.5)

    # ── TFLite block-level delta bar chart ───────────────────────────────────────
    if ax_tfl_bar is not None and tfl_block_deltas:
        n_evt_t = len(tfl_block_deltas)
        n_idx   = len(INDEX_KEYS)
        bar_w_t = 0.8 / n_idx
        for idx_i, idx_name in enumerate(INDEX_KEYS):
            x_off = (idx_i - n_idx / 2 + 0.5) * bar_w_t
            means = [tfl_block_deltas[e][2][idx_name][0] for e in range(n_evt_t)]
            stds  = [tfl_block_deltas[e][2][idx_name][1] for e in range(n_evt_t)]
            ax_tfl_bar.bar(np.arange(n_evt_t) + x_off, means, width=bar_w_t,
                           color=IDX_COLORS[idx_name], alpha=0.82,
                           label=idx_name.capitalize(),
                           yerr=stds, capsize=3,
                           error_kw={'lw': 1.0}, zorder=3)
        ax_tfl_bar.axhline(0, color='k', lw=0.8)
        ax_tfl_bar.set_xticks(np.arange(n_evt_t))
        ax_tfl_bar.set_xticklabels([bd[0] for bd in tfl_block_deltas],
                                   rotation=25, ha='right', fontsize=8)
        ax_tfl_bar.set_ylabel('TFLite Δ Index\n(vs baseline)', fontsize=8)
        ax_tfl_bar.set_xlabel('Event Block', fontsize=9)
        ax_tfl_bar.legend(loc='upper right', fontsize=7, ncol=n_idx)
        ax_tfl_bar.grid(True, alpha=0.2, axis='y')
        ax_tfl_bar.set_xlim(-0.5, n_evt_t - 0.5)

    # ── Legend + save ─────────────────────────────────────────────────────────────
    if with_events and evt_patches:
        fig.legend(handles=evt_patches, loc='lower center', ncol=4,
                   fontsize=8, bbox_to_anchor=(0.5, -0.02), framealpha=0.9)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            fig.tight_layout(rect=[0, 0.04, 1, 1])
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            fig.tight_layout()

    # Lock all sharex panels to the EEG recording time range (after tight_layout
    # so pcolormesh/xaxis_date autoscaling does not override the limit).
    ax_eeg[0].set_xlim(t_dt[0], t_dt[-1])

    suffix = '_tflite' if use_tflite else '_bp'
    outpath = os.path.join(outdir, f'{name}_{info["sn"]}_eeg{suffix}.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'       → {outpath}')


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Plot EEG signal + quality for iBrainCenter and YoGa subjects.')
    p.add_argument('--ibrain-outdir',
                   default=os.path.join(IBRAIN_DIR, 'event_verification'),
                   metavar='DIR', help='Output directory for iBrainCenter PNGs.')
    p.add_argument('--yoga-outdir',
                   default=os.path.join(YOGA_DIR, 'eeg_overview'),
                   metavar='DIR', help='Output directory for YoGa PNGs.')
    p.add_argument('--ds', type=int, default=500, metavar='N',
                   help='Downsample factor for EEG trace (default 500 → 1 pt/s).')
    p.add_argument('--no-tflite', action='store_true',
                   help='Skip TFLite model processing and comparison panels.')
    p.add_argument('--hardy2-analysis', action='store_true',
                   help='Only run Hardy_2(SN036) band/index plots with custom event markers.')
    p.add_argument('--hardy2-outdir',
                   default=os.path.join(IBRAIN_DIR, 'event_verification'),
                   metavar='DIR', help='Output directory for Hardy_2 analysis figures.')
    return p.parse_args()


def main():
    args = parse_args()

    if args.hardy2_analysis:
        plot_hardy2_band_and_indices(args.hardy2_outdir, win_sec=QEEG_WIN_SEC)
        print('\nDone.')
        return

    os.makedirs(args.ibrain_outdir, exist_ok=True)
    os.makedirs(args.yoga_outdir,   exist_ok=True)

    print('Event times (UTC µs) for verification:')
    for label, hhmm, dur, _ in EVENTS:
        us = hhmm_to_us(hhmm)
        print(f'  {hhmm}  {label:<26s}  {us}')

    print(f'\n── iBrainCenter → {args.ibrain_outdir}')
    for name, info in SUBJECTS.items():
        plot_subject(name, info, args.ibrain_outdir, args.ds,
                     base_dir=IBRAIN_DIR, with_events=True,
                     group_label='iBrainCenter',
                     use_tflite=not args.no_tflite)

    print(f'\n── YoGa → {args.yoga_outdir}')
    for name, info in YOGA_SUBJECTS.items():
        plot_subject(name, info, args.yoga_outdir, args.ds,
                     base_dir=YOGA_DIR, with_events=False,
                     group_label='YoGa',
                     use_tflite=not args.no_tflite)

    print('\nDone.')


if __name__ == '__main__':
    main()
