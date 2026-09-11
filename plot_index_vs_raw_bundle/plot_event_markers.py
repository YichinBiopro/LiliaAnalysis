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
import json
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from lilia.windowing import continuous_slices, plot_breaks, finite_runs
import numpy as np
import pandas as pd
from lilia.quality_audit import plot_diagnostic_markers, diagnostic_summary, diagnostic_label
from lilia.quality import (
    get_eeg_quality_index_v2_parametric,
    get_ibrain_device_eeg_quality_v2_params,
)
from lilia.qeeg import compute_qeeg_indices
from lilia.event_qeeg import analyze_recording, summarize_branch
from lilia.event_qeeg_io import write_event_qeeg_table, json_safe
from lilia.provenance import file_sha256
from lilia.io import read_lilia_frame, load_merged_csv as _load_merged_csv_shared, bandpass_filter as _bandpass_filter_shared, read_abs_time_offset as _read_abs_time_offset_shared
from lilia.time_utils import hhmm_to_local_dt, hhmm_to_utc_us, utc_us_to_local_dt
from lilia.pathing import get_project_root
from lilia.tflite import apply_tflite_windowed as _apply_tflite_shared

# ── Session metadata ───────────────────────────────────────────────────────────
SESSION_DATE = datetime.date(2026, 5, 12)
TZ_OFFSET_H  = 8          # Asia/Taipei = UTC+8
EPOCH        = datetime.datetime(1970, 1, 1)

BASE_DIR     = get_project_root()
IBRAIN_DIR   = os.path.join(BASE_DIR, 'iBrainCenter')
YOGA_DIR     = os.path.join(BASE_DIR, 'YoGa')

# Import shared constants; override only those specific to this module
from lilia.constants import FS, QUALITY_THRESHOLD, TFLITE_FS, TFLITE_WIN

QUALITY_WIN_SEC  = 5.0          # window length for quality scorer (5 s)
QUALITY_STEP_SEC = 5.0          # non-overlapping windows (matches app refresh)
QUALITY_WIN_SEC_LONG  = 30.0   # second quality scorer window (30 s)
# Device-calibrated preset for the iBrainCenter 4-ch headset. The previous
# `flat_spectrum_only` preset was tuned for cleaner EEG and scored normal data
# from this device at ~0.5 (40-60% of every recording rejected); the calibrated
# preset puts clean data at ~0.7-0.9 so the 0.5 threshold separates clean from
# artefact rather than bisecting the clean distribution. See eeg_quality_v2.py.
QUALITY_PARAMS    = get_ibrain_device_eeg_quality_v2_params()

TFLITE_MODEL_PATH = os.path.join(BASE_DIR, 'tiny_v4_optimized.tflite')

BP_LOW       = 0.5    # Hz — bandpass lower cutoff
BP_HIGH      = 45.0   # Hz — bandpass upper cutoff
QEEG_WIN_SEC = 5.0    # window for qEEG indices (non-overlapping, seconds)
# Fixed colorbar half-range (Δ index) shared by the BP and TFLite heatmaps so
# both panels use an identical −VABS…+VABS scale for fair visual comparison.
HEATMAP_DELTA_VABS = 0.6

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

from lilia.hardy2 import BANDS as HARDY2_BANDS


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


# Preserve imports used by external legacy plotting scripts.
def load_merged_csv(*args, **kwargs):
    return _load_merged_csv_shared(*args, **kwargs)


def bandpass_filter(*args, **kwargs):
    return _bandpass_filter_shared(*args, **kwargs)


# ── Time helpers ───────────────────────────────────────────────────────────────

def hhmm_to_us(hhmm: str) -> int:
    """Convert 'HH:MM' on SESSION_DATE (local TZ) to UTC Unix microseconds."""
    return hhmm_to_utc_us(hhmm, SESSION_DATE, TZ_OFFSET_H)


def us_to_local_dt(us: int) -> datetime.datetime:
    """Convert UTC Unix µs → local datetime (UTC+8, naive for matplotlib)."""
    return utc_us_to_local_dt(us, TZ_OFFSET_H)


def hhmm_to_dt(hhmm: str) -> datetime.datetime:
    """'HH:MM' on SESSION_DATE (local) → naive local datetime."""
    return hhmm_to_local_dt(hhmm, SESSION_DATE)


# ── CSV loading ────────────────────────────────────────────────────────────────

def read_abs_time_offset(path: str) -> int:
    """Wrapper around shared lilia.io.read_abs_time_offset for backward compatibility."""
    return _read_abs_time_offset_shared(path)


# ── EEG quality (windowed) ─────────────────────────────────────────────────────

def compute_quality_windowed(time_us: np.ndarray, data: np.ndarray,
                             win_sec: float = QUALITY_WIN_SEC, fs: float = FS):
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
    win  = int(win_sec * fs)
    step = win   # non-overlapping
    n    = len(data)

    q_dt      = []
    q_overall = []

    for start in range(0, n - win + 1, step):
        seg = data[start : start + win]            # (win, n_ch)
        mid_us = int(time_us[start + win // 2])
        result = get_eeg_quality_index_v2_parametric(
            seg.T.astype(np.float64),              # (n_ch, win)
            fs=fs,
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
                          fs: float = FS, windows=None):
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

    if windows is not None:
        windows.validate(n, fs, win, win)
    starts = windows.starts if windows is not None else range(0, n - win + 1, win)
    for start in starts:
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
    """Wrapper around shared lilia.tflite.apply_tflite_windowed for backward compatibility."""
    return _apply_tflite_shared(data, tflite_path, tflite_win=TFLITE_WIN)


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


def _time_mask(time_arr: np.ndarray,
               start_dt: datetime.datetime | None = None,
               end_dt: datetime.datetime | None = None) -> np.ndarray:
    """Build a half-open [start_dt, end_dt) mask on a datetime array."""
    mask = np.ones(len(time_arr), dtype=bool)
    if start_dt is not None:
        mask &= time_arr >= start_dt
    if end_dt is not None:
        mask &= time_arr < end_dt
    return mask


def _build_pre_event_rest_specs(evt_list: list) -> list:
    """Map each participating event to the immediately preceding non-event span."""
    specs = []
    prev_end = None
    for start_dt, end_dt, label, color, participates in evt_list:
        if participates:
            specs.append((start_dt, end_dt, label, color, prev_end, start_dt))
        if prev_end is None or end_dt > prev_end:
            prev_end = end_dt
    return specs


def _piecewise_event_delta(time_arr: np.ndarray,
                           abs_values: np.ndarray,
                           event_specs: list) -> np.ndarray:
    """Apply per-event baseline subtraction using each event's pre-event rest."""
    delta = np.full_like(abs_values, np.nan)
    for start_dt, end_dt, _label, _color, bl_start, bl_end in event_specs:
        baseline_mask = _time_mask(time_arr, bl_start, bl_end)
        segment_mask = baseline_mask | _time_mask(time_arr, start_dt, end_dt)
        if not np.any(segment_mask):
            continue
        for idx_i in range(abs_values.shape[0]):
            bl_vals = abs_values[idx_i, baseline_mask]
            if np.any(~np.isnan(bl_vals)):
                bl_ref = float(np.nanmedian(bl_vals))
                delta[idx_i, segment_mask] = abs_values[idx_i, segment_mask] - bl_ref
    return delta


def _draw_baseline_spans(ax: plt.Axes,
                         time_arr: np.ndarray,
                         baseline_mode: str,
                         participating_events: list,
                         pre_event_rest_specs: list) -> None:
    """Highlight the baseline regions used for delta comparisons."""
    if len(time_arr) == 0:
        return
    if baseline_mode == 'session-start':
        if participating_events:
            first_evt_dt = min(e[0] for e in participating_events)
            ax.axvspan(time_arr[0], first_evt_dt,
                       color='grey', alpha=0.08, label='baseline')
        return

    baseline_label_drawn = False
    for _start_dt, _end_dt, _label, _color, bl_start, bl_end in pre_event_rest_specs:
        span_start = time_arr[0] if bl_start is None else max(bl_start, time_arr[0])
        span_end = min(bl_end, time_arr[-1])
        if span_start >= span_end:
            continue
        ax.axvspan(span_start, span_end, color='grey', alpha=0.08,
                   label='baseline' if not baseline_label_drawn else None)
        baseline_label_drawn = True


def _bandpower_from_psd(freqs: np.ndarray, psd: np.ndarray,
                        fmin: float, fmax: float) -> float:
    """Integrate PSD over [fmin, fmax] using trapezoidal rule."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def _compute_hardy2_windowed_metrics(time_us: np.ndarray, data: np.ndarray,
                                     win_sec: float = 5.0, fs: float = FS):
    """Compatibility adapter; supplied data is already filtered by the caller."""
    from lilia.hardy2 import analyze_hardy2
    result = analyze_hardy2(time_us, data, fs=fs, win_sec=win_sec, use_bandpass=False)
    grid = result['grid']
    times = np.array([]) if grid is None else np.array([us_to_local_dt(t) for t in grid.columns['window_center_us']])
    return times, {k: result['metrics'][k] for k in HARDY2_BANDS}, {k: result['metrics'][k] for k in HARDY2_INDEX_KEYS}


def _annotate_hardy2_events(ax: plt.Axes, limits=None) -> None:
    """Add Hardy_2 event lines and labels."""
    y_positions = [0.965, 0.925, 0.965]
    for (label, hhmm, color), ypos in zip(HARDY2_EVENTS, y_positions):
        dt = hhmm_to_dt(hhmm)
        if limits is not None and not limits[0] <= dt <= limits[1]:
            continue
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


def _plot_hardy2_result(result, outdir, win_sec):
    """Plot only source-local runs; period shading follows accepted windows."""
    from lilia.hardy2 import BANDS, INDEX_KEYS
    grid = result['grid']
    times = np.array([us_to_local_dt(t) for t in grid.columns['window_center_us']])
    groups = grid.columns['segment_id']
    colors = ['#dddddd', '#d9ecff', '#ffe3d9', '#e6f7e6']
    artifacts = []

    def shade(ax, key=None):
        for period, color in zip(result['summary']['periods'], colors):
            spans = period['spans']
            for span in spans:
                ax.axvspan(us_to_local_dt(span['start_us']), us_to_local_dt(span['end_us']),
                           color=color, alpha=.26, zorder=0)
            if spans and key is not None:
                widest = max(spans, key=lambda s: s['end_us']-s['start_us'])
                center = us_to_local_dt((widest['start_us']+widest['end_us'])//2)
                ax.text(center, .895, f'{period["name"]}: {period["means"][key]*100:.1f}%',
                        ha='center', va='top', fontsize=10, transform=ax.get_xaxis_transform(),
                        bbox=dict(boxstyle='round,pad=0.24', fc='white', ec='none', alpha=.88))

    def finish(fig, ax, title, stem):
        ax.set_title(title, fontsize=17, fontweight='bold', pad=32)
        ax.set_xlabel('Local Time (UTC+8, HH:MM)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 3)))
        _style_presentation_axis(ax)
        limits = [us_to_local_dt(result['segments'][0]['raw_start_us']),
                  us_to_local_dt(result['segments'][-1]['raw_end_us'])]
        ax.set_xlim(limits)
        _annotate_hardy2_events(ax, limits=limits)
        fig.tight_layout()
        try:
            for extension in ('png', 'svg'):
                path = Path(outdir)/f'{stem}.{extension}'
                fig.savefig(path, dpi=180)
                artifacts.append(path)
        finally:
            plt.close(fig)

    channels = result['channels']['delta'].shape[1]
    for key in BANDS:
        fig, ax = plt.subplots(figsize=(16, 6.3))
        shade(ax, key)
        tx, y = plot_breaks(times, result['metrics'][key], groups)
        sx, sy = plot_breaks(times, result['smooth'][key], groups)
        color = HARDY2_BAND_COLORS[key]
        ax.plot(tx, y, lw=1.1, color=color, alpha=.22, marker='.')
        ax.plot(sx, sy, lw=3., color=color, marker='.', label=f'{key.capitalize()} ratio (5-window smooth)')
        ax.fill_between(tx, 0, y, color=color, alpha=.08)
        ax.text(0, 1.02, f'Median across {channels} channels, {win_sec:g} s windows; quality scoring disabled',
                transform=ax.transAxes, fontsize=10.5, color='#444444')
        ax.set_ylabel('Band Ratio (%)')
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.))
        ax.legend(loc='lower right', frameon=False, fontsize=10)
        finish(fig, ax, f'Hardy_2(SN036) {key.capitalize()} Band Ratio vs Time',
               f'Hardy_2_SN036_{key}_band_ratio_vs_time')

    fig, ax = plt.subplots(figsize=(16, 6.8))
    shade(ax)
    for key in INDEX_KEYS:
        ax.plot(*plot_breaks(times, result['metrics'][key], groups), lw=1., alpha=.2,
                color=HARDY2_INDEX_COLORS[key], marker='.')
        ax.plot(*plot_breaks(times, result['smooth'][key], groups), lw=2.7,
                color=HARDY2_INDEX_COLORS[key], marker='.', label=HARDY2_INDEX_LABELS[key])
    ax.axhline(0, color='k', lw=.8, ls='--', alpha=.5)
    ax.text(0, 1.02, f'Thin: {win_sec:g} s windows; bold: 5-window smooth within valid runs; quality scoring disabled',
            transform=ax.transAxes, fontsize=10.5, color='#444444')
    ax.set_ylabel('Index (−1 to +1)')
    ax.set_ylim(-1.1, 1.1)
    ax.legend(loc='lower right', ncol=4, frameon=False, fontsize=11)
    finish(fig, ax, 'Hardy_2(SN036) Focus / Flow / Calm / Relax vs Time',
           'Hardy_2_SN036_focus_flow_calm_relax_vs_time')
    return artifacts


def plot_hardy2_band_and_indices(outdir: str, win_sec: float = QEEG_WIN_SEC,
                                 use_bandpass: bool = True, csv_path=None):
    """Analyze the special five-band branch and persist source/period audits."""
    from lilia.hardy2 import analyze_hardy2, summarize_periods
    from lilia.hardy2_io import write_hardy2_table
    merged = Path(csv_path) if csv_path is not None else Path(IBRAIN_DIR)/'Hardy_2(SN036)'/'merged.csv'
    Path(outdir).mkdir(parents=True, exist_ok=True)
    audit_path = Path(outdir)/'Hardy_2_SN036_analysis.json'
    audit = {'kind': 'hardy2_analysis', 'schema_version': 1, 'source_path': str(merged.resolve()),
             'status': 'processing', 'quality_state': 'disabled', 'errors': [], 'artifacts': {}}
    try:
        audit['source_id'] = file_sha256(merged)
        source = read_lilia_frame(merged)
        t = source.iloc[:, 0].to_numpy(dtype=np.int64)
        raw = source.iloc[:, 1:].to_numpy(dtype=np.float32)
        params = {'fs': FS, 'win_sec': win_sec, 'step_sec': win_sec, 'index_space': 'raw_samples',
                  'channels': raw.shape[1], 'bands': HARDY2_BANDS, 'use_bandpass': use_bandpass,
                  'bandpass': [BP_LOW, BP_HIGH], 'quality_state': 'disabled',
                  'event_us': [hhmm_to_us(h) for _, h, _ in HARDY2_EVENTS],
                  'period_policy': 'complete_contained_windows', 'smooth_windows': 5,
                  'aggregation': 'median across channels per window, then mean of unsmoothed window medians',
                  'filter_policy': 'source segment; any nonfinite sample rejects filtered segment',
                  'clock_policy': 'source UTC microseconds; display UTC+8; header offset not added'}
        paths = [Path(__file__), *sorted((Path(__file__).parent/'lilia').glob('*.py'))]
        code_id = hashlib.sha256(''.join(file_sha256(p) for p in paths).encode()).hexdigest()
        audit.update(parameters=params, code_sha256=code_id, source_samples=len(t), source_epoch_us=int(t[0]))
        result = analyze_hardy2(t, raw, fs=FS, win_sec=win_sec, use_bandpass=use_bandpass, low=BP_LOW, high=BP_HIGH)
        result['summary'] = summarize_periods(result['grid'], result['metrics'], params['event_us'])
        audit.update({k: result[k] for k in ('segments', 'window_audit', 'summary')})
        audit.update(candidate_windows=len(result['valid']), valid_windows=int(result['valid'].sum()))
        if result['grid'] is not None:
            path = Path(outdir)/'Hardy_2_SN036_metrics.csv'
            write_hardy2_table(path, merged, result, params, code_id)
            for p in (path, Path(str(path)+'.meta.json')):
                audit['artifacts'][p.name] = file_sha256(p)
        if not result['valid'].any():
            raise ValueError('No valid complete Hardy_2 windows; inspect analysis audit')
        for p in _plot_hardy2_result(result, outdir, win_sec):
            audit['artifacts'][p.name] = file_sha256(p)
        for period in result['summary']['periods']:
            print(f'[Hardy_2] {period["name"]}: {len(period["accepted_rows"])} valid windows ({period["status"]})')
        audit['status'] = 'complete'
    except Exception as exc:
        audit['status'] = 'failed'
        audit['errors'].append(f'{type(exc).__name__}: {exc}')
        raise
    finally:
        audit_path.write_text(json.dumps(json_safe(audit), indent=2, allow_nan=False)+'\n')
    print(f'[Hardy_2] outputs and source audit saved: {outdir}')
    return audit

# ── Plotting ───────────────────────────────────────────────────────────────────

def _draw_segment_heatmap(ax, bins, values, cmap, v_abs):
    """One physical rectangle per complete bin; recording gaps remain blank."""
    pcm = None
    for i, item in enumerate(bins):
        edges = [us_to_local_dt(item['start_us']), us_to_local_dt(item['end_us'])]
        pcm = ax.pcolormesh(mdates.date2num(edges), np.arange(5)-.5,
                            values[:, i:i+1], cmap=cmap, vmin=-v_abs, vmax=v_abs, shading='flat')
    return pcm


def plot_subject(name: str, info: dict, outdir: str, ds: int,
                 base_dir: str = None, with_events: bool = True,
                 group_label: str = 'iBrainCenter',
                 use_tflite: bool = True,
                 baseline_mode: str = 'session-start'):
    """
    Plot EEG traces + quality panel for one subject.

    Parameters
    ----------
    base_dir    : root directory that contains info['dir'] (default: IBRAIN_DIR)
    with_events : if True, overlay iBrainCenter event spans/lines and legend
    group_label : string used in the figure title and output filename prefix
    baseline_mode : 'session-start' or 'pre-event-rest' for delta reference
    """
    if not isinstance(ds, int) or ds < 1:
        raise ValueError('Display downsample factor must be a positive integer')
    if baseline_mode not in ('session-start', 'pre-event-rest'):
        raise ValueError('Unknown baseline mode')
    if base_dir is None:
        base_dir = IBRAIN_DIR
    merged = os.path.join(base_dir, info['dir'], 'merged.csv')
    os.makedirs(outdir, exist_ok=True)
    if not os.path.isfile(merged):
        raise FileNotFoundError(f'[{name}] merged.csv not found: {merged}')
    print(f'  [{name}] loading and analyzing source segments…', flush=True)
    raw_frame = read_lilia_frame(merged)
    time_us_full = raw_frame.iloc[:, 0].to_numpy(dtype=np.int64)
    data_full = raw_frame.iloc[:, 1:].to_numpy(dtype=np.float32)
    result = analyze_recording(time_us_full, data_full, fs=FS, win_sec=QEEG_WIN_SEC,
        low=BP_LOW, high=BP_HIGH, scorer=get_eeg_quality_index_v2_parametric,
        quality_params=QUALITY_PARAMS, threshold=QUALITY_THRESHOLD,
        model_path=TFLITE_MODEL_PATH if use_tflite else None,
        model_fs=TFLITE_FS, model_window=TFLITE_WIN)
    segments = continuous_slices(time_us_full, FS)
    ds_idx = np.unique(np.concatenate([np.r_[np.arange(sl.start, sl.stop, ds), sl.stop-1]
                                      for sl in segments]))
    time_us_ds, data_ds = time_us_full[ds_idx], data_full[ds_idx]
    ds_groups = np.searchsorted([sl.stop for sl in segments], ds_idx, side='right')
    t_dt = np.array([us_to_local_dt(u) for u in time_us_ds])
    n_ch = data_full.shape[1]

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
    pre_event_rest_specs = (_build_pre_event_rest_specs(evt_list)
                            if with_events and baseline_mode == 'pre-event-rest'
                            else [])

    events = [{'start_us': hhmm_to_us(hhmm), 'end_us': hhmm_to_us(hhmm)+int(dur*60e6),
               'label': label, 'color': EVT_COLORS[i % len(EVT_COLORS)],
               'participates': participants is None or name in participants}
              for i, (label, hhmm, dur, participants) in enumerate(EVENTS)] if with_events else []
    code_paths = [Path(__file__), *sorted((Path(__file__).parent / 'lilia').glob('*.py'))]
    code_id = hashlib.sha256(''.join(file_sha256(p) for p in code_paths).encode()).hexdigest()
    suffix = '_tflite' if use_tflite else '_bp'
    baseline_tag = baseline_mode.replace('-', '_') if with_events else 'session_start'
    stem = os.path.join(outdir, f'{name}_{info["sn"]}_eeg{suffix}_{baseline_tag}')
    analysis = {'source_id': file_sha256(merged), 'code_sha256': code_id,
                'source_samples': len(time_us_full), 'source_epoch_us': int(time_us_full[0]),
                'segments': result['segments'], 'errors': result['errors'],
                'events': events, 'baseline_mode': baseline_mode, 'branches': {}}
    for branch_name in ('bp', 'tflite'):
        branch = result[branch_name]
        if branch is None:
            continue
        branch['summary'] = summarize_branch(branch, events, baseline_mode)
        parameters = {'fs': FS if branch_name == 'bp' else TFLITE_FS,
            'input_fs': FS, 'win_sec': QEEG_WIN_SEC, 'step_sec': QEEG_WIN_SEC,
            'index_space': 'raw_samples' if branch_name == 'bp' else 'retained_tflite_output',
            'bandpass': [BP_LOW, BP_HIGH], 'quality_params': QUALITY_PARAMS,
            'quality_threshold': QUALITY_THRESHOLD, 'quality_source': 'raw_all_channels_same_physical_interval',
            'quality_channels': n_ch, 'metric_channels': branch['scores']['focus'].shape[1],
            'baseline_mode': baseline_mode, 'events': events,
            'selection_policy': 'complete_containment', 'heatmap_bin_windows': 6,
            'bar_aggregation': 'mean_per_channel_delta_then_channel_mean_std',
            'heatmap_aggregation': 'median_window_channel_medians',
            'model_window': TFLITE_WIN if branch_name == 'tflite' else None,
            'model_sha256': file_sha256(TFLITE_MODEL_PATH) if branch_name == 'tflite' else None}
        table = stem + f'_{branch_name}_metrics.csv'
        write_event_qeeg_table(table, merged, branch, parameters, code_id,
                              result['timeline'] if branch_name == 'tflite' else None)
        analysis['branches'][branch_name] = {'table': os.path.basename(table),
            'table_sha256': file_sha256(table), 'valid_windows': int(branch['valid'].sum()),
            'total_windows': len(branch['grid'].starts), 'summary': branch['summary'],
            'quality_diagnostics': diagnostic_summary(branch['window_audit']),
            'window_audit': branch['window_audit']}
    if result['timeline'] is not None:
        analysis['inference'] = result['timeline'].metadata()
    audit_path = stem + '_analysis.json'
    Path(audit_path).write_text(json.dumps(json_safe(analysis), indent=2, allow_nan=False)+'\n')
    if result['bp'] is None or not result['bp']['valid'].any():
        raise ValueError(f'No quality-valid BP qEEG windows; audit saved: {audit_path}')
    bp = result['bp']
    q_overall = bp['quality']
    q_arr = np.array([us_to_local_dt(u) for u in bp['grid'].columns['window_center_us']])
    q_median, q_p25, q_p75 = np.median(q_overall, axis=1), np.percentile(q_overall,25,axis=1), np.percentile(q_overall,75,axis=1)
    low_qual = ~bp['valid']
    bp_groups = bp['grid'].columns['segment_id']
    INDEX_KEYS = ['focus', 'flow', 'calm', 'relaxation']
    IDX_LABELS = ['Focus', 'Flow', 'Calm', 'Relax']
    IDX_COLORS = {'focus': '#e6194b', 'flow': '#3cb44b', 'calm': '#4363d8',
                  'relaxation': '#f58231', 'restfulness': '#6f42c1', 'engagement': '#00a6c8'}
    qeeg_t_arr = q_arr
    smooth_trend = bp['summary']['smooth']
    heatmap_delta_ma = np.ma.masked_invalid(bp['summary']['heatmap_delta'])
    n_bins = len(bp['summary']['bins'])
    block_deltas = bp['summary']['block_deltas']
    tfl = result['tflite']
    has_tflite = tfl is not None and tfl['valid'].any()
    tfl_t_arr = np.array([us_to_local_dt(u) for u in tfl['grid'].columns['window_center_us']]) if has_tflite else np.array([])
    tfl_smooth_trend = tfl['summary']['smooth'] if has_tflite else {}
    tfl_heatmap_delta_ma = np.ma.masked_invalid(tfl['summary']['heatmap_delta']) if has_tflite else None
    n_tfl_bins = len(tfl['summary']['bins']) if has_tflite else 0
    tfl_block_deltas = tfl['summary']['block_deltas'] if has_tflite else []
    if use_tflite and not has_tflite and not result['errors']:
        result['errors'].append('No quality-valid TFLite qEEG windows')
        Path(audit_path).write_text(json.dumps(json_safe(analysis), indent=2, allow_nan=False)+'\n')

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
                            height_ratios=height_ratios, hspace=0.65, top=0.95, bottom=0.12)

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
        f'EEG display: every {ds} samples plus segment endpoints  |  Raw quality: {diagnostic_label(bp["window_audit"])}, '
        f'{QUALITY_WIN_SEC:.0f}s windows @ {FS}Hz  |  '
        f'qEEG: BP {BP_LOW}–{BP_HIGH}Hz, ch median, {QEEG_WIN_SEC:.0f}s windows',
        fontsize=12, fontweight='bold',
    )
    # Scientific context + cutoff explanation footnote (bottom, below legend row).
    fig.text(
        0.5, 0.008,
        'Indices from relative band powers — Theta 4–8 Hz, Alpha 8–13 Hz, '
        'Beta 13–30 Hz.   Focus: Beta↑ vs Alpha/Theta · Calm/Relax: '
        'Alpha/Theta↑ vs Beta · Flow: Alpha–Theta synchrony.   '
        'Gray cells = unavailable delta (quality / missing baseline); blank spans = recording gaps. Bar error bars show channel SD.',
        ha='center', va='bottom', fontsize=7, color='#555555',
    )

    # ── EEG panels ───────────────────────────────────────────────────────────────
    for ch_i, ax in enumerate(ax_eeg):
        ax.plot(*plot_breaks(t_dt, data_ds[:, ch_i], ds_groups), color='#444444', lw=0.4, alpha=0.8)
        _overlay_events(ax, evt_list, cone_stage_dt)
        ax.set_ylim(-150, 150)
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
        for run in finite_runs(q_overall, bp_groups):
            ax_quality.fill_between(q_arr[run], q_p25[run], q_p75[run], alpha=0.22, color='steelblue')
        ax_quality.plot(*plot_breaks(q_arr, q_median, bp_groups), color='#1a1a1a', lw=1.5, label='ch median')
        for i in np.flatnonzero(low_qual):
            ax_quality.axvspan(us_to_local_dt(bp['grid'].columns['window_start_us'][i]),
                               us_to_local_dt(bp['grid'].columns['window_end_us'][i]), color='red', alpha=.18)
        ax_quality.axhline(QUALITY_THRESHOLD, color='k', lw=.8, ls='--', alpha=.5,
                           label=f'threshold {QUALITY_THRESHOLD:.2f}')
    plot_diagnostic_markers(ax_quality, q_arr, bp['window_audit'], q_overall)
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
        v_abs = HEATMAP_DELTA_VABS
        pcm = _draw_segment_heatmap(ax_heatmap, bp['summary']['bins'], heatmap_delta_ma, cmap_hm, v_abs)
        cbar = plt.colorbar(pcm, ax=ax_heatmap, pad=0.012, fraction=0.015)
        cbar.set_label(f'Δ Index  (−{v_abs:.1f} → +{v_abs:.1f})', labelpad=12)
        cbar.ax.tick_params(pad=4)
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
            ax_trend.plot(*plot_breaks(qeeg_t_arr, smooth_trend[k], bp_groups),
                          color=IDX_COLORS[k], lw=lw, ls=ls, alpha=0.92,
                          label=lbl)
        ax_trend.axhline(0, color='k', lw=0.5, ls=':')
        if with_events and participating_events:
            _draw_baseline_spans(ax_trend, qeeg_t_arr, baseline_mode,
                                 participating_events, pre_event_rest_specs)
    _overlay_events(ax_trend, evt_list, cone_stage_dt)
    ax_trend.margins(y=.1)
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
            v_abs2 = HEATMAP_DELTA_VABS
            pcm2 = _draw_segment_heatmap(ax_tfl_heatmap, tfl['summary']['bins'], tfl_heatmap_delta_ma, cmap_hm2, v_abs2)
            cbar2 = plt.colorbar(pcm2, ax=ax_tfl_heatmap, pad=0.012, fraction=0.015)
            cbar2.set_label(f'Δ Index  (−{v_abs2:.1f} → +{v_abs2:.1f})', labelpad=12)
            cbar2.ax.tick_params(pad=4)
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
                ax_tfl_trend.plot(*plot_breaks(tfl_t_arr, tfl_smooth_trend[k], tfl['grid'].columns['segment_id']),
                                  color=IDX_COLORS[k], lw=lw, ls=ls, alpha=0.92,
                                  label=lbl)
            ax_tfl_trend.axhline(0, color='k', lw=0.5, ls=':')
            if with_events and participating_events:
                _draw_baseline_spans(ax_tfl_trend, tfl_t_arr, baseline_mode,
                                     participating_events, pre_event_rest_specs)
        _overlay_events(ax_tfl_trend, evt_list, cone_stage_dt)
        ax_tfl_trend.margins(y=.1)
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
                       yerr=np.nan_to_num(stds, nan=0.0), capsize=3,
                       error_kw={'lw': 1.0}, zorder=3)
        ax_bar.axhline(0, color='k', lw=0.8)
        ax_bar.set_xticks(np.arange(n_evt))
        ax_bar.set_xticklabels([bd[0] + ('\n(no baseline/event)' if not np.isfinite(bd[2]['focus'][0]) else '')
                                for bd in block_deltas],
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
                           yerr=np.nan_to_num(stds, nan=0.0), capsize=3,
                           error_kw={'lw': 1.0}, zorder=3)
        ax_tfl_bar.axhline(0, color='k', lw=0.8)
        ax_tfl_bar.set_xticks(np.arange(n_evt_t))
        ax_tfl_bar.set_xticklabels([bd[0] + ('\n(no baseline/event)' if not np.isfinite(bd[2]['focus'][0]) else '')
                                    for bd in tfl_block_deltas],
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
            fig.tight_layout(rect=[0, 0.03, 1, 1])

    # Lock all sharex panels to the EEG recording time range (after tight_layout
    # so pcolormesh/xaxis_date autoscaling does not override the limit).
    ax_eeg[0].set_xlim(t_dt[0], t_dt[-1])

    suffix = '_tflite' if use_tflite else '_bp'
    baseline_tag = baseline_mode.replace('-', '_') if with_events else 'session_start'
    outpath = os.path.join(
        outdir, f'{name}_{info["sn"]}_eeg{suffix}_{baseline_tag}.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    fig.savefig(os.path.splitext(outpath)[0] + '.svg')
    plt.close(fig)
    print(f'       → {outpath}')
    if result['errors']:
        raise ValueError('; '.join(result['errors']) + f'; partial outputs and audit saved: {audit_path}')
    return analysis



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
    p.add_argument('--hardy2-csv', metavar='PATH',
                   help='Hardy_2 source CSV (default: iBrainCenter/Hardy_2(SN036)/merged.csv).')
    p.add_argument('--hardy2-outdir',
                   default=os.path.join(IBRAIN_DIR, 'event_verification'),
                   metavar='DIR', help='Output directory for Hardy_2 analysis figures.')
    p.add_argument('--ibrain-baseline-mode',
                   choices=['pre-event-rest', 'session-start'],
                   default='pre-event-rest',
                   help=('Baseline mode for iBrainCenter delta plots: '
                         "'pre-event-rest' uses each event's immediately preceding "
                         "non-event interval; 'session-start' keeps the original "
                         'pre-first-event baseline.'))
    return p.parse_args()


def main():
    args = parse_args()

    if args.hardy2_analysis:
        plot_hardy2_band_and_indices(args.hardy2_outdir, win_sec=QEEG_WIN_SEC, csv_path=args.hardy2_csv)
        print('\nDone.')
        return

    os.makedirs(args.ibrain_outdir, exist_ok=True)
    os.makedirs(args.yoga_outdir,   exist_ok=True)

    print('Event times (UTC µs) for verification:')
    for label, hhmm, dur, _ in EVENTS:
        us = hhmm_to_us(hhmm)
        print(f'  {hhmm}  {label:<26s}  {us}')

    failures = []
    for registry, base_dir, outdir, with_events, group_label, baseline_mode in (
        (SUBJECTS, IBRAIN_DIR, args.ibrain_outdir, True, 'iBrainCenter', args.ibrain_baseline_mode),
        (YOGA_SUBJECTS, YOGA_DIR, args.yoga_outdir, False, 'YoGa', 'session-start')):
        print(f'\n── {group_label} → {outdir}')
        for name, info in registry.items():
            try:
                plot_subject(name, info, outdir, args.ds, base_dir=base_dir,
                    with_events=with_events, group_label=group_label,
                    use_tflite=not args.no_tflite, baseline_mode=baseline_mode)
            except (ValueError, OSError, RuntimeError) as exc:
                failures.append({'group': group_label, 'subject': name, 'error': str(exc)})
                print(f'  FAILED [{group_label}/{name}]: {exc}')
    failure_path = os.path.join(args.ibrain_outdir, 'event_markers_failures.json')
    Path(failure_path).write_text(json.dumps(failures, indent=2)+'\n')
    if failures:
        raise SystemExit(f'{len(failures)} subject analyses failed; see {failure_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
