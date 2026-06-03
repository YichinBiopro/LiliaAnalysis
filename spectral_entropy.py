"""
spectral_entropy.py
===================
Compute EEG band-structure entropy from lilia-format CSV files.

Method
------
1. Split the EEG signal into sliding 2-second windows.
2. Use Welch's method to estimate the power spectral density (PSD).
3. Integrate PSD within five EEG bands to obtain band energies:
   delta, theta, alpha, beta, gamma.
4. Compute total energy across the five bands.
5. Convert band energies into proportions p_k.
6. Compute band entropy:
       BandEn = -sum(p_k * log2(p_k))
7. Optionally compute left-right synchrony from non-zero-lag mutual
   information I(X(t); Y(t+tau)) using small positive delays tau.

CLI usage
---------
    python spectral_entropy.py --csv <path.csv> [--fs 500] [--ch 1]
                               [--win 2] [--step 2] [--out <dir>]
                               [--sync-pair LEFT RIGHT]
                               [--tau-ms 5 10 15 20]
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from eeg_utils import bandpass_filter, load_merged_csv

# ── iBrainCenter event definitions (optional import) ──────────────────────────
try:
    from plot_event_markers import (
        EVENTS as _IBRAIN_EVENTS,
        EVT_COLORS as _IBRAIN_EVT_COLORS,
        hhmm_to_dt as _hhmm_to_dt,
    )
    _IBRAIN_AVAILABLE = True
except ImportError:
    _IBRAIN_AVAILABLE = False

# ── Time-conversion helpers ────────────────────────────────────────────────────
_UTC_EPOCH = datetime.datetime(1970, 1, 1)
_TZ_LOCAL_H = 8   # UTC+8 (Asia/Taipei)


def _us_to_local_dt(us: int) -> datetime.datetime:
    """UTC Unix µs → naive local datetime (UTC+8)."""
    return _UTC_EPOCH + datetime.timedelta(microseconds=int(us)) + datetime.timedelta(hours=_TZ_LOCAL_H)


def _rel_times_to_dt(
    rel_seconds: np.ndarray, time_us_epoch: int
) -> list[datetime.datetime]:
    """Convert relative-second array to absolute local datetimes."""
    return [_us_to_local_dt(time_us_epoch + int(t * 1_000_000)) for t in rel_seconds]


EPSILON = 1e-12
DEFAULT_FS = 500.0
DEFAULT_WIN_SEC = 2.0
BAND_DEFINITIONS = (
    ('delta', (0.5, 4.0)),
    ('theta', (4.0, 8.0)),
    ('alpha', (8.0, 13.0)),
    ('beta', (13.0, 30.0)),
    ('gamma', (30.0, 45.0)),
)
DEFAULT_TAU_MS = (5.0, 10.0, 15.0, 20.0)
DEFAULT_MI_BINS = 16


def _smooth_series(values: np.ndarray, window: int = 5) -> np.ndarray:
    series = np.asarray(values, dtype=float).reshape(-1)
    if series.size <= 1:
        return series.copy()

    window = max(1, min(window, series.size))
    if window % 2 == 0 and window > 1:
        window -= 1
    if window <= 1:
        return series.copy()

    pad = window // 2
    padded = np.pad(series, (pad, pad), mode='edge')
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode='valid')


def _default_welch_params(segment_len: int, fs: float) -> tuple[int, int]:
    if segment_len < 8:
        raise ValueError('EEG segment is too short for Welch PSD estimation.')
    target_nperseg = max(8, int(round(fs)))
    nperseg = min(segment_len, target_nperseg)
    noverlap = nperseg // 2
    if noverlap >= nperseg:
        noverlap = nperseg - 1
    return nperseg, max(0, noverlap)


def _compute_welch_psd(
    data_col: np.ndarray,
    fs: float = DEFAULT_FS,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    segment = np.asarray(data_col, dtype=float).reshape(-1)
    if segment.size < 8:
        raise ValueError('EEG segment must contain at least 8 samples.')
    if not np.all(np.isfinite(segment)):
        raise ValueError('EEG segment contains NaN or infinite values.')

    if nperseg is None or noverlap is None:
        default_nperseg, default_noverlap = _default_welch_params(segment.size, fs)
        if nperseg is None:
            nperseg = default_nperseg
        if noverlap is None:
            noverlap = default_noverlap

    if nperseg <= 0:
        raise ValueError('nperseg must be positive.')
    if nperseg > segment.size:
        raise ValueError('nperseg must not exceed the segment length.')
    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError('noverlap must satisfy 0 <= noverlap < nperseg.')

    return signal.welch(
        segment,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        detrend='constant',
        scaling='density',
    )


def _band_power(
    freqs: np.ndarray,
    psd: np.ndarray,
    fmin: float,
    fmax: float,
    *,
    include_upper: bool,
) -> float:
    if include_upper:
        mask = (freqs >= fmin) & (freqs <= fmax)
    else:
        mask = (freqs >= fmin) & (freqs < fmax)

    band_freqs = freqs[mask]
    band_psd = psd[mask]
    if band_freqs.size == 0:
        return 0.0
    if band_freqs.size == 1:
        return float(band_psd[0])
    return float(np.trapezoid(band_psd, band_freqs))


def compute_band_energies(
    data_col: np.ndarray,
    fs: float = DEFAULT_FS,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> dict[str, np.ndarray | dict[str, float] | float]:
    """Compute Welch PSD and five-band energies for one EEG segment."""
    freqs, psd = _compute_welch_psd(
        data_col,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
    )

    energies: dict[str, float] = {}
    for idx, (name, (fmin, fmax)) in enumerate(BAND_DEFINITIONS):
        energies[name] = _band_power(
            freqs,
            psd,
            fmin,
            fmax,
            include_upper=(idx == len(BAND_DEFINITIONS) - 1),
        )

    total_energy = float(sum(energies.values()))
    return {
        'freqs': freqs,
        'psd': psd,
        'energies': energies,
        'total_energy': total_energy,
    }


def shannon_entropy(probabilities: np.ndarray, normalise: bool = False) -> float:
    """Compute Shannon entropy from a discrete probability distribution."""
    probs = np.asarray(probabilities, dtype=float).reshape(-1)
    if probs.size == 0:
        raise ValueError('Probability array must not be empty.')
    if not np.all(np.isfinite(probs)):
        raise ValueError('Probability array contains NaN or infinite values.')
    if float(np.sum(probs)) <= EPSILON:
        return 0.0

    valid = probs[probs > 0]
    if valid.size == 0:
        return 0.0

    entropy_bits = float(-np.sum(valid * np.log2(valid)))
    if not normalise:
        return entropy_bits
    if probs.size == 1:
        return 0.0
    return entropy_bits / np.log2(probs.size)


def compute_band_entropy(
    data_col: np.ndarray,
    fs: float = DEFAULT_FS,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> dict[str, np.ndarray | dict[str, float] | float]:
    """Compute five-band energy proportions and BandEn for one EEG segment."""
    result = compute_band_energies(
        data_col,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    energies = result['energies']
    total_energy = float(result['total_energy'])

    if total_energy <= EPSILON:
        proportions = {name: 0.0 for name, _ in BAND_DEFINITIONS}
    else:
        proportions = {
            name: float(energies[name] / total_energy)
            for name, _ in BAND_DEFINITIONS
        }

    probs = np.array([proportions[name] for name, _ in BAND_DEFINITIONS], dtype=float)
    band_entropy = shannon_entropy(probs, normalise=False)
    band_entropy_norm = shannon_entropy(probs, normalise=True)

    return {
        'freqs': result['freqs'],
        'psd': result['psd'],
        'energies': energies,
        'total_energy': total_energy,
        'proportions': proportions,
        'band_entropy': band_entropy,
        'band_entropy_norm': band_entropy_norm,
    }


def compute_band_entropy_windowed(
    data_col: np.ndarray,
    fs: float = DEFAULT_FS,
    win_sec: float = DEFAULT_WIN_SEC,
    step_sec: float | None = None,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> dict[str, np.ndarray]:
    """Compute band energies, proportions, and BandEn over sliding windows."""
    if win_sec <= 0:
        raise ValueError('win_sec must be positive.')
    if step_sec is None:
        step_sec = win_sec
    if step_sec <= 0:
        raise ValueError('step_sec must be positive.')

    win = int(round(win_sec * fs))
    step = int(round(step_sec * fs))
    if win < 8:
        raise ValueError('Window length is too short for Welch PSD estimation.')
    if step < 1:
        raise ValueError('Step size is too small.')

    signal_1d = np.asarray(data_col, dtype=float).reshape(-1)
    if signal_1d.size < win:
        raise ValueError('Signal is shorter than one analysis window.')

    times: list[float] = []
    total_energy: list[float] = []
    band_entropy: list[float] = []
    band_entropy_norm: list[float] = []
    energies_by_band = {name: [] for name, _ in BAND_DEFINITIONS}
    proportions_by_band = {name: [] for name, _ in BAND_DEFINITIONS}

    for start in range(0, signal_1d.size - win + 1, step):
        segment = signal_1d[start:start + win]
        result = compute_band_entropy(
            segment,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
        )
        times.append((start + win // 2) / fs)
        total_energy.append(float(result['total_energy']))
        band_entropy.append(float(result['band_entropy']))
        band_entropy_norm.append(float(result['band_entropy_norm']))

        for name, _ in BAND_DEFINITIONS:
            energies_by_band[name].append(float(result['energies'][name]))
            proportions_by_band[name].append(float(result['proportions'][name]))

    output = {
        'time': np.asarray(times, dtype=float),
        'total_energy': np.asarray(total_energy, dtype=float),
        'band_entropy': np.asarray(band_entropy, dtype=float),
        'band_entropy_norm': np.asarray(band_entropy_norm, dtype=float),
    }
    for name, _ in BAND_DEFINITIONS:
        output[f'E_{name}'] = np.asarray(energies_by_band[name], dtype=float)
        output[f'p_{name}'] = np.asarray(proportions_by_band[name], dtype=float)
    return output


def _normalise_mi_signal(values: np.ndarray) -> np.ndarray:
    signal_1d = np.asarray(values, dtype=float).reshape(-1)
    signal_1d = signal_1d - float(signal_1d.mean())
    std = float(signal_1d.std())
    if std <= EPSILON:
        return np.zeros_like(signal_1d)
    return np.clip(signal_1d / std, -5.0, 5.0)


def _histogram_mutual_information(
    sig_x: np.ndarray,
    sig_y: np.ndarray,
    *,
    bins: int = DEFAULT_MI_BINS,
) -> float:
    x = _normalise_mi_signal(sig_x)
    y = _normalise_mi_signal(sig_y)
    if x.size != y.size:
        raise ValueError('Signals must have the same length for mutual information.')
    if x.size < 8:
        raise ValueError('Signals must contain at least 8 samples for mutual information.')
    if bins < 2:
        raise ValueError('bins must be at least 2.')

    all_values = np.concatenate([x, y])
    edges = np.histogram_bin_edges(all_values, bins=bins)
    if edges.size < 3:
        return 0.0

    joint, _, _ = np.histogram2d(x, y, bins=(edges, edges))
    total = float(joint.sum())
    if total <= EPSILON:
        return 0.0

    pxy = joint / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    denom = px * py
    valid = pxy > 0
    if not np.any(valid):
        return 0.0
    return float(np.sum(pxy[valid] * np.log2(pxy[valid] / denom[valid])))


def compute_lagged_mutual_information(
    sig_x: np.ndarray,
    sig_y: np.ndarray,
    tau_ms: float,
    *,
    fs: float = DEFAULT_FS,
    bins: int = DEFAULT_MI_BINS,
) -> float:
    """Compute I(X(t); Y(t+tau)) for a strictly positive tau in milliseconds."""
    if tau_ms <= 0:
        raise ValueError('tau_ms must be positive.')
    lag = int(round(tau_ms * fs / 1000.0))
    if lag <= 0:
        raise ValueError('tau_ms is too small for the current sampling rate.')

    x = np.asarray(sig_x, dtype=float).reshape(-1)
    y = np.asarray(sig_y, dtype=float).reshape(-1)
    n = min(x.size, y.size)
    if n <= lag:
        raise ValueError('Signals are too short for the requested lag.')

    return _histogram_mutual_information(
        x[:n - lag],
        y[lag:n],
        bins=bins,
    )


def compute_lagged_interhemispheric_sync_windowed(
    left_col: np.ndarray,
    right_col: np.ndarray,
    *,
    fs: float = DEFAULT_FS,
    win_sec: float = DEFAULT_WIN_SEC,
    step_sec: float | None = None,
    tau_ms_list: tuple[float, ...] | list[float] = DEFAULT_TAU_MS,
    bins: int = DEFAULT_MI_BINS,
    apply_bandpass: bool = True,
) -> dict[str, np.ndarray]:
    """
    Compute non-zero-lag left-right synchrony from bidirectional lagged MI.

    For each tau > 0, this computes:
      I(L(t); R(t+tau)) and I(R(t); L(t+tau))
    and stores their mean as a symmetric synchrony estimate.
    """
    if win_sec <= 0:
        raise ValueError('win_sec must be positive.')
    if step_sec is None:
        step_sec = win_sec
    if step_sec <= 0:
        raise ValueError('step_sec must be positive.')
    if bins < 2:
        raise ValueError('bins must be at least 2.')

    tau_values = tuple(float(tau) for tau in tau_ms_list)
    if not tau_values:
        raise ValueError('tau_ms_list must not be empty.')
    if any(tau <= 0 for tau in tau_values):
        raise ValueError('All tau_ms values must be positive.')

    left = np.asarray(left_col, dtype=float).reshape(-1)
    right = np.asarray(right_col, dtype=float).reshape(-1)
    n = min(left.size, right.size)
    if n == 0:
        raise ValueError('Input signals must not be empty.')

    left = left[:n]
    right = right[:n]
    if apply_bandpass:
        stacked = np.column_stack([left, right])
        filtered = bandpass_filter(stacked, fs=fs)
        left = filtered[:, 0].astype(float)
        right = filtered[:, 1].astype(float)

    win = int(round(win_sec * fs))
    step = int(round(step_sec * fs))
    if win < 8:
        raise ValueError('Window length is too short for lagged mutual information.')
    if step < 1:
        raise ValueError('Step size is too small.')
    if n < win:
        raise ValueError('Signal is shorter than one analysis window.')

    max_lag = max(int(round(tau * fs / 1000.0)) for tau in tau_values)
    if max_lag <= 0:
        raise ValueError('tau_ms values are too small for the current sampling rate.')
    if win <= max_lag:
        raise ValueError('Analysis window must be longer than the maximum requested lag.')

    times: list[float] = []
    sync_mean: list[float] = []
    sync_max: list[float] = []
    best_tau_ms: list[float] = []
    sync_by_tau: dict[str, list[float]] = {
        f'lagged_mi_tau_{tau:g}ms': [] for tau in tau_values
    }

    for start in range(0, n - win + 1, step):
        left_seg = left[start:start + win]
        right_seg = right[start:start + win]
        times.append((start + win // 2) / fs)

        tau_sync_values = []
        for tau in tau_values:
            lr = compute_lagged_mutual_information(left_seg, right_seg, tau, fs=fs, bins=bins)
            rl = compute_lagged_mutual_information(right_seg, left_seg, tau, fs=fs, bins=bins)
            sync_value = float((lr + rl) / 2.0)
            sync_by_tau[f'lagged_mi_tau_{tau:g}ms'].append(sync_value)
            tau_sync_values.append(sync_value)

        tau_sync_arr = np.asarray(tau_sync_values, dtype=float)
        sync_mean.append(float(np.mean(tau_sync_arr)))
        sync_max.append(float(np.max(tau_sync_arr)))
        best_tau_ms.append(float(tau_values[int(np.argmax(tau_sync_arr))]))

    result = {
        'time': np.asarray(times, dtype=float),
        'lagged_mi_mean': np.asarray(sync_mean, dtype=float),
        'lagged_mi_max': np.asarray(sync_max, dtype=float),
        'lagged_mi_best_tau_ms': np.asarray(best_tau_ms, dtype=float),
    }
    for key, values in sync_by_tau.items():
        result[key] = np.asarray(values, dtype=float)
    return result


def _overlay_ibrain_events(ax: plt.Axes, t_min, t_max, use_abs: bool) -> None:
    """Draw iBrainCenter event spans and labels on *ax*.

    Parameters
    ----------
    t_min, t_max : limits of the time axis (datetime objects when use_abs=True,
                   float seconds otherwise).
    use_abs      : True when x-axis holds datetime objects.
    """
    if not _IBRAIN_AVAILABLE:
        return
    for evt_idx, (name, start_hhmm, dur_min, _) in enumerate(_IBRAIN_EVENTS):
        dt_start = _hhmm_to_dt(start_hhmm)
        dt_end = dt_start + datetime.timedelta(minutes=dur_min)
        x0 = dt_start if use_abs else None
        x1 = dt_end if use_abs else None
        if not use_abs:
            continue  # only overlay when absolute time is available
        color = _IBRAIN_EVT_COLORS[evt_idx % len(_IBRAIN_EVT_COLORS)]
        ax.axvspan(x0, x1, alpha=0.12, color=color, zorder=0)
        ax.axvline(x=x0, color=color, lw=1.0, ls='--', alpha=0.7, zorder=1)
        # place label inside the axes y-range
        y_lo, y_hi = ax.get_ylim()
        ax.text(
            x0,
            y_hi - (y_hi - y_lo) * 0.05,
            f' {name}',
            color=color,
            fontsize=6,
            va='top',
            ha='left',
            clip_on=True,
            rotation=90,
        )


def plot_band_entropy(
    entropy_result: dict[str, np.ndarray],
    title: str,
    outpath: str,
    t_offset: float = 0.0,
    sync_result: dict[str, np.ndarray] | None = None,
    time_us_epoch: int | None = None,
    ibrain_events: bool = False,
) -> None:
    """Plot each band separately plus BandEn over time.

    Parameters
    ----------
    time_us_epoch : UTC Unix µs of sample 0 in the recording.  When provided,
                    the x-axis is converted to absolute local time (UTC+8).
    ibrain_events : Overlay iBrainCenter session event spans/labels when True.
                    Requires *time_us_epoch* and plot_event_markers to be importable.
    """
    use_abs = time_us_epoch is not None
    if use_abs:
        t = _rel_times_to_dt(entropy_result['time'] + t_offset, time_us_epoch)
    else:
        t = entropy_result['time'] + t_offset
    has_sync = sync_result is not None
    n_rows = len(BAND_DEFINITIONS) + 1 + (1 if has_sync else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.2 * n_rows), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    colors = {
        'delta': '#1f77b4',
        'theta': '#9467bd',
        'alpha': '#2ca02c',
        'beta': '#ff7f0e',
        'gamma': '#d62728',
    }
    for idx, (name, _) in enumerate(BAND_DEFINITIONS):
        ax = axes[idx]
        values = entropy_result[f'p_{name}']
        smooth = _smooth_series(values)
        ax.plot(t, values, color=colors[name], lw=1.1, alpha=0.4, label=f'{name} raw')
        ax.plot(t, smooth, color=colors[name], lw=2.0, label=f'{name} smooth')
        ax.set_ylabel(name)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f'{name.capitalize()} Band Proportion')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        if ibrain_events and use_abs:
            _overlay_ibrain_events(ax, t[0], t[-1], use_abs)

    entropy_axis_idx = -2 if has_sync else -1
    ax_entropy = axes[entropy_axis_idx]
    entropy_bits = entropy_result['band_entropy']
    entropy_bits_smooth = _smooth_series(entropy_bits)
    entropy_norm = entropy_result['band_entropy_norm']
    entropy_norm_smooth = _smooth_series(entropy_norm)
    ax_entropy.plot(t, entropy_bits, color='#111111', lw=1.1, alpha=0.4, label='BandEn (bits)')
    ax_entropy.plot(t, entropy_bits_smooth, color='#111111', lw=2.0, label='BandEn smooth')
    ax_entropy.plot(t, entropy_norm, color='#d62728', lw=1.0, alpha=0.35, label='BandEn norm')
    ax_entropy.plot(t, entropy_norm_smooth, color='#d62728', lw=1.8, label='BandEn norm smooth')
    ax_entropy.set_ylabel('Entropy')
    if not has_sync and not use_abs:
        ax_entropy.set_xlabel('Time (s)')
    ax_entropy.set_ylim(0.0, max(1.05, np.log2(len(BAND_DEFINITIONS)) + 0.05))
    ax_entropy.set_title('Band Structure Entropy')
    ax_entropy.legend(loc='upper right', fontsize=9)
    ax_entropy.grid(True, alpha=0.3)
    if ibrain_events and use_abs:
        _overlay_ibrain_events(ax_entropy, t[0], t[-1], use_abs)

    if has_sync:
        ax_sync = axes[-1]
        if use_abs:
            t_sync = _rel_times_to_dt(sync_result['time'] + t_offset, time_us_epoch)
        else:
            t_sync = sync_result['time'] + t_offset
        tau_keys = sorted(
            [key for key in sync_result if key.startswith('lagged_mi_tau_')],
            key=lambda key: float(key.split('_tau_')[1].replace('ms', '')),
        )
        tau_colors = ['#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']
        for idx, key in enumerate(tau_keys):
            tau_label = key.split('_tau_')[1]
            ax_sync.plot(
                t_sync,
                sync_result[key],
                color=tau_colors[idx % len(tau_colors)],
                lw=1.0,
                alpha=0.35,
                label=f'MI {tau_label}',
            )

        sync_mean = sync_result['lagged_mi_mean']
        sync_max = sync_result['lagged_mi_max']
        ax_sync.plot(
            t_sync,
            _smooth_series(sync_mean),
            color='#111111',
            lw=2.0,
            label='Lagged MI mean smooth',
        )
        ax_sync.plot(
            t_sync,
            _smooth_series(sync_max),
            color='#d62728',
            lw=1.8,
            ls='--',
            label='Lagged MI max smooth',
        )
        ax_sync.set_ylabel('MI (bits)')
        if not use_abs:
            ax_sync.set_xlabel('Time (s)')
        ax_sync.set_ylim(bottom=0.0)
        ax_sync.set_title('Left-Right Synchrony (non-zero-lag mutual information)')
        ax_sync.legend(loc='upper right', fontsize=8, ncol=2)
        ax_sync.grid(True, alpha=0.3)
        if ibrain_events and use_abs:
            _overlay_ibrain_events(ax_sync, t_sync[0], t_sync[-1], use_abs)

    # ── x-axis formatting ──────────────────────────────────────────────────────
    if use_abs:
        fmt = mdates.DateFormatter('%H:%M:%S')
        axes[-1].xaxis.set_major_formatter(fmt)
        axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
        axes[-1].set_xlabel('Time (local, UTC+8)')
        fig.autofmt_xdate(rotation=30, ha='right')

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f'Saved: {outpath}')


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compute EEG band-structure entropy from lilia-format EEG CSV files.',
    )
    parser.add_argument('--csv', required=True, metavar='PATH',
                        help='Input CSV file (lilia format, 4-row header).')
    parser.add_argument('--fs', type=float, default=DEFAULT_FS, metavar='HZ',
                        help=f'Sampling frequency (default {DEFAULT_FS} Hz).')
    parser.add_argument('--ch', type=int, default=1, metavar='N',
                        help='1-based channel index to analyse (default 1).')
    parser.add_argument('--win', type=float, default=DEFAULT_WIN_SEC, metavar='SEC',
                        help=f'Window length in seconds (default {DEFAULT_WIN_SEC} s).')
    parser.add_argument('--step', type=float, metavar='SEC',
                        help='Step size in seconds (default = window length).')
    parser.add_argument('--sync-pair', type=int, nargs=2, metavar=('LEFT', 'RIGHT'),
                        help='Optional 1-based left/right channel pair for lagged-MI synchrony.')
    parser.add_argument('--tau-ms', type=float, nargs='+', default=list(DEFAULT_TAU_MS),
                        metavar='MS',
                        help=('Positive non-zero delays in milliseconds for lagged MI '
                              f'(default: {" ".join(f"{tau:g}" for tau in DEFAULT_TAU_MS)}).'))
    parser.add_argument('--mi-bins', type=int, default=DEFAULT_MI_BINS, metavar='N',
                        help=f'Histogram bins for mutual-information estimation (default {DEFAULT_MI_BINS}).')
    parser.add_argument('--out', metavar='DIR',
                        help='Output directory (default: same dir as CSV).')
    parser.add_argument('--ibrain-events', action='store_true', default=False,
                        help=('Overlay iBrainCenter session event markers and '
                              'convert x-axis to absolute local time (UTC+8).'))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outdir = args.out or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(outdir, exist_ok=True)

    print(f'Loading: {args.csv}')
    time_us, data = load_merged_csv(args.csv)

    ch_idx = args.ch - 1
    if ch_idx < 0:
        sys.exit('Error: --ch must be >= 1.')
    if ch_idx >= data.shape[1]:
        sys.exit(f'Error: channel {args.ch} not found (file has {data.shape[1]} channels).')

    sync_result = None
    sync_suffix = ''
    if args.sync_pair is not None:
        left_ch, right_ch = args.sync_pair
        left_idx = left_ch - 1
        right_idx = right_ch - 1
        if left_idx < 0 or right_idx < 0:
            sys.exit('Error: --sync-pair channels must be >= 1.')
        if left_idx >= data.shape[1] or right_idx >= data.shape[1]:
            sys.exit(
                f'Error: --sync-pair channel not found (file has {data.shape[1]} channels).'
            )
        if left_idx == right_idx:
            sys.exit('Error: --sync-pair must specify two different channels.')
        sync_suffix = f'_sync_ch{left_ch}_ch{right_ch}'

    print(
        'Computing band entropy '
        f'— ch{args.ch}, win={args.win}s, step={args.step or args.win}s, '
        f'bands=delta/theta/alpha/beta/gamma, fs={args.fs}Hz'
    )
    entropy_result = compute_band_entropy_windowed(
        data[:, ch_idx],
        fs=args.fs,
        win_sec=args.win,
        step_sec=args.step,
    )

    if args.sync_pair is not None:
        print(
            'Computing lagged interhemispheric synchrony '
            f'— ch{left_ch}↔ch{right_ch}, tau={", ".join(f"{tau:g}" for tau in args.tau_ms)} ms, '
            f'bins={args.mi_bins}'
        )
        sync_result = compute_lagged_interhemispheric_sync_windowed(
            data[:, left_idx],
            data[:, right_idx],
            fs=args.fs,
            win_sec=args.win,
            step_sec=args.step,
            tau_ms_list=args.tau_ms,
            bins=args.mi_bins,
        )

    basename = os.path.splitext(os.path.basename(args.csv))[0]
    stem = f'{basename}_band_entropy_ch{args.ch}{sync_suffix}'
    csv_out = os.path.join(outdir, f'{stem}.csv')
    png_out = os.path.join(outdir, f'{stem}.png')

    time_s = (time_us.astype(float) - float(time_us[0])) / 1e6
    t_offset = float(time_s[0]) if time_s.size else 0.0

    csv_data = {'time_s': entropy_result['time'] + t_offset}
    for name, _ in BAND_DEFINITIONS:
        csv_data[f'E_{name}'] = entropy_result[f'E_{name}']
    csv_data['E_total'] = entropy_result['total_energy']
    for name, _ in BAND_DEFINITIONS:
        csv_data[f'p_{name}'] = entropy_result[f'p_{name}']
    csv_data['band_entropy'] = entropy_result['band_entropy']
    csv_data['band_entropy_norm'] = entropy_result['band_entropy_norm']
    if sync_result is not None:
        if not np.allclose(sync_result['time'], entropy_result['time']):
            sys.exit('Error: lagged-MI time axis does not match band-entropy windows.')
        for key, values in sync_result.items():
            if key != 'time':
                csv_data[key] = values

    pd.DataFrame(csv_data).to_csv(csv_out, index=False)
    print(f'Saved: {csv_out}')

    title = f'Band Entropy — {os.path.basename(args.csv)} ch{args.ch}'
    if args.sync_pair is not None:
        title += f' | sync ch{left_ch}↔ch{right_ch}'
    plot_band_entropy(
        entropy_result,
        title=title,
        outpath=png_out,
        t_offset=t_offset,
        sync_result=sync_result,
        time_us_epoch=int(time_us[0]) if args.ibrain_events else None,
        ibrain_events=args.ibrain_events,
    )

    bits = entropy_result['band_entropy']
    norm = entropy_result['band_entropy_norm']
    print('\nSummary:')
    print(f'  band_entropy      : mean={bits.mean():.4f}, std={bits.std():.4f}, '
          f'min={bits.min():.4f}, max={bits.max():.4f}')
    print(f'  band_entropy_norm : mean={norm.mean():.4f}, std={norm.std():.4f}, '
          f'min={norm.min():.4f}, max={norm.max():.4f}')
    if sync_result is not None:
        sync_mean = sync_result['lagged_mi_mean']
        sync_max = sync_result['lagged_mi_max']
        print(f'  lagged_mi_mean    : mean={sync_mean.mean():.4f}, std={sync_mean.std():.4f}, '
              f'min={sync_mean.min():.4f}, max={sync_mean.max():.4f}')
        print(f'  lagged_mi_max     : mean={sync_max.mean():.4f}, std={sync_max.std():.4f}, '
              f'min={sync_max.min():.4f}, max={sync_max.max():.4f}')


if __name__ == '__main__':
    main()
