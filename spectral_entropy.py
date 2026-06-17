"""
spectral_entropy.py
===================
Compute EEG band-structure entropy from lilia-format CSV files.

Pipeline (aligned with ``plot_tflite_summary.py`` / ``plot_event_markers.py``)
-----------------------------------------------------------------------------
The raw-data front-end mirrors the canonical qEEG pipeline so that this
analysis is directly comparable to the Flow/Focus/Calm/Relax indices:

    load_merged_csv(...)                       # 4-row-header lilia CSV → (time_us, data @ FS)
    bandpass_filter(data, fs=FS, lo=0.5, hi=45)# zero-phase Butterworth, BP_LOW–BP_HIGH

Unlike the TFLite branch, no 500→200 Hz resampling is applied: spectral
entropy is a *direct* PSD measurement (not a model input), so we keep the full
500 Hz record for the best frequency resolution. The 0.5–45 Hz bandpass is the
shared step that matters, and it is reused here from ``eeg_utils.bandpass_filter``.

Method
------
1. Split the (bandpass-filtered) EEG into sliding windows.
2. Use Welch's method to estimate the power spectral density (PSD).
3. Integrate the PSD within the THREE task-relevant EEG bands —
   theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz) — to obtain band energies.
   Delta (0.5-4 Hz) and gamma (30-45 Hz) are deliberately DISCARDED; see the
   ``BAND_DEFINITIONS`` note below for the methodological justification.
4. Convert band energies into proportions p_k (sum to 1).
5. Compute band entropy:  BandEn = -sum(p_k * log2(p_k)).  Max = log2(3) bits.
6. Optionally compute left-right synchrony from non-zero-lag mutual
   information I(X(t); Y(t+tau)) using small positive delays tau.

Baseline-vs-event mode (``--baseline`` / ``--event``)
-----------------------------------------------------
Instead of (only) a per-window entropy time series, this mode estimates a
single entropy for a *baseline* interval and an *event* interval, where the
probability distribution p_k is derived from the **pooled power distribution**
over that interval (the averaged Welch PSD across all clean windows). It also
reports the per-window entropy distribution and a non-parametric test of the
baseline-vs-event difference. See ``compute_state_entropy`` for the rigour
caveats (pooling, stationarity, artefact rejection, entropy non-linearity).

With ``--clean`` the baseline/event intervals are micro-epoched and only
artefact-free epochs are kept (ADC-saturation guard + eeg_quality_v2 score),
reusing the exact "clean" definition of the qEEG baseline builder in
``plot_tflite_summary``; per-epoch PSDs are then averaged into the pooled
distribution. This is the peer-review-grade path.

CLI usage
---------
    python spectral_entropy.py --csv <path.csv> [--fs 500] [--ch 1]
                               [--win 2] [--step 2] [--out <dir>]
                               [--no-bandpass]
                               [--sync-pair LEFT RIGHT]
                               [--tau-ms 5 10 15 20]
                               [--baseline START_S END_S --event START_S END_S]
                               [--clean] [--quality-threshold 0.5]
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

# ── Quality-control machinery for the --clean baseline/event mode (optional) ───
# Reuse the exact artefact-rejection primitives that build the qEEG baselines in
# plot_tflite_summary so the "clean" entropy estimates share one definition of
# "clean": ADC-saturation rejection on the RAW signal + eeg_quality_v2 scoring
# on the bandpass-filtered signal (channel-median ≥ threshold).
try:
    from plot_event_markers import QUALITY_PARAMS as _QUALITY_PARAMS, \
        QUALITY_THRESHOLD as _QUALITY_THRESHOLD
    from eeg_quality_v2 import get_eeg_quality_index_v2_parametric \
        as _eeg_quality_v2
    from plot_tflite_summary import _saturation_frac, SAT_FRAC_MAX as _SAT_FRAC_MAX
    _QC_AVAILABLE = True
except Exception:  # pragma: no cover - defensive fallback
    _QC_AVAILABLE = False
    _QUALITY_THRESHOLD = 0.5

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

# ── Bandpass front-end (single source of truth = plot_event_markers) ───────────
# Reuse the project-wide cutoffs so spectral entropy is computed on exactly the
# same passband as the qEEG indices. Fall back to the documented defaults
# (0.5–45 Hz) if plot_event_markers is unavailable.
try:
    from plot_event_markers import BP_LOW as _BP_LOW, BP_HIGH as _BP_HIGH
    DEFAULT_BP_LOW = float(_BP_LOW)
    DEFAULT_BP_HIGH = float(_BP_HIGH)
except Exception:  # pragma: no cover - defensive fallback
    DEFAULT_BP_LOW = 0.5
    DEFAULT_BP_HIGH = 45.0

# ── Band definitions ───────────────────────────────────────────────────────────
# Only the three task-relevant bands are kept. Delta and gamma are DISCARDED.
#
#   Why this is correct (and consistent with the rest of the pipeline)
#   ------------------------------------------------------------------
#   * Consistency: qeeg_indices.compute_relative_powers (§3.1) already normalises
#     over θ/α/β ONLY and excludes delta & gamma "to minimise motion/EMG
#     artefacts". The Flow index (flow_index) is a function of θ/α/β alone.
#     Computing band entropy over the same three bands keeps this measure
#     directly comparable to Flow/Focus/Calm/Relax rather than mixing in bands
#     the indices never see.
#   * Delta (0.5–4 Hz): on a dry-electrode wearable during *active* tasks
#     (e.g. Agility Ladder) this band is dominated by movement, sweat potentials
#     and baseline drift — low-frequency artefact, not cortical "flow" signal.
#   * Gamma (30–45 Hz): dominated by EMG (muscle) and approaches the line-noise
#     region; an unreliable cortical estimate on consumer EEG, again especially
#     during movement.
#   Net effect: BandEn now describes the spectral balance of the flow-relevant
#   bands; maximum entropy is log2(3) ≈ 1.585 bits (was log2(5) ≈ 2.322 bits).
BAND_DEFINITIONS = (
    ('theta', (4.0, 8.0)),
    ('alpha', (8.0, 13.0)),
    ('beta', (13.0, 30.0)),
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
        # Integrate the lone bin over the PSD frequency resolution so the result
        # is a power (uV^2), consistent with the trapezoid branch — not a raw
        # PSD density (uV^2/Hz).
        df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 1.0
        return float(band_psd[0] * df)
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


def compute_state_entropy(
    data_col: np.ndarray,
    fs: float = DEFAULT_FS,
    win_sec: float = DEFAULT_WIN_SEC,
    step_sec: float | None = None,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> dict[str, np.ndarray | dict[str, float] | float]:
    """Estimate one band entropy for a whole *state* (e.g. baseline or event).

    Two distinct entropies are returned, and the distinction is methodologically
    important — they answer different questions and must not be conflated:

    * ``pooled_entropy`` — the probability distribution p_k is taken from the
      **pooled power distribution** of the interval: the per-window Welch PSDs
      are *averaged* (equivalently, band energies are summed) across every
      window in the interval, then integrated into θ/α/β and normalised. This
      is simply a longer Welch average and is the natural reading of "the power
      distribution during baseline / event-time". It summarises the spectral
      shape of the aggregate state in a single number.

    * ``mean_window_entropy`` (± ``std_window_entropy``) — the mean of the
      per-window entropies. Because entropy is a *non-linear* function of p_k,
      entropy(mean PSD) ≠ mean(entropy). The per-window distribution is what a
      statistical test should operate on (see ``compare_baseline_event``).

    Rigour caveats
    --------------
    * **Stationarity / pooling**: averaging PSDs across windows is valid only if
      the interval is quasi-stationary and artefact-free. Feed clean, bandpass-
      filtered data and keep baseline/event durations comparable.
    * **Comparability**: baseline and event MUST use identical bands, Welch
      parameters and ``fs``; otherwise the comparison is confounded. This is
      guaranteed by calling this function with the same arguments for both.
    * **Coarse distribution**: with only three bands the maximum entropy is
      log2(3) ≈ 1.585 bits, so absolute differences are small — always report
      ΔEntropy together with the per-window test, not the pooled value alone.

    Returns a dict with the mean PSD, pooled band proportions/energies,
    ``pooled_entropy`` (bits + normalised) and the per-window entropy array.
    """
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

    signal_1d = np.asarray(data_col, dtype=float).reshape(-1)
    if signal_1d.size < win:
        raise ValueError('Interval is shorter than one analysis window.')

    psd_accum: np.ndarray | None = None
    freqs_ref: np.ndarray | None = None
    n_windows = 0
    per_window_entropy: list[float] = []
    per_window_entropy_norm: list[float] = []

    for start in range(0, signal_1d.size - win + 1, step):
        segment = signal_1d[start:start + win]
        freqs, psd = _compute_welch_psd(segment, fs=fs, nperseg=nperseg, noverlap=noverlap)
        if psd_accum is None:
            psd_accum = np.zeros_like(psd)
            freqs_ref = freqs
        psd_accum += psd
        n_windows += 1

        # Per-window band entropy (for the distribution-level statistics).
        win_result = compute_band_entropy(segment, fs=fs, nperseg=nperseg, noverlap=noverlap)
        per_window_entropy.append(float(win_result['band_entropy']))
        per_window_entropy_norm.append(float(win_result['band_entropy_norm']))

    if psd_accum is None or freqs_ref is None or n_windows == 0:
        raise ValueError('No analysis windows were produced for this interval.')

    return _assemble_state_entropy(
        freqs_ref, psd_accum / n_windows,
        per_window_entropy, per_window_entropy_norm, n_windows)


def _assemble_state_entropy(
    freqs_ref: np.ndarray,
    mean_psd: np.ndarray,
    per_window_entropy: list[float],
    per_window_entropy_norm: list[float],
    n_windows: int,
) -> dict[str, np.ndarray | dict[str, float] | float]:
    """Build the state-entropy result dict from an averaged PSD and the
    per-window entropy list. Shared by ``compute_state_entropy`` (sliding
    windows over a contiguous interval) and ``compute_state_entropy_from_epochs``
    (a set of pre-screened clean epochs) so both report identically."""
    energies: dict[str, float] = {}
    for idx, (name, (fmin, fmax)) in enumerate(BAND_DEFINITIONS):
        energies[name] = _band_power(
            freqs_ref, mean_psd, fmin, fmax,
            include_upper=(idx == len(BAND_DEFINITIONS) - 1),
        )
    total_energy = float(sum(energies.values()))
    if total_energy <= EPSILON:
        proportions = {name: 0.0 for name, _ in BAND_DEFINITIONS}
    else:
        proportions = {name: float(energies[name] / total_energy)
                       for name, _ in BAND_DEFINITIONS}
    probs = np.array([proportions[name] for name, _ in BAND_DEFINITIONS], dtype=float)

    per_window_arr = np.asarray(per_window_entropy, dtype=float)
    return {
        'freqs': freqs_ref,
        'mean_psd': mean_psd,
        'energies': energies,
        'total_energy': total_energy,
        'proportions': proportions,
        'pooled_entropy': shannon_entropy(probs, normalise=False),
        'pooled_entropy_norm': shannon_entropy(probs, normalise=True),
        'per_window_entropy': per_window_arr,
        'per_window_entropy_norm': np.asarray(per_window_entropy_norm, dtype=float),
        'mean_window_entropy': float(per_window_arr.mean()) if per_window_arr.size else 0.0,
        'std_window_entropy': float(per_window_arr.std()) if per_window_arr.size else 0.0,
        'n_windows': int(n_windows),
    }


def compute_state_entropy_from_epochs(
    epochs: list[np.ndarray],
    fs: float = DEFAULT_FS,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> dict[str, np.ndarray | dict[str, float] | float]:
    """Like ``compute_state_entropy`` but over a list of pre-screened *clean*
    epochs (each a 1-D channel segment) instead of contiguous sliding windows.

    Each epoch contributes one Welch PSD; the PSDs are averaged into the pooled
    distribution and each epoch yields one per-window entropy. Because epochs
    are scored and selected independently (see ``collect_clean_epochs``), no
    sliding window ever straddles a discarded/artefactual span — the join
    between non-contiguous epochs is never spanned by an FFT window.
    """
    if not epochs:
        raise ValueError('No clean epochs were supplied.')

    psd_accum: np.ndarray | None = None
    freqs_ref: np.ndarray | None = None
    per_window_entropy: list[float] = []
    per_window_entropy_norm: list[float] = []
    for epoch in epochs:
        freqs, psd = _compute_welch_psd(epoch, fs=fs, nperseg=nperseg, noverlap=noverlap)
        if psd_accum is None:
            psd_accum = np.zeros_like(psd)
            freqs_ref = freqs
        psd_accum += psd
        win_result = compute_band_entropy(epoch, fs=fs, nperseg=nperseg, noverlap=noverlap)
        per_window_entropy.append(float(win_result['band_entropy']))
        per_window_entropy_norm.append(float(win_result['band_entropy_norm']))

    return _assemble_state_entropy(
        freqs_ref, psd_accum / len(epochs),
        per_window_entropy, per_window_entropy_norm, len(epochs))


def collect_clean_epochs(
    time_us_full: np.ndarray,
    data_filt_full: np.ndarray,
    data_raw_full: np.ndarray | None,
    lo_us: float,
    hi_us: float,
    ch_idx: int,
    *,
    fs: float = DEFAULT_FS,
    epoch_sec: float = DEFAULT_WIN_SEC,
    quality_threshold: float = _QUALITY_THRESHOLD,
) -> tuple[list[np.ndarray], dict]:
    """Cut [lo_us, hi_us) into non-overlapping ``epoch_sec`` epochs and return
    every epoch that survives artefact rejection (for channel ``ch_idx``).

    Rejection mirrors the qEEG baseline builder in plot_tflite_summary:
      1. ADC-saturation guard on the RAW window (bandpass smears clipping, so
         saturation must be detected pre-filter);
      2. eeg_quality_v2 score on the bandpass-filtered window, channel-median
         ≥ ``quality_threshold``.

    Unlike ``build_baseline_epochs`` we keep ALL clean epochs (no blind
    subsampling): there is no subset to bias, every QC-passing epoch is used,
    which also maximises statistical power for the per-epoch test. The procedure
    is fully deterministic, hence reproducible.
    """
    if not _QC_AVAILABLE:
        raise RuntimeError(
            'Quality-control modules unavailable; --clean mode requires '
            'plot_event_markers / eeg_quality_v2 / plot_tflite_summary to import.')

    epoch_n = int(round(epoch_sec * fs))
    in_win = np.where((time_us_full >= lo_us) & (time_us_full < hi_us))[0]

    clean: list[np.ndarray] = []
    n_total = n_saturated = n_lowq = 0
    if in_win.size >= epoch_n:
        i0, i1 = in_win[0], in_win[-1] + 1
        for s in range(i0, i1 - epoch_n + 1, epoch_n):
            seg = data_filt_full[s:s + epoch_n]            # (epoch_n, n_ch)
            n_total += 1
            if data_raw_full is not None and \
                    _saturation_frac(data_raw_full[s:s + epoch_n]) > _SAT_FRAC_MAX:
                n_saturated += 1
                continue
            res = _eeg_quality_v2(seg.T.astype(np.float64), fs=fs, params=_QUALITY_PARAMS)
            if float(np.median(res['overall'])) >= quality_threshold:
                clean.append(seg[:, ch_idx].astype(float))
            else:
                n_lowq += 1
    meta = {'n_total_epochs': n_total, 'n_saturated_epochs': n_saturated,
            'n_lowquality_epochs': n_lowq, 'n_clean_epochs': len(clean)}
    return clean, meta


def _finalise_comparison(
    baseline_state: dict,
    event_state: dict,
    baseline_range: tuple[float, float],
    event_range: tuple[float, float],
) -> dict[str, object]:
    """Compute ΔEntropy and the per-window Mann–Whitney U test from two
    pre-computed state-entropy dicts. Shared by the raw and --clean paths."""
    from scipy import stats
    bw = baseline_state['per_window_entropy']
    ew = event_state['per_window_entropy']
    if bw.size >= 1 and ew.size >= 1 and (bw.size + ew.size) >= 3:
        u_stat, p_value = stats.mannwhitneyu(ew, bw, alternative='two-sided')
        u_stat, p_value = float(u_stat), float(p_value)
    else:
        u_stat, p_value = float('nan'), float('nan')
    return {
        'baseline': baseline_state,
        'event': event_state,
        'baseline_range': (float(baseline_range[0]), float(baseline_range[1])),
        'event_range': (float(event_range[0]), float(event_range[1])),
        'delta_pooled_entropy': float(
            event_state['pooled_entropy'] - baseline_state['pooled_entropy']),
        'delta_mean_window_entropy': float(
            event_state['mean_window_entropy'] - baseline_state['mean_window_entropy']),
        'mannwhitneyu_u': u_stat,
        'mannwhitneyu_p': p_value,
    }


def compare_baseline_event(
    data_col: np.ndarray,
    baseline_range: tuple[float, float],
    event_range: tuple[float, float],
    fs: float = DEFAULT_FS,
    win_sec: float = DEFAULT_WIN_SEC,
    step_sec: float | None = None,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> dict[str, object]:
    """Compute baseline vs event band entropy and test their difference.

    *baseline_range* / *event_range* are ``(start_s, end_s)`` intervals in
    seconds relative to the first sample of *data_col*. Both intervals are cut
    from the same (already bandpass-filtered) channel, analysed with identical
    Welch/band settings, and compared via:

    * ΔEntropy(pooled) = event.pooled_entropy − baseline.pooled_entropy, and
    * a two-sided Mann–Whitney U test on the per-window entropy distributions
      (non-parametric: window entropies are bounded and not guaranteed normal).

    The pooled value is the headline summary; the U test tells you whether the
    window-level distributions actually differ. Reporting both is what makes the
    baseline/event comparison defensible.
    """
    signal_1d = np.asarray(data_col, dtype=float).reshape(-1)

    def _slice(rng: tuple[float, float], which: str) -> np.ndarray:
        lo_s, hi_s = float(rng[0]), float(rng[1])
        if hi_s <= lo_s:
            raise ValueError(f'{which} range end must be greater than start.')
        lo = max(0, int(round(lo_s * fs)))
        hi = min(signal_1d.size, int(round(hi_s * fs)))
        seg = signal_1d[lo:hi]
        if seg.size < int(round(win_sec * fs)):
            raise ValueError(
                f'{which} interval [{lo_s:g}, {hi_s:g}]s yields fewer than one '
                f'{win_sec:g}s window after clipping to the recording.')
        return seg

    baseline = compute_state_entropy(
        _slice(baseline_range, 'baseline'), fs=fs, win_sec=win_sec,
        step_sec=step_sec, nperseg=nperseg, noverlap=noverlap)
    event = compute_state_entropy(
        _slice(event_range, 'event'), fs=fs, win_sec=win_sec,
        step_sec=step_sec, nperseg=nperseg, noverlap=noverlap)

    return _finalise_comparison(baseline, event, baseline_range, event_range)


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
        'theta': '#9467bd',
        'alpha': '#2ca02c',
        'beta': '#ff7f0e',
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
    parser.add_argument('--no-bandpass', action='store_true', default=False,
                        help=(f'Skip the {DEFAULT_BP_LOW:g}-{DEFAULT_BP_HIGH:g} Hz bandpass '
                              'front-end. By default the same zero-phase Butterworth '
                              'bandpass used by the qEEG pipeline is applied first.'))
    parser.add_argument('--baseline', type=float, nargs=2, metavar=('START_S', 'END_S'),
                        help=('Baseline interval (seconds, relative to recording start) '
                              'for baseline-vs-event entropy mode. Requires --event.'))
    parser.add_argument('--event', type=float, nargs=2, metavar=('START_S', 'END_S'),
                        help=('Event interval (seconds, relative to recording start) '
                              'for baseline-vs-event entropy mode. Requires --baseline.'))
    parser.add_argument('--clean', action='store_true', default=False,
                        help=('Baseline-vs-event mode only: micro-epoch each interval '
                              'and keep only artefact-free epochs (ADC-saturation guard '
                              '+ eeg_quality_v2 ≥ threshold), matching the qEEG baseline '
                              'builder in plot_tflite_summary. Requires the QC modules.'))
    parser.add_argument('--quality-threshold', type=float, default=_QUALITY_THRESHOLD,
                        metavar='Q',
                        help=('Channel-median EEG quality (0-1) an epoch must reach to be '
                              f'kept in --clean mode (default {_QUALITY_THRESHOLD:g}).'))
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


def _run_baseline_event_mode(args: argparse.Namespace,
                             time_us: np.ndarray,
                             data_filt: np.ndarray,
                             data_raw: np.ndarray | None,
                             ch_idx: int,
                             outdir: str) -> None:
    """Run and report the baseline-vs-event band-entropy comparison.

    Two sampling regimes:
      * default     — contiguous time slices (each interval taken as-is);
      * ``--clean``  — micro-epoch each interval and keep only artefact-free
                       epochs (ADC-saturation + eeg_quality_v2), then average
                       per-epoch PSDs. This is the peer-review-grade path,
                       sharing the qEEG baseline builder's definition of clean.
    """
    band_names = '/'.join(name for name, _ in BAND_DEFINITIONS)
    mode_desc = 'clean micro-epochs' if args.clean else 'contiguous slices'
    print(
        'Baseline-vs-event entropy '
        f'— ch{args.ch}, win={args.win}s, step={args.step or args.win}s, '
        f'bands={band_names}, fs={args.fs}Hz, sampling={mode_desc}\n'
        f'  baseline = [{args.baseline[0]:g}, {args.baseline[1]:g}] s | '
        f'event = [{args.event[0]:g}, {args.event[1]:g}] s'
    )

    if args.clean:
        if not _QC_AVAILABLE:
            sys.exit('Error: --clean requires plot_event_markers / eeg_quality_v2 / '
                     'plot_tflite_summary to be importable.')
        t0 = int(time_us[0])

        def _clean_state(rng, which):
            lo_us = t0 + int(round(float(rng[0]) * 1e6))
            hi_us = t0 + int(round(float(rng[1]) * 1e6))
            epochs, meta = collect_clean_epochs(
                time_us, data_filt, data_raw, lo_us, hi_us, ch_idx,
                fs=args.fs, epoch_sec=args.win,
                quality_threshold=args.quality_threshold)
            print(f'  [{which}] clean {meta["n_clean_epochs"]}/{meta["n_total_epochs"]} '
                  f'epochs (rejected: {meta["n_saturated_epochs"]} saturated, '
                  f'{meta["n_lowquality_epochs"]} low-quality)')
            if not epochs:
                sys.exit(f'Error: no clean {args.win:g}s epochs in the {which} interval '
                         f'(try a longer interval or a lower --quality-threshold).')
            return compute_state_entropy_from_epochs(epochs, fs=args.fs)

        baseline_state = _clean_state(args.baseline, 'baseline')
        event_state = _clean_state(args.event, 'event')
        result = _finalise_comparison(
            baseline_state, event_state, tuple(args.baseline), tuple(args.event))
    else:
        result = compare_baseline_event(
            data_filt[:, ch_idx],
            baseline_range=tuple(args.baseline),
            event_range=tuple(args.event),
            fs=args.fs,
            win_sec=args.win,
            step_sec=args.step,
        )

    rows = []
    for label, state in (('baseline', result['baseline']), ('event', result['event'])):
        row = {
            'state': label,
            'sampling': 'clean_epochs' if args.clean else 'contiguous',
            'range_start_s': result[f'{label}_range'][0],
            'range_end_s': result[f'{label}_range'][1],
            'n_windows': state['n_windows'],
            'pooled_entropy': state['pooled_entropy'],
            'pooled_entropy_norm': state['pooled_entropy_norm'],
            'mean_window_entropy': state['mean_window_entropy'],
            'std_window_entropy': state['std_window_entropy'],
        }
        for name, _ in BAND_DEFINITIONS:
            row[f'p_{name}'] = state['proportions'][name]
        rows.append(row)

    basename = os.path.splitext(os.path.basename(args.csv))[0]
    csv_out = os.path.join(outdir, f'{basename}_baseline_event_entropy_ch{args.ch}.csv')
    pd.DataFrame(rows).to_csv(csv_out, index=False)
    print(f'Saved: {csv_out}')

    print('\nSummary (pooled-PSD band entropy, bits; max = log2(3) ≈ 1.585):')
    for label, state in (('baseline', result['baseline']), ('event', result['event'])):
        props = ', '.join(f'{n}={state["proportions"][n]:.3f}' for n, _ in BAND_DEFINITIONS)
        print(f'  {label:<8s}: pooled={state["pooled_entropy"]:.4f}  '
              f'per-window={state["mean_window_entropy"]:.4f}±{state["std_window_entropy"]:.4f}  '
              f'(n={state["n_windows"]})  [{props}]')
    print(f'  Δ pooled entropy (event − baseline)      : '
          f'{result["delta_pooled_entropy"]:+.4f} bits')
    print(f'  Δ mean per-window entropy (event − base) : '
          f'{result["delta_mean_window_entropy"]:+.4f} bits')
    print(f'  Mann–Whitney U (per-window, two-sided)   : '
          f'U={result["mannwhitneyu_u"]:.1f}, p={result["mannwhitneyu_p"]:.4g}')


def main() -> None:
    args = _parse_args()
    outdir = args.out or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(outdir, exist_ok=True)

    print(f'Loading: {args.csv}')
    time_us, data_raw = load_merged_csv(args.csv)

    # ── Bandpass front-end (same step as plot_tflite_summary / qEEG pipeline) ───
    # All downstream measures (band entropy AND lagged-MI synchrony) run on the
    # filtered signal, so the sync path below is told not to filter again. The
    # raw (unfiltered) copy is retained for the --clean ADC-saturation guard.
    if args.no_bandpass:
        print('Bandpass: DISABLED (--no-bandpass) — analysing raw channels.')
        data = data_raw
    else:
        print(f'Bandpass: {DEFAULT_BP_LOW:g}-{DEFAULT_BP_HIGH:g} Hz zero-phase Butterworth '
              f'(fs={args.fs:g}Hz).')
        data = bandpass_filter(data_raw, fs=args.fs, lo=DEFAULT_BP_LOW, hi=DEFAULT_BP_HIGH)

    ch_idx = args.ch - 1
    if ch_idx < 0:
        sys.exit('Error: --ch must be >= 1.')
    if ch_idx >= data.shape[1]:
        sys.exit(f'Error: channel {args.ch} not found (file has {data.shape[1]} channels).')

    # ── Baseline-vs-event entropy mode ──────────────────────────────────────────
    if (args.baseline is None) != (args.event is None):
        sys.exit('Error: --baseline and --event must be supplied together.')
    if args.baseline is not None:
        _run_baseline_event_mode(args, time_us, data, data_raw, ch_idx, outdir)
        return

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

    band_names = '/'.join(name for name, _ in BAND_DEFINITIONS)
    print(
        'Computing band entropy '
        f'— ch{args.ch}, win={args.win}s, step={args.step or args.win}s, '
        f'bands={band_names}, fs={args.fs}Hz'
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
            # data is already bandpassed above (unless --no-bandpass); avoid
            # filtering twice. When --no-bandpass is set, leave the signal raw too.
            apply_bandpass=False,
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
