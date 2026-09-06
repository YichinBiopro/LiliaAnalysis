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

Joint-distribution / mutual-information mode (``--joint-mi``)
------------------------------------------------------------
Estimate the 2-D **joint probability distribution** P(X, Y) of two channels and
the mutual information it implies, I(X;Y) = H(X) + H(Y) − H(X,Y). With
``--denoise`` the two channels are the outputs of the TinyUNetV4 neural denoiser
(4 raw channels in → 2 *denoised* channels out @ 200 Hz, matching the
``data_analysis`` inference pipeline) — i.e. "denoised ch1 vs denoised ch2".
By default the joint histogram uses **equiprobable (quantile) binning**
(``--mi-binning quantile``): per-axis edges at data quantiles keep every bin
populated, which is essential on heavy-tailed dry-electrode EEG where equal-width
bins collapse into a few central cells and badly under-estimate MI.
Plug-in histogram MI is positively biased, so the Miller–Madow correction and a
circular-shift surrogate null (p-value + z) are reported alongside it. Outputs:
a P(X,Y) heatmap with marginals, a sliding-window zero-lag MI series, and a
one-row summary CSV. See ``compute_joint_probability``.

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
                               [--joint-mi [--denoise | --joint-pair CH_X CH_Y]
                                [--mi-bins 16] [--mi-binning quantile|uniform]
                                [--mi-surrogates 200]]
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

from lilia.io import bandpass_filter, load_merged_csv
from lilia.windowing import require_continuous
from lilia.time_utils import utc_us_to_local_dt

# ── qEEG Focus/Relax indices (optional import) ────────────────────────────────
# The per-window (p_θ, p_α, p_β) proportions this module already computes are
# *exactly* the relative band powers qeeg_indices.compute_relative_powers feeds
# the wellness indices, so we can derive Focus/Relax per window with no extra
# PSD work — just the closed-form index formulas.
try:
    from lilia.qeeg import focus_index, relaxation_index
    _QEEG_AVAILABLE = True
except ImportError:
    _QEEG_AVAILABLE = False

# ── iBrainCenter event definitions (optional import) ──────────────────────────
try:
    from plot_event_markers import (
        EVENTS as _IBRAIN_EVENTS,
        EVT_COLORS as _IBRAIN_EVT_COLORS,
        hhmm_to_dt as _hhmm_to_dt,
        hhmm_to_us as _hhmm_to_us,
    )
    _IBRAIN_AVAILABLE = True
except ImportError:
    _IBRAIN_AVAILABLE = False

# ── KNN/KSG mutual-information estimators for band-power × event MI (optional) ─
# The band-power-vs-event joint MI mode (I(θ,α,β power ; pre/post-event)) uses
# scikit-learn's KNN entropy estimators. sklearn is only needed for that mode,
# so it is imported defensively — the histogram-based channel-vs-channel MI that
# the rest of the module uses has no such dependency.
try:
    from scipy.special import digamma as _digamma
    from sklearn.feature_selection import mutual_info_classif as _sk_mutual_info_classif
    from sklearn.neighbors import KDTree as _SKKDTree, NearestNeighbors as _SKNearestNeighbors
    from sklearn.preprocessing import scale as _sk_scale
    _SKLEARN_MI_AVAILABLE = True
except ImportError:
    _SKLEARN_MI_AVAILABLE = False

# ── Quality-control machinery for the --clean baseline/event mode (optional) ───
# Reuse the exact artefact-rejection primitives that build the qEEG baselines in
# plot_tflite_summary so the "clean" entropy estimates share one definition of
# "clean": ADC-saturation rejection on the RAW signal + eeg_quality_v2 scoring
# on the bandpass-filtered signal (channel-median ≥ threshold).
try:
    from plot_event_markers import QUALITY_PARAMS as _QUALITY_PARAMS, \
        QUALITY_THRESHOLD as _QUALITY_THRESHOLD
    from lilia.quality import get_eeg_quality_index_v2_parametric \
        as _eeg_quality_v2
    from plot_tflite_summary import _saturation_frac, SAT_FRAC_MAX as _SAT_FRAC_MAX
    _QC_AVAILABLE = True
except Exception:  # pragma: no cover - defensive fallback
    _QC_AVAILABLE = False
    _QUALITY_THRESHOLD = 0.5

# ── Interactive plotting back-end (optional) ───────────────────────────────────
# Only the interactive peri-event MI animation (--peri-event-html) needs Plotly;
# every static figure uses matplotlib, so Plotly is imported defensively and the
# HTML export is silently skipped (with a hint) when it is unavailable.
try:
    import plotly.graph_objects as _go
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

# ── Time-conversion helpers ────────────────────────────────────────────────────
_TZ_LOCAL_H = 8   # UTC+8 (Asia/Taipei)


def _us_to_local_dt(us: int) -> datetime.datetime:
    """UTC Unix µs → naive local datetime (UTC+8)."""
    return utc_us_to_local_dt(us, _TZ_LOCAL_H)


def _rel_times_to_dt(
    rel_seconds: np.ndarray, time_us_epoch: int
) -> list[datetime.datetime]:
    """Convert relative-second array to absolute local datetimes."""
    return [_us_to_local_dt(time_us_epoch + int(t * 1_000_000)) for t in rel_seconds]


EPSILON = 1e-12
DEFAULT_FS = 500.0
DEFAULT_WIN_SEC = 2.0
# Laplace-style smoothing added to every band power before normalising the BASD
# probability vectors, so a band that drops to zero cannot produce log(0) / a
# divide-by-zero in the KL sum. Small relative to a unit-normalised mass of 1.
DEFAULT_BASD_EPSILON = 1e-9

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

# Centred moving-average length (in windows) for every smoothed trace. Module-
# level so the CLI ``--smooth`` flag can retune it once, globally, rather than
# threading a parameter through every plotting helper.
DEFAULT_SMOOTH_WINDOW = 5
SMOOTH_WINDOW = DEFAULT_SMOOTH_WINDOW

# Colour-blind-safe qualitative palette (Wong, 2011, *Nature Methods*) for the
# θ/α/β bands, replacing the purple/green/orange triple that is not safe under
# deuteranopia/protanopia. Reused for the stacked-area and ternary views so the
# band identity is consistent across every figure.
BAND_COLORS = {
    'theta': '#0072B2',   # blue
    'alpha': '#009E73',   # bluish green
    'beta':  '#D55E00',   # vermillion
}


def _smooth_series(values: np.ndarray, window: int | None = None) -> np.ndarray:
    series = np.asarray(values, dtype=float).reshape(-1)
    if series.size <= 1:
        return series.copy()

    window = SMOOTH_WINDOW if window is None else window
    window = max(1, min(window, series.size))
    if window % 2 == 0 and window > 1:
        window -= 1
    if window <= 1:
        return series.copy()

    # NaN-aware centred moving average. Low-quality windows are NaN-masked, so
    # they are ignored when averaging (min_periods=1); the masked positions are
    # then restored to NaN so a discarded window is never "filled in" from its
    # neighbours — mirrors the smoothing/re-mask in plot_tflite_summary.
    smoothed = (pd.Series(series)
                .rolling(window, center=True, min_periods=1)
                .mean()
                .to_numpy())
    smoothed[np.isnan(series)] = np.nan
    return smoothed


def _rolling_std(values: np.ndarray, window: int | None = None) -> np.ndarray:
    """NaN-aware centred rolling standard deviation, aligned 1:1 with
    ``_smooth_series``. Used to shade a ±1σ dispersion band around a smoothed
    trace so a reader can tell a real excursion from window-to-window estimator
    jitter. Masked (NaN) positions are restored to NaN, never interpolated."""
    series = np.asarray(values, dtype=float).reshape(-1)
    if series.size <= 1:
        return np.zeros_like(series)
    window = SMOOTH_WINDOW if window is None else window
    window = max(1, min(window, series.size))
    if window % 2 == 0 and window > 1:
        window -= 1
    if window <= 1:
        return np.zeros_like(series)
    std = (pd.Series(series)
           .rolling(window, center=True, min_periods=2)
           .std()
           .to_numpy())
    std[np.isnan(series)] = np.nan
    return std


def _git_commit() -> str:
    """Short git commit hash of the working tree, or 'unknown' if unavailable.
    Embedded in figure footers so every PNG is self-documenting / reproducible."""
    import subprocess
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.run(
            ['git', '-C', here, 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5)
        sha = out.stdout.strip()
        return sha if sha else 'unknown'
    except Exception:
        return 'unknown'


def _provenance(fs: float, win_sec: float, step_sec: float | None,
                extra: str = '') -> str:
    """One-line provenance string for a figure footer: passband, window grid,
    sampling rate, smoothing, git commit, timestamp."""
    bp = ('raw (no bandpass)' if _PROV.get('no_bandpass')
          else f'BP {DEFAULT_BP_LOW:g}-{DEFAULT_BP_HIGH:g}Hz')
    step = win_sec if step_sec is None else step_sec
    stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    bits = [bp, f'fs={fs:g}Hz', f'win={win_sec:g}s/step={step:g}s',
            f'smooth={SMOOTH_WINDOW}w', f'git {_git_commit()}', stamp]
    if extra:
        bits.insert(0, extra)
    return '  |  '.join(bits)


def _add_footer(fig: plt.Figure, text: str) -> None:
    """Stamp a small grey provenance footer along the bottom of *fig*."""
    fig.text(0.005, 0.002, text, fontsize=6, color='#888888',
             ha='left', va='bottom', family='monospace')


# Lightweight run-context shared with the footer builder (set in main()).
_PROV: dict[str, object] = {'no_bandpass': False}


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


def compute_quality_windowed_aligned(
    data_full: np.ndarray,
    fs: float = DEFAULT_FS,
    win_sec: float = DEFAULT_WIN_SEC,
    step_sec: float | None = None,
) -> np.ndarray:
    """Channel-median EEG quality for each analysis window, aligned 1:1 with
    ``compute_band_entropy_windowed`` (and the lagged-MI sync windows).

    Quality is scored with the same method as plot_tflite_summary —
    ``get_eeg_quality_index_v2_parametric`` (with ``QUALITY_PARAMS``) per window,
    then the per-channel ``overall`` scores are reduced by the channel median.
    The only difference is the window grid: plot_tflite_summary scores fixed 5 s
    quality windows and maps them onto the 5 s qEEG windows, whereas here the
    band-entropy windows are ``win_sec`` (default 2 s), so we score on those
    exact windows to guarantee an exact 1:1 alignment for masking.

    Returns a ``(n_windows,)`` array of channel-median quality in [0, 1].
    """
    if not _QC_AVAILABLE:
        raise RuntimeError(
            'Quality masking requires plot_event_markers / eeg_quality_v2 / '
            'plot_tflite_summary to be importable.')
    if step_sec is None:
        step_sec = win_sec

    win = int(round(win_sec * fs))
    step = int(round(step_sec * fs))
    data2d = np.asarray(data_full, dtype=float)
    if data2d.ndim == 1:
        data2d = data2d[:, None]
    n = data2d.shape[0]

    quality: list[float] = []
    for start in range(0, n - win + 1, step):
        seg = data2d[start:start + win]                       # (win, n_ch)
        res = _eeg_quality_v2(seg.T, fs=fs, params=_QUALITY_PARAMS)
        quality.append(float(np.median(res['overall'])))
    return np.asarray(quality, dtype=float)


def _mask_low_quality(result: dict, mask: np.ndarray, keys) -> None:
    """Set masked (low-quality) windows to NaN, in place, for each of *keys*
    present in *result*. ``time``/``quality`` axes are intentionally left whole
    so the discarded windows still carry a timestamp and a quality value."""
    for key in keys:
        if key in result:
            arr = np.asarray(result[key], dtype=float).copy()
            if arr.shape[:1] == mask.shape:
                arr[mask] = np.nan
            result[key] = arr


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


# ── Baseline-Anchored Spectral Divergence (BASD) ──────────────────────────────
# BASD answers a different question from the band entropy above. Shannon entropy
# H(P) = −Σ p_k log2 p_k is a *single-state* scalar: it measures how flat the
# θ/α/β split is *within* one interval, and it is blind to *which* band carries
# the power — (θ=0.7, α=0.2, β=0.1) and (θ=0.1, α=0.2, β=0.7) have identical
# entropy. For a baseline→event contrast that property is exactly wrong: a
# meditation/eyes-closed shift that moves mass from β into α barely changes the
# entropy yet is the whole signal of interest.
#
# KL divergence (relative entropy) D_KL(P_event ‖ P_base) = Σ P_event(k)
# log2[P_event(k)/P_base(k)] is the natural fix. It is *directed* and
# *anchored*: it scores how many bits are wasted coding the event spectrum with
# a code optimised for the baseline spectrum, i.e. how far the event distribution
# has moved away from the subject's own resting prior. It is 0 iff the two
# distributions are identical, strictly positive otherwise, and — unlike a
# difference of entropies — it is sensitive to *band identity*, so a θ→α→β
# redistribution registers even when overall flatness is preserved. The event is
# the first argument (P_event ‖ P_base) on purpose: we want the expectation taken
# under the event, "how surprising is the event under the baseline prior", which
# is the asymmetric, baseline-anchored reading the name BASD denotes.


def _band_power_matrix(data: object) -> np.ndarray:
    """Coerce band-power input into a ``(n_windows, n_bands)`` float matrix in
    canonical ``BAND_DEFINITIONS`` (θ, α, β) order.

    Accepts:
    * a ``{band_name: power}`` mapping (or ``{band_name: [power_t, …]}``),
    * a 1-D sequence of ``n_bands`` powers (a single block), or
    * a 2-D ``(n_windows, n_bands)`` array of time-resolved band powers.
    """
    band_names = [name for name, _ in BAND_DEFINITIONS]
    n_bands = len(band_names)

    if isinstance(data, dict):
        missing = [b for b in band_names if b not in data]
        if missing:
            raise ValueError(f'Band-power mapping is missing bands: {missing}.')
        cols = [np.atleast_1d(np.asarray(data[b], dtype=float)) for b in band_names]
        lengths = {c.size for c in cols}
        if len(lengths) != 1:
            raise ValueError('All bands must carry the same number of samples.')
        return np.column_stack(cols)

    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != n_bands:
        raise ValueError(
            f'Band-power array must have {n_bands} columns (θ, α, β); got shape '
            f'{arr.shape}.')
    return arr


def _normalise_distribution(powers: np.ndarray, epsilon: float) -> np.ndarray:
    """Turn a non-negative band-power vector into a probability distribution
    (Σ p = 1) after adding *epsilon* smoothing so no band is exactly zero."""
    powers = np.asarray(powers, dtype=float)
    if np.any(powers < 0):
        raise ValueError('Band powers must be non-negative.')
    smoothed = powers + epsilon
    total = float(smoothed.sum())
    if total <= 0:
        raise ValueError('Total band power is non-positive even after smoothing.')
    return smoothed / total


def kl_divergence(
    p_event: np.ndarray,
    p_base: np.ndarray,
    epsilon: float = DEFAULT_BASD_EPSILON,
) -> float:
    """Kullback–Leibler divergence D_KL(P_event ‖ P_base) in **bits**.

    Both inputs are treated as (already smoothed) probability vectors of the same
    length; *epsilon* is re-applied defensively in case a caller passes a raw or
    un-smoothed distribution, guaranteeing the log and the ratio stay finite.
    The result is asymmetric (``kl_divergence(a, b) != kl_divergence(b, a)``) and
    non-negative.
    """
    pe = _normalise_distribution(np.asarray(p_event, dtype=float), epsilon)
    pb = _normalise_distribution(np.asarray(p_base, dtype=float), epsilon)
    if pe.shape != pb.shape:
        raise ValueError('P_event and P_base must have the same shape.')
    # Σ pe * log2(pe / pb); both strictly positive after smoothing.
    return float(np.sum(pe * np.log2(pe / pb)))


def compute_basd(
    baseline_data: object,
    event_data: object,
    time_resolved: bool = False,
    epsilon: float = DEFAULT_BASD_EPSILON,
) -> dict[str, object]:
    """Baseline-Anchored Spectral Divergence between a baseline and an event.

    Parameters
    ----------
    baseline_data, event_data
        θ/α/β band powers for the two states, in any form accepted by
        ``_band_power_matrix``: a ``{band: power}`` mapping, a 1-D 3-vector, or a
        2-D ``(n_windows, 3)`` array of per-window powers. The baseline is always
        collapsed to a single prior P_base by averaging its band powers over the
        baseline duration, then normalising so Σ P_base = 1.
    time_resolved
        If ``False`` (default) the event is likewise block-averaged into one
        P_event and a single scalar BASD is returned. If ``True`` the event must
        be 2-D; each event window is normalised on its own and scored against the
        shared P_base, yielding a BASD *time series* (plus its mean / std / peak).
    epsilon
        Laplace smoothing added to every band power before normalisation
        (see ``DEFAULT_BASD_EPSILON``) — prevents log(0) / divide-by-zero.

    Returns
    -------
    dict with:
        ``bands``            – band order, ``['theta', 'alpha', 'beta']``.
        ``p_base``           – baseline probability vector (sums to 1).
        ``p_event``          – event probability vector, or ``(n_windows, 3)``
                               matrix when ``time_resolved``.
        ``basd_bits``        – the headline divergence in bits: the block KL, or
                               the mean of the per-window KL when time-resolved.
        ``basd_per_window``  – per-window KL array (``time_resolved`` only).
        ``basd_std`` / ``basd_peak`` – dispersion / maximum of the series
                               (``time_resolved`` only).
        ``epsilon``          – the smoothing constant actually used.

    Interpretation: BASD is in bits and bounded below by 0 (event ≡ baseline).
    There is no fixed upper bound, but with three bands and ε smoothing it is
    finite; larger values mean the event spectrum sits further from the subject's
    resting θ/α/β prior. Report it alongside which band gained/lost mass
    (``p_event − p_base``), since the scalar alone does not name the direction.
    """
    base_matrix = _band_power_matrix(baseline_data)
    p_base = _normalise_distribution(base_matrix.mean(axis=0), epsilon)
    band_names = [name for name, _ in BAND_DEFINITIONS]

    if not time_resolved:
        event_matrix = _band_power_matrix(event_data)
        p_event = _normalise_distribution(event_matrix.mean(axis=0), epsilon)
        return {
            'bands': band_names,
            'p_base': p_base,
            'p_event': p_event,
            'basd_bits': kl_divergence(p_event, p_base, epsilon=epsilon),
            'epsilon': float(epsilon),
        }

    event_matrix = _band_power_matrix(event_data)
    if event_matrix.shape[0] < 1:
        raise ValueError('Time-resolved BASD needs at least one event window.')
    p_event = np.vstack([
        _normalise_distribution(row, epsilon) for row in event_matrix])
    per_window = np.array([
        kl_divergence(row, p_base, epsilon=epsilon) for row in p_event],
        dtype=float)
    return {
        'bands': band_names,
        'p_base': p_base,
        'p_event': p_event,
        'basd_bits': float(per_window.mean()),
        'basd_per_window': per_window,
        'basd_std': float(per_window.std()),
        'basd_peak': float(per_window.max()),
        'epsilon': float(epsilon),
    }


def compute_basd_from_segments(
    data_col: np.ndarray,
    baseline_range: tuple[float, float],
    event_range: tuple[float, float],
    fs: float = DEFAULT_FS,
    win_sec: float = DEFAULT_WIN_SEC,
    step_sec: float | None = None,
    nperseg: int | None = None,
    noverlap: int | None = None,
    time_resolved: bool = False,
    epsilon: float = DEFAULT_BASD_EPSILON,
) -> dict[str, object]:
    """Convenience wrapper that computes BASD straight from a raw channel.

    The baseline / event ``(start_s, end_s)`` intervals are cut from
    *data_col*, Welch-analysed with identical settings (so the comparison is not
    confounded), and reduced to θ/α/β band energies via ``compute_state_entropy``
    — the same pooled / per-window machinery used by ``compare_baseline_event``.
    The pooled band energies form each block distribution; in ``time_resolved``
    mode the event's per-window band energies (one distribution per window) are
    taken from the windowed pass instead of the pooled average.

    Returns the ``compute_basd`` dict augmented with ``baseline_range`` /
    ``event_range`` for provenance.
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

    base_state = compute_state_entropy(
        _slice(baseline_range, 'baseline'), fs=fs, win_sec=win_sec,
        step_sec=step_sec, nperseg=nperseg, noverlap=noverlap)
    baseline_data = base_state['energies']

    if time_resolved:
        ev = compute_band_entropy_windowed(
            _slice(event_range, 'event'), fs=fs, win_sec=win_sec,
            step_sec=step_sec, nperseg=nperseg, noverlap=noverlap)
        band_names = [name for name, _ in BAND_DEFINITIONS]
        event_data = np.column_stack([ev[f'E_{b}'] for b in band_names])
    else:
        event_state = compute_state_entropy(
            _slice(event_range, 'event'), fs=fs, win_sec=win_sec,
            step_sec=step_sec, nperseg=nperseg, noverlap=noverlap)
        event_data = event_state['energies']

    result = compute_basd(
        baseline_data, event_data, time_resolved=time_resolved, epsilon=epsilon)
    result['baseline_range'] = (float(baseline_range[0]), float(baseline_range[1]))
    result['event_range'] = (float(event_range[0]), float(event_range[1]))
    return result


def _normalise_mi_signal(values: np.ndarray) -> np.ndarray:
    signal_1d = np.asarray(values, dtype=float).reshape(-1)
    signal_1d = signal_1d - float(signal_1d.mean())
    std = float(signal_1d.std())
    if std <= EPSILON:
        return np.zeros_like(signal_1d)
    return np.clip(signal_1d / std, -5.0, 5.0)


def _joint_bin_edges(
    x: np.ndarray,
    y: np.ndarray,
    bins: int,
    binning: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the (x_edges, y_edges) for the 2-D histogram.

    ``'uniform'`` — a single shared set of equal-width edges spanning the pooled
    range of both signals (legacy behaviour; fine for clean, comparably-scaled
    signals but wastes bins when artefacts inflate the range).

    ``'quantile'`` — per-axis *equiprobable* edges placed at data quantiles, so
    every marginal bin holds ≈ the same number of samples. This adapts to the
    real distribution, keeps all bins populated (each marginal entropy → its
    log2(bins) ceiling) and is robust to the heavy-tailed artefacts typical of
    dry-electrode EEG. Degenerate quantiles (ties / saturation) collapse via
    ``np.unique``, which simply lowers the effective bin count for that axis.
    """
    if binning == 'uniform':
        edges = np.histogram_bin_edges(np.concatenate([x, y]), bins=bins)
        return edges, edges
    if binning == 'quantile':
        q = np.linspace(0.0, 1.0, bins + 1)
        return np.unique(np.quantile(x, q)), np.unique(np.quantile(y, q))
    raise ValueError("binning must be 'uniform' or 'quantile'.")


def compute_joint_probability(
    sig_x: np.ndarray,
    sig_y: np.ndarray,
    *,
    bins: int = DEFAULT_MI_BINS,
    binning: str = 'uniform',
    normalise_signals: bool = True,
) -> dict[str, np.ndarray | float | int]:
    """Estimate the 2-D joint probability distribution P(X, Y) of two equal-
    length signals and the mutual information derived from it.

    Method
    ------
    The two signals are (by default) z-scored and clipped to ±5 σ
    (``_normalise_mi_signal``); the joint distribution ``pxy`` is then the
    normalised 2-D histogram and the marginals ``px`` / ``py`` are its
    row / column sums, so by construction ``pxy.sum() == 1`` and the marginals
    are mutually consistent with the joint. ``binning`` selects the bin edges
    (see ``_joint_bin_edges``): ``'quantile'`` (equiprobable, recommended for
    real EEG) or ``'uniform'`` (equal-width, the default for backward
    compatibility with the lagged-MI synchrony path).

    Mutual information is read straight off the same distribution as

        I(X; Y) = H(X) + H(Y) - H(X, Y)                         [bits]

    which is algebraically identical to ``Σ P(x,y) log2[P(x,y)/(P(x)P(y))]`` but
    keeps the joint and marginal entropies available for inspection.

    Bias
    ----
    The plug-in (maximum-likelihood) entropy estimator is negatively biased, so
    plug-in MI is *positively* biased — it grows with ``bins`` and shrinks with
    sample size even for independent signals. The Miller–Madow bias-corrected MI
    (``mutual_information_mm``) is therefore also returned; for an absolute,
    defensible MI value prefer it and/or compare against the shuffled-surrogate
    null in ``compute_joint_mi_significance``.

    Returns
    -------
    dict with the joint PMF ``pxy`` (nx × ny), marginals ``px`` / ``py``, the
    per-axis bin edges ``x_edges`` / ``y_edges``, the three entropies (bits),
    the plug-in and Miller–Madow ``mutual_information`` / ``mutual_information_mm``
    (bits), the normalised MI (plug-in MI ÷ min(H(X), H(Y))), the effective bin
    counts ``n_bins_x`` / ``n_bins_y`` and the sample count.
    """
    if normalise_signals:
        x = _normalise_mi_signal(sig_x)
        y = _normalise_mi_signal(sig_y)
    else:
        x = np.asarray(sig_x, dtype=float).reshape(-1)
        y = np.asarray(sig_y, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError('Signals must have the same length for mutual information.')
    if x.size < 8:
        raise ValueError('Signals must contain at least 8 samples for mutual information.')
    if bins < 2:
        raise ValueError('bins must be at least 2.')

    x_edges, y_edges = _joint_bin_edges(x, y, bins, binning)

    def _empty() -> dict[str, np.ndarray | float | int]:
        nx = max(1, x_edges.size - 1)
        ny = max(1, y_edges.size - 1)
        return {
            'pxy': np.zeros((nx, ny), dtype=float),
            'px': np.zeros(nx, dtype=float), 'py': np.zeros(ny, dtype=float),
            'x_edges': x_edges, 'y_edges': y_edges,
            'entropy_x': 0.0, 'entropy_y': 0.0, 'entropy_xy': 0.0,
            'mutual_information': 0.0, 'mutual_information_mm': 0.0,
            'mutual_information_norm': 0.0,
            'n_bins_x': nx, 'n_bins_y': ny, 'n_samples': int(x.size),
        }

    if x_edges.size < 3 or y_edges.size < 3:
        return _empty()

    joint, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges))
    total = float(joint.sum())
    if total <= EPSILON:
        return _empty()

    pxy = joint / total
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    h_x = shannon_entropy(px, normalise=False)
    h_y = shannon_entropy(py, normalise=False)
    h_xy = shannon_entropy(pxy.reshape(-1), normalise=False)
    mi = max(0.0, h_x + h_y - h_xy)

    # Miller–Madow correction: Ĥ_MM = Ĥ + (m̂ − 1) / (2N), where m̂ is the number
    # of occupied bins (nats); the /ln2 converts the correction term to bits.
    inv2n_ln2 = 1.0 / (2.0 * total * np.log(2.0))
    m_x = int(np.count_nonzero(px))
    m_y = int(np.count_nonzero(py))
    m_xy = int(np.count_nonzero(pxy))
    h_x_mm = h_x + (m_x - 1) * inv2n_ln2
    h_y_mm = h_y + (m_y - 1) * inv2n_ln2
    h_xy_mm = h_xy + (m_xy - 1) * inv2n_ln2
    mi_mm = max(0.0, h_x_mm + h_y_mm - h_xy_mm)

    denom = min(h_x, h_y)
    mi_norm = float(mi / denom) if denom > EPSILON else 0.0

    return {
        'pxy': pxy,
        'px': px,
        'py': py,
        'x_edges': x_edges,
        'y_edges': y_edges,
        'entropy_x': float(h_x),
        'entropy_y': float(h_y),
        'entropy_xy': float(h_xy),
        'mutual_information': float(mi),
        'mutual_information_mm': float(mi_mm),
        'mutual_information_norm': mi_norm,
        'n_bins_x': int(px.size),
        'n_bins_y': int(py.size),
        'n_samples': int(x.size),
    }


def _histogram_mutual_information(
    sig_x: np.ndarray,
    sig_y: np.ndarray,
    *,
    bins: int = DEFAULT_MI_BINS,
    binning: str = 'uniform',
) -> float:
    """Plug-in histogram MI in bits (thin wrapper over the joint distribution)."""
    return float(
        compute_joint_probability(sig_x, sig_y, bins=bins, binning=binning)[
            'mutual_information']
    )


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


def compute_joint_mi_windowed(
    sig_x: np.ndarray,
    sig_y: np.ndarray,
    *,
    fs: float = DEFAULT_FS,
    win_sec: float = DEFAULT_WIN_SEC,
    step_sec: float | None = None,
    bins: int = DEFAULT_MI_BINS,
    binning: str = 'uniform',
) -> dict[str, np.ndarray]:
    """Per-window *zero-lag* mutual information I(X(t); Y(t)) between two channels.

    Companion to ``compute_lagged_interhemispheric_sync_windowed`` but at τ = 0:
    it tracks the instantaneous shared information between the two (denoised)
    channels by re-estimating their joint probability distribution in each
    sliding window. Each window is normalised independently, so the series is
    insensitive to slow amplitude drift between windows.

    Returns ``time`` plus ``joint_mi`` (bits) and ``joint_mi_norm``.
    """
    if win_sec <= 0:
        raise ValueError('win_sec must be positive.')
    if step_sec is None:
        step_sec = win_sec
    if step_sec <= 0:
        raise ValueError('step_sec must be positive.')
    if bins < 2:
        raise ValueError('bins must be at least 2.')

    x = np.asarray(sig_x, dtype=float).reshape(-1)
    y = np.asarray(sig_y, dtype=float).reshape(-1)
    n = min(x.size, y.size)
    if n == 0:
        raise ValueError('Input signals must not be empty.')
    x, y = x[:n], y[:n]

    win = int(round(win_sec * fs))
    step = int(round(step_sec * fs))
    if win < 8:
        raise ValueError('Window length is too short for mutual information.')
    if step < 1:
        raise ValueError('Step size is too small.')
    if n < win:
        raise ValueError('Signal is shorter than one analysis window.')

    times: list[float] = []
    joint_mi: list[float] = []
    joint_mi_norm: list[float] = []
    for start in range(0, n - win + 1, step):
        joint = compute_joint_probability(
            x[start:start + win], y[start:start + win], bins=bins, binning=binning)
        times.append((start + win // 2) / fs)
        joint_mi.append(float(joint['mutual_information']))
        joint_mi_norm.append(float(joint['mutual_information_norm']))

    return {
        'time': np.asarray(times, dtype=float),
        'joint_mi': np.asarray(joint_mi, dtype=float),
        'joint_mi_norm': np.asarray(joint_mi_norm, dtype=float),
    }


def compute_joint_mi_significance(
    sig_x: np.ndarray,
    sig_y: np.ndarray,
    *,
    bins: int = DEFAULT_MI_BINS,
    binning: str = 'uniform',
    n_surrogates: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """Assess whether the observed MI exceeds chance via a circular-shift null.

    Plug-in histogram MI is positively biased, so a non-zero value is not by
    itself evidence of dependence. Each surrogate circularly shifts ``Y`` by a
    random offset — this destroys the cross-channel coupling while preserving
    each channel's own autocorrelation and amplitude distribution — and MI is
    recomputed. The one-sided p-value is the fraction of surrogates whose MI is
    ≥ the observed MI (add-one smoothed), and ``z`` is the standardised excess
    over the surrogate mean.

    Returns the observed MI, surrogate mean/std, ``p_value`` and ``z``.
    """
    x = np.asarray(sig_x, dtype=float).reshape(-1)
    y = np.asarray(sig_y, dtype=float).reshape(-1)
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    observed = _histogram_mutual_information(x, y, bins=bins, binning=binning)
    if n_surrogates <= 0 or n < 16:
        return {'mutual_information': float(observed), 'surrogate_mean': float('nan'),
                'surrogate_std': float('nan'), 'p_value': float('nan'),
                'z': float('nan'), 'n_surrogates': 0,
                'surrogates': np.empty(0, dtype=float)}

    rng = np.random.default_rng(seed)
    # Avoid trivial (near-zero) shifts that barely perturb the alignment.
    shifts = rng.integers(low=max(1, n // 100), high=n, size=int(n_surrogates))
    surrogate = np.empty(int(n_surrogates), dtype=float)
    for i, shift in enumerate(shifts):
        surrogate[i] = _histogram_mutual_information(
            x, np.roll(y, int(shift)), bins=bins, binning=binning)

    s_mean = float(surrogate.mean())
    s_std = float(surrogate.std())
    p_value = float((np.sum(surrogate >= observed) + 1) / (surrogate.size + 1))
    z = float((observed - s_mean) / s_std) if s_std > EPSILON else float('nan')
    return {'mutual_information': float(observed), 'surrogate_mean': s_mean,
            'surrogate_std': s_std, 'p_value': p_value, 'z': z,
            'n_surrogates': int(surrogate.size), 'surrogates': surrogate}


def denoise_channels(
    data_raw: np.ndarray,
    fs: float = DEFAULT_FS,
) -> tuple[np.ndarray, float]:
    """Produce the two denoised EEG channels with the TinyUNetV4 neural model.

    Mirrors the inference pipeline in ``data_analysis`` exactly so the denoised
    channels here match the rest of the project: filter (bandpass → notch →
    bandstop) at *fs*, resample to the model's 200 Hz training rate, then run
    overlap-add inference (4 raw channels in → 2 denoised channels out).

    Returns ``(denoised (N, 2) float array, fs_out)`` where ``fs_out`` is the
    model's output sampling rate (200 Hz). Requires PyTorch and the
    ``eeg_denoise`` model package (imported lazily via ``data_analysis``).
    """
    try:
        import data_analysis as da  # noqa: WPS433 (lazy: heavy torch dependency)
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            'Neural denoising requires data_analysis (PyTorch + the eeg_denoise '
            'TinyUNetV4 package) to be importable.') from exc

    data2d = np.asarray(data_raw, dtype=float)
    if data2d.ndim == 1:
        data2d = data2d[:, None]
    if data2d.shape[1] < da.N_CH:
        raise ValueError(
            f'Denoiser expects {da.N_CH} input channels, got {data2d.shape[1]}.')

    filt = da.apply_filters(data2d[:, :da.N_CH])
    t_idx = np.arange(filt.shape[0], dtype=float)
    _, filt_ds = da.downsample_data(t_idx, filt, fs_in=fs, fs_out=da.DOWNSAMPLED_FS)
    model = da.load_model()
    denoised = da.run_model(model, filt_ds)
    return np.asarray(denoised, dtype=float), float(da.DOWNSAMPLED_FS)


def compute_event_pre_onset_joint_mi(
    sig_x: np.ndarray,
    sig_y: np.ndarray,
    time_us_epoch: int,
    *,
    fs: float = DEFAULT_FS,
    pre_sec: float = 30.0,
    onset_sec: float = 30.0,
    bins: int = DEFAULT_MI_BINS,
    binning: str = 'quantile',
    n_surrogates: int = 100,
    subject: str | None = None,
) -> list[dict]:
    """For each iBrainCenter event, compare the two channels' joint distribution
    *just before* the event with the distribution *at its onset*.

    For an event starting at absolute time ``onset_us`` the two intervals are
        pre-event : [onset − pre_sec, onset)
        onset     : [onset,          onset + onset_sec)
    both relative to ``time_us_epoch`` (the UTC µs of sample 0). Each interval
    yields a joint probability distribution P(X, Y) and the MI it implies, so a
    shift in interhemispheric coupling around the event onset is directly
    visible (ΔMI = onset − pre).

    If subject is omitted, only events open to all participants are used.
    Events whose pre- or onset-window falls outside the recording (fewer than
    ~1 s of data) are skipped. Returns one dict per usable event with the two
    ``compute_joint_probability`` results and the headline MI values.
    """
    if not _IBRAIN_AVAILABLE:
        raise RuntimeError('iBrainCenter event definitions are unavailable.')

    n = min(sig_x.size, sig_y.size)
    min_n = max(8, int(round(fs)))            # need ≳ 1 s per window
    results: list[dict] = []
    for name, start_hhmm, _dur_min, _participants in _IBRAIN_EVENTS:
        if _participants is not None and (subject is None or subject not in _participants):
            continue
        onset_rel_s = (_hhmm_to_us(start_hhmm) - int(time_us_epoch)) / 1e6
        onset_idx = int(round(onset_rel_s * fs))
        pre_lo = max(0, onset_idx - int(round(pre_sec * fs)))
        pre_hi = min(n, onset_idx)
        on_lo = max(0, onset_idx)
        on_hi = min(n, onset_idx + int(round(onset_sec * fs)))
        if (pre_hi - pre_lo) < min_n or (on_hi - on_lo) < min_n:
            continue

        pre = compute_joint_probability(
            sig_x[pre_lo:pre_hi], sig_y[pre_lo:pre_hi], bins=bins, binning=binning)
        onset = compute_joint_probability(
            sig_x[on_lo:on_hi], sig_y[on_lo:on_hi], bins=bins, binning=binning)
        # Per-interval circular-shift significance, so each bar can carry a
        # surrogate p/z rather than an un-tested raw MI (plug-in MI is positively
        # biased; a bare value is not evidence of coupling).
        pre_sig = compute_joint_mi_significance(
            sig_x[pre_lo:pre_hi], sig_y[pre_lo:pre_hi], bins=bins,
            binning=binning, n_surrogates=n_surrogates)
        onset_sig = compute_joint_mi_significance(
            sig_x[on_lo:on_hi], sig_y[on_lo:on_hi], bins=bins,
            binning=binning, n_surrogates=n_surrogates)
        results.append({
            'name': name,
            'onset_rel_s': float(onset_rel_s),
            'pre': pre,
            'onset': onset,
            'pre_mi': float(pre['mutual_information']),
            'onset_mi': float(onset['mutual_information']),
            'delta_mi': float(onset['mutual_information'] - pre['mutual_information']),
            'pre_p': float(pre_sig['p_value']), 'pre_z': float(pre_sig['z']),
            'onset_p': float(onset_sig['p_value']), 'onset_z': float(onset_sig['z']),
            'pre_n': int(pre['n_samples']), 'onset_n': int(onset['n_samples']),
        })
    return results


def plot_event_pre_onset_comparison(
    events: list[dict],
    title: str,
    outpath: str,
    label_x: str = 'X',
    label_y: str = 'Y',
) -> None:
    """Bar chart of pre-event vs onset joint MI for each iBrainCenter event,
    so the change in coupling at every onset is comparable at a glance."""
    if not events:
        return
    names = [e['name'] for e in events]
    pre_mi = [e['pre_mi'] for e in events]
    onset_mi = [e['onset_mi'] for e in events]
    x = np.arange(len(names))
    w = 0.4

    # Surrogate p-values (if present) gate a significance star on each bar, so a
    # ΔMI within surrogate noise is not over-read.
    pre_p = [e.get('pre_p', float('nan')) for e in events]
    onset_p = [e.get('onset_p', float('nan')) for e in events]

    def _star(p):
        if not np.isfinite(p):
            return ''
        return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

    fig, ax = plt.subplots(figsize=(max(8, 1.7 * len(names)), 5.0))
    ax.bar(x - w / 2, pre_mi, width=w, color='#8c9bb5', label='pre-event')
    ax.bar(x + w / 2, onset_mi, width=w, color='#d62728', label='onset')
    for xi, (p, o, pp, op) in enumerate(zip(pre_mi, onset_mi, pre_p, onset_p)):
        ax.annotate(_star(pp), (xi - w / 2, p), ha='center', va='bottom',
                    fontsize=7, color='#555555')
        ax.annotate(_star(op), (xi + w / 2, o), ha='center', va='bottom',
                    fontsize=7, color='#555555')
        ax.annotate(f'Δ{o - p:+.3f}', (xi, max(p, o)), ha='center', va='bottom',
                    fontsize=8, color='#333333', xytext=(0, 9),
                    textcoords='offset points')
    # Flag unequal N (plug-in MI bias is N-dependent → unequal-N comparisons are
    # subtly confounded).
    n_note = ''
    if events and 'pre_n' in events[0]:
        ns = {(e['pre_n'], e['onset_n']) for e in events}
        if any(a != b for a, b in ns):
            n_note = '  [⚠ unequal pre/onset N — see CSV]'
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('joint MI (bits)')
    # Headroom so the ΔMI annotation above the tallest bar is not clipped.
    top = max([*pre_mi, *onset_mi, 1e-6])
    ax.set_ylim(0.0, top * 1.18)
    ax.set_title(f'{title}\nPre-event vs onset joint MI — {label_x} vs {label_y} '
                 f'(ΔMI annotated; * = surrogate p<.05){n_note}')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    _add_footer(fig, _provenance(DEFAULT_FS, DEFAULT_WIN_SEC, None,
                                 extra='pre/onset MI'))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(outpath, dpi=150)
    fig.savefig(os.path.splitext(outpath)[0] + '.svg')
    plt.close(fig)
    print(f'Saved: {outpath}')


# ═════════════════════════════════════════════════════════════════════════════
# Band-power × Event joint MI:  I(θ-power, α-power, β-power ; pre/post-event)
# ═════════════════════════════════════════════════════════════════════════════
# Ported from the standalone joint_mi_band_analysis.py and adapted to this
# module's real, *continuous* recordings. Where the standalone script assumed
# trial-epoched arrays (n_trials samples per pre/post class), a lilia recording
# is one long signal with event onsets, so samples are drawn by *sub-epoch
# tiling*: each pre/post window is cut into short (overlapping) sub-epochs and
# each sub-epoch's mean θ/α/β envelope amplitude is one 3-D feature vector. When
# --ibrain-events is used the sub-epochs are pooled across every session event,
# which is the faithful analogue of the standalone script's "trials".
#
# Two estimates of I(θ,α,β power ; event) are reported, both in bits:
#   * sum-of-per-band  -- Σ_j I(X_j ; Y) from sklearn.mutual_info_classif; equals
#     the joint MI only if the bands are conditionally independent given the
#     label, so for correlated bands it over-counts shared information.
#   * true joint KSG   -- I([θ,α,β] ; Y) from compute_mi_cd_multivariate, the
#     Ross (2014) mixed estimator measuring neighbour radii in the full 3-D
#     feature space. This is the value to trust; the gap to the sum diagnoses
#     inter-band redundancy.
# In keeping with the module's surrogate-gated MI culture, a label-shuffle
# permutation null can gate the joint estimate (--mi-surrogates).

_NATS_TO_BITS = 1.0 / np.log(2.0)


def extract_band_envelopes(sig: np.ndarray, fs: float = DEFAULT_FS) -> dict[str, np.ndarray]:
    """Instantaneous θ/α/β amplitude envelopes via bandpass + Hilbert transform.

    Each canonical band in ``BAND_DEFINITIONS`` is isolated with the same
    zero-phase Butterworth bandpass the qEEG pipeline uses (``bandpass_filter``),
    then the analytic-signal magnitude gives the instantaneous amplitude.

    Returns a dict band-name → envelope (1-D, same length as ``sig``).
    """
    x = np.asarray(sig, dtype=float).reshape(-1)
    envelopes: dict[str, np.ndarray] = {}
    for name, (lo, hi) in BAND_DEFINITIONS:
        filtered = np.asarray(bandpass_filter(x, fs=fs, lo=lo, hi=hi), dtype=float)
        envelopes[name] = np.abs(signal.hilbert(filtered))
    return envelopes


def _subepoch_features(
    envelopes: dict[str, np.ndarray],
    lo: int,
    hi: int,
    sub_len: int,
    sub_step: int,
) -> np.ndarray:
    """Mean θ/α/β envelope over each overlapping sub-epoch in ``[lo, hi)``.

    Returns an ``(n_subepochs, 3)`` matrix (band columns in BAND_DEFINITIONS
    order). If the window is shorter than one sub-epoch it degrades gracefully
    to a single mean over the whole window.
    """
    band_names = [name for name, _ in BAND_DEFINITIONS]
    rows: list[list[float]] = []
    start = lo
    while start + sub_len <= hi:
        rows.append([float(envelopes[n][start:start + sub_len].mean()) for n in band_names])
        start += sub_step
    if not rows and hi > lo:
        rows.append([float(envelopes[n][lo:hi].mean()) for n in band_names])
    return np.asarray(rows, dtype=float)


def _preprocess_continuous_features(X: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Replicate sklearn's continuous-feature preprocessing (unit-variance scale
    + tiny tie-breaking noise) so the joint-KSG estimate is directly comparable
    to ``mutual_info_classif`` and matches it exactly in the 1-D limit."""
    X = X.astype(np.float64, copy=True)
    X = _sk_scale(X, with_mean=False, copy=False)
    means = np.maximum(1.0, np.mean(np.abs(X), axis=0))
    X += 1e-10 * means * rng.standard_normal(size=X.shape)
    return X


def compute_mi_cd_multivariate(c: np.ndarray, d: np.ndarray, n_neighbors: int = 3) -> float:
    """True joint MI I(C ; D): multivariate continuous C vs. discrete D, in nats.

    Ross (2014) KSG-style mixed estimator — mathematically identical to
    scikit-learn's private ``_compute_mi_cd`` but generalised to accept a
    *multivariate* continuous variable of shape ``(n_samples, n_features)``.
    sklearn hard-codes ``c.reshape((-1, 1))`` and so only supports one continuous
    dimension; the sole change here is to keep ``c`` a joint vector, so
    nearest-neighbour radii are measured in the full 3-D (θ, α, β) space rather
    than per band.

    Reference: B. C. Ross, "Mutual Information between Discrete and Continuous
    Data Sets", PLoS ONE 9(2), 2014.
    """
    c = np.asarray(c, dtype=np.float64)
    if c.ndim == 1:
        c = c.reshape(-1, 1)
    n_samples = c.shape[0]

    radius = np.empty(n_samples)
    label_counts = np.empty(n_samples)
    k_all = np.empty(n_samples)

    nn = _SKNearestNeighbors()
    for label in np.unique(d):
        mask = d == label
        count = int(np.sum(mask))
        if count > 1:
            k = min(n_neighbors, count - 1)
            nn.set_params(n_neighbors=k)
            nn.fit(c[mask])
            r = nn.kneighbors()[0]
            radius[mask] = np.nextafter(r[:, -1], 0)
            k_all[mask] = k
        label_counts[mask] = count

    # Ignore points whose label occurs only once (they carry no information).
    mask = label_counts > 1
    n_samples = int(np.sum(mask))
    if n_samples == 0:
        return 0.0
    label_counts = label_counts[mask]
    k_all = k_all[mask]
    c = c[mask]
    radius = radius[mask]

    kd = _SKKDTree(c)
    m_all = kd.query_radius(c, radius, count_only=True, return_distance=False)
    m_all = np.asarray(m_all)

    mi = (
        _digamma(n_samples)
        + np.mean(_digamma(k_all))
        - np.mean(_digamma(label_counts))
        - np.mean(_digamma(m_all))
    )
    return max(0.0, float(mi))


def compute_band_event_joint_mi(
    envelopes: dict[str, np.ndarray],
    onset_indices: list[int],
    window_samples: int,
    *,
    fs: float = DEFAULT_FS,
    sub_sec: float = 1.0,
    sub_step_sec: float = 0.5,
    n_neighbors: int = 3,
    n_surrogates: int = 0,
    random_state: int = 0,
) -> dict | None:
    """Joint MI I(θ,α,β power ; pre/post-event) for one window size.

    For every onset in ``onset_indices`` the pre-event window
    ``[onset − window_samples, onset)`` (label 0) and post-event window
    ``[onset, onset + window_samples)`` (label 1) are tiled into sub-epochs; all
    sub-epochs are pooled across onsets into ``X (n, 3)`` / ``Y (n,)``.

    Returns ``None`` when there are too few usable sub-epochs; otherwise a dict
    with ``sum_mi_bits`` (Σ per-band), ``joint_mi_bits`` (true KSG), sample
    counts, and — if ``n_surrogates > 0`` — a label-shuffle permutation null.
    """
    if not _SKLEARN_MI_AVAILABLE:
        raise RuntimeError('scikit-learn is required for band-power × event MI.')

    sub_len = max(1, int(round(sub_sec * fs)))
    sub_step = max(1, int(round(sub_step_sec * fs)))
    n_times = len(next(iter(envelopes.values())))

    pre_blocks, post_blocks = [], []
    for onset in onset_indices:
        pre_lo, pre_hi = onset - window_samples, onset
        post_lo, post_hi = onset, onset + window_samples
        if pre_lo < 0 or post_hi > n_times:
            continue  # window falls outside the recording → skip this onset
        pre_blocks.append(_subepoch_features(envelopes, pre_lo, pre_hi, sub_len, sub_step))
        post_blocks.append(_subepoch_features(envelopes, post_lo, post_hi, sub_len, sub_step))

    pre_blocks = [b for b in pre_blocks if b.size]
    post_blocks = [b for b in post_blocks if b.size]
    if not pre_blocks or not post_blocks:
        return None

    x_pre = np.vstack(pre_blocks)
    x_post = np.vstack(post_blocks)
    n_pre, n_post = len(x_pre), len(x_post)

    # KNN estimators need at least n_neighbors+1 points in the smaller class.
    k = min(n_neighbors, n_pre - 1, n_post - 1)
    if k < 1:
        return None

    X = np.vstack([x_pre, x_post])
    Y = np.concatenate([np.zeros(n_pre), np.ones(n_post)]).astype(int)

    # (a) sklearn per-band MIs, summed (nats → bits).
    sum_mi = float(
        _sk_mutual_info_classif(
            X, Y, discrete_features=False, n_neighbors=k, random_state=random_state
        ).sum()
    ) * _NATS_TO_BITS

    # (b) true joint KSG estimate over the full 3-D feature vector (nats → bits).
    rng = np.random.RandomState(random_state)
    X_pre = _preprocess_continuous_features(X, rng)
    joint_mi = compute_mi_cd_multivariate(X_pre, Y, n_neighbors=k) * _NATS_TO_BITS

    out: dict = {
        'window_samples': int(window_samples),
        'n_pre': int(n_pre), 'n_post': int(n_post), 'n_neighbors': int(k),
        'sum_mi_bits': sum_mi, 'joint_mi_bits': joint_mi,
    }

    # Label-shuffle permutation null for the joint estimate (surrogate-gated MI).
    if n_surrogates and n_surrogates > 0:
        null = np.empty(int(n_surrogates))
        for i in range(int(n_surrogates)):
            y_shuf = rng.permutation(Y)
            null[i] = compute_mi_cd_multivariate(X_pre, y_shuf, n_neighbors=k) * _NATS_TO_BITS
        s_mean = float(null.mean())
        s_std = float(null.std(ddof=1)) if n_surrogates > 1 else 0.0
        out.update({
            'surrogate_n': int(n_surrogates),
            'surrogate_mean_bits': s_mean,
            'surrogate_std_bits': s_std,
            # +1 correction: observed is one draw from the null under H0.
            'surrogate_p_value': float((np.sum(null >= joint_mi) + 1) / (n_surrogates + 1)),
            'surrogate_z': float((joint_mi - s_mean) / s_std) if s_std > 0 else float('nan'),
        })
    return out


def run_band_event_mi_pipeline(
    signals_by_channel: dict[str, np.ndarray],
    onset_indices: list[int],
    *,
    fs: float = DEFAULT_FS,
    windows_sec: tuple[float, ...] = (5.0, 10.0, 15.0, 30.0),
    sub_sec: float = 1.0,
    sub_step_sec: float = 0.5,
    n_neighbors: int = 3,
    n_surrogates: int = 0,
    random_state: int = 0,
) -> pd.DataFrame:
    """Iterate window sizes × channels; return a tidy DataFrame.

    Columns: ``Window_Size``, ``Channel``, ``Joint_MI_Sum_Bits``,
    ``Joint_MI_KSG_Bits``, ``N_Pre``, ``N_Post``, ``K_Neighbors`` and, when
    ``n_surrogates > 0``, the surrogate-null columns.
    """
    records: list[dict] = []
    for ch_label, sig in signals_by_channel.items():
        envelopes = extract_band_envelopes(sig, fs=fs)
        for w_sec in windows_sec:
            w_samp = int(round(w_sec * fs))
            res = compute_band_event_joint_mi(
                envelopes, onset_indices, w_samp, fs=fs,
                sub_sec=sub_sec, sub_step_sec=sub_step_sec,
                n_neighbors=n_neighbors, n_surrogates=n_surrogates,
                random_state=random_state,
            )
            if res is None:
                print(f'  [skip] {ch_label} @ {w_sec:g}s — too few usable sub-epochs.')
                continue
            rec = {
                'Window_Size': float(w_sec), 'Channel': ch_label,
                'Joint_MI_Sum_Bits': res['sum_mi_bits'],
                'Joint_MI_KSG_Bits': res['joint_mi_bits'],
                'N_Pre': res['n_pre'], 'N_Post': res['n_post'],
                'K_Neighbors': res['n_neighbors'],
            }
            for key in ('surrogate_mean_bits', 'surrogate_std_bits',
                        'surrogate_p_value', 'surrogate_z'):
                if key in res:
                    rec[key] = res[key]
            records.append(rec)
    return pd.DataFrame.from_records(records)


def plot_band_event_mi(df: pd.DataFrame, title: str, outpath: str) -> None:
    """Line plot of joint MI (bits) vs. window size, per channel.

    Solid = true multivariate KSG joint MI; dashed = summed per-band MI, so the
    over-counting gap between the estimators is visible. A distinct colour per
    channel. If a surrogate p-value column is present, KSG points that clear
    p<.05 are ring-marked so a value within surrogate noise is not over-read.
    """
    if df.empty:
        print('  [warn] band-event MI DataFrame is empty — nothing to plot.')
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    palette = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
    has_p = 'surrogate_p_value' in df.columns
    for ci, (ch_label, sub) in enumerate(df.groupby('Channel')):
        sub = sub.sort_values('Window_Size')
        color = palette[ci % len(palette)]
        ax.plot(sub['Window_Size'], sub['Joint_MI_KSG_Bits'], linestyle='-',
                marker='o', linewidth=2, color=color, label=f'{ch_label} (joint KSG)')
        ax.plot(sub['Window_Size'], sub['Joint_MI_Sum_Bits'], linestyle='--',
                marker='s', linewidth=1.6, color=color, alpha=0.8,
                label=f'{ch_label} (Σ per-band)')
        if has_p:
            sig_pts = sub[sub['surrogate_p_value'] < 0.05]
            if not sig_pts.empty:
                ax.scatter(sig_pts['Window_Size'], sig_pts['Joint_MI_KSG_Bits'],
                           s=140, facecolors='none', edgecolors=color, linewidths=1.8,
                           zorder=5)

    ax.set_xlabel('Window Size (s)')
    ax.set_ylabel('Joint MI  I(θ, α, β power ; Event)  [bits]')
    star_note = '  (ringed = surrogate p<.05)' if has_p else ''
    ax.set_title(f'{title}\nJoint band-power mutual information vs. window size{star_note}')
    ax.set_xticks(sorted(df['Window_Size'].unique()))
    ax.grid(True, alpha=0.3)
    ax.legend(title='Channel / estimator', fontsize=8)
    _add_footer(fig, _provenance(DEFAULT_FS, DEFAULT_WIN_SEC, None,
                                 extra='band-power × event joint MI'))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(outpath, dpi=150)
    fig.savefig(os.path.splitext(outpath)[0] + '.svg')
    plt.close(fig)
    print(f'Saved: {outpath}')


def plot_peri_event_mi(
    win_result: dict[str, np.ndarray],
    event_onsets: list[tuple[str, float]],
    title: str,
    outpath: str,
    pre_sec: float = 30.0,
    post_sec: float = 60.0,
    fs: float = DEFAULT_FS,
    win_sec: float = DEFAULT_WIN_SEC,
    step_sec: float | None = None,
) -> None:
    """Peri-event MI time course: the zero-lag MI series re-expressed relative to
    each event onset (t=0) and overlaid, so the *latency and duration* of any
    coupling change is visible — not just the single pre→onset step the bar chart
    collapses it to. Reuses the already-computed global ``win_result`` (no extra
    MI estimation). A bold black line shows the across-event mean ±1σ.
    """
    t = np.asarray(win_result['time'], dtype=float)
    mi = np.asarray(win_result['joint_mi'], dtype=float)
    if not event_onsets:
        return
    # Common relative-time grid for averaging across events.
    grid = np.arange(-pre_sec, post_sec + 1e-9, step_sec or win_sec)
    stack = []
    fig, ax = plt.subplots(figsize=(11, 4.5))
    cmap = plt.get_cmap('tab10')
    for i, (name, onset_s) in enumerate(event_onsets):
        rel = t - onset_s
        sel = (rel >= -pre_sec) & (rel <= post_sec)
        if not np.any(sel):
            continue
        ax.plot(rel[sel], mi[sel], color=cmap(i % 10), lw=1.0, alpha=0.5,
                label=name)
        # Resample onto the common grid for the mean curve (NaN-safe).
        valid = sel & np.isfinite(mi)
        if np.count_nonzero(valid) >= 2:
            stack.append(np.interp(grid, rel[valid], mi[valid],
                                   left=np.nan, right=np.nan))
    ax.axvline(0.0, color='k', lw=1.2, ls='--', alpha=0.7)
    if stack:
        arr = np.vstack(stack)
        with np.errstate(invalid='ignore'):
            allnan = np.all(~np.isfinite(arr), axis=0)
            mean = np.full(grid.shape, np.nan)
            sd = np.full(grid.shape, np.nan)
            mean[~allnan] = np.nanmean(arr[:, ~allnan], axis=0)
            sd[~allnan] = np.nanstd(arr[:, ~allnan], axis=0)
        ax.plot(grid, mean, color='#111111', lw=2.5, label='event mean')
        ax.fill_between(grid, mean - sd, mean + sd, color='#111111', alpha=0.15)
    ax.set_xlabel('time relative to event onset (s)')
    ax.set_ylabel('joint MI (bits)')
    ax.set_ylim(bottom=0.0)
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    _add_footer(fig, _provenance(fs, win_sec, step_sec, extra='peri-event MI'))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(outpath, dpi=150)
    fig.savefig(os.path.splitext(outpath)[0] + '.svg')
    plt.close(fig)
    print(f'Saved: {outpath}')


# Colour-blind-friendly qualitative palette for the interactive per-event traces.
_PERI_EVENT_COLORS = (
    '#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7',
    '#56B4E9', '#F0E442', '#999999', '#8C564B', '#17BECF',
)


def _peri_event_series_on_grid(
    win_result: dict[str, np.ndarray],
    event_onsets: list[tuple[str, float]],
    grid: np.ndarray,
    pre_sec: float,
    post_sec: float,
) -> tuple[list[tuple[str, np.ndarray]], np.ndarray, np.ndarray]:
    """Resample every event's zero-lag MI trace onto the shared relative-time
    ``grid`` (NaN outside coverage) and return ``[(name, series), …]`` plus the
    across-event mean and ±1σ. The mean/σ replicate the static view's
    computation, so the interactive figure plots the *same* real data."""
    t = np.asarray(win_result['time'], dtype=float)
    mi = np.asarray(win_result['joint_mi'], dtype=float)
    per_event: list[tuple[str, np.ndarray]] = []
    stack: list[np.ndarray] = []
    for name, onset_s in event_onsets:
        rel = t - onset_s
        sel = (rel >= -pre_sec) & (rel <= post_sec) & np.isfinite(mi)
        if np.count_nonzero(sel) < 2:
            continue
        series = np.interp(grid, rel[sel], mi[sel], left=np.nan, right=np.nan)
        per_event.append((name, series))
        stack.append(series)
    if stack:
        arr = np.vstack(stack)
        with np.errstate(invalid='ignore'):
            allnan = np.all(~np.isfinite(arr), axis=0)
            mean = np.full(grid.shape, np.nan)
            sd = np.full(grid.shape, np.nan)
            mean[~allnan] = np.nanmean(arr[:, ~allnan], axis=0)
            sd[~allnan] = np.nanstd(arr[:, ~allnan], axis=0)
    else:
        mean = np.full(grid.shape, np.nan)
        sd = np.full(grid.shape, np.nan)
    return per_event, mean, sd


def plot_peri_event_mi_interactive(
    win_result: dict[str, np.ndarray],
    event_onsets: list[tuple[str, float]],
    title: str,
    outpath: str,
    pre_sec: float = 30.0,
    post_sec: float = 60.0,
    fs: float = DEFAULT_FS,
    win_sec: float = DEFAULT_WIN_SEC,
    step_sec: float | None = None,
) -> bool:
    """Interactive (Plotly) counterpart of :func:`plot_peri_event_mi`.

    Renders the same real per-event zero-lag MI traces — resampled onto a shared
    relative-time grid — as an animated HTML figure: a Play button + slider grow
    every line left→right in time, a dashed "Event Onset" marker sits at t=0, and
    the legend toggles individual events. The across-event mean is drawn as a bold
    black line. No new MI is estimated; the global ``win_result`` is reused.

    Returns ``True`` if the HTML was written, ``False`` if Plotly is unavailable
    or there is nothing to plot (callers keep the static PNG regardless).
    """
    if not _PLOTLY_AVAILABLE:
        print('  [warn] --peri-event-html needs Plotly (pip install plotly) — '
              'skipping interactive export; the static PNG was still written.')
        return False
    if not event_onsets:
        return False

    grid = np.arange(-pre_sec, post_sec + 1e-9, step_sec or win_sec)
    per_event, mean, sd = _peri_event_series_on_grid(
        win_result, event_onsets, grid, pre_sec, post_sec)
    if not per_event:
        print('  [warn] no event has ≥2 in-window MI samples — no interactive plot.')
        return False

    # Traces: one per event, then the across-event mean (drawn last / on top).
    traces = list(per_event) + [('event mean', mean)]

    def _scatter(idx: int, name: str, y: np.ndarray, upto: int) -> '_go.Scatter':
        is_mean = name == 'event mean'
        color = '#111111' if is_mean else _PERI_EVENT_COLORS[idx % len(_PERI_EVENT_COLORS)]
        return _go.Scatter(
            x=grid[:upto], y=y[:upto], mode='lines', name=name,
            legendgroup=name,
            line=dict(color=color, width=3.0 if is_mean else 1.8,
                      dash='solid'),
            opacity=1.0 if is_mean else 0.75,
            hovertemplate=f'{name}<br>t=%{{x:.1f}}s<br>MI=%{{y:.3f}} bits<extra></extra>',
        )

    base = [_scatter(i, name, y, 1) for i, (name, y) in enumerate(traces)]
    frames = [
        _go.Frame(name=f'{grid[k]:.1f}',
                  data=[_go.Scatter(x=grid[:k + 1], y=y[:k + 1])
                        for _, y in traces])
        for k in range(grid.size)
    ]

    # Y-range from the real data (headroom above the largest finite value).
    finite_vals = np.concatenate([y[np.isfinite(y)] for _, y in traces
                                  if np.any(np.isfinite(y))] or [np.array([0.0])])
    ymax = float(np.nanmax(finite_vals)) if finite_vals.size else 1.0
    y_top = max(0.1, ymax * 1.15)

    play_args = dict(frame=dict(duration=90, redraw=True),
                     transition=dict(duration=0), fromcurrent=True, mode='immediate')
    slider_steps = [
        dict(method='animate', label=f'{grid[k]:.0f}',
             args=[[f'{grid[k]:.1f}'],
                   dict(mode='immediate', frame=dict(duration=0, redraw=True),
                        transition=dict(duration=0))])
        for k in range(grid.size)
    ]

    fig = _go.Figure(data=base, frames=frames)
    fig.update_layout(
        title=title,
        template='plotly_white',
        xaxis=dict(title='time relative to event onset (s)',
                   range=[-pre_sec, post_sec], zeroline=False),
        yaxis=dict(title='joint MI (bits)', range=[0.0, y_top]),
        legend=dict(title='event  (click to toggle)', x=1.02, y=1.0),
        hovermode='x unified',
        updatemenus=[dict(
            type='buttons', direction='left', showactive=False,
            x=0.0, y=1.14, xanchor='left', yanchor='top',
            buttons=[
                dict(label='▶ Play', method='animate', args=[None, play_args]),
                dict(label='⏸ Pause', method='animate',
                     args=[[None], dict(mode='immediate',
                                        frame=dict(duration=0, redraw=False),
                                        transition=dict(duration=0))]),
            ])],
        sliders=[dict(active=0, x=0.08, len=0.92, y=0.0, xanchor='left',
                      currentvalue=dict(prefix='t = ', suffix=' s'),
                      pad=dict(t=40), steps=slider_steps)],
        margin=dict(t=95, r=180),
    )
    fig.add_vline(x=0.0, line=dict(color='crimson', width=2, dash='dash'),
                  annotation_text='Event Onset', annotation_position='top',
                  annotation=dict(font=dict(color='crimson')))
    fig.add_annotation(text=_provenance(fs, win_sec, step_sec, extra='peri-event MI'),
                       xref='paper', yref='paper', x=0.0, y=-0.16, showarrow=False,
                       font=dict(size=9, color='#666666'), align='left')
    fig.write_html(outpath, include_plotlyjs='cdn', auto_open=False)
    print(f'Saved: {outpath}')
    return True


def plot_joint_distribution(
    joint: dict[str, np.ndarray | float],
    title: str,
    outpath: str,
    label_x: str = 'X',
    label_y: str = 'Y',
    sig: dict[str, float] | None = None,
) -> None:
    """Plot the 2-D joint probability distribution P(X, Y) as a heatmap with the
    two marginals, annotated with the mutual information it implies."""
    from matplotlib.gridspec import GridSpec

    pxy = np.asarray(joint['pxy'], dtype=float)
    px = np.asarray(joint['px'], dtype=float)
    py = np.asarray(joint['py'], dtype=float)
    x_edges = np.asarray(joint['x_edges'], dtype=float)
    y_edges = np.asarray(joint['y_edges'], dtype=float)
    nx, ny = pxy.shape

    fig = plt.figure(figsize=(8.5, 8.5))
    gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                  wspace=0.05, hspace=0.05)
    ax_joint = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_joint)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_joint)

    # Plot the joint *mass* P(X,Y) in equal-cell bin-index space (the copula /
    # rank view). With quantile binning the cells have very unequal widths in
    # signal units, so a signal-axis density view collapses visually; in
    # bin-index space every cell is equal, the marginals are flat by
    # construction, and any off-diagonal structure (the actual dependence that
    # MI measures) is directly visible. Tick labels carry the real edge values.
    im = ax_joint.imshow(pxy.T, origin='lower', aspect='auto', cmap='magma',
                         extent=[0, nx, 0, ny])
    ax_joint.set_xlabel(f'{label_x} (z-scored; bin index)')
    ax_joint.set_ylabel(f'{label_y} (z-scored; bin index)')

    def _edge_ticks(ax, edges, n, axis):
        pos = np.linspace(0, n, min(n + 1, 6))
        labels = [f'{np.interp(p, np.arange(edges.size), edges):.2f}' for p in pos]
        (ax.set_xticks if axis == 'x' else ax.set_yticks)(pos)
        (ax.set_xticklabels if axis == 'x' else ax.set_yticklabels)(labels)

    _edge_ticks(ax_joint, x_edges, nx, 'x')
    _edge_ticks(ax_joint, y_edges, ny, 'y')

    ax_top.bar(np.arange(nx) + 0.5, px, width=1.0, color='#4292c6')
    ax_top.tick_params(labelbottom=False)
    ax_top.set_ylabel('P(X)')
    ax_right.barh(np.arange(ny) + 0.5, py, height=1.0, color='#41ab5d')
    ax_right.tick_params(labelleft=False)
    ax_right.set_xlabel('P(Y)')

    cax = fig.add_axes([0.13, 0.06, 0.5, 0.015])
    fig.colorbar(im, cax=cax, orientation='horizontal', label='P(X, Y) per cell')

    lines = [
        f"I(X;Y) = {float(joint['mutual_information']):.4f} bits",
        f"I_MM   = {float(joint['mutual_information_mm']):.4f} bits",
        f"I_norm = {float(joint['mutual_information_norm']):.4f}",
        f"H(X) = {float(joint['entropy_x']):.3f}  H(Y) = {float(joint['entropy_y']):.3f}",
        f"H(X,Y) = {float(joint['entropy_xy']):.3f}  N = {int(joint['n_samples'])}",
    ]
    if sig is not None and sig.get('n_surrogates', 0):
        lines.append(
            f"surrogate null: p = {sig['p_value']:.3g}, z = {sig['z']:.2f}")
    ax_top.text(0.02, 0.95, '\n'.join(lines), transform=ax_top.transAxes,
                va='top', ha='left', fontsize=9, family='monospace',
                bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.85))

    # Surrogate-null inset (top-right cell, otherwise empty): the circular-shift
    # MI null with the observed MI marked, so the reader can *see* whether the
    # observed value sits in the tail rather than trusting the printed p alone.
    if sig is not None and sig.get('n_surrogates', 0) and \
            np.asarray(sig.get('surrogates', [])).size:
        ax_sig = fig.add_subplot(gs[0, 1])
        sur = np.asarray(sig['surrogates'], dtype=float)
        ax_sig.hist(sur, bins=min(30, max(5, sur.size // 5)),
                    color='#9ecae1', edgecolor='0.5', linewidth=0.4)
        ax_sig.axvline(float(sig['mutual_information']), color='#d62728', lw=1.6,
                       label='observed')
        ax_sig.set_title('surrogate null', fontsize=8)
        ax_sig.set_xlabel('MI (bits)', fontsize=7)
        ax_sig.tick_params(labelsize=6)
        ax_sig.legend(fontsize=6, loc='upper right')

    fig.suptitle(title, fontsize=13, fontweight='bold')
    _add_footer(fig, _provenance(float(joint.get('fs', DEFAULT_FS)),
                                 DEFAULT_WIN_SEC, None, extra='joint-MI'))
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    fig.savefig(os.path.splitext(outpath)[0] + '.svg')
    plt.close(fig)
    print(f'Saved: {outpath}')


def plot_joint_excess(
    joint: dict[str, np.ndarray | float],
    title: str,
    outpath: str,
    label_x: str = 'X',
    label_y: str = 'Y',
) -> None:
    """Excess-mass view of the joint distribution: P(X,Y) − P(X)P(Y) in
    bin-index space, on a zero-centred diverging colourmap.

    MI measures exactly the divergence of P(X,Y) from the independence product
    P(X)P(Y); plotting that *difference* (rather than the raw mass) makes the
    dependence structure pop — independence reads as flat mid-grey, positive
    association as a warm diagonal, negative/non-monotone coupling as cool
    off-diagonal cells — which the raw-mass magma heatmap cannot show because
    the marginals are flat by construction under quantile binning.
    """
    pxy = np.asarray(joint['pxy'], dtype=float)
    px = np.asarray(joint['px'], dtype=float)
    py = np.asarray(joint['py'], dtype=float)
    x_edges = np.asarray(joint['x_edges'], dtype=float)
    y_edges = np.asarray(joint['y_edges'], dtype=float)
    nx, ny = pxy.shape
    excess = pxy - np.outer(px, py)
    vmax = float(np.max(np.abs(excess))) or 1e-9

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(excess.T, origin='lower', aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, extent=[0, nx, 0, ny])

    def _edge_ticks(edges, n, axis):
        pos = np.linspace(0, n, min(n + 1, 6))
        labels = [f'{np.interp(p, np.arange(edges.size), edges):.2f}' for p in pos]
        (ax.set_xticks if axis == 'x' else ax.set_yticks)(pos)
        (ax.set_xticklabels if axis == 'x' else ax.set_yticklabels)(labels)

    _edge_ticks(x_edges, nx, 'x')
    _edge_ticks(y_edges, ny, 'y')
    ax.set_xlabel(f'{label_x} (z-scored; bin index)')
    ax.set_ylabel(f'{label_y} (z-scored; bin index)')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('P(X,Y) − P(X)P(Y)  (excess over independence)')
    ax.set_title(f'{title}\nI(X;Y) = {float(joint["mutual_information"]):.4f} bits '
                 f'(MM {float(joint["mutual_information_mm"]):.4f})', fontsize=11)
    _add_footer(fig, _provenance(float(joint.get('fs', DEFAULT_FS)),
                                 DEFAULT_WIN_SEC, None, extra='joint-MI excess'))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    fig.savefig(os.path.splitext(outpath)[0] + '.svg')
    plt.close(fig)
    print(f'Saved: {outpath}')


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
    fs: float = DEFAULT_FS,
    win_sec: float = DEFAULT_WIN_SEC,
    step_sec: float | None = None,
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

    colors = BAND_COLORS
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
    # ±1σ dispersion band: a centred rolling std of the per-window BandEn, so a
    # genuine entropy excursion is visually separable from window-to-window
    # estimator jitter. NaN-masked (low-quality) windows leave gaps in the band.
    bits_std = _rolling_std(entropy_bits)
    t_arr = np.asarray(t)
    with np.errstate(invalid='ignore'):
        lo_band = entropy_bits_smooth - bits_std
        hi_band = entropy_bits_smooth + bits_std
    ax_entropy.fill_between(t_arr, lo_band, hi_band, color='#111111', alpha=0.12,
                            linewidth=0, label='BandEn ±1σ')
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

        # Dominant lag (argmax-τ) on a twin axis: reveals whether the lag that
        # carries the strongest coupling drifts over time — a phenomenon the
        # mean/max traces alone hide. Drawn as a faint scatter so it never
        # competes with the MI curves.
        if 'lagged_mi_best_tau_ms' in sync_result:
            ax_tau = ax_sync.twinx()
            ax_tau.scatter(t_sync, sync_result['lagged_mi_best_tau_ms'],
                           s=6, color='#7b3294', alpha=0.45, label='best τ')
            ax_tau.set_ylabel('best τ (ms)', color='#7b3294')
            ax_tau.tick_params(axis='y', labelcolor='#7b3294')
            ax_tau.set_ylim(bottom=0.0)

    # ── x-axis formatting ──────────────────────────────────────────────────────
    if use_abs:
        fmt = mdates.DateFormatter('%H:%M:%S')
        axes[-1].xaxis.set_major_formatter(fmt)
        axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
        axes[-1].set_xlabel('Time (local, UTC+8)')
        fig.autofmt_xdate(rotation=30, ha='right')

    fig.tight_layout(rect=(0, 0.015, 1, 1))
    _add_footer(fig, _provenance(fs, win_sec, step_sec, extra='band-entropy'))
    fig.savefig(outpath, dpi=150)
    fig.savefig(os.path.splitext(outpath)[0] + '.svg')
    plt.close(fig)
    print(f'Saved: {outpath}')


def plot_band_composition(
    entropy_result: dict[str, np.ndarray],
    title: str,
    outpath: str,
    t_offset: float = 0.0,
    time_us_epoch: int | None = None,
    ibrain_events: bool = False,
    fs: float = DEFAULT_FS,
    win_sec: float = DEFAULT_WIN_SEC,
    step_sec: float | None = None,
) -> None:
    """Stacked-area view of the θ/α/β proportions on a single axis.

    The three proportions sum to 1 at every window, so a stacked area makes the
    conservation-of-mass structure explicit (which the three separate panels in
    ``plot_band_entropy`` obscure) and lets a reader read *redistribution*
    between bands at a glance. Low-quality (NaN) windows are dropped from the
    stack so masked spans appear as gaps rather than zero-height slabs.
    """
    use_abs = time_us_epoch is not None
    rel_t = np.asarray(entropy_result['time'], dtype=float) + t_offset
    band_names = [name for name, _ in BAND_DEFINITIONS]
    stack = np.vstack([np.asarray(entropy_result[f'p_{n}'], dtype=float)
                       for n in band_names])
    # Smooth each proportion for a legible band; keep NaN gaps.
    stack_s = np.vstack([_smooth_series(stack[i]) for i in range(stack.shape[0])])
    valid = np.all(np.isfinite(stack_s), axis=0)

    t = _rel_times_to_dt(rel_t, time_us_epoch) if use_abs else rel_t
    t_arr = np.asarray(t, dtype=object if use_abs else float)

    fig, ax = plt.subplots(figsize=(14, 4.0))
    # stackplot cannot span NaN gaps, so plot contiguous valid runs separately.
    idx = np.where(valid)[0]
    if idx.size:
        splits = np.where(np.diff(idx) > 1)[0] + 1
        first = True
        for run in np.split(idx, splits):
            ax.stackplot(
                t_arr[run], *[stack_s[i, run] for i in range(stack.shape[0])],
                labels=band_names if first else ['_nolegend_'] * stack.shape[0],
                colors=[BAND_COLORS[n] for n in band_names], alpha=0.85)
            first = False
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('cumulative proportion')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=9, ncol=3)
    ax.grid(True, alpha=0.25)
    if ibrain_events and use_abs:
        _overlay_ibrain_events(ax, t_arr[0], t_arr[-1], use_abs)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.set_xlabel('Time (local, UTC+8)')
        fig.autofmt_xdate(rotation=30, ha='right')
    else:
        ax.set_xlabel('Time (s)')
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _add_footer(fig, _provenance(fs, win_sec, step_sec, extra='band-composition'))
    fig.savefig(outpath, dpi=150)
    fig.savefig(os.path.splitext(outpath)[0] + '.svg')
    plt.close(fig)
    print(f'Saved: {outpath}')


def _simplex_xy(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map rows of a (N,3) probability matrix (θ,α,β) to 2-D barycentric
    coordinates of an equilateral triangle (θ=bottom-left, α=bottom-right,
    β=top)."""
    p = np.asarray(p, dtype=float)
    x = p[:, 1] + 0.5 * p[:, 2]
    y = (np.sqrt(3.0) / 2.0) * p[:, 2]
    return x, y


def plot_band_ternary(
    entropy_result: dict[str, np.ndarray],
    title: str,
    outpath: str,
    fs: float = DEFAULT_FS,
    win_sec: float = DEFAULT_WIN_SEC,
    step_sec: float | None = None,
) -> None:
    """Ternary (2-simplex) view of the per-window (p_θ, p_α, p_β) trajectory.

    Because the three proportions sum to 1 they live on a 2-simplex; plotting
    them in the triangle reveals *attractor states* (e.g. an α-corner rest vs. a
    β-corner task state) and the path between them — structure the time series
    only hints at, and the geometric intuition behind why BASD (band-aware)
    carries information BandEn (band-blind) discards. Points are coloured by
    time; the mean (centroid) is marked.
    """
    band_names = [name for name, _ in BAND_DEFINITIONS]
    P = np.column_stack([np.asarray(entropy_result[f'p_{n}'], dtype=float)
                         for n in band_names])
    finite = np.all(np.isfinite(P), axis=0) if P.ndim == 1 else np.all(np.isfinite(P), axis=1)
    P = P[finite]
    tvec = np.asarray(entropy_result['time'], dtype=float)[finite]
    if P.shape[0] == 0:
        print(f'  [warn] ternary: no clean windows to plot — skipping {outpath}.')
        return

    x, y = _simplex_xy(P)
    # Triangle vertices.
    vx = [0.0, 1.0, 0.5, 0.0]
    vy = [0.0, 0.0, np.sqrt(3.0) / 2.0, 0.0]

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    ax.plot(vx, vy, color='0.4', lw=1.2)
    # Light iso-proportion grid lines every 0.2 (constant α, constant β, constant θ).
    h = np.sqrt(3) / 2
    for f in np.arange(0.2, 1.0, 0.2):
        # constant α = f
        ax.plot([f, f + 0.5 * (1 - f)], [0, h * (1 - f)], color='0.85', lw=0.6)
        # constant β = f (horizontal)
        ax.plot([0.5 * f, 1 - 0.5 * f], [h * f, h * f], color='0.85', lw=0.6)
        # constant θ = f
        ax.plot([0.5 * (1 - f), 1 - f], [h * (1 - f), 0], color='0.85', lw=0.6)
    sc = ax.scatter(x, y, c=tvec, cmap='viridis', s=10, alpha=0.6, zorder=3)
    # Centroid.
    cx, cy = _simplex_xy(P.mean(axis=0)[None, :])
    ax.scatter(cx, cy, marker='*', s=320, color='#d62728', edgecolor='k',
               zorder=4, label='mean')
    ax.annotate('θ', (0, -0.04), ha='center', va='top', fontsize=14, color=BAND_COLORS['theta'])
    ax.annotate('α', (1, -0.04), ha='center', va='top', fontsize=14, color=BAND_COLORS['alpha'])
    ax.annotate('β', (0.5, np.sqrt(3) / 2 + 0.03), ha='center', va='bottom', fontsize=14, color=BAND_COLORS['beta'])
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('time (s)')
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _add_footer(fig, _provenance(fs, win_sec, step_sec, extra='band-ternary'))
    fig.savefig(outpath, dpi=150)
    fig.savefig(os.path.splitext(outpath)[0] + '.svg')
    plt.close(fig)
    print(f'Saved: {outpath}')


def plot_focus_relax_scatter(
    entropy_result: dict[str, np.ndarray],
    title: str,
    outpath: str,
    t_offset: float = 0.0,
    fs: float = DEFAULT_FS,
    win_sec: float = DEFAULT_WIN_SEC,
    step_sec: float | None = None,
) -> None:
    """Scatter of per-window Focus vs Relaxation qEEG indices, coloured by time.

    Each point is one analysis window placed at (Focus, Relaxation); colour
    encodes the window-centre time and a faint line connects consecutive windows,
    so the figure traces the *trajectory* through Focus–Relax space rather than a
    static cloud. The two indices are derived from the same per-window
    (p_θ, p_α, p_β) relative band powers already computed for the band-entropy
    series — these are exactly the inputs qeeg_indices uses — so no PSDs are
    recomputed. Both indices live in ≈[−1, 1] (the bounded-ratio range); the grey
    cross marks the neutral origin.
    """
    if not _QEEG_AVAILABLE:
        print(f'  [warn] focus/relax scatter: qeeg_indices unavailable — '
              f'skipping {outpath}.')
        return

    band_names = [name for name, _ in BAND_DEFINITIONS]  # theta, alpha, beta
    P = np.column_stack([np.asarray(entropy_result[f'p_{n}'], dtype=float)
                         for n in band_names])
    finite = np.all(np.isfinite(P), axis=1)
    P = P[finite]
    tvec = np.asarray(entropy_result['time'], dtype=float)[finite] + t_offset
    if P.shape[0] == 0:
        print(f'  [warn] focus/relax scatter: no clean windows to plot — '
              f'skipping {outpath}.')
        return

    focus = np.array([focus_index(th, al, be) for th, al, be in P])
    relax = np.array([relaxation_index(th, al, be) for th, al, be in P])

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    # Neutral-point cross (bounded ratio = 0).
    ax.axhline(0.0, color='0.85', lw=0.8, zorder=0)
    ax.axvline(0.0, color='0.85', lw=0.8, zorder=0)
    # Faint temporal trajectory through the (Focus, Relax) plane.
    ax.plot(focus, relax, color='0.6', lw=0.5, alpha=0.4, zorder=2)
    sc = ax.scatter(focus, relax, c=tvec, cmap='viridis', s=18, alpha=0.75,
                    edgecolor='none', zorder=3)
    # Start / end markers so the trajectory direction is unambiguous.
    ax.scatter(focus[0], relax[0], marker='o', s=130, facecolor='none',
               edgecolor='#2ca02c', lw=2.0, zorder=4, label='start')
    ax.scatter(focus[-1], relax[-1], marker='s', s=130, facecolor='none',
               edgecolor='#d62728', lw=2.0, zorder=4, label='end')
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('time (s)')

    lim = max(1.05, float(np.nanmax(np.abs(np.concatenate([focus, relax])))) * 1.05)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('Focus index')
    ax.set_ylabel('Relaxation index')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _add_footer(fig, _provenance(fs, win_sec, step_sec, extra='focus-relax'))
    fig.savefig(outpath, dpi=150)
    fig.savefig(os.path.splitext(outpath)[0] + '.svg')
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
    parser.add_argument('--smooth', type=int, default=DEFAULT_SMOOTH_WINDOW,
                        metavar='N',
                        help=('Centred moving-average length (in windows) for every '
                              f'smoothed trace (default {DEFAULT_SMOOTH_WINDOW}). The '
                              'effective duration is N×step seconds.'))
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
                        help=('Channel-median EEG quality (0-1) threshold. Windows below '
                              'this are NaN-masked in the band-entropy/sync time series, '
                              'and it is also the keep threshold for --clean micro-epochs '
                              f'(default {_QUALITY_THRESHOLD:g}).'))
    parser.add_argument('--no-quality-mask', action='store_true', default=False,
                        help=('Do not mask low-quality windows in the band-entropy/sync '
                              'time series. By default each analysis window is scored with '
                              'eeg_quality_v2 (same method as plot_tflite_summary) and '
                              'windows below --quality-threshold are discarded (set NaN).'))
    parser.add_argument('--sync-pair', type=int, nargs=2, metavar=('LEFT', 'RIGHT'),
                        help='Optional 1-based left/right channel pair for lagged-MI synchrony.')
    parser.add_argument('--tau-ms', type=float, nargs='+', default=list(DEFAULT_TAU_MS),
                        metavar='MS',
                        help=('Positive non-zero delays in milliseconds for lagged MI '
                              f'(default: {" ".join(f"{tau:g}" for tau in DEFAULT_TAU_MS)}).'))
    parser.add_argument('--mi-bins', type=int, default=DEFAULT_MI_BINS, metavar='N',
                        help=f'Histogram bins for mutual-information estimation (default {DEFAULT_MI_BINS}).')
    parser.add_argument('--joint-mi', action='store_true', default=False,
                        help=('Joint-distribution mode: estimate the 2-D joint probability '
                              'distribution P(X,Y) of two channels and the mutual information '
                              'it implies (whole-recording + sliding-window series + a 2-D '
                              'heatmap). By default uses the two denoised channels (see '
                              '--denoise); otherwise the --joint-pair channels.'))
    parser.add_argument('--joint-pair', type=int, nargs=2, metavar=('CH_X', 'CH_Y'),
                        default=[1, 2],
                        help=('1-based channel pair for --joint-mi when NOT denoising '
                              '(default: 1 2). Ignored under --denoise (the two model '
                              'outputs are always used).'))
    parser.add_argument('--denoise', action='store_true', default=False,
                        help=('--joint-mi only: produce the two channels with the TinyUNetV4 '
                              'neural denoiser (4 raw ch in → 2 denoised ch out @200 Hz), '
                              'matching the data_analysis inference pipeline. Requires PyTorch '
                              'and the eeg_denoise model package.'))
    parser.add_argument('--mi-binning', choices=('quantile', 'uniform'),
                        default='quantile',
                        help=('--joint-mi only: histogram bin-edge strategy. "quantile" '
                              '(default) places per-axis equiprobable edges so every bin '
                              'is populated — robust to EEG artefacts and recommended. '
                              '"uniform" uses equal-width edges (collapses when artefacts '
                              'inflate the amplitude range).'))
    parser.add_argument('--mi-surrogates', type=int, default=200, metavar='N',
                        help=('--joint-mi only: number of circular-shift surrogates for the '
                              'MI significance test (0 disables; default 200).'))
    parser.add_argument('--out', metavar='DIR',
                        help='Output directory (default: same dir as CSV).')
    parser.add_argument('--subject', default=None, help='Participant key for iBrainCenter event analysis')
    parser.add_argument('--ibrain-events', action='store_true', default=False,
                        help=('Overlay iBrainCenter session event markers and '
                              'convert x-axis to absolute local time (UTC+8).'))
    parser.add_argument('--peri-event-html', action='store_true', default=False,
                        help=('--joint-mi with --ibrain-events: also write an '
                              'interactive Plotly peri-event MI animation '
                              '(*_peri_event.html) alongside the static PNG — a '
                              'Play button/slider grows each event\'s real MI '
                              'trace left→right around onset. Requires plotly.'))
    parser.add_argument('--band-event-mi', action='store_true', default=False,
                        help=('Band-power × event mode: estimate the joint MI '
                              'I(θ,α,β power ; pre/post-event) across window sizes '
                              '(--mi-windows) for two channels (--band-mi-channels). '
                              'Reports both the summed per-band and the true '
                              'multivariate KSG estimate (bits). Onsets come from '
                              '--event-onset or, with --ibrain-events, every session '
                              'event pooled together. Requires scikit-learn.'))
    parser.add_argument('--event-onset', type=float, metavar='SEC',
                        help=('--band-event-mi: single event onset, in seconds from '
                              'recording start. Ignored when --ibrain-events pools '
                              'onsets over all session events.'))
    parser.add_argument('--mi-windows', type=float, nargs='+',
                        default=[5.0, 10.0, 15.0, 30.0], metavar='SEC',
                        help=('--band-event-mi: pre/post window sizes in seconds '
                              '(default: 5 10 15 30).'))
    parser.add_argument('--band-mi-channels', type=int, nargs='+', default=[1, 2],
                        metavar='N',
                        help=('--band-event-mi: 1-based channels to analyse '
                              '(default: 1 2).'))
    parser.add_argument('--mi-sub-sec', type=float, default=1.0, metavar='SEC',
                        help=('--band-event-mi: sub-epoch length in seconds used to '
                              'draw feature samples inside each window (default 1.0).'))
    parser.add_argument('--mi-sub-step', type=float, default=0.5, metavar='SEC',
                        help=('--band-event-mi: sub-epoch step in seconds; < --mi-sub-sec '
                              'means overlapping sub-epochs → more samples (default 0.5).'))
    return parser.parse_args()


def _run_joint_mi_mode(args: argparse.Namespace,
                       time_us: np.ndarray,
                       data_filt: np.ndarray,
                       data_raw: np.ndarray | None,
                       outdir: str) -> None:
    """Estimate the joint probability distribution of two channels and the
    mutual information it implies.

    Two channel sources:
      * ``--denoise`` — the two outputs of the TinyUNetV4 neural denoiser
        (4 raw ch → 2 denoised ch @ 200 Hz). This is the requested path:
        "denoised ch1 / denoised ch2".
      * otherwise     — the two ``--joint-pair`` channels of the (bandpassed)
        signal at the recording's own sampling rate.

    Outputs: a whole-recording joint-distribution heatmap (P(X,Y) + marginals +
    MI), a sliding-window zero-lag MI series (CSV + PNG), and a one-row summary
    CSV with the MI, its Miller–Madow correction and the surrogate-null test.

    With ``--ibrain-events`` the MI time series additionally carries the
    iBrainCenter event overlay (absolute local time), and a per-event
    *pre-event vs onset* joint-distribution comparison is produced (CSV + bar
    plot) so the change in coupling at each event onset is quantified.
    """
    bins = args.mi_bins
    binning = args.mi_binning
    if args.denoise:
        if data_raw is None:
            sys.exit('Error: --denoise needs the raw channels (do not combine with '
                     'a path that drops them).')
        print('Denoising channels with TinyUNetV4 '
              '(4 raw ch → 2 denoised ch @200 Hz)…', flush=True)
        try:
            chans, fs_eff = denoise_channels(data_raw, fs=args.fs)
        except (RuntimeError, ValueError) as exc:
            sys.exit(f'Error: {exc}')
        if chans.shape[1] < 2:
            sys.exit('Error: denoiser returned fewer than two channels.')
        sig_x, sig_y = chans[:, 0], chans[:, 1]
        label_x, label_y = 'denoised ch1', 'denoised ch2'
        pair_suffix = 'denoised_ch1_ch2'
    else:
        cx, cy = args.joint_pair
        ix, iy = cx - 1, cy - 1
        if ix < 0 or iy < 0:
            sys.exit('Error: --joint-pair channels must be >= 1.')
        if ix >= data_filt.shape[1] or iy >= data_filt.shape[1]:
            sys.exit(f'Error: --joint-pair channel not found (file has '
                     f'{data_filt.shape[1]} channels).')
        if ix == iy:
            sys.exit('Error: --joint-pair must specify two different channels.')
        sig_x, sig_y = data_filt[:, ix], data_filt[:, iy]
        fs_eff = args.fs
        label_x, label_y = f'ch{cx}', f'ch{cy}'
        pair_suffix = f'ch{cx}_ch{cy}'

    print(f'Computing joint probability distribution & mutual information '
          f'— {label_x} vs {label_y}, bins={bins} ({binning}), fs={fs_eff:g}Hz')

    joint = compute_joint_probability(sig_x, sig_y, bins=bins, binning=binning)
    joint['fs'] = fs_eff
    sig = compute_joint_mi_significance(
        sig_x, sig_y, bins=bins, binning=binning, n_surrogates=args.mi_surrogates)
    win_result = compute_joint_mi_windowed(
        sig_x, sig_y, fs=fs_eff, win_sec=args.win, step_sec=args.step,
        bins=bins, binning=binning)

    # ── Quality masking of the windowed MI series (non-denoise path only) ────────
    # Route the same eeg_quality_v2 mask used by the band-entropy series through
    # the zero-lag MI series so artefactual windows cannot manufacture spurious
    # MI spikes. Skipped under --denoise (model outputs are not raw device
    # channels, so the device-tuned quality params do not apply) and when the
    # caller opts out or the QC modules are unavailable.
    mi_quality = None
    if (not args.denoise) and (not args.no_quality_mask) and _QC_AVAILABLE:
        two_ch = np.column_stack([sig_x, sig_y]).astype(float)
        mi_quality = compute_quality_windowed_aligned(
            two_ch, fs=fs_eff, win_sec=args.win, step_sec=args.step)
        if mi_quality.shape[0] == win_result['time'].shape[0]:
            mmask = mi_quality < args.quality_threshold
            _mask_low_quality(win_result, mmask, ['joint_mi', 'joint_mi_norm'])
            print(f'  Masked {int(mmask.sum())}/{mmask.size} MI windows '
                  f'({100.0 * mmask.mean():.1f}%) below quality '
                  f'{args.quality_threshold:g}.')
        else:
            print('  [warn] MI quality-window count mismatch — skipping MI masking.')
            mi_quality = None

    basename = os.path.splitext(os.path.basename(args.csv))[0]
    stem = f'{basename}_joint_mi_{pair_suffix}'

    # ── Whole-recording summary (one row) ───────────────────────────────────────
    summary = {
        'channel_x': label_x, 'channel_y': label_y, 'fs_hz': fs_eff,
        'mi_bins': bins, 'mi_binning': binning,
        'n_bins_x': joint['n_bins_x'], 'n_bins_y': joint['n_bins_y'],
        'n_samples': joint['n_samples'],
        'mutual_information_bits': joint['mutual_information'],
        'mutual_information_mm_bits': joint['mutual_information_mm'],
        'mutual_information_norm': joint['mutual_information_norm'],
        'entropy_x_bits': joint['entropy_x'], 'entropy_y_bits': joint['entropy_y'],
        'entropy_xy_bits': joint['entropy_xy'],
        'surrogate_n': sig['n_surrogates'], 'surrogate_mean_bits': sig['surrogate_mean'],
        'surrogate_std_bits': sig['surrogate_std'],
        'surrogate_p_value': sig['p_value'], 'surrogate_z': sig['z'],
    }
    summary_csv = os.path.join(outdir, f'{stem}_summary.csv')
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    print(f'Saved: {summary_csv}')

    # ── Sliding-window MI series ────────────────────────────────────────────────
    series_csv = os.path.join(outdir, f'{stem}_timeseries.csv')
    series_df = {
        'time_s': win_result['time'],
        'joint_mi': win_result['joint_mi'],
        'joint_mi_norm': win_result['joint_mi_norm'],
    }
    if mi_quality is not None:
        series_df['quality'] = mi_quality
    pd.DataFrame(series_df).to_csv(series_csv, index=False)
    print(f'Saved: {series_csv}')

    # ── Plots ───────────────────────────────────────────────────────────────────
    heatmap_png = os.path.join(outdir, f'{stem}_distribution.png')
    plot_joint_distribution(
        joint,
        title=(f'Joint P({label_x}, {label_y}) — {os.path.basename(args.csv)} '
               f'[{binning}, {bins} bins]'),
        outpath=heatmap_png, label_x=label_x, label_y=label_y, sig=sig)

    excess_png = os.path.join(outdir, f'{stem}_excess.png')
    plot_joint_excess(
        joint,
        title=(f'Excess mass P−P·P ({label_x}, {label_y}) — '
               f'{os.path.basename(args.csv)} [{binning}, {bins} bins]'),
        outpath=excess_png, label_x=label_x, label_y=label_y)

    # iBrainCenter event overlay (absolute local time) on the MI time series.
    use_events = bool(args.ibrain_events)
    if use_events and not _IBRAIN_AVAILABLE:
        print('  [warn] --ibrain-events set but plot_event_markers unavailable — '
              'skipping event overlay.')
        use_events = False
    t0_us = int(time_us[0])

    series_png = os.path.join(outdir, f'{stem}_timeseries.png')
    fig, ax = plt.subplots(figsize=(14, 3.2))
    if use_events:
        t_axis = _rel_times_to_dt(win_result['time'], t0_us)
    else:
        t_axis = win_result['time']
    ax.plot(t_axis, win_result['joint_mi'], color='#111111', lw=1.0,
            alpha=0.4, label='joint MI (bits)')
    ax.plot(t_axis, _smooth_series(win_result['joint_mi']),
            color='#111111', lw=2.0, label='joint MI smooth')
    ax.set_ylabel('MI (bits)')
    ax.set_ylim(bottom=0.0)
    ax.set_title(f'Zero-lag mutual information — {label_x} vs {label_y} '
                 f'(win={args.win:g}s)')
    ax.grid(True, alpha=0.3)
    if use_events:
        _overlay_ibrain_events(ax, t_axis[0], t_axis[-1], use_abs=True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.set_xlabel('Time (local, UTC+8)')
        fig.autofmt_xdate(rotation=30, ha='right')
    else:
        ax.set_xlabel('Time (s)')
    ax.legend(loc='upper right', fontsize=9)
    _add_footer(fig, _provenance(fs_eff, args.win, args.step,
                                 extra=f'joint-MI series ({binning},{bins}b)'))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(series_png, dpi=150)
    fig.savefig(os.path.splitext(series_png)[0] + '.svg')
    plt.close(fig)
    print(f'Saved: {series_png}')

    # ── Pre-event vs onset joint-distribution comparison (--ibrain-events) ───────
    if use_events:
        events = compute_event_pre_onset_joint_mi(
            sig_x, sig_y, t0_us, fs=fs_eff, bins=bins, binning=binning, subject=args.subject,
            n_surrogates=min(100, args.mi_surrogates) if args.mi_surrogates else 0)
        if not events:
            print('  [warn] no iBrainCenter events fall within this recording — '
                  'skipping pre-event/onset comparison.')
        else:
            ev_rows = [{
                'event': e['name'], 'onset_rel_s': e['onset_rel_s'],
                'pre_mi_bits': e['pre']['mutual_information'],
                'pre_mi_mm_bits': e['pre']['mutual_information_mm'],
                'pre_mi_norm': e['pre']['mutual_information_norm'],
                'pre_n_samples': e['pre']['n_samples'],
                'pre_surrogate_p': e['pre_p'], 'pre_surrogate_z': e['pre_z'],
                'onset_mi_bits': e['onset']['mutual_information'],
                'onset_mi_mm_bits': e['onset']['mutual_information_mm'],
                'onset_mi_norm': e['onset']['mutual_information_norm'],
                'onset_n_samples': e['onset']['n_samples'],
                'onset_surrogate_p': e['onset_p'], 'onset_surrogate_z': e['onset_z'],
                'delta_mi_bits': e['delta_mi'],
            } for e in events]
            events_csv = os.path.join(outdir, f'{stem}_events.csv')
            pd.DataFrame(ev_rows).to_csv(events_csv, index=False)
            print(f'Saved: {events_csv}')

            events_png = os.path.join(outdir, f'{stem}_events.png')
            plot_event_pre_onset_comparison(
                events,
                title=f'{os.path.basename(args.csv)} [{binning}, {bins} bins]',
                outpath=events_png, label_x=label_x, label_y=label_y)

            # Peri-event MI time course (reuses the global windowed series).
            peri_onsets = [(e['name'], e['onset_rel_s']) for e in events]
            peri_title = (f'Peri-event zero-lag MI — {label_x} vs {label_y} '
                          f'[{os.path.basename(args.csv)}]')
            peri_png = os.path.join(outdir, f'{stem}_peri_event.png')
            plot_peri_event_mi(
                win_result, peri_onsets, title=peri_title,
                outpath=peri_png, fs=fs_eff, win_sec=args.win, step_sec=args.step)

            # Optional interactive (Plotly) animation of the same real traces.
            if args.peri_event_html:
                peri_html = os.path.join(outdir, f'{stem}_peri_event.html')
                plot_peri_event_mi_interactive(
                    win_result, peri_onsets, title=peri_title,
                    outpath=peri_html, fs=fs_eff, win_sec=args.win,
                    step_sec=args.step)

            print('\nPre-event vs onset joint MI (bits):')
            for e in events:
                print(f'  {e["name"]:<24s}: pre={e["pre_mi"]:.4f}  '
                      f'onset={e["onset_mi"]:.4f}  Δ={e["delta_mi"]:+.4f}  '
                      f'(onset surrogate p={e["onset_p"]:.3g})')

    # ── Summary to stdout ───────────────────────────────────────────────────────
    print(f'\nJoint MI ({label_x} ↔ {label_y}):')
    print(f'  I(X;Y)        : {joint["mutual_information"]:.4f} bits  '
          f'(Miller–Madow {joint["mutual_information_mm"]:.4f}; '
          f'norm {joint["mutual_information_norm"]:.4f})')
    if sig['n_surrogates']:
        print(f'  surrogate null: mean={sig["surrogate_mean"]:.4f}±'
              f'{sig["surrogate_std"]:.4f} bits, p={sig["p_value"]:.3g}, '
              f'z={sig["z"]:.2f} (n={sig["n_surrogates"]})')
    finite = win_result['joint_mi'][np.isfinite(win_result['joint_mi'])]
    if finite.size:
        print(f'  windowed MI   : mean={finite.mean():.4f}, std={finite.std():.4f}, '
              f'min={finite.min():.4f}, max={finite.max():.4f} '
              f'(n={finite.size} windows)')


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


def _resolve_event_onsets(args: argparse.Namespace, time_us: np.ndarray,
                          n_times: int) -> list[int]:
    """Resolve pre/post-event onset sample indices for --band-event-mi.

    With --ibrain-events every session event onset (converted from its HH:MM to
    a sample index relative to the recording's epoch) is pooled. Otherwise a
    single --event-onset (seconds from recording start) is used.
    """
    if args.ibrain_events:
        if not _IBRAIN_AVAILABLE:
            sys.exit('Error: --ibrain-events needs plot_event_markers (unavailable).')
        epoch_us = int(time_us[0])
        onsets = []
        subject = getattr(args, 'subject', None)
        if not subject:
            raise ValueError('Participant subject is required for pooled iBrainCenter events')
        for name, start_hhmm, _duration, participants in _IBRAIN_EVENTS:
            if participants is not None and subject not in participants:
                continue
            idx = int(round((_hhmm_to_us(start_hhmm) - epoch_us) / 1e6 * args.fs))
            if 0 <= idx < n_times:
                onsets.append(idx)
        if not onsets:
            sys.exit('Error: no iBrainCenter event onset falls within the recording.')
        print(f'  Pooling {len(onsets)} iBrainCenter event onset(s).')
        return onsets
    if args.event_onset is None:
        sys.exit('Error: --band-event-mi needs --event-onset SEC (or --ibrain-events).')
    idx = int(round(args.event_onset * args.fs))
    if not (0 <= idx < n_times):
        sys.exit(f'Error: --event-onset {args.event_onset:g}s is outside the recording.')
    return [idx]


def _run_band_event_mi_mode(args: argparse.Namespace,
                            time_us: np.ndarray,
                            data: np.ndarray,
                            outdir: str) -> None:
    """Estimate I(θ,α,β power ; pre/post-event) across window sizes for the
    requested channels, write the results CSV, and plot MI vs. window size.

    See the "Band-power × Event joint MI" section for the method: sub-epoch
    tiling of each pre/post window (pooled across onsets) feeds both the summed
    per-band and the true multivariate KSG estimator, reported in bits.
    """
    if not _SKLEARN_MI_AVAILABLE:
        sys.exit('Error: --band-event-mi requires scikit-learn (import failed).')

    n_times = data.shape[0]
    onsets = _resolve_event_onsets(args, time_us, n_times)

    signals: dict[str, np.ndarray] = {}
    for ch in args.band_mi_channels:
        idx = ch - 1
        if idx < 0 or idx >= data.shape[1]:
            sys.exit(f'Error: --band-mi-channels {ch} not found '
                     f'(file has {data.shape[1]} channels).')
        signals[f'ch{ch}'] = data[:, idx]

    print(f'Band-power × event joint MI — channels={list(signals)}, '
          f'windows={[f"{w:g}s" for w in args.mi_windows]}, '
          f'sub={args.mi_sub_sec:g}s/step={args.mi_sub_step:g}s, '
          f'surrogates={args.mi_surrogates}, fs={args.fs:g}Hz')

    df = run_band_event_mi_pipeline(
        signals, onsets, fs=args.fs,
        windows_sec=tuple(args.mi_windows),
        sub_sec=args.mi_sub_sec, sub_step_sec=args.mi_sub_step,
        n_neighbors=3,   # KSG/KNN k; auto-capped to the smaller class per window
        n_surrogates=args.mi_surrogates,
    )
    if df.empty:
        sys.exit('Error: no window/channel produced enough sub-epochs for an MI '
                 'estimate — try larger --mi-windows or a smaller --mi-sub-sec.')

    print('\n=== Joint MI results (bits) ===')
    print(df.to_string(index=False))

    basename = os.path.splitext(os.path.basename(args.csv))[0]
    csv_out = os.path.join(outdir, f'{basename}_band_event_mi.csv')
    df.to_csv(csv_out, index=False)
    print(f'\nSaved: {csv_out}')

    png_out = os.path.join(outdir, f'{basename}_band_event_mi.png')
    plot_band_event_mi(df, title=os.path.basename(args.csv), outpath=png_out)


def main() -> None:
    global SMOOTH_WINDOW
    args = _parse_args()
    outdir = args.out or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(outdir, exist_ok=True)

    # Apply global run-context (smoothing length + passband state) so every
    # smoothed trace and every figure footer reflect the actual CLI settings.
    SMOOTH_WINDOW = max(1, int(args.smooth))
    _PROV['no_bandpass'] = bool(args.no_bandpass)

    print(f'Loading: {args.csv}')
    time_us, data_raw = load_merged_csv(args.csv)
    require_continuous(time_us, args.fs, 'spectral_entropy.py')
    if args.ibrain_events:
        args.subject = args.subject or os.path.basename(os.path.dirname(os.path.abspath(args.csv))).split('(')[0].strip()
        from plot_event_markers import SUBJECTS
        if args.subject not in SUBJECTS:
            raise ValueError('--ibrain-events requires --subject for unrecognized recording paths')

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

    # ── Joint-distribution / mutual-information mode ─────────────────────────────
    if args.joint_mi:
        _run_joint_mi_mode(args, time_us, data, data_raw, outdir)
        return

    # ── Band-power × event joint-MI mode ─────────────────────────────────────────
    if args.band_event_mi:
        _run_band_event_mi_mode(args, time_us, data, outdir)
        return

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

    # ── Quality masking (same scoring method as plot_tflite_summary) ────────────
    # Score every analysis window with eeg_quality_v2, take the channel median,
    # and discard (NaN-mask) windows below the threshold. Masked windows become
    # gaps in the plot and blank cells in the CSV, so no band-entropy / sync
    # value is ever reported for a window the quality scorer rejects.
    quality = None
    if not args.no_quality_mask:
        if not _QC_AVAILABLE:
            print('  [warn] quality modules unavailable — skipping quality masking '
                  '(pass --no-quality-mask to silence).')
        else:
            print(f'  Scoring window quality (eeg_quality_v2, ch median) and masking '
                  f'< {args.quality_threshold:g}…', flush=True)
            quality = compute_quality_windowed_aligned(
                data, fs=args.fs, win_sec=args.win, step_sec=args.step)
            if quality.shape[0] != entropy_result['time'].shape[0]:
                sys.exit('Error: quality window count does not match band-entropy windows.')
            mask = quality < args.quality_threshold
            entropy_keys = (['band_entropy', 'band_entropy_norm', 'total_energy']
                            + [f'E_{n}' for n, _ in BAND_DEFINITIONS]
                            + [f'p_{n}' for n, _ in BAND_DEFINITIONS])
            _mask_low_quality(entropy_result, mask, entropy_keys)
            if sync_result is not None:
                sync_keys = [k for k in sync_result if k != 'time']
                _mask_low_quality(sync_result, mask, sync_keys)
            print(f'    masked {int(mask.sum())}/{mask.size} windows '
                  f'({100.0 * mask.mean():.1f}%) below quality {args.quality_threshold:g}.')

    basename = os.path.splitext(os.path.basename(args.csv))[0]
    stem = f'{basename}_band_entropy_ch{args.ch}{sync_suffix}'
    csv_out = os.path.join(outdir, f'{stem}.csv')
    png_out = os.path.join(outdir, f'{stem}.png')

    time_s = (time_us.astype(float) - float(time_us[0])) / 1e6
    t_offset = float(time_s[0]) if time_s.size else 0.0

    csv_data = {'time_s': entropy_result['time'] + t_offset}
    if quality is not None:
        csv_data['quality'] = quality
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
        fs=args.fs,
        win_sec=args.win,
        step_sec=args.step,
    )

    # ── Compositional views: stacked-area simplex + ternary trajectory ──────────
    comp_png = os.path.join(outdir, f'{stem}_composition.png')
    plot_band_composition(
        entropy_result, title=f'{title} — θ/α/β composition',
        outpath=comp_png, t_offset=t_offset,
        time_us_epoch=int(time_us[0]) if args.ibrain_events else None,
        ibrain_events=args.ibrain_events,
        fs=args.fs, win_sec=args.win, step_sec=args.step)
    ternary_png = os.path.join(outdir, f'{stem}_ternary.png')
    plot_band_ternary(
        entropy_result, title=f'{title} — θ/α/β simplex trajectory',
        outpath=ternary_png, fs=args.fs, win_sec=args.win, step_sec=args.step)

    # ── Focus vs Relax scatter (qEEG indices), coloured by time ─────────────────
    focus_relax_png = os.path.join(outdir, f'{stem}_focus_relax.png')
    plot_focus_relax_scatter(
        entropy_result, title=f'{title} — Focus vs Relax trajectory',
        outpath=focus_relax_png, t_offset=t_offset,
        fs=args.fs, win_sec=args.win, step_sec=args.step)

    # Summary statistics ignore NaN-masked (low-quality) windows.
    def _stats(arr: np.ndarray) -> str:
        a = np.asarray(arr, dtype=float)
        if not np.any(np.isfinite(a)):
            return 'no valid (all windows masked)'
        return (f'mean={np.nanmean(a):.4f}, std={np.nanstd(a):.4f}, '
                f'min={np.nanmin(a):.4f}, max={np.nanmax(a):.4f}')

    print('\nSummary:')
    if quality is not None:
        n_valid = int(np.sum(np.isfinite(entropy_result['band_entropy'])))
        print(f'  valid windows     : {n_valid}/{quality.size} '
              f'(quality ≥ {args.quality_threshold:g})')
    print(f'  band_entropy      : {_stats(entropy_result["band_entropy"])}')
    print(f'  band_entropy_norm : {_stats(entropy_result["band_entropy_norm"])}')
    if sync_result is not None:
        print(f'  lagged_mi_mean    : {_stats(sync_result["lagged_mi_mean"])}')
        print(f'  lagged_mi_max     : {_stats(sync_result["lagged_mi_max"])}')


if __name__ == '__main__':
    main()
