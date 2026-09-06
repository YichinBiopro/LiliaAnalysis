"""Shared signal processing utilities across lilia_analysis scripts."""

from __future__ import annotations

import numpy as np
from fractions import Fraction
from lilia.windowing import continuous_slices
from scipy import signal as sp_signal
from typing import Tuple


def bandpass(data: np.ndarray,
             fs: float = 500.0,
             low: float = 0.5,
             high: float = 45.0,
             order: int = 4, time_us=None) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter column-wise.

    Parameters
    ----------
    data  : (N,) or (N, n_ch) float array
    fs    : sample rate in Hz
    low   : lower cut-off frequency in Hz
    high  : upper cut-off frequency in Hz
    order : filter order (default 4)

    Returns
    -------
    (N,) or (N, n_ch) float array
    """
    sos = sp_signal.butter(order, [low, high], btype='bandpass', fs=fs, output='sos')
    return _filter_segments(data, lambda x: sp_signal.sosfiltfilt(sos, x, axis=0), time_us, fs)


def notch(data: np.ndarray,
          fs: float = 500.0,
          freq: float = 60.0,
          q: float = 30.0, time_us=None) -> np.ndarray:
    """Apply zero-phase IIR notch filter column-wise.

    Parameters
    ----------
    data : (N,) or (N, n_ch) float array
    fs   : sample rate in Hz
    freq : notch frequency in Hz
    q    : quality factor

    Returns
    -------
    (N,) or (N, n_ch) float array
    """
    b, a = sp_signal.iirnotch(freq, q, fs=fs)
    return _filter_segments(data, lambda x: sp_signal.filtfilt(b, a, x, axis=0), time_us, fs)


def bandstop(data: np.ndarray,
             fs: float = 500.0,
             center: float = 33.25,
             bw: float = 1.0,
             order: int = 4, time_us=None) -> np.ndarray:
    """Apply zero-phase Butterworth bandstop filter column-wise.

    Parameters
    ----------
    data   : (N,) or (N, n_ch) float array
    fs     : sample rate in Hz
    center : center frequency in Hz
    bw     : bandwidth in Hz
    order  : filter order (default 4)

    Returns
    -------
    (N,) or (N, n_ch) float array
    """
    sos = sp_signal.butter(order, [center - bw/2, center + bw/2],
                          btype='bandstop', fs=fs, output='sos')
    return _filter_segments(data, lambda x: sp_signal.sosfiltfilt(sos, x, axis=0), time_us, fs)


def apply_filters(data: np.ndarray,
                  fs: float = 500.0,
                  bandpass_low: float = 0.5,
                  bandpass_high: float = 45.0,
                  notch_freq: float = 60.0,
                  notch_q: float = 30.0,
                  bandstop_center: float = 33.25,
                  bandstop_bw: float = 1.0, time_us=None) -> np.ndarray:
    """Apply bandpass → notch → bandstop filters sequentially.

    Parameters
    ----------
    data              : (N,) or (N, n_ch) float array
    fs                : sample rate in Hz
    bandpass_low      : bandpass lower cutoff (Hz)
    bandpass_high     : bandpass upper cutoff (Hz)
    notch_freq        : notch center frequency (Hz)
    notch_q           : notch quality factor
    bandstop_center   : bandstop center frequency (Hz)
    bandstop_bw       : bandstop bandwidth (Hz)

    Returns
    -------
    (N,) or (N, n_ch) filtered float array
    """
    data = bandpass(data, fs=fs, low=bandpass_low, high=bandpass_high, time_us=time_us)
    data = notch(data, fs=fs, freq=notch_freq, q=notch_q, time_us=time_us)
    data = bandstop(data, fs=fs, center=bandstop_center, bw=bandstop_bw, time_us=time_us)
    return data


def _filter_segments(data, transform, time_us, fs):
    data = np.asarray(data)
    if not np.all(np.isfinite(data)):
        raise ValueError('signal contains non-finite values')
    if time_us is None:
        return transform(data)
    if len(time_us) != len(data):
        raise ValueError('timestamp and signal lengths differ')
    out = np.empty_like(data, dtype=np.float64)
    for sl in continuous_slices(time_us, fs):
        try:
            out[sl] = transform(data[sl])
        except ValueError as exc:
            raise ValueError(f'cannot filter continuous segment [{sl.start}:{sl.stop}]: {exc}') from exc
    return out


def _resampling_ratio(fs_in, fs_out):
    if not all(np.isfinite(f) and f > 0 for f in (fs_in, fs_out)):
        raise ValueError('sample rates must be finite and positive')
    ratio = (Fraction(str(fs_out)) / Fraction(str(fs_in))).limit_denominator(1000000)
    if not np.isclose(float(ratio), fs_out / fs_in, rtol=1e-12, atol=0):
        raise ValueError('sample-rate ratio cannot be represented accurately')
    return ratio.numerator, ratio.denominator


def resample_polyphase(data: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Resample a continuous array; fractional sample rates are supported."""
    up, dn = _resampling_ratio(fs_in, fs_out)
    data = np.asarray(data)
    if not len(data) or up == dn:
        return data.astype(np.float32)
    return sp_signal.resample_poly(data, up, dn, axis=0).astype(np.float32)


def resample_with_time(time_us: np.ndarray, data: np.ndarray,
                       fs_in: float, fs_out: float) -> Tuple[np.ndarray, np.ndarray]:
    """Resample each continuous segment, preserving integer microsecond time.

    Relative microseconds and UTC microseconds are both accepted. Floating-point
    seconds must use resample_seconds instead. Each segment's final fractional
    sample position is extrapolated at the nominal rate, never clamped/repeated.
    """
    t = np.asarray(time_us)
    x = np.asarray(data)
    if t.dtype.kind not in 'iu':
        raise ValueError('time_us must be integer microseconds; use resample_seconds for seconds')
    if len(t) != len(x):
        raise ValueError('timestamp and signal lengths differ')
    up, dn = _resampling_ratio(fs_in, fs_out)
    slices = continuous_slices(t, fs_in)
    if not slices:
        return t.astype(np.int64), x.astype(np.float32)
    times, signals = [], []
    for sl in slices:
        part = resample_polyphase(x[sl], fs_in, fs_out)
        original = t[sl]
        positions = np.arange(len(part), dtype=float) * dn / up
        relative = (original - original[0]).astype(float)
        rel_new = np.interp(positions, np.arange(len(original)), relative)
        beyond = positions > len(original) - 1
        rel_new[beyond] = relative[-1] + (positions[beyond] - len(original) + 1) * 1e6 / fs_in
        times.append(int(original[0]) + np.rint(rel_new).astype(np.int64))
        signals.append(part)
    return np.concatenate(times), np.concatenate(signals, axis=0)


def resample_seconds(time_s, data, fs_in, fs_out):
    """Explicit seconds adapter, retaining sub-second timestamps."""
    t = np.asarray(time_s, dtype=float)
    if not np.all(np.isfinite(t)):
        raise ValueError('time_s must be finite')
    time_us, result = resample_with_time(np.rint(t * 1e6).astype(np.int64), data, fs_in, fs_out)
    return time_us.astype(float) / 1e6, result
