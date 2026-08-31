"""Shared signal processing utilities across lilia_analysis scripts."""

from __future__ import annotations

import numpy as np
from math import gcd
from scipy import signal as sp_signal
from typing import Tuple


def bandpass(data: np.ndarray,
             fs: float = 500.0,
             low: float = 0.5,
             high: float = 45.0,
             order: int = 4) -> np.ndarray:
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
    return sp_signal.sosfiltfilt(sos, data, axis=0)


def notch(data: np.ndarray,
          fs: float = 500.0,
          freq: float = 60.0,
          q: float = 30.0) -> np.ndarray:
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
    return sp_signal.filtfilt(b, a, data, axis=0)


def bandstop(data: np.ndarray,
             fs: float = 500.0,
             center: float = 33.25,
             bw: float = 1.0,
             order: int = 4) -> np.ndarray:
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
    return sp_signal.sosfiltfilt(sos, data, axis=0)


def apply_filters(data: np.ndarray,
                  fs: float = 500.0,
                  bandpass_low: float = 0.5,
                  bandpass_high: float = 45.0,
                  notch_freq: float = 60.0,
                  notch_q: float = 30.0,
                  bandstop_center: float = 33.25,
                  bandstop_bw: float = 1.0) -> np.ndarray:
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
    data = bandpass(data, fs=fs, low=bandpass_low, high=bandpass_high)
    data = notch(data, fs=fs, freq=notch_freq, q=notch_q)
    data = bandstop(data, fs=fs, center=bandstop_center, bw=bandstop_bw)
    return data


def resample_polyphase(data: np.ndarray,
                       fs_in: float,
                       fs_out: float) -> np.ndarray:
    """Resample data using polyphase filtering (scipy.signal.resample_poly).

    Parameters
    ----------
    data   : (N,) or (N, n_ch) float array
    fs_in  : input sample rate in Hz
    fs_out : output sample rate in Hz

    Returns
    -------
    (M,) or (M, n_ch) resampled float32 array (M ≈ N * fs_out / fs_in)
    """
    if fs_in == fs_out:
        return data.astype(np.float32)

    up = int(fs_out)
    dn = int(fs_in)
    g = gcd(up, dn)
    up //= g
    dn //= g

    return sp_signal.resample_poly(data, up, dn, axis=0).astype(np.float32)


def resample_with_time(time_us: np.ndarray,
                       data: np.ndarray,
                       fs_in: float,
                       fs_out: float) -> Tuple[np.ndarray, np.ndarray]:
    """Resample data and interpolate time axis together.

    Parameters
    ----------
    time_us : (N,) int64 array – absolute time in microseconds
    data    : (N,) or (N, n_ch) float array
    fs_in   : input sample rate in Hz
    fs_out  : output sample rate in Hz

    Returns
    -------
    time_us_resampled : (M,) int64 array
    data_resampled    : (M,) or (M, n_ch) float32 array
    """
    if fs_in == fs_out:
        return time_us.astype(np.int64), data.astype(np.float32)

    up = int(fs_out)
    dn = int(fs_in)
    g = gcd(up, dn)
    up //= g
    dn //= g

    data_resampled = sp_signal.resample_poly(data, up, dn, axis=0).astype(np.float32)

    # Interpolate time axis
    t_orig = np.arange(len(time_us), dtype=np.float64)
    t_new = np.arange(len(data_resampled), dtype=np.float64) * (dn / up)
    time_us_resampled = np.interp(t_new, t_orig, time_us.astype(np.float64)).astype(np.int64)

    return time_us_resampled, data_resampled
