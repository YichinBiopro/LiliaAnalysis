"""
eeg_utils.py
============
Shared low-level utilities reused across the lilia_analysis scripts.

  load_merged_csv    – load a 4-row-header lilia merged.csv
  bandpass_filter    – zero-phase Butterworth bandpass, column-wise
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from typing import Tuple


def load_merged_csv(path: str, downsample: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Load a lilia merged.csv (4-row header) and return (time_us, data).

    Parameters
    ----------
    path       : path to the merged.csv file
    downsample : keep every *downsample*-th row (1 = no downsampling)

    Returns
    -------
    time_us : (N,) int64 array – absolute UTC Unix microseconds
    data    : (N, n_ch) float32 array – EEG channel values
    """
    df = pd.read_csv(path, skiprows=4)
    if downsample > 1:
        df = df.iloc[::downsample].reset_index(drop=True)
    time_us = df.iloc[:, 0].values.astype(np.int64)
    data    = df.iloc[:, 1:].values.astype(np.float32)
    return time_us, data


def bandpass_filter(data: np.ndarray,
                    fs: float = 500.0,
                    lo: float = 0.5,
                    hi: float = 45.0,
                    order: int = 4) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter column-wise.

    Parameters
    ----------
    data  : (N,) or (N, n_ch) float array
    fs    : sample rate in Hz
    lo    : lower cut-off frequency in Hz
    hi    : upper cut-off frequency in Hz
    order : filter order (default 4)

    Returns
    -------
    (N,) or (N, n_ch) float32 array
    """
    sos = butter(order, [lo, hi], btype='bandpass', fs=fs, output='sos')
    return sosfiltfilt(sos, data, axis=0).astype(np.float32)
