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
import csv
import warnings
from decimal import Decimal
from lilia.signal import bandpass
from typing import Tuple


def read_lilia_frame(path: str, *, drop_invalid_time=False) -> pd.DataFrame:
    """Read declared columns without pandas inferring an implicit row index.

    Device exports can carry undeclared trailing values; preserve the declared
    signal columns, as the merge tool historically does. Time units/origin are
    unchanged. Invalid timestamps are rejected unless merge explicitly opts in.
    """
    with open(path, encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle)
        try:
            for _ in range(4):
                next(reader)
            names = [v.strip() for v in next(reader)]
            first = next(reader, [])
        except StopIteration as exc:
            raise ValueError(f'Not a four-header-row Lilia recording: {path}') from exc
    if len(names) < 2 or names[0].lower() != 'time[us]':
        raise ValueError(f'Expected Time[us] and EEG columns: {path}')
    # Repeated device labels such as "value" need deterministic unique names.
    unique, seen = [], {}
    for name in names:
        count = seen.get(name, 0)
        unique.append(name if count == 0 else f'{name}.{count}')
        seen[name] = count + 1
    if len(first) > len(names):
        warnings.warn(f'{path}: ignoring undeclared trailing columns', UserWarning, stacklevel=2)
    df = pd.read_csv(path, skiprows=5, header=None, names=unique,
                     usecols=range(len(unique)), encoding='utf-8-sig')
    numeric = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    valid = np.isfinite(numeric) & (numeric == np.floor(numeric))
    if not valid.all():
        if not drop_invalid_time:
            raise ValueError(f'Invalid microsecond timestamps: {path}')
        warnings.warn(f'{path}: dropped {int((~valid).sum())} invalid timestamps', UserWarning)
        df = df.loc[valid].copy()
    df[unique[0]] = numeric.loc[df.index].astype(np.int64)
    if df.empty:
        raise ValueError(f'Recording contains no valid samples: {path}')
    return df


def load_merged_csv(path: str, downsample: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Return microsecond timestamps (origin unchanged) and declared EEG columns.

    merged.csv uses UTC; raw device files may use relative microseconds. This
    loader never guesses the origin or automatically adds an offset.
    downsample is display decimation, not an anti-aliasing resampler.
    """
    if not isinstance(downsample, (int, np.integer)) or downsample < 1:
        raise ValueError('downsample must be a positive integer')
    df = read_lilia_frame(path).iloc[::downsample]
    time_us = df.iloc[:, 0].to_numpy(dtype=np.int64)
    data = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    if not np.all(np.isfinite(data)):
        raise ValueError(f'Non-finite EEG samples: {path}')
    return time_us, data


def bandpass_filter(data: np.ndarray,
                    fs: float = 500.0,
                    lo: float = 0.5,
                    hi: float = 45.0,
                    order: int = 4, time_us=None) -> np.ndarray:
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
    return bandpass(data, fs=fs, low=lo, high=hi, order=order, time_us=time_us).astype(np.float32)


def read_abs_time_offset(path: str) -> int:
    """Read the Abs Time Offset[us] value from row 1 of a merged CSV.

    Parses the header row 1 (format: Amp Gain, 500, Abs Time Offset[us], <value>, ...)
    Tolerates float / quoted / whitespace-padded values.

    Parameters
    ----------
    path : path to the merged.csv file

    Returns
    -------
    int – the Abs Time Offset[us] value (UTC Unix microseconds of first sample)

    Raises
    ------
    ValueError if the offset cannot be read
    """
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 1:
                parts = line.strip().split(',')
                # Tolerate float / quoted / whitespace-padded values.
                return int(Decimal(parts[3].strip().strip('"').strip("'")))
    raise ValueError(f'Cannot read offset from {path}')
