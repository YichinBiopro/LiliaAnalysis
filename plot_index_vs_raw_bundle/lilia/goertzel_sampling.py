"""Shared helpers for selecting Goertzel windows across subjects/channels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lilia.quality_policy import valid_goertzel_rows
from lilia.subject_paths import goertzel_csv_path, iter_subject_dirs


def collect_matches_global(
    root: str,
    channels: list[int],
    stem: str,
    target_freq: float,
    target_db: float,
    tol_db: float,
    quality_threshold: float | None,
) -> pd.DataFrame:
    """Collect all subject/channel windows near ``target_db`` within ``tol_db``."""
    rows: list[pd.DataFrame] = []
    for subject in iter_subject_dirs(root):
        for ch in channels:
            path = goertzel_csv_path(root, subject, stem, ch, target_freq)
            try:
                df = pd.read_csv(path)
            except FileNotFoundError:
                continue
            need = {"time_s", "goertzel_db", "quality"}
            if not need.issubset(df.columns):
                continue

            db = df["goertzel_db"].to_numpy(dtype=float)
            quality_col = 'quality_final' if 'quality_final' in df else 'quality'
            q = df[quality_col].to_numpy(dtype=float)
            keep = valid_goertzel_rows(df, quality_threshold)
            keep &= np.abs(db - target_db) <= tol_db

            if not np.any(keep):
                continue

            selected = df.loc[keep].copy()
            selected['subject'] = subject
            selected['channel'] = ch
            selected['quality'] = q[keep]
            rows.append(selected)


    if not rows:
        return pd.DataFrame(columns=["subject", "channel", "time_s", "goertzel_db", "quality"])
    return pd.concat(rows, ignore_index=True)


def collect_candidates_per_subject(
    root: str,
    subject: str,
    channels: list[int],
    stem: str,
    target_freq: float,
    target_db: float,
    quality_threshold: float,
    fs: float,
) -> pd.DataFrame:
    """Collect quality-filtered windows for one subject and rank by dB distance."""
    frames: list[pd.DataFrame] = []
    for ch in channels:
        path = goertzel_csv_path(root, subject, stem, ch, target_freq)
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            continue
        need = {"time_s", "goertzel_db", "quality"}
        if not need.issubset(df.columns):
            continue

        keep = valid_goertzel_rows(df, quality_threshold)
        if not np.any(keep):
            continue

        sub = df.loc[keep].copy()
        sub["subject"] = subject
        sub["channel"] = ch
        # time_s is real elapsed time, whereas indexes count retained samples.
        # Derive durations from indexes without replacing the physical centre.
        duration = np.full(len(sub), 5.0)
        if "window_start_idx" in sub and "window_end_idx" in sub:
            starts = sub["window_start_idx"].to_numpy(dtype=float)
            ends = sub["window_end_idx"].to_numpy(dtype=float)
            valid = np.isfinite(starts) & np.isfinite(ends) & (ends > starts)
            duration[valid] = (ends[valid] - starts[valid]) / fs
        sub["window_center_s"] = sub["time_s"].astype(float)
        sub["window_duration_s"] = duration
        sub["window_start_s"] = sub["window_center_s"] - duration / 2
        sub["window_end_s"] = sub["window_start_s"] + duration
        sub["distance_db"] = np.abs(sub["goertzel_db"].to_numpy(dtype=float) - target_db)
        frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=["subject", "channel", "time_s", "goertzel_db", "quality", "distance_db"])
    return pd.concat(frames, ignore_index=True)


def sample_rows(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample up to ``n`` rows from a dataframe without replacement."""
    if df.empty or n <= 0:
        return df.head(0).copy()
    rng = np.random.default_rng(seed)
    take = min(n, len(df))
    idx = rng.choice(len(df), size=take, replace=False)
    return df.iloc[np.sort(idx)].reset_index(drop=True)
