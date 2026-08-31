"""Shared helpers for selecting Goertzel windows across subjects/channels."""

from __future__ import annotations

import numpy as np
import pandas as pd

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
            except Exception:
                continue
            need = {"time_s", "goertzel_db", "quality"}
            if not need.issubset(df.columns):
                continue

            db = df["goertzel_db"].to_numpy(dtype=float)
            q = df["quality"].to_numpy(dtype=float)

            keep = np.isfinite(db)
            keep &= np.abs(db - target_db) <= tol_db
            if quality_threshold is not None:
                keep &= np.isfinite(q) & (q > quality_threshold)

            if not np.any(keep):
                continue

            rows.append(pd.DataFrame({
                "subject": subject,
                "channel": ch,
                "time_s": df.loc[keep, "time_s"].to_numpy(dtype=float),
                "goertzel_db": db[keep],
                "quality": q[keep],
            }))

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
        except Exception:
            continue
        need = {"time_s", "goertzel_db", "quality"}
        if not need.issubset(df.columns):
            continue

        quality_col = "quality_final" if "quality_final" in df.columns else "quality"
        q = df[quality_col].to_numpy(dtype=float)
        keep = np.isfinite(q) & (q > quality_threshold)
        if "artifact_hard_clip" in df.columns:
            keep &= (df["artifact_hard_clip"].to_numpy(dtype=float) < 0.5)
        if not np.any(keep):
            continue

        sub = df.loc[keep].copy()
        sub["subject"] = subject
        sub["channel"] = ch
        if "window_start_idx" in sub.columns and "window_end_idx" in sub.columns:
            sub["window_start_idx"] = sub["window_start_idx"].astype(int)
            sub["window_end_idx"] = sub["window_end_idx"].astype(int)
            sub["window_start_s"] = sub["window_start_idx"].astype(float) / fs
            sub["window_end_s"] = sub["window_end_idx"].astype(float) / fs
            sub["window_center_s"] = 0.5 * (sub["window_start_s"] + sub["window_end_s"])
            sub["window_duration_s"] = sub["window_end_s"] - sub["window_start_s"]
        else:
            sub["window_start_s"] = sub["time_s"].astype(float) - 2.5
            sub["window_end_s"] = sub["time_s"].astype(float) + 2.5
            sub["window_center_s"] = sub["time_s"].astype(float)
            sub["window_duration_s"] = 5.0
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
