"""Shared utilities for Goertzel distribution summary and histogram scripts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lilia.subject_paths import goertzel_csv_path, iter_subject_dirs


@dataclass
class GroupData:
    subject: str
    channel: str
    power: np.ndarray
    power_db: np.ndarray
    n_total: int
    n_kept: int


def collect_group_data(
    root: str,
    channels: list[int],
    stem: str,
    target_freq: float,
    threshold: float,
    exclude_hard_artifact: bool,
    require_power: bool,
) -> tuple[list[GroupData], list[str]]:
    """Collect quality-filtered per-subject/channel Goertzel rows.

    Returns ``(groups, missing_files)``.
    """
    groups: list[GroupData] = []
    missing_files: list[str] = []

    for subject in iter_subject_dirs(root):
        for ch in channels:
            csv_path = goertzel_csv_path(root, subject, stem, ch, target_freq)
            try:
                df = pd.read_csv(csv_path)
            except FileNotFoundError:
                missing_files.append(csv_path)
                continue

            required = {"goertzel_db", "quality"}
            if require_power:
                required.add("goertzel_power")
            if not required.issubset(df.columns):
                missing = sorted(required - set(df.columns))
                raise ValueError(f"Input CSV missing required columns {missing}: {csv_path}")

            quality_col = "quality_final" if "quality_final" in df.columns else "quality"
            q = df[quality_col].to_numpy(dtype=float)
            keep = np.isfinite(q) & (q > threshold)
            if exclude_hard_artifact and "artifact_hard_clip" in df.columns:
                keep &= (df["artifact_hard_clip"].to_numpy(dtype=float) < 0.5)

            if require_power:
                pwr = df["goertzel_power"].to_numpy(dtype=float)[keep]
            else:
                pwr = np.array([], dtype=float)
            db = df["goertzel_db"].to_numpy(dtype=float)[keep]

            groups.append(
                GroupData(
                    subject=subject,
                    channel=f"ch{ch}",
                    power=pwr,
                    power_db=db,
                    n_total=int(len(df)),
                    n_kept=int(np.sum(keep)),
                )
            )
    return groups, missing_files


def combine_groups(groups: list[GroupData], subject: str, channel: str) -> GroupData:
    """Combine multiple ``GroupData`` rows into one aggregate row."""
    if not groups:
        return GroupData(
            subject=subject,
            channel=channel,
            power=np.array([], dtype=float),
            power_db=np.array([], dtype=float),
            n_total=0,
            n_kept=0,
        )
    return GroupData(
        subject=subject,
        channel=channel,
        power=(
            np.concatenate([g.power for g in groups])
            if any(g.power.size for g in groups)
            else np.array([], dtype=float)
        ),
        power_db=(
            np.concatenate([g.power_db for g in groups])
            if any(g.power_db.size for g in groups)
            else np.array([], dtype=float)
        ),
        n_total=int(sum(g.n_total for g in groups)),
        n_kept=int(sum(g.n_kept for g in groups)),
    )


def with_aggregates(base_groups: list[GroupData], channels: list[int]) -> list[GroupData]:
    """Return ``base_groups`` plus per-channel and all-channel ALL_SUBJECTS rows."""
    out = list(base_groups)
    per_channel: list[GroupData] = []
    for ch in channels:
        subset = [g for g in base_groups if g.channel == f"ch{ch}"]
        agg = combine_groups(subset, subject="ALL_SUBJECTS", channel=f"ch{ch}")
        per_channel.append(agg)
        out.append(agg)
    out.append(combine_groups(per_channel, subject="ALL_SUBJECTS", channel="ALL_CHANNELS"))
    return out


def common_db_edges(groups: list[GroupData], bins: int) -> np.ndarray:
    """Build one set of histogram edges for all groups."""
    values = [g.power_db for g in groups if g.power_db.size > 0]
    if not values:
        return np.linspace(-120.0, 0.0, bins + 1)

    db_all = np.concatenate(values)
    lo = float(np.min(db_all))
    hi = float(np.max(db_all))
    if np.isclose(lo, hi):
        lo -= 1.0
        hi += 1.0
    return np.linspace(lo, hi, bins + 1)
