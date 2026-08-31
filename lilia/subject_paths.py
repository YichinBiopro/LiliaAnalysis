"""Shared helpers for iterating subject folders and locating analysis CSV files."""

from __future__ import annotations

import os
from typing import Iterable, Iterator


def iter_subject_dirs(root: str) -> Iterator[str]:
    """Yield subject directory names that contain ``merged.csv`` under ``root``."""
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "merged.csv")):
            yield name


def goertzel_csv_path(root: str, subject: str, stem: str, ch: int, target_freq: float) -> str:
    """Return expected per-channel Goertzel CSV path for a subject."""
    return os.path.join(root, subject, f"{stem}_ch{ch}_{target_freq:g}Hz.csv")


def iter_group_merged_csvs(
    base_dir: str,
    groups: Iterable[str] = ("iBrainCenter", "YoGa"),
) -> Iterator[tuple[str, str, str]]:
    """Yield ``(group, subject, merged_csv_path)`` for all discovered subjects."""
    for group in groups:
        root = os.path.join(base_dir, group)
        if not os.path.isdir(root):
            continue
        for subject in iter_subject_dirs(root):
            merged_csv = os.path.join(root, subject, "merged.csv")
            yield group, subject, merged_csv
