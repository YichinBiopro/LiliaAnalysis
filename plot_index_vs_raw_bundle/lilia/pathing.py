"""Helpers for portable project path and optional external dependency lookup."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_project_root() -> str:
    """Return absolute path to repository root."""
    return str(PROJECT_ROOT)


def configure_extra_import_paths() -> None:
    """Add optional external source paths to ``sys.path`` when present.

    Resolution order:
    1) ``LILIA_EXTRA_PYTHONPATH`` (os.pathsep-separated list)
    2) ``TOMMY_REPO``
    3) ``../tommy`` relative to this repository
    """
    candidates: list[str] = []

    env_paths = os.environ.get("LILIA_EXTRA_PYTHONPATH", "")
    if env_paths:
        candidates.extend(p for p in env_paths.split(os.pathsep) if p)

    tommy_repo = os.environ.get("TOMMY_REPO", "")
    if tommy_repo:
        candidates.append(tommy_repo)

    candidates.append(str(PROJECT_ROOT.parent / "tommy"))

    for path in candidates:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def import_tinyunetv4():
    """Import and return ``TinyUNetV4`` from external ``eeg_denoise`` package."""
    configure_extra_import_paths()
    try:
        module = importlib.import_module("eeg_denoise.tiny_model_v4")
    except Exception as exc:  # pragma: no cover - exercised in CLI runtime
        raise ModuleNotFoundError(
            "Cannot import eeg_denoise.tiny_model_v4.TinyUNetV4. "
            "Set TOMMY_REPO or LILIA_EXTRA_PYTHONPATH to a directory containing "
            "the eeg_denoise package."
        ) from exc
    return module.TinyUNetV4
