"""Deprecated shim — moved to :mod:`lilia.io`. Prefer ``from lilia.io import ...``."""
from lilia.io import load_merged_csv, bandpass_filter

__all__ = ["load_merged_csv", "bandpass_filter"]
