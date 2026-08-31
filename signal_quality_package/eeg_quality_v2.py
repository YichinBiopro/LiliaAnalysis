"""Deprecated shim — moved to :mod:`lilia.quality`. Prefer ``from lilia.quality import ...``."""
from .quality import (
    get_default_eeg_quality_v2_params,
    get_best_eeg_quality_v2_flat_spectrum_only_params,
    get_ibrain_device_eeg_quality_v2_params,
    get_eeg_quality_index_v2_parametric,
)

__all__ = [
    "get_default_eeg_quality_v2_params",
    "get_best_eeg_quality_v2_flat_spectrum_only_params",
    "get_ibrain_device_eeg_quality_v2_params",
    "get_eeg_quality_index_v2_parametric",
]
