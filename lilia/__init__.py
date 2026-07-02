"""
lilia — core library for the lilia EEG/biosignal analysis scripts.

Submodules
----------
    lilia.io       load_merged_csv, bandpass_filter        (was eeg_utils)
    lilia.qeeg     qEEG band powers + wellness indices      (was qeeg_indices)
    lilia.quality  EEG quality v2 scorer                    (was eeg_quality_v2)

The headline helpers are re-exported here for convenience, e.g.::

    from lilia import load_merged_csv, bandpass_filter, compute_qeeg_indices
"""
from lilia.io import load_merged_csv, bandpass_filter
from lilia.qeeg import (
    compute_qeeg_indices, compute_qeeg_indices_windowed,
    focus_index, flow_index, calm_index, relaxation_index,
)
from lilia.quality import get_eeg_quality_index_v2_parametric

__all__ = [
    "load_merged_csv", "bandpass_filter",
    "compute_qeeg_indices", "compute_qeeg_indices_windowed",
    "focus_index", "flow_index", "calm_index", "relaxation_index",
    "get_eeg_quality_index_v2_parametric",
]
