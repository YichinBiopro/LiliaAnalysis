"""Deprecated shim — moved to :mod:`lilia.qeeg`. Prefer ``from lilia.qeeg import ...``."""
from lilia.qeeg import (
    compute_relative_powers, bounded_ratio,
    focus_index, flow_index, calm_index, relaxation_index,
    compute_qeeg_indices, compute_qeeg_indices_windowed, plot_qeeg_indices,
)

__all__ = [
    "compute_relative_powers", "bounded_ratio",
    "focus_index", "flow_index", "calm_index", "relaxation_index",
    "compute_qeeg_indices", "compute_qeeg_indices_windowed", "plot_qeeg_indices",
]

if __name__ == "__main__":   # preserve `python qeeg_indices.py ...`
    from lilia.qeeg import main
    main()
