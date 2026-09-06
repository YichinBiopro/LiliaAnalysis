"""Goertzel single-frequency power utilities."""

from __future__ import annotations

import math
from typing import Sequence


def goertzel_power(
    data: Sequence[float],
    target_freq: float,
    sample_rate: float,
    remove_dc: bool = True,
    apply_hann: bool = True,
) -> float:
    """Estimate signal power at ``target_freq`` with the Goertzel recurrence.

    This keeps parity with the original C implementation in power_spectral_app.c:
    optional DC removal + Hann windowing, followed by the canonical power term.
    """
    n_samples = len(data)
    if n_samples == 0:
        return 0.0

    if remove_dc:
        mean = sum(float(v) for v in data) / n_samples
    else:
        mean = 0.0

    w = 2.0 * math.pi * (target_freq / sample_rate)
    coeff = 2.0 * math.cos(w)

    s_prev = 0.0
    s_prev2 = 0.0

    for n, x in enumerate(data):
        sample = float(x) - mean
        if apply_hann and n_samples > 1:
            window = 0.5 * (1.0 - math.cos(2.0 * math.pi * n / (n_samples - 1)))
            sample *= window

        s = sample + (coeff * s_prev) - s_prev2
        s_prev2 = s_prev
        s_prev = s

    return s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2
