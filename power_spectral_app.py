#!/usr/bin/env python3
"""Goertzel-based single-frequency power estimation.

Converted from power_spectral_app.c.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from lilia.goertzel import goertzel_power as _goertzel_power


def goertzel_power(data: Sequence[float], target_freq: float, sample_rate: float) -> float:
    """Backward-compatible wrapper around :mod:`lilia.goertzel`."""
    return float(
        _goertzel_power(
            data,
            target_freq=target_freq,
            sample_rate=sample_rate,
            remove_dc=True,
            apply_hann=True,
        )
    )


def _parse_samples(csv_text: str) -> list[float]:
    return [float(v.strip()) for v in csv_text.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Goertzel power at a target frequency.")
    parser.add_argument("--samples", required=True, help="Comma-separated samples, e.g. '0.1,0.2,0.15'")
    parser.add_argument("--target-freq", required=True, type=float, help="Target frequency in Hz")
    parser.add_argument("--sample-rate", required=True, type=float, help="Sampling rate in Hz")
    args = parser.parse_args()

    data = _parse_samples(args.samples)
    power = goertzel_power(data, target_freq=args.target_freq, sample_rate=args.sample_rate)
    print(f"{power:.12f}")


if __name__ == "__main__":
    main()
