#!/usr/bin/env python3
"""
Standalone EEG quality v2 scorer extracted from SleepStage.

This module centers on `get_eeg_quality_index_v2_parametric()` and the
parameter presets / helpers it depends on, so it can be reused directly from
this repository without importing the original SleepStage codebase.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import kurtosis


DEFAULT_EEG_QUALITY_V2_PARAMS = {
    "target_score": 0.8,
    "flat_ratio_start": 0.3,
    "flat_penalty_floor": 0.2,
    "spike_ratio_start": 0.0001,
    "spike_ratio_end": 0.0015,
    "spike_penalty_floor": 0.2,
    "clip_ratio_start": 0.001,
    "clip_ratio_end": 0.01,
    "clip_penalty_floor": 0.2,
    "range_start": 6.0,
    "range_end": 12.0,
    "range_penalty_floor": 0.3,
    "slope_good_low": -3.5,
    "slope_center": -2.0,
    "slope_good_high": -0.5,
    "slope_edge_score": 0.4,
    "slope_outer_floor": 0.2,
    "alpha_dpr_good": 0.55,
    "alpha_dpr_bad": 0.80,
    "alpha_artifact_floor": 0.5,
    "nonalpha_dpr_good": 0.25,
    "nonalpha_dpr_bad": 0.50,
    "nonalpha_artifact_floor": 0.1,
    "low_freq_penalty_floor": 0.3,
    "low_freq_penalty_ceiling": 0.7,
    "line_noise_penalty_floor": 0.4,
    "line_noise_penalty_ceiling": 0.7,
    "kurtosis_good_low": 2.0,
    "kurtosis_good_high": 8.0,
    "kurtosis_bad_high": 15.0,
    "kurtosis_floor": 0.2,
    "corr_low": 0.2,
    "corr_mid": 0.4,
    "corr_high": 0.9,
    "corr_floor": 0.3,
    "corr_low_score": 0.75,
    "corr_mid_score": 0.90,
    "flat_weight": 1.0,
    "spectrum_weight": 1.0,
    "kurtosis_weight": 1.0,
    "corr_weight": 0.0,
}

BEST_EEG_QUALITY_V2_MEAN_ABS_CORR_PARAMS = {
    "target_score": 0.78,
    "flat_ratio_start": 0.3,
    "flat_penalty_floor": 0.2,
    "spike_ratio_start": 0.0001,
    "spike_ratio_end": 0.0015,
    "spike_penalty_floor": 0.2,
    "clip_ratio_start": 0.001,
    "clip_ratio_end": 0.01,
    "clip_penalty_floor": 0.2,
    "range_start": 6.0,
    "range_end": 12.0,
    "range_penalty_floor": 0.3,
    "slope_good_low": -3.5,
    "slope_center": -2.0,
    "slope_good_high": -0.5,
    "slope_edge_score": 0.4,
    "slope_outer_floor": 0.2,
    "alpha_dpr_good": 0.55,
    "alpha_dpr_bad": 0.80,
    "alpha_artifact_floor": 0.5,
    "nonalpha_dpr_good": 0.25,
    "nonalpha_dpr_bad": 0.44,
    "nonalpha_artifact_floor": 0.1,
    "low_freq_penalty_floor": 0.3,
    "line_noise_penalty_floor": 0.4,
    "low_freq_penalty_ceiling": 0.7,
    "line_noise_penalty_ceiling": 0.7,
    "kurtosis_good_low": 2.0,
    "kurtosis_good_high": 8.0,
    "kurtosis_bad_high": 15.0,
    "kurtosis_floor": 0.2,
    "corr_low": 0.2,
    "corr_mid": 0.4,
    "corr_high": 0.84,
    "corr_floor": 0.3,
    "corr_low_score": 0.75,
    "corr_mid_score": 0.88,
    "flat_weight": 1.0,
    "spectrum_weight": 0.9,
    "kurtosis_weight": 0.75,
    "corr_weight": 0.15,
}


def get_default_eeg_quality_v2_params(target_score: float = 0.8) -> Dict[str, float]:
    params = dict(DEFAULT_EEG_QUALITY_V2_PARAMS)
    params["target_score"] = float(target_score)
    return params


def get_best_eeg_quality_v2_flat_spectrum_only_params() -> Dict[str, float]:
    params = dict(BEST_EEG_QUALITY_V2_MEAN_ABS_CORR_PARAMS)
    params["kurtosis_weight"] = 0.0
    params["corr_weight"] = 0.0
    return params


def _resolve_eeg_quality_v2_params(
    params: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    resolved = get_default_eeg_quality_v2_params()
    if params is not None:
        resolved.update(params)
    return resolved


def _linear_falloff(value: float, start: float, end: float, floor: float) -> float:
    if value <= start:
        return 1.0
    if end <= start:
        return float(floor)
    score = 1.0 - (1.0 - floor) * ((value - start) / (end - start))
    return float(np.clip(score, floor, 1.0))


def _piecewise_linear_correlation_score(
    corr_val: float,
    params: Mapping[str, float],
) -> float:
    corr_floor = float(params["corr_floor"])
    corr_low = float(params["corr_low"])
    corr_mid = float(params["corr_mid"])
    corr_high = float(params["corr_high"])
    corr_low_score = float(params["corr_low_score"])
    corr_mid_score = float(params["corr_mid_score"])

    if corr_val < corr_low:
        score = corr_floor + (corr_low_score - corr_floor) * (
            max(corr_val, 0.0) / max(corr_low, 1e-12)
        )
    elif corr_val < corr_mid:
        score = corr_low_score + (corr_mid_score - corr_low_score) * (
            (corr_val - corr_low) / max(corr_mid - corr_low, 1e-12)
        )
    elif corr_val <= corr_high:
        score = corr_mid_score + (1.0 - corr_mid_score) * (
            (corr_val - corr_mid) / max(corr_high - corr_mid, 1e-12)
        )
    else:
        score = 1.0 - (1.0 - corr_floor) * (
            (corr_val - corr_high) / max(1.0 - corr_high, 1e-12)
        )
    return float(np.clip(score, corr_floor, 1.0))


def _weighted_geometric_quality(
    component_scores: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
) -> np.ndarray:
    active_items = [(name, weight) for name, weight in weights.items() if weight > 0]
    if not active_items:
        raise ValueError("At least one v2 component weight must be > 0")

    weight_sum = sum(weight for _, weight in active_items)
    overall_log = np.zeros_like(
        next(iter(component_scores.values())),
        dtype=np.float64,
    )
    for name, weight in active_items:
        overall_log += (weight / weight_sum) * np.log(
            np.clip(component_scores[name], 1e-12, 1.0)
        )
    return np.exp(overall_log)


def get_eeg_quality_index_v2_parametric(
    data,
    fs: int = 200,
    params: Optional[Mapping[str, float]] = None,
):
    """
    Score one EEG segment of shape (n_channels, n_samples).

    Only the components with a positive weight in `params` are computed; the
    overall quality is their weighted geometric mean (a single bad component
    drags the whole score down).

    Returns:
        {
            "overall": np.ndarray shape (n_channels,),
            "detail": {<component>: np.ndarray, ...}  # active components only
        }
    """
    params = _resolve_eeg_quality_v2_params(params)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("data must have shape (n_channels, n_samples)")

    n_channels, n_samples = data.shape

    weights = {
        "flat": max(float(params["flat_weight"]), 0.0),
        "spectrum": max(float(params["spectrum_weight"]), 0.0),
        "kurtosis": max(float(params["kurtosis_weight"]), 0.0),
        "corr": max(float(params["corr_weight"]), 0.0),
    }

    def check_flat_and_sat_v2(ch_data: np.ndarray) -> float:
        if ch_data is None or np.size(ch_data) == 0:
            return 0.0
        if np.ptp(ch_data) < 1e-6:
            return 0.0

        window_size = max(int(0.5 * fs), 1)
        step = max(window_size // 2, 1)
        win_stds = []
        for i in range(0, max(n_samples - window_size, 0), step):
            win = ch_data[i : i + window_size]
            if win.size == 0:
                continue
            win_stds.append(np.std(win))

        if not win_stds:
            return 0.0

        win_stds = np.array(win_stds)
        positive_stds = win_stds[win_stds > 0]
        median_std = np.median(positive_stds) if positive_stds.size else 0.0

        activity_ratio = win_stds / max(median_std, 1e-7)
        activity_k = 0.8
        window_activity_score = 1.0 - np.exp(-activity_ratio / activity_k)
        base_activity_score = float(
            np.mean(np.clip(window_activity_score, 0.0, 1.0))
        )

        q10_std = float(np.percentile(win_stds, 10))
        q10_ratio = q10_std / max(median_std, 1e-7)
        q10_score = 1.0 - np.exp(-q10_ratio / 0.9)
        q10_score = float(np.clip(q10_score, 0.0, 1.0))

        soft_flat_ratio = 1.0 - base_activity_score

        median = np.median(ch_data)
        mad = np.median(np.abs(ch_data - median))
        robust_sigma = 1.4826 * mad
        if robust_sigma < 1e-7:
            robust_sigma = np.std(ch_data)
        spike_th = max(robust_sigma * 8.0, 1e-6)
        spike_ratio = float(np.mean(np.abs(ch_data - median) > spike_th))

        max_abs = np.max(np.abs(ch_data))
        clip_ratio = (
            float(np.mean(np.abs(ch_data) >= (max_abs * 0.98))) if max_abs > 0 else 0.0
        )

        signal_range = np.max(ch_data) - np.min(ch_data)
        if robust_sigma > 1e-7:
            normalized_range = signal_range / (robust_sigma * 6.0)
        else:
            normalized_range = 1.0

        score = 0.90 * base_activity_score + 0.10 * q10_score
        score = float(np.clip(score, 0.0, 1.0))

        if soft_flat_ratio > params["flat_ratio_start"]:
            flat_penalty = 1.0 - (
                (soft_flat_ratio - params["flat_ratio_start"])
                / max(1.0 - params["flat_ratio_start"], 1e-12)
                * (1.0 - params["flat_penalty_floor"])
            )
            flat_penalty = max(flat_penalty, params["flat_penalty_floor"])
            score *= flat_penalty

        if spike_ratio > params["spike_ratio_start"]:
            score *= _linear_falloff(
                spike_ratio,
                params["spike_ratio_start"],
                params["spike_ratio_end"],
                params["spike_penalty_floor"],
            )

        if clip_ratio > params["clip_ratio_start"]:
            score *= _linear_falloff(
                clip_ratio,
                params["clip_ratio_start"],
                params["clip_ratio_end"],
                params["clip_penalty_floor"],
            )

        if normalized_range > params["range_start"]:
            score *= _linear_falloff(
                normalized_range,
                params["range_start"],
                params["range_end"],
                params["range_penalty_floor"],
            )

        return max(score, 0.0)

    def check_spectrum_v2(ch_data: np.ndarray) -> float:
        try:
            f, psd = sp_signal.welch(ch_data, fs, nperseg=fs * 2)
            mask = (f > 1) & (f < 45)
            if np.sum(mask) < 2:
                return 0.5

            log_f = np.log10(f[mask])
            log_psd = np.log10(psd[mask] + 1e-10)
            slope, _ = np.polyfit(log_f, log_psd, 1)

            good_low = float(params["slope_good_low"])
            good_high = float(params["slope_good_high"])
            slope_center = float(params["slope_center"])
            slope_edge_score = float(params["slope_edge_score"])
            slope_outer_floor = float(params["slope_outer_floor"])
            half_span = max(
                max(slope_center - good_low, good_high - slope_center),
                1e-12,
            )
            if good_low <= slope <= good_high:
                slope_score = 1.0 - (1.0 - slope_edge_score) * (
                    abs(slope - slope_center) / half_span
                )
            elif slope < good_low:
                slope_score = slope_edge_score - (
                    slope_edge_score - slope_outer_floor
                ) * min(((good_low - slope) / half_span), 1.0)
            else:
                slope_score = slope_edge_score - (
                    slope_edge_score - slope_outer_floor
                ) * min(((slope - good_high) / half_span), 1.0)
            return float(np.clip(slope_score, slope_outer_floor, 1.0))
        except Exception:
            return 0.5

    def check_kurt_v2(ch_data: np.ndarray) -> float:
        try:
            k = kurtosis(ch_data, fisher=False)
            if not np.isfinite(k):
                return float(params["kurtosis_floor"])
            good_low = float(params["kurtosis_good_low"])
            good_high = float(params["kurtosis_good_high"])
            bad_high = float(params["kurtosis_bad_high"])
            floor = float(params["kurtosis_floor"])

            center = 0.5 * (good_low + good_high)
            half_span = max(0.5 * (good_high - good_low), 1e-12)
            inner_edge_score = 0.97

            if good_low <= k <= good_high:
                inner_t = min(abs(k - center) / half_span, 1.0)
                kurt_score = 1.0 - (1.0 - inner_edge_score) * (inner_t ** 1.5)
            elif k < good_low:
                t = (good_low - k) / half_span
                kurt_score = inner_edge_score - (inner_edge_score - floor) * (
                    1.0 - np.exp(-t)
                )
            elif k <= bad_high:
                t = (k - good_high) / max(bad_high - good_high, 1e-12)
                kurt_score = inner_edge_score - (inner_edge_score - floor) * t
            else:
                excess = (k - bad_high) / max(bad_high, 1e-12)
                kurt_score = floor + 0.05 * np.exp(-8.0 * excess)

            return float(np.clip(kurt_score, floor, 1.0))
        except Exception:
            return 0.5

    def check_corr_v2() -> np.ndarray:
        if n_channels <= 1:
            return np.ones(n_channels, dtype=np.float64)
        try:
            corr_matrix = np.abs(np.corrcoef(data))
        except Exception:
            corr_matrix = np.eye(n_channels)
        mean_abs_corr = (np.sum(corr_matrix, axis=1) - 1) / (n_channels - 1)
        return np.array(
            [_piecewise_linear_correlation_score(c, params) for c in mean_abs_corr]
        )

    detail: Dict[str, np.ndarray] = {}
    if weights["flat"] > 0:
        detail["flat"] = np.array(
            [check_flat_and_sat_v2(data[i]) for i in range(n_channels)]
        )
    if weights["spectrum"] > 0:
        detail["spectrum"] = np.array(
            [check_spectrum_v2(data[i]) for i in range(n_channels)]
        )
    if weights["kurtosis"] > 0:
        detail["kurtosis"] = np.array(
            [check_kurt_v2(data[i]) for i in range(n_channels)]
        )
    if weights["corr"] > 0:
        detail["corr"] = check_corr_v2()

    overall_quality = _weighted_geometric_quality(detail, weights)

    return {
        "overall": overall_quality,
        "detail": detail,
    }


__all__ = [
    "BEST_EEG_QUALITY_V2_MEAN_ABS_CORR_PARAMS",
    "DEFAULT_EEG_QUALITY_V2_PARAMS",
    "get_best_eeg_quality_v2_flat_spectrum_only_params",
    "get_default_eeg_quality_v2_params",
    "get_eeg_quality_index_v2_parametric",
]
