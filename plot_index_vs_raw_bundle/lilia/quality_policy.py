"""One explicit validity policy for Goertzel tables."""
import numpy as np


def valid_goertzel_rows(frame, threshold=0.5, exclude_hard=True, require_power=False):
    key = 'quality_final' if exclude_hard and 'quality_final' in frame else 'quality'
    q = frame[key].to_numpy(dtype=float)
    keep = np.isfinite(q) & np.isfinite(frame['goertzel_db'].to_numpy(dtype=float))
    if threshold is not None:
        keep &= q > threshold
    if 'time_s' in frame:
        keep &= np.isfinite(frame['time_s'].to_numpy(dtype=float))
    if require_power:
        power = frame['goertzel_power'].to_numpy(dtype=float)
        keep &= np.isfinite(power) & (power >= 0)
    if exclude_hard and 'artifact_hard_clip' in frame:
        hard = frame['artifact_hard_clip'].to_numpy(dtype=float)
        keep &= np.isfinite(hard) & (hard < 0.5)
    return keep
