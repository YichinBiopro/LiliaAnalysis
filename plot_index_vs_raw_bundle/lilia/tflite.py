"""Shared TFLite model inference utilities."""

from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np
from lilia.signal import _resampling_ratio, resample_segment_time_us, resample_polyphase
from lilia.windowing import continuous_slices, build_window_grid


def apply_tflite_windowed(
    data: np.ndarray,
    tflite_path: str,
    tflite_win: int = 400,
) -> np.ndarray:
    """Run TFLite model on (N, n_ch) data in non-overlapping windows.

    Parameters
    ----------
    data        : (N, n_ch) float32 signal (typically 4-channel EEG)
    tflite_path : path to the .tflite model file
    tflite_win  : window size in samples (default 400)

    Returns
    -------
    out : (M, 2) float32, M = (N // tflite_win) * tflite_win
          Model output (typically 2-channel denoised signal)

    Notes
    -----
    Per-window RMS normalization is applied to match the model's training
    convention: each window is normalized by its RMS, the normalized window
    is run through the model, and the output is rescaled by the RMS.
    """
    if tflite_win < 1:
        raise ValueError('tflite_win must be positive')
    data = np.asarray(data)
    if data.ndim != 2 or not np.all(np.isfinite(data)):
        raise ValueError('model data must be a finite (samples, channels) array')
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    if tuple(inp_det['shape']) != (1, tflite_win, data.shape[1]):
        raise ValueError(f"Model input {inp_det['shape']} does not match (1, {tflite_win}, {data.shape[1]})")
    if inp_det['dtype'] != np.float32:
        raise ValueError('This RMS inference backend requires a float32 model')
    if tuple(out_det['shape']) != (1, tflite_win, 2) or out_det['dtype'] != np.float32:
        raise ValueError('Model output must be float32 (1, window, 2)')
    n_win = len(data) // tflite_win
    chunks = []

    for i in range(n_win):
        seg = data[i * tflite_win : (i + 1) * tflite_win][np.newaxis].astype(np.float32)
        # Per-window RMS normalization
        seg_rms = np.sqrt(np.mean(seg.astype(np.float64) ** 2)) + 1e-8
        seg_norm = (seg / np.float32(seg_rms)).astype(np.float32, copy=False)
        # Run model
        interp.set_tensor(inp_det['index'], seg_norm)
        interp.invoke()
        # Rescale output by RMS
        pred = interp.get_tensor(out_det['index'])[0] * np.float32(seg_rms)
        if pred.shape != (tflite_win, 2) or not np.isfinite(pred).all():
            raise ValueError('TFLite output shape or values are invalid')
        chunks.append(pred.astype(np.float32, copy=False))

    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 2), np.float32)


def apply_tflite_with_time(time_us, data, tflite_path, tflite_win=400, fs=200, *, segment_ids=None):
    """Infer complete windows within each segment and return their true times.

    Trim each segment separately before packing windows into one interpreter
    invocation loop. No model window contains samples from both sides of a gap.
    """
    t = np.asarray(time_us)
    if len(t) != len(data):
        raise ValueError('timestamp and signal lengths differ')
    if tflite_win < 1:
        raise ValueError('tflite_win must be positive')
    selected = []
    for sl in continuous_slices(t, fs, segment_ids=segment_ids):
        keep = ((sl.stop - sl.start) // tflite_win) * tflite_win
        if keep:
            selected.append(np.arange(sl.start, sl.start + keep))
    idx = np.concatenate(selected) if selected else np.empty(0, dtype=int)
    return t[idx], apply_tflite_windowed(np.asarray(data)[idx], tflite_path, tflite_win)


@dataclass
class TFLiteTimeline:
    time_us: np.ndarray
    segment_ids: np.ndarray
    segments: list
    model_windows: list
    source_samples: int
    source_epoch_us: int
    fs_in: float
    fs_out: float
    model_window: int

    def metadata(self):
        return {k: getattr(self, k) for k in ('segments', 'model_windows', 'source_samples',
                'source_epoch_us', 'fs_in', 'fs_out', 'model_window')}

    def grid(self, win_sec, step_sec=None):
        grid = build_window_grid(self.time_us, self.fs_out, win_sec, step_sec,
                    segment_ids=self.segment_ids, epoch_us=self.source_epoch_us, reset_per_segment=True)
        ends = {s['segment_id']: s['raw_end_us'] for s in self.segments}
        capped = np.minimum(grid.columns['window_end_us'], [ends[int(s)] for s in grid.columns['segment_id']])
        capped.flags.writeable = False
        return replace(grid, columns={**grid.columns, 'window_end_us': capped})


def build_tflite_timeline(time_us, fs_in=500., fs_out=200., model_window=400):
    """Map complete nonoverlapping model windows, trimming each source segment."""
    t = np.asarray(time_us)
    if t.dtype.kind not in 'iu' or not len(t):
        raise ValueError('TFLite timestamps must be nonempty integer microseconds')
    if not isinstance(model_window, int) or model_window < 1:
        raise ValueError('Model window must be a positive integer')
    up, dn = _resampling_ratio(fs_in, fs_out)
    segments, windows, times, groups = [], [], [], []
    offset = 0
    for sid, sl in enumerate(continuous_slices(t, fs_in)):
        n = ((sl.stop - sl.start) * up + dn - 1) // dn
        keep = n // model_window * model_window
        end_us = int(t[sl.stop - 1]) + int(round(1e6 / fs_in))
        row = dict(segment_id=sid, raw_start_idx=sl.start, raw_end_idx=sl.stop,
                   raw_start_us=int(t[sl.start]), raw_end_us=end_us,
                   resampled_samples=n, retained_samples=keep, trimmed_samples=n - keep,
                   ratio_up=up, ratio_down=dn, status='retained' if keep else 'excluded',
                   reason='' if keep else 'shorter_than_model_window')
        if keep:
            part = resample_segment_time_us(t[sl], fs_in, fs_out)[:keep]
            row.update(output_start_idx=offset, output_end_idx=offset + keep)
            times.append(part)
            groups.append(np.full(keep, sid, dtype=np.int64))
            for a in range(0, keep, model_window):
                b = a + model_window
                lo = int(part[a])
                hi = min(end_us, int(part[b - 1]) + int(round(1e6 / fs_out)))
                windows.append(dict(model_window_id=len(windows), segment_id=sid,
                    output_start_idx=offset + a, output_end_idx=offset + b,
                    window_start_us=lo, window_end_us=hi,
                    raw_start_idx=int(np.searchsorted(t, lo)), raw_end_idx=int(np.searchsorted(t, hi))))
            offset += keep
        segments.append(row)
    return TFLiteTimeline(np.concatenate(times) if times else np.empty(0, dtype=np.int64),
                         np.concatenate(groups) if groups else np.empty(0, dtype=np.int64),
                         segments, windows, len(t), int(t[0]), float(fs_in), float(fs_out), model_window)


def run_tflite_recording(time_us, filtered_data, model_path, fs_in=500., fs_out=200., model_window=400):
    """Resample source segments separately and infer their complete windows once.

    ``filtered_data`` must have been filtered separately within raw segments.
    Before/after arrays use exactly the same retained output sample positions.
    """
    timeline = build_tflite_timeline(time_us, fs_in, fs_out, model_window)
    data = np.asarray(filtered_data)
    if data.ndim != 2 or data.shape != (len(time_us), 4):
        raise ValueError('TFLite recording needs aligned timestamps and four channels')
    if not len(timeline.time_us):
        raise ValueError('No complete TFLite model window in any source segment')
    parts = []
    for row in timeline.segments:
        if row['status'] == 'retained':
            part = data[row['raw_start_idx']:row['raw_end_idx']]
            if not np.isfinite(part).all():
                raise ValueError('Non-finite input in a retained TFLite segment')
            parts.append(resample_polyphase(part, fs_in, fs_out)[:row['retained_samples']])
    pre = np.concatenate(parts)
    out = apply_tflite_windowed(pre, model_path, model_window)
    if out.shape != (len(pre), 2) or not np.isfinite(out).all():
        raise ValueError('TFLite output does not match its retained timeline')
    return timeline, pre, out
