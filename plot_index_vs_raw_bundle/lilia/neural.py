"""Segment-preserving timeline and PyTorch overlap-add denoising."""
from __future__ import annotations

from dataclasses import dataclass, replace
import inspect

import numpy as np

from lilia.provenance import file_sha256
from lilia.signal import resample_segment_time_us, resample_polyphase, _resampling_ratio
from lilia.windowing import continuous_slices, build_window_grid


@dataclass
class InferenceTimeline:
    time_us: np.ndarray
    segment_ids: np.ndarray
    source_samples: int
    source_epoch_us: int
    fs_in: float
    fs_out: float
    model_window: int
    model_hop: int
    segments: list
    model_windows: list

    def metadata(self):
        return {key: getattr(self, key) for key in (
            'source_samples', 'source_epoch_us', 'fs_in', 'fs_out',
            'model_window', 'model_hop', 'segments', 'model_windows')}

    def grid(self, win_sec, step_sec=None):
        grid = build_window_grid(self.time_us, self.fs_out, win_sec, step_sec,
                                 segment_ids=self.segment_ids, epoch_us=self.source_epoch_us, reset_per_segment=True)
        ends = {s['segment_id']: s['raw_end_us'] for s in self.segments}
        capped = np.minimum(grid.columns['window_end_us'], [ends[int(s)] for s in grid.columns['segment_id']])
        capped.flags.writeable = False
        return replace(grid, columns={**grid.columns, 'window_end_us': capped})


def build_inference_timeline(time_us, fs_in, fs_out=200., model_window=400, model_hop=200):
    """Map retained output samples and mirrored model windows to source segments.

    Each segment needs a complete real model window after polyphase resampling.
    Output indexes address the packed resampled axis, never the raw source.
    Model input intervals are relative to that segment's resampled array;
    negative/out-of-range indexes refer to symmetric mirrored edge padding.
    """
    t = np.asarray(time_us)
    if t.dtype.kind not in 'iu' or not len(t):
        raise ValueError('Inference timestamps must be nonempty integer microseconds')
    if (not isinstance(model_window, int) or not isinstance(model_hop, int)
            or model_window < 4 or not 1 <= model_hop < model_window):
        raise ValueError('Invalid model window or hop')
    up, dn = _resampling_ratio(fs_in, fs_out)
    times, groups, segments, model_windows = [], [], [], []
    offset = 0
    for sid, sl in enumerate(continuous_slices(t, fs_in)):
        n_out = ((sl.stop - sl.start) * up + dn - 1) // dn
        record = dict(segment_id=sid, raw_start_idx=sl.start, raw_end_idx=sl.stop,
                      raw_start_us=int(t[sl.start]), raw_end_us=int(t[sl.stop - 1]) + int(round(1e6 / fs_in)),
                      resampled_samples=n_out, ratio_up=up, ratio_down=dn,
                      status='excluded', reason='shorter_than_model_window')
        if n_out >= model_window:
            part = resample_segment_time_us(t[sl], fs_in, fs_out)
            record.update(status='retained', reason='', output_start_idx=offset, output_end_idx=offset + n_out)
            times.append(part)
            groups.append(np.full(n_out, sid, dtype=np.int64))
            count = (n_out + 2 * model_hop - model_window) // model_hop + 1
            for i in range(count):
                start = i * model_hop - model_hop
                model_windows.append(dict(segment_id=sid, input_start_idx=start,
                                          input_end_idx=start + model_window,
                                          output_start_idx=offset + max(0, start),
                                          output_end_idx=offset + min(n_out, start + model_window)))
            offset += n_out
        segments.append(record)
    output_t = np.concatenate(times) if times else np.empty(0, dtype=np.int64)
    output_groups = np.concatenate(groups) if groups else np.empty(0, dtype=np.int64)
    # Explicit source IDs preserve even gaps smaller than 3 output periods.
    continuous_slices(output_t, fs_out, segment_ids=output_groups)
    return InferenceTimeline(output_t, output_groups, len(t), int(t[0]),
                             float(fs_in), float(fs_out), model_window, model_hop, segments, model_windows)


def denoise_with_time(time_us, data_raw, fs=500., *, model=None):
    """Filter/resample/infer within each retained raw segment, returning its map."""
    import data_analysis as da
    data = np.asarray(data_raw, dtype=float)
    if data.ndim != 2 or data.shape[1] < da.N_CH or len(data) != len(time_us):
        raise ValueError('Denoising needs aligned timestamps and at least four raw channels')
    if not np.isfinite(data).all():
        raise ValueError('Denoising input contains non-finite samples')
    timeline = build_inference_timeline(time_us, fs, da.DOWNSAMPLED_FS, da.MODEL_WINDOW, da.MODEL_WINDOW // 2)
    if not len(timeline.time_us):
        raise ValueError('No continuous segment contains a complete model window after resampling')
    if model is None:
        model = da.load_model()
    out = []
    for record in timeline.segments:
        if record['status'] != 'retained':
            continue
        a, b = record['raw_start_idx'], record['raw_end_idx']
        filtered = da.apply_filters(data[a:b, :da.N_CH], fs=fs)
        resampled = resample_polyphase(filtered, fs, timeline.fs_out)
        prediction = np.asarray(da.run_model(model, resampled, window=timeline.model_window,
                                            hop=timeline.model_hop))
        if prediction.shape != (record['resampled_samples'], da.N_CH_OUT) or not np.isfinite(prediction).all():
            raise ValueError('Model output does not match retained inference timestamps')
        out.append(prediction)
    return timeline, np.concatenate(out, axis=0)


def model_provenance():
    """Record checkpoint and external architecture fingerprints without copying either."""
    import data_analysis as da
    architecture = inspect.getfile(da.import_tinyunetv4())
    return {'checkpoint_sha256': file_sha256(da.MODEL_PATH),
            'architecture_sha256': file_sha256(architecture),
            'backend': 'pytorch', 'normalization': 'per-window RMS',
            'padding': 'symmetric mirrored edges, hop samples on each side',
            'overlap_add': 'Hann weighted, crop to real resampled samples',
            'filters': {'bandpass': [da.BANDPASS_LOW, da.BANDPASS_HIGH],
                        'notch': [da.NOTCH_FREQ, da.NOTCH_Q],
                        'bandstop': [da.ARTIFACT_PEAK_HZ, da.ARTIFACT_PEAK_BW]}}
