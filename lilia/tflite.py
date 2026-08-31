"""Shared TFLite model inference utilities."""

from __future__ import annotations

import numpy as np


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
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

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
        chunks.append(pred.astype(np.float32, copy=False))

    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 2), np.float32)
