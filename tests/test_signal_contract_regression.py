import unittest
from unittest.mock import patch

import numpy as np
from scipy import signal

from lilia.qeeg import compute_relative_powers, compute_qeeg_indices_windowed
from lilia.signal import bandpass, resample_with_time, resample_polyphase
from lilia.tflite import apply_tflite_with_time
from lilia.windowing import continuous_slices, require_continuous, window_starts
from lilia.segment_sampling import pick_non_overlapping_segments


class SignalContractTests(unittest.TestCase):
    def test_seconds_call_site_keeps_fractional_time(self):
        from data_analysis import downsample_data
        t, y = downsample_data(np.arange(1000) / 500, np.ones((1000, 4)))
        np.testing.assert_allclose(t, np.arange(400) / 200, atol=1e-12)
        self.assertEqual(y.shape, (400, 4))

    def test_integer_microsecond_api_rejects_seconds(self):
        with self.assertRaisesRegex(ValueError, 'integer microseconds'):
            resample_with_time(np.arange(20) / 500, np.ones(20), 500, 200)

    def test_fractional_rate_is_not_truncated(self):
        self.assertEqual(len(resample_polyphase(np.ones(5009), 500.9, 200)), 2000)

    def test_resampling_preserves_epoch_gap_and_segment_values(self):
        epoch = 1778566248498593
        t = epoch + np.r_[np.arange(500) * 2000, 5000000 + np.arange(500) * 2000]
        x = np.r_[np.ones(500), np.ones(500) * 10]
        new_t, y = resample_with_time(t, x, 500, 200)
        np.testing.assert_array_equal(new_t[:200], epoch + np.arange(200) * 5000)
        self.assertEqual(new_t[200], epoch + 5000000)
        np.testing.assert_allclose(y[:200], resample_polyphase(x[:500], 500, 200))
        np.testing.assert_allclose(y[200:], resample_polyphase(x[500:], 500, 200))

    def test_filter_does_not_mix_disconnected_segments(self):
        x = np.random.default_rng(42).normal(size=(1000, 2))
        x[500:] += 100
        t = np.r_[np.arange(500) * 2000, 5000000 + np.arange(500) * 2000]
        expected = np.concatenate([bandpass(x[:500]), bandpass(x[500:])])
        np.testing.assert_allclose(bandpass(x, time_us=t), expected)

    def test_gap_windows_and_legacy_guard(self):
        t = np.r_[np.arange(7) * 2000, 1000000 + np.arange(9) * 2000]
        self.assertEqual(list(window_starts(16, 4, 2, t, 500)), [0, 2, 8, 10, 12])
        with self.assertRaisesRegex(ValueError, 'split at timestamp gaps'):
            require_continuous(t, 500)
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            continuous_slices(np.array([0, 0]), 500)

    def test_tflite_trims_each_segment_and_keeps_true_timestamps(self):
        t = np.r_[np.arange(500) * 5000, 5000000 + np.arange(500) * 5000]
        x = np.arange(4000).reshape(1000, 4)
        with patch('lilia.tflite.apply_tflite_windowed', side_effect=lambda data, *_: data[:, :2]) as infer:
            actual_t, out = apply_tflite_with_time(t, x, 'unused.tflite')
        idx = np.r_[0:400, 500:900]
        np.testing.assert_array_equal(actual_t, t[idx])
        np.testing.assert_array_equal(out, x[idx, :2])
        self.assertEqual(infer.call_count, 1)

    def test_short_welch_windows_are_finite_and_long_profile_unchanged(self):
        x = np.random.default_rng(7).normal(size=2500)
        for seconds in (1, 2, 5):
            self.assertTrue(np.isfinite(compute_relative_powers(x[:seconds * 500])).all())
        f, p = signal.welch(x, fs=500, nperseg=2000, noverlap=1000, window='hann')
        powers = np.array([np.trapezoid(p[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)])
                           for lo, hi in [(4, 8), (8, 13), (13, 30)]])
        from lilia.qeeg import EPSILON
        np.testing.assert_allclose(compute_relative_powers(x), powers / (powers.sum() + EPSILON))
        with self.assertRaisesRegex(ValueError, 'at least one sample'):
            compute_qeeg_indices_windowed(x, step_sec=0.0001)

    def test_summary_resampling_branch_resolves_import(self):
        from plot_tflite_summary import _resample_500_to_200
        self.assertEqual(_resample_500_to_200(np.ones((1000, 4))).shape, (400, 4))

    def test_fully_packed_sampling_always_finds_capacity(self):
        for seed in range(10):
            self.assertEqual(pick_non_overlapping_segments(200, 100, 2, np.random.default_rng(seed)), [0, 100])
