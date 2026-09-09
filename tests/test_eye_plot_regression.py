"""Numerical and rendered boundaries for eye TD/STFT plots."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from matplotlib.figure import Figure
import numpy as np

import process_lilia_eye_open_close as eye


class EyePlotTests(unittest.TestCase):
    def timeline(self, lengths=(998, 998), starts=(0, 2002000), epoch=9000000000000001):
        time_us = epoch + np.concatenate([start + np.arange(n) * 2000
                                          for start, n in zip(starts, lengths)])
        return eye.build_inference_timeline(time_us, 500, 200, 400, 200)

    def test_segmented_stft_preserves_frozen_real_and_synthetic_values(self):
        for name in ('real', 'synthetic'):
            with self.subTest(name=name), np.load(Path(__file__).parent / 'fixtures' / f'eye_{name}_continuous_reference.npz') as ref:
                timeline = eye.build_inference_timeline(ref['raw_time_us'], 500, 200, 400, 200)
                for stage, values in (('before', ref['before'][:, [0, 1, 4, 5]]), ('after', ref['processed'])):
                    panel, = eye.segmented_stft(timeline, values, 50)
                    for col, channel in enumerate((1, 2, 5, 6)):
                        np.testing.assert_array_equal(panel['frequency_hz'], ref[f'{stage}_ch{channel}_f'])
                        np.testing.assert_allclose(panel['stft_time_s'], ref[f'{stage}_ch{channel}_t'], rtol=0, atol=1e-12)
                        np.testing.assert_array_equal(panel['db'][col], ref[f'{stage}_ch{channel}_db'])

    def test_small_gap_has_two_independent_stfts_and_padding_is_clipped(self):
        timeline = self.timeline()
        values = np.random.default_rng(153).normal(size=(800, 2))
        panels = eye.segmented_stft(timeline, values, 50)
        self.assertEqual(len(panels), 2)
        self.assertEqual(panels[0]['clip_end_s'], 1.996)
        self.assertEqual(panels[1]['clip_start_s'], 2.002)
        self.assertGreater(panels[0]['stft_time_s'][-1], panels[0]['clip_end_s'])
        for i, panel in enumerate(panels):
            for col in range(2):
                _, local_time, db = eye.compute_stft_db(values[i * 400:(i + 1) * 400, col], 200, 50)
                np.testing.assert_array_equal(panel['db'][col], db)
                np.testing.assert_allclose(panel['stft_time_s'], local_time + i * 2.002, rtol=0, atol=1e-12)
        values[:400] *= 1e6
        changed = eye.segmented_stft(timeline, values, 50)
        np.testing.assert_array_equal(changed[1]['db'], panels[1]['db'])

    def test_short_prefix_tail_and_large_epoch_keep_source_elapsed_extent(self):
        timeline = self.timeline((10, 1503, 997), (0, 10000000, 20000000))
        meta = eye.plot_metadata(timeline, 50)
        self.assertEqual(meta['xlim_s'], [0., 21.994])
        self.assertEqual(meta['segments'][0]['segment_id'], 1)
        self.assertEqual(meta['segments'][0]['clip_start_s'], 10.)
        self.assertEqual(meta['segments'][0]['clip_end_s'], 13.006)
        self.assertEqual([s['segment_id'] for s in meta['excluded_spans']], [0, 2])
        self.assertEqual(meta['missing_spans'], [{'start_s': .02, 'end_s': 10.},
                                               {'start_s': 13.006, 'end_s': 20.}])
        coords, = eye.plot_coordinates(timeline)
        np.testing.assert_allclose(coords['sample_time_s'], 10. + np.arange(602) / 200, rtol=0, atol=1e-12)

    def test_stft_centers_follow_timestamp_jitter_without_changing_spectrum(self):
        timestamps = 9000000000000001 + np.arange(1503) * 2000
        timestamps[500:] += 1000
        timeline = eye.build_inference_timeline(timestamps, 500, 200, 400, 200)
        panel, = eye.segmented_stft(timeline, np.ones((602, 1)), 50)
        positions = panel['stft_sample_positions']
        inside = positions < 602
        np.testing.assert_array_equal(panel['stft_time_s'][inside],
                                      (timeline.time_us[positions[inside]] - timestamps[0]) / 1e6)
        _, _, expected = eye.compute_stft_db(np.ones(602), 200, 50)
        np.testing.assert_array_equal(panel['db'][0], expected)

    def test_rendered_comparison_uses_ch5_6_and_never_draws_across_small_gap(self):
        timeline = self.timeline()
        before = np.broadcast_to(np.arange(1, 9), (800, 8)).copy()
        after = np.broadcast_to(np.array([11, 12, 15, 16]), (800, 4)).copy()
        captured = []

        def inspect(fig, *args, **kwargs):
            fig.canvas.draw()
            captured.append(fig)
            axes = [ax for ax in fig.axes if ax.get_xlabel() == 'Elapsed time (s)']
            self.assertEqual(len(axes), 8)
            widths = [ax.get_position().width for ax in axes]
            np.testing.assert_allclose(widths, widths[0], rtol=0, atol=1e-10)
            self.assertIn('channels 5-6', fig._suptitle.get_text())
            for ax in axes:
                np.testing.assert_array_equal(ax.get_xlim(), [0., 3.998])
                channel = 5 if 'Ch5' in ax.get_title() else 6
                if '(Time)' in ax.get_title():
                    signals = [line for line in ax.lines if len(line.get_xdata()) == 400]
                    self.assertEqual(len(signals), 2)
                    expected = channel if 'Before' in ax.get_title() else channel + 10
                    for line, start in zip(signals, (0., 2.002)):
                        np.testing.assert_array_equal(line.get_ydata(), np.full(400, expected))
                        self.assertEqual(line.get_xdata()[0], start)
                else:
                    self.assertEqual(len(ax.collections), 2)
                    for mesh, bounds in zip(ax.collections, ((0., 1.996), (2.002, 3.998))):
                        clip = mesh.get_clip_box().transformed(ax.transData.inverted())
                        np.testing.assert_allclose(clip.extents[[0, 2]], bounds, rtol=0, atol=1e-12)

        with tempfile.TemporaryDirectory() as tmp, patch.object(Figure, 'savefig', inspect):
            eye.plot_before_after_channels(timeline, before, after, tmp, 'mapping', 50, 2, 'ch5_6')
        self.assertEqual(len(captured), 1)
        self.assertFalse(eye.plt.fignum_exists(captured[0].number))

    def test_rendered_output_keeps_short_spans_and_closes_on_save_failure(self):
        timeline = self.timeline((10, 998, 997), (0, 10000000, 20000000))
        captured = []

        def fail(fig, *args, **kwargs):
            captured.append(fig.number)
            for ax in fig.axes:
                if ax.get_xlabel() == 'Elapsed time (s)':
                    self.assertEqual(len(ax.patches), 2)
                    self.assertNotIn('Ch3', ax.get_title())
                    self.assertNotIn('Ch4', ax.get_title())
                    self.assertEqual(ax.get_xlim()[1], 21.994)
            raise OSError('simulated disk failure')

        with tempfile.TemporaryDirectory() as tmp, patch.object(Figure, 'savefig', fail):
            with self.assertRaisesRegex(OSError, 'disk failure'):
                eye.plot_output_channels(timeline, np.ones((400, 4)), tmp, 'short', 50, 2, 'ch5_6')
        self.assertFalse(eye.plt.fignum_exists(captured[0]))


if __name__ == '__main__':
    unittest.main()
