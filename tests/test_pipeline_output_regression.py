import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd

from test_csv_contract_regression import write_recording


class PipelineOutputTests(unittest.TestCase):
    def test_eye_output_header_matches_resampled_channels_and_time(self):
        import process_lilia_eye_open_close as eye
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / 'eye.csv'
            rows = ''.join(f'{i * 2000},' + ','.join(str(np.sin(i / 8 + ch)) for ch in range(8)) + '\n'
                           for i in range(1000))
            write_recording(raw, rows, offset=0, channels=','.join(['value'] * 8))
            args = argparse.Namespace(csv=str(raw), outdir=str(root), fmax=50)
            with patch.object(eye, 'parse_args', return_value=args), \
                 patch.object(eye, 'load_model', return_value=object()), \
                 patch.object(eye, 'run_model', side_effect=lambda model, x: x[:, :2]), \
                 patch.object(eye, 'plot_output_channels', return_value='mock.png'), \
                 patch.object(eye, 'plot_before_after_channels', return_value='mock.png'), \
                 contextlib.redirect_stdout(io.StringIO()):
                eye.main()
            output = root / 'eye_tinyv4_output.csv'
            lines = output.read_text().splitlines()
            self.assertEqual(lines[2], 'Channels,1,2,5,6')
            self.assertEqual(lines[3], 'Sample Rate (per channel),200,200,200,200')
            self.assertEqual(lines[4], 'Time[us],ch1,ch2,ch5,ch6')
            df = pd.read_csv(output, skiprows=4)
            self.assertEqual(df.shape, (400, 5))
            np.testing.assert_array_equal(df.iloc[:, 0], np.arange(400) * 5000)

    def test_custom_markers_at_200hz_apply_quality_in_aligned_windows(self):
        import plot_index_vs_raw as plot
        epoch, _ = plot._parse_marker_us('12:00:00', '2026-05-12')
        t = epoch + np.arange(6000) * 5000
        data = np.sin(np.arange(6000)[:, None] / 4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def score(x, fs, params):
                self.assertEqual(fs, 200)
                self.assertEqual(x.shape, (1, 1000))
                return {'overall': np.zeros(1)}
            with patch.object(plot, 'load_merged_csv', return_value=(t, data)), \
                 patch.object(plot.pem, 'get_eeg_quality_index_v2_parametric', side_effect=score) as scorer, \
                 contextlib.redirect_stdout(io.StringIO()):
                plot.plot_custom_markers('unused.csv', 'Demo', 1, ['12:00:15'], '2026-05-12',
                                         10, 10, str(root / 'out.png'), str(root / 'out.csv'), fs=200)
            self.assertEqual(scorer.call_count, 6)
            df = pd.read_csv(root / 'out.csv')
            self.assertEqual(df.loc[0, 'baseline_windows'], 0)
            self.assertEqual(df.loc[0, 'post_windows'], 0)
            self.assertTrue(np.isnan(df.loc[0, 'focus_delta']))
            self.assertTrue((root / 'out.png').is_file())

    def test_event_selection_preserves_prior_event_baseline_anchor(self):
        import plot_index_vs_raw as plot
        with patch.object(plot.pem, 'EVENTS', [('first', '12:00', 1, None), ('second', '12:01', 1, None)]), \
             patch.object(plot.pem, 'hhmm_to_us', side_effect=lambda clock: {'12:00': 60000000, '12:01': 120000000}[clock]):
            windows = plot._event_windows('Demo', 0, event_name='second')
            self.assertEqual([w['name'] for w in windows], ['second'])
            self.assertEqual(windows[0]['anchor_name'], 'first')
            with self.assertRaisesRegex(ValueError, 'No participating event'):
                plot._event_windows('Demo', 0, event_name='missing')

    def test_negative_entropy_delta_remains_visible(self):
        import plot_index_vs_raw as plot
        from matplotlib.figure import Figure
        captured = []
        def capture(fig, *args, **kwargs):
            captured.append(fig.axes[0].get_ylim())
        with patch.object(plot, '_load_signals', return_value=(np.array([0., 1., 2.]), {'entropy': np.array([0.8, 0.2, 0.1])}, np.ones(3))), \
             patch.object(plot, '_load_raw', return_value=(np.arange(3), np.zeros(3), 0)), \
             patch.object(plot, '_event_windows', return_value=[dict(name='event', onset=1., end=3., pre_lo=0., pre_hi=1., post_lo=1., post_hi=2., continuous=False)]), \
             patch.object(Figure, 'savefig', capture), contextlib.redirect_stdout(io.StringIO()):
            plot.plot_single_signal('unused', 'unused', 'Demo', 1, 'entropy', 'Entropy', 'blue', '/tmp/unused.png', smooth_win=1)
        self.assertTrue(all(lo < -0.7 for lo, hi in captured))
